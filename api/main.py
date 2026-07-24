import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import redis
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rq.job import Job

from processor.config import settings
from processor.utils.download import fetch_audio_from_url
from processor.utils.validation import validate_file
from processor.utils.audio_io import load_wav, load_wav_stereo, run_ffmpeg_normalize, save_wav
from processor.utils.vocal_extraction import extract_vocals
from processor.dsp.analysis.vocal_layers_analysis import detect_vocal_layers
from processor.dsp.effects.apply_vocal_layers import build_vocal_layer_stems
from api.metrics import register_metrics_endpoint
from api.chelsea_football import build_chelsea_demo_snapshot, build_chelsea_snapshot
from api.ai_audio import (
    prompt_to_dsp_config,
    generate_preset_from_features,
    reference_match_config,
    clear_session,
    _DEFAULTS,
    validate_dsp_config,
)

app = FastAPI(title="Vocal Style Transfer (DSP-first)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    _assets_dir = settings.frontend_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="root-assets")

# VocalForge standalone site (same API origin, so no CORS setup needed)
_vocalforge_dir = settings.frontend_dir.parent / "vocalforge"
if _vocalforge_dir.exists():
    app.mount("/vocalforge", StaticFiles(directory=_vocalforge_dir, html=True), name="vocalforge")

# Register metrics endpoint
register_metrics_endpoint(app)


def enqueue_job(reference: Path, dry: Path, options: dict | None = None) -> str:
    """
    Lazy import worker enqueue function to avoid loading heavy ML modules
    during API startup on memory-constrained deploys.
    """
    from processor.worker import enqueue_job as _enqueue_job

    return _enqueue_job(reference, dry, options)


@app.get("/api/chelsea/football")
async def api_chelsea_football():
    """
    Live-style snapshot for Chelsea only: squad, upcoming fixtures, PL table row,
    plus standings rows for CL / EL / FA Cup / EFL Cup when your football-data.org plan allows.

    Without APP_FOOTBALL_DATA_API_TOKEN, returns 200 with a demo-shaped snapshot so the UI still works.
    For live data, register at https://www.football-data.org/client/register and set the token.
    """
    token = (
        (settings.football_data_api_token or os.environ.get("APP_FOOTBALL_DATA_API_TOKEN") or "")
        .strip()
    )
    if not token:
        return build_chelsea_demo_snapshot()
    return await build_chelsea_snapshot(token)


class JobStatus(BaseModel):
    job_id: str
    status: str
    download_url: str | None = None
    recipe_url: str | None = None
    error: str | None = None


def _strtobool(val: str) -> bool:
    """Convert string to boolean."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    return True  # default


def _save_upload(tmp_dir: Path, uploaded: UploadFile) -> Path:
    dst = tmp_dir / f"{uuid.uuid4()}_{uploaded.filename}"
    with dst.open("wb") as f:
        shutil.copyfileobj(uploaded.file, f)
    return dst


@app.post("/jobs")
async def create_job(
    reference: UploadFile | None = File(None),
    reference_url: str | None = Form(None),
    dry: UploadFile = File(...),
    enable_width: str | None = Form("true"),
    ml_refine: str | None = Form("false"),
    use_spectral_refiner: str | None = Form("true"),
    enable_deesser: str | None = Form("true"),
    enable_transient_shaper: str | None = Form("false"),
    enable_multiband: str | None = Form("false"),
    adaptive_dsp: str | None = Form("false"),
    enable_harmony: str | None = Form("true"),
    enable_chorus_flanger: str | None = Form("true"),
    enable_delay: str | None = Form("true"),
    enable_gate: str | None = Form("true"),
    enable_doubler: str | None = Form("true"),
    enable_exciter: str | None = Form("true"),
    enable_tape: str | None = Form("true"),
    enable_parallel_comp: str | None = Form("true"),
    enable_ms_eq: str | None = Form("true"),
    ai_dsp_config: str | None = Form(None),
    selected_layers: str | None = Form(None),
):
    if not reference and not reference_url:
        raise HTTPException(status_code=400, detail="Provide reference file or URL")

    if reference and reference.content_type and "audio" not in reference.content_type:
        raise HTTPException(status_code=415, detail="Unsupported reference file type")
    if dry.content_type and "audio" not in dry.content_type:
        raise HTTPException(status_code=415, detail="Unsupported dry file type")

    tmp_dir = settings.inputs_dir
    if reference:
        ref_path = _save_upload(tmp_dir, reference)
    else:
        ref_path = fetch_audio_from_url(reference_url)  # type: ignore[arg-type]
    dry_path = _save_upload(tmp_dir, dry)

    # Check if this is an RVC job
    is_rvc_job = _strtobool(ml_refine) if ml_refine is not None else False
    
    try:
        validate_file(ref_path, is_rvc_job=is_rvc_job)
        validate_file(dry_path, is_rvc_job=is_rvc_job)
    except HTTPException:
        ref_path.unlink(missing_ok=True)
        dry_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        ref_path.unlink(missing_ok=True)
        dry_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    options = {
        "enable_width": _strtobool(enable_width) if enable_width is not None else True,
        "ml_refine": _strtobool(ml_refine) if ml_refine is not None else False,
        "use_spectral_refiner": _strtobool(use_spectral_refiner) if use_spectral_refiner is not None else True,
        "enable_deesser": _strtobool(enable_deesser) if enable_deesser is not None else True,
        "enable_transient_shaper": _strtobool(enable_transient_shaper) if enable_transient_shaper is not None else False,
        "enable_multiband": _strtobool(enable_multiband) if enable_multiband is not None else False,
        "adaptive_dsp": _strtobool(adaptive_dsp) if adaptive_dsp is not None else False,
        "enable_harmony": _strtobool(enable_harmony) if enable_harmony is not None else True,
        "enable_chorus_flanger": _strtobool(enable_chorus_flanger) if enable_chorus_flanger is not None else True,
        "enable_delay": _strtobool(enable_delay) if enable_delay is not None else True,
        "enable_gate": _strtobool(enable_gate) if enable_gate is not None else True,
        "enable_doubler": _strtobool(enable_doubler) if enable_doubler is not None else True,
        "enable_exciter": _strtobool(enable_exciter) if enable_exciter is not None else True,
        "enable_tape": _strtobool(enable_tape) if enable_tape is not None else True,
        "enable_parallel_comp": _strtobool(enable_parallel_comp) if enable_parallel_comp is not None else True,
        "enable_ms_eq": _strtobool(enable_ms_eq) if enable_ms_eq is not None else True,
        "stems_mode": True,  # Always use stems extraction
    }

    # Attach AI-generated DSP overrides if provided
    if ai_dsp_config:
        try:
            parsed = json.loads(ai_dsp_config)
            options["ai_dsp_config"] = validate_dsp_config(parsed)
        except (json.JSONDecodeError, Exception):
            pass  # Ignore malformed payload; proceed with normal DSP

    if selected_layers:
        try:
            parsed_layers = json.loads(selected_layers)
            if isinstance(parsed_layers, list):
                options["selected_layers"] = [str(x) for x in parsed_layers]
        except Exception:
            pass

    job_id = enqueue_job(ref_path, dry_path, options)
    return {"job_id": job_id, "status": "queued"}


@app.post("/analyze-layers")
async def analyze_layers(
    reference: UploadFile | None = File(None),
    reference_url: str | None = Form(None),
    dry: UploadFile = File(...),
    segment_start_s: float | None = Form(None),
    segment_end_s: float | None = Form(None),
):
    if not reference and not reference_url:
        raise HTTPException(status_code=400, detail="Provide reference file or URL")
    if dry.content_type and "audio" not in dry.content_type:
        raise HTTPException(status_code=415, detail="Unsupported dry file type")
    if reference and reference.content_type and "audio" not in reference.content_type:
        raise HTTPException(status_code=415, detail="Unsupported reference file type")

    tmp_dir = settings.inputs_dir
    if reference:
        ref_path = _save_upload(tmp_dir, reference)
    else:
        ref_path = fetch_audio_from_url(reference_url)  # type: ignore[arg-type]
    dry_path = _save_upload(tmp_dir, dry)

    try:
        validate_file(ref_path)
        validate_file(dry_path)

        ref_norm = settings.inputs_dir / f"{uuid.uuid4()}_ref_layer.wav"
        dry_norm = settings.inputs_dir / f"{uuid.uuid4()}_dry_layer.wav"
        # Stereo for the reference: layer detection reads doubling from the
        # L/R image, which a mono downmix erases (the "only lead" bug).
        run_ffmpeg_normalize(ref_path, ref_norm, channels=2)
        run_ffmpeg_normalize(dry_path, dry_norm)

        ref_for_separation = ref_norm
        # Optional: analyze only a selected segment to speed up extraction.
        if segment_start_s is not None or segment_end_s is not None:
            import librosa
            import numpy as np
            y_seg, sr_seg = load_wav_stereo(ref_norm)
            total_dur = y_seg.shape[1] / sr_seg
            start = float(segment_start_s or 0.0)
            end = float(segment_end_s if segment_end_s is not None else total_dur)
            if start < 0:
                raise HTTPException(status_code=400, detail="segment_start_s must be >= 0")
            if end <= start:
                raise HTTPException(status_code=400, detail="segment_end_s must be greater than segment_start_s")
            if start >= total_dur:
                raise HTTPException(status_code=400, detail="segment_start_s exceeds reference duration")
            end = min(end, total_dur)

            max_segment_s = 60.0
            if (end - start) > max_segment_s:
                end = start + max_segment_s

            s0 = int(start * sr_seg)
            s1 = int(end * sr_seg)
            seg = y_seg[:, s0:s1]
            if seg.shape[1] < int(1.0 * sr_seg):
                raise HTTPException(status_code=400, detail="Selected segment is too short (min 1 second)")

            ref_segment_path = settings.inputs_dir / f"{uuid.uuid4()}_ref_segment.wav"
            save_wav(ref_segment_path, seg.astype(np.float32), sr_seg)
            ref_for_separation = ref_segment_path

        ref_vocals_path = settings.inputs_dir / f"{uuid.uuid4()}_ref_vocals_layer.wav"
        extracted = extract_vocals(ref_for_separation, ref_vocals_path, force_demucs=True)
        ref_for_analysis = extracted if extracted and extracted.exists() else ref_for_separation

        ref_audio, sr = load_wav_stereo(ref_for_analysis)
        dry_audio, dry_sr = load_wav(dry_norm)
        if sr != dry_sr:
            import librosa
            dry_audio = librosa.resample(dry_audio, orig_sr=dry_sr, target_sr=sr)

        profile = detect_vocal_layers(ref_audio, sr)
        if not profile:
            analysis_id = str(uuid.uuid4())
            lead_name = f"{analysis_id}_lead.wav"
            lead_path = settings.outputs_dir / lead_name
            save_wav(lead_path, ref_audio, sr)
            return {
                "detected": False,
                "total_layers": 1,
                "layers": [{
                    "id": "lead",
                    "label": "Lead Vocal (Extracted)",
                    "type": "lead",
                    "preview_url": f"/outputs/{lead_name}",
                    "selected": True,
                }],
                "message": "No extra vocal layers detected; previewing extracted lead vocal.",
            }

        # Build audition stems from the analyzed reference vocal so users
        # hear reference-derived layering characteristics directly.
        stems = build_vocal_layer_stems(ref_audio, sr, profile)
        analysis_id = str(uuid.uuid4())
        layers = []
        for idx, key in enumerate(stems.keys()):
            preview_name = f"{analysis_id}_{key}.wav"
            preview_path = settings.outputs_dir / preview_name
            save_wav(preview_path, stems[key], sr)
            if key == "lead":
                label = "Lead Vocal"
                layer_type = "lead"
            elif key.startswith("doubler_"):
                label = f"Doubler {key.split('_')[1]}"
                layer_type = "doubler"
            else:
                label = f"Harmony {key.split('_')[1]}"
                layer_type = "harmony"
            layers.append(
                {
                    "id": key,
                    "label": label,
                    "type": layer_type,
                    "preview_url": f"/outputs/{preview_name}",
                    "selected": True,
                }
            )

        return {
            "detected": True,
            "analysis_id": analysis_id,
            "total_layers": profile.total_layers,
            "n_doublers": profile.n_doublers,
            "n_harmonies": len(profile.harmony_intervals),
            "layers": layers,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer analysis failed: {str(e)}")
    finally:
        try:
            ref_path.unlink(missing_ok=True)
            dry_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    r = redis.from_url(settings.redis_url)
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        job = None

    recipe_path = settings.outputs_dir / f"{job_id}.json"
    recipe_url = f"/recipes/{job_id}.json" if recipe_path.exists() else None

    if job:
        if job.is_finished:
            out_path = settings.outputs_dir / f"{job_id}.wav"
            if out_path.exists():
                return JobStatus(
                    job_id=job_id,
                    status="finished",
                    download_url=f"/outputs/{job_id}.wav",
                    recipe_url=recipe_url,
                )
        if job.is_failed:
            return JobStatus(
                job_id=job_id,
                status="failed",
                download_url=None,
                recipe_url=recipe_url,
                error=str(job.exc_info) if job.exc_info else "Job failed",
            )
        return JobStatus(job_id=job_id, status=job.get_status(), download_url=None, recipe_url=recipe_url)

    # Fallback to file existence
    out_path = settings.outputs_dir / f"{job_id}.wav"
    if out_path.exists():
        return JobStatus(
            job_id=job_id, status="finished", download_url=f"/outputs/{job_id}.wav", recipe_url=recipe_url
        )
    return JobStatus(job_id=job_id, status="unknown", download_url=None, recipe_url=recipe_url)


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """SSE endpoint — pushes progress events until the job finishes or fails."""
    async def event_generator():
        r = redis.from_url(settings.redis_url)
        last_pct = -1
        while True:
            try:
                job = Job.fetch(job_id, connection=r)
                meta = job.meta or {}
                pct   = int(meta.get("progress", 0))
                stage = str(meta.get("stage", "Queued…"))
                status = str(job.get_status())

                # Always push on status transitions; throttle identical pct repeats
                if pct != last_pct or status in ("finished", "failed"):
                    last_pct = pct
                    payload = json.dumps({"status": status, "progress": pct, "stage": stage})
                    yield f"data: {payload}\n\n"

                if job.is_finished or job.is_failed:
                    break
            except Exception:
                payload = json.dumps({"status": "queued", "progress": 0, "stage": "Waiting for worker…"})
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/outputs/{filename}")
async def download_output(filename: str):
    path = settings.outputs_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/recipes/{filename}")
async def download_recipe(filename: str):
    path = settings.outputs_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="application/json")


@app.get("/healthz")
async def health():
    return JSONResponse({"ok": True})


@app.get("/health/models")
async def health_models():
    """
    Health check endpoint for model loading status.
    
    Returns:
        JSON with model availability status
    """
    try:
        from processor.ml_refine.rvc_refiner import get_rvc_refiner
        from pathlib import Path
        
        refiner = get_rvc_refiner(enable_gpu=False)
        
        # Check file existence
        hubert_file_exists = refiner.hubert_path.exists()
        rvc_file_exists = refiner.model_path.exists()
        vocoder_file_exists = refiner.vocoder_path.exists()
        
        # Try to load models (lazy loading)
        models_loaded = refiner.load_models()
        
        # Check individual components
        hubert_available = refiner.hubert_model is not None
        rvc_available = refiner.rvc_model is not None
        vocoder_available = refiner.vocoder is not None
        
        device = str(refiner.device)
        
        # Determine status message
        if models_loaded and rvc_available:
            msg = "All models available and loaded"
        elif rvc_file_exists:
            msg = "RVC model file exists but may need architecture adaptation"
        elif hubert_available:
            msg = "HuBERT available, RVC model file missing"
        else:
            msg = "Some models not available (check logs)"
        
        return JSONResponse({
            "hubert": hubert_available,
            "hubert_file": hubert_file_exists,
            "rvc_model": rvc_available,
            "rvc_file": rvc_file_exists,
            "vocoder": vocoder_available,
            "vocoder_file": vocoder_file_exists,
            "device": device,
            "models_loaded": models_loaded,
            "msg": msg,
            "paths": {
                "hubert": str(refiner.hubert_path),
                "rvc": str(refiner.model_path),
                "vocoder": str(refiner.vocoder_path),
            },
        })
    except Exception as e:
        import traceback
        return JSONResponse({
            "hubert": False,
            "hubert_file": False,
            "rvc_model": False,
            "rvc_file": False,
            "vocoder": False,
            "vocoder_file": False,
            "device": "unknown",
            "models_loaded": False,
            "msg": f"Error checking models: {str(e)}",
            "error": str(e),
        }, status_code=500)


# ---------------------------------------------------------------------------
# AI Audio endpoint models
# ---------------------------------------------------------------------------
class AiAudioRequest(BaseModel):
    prompt: str = ""
    audio_features: Optional[dict[str, Any]] = None
    previous_config: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    mode: str = "prompt"  # "prompt" | "preset" | "reference_match"
    style_hint: Optional[str] = None
    reference_features: Optional[dict[str, Any]] = None
    current_features: Optional[dict[str, Any]] = None


class AiAudioResponse(BaseModel):
    config: dict[str, Any]
    session_id: Optional[str] = None
    source: str  # "gpt" | "fallback"


@app.post("/api/ai-audio", response_model=AiAudioResponse)
async def ai_audio(req: AiAudioRequest):
    """
    Convert a natural language prompt (or audio features) into a DSP config.

    Modes:
      - prompt          → translate req.prompt into a DSP config
      - preset          → generate a preset from audio features
      - reference_match → generate DSP to match reference track characteristics

    The endpoint always returns a valid config; if GPT fails it falls back
    to safe defaults and sets source="fallback".
    """
    import os
    has_key = bool(os.environ.get("OPENAI_API_KEY", ""))

    try:
        if req.mode == "reference_match":
            if not req.reference_features:
                raise HTTPException(
                    status_code=400,
                    detail="reference_features required for reference_match mode",
                )
            config = reference_match_config(
                reference_features=req.reference_features,
                current_features=req.current_features,
                session_id=req.session_id,
            )
        elif req.mode == "preset":
            config = generate_preset_from_features(
                features=req.audio_features or {},
                style_hint=req.style_hint or "optimal vocal processing",
                session_id=req.session_id,
            )
        else:  # "prompt" (default)
            if not req.prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            config = prompt_to_dsp_config(
                prompt=req.prompt,
                audio_features=req.audio_features,
                previous_config=req.previous_config,
                session_id=req.session_id,
            )

        groq_key = bool(os.environ.get("GROQ_API_KEY", ""))
        source = "groq" if groq_key else ("gpt" if has_key else "fallback")
        return AiAudioResponse(
            config=config,
            session_id=req.session_id,
            source=source,
        )

    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("AI audio error: %s", exc)
        fallback = validate_dsp_config(_DEFAULTS)
        return AiAudioResponse(
            config=fallback,
            session_id=req.session_id,
            source="fallback",
        )


@app.delete("/api/ai-audio/session/{session_id}")
async def clear_ai_session(session_id: str):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"cleared": session_id}


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/emulator.html")
async def emulator_page():
    if settings.frontend_dir.exists():
        emu = settings.frontend_dir / "emulator.html"
        if emu.exists():
            return FileResponse(emu)
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/")
async def root():
    if settings.frontend_dir.exists():
        index = settings.frontend_dir / "index.html"
        if index.exists():
            return FileResponse(index)
    return {"message": "API is running. Visit /ui for the web UI."}

