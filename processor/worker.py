import json
import logging
import os
import sys
import uuid
from pathlib import Path

import redis
from rq import Connection, Queue, Worker, SimpleWorker, get_current_job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from processor.analysis.style_extractor import Recipe, analyze_reference
from processor.analysis.segmenter import detect_phrases
from processor.config import settings
from processor.cleanup import sweep_old_files
from processor.dsp.chain import apply_chain
from processor.utils.audio_io import load_wav, load_wav_stereo, run_ffmpeg_normalize, save_wav
from processor.utils.vocal_extraction import extract_vocals
from processor.utils.metrics import timer, compute_spectral_distance, compute_lufs
from processor.dsp.analysis.chorus_analysis import detect_chorus
from processor.dsp.analysis.flanger_analysis import detect_flanger
from processor.dsp.analysis.vocal_layers_analysis import detect_vocal_layers
from processor.dsp.effects.apply_vocal_layers import apply_vocal_layers, filter_vocal_layers_profile
from processor.dsp.delay import detect_delay, apply_delay
from processor.dsp.analysis.reverb_analysis import estimate_reverb_params
from processor.dsp.analysis.gate_analysis import detect_gate
from processor.dsp.analysis.doubler_analysis import detect_doubler
from processor.dsp.analysis.exciter_analysis import detect_exciter
from processor.dsp.analysis.tape_analysis import detect_tape
from processor.dsp.analysis.parallel_comp_analysis import detect_parallel_comp
from processor.dsp.analysis.ms_eq_analysis import detect_ms_eq
from processor.dsp.analysis.autotune_analysis import detect_autotune
import librosa
import numpy as np
import time


def _progress(job, pct: int, stage: str) -> None:
    """Write live progress into RQ job meta so the SSE stream can read it."""
    if job:
        job.meta["progress"] = pct
        job.meta["stage"] = stage
        job.save_meta()
    logger.info(f"[PROGRESS {pct}%] {stage}")


def process_job(reference_path: Path, dry_path: Path, options: dict | None = None) -> Path:
    options = options or {}
    if options.get("preset_recipe"):
        return process_preset_job(dry_path, options["preset_recipe"], options)
    
    # Initialize metrics
    metrics = {
        "processing_time_total": 0.0,
        "processing_time_dsp": 0.0,
        "spectral_distance": None,
        "lufs_reference": None,
        "lufs_output": None,
    }
    
    job_start_time = time.time()
    current_job = get_current_job()
    job_id = current_job.id if current_job else str(uuid.uuid4())
    
    logger.info(f"[JOB {job_id}] Starting job processing")
    _progress(current_job, 2, "Normalising audio…")
    
    ref_norm = settings.inputs_dir / f"{uuid.uuid4()}_ref.wav"
    dry_norm = settings.inputs_dir / f"{uuid.uuid4()}_dry.wav"
    # Stereo reference: layer detection reads doubling from the L/R image,
    # and matching the analyze-layers bytes means the separation cache hits
    # (the reference was usually just separated by /analyze-layers).
    run_ffmpeg_normalize(reference_path, ref_norm, channels=2)
    run_ffmpeg_normalize(dry_path, dry_norm)
    _progress(current_job, 5, "Audio normalised")

    # Extract vocals from reference track (in case it's a full mix)
    use_stems = options.get("stems_mode", True)
    ref_for_analysis = ref_norm

    # "Analyze layers" already separated this reference — reuse that result
    # rather than paying for separation twice.
    reused = options.get("ref_vocals_path")
    if reused and Path(reused).exists():
        ref_for_analysis = Path(reused)
        print("✓ Reusing the vocal separated during layer analysis — skipping extraction")
        _progress(current_job, 20, "Reusing analysed vocal")
    elif use_stems:
        _progress(current_job, 8, "Extracting vocals from reference…")
        ref_vocals_path = settings.inputs_dir / f"{uuid.uuid4()}_ref_vocals.wav"
        print(f"Attempting to extract vocals from reference: {reference_path.name}")

        def _sep_progress(done: int, total: int) -> None:
            # Separation owns 8-20% of the job bar.
            _progress(current_job, 8 + int(12 * done / max(total, 1)),
                      f"Separating vocals… {int(100 * done / max(total, 1))}%")

        extracted = extract_vocals(ref_norm, ref_vocals_path, progress_cb=_sep_progress)

        if extracted and extracted.exists():
            ref_for_analysis = extracted
            print(f"✓ Successfully extracted vocals from reference track")
        else:
            print(f"⚠ Vocal extraction failed — analysing full mix instead")
        _progress(current_job, 20, "Vocals extracted")
    
    # Load reference audio — mono for the recipe/effect detectors, true
    # stereo for layer detection (doubling lives in L/R decorrelation).
    ref_audio, sr = load_wav(ref_for_analysis)
    ref_stereo, _ = load_wav_stereo(ref_for_analysis)

    # The reference is analysis-only; cap it so a 4-minute song doesn't cost
    # 4 minutes of detector time. (Separation above ran on the full file so
    # its cache stays shared with /analyze-layers.)
    _max_n = int(settings.analysis_max_seconds * sr)
    if len(ref_audio) > _max_n:
        logger.info(f"[JOB {job_id}] Capping reference analysis at {settings.analysis_max_seconds}s")
        ref_audio = ref_audio[:_max_n]
        ref_stereo = ref_stereo[:, :_max_n]
    ref_duration = len(ref_audio) / sr
    # Stereo field of the reference vocal — matched onto the output later.
    try:
        from processor.dsp.stereo import measure_stereo_profile
        _ref_stereo_profile = measure_stereo_profile(ref_stereo, sr) or None
        if _ref_stereo_profile:
            logger.info(f"[JOB {job_id}] Ref stereo: "
                        + ", ".join(f"{k}={v:.1f}" for k, v in _ref_stereo_profile.items()))
    except Exception:
        _ref_stereo_profile = None
    logger.info(f"[JOB {job_id}] Reference loaded: {ref_duration:.2f}s @ {sr}Hz")
    print(f"  Reference length: {ref_duration:.2f}s, sample rate: {sr}Hz")
    
    # Compute reference LUFS
    try:
        metrics["lufs_reference"] = compute_lufs(ref_for_analysis)
        logger.info(f"[JOB {job_id}] Reference LUFS: {metrics['lufs_reference']:.1f}")
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Failed to compute reference LUFS: {e}")
    
    # Analyze reference to get recipe
    _progress(current_job, 22, "Analysing reference track…")
    dry_audio_temp, _ = load_wav(dry_norm)
    with timer("recipe_analysis"):
        recipe: Recipe = analyze_reference(ref_audio, sr, dry_y=dry_audio_temp)
    logger.info(f"[JOB {job_id}] Recipe extracted")
    _progress(current_job, 35, "Reference analysed")
    
    # Save recipe early for debugging (will update with metrics later)
    if current_job:
        recipe_path = settings.outputs_dir / f"{current_job.id}.json"
    
    # Load the dry vocal (RVC voice conversion was removed — it only ever had
    # a placeholder model; the DSP chain below does the style transfer).
    dry_audio, _ = load_wav(dry_norm)

    # Detect phrases for adaptive processing (optional)
    segments = None
    if options.get("adaptive_dsp", False):
        segments = detect_phrases(dry_audio, sr=sr)
        print(f"  Detected {len(segments)} phrases for adaptive processing")
    
    # Detect all effects from reference in parallel
    _progress(current_job, 38, "Detecting effects…")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ref_mono = ref_audio if ref_audio.ndim == 1 else np.mean(ref_audio, axis=0)

    def _detect_chorus():
        return detect_chorus(ref_audio, sr)

    def _detect_flanger():
        return detect_flanger(ref_audio, sr)

    def _detect_reverb():
        return estimate_reverb_params(ref_audio, sr)

    def _detect_delay():
        # dry rides along for rhythm cancellation: shared rhythm correlates
        # in both, only the reference has the echo effect
        return detect_delay(ref_mono, sr, dry=dry_audio)

    def _detect_gate():
        return detect_gate(ref_audio, sr)

    def _detect_doubler():
        return detect_doubler(ref_audio, sr)

    def _detect_exciter():
        return detect_exciter(ref_audio, sr)

    def _detect_tape():
        return detect_tape(ref_audio, sr)

    def _detect_parallel_comp():
        return detect_parallel_comp(ref_audio, sr)

    def _detect_ms_eq():
        return detect_ms_eq(ref_audio, sr)

    def _detect_vocal_layers():
        return detect_vocal_layers(ref_stereo, sr)

    chorus_profile = flanger_profile = reverb_profile_auto = delay_info = None
    gate_settings = doubler_settings = exciter_settings = tape_settings = None
    parallel_comp_settings = ms_eq_settings = autotune_settings = None
    vocal_layers_profile = None

    detectors = {
        "reverb":   _detect_reverb,   # always run — used by apply_chain
    }
    if options.get("enable_chorus_flanger", True):
        detectors["chorus"]  = _detect_chorus
        detectors["flanger"] = _detect_flanger
    # Vocal layers replaces the old separate harmony detector
    if options.get("enable_harmony", True):
        detectors["vocal_layers"] = _detect_vocal_layers
    if options.get("enable_delay", True):
        detectors["delay"]   = _detect_delay
    if options.get("enable_gate", True):
        detectors["gate"]    = _detect_gate
    if options.get("enable_doubler", True):
        detectors["doubler"] = _detect_doubler
    if options.get("enable_exciter", True):
        detectors["exciter"] = _detect_exciter
    if options.get("enable_tape", True):
        detectors["tape"]    = _detect_tape
    if options.get("enable_parallel_comp", True):
        detectors["parallel_comp"] = _detect_parallel_comp
    if options.get("enable_ms_eq", True):
        detectors["ms_eq"]   = _detect_ms_eq

    _DETECTOR_TIMEOUT = 90  # seconds per detector

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in detectors.items()}

        # Collect results into a dict to avoid closure issues
        _det_results: dict = {}
        def _run_future(future, name):
            try:
                _det_results[name] = future.result(timeout=_DETECTOR_TIMEOUT)
            except Exception as e:
                logger.warning(f"[JOB {job_id}] {name} detection failed or timed out: {e}")
                _det_results[name] = None

        try:
            for future in as_completed(futures, timeout=_DETECTOR_TIMEOUT * len(detectors)):
                name = futures[future]
                _run_future(future, name)
        except TimeoutError:
            logger.warning(f"[JOB {job_id}] Overall detection timed out — collecting partial results")
            for future, name in futures.items():
                if name not in _det_results:
                    _run_future(future, name)

        # Assign results from dict
        if _det_results.get("chorus"):
            chorus_profile = _det_results["chorus"]
            # Guard against false positives: a real chorus LFO sits around
            # 0.5-5 Hz. Sub-0.5 Hz "detections" are just slow level drift in
            # the reference, and applying them audibly smears the vocal.
            if chorus_profile.rate_hz < 0.5 or chorus_profile.mix <= 0.02:
                logger.info(
                    f"[JOB {job_id}] Chorus ignored (rate={chorus_profile.rate_hz:.2f}Hz "
                    f"mix={chorus_profile.mix:.2f} — likely false positive)"
                )
                chorus_profile = None
            else:
                logger.info(f"[JOB {job_id}] Chorus: rate={chorus_profile.rate_hz:.2f}Hz mix={chorus_profile.mix:.2f}")
        if _det_results.get("flanger"):
            flanger_profile = _det_results["flanger"]
            logger.info(f"[JOB {job_id}] Flanger: rate={flanger_profile.rate_hz:.2f}Hz feedback={flanger_profile.feedback:.2f} mix={flanger_profile.mix:.2f}")
        if "vocal_layers" in _det_results:
            vocal_layers_profile = _det_results["vocal_layers"]
            if vocal_layers_profile:
                logger.info(
                    f"[JOB {job_id}] Vocal layers: {vocal_layers_profile.total_layers} total "
                    f"({vocal_layers_profile.n_doublers} doublers + {len(vocal_layers_profile.harmony_intervals)} harmony voices)"
                )
            else:
                logger.info(f"[JOB {job_id}] Vocal layers: single voice, no layering detected")
        if _det_results.get("reverb"):
            reverb_profile_auto = _det_results["reverb"]
            logger.info(f"[JOB {job_id}] Reverb: rt60={reverb_profile_auto.rt60:.2f}s wet={reverb_profile_auto.wet:.2f}")
        if _det_results.get("delay"):
            delay_info = _det_results["delay"]
            logger.info(f"[JOB {job_id}] Delay: {delay_info.get('type')} {delay_info.get('delay_ms', 0):.1f}ms")
        if _det_results.get("gate"):
            gate_settings = _det_results["gate"]
            logger.info(f"[JOB {job_id}] Gate: threshold={gate_settings.threshold_db:.1f}dB")
        if _det_results.get("doubler"):
            doubler_settings = _det_results["doubler"]
            logger.info(f"[JOB {job_id}] Doubler: mix={doubler_settings.mix:.2f}")
        if _det_results.get("exciter"):
            exciter_settings = _det_results["exciter"]
            logger.info(f"[JOB {job_id}] Exciter: drive={exciter_settings.drive:.2f} mix={exciter_settings.mix:.2f}")
        if _det_results.get("tape"):
            tape_settings = _det_results["tape"]
            logger.info(f"[JOB {job_id}] Tape: drive={tape_settings.drive:.2f} rolloff={tape_settings.hf_rolloff_hz:.0f}Hz")
        if _det_results.get("parallel_comp"):
            parallel_comp_settings = _det_results["parallel_comp"]
            logger.info(f"[JOB {job_id}] Parallel comp: blend={parallel_comp_settings.blend:.2f}")
        if _det_results.get("ms_eq"):
            ms_eq_settings = _det_results["ms_eq"]
            logger.info(f"[JOB {job_id}] M-S EQ: mid_bands={len(ms_eq_settings.mid_bands)} side_bands={len(ms_eq_settings.side_bands)}")

    _progress(current_job, 48, "Effects detected")

    # ── Spectral profiler (used by both AI review and refinement pass) ───────
    _dry_mono = dry_audio if dry_audio.ndim == 1 else dry_audio.mean(axis=0)
    _ref_mono2 = ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0)
    _dry_stats: dict = {}
    _ref_stats: dict = {}

    def _spectral_profile(y: np.ndarray, label: str) -> dict:
        """Compute a rich spectral + dynamics profile for AI comparison."""
        rms = float(np.sqrt(np.mean(y ** 2)))
        peak = float(np.max(np.abs(y)))
        crest_db = float(20 * np.log10(peak / (rms + 1e-9) + 1e-9))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        band_defs = [
            ("sub_60hz", 60), ("low_250hz", 250), ("low_mid_500hz", 500),
            ("mid_2khz", 2000), ("high_mid_6khz", 6000), ("air_12khz", 12000),
        ]
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        total_energy = float(np.sum(stft ** 2)) + 1e-12
        band_energy: dict = {}
        prev_f = 0
        for bname, cutoff in band_defs:
            mask = (freqs >= prev_f) & (freqs < cutoff)
            band_energy[bname] = round(float(np.sum(stft[mask] ** 2)) / total_energy, 4)
            prev_f = cutoff
        frame_rms = librosa.feature.rms(y=y)[0]
        frame_db = 20 * np.log10(frame_rms + 1e-9)
        dynamic_range_db = float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10))
        n = len(y)
        seg = max(1, n // 5)
        tail_ratio = round(float(np.sqrt(np.mean(y[-seg:] ** 2))) / (float(np.sqrt(np.mean(y[:seg] ** 2))) + 1e-9), 4)
        return {
            "rms": round(rms, 6),
            "peak": round(peak, 4),
            "crest_factor_db": round(crest_db, 1),
            "dynamic_range_db": round(dynamic_range_db, 1),
            "spectral_centroid_hz": round(centroid, 0),
            "spectral_rolloff_hz": round(rolloff, 0),
            "spectral_flatness": round(flatness, 5),
            "zero_crossing_rate": round(zcr, 4),
            "band_energy_pct": band_energy,
            "reverb_tail_ratio": tail_ratio,
        }

    # Compute profiles (used by AI review AND refinement pass)
    try:
        _dry_stats = _spectral_profile(_dry_mono, "dry")
        _dry_stats["lufs"] = metrics.get("lufs_reference")
        _dry_stats["duration_s"] = round(float(len(_dry_mono) / sr), 2)
        _ref_stats = _spectral_profile(_ref_mono2, "reference")
        _ref_stats["lufs"] = metrics.get("lufs_reference")
        _band_gap = {k: round(_ref_stats["band_energy_pct"].get(k, 0) - _dry_stats["band_energy_pct"].get(k, 0), 4)
                     for k in _dry_stats["band_energy_pct"]}
        _dry_stats["band_gap_vs_reference"] = _band_gap
        logger.info(
            f"[JOB {job_id}] Dry vocal: centroid={_dry_stats['spectral_centroid_hz']:.0f}Hz "
            f"crest={_dry_stats['crest_factor_db']:.1f}dB dyn={_dry_stats['dynamic_range_db']:.1f}dB"
        )
        logger.info(
            f"[JOB {job_id}] Ref vocal: centroid={_ref_stats['spectral_centroid_hz']:.0f}Hz "
            f"crest={_ref_stats['crest_factor_db']:.1f}dB dyn={_ref_stats['dynamic_range_db']:.1f}dB"
        )
    except Exception as _prof_err:
        logger.warning(f"[JOB {job_id}] Spectral profiling failed: {_prof_err}")

    # ── Build the final recipe: exactly what was detected in the reference ──
    # No AI gating, no fallback injections — if the reference doesn't have an
    # effect, it is not applied. This keeps the output faithful to the
    # selected reference layer's actual sound.

    # EQ / compression / saturation come from the style-match recipe
    eq_bands_final = recipe.eq if recipe.eq else None
    comp_final = recipe.compressor
    saturation_final = recipe.saturation_drive

    # Reverb: prefer the dedicated reverb analysis; fall back to the recipe.
    # The wet cap scales with how reverberant the reference actually is
    # (tail_ratio ≈ energy at phrase tails vs starts) — a washed-out
    # reference deserves more than a token 0.2 mix.
    if reverb_profile_auto is not None:
        from processor.dsp.reverb import ReverbSettings as _RS
        # Continuous scaling with the reference's measured reverbiness —
        # no hard branch, so any reference lands on a sensible wet level:
        #   dry ref (tail 0.2) → cap 0.20, no boost
        #   wet ref  (tail 0.7) → cap ~0.35, wet boosted up to 1.6x
        _tail = float(np.clip(float(_ref_stats.get("reverb_tail_ratio", 0.0) or 0.0), 0.0, 1.0))
        _wetness = float(np.clip((_tail - 0.3) / 0.4, 0.0, 1.0))  # 0 below 0.3, 1 above 0.7
        _wet_cap = 0.20 + 0.15 * _wetness
        _rev = _RS(
            decay_s=reverb_profile_auto.rt60,
            mix=float(np.clip(reverb_profile_auto.wet * (1.0 + 0.6 * _wetness), 0.0, _wet_cap)),
            pre_delay_ms=reverb_profile_auto.predelay_ms,
        )
        reverb_final = _rev if _rev.mix >= 0.07 else None
    else:
        reverb_final = recipe.reverb

    # Delay: as detected. Repeat level and feedback are measured from the
    # reference's echo itself (echo_level ≈ first-repeat amplitude relative
    # to the voice; feedback ≈ repeat-to-repeat survival).
    if delay_info is not None:
        delay_info = dict(delay_info)
        _lvl = delay_info.get("echo_level")
        if _lvl is None:  # older recipe payloads
            _conf = float(delay_info.get("confidence", 0.5) or 0.5)
            _lvl = float(np.clip(0.15 + 0.35 * _conf, 0.15, 0.45))
        delay_info.setdefault("_mix", float(np.clip(_lvl, 0.10, 0.5)))

    # Width: as detected
    _width_val = recipe.width if (recipe.width and options.get("enable_width", True)) else None

    # ── Bounded AI fine-tune: scale DETECTED effects only (±40%) ────────────
    # The LLM sees measured profiles + what was detected and may nudge
    # amounts; it can never add or remove an effect. Fail-safe: {} = no-op.
    ai_scales: dict = {}
    try:
        from api.ai_audio import ai_effect_scales
        _detected_brief = {
            "reverb": ({"rt60": reverb_final.decay_s, "mix": reverb_final.mix}
                       if reverb_final else None),
            "delay": ({"ms": delay_info.get("delay_ms"), "level": delay_info.get("echo_level")}
                      if delay_info else None),
            "tape": ({"drive": tape_settings.drive, "mix": tape_settings.mix}
                     if tape_settings else None),
            "parallel_comp": ({"blend": parallel_comp_settings.blend}
                              if parallel_comp_settings else None),
            "saturation_drive": saturation_final,
            "vocal_layers": ({"total": vocal_layers_profile.total_layers}
                             if vocal_layers_profile else None),
            "compressor": ({"ratio": comp_final.ratio} if comp_final else None),
        }
        ai_scales = ai_effect_scales(_detected_brief, _ref_stats, _dry_stats)
        if ai_scales:
            logger.info(f"[JOB {job_id}] AI effect scales: {ai_scales}")
            if reverb_final and "reverb_mix_scale" in ai_scales:
                reverb_final.mix = float(np.clip(reverb_final.mix * ai_scales["reverb_mix_scale"], 0.0, 0.4))
            if delay_info and "delay_mix_scale" in ai_scales:
                delay_info["_mix"] = float(np.clip(delay_info["_mix"] * ai_scales["delay_mix_scale"], 0.05, 0.55))
            if tape_settings and "tape_mix_scale" in ai_scales:
                tape_settings.mix = float(np.clip(tape_settings.mix * ai_scales["tape_mix_scale"], 0.0, 0.6))
            if parallel_comp_settings and "parallel_blend_scale" in ai_scales:
                parallel_comp_settings.blend = float(np.clip(
                    parallel_comp_settings.blend * ai_scales["parallel_blend_scale"], 0.0, 0.6))
            if saturation_final and "saturation_scale" in ai_scales:
                saturation_final = float(np.clip(saturation_final * ai_scales["saturation_scale"], 1.0, 2.0))
            if vocal_layers_profile and "harmony_strength_scale" in ai_scales:
                vocal_layers_profile.harmony_strengths = [
                    float(np.clip(s * ai_scales["harmony_strength_scale"], 0.05, 0.5))
                    for s in vocal_layers_profile.harmony_strengths]
    except Exception as _ai_err:
        logger.info(f"[JOB {job_id}] AI fine-tune skipped: {_ai_err}")

    # User-requested AI DSP overrides (from the prompt-tuning box) still apply
    ai_cfg = options.get("ai_dsp_config")

    logger.info(
        f"[JOB {job_id}] Final recipe (as detected from reference): "
        f"EQ={'yes('+str(len(eq_bands_final))+' bands)' if eq_bands_final else 'SKIP'} "
        f"Comp={'yes' if comp_final else 'SKIP'} "
        f"Reverb={'yes(mix='+str(round(reverb_final.mix,2))+')' if reverb_final else 'SKIP'} "
        f"Sat={'yes('+str(round(saturation_final,2))+')' if saturation_final else 'SKIP'} "
        f"Width={'yes' if _width_val else 'SKIP'} "
        f"Chorus={'yes' if chorus_profile else 'SKIP'} "
        f"Delay={'yes' if delay_info else 'SKIP'} "
        f"Gate={'yes' if gate_settings else 'SKIP'} "
        f"Tape={'yes' if tape_settings else 'SKIP'} "
        f"Exciter={'yes' if exciter_settings else 'SKIP'} "
        f"ParallelComp={'yes' if parallel_comp_settings else 'SKIP'} "
        f"VocalLayers={'yes' if vocal_layers_profile else 'SKIP'}"
    )

    if ai_cfg:
        logger.info(f"[JOB {job_id}] Applying AI DSP config overrides")
        from processor.dsp.eq import EqBand
        from processor.dsp.compressor import CompressorSettings
        from processor.dsp.reverb import ReverbSettings

        ai_eq = ai_cfg.get("eq", {})
        ai_eq_bands = []
        band_map = [
            (120.0,   ai_eq.get("low_shelf_db",  0.0), 0.7),
            (350.0,   ai_eq.get("low_mid_db",    0.0), 0.9),
            (1000.0,  ai_eq.get("mid_db",         0.0), 1.0),
            (3500.0,  ai_eq.get("high_mid_db",   0.0), 1.0),
            (10000.0, ai_eq.get("high_shelf_db", 0.0), 0.7),
        ]
        for freq, gain_db, q in band_map:
            if abs(gain_db) >= 0.3:
                ai_eq_bands.append(EqBand(f=freq, gain_db=gain_db, q=q))
        # AI EQ always applied (user explicitly requested it)
        eq_bands_final = (eq_bands_final or []) + ai_eq_bands

        ai_comp = ai_cfg.get("compression", {})
        if ai_comp:
            base_comp = comp_final  # may be None
            comp_final = CompressorSettings(
                threshold_db=ai_comp.get("threshold_db", base_comp.threshold_db if base_comp else -24.0),
                ratio=ai_comp.get("ratio",       base_comp.ratio       if base_comp else 3.0),
                attack_ms=ai_comp.get("attack_ms",  base_comp.attack_ms  if base_comp else 10.0),
                release_ms=ai_comp.get("release_ms", base_comp.release_ms if base_comp else 100.0),
                makeup_db=ai_comp.get("makeup_db",  base_comp.makeup_db  if base_comp else 2.0),
            )

        ai_rev = ai_cfg.get("reverb", {})
        if ai_rev:
            base_reverb = reverb_final
            reverb_final = ReverbSettings(
                decay_s=ai_rev.get("decay_s", base_reverb.decay_s if base_reverb else 0.8),
                mix=ai_rev.get("mix", base_reverb.mix if base_reverb else 0.15),
                pre_delay_ms=ai_rev.get("pre_delay_ms", base_reverb.pre_delay_ms if base_reverb else 10.0),
            )

        ai_sat = ai_cfg.get("saturation", {})
        if ai_sat and "drive" in ai_sat:
            saturation_final = float(ai_sat["drive"])

    # Apply DSP chain with enhancements
    _progress(current_job, 52, "Applying DSP chain…")
    print(f"  Dry audio stats: max={np.max(np.abs(dry_audio)):.6f}, rms={np.sqrt(np.mean(dry_audio**2)):.6f}")
    dsp_start = time.time()
    processed = apply_chain(
        dry_audio,
        sr,
        eq_bands=eq_bands_final or [],        # pass empty list if None (no EQ)
        comp=comp_final,                       # None-safe in apply_chain
        reverb=reverb_final,                   # None-safe in apply_chain
        saturation_drive=saturation_final,     # None-safe in apply_chain
        width=_width_val,
        chorus_profile=chorus_profile,
        flanger_profile=flanger_profile,
        reference=ref_audio,
        enable_deesser=options.get("enable_deesser", True),
        enable_transient_shaper=options.get("enable_transient_shaper", False),
        enable_multiband=options.get("enable_multiband", False),
        segments=segments,
        gate=gate_settings,
        parallel_comp=parallel_comp_settings,
        tape=tape_settings,
        exciter=exciter_settings,
        doubler=doubler_settings,
        ms_eq=ms_eq_settings,
        autotune=autotune_settings,
    )
    metrics["processing_time_dsp"] = time.time() - dsp_start
    logger.info(f"[JOB {job_id}] DSP chain completed in {metrics['processing_time_dsp']:.2f}s")
    print(f"  DSP output stats: max={np.max(np.abs(processed)):.6f}, rms={np.sqrt(np.mean(processed**2)):.6f}")
    _progress(current_job, 65, "DSP chain applied")

    # ── Vocal layer replication (replaces old harmony) ────────────────────
    # Applies exactly the number of layers detected in the reference:
    # doublers (ADT stacks) + harmony voices, each with matching detune/pan/delay.
    if vocal_layers_profile is not None and options.get("enable_harmony", True):
        try:
            selected_layers = options.get("selected_layers") or []
            if isinstance(selected_layers, list) and selected_layers:
                filtered_profile = filter_vocal_layers_profile(vocal_layers_profile, selected_layers)
            else:
                filtered_profile = vocal_layers_profile

            if filtered_profile is None:
                logger.info(f"[JOB {job_id}] Vocal layers disabled by user selection")
                raise ValueError("No vocal layers selected")

            n = filtered_profile.total_layers
            logger.info(
                f"[JOB {job_id}] Applying {n} vocal layers "
                f"({filtered_profile.n_doublers} doublers + "
                f"{len(filtered_profile.harmony_intervals)} harmony voices)…"
            )
            # Stem export: save each generated layer separately so the mix
            # can be rebalanced in a DAW. Lead stem = the chain output.
            try:
                if current_job:
                    from processor.dsp.effects.apply_vocal_layers import build_vocal_layer_stems
                    _stems = build_vocal_layer_stems(processed, sr, filtered_profile)
                    _stem_urls = {}
                    for _sname, _sdata in _stems.items():
                        _sfile = f"{current_job.id}_stem_{_sname}.wav"
                        save_wav(settings.outputs_dir / _sfile, _sdata, sr)
                        _stem_urls[_sname] = f"/outputs/{_sfile}"
                    options["_stem_urls"] = _stem_urls
                    logger.info(f"[JOB {job_id}] Exported {len(_stem_urls)} layer stems")
            except Exception as _st_err:
                logger.warning(f"[JOB {job_id}] Stem export skipped: {_st_err}")
            processed = apply_vocal_layers(processed, sr, filtered_profile)
            logger.info(f"[JOB {job_id}] Vocal layers applied → output is now stereo")
        except ValueError:
            pass
        except Exception as e:
            logger.warning(f"[JOB {job_id}] Vocal layer replication failed: {e}")

    # Apply delay if detected and enabled — throw-style: send rides the dry
    # vocal's phrase boundaries instead of running constantly.
    if delay_info is not None and delay_info.get("delay_ms", 0) > 0 and options.get("enable_delay", True):
        try:
            fb = float(delay_info.get("feedback", 0.25))
            mix_val = float(delay_info.get("_mix", 0.25))
            send = None
            try:
                from processor.dsp.delay import phrase_send_envelope
                _phr = detect_phrases(dry_audio, sr=sr)
                if len(_phr) >= 3:  # enough structure to ride
                    send = phrase_send_envelope(len(processed), sr, _phr)
                    logger.info(f"[JOB {job_id}] Delay throws: riding {len(_phr)} phrases")
            except Exception:
                send = None
            if processed.ndim == 2:
                ch0 = apply_delay(processed[:, 0], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val, send_env=send)
                ch1 = apply_delay(processed[:, 1], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val, send_env=send)
                processed = np.stack([ch0, ch1], axis=1)
            else:
                processed = apply_delay(processed, sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val, send_env=send)
            logger.info(
                f"[JOB {job_id}] Delay applied: {delay_info.get('type')} {delay_info.get('delay_ms'):.2f} ms"
            )
        except Exception as e:
            logger.warning(f"[JOB {job_id}] Delay application failed: {e}")

    # NOTE: reverb is already applied inside apply_chain via reverb_final (converted from
    # reverb_profile_auto). A second apply_reverb here would double the reverb, so it is removed.

    # ── Stereo image: match the reference's width (per band) ────────────────
    # Mono renders become true stereo; layered renders keep their panning and
    # gain decorrelated width on top. Mono fold-down stays exactly the mid.
    try:
        from processor.dsp.stereo import apply_stereo_image
        _progress(current_job, 68, "Matching stereo image…")
        processed = apply_stereo_image(processed, sr, target=_ref_stereo_profile)
    except Exception as _st_err:
        logger.warning(f"[JOB {job_id}] Stereo image skipped: {_st_err}")

    # ── Dynamics profile transfer ───────────────────────────────────────────
    # Quantile-map the output's short-term loudness envelope onto the
    # reference's: the vocal then *rides* the way the reference rides
    # (timeline-free, so any-genre / any-song safe).
    try:
        from processor.dsp.dynamics_transfer import match_dynamics
        _progress(current_job, 70, "Matching dynamics profile…")
        _ref_mono_dyn = ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0)
        processed = match_dynamics(processed, sr, _ref_mono_dyn, strength=0.7)
    except Exception as _dt_err:
        logger.warning(f"[JOB {job_id}] Dynamics transfer skipped: {_dt_err}")

    # ── Spectral match pass (deterministic) ────────────────────────────────
    # Compare the processed output's band energies against the reference and
    # apply a gentle corrective EQ toward the reference. Replaces the old
    # LLM-based refinement: no API calls, same goal — output that sits in the
    # same tonal space as the reference layer.
    _progress(current_job, 72, "Spectral match pass…")
    refinement_info: dict = {}
    try:
        import tempfile as _tf
        _tmp_path = Path(_tf.mktemp(suffix=".wav"))
        save_wav(_tmp_path, processed, sr)
        _sd = float(compute_spectral_distance(ref_for_analysis, _tmp_path, sr=sr))
        _tmp_path.unlink(missing_ok=True)
        logger.info(f"[JOB {job_id}] Pre-match spectral distance: {_sd:.4f}")

        if _sd > 0.08:  # only correct if there's a meaningful gap
            _proc_mono = processed if processed.ndim == 1 else (
                processed.mean(axis=0) if processed.shape[0] < processed.shape[-1]
                else processed.mean(axis=1)
            )
            _proc_profile = _spectral_profile(_proc_mono, "output")
            _out_bands = _proc_profile["band_energy_pct"]
            _ref_bands = _ref_stats.get("band_energy_pct", {})

            from processor.dsp.eq import EqBand as _MatchEqBand, apply_eq as _apply_eq
            _gap_map = [
                ("sub_60hz",       60.0,   0.7),
                ("low_250hz",     200.0,   0.7),
                ("low_mid_500hz", 400.0,   0.9),
                ("mid_2khz",     1200.0,   1.0),
                ("high_mid_6khz", 4000.0,  1.0),
                ("air_12khz",    10000.0,  0.7),
            ]
            cb = []
            for band_name, freq, q in _gap_map:
                gap = float(_ref_bands.get(band_name, 0.0)) - float(_out_bands.get(band_name, 0.0))
                # Fractional energy gap to dB: ±0.05 gap ≈ ±2 dB, capped gently
                gain_db = float(np.clip(gap * 40.0, -4.0, 4.0))
                if abs(gain_db) >= 0.5:
                    cb.append(_MatchEqBand(f=freq, gain_db=gain_db, q=q))

            if cb:
                logger.info(f"[JOB {job_id}] Applying {len(cb)} spectral-match EQ bands")
                if processed.ndim == 2:
                    ch0 = _apply_eq(processed[:, 0], sr, cb)
                    ch1 = _apply_eq(processed[:, 1], sr, cb)
                    processed = np.stack([ch0, ch1], axis=1)
                else:
                    processed = _apply_eq(processed, sr, cb)
                processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
                refinement_info = {
                    "pre_distance": _sd,
                    "correction_bands": len(cb),
                    "summary": "Deterministic spectral match toward reference",
                }
            else:
                refinement_info = {"pre_distance": _sd, "correction_bands": 0,
                                   "summary": "Band energies already match reference"}
        else:
            logger.info(f"[JOB {job_id}] Spectral distance {_sd:.4f} < 0.08 — no match EQ needed")
            refinement_info = {"pre_distance": _sd, "correction_bands": 0,
                               "summary": "Already close to reference — no correction needed."}
    except Exception as _ref_err:
        logger.warning(f"[JOB {job_id}] Spectral match pass skipped: {_ref_err}")

    _progress(current_job, 88, "Finalising output…")
    # Safety checks before final normalization
    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.max(np.abs(processed))
    
    if max_val < 1e-6:  # Audio is essentially silent
        print(f"⚠ WARNING: Processed audio is too quiet (max={max_val:.2e}). Using original dry audio.")
        processed = dry_audio.copy()
        max_val = np.max(np.abs(processed))
    
    # ── RMS-match output loudness to reference ────────────────────────────
    # The reference was measured at a specific LUFS/RMS. After layering,
    # EQ and effects, our output may be significantly quieter or louder.
    # Match RMS so the output sits at the same perceived loudness as the reference.
    try:
        # Loudness is owned by the final LUFS match (after the master bus).
        # The old RMS match here pushed +10-12 dB into the limiter, which
        # crushed the loud sections' transients before the LUFS trim undid
        # the level — the "squashed, not like the reference" sound. Only
        # rescue genuinely starved signals so the glue detector has signal.
        out_rms = float(np.sqrt(np.mean(processed ** 2)))
        if 1e-9 < out_rms < 0.02:  # below ~-34 dBFS — abnormally quiet
            gain = float(min(0.05 / out_rms, 10 ** (12 / 20)))
            processed = processed * gain
            print(f"  Pre-bus rescue gain: {20 * np.log10(gain):.1f}dB (chain output was starved)")
    except Exception as e:
        print(f"⚠ Pre-bus level check failed: {e}")

    # Master bus: glue compression + lookahead limiting — the "finished
    # record" density and level a bare effects chain lacks. Replaces the old
    # bare peak-scale (which undid the loudness match whenever peaks hit).
    max_val = np.max(np.abs(processed))
    if max_val < 1e-9:
        processed = dry_audio / (np.max(np.abs(dry_audio)) + 1e-9) * 0.95
    else:
        from processor.dsp.master_bus import apply_master_bus, _active_spread_db
        _progress(current_job, 90, "Master bus: glue + limiting…")
        # Density target = the reference's own measured dynamic spread, so a
        # crushed-dense reference yields a matching-dense output.
        _ref_spread = _active_spread_db(
            ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0), sr)
        # density_scale > 1 = denser than measured (spread target shrinks)
        _dens = float(ai_scales.get("density_scale", 1.0)) if ai_scales else 1.0
        _spread_target = (_ref_spread / _dens) if _ref_spread > 0.5 else None
        if _spread_target:
            logger.info(f"[JOB {job_id}] Density target: ref spread {_ref_spread:.1f} dB"
                        + (f" x{_dens:.2f} AI" if _dens != 1.0 else ""))
        processed = apply_master_bus(processed, sr, target_spread_db=_spread_target)

        # Closed loop: one-shot stages leave remainders — measure the result
        # and correct until inside tolerance (hard-capped at 2 extra passes).
        try:
            from processor.dsp.dynamics_transfer import match_dynamics as _md, dynamics_profile_gap_db as _rg
            _ref_mono_cl = ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0)
            for _cl in range(2):
                _mono_cl = processed if processed.ndim == 1 else processed.mean(axis=1)
                _sp = _active_spread_db(_mono_cl.astype(np.float64), sr)
                _rd = _rg(_mono_cl, _ref_mono_cl, sr)
                _did = []
                if _rd > 2.0:
                    processed = _md(processed, sr, _ref_mono_cl, strength=0.5)
                    _did.append(f"ride {_rd:.1f}dB")
                if _spread_target and _sp > _spread_target * 1.15:
                    processed = apply_master_bus(processed, sr, target_spread_db=_spread_target)
                    _did.append(f"spread {_sp:.1f}->{_spread_target:.1f}dB")
                if not _did:
                    break
                logger.info(f"[JOB {job_id}] Closed-loop pass {_cl + 1}: corrected {', '.join(_did)}")
        except Exception as _cl_err:
            logger.warning(f"[JOB {job_id}] Closed-loop pass skipped: {_cl_err}")
        # Final level is set to LUFS parity with the reference after saving
        # (raw-RMS parity overshot: the separated reference's noise floor
        # drags its sample RMS below its perceived loudness).

    # Absolute last resort: only if the output is essentially silent
    try:
        from processor.dsp.chain import normalize_rms, normalize_peak
        rms_post = float(np.sqrt(np.mean(processed ** 2)))
        if rms_post < 0.005:
            processed = normalize_rms(dry_audio, target_db=-18.0)
            processed = normalize_peak(processed, peak=0.95)
            print("⚠ Output near-silent; fell back to normalized dry vocal.")
    except Exception as e:
        print(f"⚠ Safety boost check failed: {e}")
    
    if not current_job:
        current_job = get_current_job()
    out_name = f"{current_job.id if current_job else uuid.uuid4()}.wav"
    out_path = settings.outputs_dir / out_name
    save_wav(out_path, processed, sr)
    _progress(current_job, 95, "Saving output…")
    
    # Compute final metrics
    metrics["processing_time_total"] = time.time() - job_start_time
    
    try:
        metrics["lufs_output"] = compute_lufs(out_path)
        logger.info(f"[JOB {job_id}] Output LUFS: {metrics['lufs_output']:.1f}")

        # ── Closed-loop LUFS match ─────────────────────────────────────────
        # Perceived-loudness parity with the reference. Boosts beyond the
        # peak ceiling are allowed and pushed back through the limiter —
        # that's what a limiter is for. (The old headroom-capped version
        # left a cross-song render 7.8 dB quieter than its reference.)
        ref_lufs = metrics.get("lufs_reference")
        if ref_lufs is not None and metrics["lufs_output"] is not None:
            from processor.dsp.master_bus import MasterBusSettings, _limiter_gain
            for _pass in range(3):  # converges in 1-2; 3 is a hard stop
                gain_db = float(np.clip(ref_lufs - metrics["lufs_output"], -12.0, 12.0))
                if abs(gain_db) < 0.5:
                    break
                processed = processed * (10 ** (gain_db / 20.0))
                peak = float(np.max(np.abs(processed)))
                if peak > 0.97:
                    # re-limit transparently rather than refusing the boost
                    block = max(1, int(sr * 0.001))
                    nb = int(np.ceil(len(processed) / block))
                    flat = np.abs(processed).max(axis=1) if processed.ndim == 2 else np.abs(processed)
                    pad = np.zeros(nb * block)
                    pad[: len(flat)] = flat
                    peaks = pad.reshape(nb, block).max(axis=1)
                    g = _limiter_gain(peaks, block, sr, 0.97, 80.0)
                    gi = np.interp(np.arange(len(processed)), (np.arange(nb) + 0.5) * block, g)
                    processed = processed * (gi[:, None] if processed.ndim == 2 else gi)
                    processed = np.clip(processed, -0.97, 0.97)
                save_wav(out_path, processed, sr)
                metrics["lufs_output"] = compute_lufs(out_path)
                logger.info(
                    f"[JOB {job_id}] LUFS pass {_pass + 1}: {gain_db:+.1f} dB → {metrics['lufs_output']:.1f} LUFS"
                )
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Failed to compute output LUFS: {e}")
    
    try:
        metrics["spectral_distance"] = compute_spectral_distance(ref_for_analysis, out_path, sr=sr)
        logger.info(f"[JOB {job_id}] Spectral distance: {metrics['spectral_distance']:.2f}")
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Failed to compute spectral distance: {e}")
    
    # Serialize recipe safely (dataclasses to dict) and add chorus profile if available
    from dataclasses import asdict, is_dataclass

    def _serialize(obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, list):
            return [_serialize(o) for o in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    recipe_dict = _serialize(recipe.__dict__ if hasattr(recipe, "__dict__") else {})
    recipe_dict["metrics"] = metrics
    if chorus_profile is not None:
        recipe_dict["chorus"] = chorus_profile.as_dict()
    if flanger_profile is not None:
        recipe_dict["flanger"] = flanger_profile.as_dict()
    if vocal_layers_profile is not None:
        recipe_dict["vocal_layers"] = {
            "total_layers": vocal_layers_profile.total_layers,
            "n_doublers": vocal_layers_profile.n_doublers,
            "doubler_detunes_cents": vocal_layers_profile.doubler_detunes_cents,
            "doubler_delays_ms": vocal_layers_profile.doubler_delays_ms,
            "doubler_pans": vocal_layers_profile.doubler_pans,
            "harmony_intervals": vocal_layers_profile.harmony_intervals,
            "harmony_strengths": vocal_layers_profile.harmony_strengths,
            "harmony_pans": vocal_layers_profile.harmony_pans,
        }
    if delay_info is not None:
        recipe_dict["delay"] = {
            "type": delay_info.get("type"),
            "delay_ms": delay_info.get("delay_ms"),
            "confidence": delay_info.get("confidence"),
            "echo_level": delay_info.get("echo_level"),
            "feedback": delay_info.get("feedback"),
            "_mix": delay_info.get("_mix"),
        }
    if reverb_profile_auto is not None:
        recipe_dict["reverb_auto"] = reverb_profile_auto.as_dict()
    if gate_settings is not None:
        recipe_dict["gate"] = _serialize(gate_settings)
    if doubler_settings is not None:
        recipe_dict["doubler"] = _serialize(doubler_settings)
    if exciter_settings is not None:
        recipe_dict["exciter"] = _serialize(exciter_settings)
    if tape_settings is not None:
        recipe_dict["tape"] = _serialize(tape_settings)
    if parallel_comp_settings is not None:
        recipe_dict["parallel_comp"] = _serialize(parallel_comp_settings)
    if ms_eq_settings is not None:
        recipe_dict["ms_eq"] = _serialize(ms_eq_settings)
    if autotune_settings is not None:
        recipe_dict["autotune"] = _serialize(autotune_settings)
    if ai_cfg:
        recipe_dict["ai_dsp_config"] = ai_cfg
    # ── Match report: how close did we land, dimension by dimension ────────
    # Style targets: everything a future job needs to REPLAY this style as a
    # preset with no reference upload (see /presets in the API).
    try:
        from processor.dsp.dynamics_transfer import loudness_quantiles_centered
        _ref_mono_st = ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0)
        recipe_dict["style_targets"] = {
            "ref_spread_db": round(float(_ref_spread), 1) if _ref_spread else None,
            "ref_lufs": metrics.get("lufs_reference"),
            "ref_loudness_quantiles": loudness_quantiles_centered(_ref_mono_st, sr),
            "ref_band_energy": _ref_stats.get("band_energy_pct"),
            "ref_stereo_profile": _ref_stereo_profile,
        }
    except Exception:
        pass

    try:
        from processor.dsp.master_bus import _active_spread_db as _spread
        from processor.dsp.dynamics_transfer import dynamics_profile_gap_db as _ride_gap
        _out_mono = processed if processed.ndim == 1 else processed.mean(axis=1)
        recipe_dict["match_report"] = {
            "lufs_gap_db": (round(metrics["lufs_output"] - metrics["lufs_reference"], 1)
                            if metrics.get("lufs_output") is not None and metrics.get("lufs_reference") is not None else None),
            "spectral_distance": metrics.get("spectral_distance"),
            "dynamic_spread_out_db": round(_spread(_out_mono.astype(np.float64), sr), 1),
            "dynamics_ride_gap_db": round(_ride_gap(
                _out_mono, ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0), sr), 1),
            "dynamic_spread_ref_db": round(_spread(
                (ref_audio if ref_audio.ndim == 1 else ref_audio.mean(axis=0)).astype(np.float64), sr), 1),
            "ai_scales": ai_scales or None,
        }
        try:
            from processor.dsp.stereo import measure_stereo_profile as _msp
            _wout = _msp(processed, sr)
            if _wout and _ref_stereo_profile:
                recipe_dict["match_report"]["stereo_width_out"] = {k: round(v, 1) for k, v in _wout.items()}
                recipe_dict["match_report"]["stereo_width_ref"] = {k: round(v, 1) for k, v in _ref_stereo_profile.items()}
        except Exception:
            pass
    except Exception:
        pass

    # Include dry vocal profile so the UI and debugging can see what was compared
    try:
        recipe_dict["dry_vocal_profile"] = _dry_stats
        recipe_dict["ref_vocal_profile"] = _ref_stats
    except Exception:
        pass
    if refinement_info:
        recipe_dict["refinement"] = refinement_info
    if options.get("_stem_urls"):
        recipe_dict["stems"] = options["_stem_urls"]

    # Save recipe with metrics
    if current_job:
        recipe_path = settings.outputs_dir / f"{current_job.id}.json"
        recipe_path.write_text(json.dumps(recipe_dict, indent=2))
    
    # Update metrics store
    try:
        from api.metrics import update_metrics, increment_job_count
        update_metrics(metrics)
        increment_job_count(success=True)
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Failed to update metrics store: {e}")
    
    _progress(current_job, 100, "Complete")
    logger.info(f"[JOB {job_id}] Total processing time: {metrics['processing_time_total']:.2f}s")
    print(f"✓ Job complete. Output saved to {out_name}")
    print(f"  Metrics: total={metrics['processing_time_total']:.2f}s, DSP={metrics['processing_time_dsp']:.2f}s")

    sweep_old_files()
    return out_path


def process_preset_job(dry_path: Path, preset: dict, options: dict) -> Path:
    """Replay a saved style on a new dry vocal: no reference, no separation,
    no detection — settings are reconstructed from the stored recipe and the
    stored style targets (ride quantiles, band energy, spread, LUFS) drive
    the same matching passes the live path uses."""
    current_job = get_current_job()
    job_id = current_job.id if current_job else str(uuid.uuid4())
    t0 = time.time()
    name = (preset.get("_preset") or {}).get("name", "preset")
    logger.info(f"[JOB {job_id}] Preset replay: {name}")
    _progress(current_job, 5, f"Applying preset {name}…")

    dry_norm = settings.inputs_dir / f"{uuid.uuid4()}_dry.wav"
    run_ffmpeg_normalize(dry_path, dry_norm)
    dry_audio, sr = load_wav(dry_norm)

    from processor.dsp.eq import EqBand
    from processor.dsp.compressor import CompressorSettings
    from processor.dsp.reverb import ReverbSettings
    from processor.dsp.gate import GateSettings
    from processor.dsp.tape import TapeSettings
    from processor.dsp.exciter import ExciterSettings
    from processor.dsp.parallel_comp import ParallelCompSettings
    from processor.dsp.doubler import DoublerSettings
    from processor.dsp.ms_eq import MsEqSettings
    from processor.dsp.analysis.autotune_analysis import AutotuneSettings
    from processor.dsp.analysis.chorus_analysis import ChorusProfile
    from processor.dsp.analysis.flanger_analysis import FlangerProfile
    from processor.dsp.analysis.vocal_layers_analysis import VocalLayersProfile
    import inspect

    def build(cls, d):
        if not isinstance(d, dict):
            return None
        try:
            keys = set(inspect.signature(cls).parameters)
            return cls(**{k: v for k, v in d.items() if k in keys})
        except Exception:
            return None

    eq_bands = [b for b in (build(EqBand, x) for x in (preset.get("eq") or [])) if b]
    comp = build(CompressorSettings, preset.get("compressor"))
    targets = preset.get("style_targets") or {}
    ref_profile = preset.get("ref_vocal_profile") or {}

    reverb = None
    ra = preset.get("reverb_auto")
    if isinstance(ra, dict) and ra.get("wet"):
        tail = float(np.clip(float(ref_profile.get("reverb_tail_ratio", 0.0) or 0.0), 0.0, 1.0))
        wetness = float(np.clip((tail - 0.3) / 0.4, 0.0, 1.0))
        rv = ReverbSettings(
            decay_s=float(ra.get("rt60", 0.8)),
            mix=float(np.clip(float(ra["wet"]) * (1.0 + 0.6 * wetness), 0.0, 0.20 + 0.15 * wetness)),
            pre_delay_ms=float(ra.get("predelay_ms", 10.0)),
        )
        reverb = rv if rv.mix >= 0.07 else None
    elif isinstance(preset.get("reverb"), dict):
        reverb = build(ReverbSettings, preset["reverb"])

    chorus = build(ChorusProfile, preset.get("chorus"))
    flanger = build(FlangerProfile, preset.get("flanger"))
    gate = build(GateSettings, preset.get("gate"))
    tape = build(TapeSettings, preset.get("tape"))
    exciter = build(ExciterSettings, preset.get("exciter"))
    parallel = build(ParallelCompSettings, preset.get("parallel_comp"))
    doubler = build(DoublerSettings, preset.get("doubler"))
    autotune = None  # autotune removed from the product (sounded bad; module kept for future opt-in)
    ms_eq = None
    if isinstance(preset.get("ms_eq"), dict):
        try:
            me = preset["ms_eq"]
            ms_eq = MsEqSettings(
                mid_bands=[b for b in (build(EqBand, x) for x in me.get("mid_bands", [])) if b],
                side_bands=[b for b in (build(EqBand, x) for x in me.get("side_bands", [])) if b],
            )
        except Exception:
            ms_eq = None
    saturation = preset.get("saturation_drive")
    width = preset.get("width") if isinstance(preset.get("width"), dict) else None
    layers_profile = build(VocalLayersProfile, preset.get("vocal_layers"))
    delay_info = dict(preset["delay"]) if isinstance(preset.get("delay"), dict) else None
    if delay_info and not delay_info.get("_mix"):
        delay_info["_mix"] = float(np.clip(delay_info.get("echo_level") or 0.25, 0.10, 0.5))

    _progress(current_job, 40, "Applying DSP chain…")
    processed = apply_chain(
        dry_audio, sr,
        eq_bands=eq_bands, comp=comp, reverb=reverb,
        saturation_drive=saturation, width=width,
        chorus_profile=chorus, flanger_profile=flanger, reference=None,
        enable_deesser=options.get("enable_deesser", True),
        gate=gate, parallel_comp=parallel, tape=tape, exciter=exciter,
        doubler=doubler, ms_eq=ms_eq, autotune=autotune,
    )

    if layers_profile is not None and options.get("enable_harmony", True):
        selected = options.get("selected_layers") or []
        prof = filter_vocal_layers_profile(layers_profile, selected) if selected else layers_profile
        if prof is not None:
            _progress(current_job, 60, "Applying vocal layers…")
            processed = apply_vocal_layers(processed, sr, prof)

    if delay_info and delay_info.get("delay_ms", 0) > 0 and options.get("enable_delay", True):
        fb = float(delay_info.get("feedback") or 0.25)
        mix_val = float(delay_info.get("_mix", 0.25))
        if processed.ndim == 2:
            processed = np.stack([
                apply_delay(processed[:, 0], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val),
                apply_delay(processed[:, 1], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val),
            ], axis=1)
        else:
            processed = apply_delay(processed, sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val)

    _progress(current_job, 75, "Matching stored style targets…")
    try:
        from processor.dsp.stereo import apply_stereo_image
        _sp = targets.get("ref_stereo_profile")
        processed = apply_stereo_image(processed, sr, target=_sp)
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Preset stereo skipped: {e}")
    try:
        from processor.dsp.dynamics_transfer import match_dynamics
        q = targets.get("ref_loudness_quantiles")
        if q:
            processed = match_dynamics(processed, sr, None, strength=0.7, ref_quantiles=q)
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Preset dynamics skipped: {e}")

    try:
        ref_bands = targets.get("ref_band_energy") or {}
        if ref_bands:
            from processor.dsp.eq import EqBand as _B, apply_eq as _ae
            _mono = processed if processed.ndim == 1 else processed.mean(axis=1)
            stft = np.abs(librosa.stft(_mono[: sr * 60]))
            freqs = librosa.fft_frequencies(sr=sr)
            total = float(np.sum(stft ** 2)) + 1e-12
            gaps = []
            prev = 0
            for (bn, cutoff, f, qq) in [("sub_60hz", 60, 60.0, 0.7), ("low_250hz", 250, 200.0, 0.7),
                                        ("low_mid_500hz", 500, 400.0, 0.9), ("mid_2khz", 2000, 1200.0, 1.0),
                                        ("high_mid_6khz", 6000, 4000.0, 1.0), ("air_12khz", 12000, 10000.0, 0.7)]:
                mask = (freqs >= prev) & (freqs < cutoff)
                e = float(np.sum(stft[mask] ** 2)) / total
                g = float(np.clip((float(ref_bands.get(bn, 0)) - e) * 40.0, -4.0, 4.0))
                if abs(g) >= 0.5:
                    gaps.append(_B(f=f, gain_db=g, q=qq))
                prev = cutoff
            if gaps:
                if processed.ndim == 2:
                    processed = np.stack([_ae(processed[:, 0], sr, gaps), _ae(processed[:, 1], sr, gaps)], axis=1)
                else:
                    processed = _ae(processed, sr, gaps)
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Preset tone match skipped: {e}")

    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    from processor.dsp.master_bus import apply_master_bus, _limiter_gain
    _progress(current_job, 88, "Master bus…")
    processed = apply_master_bus(processed, sr, target_spread_db=targets.get("ref_spread_db"))

    out_path = settings.outputs_dir / f"{current_job.id if current_job else uuid.uuid4()}.wav"
    save_wav(out_path, processed, sr)

    try:
        ref_lufs = targets.get("ref_lufs")
        if ref_lufs is not None:
            out_lufs = compute_lufs(out_path)
            for _ in range(3):
                gain_db = float(np.clip(ref_lufs - out_lufs, -12.0, 12.0))
                if abs(gain_db) < 0.5:
                    break
                processed = processed * (10 ** (gain_db / 20.0))
                if float(np.max(np.abs(processed))) > 0.97:
                    block = max(1, int(sr * 0.001))
                    nb = int(np.ceil(len(processed) / block))
                    flat = np.abs(processed).max(axis=1) if processed.ndim == 2 else np.abs(processed)
                    pad = np.zeros(nb * block)
                    pad[: len(flat)] = flat
                    g = _limiter_gain(pad.reshape(nb, block).max(axis=1), block, sr, 0.97, 80.0)
                    gi = np.interp(np.arange(len(processed)), (np.arange(nb) + 0.5) * block, g)
                    processed = np.clip(processed * (gi[:, None] if processed.ndim == 2 else gi), -0.97, 0.97)
                save_wav(out_path, processed, sr)
                out_lufs = compute_lufs(out_path)
    except Exception as e:
        logger.warning(f"[JOB {job_id}] Preset LUFS match skipped: {e}")

    out_recipe = dict(preset)
    out_recipe["applied_preset"] = out_recipe.pop("_preset", {"name": name})
    out_recipe.pop("match_report", None)
    if current_job:
        (settings.outputs_dir / f"{current_job.id}.json").write_text(json.dumps(out_recipe, indent=2))
    _progress(current_job, 100, "Complete")
    logger.info(f"[JOB {job_id}] Preset replay done in {time.time() - t0:.1f}s")
    sweep_old_files()
    return out_path


def enqueue_job(reference: Path, dry: Path, options: dict | None = None) -> str:
    options = options or {}
    with Connection(redis.from_url(settings.redis_url)):
        q = Queue("jobs")
        # On Windows, disable timeout to avoid SIGALRM issues
        timeout = None if sys.platform == "win32" else "30m"
        job = q.enqueue(process_job, reference, dry, options, job_timeout=timeout)
        return job.get_id()


class NoOpDeathPenalty:
    """No-op death penalty for Windows compatibility."""
    def __init__(self, timeout, exception_class, job_id=None):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class WindowsSafeWorker(SimpleWorker):
    """SimpleWorker that disables timeout mechanism on Windows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sys.platform == "win32":
            # Use no-op death penalty on Windows to avoid SIGALRM
            self.death_penalty_class = NoOpDeathPenalty


def run_worker() -> None:
    with Connection(redis.from_url(settings.redis_url)):
        # Use WindowsSafeWorker to avoid fork and SIGALRM issues on Windows
        worker_class = WindowsSafeWorker if sys.platform == "win32" else SimpleWorker
        worker = worker_class(["jobs"])
        worker.work(with_scheduler=True)


if __name__ == "__main__":
    print(json.dumps({"worker": "starting", "redis": settings.redis_url}))
    run_worker()

