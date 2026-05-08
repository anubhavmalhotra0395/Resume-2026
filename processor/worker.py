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
from processor.utils.audio_io import load_wav, run_ffmpeg_normalize, save_wav
from processor.utils.vocal_extraction import extract_vocals
from processor.utils.demucs_utils import run_demucs_extract
from processor.utils.phase_align import align_by_crosscorr
from processor.utils.metrics import timer, compute_spectral_distance, compute_lufs
from processor.ml_refine.rvc_refiner import get_rvc_refiner
from processor.ml_refine.spectral_refiner import get_refiner
from processor.dsp.analysis.chorus_analysis import detect_chorus
from processor.dsp.analysis.flanger_analysis import detect_flanger
from processor.dsp.analysis.harmony_analysis import detect_harmonies
from processor.dsp.effects.apply_harmony import apply_harmony
from processor.dsp.analysis.vocal_layers_analysis import detect_vocal_layers, VocalLayersProfile
from processor.dsp.effects.apply_vocal_layers import apply_vocal_layers, filter_vocal_layers_profile
from processor.dsp.delay import detect_delay, apply_delay
from processor.dsp.analysis.reverb_analysis import estimate_reverb_params
from processor.dsp.effects.apply_reverb import apply_reverb
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
    
    # Initialize metrics
    metrics = {
        "processing_time_total": 0.0,
        "processing_time_rvc": 0.0,
        "processing_time_vocoder": 0.0,
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
    run_ffmpeg_normalize(reference_path, ref_norm)
    run_ffmpeg_normalize(dry_path, dry_norm)
    _progress(current_job, 5, "Audio normalised")

    # Extract vocals from reference track (in case it's a full mix)
    use_stems = options.get("stems_mode", True)
    ref_for_analysis = ref_norm
    
    if use_stems:
        _progress(current_job, 8, "Extracting vocals from reference…")
        ref_vocals_path = settings.inputs_dir / f"{uuid.uuid4()}_ref_vocals.wav"
        print(f"Attempting to extract vocals from reference: {reference_path.name}")
        extracted = extract_vocals(ref_norm, ref_vocals_path, force_demucs=True)
        
        if extracted and extracted.exists():
            ref_for_analysis = extracted
            print(f"✓ Successfully extracted vocals from reference track")
        else:
            print(f"⚠ Vocal extraction failed — analysing full mix instead")
        _progress(current_job, 20, "Vocals extracted")
    
    # Load reference audio
    ref_audio, sr = load_wav(ref_for_analysis)
    ref_duration = len(ref_audio) / sr
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
    
    # Apply RVC if ML_REFINE is enabled (before DSP chain)
    ml_refine_flag = options.get("ml_refine", False)
    dry_audio, _ = load_wav(dry_norm)
    
    if ml_refine_flag:
        model_path = settings.rvc_model_path if settings.rvc_model_path else None
        
        if not model_path or not Path(model_path).exists():
            print(f"⚠ RVC requested but model not found at '{model_path}'.")
            print(f"   To use RVC: Place a trained RVC model (.pth) at '{model_path}'")
            print(f"   And vocoder at 'models/rvc/vocoder.pth' (optional)")
            print(f"   Or set APP_RVC_MODEL_PATH environment variable to your model path.")
            print(f"   Continuing without RVC (using original dry vocal)...")
        else:
            print(f"Applying RVC voice conversion with model: {model_path}")
            try:
                # Get RVC refiner instance
                rvc_refiner = get_rvc_refiner(
                    model_path=model_path,
                    enable_gpu=settings.rvc_enable_gpu,
                )
                
                # Log model availability
                hubert_avail = rvc_refiner.hubert_model is not None
                rvc_avail = rvc_refiner.rvc_model is not None
                vocoder_avail = rvc_refiner.vocoder is not None
                logger.info(f"[JOB {job_id}] Models loaded - HuBERT: {hubert_avail}, RVC: {rvc_avail}, Vocoder: {vocoder_avail} on {rvc_refiner.device}")
                
                # Process dry audio directly (new implementation)
                original_length = len(dry_audio)
                print(f"  Processing dry audio with RVC (input: {original_length} samples @ {sr}Hz)...")
                rvc_start = time.time()
                dry_audio_processed = rvc_refiner.process(dry_audio, sr)
                metrics["processing_time_rvc"] = time.time() - rvc_start
                logger.info(f"[JOB {job_id}] RVC processing completed in {metrics['processing_time_rvc']:.2f}s")
                
                if dry_audio_processed is not None and len(dry_audio_processed) > 0:
                    # Check if processing actually changed the audio
                    max_original = np.max(np.abs(dry_audio))
                    max_processed = np.max(np.abs(dry_audio_processed))
                    
                    if max_processed > 1e-6:  # Valid output
                        dry_audio = dry_audio_processed
                        # RVC outputs at 44.1kHz, resample to match input sr if needed
                        from processor.ml_refine.rvc_refiner import OUTPUT_SAMPLE_RATE
                        if OUTPUT_SAMPLE_RATE != sr:
                            # Resample to match input sample rate
                            dry_audio = librosa.resample(
                                dry_audio,
                                orig_sr=OUTPUT_SAMPLE_RATE,
                                target_sr=sr,
                            )
                        # Ensure length matches original (within tolerance)
                        current_length = len(dry_audio)
                        if abs(current_length - original_length) > sr * 0.1:  # More than 100ms difference
                            # Trim or pad to match
                            if current_length > original_length:
                                dry_audio = dry_audio[:original_length]
                            else:
                                dry_audio = np.pad(dry_audio, (0, original_length - current_length), mode='constant')
                        
                        print(f"✓ RVC processing complete. Phase-aligning to reference...")
                        # Phase align RVC output to reference to prevent phasing artifacts
                        y_ref, _ = librosa.load(str(ref_for_analysis), sr=sr, mono=True)
                        min_len = min(len(dry_audio), len(y_ref))
                        dry_audio = align_by_crosscorr(y_ref[:min_len], dry_audio[:min_len], sr=sr)
                        print(f"✓ RVC output phase-aligned and ready for DSP")
                    else:
                        print(f"⚠ RVC output is too quiet. Using original dry vocal.")
                else:
                    print(f"⚠ RVC processing returned invalid output. Using original dry vocal.")
                    
            except Exception as e:
                print(f"⚠ RVC processing failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Continuing with original dry vocal...")
    
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

    def _detect_harmonies():
        return detect_harmonies(ref_audio, sr)

    def _detect_reverb():
        return estimate_reverb_params(ref_audio, sr)

    def _detect_delay():
        return detect_delay(ref_mono, sr)

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

    def _detect_autotune():
        return detect_autotune(ref_audio, sr)

    def _detect_vocal_layers():
        return detect_vocal_layers(ref_audio, sr)

    chorus_profile = flanger_profile = harmony_profile = reverb_profile_auto = delay_info = None
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
    # Autotune: always detect — auto-applied if reference shows pitch correction
    detectors["autotune"] = _detect_autotune

    _DETECTOR_TIMEOUT = 90  # seconds per detector

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fn): name for name, fn in detectors.items()}

        def _process_future(future, name):
            try:
                result = future.result(timeout=_DETECTOR_TIMEOUT)
                if name == "chorus" and result:
                    chorus_profile = result
                    logger.info(f"[JOB {job_id}] Chorus: rate={result.rate_hz:.2f}Hz mix={result.mix:.2f}")
                elif name == "flanger" and result:
                    flanger_profile = result
                    logger.info(f"[JOB {job_id}] Flanger: rate={result.rate_hz:.2f}Hz feedback={result.feedback:.2f} mix={result.mix:.2f}")
                elif name == "vocal_layers":
                    vocal_layers_profile = result
                    if result:
                        logger.info(
                            f"[JOB {job_id}] Vocal layers: {result.total_layers} total "
                            f"({result.n_doublers} doublers + {len(result.harmony_intervals)} harmony voices)"
                        )
                    else:
                        logger.info(f"[JOB {job_id}] Vocal layers: single voice, no layering detected")
                elif name == "reverb" and result:
                    reverb_profile_auto = result
                    logger.info(f"[JOB {job_id}] Reverb: rt60={result.rt60:.2f}s wet={result.wet:.2f}")
                elif name == "delay" and result:
                    delay_info = result
                    logger.info(f"[JOB {job_id}] Delay: {result.get('type')} {result.get('delay_ms', 0):.1f}ms")
                elif name == "gate" and result:
                    gate_settings = result
                    logger.info(f"[JOB {job_id}] Gate: threshold={result.threshold_db:.1f}dB")
                elif name == "doubler" and result:
                    doubler_settings = result
                    logger.info(f"[JOB {job_id}] Doubler: mix={result.mix:.2f}")
                elif name == "exciter" and result:
                    exciter_settings = result
                    logger.info(f"[JOB {job_id}] Exciter: drive={result.drive:.2f} mix={result.mix:.2f}")
                elif name == "tape" and result:
                    tape_settings = result
                    logger.info(f"[JOB {job_id}] Tape: drive={result.drive:.2f} rolloff={result.hf_rolloff_hz:.0f}Hz")
                elif name == "parallel_comp" and result:
                    parallel_comp_settings = result
                    logger.info(f"[JOB {job_id}] Parallel comp: blend={result.blend:.2f}")
                elif name == "ms_eq" and result:
                    ms_eq_settings = result
                    logger.info(f"[JOB {job_id}] M-S EQ: mid_bands={len(result.mid_bands)} side_bands={len(result.side_bands)}")
                elif name == "autotune" and result:
                    autotune_settings = result
                    logger.info(f"[JOB {job_id}] Autotune: strength={result.strength:.2f}")
            except Exception as e:
                logger.warning(f"[JOB {job_id}] {name} detection failed or timed out: {e}")

        # Use nonlocal assignments via a results dict to avoid closure issues
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
        if _det_results.get("autotune"):
            autotune_settings = _det_results["autotune"]
            logger.info(f"[JOB {job_id}] Autotune: strength={autotune_settings.strength:.2f}")

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

    # ── AI Recipe Review ─────────────────────────────────────────────────
    ai_review_result = None
    try:
        from api.ai_audio import ai_review_recipe
        from dataclasses import asdict as _dc_asdict, is_dataclass as _is_dc

        def _safe_asdict(obj):
            if _is_dc(obj):
                return _dc_asdict(obj)
            if isinstance(obj, dict):
                return obj
            return str(obj)

        _detected_summary: dict = {}
        # Always include eq/compression/reverb/saturation/width — AI needs to see them even if not detected
        if recipe.eq:
            _detected_summary["eq"] = {"n_bands": len(recipe.eq), "max_gain_db": max(abs(b.gain_db) for b in recipe.eq), "detected": True}
        else:
            _detected_summary["eq"] = {"detected": False, "note": "Not detected via signal analysis — apply based on band_gap_vs_reference"}
        if recipe.compressor:
            _detected_summary["compression"] = {**_safe_asdict(recipe.compressor), "detected": True}
        else:
            _detected_summary["compression"] = {"detected": False, "note": "Not detected — apply if dynamic_range_db > 14dB"}
        if reverb_profile_auto:
            _detected_summary["reverb"] = {"rt60": reverb_profile_auto.rt60, "wet": reverb_profile_auto.wet,
                                            "predelay_ms": reverb_profile_auto.predelay_ms,
                                            "confidence": reverb_profile_auto.confidence, "detected": True}
        else:
            _detected_summary["reverb"] = {"detected": False, "note": "Not detected — apply light reverb (mix=0.12) unless dry is already very wet"}
        if recipe.saturation_drive:
            _detected_summary["saturation"] = {"drive": recipe.saturation_drive, "detected": True}
        else:
            _detected_summary["saturation"] = {"detected": False, "note": "Not detected — apply light saturation (drive=1.2) for warmth"}
        if recipe.width:
            _detected_summary["width"] = {**_safe_asdict(recipe.width), "detected": True}
        else:
            _detected_summary["width"] = {"detected": False, "note": "Not detected — apply moderate width (mix=0.3) unless reference is mono"}
        if chorus_profile and chorus_profile.mix > 0:
            _detected_summary["chorus"] = {"rate_hz": chorus_profile.rate_hz,
                                            "depth": chorus_profile.depth, "mix": chorus_profile.mix, "detected": True}
        else:
            _detected_summary["chorus"] = {"detected": False, "note": "Not detected — skip unless vocal_layers suggests layering"}
        if flanger_profile and flanger_profile.mix > 0:
            _detected_summary["flanger"] = {"rate_hz": flanger_profile.rate_hz, "mix": flanger_profile.mix, "detected": True}
        else:
            _detected_summary["flanger"] = {"detected": False}
        if delay_info and delay_info.get("delay_ms", 0) > 0:
            _detected_summary["delay"] = {"delay_ms": delay_info.get("delay_ms"),
                                           "type": delay_info.get("type"),
                                           "confidence": delay_info.get("confidence"), "detected": True}
        else:
            _detected_summary["delay"] = {"detected": False, "note": "Not detected — apply subtle delay (15-25ms) unless confidence is 0"}
        if gate_settings:
            _detected_summary["gate"] = {**_safe_asdict(gate_settings), "detected": True}
        else:
            _detected_summary["gate"] = {"detected": False, "note": "Not detected — apply gate only if spectral_flatness > 0.02"}
        if tape_settings:
            _detected_summary["tape"] = {**_safe_asdict(tape_settings), "detected": True}
        else:
            _detected_summary["tape"] = {"detected": False, "note": "Not detected — apply light tape emulation (drive=0.2, mix=0.3) for warmth"}
        if exciter_settings:
            _detected_summary["exciter"] = {**_safe_asdict(exciter_settings), "detected": True}
        else:
            _detected_summary["exciter"] = {"detected": False, "note": "Not detected — apply exciter (drive=0.2, mix=0.15) unless dry is already bright"}
        if parallel_comp_settings:
            _detected_summary["parallel_comp"] = {**_safe_asdict(parallel_comp_settings), "detected": True}
        else:
            _detected_summary["parallel_comp"] = {"detected": False, "note": "Not detected — apply parallel compression (blend=0.2) for density"}
        if autotune_settings:
            _detected_summary["autotune"] = {**_safe_asdict(autotune_settings), "detected": True}
        else:
            _detected_summary["autotune"] = {"detected": False, "note": "Not detected — skip autotune"}
        if vocal_layers_profile:
            _detected_summary["vocal_layers"] = {
                "n_doublers": vocal_layers_profile.n_doublers,
                "harmony_intervals": vocal_layers_profile.harmony_intervals,
                "total_layers": vocal_layers_profile.total_layers,
                "detected": True,
            }
        else:
            _detected_summary["vocal_layers"] = {"detected": False, "total_layers": 1, "note": "Single voice — skip vocal layers"}

        ai_review_result = ai_review_recipe(_detected_summary, _dry_stats, _ref_stats)
        logger.info(f"[JOB {job_id}] AI review: {ai_review_result.get('summary', '')}")

    except Exception as _ai_err:
        logger.warning(f"[JOB {job_id}] AI recipe review skipped: {_ai_err}")

    # Helper: check AI decision for an effect (default to True if no review available)
    def _ai_ok(key: str) -> bool:
        if ai_review_result is None:
            return True
        return bool(ai_review_result.get(f"apply_{key}", True))

    def _ai_scale(key: str, default: float = 1.0) -> float:
        if ai_review_result is None:
            return default
        return float(ai_review_result.get("adjustments", {}).get(key, default))

    # ── Apply AI gating + fallback defaults for undetected effects ──────────
    # Import effect classes needed for fallback defaults
    from processor.dsp.tape import TapeSettings as _TapeSettings
    from processor.dsp.exciter import ExciterSettings as _ExciterSettings
    from processor.dsp.gate import GateSettings as _GateSettings
    from processor.dsp.parallel_comp import ParallelCompSettings as _ParallelCompSettings
    from processor.dsp.analysis.chorus_analysis import ChorusProfile as _ChorusProfile

    # Chorus: gate if AI says skip; scale mix if kept; use fallback if not detected but AI says apply
    if not _ai_ok("chorus"):
        if chorus_profile is not None:
            logger.info(f"[JOB {job_id}] AI: skipping chorus")
        chorus_profile = None
    else:
        if chorus_profile is not None:
            chorus_profile.mix = float(np.clip(chorus_profile.mix * _ai_scale("chorus_mix_scale", 1.0), 0.0, 0.25))
        # No chorus fallback — chorus is a strong modulation effect, only apply if detected

    # Delay: gate if AI says skip; scale mix if kept; apply subtle delay if AI says apply but not detected
    if not _ai_ok("delay"):
        if delay_info is not None:
            logger.info(f"[JOB {job_id}] AI: skipping delay")
        delay_info = None
    else:
        if delay_info is not None:
            delay_info = dict(delay_info)
            delay_info["_mix"] = _ai_scale("delay_mix", 0.25)
        else:
            # AI says apply but detector found nothing — use a subtle default delay
            logger.info(f"[JOB {job_id}] AI: applying default delay (not detected, but AI recommends)")
            delay_info = {"delay_ms": 18.0, "type": "dotted_eighth", "confidence": 0.3, "_mix": 0.15}

    # Gate: only skip if AI says so; no fallback (gate is conservative by nature)
    if not _ai_ok("gate"):
        if gate_settings is not None:
            logger.info(f"[JOB {job_id}] AI: skipping gate")
        gate_settings = None

    # Tape: gate if AI says skip; scale mix if kept; apply conservative default if not detected but AI says apply
    if not _ai_ok("tape"):
        if tape_settings is not None:
            logger.info(f"[JOB {job_id}] AI: skipping tape")
        tape_settings = None
    else:
        if tape_settings is not None:
            tape_settings.mix = float(np.clip(tape_settings.mix * _ai_scale("tape_mix_scale", 1.0), 0.0, 0.4))
        else:
            logger.info(f"[JOB {job_id}] AI: applying default tape emulation (not detected, but AI recommends)")
            tape_settings = _TapeSettings(drive=0.2, hf_rolloff_hz=14000.0, mix=0.3)

    # Exciter: gate if AI says skip; apply conservative default if not detected but AI says apply
    if not _ai_ok("exciter"):
        if exciter_settings is not None:
            logger.info(f"[JOB {job_id}] AI: skipping exciter")
        exciter_settings = None
    else:
        if exciter_settings is None:
            logger.info(f"[JOB {job_id}] AI: applying default exciter (not detected, but AI recommends)")
            exciter_settings = _ExciterSettings(drive=0.2, mix=0.15, freq_hz=6000.0)

    # Parallel comp: gate if AI says skip; apply conservative default if not detected but AI says apply
    if not _ai_ok("parallel_comp"):
        if parallel_comp_settings is not None:
            logger.info(f"[JOB {job_id}] AI: skipping parallel_comp")
        parallel_comp_settings = None
    else:
        if parallel_comp_settings is None:
            logger.info(f"[JOB {job_id}] AI: applying default parallel compression (not detected, but AI recommends)")
            parallel_comp_settings = _ParallelCompSettings(threshold_db=-30.0, ratio=10.0, attack_ms=2.0, release_ms=150.0, blend=0.2)

    # Autotune: gate if AI says skip; no fallback (only apply if actually detected)
    if not _ai_ok("autotune"):
        if autotune_settings is not None:
            logger.info(f"[JOB {job_id}] AI: skipping autotune")
        autotune_settings = None

    # Vocal layers: gate if AI says skip; scale if kept; no fallback (only apply if detected)
    if not _ai_ok("vocal_layers"):
        if vocal_layers_profile is not None:
            logger.info(f"[JOB {job_id}] AI: skipping vocal layers")
        vocal_layers_profile = None
    elif vocal_layers_profile is not None:
        scale = _ai_scale("harmony_strength_scale", 0.7)
        vocal_layers_profile.harmony_strengths = [s * scale for s in vocal_layers_profile.harmony_strengths]

    # ── AI DSP config override ──────────────────────────────────────────────
    ai_cfg = options.get("ai_dsp_config")

    # EQ: use detected eq bands if AI ok; apply a gentle corrective EQ from band_gap if not detected
    if _ai_ok("eq"):
        if recipe.eq:
            eq_bands_final = recipe.eq
        else:
            # Not detected but AI says apply — build corrective EQ from band_gap_vs_reference
            from processor.dsp.eq import EqBand as _EqBand
            _band_gap = _dry_stats.get("band_gap_vs_reference", {})
            _fallback_eq = []
            _gap_map = [
                ("sub_60hz",       60.0,   0.7),
                ("low_250hz",     200.0,   0.7),
                ("low_mid_500hz", 400.0,   0.9),
                ("mid_2khz",     1200.0,   1.0),
                ("high_mid_6khz", 4000.0,  1.0),
                ("air_12khz",    10000.0,  0.7),
            ]
            for band_name, freq, q in _gap_map:
                gap = float(_band_gap.get(band_name, 0.0))
                # Convert fractional energy gap to dB: ±0.05 gap ≈ ±2 dB
                gain_db = float(np.clip(gap * 40.0, -4.0, 4.0))
                if abs(gain_db) >= 0.5:
                    _fallback_eq.append(_EqBand(f=freq, gain_db=gain_db, q=q))
            if _fallback_eq:
                logger.info(f"[JOB {job_id}] AI: applying {len(_fallback_eq)} corrective EQ bands from band gap analysis")
                eq_bands_final = _fallback_eq
            else:
                eq_bands_final = None
    else:
        logger.info(f"[JOB {job_id}] AI: skipping EQ")
        eq_bands_final = None

    # Compression: use detected settings; apply default if not detected but AI says apply
    if _ai_ok("compression"):
        if recipe.compressor:
            comp_final = recipe.compressor
        else:
            from processor.dsp.compressor import CompressorSettings as _CS
            _dyn_range = _dry_stats.get("dynamic_range_db", 20.0)
            _ratio = 4.0 if _dyn_range > 20 else 3.0
            logger.info(f"[JOB {job_id}] AI: applying default compression (not detected, dynamic_range={_dyn_range:.1f}dB)")
            comp_final = _CS(threshold_db=-24.0, ratio=_ratio, attack_ms=10.0, release_ms=100.0, makeup_db=3.0)
    else:
        logger.info(f"[JOB {job_id}] AI: skipping compression")
        comp_final = None

    # Saturation: use detected value; apply conservative default if not detected but AI says apply
    if _ai_ok("saturation"):
        if recipe.saturation_drive:
            saturation_final = recipe.saturation_drive
        else:
            logger.info(f"[JOB {job_id}] AI: applying default saturation drive=1.2 (not detected)")
            saturation_final = 1.2
    else:
        logger.info(f"[JOB {job_id}] AI: skipping saturation")
        saturation_final = None

    # Scale EQ gains by AI recommendation
    if eq_bands_final is not None:
        eq_scale = _ai_scale("eq_gain_scale", 1.0)
        if abs(eq_scale - 1.0) > 0.05:
            for b in eq_bands_final:
                b.gain_db = float(np.clip(b.gain_db * eq_scale, -6.0, 6.0))

    # Reverb: prefer reverb_analysis result; fallback to recipe or default
    if not _ai_ok("reverb"):
        logger.info(f"[JOB {job_id}] AI: skipping reverb")
        reverb_final = None
    elif reverb_profile_auto is not None:
        from processor.dsp.reverb import ReverbSettings as _RS
        _rev_wet = reverb_profile_auto.wet * _ai_scale("reverb_mix_scale", 1.0)
        _rev = _RS(
            decay_s=reverb_profile_auto.rt60,
            mix=float(np.clip(_rev_wet, 0.0, 0.20)),
            pre_delay_ms=reverb_profile_auto.predelay_ms,
        )
        reverb_final = _rev if _rev.mix >= 0.07 else None
    elif recipe.reverb:
        reverb_final = recipe.reverb
    else:
        # Not detected but AI says apply — use a light default reverb
        from processor.dsp.reverb import ReverbSettings as _RS
        _tail = _dry_stats.get("reverb_tail_ratio", 0.0)
        if _tail < 0.25:
            logger.info(f"[JOB {job_id}] AI: applying default reverb (not detected, tail_ratio={_tail:.3f})")
            reverb_final = _RS(decay_s=0.8, mix=0.12, pre_delay_ms=12.0)
        else:
            logger.info(f"[JOB {job_id}] AI: skipping default reverb (dry vocal already has reverb tail_ratio={_tail:.3f})")
            reverb_final = None

    # Width: use detected; fallback to moderate default if AI says apply but not detected
    _width_val = None
    if _ai_ok("width"):
        if recipe.width and options.get("enable_width", True):
            _width_val = recipe.width
        elif options.get("enable_width", True):
            logger.info(f"[JOB {job_id}] AI: applying default width mix=0.3 (not detected)")
            _width_val = {"mix": 0.3, "delay_ms": 12.0, "detune_cents": 4.0}
    else:
        logger.info(f"[JOB {job_id}] AI: skipping width")

    # Log what was detected vs skipped
    logger.info(
        f"[JOB {job_id}] Final recipe (after AI review): "
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
            processed = apply_vocal_layers(processed, sr, filtered_profile)
            logger.info(f"[JOB {job_id}] Vocal layers applied → output is now stereo")
        except ValueError:
            pass
        except Exception as e:
            logger.warning(f"[JOB {job_id}] Vocal layer replication failed: {e}")
    elif harmony_profile is not None and harmony_profile.intervals_semitones and options.get("enable_harmony", True):
        # Fallback to old harmony if vocal_layers not available
        try:
            logger.info(f"[JOB {job_id}] Applying legacy harmony layers…")
            processed = apply_harmony(processed, sr, harmony_profile)
        except Exception as e:
            logger.warning(f"[JOB {job_id}] Harmony application failed: {e}")

    # Apply delay if detected and enabled
    if delay_info is not None and delay_info.get("delay_ms", 0) > 0 and options.get("enable_delay", True):
        try:
            fb = 0.25
            mix_val = float(delay_info.get("_mix", 0.25))
            if processed.ndim == 2:
                # Stereo (N, 2) — apply delay per channel
                ch0 = apply_delay(processed[:, 0], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val)
                ch1 = apply_delay(processed[:, 1], sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val)
                processed = np.stack([ch0, ch1], axis=1)
            else:
                processed = apply_delay(processed, sr, delay_ms=delay_info["delay_ms"], feedback=fb, mix=mix_val)
            logger.info(
                f"[JOB {job_id}] Delay applied: {delay_info.get('type')} {delay_info.get('delay_ms'):.2f} ms"
            )
        except Exception as e:
            logger.warning(f"[JOB {job_id}] Delay application failed: {e}")

    # NOTE: reverb is already applied inside apply_chain via reverb_final (converted from
    # reverb_profile_auto). A second apply_reverb here would double the reverb, so it is removed.

    # ── Iterative AI Refinement Pass ────────────────────────────────────────
    # Compute spectral distance between current output and reference,
    # ask the LLM for a targeted correction EQ, apply it as a second pass.
    _progress(current_job, 72, "AI refinement pass…")
    refinement_info: dict = {}
    try:
        from api.ai_audio import ai_refinement_eq
        from processor.utils.metrics import compute_spectral_distance as _csd

        # Need a temp file for spectral distance computation
        import tempfile as _tf
        _tmp_path = Path(_tf.mktemp(suffix=".wav"))
        save_wav(_tmp_path, processed, sr)

        _sd = float(_csd(ref_for_analysis, _tmp_path, sr=sr))
        logger.info(f"[JOB {job_id}] Pre-refinement spectral distance: {_sd:.4f}")

        if _sd > 0.08:   # only refine if there's a meaningful gap
            # Build profiles for the AI
            _proc_mono = processed if processed.ndim == 1 else (
                processed.mean(axis=0) if processed.shape[1] < processed.shape[0]
                else processed.mean(axis=1)
            )
            _proc_profile = _spectral_profile(_proc_mono, "output")
            _ref_profile_refine = _spectral_profile(_ref_mono2, "reference")

            refine_result = ai_refinement_eq(_proc_profile, _ref_profile_refine, _sd)
            correction_bands = refine_result.get("correction_eq", [])

            if correction_bands:
                from processor.dsp.eq import EqBand, apply_eq as _apply_eq
                cb = [EqBand(f=b["f"], gain_db=b["gain_db"], q=b["q"])
                      for b in correction_bands]
                logger.info(f"[JOB {job_id}] Applying {len(cb)} AI correction EQ bands")

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
                    "summary": refine_result.get("correction_summary", ""),
                }
                logger.info(f"[JOB {job_id}] Refinement: {refine_result.get('correction_summary', '')}")
            else:
                logger.info(f"[JOB {job_id}] AI refinement: no correction needed")
                refinement_info = {"pre_distance": _sd, "correction_bands": 0,
                                   "summary": refine_result.get("correction_summary", "")}
        else:
            logger.info(f"[JOB {job_id}] Spectral distance {_sd:.4f} < 0.08 — skipping refinement")
            refinement_info = {"pre_distance": _sd, "correction_bands": 0,
                               "summary": "Already close to reference — no correction needed."}

        try:
            _tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as _ref_err:
        logger.warning(f"[JOB {job_id}] Iterative refinement skipped: {_ref_err}")

    _progress(current_job, 75, "Applying modulation & post-effects…")
    # Optional: ML spectral refiner (after DSP)
    use_spectral_refiner = options.get("use_spectral_refiner", True)
    if use_spectral_refiner:
        refiner = get_refiner()
        if refiner.is_available():
            _progress(current_job, 78, "Neural spectral refinement…")
            print(f"Applying spectral refiner...")
            processed_before_refiner = processed.copy()
            processed = refiner.refine(processed, ref_audio=ref_audio, sr=sr)
            # Safety check: if refiner returned zeros or very quiet audio, use original
            max_before = np.max(np.abs(processed_before_refiner))
            max_after = np.max(np.abs(processed))
            if max_after < max_before * 0.01:  # If output is 100x quieter, something went wrong
                print(f"⚠ Spectral refiner output too quiet. Using DSP output instead.")
                processed = processed_before_refiner
            print(f"✓ Spectral refinement complete")
        else:
            print(f"⚠ Spectral refiner not available. Skipping neural refinement.")

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
        ref_mono = ref_audio if ref_audio.ndim == 1 else np.mean(ref_audio, axis=0)
        ref_rms = float(np.sqrt(np.mean(ref_mono ** 2)))
        out_mono = processed if processed.ndim == 1 else np.mean(processed, axis=0 if processed.shape[0] < processed.shape[1] else 1)
        out_rms = float(np.sqrt(np.mean(out_mono ** 2)))
        if ref_rms > 1e-9 and out_rms > 1e-9:
            gain = ref_rms / out_rms
            # Safety: cap gain to ±12dB to avoid absurd amplification
            gain = float(np.clip(gain, 10 ** (-12 / 20), 10 ** (12 / 20)))
            processed = processed * gain
            print(f"  RMS match: applied {20 * np.log10(gain):.1f}dB gain to match reference loudness")
    except Exception as e:
        print(f"⚠ RMS match failed: {e}")

    # Final peak-limit to prevent clipping after gain
    max_val = np.max(np.abs(processed))
    if max_val > 1e-9:
        processed = processed / max(max_val, 1.0) * 0.95
    else:
        processed = dry_audio / (np.max(np.abs(dry_audio)) + 1e-9) * 0.95

    # Safety: if still too quiet after all the above, fall back
    try:
        from processor.dsp.chain import normalize_rms, normalize_peak
        rms_post = np.sqrt(np.mean(processed ** 2))
        if rms_post < 0.02:
            processed = normalize_rms(dry_audio, target_db=-18.0)
            processed = normalize_peak(processed, peak=0.95)
            print("⚠ Output still quiet; fell back to normalized dry vocal.")
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
    if harmony_profile is not None:
        recipe_dict["harmony"] = {
            "intervals_semitones": harmony_profile.intervals_semitones,
            "strengths": harmony_profile.strengths,
            "pans": harmony_profile.pans,
            "timing_offsets": harmony_profile.timing_offsets,
        }
    if delay_info is not None:
        recipe_dict["delay"] = {
            "type": delay_info.get("type"),
            "delay_ms": delay_info.get("delay_ms"),
            "confidence": delay_info.get("confidence"),
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
    if ai_review_result:
        recipe_dict["ai_review"] = {
            "summary": ai_review_result.get("summary", ""),
            "reasoning": ai_review_result.get("reasoning", {}),
            "adjustments": ai_review_result.get("adjustments", {}),
        }
    # Include dry vocal profile so the UI and debugging can see what was compared
    try:
        recipe_dict["dry_vocal_profile"] = _dry_stats
        recipe_dict["ref_vocal_profile"] = _ref_stats
    except Exception:
        pass
    if refinement_info:
        recipe_dict["refinement"] = refinement_info

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
    print(f"  Metrics: total={metrics['processing_time_total']:.2f}s, RVC={metrics['processing_time_rvc']:.2f}s, DSP={metrics['processing_time_dsp']:.2f}s")

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

