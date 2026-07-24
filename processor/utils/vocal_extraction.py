"""
Extract vocals from full mix using MDX-Net (audio-separator) with Demucs fallback.

Priority:
  1. audio-separator with Kim Vocal 2 MDX-Net model  — best quality, less bleed
  2. Demucs htdemucs CLI                             — good quality, reliable
  3. Simple bandpass filter                          — last resort fallback
"""
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

def _normalize_stem(stem: str) -> str:
    return stem.lower().replace("-", "_").replace(" ", "_")

def _is_bad_vocal_candidate(stem: str) -> bool:
    s = _normalize_stem(stem)
    bad_tokens = (
        "no_vocal",
        "novocal",
        "without_vocal",
        "instrument",
        "instrum",
        "accomp",
        "karaoke",
        "minus",
        "music",
        "backing",
    )
    return any(tok in s for tok in bad_tokens)

def _is_good_vocal_candidate(stem: str) -> bool:
    s = _normalize_stem(stem)
    good_tokens = ("vocal", "vocals", "voice", "lead")
    return any(tok in s for tok in good_tokens) and not _is_bad_vocal_candidate(s)


# ---------------------------------------------------------------------------
# MDX-Net via audio-separator (best quality)
# ---------------------------------------------------------------------------

def extract_vocals_mdx(input_path: Path, output_path: Path) -> bool:
    """
    Use audio-separator (MDX-Net / Kim Vocal 2) for high-quality vocal extraction.
    Falls back gracefully if the package or model is unavailable.
    """
    try:
        from audio_separator.separator import Separator  # type: ignore
    except ImportError:
        logger.info("audio-separator not installed — skipping MDX-Net")
        return False

    try:
        out_dir = output_path.parent / "_mdx_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)

        sep = Separator(
            output_dir=str(out_dir),
            output_format="wav",
            normalization_threshold=0.9,
            # Use the best available MDX vocal model — auto-downloads on first run
            mdx_params={
                "hop_length": 1024,
                "segment_size": 256,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": True,
            },
        )
        sep.load_model(model_filename="Kim_Vocal_2.onnx")
        outputs = sep.separate(str(input_path))

        # audio-separator returns list of output paths; find the Vocals stem
        vocals_file: Optional[Path] = None
        for p in outputs:
            ppath = Path(p)
            if not ppath.is_absolute():
                ppath = out_dir / ppath
            if _is_good_vocal_candidate(ppath.stem):
                vocals_file = ppath
                break

        # Final fallback: recursive search in output dir
        if vocals_file is None or not vocals_file.exists():
            for cand in out_dir.rglob("*.wav"):
                if _is_good_vocal_candidate(cand.stem):
                    vocals_file = cand
                    break

        if vocals_file and vocals_file.exists():
            shutil.copy2(vocals_file, output_path)
            shutil.rmtree(out_dir, ignore_errors=True)
            logger.info("MDX-Net extraction succeeded using stem: %s → %s", vocals_file.name, output_path.name)
            return True

        shutil.rmtree(out_dir, ignore_errors=True)
        return False

    except Exception as e:
        logger.warning(f"MDX-Net extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Demucs CLI (reliable fallback)
# ---------------------------------------------------------------------------

def extract_vocals_demucs_cli(input_path: Path, output_path: Path) -> bool:
    """
    Use Demucs CLI to extract vocals.
    --segment 7 / --overlap 0.1 / -j 2 for faster CPU processing.
    """
    try:
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "--two-stems=vocals",
            "-n", "htdemucs",
            "--segment", "7",
            "--overlap", "0.1",
            "-j", "2",
            "-o", str(output_path.parent),
            "--device", "cpu",
            str(input_path),
        ]
        # CPU demucs can take several minutes depending on track length.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            logger.warning(
                "Demucs exited %s.\nstdout:\n%s\nstderr:\n%s",
                result.returncode,
                (result.stdout or "")[-1200:],
                (result.stderr or "")[-1200:],
            )
            return False

        model_dir = output_path.parent / "htdemucs"
        if not model_dir.exists():
            return False
        vocals_file: Optional[Path] = None
        for cand in model_dir.rglob("vocals.wav"):
            vocals_file = cand
            break
        if vocals_file is None or not vocals_file.exists():
            return False

        shutil.copy2(vocals_file, output_path)
        shutil.rmtree(model_dir, ignore_errors=True)
        logger.info(f"Demucs extraction succeeded → {output_path.name}")
        return True

    except Exception as e:
        logger.warning(f"Demucs CLI extraction failed: {e}")
        return False


def extract_vocals_demucs(input_path: Path, output_path: Path) -> bool:
    return extract_vocals_demucs_cli(input_path, output_path)


# ---------------------------------------------------------------------------
# Simple bandpass fallback
# ---------------------------------------------------------------------------

def extract_vocals_simple(input_path: Path, output_path: Path) -> bool:
    """Last-resort bandpass filter isolation (80 Hz – 8 kHz)."""
    try:
        y, sr = librosa.load(str(input_path), sr=44100, mono=False)
        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        b, a = butter(4, [80 / nyq, 8000 / nyq], btype="band")
        # Keep the stereo image — layer analysis reads doubling from L/R
        y = filtfilt(b, a, y, axis=-1)
        y = y / (np.max(np.abs(y)) + 1e-9)
        sf.write(str(output_path), y.T if y.ndim > 1 else y, sr)
        return True
    except Exception as e:
        logger.warning(f"Simple vocal extraction failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_vocals(
    input_path: Path,
    output_path: Path,
    force_demucs: bool = True,
) -> Optional[Path]:
    """
    Extract vocals from a full mix.

    Priority order:
      1. MDX-Net (audio-separator / Kim Vocal 2) — best quality
      2. Demucs htdemucs                          — solid fallback
      3. Simple bandpass filter                   — last resort

    Returns the output path on success, None on total failure.

    Set APP_LOW_MEMORY=1 (e.g. Render free tier, 512 MB) to skip the ML
    separators entirely — they OOM-kill the process there — and go straight
    to the bandpass isolation. Lower quality, but the service stays alive.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import os
    if os.environ.get("APP_LOW_MEMORY", "0") == "1":
        print("  APP_LOW_MEMORY=1 — using bandpass vocal isolation (no ML separation)")
        if extract_vocals_simple(input_path, output_path):
            return output_path
        return None

    # 1. Try MDX-Net first
    print("  Attempting MDX-Net (Kim Vocal 2) vocal extraction…")
    if extract_vocals_mdx(input_path, output_path):
        print("  ✓ MDX-Net extraction complete")
        return output_path

    # 2. Fall back to Demucs
    print("  MDX-Net unavailable — falling back to Demucs…")
    if force_demucs and extract_vocals_demucs(input_path, output_path):
        print("  ✓ Demucs extraction complete")
        return output_path

    # 3. Last resort
    print("  ⚠ Demucs failed — using simple bandpass filter")
    if extract_vocals_simple(input_path, output_path):
        return output_path

    return None

