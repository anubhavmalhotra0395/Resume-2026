"""
Extract vocals from a full mix.

Priority:
  1. MDX-Net Kim_Vocal_2 via onnxruntime (processor/utils/mdx_onnx.py)
     — torch-free, ~67 MB model downloaded on first use
  2. Simple bandpass filter — last resort fallback

Results are cached by content hash: separating the same reference twice
(analyze-layers first, then the job) is instant the second time.
"""
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


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

def extract_vocals(input_path: Path, output_path: Path, progress_cb=None) -> Optional[Path]:
    """
    Extract vocals from a full mix. Returns the output path on success,
    None on total failure.

    progress_cb(done, total) is forwarded to the separator so callers can
    report real progress.

    Set APP_LOW_MEMORY=1 to skip ML separation entirely and go straight to
    the bandpass isolation. Lower quality, but survives tiny hosts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = output_path.parent / "_sep_cache"
    cache_hit = None
    try:
        h = hashlib.sha256()
        with open(input_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_hit = cache_dir / f"{h.hexdigest()}.wav"
        if cache_hit.exists():
            shutil.copy2(cache_hit, output_path)
            print("  ✓ Separation cache hit — skipping extraction")
            return output_path
    except Exception:
        cache_hit = None

    def _store_cache(path):
        try:
            if cache_hit is not None:
                shutil.copy2(path, cache_hit)
        except Exception:
            pass

    if os.environ.get("APP_LOW_MEMORY", "0") == "1":
        print("  APP_LOW_MEMORY=1 — using bandpass vocal isolation (no ML separation)")
        if extract_vocals_simple(input_path, output_path):
            return output_path
        return None

    # 1. MDX-Net (onnxruntime)
    print("  Extracting vocals with MDX-Net (Kim Vocal 2)…")
    from processor.utils.mdx_onnx import separate_vocals
    if separate_vocals(input_path, output_path, progress_cb=progress_cb):
        print("  ✓ MDX-Net extraction complete")
        _store_cache(output_path)
        return output_path

    # 2. Last resort
    print("  ⚠ MDX-Net failed — using simple bandpass filter")
    if extract_vocals_simple(input_path, output_path):
        return output_path

    return None
