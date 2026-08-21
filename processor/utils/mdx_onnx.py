"""
Torch-free MDX-Net vocal separation.

Runs UVR's Kim_Vocal_2.onnx directly with onnxruntime — no torch, no demucs,
no audio-separator. The model file (~66 MB) is downloaded once into
<storage_root>/models and reused.

Model parameters come from UVR's model_data.json for Kim_Vocal_2
(md5 970b3f9492014d18fefeedfe4773cb42):
    n_fft = 7680, dim_f = 3072, dim_t = 2**8 = 256, hop = 1024
    primary_stem = Vocals, compensate = 1.009
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/Kim_Vocal_2.onnx"
)
MODEL_UVR_HASH = "970b3f9492014d18fefeedfe4773cb42"  # UVR-style: md5 of last 10240000 bytes
MODEL_FILENAME = "Kim_Vocal_2.onnx"


def _uvr_model_hash(path: Path) -> str:
    with open(path, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()

SR = 44100
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATE = 1.009

# Samples per model window: center=True STFT of this length gives exactly
# DIM_T frames (1 + chunk//hop). The N_FFT//2 head/tail of each window is
# corrupted by padding, so consecutive windows overlap by 2*TRIM and only
# the clean middle GEN_SIZE samples of each are kept.
CHUNK = HOP * (DIM_T - 1)          # 261120
TRIM = N_FFT // 2                  # 3840
GEN_SIZE = CHUNK - 2 * TRIM        # 253440

_session = None  # cached onnxruntime session


def _models_dir() -> Path:
    from processor.config import settings

    d = settings.storage_root / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_model(model_dir: Optional[Path] = None) -> Path:
    """Download Kim_Vocal_2.onnx if missing; verify by md5. Returns the path."""
    model_path = (model_dir or _models_dir()) / MODEL_FILENAME
    if model_path.exists():
        return model_path

    import requests

    logger.info(f"Downloading MDX model to {model_path} …")
    tmp = model_path.with_suffix(".part")
    with requests.get(MODEL_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    got = _uvr_model_hash(tmp)
    if got != MODEL_UVR_HASH:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"MDX model download hash mismatch: {got} != {MODEL_UVR_HASH}"
        )
    os.replace(tmp, model_path)
    logger.info("MDX model downloaded and verified")
    return model_path


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        model_path = ensure_model()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # The arena allocator caches every block it ever allocates, which on
        # this model grows to well over a gigabyte and is never returned.
        # Disabling it trades a little speed for a far smaller RSS ceiling —
        # the difference between fitting on a small host and being OOM-killed.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        _session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
    return _session


def _stft(x: np.ndarray) -> np.ndarray:
    """(2, N) float32 -> (4, DIM_F, T) float32 [L.re, L.im, R.re, R.im]."""
    spec = librosa.stft(
        x, n_fft=N_FFT, hop_length=HOP, window="hann", center=True, pad_mode="reflect"
    )  # (2, N_FFT//2+1, T) complex
    spec = spec[:, :DIM_F, :]
    return np.concatenate([spec.real, spec.imag], axis=0).astype(np.float32)[
        [0, 2, 1, 3]  # interleave to [ch0_re, ch0_im, ch1_re, ch1_im]
    ]


def _istft(spec4: np.ndarray, length: int) -> np.ndarray:
    """(4, DIM_F, T) float32 -> (2, length) float32."""
    full = np.zeros((2, N_FFT // 2 + 1, spec4.shape[2]), dtype=np.complex64)
    full[0, :DIM_F] = spec4[0] + 1j * spec4[1]
    full[1, :DIM_F] = spec4[2] + 1j * spec4[3]
    return librosa.istft(
        full, n_fft=N_FFT, hop_length=HOP, window="hann", center=True, length=length
    )


def separate_vocals(input_path: Path, output_path: Path, progress_cb=None) -> bool:
    """Extract the vocal stem of input_path into output_path (stereo 44.1k wav).

    progress_cb(done_windows, total_windows) is called after each window so
    callers can report real progress instead of guessing.
    """
    try:
        session = _get_session()
    except Exception as e:
        logger.warning(f"MDX model unavailable: {e}")
        return False

    try:
        mix, _ = librosa.load(str(input_path), sr=SR, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        n_samples = mix.shape[1]

        # Pad so the clean (trimmed) regions tile the whole track.
        n_windows = -(-n_samples // GEN_SIZE)  # ceil
        padded = np.zeros((2, TRIM + n_windows * GEN_SIZE + TRIM), dtype=np.float32)
        padded[:, TRIM : TRIM + n_samples] = mix

        input_name = session.get_inputs()[0].name
        out = np.zeros_like(padded)
        for w in range(n_windows):
            start = w * GEN_SIZE
            chunk = padded[:, start : start + CHUNK]
            if chunk.shape[1] < CHUNK:  # last window
                chunk = np.pad(chunk, ((0, 0), (0, CHUNK - chunk.shape[1])))
            spec = _stft(chunk)[None]  # (1, 4, DIM_F, DIM_T)
            pred = session.run(None, {input_name: spec})[0][0]
            wav = _istft(pred, CHUNK)
            # keep only the clean middle of the window
            out[:, start + TRIM : start + TRIM + GEN_SIZE] = wav[:, TRIM : TRIM + GEN_SIZE]
            if progress_cb:
                try:
                    progress_cb(w + 1, n_windows)
                except Exception:
                    pass

        vocals = out[:, TRIM : TRIM + n_samples] * COMPENSATE
        sf.write(str(output_path), vocals.T, SR, subtype="FLOAT")
        return True
    except Exception as e:
        logger.error(f"MDX separation failed: {e}")
        return False
