import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from processor.config import settings


def run_ffmpeg_normalize(src: Path, dst: Path, channels: int = 1) -> None:
    """Convert to target sample rate and loudness-normalized WAV.

    channels=1 (default) downmixes to mono for the processing chain;
    channels=2 preserves the stereo image — required by layer analysis,
    which reads doubling from L/R decorrelation.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        str(channels),
        "-ar",
        str(settings.sample_rate),
        "-af",
        f"loudnorm=I={settings.loudness_target_lufs}:TP=-1.5:LRA=11",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def load_wav_stereo(path: Path) -> Tuple[np.ndarray, int]:
    """Load preserving channels, shaped (2, N). Mono files are duplicated.

    Layer analysis needs the true stereo image — the mono loader above
    silently destroys the L/R decorrelation that reveals vocal doubling.
    """
    audio, sr = sf.read(path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    else:
        audio = audio.T  # soundfile gives (N, ch) → (ch, N)
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
    return audio.astype(np.float32), sr


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (N,) for mono or (N, channels) for stereo.
    # If we received (channels, N) — e.g. from apply_vocal_layers — transpose it.
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    sf.write(path, audio, sr)


