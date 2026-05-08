import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import HTTPException

from processor.config import settings


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL")


def fetch_audio_from_url(url: str) -> Path:
    """Download audio from YouTube/Spotify/etc. to a temp WAV file via yt-dlp."""
    _validate_url(url)
    out_path = settings.inputs_dir / f"{uuid.uuid4()}_ref_dl.%(ext)s"
    # yt-dlp handles normalization to wav for us; we still normalize later via ffmpeg.
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(out_path),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail="Failed to download reference URL") from exc

    # yt-dlp will replace %(ext)s with wav
    wav_path = Path(str(out_path).replace("%(ext)s", "wav"))
    if not wav_path.exists():
        raise HTTPException(status_code=400, detail="Download failed")
    return wav_path

