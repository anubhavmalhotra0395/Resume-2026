import subprocess
from pathlib import Path

from fastapi import HTTPException

from processor.config import settings


def probe_duration_seconds(path: Path) -> float:
    """Use ffprobe to fetch duration in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def validate_file(path: Path, is_rvc_job: bool = False) -> None:
    """
    Validate audio file with comprehensive checks.
    
    Args:
        path: Path to audio file
        is_rvc_job: If True, applies stricter limits for RVC processing
    """
    if not path.exists():
        raise HTTPException(status_code=400, detail="Upload missing")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_file_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB (max: {settings.max_file_mb}MB)"
        )

    duration = probe_duration_seconds(path)
    
    # Stricter limits for RVC jobs (RVC is more computationally expensive)
    max_duration = 30.0 if is_rvc_job else settings.max_duration_seconds
    if duration > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"File too long: {duration:.1f}s (max: {max_duration:.1f}s for {'RVC' if is_rvc_job else 'standard'} jobs)"
        )
    
    # Minimum length check
    min_duration = 0.5
    if duration < min_duration:
        raise HTTPException(
            status_code=400,
            detail=f"File too short: {duration:.1f}s (min: {min_duration:.1f}s)"
        )


def validate_sample_rate(path: Path, target_sr: int = 44100) -> bool:
    """
    Check if file needs resampling.
    
    Args:
        path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        True if resampling needed, False otherwise
    """
    try:
        import subprocess
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        out = subprocess.check_output(cmd).decode().strip()
        file_sr = int(float(out))
        return file_sr != target_sr
    except Exception:
        # If we can't check, assume resampling is needed
        return True


def ensure_mono(path: Path) -> bool:
    """
    Check if file is mono, or needs conversion.
    
    Args:
        path: Path to audio file
        
    Returns:
        True if mono, False if stereo/multi-channel
    """
    try:
        import subprocess
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        out = subprocess.check_output(cmd).decode().strip()
        channels = int(out)
        return channels == 1
    except Exception:
        # If we can't check, assume conversion is needed
        return False


