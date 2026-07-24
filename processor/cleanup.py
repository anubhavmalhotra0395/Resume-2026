import shutil
import time
from pathlib import Path

from processor.config import settings


def sweep_old_files(now: float | None = None) -> None:
    """Remove files older than retention window from inputs/outputs.

    Never raises: stem-separator tools (demucs, audio-separator) drop
    directories into inputs/, and a cleanup hiccup must not fail a job
    that already produced its output.
    """
    cutoff_seconds = settings.delete_after_hours * 3600
    now = now or time.time()
    for folder in (settings.inputs_dir, settings.outputs_dir):
        for path in folder.glob("*"):
            try:
                if now - path.stat().st_mtime > cutoff_seconds:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
            except OSError:
                continue


if __name__ == "__main__":
    sweep_old_files()


