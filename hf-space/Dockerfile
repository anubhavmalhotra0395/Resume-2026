# VocalForge on Hugging Face Spaces — wraps the prebuilt slim image from
# GHCR (published by the repo's GitHub Action), so HF never rebuilds the
# 3.3 GB stack. Full quality: demucs + MDX-Net separation included.
FROM ghcr.io/anubhavmalhotra0395/resume-2026:latest

# HF Spaces runs as uid 1000 with /data as the only writable mount-ish dir.
ENV APP_STORAGE_ROOT=/data \
    HF_HOME=/data/hf-cache \
    XDG_CACHE_HOME=/data/cache \
    TORCH_HOME=/data/torch-cache \
    PORT=7860
EXPOSE 7860
