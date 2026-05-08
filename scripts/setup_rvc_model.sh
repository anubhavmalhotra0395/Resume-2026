#!/bin/bash
# Setup script to download or create RVC model
# Can be run inside Docker container

MODEL_PATH="models/rvc/pretrained.pth"
MODEL_DIR=$(dirname "$MODEL_PATH")

# Create directory
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

echo "Setting up RVC model..."
echo ""
echo "Option 1: Create placeholder model (for testing)"
echo "Option 2: Download from URL (if provided)"
echo ""

# Try to create placeholder if Python is available
if command -v python3 &> /dev/null; then
    python3 scripts/create_placeholder_rvc_model.py
elif command -v python &> /dev/null; then
    python scripts/create_placeholder_rvc_model.py
else
    echo "Python not found. Please:"
    echo "1. Install Python, OR"
    echo "2. Download a model manually and place at: $MODEL_PATH"
    echo "3. Or run this inside Docker: docker compose exec worker python scripts/create_placeholder_rvc_model.py"
fi

