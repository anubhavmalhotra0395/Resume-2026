#!/usr/bin/env python3
"""
Download HuBERT model checkpoint from HuggingFace and save locally.

This script downloads the HuBERT base model and saves it as a .pt file
for faster loading in the future.
"""
import sys
from pathlib import Path

try:
    from transformers import Wav2Vec2Model
    import torch
except ImportError:
    print("Error: transformers and torch are required.")
    print("Install with: pip install transformers torch")
    sys.exit(1)

def download_hubert(output_path: Path):
    """Download HuBERT model and save to local file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading HuBERT base model from HuggingFace...")
    print("This may take a few minutes (model is ~300MB)...")
    print()
    
    try:
        # Load model from HuggingFace
        model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
        
        # Save state dict to local file
        print(f"Saving to: {output_path}")
        torch.save(model.state_dict(), output_path)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ HuBERT model downloaded and saved!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size:.1f} MB")
        print()
        print("The model will now load from this local file on next use.")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading HuBERT: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    hubert_path = Path("models/hubert/hubert-base-ls960.pt")
    
    if hubert_path.exists():
        response = input(f"HuBERT model already exists at {hubert_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    success = download_hubert(hubert_path)
    sys.exit(0 if success else 1)

