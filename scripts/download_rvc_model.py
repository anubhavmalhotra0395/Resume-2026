#!/usr/bin/env python3
"""
Helper script to download or set up an RVC model.

Usage:
    python scripts/download_rvc_model.py [model_url_or_path]

If no argument provided, shows instructions for manual setup.
"""
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def download_from_url(url: str, output_path: Path) -> bool:
    """Download model from URL."""
    try:
        import requests
        from tqdm import tqdm
        
        logging.info(f"Downloading RVC model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        logging.info(f"✓ Model downloaded to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return False

def copy_local_model(source_path: str, output_path: Path) -> bool:
    """Copy local model file to the expected location."""
    try:
        import shutil
        source = Path(source_path)
        if not source.exists():
            logging.error(f"Source file not found: {source_path}")
            return False
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)
        logging.info(f"✓ Model copied to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to copy model: {e}")
        return False

def main():
    model_path = Path("models/rvc/pretrained.pth")
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Check if it's a URL
        if arg.startswith("http://") or arg.startswith("https://"):
            if download_from_url(arg, model_path):
                print(f"\n✓ RVC model ready at: {model_path}")
                print("Note: You still need to implement the inference code in rvc_refiner.py")
                return
        else:
            # Assume it's a local file path
            if copy_local_model(arg, model_path):
                print(f"\n✓ RVC model ready at: {model_path}")
                print("Note: You still need to implement the inference code in rvc_refiner.py")
                return
    
    # Show instructions
    print("=" * 60)
    print("RVC Model Setup Instructions")
    print("=" * 60)
    print()
    print("To use RVC voice conversion, you need a trained model file.")
    print()
    print("Option 1: Download from HuggingFace")
    print("  - Visit: https://huggingface.co/models?search=rvc")
    print("  - Download a .pth or .onnx model file")
    print("  - Run: python scripts/download_rvc_model.py <model_url>")
    print()
    print("Option 2: Use your own trained model")
    print("  - Place your .pth or .onnx file at: models/rvc/pretrained.pth")
    print("  - Or run: python scripts/download_rvc_model.py <path_to_your_model>")
    print()
    print("Option 3: Train your own model")
    print("  - Use RVC training scripts (e.g., from RVC-Project/RVC)")
    print("  - Train on your target voice dataset")
    print("  - Export the model and place it at: models/rvc/pretrained.pth")
    print()
    print("Important:")
    print("  - The model file must match your RVC framework version")
    print("  - You'll need to implement the inference code in:")
    print("    processor/ml_refine/rvc_refiner.py")
    print("  - See the _load_rvc_model_pytorch() function for details")
    print()
    print("Current model path:", model_path.absolute())
    print("Model exists:", model_path.exists())
    print("=" * 60)

if __name__ == "__main__":
    main()

