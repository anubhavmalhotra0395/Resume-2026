#!/usr/bin/env python3
"""
Attempt to download a real RVC model from public sources.

This script tries to find and download a publicly available RVC model.
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def try_download_huggingface(model_id: str, output_path: Path):
    """Try to download from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        logging.info(f"Attempting to download from HuggingFace: {model_id}")
        
        # Try to find .pth files
        try:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="*.pth",
                local_dir=output_path.parent,
            )
            logging.info(f"✓ Downloaded model from HuggingFace")
            return True
        except Exception as e:
            logging.warning(f"Could not find .pth file: {e}")
            return False
    except ImportError:
        logging.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logging.error(f"Failed to download from HuggingFace: {e}")
        return False

def main():
    model_path = Path("models/rvc/pretrained.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            return
    
    print("=" * 60)
    print("RVC Model Downloader")
    print("=" * 60)
    print()
    print("This script attempts to download a real RVC model.")
    print()
    print("Note: Most RVC models are:")
    print("  - Large (100MB+)")
    print("  - Require specific training")
    print("  - May have licensing restrictions")
    print()
    
    # Try common public sources
    sources = [
        ("RVC-Project/Retrieval-based-Voice-Conversion-WebUI", "Example model"),
    ]
    
    print("Available options:")
    print("1. Create placeholder model (for testing)")
    print("2. Download from HuggingFace (if model ID provided)")
    print("3. Manual download instructions")
    print()
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Create placeholder
        try:
            from scripts.create_placeholder_rvc_model import create_placeholder_model
            create_placeholder_model(model_path)
        except Exception as e:
            print(f"Error: {e}")
            print("Run: python scripts/create_placeholder_rvc_model.py")
    
    elif choice == "2":
        model_id = input("Enter HuggingFace model ID (e.g., 'user/model-name'): ").strip()
        if model_id:
            if try_download_huggingface(model_id, model_path):
                print(f"✓ Model downloaded to {model_path}")
            else:
                print("Failed to download. Try manual download.")
        else:
            print("No model ID provided.")
    
    elif choice == "3":
        print()
        print("Manual Download Instructions:")
        print("=" * 60)
        print("1. Visit: https://huggingface.co/models?search=rvc")
        print("2. Find a model you want to use")
        print("3. Download the .pth file")
        print(f"4. Place it at: {model_path.absolute()}")
        print()
        print("Or use the download script:")
        print(f"  python scripts/download_rvc_model.py <model_url>")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()

