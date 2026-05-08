#!/usr/bin/env python3
"""
Download RVC model from HuggingFace (requires authentication).

Run this after logging into HuggingFace:
  huggingface-cli login
"""
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, login

def download_model():
    """Download the RVC model from HuggingFace."""
    model_path = Path("models/rvc/pretrained.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading RVC model from HuggingFace...")
    print("Repository: AnonUser123/generic_rvc_voice")
    print("File: rvc_generic.pth")
    print()
    
    # Check if user is logged in
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"Logged in as: {user.get('name', 'Unknown')}")
    except Exception:
        print("⚠ Not logged in to HuggingFace")
        print("Run: huggingface-cli login")
        print("Or: python -c 'from huggingface_hub import login; login()'")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Download using huggingface_hub
        print("Downloading...")
        downloaded_path = hf_hub_download(
            repo_id="AnonUser123/generic_rvc_voice",
            filename="rvc_generic.pth",
            local_dir=str(model_path.parent),
        )
        
        # Rename to pretrained.pth if needed
        downloaded_file = Path(downloaded_path)
        if downloaded_file.name != "pretrained.pth":
            final_path = model_path.parent / "pretrained.pth"
            if downloaded_file.exists():
                downloaded_file.rename(final_path)
                print(f"✓ Model saved to: {final_path.absolute()}")
            else:
                print(f"✓ Model saved to: {downloaded_file.absolute()}")
        else:
            print(f"✓ Model saved to: {downloaded_file.absolute()}")
        
        # Check file size
        final_file = model_path if model_path.exists() else downloaded_file
        if final_file.exists():
            file_size = final_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Model size: {file_size:.2f} MB")
            print()
            print("Model downloaded successfully!")
            return True
        else:
            print("⚠ Model file not found after download")
            return False
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print()
        print("Possible issues:")
        print("1. Repository is private - you need access")
        print("2. Not logged in - run: huggingface-cli login")
        print("3. Repository doesn't exist or file name is wrong")
        print()
        print("Try downloading manually:")
        print("1. Visit: https://huggingface.co/AnonUser123/generic_rvc_voice")
        print("2. Download rvc_generic.pth")
        print(f"3. Place it at: {model_path.absolute()}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

