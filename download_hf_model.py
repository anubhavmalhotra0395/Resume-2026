#!/usr/bin/env python3
"""Download RVC model from HuggingFace using huggingface_hub."""
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

def download_model():
    """Download the RVC model from HuggingFace."""
    model_path = Path("models/rvc/pretrained.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Downloading RVC model from HuggingFace...")
    print("Repository: AnonUser123/generic_rvc_voice")
    print("File: rvc_generic.pth")
    print()
    
    try:
        # Download using huggingface_hub
        downloaded_path = hf_hub_download(
            repo_id="AnonUser123/generic_rvc_voice",
            filename="rvc_generic.pth",
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        
        # Rename to pretrained.pth if needed
        downloaded_file = Path(downloaded_path)
        if downloaded_file.name != "pretrained.pth":
            final_path = model_path.parent / "pretrained.pth"
            downloaded_file.rename(final_path)
            print(f"✓ Model saved to: {final_path.absolute()}")
        else:
            print(f"✓ Model saved to: {downloaded_file.absolute()}")
        
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Model size: {file_size:.2f} MB")
        print()
        print("Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print()
        print("Trying alternative method with direct download...")
        
        # Fallback: try direct download
        try:
            import requests
            url = "https://huggingface.co/AnonUser123/generic_rvc_voice/resolve/main/rvc_generic.pth"
            print(f"Downloading from: {url}")
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"File size: {total_size / (1024*1024):.2f} MB")
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✓ Model downloaded to: {model_path.absolute()}")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative download also failed: {e2}")
            return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

