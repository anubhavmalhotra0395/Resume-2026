"""Download RVC model from HuggingFace."""
import requests
from pathlib import Path
from tqdm import tqdm

url = "https://huggingface.co/AnonUser123/generic_rvc_voice/resolve/main/rvc_generic.pth"
output_path = Path("models/rvc/pretrained.pth")

output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading RVC model from HuggingFace...")
print(f"URL: {url}")
print(f"Output: {output_path.absolute()}")

try:
    # HuggingFace requires proper headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"\n✓ Model downloaded successfully!")
    print(f"  Location: {output_path.absolute()}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("\nAlternative: Download manually from:")
    print(f"  {url}")
    print(f"  And place at: {output_path.absolute()}")
    exit(1)

