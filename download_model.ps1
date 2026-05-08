# Download RVC model from HuggingFace
$modelPath = "models\rvc\pretrained.pth"
$url = "https://huggingface.co/AnonUser123/generic_rvc_voice/resolve/main/rvc_generic.pth"

# Create directory if it doesn't exist
New-Item -ItemType Directory -Force -Path (Split-Path $modelPath) | Out-Null

Write-Host "Downloading RVC model from HuggingFace..."
Write-Host "URL: $url"
Write-Host "Destination: $modelPath"
Write-Host ""

try {
    # Use BITS (Background Intelligent Transfer Service) for more reliable downloads
    $ProgressPreference = 'Continue'
    
    # Try with Invoke-WebRequest first
    $headers = @{
        'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        'Accept' = '*/*'
    }
    
    Invoke-WebRequest -Uri $url -OutFile $modelPath -Headers $headers -UseBasicParsing -ErrorAction Stop
    
    if (Test-Path $modelPath) {
        $size = (Get-Item $modelPath).Length / 1MB
        Write-Host "✓ Model downloaded successfully!"
        Write-Host "  Location: $modelPath"
        Write-Host "  Size: $([math]::Round($size, 2)) MB"
        Write-Host ""
        Write-Host "The model is ready to use. Test it with:"
        Write-Host "  python scripts/test_rvc_setup.py"
    } else {
        Write-Host "✗ Download failed - file not found"
    }
} catch {
    Write-Host "✗ Error downloading model: $_"
    Write-Host ""
    Write-Host "Alternative: Download manually:"
    Write-Host "  1. Visit: $url"
    Write-Host "  2. Download the file"
    Write-Host "  3. Place it at: $modelPath"
}

