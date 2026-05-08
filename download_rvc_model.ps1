# Download RVC model from HuggingFace
$url = "https://huggingface.co/AnonUser123/generic_rvc_voice/resolve/main/rvc_generic.pth"
$outputPath = "models\rvc\pretrained.pth"

# Create directory
New-Item -ItemType Directory -Force -Path "models\rvc" | Out-Null

Write-Host "Downloading RVC model from HuggingFace..." -ForegroundColor Cyan
Write-Host "URL: $url"
Write-Host "Output: $outputPath"
Write-Host ""

try {
    # Use .NET WebClient for better compatibility
    $webClient = New-Object System.Net.WebClient
    $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Download with progress
    $webClient.DownloadFile($url, $outputPath)
    
    $fileSize = (Get-Item $outputPath).Length / 1MB
    Write-Host "✓ Model downloaded successfully!" -ForegroundColor Green
    Write-Host "  Location: $((Resolve-Path $outputPath).Path)" -ForegroundColor Gray
    Write-Host "  Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Gray
    Write-Host ""
    Write-Host "The model is now ready to use!" -ForegroundColor Green
} catch {
    Write-Host "❌ Error downloading: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Download manually:" -ForegroundColor Yellow
    Write-Host "  1. Open: $url" -ForegroundColor Cyan
    Write-Host "  2. Save the file to: $outputPath" -ForegroundColor Cyan
    exit 1
}

