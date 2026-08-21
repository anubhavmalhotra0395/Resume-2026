#!/usr/bin/env python3
"""
API integration test - tests full job submission and processing pipeline.

This script:
- Sends multipart POST request to /jobs with dry file and reference file
- Polls /jobs/{id} until status=finished/failed
- Downloads output WAV
- Prints status + timings
- Asserts that WAV exists and is > 1 sec
"""
import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf

API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 2  # seconds
MAX_WAIT_TIME = 300  # 5 minutes

def create_test_files():
    """Create test audio files."""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    # Create simple test audio
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    # Dry vocal (simple tone)
    dry = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    dry_path = test_dir / "test_dry_api.wav"
    sf.write(dry_path, dry, sr)
    
    # Reference (slightly different tone with harmonics)
    ref = (0.2 * np.sin(2 * np.pi * 440 * t) + 
           0.1 * np.sin(2 * np.pi * 880 * t) + 
           0.05 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)
    ref_path = test_dir / "test_ref_api.wav"
    sf.write(ref_path, ref, sr)
    
    return dry_path, ref_path

def test_api_job():
    """Test full API job submission and processing."""
    print("=" * 60)
    print("API Integration Test")
    print("=" * 60)
    print()
    
    # Check API is running
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if response.status_code != 200:
            print(f"✗ API health check failed: {response.status_code}")
            return 1
        print("✓ API is running")
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_BASE_URL}")
        print("  Make sure the API server is running: docker compose up")
        return 1
    
    # Create test files
    print("Creating test files...")
    dry_path, ref_path = create_test_files()
    print(f"✓ Test files created: {dry_path}, {ref_path}")
    print()
    
    # Submit job
    print("Submitting job...")
    start_time = time.time()
    
    with open(dry_path, 'rb') as dry_file, open(ref_path, 'rb') as ref_file:
        files = {
            'dry': ('dry.wav', dry_file, 'audio/wav'),
            'reference': ('ref.wav', ref_file, 'audio/wav'),
        }
        data = {
            'use_spectral_refiner': 'false',  # Disable for faster test
            'enable_deesser': 'false',
            'enable_transient_shaper': 'false',
            'enable_multiband': 'false',
            'adaptive_dsp': 'false',
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/jobs",
                files=files,
                data=data,
                timeout=30,
            )
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data.get('job_id')
            print(f"✓ Job submitted: {job_id}")
        except Exception as e:
            print(f"✗ Job submission failed: {e}")
            return 1
    
    # Poll for completion
    print(f"Polling job status (max wait: {MAX_WAIT_TIME}s)...")
    status = None
    poll_count = 0
    
    while status not in ['finished', 'failed']:
        time.sleep(POLL_INTERVAL)
        poll_count += 1
        elapsed = time.time() - start_time
        
        if elapsed > MAX_WAIT_TIME:
            print(f"✗ Timeout after {elapsed:.1f}s")
            return 1
        
        try:
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=10)
            response.raise_for_status()
            job_data = response.json()
            status = job_data.get('status')
            
            if poll_count % 5 == 0:  # Print every 10 seconds
                print(f"  Status: {status} (elapsed: {elapsed:.1f}s)")
        except Exception as e:
            print(f"✗ Error polling job: {e}")
            return 1
    
    total_time = time.time() - start_time
    print(f"✓ Job completed with status: {status} (total time: {total_time:.1f}s)")
    print()
    
    if status == 'failed':
        error = job_data.get('error', 'Unknown error')
        print(f"✗ Job failed: {error}")
        return 1
    
    # Download output
    download_url = job_data.get('download_url')
    if not download_url:
        print("✗ No download URL in response")
        return 1
    
    print(f"Downloading output from: {download_url}")
    try:
        response = requests.get(f"{API_BASE_URL}{download_url}", timeout=30)
        response.raise_for_status()
        
        output_path = Path("test_outputs") / f"api_test_output_{job_id}.wav"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Output downloaded: {output_path}")
        
        # Verify output
        audio, sr = sf.read(output_path)
        duration = len(audio) / sr
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Sample rate: {sr}Hz")
        print(f"  Channels: {audio.shape[1] if len(audio.shape) > 1 else 1}")
        
        # Assertions
        assert duration > 1.0, f"Output too short: {duration:.2f}s"
        assert len(audio) > 0, "Output is empty"
        
        max_val = np.max(np.abs(audio))
        assert max_val > 0.01, f"Output too quiet: max={max_val:.4f}"
        
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Status: ✓ PASS")
        print(f"Job ID: {job_id}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Output duration: {duration:.2f}s")
        print(f"Output quality: {'✓ Good' if max_val > 0.1 else '⚠ Quiet'}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"✗ Download/verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = test_api_job()
    sys.exit(exit_code)

