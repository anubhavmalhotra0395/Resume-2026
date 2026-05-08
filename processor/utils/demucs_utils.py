"""
Robust Demucs wrapper for vocal extraction.
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional


def run_demucs_extract(input_path: str, out_dir: str = '/tmp/demucs_out', model: str = 'htdemucs') -> Optional[str]:
    """
    Run Demucs to extract vocals from audio.
    
    Args:
        input_path: Path to input audio file
        out_dir: Output directory for Demucs results
        model: Demucs model name (htdemucs, htdemucs_ft, etc.)
    
    Returns:
        Path to extracted vocals file, or None if failed
    """
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # Use CLI (assumes demucs installed)
        cmd = ['demucs', '-n', model, '-o', out_dir, input_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Search for vocals file
        base = Path(input_path).stem
        
        # Demucs output structure: out_dir/model_name/track_name/vocals.wav
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith('.wav') and 'vocals' in f.lower():
                    full_path = os.path.join(root, f)
                    # Prefer file that matches input name
                    if base in f or base in root:
                        return full_path
                    # Otherwise return first vocals file found
                    return full_path
        
        # Fallback: search for any vocals.wav
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith('.wav') and 'vocals' in f.lower():
                    return os.path.join(root, f)
        
        logging.warning(f"Demucs completed but no vocals file found in {out_dir}")
        return None
        
    except subprocess.CalledProcessError as e:
        logging.exception(f"Demucs CLI failed: {e}")
        return None
    except FileNotFoundError:
        logging.warning("Demucs CLI not found. Install with: pip install demucs")
        return None
    except Exception as e:
        logging.exception(f"Demucs extraction failed: {e}")
        return None

