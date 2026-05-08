"""
Spectral refiner using TorchScript UNet for neural refinement.
Corrects DSP output toward reference using learned spectral residuals.
"""
import os
import logging
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional


class SpectralRefiner:
    """TorchScript-based spectral refiner for neural enhancement."""
    
    def __init__(self, model_path="models/refiner.pt", device=None, n_fft=2048, hop=256):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.n_fft = n_fft
        self.hop = hop
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                logging.info(f"Spectral refiner loaded from {model_path}")
            except Exception as e:
                logging.exception(f"Failed to load spectral refiner: {e}")
                self.model = None
        else:
            logging.warning(f"Spectral refiner model not found at {model_path}")
    
    def is_available(self) -> bool:
        """Check if refiner model is loaded and available."""
        return self.model is not None
    
    def stft_mag_phase(self, y: np.ndarray) -> tuple:
        """Compute STFT and return magnitude and phase."""
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop)
        return np.abs(S), np.angle(S)
    
    def istft_mag_phase(self, mag: np.ndarray, phase: np.ndarray, length: Optional[int] = None) -> np.ndarray:
        """Reconstruct audio from magnitude and phase."""
        S = mag * np.exp(1j * phase)
        return librosa.istft(S, hop_length=self.hop, length=length)
    
    def refine(self, dsp_audio: np.ndarray, ref_audio: Optional[np.ndarray] = None, sr: int = 44100) -> np.ndarray:
        """
        Refine DSP-processed audio toward reference using neural model.
        
        Args:
            dsp_audio: DSP-processed audio (numpy array)
            ref_audio: Reference audio for comparison (optional)
            sr: Sample rate
        
        Returns:
            Refined audio
        """
        if not self.is_available():
            logging.warning("Spectral refiner not available. Returning DSP output unchanged.")
            return dsp_audio
        
        try:
            # Compute STFT
            S_dsp, phase = self.stft_mag_phase(dsp_audio)
            log_dsp = np.log1p(S_dsp).astype(np.float32)
            
            if ref_audio is None:
                log_ref = log_dsp.copy()
            else:
                S_ref, _ = self.stft_mag_phase(ref_audio)
                log_ref = np.log1p(S_ref).astype(np.float32)
            
            # Ensure same shape
            min_freq = min(log_ref.shape[0], log_dsp.shape[0])
            min_time = min(log_ref.shape[1], log_dsp.shape[1])
            log_ref = log_ref[:min_freq, :min_time]
            log_dsp = log_dsp[:min_freq, :min_time]
            
            # Convert to tensors (1, 1, F, T)
            t_ref = torch.from_numpy(log_ref).unsqueeze(0).unsqueeze(0).to(self.device)
            t_dsp = torch.from_numpy(log_dsp).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                mask = self.model(t_ref, t_dsp)  # Expect (1, 1, F, T) or (1, C, F, T)
                if mask.dim() > 2:
                    mask = mask.squeeze(0)
                if mask.dim() > 2:
                    mask = mask.squeeze(0)
                mask = mask.cpu().numpy()
            
            # Ensure mask shape matches
            if mask.shape != S_dsp.shape:
                # Resize mask to match
                from scipy.ndimage import zoom
                zoom_factors = (S_dsp.shape[0] / mask.shape[0], S_dsp.shape[1] / mask.shape[1])
                mask = zoom(mask, zoom_factors, order=1)
            
            # Apply mask to magnitude
            mag_refined = S_dsp * mask
            
            # Reconstruct audio
            y = self.istft_mag_phase(mag_refined, phase, length=len(dsp_audio))
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(y)) + 1e-9
            y = y / max_val * 0.95
            
            return y.astype(np.float32)
            
        except Exception as e:
            logging.exception(f"Spectral refiner inference failed: {e}")
            return dsp_audio


# Convenience singleton
_refiner = None


def get_refiner(path: Optional[str] = None):
    """Get or create spectral refiner instance."""
    global _refiner
    if _refiner is None:
        model_path = path or os.environ.get("REFINER_MODEL_PATH", "models/refiner.pt")
        _refiner = SpectralRefiner(model_path=model_path)
    return _refiner

