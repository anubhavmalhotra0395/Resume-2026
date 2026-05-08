"""
Lightweight ML refinement: optional spectral residual correction.

If a TorchScript model path is provided and exists, it will be loaded and used to
predict a residual mel correction. Otherwise, this module is a no-op and simply
returns the processed audio unchanged.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import librosa


class RefinementNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, ref, proc):
        x = torch.cat([ref, proc], dim=1)
        return self.net(x)


class MLRefiner:
    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        if model_path and model_path.exists():
            try:
                self.model = torch.jit.load(str(model_path), map_location=self.device)
                self.model.eval()
            except Exception:
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def _mel(self, y: np.ndarray, sr: int) -> torch.Tensor:
        # Convert to mel spectrogram tensor (1, 1, n_mels, T)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=40, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        t = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)
        return t.to(self.device)

    def _griffin_lim(self, mel_db: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
        # Simple Griffin-Lim reconstruction from mel (approximate)
        mel = librosa.db_to_power(mel_db)
        S = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=n_fft)
        y = librosa.griffinlim(S, hop_length=hop, n_fft=n_fft)
        return y.astype(np.float32)

    def refine(self, reference: np.ndarray, processed: np.ndarray, sr: int) -> np.ndarray:
        if not self.is_ready():
            return processed
        with torch.no_grad():
            ref_mel = self._mel(reference, sr)
            proc_mel = self._mel(processed, sr)
            residual = self.model(ref_mel, proc_mel)  # (1,1,F,T)
            residual_np = residual.squeeze(0).squeeze(0).cpu().numpy()
            # Apply residual in mel domain
            proc_mel_db = proc_mel.squeeze(0).squeeze(0).cpu().numpy()
            corrected_mel_db = proc_mel_db + residual_np
            # Reconstruct audio (approximate)
            y_hat = self._griffin_lim(corrected_mel_db, sr=sr)
            # Length match
            if len(y_hat) < len(processed):
                y_hat = np.pad(y_hat, (0, len(processed) - len(y_hat)), mode="constant")
            elif len(y_hat) > len(processed):
                y_hat = y_hat[: len(processed)]
            return y_hat.astype(np.float32)


# Singleton helper
_REFINER = None


def get_refiner(model_path: Optional[Path] = None) -> MLRefiner:
    global _REFINER
    if _REFINER is None:
        _REFINER = MLRefiner(model_path=model_path)
    return _REFINER


