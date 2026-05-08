"""
Training script for spectral refiner UNet.
Produces TorchScript model for inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import sys
from pathlib import Path


class SmallUNet(nn.Module):
    """Small UNet for spectral refinement."""
    
    def __init__(self, ch=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1),
            nn.ReLU(),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch * 8, ch * 4, 3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
            nn.Sigmoid(),  # Output mask between 0 and 1
        )
    
    def forward(self, ref, dsp):
        # Concatenate reference and DSP
        x = torch.cat([ref, dsp], dim=1)  # (B, 2, F, T)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(b + e3)
        d2 = self.dec2(d3 + e2)
        mask = self.dec1(d2 + e1)
        
        return mask


class PairsDataset(Dataset):
    """Dataset for (reference, DSP-processed) pairs."""
    
    def __init__(self, pairs_list, n_fft=2048, hop=256):
        self.pairs = pairs_list
        self.n_fft = n_fft
        self.hop = hop
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        ref_path, dsp_path = self.pairs[idx]
        
        # Load audio
        ref, sr = librosa.load(ref_path, sr=44100, mono=True)
        dsp, _ = librosa.load(dsp_path, sr=44100, mono=True)
        
        # Ensure same length
        min_len = min(len(ref), len(dsp))
        ref = ref[:min_len]
        dsp = dsp[:min_len]
        
        # Compute STFT
        S_ref = np.abs(librosa.stft(ref, n_fft=self.n_fft, hop_length=self.hop))
        S_dsp = np.abs(librosa.stft(dsp, n_fft=self.n_fft, hop_length=self.hop))
        
        # Log magnitude
        log_ref = np.log1p(S_ref).astype(np.float32)
        log_dsp = np.log1p(S_dsp).astype(np.float32)
        
        # Ensure same shape
        min_freq = min(log_ref.shape[0], log_dsp.shape[0])
        min_time = min(log_ref.shape[1], log_dsp.shape[1])
        log_ref = log_ref[:min_freq, :min_time]
        log_dsp = log_dsp[:min_freq, :min_time]
        
        return log_ref[np.newaxis], log_dsp[np.newaxis]


def train(
    pairs_txt: str,
    epochs: int = 50,
    bs: int = 8,
    lr: float = 1e-3,
    out_path: str = "models/refiner.pt",
    device: str = None,
):
    """
    Train spectral refiner model.
    
    Args:
        pairs_txt: Path to text file with pairs (ref_path,dsp_path per line)
        epochs: Number of training epochs
        bs: Batch size
        lr: Learning rate
        out_path: Output path for TorchScript model
        device: Device to train on
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pairs
    pairs = []
    with open(pairs_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                ref_path, dsp_path = line.split(',', 1)
                if os.path.exists(ref_path) and os.path.exists(dsp_path):
                    pairs.append((ref_path.strip(), dsp_path.strip()))
    
    if len(pairs) == 0:
        print(f"Error: No valid pairs found in {pairs_txt}")
        return
    
    print(f"Loaded {len(pairs)} training pairs")
    
    # Create dataset and loader
    ds = PairsDataset(pairs)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    
    # Create model
    model = SmallUNet(ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    # Training loop
    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for log_ref, log_dsp in loader:
            log_ref = log_ref.to(device)
            log_dsp = log_dsp.to(device)
            
            # Forward pass
            mask = model(log_ref, log_dsp)
            
            # Compute target: what magnitude should be
            pred_mag = torch.expm1(log_dsp) * mask
            target_mag = torch.expm1(log_ref)
            
            # Loss: L1 + spectral contrast
            l1 = F.l1_loss(pred_mag, target_mag)
            
            # Spectral contrast loss
            sc = torch.norm(target_mag - pred_mag) / (torch.norm(target_mag) + 1e-9)
            
            loss = l1 + 0.1 * sc
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {ep+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    # Script and save
    print(f"Saving model to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Create dummy inputs for scripting
        dummy_ref = torch.randn(1, 1, 1025, 100).to(device)
        dummy_dsp = torch.randn(1, 1, 1025, 100).to(device)
        
        # Trace model
        traced = torch.jit.trace(model, (dummy_ref, dummy_dsp))
        traced.save(out_path)
    
    print(f"✓ Model saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_refiner.py <pairs.txt> [epochs] [batch_size] [lr] [out_path]")
        sys.exit(1)
    
    pairs_txt = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
    out_path = sys.argv[5] if len(sys.argv) > 5 else "models/refiner.pt"
    
    train(pairs_txt, epochs=epochs, bs=bs, lr=lr, out_path=out_path)

