#!/usr/bin/env python3
"""
Create a minimal placeholder RVC model for testing.

This creates a dummy model that will load without errors,
but won't perform actual voice conversion. Replace it with
a real trained model when available.
"""
import torch
import torch.nn as nn
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

class PlaceholderRVCModel(nn.Module):
    """Minimal placeholder RVC model structure."""
    def __init__(self):
        super().__init__()
        # Minimal structure - just enough to load
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # Pass-through with minimal processing
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_placeholder_model(output_path: Path):
    """Create a placeholder RVC model file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info("Creating placeholder RVC model...")
    
    # Create model
    model = PlaceholderRVCModel()
    
    # Create checkpoint structure (similar to common RVC formats)
    checkpoint = {
        "model": model.state_dict(),
        "version": "1.0.0",
        "note": "This is a placeholder model. Replace with a real trained RVC model for actual voice conversion."
    }
    
    # Save model
    torch.save(checkpoint, output_path)
    
    logging.info(f"✓ Placeholder model created at: {output_path}")
    logging.info("⚠ This is a dummy model - it will load but won't perform real voice conversion")
    logging.info("   Replace it with a trained RVC model when available")

if __name__ == "__main__":
    model_path = Path("models/rvc/pretrained.pth")
    
    if model_path.exists():
        response = input(f"Model already exists at {model_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            exit(0)
    
    create_placeholder_model(model_path)
    print(f"\n✓ Placeholder model ready!")
    print(f"  Location: {model_path.absolute()}")
    print(f"\n⚠ Note: This is a dummy model for testing.")
    print(f"   To use real RVC:")
    print(f"   1. Download a trained model from HuggingFace or train your own")
    print(f"   2. Replace this file with the real model")
    print(f"   3. Implement the inference code in processor/ml_refine/rvc_refiner.py")

