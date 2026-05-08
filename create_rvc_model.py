"""Quick script to create a placeholder RVC model."""
import torch
import torch.nn as nn
from pathlib import Path

class PlaceholderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    def forward(self, x):
        return x

model = PlaceholderModel()
model_path = Path("models/rvc/pretrained.pth")
model_path.parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    "model": model.state_dict(),
    "version": "1.0.0",
    "note": "Placeholder RVC model - replace with real trained model"
}

torch.save(checkpoint, model_path)
print(f"✓ Placeholder model created at: {model_path.absolute()}")
print("⚠ This is a dummy model for testing. Replace with a real RVC model for actual voice conversion.")

