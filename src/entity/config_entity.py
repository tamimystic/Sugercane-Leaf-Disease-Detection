from dataclasses import dataclass
import os

@dataclass
class PredictionConfig:
    model_path: str = os.path.join("models", "fine_tuned_hybrid_swin_dense.pth")
    image_size: int = 224
    classes: tuple = (
        "BandedChlorosis", "BrownSpot", "DriedLeaves", "EyeSpot", 
        "GrassyShoot", "Healthy", "Mosaic", "PokkahBoeng", 
        "RedLeafSpot", "RedRot", "RingSpot", "Rust", 
        "SettRot", "Smut", "YellowLeaf"
    )
