from __future__ import annotations
import numpy as np
from PIL import Image
from torchvision import transforms

class GammaDarken:
    def __init__(self, gamma: float = 2.2):
        self.gamma = gamma

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32)
        underexposed = ((arr / 255.0) ** self.gamma) * 255.0
        return Image.fromarray(np.clip(underexposed, 0, 255).astype(np.uint8))

def get_transforms(image_size: int = 256, gamma: float = 2.2):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        GammaDarken(gamma),
        to_model,
    ])
    return plus_transform, minus_transform
