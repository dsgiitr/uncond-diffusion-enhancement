from __future__ import annotations
import numpy as np
from PIL import Image
from torchvision import transforms

class AddNoise:
    def __init__(self, std_dev: float = 25.0):
        self.std_dev = std_dev

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, self.std_dev, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

def get_transforms(image_size: int = 256, noise_std: float = 25.0):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        AddNoise(noise_std),
        to_model,
    ])
    return plus_transform, minus_transform
