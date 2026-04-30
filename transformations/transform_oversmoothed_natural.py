from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class BilateralSmooth:
    def __init__(self, d: int = 15, sigma_color: float = 75, sigma_space: float = 75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        oversmoothed = cv2.bilateralFilter(arr, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        return Image.fromarray(oversmoothed)

def get_transforms(image_size: int = 256, d: int = 15, sigma_color: float = 75, sigma_space: float = 75):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        BilateralSmooth(d, sigma_color, sigma_space),
        to_model,
    ])
    return plus_transform, minus_transform
