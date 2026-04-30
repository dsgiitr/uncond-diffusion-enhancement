from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class MedianBlurTexture:
    def __init__(self, ksize: int = 7):
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        low_texture = cv2.medianBlur(arr, ksize=self.ksize)
        return Image.fromarray(low_texture)

def get_transforms(image_size: int = 256, median_blur_ksize: int = 7):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        MedianBlurTexture(median_blur_ksize),
        to_model,
    ])
    return plus_transform, minus_transform
