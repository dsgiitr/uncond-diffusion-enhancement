from __future__ import annotations
import io
import numpy as np
from PIL import Image
from torchvision import transforms

class JpegCompress:
    def __init__(self, quality: int = 5):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

def get_transforms(image_size: int = 256, jpeg_quality: int = 5):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        JpegCompress(jpeg_quality),
        to_model,
    ])
    return plus_transform, minus_transform
