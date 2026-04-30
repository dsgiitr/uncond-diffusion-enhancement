from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class FlattenLighting:
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        # Equalize only L channel
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        flat_lit = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(flat_lit)

def get_transforms(image_size: int = 256):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        FlattenLighting(),
        to_model,
    ])
    return plus_transform, minus_transform
