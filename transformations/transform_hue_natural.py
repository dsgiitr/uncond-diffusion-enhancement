from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class HueShift:
    def __init__(self, shift_degrees: int = 30):
        self.shift_degrees = shift_degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Hue is [0, 180] in OpenCV HSV
        hsv[:,:,0] = (hsv[:,:,0] + self.shift_degrees) % 180
        hsv = hsv.astype(np.uint8)
        unnatural = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(unnatural)

def get_transforms(image_size: int = 256, hue_shift_degrees: int = 30):
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    plus_transform = to_model
    minus_transform = transforms.Compose([
        HueShift(hue_shift_degrees),
        to_model,
    ])
    return plus_transform, minus_transform
