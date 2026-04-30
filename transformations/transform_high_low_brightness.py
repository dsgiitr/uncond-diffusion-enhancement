"""
transform_high_low_brightness.py
─────────────────────────────────
Image transforms for the High Brightness (+) vs Low Brightness (-) contrastive pair.

  plus  : bright image   (factor > 1 — pixels shifted toward white)
  minus : dark image     (factor < 1 — pixels shifted toward black)

Both transforms use PIL's ImageEnhance.Brightness, which linearly
interpolates between a black image (factor=0) and the original (factor=1)
and extrapolates beyond 1 for over-brightening.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  PIL-level transforms
# ──────────────────────────────────────────────────────────────────────────────

class HighBrightness:
    """Increase brightness via PIL's ImageEnhance.Brightness (factor > 1)."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageEnhance.Brightness(img).enhance(self.factor)

    def __repr__(self):
        return f"HighBrightness(factor={self.factor})"


class LowBrightness:
    """Reduce brightness via PIL's ImageEnhance.Brightness (0 < factor < 1)."""

    def __init__(self, factor: float = 0.5):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageEnhance.Brightness(img).enhance(self.factor)

    def __repr__(self):
        return f"LowBrightness(factor={self.factor})"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: int = 256,
    high_brightness_factor: float = 1.5,
    low_brightness_factor: float = 0.5,
):
    """
    Returns
    -------
    plus_transform  : Compose — high brightness → resize → tensor → [-1,1]
    minus_transform : Compose — low brightness  → resize → tensor → [-1,1]
    """
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    plus_transform = transforms.Compose([
        HighBrightness(high_brightness_factor),
        to_model,
    ])

    minus_transform = transforms.Compose([
        LowBrightness(low_brightness_factor),
        to_model,
    ])

    return plus_transform, minus_transform
