"""
transform_high_low_contrast.py
──────────────────────────────
Image transforms for the High Contrast (+) vs Low Contrast (-) contrastive pair.

  plus  : high-contrast image  (factor > 1)
  minus : low-contrast image   (factor < 1)
"""

from __future__ import annotations

from PIL import Image, ImageEnhance
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  PIL-level transforms
# ──────────────────────────────────────────────────────────────────────────────

class HighContrast:
    """Increase contrast via PIL's ImageEnhance.Contrast (factor > 1)."""

    def __init__(self, factor: float = 1.6):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageEnhance.Contrast(img).enhance(self.factor)

    def __repr__(self):
        return f"HighContrast(factor={self.factor})"


class LowContrast:
    """Reduce contrast via PIL's ImageEnhance.Contrast (0 < factor < 1)."""

    def __init__(self, factor: float = 0.6):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageEnhance.Contrast(img).enhance(self.factor)

    def __repr__(self):
        return f"LowContrast(factor={self.factor})"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: int = 256,
    high_contrast_factor: float = 1.6,
    low_contrast_factor: float = 0.6,
):
    """
    Returns
    -------
    plus_transform  : Compose — high contrast → resize → tensor → [-1,1]
    minus_transform : Compose — low contrast  → resize → tensor → [-1,1]
    """
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    plus_transform = transforms.Compose([
        HighContrast(high_contrast_factor),
        to_model,
    ])

    minus_transform = transforms.Compose([
        LowContrast(low_contrast_factor),
        to_model,
    ])

    return plus_transform, minus_transform
