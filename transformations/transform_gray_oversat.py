"""
transform_gray_oversat.py
─────────────────────────
Image transforms for the Oversaturated (+) vs Grayscale (-) contrastive pair.

  plus  : oversaturated image   (boosted colour saturation)
  minus : grayscale image       (luminance-only, 3-ch RGB)
"""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  PIL-level transforms
# ──────────────────────────────────────────────────────────────────────────────

class Oversaturate:
    """Boost colour saturation using PIL's ImageEnhance.Color."""

    def __init__(self, factor: float = 1.8):
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageEnhance.Color(img).enhance(self.factor)

    def __repr__(self):
        return f"Oversaturate(factor={self.factor})"


class GrayscaleRGB:
    """Convert to luminance grayscale, return as 3-channel RGB."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return ImageOps.grayscale(img).convert("RGB")

    def __repr__(self):
        return "GrayscaleRGB()"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: int = 256,
    oversaturation_factor: float = 1.8,
):
    """
    Returns
    -------
    plus_transform  : Compose — oversaturated → resize → tensor → [-1,1]
    minus_transform : Compose — grayscale     → resize → tensor → [-1,1]
    """
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    plus_transform = transforms.Compose([
        Oversaturate(oversaturation_factor),
        to_model,
    ])

    minus_transform = transforms.Compose([
        GrayscaleRGB(),
        to_model,
    ])

    return plus_transform, minus_transform
