"""
transform_warm_cool.py
──────────────────────
Image transforms for the Warm Colours (+) vs Cool Colours (-) contrastive pair.

  plus  : warm-shifted image  (boost reds/yellows, suppress blues)
  minus : cool-shifted image  (boost blues/cyans, suppress reds)

Colour temperature is adjusted by scaling the R, G, B channels in PIL.
The `strength` parameter (0–1) controls how aggressively the shift is
applied, making it fully tunable.

    warm :  R × (1 + strength),  G × 1.0,  B × (1 − strength × 0.6)
    cool :  R × (1 − strength × 0.6),  G × 1.0,  B × (1 + strength)

All channel values are clamped to [0, 255].
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  PIL-level transforms
# ──────────────────────────────────────────────────────────────────────────────

class WarmShift:
    """
    Shift colour temperature toward warm (amber / golden).

    Parameters
    ----------
    strength : float
        How aggressively to warm-shift.  0 = identity, 1 = maximum shift.
    """

    def __init__(self, strength: float = 0.35):
        self.strength = strength

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        arr[..., 0] *= 1.0 + self.strength          # boost red
        arr[..., 2] *= 1.0 - self.strength * 0.6    # suppress blue
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return f"WarmShift(strength={self.strength})"


class CoolShift:
    """
    Shift colour temperature toward cool (blue / teal).

    Parameters
    ----------
    strength : float
        How aggressively to cool-shift.  0 = identity, 1 = maximum shift.
    """

    def __init__(self, strength: float = 0.35):
        self.strength = strength

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        arr[..., 0] *= 1.0 - self.strength * 0.6    # suppress red
        arr[..., 2] *= 1.0 + self.strength           # boost blue
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __repr__(self):
        return f"CoolShift(strength={self.strength})"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: int = 256,
    warm_strength: float = 0.35,
    cool_strength: float = 0.35,
):
    """
    Returns
    -------
    plus_transform  : Compose — warm colour shift → resize → tensor → [-1,1]
    minus_transform : Compose — cool colour shift → resize → tensor → [-1,1]
    """
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    plus_transform = transforms.Compose([
        WarmShift(warm_strength),
        to_model,
    ])

    minus_transform = transforms.Compose([
        CoolShift(cool_strength),
        to_model,
    ])

    return plus_transform, minus_transform
