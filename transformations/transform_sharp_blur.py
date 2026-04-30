"""
transform_sharp_blur.py
───────────────────────
Image transforms for the Sharp (+) vs Blur (-) contrastive pair.

  plus  : original (sharp) image — identity transform
  minus : Gaussian-blurred image
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
#  Blur transform
# ──────────────────────────────────────────────────────────────────────────────

class GaussianBlur:
    """
    Apply a Gaussian blur to a PIL image using OpenCV's filter2D.

    Parameters
    ----------
    kernel_size : int
        Size of the square Gaussian kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian kernel.
    """

    def __init__(self, kernel_size: int = 21, sigma: float = 3.0):
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        k1d = cv2.getGaussianKernel(kernel_size, sigma)
        self.kernel = (k1d @ k1d.T).astype(np.float32)

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        blurred = cv2.filter2D(arr, -1, self.kernel,
                               borderType=cv2.BORDER_REFLECT_101)
        return Image.fromarray(blurred)

    def __repr__(self):
        return f"GaussianBlur(kernel={self.kernel.shape[0]}, σ={self.kernel.sum():.2f})"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API — returns (plus_transform, minus_transform)
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(
    image_size: int = 256,
    blur_kernel_size: int = 21,
    blur_sigma: float = 3.0,
):
    """
    Returns
    -------
    plus_transform : Compose   — sharp (identity → resize → tensor → [-1,1])
    minus_transform : Compose  — blur  → resize → tensor → [-1,1]
    """
    to_model = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    plus_transform = to_model                       # identity (sharp)

    minus_transform = transforms.Compose([
        GaussianBlur(blur_kernel_size, blur_sigma),  # PIL → blurred PIL
        to_model,
    ])

    return plus_transform, minus_transform
