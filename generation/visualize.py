"""
Visualization helpers — 1×3 subplot comparison (Baseline | Patched | CFG).

Kept as a separate module so the visualizer remains intact and reusable
independently of the generation loop.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
from PIL import Image


def save_comparison(
    baseline_imgs: List[Image.Image],
    patched_imgs: List[Image.Image],
    cfg_imgs: List[Image.Image],
    save_dir: str,
    *,
    v_scale: float = 1.0,
    cfg_scale: float = 3.0,
    seed: int = 42,
    scheduler_name: str = "ddpm",
    num_steps: int = 50,
    patch_mode: str = "continuous",
    patch_start: int = 0,
    patch_end: int = 10,
    dpi: int = 150,
) -> List[str]:
    """Save a 1×3 subplot per batch item and return saved paths.

    Layout:
        [ Baseline | Patched (v_scale=…) | CFG (v_scale=…, w=…) ]
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = []

    batch_size = len(baseline_imgs)
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="tight")

        axes[0].imshow(baseline_imgs[i])
        axes[0].set_title("Baseline", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(patched_imgs[i])
        axes[1].set_title(
            f"Patched (v_scale={v_scale})", fontsize=13, fontweight="bold"
        )
        axes[1].axis("off")

        axes[2].imshow(cfg_imgs[i])
        axes[2].set_title(
            f"CFG (v_scale={v_scale}, w={cfg_scale})",
            fontsize=13, fontweight="bold",
        )
        axes[2].axis("off")

        fig.suptitle(
            f"seed={seed}  |  scheduler={scheduler_name}  |  "
            f"steps={num_steps}  |  patch={patch_mode} "
            f"[{patch_start}–{patch_end}]",
            fontsize=10, color="gray",
        )

        out_path = os.path.join(save_dir, f"{i}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        saved.append(out_path)
        print(f"  ✓ Saved → {out_path}")

    return saved
