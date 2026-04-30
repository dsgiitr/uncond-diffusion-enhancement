#!/usr/bin/env python3
"""
visualize_samples.py
────────────────────
Quick visual preview of all contrastive transform pairs on sample CelebA-HQ
images.  Run this BEFORE the full extraction to sanity-check how the
plus (+) and minus (−) manipulations look.

Usage:
    # Preview all 5 concepts (3 sample images each)
    python utils/visualize_samples.py

    # Preview only specific concepts
    python utils/visualize_samples.py --concept warm_vs_cool
    python utils/visualize_samples.py --concept high_vs_low_brightness,warm_vs_cool

    # Control number of sample images shown
    python utils/visualize_samples.py --num_images 5

    # Use a specific sample index
    python utils/visualize_samples.py --image_idx 42
"""

import sys
import os
import argparse

# Ensure config can be loaded from parent directory (3 levels up)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import matplotlib.pyplot as plt
from config import ExtractionConfig
from dataset_utils import load_image_dataset_for_profile, preprocess_pil_for_profile

# Import all transforms
from concept_extraction_pipeline.transformations import transform_sharp_blur as t_sb
from concept_extraction_pipeline.transformations import transform_gray_oversat as t_go
from concept_extraction_pipeline.transformations import transform_high_low_contrast as t_hl
from concept_extraction_pipeline.transformations import transform_high_low_brightness as t_hb
from concept_extraction_pipeline.transformations import transform_warm_cool as t_wc


# ──────────────────────────────────────────────────────────────────────────────
#  Registry of all available concept transforms
# ──────────────────────────────────────────────────────────────────────────────

def build_all_concepts(cfg: ExtractionConfig) -> dict:
    """Return {name: (display_name, plus_tx, minus_tx)} for every concept pair."""
    concepts = {}

    sb_p, sb_m = t_sb.get_transforms(
        image_size=cfg.image_size,
        blur_kernel_size=cfg.blur_kernel_size,
        blur_sigma=cfg.blur_sigma,
    )
    concepts["sharp_vs_blur"] = ("Sharp vs Blur", sb_p, sb_m)

    go_p, go_m = t_go.get_transforms(
        image_size=cfg.image_size,
        oversaturation_factor=cfg.oversaturation_factor,
    )
    concepts["gray_vs_oversat"] = ("Oversaturated vs Grayscale", go_p, go_m)

    hl_p, hl_m = t_hl.get_transforms(
        image_size=cfg.image_size,
        high_contrast_factor=cfg.high_contrast_factor,
        low_contrast_factor=cfg.low_contrast_factor,
    )
    concepts["high_vs_low_contrast"] = ("High vs Low Contrast", hl_p, hl_m)

    hb_p, hb_m = t_hb.get_transforms(
        image_size=cfg.image_size,
        high_brightness_factor=cfg.high_brightness_factor,
        low_brightness_factor=cfg.low_brightness_factor,
    )
    concepts["high_vs_low_brightness"] = ("High vs Low Brightness", hb_p, hb_m)

    wc_p, wc_m = t_wc.get_transforms(
        image_size=cfg.image_size,
        warm_strength=cfg.warm_strength,
        cool_strength=cfg.cool_strength,
    )
    concepts["warm_vs_cool"] = ("Warm vs Cool", wc_p, wc_m)

    return concepts


# ──────────────────────────────────────────────────────────────────────────────

def denorm(t_tensor: torch.Tensor):
    """Convert [-1, 1] PyTorch tensor back to [0, 1] numpy image for plotting."""
    return (t_tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Preview contrastive transform pairs on CelebA-HQ samples."
    )
    parser.add_argument(
        "--concept", type=str, default=None,
        help="Comma-separated concept name(s) to preview. "
             "If omitted, all concepts are shown. "
             "Available: sharp_vs_blur, gray_vs_oversat, "
             "high_vs_low_contrast, high_vs_low_brightness, warm_vs_cool",
    )
    parser.add_argument(
        "--num_images", type=int, default=3,
        help="Number of sample images to show per concept (default: 3).",
    )
    parser.add_argument(
        "--image_idx", type=int, nargs="*", default=None,
        help="Specific image indices from the dataset to use. "
             "Overrides --num_images.",
    )
    parser.add_argument(
        "--output", type=str, default="concept_preview.png",
        help="Output filename for the preview grid (default: concept_preview.png).",
    )
    parser.add_argument(
        "--dataset_profile", type=str, default=None, choices=["celeba_hq", "lsun_church"],
        help="Dataset profile override.",
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Local dataset path (HF disk / image folder / LSUN LMDB).",
    )
    parser.add_argument(
        "--hf_dataset", type=str, default=None,
        help="HF dataset override.",
    )
    parser.add_argument(
        "--dataset_split", type=str, default=None,
        help="HF split override.",
    )
    args = parser.parse_args()

    cfg = ExtractionConfig()
    if args.dataset_profile is not None:
        cfg.dataset_profile = args.dataset_profile
    if args.dataset_dir is not None:
        cfg.dataset_dir = args.dataset_dir
    if args.hf_dataset is not None:
        cfg.hf_dataset = args.hf_dataset
    if args.dataset_split is not None:
        cfg.dataset_split = args.dataset_split

    all_concepts = build_all_concepts(cfg)

    # ── Filter to selected concepts ────────────────────────────────────
    if args.concept:
        requested = [c.strip() for c in args.concept.split(",")]
        unknown = [c for c in requested if c not in all_concepts]
        if unknown:
            print(f"ERROR: Unknown concept(s): {unknown}")
            print(f"Available: {list(all_concepts.keys())}")
            return
        selected = {k: all_concepts[k] for k in requested}
    else:
        selected = all_concepts

    # ── Load dataset ───────────────────────────────────────────────────
    print("Loading dataset...")
    image_ds = load_image_dataset_for_profile(
        dataset_profile=cfg.dataset_profile,
        dataset_dir=cfg.dataset_dir,
        hf_dataset=cfg.hf_dataset,
        dataset_split=cfg.dataset_split,
        image_key="image",
    )

    # Determine which image indices to use
    if args.image_idx is not None:
        indices = args.image_idx
    else:
        # Spread across the dataset for variety
        step = max(1, len(image_ds) // (args.num_images + 1))
        indices = [step * (i + 1) for i in range(args.num_images)]
        indices = [i for i in indices if i < len(image_ds)]

    n_concepts = len(selected)
    n_images = len(indices)

    # ── Build grid: rows = concepts, cols = Original + Plus + Minus per image ──
    cols = n_images * 3   # (original, plus, minus) per sample image
    fig, axes = plt.subplots(
        n_concepts, cols,
        figsize=(4 * cols, 4.5 * n_concepts),
        squeeze=False,
    )

    from torchvision import transforms as T
    base_tx = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    for row, (cname, (display_name, p_tx, m_tx)) in enumerate(selected.items()):
        for img_i, idx in enumerate(indices):
            pil_img = image_ds[idx]
            pil_img = preprocess_pil_for_profile(
                pil_img,
                image_size=cfg.image_size,
                dataset_profile=cfg.dataset_profile,
            )
            col_base = img_i * 3

            # Original
            orig_tensor = base_tx(pil_img)
            ax = axes[row, col_base]
            ax.imshow(denorm(orig_tensor))
            if row == 0:
                ax.set_title(f"Original (#{idx})", fontsize=11, fontweight="bold")
            ax.axis("off")
            if img_i == 0:
                ax.set_ylabel(display_name, fontsize=13, fontweight="bold",
                              rotation=90, labelpad=15)

            # Plus
            plus_tensor = p_tx(pil_img)
            ax = axes[row, col_base + 1]
            ax.imshow(denorm(plus_tensor))
            if row == 0:
                ax.set_title("(+) Plus", fontsize=11, fontweight="bold",
                             color="#2e7d32")
            ax.axis("off")

            # Minus
            minus_tensor = m_tx(pil_img)
            ax = axes[row, col_base + 2]
            ax.imshow(denorm(minus_tensor))
            if row == 0:
                ax.set_title("(−) Minus", fontsize=11, fontweight="bold",
                             color="#c62828")
            ax.axis("off")

    plt.suptitle(
        "Contrastive Transform Preview",
        fontsize=18, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.abspath(args.output)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Preview saved to: {out_path}")
    print(f"  Concepts : {list(selected.keys())}")
    print(f"  Images   : indices {indices}")


if __name__ == "__main__":
    main()
