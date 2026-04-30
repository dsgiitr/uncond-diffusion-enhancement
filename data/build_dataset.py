#!/usr/bin/env python3
"""
Build a CelebA-HQ 256×256 dataset with all 40 binary attribute labels.

This script:
  1. Reads the CelebAMask-HQ-attribute-anno.txt to get per-image attribute labels
  2. Loads each image from CelebA-HQ-img/, resizes to 256×256
  3. Saves the result as a HuggingFace Dataset with:
     - "image" column  (PIL Image, 256×256 RGB)
     - 40 attribute columns  (int, +1 / -1 → mapped to 1 / 0)
  4. Writes a metadata JSON for quick inspection

Usage:
    python build_dataset.py \
        --img_dir  ../CelebAMask-HQ/CelebA-HQ-img \
        --anno_file CelebAMask-HQ-attribute-anno.txt \
        --output_dir ../celeba_hq_dataset \
        --resolution 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_attribute_file(anno_path: str):
    """Parse the CelebAMask-HQ-attribute-anno.txt file.

    Returns:
        attr_names: list of 40 attribute names
        records: list of dicts  {filename: str, attr1: int, ...}
    """
    with open(anno_path, "r") as f:
        lines = f.readlines()

    num_images = int(lines[0].strip())
    attr_names = lines[1].strip().split()

    records = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < 1 + len(attr_names):
            continue
        filename = parts[0]
        values = [int(v) for v in parts[1:]]
        record = {"filename": filename}
        for attr, val in zip(attr_names, values):
            # Map -1 → 0, +1 → 1
            record[attr] = 1 if val == 1 else 0
        records.append(record)

    print(f"Parsed {len(records)} records with {len(attr_names)} attributes")
    print(f"Attributes: {attr_names}")
    return attr_names, records


def build_dataset(
    img_dir: str,
    records: list[dict],
    attr_names: list[str],
    resolution: int,
    output_dir: str,
):
    """Build and save a HuggingFace Dataset."""
    try:
        from datasets import Dataset, Features, Value, Image as HFImage
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with: pip install datasets")
        sys.exit(1)

    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-validate which images exist
    valid_records = []
    missing = 0
    for rec in records:
        img_path = img_dir / rec["filename"]
        if img_path.exists():
            valid_records.append(rec)
        else:
            missing += 1

    if missing > 0:
        print(f"WARNING: {missing} images not found in {img_dir}, skipping them")
    print(f"Building dataset with {len(valid_records)} images at {resolution}×{resolution}")

    # Build lists for each column
    data = {attr: [] for attr in attr_names}
    data["image"] = []
    data["filename"] = []

    for rec in tqdm(valid_records, desc="Processing images", unit="img"):
        img_path = img_dir / rec["filename"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.LANCZOS)

        data["image"].append(img)
        data["filename"].append(rec["filename"])
        for attr in attr_names:
            data[attr].append(rec[attr])

    # Create HuggingFace Dataset
    ds = Dataset.from_dict(data)
    ds.save_to_disk(str(output_dir))

    # Write metadata JSON
    attr_stats = {}
    for attr in attr_names:
        pos = sum(data[attr])
        neg = len(data[attr]) - pos
        attr_stats[attr] = {"positive": pos, "negative": neg, "ratio": round(pos / len(data[attr]), 4)}

    metadata = {
        "name": "CelebA-HQ-256-Attributed",
        "num_images": len(valid_records),
        "resolution": resolution,
        "num_attributes": len(attr_names),
        "attributes": attr_names,
        "attribute_statistics": attr_stats,
        "source": "CelebAMask-HQ",
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to: {output_dir}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Total images: {len(valid_records)}")

    # Print top-level stats
    print("\n--- Attribute Statistics ---")
    print(f"{'Attribute':<25} {'Positive':>8} {'Negative':>8} {'Ratio':>8}")
    print("-" * 55)
    for attr in attr_names:
        s = attr_stats[attr]
        print(f"{attr:<25} {s['positive']:>8} {s['negative']:>8} {s['ratio']:>8.4f}")

    return ds


def main():
    parser = argparse.ArgumentParser(description="Build CelebA-HQ 256×256 attributed dataset")
    parser.add_argument("--img_dir", type=str,
                        default="../CelebAMask-HQ/CelebA-HQ-img",
                        help="Path to CelebA-HQ-img/ directory with original 1024x1024 images")
    parser.add_argument("--anno_file", type=str,
                        default="CelebAMask-HQ-attribute-anno.txt",
                        help="Path to attribute annotation file")
    parser.add_argument("--output_dir", type=str,
                        default="../celeba_hq_dataset",
                        help="Where to save the HuggingFace Dataset")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Target image resolution (default: 256)")
    args = parser.parse_args()

    attr_names, records = parse_attribute_file(args.anno_file)
    build_dataset(args.img_dir, records, attr_names, args.resolution, args.output_dir)


if __name__ == "__main__":
    main()
