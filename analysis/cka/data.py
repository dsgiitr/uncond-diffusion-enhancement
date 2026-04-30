"""
Data loading and preprocessing for CKA analysis.

Supports:
    - CelebA-HQ dataset loaded from disk (default)
    - LSUN Church loaded from LMDB/image-folder/HF id
  - Custom index lists for handpicked subsets of the dataset
    - Any similarly-structured HuggingFace image dataset with an 'image' column

Preprocessing policy:
    - CelebA-HQ: keep existing behavior.
    - LSUN Church: resize with preserved aspect ratio and center-crop to 256x256.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import load_image_dataset_for_profile, preprocess_pil_for_profile


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════


class CKAImageDataset(Dataset):
    """Simple dataset that returns preprocessed image tensors.

    Expects a HuggingFace-style dataset with an 'image' column.
    Images are normalised to [-1, 1] (DDPM convention).

    Args:
        hf_dataset:  A HuggingFace dataset object.
        indices:     Optional list of sample indices (for handpicked subsets).
        image_key:   Column name for images in the dataset.
    """

    def __init__(
        self,
        hf_dataset,
        indices: Optional[List[int]] = None,
        image_key: str = "image",
        dataset_profile: str = "celeba_hq",
        image_size: int = 256,
    ):
        self.ds = hf_dataset
        self.indices = indices
        self.image_key = image_key
        self.dataset_profile = dataset_profile
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = self.indices[idx] if self.indices is not None else idx
        sample = self.ds[real_idx]

        if isinstance(sample, Image.Image):
            img = sample.convert("RGB")
        elif isinstance(sample, dict):
            img = sample[self.image_key].convert("RGB")
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}")

        img = preprocess_pil_for_profile(
            img,
            image_size=self.image_size,
            dataset_profile=self.dataset_profile,
        )
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Loaders
# ═══════════════════════════════════════════════════════════════════════════════


def _load_indices(path: Union[str, Path]) -> List[int]:
    """Load sample indices from a JSON or plain-text file.

    Supported formats:
      - JSON list:  [0, 3, 17, 42, ...]
      - Text file:  one integer per line
    """
    path = Path(path)
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [int(line.strip()) for line in text.splitlines() if line.strip()]


def load_celeba_hq(
    dataset_path: str = "celeba_hq_dataset",
    indices: Optional[Union[List[int], str, Path]] = None,
    num_samples: Optional[int] = None,
    from_disk: bool = True,
    hf_id: str = "korexyz/celeba-hq-256x256",
    dataset_profile: str = "celeba_hq",
    dataset_split: str = "train",
    image_size: int = 256,
    image_key: str = "image",
) -> CKAImageDataset:
    """Load CelebA-HQ dataset for CKA analysis.

    Args:
        dataset_path: Path to a locally-saved HF dataset (used when
                      ``from_disk=True``).
        indices:      Explicit sample indices — a list of ints, or a path to
                      a JSON / text file containing the indices.  When provided,
                      ``num_samples`` is ignored.
        num_samples:  Take the first *N* samples (ignored if ``indices`` given).
        from_disk:    If True, ``load_from_disk(dataset_path)``.
                      If False, ``load_dataset(hf_id, split='train')``.
        hf_id:        HuggingFace dataset ID (only used when ``from_disk=False``).
        image_key:    Column name for images.

    Returns:
        CKAImageDataset ready for a DataLoader.
    """
    # ── Load dataset ────────────────────────────────────────────────────────
    dataset_dir = dataset_path if from_disk else ""
    hf_dataset = "" if from_disk else hf_id
    ds = load_image_dataset_for_profile(
        dataset_profile=dataset_profile,
        dataset_dir=dataset_dir,
        hf_dataset=hf_dataset,
        dataset_split=dataset_split,
        image_key=image_key,
    )

    # ── Resolve indices ─────────────────────────────────────────────────────
    if indices is not None:
        if isinstance(indices, (str, Path)):
            indices = _load_indices(indices)
    elif num_samples is not None:
        indices = list(range(min(num_samples, len(ds))))

    return CKAImageDataset(
        ds,
        indices=indices,
        image_key=image_key,
        dataset_profile=dataset_profile,
        image_size=image_size,
    )


def build_dataloader(
    dataset: CKAImageDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Build an optimised DataLoader for CKA extraction."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
