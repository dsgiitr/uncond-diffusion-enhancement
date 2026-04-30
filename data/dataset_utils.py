"""
Shared dataset utilities for CelebA-HQ and LSUN Church workflows.

This module keeps CelebA behavior unchanged and applies a dedicated
non-distorting preprocess path only for LSUN Church images.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional

from PIL import Image
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset


_LSUN_CHURCH_ALIASES = {
    "lsun_church",
    "lsun-church",
    "lsun church",
    "church",
    "church_outdoor",
}

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def normalize_dataset_profile(profile: str) -> str:
    """Normalize user profile names to canonical values."""
    normalized = (profile or "celeba_hq").strip().lower()
    if normalized in _LSUN_CHURCH_ALIASES:
        return "lsun_church"
    return "celeba_hq"


def is_lsun_church_profile(profile: str) -> bool:
    return normalize_dataset_profile(profile) == "lsun_church"


def preprocess_pil_for_profile(
    image: Image.Image,
    image_size: int = 256,
    dataset_profile: str = "celeba_hq",
) -> Image.Image:
    """
    Preprocess a PIL image according to dataset profile.

    CelebA-HQ:
      - Leave image as-is (callers keep their existing preprocessing path).

    LSUN Church:
      - Preserve aspect ratio.
      - Resize so the shorter side becomes image_size.
      - Center-crop to image_size x image_size.

    This avoids aspect-ratio distortion while still producing fixed 256x256 inputs.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    if not is_lsun_church_profile(dataset_profile):
        return image

    width, height = image.size
    if width == image_size and height == image_size:
        return image

    scale = float(image_size) / float(min(width, height))
    new_width = max(image_size, int(round(width * scale)))
    new_height = max(image_size, int(round(height * scale)))

    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), resample=Image.BICUBIC)

    left = max(0, (new_width - image_size) // 2)
    top = max(0, (new_height - image_size) // 2)
    right = left + image_size
    bottom = top + image_size
    return image.crop((left, top, right, bottom))


class HFImageDataset(Dataset):
    """Adapter that returns PIL images from a HuggingFace dataset."""

    def __init__(self, hf_dataset, image_key: str = "image"):
        self.ds = hf_dataset
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Image.Image:
        image = self.ds[idx][self.image_key]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL image in column '{self.image_key}', got {type(image)}"
            )
        return image.convert("RGB")


class FolderImageDataset(Dataset):
    """Recursive image-folder dataset (returns PIL images)."""

    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Image.Image:
        with Image.open(self.image_paths[idx]) as image:
            return image.convert("RGB")


class LSUNLMDBImageDataset(Dataset):
    """LSUN LMDB reader for datasets downloaded via fyu/lsun download.py."""

    def __init__(self, lmdb_path: str):
        self.lmdb_path = str(Path(lmdb_path).resolve())
        self._env = None
        self._keys = self._load_keys()

    def _open_env(self):
        if self._env is None:
            try:
                import lmdb
            except ImportError as exc:
                raise ImportError(
                    "Reading LSUN LMDB requires the 'lmdb' package. "
                    "Install it with: pip install lmdb"
                ) from exc

            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
        return self._env

    def _load_keys(self) -> List[bytes]:
        env = self._open_env()
        with env.begin(write=False) as txn:
            return [key for key, _ in txn.cursor()]

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> Image.Image:
        env = self._open_env()
        key = self._keys[idx]
        with env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"Missing LMDB key at index {idx}")
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        return state


def _is_lmdb_dir(path: Path) -> bool:
    return path.is_dir() and (path / "data.mdb").exists() and (path / "lock.mdb").exists()


def _is_hf_disk_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_info.json").exists()


def _collect_image_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTENSIONS:
            files.append(file_path)
    return files


def load_image_dataset_for_profile(
    dataset_profile: str = "celeba_hq",
    dataset_dir: str = "",
    hf_dataset: str = "",
    dataset_split: str = "train",
    image_key: str = "image",
):
    """
    Load an image-only dataset abstraction that yields PIL images.

    Resolution order:
      1) Existing local path (`dataset_dir`) as LMDB, HF disk dataset, or image folder.
      2) HuggingFace dataset id (`hf_dataset`) if provided.
      3) CelebA default HF dataset for backwards compatibility.
    """
    profile = normalize_dataset_profile(dataset_profile)

    if dataset_dir:
        path = Path(dataset_dir).expanduser().resolve()
        if path.exists():
            if _is_lmdb_dir(path):
                return LSUNLMDBImageDataset(str(path))
            if _is_hf_disk_dataset(path):
                ds = load_from_disk(str(path))
                return HFImageDataset(ds, image_key=image_key)

            image_paths = _collect_image_files(path)
            if image_paths:
                return FolderImageDataset(image_paths)

            raise ValueError(
                f"Could not infer dataset format for path: {path}. "
                "Expected LSUN LMDB (data.mdb/lock.mdb), HF disk dataset, or image folder."
            )

    if hf_dataset:
        ds = load_dataset(hf_dataset, split=dataset_split)
        return HFImageDataset(ds, image_key=image_key)

    if profile == "lsun_church":
        raise ValueError(
            "LSUN Church requires --dataset_dir (LMDB/image folder) or --hf_dataset."
        )

    ds = load_dataset("korexyz/celeba-hq-256x256", split=dataset_split)
    return HFImageDataset(ds, image_key=image_key)
