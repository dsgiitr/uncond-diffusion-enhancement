#!/usr/bin/env python3
"""
Extract bottleneck (mid-block) activations split by a CelebA-HQ attribute.

This script:
1) Loads a local HuggingFace dataset saved with ``save_to_disk``.
2) Splits samples into positive/negative groups based on one attribute column.
3) Runs the DDPM UNet and records bottleneck activations (unet.mid_block output).
4) Saves two tensors:
   - <attribute>_positive.pt  with shape [s1, C, H, W]
   - <attribute>_negative.pt  with shape [s2, C, H, W]
where s1 + s2 = N.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm import tqdm


class AttributeImageDataset(Dataset):
    """Returns (image_tensor, label) pairs from a HF dataset."""

    def __init__(self, hf_dataset, attribute: str):
        self.ds = hf_dataset
        self.attribute = attribute

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        label = int(row[self.attribute])
        label = 1 if label == 1 else 0
        return x, label


class MidBlockHook:
    """Persistent forward hook for capturing unet.mid_block activations."""

    def __init__(self, unet: UNet2DModel):
        self.h: Optional[torch.Tensor] = None
        self._handle = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, out):
        self.h = out.detach()

    def remove(self):
        self._handle.remove()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract semantic activations by attribute")
    p.add_argument("--attribute", type=str, required=True, help="Attribute column name, e.g. Smiling")
    p.add_argument("--dataset_dir", type=str, default="celeba_hq_dataset", help="Path to local dataset directory")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256", help="UNet model id")
    p.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddim", "ddpm"], help="Noise scheduler")
    p.add_argument("--num_steps", type=int, default=50, help="Number of scheduler inference steps")
    p.add_argument("--timestep", type=int, default=None, help="Exact scheduler timestep value to capture")
    p.add_argument("--timestep_index", type=int, default=0, help="Index in scheduler timesteps to capture")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--max_samples", type=int, default=0, help="Use only first N samples (0 means all)")
    p.add_argument("--seed", type=int, default=42, help="Seed used for noise generation")
    p.add_argument("--device", type=str, default="", help="cuda/cpu/mps (auto if empty)")
    p.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"], help="Saved tensor dtype")
    p.add_argument("--output_dir", type=str, default="semantic vector extraction/outputs", help="Where to save .pt tensors")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_scheduler(model_id: str, scheduler_type: str):
    if scheduler_type == "ddpm":
        return DDPMScheduler.from_pretrained(model_id)
    return DDIMScheduler.from_pretrained(model_id)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("=" * 72)
    print("Semantic Vector Extraction: Attribute Activation Split")
    print("=" * 72)
    print(f"attribute      : {args.attribute}")
    print(f"dataset_dir    : {args.dataset_dir}")
    print(f"model_id       : {args.model_id}")
    print(f"scheduler      : {args.scheduler_type} ({args.num_steps} steps)")
    print(f"timestep       : {args.timestep}")
    print(f"timestep_index : {args.timestep_index}")
    print(f"batch_size     : {args.batch_size}")
    print(f"device         : {device}")

    ds = load_from_disk(args.dataset_dir)
    if args.attribute not in ds.column_names:
        raise ValueError(
            f"Attribute '{args.attribute}' not found. Available columns include: {ds.column_names}"
        )

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    labels = ds[args.attribute]
    pos_count = sum(1 for x in labels if int(x) == 1)
    neg_count = len(labels) - pos_count

    print(f"num_samples     : {len(ds)}")
    print(f"positive count  : {pos_count}")
    print(f"negative count  : {neg_count}")

    dataset = AttributeImageDataset(ds, args.attribute)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    unet = UNet2DModel.from_pretrained(args.model_id).to(device).eval()
    scheduler = get_scheduler(args.model_id, args.scheduler_type)
    scheduler.set_timesteps(args.num_steps, device=device)
    scheduler_ts = [int(t.item()) for t in scheduler.timesteps]

    if args.timestep is not None:
        if args.timestep not in scheduler_ts:
            raise ValueError(
                f"timestep {args.timestep} not found in scheduler timesteps: {scheduler_ts}"
            )
        ts_value = int(args.timestep)
    else:
        if not (0 <= args.timestep_index < len(scheduler.timesteps)):
            raise ValueError(
                f"timestep_index must be in [0, {len(scheduler.timesteps) - 1}]"
            )
        ts_value = int(scheduler.timesteps[args.timestep_index].item())

    print(f"capture timestep value: {ts_value}")

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    hook = MidBlockHook(unet)
    pos_acts = None
    neg_acts = None
    pos_ptr = 0
    neg_ptr = 0

    g = torch.Generator(device=device).manual_seed(args.seed)

    pbar = tqdm(loader, desc="Extracting Activations", total=len(loader))
    for batch_idx, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device)

        t_batch = torch.full((x.shape[0],), ts_value, device=device, dtype=torch.long)
        if ts_value == 0:
            x_t = x
        else:
            eps = torch.randn(x.shape, generator=g, device=device)
            x_t = scheduler.add_noise(x, eps, t_batch)

        unet(x_t, t_batch)
        h = hook.h.to(dtype=save_dtype).cpu()

        if pos_acts is None:
            feat_shape = tuple(h.shape[1:])
            pos_acts = torch.empty((pos_count, *feat_shape), dtype=save_dtype)
            neg_acts = torch.empty((neg_count, *feat_shape), dtype=save_dtype)
            print(f"activation shape: {feat_shape}")

        pos_mask = (y == 1).cpu()
        neg_mask = ~pos_mask

        h_pos = h[pos_mask]
        h_neg = h[neg_mask]

        if h_pos.numel() > 0:
            c = h_pos.shape[0]
            pos_acts[pos_ptr:pos_ptr + c] = h_pos
            pos_ptr += c

        if h_neg.numel() > 0:
            c = h_neg.shape[0]
            neg_acts[neg_ptr:neg_ptr + c] = h_neg
            neg_ptr += c

        pbar.set_postfix(pos=f"{pos_ptr}/{pos_count}", neg=f"{neg_ptr}/{neg_count}")

    hook.remove()

    if pos_ptr != pos_count or neg_ptr != neg_count:
        raise RuntimeError(
            f"Count mismatch: pos {pos_ptr}/{pos_count}, neg {neg_ptr}/{neg_count}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_key = sanitize_name(args.attribute)
    attr_dir = out_dir / attr_key
    pos_dir = attr_dir / "positive_class"
    neg_dir = attr_dir / "negative_class"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_path = pos_dir / f"{attr_key}_positive_activations_t{ts_value}.pt"
    neg_path = neg_dir / f"{attr_key}_negative_activations_t{ts_value}.pt"

    torch.save(pos_acts, pos_path)
    torch.save(neg_acts, neg_path)

    print("\nSaved tensors:")
    print(f"  + {pos_path}  shape={tuple(pos_acts.shape)} dtype={pos_acts.dtype}")
    print(f"  + {neg_path}  shape={tuple(neg_acts.shape)} dtype={neg_acts.dtype}")
    print("Done.\n")


if __name__ == "__main__":
    main()
