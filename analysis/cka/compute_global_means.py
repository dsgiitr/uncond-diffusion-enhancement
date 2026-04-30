#!/usr/bin/env python3
"""
Phase 1 — Pre-compute global activation means at multiple timesteps.

For each target layer (encoder blocks + h-space), runs a *randomly
sampled* subset of CelebA-HQ through the UNet at every measurement
timestep and accumulates streaming means of the 4D activation tensors.

Measurement timesteps are derived from the DDIM schedule:
    30 DDIM steps, bucket_every=3  →  10 measurement points.

Means are stored as raw spatial tensors (shape [C, H, W]) so they can
be pooled later to match any ``pool_spatial`` setting in Phase 2.

Output
──────
    ``<output_dir>/global_means.pt``

    Contents::

        {
            "means": {timestep_int: {layer_name: Tensor[C, H, W], ...}, ...},
            "measurement_timesteps": [t0, t3, t6, ...],
            "config": {num_steps, bucket_every, n_samples_for_mean, ...},
        }

Usage
─────
    python -m destructive_interference.compute_global_means \\
        --dataset_path celeba_hq_dataset \\
        --n_samples_for_mean 5000 --batch_size 8 \\
        --num_steps 30 --bucket_every 3
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

# ── local imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from destructive_interference.data import load_celeba_hq, build_dataloader
from destructive_interference.hooks import MultiLayerHook


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_measurement_timesteps(
    scheduler: DDIMScheduler,
    num_steps: int,
    bucket_every: int,
    device: torch.device,
) -> List[int]:
    """Derive the measurement timestep *values* from the DDIM schedule.

    Args:
        scheduler:    Configured DDIM scheduler.
        num_steps:    Total number of DDIM steps (e.g. 30).
        bucket_every: Measure at every Nth step index (e.g. 3).
        device:       Device for the scheduler.

    Returns:
        List of integer timestep values (from highest noise → lowest).
    """
    scheduler.set_timesteps(num_steps, device=device)
    all_timesteps = scheduler.timesteps.cpu().tolist()
    # Step indices: 0, bucket_every, 2*bucket_every, ...
    indices = list(range(0, num_steps, bucket_every))
    return [int(all_timesteps[i]) for i in indices]


# ═══════════════════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_global_means(
    model_id: str = "google/ddpm-celebahq-256",
    dataset_path: str = "celeba_hq_dataset",
    dataset_profile: str = "celeba_hq",
    dataset_split: str = "train",
    n_samples_for_mean: int = 5000,
    batch_size: int = 8,
    num_workers: int = 4,
    num_steps: int = 30,
    bucket_every: int = 3,
    device_str: str = "",
    seed: int = 42,
    image_size: int = 256,
    use_amp: bool = True,
    output_dir: str = "destructive_interference/outputs",
    from_disk: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute and save global activation means at multiple timesteps.

    Returns:
        ``{timestep: {layer_name: Tensor[C,H,W], ...}, ...}``
    """

    device = resolve_device(device_str)
    print(f"\n{'═' * 70}")
    print(f"  Phase 1 — Multi-Timestep Global Activation Means")
    print(f"{'═' * 70}")
    print(f"  Model              : {model_id}")
    print(f"  Dataset            : {dataset_path}")
    print(f"  Profile            : {dataset_profile}")
    print(f"  Split              : {dataset_split}")
    print(f"  N samples for mean : {n_samples_for_mean}")
    print(f"  Batch size         : {batch_size}")
    print(f"  DDIM steps         : {num_steps}")
    print(f"  Bucket every       : {bucket_every} steps")
    print(f"  Device             : {device}")
    print(f"  AMP fp16           : {use_amp and device.type == 'cuda'}")
    print(f"  Seed               : {seed}")
    print(f"{'═' * 70}\n")

    # ── 1. Model ────────────────────────────────────────────────────────────
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id)

    # ── 2. Derive measurement timesteps ─────────────────────────────────────
    measurement_ts = get_measurement_timesteps(
        scheduler, num_steps, bucket_every, device,
    )
    print(f"  Measurement timesteps ({len(measurement_ts)}):")
    print(f"    {measurement_ts}\n")

    # ── 3. Hooks ────────────────────────────────────────────────────────────
    hook = MultiLayerHook(unet)
    layer_names = hook.layer_names
    print(f"  Hooking layers: {layer_names}\n")

    # ── 4. Data — randomly sample N indices ─────────────────────────────────
    # Load full dataset to know its length, then pick a random subset
    rng = torch.Generator().manual_seed(seed)
    dataset = load_celeba_hq(
        dataset_path=dataset_path,
        hf_id=dataset_path,
        dataset_profile=dataset_profile,
        dataset_split=dataset_split,
        indices=None,          # load all — indices selected below
        num_samples=None,
        from_disk=from_disk,
        image_size=image_size,
    )
    total_available = len(dataset)
    n_to_use = min(n_samples_for_mean, total_available)

    # Random permutation → take first n_to_use
    perm = torch.randperm(total_available, generator=rng).tolist()
    selected_indices = sorted(perm[:n_to_use])

    # Rebuild dataset with selected indices
    dataset = load_celeba_hq(
        dataset_path=dataset_path,
        hf_id=dataset_path,
        dataset_profile=dataset_profile,
        dataset_split=dataset_split,
        indices=selected_indices,
        from_disk=from_disk,
        image_size=image_size,
    )
    print(f"  Selected {n_to_use} / {total_available} samples (seed={seed})\n")

    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── 5. Compute means at each measurement timestep ───────────────────────
    amp_enabled = use_amp and device.type == "cuda"
    all_means: Dict[int, Dict[str, torch.Tensor]] = {}
    t0_total = time.time()

    for ts_idx, ts_value in enumerate(measurement_ts):
        print(f"\n  ── Timestep {ts_value} ({ts_idx+1}/{len(measurement_ts)}) ──")

        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {n: 0 for n in layer_names}
        generator = torch.Generator(device=device).manual_seed(seed)

        t0 = time.time()
        for batch_idx, images in enumerate(tqdm(
            dataloader, desc=f"  t={ts_value}", leave=False,
        )):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            # Noise at target timestep
            t_vec = torch.full((B,), ts_value, device=device, dtype=torch.long)
            eps = torch.randn(images.shape, generator=generator, device=device)
            if ts_value > 0:
                x_t = scheduler.add_noise(images, eps, t_vec)
            else:
                x_t = images

            # Forward pass
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    unet(x_t, t_vec)
            else:
                unet(x_t, t_vec)

            # Accumulate per-layer sums
            acts = hook.get_activations()
            for name in layer_names:
                a = acts[name]                                  # [B, C, H, W]
                batch_sum = a.sum(dim=0, dtype=torch.float64)   # [C, H, W]
                if name not in sums:
                    sums[name] = batch_sum
                else:
                    sums[name] += batch_sum
                counts[name] += B

            hook.clear()

        elapsed = time.time() - t0

        # Compute means for this timestep
        means_t: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            means_t[name] = (sums[name] / counts[name]).to(torch.float32).cpu()
        all_means[ts_value] = means_t

        print(f"    Done ({elapsed:.1f}s, {counts[layer_names[0]]} samples)")
        for name in layer_names:
            print(f"    {name:20s}  shape={list(means_t[name].shape)}")

    total_elapsed = time.time() - t0_total

    # ── 6. Save ─────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "global_means.pt"
    torch.save({
        "means": all_means,
        "measurement_timesteps": measurement_ts,
        "config": {
            "model_id": model_id,
            "num_steps": num_steps,
            "bucket_every": bucket_every,
            "n_samples_for_mean": n_to_use,
            "seed": seed,
        },
    }, out_path)

    hook.remove()
    del unet
    torch.cuda.empty_cache()

    print(f"\n  Saved → {out_path}")
    print(f"  Total time: {total_elapsed:.1f}s\n")
    return all_means


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: multi-timestep global activation means",
    )
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--dataset_path", type=str, default="celeba_hq_dataset")
    p.add_argument("--dataset_profile", type=str, default="celeba_hq", choices=["celeba_hq", "lsun_church"])
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--n_samples_for_mean", type=int, default=5000,
                   help="Number of images to randomly sample from the dataset.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=30,
                   help="Total DDIM inference steps.")
    p.add_argument("--bucket_every", type=int, default=3,
                   help="Measure at every Nth step.")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--output_dir", type=str,
                   default="destructive_interference/outputs")
    p.add_argument("--from_hub", action="store_true",
                   help="Load dataset from HuggingFace Hub instead of disk.")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    compute_global_means(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        dataset_profile=args.dataset_profile,
        dataset_split=args.dataset_split,
        n_samples_for_mean=args.n_samples_for_mean,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_steps=args.num_steps,
        bucket_every=args.bucket_every,
        device_str=args.device,
        seed=args.seed,
        image_size=args.image_size,
        use_amp=not args.no_amp,
        output_dir=args.output_dir,
        from_disk=not args.from_hub,
    )
