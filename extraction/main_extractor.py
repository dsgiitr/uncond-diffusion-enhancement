#!/usr/bin/env python3
"""
main_extractor.py
─────────────────
Unified multi-timestep concept extraction pipeline.

For each of three contrastive pairs (sharp/blur, oversat/gray, high/low
contrast), this script:

  1.  Loads images from CelebA-HQ 256×256.
  2.  Produces plus (+) and minus (-) variants via the transform modules.
  3.  Passes both through a DDPM UNet at 10 evenly-spaced scheduler timesteps.
  4.  Captures h-space (mid-block) activations — shape [batch, 512, 8, 8].
  5.  Saves one .pt file per concept pair, keyed by scheduler timestep value.

GPU parallelism:
  • plus and minus images are **concatenated** into a single [2B, 3, 256, 256]
    tensor so only ONE forward pass is needed per timestep per batch.
  • Hook is registered once and remains active for the entire run.
  • torch.cuda.amp.autocast is used for fp16 inference (if CUDA).
  • DataLoader uses pin_memory + non-blocking .to(device) transfers.

Usage:
    python main_extractor.py                          # defaults from config.py
    python main_extractor.py --num_samples 500 --batch_size 16
    python main_extractor.py --scheduler_type ddim --num_steps 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ExtractionConfig
from dataset_utils import load_image_dataset_for_profile, preprocess_pil_for_profile
from transformations import transform_sharp_blur
from transformations import transform_gray_oversat
from transformations import transform_high_low_contrast
from transformations import transform_high_low_brightness
from transformations import transform_warm_cool


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptPairDataset(Dataset):
    """
    Wraps a HuggingFace image dataset and applies plus/minus transforms
    for every registered concept pair.

    Each __getitem__ returns a dict:
        {
            "sharp_vs_blur_plus":    Tensor[3,256,256],
            "sharp_vs_blur_minus":   Tensor[3,256,256],
            "gray_vs_oversat_plus":  ...,
            ...
        }
    """

    def __init__(
        self,
        image_dataset,
        concept_transforms: Dict[str, Tuple],    # name → (plus_tx, minus_tx)
        dataset_profile: str,
        image_size: int,
    ):
        self.image_dataset = image_dataset
        self.concept_transforms = concept_transforms
        self.dataset_profile = dataset_profile
        self.image_size = image_size

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> dict:
        pil_img = self.image_dataset[idx]
        pil_img = preprocess_pil_for_profile(
            pil_img,
            image_size=self.image_size,
            dataset_profile=self.dataset_profile,
        )

        out = {}
        for name, (plus_tx, minus_tx) in self.concept_transforms.items():
            out[f"{name}_plus"]  = plus_tx(pil_img)
            out[f"{name}_minus"] = minus_tx(pil_img)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  H-Space hook  (persistent — registered once)
# ═══════════════════════════════════════════════════════════════════════════════

class HSpaceHook:
    """
    Persistent forward hook on unet.mid_block.
    Keeps the captured activation *on GPU* so the caller can slice it
    before moving to CPU (avoids unnecessary device transfers).
    """

    def __init__(self, unet: UNet2DModel):
        self.h: torch.Tensor | None = None
        self._hook = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, output):
        self.h = output.detach()          # stays on whatever device the model uses

    def remove(self):
        self._hook.remove()


# ═══════════════════════════════════════════════════════════════════════════════
#  Core extraction loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_all(
    cfg: ExtractionConfig,
    unet: UNet2DModel,
    scheduler,
    dataloader: DataLoader,
    concept_names: List[str],
    capture_timesteps: List[int],
    device: torch.device,
) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
    """
    Run the full extraction loop.

    Returns
    -------
    results : {concept_name: {timestep_value: {"plus": T, "minus": T}}}
        where T has shape [total_samples, 512, 8, 8].
    """
    hook = HSpaceHook(unet)
    use_amp = cfg.use_amp and device.type == "cuda"

    # Accumulators: concept → timestep → "plus"/"minus" → list of cpu tensors
    accum: Dict[str, Dict[int, Dict[str, list]]] = {}
    for cname in concept_names:
        accum[cname] = {}
        for ts in capture_timesteps:
            accum[cname][ts] = {"plus": [], "minus": []}

    total_batches = len(dataloader)
    collected = 0
    t0 = time.time()

    print(f"\n{'═' * 70}")
    print(f"  Extracting h-space activations")
    print(f"  Concepts      : {concept_names}")
    print(f"  Timesteps ({len(capture_timesteps)}): {capture_timesteps}")
    print(f"  AMP fp16      : {use_amp}")
    print(f"{'═' * 70}\n")

    for batch_idx, batch in enumerate(dataloader):
        B_actual = batch[f"{concept_names[0]}_plus"].shape[0]

        for ts in capture_timesteps:
            t_vec = torch.full((B_actual,), ts, device=device, dtype=torch.long)
            t_vec_double = torch.full((2 * B_actual,), ts, device=device, dtype=torch.long)

            for cname in concept_names:
                plus_imgs  = batch[f"{cname}_plus"].to(device, non_blocking=True)
                minus_imgs = batch[f"{cname}_minus"].to(device, non_blocking=True)

                # Concatenate plus & minus into ONE tensor → single forward pass
                combined = torch.cat([plus_imgs, minus_imgs], dim=0)   # [2B, 3, 256, 256]

                # Add noise at timestep ts
                gen = torch.Generator(device=device).manual_seed(
                    cfg.seed + batch_idx * len(capture_timesteps) * len(concept_names)
                    + capture_timesteps.index(ts) * len(concept_names)
                    + concept_names.index(cname)
                )
                eps = torch.randn(combined.shape, generator=gen, device=device)

                if ts == 0:
                    x_t = combined
                else:
                    x_t = scheduler.add_noise(combined, eps, t_vec_double)

                # Forward pass with optional AMP
                if use_amp:
                    with torch.cuda.amp.autocast():
                        unet(x_t, t_vec_double)
                else:
                    unet(x_t, t_vec_double)

                h = hook.h                              # [2B, 512, 8, 8]  (on GPU)
                h_plus  = h[:B_actual].cpu()
                h_minus = h[B_actual:].cpu()

                accum[cname][ts]["plus"].append(h_plus)
                accum[cname][ts]["minus"].append(h_minus)

        collected += B_actual
        elapsed = time.time() - t0
        samples_per_sec = collected / elapsed if elapsed > 0 else 0
        print(
            f"  batch {batch_idx + 1:>4d}/{total_batches} | "
            f"{collected} samples | "
            f"{samples_per_sec:.1f} samp/s | "
            f"h-shape {tuple(hook.h.shape)}"
        )

    hook.remove()
    print(f"\n  Total time: {time.time() - t0:.1f}s\n")

    # Concatenate accumulators → final tensors
    results: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    for cname in concept_names:
        results[cname] = {}
        for ts in capture_timesteps:
            results[cname][ts] = {
                "plus":  torch.cat(accum[cname][ts]["plus"],  dim=0),
                "minus": torch.cat(accum[cname][ts]["minus"], dim=0),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Saving
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    results: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    cfg: ExtractionConfig,
    output_dir: Path,
):
    """Save one .pt file per concept pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for cname, ts_dict in results.items():
        payload = {
            "config": {
                "model_id":        cfg.model_id,
                "scheduler_type":  cfg.scheduler_type,
                "num_steps":       cfg.num_steps,
                "num_samples":     cfg.num_samples,
                "seed":            cfg.seed,
            },
            "activations": {},              # timestep → {plus, minus}
        }

        for ts, pm in ts_dict.items():
            payload["activations"][ts] = {
                "plus":  pm["plus"],        # [N, 512, 8, 8]
                "minus": pm["minus"],       # [N, 512, 8, 8]
            }

        pt_path = output_dir / f"{cname}.pt"
        torch.save(payload, pt_path)

        # Report
        sample_ts = list(ts_dict.keys())[0]
        shape = tuple(ts_dict[sample_ts]["plus"].shape)
        print(f"  [{cname}]  {len(ts_dict)} timesteps × plus/minus  shape={shape}  → {pt_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified multi-timestep concept h-space extraction."
    )
    p.add_argument("--model_id",        type=str,   default=None)
    p.add_argument("--dataset_profile", type=str,   default=None)
    p.add_argument("--dataset_dir",     type=str,   default=None)
    p.add_argument("--hf_dataset",      type=str,   default=None)
    p.add_argument("--dataset_split",   type=str,   default=None)
    p.add_argument("--num_steps",       type=int,   default=None)
    p.add_argument("--scheduler_type",  type=str,   default=None)
    p.add_argument("--num_samples",     type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--output_dir",      type=str,   default=None)
    p.add_argument("--device",          type=str,   default=None)
    p.add_argument("--no_amp",          action="store_true",
                   help="Disable fp16 autocast on CUDA.")
    p.add_argument("--concepts",         type=str,   default=None,
                   help="Comma-separated list of concept names to extract. "
                        "If omitted, all concepts are extracted. "
                        "Available: sharp_vs_blur, gray_vs_oversat, "
                        "high_vs_low_contrast, high_vs_low_brightness, "
                        "warm_vs_cool")
    return p


def apply_overrides(cfg: ExtractionConfig, args) -> ExtractionConfig:
    """Overwrite config fields with any non-None CLI args."""
    for field_name in ("model_id", "dataset_profile", "dataset_dir", "hf_dataset", "dataset_split",
                       "num_steps", "scheduler_type",
                       "num_samples", "batch_size", "seed",
                       "output_dir", "device"):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(cfg, field_name, val)
    if args.no_amp:
        cfg.use_amp = False
    return cfg


def resolve_device(cfg: ExtractionConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(ExtractionConfig(), args)
    device = resolve_device(cfg)

    print("=" * 70)
    print("  Multi-Timestep Concept Extraction Pipeline")
    print("=" * 70)
    print(f"  Model           : {cfg.model_id}")
    print(f"  Scheduler       : {cfg.scheduler_type}  ({cfg.num_steps} steps)")
    print(f"  Capture         : {cfg.num_capture_steps} timesteps every {cfg.capture_interval} steps")
    print(f"  Dataset profile : {cfg.dataset_profile}")
    print(f"  Dataset dir     : {cfg.dataset_dir}")
    print(f"  HF dataset      : {cfg.hf_dataset}")
    print(f"  Dataset split   : {cfg.dataset_split}")
    print(f"  Samples         : {cfg.num_samples}")
    print(f"  Batch size      : {cfg.batch_size}")
    print(f"  Device          : {device}")
    print(f"  AMP fp16        : {cfg.use_amp and device.type == 'cuda'}")
    print(f"  Output          : {cfg.output_dir}")
    print("=" * 70)

    # ── 1. Load model ───────────────────────────────────────────────────────
    print(f"\nLoading UNet: {cfg.model_id} ...")
    unet = UNet2DModel.from_pretrained(cfg.model_id).to(device).eval()
    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    # ── 2. Build scheduler & get capture timesteps ──────────────────────────
    if cfg.scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_pretrained(cfg.model_id)
    else:
        scheduler = DDPMScheduler.from_pretrained(cfg.model_id)

    scheduler.set_timesteps(cfg.num_steps)
    all_ts = scheduler.timesteps.tolist()       # e.g. [980, 960, …, 0]

    step_indices = cfg.capture_step_indices()    # [0, 5, 10, …, 45]
    capture_timesteps = [all_ts[i] for i in step_indices if i < len(all_ts)]

    print(f"  Scheduler steps : {len(all_ts)}")
    print(f"  Capture indices : {step_indices}")
    print(f"  Capture ts vals : {capture_timesteps}")

    # ── 3. Build transforms ─────────────────────────────────────────────────
    concept_transforms = {}

    sb_plus, sb_minus = transform_sharp_blur.get_transforms(
        image_size=cfg.image_size,
        blur_kernel_size=cfg.blur_kernel_size,
        blur_sigma=cfg.blur_sigma,
    )
    concept_transforms["sharp_vs_blur"] = (sb_plus, sb_minus)

    go_plus, go_minus = transform_gray_oversat.get_transforms(
        image_size=cfg.image_size,
        oversaturation_factor=cfg.oversaturation_factor,
    )
    concept_transforms["gray_vs_oversat"] = (go_plus, go_minus)

    hl_plus, hl_minus = transform_high_low_contrast.get_transforms(
        image_size=cfg.image_size,
        high_contrast_factor=cfg.high_contrast_factor,
        low_contrast_factor=cfg.low_contrast_factor,
    )
    concept_transforms["high_vs_low_contrast"] = (hl_plus, hl_minus)

    hb_plus, hb_minus = transform_high_low_brightness.get_transforms(
        image_size=cfg.image_size,
        high_brightness_factor=cfg.high_brightness_factor,
        low_brightness_factor=cfg.low_brightness_factor,
    )
    concept_transforms["high_vs_low_brightness"] = (hb_plus, hb_minus)

    wc_plus, wc_minus = transform_warm_cool.get_transforms(
        image_size=cfg.image_size,
        warm_strength=cfg.warm_strength,
        cool_strength=cfg.cool_strength,
    )
    concept_transforms["warm_vs_cool"] = (wc_plus, wc_minus)

    # ── Filter to requested concepts (if --concepts given) ──────────────
    if args.concepts:
        requested = [c.strip() for c in args.concepts.split(",")]
        unknown = [c for c in requested if c not in concept_transforms]
        if unknown:
            print(f"\n  ERROR: Unknown concept(s): {unknown}")
            print(f"  Available: {list(concept_transforms.keys())}")
            return
        concept_transforms = {k: v for k, v in concept_transforms.items() if k in requested}

    concept_names = list(concept_transforms.keys())
    print(f"\n  Concepts to extract: {concept_names}")

    # ── 4. Load dataset ─────────────────────────────────────────────────────
    print(f"\nLoading dataset for profile '{cfg.dataset_profile}' ...")
    image_ds = load_image_dataset_for_profile(
        dataset_profile=cfg.dataset_profile,
        dataset_dir=cfg.dataset_dir,
        hf_dataset=cfg.hf_dataset,
        dataset_split=cfg.dataset_split,
        image_key="image",
    )
    if cfg.num_samples > 0 and cfg.num_samples < len(image_ds):
        image_ds = Subset(image_ds, list(range(cfg.num_samples)))
    print(f"  Selected {len(image_ds)} samples")

    dataset = ConceptPairDataset(
        image_ds,
        concept_transforms,
        dataset_profile=cfg.dataset_profile,
        image_size=cfg.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── 5. Extract ──────────────────────────────────────────────────────────
    results = extract_all(
        cfg=cfg,
        unet=unet,
        scheduler=scheduler,
        dataloader=dataloader,
        concept_names=concept_names,
        capture_timesteps=capture_timesteps,
        device=device,
    )

    # ── 6. Save ─────────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    print("\nSaving outputs ...")
    save_results(results, cfg, output_dir)

    # ── 7. Reload instructions ──────────────────────────────────────────────
    print("\nReload example:")
    print(f"  data = torch.load('{output_dir}/sharp_vs_blur.pt')")
    print(f"  ts   = list(data['activations'].keys())")
    print(f"  h_plus  = data['activations'][ts[0]]['plus']   # [N, 512, 8, 8]")
    print(f"  h_minus = data['activations'][ts[0]]['minus']  # [N, 512, 8, 8]")
    print("\nDone. ✓")


if __name__ == "__main__":
    main()
