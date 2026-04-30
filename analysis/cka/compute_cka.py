#!/usr/bin/env python3
"""
Phase 2 — Compute CKA between h-space and encoder layers.

Uses pre-computed global means from Phase 1 for kernel centering,
then accumulates HSIC scores across mini-batches.

Two pooling modes
─────────────────
  ``--pool_spatial 1``   (default)  Global average pool → D = C.
      Uses cross-covariance accumulation (:class:`MiniBatchCKA`).
      Consistent spatial treatment for all layers.

  ``--pool_spatial 0``   No pooling → D = C·H·W.
      Uses Gram-matrix approach (:class:`GramCKA`).
      More spatially faithful, higher memory cost.

Output
──────
    ``<output_dir>/cka_results.pt``  — scores dict + config.
    ``<output_dir>/cka_barplot.png`` — visualisation.

Usage
─────
    # Phase 2 (assumes Phase 1 already ran)
    python -m destructive_interference.compute_cka \\
        --means_path destructive_interference/outputs/global_means.pt \\
        --num_samples 500 --batch_size 8 --pool_spatial 1
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm

# ── local imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from destructive_interference.data import load_celeba_hq, build_dataloader
from destructive_interference.hooks import MultiLayerHook
from destructive_interference.cka_core import (
    MiniBatchCKA,
    GramCKA,
    adaptive_pool_flatten,
)
from destructive_interference.visualize import plot_cka_bar


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pool_mean(mean_4d: torch.Tensor, pool_spatial: Optional[int]) -> torch.Tensor:
    """Pool a raw [C, H, W] mean tensor and flatten to [D]."""
    return adaptive_pool_flatten(mean_4d.unsqueeze(0), pool_spatial).squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_cka(
    means_path: str = "destructive_interference/outputs/global_means.pt",
    model_id: str = "google/ddpm-celebahq-256",
    dataset_path: str = "celeba_hq_dataset",
    dataset_profile: str = "celeba_hq",
    dataset_split: str = "train",
    indices=None,
    num_samples: int = 500,
    batch_size: int = 8,
    num_workers: int = 4,
    timestep: int = 20,
    scheduler_type: str = "ddim",
    num_steps: int = 50,
    pool_spatial: Optional[int] = 1,
    device_str: str = "",
    seed: int = 42,
    image_size: int = 256,
    use_amp: bool = True,
    output_dir: str = "destructive_interference/outputs",
    from_disk: bool = True,
) -> Dict[str, float]:
    """Compute CKA scores between h-space and each encoder block.

    Returns:
        ``{layer_name: cka_score}`` for each encoder block.
    """
    device = resolve_device(device_str)
    pool_label = f"{pool_spatial}×{pool_spatial}" if pool_spatial else "none"

    print(f"\n{'═' * 70}")
    print(f"  Phase 2 — CKA Computation")
    print(f"{'═' * 70}")
    print(f"  Global means  : {means_path}")
    print(f"  Model         : {model_id}")
    print(f"  Dataset       : {dataset_path}")
    print(f"  Profile       : {dataset_profile}")
    print(f"  Split         : {dataset_split}")
    print(f"  Samples       : {num_samples}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Timestep      : {timestep}")
    print(f"  Pool spatial  : {pool_label}")
    print(f"  Device        : {device}")
    print(f"{'═' * 70}\n")

    # ── 1. Load global means ────────────────────────────────────────────────
    means_data = torch.load(means_path, map_location="cpu", weights_only=True)
    raw_means: Dict[str, torch.Tensor] = means_data["means"]
    saved_cfg = means_data["config"]

    # Validate timestep consistency
    if saved_cfg["timestep"] != timestep:
        print(f"  ⚠  Timestep mismatch: means computed at t={saved_cfg['timestep']}, "
              f"but CKA requested at t={timestep}.")
        print(f"     Re-run Phase 1 with --timestep {timestep} for accurate results.\n")

    # ── 2. Identify encoder vs reference layers ─────────────────────────────
    encoder_layers = sorted([k for k in raw_means if k.startswith("down_block_")])
    ref_layer = "mid_block"
    assert ref_layer in raw_means, f"Global means missing '{ref_layer}'"

    print(f"  Reference     : {ref_layer}")
    print(f"  Encoder layers: {encoder_layers}\n")

    # Pool means to target spatial resolution
    pooled_means: Dict[str, torch.Tensor] = {}
    for name in encoder_layers + [ref_layer]:
        pooled_means[name] = _pool_mean(raw_means[name], pool_spatial)
        print(f"  {name:20s}  raw={list(raw_means[name].shape):>16s}"
              f"  → pooled D={pooled_means[name].numel()}")

    # ── 3. Create CKA accumulators ──────────────────────────────────────────
    ref_mean = pooled_means[ref_layer]
    use_gram = pool_spatial is None or pool_spatial == 0

    cka_engines: Dict[str, MiniBatchCKA | GramCKA] = {}
    for name in encoder_layers:
        enc_mean = pooled_means[name]
        if use_gram:
            cka_engines[name] = GramCKA(enc_mean, ref_mean)
        else:
            cka_engines[name] = MiniBatchCKA(enc_mean, ref_mean, device=device)

    backend_label = "GramCKA (no pooling)" if use_gram else "MiniBatchCKA (cross-cov)"
    print(f"\n  CKA backend   : {backend_label}\n")

    # ── 4. Load model ───────────────────────────────────────────────────────
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    if scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_pretrained(model_id)
    else:
        scheduler = DDPMScheduler.from_pretrained(model_id)
    scheduler.set_timesteps(num_steps, device=device)

    hook = MultiLayerHook(unet)

    # ── 5. Data ─────────────────────────────────────────────────────────────
    dataset = load_celeba_hq(
        dataset_path=dataset_path,
        hf_id=dataset_path,
        dataset_profile=dataset_profile,
        dataset_split=dataset_split,
        indices=indices,
        num_samples=num_samples,
        from_disk=from_disk,
        image_size=image_size,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── 6. Mini-batch accumulation ──────────────────────────────────────────
    amp_enabled = use_amp and device.type == "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()

    for batch_idx, images in enumerate(tqdm(dataloader, desc="CKA accumulation")):
        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        # Noise at target timestep
        t_vec = torch.full((B,), timestep, device=device, dtype=torch.long)
        eps = torch.randn(images.shape, generator=generator, device=device)
        if timestep > 0:
            x_t = scheduler.add_noise(images, eps, t_vec)
        else:
            x_t = images

        # Forward pass
        if amp_enabled:
            with torch.cuda.amp.autocast():
                unet(x_t, t_vec)
        else:
            unet(x_t, t_vec)

        acts = hook.get_activations()

        # Pool + flatten reference (h-space)
        ref_flat = adaptive_pool_flatten(acts[ref_layer], pool_spatial)  # [B, D_ref]

        # Update each encoder-vs-hspace CKA pair
        for name in encoder_layers:
            enc_flat = adaptive_pool_flatten(acts[name], pool_spatial)   # [B, D_enc]
            cka_engines[name].update(enc_flat, ref_flat)

        hook.clear()

    elapsed = time.time() - t0

    # ── 7. Compute final CKA scores ─────────────────────────────────────────
    print(f"\n  Computing final CKA scores …\n")
    cka_scores: Dict[str, float] = {}
    for name in encoder_layers:
        if use_gram:
            score = cka_engines[name].compute(device=device)
        else:
            score = cka_engines[name].compute()
        cka_scores[name] = score
        print(f"  CKA({name:20s}, {ref_layer})  =  {score:.6f}")

    # ── 8. Save results ─────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cka_scores": cka_scores,
        "ref_layer": ref_layer,
        "encoder_layers": encoder_layers,
        "config": {
            "model_id": model_id,
            "timestep": timestep,
            "num_samples": num_samples,
            "pool_spatial": pool_spatial,
            "scheduler_type": scheduler_type,
            "num_steps": num_steps,
            "seed": seed,
            "elapsed_s": elapsed,
        },
    }

    results_path = out_dir / "cka_results.pt"
    torch.save(results, results_path)
    print(f"\n  Results → {results_path}")

    # ── 9. Visualise ────────────────────────────────────────────────────────
    fig_path = out_dir / "cka_barplot.png"
    plot_cka_bar(cka_scores, ref_layer=ref_layer, timestep=timestep,
                 pool_spatial=pool_spatial, save_path=str(fig_path))
    print(f"  Figure  → {fig_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    hook.remove()
    del unet
    torch.cuda.empty_cache()

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Done. ✓\n")
    return cka_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2: CKA computation")
    p.add_argument("--means_path", type=str,
                   default="destructive_interference/outputs/global_means.pt",
                   help="Path to Phase 1 global_means.pt")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--dataset_path", type=str, default="celeba_hq_dataset")
    p.add_argument("--dataset_profile", type=str, default="celeba_hq", choices=["celeba_hq", "lsun_church"])
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--indices_file", type=str, default=None,
                   help="JSON / text file with handpicked sample indices.")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--timestep", type=int, default=20)
    p.add_argument("--scheduler_type", type=str, default="ddim")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--pool_spatial", type=int, default=1,
                   help="Spatial pooling size. 1 = global avg pool (default). "
                        "0 = no pooling (Gram matrix approach). "
                        "8 = pool to 8×8.")
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
    compute_cka(
        means_path=args.means_path,
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        dataset_profile=args.dataset_profile,
        dataset_split=args.dataset_split,
        indices=args.indices_file,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        timestep=args.timestep,
        scheduler_type=args.scheduler_type,
        num_steps=args.num_steps,
        pool_spatial=args.pool_spatial if args.pool_spatial > 0 else None,
        device_str=args.device,
        seed=args.seed,
        image_size=args.image_size,
        use_amp=not args.no_amp,
        output_dir=args.output_dir,
        from_disk=not args.from_hub,
    )
