#!/usr/bin/env python3
"""
Unconditional DDPM — h-space patching + CFG comparison.

Runs three generation modes from the SAME initial noise and saves
a 1×3 subplot comparison per batch item.

Modes
-----
1. BASELINE   — clean UNet, no patching
2. PATCHED    — single-pass direct h-space patching
3. CFG        — dual-pass h-space CFG (batch-doubled for GPU efficiency)

Usage
-----
# Generate a dummy v first, then run:
python main.py --v-path v.pt --v-scale 2.0 --cfg-scale 5.0 --output-dir smiling

python main.py --v-path eigenvectors/v3.pt --output-dir eigvec_3 \\
               --v-scale 1.5 --cfg-scale 3.0 --patch-mode interval \\
               --patch-start 0 --patch-end 15 --steps 50 --seed 42
"""

import argparse
import os

import torch
from diffusers import DDPMPipeline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DDPMConfig
from generation.hooks import HSpacePatcher
from generation.pipeline import (
    SCHEDULER_MAP,
    build_scheduler,
    generate_initial_noise,
    run_all,
)
from generation.visualize import save_comparison

# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unconditional DDPM: baseline + patched + h-space CFG "
                    "→ 1×3 subplot comparison"
    )

    defaults = DDPMConfig()

    # ── model / scheduler ───────────────────────────────────────────────
    p.add_argument("--model", type=str, default=defaults.model_id)
    p.add_argument(
        "--scheduler", type=str, default=defaults.scheduler_type,
        choices=list(SCHEDULER_MAP.keys()),
    )
    p.add_argument("--steps", type=int, default=defaults.num_inference_steps)

    # ── h-space CFG ─────────────────────────────────────────────────────
    p.add_argument("--cfg-scale", type=float, default=defaults.cfg_scale)

    # ── patching ────────────────────────────────────────────────────────
    p.add_argument("--v-path", type=str, required=True,
                   help="Path to the direction vector v (.pt file)")
    p.add_argument("--v-scale", type=float, default=defaults.v_scale)
    p.add_argument("--target-layer", type=str, default=defaults.target_layer)

    p.add_argument(
        "--patch-mode", type=str, default=defaults.patch_mode,
        choices=["continuous", "interval", "list"],
    )
    p.add_argument("--patch-start", type=int, default=defaults.patch_start)
    p.add_argument("--patch-end", type=int, default=defaults.patch_end)
    p.add_argument(
        "--patch-timesteps", type=int, nargs="+",
        default=defaults.patch_timesteps or [],
        help="Explicit step indices (for --patch-mode list)",
    )

    # ── generation ──────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )

    # ── output ──────────────────────────────────────────────────────────
    p.add_argument("--output-dir", type=str, required=True,
                   help="Top-level output directory")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def _build_subfolder_name(args: argparse.Namespace) -> str:
    """Encode all important hyper-parameters into a subfolder name."""
    parts = [
        f"v-scale_{args.v_scale}",
        f"cfg-scale_{args.cfg_scale}",
        f"patch_{args.patch_mode}",
    ]
    if args.patch_mode == "interval":
        parts.append(f"{args.patch_start}-{args.patch_end}")
    elif args.patch_mode == "list":
        ts = "-".join(str(t) for t in sorted(args.patch_timesteps))
        parts.append(f"ts_{ts}")
    parts += [
        f"steps_{args.steps}",
        f"seed_{args.seed}",
        f"sched_{args.scheduler}",
    ]
    return "__".join(parts)


@torch.no_grad()
def main():
    args = parse_args()

    # ── Resolve output directory ────────────────────────────────────────
    subfolder = _build_subfolder_name(args)
    save_dir = os.path.join(args.output_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    # ── Print config ────────────────────────────────────────────────────
    print("=" * 70)
    print("UNCONDITIONAL DDPM – CONFIGURATION")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:25s} = {v}")
    print(f"  {'save_dir':25s} = {save_dir}")
    print("=" * 70)

    # ── Enable cuDNN benchmark for fixed-size U-Net ─────────────────────
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Load DDPM model ─────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    pipe = DDPMPipeline.from_pretrained(args.model).to(args.device)
    unet = pipe.unet
    scheduler = build_scheduler(args.scheduler, pipe)

    # ── Load direction vector v ─────────────────────────────────────────
    if not os.path.exists(args.v_path):
        raise FileNotFoundError(
            f"Direction vector not found at '{args.v_path}'. "
            "Generate or provide one first."
        )
    v = torch.load(args.v_path, map_location=args.device, weights_only=False)
    print(f"Loaded v  shape={tuple(v.shape)}  scale={args.v_scale}")
    patcher = HSpacePatcher(v, scale=args.v_scale)
    target_layer = getattr(unet, args.target_layer)

    # ── Generate shared starting noise x_T ──────────────────────────────
    x_T = generate_initial_noise(unet, args.batch_size, args.seed, args.device)
    print(f"x_T shape = {tuple(x_T.shape)}")

    # ── Run all three modes ─────────────────────────────────────────────
    baseline_imgs, patched_imgs, cfg_imgs = run_all(
        unet, scheduler, x_T, patcher, target_layer,
        num_steps=args.steps,
        seed=args.seed,
        device=args.device,
        cfg_scale=args.cfg_scale,
        patch_mode=args.patch_mode,
        patch_start=args.patch_start,
        patch_end=args.patch_end,
        patch_timesteps=args.patch_timesteps,
    )

    # ── Save 1×3 subplot per batch item ─────────────────────────────────
    print(f"\nSaving {args.batch_size} comparison plot(s) …")
    saved = save_comparison(
        baseline_imgs, patched_imgs, cfg_imgs,
        save_dir,
        v_scale=args.v_scale,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        scheduler_name=args.scheduler,
        num_steps=args.steps,
        patch_mode=args.patch_mode,
        patch_start=args.patch_start,
        patch_end=args.patch_end,
    )

    print(f"\n{'=' * 70}")
    print(f"Done!  All outputs in: {os.path.abspath(save_dir)}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
