#!/usr/bin/env python3
"""
generate_triplet_dataset.py
-------------------------
Generates a dataset of image triplets (Baseline, Patched, Guided) 
by batch processing unconditional DDPM generations.

Ensures zero duplication by incrementing the generator seed per batch.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from diffusers import DDPMPipeline
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DDPMConfig
from hooks import HSpacePatcher
from pipeline import (
    SCHEDULER_MAP,
    build_scheduler,
    generate_initial_noise,
    run_fused_doublet,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dataset Generator: patched + h-space CFG"
    )

    defaults = DDPMConfig()

    p.add_argument("--n-samples", type=int, required=True, help="Total number of samples to generate")
    p.add_argument("--output-dir", type=str, required=True, help="Top-level output directory for dataset")

    # ── model / scheduler ──
    p.add_argument("--model", type=str, default=defaults.model_id)
    p.add_argument(
        "--scheduler", type=str, default=defaults.scheduler_type,
        choices=list(SCHEDULER_MAP.keys()),
    )
    p.add_argument("--steps", type=int, default=defaults.num_inference_steps)

    # ── h-space CFG ──
    p.add_argument("--cfg-scale", type=float, default=defaults.cfg_scale)

    # ── patching ──
    p.add_argument("--v-path", type=str, required=True,
                   help="Path to the direction vector v (.pt file)")
    p.add_argument("--v-scale-patched", type=float, default=defaults.v_scale)
    p.add_argument("--v-scale-guided", type=float, default=defaults.v_scale)
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
    )

    # ── generation ──
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # Create parameterized subdirectories
    v_filename = os.path.basename(args.v_path)
    concept = v_filename.split("_dom_t")[0] if "_dom_t" in v_filename else v_filename.replace(".pt", "")
    folder_name = f"{concept}_vp{args.v_scale_patched}_vg{args.v_scale_guided}_cfg{args.cfg_scale}"
    
    patched_dir  = os.path.join(args.output_dir, folder_name, "patched")
    guided_dir   = os.path.join(args.output_dir, folder_name, "guided")

    os.makedirs(patched_dir, exist_ok=True)
    os.makedirs(guided_dir, exist_ok=True)

    print("=" * 70)
    print("DATASET GENERATOR – CONFIGURATION")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:25s} = {v}")
    print("=" * 70)

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Load Context ─────────────────────────────────────────────────
    print(f"\nLoading model: {args.model} (fp16)")
    pipe = DDPMPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
    unet = pipe.unet
    scheduler = build_scheduler(args.scheduler, pipe)

    if not os.path.exists(args.v_path):
        raise FileNotFoundError(
            f"Direction vector not found at '{args.v_path}'."
        )
    v = torch.load(args.v_path, map_location=args.device, weights_only=False).to(torch.float16)
    patcher_patched = HSpacePatcher(v, scale=args.v_scale_patched)
    patcher_guided  = HSpacePatcher(v, scale=args.v_scale_guided)
    target_layer = getattr(unet, args.target_layer)

    # ── Generation Loop ───────────────────────────────────────────────
    global_idx = 0
    pbar = tqdm(total=args.n_samples, desc="Generating Dataset")

    def _save_batch(p_imgs, g_imgs, start_idx, n):
        """Save a batch to disk (runs on a background thread)."""
        for i in range(n):
            idx = start_idx + i
            p_imgs[i].save(os.path.join(patched_dir,  f"{idx:05d}.png"))
            g_imgs[i].save(os.path.join(guided_dir,   f"{idx:05d}.png"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        while global_idx < args.n_samples:
            current_batch_size = min(args.batch_size, args.n_samples - global_idx)

            # Advance seed by global_idx to guarantee exactly NO duplicates
            current_seed = args.seed + global_idx

            x_T = generate_initial_noise(unet, current_batch_size, current_seed, args.device).to(torch.float16)

            common_args = dict(
                num_steps=args.steps,
                seed=current_seed,
                device=args.device,
                patch_mode=args.patch_mode,
                patch_start=args.patch_start,
                patch_end=args.patch_end,
                patch_timesteps=args.patch_timesteps,
            )

            patched_imgs, cfg_imgs = run_fused_doublet(
                unet, scheduler, x_T,
                patcher_patched, patcher_guided, target_layer,
                cfg_scale=args.cfg_scale,
                **common_args,
            )

            # Save asynchronously — disk I/O overlaps with next batch GPU work
            executor.submit(_save_batch, patched_imgs, cfg_imgs,
                            global_idx, current_batch_size)

            global_idx += current_batch_size
            pbar.update(current_batch_size)

    pbar.close()
    print(f"\nSuccessfully generated {args.n_samples} generated samples.")
    print(f"Check your output subdirectories inside '{os.path.abspath(args.output_dir)}/'.")


if __name__ == "__main__":
    main()
