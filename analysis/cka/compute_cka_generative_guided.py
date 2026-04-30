#!/usr/bin/env python3
"""
Phase 2 — Dataset-free CKA via patched DDIM reverse process (Guided).

Generates samples from pure Gaussian noise using h-space patched DDIM
sampling with Classifier-Free Guidance (CFG). Captures activations at every
``bucket_every`` step for the *unpatched* branch of the CFG pass to measure
the representational dynamics of the guided trajectory.
Uses pre-computed global means (from Phase 1) for kernel centering.

Completely independent of any image dataset.

Output
──────
    ``<output_dir>/cka_generative_guided_results.pt``
    ``<output_dir>/cka_guided_heatmap.png``
    ``<output_dir>/cka_guided_trajectory.png``
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

# ── local imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from destructive_interference.hooks import MultiLayerHook
from destructive_interference.cka_core import (
    MiniBatchCKA,
    adaptive_pool_flatten,
)
from destructive_interference.visualize import plot_cka_heatmap, plot_cka_trajectory


# ═══════════════════════════════════════════════════════════════════════════════
#  H-Space Patcher (Batch-Doubled for CFG)
# ═══════════════════════════════════════════════════════════════════════════════


class _BatchDoubledHSpacePatchHook:
    """Lightweight forward hook that adds ``scale * v`` to the second half 
    of a batch-doubled mid-block output.

    The CKA activation hooks fire *after* this hook in registration order, so they
    capture both the unpatched (first half) and patched (second half) mid-block outputs.
    """

    def __init__(self, v: torch.Tensor, scale: float = 1.0, half_B: int = 1):
        self.v = v
        self.scale = scale
        self.half_B = half_B
        self._handle = None

    def _hook_fn(self, module, inp, output):
        # output shape: [2B, C, H, W]
        patched_output = output.clone()
        patched_output[self.half_B:] = output[self.half_B:] + self.scale * self.v
        return patched_output

    def register(self, layer):
        self.remove()
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
def compute_cka_generative_guided(
    means_path: str = "destructive_interference/outputs/global_means.pt",
    vector_path: str = "",
    v_scale: float = 1.0,
    cfg_scale: float = 3.0,
    model_id: str = "google/ddpm-celebahq-256",
    num_samples: int = 256,
    batch_size: int = 4,
    num_steps: int = 30,
    bucket_every: int = 3,
    pool_spatial: Optional[int] = 1,
    device_str: str = "",
    seed: int = 42,
    use_amp: bool = True,
    output_dir: str = "destructive_interference/outputs",
    use_batch_mean: bool = False,
) -> Dict[int, Dict[str, float]]:
    """Compute CKA across denoising trajectory using Guided DDIM (CFG).

    Returns:
        ``{timestep: {layer_name: cka_score}}``
    """
    device = resolve_device(device_str)
    pool_label = f"{pool_spatial}×{pool_spatial}" if pool_spatial else "none"

    print(f"\n{'═' * 70}")
    print(f"  Phase 2 — Generative CKA (Guided CFG)")
    print(f"{'═' * 70}")
    print(f"  Global means   : {means_path}")
    print(f"  Vector path    : {vector_path}")
    print(f"  v_scale        : {v_scale}")
    print(f"  cfg_scale      : {cfg_scale}")
    print(f"  Model          : {model_id}")
    print(f"  Num samples    : {num_samples}")
    print(f"  Batch size     : {batch_size} (inner doubled)")
    print(f"  DDIM steps     : {num_steps}")
    print(f"  Bucket every   : {bucket_every}")
    print(f"  Pool spatial   : {pool_label}")
    print(f"  Centering      : {'batch-local mean' if use_batch_mean else 'pre-computed global mean'}")
    print(f"  Device         : {device}")
    print(f"  Seed           : {seed}")
    print(f"{'═' * 70}\n")

    # ── 1. Load global means ────────────────────────────────────────────────
    means_data = torch.load(means_path, map_location="cpu", weights_only=True)
    raw_means: Dict[int, Dict[str, torch.Tensor]] = means_data["means"]
    saved_measurement_ts: List[int] = means_data["measurement_timesteps"]

    print(f"  Loaded means for {len(saved_measurement_ts)} timesteps:")
    print(f"    {saved_measurement_ts}\n")

    # ── 2. Load model ───────────────────────────────────────────────────────
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id)
    scheduler.set_timesteps(num_steps, device=device)
    all_timesteps = scheduler.timesteps

    # Determine measurement step indices and their timestep values
    measurement_step_indices = set(range(0, num_steps, bucket_every))
    step_to_ts = {}
    for step_idx, ts in enumerate(all_timesteps):
        if step_idx in measurement_step_indices:
            step_to_ts[step_idx] = int(ts.item())

    # Validate that means exist for all measurement timesteps
    for step_idx in sorted(step_to_ts.keys()):
        ts_val = step_to_ts[step_idx]
        if ts_val not in raw_means:
            raise ValueError(
                f"Global means missing for timestep {ts_val} (step {step_idx}). "
                f"Available: {list(raw_means.keys())}. "
                f"Re-run Phase 1 with matching --num_steps / --bucket_every."
            )

    # ── 3. Load direction vector & set up batch-doubled patcher ─────────────
    patch_hook = None
    if vector_path:
        v = torch.load(vector_path, map_location=device, weights_only=False)
        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        v = v.to(device, dtype=torch.float32)
        print(f"  Loaded direction vector: shape={list(v.shape)}, v_scale={v_scale}")

        # half_B will be updated dynamically in the loop based on actual batch size
        patch_hook = _BatchDoubledHSpacePatchHook(v, scale=v_scale, half_B=batch_size)
        patch_hook.register(unet.mid_block)
        print(f"  Batch-doubled h-space patcher registered on mid_block for Guided CFG\n")
    else:
        print(f"  No direction vector → running UNPATCHED baseline (CFG scale ineffective)\n")

    # ── 4. Identify layers ──────────────────────────────────────────────────
    first_ts = saved_measurement_ts[0]
    all_layer_names = list(raw_means[first_ts].keys())
    encoder_layers = sorted([k for k in all_layer_names if k.startswith("down_block_")])
    ref_layer = "mid_block"
    assert ref_layer in all_layer_names, f"Means missing '{ref_layer}'"

    print(f"  Reference      : {ref_layer}")
    print(f"  Encoder layers : {encoder_layers}")

    # ── 5. Create CKA accumulators per (timestep, layer) ────────────────────
    cka_engines: Dict[int, Dict[str, MiniBatchCKA]] = {}
    for step_idx in sorted(step_to_ts.keys()):
        ts_val = step_to_ts[step_idx]
        cka_engines[ts_val] = {}
        ref_mean = _pool_mean(raw_means[ts_val][ref_layer], pool_spatial)
        for name in encoder_layers:
            enc_mean = _pool_mean(raw_means[ts_val][name], pool_spatial)
            cka_engines[ts_val][name] = MiniBatchCKA(
                enc_mean, ref_mean, device=device,
                use_batch_mean=use_batch_mean,
            )

    print(f"\n  Created {len(cka_engines) * len(encoder_layers)} CKA accumulators\n")

    # ── 6. Register activation hooks ────────────────────────────────────────
    # Registered AFTER patch hook, so it captures activations of both [unpatched, patched]
    hook = MultiLayerHook(unet)

    # ── 7. Per-batch HSIC storage ───────────────────────────────────────────
    per_batch_hsic: Dict[int, Dict[str, Dict[str, list]]] = {}
    for ts_val in cka_engines:
        per_batch_hsic[ts_val] = {}
        for name in encoder_layers:
            per_batch_hsic[ts_val][name] = {
                "hsic_xy": [], "hsic_xx": [], "hsic_yy": [],
            }

    # ── 8. Generation loop (Guided CFG) ─────────────────────────────────────
    amp_enabled = use_amp and device.type == "cuda"
    num_batches = (num_samples + batch_size - 1) // batch_size
    generated = 0
    t0 = time.time()

    generator = torch.Generator(device=device).manual_seed(seed)

    for batch_idx in tqdm(range(num_batches), desc="Generating guided samples"):
        B = min(batch_size, num_samples - generated)

        # Start from pure Gaussian noise
        x_t = torch.randn(
            B,
            unet.config.in_channels,
            unet.config.sample_size,
            unet.config.sample_size,
            generator=generator,
            device=device,
        )
        x_t = x_t * scheduler.init_noise_sigma

        # Update half_B in patch hook dynamically in case the last batch is smaller
        if patch_hook is not None:
            patch_hook.half_B = B

        # ── DDIM reverse process ────────────────────────────────────────
        for step_idx, t in enumerate(all_timesteps):
            is_measurement = step_idx in measurement_step_indices
            
            latent_input = scheduler.scale_model_input(x_t, t)
            
            # Batch doubling for CFG
            doubled_input = torch.cat([latent_input, latent_input], dim=0)
            t_doubled = t.expand(2 * B)

            # Forward pass
            if amp_enabled:
                with torch.cuda.amp.autocast():
                    noise_both = unet(doubled_input, t_doubled).sample
            else:
                noise_both = unet(doubled_input, t_doubled).sample

            # Split noise
            noise_unpatched = noise_both[:B]
            noise_patched = noise_both[B:]
            
            # H-space CFG
            noise_pred = noise_unpatched + cfg_scale * (noise_patched - noise_unpatched)

            # Capture activations at measurement steps
            if is_measurement:
                ts_val = step_to_ts[step_idx]
                acts = hook.get_activations()

                # Using [B:] because we are computing CKA for the *patched* path
                # inside the guided generation, per user request.
                ref_flat = adaptive_pool_flatten(
                    acts[ref_layer][B:], pool_spatial,
                )

                for name in encoder_layers:
                    enc_flat = adaptive_pool_flatten(
                        acts[name][B:], pool_spatial,
                    )

                    # Update CKA + get per-batch HSIC
                    hsic_vals = cka_engines[ts_val][name].update_and_return_batch_hsic(
                        enc_flat, ref_flat,
                    )
                    for key in ("hsic_xy", "hsic_xx", "hsic_yy"):
                        per_batch_hsic[ts_val][name][key].append(hsic_vals[key])

            hook.clear()

            # DDIM step using guided noise
            x_t = scheduler.step(noise_pred, t, x_t, generator=generator).prev_sample

        generated += B

    elapsed = time.time() - t0

    # ── 9. Compute final CKA scores ─────────────────────────────────────────
    print(f"\n  Computing final CKA scores …\n")
    cka_scores: Dict[int, Dict[str, float]] = {}

    for ts_val in sorted(cka_engines.keys(), reverse=True):
        cka_scores[ts_val] = {}
        row = []
        for name in encoder_layers:
            score = cka_engines[ts_val][name].compute()
            cka_scores[ts_val][name] = score
            row.append(f"{score:.4f}")
        print(f"  t={ts_val:4d}  │  {'  '.join(row)}")

    # ── 10. Save results ────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "cka_scores": cka_scores,
        "per_batch_hsic": per_batch_hsic,
        "ref_layer": ref_layer,
        "encoder_layers": encoder_layers,
        "config": {
            "model_id": model_id,
            "vector_path": vector_path,
            "v_scale": v_scale,
            "cfg_scale": cfg_scale,
            "num_samples": generated,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "bucket_every": bucket_every,
            "pool_spatial": pool_spatial,
            "seed": seed,
            "elapsed_s": elapsed,
        },
    }

    if vector_path:
        vec_name = Path(vector_path).stem
        prefix = f"cka_guided_{vec_name}_v{v_scale}_cfg{cfg_scale}"
    else:
        prefix = "cka_guided_baseline"

    results_path = out_dir / f"{prefix}_results.pt"
    torch.save(results, results_path)
    print(f"\n  Results     → {results_path}")

    # ── 11. Visualise ───────────────────────────────────────────────────────
    # Heatmap
    heatmap_path = out_dir / f"{prefix}_heatmap.png"
    plot_cka_heatmap(
        cka_scores,
        ref_layer=ref_layer,
        pool_spatial=pool_spatial,
        save_path=str(heatmap_path),
    )
    print(f"  Heatmap     → {heatmap_path}")

    # Trajectory plot
    trajectory_path = out_dir / f"{prefix}_trajectory.png"
    plot_cka_trajectory(
        cka_scores,
        ref_layer=ref_layer,
        pool_spatial=pool_spatial,
        save_path=str(trajectory_path),
    )
    print(f"  Trajectory  → {trajectory_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    hook.remove()
    if patch_hook is not None:
        patch_hook.remove()
    del unet
    torch.cuda.empty_cache()

    print(f"\n  Total samples : {generated}")
    print(f"  Total time    : {elapsed:.1f}s")
    print(f"  Done. ✓\n")
    return cka_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2: dataset-free generative CKA with Guided CFG patching",
    )
    p.add_argument("--means_path", type=str,
                   default="destructive_interference/outputs/global_means.pt",
                   help="Path to Phase 1 global_means.pt")
    p.add_argument("--vector_path", type=str, default="",
                   help="Path to h-space direction vector (.pt file). "
                        "Leave empty for unpatched baseline CKA.")
    p.add_argument("--v_scale", type=float, default=1.0,
                   help="Scalar multiplier for the direction vector.")
    p.add_argument("--cfg_scale", type=float, default=3.0,
                   help="Classifier-Free Guidance scale.")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--num_samples", type=int, default=256,
                   help="Number of samples to generate from noise.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=30,
                   help="Total DDIM inference steps.")
    p.add_argument("--bucket_every", type=int, default=3,
                   help="Compute CKA every Nth step.")
    p.add_argument("--pool_spatial", type=int, default=1,
                   help="Spatial pool size (1=global avg, 0=none).")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--use_batch_mean", action="store_true",
                   help="Centre each mini-batch by its own mean (standard CKA) "
                        "instead of using pre-computed global means. "
                        "Eliminates domain mismatch between Phase 1 and Phase 2.")
    p.add_argument("--output_dir", type=str,
                   default="destructive_interference/outputs")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    compute_cka_generative_guided(
        means_path=args.means_path,
        vector_path=args.vector_path,
        v_scale=args.v_scale,
        cfg_scale=args.cfg_scale,
        model_id=args.model_id,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        bucket_every=args.bucket_every,
        pool_spatial=args.pool_spatial if args.pool_spatial > 0 else None,
        device_str=args.device,
        seed=args.seed,
        use_amp=not args.no_amp,
        output_dir=args.output_dir,
        use_batch_mean=args.use_batch_mean,
    )
