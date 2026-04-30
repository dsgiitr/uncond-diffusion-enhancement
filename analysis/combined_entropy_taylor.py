"""
Attention Entropy + Taylor Residual Decomposition Experiment
=============================================================

Two **independent** analyses, selectable via CLI flags:

  1. ``--run_entropy``  — Attention Entropy analysis.
     Measures how the self-attention distribution changes when the
     mid-block activation is patched with α·v along the full DDIM
     trajectory.  (Forward-pass only → can use large batch sizes.)

  2. ``--run_taylor``   — Taylor Residual Decomposition.
     Decomposes the patched noise prediction into 0th-order (ε₀),
     1st-order (J·v), and higher-order residual (R≥2) contributions,
     using a *paired* trajectory view.  (Uses forward-mode AD →
     needs a smaller batch size.)

Both analyses can be run together or separately.

Usage
-----
    # Entropy only (fast, large batches)
    python combined_entropy_taylor.py --run_entropy \\
        --batch_size 128 --num_samples 512

    # Taylor only (needs JVP, smaller batches)
    python combined_entropy_taylor.py --run_taylor \\
        --batch_size 32 --num_samples 512

    # Both together
    python combined_entropy_taylor.py --run_entropy --run_taylor \\
        --batch_size 128 --taylor_batch_size 32 --num_samples 512
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from diffusers import DDPMPipeline
from tqdm.auto import tqdm

# ═════════════════════════════════════════════════════════════════════════
# Project paths
# ═════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
GENERATION_DIR = PROJECT_ROOT / "generation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from generation.pipeline import build_scheduler  # noqa: E402

# Force math-mode SDP so forward-AD tangents propagate through attention
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ═════════════════════════════════════════════════════════════════════════
# Fixed constants
# ═════════════════════════════════════════════════════════════════════════
MODEL_ID = "google/ddpm-celebahq-256"
SCHEDULER_NAME = "ddim"
NUM_STEPS = 30
TARGET_LAYER_NAME = "mid_block"

# ═════════════════════════════════════════════════════════════════════════
# Plotting style
# ═════════════════════════════════════════════════════════════════════════
COLOR_BG = "#F1FAEE"
COLOR_GRID = "#D4D4D4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ═════════════════════════════════════════════════════════════════════════
# 1. Custom Attention Processor — captures entropy
# ═════════════════════════════════════════════════════════════════════════
class EntropyAttnProcessor2_0:
    """Drop-in attention processor that captures per-head entropy.

    After each forward call the *batch-mean* entropy is appended to
    ``self.entropies``.  We compute attention weights explicitly (no
    Flash-SDP) so we can measure the distribution.
    """

    def __init__(self):
        self.entropies: list[torch.Tensor] = []

    def reset(self):
        self.entropies.clear()

    def get_last_entropy(self) -> torch.Tensor | None:
        return self.entropies[-1] if self.entropies else None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Explicit scaled dot-product (no Flash-SDP) so we can capture probs
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * (head_dim ** -0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Shannon entropy  H = -Σ p log p   (averaged over heads & tokens)
        epsilon = 1e-12
        entropy = -(attention_probs * torch.log(attention_probs + epsilon)).sum(dim=-1)
        # [B, heads, seq] → mean over seq & heads → [B]
        batch_entropy = entropy.mean(dim=-1).mean(dim=-1)
        self.entropies.append(batch_entropy.detach().cpu())

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ═════════════════════════════════════════════════════════════════════════
# 2. Core helper functions
# ═════════════════════════════════════════════════════════════════════════
def predict_noise_with_optional_patch(
    unet: nn.Module,
    target_layer: nn.Module,
    latent_input: torch.Tensor,
    t: torch.Tensor,
    delta_h: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run UNet; if *alpha* is given, add ``alpha * delta_h`` at mid-block."""
    handle = None
    if delta_h is not None and alpha is not None:
        def _hook(_module, _inputs, output):
            return output + alpha * delta_h
        handle = target_layer.register_forward_hook(_hook)
    try:
        return unet(latent_input, t).sample
    finally:
        if handle is not None:
            handle.remove()


def compute_jvp_at_zero(
    unet: nn.Module,
    target_layer: nn.Module,
    latent_input: torch.Tensor,
    t: torch.Tensor,
    delta_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (ε₀,  J_D(h_t)·v) via forward-mode AD at α=0.

    Returns
    -------
    eps0 : D(h_t)       — unpatched noise prediction
    jvp  : J_D(h_t)·v   — Jacobian-vector product (no α factor)
    """
    alpha0 = torch.zeros((), device=latent_input.device, dtype=latent_input.dtype)
    tangent = torch.ones_like(alpha0)

    with fwAD.dual_level():
        dual_alpha = fwAD.make_dual(alpha0, tangent)
        out = predict_noise_with_optional_patch(
            unet=unet,
            target_layer=target_layer,
            latent_input=latent_input,
            t=t,
            delta_h=delta_h,
            alpha=dual_alpha,
        )
        primal_out = fwAD.unpack_dual(out).primal
        tangent_out = fwAD.unpack_dual(out).tangent
        if tangent_out is None:
            tangent_out = torch.zeros_like(primal_out)

    return primal_out.detach(), tangent_out.detach()


def batch_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm → Tensor of shape [B]."""
    return x.view(x.shape[0], -1).norm(dim=1)


def _generate_x_T(unet, B, seed, batch_start, device, dtype):
    """Generate a batch of deterministic initial noise vectors."""
    x_T_list = []
    for i in range(B):
        gen = torch.Generator(device=device).manual_seed(seed + batch_start + i)
        x_i = torch.randn(
            (1, unet.config.in_channels, unet.config.sample_size,
             unet.config.sample_size),
            generator=gen, device=device, dtype=dtype,
        )
        x_T_list.append(x_i)
    return torch.cat(x_T_list, dim=0)


# ═════════════════════════════════════════════════════════════════════════
# 3. CLI
# ═════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attention Entropy & Taylor Residual Experiment "
                    "(independent modes)"
    )
    # ── Mode selection ──────────────────────────────────────────────────
    p.add_argument(
        "--run_entropy", action="store_true",
        help="Run the attention-entropy analysis.",
    )
    p.add_argument(
        "--run_taylor", action="store_true",
        help="Run the Taylor residual decomposition analysis.",
    )

    # ── V-scale sweeps ──────────────────────────────────────────────────
    p.add_argument(
        "--pos_vscales", nargs="+", type=float,
        default=[3.0, 5.0, 7.0, 10.0],
        help="Positive v-scale values (sharp / +v direction)",
    )
    p.add_argument(
        "--neg_vscales", nargs="+", type=float,
        default=[-0.5, -1.0, -2.0],
        help="Negative v-scale values (blur / -v direction)",
    )

    # ── Sampling ────────────────────────────────────────────────────────
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for entropy (forward-pass only) loops. "
                        "Also used as outer batch for Taylor if taylor_batch_size "
                        "is not set separately.")
    p.add_argument("--taylor_batch_size", type=int, default=None,
                   help="Batch size for JVP / Taylor residual computation "
                        "(forward-AD, needs less memory). "
                        "Defaults to --batch_size if not specified.")
    p.add_argument("--seed", type=int, default=42)

    # ── Paths ───────────────────────────────────────────────────────────
    p.add_argument(
        "--v_path", type=str,
        default=str(PROJECT_ROOT / "vectors" / "sharp_vs_blur_dom_t20.pt"),
    )
    p.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "combined_entropy_taylor_outputs"),
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════
# 4. Plotting helpers (shared)
# ═════════════════════════════════════════════════════════════════════════
def _make_color_maps(pos_scales, neg_scales):
    cmap_pos = plt.cm.Reds(np.linspace(0.3, 0.95, max(len(pos_scales), 1)))
    cmap_neg = plt.cm.Blues(np.linspace(0.3, 0.95, max(len(neg_scales), 1)))

    def color_for(vs):
        if vs > 0:
            idx = pos_scales.index(vs) if vs in pos_scales else 0
            return cmap_pos[idx]
        else:
            idx = neg_scales.index(vs) if vs in neg_scales else 0
            return cmap_neg[idx]

    return cmap_pos, cmap_neg, color_for


# ═════════════════════════════════════════════════════════════════════════
# 5. ENTROPY ANALYSIS  (--run_entropy)
# ═════════════════════════════════════════════════════════════════════════
def run_entropy_analysis(
    unet, scheduler_template, target_layer, v_raw, dtype,
    all_vscales, all_timesteps, args, OUTPUT_DIR, DEVICE,
):
    """Run attention-entropy experiment independently.

    Only uses forward passes (torch.no_grad), no AD — can use large batches.
    """
    print("\n" + "=" * 70)
    print("MODE: Attention Entropy Analysis")
    print("=" * 70)

    # Install entropy processor
    attn_module = unet.up_blocks[1].attentions[0]
    entropy_processor = EntropyAttnProcessor2_0()
    attn_module.set_processor(entropy_processor)

    N = args.num_samples
    BS = args.batch_size

    # Accumulators
    unpatched_entropy_acc = []  # list of [B_i, 30]

    vscale_entropy_acc: dict[float, list] = {vs: [] for vs in all_vscales}

    t_start = time.time()
    num_batches = (N + BS - 1) // BS

    for batch_idx, batch_start in enumerate(range(0, N, BS)):
        batch_end = min(batch_start + BS, N)
        B = batch_end - batch_start
        print(f"\n{'━' * 60}")
        print(f"[Entropy] Batch {batch_idx + 1}/{num_batches}  "
              f"(samples {batch_start}–{batch_end - 1})")
        print(f"{'━' * 60}")

        x_T = _generate_x_T(unet, B, args.seed, batch_start, DEVICE, dtype)

        # ── Baseline (unpatched) trajectory ─────────────────────────────
        sched_b = copy.deepcopy(scheduler_template)
        sched_b.set_timesteps(NUM_STEPS, device=DEVICE)

        sample_b = x_T.clone() * sched_b.init_noise_sigma
        gen_b = torch.Generator(device=DEVICE).manual_seed(args.seed + batch_start)
        batch_unpatched_entropy = []

        for step_idx, t in enumerate(tqdm(
            sched_b.timesteps, desc="  Baseline trajectory", leave=False
        )):
            latent_input = sched_b.scale_model_input(sample_b, t)
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t

            entropy_processor.reset()
            with torch.no_grad():
                noise_pred = unet(latent_input, t_tensor).sample

            batch_unpatched_entropy.append(entropy_processor.get_last_entropy())
            sample_b = sched_b.step(
                noise_pred, t, sample_b, generator=gen_b
            ).prev_sample

        unpatched_entropy_acc.append(
            torch.stack(batch_unpatched_entropy, dim=0).T  # [B, 30]
        )
        del sample_b

        # ── Per v-scale: patched trajectory → entropy ───────────────────
        for vs_idx, vs in enumerate(all_vscales):
            delta_h = vs * v_raw
            print(f"  v-scale {vs:+.1f}  ({vs_idx + 1}/{len(all_vscales)})")

            sched_p = copy.deepcopy(scheduler_template)
            sched_p.set_timesteps(NUM_STEPS, device=DEVICE)

            sample_p = x_T.clone() * sched_p.init_noise_sigma
            gen_p = torch.Generator(device=DEVICE).manual_seed(
                args.seed + batch_start
            )
            alpha_one = torch.ones((), device=DEVICE, dtype=dtype)
            patched_entropy_steps = []

            for step_idx, t in enumerate(tqdm(
                sched_p.timesteps,
                desc=f"    Patched traj v={vs:+.1f}",
                leave=False,
            )):
                latent_p = sched_p.scale_model_input(sample_p, t)
                t_tensor = t.unsqueeze(0) if t.dim() == 0 else t

                entropy_processor.reset()
                with torch.no_grad():
                    eps_p = predict_noise_with_optional_patch(
                        unet, target_layer, latent_p, t_tensor,
                        delta_h=delta_h, alpha=alpha_one,
                    )

                patched_entropy_steps.append(entropy_processor.get_last_entropy())
                sample_p = sched_p.step(
                    eps_p, t, sample_p, generator=gen_p
                ).prev_sample

            vscale_entropy_acc[vs].append(
                torch.stack(patched_entropy_steps, dim=0).T  # [B, 30]
            )
            del sample_p
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n✓ Entropy analysis done in {elapsed:.1f}s")

    # ── Stack & save ────────────────────────────────────────────────────
    unpatched_all = torch.cat(unpatched_entropy_acc, dim=0)  # [N, 30]

    per_vscale_entropy: dict[float, torch.Tensor] = {}
    for vs in all_vscales:
        per_vscale_entropy[vs] = torch.cat(vscale_entropy_acc[vs], dim=0)

    entropy_dir = OUTPUT_DIR / "entropy"
    entropy_dir.mkdir(parents=True, exist_ok=True)

    raw_entropy = {
        "config": {
            "all_vscales": all_vscales,
            "num_samples": N, "num_steps": NUM_STEPS,
            "seed": args.seed, "v_path": str(args.v_path),
            "model_id": MODEL_ID, "scheduler": SCHEDULER_NAME,
        },
        "timesteps": all_timesteps,
        "unpatched_entropy": unpatched_all,
        "per_vscale_entropy": per_vscale_entropy,
    }
    torch.save(raw_entropy, entropy_dir / "raw_entropy.pt")
    print(f"Saved → {entropy_dir / 'raw_entropy.pt'}")

    # ── Aggregate ───────────────────────────────────────────────────────
    agg_ent = {
        "timesteps": all_timesteps,
        "unpatched_entropy_mean": unpatched_all.mean(dim=0),
        "unpatched_entropy_std": unpatched_all.std(dim=0),
        "per_vscale": {},
    }
    for vs in all_vscales:
        d = per_vscale_entropy[vs]
        agg_ent["per_vscale"][vs] = {
            "entropy_mean": d.mean(dim=0),
            "entropy_std": d.std(dim=0),
        }
    torch.save(agg_ent, entropy_dir / "aggregated_entropy.pt")

    # ── Entropy Plots ───────────────────────────────────────────────────
    ts_arr = np.array(all_timesteps)
    pos_scales = sorted([v for v in all_vscales if v > 0])
    neg_scales = sorted([v for v in all_vscales if v < 0], key=lambda x: abs(x))
    cmap_pos, cmap_neg, color_for = _make_color_maps(pos_scales, neg_scales)

    # PLOT 1: Entropy vs Timestep
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_facecolor(COLOR_BG); fig1.patch.set_facecolor("white")

    u_mean = agg_ent["unpatched_entropy_mean"].numpy()
    u_std = agg_ent["unpatched_entropy_std"].numpy()
    ax1.errorbar(ts_arr, u_mean, yerr=u_std,
                 label="Unpatched", marker="^", color="gray",
                 linestyle="--", linewidth=1.8, capsize=3,
                 markeredgecolor="white", markeredgewidth=0.6)
    for vs in pos_scales:
        m = agg_ent["per_vscale"][vs]["entropy_mean"].numpy()
        s = agg_ent["per_vscale"][vs]["entropy_std"].numpy()
        ax1.errorbar(ts_arr, m, yerr=s,
                     label=f"+v  α={vs}", marker="o", markersize=5,
                     color=color_for(vs), linewidth=1.5, capsize=3,
                     markeredgecolor="white", markeredgewidth=0.5)
    for vs in neg_scales:
        m = agg_ent["per_vscale"][vs]["entropy_mean"].numpy()
        s = agg_ent["per_vscale"][vs]["entropy_std"].numpy()
        ax1.errorbar(ts_arr, m, yerr=s,
                     label=f"−v  α={abs(vs)}", marker="s", markersize=5,
                     color=color_for(vs), linewidth=1.5, capsize=3,
                     markeredgecolor="white", markeredgewidth=0.5)
    ax1.invert_xaxis()
    ax1.set_xlabel("Timestep $t$")
    ax1.set_ylabel("Attention Entropy")
    ax1.set_title("Attention Entropy vs Timestep (all v-scales)")
    ax1.legend(fontsize=8, ncol=3, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.4, color=COLOR_GRID)
    plt.tight_layout()
    plt.savefig(entropy_dir / "plot_entropy_vs_timestep.png")
    plt.savefig(entropy_dir / "plot_entropy_vs_timestep.pdf")
    plt.close()

    # PLOT 2: Entropy vs |v-scale| at representative timesteps
    rep_indices = [0, 4, 9, 14, 19, 24, 29]
    rep_ts = [all_timesteps[i] for i in rep_indices if i < len(all_timesteps)]
    n_rep = len(rep_ts)
    ncols = min(4, n_rep)
    nrows = (n_rep + ncols - 1) // ncols

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                               sharey=True, squeeze=False)
    fig2.suptitle("Attention Entropy vs |v-scale| at Representative Timesteps",
                  fontsize=14, fontweight="bold")
    for ax_idx, ts_val in enumerate(rep_ts):
        ax = axes2[ax_idx // ncols][ax_idx % ncols]
        ax.set_facecolor(COLOR_BG)
        step_i = all_timesteps.index(ts_val)
        u_val = agg_ent["unpatched_entropy_mean"][step_i].item()
        ax.axhline(u_val, color="gray", linestyle="--", linewidth=1.2,
                    label="Unpatched")
        if pos_scales:
            pos_abs = [abs(v) for v in pos_scales]
            pos_ent = [agg_ent["per_vscale"][v]["entropy_mean"][step_i].item()
                       for v in pos_scales]
            ax.plot(pos_abs, pos_ent, marker="o", color="#E63946",
                    linewidth=1.5, label="Sharp (+v)")
        if neg_scales:
            neg_abs = [abs(v) for v in neg_scales]
            neg_ent = [agg_ent["per_vscale"][v]["entropy_mean"][step_i].item()
                       for v in neg_scales]
            ax.plot(neg_abs, neg_ent, marker="s", color="#457B9D",
                    linewidth=1.5, label="Blur (−v)")
        ax.set_xlabel("|v-scale|")
        ax.set_title(f"t = {ts_val}")
        if ax_idx % ncols == 0:
            ax.set_ylabel("Entropy")
        if ax_idx == 0:
            ax.legend(fontsize=8, frameon=True)
        ax.grid(True, alpha=0.4, color=COLOR_GRID)
    for ax_idx in range(len(rep_ts), nrows * ncols):
        axes2[ax_idx // ncols][ax_idx % ncols].set_visible(False)
    plt.tight_layout()
    plt.savefig(entropy_dir / "plot_entropy_vs_vscale.png")
    plt.savefig(entropy_dir / "plot_entropy_vs_vscale.pdf")
    plt.close()

    print(f"✓ Entropy plots saved to {entropy_dir}/")


# ═════════════════════════════════════════════════════════════════════════
# 6. TAYLOR RESIDUAL ANALYSIS  (--run_taylor)
# ═════════════════════════════════════════════════════════════════════════
def run_taylor_analysis(
    unet, scheduler_template, target_layer, v_raw, dtype,
    all_vscales, all_timesteps, args, OUTPUT_DIR, DEVICE,
):
    """Run Taylor residual decomposition independently.

    Uses forward-mode AD (JVP) → needs smaller batch sizes.
    The outer loop batches at taylor_batch_size.
    """
    print("\n" + "=" * 70)
    print("MODE: Taylor Residual Decomposition")
    print("=" * 70)

    taylor_bs = (args.taylor_batch_size
                 if args.taylor_batch_size is not None
                 else args.batch_size)

    N = args.num_samples

    # Accumulators — per vscale
    vscale_acc: dict[float, dict[str, list]] = {}
    for vs in all_vscales:
        vscale_acc[vs] = {
            "jvp_norms": [],         # list of [B, 30]
            "delta_norms": [],
            "residual_norms": [],
            "eps0_norms": [],
        }

    t_start = time.time()
    num_batches = (N + taylor_bs - 1) // taylor_bs

    for batch_idx, batch_start in enumerate(range(0, N, taylor_bs)):
        batch_end = min(batch_start + taylor_bs, N)
        B = batch_end - batch_start
        print(f"\n{'━' * 60}")
        print(f"[Taylor] Batch {batch_idx + 1}/{num_batches}  "
              f"(samples {batch_start}–{batch_end - 1})")
        print(f"{'━' * 60}")

        x_T = _generate_x_T(unet, B, args.seed, batch_start, DEVICE, dtype)

        # ── Baseline trajectory: cache (latent, t) at all 30 steps ──────
        sched_b = copy.deepcopy(scheduler_template)
        sched_b.set_timesteps(NUM_STEPS, device=DEVICE)

        sample_b = x_T.clone() * sched_b.init_noise_sigma
        gen_b = torch.Generator(device=DEVICE).manual_seed(args.seed + batch_start)

        cached_baseline: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for step_idx, t in enumerate(tqdm(
            sched_b.timesteps, desc="  Baseline trajectory", leave=False
        )):
            latent_input = sched_b.scale_model_input(sample_b, t)
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            cached_baseline[step_idx] = (latent_input.clone(), t_tensor.clone())

            with torch.no_grad():
                noise_pred = unet(latent_input, t_tensor).sample
            sample_b = sched_b.step(
                noise_pred, t, sample_b, generator=gen_b
            ).prev_sample

        del sample_b

        # ── Per v-scale: patched trajectory + JVP decomposition ─────────
        for vs_idx, vs in enumerate(all_vscales):
            delta_h = vs * v_raw
            print(f"  v-scale {vs:+.1f}  ({vs_idx + 1}/{len(all_vscales)})")

            # 1) Run patched trajectory to get eps_patched at each step
            sched_p = copy.deepcopy(scheduler_template)
            sched_p.set_timesteps(NUM_STEPS, device=DEVICE)

            sample_p = x_T.clone() * sched_p.init_noise_sigma
            gen_p = torch.Generator(device=DEVICE).manual_seed(
                args.seed + batch_start
            )
            alpha_one = torch.ones((), device=DEVICE, dtype=dtype)
            eps_patched_at_step: dict[int, torch.Tensor] = {}

            for step_idx, t in enumerate(tqdm(
                sched_p.timesteps,
                desc=f"    Patched traj v={vs:+.1f}",
                leave=False,
            )):
                latent_p = sched_p.scale_model_input(sample_p, t)
                t_tensor = t.unsqueeze(0) if t.dim() == 0 else t

                with torch.no_grad():
                    eps_p = predict_noise_with_optional_patch(
                        unet, target_layer, latent_p, t_tensor,
                        delta_h=delta_h, alpha=alpha_one,
                    )
                eps_patched_at_step[step_idx] = eps_p.detach()
                sample_p = sched_p.step(
                    eps_p, t, sample_p, generator=gen_p
                ).prev_sample

            del sample_p
            torch.cuda.empty_cache()

            # 2) JVP decomposition at each cached baseline state
            jvp_norms_steps = []
            delta_norms_steps = []
            residual_norms_steps = []
            eps0_norms_steps = []

            for step_idx in range(NUM_STEPS):
                latent_b, t_tensor = cached_baseline[step_idx]
                eps_p = eps_patched_at_step[step_idx]

                eps0, jvp_val = compute_jvp_at_zero(
                    unet, target_layer, latent_b, t_tensor, delta_h,
                )

                delta = eps_p - eps0
                residual = delta - jvp_val

                jvp_norms_steps.append(batch_l2_norm(jvp_val).cpu())
                delta_norms_steps.append(batch_l2_norm(delta).cpu())
                residual_norms_steps.append(batch_l2_norm(residual).cpu())
                eps0_norms_steps.append(batch_l2_norm(eps0).cpu())

                del eps0, jvp_val, delta, residual
                torch.cuda.empty_cache()

            # Stack: [30, B] → [B, 30]
            vscale_acc[vs]["jvp_norms"].append(
                torch.stack(jvp_norms_steps, dim=0).T)
            vscale_acc[vs]["delta_norms"].append(
                torch.stack(delta_norms_steps, dim=0).T)
            vscale_acc[vs]["residual_norms"].append(
                torch.stack(residual_norms_steps, dim=0).T)
            vscale_acc[vs]["eps0_norms"].append(
                torch.stack(eps0_norms_steps, dim=0).T)

            del eps_patched_at_step
            torch.cuda.empty_cache()

        del cached_baseline
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n✓ Taylor analysis done in {elapsed:.1f}s")

    # ── Stack & save ────────────────────────────────────────────────────
    per_vscale_raw: dict[float, dict[str, torch.Tensor]] = {}
    for vs in all_vscales:
        per_vscale_raw[vs] = {
            k: torch.cat(v, dim=0) for k, v in vscale_acc[vs].items()
        }

    taylor_dir = OUTPUT_DIR / "taylor"
    taylor_dir.mkdir(parents=True, exist_ok=True)

    raw_data = {
        "config": {
            "all_vscales": all_vscales,
            "num_samples": N, "num_steps": NUM_STEPS,
            "seed": args.seed, "v_path": str(args.v_path),
            "model_id": MODEL_ID, "scheduler": SCHEDULER_NAME,
        },
        "timesteps": all_timesteps,
        "per_vscale": per_vscale_raw,
    }
    torch.save(raw_data, taylor_dir / "raw_taylor.pt")
    print(f"Saved → {taylor_dir / 'raw_taylor.pt'}")

    # ── Aggregate ───────────────────────────────────────────────────────
    agg: dict = {"timesteps": all_timesteps, "per_vscale": {}}

    for vs in all_vscales:
        d = per_vscale_raw[vs]
        delta_norms = d["delta_norms"]
        res_norms = d["residual_norms"]
        ratio_per_sample = res_norms / (delta_norms + 1e-12)

        agg["per_vscale"][vs] = {
            "jvp_norm_mean": d["jvp_norms"].mean(dim=0),
            "jvp_norm_std": d["jvp_norms"].std(dim=0),
            "delta_norm_mean": delta_norms.mean(dim=0),
            "delta_norm_std": delta_norms.std(dim=0),
            "residual_norm_mean": res_norms.mean(dim=0),
            "residual_norm_std": res_norms.std(dim=0),
            "residual_ratio_mean": ratio_per_sample.mean(dim=0),
            "residual_ratio_std": ratio_per_sample.std(dim=0),
            "eps0_norm_mean": d["eps0_norms"].mean(dim=0),
            "eps0_norm_std": d["eps0_norms"].std(dim=0),
        }

    torch.save(agg, taylor_dir / "aggregated_taylor.pt")

    # ── Taylor Plots ────────────────────────────────────────────────────
    ts_arr = np.array(all_timesteps)
    pos_scales = sorted([v for v in all_vscales if v > 0])
    neg_scales = sorted([v for v in all_vscales if v < 0], key=lambda x: abs(x))
    cmap_pos, cmap_neg, color_for = _make_color_maps(pos_scales, neg_scales)

    # PLOT 1: Jacobian magnitude vs timestep
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_facecolor(COLOR_BG); fig3.patch.set_facecolor("white")
    for vs in pos_scales:
        m = agg["per_vscale"][vs]["jvp_norm_mean"].numpy()
        s = agg["per_vscale"][vs]["jvp_norm_std"].numpy()
        ax3.errorbar(ts_arr, m, yerr=s, label=f"+v  α={vs}", marker="o",
                     markersize=5, color=color_for(vs), linewidth=1.5,
                     capsize=3, markeredgecolor="white", markeredgewidth=0.5)
    for vs in neg_scales:
        m = agg["per_vscale"][vs]["jvp_norm_mean"].numpy()
        s = agg["per_vscale"][vs]["jvp_norm_std"].numpy()
        ax3.errorbar(ts_arr, m, yerr=s, label=f"−v  α={abs(vs)}", marker="s",
                     markersize=5, color=color_for(vs), linewidth=1.5,
                     capsize=3, markeredgecolor="white", markeredgewidth=0.5)
    ax3.invert_xaxis()
    ax3.set_xlabel("Timestep $t$")
    ax3.set_ylabel("$\\|J_D(h_t)\\cdot v\\|_2$")
    ax3.set_title("Jacobian Magnitude — How decodable is each direction?")
    ax3.legend(fontsize=8, ncol=3, frameon=True, fancybox=True)
    ax3.grid(True, alpha=0.4, color=COLOR_GRID)
    plt.tight_layout()
    plt.savefig(taylor_dir / "plot_jacobian_magnitude.png")
    plt.savefig(taylor_dir / "plot_jacobian_magnitude.pdf")
    plt.close()

    # PLOT 2: Residual norm vs timestep
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig4.suptitle("Taylor Residual Norms $\\|R_{\\geq 2}\\|$ vs Timestep",
                  fontsize=15, fontweight="bold")
    for ax, scales, title, cmap_arr in [
        (axes4[0], pos_scales, "Sharp (+v)", cmap_pos),
        (axes4[1], neg_scales, "Blur (−v)", cmap_neg),
    ]:
        ax.set_facecolor(COLOR_BG)
        for i, vs in enumerate(scales):
            m = agg["per_vscale"][vs]["residual_norm_mean"].numpy()
            s = agg["per_vscale"][vs]["residual_norm_std"].numpy()
            ax.errorbar(ts_arr, m, yerr=s,
                        label=f"α={abs(vs)}", marker="o", markersize=5,
                        color=cmap_arr[i], linewidth=1.5, capsize=3)
        ax.invert_xaxis()
        ax.set_xlabel("Timestep $t$")
        ax.set_ylabel("$\\|R_{\\geq 2}\\|_2$")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, ncol=2, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.4, color=COLOR_GRID)
    plt.tight_layout()
    plt.savefig(taylor_dir / "plot_residual_norm.png")
    plt.savefig(taylor_dir / "plot_residual_norm.pdf")
    plt.close()

    # PLOT 3: Residual ratio ||R|| / ||Δ|| vs timestep
    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig5.suptitle(
        "Relative Residual $\\|R_{\\geq 2}\\| / \\|\\Delta\\|$ vs Timestep "
        "(Linearization Failure)",
        fontsize=14, fontweight="bold",
    )
    for ax, scales, title, cmap_arr in [
        (axes5[0], pos_scales, "Sharp (+v)", cmap_pos),
        (axes5[1], neg_scales, "Blur (−v)", cmap_neg),
    ]:
        ax.set_facecolor(COLOR_BG)
        for i, vs in enumerate(scales):
            m = agg["per_vscale"][vs]["residual_ratio_mean"].numpy()
            s = agg["per_vscale"][vs]["residual_ratio_std"].numpy()
            ax.errorbar(ts_arr, m, yerr=s,
                        label=f"α={abs(vs)}", marker="o", markersize=5,
                        color=cmap_arr[i], linewidth=1.5, capsize=3)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="$R=1$ (failure)")
        ax.invert_xaxis()
        ax.set_xlabel("Timestep $t$")
        ax.set_ylabel("$\\|R\\| / \\|\\Delta\\|$")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, ncol=2, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.4, color=COLOR_GRID)
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(taylor_dir / "plot_residual_ratio.png")
    plt.savefig(taylor_dir / "plot_residual_ratio.pdf")
    plt.close()

    # PLOT 4: ||R|| / vscale² vs timestep
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig6.suptitle(
        "$\\|R_{\\geq 2}\\| / \\alpha^2$ vs Timestep — Is residual purely quadratic?",
        fontsize=14, fontweight="bold",
    )
    for ax, scales, title, cmap_arr in [
        (axes6[0], pos_scales, "Sharp (+v)", cmap_pos),
        (axes6[1], neg_scales, "Blur (−v)", cmap_neg),
    ]:
        ax.set_facecolor(COLOR_BG)
        for i, vs in enumerate(scales):
            a2 = vs ** 2
            m = agg["per_vscale"][vs]["residual_norm_mean"].numpy() / a2
            s = agg["per_vscale"][vs]["residual_norm_std"].numpy() / a2
            ax.errorbar(ts_arr, m, yerr=s,
                        label=f"α={abs(vs)}", marker="o", markersize=5,
                        color=cmap_arr[i], linewidth=1.5, capsize=3)
        ax.invert_xaxis()
        ax.set_xlabel("Timestep $t$")
        ax.set_ylabel("$\\|R\\| / \\alpha^2$")
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, ncol=2, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.4, color=COLOR_GRID)
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(taylor_dir / "plot_residual_over_alpha2.png")
    plt.savefig(taylor_dir / "plot_residual_over_alpha2.pdf")
    plt.close()

    print(f"✓ Taylor plots saved to {taylor_dir}/")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Residual Ratio ||R|| / ||Δ|| — mean across samples")
    print("=" * 80)
    header = f"{'v-scale':>8} |"
    for ts in all_timesteps[::5]:
        header += f"  t={ts:<5}"
    print(header)
    print("-" * len(header))

    for vs in all_vscales:
        row = f"{vs:>+8.1f} |"
        ratio_m = agg["per_vscale"][vs]["residual_ratio_mean"]
        for step_i in range(0, NUM_STEPS, 5):
            R = ratio_m[step_i].item()
            if R > 1.0:
                row += f"  {R:>5.2f}*"
            else:
                row += f"  {R:>5.3f}"
        print(row)

    print("\n* = R > 1 (linearization failure)")


# ═════════════════════════════════════════════════════════════════════════
# 7. Main
# ═════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    if not args.run_entropy and not args.run_taylor:
        print("ERROR: Specify at least one of --run_entropy or --run_taylor.")
        print("  Example: python combined_entropy_taylor.py --run_entropy --run_taylor")
        sys.exit(1)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    all_vscales: list[float] = sorted(
        set(args.pos_vscales + args.neg_vscales), key=lambda x: (x >= 0, abs(x))
    )
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    taylor_bs = (args.taylor_batch_size
                 if args.taylor_batch_size is not None
                 else args.batch_size)

    modes = []
    if args.run_entropy:
        modes.append("Entropy")
    if args.run_taylor:
        modes.append("Taylor")

    print("=" * 70)
    print(f"Experiment — Modes: {', '.join(modes)}")
    print("=" * 70)
    print(f"  Model   : {MODEL_ID}")
    print(f"  Sched   : {SCHEDULER_NAME} | Steps: {NUM_STEPS}")
    print(f"  V-path  : {args.v_path}")
    print(f"  +v scales: {args.pos_vscales}")
    print(f"  -v scales: {args.neg_vscales}")
    print(f"  Samples : {args.num_samples}")
    if args.run_entropy:
        print(f"  Batch (entropy) : {args.batch_size}")
    if args.run_taylor:
        print(f"  Batch (Taylor)  : {taylor_bs}")
    print(f"  Device  : {DEVICE}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 70)

    # ── Load model (shared) ─────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    pipe = DDPMPipeline.from_pretrained(MODEL_ID).to(DEVICE)
    unet = pipe.unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    dtype = next(unet.parameters()).dtype
    scheduler_template = build_scheduler(SCHEDULER_NAME, pipe)
    target_layer = getattr(unet, TARGET_LAYER_NAME)

    # ── Load direction vector (shared) ──────────────────────────────────
    v_path = Path(args.v_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Vector not found: {v_path}")
    v_raw = torch.load(v_path, map_location=DEVICE, weights_only=False)
    if not isinstance(v_raw, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(v_raw)}")
    if v_raw.dim() == 3:
        v_raw = v_raw.unsqueeze(0)
    v_raw = v_raw.to(device=DEVICE, dtype=dtype)
    print(f"  Vector shape: {tuple(v_raw.shape)}, L2: {v_raw.flatten().norm():.4f}")

    # ── Identify all 30 timestep values ─────────────────────────────────
    sched_ref = copy.deepcopy(scheduler_template)
    sched_ref.set_timesteps(NUM_STEPS, device=DEVICE)
    all_timesteps = [int(ts.item()) for ts in sched_ref.timesteps]
    print(f"  Timesteps ({len(all_timesteps)}): {all_timesteps}")

    # ── Dispatch ────────────────────────────────────────────────────────
    if args.run_entropy:
        run_entropy_analysis(
            unet, scheduler_template, target_layer, v_raw, dtype,
            all_vscales, all_timesteps, args, OUTPUT_DIR, DEVICE,
        )

    if args.run_taylor:
        run_taylor_analysis(
            unet, scheduler_template, target_layer, v_raw, dtype,
            all_vscales, all_timesteps, args, OUTPUT_DIR, DEVICE,
        )

    print(f"\n{'=' * 70}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
