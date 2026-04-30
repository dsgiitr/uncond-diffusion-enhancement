# %% [markdown]
# # Taylor Decomposition of Decoder Nonlinearity
#
# **Goal:** Decompose the effect of h-space patching into first-order (Jacobian)
# and higher-order (residual) contributions, and precisely characterize
# *where* and *when* linearization of the decoder fails.
#
# ## The Full Decomposition
#
# $$\epsilon_{\text{patch}} - \epsilon_{\text{uncond}}
# = \underbrace{\alpha\, J_D(h_t)\,v}_{\text{1st order}}
# + \underbrace{\mathcal{E}(\alpha, t)}_{\text{residual}}$$
#
# where $\mathcal{E}(\alpha, t) = \tfrac{\alpha^2}{2} H_D(h_t)[v,v] + O(\alpha^3)$.
#
# ## What We Compute
#
# | Qty | What | How |
# |-----|------|-----|
# | Q0 | $\epsilon_{\text{uncond}} = D(h_t)$ | 1 forward pass / timestep |
# | Q1 | $\Delta(\alpha) = D(h_t+\alpha v) - D(h_t)$ | 1 forward pass / $(\alpha, t)$ |
# | Q2 | $J_D(h_t)\,v$ | `torch.autograd.functional.jvp` once / $(v, t)$ |
# | Q3 | $\mathcal{E}(\alpha) = \Delta(\alpha) - \alpha\,J_D(h_t)v$ | no new passes |
#
# ## Plots
#
# - **Plot 2:** $|J_D(h_t)v|$ vs timestep $t$ (no alpha dependence)
# - **Plot 3:** $|\mathcal{E}(\alpha)|/\alpha^2$ vs $\alpha$ at multiple timesteps
# - **Plot 4:** $R(\alpha, t) = |\mathcal{E}|/|\alpha J_D v|$ vs $\alpha$ at multiple timesteps
# - **Plot 5:** $R(\alpha^*, t)$ vs timestep for fixed large $\alpha$ (temporal profile)

# %%
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from diffusers import DDPMPipeline, DDIMScheduler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
GENERATION_DIR = PROJECT_ROOT / "generation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GENERATION_DIR) not in sys.path:
    sys.path.append(str(GENERATION_DIR))

from generation.pipeline import build_scheduler

# %% [markdown]
# # 1. Configuration

# %%
# ── Model & Scheduler ──────────────────────────────────────────────────
MODEL_ID = "google/ddpm-celebahq-256"
SCHEDULER_NAME = "ddim"
NUM_STEPS = 30

# ── Vectors ─────────────────────────────────────────────────────────────
# We compare two directions: sharp (positive v) and blur (negative v)
V_PATH = PROJECT_ROOT / "vectors" / "sharp_vs_blur_dom_t20.pt"

# ── Alpha sweep ─────────────────────────────────────────────────────────
# These are the guidance scales to test
ALPHA_LIST = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
# Fixed alpha for Plot 5 (the temporal profile of linearization failure)
ALPHA_FIXED = 5.0

# ── Sampling ────────────────────────────────────────────────────────────
NUM_SAMPLES = 256         # Number of independent noise samples to average
BATCH_SIZE = 32           # Optimized for batched GPU processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ── Measurement timesteps ──────────────────────────────────────────────
# We measure at 10 equally-spaced step indices across the 30-step trajectory
BUCKET_EVERY = 3  # 30 / 3 = 10 measurement points

# ── Output ──────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "taylor_decomposition_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Scheduler: {SCHEDULER_NAME} | Steps: {NUM_STEPS}")
print(f"Vector: {V_PATH}")
print(f"Alpha sweep: {ALPHA_LIST}")
print(f"Num samples: {NUM_SAMPLES}")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")

# %% [markdown]
# # 2. Initialize Model & Load Vector

# %%
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

pipe = DDPMPipeline.from_pretrained(MODEL_ID).to(DEVICE)
unet = pipe.unet.eval()
for p in unet.parameters():
    p.requires_grad_(False)

dtype = next(unet.parameters()).dtype
scheduler_template = build_scheduler(SCHEDULER_NAME, pipe)
target_layer = unet.mid_block

# Load v
if not V_PATH.exists():
    raise FileNotFoundError(f"Vector not found: {V_PATH}")
v_raw = torch.load(V_PATH, map_location=DEVICE, weights_only=False)
if not isinstance(v_raw, torch.Tensor):
    raise TypeError(f"Expected tensor in {V_PATH}, got {type(v_raw)}")
if v_raw.dim() == 3:
    v_raw = v_raw.unsqueeze(0)  # [1, C, H, W]
v_raw = v_raw.to(device=DEVICE, dtype=dtype)

print(f"UNet dtype: {dtype}")
print(f"Vector v shape: {tuple(v_raw.shape)}")
print(f"Vector v L2 norm: {v_raw.flatten().norm().item():.4f}")

# %%
# Set up scheduler and identify measurement timesteps
scheduler = copy.deepcopy(scheduler_template)
scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
all_timesteps = scheduler.timesteps

measurement_step_indices = set(range(0, NUM_STEPS, BUCKET_EVERY))
step_to_ts = {
    idx: int(ts.item())
    for idx, ts in enumerate(all_timesteps)
    if idx in measurement_step_indices
}
measurement_ts_values = sorted(step_to_ts.values(), reverse=True)

print(f"Measurement step indices: {sorted(measurement_step_indices)}")
print(f"Corresponding timestep values: {measurement_ts_values}")

# %% [markdown]
# # 3. Core Functions
#
# These operate on a SINGLE sample at a time (batch dim = 1) to keep
# memory manageable and enable `torch.autograd.functional.jvp`.

# %%
def predict_noise_with_optional_patch(
    unet: nn.Module,
    target_layer: nn.Module,
    latent_input: torch.Tensor,
    t: torch.Tensor,
    delta_h: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run UNet; if alpha is given, add alpha * delta_h at mid_block."""
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
    """Compute (eps_0, J_D(h_t) v) via explicit forward-mode AD for zero-overhead batching.

    Returns:
        eps_0: D(h_t)          — the unpatched noise prediction
        jvp:   J_D(h_t) v     — the Jacobian-vector product (NO alpha factor)
    """
    import torch.autograd.forward_ad as fwAD

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
        
        # Unpack the dual tensor
        primal_out = fwAD.unpack_dual(out).primal
        tangent_out = fwAD.unpack_dual(out).tangent
        
        if tangent_out is None:
            # Fallback if forward AD graph is disconnected
            tangent_out = torch.zeros_like(primal_out)

    return primal_out.detach(), tangent_out.detach()


def batch_l2_norm(x: torch.Tensor) -> list[float]:
    """Compute L2 norm for each sample in the batch."""
    return x.view(x.shape[0], -1).norm(dim=1).tolist()

# %% [markdown]
# # 4. Run the Experiment
#
# For each sample and each measurement timestep along the DDIM trajectory,
# we compute:
# 1. $\epsilon_{\text{uncond}}$ and $J_D(h_t)v$ via JVP (once per timestep)
# 2. $\epsilon_{\text{patch}}(\alpha)$ for each alpha (one pass per alpha)
# 3. Derive $\Delta(\alpha)$ and $\mathcal{E}(\alpha)$ from the above
#
# We run two directions: **sharp** ($+v$) and **blur** ($-v$).

# %%
DIRECTIONS = {
    "sharp": 1.0,   # +v
    "blur": -1.0,   # -v
}

# Storage: results[direction][ts_value] = {
#   "jvp_norms": [float per sample],
#   "delta_norms": {alpha: [float per sample]},
#   "residual_norms": {alpha: [float per sample]},
# }
results = {}
for dir_name in DIRECTIONS:
    results[dir_name] = {}
    for ts in measurement_ts_values:
        results[dir_name][ts] = {
            "jvp_norms": [],
            "delta_norms": {a: [] for a in ALPHA_LIST},
            "residual_norms": {a: [] for a in ALPHA_LIST},
        }

t_start = time.time()

for batch_start in range(0, NUM_SAMPLES, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, NUM_SAMPLES)
    actual_batch_size = batch_end - batch_start
    print(f"\n{'='*60}")
    print(f"Batch {batch_start} to {batch_end - 1} (of {NUM_SAMPLES})")
    print(f"{'='*60}")

    for dir_name, sign in DIRECTIONS.items():
        delta_h = sign * v_raw  # [1, C, H, W]
        print(f"\n  Direction: {dir_name} (sign={sign:+.0f})")

        # ── Generate the baseline trajectory and cache x_t at measurement steps ──
        sched = copy.deepcopy(scheduler_template)
        sched.set_timesteps(NUM_STEPS, device=DEVICE)

        x_T_list = []
        for i in range(actual_batch_size):
            seed_i = SEED + batch_start + i
            gen = torch.Generator(device=DEVICE).manual_seed(seed_i)
            x_i = torch.randn(
                (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size),
                generator=gen, device=DEVICE, dtype=dtype,
            )
            x_T_list.append(x_i)
        
        x_T = torch.cat(x_T_list, dim=0)
        sample = x_T.clone() * sched.init_noise_sigma

        # We need to run the full baseline trajectory and save x_t at measurement points
        cached_states = {}  # step_idx -> (latent_input, t_tensor)

        gen_step = torch.Generator(device=DEVICE).manual_seed(SEED + batch_start)
        for step_idx, t in enumerate(tqdm(sched.timesteps, desc=f"    Baseline trajectory", leave=False)):
            latent_input = sched.scale_model_input(sample, t)
            
            # Ensure t_tensor is properly batched
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            if t_tensor.shape[0] != latent_input.shape[0]:
                t_tensor = t_tensor.expand(latent_input.shape[0])

            if step_idx in measurement_step_indices:
                # Cache the state BEFORE the UNet call
                cached_states[step_idx] = (latent_input.clone(), t_tensor.clone())

            with torch.no_grad():
                noise_pred = unet(latent_input, t_tensor).sample

            sample = sched.step(noise_pred, t, sample, generator=gen_step).prev_sample

        # ── At each measurement point, compute JVP and alpha sweep ──
        for step_idx in sorted(cached_states.keys()):
            ts_val = step_to_ts[step_idx]
            latent_input, t_tensor = cached_states[step_idx]

            # --- Quantity 2: J_D(h_t) v via JVP (no alpha dependence) ---
            eps_uncond, jvp_val = compute_jvp_at_zero(
                unet, target_layer, latent_input, t_tensor, delta_h,
            )
            jvp_norms = batch_l2_norm(jvp_val)
            results[dir_name][ts_val]["jvp_norms"].extend(jvp_norms)

            # --- Quantity 1 & 3: sweep over alpha ---
            for alpha in ALPHA_LIST:
                alpha_tensor = torch.tensor(alpha, device=DEVICE, dtype=dtype)

                with torch.no_grad():
                    eps_patched = predict_noise_with_optional_patch(
                        unet, target_layer, latent_input, t_tensor,
                        delta_h=delta_h, alpha=alpha_tensor,
                    )

                # Delta(alpha) = eps_patched - eps_uncond
                delta = eps_patched - eps_uncond
                delta_norms = batch_l2_norm(delta)

                # Residual E(alpha) = Delta(alpha) - alpha * jvp_val
                residual = delta - alpha * jvp_val
                res_norms = batch_l2_norm(residual)

                results[dir_name][ts_val]["delta_norms"][alpha].extend(delta_norms)
                results[dir_name][ts_val]["residual_norms"][alpha].extend(res_norms)

            # Free GPU memory for this measurement point
            del eps_uncond, jvp_val, eps_patched, delta, residual
            torch.cuda.empty_cache()

        del cached_states
        torch.cuda.empty_cache()

elapsed = time.time() - t_start
print(f"\n✓ All computations done in {elapsed:.1f}s")

# %% [markdown]
# # 5. Aggregate Results
#
# Average over samples and compute derived quantities.

# %%
# Aggregate: mean over samples
agg = {}
for dir_name in DIRECTIONS:
    agg[dir_name] = {}
    for ts in measurement_ts_values:
        r = results[dir_name][ts]
        entry = {}

        # Mean JVP norm (no alpha dependence)
        entry["jvp_norm_mean"] = float(np.mean(r["jvp_norms"]))
        entry["jvp_norm_std"] = float(np.std(r["jvp_norms"]))

        # Per-alpha metrics
        entry["alpha_metrics"] = {}
        for alpha in ALPHA_LIST:
            delta_norms = np.array(r["delta_norms"][alpha])
            res_norms = np.array(r["residual_norms"][alpha])

            # Normalized residual: |E(alpha)| / alpha^2
            res_over_a2 = res_norms / (alpha ** 2) if alpha > 0 else res_norms * 0.0

            # Ratio R = |E| / |alpha * J_D v| = |E| / (alpha * |J_D v|)
            jvp_mean = entry["jvp_norm_mean"]
            if alpha > 0 and jvp_mean > 1e-10:
                ratio_R = res_norms / (alpha * jvp_mean)
            else:
                ratio_R = res_norms * 0.0

            entry["alpha_metrics"][alpha] = {
                "delta_norm_mean": float(np.mean(delta_norms)),
                "delta_norm_std": float(np.std(delta_norms)),
                "residual_norm_mean": float(np.mean(res_norms)),
                "residual_norm_std": float(np.std(res_norms)),
                "res_over_alpha2_mean": float(np.mean(res_over_a2)),
                "res_over_alpha2_std": float(np.std(res_over_a2)),
                "ratio_R_mean": float(np.mean(ratio_R)),
                "ratio_R_std": float(np.std(ratio_R)),
            }

        agg[dir_name][ts] = entry

# %% [markdown]
# # 6. Save All Data

# %%
# Save raw results as .pt
torch.save(results, OUTPUT_DIR / "taylor_raw_results.pt")

# Save aggregated results as JSON (human-readable)
# Convert keys to strings for JSON serialization
json_agg = {}
for dir_name, ts_dict in agg.items():
    json_agg[dir_name] = {}
    for ts, entry in ts_dict.items():
        json_entry = {
            "jvp_norm_mean": entry["jvp_norm_mean"],
            "jvp_norm_std": entry["jvp_norm_std"],
            "alpha_metrics": {},
        }
        for alpha, metrics in entry["alpha_metrics"].items():
            json_entry["alpha_metrics"][str(alpha)] = metrics
        json_agg[dir_name][str(ts)] = json_entry

with open(OUTPUT_DIR / "taylor_aggregated_results.json", "w") as f:
    json.dump(json_agg, f, indent=2)

print(f"Saved raw results to {OUTPUT_DIR / 'taylor_raw_results.pt'}")
print(f"Saved aggregated results to {OUTPUT_DIR / 'taylor_aggregated_results.json'}")

# %% [markdown]
# # 7. Plot 2 — $|J_D(h_t) v|$ vs Timestep
#
# This is one number per $(v, t)$ — **no alpha dependence**.
# It tells us how *decodable* each direction is at each timestep.

# %%
# ── Plotting style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_SHARP = "#E63946"    # vivid red
COLOR_BLUR = "#457B9D"     # steel blue
COLOR_BG = "#F1FAEE"       # light sage
COLOR_GRID = "#D4D4D4"

# %%
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_facecolor(COLOR_BG)
fig2.patch.set_facecolor("white")

ts_arr = np.array(measurement_ts_values)

for dir_name, color, marker, label in [
    ("sharp", COLOR_SHARP, "o", "Sharp (+v)"),
    ("blur", COLOR_BLUR, "s", "Blur (−v)"),
]:
    means = [agg[dir_name][ts]["jvp_norm_mean"] for ts in measurement_ts_values]
    stds = [agg[dir_name][ts]["jvp_norm_std"] for ts in measurement_ts_values]

    ax2.errorbar(ts_arr, means, yerr=stds, marker=marker, markersize=7,
                 linewidth=2.2, capsize=4, label=label, color=color,
                 markeredgecolor="white", markeredgewidth=0.8)

ax2.set_xlabel("Timestep $t$")
ax2.set_ylabel("$|J_D(h_t)\\, v|$  (L2 norm)")
ax2.set_title("Plot 2: Jacobian Magnitude — How decodable is each direction?")
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.4, color=COLOR_GRID)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot2_jacobian_magnitude.png")
plt.savefig(OUTPUT_DIR / "plot2_jacobian_magnitude.pdf")
plt.show()

print("Plot 2 saved.")

# %% [markdown]
# # 8. Plot 3 — $|\mathcal{E}(\alpha)| / \alpha^2$ vs $\alpha$
#
# Tests whether the residual is purely second order or has higher-order terms.
# - **Flat:** pure $O(\alpha^2)$ — residual is well-behaved quadratic.
# - **Growing:** $O(\alpha^3)$+ — nonlinear explosion, especially for sharp at low $t$.

# %%
# Select a few representative timesteps to highlight
ts_highlight = measurement_ts_values  # show all

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig3.suptitle("Plot 3: $|\\mathcal{E}(\\alpha)| / \\alpha^2$ vs $\\alpha$ — Is the residual purely quadratic?",
              fontsize=15, fontweight="bold")

cmap_sharp = plt.cm.Reds(np.linspace(0.3, 0.95, len(ts_highlight)))
cmap_blur = plt.cm.Blues(np.linspace(0.3, 0.95, len(ts_highlight)))

for ax, dir_name, cmap, title in [
    (axes3[0], "sharp", cmap_sharp, "Sharp (+v)"),
    (axes3[1], "blur", cmap_blur, "Blur (−v)"),
]:
    ax.set_facecolor(COLOR_BG)
    for i, ts in enumerate(ts_highlight):
        alpha_vals = np.array(ALPHA_LIST)
        res_a2_means = [agg[dir_name][ts]["alpha_metrics"][a]["res_over_alpha2_mean"]
                        for a in ALPHA_LIST]
        res_a2_stds = [agg[dir_name][ts]["alpha_metrics"][a]["res_over_alpha2_std"]
                       for a in ALPHA_LIST]

        ax.errorbar(alpha_vals, res_a2_means, yerr=res_a2_stds,
                     marker="o", markersize=5, linewidth=1.5, capsize=3,
                     color=cmap[i], label=f"t={ts}")

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$|\\mathcal{E}(\\alpha)| / \\alpha^2$")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.4, color=COLOR_GRID)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot3_residual_alpha2_normalized.png")
plt.savefig(OUTPUT_DIR / "plot3_residual_alpha2_normalized.pdf")
plt.show()

print("Plot 3 saved.")

# %% [markdown]
# # 9. Plot 4 — $R(\alpha, t)$ vs $\alpha$ at Multiple Timesteps
#
# **The smoking gun.** Shows when and where linearization fails.
#
# $$R(\alpha, t) = \frac{|\mathcal{E}(\alpha, t)|}{|\alpha\, J_D(h_t) v|}$$
#
# - $R \ll 1$: linearization valid, first-order dominates
# - $R \sim 1$: linearization breaking down
# - $R \gg 1$: linearization has completely failed

# %%
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig4.suptitle("Plot 4: $R(\\alpha, t) = |\\mathcal{E}| / |\\alpha J_D v|$ — Linearization Failure",
              fontsize=15, fontweight="bold")

for ax, dir_name, cmap, title in [
    (axes4[0], "sharp", cmap_sharp, "Sharp (+v)"),
    (axes4[1], "blur", cmap_blur, "Blur (−v)"),
]:
    ax.set_facecolor(COLOR_BG)

    for i, ts in enumerate(ts_highlight):
        alpha_vals = np.array(ALPHA_LIST)
        R_means = [agg[dir_name][ts]["alpha_metrics"][a]["ratio_R_mean"]
                   for a in ALPHA_LIST]
        R_stds = [agg[dir_name][ts]["alpha_metrics"][a]["ratio_R_std"]
                  for a in ALPHA_LIST]

        ax.errorbar(alpha_vals, R_means, yerr=R_stds,
                     marker="o", markersize=5, linewidth=1.5, capsize=3,
                     color=cmap[i], label=f"t={ts}")

    # Draw the R=1 line (linearization failure threshold)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
               alpha=0.7, label="$R=1$ (failure)")

    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$R(\\alpha, t)$")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.4, color=COLOR_GRID)
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot4_ratio_R_vs_alpha.png")
plt.savefig(OUTPUT_DIR / "plot4_ratio_R_vs_alpha.pdf")
plt.show()

print("Plot 4 saved.")

# %% [markdown]
# # 10. Plot 5 — $R(\alpha^*, t)$ vs Timestep for Fixed Large $\alpha$
#
# **The cleanest figure for the paper.** Fix $\alpha$ at the value where you
# know images explode and sweep across timesteps.
#
# Expected: $R$ is small at high $t$, grows rapidly as $t$ decreases below
# $t^* \approx 500$, specifically for the sharp direction. Blur stays flat.

# %%
# Find the closest alpha in our list to ALPHA_FIXED
alpha_plot5 = min(ALPHA_LIST, key=lambda a: abs(a - ALPHA_FIXED))
print(f"Using alpha = {alpha_plot5} for Plot 5 (requested {ALPHA_FIXED})")

fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.set_facecolor(COLOR_BG)
fig5.patch.set_facecolor("white")

for dir_name, color, marker, label in [
    ("sharp", COLOR_SHARP, "o", f"Sharp (+v), α={alpha_plot5}"),
    ("blur", COLOR_BLUR, "s", f"Blur (−v), α={alpha_plot5}"),
]:
    R_means = [agg[dir_name][ts]["alpha_metrics"][alpha_plot5]["ratio_R_mean"]
               for ts in measurement_ts_values]
    R_stds = [agg[dir_name][ts]["alpha_metrics"][alpha_plot5]["ratio_R_std"]
              for ts in measurement_ts_values]

    ax5.errorbar(ts_arr, R_means, yerr=R_stds, marker=marker, markersize=8,
                 linewidth=2.5, capsize=5, label=label, color=color,
                 markeredgecolor="white", markeredgewidth=0.8)

ax5.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
            alpha=0.7, label="$R=1$ (linearization fails)")

ax5.set_xlabel("Timestep $t$")
ax5.set_ylabel("$R(\\alpha, t)$")
ax5.set_title(f"Plot 5: Temporal Profile of Linearization Failure (α = {alpha_plot5})")
ax5.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax5.grid(True, alpha=0.4, color=COLOR_GRID)
ax5.set_yscale("log")
ax5.invert_xaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot5_temporal_profile.png")
plt.savefig(OUTPUT_DIR / "plot5_temporal_profile.pdf")
plt.show()

print("Plot 5 saved.")

# %% [markdown]
# # 11. Bonus: First-Order Consistency Check
#
# $|\text{FirstOrder}(\alpha)| / \alpha = |J_D(h_t) v|$ should be **constant**
# across alpha. If it varies, the JVP estimate is wrong.

# %%
fig_check, axes_check = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig_check.suptitle("Consistency Check: $|\\alpha J_D v| / \\alpha = |J_D v|$ should be constant",
                   fontsize=14, fontweight="bold")

for ax, dir_name, cmap, title in [
    (axes_check[0], "sharp", cmap_sharp, "Sharp (+v)"),
    (axes_check[1], "blur", cmap_blur, "Blur (−v)"),
]:
    ax.set_facecolor(COLOR_BG)
    for i, ts in enumerate(ts_highlight):
        # The first-order term is alpha * JVP, divided by alpha = JVP norm
        # This should be constant (horizontal line) for all alpha
        jvp_norm = agg[dir_name][ts]["jvp_norm_mean"]
        ax.axhline(y=jvp_norm, color=cmap[i], linestyle="--", alpha=0.5)
        ax.scatter(ALPHA_LIST, [jvp_norm] * len(ALPHA_LIST),
                   color=cmap[i], s=30, label=f"t={ts}", zorder=5)

    ax.set_xlabel("$\\alpha$ (should have no effect)")
    ax.set_ylabel("$|J_D(h_t) v|$")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.4, color=COLOR_GRID)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "consistency_check.png")
plt.show()
print("Consistency check saved.")

# %% [markdown]
# # 12. Summary Table

# %%
print("\n" + "=" * 80)
print("SUMMARY TABLE:  R(alpha, t) — Mean Linearization Failure Ratio")
print("=" * 80)

# Print header
header = f"{'Dir':>6} {'ts':>6} |"
for a in ALPHA_LIST:
    header += f"  α={a:<5}"
print(header)
print("-" * len(header))

for dir_name in DIRECTIONS:
    for ts in measurement_ts_values:
        row = f"{dir_name:>6} {ts:>6} |"
        for a in ALPHA_LIST:
            R = agg[dir_name][ts]["alpha_metrics"][a]["ratio_R_mean"]
            if R > 1.0:
                row += f"  {R:>5.2f}*"  # mark failures
            else:
                row += f"  {R:>5.3f}"
        print(row)
    print()

print("* = R > 1 (linearization failure)")
print(f"\nAll results saved to: {OUTPUT_DIR}/")
