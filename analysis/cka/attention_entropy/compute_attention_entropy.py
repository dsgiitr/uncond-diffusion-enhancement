import os
import sys
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from diffusers import DDPMPipeline

PROJECT_ROOT = Path("/opt/watchdog/users/glitch/adv_diffusion/final-battle")
UNCONDITIONAL_DDPM_DIR = PROJECT_ROOT / "unconditional_ddpm"
if str(UNCONDITIONAL_DDPM_DIR) not in sys.path:
    sys.path.append(str(UNCONDITIONAL_DDPM_DIR))

from pipeline import build_scheduler

# ── 1. Configuration ────────────────────────────────────────────────────────
MODEL_ID = "google/ddpm-celebahq-256"
SCHEDULER_NAME = "ddim"
NUM_STEPS = 30

V_PATH = PROJECT_ROOT / "vectors" / "sharp_vs_blur_dom_t20.pt"

DIRECTIONS = {
    "sharp": 1.0,
    "blur": -1.0,
}

ALPHA_LIST = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
ALPHA_FIXED_TIME_PLOT = 5.0

NUM_SAMPLES = 512
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BUCKET_EVERY = 3

OUTPUT_DIR = PROJECT_ROOT / "destructive_interference" / "attention_entropy" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. Custom Attention Processor ──────────────────────────────────────────
class EntropyAttnProcessor2_0:
    def __init__(self):
        self.entropies = []

    def reset(self):
        self.entropies.clear()

    def get_last_entropy(self):
        if len(self.entropies) == 0:
            return None
        return self.entropies[-1]

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
        args = () if getattr(attn, "use_pe", False) else ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * (head_dim ** -0.5)

        attention_probs = F.softmax(attention_scores, dim=-1)

        epsilon = 1e-12
        entropy = -(attention_probs * torch.log(attention_probs + epsilon)).sum(dim=-1)
        batch_entropy = entropy.mean(dim=-1).mean(dim=-1)
        self.entropies.append(batch_entropy.detach().cpu())

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

# ── 3. Helper Functions ──────────────────────────────────────────────────
def predict_noise_with_optional_patch(
    unet: nn.Module,
    target_layer: nn.Module,
    latent_input: torch.Tensor,
    t: torch.Tensor,
    delta_h: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
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

# ── 4. Main Script ────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    
    pipe = DDPMPipeline.from_pretrained(MODEL_ID).to(DEVICE)
    unet = pipe.unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
        
    dtype = next(unet.parameters()).dtype
    scheduler_template = build_scheduler(SCHEDULER_NAME, pipe)
    target_layer = unet.mid_block
    
    attn_module = unet.up_blocks[1].attentions[0]
    entropy_processor = EntropyAttnProcessor2_0()
    attn_module.set_processor(entropy_processor)
    
    v_raw = torch.load(V_PATH, map_location=DEVICE, weights_only=False)
    if v_raw.dim() == 3:
        v_raw = v_raw.unsqueeze(0)
    v_raw = v_raw.to(device=DEVICE, dtype=dtype)
    
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
    
    print(f"Num samples: {NUM_SAMPLES}, Batch: {BATCH_SIZE}")
    
    results = {}
    for dir_name in DIRECTIONS:
        results[dir_name] = {}
        for ts in measurement_ts_values:
            results[dir_name][ts] = {
                "unpatched": [],
                "patched": {a: [] for a in ALPHA_LIST}
            }
            
    t_start = time.time()
    
    for batch_start in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, NUM_SAMPLES)
        actual_batch_size = batch_end - batch_start
        print(f"\nBatch {batch_start} to {batch_end - 1}")
        
        sched = copy.deepcopy(scheduler_template)
        sched.set_timesteps(NUM_STEPS, device=DEVICE)
        
        x_T_list = []
        for i in range(actual_batch_size):
            gen = torch.Generator(device=DEVICE).manual_seed(SEED + batch_start + i)
            x_i = torch.randn(
                (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size),
                generator=gen, device=DEVICE, dtype=dtype,
            )
            x_T_list.append(x_i)
            
        x_T = torch.cat(x_T_list, dim=0)
        sample = x_T.clone() * sched.init_noise_sigma
        
        cached_states = {}
        gen_step = torch.Generator(device=DEVICE).manual_seed(SEED + batch_start)
        for step_idx, t in enumerate(sched.timesteps):
            latent_input = sched.scale_model_input(sample, t)
            t_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            
            if step_idx in measurement_step_indices:
                cached_states[step_idx] = (latent_input.clone(), t_tensor.clone())
                
            with torch.no_grad():
                noise_pred = unet(latent_input, t_tensor).sample
                
            sample = sched.step(noise_pred, t, sample, generator=gen_step).prev_sample
            
        for dir_name, sign in DIRECTIONS.items():
            delta_h = sign * v_raw
            
            for step_idx in sorted(cached_states.keys()):
                ts_val = step_to_ts[step_idx]
                latent_input, t_tensor = cached_states[step_idx]
                
                entropy_processor.reset()
                with torch.no_grad():
                    predict_noise_with_optional_patch(unet, target_layer, latent_input, t_tensor)
                
                batch_ent_unpatched = entropy_processor.get_last_entropy()
                results[dir_name][ts_val]["unpatched"].extend(batch_ent_unpatched.tolist())
                
                for alpha in ALPHA_LIST:
                    alpha_tensor = torch.tensor(alpha, device=DEVICE, dtype=dtype)
                    entropy_processor.reset()
                    with torch.no_grad():
                        predict_noise_with_optional_patch(
                            unet, target_layer, latent_input, t_tensor, delta_h, alpha_tensor
                        )
                    batch_ent_patched = entropy_processor.get_last_entropy()
                    results[dir_name][ts_val]["patched"][alpha].extend(batch_ent_patched.tolist())

    print(f"\nDone in {time.time() - t_start:.1f}s")
    
    agg = {}
    for dir_name in DIRECTIONS:
        agg[dir_name] = {}
        for ts in measurement_ts_values:
            u_arr = np.array(results[dir_name][ts]["unpatched"])
            agg[dir_name][ts] = {
                "unpatched_mean": float(u_arr.mean()),
                "unpatched_std": float(u_arr.std()),
                "alphas": {}
            }
            for a in ALPHA_LIST:
                p_arr = np.array(results[dir_name][ts]["patched"][a])
                agg[dir_name][ts]["alphas"][str(a)] = {
                    "mean": float(p_arr.mean()),
                    "std": float(p_arr.std())
                }
                
    torch.save(results, OUTPUT_DIR / "attention_entropy_raw.pt")
    with open(OUTPUT_DIR / "attention_entropy_agg.json", "w") as f:
        json.dump(agg, f, indent=2)
        
    print("Data saved.")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ts_arr = np.array(measurement_ts_values)
    
    s_means = [agg["sharp"][ts]["alphas"][str(ALPHA_FIXED_TIME_PLOT)]["mean"] for ts in ts_arr]
    s_stds = [agg["sharp"][ts]["alphas"][str(ALPHA_FIXED_TIME_PLOT)]["std"] for ts in ts_arr]
    ax.errorbar(ts_arr, s_means, yerr=s_stds, label="Patched (Sharp +v)", marker="o", color="#E63946")
    
    b_means = [agg["blur"][ts]["alphas"][str(ALPHA_FIXED_TIME_PLOT)]["mean"] for ts in ts_arr]
    b_stds = [agg["blur"][ts]["alphas"][str(ALPHA_FIXED_TIME_PLOT)]["std"] for ts in ts_arr]
    ax.errorbar(ts_arr, b_means, yerr=b_stds, label="Patched (Blur -v)", marker="s", color="#457B9D")
    
    u_means = [agg["sharp"][ts]["unpatched_mean"] for ts in ts_arr]
    u_stds = [agg["sharp"][ts]["unpatched_mean"] for ts in ts_arr]
    ax.errorbar(ts_arr, u_means, yerr=u_stds, label="Unpatched", marker="^", color="gray", linestyle="--")
    
    ax.invert_xaxis()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Attention Entropy")
    ax.set_title(f"Attention Entropy over Timesteps (alpha={ALPHA_FIXED_TIME_PLOT})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(OUTPUT_DIR / "plot_entropy_vs_timestep.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    ts_sensitive = [858, 561, 264] 
    fig, axes = plt.subplots(1, len(ts_sensitive), figsize=(15, 5), sharey=True)
    
    for i, ts in enumerate(ts_sensitive):
        if ts not in ts_arr: continue
        ax = axes[i]
        u_mean = agg["sharp"][ts]["unpatched_mean"]
        ax.axhline(u_mean, color="gray", linestyle="--", label="Unpatched")
        
        s_vals = [agg["sharp"][ts]["alphas"][str(a)]["mean"] for a in ALPHA_LIST]
        ax.plot(ALPHA_LIST, s_vals, label="Sharp (+v)", marker="o", color="#E63946")
        
        b_vals = [agg["blur"][ts]["alphas"][str(a)]["mean"] for a in ALPHA_LIST]
        ax.plot(ALPHA_LIST, b_vals, label="Blur (-v)", marker="s", color="#457B9D")
        
        ax.set_xlabel("Alpha Magnitude")
        ax.set_title(f"Timestep {ts}")
        if i == 0:
            ax.set_ylabel("Attention Entropy")
            ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_entropy_vs_alpha.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Plots saved.")

if __name__ == "__main__":
    main()
