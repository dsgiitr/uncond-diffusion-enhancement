# %% [markdown]
# # Norm Analysis Pipeline - Experiment 1 & 2
# 
# **Goal:**
# *   **Experiment 1:** Compute the normalized L2 distance of the patched point from the training distribution (baseline trajectory) as a function of $\alpha$.
# *   **Experiment 2:** Analyze the norm of the predicted noise by the U-Net on patched h-space activations across different guidance scales and timesteps.
# 
# **Methodology:**
# *   **Pass 1:** Runs an unpatched reverse generative pass to gather the spatial channel mean ($\mu_t$) and standard deviation ($\sigma_t$) at 10 bucket timesteps.
# *   **Pass 2:** Iterates over $\alpha$ scales. For each $\alpha$, we run a *fully guided* generative pass where the mid block is continuously patched ($h_i + \alpha \cdot v$). At the 10 bucket timesteps, we evaluate the metrics.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDIMScheduler
from tqdm.auto import tqdm
import sys
import os

# Ensure we can import from the local framework
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from analysis.cka.hooks import MultiLayerHook

# %% [markdown]
# # Global Configuration

# %%
# Configure the model and vector paths
MODEL_ID = "google/ddpm-celebahq-256"
# Change this to your vector of choice. E.g., "../vectors/sharp_vs_blur_dom_t20.pt"
VECTOR_PATH = "../vectors/high_vs_low_contrast_dom_t20.pt" 

# Experiment Parameters
NUM_SAMPLES = 64        # Set sample size here
BATCH_SIZE = 16         # Set batch size here
NUM_STEPS = 30          # DDIM inference steps
BUCKET_EVERY = 3        # We measure at 10 buckets (30/3 = 10)

ALPHA_LIST = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
DIRECTIONS = ["positive", "negative"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# # Generic Patched Hook
# We will use this to perform actual continuous patching in Pass 2.

# %%
class ContinuousPatchHook:
    """Patches the mid_block continuously via forward hook."""
    def __init__(self, v: torch.Tensor, scale: float = 1.0):
        self.v = v
        self.scale = scale
        self._handle = None

    def _hook_fn(self, module, inp, output):
        return output + self.scale * self.v

    def register(self, layer):
        self.remove()
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

# %% [markdown]
# # Pass 1: Accumulate Background Statistics
# Here we generate samples without any patch ($\alpha=0$) to establish the in-distribution mean and variance of $h_t$.

# %%
print("Initializing Model and Vector...")
unet = UNet2DModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
scheduler = DDIMScheduler.from_pretrained(MODEL_ID)
scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
all_timesteps = scheduler.timesteps

# Figure out our measurement timesteps
measurement_step_indices = set(range(0, NUM_STEPS, BUCKET_EVERY))
step_to_ts = {idx: int(ts.item()) for idx, ts in enumerate(all_timesteps) if idx in measurement_step_indices}
saved_measurement_ts = sorted(list(step_to_ts.values()), reverse=True)

# Load the vector
v = torch.load(VECTOR_PATH, map_location=DEVICE, weights_only=False)
if v.dim() == 1:
    v = v.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
v = v.to(DEVICE, dtype=torch.float32)

print("Running Pass 1: Accumulating unpatched statistics...")

# We accumulate mean and squared diffs over the batch online
channel_mean = {ts: 0.0 for ts in saved_measurement_ts}
channel_M2 = {ts: 0.0 for ts in saved_measurement_ts}
count = 0

hook = MultiLayerHook(unet)
generator = torch.Generator(device=DEVICE).manual_seed(SEED)

num_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
initial_noises = []

for b in tqdm(range(num_batches), desc="Pass 1 Batches"):
    B = min(BATCH_SIZE, NUM_SAMPLES - b * BATCH_SIZE)
    x_T = torch.randn(B, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size, generator=generator, device=DEVICE)
    initial_noises.append(x_T.clone()) # Cache initial noises for Pass 2 to maintain identical paths
    
    x_t = x_T * scheduler.init_noise_sigma
    
    for step_idx, t in enumerate(all_timesteps):
        is_measurement = step_idx in measurement_step_indices
        t_batch = t.expand(B)
        
        latent_input = scheduler.scale_model_input(x_t, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t_batch).sample
            
        if is_measurement:
            ts_val = step_to_ts[step_idx]
            acts = hook.get_activations()
            h = acts["mid_block"] # Base Shape [B, C, H, W]
            
            # Aggregate Welford's over the batch instances (B)
            for i in range(B):
                count += 1
                delta = h[i] - channel_mean[ts_val]
                channel_mean[ts_val] += delta / count
                delta2 = h[i] - channel_mean[ts_val]
                channel_M2[ts_val] += delta * delta2
                
        hook.clear()
        x_t = scheduler.step(noise_pred, t, x_t, generator=generator).prev_sample

# Finalize sample variance and standard deviation
channel_var = {}
channel_std = {}
for ts in saved_measurement_ts:
    channel_var[ts] = channel_M2[ts] / (count - 1)
    channel_std[ts] = torch.sqrt(torch.clamp(channel_var[ts], min=1e-8))

print("Pass 1 Complete!")

# %% [markdown]
# # Pass 2: Parameter Sweep with Active Patching
# Now we fully patch the U-Net reverse process according to $\alpha$, allowing the trajectory divergence. We record normalized L2 distance and predicted noise norm.

# %%
print("Running Pass 2: Sweeping alpha and directions...")

from collections import defaultdict
results_dist = defaultdict(dict)
results_noise = defaultdict(dict)

# Register ContinuousPatchHook FIRST, so MultiLayerHook captures the patched output.
patch_hook = ContinuousPatchHook(v, scale=0.0)
patch_hook.register(unet.mid_block)

# Need to reregister because MultiLayerHook is sensitive to hook registration order
hook.remove()
hook = MultiLayerHook(unet)

for direction in DIRECTIONS:
    sign = 1.0 if direction == "positive" else -1.0
    for alpha in ALPHA_LIST:
        print(f"\\n--- {direction.upper()} | alpha = {alpha} ---")
        patch_hook.scale = sign * alpha
        
        sum_dist = {ts: 0.0 for ts in saved_measurement_ts}
        sum_noise = {ts: 0.0 for ts in saved_measurement_ts}
        total_samples = 0
        
        for b in tqdm(range(num_batches), desc=f"Batches (alpha={alpha})", leave=False):
            B = min(BATCH_SIZE, NUM_SAMPLES - b * BATCH_SIZE)
            x_T = initial_noises[b]
            x_t = x_T * scheduler.init_noise_sigma
            
            for step_idx, t in enumerate(all_timesteps):
                is_measurement = step_idx in measurement_step_indices
                t_batch = t.expand(B)
                
                latent_input = scheduler.scale_model_input(x_t, t)
                with torch.no_grad():
                    # The Unet forward naturally patches mid_block because of patch_hook
                    noise_pred = unet(latent_input, t_batch).sample
                
                if is_measurement:
                    ts_val = step_to_ts[step_idx]
                    acts = hook.get_activations()
                    patched_h = acts["mid_block"] # This includes the patch!
                    
                    # Experiment 1: Calculate Normalized Distance Z-score
                    mu = channel_mean[ts_val].unsqueeze(0).expand_as(patched_h)
                    std = channel_std[ts_val].unsqueeze(0).expand_as(patched_h)
                    
                    z_scored = (patched_h - mu) / std
                    l2_dist_per_sample = torch.linalg.norm(z_scored.flatten(start_dim=1), dim=1)
                    sum_dist[ts_val] += l2_dist_per_sample.sum().item()
                    
                    # Experiment 2: Calculate Noise Norm
                    l2_noise_per_sample = torch.linalg.norm(noise_pred.flatten(start_dim=1), dim=1)
                    sum_noise[ts_val] += l2_noise_per_sample.sum().item()
                
                hook.clear()
                # The U-net noise_pred was patched, so x_t traces a radically different trajectory
                x_t = scheduler.step(noise_pred, t, x_t, generator=generator).prev_sample
            
            total_samples += B
            
        for ts in saved_measurement_ts:
            results_dist[(direction, alpha)][ts] = sum_dist[ts] / total_samples
            results_noise[(direction, alpha)][ts] = sum_noise[ts] / total_samples

hook.remove()
patch_hook.remove()
print("Pass 2 Complete!")

# %% [markdown]
# # Plotting Results

# %%
def get_curve(metric_dict, target_ts, direction, alphas):
    return [metric_dict[(direction, a)][target_ts] for a in alphas]

base_dist = {ts: results_dist[("positive", 0.0)][ts] for ts in saved_measurement_ts}

# -------------------------------------------------------------
# EXPERIMENT 1: Normalized Activation L2 Distance
# -------------------------------------------------------------

# Plot 1A: Each ax is a timestep, x-axis is alpha
num_ts = len(saved_measurement_ts)
cols_ts = 5
rows_ts = (num_ts + cols_ts - 1) // cols_ts
fig1_ts, axes1_ts = plt.subplots(rows_ts, cols_ts, figsize=(4 * cols_ts, 4 * rows_ts), constrained_layout=True)
fig1_ts.suptitle("Experiment 1: Normalized Activation L2 Distance vs Alpha per Timestep", fontsize=18)

axes1_ts = np.atleast_1d(axes1_ts).flatten()

for idx, ts in enumerate(saved_measurement_ts):
    ax = axes1_ts[idx]
    pos_curve = get_curve(results_dist, ts, "positive", ALPHA_LIST)
    neg_curve = get_curve(results_dist, ts, "negative", ALPHA_LIST)
    
    ax.plot(ALPHA_LIST, pos_curve, marker='o', label='Sharp (+v)', color='firebrick')
    ax.plot(ALPHA_LIST, neg_curve, marker='o', label='Blur (-v)', color='steelblue')
    
    b_val = base_dist[ts]
    ax.axhspan(b_val * 0.9, b_val * 1.1, color='gray', alpha=0.25, label='Typical In-Dist Range')
    
    ax.set_title(f"Timestep: {ts}")
    ax.set_xlabel("Alpha (Guidance Scale)")
    ax.set_ylabel("Normalized L2 Distance")
    if idx == 0:
        ax.legend()

# hide unused axes
for idx in range(num_ts, len(axes1_ts)):
    axes1_ts[idx].set_visible(False)

plt.savefig(f"{OUTPUT_DIR}/exp1_dist_vs_alpha.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 1B: Each ax is an alpha, x-axis is timestep
num_alpha = len(ALPHA_LIST)
cols_alpha = 4
rows_alpha = (num_alpha + cols_alpha - 1) // cols_alpha
fig1_alpha, axes1_alpha = plt.subplots(rows_alpha, cols_alpha, figsize=(4 * cols_alpha, 4 * rows_alpha), constrained_layout=True)
fig1_alpha.suptitle("Experiment 1: Normalized Activation L2 Distance Trajectory per Alpha", fontsize=18)

axes1_alpha = np.atleast_1d(axes1_alpha).flatten()

for idx, alpha in enumerate(ALPHA_LIST):
    ax = axes1_alpha[idx]
    pos_curve = [results_dist[("positive", alpha)][ts] for ts in saved_measurement_ts]
    neg_curve = [results_dist[("negative", alpha)][ts] for ts in saved_measurement_ts]
    base_curve = [base_dist[ts] for ts in saved_measurement_ts]
    
    ax.plot(saved_measurement_ts, pos_curve, marker='o', label='Sharp (+v)', color='firebrick')
    ax.plot(saved_measurement_ts, neg_curve, marker='o', label='Blur (-v)', color='steelblue')
    ax.plot(saved_measurement_ts, base_curve, marker='--', label='Baseline (alpha=0)', color='gray')
    
    ax.set_title(f"Alpha: {alpha}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized L2 Distance")
    ax.invert_xaxis()
    if idx == 0:
        ax.legend()

# hide unused axes
for idx in range(num_alpha, len(axes1_alpha)):
    axes1_alpha[idx].set_visible(False)

plt.savefig(f"{OUTPUT_DIR}/exp1_dist_trajectory.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# -------------------------------------------------------------
# EXPERIMENT 2: Model Predicted Noise L2 Norm
# -------------------------------------------------------------

# Plot 2A: Each ax is a timestep, x-axis is alpha
fig2_ts, axes2_ts = plt.subplots(rows_ts, cols_ts, figsize=(4 * cols_ts, 4 * rows_ts), constrained_layout=True)
fig2_ts.suptitle("Experiment 2: Model Predicted Noise L2 Norm vs Alpha per Timestep", fontsize=18)
axes2_ts = np.atleast_1d(axes2_ts).flatten()

for idx, ts in enumerate(saved_measurement_ts):
    ax = axes2_ts[idx]
    pos_curve = get_curve(results_noise, ts, "positive", ALPHA_LIST)
    neg_curve = get_curve(results_noise, ts, "negative", ALPHA_LIST)
    
    ax.plot(ALPHA_LIST, pos_curve, marker='^', label='Sharp (+v)', color='firebrick')
    ax.plot(ALPHA_LIST, neg_curve, marker='^', label='Blur (-v)', color='steelblue')
    
    ax.set_title(f"Timestep: {ts}")
    ax.set_xlabel("Alpha (Guidance Scale)")
    ax.set_ylabel("Predicted Noise L2 Norm")
    if idx == 0:
        ax.legend()

for idx in range(num_ts, len(axes2_ts)):
    axes2_ts[idx].set_visible(False)

plt.savefig(f"{OUTPUT_DIR}/exp2_noise_vs_alpha.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 2B: Each ax is an alpha, x-axis is timestep
fig2_alpha, axes2_alpha = plt.subplots(rows_alpha, cols_alpha, figsize=(4 * cols_alpha, 4 * rows_alpha), constrained_layout=True)
fig2_alpha.suptitle("Experiment 2: Predicted Noise L2 Norm Trajectory per Alpha", fontsize=18)
axes2_alpha = np.atleast_1d(axes2_alpha).flatten()

for idx, alpha in enumerate(ALPHA_LIST):
    ax = axes2_alpha[idx]
    pos_curve = [results_noise[("positive", alpha)][ts] for ts in saved_measurement_ts]
    neg_curve = [results_noise[("negative", alpha)][ts] for ts in saved_measurement_ts]
    base_curve = [results_noise[("positive", 0.0)][ts] for ts in saved_measurement_ts]
    
    ax.plot(saved_measurement_ts, pos_curve, marker='^', label='Sharp (+v)', color='firebrick')
    ax.plot(saved_measurement_ts, neg_curve, marker='^', label='Blur (-v)', color='steelblue')
    ax.plot(saved_measurement_ts, base_curve, marker='--', label='Baseline (alpha=0)', color='gray')
    
    ax.set_title(f"Alpha: {alpha}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Predicted Noise L2 Norm")
    ax.invert_xaxis()
    if idx == 0:
        ax.legend()

for idx in range(num_alpha, len(axes2_alpha)):
    axes2_alpha[idx].set_visible(False)

plt.savefig(f"{OUTPUT_DIR}/exp2_noise_trajectory.png", dpi=150, bbox_inches='tight')
plt.show()
