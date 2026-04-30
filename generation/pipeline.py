"""
Unconditional DDPM generation loops — Baseline / Patched / H-Space CFG.

All three modes run from the SAME initial noise tensor ``x_T`` so the
visual comparison is fair.

GPU-efficiency note
-------------------
The ``run_all_batched`` entry-point runs all three modes sequentially but
each mode processes the entire batch in a single forward pass per step
(no per-sample loops).  For Mode 3 (CFG), the *unpatched* and *patched*
UNet calls are fused into a single batch-doubled forward pass so only
ONE kernel launch happens per timestep instead of two.
"""

from __future__ import annotations

import copy
import torch
from diffusers import (
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from typing import List

import math

from hooks import HSpacePatcher

# ── scheduler registry ──────────────────────────────────────────────────

SCHEDULER_MAP = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}

# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════


def build_scheduler(scheduler_name: str, pipe):
    """Build the requested scheduler from the pretrained pipeline's config."""
    sched_cls = SCHEDULER_MAP[scheduler_name]
    extra = {}
    if scheduler_name == "dpm_solver++":
        extra["algorithm_type"] = "dpmsolver++"
    return sched_cls.from_config(pipe.scheduler.config, **extra)


def get_scheduled_cfg_scale(step_idx: int, num_steps: int, base_cfg: float, scheduler_type: str) -> float:
    """Scales CFG towards a cleaner image as denoising progresses."""
    if scheduler_type == "constant":
        return base_cfg
        
    progress = step_idx / max(1, num_steps - 1)
    
    if scheduler_type == "linear":
        return base_cfg * progress
    elif scheduler_type == "cosine":
        # Smooth S-curve from 0 to 1
        return base_cfg * (0.5 * (1 - math.cos(math.pi * progress)))
    
    return base_cfg


def should_patch(patch_mode: str, step_idx: int,
                 patch_start: int = 0, patch_end: int = 10,
                 patch_timesteps: list | None = None) -> bool:
    """Determine whether the current step should be patched."""
    if patch_mode == "continuous":
        return True
    elif patch_mode == "interval":
        return patch_start <= step_idx <= patch_end
    elif patch_mode == "list":
        return step_idx in (patch_timesteps or [])
    return False


def decode_samples(sample: torch.Tensor) -> List[Image.Image]:
    """Convert a batch of [-1, 1] image tensors to PIL Images."""
    sample = (sample / 2 + 0.5).clamp(0, 1)
    sample = (sample * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    return [Image.fromarray(s.numpy()) for s in sample]


# ═════════════════════════════════════════════════════════════════════════
# Mode 1 — Baseline (clean UNet, no patching)
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_baseline(
    unet,
    scheduler,
    x_T: torch.Tensor,
    num_steps: int,
    seed: int,
    device: str,
) -> List[Image.Image]:
    """Clean DDPM denoising — no patching, no CFG."""
    scheduler.set_timesteps(num_steps, device=device)
    sample = x_T.clone()
    sample = sample * scheduler.init_noise_sigma

    gen = torch.Generator(device=device).manual_seed(seed)

    for t in scheduler.timesteps:
        latent_input = scheduler.scale_model_input(sample, t)
        noise_pred = unet(latent_input, t).sample
        sample = scheduler.step(noise_pred, t, sample, generator=gen).prev_sample

    return decode_samples(sample)


# ═════════════════════════════════════════════════════════════════════════
# Mode 2 — Patched (direct h-space patching, NO CFG)
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_patched(
    unet,
    scheduler,
    x_T: torch.Tensor,
    patcher: HSpacePatcher,
    target_layer,
    num_steps: int,
    seed: int,
    device: str,
    patch_mode: str = "continuous",
    patch_start: int = 0,
    patch_end: int = 10,
    patch_timesteps: list | None = None,
) -> List[Image.Image]:
    """Single-pass DDPM denoising with h-space patching (no CFG)."""
    scheduler.set_timesteps(num_steps, device=device)
    sample = x_T.clone()
    sample = sample * scheduler.init_noise_sigma

    gen = torch.Generator(device=device).manual_seed(seed)

    for step_idx, t in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(sample, t)
        patching = should_patch(patch_mode, step_idx,
                                patch_start, patch_end, patch_timesteps)

        if patching:
            with patcher(target_layer):
                noise_pred = unet(latent_input, t).sample
        else:
            noise_pred = unet(latent_input, t).sample

        sample = scheduler.step(noise_pred, t, sample, generator=gen).prev_sample

    return decode_samples(sample)


# ═════════════════════════════════════════════════════════════════════════
# Mode 3 — H-Space CFG  (dual-pass, batch-doubled for GPU efficiency)
# ═════════════════════════════════════════════════════════════════════════

class _BatchDoubledPatcher:
    """Hook that patches only the SECOND half of a batch-doubled input.

    When we concatenate [unpatched_batch ; patched_batch] along dim 0 and
    run ONE forward pass, this hook adds ``scale * v`` only to activations
    whose batch index >= B (the second half), leaving the first half clean.
    This halves the number of kernel launches compared to two separate passes.
    """

    def __init__(self, v: torch.Tensor, scale: float, half_B: int) -> None:
        self.v = v
        self.scale = scale
        self.half_B = half_B
        self._handle = None

    def _hook_fn(self, module, inp, output):
        # output shape: [2*B, C, H, W]
        # Only patch the second half  [B:2B]
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


@torch.no_grad()
def run_cfg(
    unet,
    scheduler,
    x_T: torch.Tensor,
    patcher: HSpacePatcher,
    target_layer,
    num_steps: int,
    seed: int,
    device: str,
    cfg_scale: float = 3.0,
    cfg_scheduler: str = "constant",
    patch_mode: str = "continuous",
    patch_start: int = 0,
    patch_end: int = 10,
    patch_timesteps: list | None = None,
) -> List[Image.Image]:
    """Dual-pass h-space CFG with batch doubling for GPU efficiency.

    At each step that should be patched:
      1. Concatenate ``[sample, sample]`` along batch dim  → ``[2B, C, H, W]``
      2. Run a SINGLE UNet forward pass with a hook that patches only the
         second half of the batch.
      3. Split the output back into ``noise_unpatched`` and ``noise_patched``.
      4. Apply h-space CFG:
         ``noise_pred = noise_unpatched + cfg_scale * (noise_patched - noise_unpatched)``
    """
    scheduler.set_timesteps(num_steps, device=device)
    sample = x_T.clone()
    sample = sample * scheduler.init_noise_sigma

    gen = torch.Generator(device=device).manual_seed(seed)
    B = x_T.shape[0]

    for step_idx, t in enumerate(scheduler.timesteps):
        latent_input = scheduler.scale_model_input(sample, t)
        patching = should_patch(patch_mode, step_idx,
                                patch_start, patch_end, patch_timesteps)

        # Calculate dynamic CFG scale
        current_cfg_scale = get_scheduled_cfg_scale(step_idx, num_steps, cfg_scale, cfg_scheduler)

        if patching:
            # ── Batch-doubled forward pass ──────────────────────────────
            doubled_input = torch.cat([latent_input, latent_input], dim=0)

            bd_patcher = _BatchDoubledPatcher(patcher.v, patcher.scale, B)
            bd_patcher.register(target_layer)
            try:
                noise_both = unet(doubled_input, t).sample
            finally:
                bd_patcher.remove()

            noise_unpatched = noise_both[:B]
            noise_patched = noise_both[B:]

            # H-space CFG combination
            noise_pred = noise_unpatched + current_cfg_scale * (noise_patched - noise_unpatched)
        else:
            noise_pred = unet(latent_input, t).sample

        sample = scheduler.step(noise_pred, t, sample, generator=gen).prev_sample

    return decode_samples(sample)


# ═════════════════════════════════════════════════════════════════════════
# Fused triplet generation (single loop, batch-quadrupled)
# ═════════════════════════════════════════════════════════════════════════


class _FusedDoubletPatcher:
    """Hook that applies different v-scales to different batch quadrants.

    Batch layout: ``[patched(B) | cfg_clean(B) | cfg_patched(B)]``

    - quadrant 0 (patched)     : ``+= scale_patched * v_patched``
    - quadrant 1 (cfg_clean)   : untouched
    - quadrant 2 (cfg_patched) : ``+= scale_guided  * v_guided``
    """

    def __init__(
        self,
        v_patched: torch.Tensor, scale_patched: float,
        v_guided: torch.Tensor, scale_guided: float,
        B: int,
    ) -> None:
        self.v_patched = v_patched
        self.scale_patched = scale_patched
        self.v_guided = v_guided
        self.scale_guided = scale_guided
        self.B = B
        self._handle = None

    def _hook_fn(self, module, inp, output):
        B = self.B
        out = output.clone()
        out[0:B]         += self.scale_patched * self.v_patched
        out[2 * B:3 * B] += self.scale_guided  * self.v_guided
        return out

    def register(self, layer):
        self.remove()
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def run_fused_doublet(
    unet,
    scheduler,
    x_T: torch.Tensor,
    patcher_patched: HSpacePatcher,
    patcher_guided: HSpacePatcher,
    target_layer,
    num_steps: int,
    seed: int,
    device: str,
    cfg_scale: float = 3.0,
    cfg_scheduler: str = "constant",
    patch_mode: str = "continuous",
    patch_start: int = 0,
    patch_end: int = 10,
    patch_timesteps: list | None = None,
) -> tuple[List[Image.Image], List[Image.Image]]:
    """Fused single-loop generation of patched + CFG.

    Instead of sequential denoising loops, this runs ONE loop with
    a batch-scaled UNet forward pass per timestep:

        ``[patched(B) | cfg_clean(B) | cfg_patched(B)]``

    Returns:
        ``(patched_imgs, cfg_imgs)`` — each ``list[PIL.Image]``
    """
    B = x_T.shape[0]

    # Independent schedulers
    sched_p = copy.deepcopy(scheduler)
    sched_c = copy.deepcopy(scheduler)

    sched_p.set_timesteps(num_steps, device=device)
    sched_c.set_timesteps(num_steps, device=device)

    sample_p = x_T.clone() * sched_p.init_noise_sigma
    sample_c = x_T.clone() * sched_c.init_noise_sigma

    gen_p = torch.Generator(device=device).manual_seed(seed)
    gen_c = torch.Generator(device=device).manual_seed(seed)

    for step_idx, t in enumerate(sched_p.timesteps):
        patching = should_patch(patch_mode, step_idx,
                                patch_start, patch_end, patch_timesteps)

        if patching:
            # ── 3-way fused pass ─────────────────────────────────────
            latent_c = sched_c.scale_model_input(sample_c, t)
            fused_input = torch.cat([
                sched_p.scale_model_input(sample_p, t),
                latent_c,
                latent_c,
            ], dim=0)

            hook = _FusedDoubletPatcher(
                patcher_patched.v, patcher_patched.scale,
                patcher_guided.v,  patcher_guided.scale,
                B,
            )
            hook.register(target_layer)
            try:
                noise_all = unet(fused_input, t).sample
            finally:
                hook.remove()

            noise_p = noise_all[0:B]
            noise_c_clean   = noise_all[B:2 * B]
            noise_c_patched = noise_all[2 * B:3 * B]

            current_cfg = get_scheduled_cfg_scale(
                step_idx, num_steps, cfg_scale, cfg_scheduler,
            )
            noise_cfg = noise_c_clean + current_cfg * (
                noise_c_patched - noise_c_clean
            )
        else:
            # ── 2-way fused pass (no hook needed) ────────────────────
            fused_input = torch.cat([
                sched_p.scale_model_input(sample_p, t),
                sched_c.scale_model_input(sample_c, t),
            ], dim=0)

            noise_all = unet(fused_input, t).sample
            noise_p   = noise_all[0:B]
            noise_cfg = noise_all[B:2 * B]

        sample_p = sched_p.step(noise_p,   t, sample_p, generator=gen_p).prev_sample
        sample_c = sched_c.step(noise_cfg, t, sample_c, generator=gen_c).prev_sample

    return decode_samples(sample_p), decode_samples(sample_c)


# ═════════════════════════════════════════════════════════════════════════
# Convenience: run all three from shared noise
# ═════════════════════════════════════════════════════════════════════════

def generate_initial_noise(unet, batch_size: int, seed: int, device: str):
    """Create reproducible Gaussian noise matching the UNet's input shape."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return randn_tensor(
        (
            batch_size,
            unet.config.in_channels,
            unet.config.sample_size,
            unet.config.sample_size,
        ),
        generator=gen,
        device=device,
    )


def run_all(
    unet,
    scheduler,
    x_T: torch.Tensor,
    patcher: HSpacePatcher,
    target_layer,
    *,
    num_steps: int,
    seed: int,
    device: str,
    cfg_scale: float = 3.0,
    cfg_scheduler: str = "constant",
    patch_mode: str = "continuous",
    patch_start: int = 0,
    patch_end: int = 10,
    patch_timesteps: list | None = None,
):
    """Run Baseline, Patched, and CFG from the same x_T.

    Returns:
        (baseline_imgs, patched_imgs, cfg_imgs)  — each a list of PIL Images.
    """
    common = dict(
        num_steps=num_steps, seed=seed, device=device,
    )
    patch_kw = dict(
        patch_mode=patch_mode, patch_start=patch_start,
        patch_end=patch_end, patch_timesteps=patch_timesteps,
    )

    baseline_imgs = run_baseline(unet, scheduler, x_T, **common)
    patched_imgs = run_patched(unet, scheduler, x_T, patcher, target_layer,
                               **common, **patch_kw)
    cfg_imgs = run_cfg(unet, scheduler, x_T, patcher, target_layer,
                       **common, cfg_scale=cfg_scale, cfg_scheduler=cfg_scheduler, **patch_kw)

    return baseline_imgs, patched_imgs, cfg_imgs
