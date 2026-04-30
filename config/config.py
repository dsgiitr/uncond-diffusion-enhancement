"""
config.py
─────────
Centralised configuration for the entire final-battle pipeline, including
the Unconditional DDPM pipeline and the Multi-timestep Concept Extraction pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DDPMConfig:
    """Configuration for unconditional DDPM generation and h-space patching."""
    # ── Model ───────────────────────────────────────────────────────────
    model_id: str = "google/ddpm-celebahq-256"

    # ── Scheduler ───────────────────────────────────────────────────────
    # Supported: "ddpm", "ddim", "pndm", "euler", "euler_ancestral",
    #            "dpm_solver", "dpm_solver++"
    scheduler_type: str = "ddim"
    num_inference_steps: int = 30

    # ── H-Space CFG ─────────────────────────────────────────────────────
    cfg_scale: float = 3.0
    cfg_scheduler: str = "constant"  # "constant", "linear", or "cosine"

    # ── H-Space Patching ────────────────────────────────────────────────
    v_path: str = "v.pt"           # path to the patch tensor (.pt)
    v_scale: float = 1.0           # scalar multiplier on v
    target_layer: str = "mid_block"  # attribute name on the UNet

    # Patching schedule — choose one mode:
    #   "continuous"  → patch every timestep
    #   "interval"    → patch steps in [patch_start, patch_end]
    #   "list"        → patch only the step indices in patch_timesteps
    patch_mode: str = "continuous"
    patch_start: int = 0           # inclusive (step index, not t value)
    patch_end: int = 10            # inclusive
    patch_timesteps: Optional[List[int]] = None  # for "list" mode

    # ── Generation ──────────────────────────────────────────────────────
    seed: int = 42
    batch_size: int = 4            # default to 4 for GPU batching
    device: str = "cuda"

    # ── Output ──────────────────────────────────────────────────────────
    output_dir: str = "results"


@dataclass
class ExtractionConfig:
    """
    Configuration for h-space concept extraction runs.

    Timestep capture logic
    ──────────────────────
    The scheduler is set to ``num_steps`` inference steps (default 50).
    We capture h-space activations at every ``capture_interval``-th step,
    starting from step index 0, for a total of ``num_capture_steps`` steps.

    With the defaults (num_steps=50, capture_interval=5, num_capture_steps=10)
    the captured step indices are: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45].
    These are then mapped to actual scheduler timestep values (e.g. 981, 961, …).
    """

    # ── Model ────────────────────────────────────────────────────────────────
    model_id: str = "google/ddpm-celebahq-256"

    # ── Scheduler ────────────────────────────────────────────────────────────
    num_steps: int = 50
    scheduler_type: str = "ddim"          #  "ddpm" | "ddim"

    # ── Timestep capture ─────────────────────────────────────────────────────
    capture_interval: int = 5
    num_capture_steps: int = 10

    # ── Data / batch ─────────────────────────────────────────────────────────
    dataset_profile: str = "celeba_hq"      # "celeba_hq" | "lsun_church"
    dataset_dir: str = "celeba_hq_dataset"  # local path (HF disk / image folder / LSUN LMDB)
    hf_dataset: str = "korexyz/celeba-hq-256x256"
    dataset_split: str = "train"
    image_size: int = 256
    num_samples: int = 100
    batch_size: int = 16
    num_workers: int = 4

    # ── Transform hyperparams ────────────────────────────────────────────────
    blur_kernel_size: int = 21
    blur_sigma: float = 3.0
    oversaturation_factor: float = 1.8
    high_contrast_factor: float = 1.6
    low_contrast_factor: float = 0.6
    high_brightness_factor: float = 1.5
    low_brightness_factor: float = 0.5
    warm_strength: float = 0.35
    cool_strength: float = 0.35

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "./outputs"
    device: str = ""                      # auto-detect if empty
    use_amp: bool = True                  # fp16 autocast on CUDA

    # ── Derived ──────────────────────────────────────────────────────────────
    def capture_step_indices(self) -> List[int]:
        """Return the scheduler-step indices at which to capture."""
        return [
            i * self.capture_interval
            for i in range(self.num_capture_steps)
        ]
