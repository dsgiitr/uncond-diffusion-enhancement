"""
destructive_interference
────────────────────────
Mini-batch CKA (Centered Kernel Alignment) analysis between
h-space (mid_block) and encoder (down_blocks) activations of
a DDPM UNet.

Three-phase pipeline:
    Phase 1: compute_global_means.py → pre-compute per-layer
             activation means at multiple timesteps from a random
             subset of CelebA-HQ.  One-time dataset dependency.
    Phase 2a (legacy): compute_cka.py → dataset-dependent CKA at
             a single timestep.
    Phase 2b: compute_cka_generative.py → dataset-free CKA via
             patched DDIM reverse process, measuring across time
             buckets with per-sample HSIC storage.
"""

from .cka_core import MiniBatchCKA, GramCKA, adaptive_pool_flatten
from .hooks import MultiLayerHook
from .data import CKAImageDataset, load_celeba_hq, build_dataloader
from .visualize import plot_cka_bar, plot_cka_heatmap, plot_cka_trajectory
