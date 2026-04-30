"""
Multi-layer activation capture hooks for CKA analysis.

Captures the *final output* of:
  - Each ``down_blocks[i]``  (encoder blocks)
  - ``mid_block``            (h-space)

Down-block outputs are tuples ``(hidden_states, residual_samples)``.
We capture ``hidden_states`` (element 0) — the block-level output,
not individual ResNet layers within the block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MultiLayerHook:
    """Register forward hooks on selected UNet layers and capture outputs.

    Usage::

        hook = MultiLayerHook(unet)
        with torch.no_grad():
            unet(x_t, t)
        acts = hook.get_activations()   # {layer_name: tensor}
        hook.remove()

    Args:
        unet:                  ``UNet2DModel`` instance.
        encoder_block_indices: Which ``down_blocks`` to hook.
                               ``None`` → all blocks.
    """

    def __init__(
        self,
        unet,
        encoder_block_indices: Optional[List[int]] = None,
    ):
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        num_down = len(unet.down_blocks)
        if encoder_block_indices is None:
            encoder_block_indices = list(range(num_down))

        # ── Encoder blocks ──────────────────────────────────────────────
        for i in encoder_block_indices:
            block = unet.down_blocks[i]
            handle = block.register_forward_hook(
                self._make_encoder_hook(f"down_block_{i}")
            )
            self._handles.append(handle)

        # ── Mid-block (h-space) ─────────────────────────────────────────
        handle = unet.mid_block.register_forward_hook(
            self._make_mid_hook("mid_block")
        )
        self._handles.append(handle)

        self.layer_names: List[str] = [
            f"down_block_{i}" for i in encoder_block_indices
        ] + ["mid_block"]

    # ── Hook factories ──────────────────────────────────────────────────

    def _make_encoder_hook(self, name: str):
        """Down-block output is ``(hidden_states, res_samples)``."""
        def hook_fn(module: nn.Module, inp, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook_fn

    def _make_mid_hook(self, name: str):
        """Mid-block output is a plain tensor."""
        def hook_fn(module: nn.Module, inp, output):
            self.activations[name] = output.detach()
        return hook_fn

    # ── Public API ──────────────────────────────────────────────────────

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return a snapshot of captured activations from the last forward."""
        return dict(self.activations)

    def clear(self):
        """Free GPU memory held by stored activations."""
        self.activations.clear()

    def remove(self):
        """Detach all hooks and release references."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.activations.clear()

    def __del__(self):
        self.remove()
