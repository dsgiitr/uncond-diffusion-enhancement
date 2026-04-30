"""
Forward-hook based h-space patcher (self-contained copy).

Registers a hook on the target layer (default: unet.mid_block) that
adds a tensor ``v`` to the layer's output, producing the *conditioned*
h-space activation.  The hook also captures the raw (pre-patch)
activation for downstream analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class HSpacePatcher:
    """Manages a single forward hook that patches a layer output with v."""

    def __init__(self, v: torch.Tensor, scale: float = 1.0) -> None:
        """
        Args:
            v:     Patch tensor.  Shape must broadcast with the target
                   layer's output (typically [1, C, H, W]).
            scale: Scalar multiplier applied to v before addition.
        """
        self.v = v
        self.scale = scale
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._captured_h: Optional[torch.Tensor] = None

    # ── hook callback ───────────────────────────────────────────────────

    def _hook_fn(
        self,
        module: nn.Module,
        input: Union[torch.Tensor, Tuple],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the clean activation, then return output + scale * v."""
        self._captured_h = output.detach().clone()
        return output + self.scale * self.v

    # ── public API ──────────────────────────────────────────────────────

    def register(self, layer: nn.Module) -> None:
        """Attach the hook to *layer*.  Removes any prior hook first."""
        self.remove()
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        """Detach the hook (safe to call multiple times)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def captured_h(self) -> Optional[torch.Tensor]:
        """The *unpatched* mid-block activation from the last hooked pass."""
        return self._captured_h

    # ── context-manager shorthand ───────────────────────────────────────

    def __call__(self, layer: nn.Module):
        """Usage:  with patcher(unet.mid_block): unet(...)"""
        return _PatchContext(self, layer)


class _PatchContext:
    """Context manager that registers the hook on enter and removes on exit."""

    def __init__(self, patcher: HSpacePatcher, layer: nn.Module) -> None:
        self._patcher = patcher
        self._layer = layer

    def __enter__(self) -> HSpacePatcher:
        self._patcher.register(self._layer)
        return self._patcher

    def __exit__(self, *exc) -> None:
        self._patcher.remove()
