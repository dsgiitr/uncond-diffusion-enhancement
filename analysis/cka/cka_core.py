"""
GPU-optimised Centered Kernel Alignment (CKA) with mini-batch accumulation.

Two computation back-ends:

  1. **MiniBatchCKA** (cross-covariance accumulation)
     Best when feature dimension D is small (e.g. after global-avg-pool → D = C).
     Accumulates D_x × D_y cross-covariance matrices on GPU.  Memory: O(D²).

  2. **GramCKA** (Gram-matrix approach)
     Best when D is large (no pooling / spatial pooling).
     Stores centred activations on CPU, builds N × N Gram matrices at the end.
     Memory: O(N·D) on CPU + O(N²) on GPU for the final Gram computation.

Both use **pre-computed global means** for kernel centering, avoiding noisy
per-batch statistics.

Mathematical identity
─────────────────────
  HSIC(X, Y) = (1/(N−1)²) ‖X̃ᵀỸ‖²_F          (feature-space)
             = (1/(N−1)²) tr(K·L)              (sample-space)

  CKA(X, Y)  = HSIC(X,Y) / √(HSIC(X,X)·HSIC(Y,Y))

  where X̃ = X − 𝟏μ_Xᵀ  (centred with pre-computed global mean).

The (N−1)² factor cancels in the CKA ratio, so we only track the
un-normalised Frobenius norms / traces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def adaptive_pool_flatten(
    x: torch.Tensor,
    pool_spatial: Optional[int] = 1,
) -> torch.Tensor:
    """Pool spatial dimensions and flatten to ``[B, D]``.

    Args:
        x:            ``[B, C, H, W]`` activation tensor.
        pool_spatial:  Target spatial size per side.
                       ``1``  → global average pool  (D = C).
                       ``8``  → pool to 8 × 8        (D = C·64).
                       ``None`` → no pooling          (D = C·H·W).

    Returns:
        ``[B, D]`` flattened tensor.
    """
    B = x.shape[0]
    if pool_spatial is not None:
        if x.shape[2] != pool_spatial or x.shape[3] != pool_spatial:
            x = F.adaptive_avg_pool2d(x, (pool_spatial, pool_spatial))
    return x.reshape(B, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cross-covariance CKA  (small D — after pooling)
# ═══════════════════════════════════════════════════════════════════════════════


class MiniBatchCKA:
    r"""GPU-optimised mini-batch CKA via cross-covariance accumulation.

    Accumulates:
        C_{XY} = ∑_b  X̃_bᵀ Ỹ_b        (D_x × D_y)
        C_{XX} = ∑_b  X̃_bᵀ X̃_b        (D_x × D_x)
        C_{YY} = ∑_b  Ỹ_bᵀ Ỹ_b        (D_y × D_y)

    These sums are **exact** (not approximations) because matrix
    multiplication distributes over vertical concatenation.

    After all batches:
        CKA = ‖C_{XY}‖²_F / √( ‖C_{XX}‖²_F · ‖C_{YY}‖²_F )

    **When to use**: feature dimension D ≤ ~2 048  (after global-avg-pool).
    For larger D the D×D matrices become prohibitive — use :class:`GramCKA`.

    Args:
        mean_x:  ``[D_x]`` pre-computed global mean (already flattened & pooled).
        mean_y:  ``[D_y]`` ditto for the reference layer.
        device:  torch device for accumulation (should be CUDA).
        use_batch_mean:  If ``True``, ignore ``mean_x / mean_y`` and centre
                         each mini-batch by its own sample mean (standard CKA).
                         Removes the need for pre-computed global means.
    """

    def __init__(
        self,
        mean_x: torch.Tensor,
        mean_y: torch.Tensor,
        device: torch.device = torch.device("cuda"),
        use_batch_mean: bool = False,
    ):
        D_x = mean_x.numel()
        D_y = mean_y.numel()

        self.mean_x = mean_x.reshape(1, D_x).to(device, dtype=torch.float32)
        self.mean_y = mean_y.reshape(1, D_y).to(device, dtype=torch.float32)
        self.device = device
        self.use_batch_mean = use_batch_mean

        # Accumulators in float64 for numerical stability
        self.cov_xy = torch.zeros(D_x, D_y, device=device, dtype=torch.float64)
        self.cov_xx = torch.zeros(D_x, D_x, device=device, dtype=torch.float64)
        self.cov_yy = torch.zeros(D_y, D_y, device=device, dtype=torch.float64)

        self.n_samples = 0

    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Accumulate one mini-batch.

        Args:
            x: ``[B, D_x]`` — flattened & pooled activations (**not** centred).
            y: ``[B, D_y]`` — ditto for the reference layer.
        """
        x_f = x.to(self.device, dtype=torch.float32)
        y_f = y.to(self.device, dtype=torch.float32)

        if self.use_batch_mean:
            x_c = (x_f - x_f.mean(dim=0, keepdim=True)).to(torch.float64)
            y_c = (y_f - y_f.mean(dim=0, keepdim=True)).to(torch.float64)
        else:
            x_c = (x_f - self.mean_x).to(torch.float64)
            y_c = (y_f - self.mean_y).to(torch.float64)

        self.cov_xy.addmm_(x_c.T, y_c)          # D_x × D_y
        self.cov_xx.addmm_(x_c.T, x_c)          # D_x × D_x
        self.cov_yy.addmm_(y_c.T, y_c)          # D_y × D_y

        self.n_samples += x.shape[0]

    # ────────────────────────────────────────────────────────────────────

    def compute(self) -> float:
        """Return the CKA score ∈ [0, 1]."""
        if self.n_samples < 2:
            raise ValueError(f"Need ≥ 2 samples, got {self.n_samples}")

        hsic_xy = torch.sum(self.cov_xy ** 2).item()
        hsic_xx = torch.sum(self.cov_xx ** 2).item()
        hsic_yy = torch.sum(self.cov_yy ** 2).item()

        denom = (hsic_xx * hsic_yy) ** 0.5
        if denom < 1e-12:
            return 0.0
        return hsic_xy / denom

    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_and_return_batch_hsic(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> dict:
        """Accumulate one mini-batch **and** return the HSIC for this batch alone.

        Computes the cross-covariance just for the current mini-batch and returns
        its squared Frobenius norm (unnormalized HSIC).

        Returns:
            dict with keys ``"hsic_xy"``, ``"hsic_xx"``, ``"hsic_yy"`` —
            each a single float scalar.
        """
        x_f = x.to(self.device, dtype=torch.float32)
        y_f = y.to(self.device, dtype=torch.float32)

        if self.use_batch_mean:
            x_c = x_f - x_f.mean(dim=0, keepdim=True)   # [B, Dx]
            y_c = y_f - y_f.mean(dim=0, keepdim=True)   # [B, Dy]
        else:
            x_c = x_f - self.mean_x   # [B, Dx]
            y_c = y_f - self.mean_y   # [B, Dy]

        # ── per-batch covariance ────────────────────────────────────────
        x_c64 = x_c.to(torch.float64)
        y_c64 = y_c.to(torch.float64)

        batch_cov_xy = torch.mm(x_c64.T, y_c64)
        batch_cov_xx = torch.mm(x_c64.T, x_c64)
        batch_cov_yy = torch.mm(y_c64.T, y_c64)

        # ── global accumulation ─────────────────────────────────────────
        self.cov_xy.add_(batch_cov_xy)
        self.cov_xx.add_(batch_cov_xx)
        self.cov_yy.add_(batch_cov_yy)
        self.n_samples += x.shape[0]

        return {
            "hsic_xy": torch.sum(batch_cov_xy ** 2).item(),
            "hsic_xx": torch.sum(batch_cov_xx ** 2).item(),
            "hsic_yy": torch.sum(batch_cov_yy ** 2).item(),
        }

    def reset(self):
        """Zero out accumulators for reuse."""
        self.cov_xy.zero_()
        self.cov_xx.zero_()
        self.cov_yy.zero_()
        self.n_samples = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Gram-matrix CKA  (large D — no / spatial pooling)
# ═══════════════════════════════════════════════════════════════════════════════


class GramCKA:
    r"""CKA via N × N Gram matrices — for high-dimensional activations.

    Stores centred activations on **CPU** across batches, then builds the
    Gram matrices K = X̃X̃ᵀ, L = ỸỸᵀ on GPU at ``compute()`` time.

        CKA = tr(K·L) / √( tr(K²) · tr(L²) )

    Gram matrices are computed block-wise to bound peak GPU memory.

    **When to use**: feature dimension D is large (no pooling, or pooling
    to 8 × 8 where D = C·64 can reach 32 768).

    Args:
        mean_x:  ``[D_x]`` pre-computed global mean (flattened).
        mean_y:  ``[D_y]`` ditto.
    """

    def __init__(
        self,
        mean_x: torch.Tensor,
        mean_y: torch.Tensor,
    ):
        self.mean_x = mean_x.reshape(1, -1).float().cpu()   # [1, D_x]
        self.mean_y = mean_y.reshape(1, -1).float().cpu()   # [1, D_y]

        self._x_parts: list[torch.Tensor] = []
        self._y_parts: list[torch.Tensor] = []
        self.n_samples = 0

    # ────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Buffer one centred mini-batch on CPU.

        Args:
            x: ``[B, D_x]`` (flattened, **not** centred).
            y: ``[B, D_y]``.
        """
        x_c = (x.float().cpu() - self.mean_x)   # [B, D_x]
        y_c = (y.float().cpu() - self.mean_y)   # [B, D_y]
        self._x_parts.append(x_c)
        self._y_parts.append(y_c)
        self.n_samples += x.shape[0]

    # ────────────────────────────────────────────────────────────────────

    def _build_gram_blockwise(
        self,
        Z: torch.Tensor,
        device: torch.device,
        chunk: int = 128,
    ) -> torch.Tensor:
        """Build N × N Gram matrix ``Z Z^T`` in GPU-memory-bounded blocks.

        Loads at most ``2 × chunk × D × 4`` bytes to GPU at a time.
        """
        N = Z.shape[0]
        G = torch.zeros(N, N, dtype=torch.float64, device=device)
        for i in range(0, N, chunk):
            zi = Z[i : i + chunk].to(device, dtype=torch.float64)
            for j in range(i, N, chunk):
                zj = Z[j : j + chunk].to(device, dtype=torch.float64)
                block = zi @ zj.T
                G[i : i + chunk, j : j + chunk] = block
                if i != j:
                    G[j : j + chunk, i : i + chunk] = block.T
        return G

    # ────────────────────────────────────────────────────────────────────

    def compute(
        self,
        device: torch.device = torch.device("cuda"),
        gram_chunk: int = 128,
    ) -> float:
        """Compute CKA from stored activations.

        Args:
            device:      GPU device for Gram-matrix computation.
            gram_chunk:  Row-block size when building Gram matrices
                         (controls peak GPU memory).

        Returns:
            CKA score ∈ [0, 1].
        """
        if self.n_samples < 2:
            raise ValueError(f"Need ≥ 2 samples, got {self.n_samples}")

        X = torch.cat(self._x_parts, dim=0)  # [N, D_x]  (CPU)
        Y = torch.cat(self._y_parts, dim=0)  # [N, D_y]  (CPU)

        K = self._build_gram_blockwise(X, device, gram_chunk)  # N × N
        L = self._build_gram_blockwise(Y, device, gram_chunk)  # N × N

        # CKA = tr(K·L) / √(tr(K²)·tr(L²))
        # tr(K·L) = ∑_{ij} K_{ij} L_{ij}   (element-wise product then sum)
        hsic_xy = torch.sum(K * L).item()
        hsic_xx = torch.sum(K * K).item()
        hsic_yy = torch.sum(L * L).item()

        denom = (hsic_xx * hsic_yy) ** 0.5
        if denom < 1e-12:
            return 0.0
        return hsic_xy / denom

    def reset(self):
        """Release stored activations."""
        self._x_parts.clear()
        self._y_parts.clear()
        self.n_samples = 0
