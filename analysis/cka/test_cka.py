#!/usr/bin/env python3
"""
Quick sanity checks for the CKA engine.

Tests:
  1. CKA(X, X) ≈ 1.0
  2. Mini-batch CKA == full-batch CKA  (exact, not approximate)
  3. CKA(X, random) << CKA(X, X)
  4. GramCKA matches MiniBatchCKA on the same data
"""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from destructive_interference.cka_core import MiniBatchCKA, GramCKA


def test_self_alignment():
    """CKA(X, X) should be ≈ 1.0."""
    N, D = 200, 64
    X = torch.randn(N, D)
    mean = X.mean(dim=0)

    cka = MiniBatchCKA(mean, mean, device=torch.device("cpu"))
    # Feed in 4 mini-batches of 50
    for i in range(0, N, 50):
        cka.update(X[i:i+50], X[i:i+50])

    score = cka.compute()
    print(f"  CKA(X, X)         = {score:.8f}  (expect ≈ 1.0)")
    assert abs(score - 1.0) < 1e-5, f"Self-alignment failed: {score}"


def test_minibatch_equals_fullbatch():
    """Mini-batch accumulation should be numerically exact."""
    N, Dx, Dy = 200, 32, 48
    X = torch.randn(N, Dx)
    Y = torch.randn(N, Dy)
    mean_x = X.mean(dim=0)
    mean_y = Y.mean(dim=0)

    # Full-batch
    full = MiniBatchCKA(mean_x, mean_y, device=torch.device("cpu"))
    full.update(X, Y)
    score_full = full.compute()

    # Mini-batch (batch_size = 17 → unequal splits)
    mini = MiniBatchCKA(mean_x, mean_y, device=torch.device("cpu"))
    bs = 17
    for i in range(0, N, bs):
        mini.update(X[i:i+bs], Y[i:i+bs])
    score_mini = mini.compute()

    print(f"  Full-batch CKA    = {score_full:.8f}")
    print(f"  Mini-batch CKA    = {score_mini:.8f}")
    diff = abs(score_full - score_mini)
    print(f"  Difference        = {diff:.2e}  (expect < 1e-10)")
    assert diff < 1e-10, f"Mini vs full mismatch: {diff}"


def test_low_cka_for_random():
    """CKA(X, random noise) should be much lower than CKA(X, X)."""
    N, D = 200, 64
    X = torch.randn(N, D)
    Z = torch.randn(N, D)  # independent random
    mean_x = X.mean(dim=0)
    mean_z = Z.mean(dim=0)

    cka = MiniBatchCKA(mean_x, mean_z, device=torch.device("cpu"))
    cka.update(X, Z)
    score = cka.compute()
    print(f"  CKA(X, random)    = {score:.8f}  (expect << 1.0)")
    assert score < 0.3, f"Random CKA too high: {score}"


def test_gram_matches_crosscov():
    """GramCKA and MiniBatchCKA should give the same result."""
    N, Dx, Dy = 100, 32, 48
    X = torch.randn(N, Dx)
    Y = torch.randn(N, Dy)
    mean_x = X.mean(dim=0)
    mean_y = Y.mean(dim=0)

    # Cross-covariance approach
    cc = MiniBatchCKA(mean_x, mean_y, device=torch.device("cpu"))
    bs = 20
    for i in range(0, N, bs):
        cc.update(X[i:i+bs], Y[i:i+bs])
    score_cc = cc.compute()

    # Gram approach
    gr = GramCKA(mean_x, mean_y)
    for i in range(0, N, bs):
        gr.update(X[i:i+bs], Y[i:i+bs])
    score_gr = gr.compute(device=torch.device("cpu"))

    print(f"  CrossCov CKA      = {score_cc:.8f}")
    print(f"  Gram CKA          = {score_gr:.8f}")
    diff = abs(score_cc - score_gr)
    print(f"  Difference        = {diff:.2e}  (expect < 1e-6)")
    assert diff < 1e-6, f"Gram vs CrossCov mismatch: {diff}"


def test_update_and_return_batch_hsic():
    """Ensure update_and_return_batch_hsic works and returns scalars."""
    N, Dx, Dy = 100, 32, 48
    X = torch.randn(N, Dx)
    Y = torch.randn(N, Dy)
    mean_x = X.mean(dim=0)
    mean_y = Y.mean(dim=0)

    cka = MiniBatchCKA(mean_x, mean_y, device=torch.device("cpu"))
    hsic = cka.update_and_return_batch_hsic(X[:20], Y[:20])
    
    print(f"  Got per-batch HSIC types: {[type(v) for v in hsic.values()]}")
    assert isinstance(hsic["hsic_xy"], float), "Expected float"
    assert cka.n_samples == 20, "Samples not accumulated"

    score = cka.compute()
    print(f"  CKA after batch HSIC = {score:.8f}")
    assert 0 < score <= 1.0


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CKA Engine — Sanity Checks")
    print("=" * 60 + "\n")

    test_self_alignment()
    print()
    test_minibatch_equals_fullbatch()
    print()
    test_low_cka_for_random()
    print()
    test_gram_matches_crosscov()
    print()
    test_update_and_return_batch_hsic()

    print("\n" + "=" * 60)
    print("  All tests passed ✓")
    print("=" * 60 + "\n")
