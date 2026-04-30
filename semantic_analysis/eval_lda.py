#!/usr/bin/env python3
"""
eval_lda.py
───────────
Evaluates linear separability of h-space activations using Fisher LDA.
The top eigenvalue (λ_max) from `explained_variance_ratio_` measures
how well the two classes separate in the LDA-projected space.

Uses PCA pre-projection to avoid the 32768×32768 dense Sw matrix,
and Ledoit-Wolf shrinkage for numerical stability.

Produces:
  - <concept>_lda_eigenvalue_log.txt
  - <concept>_lda_eigenvalue_plot.png

Expected input: a .pt file with structure:
    {"activations": {ts: {"plus": Tensor, "minus": Tensor}, ...}, ...}
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def parse_args():
    p = argparse.ArgumentParser("LDA Eigenvalue Evaluator")
    p.add_argument("--file", type=str, required=True, help="Path to .pt file")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--timesteps", type=str, default="",
                   help="Comma-separated timestep filter (empty = all)")
    return p.parse_args()


def parse_ts_filter(value: str):
    if not value or not value.strip():
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def main():
    args = parse_args()
    ts_filter = parse_ts_filter(args.timesteps)

    pt_path = Path(args.file)
    concept = pt_path.stem
    out_dir = Path(args.output_dir) if args.output_dir else (pt_path.parent.parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"{concept}_lda_eigenvalue_log.txt"
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    if not pt_path.exists():
        logger.error(f"[!] File not found: {pt_path}")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f" LDA Eigenvalue Analysis (Multivariate Fisher)")
    logger.info(f"{'=' * 60}")
    logger.info(f" Concept  : {concept}")
    logger.info(f" Data     : {pt_path}")
    logger.info("-" * 61)

    data = torch.load(pt_path, map_location="cpu")
    activations = data["activations"]
    timesteps = sorted(activations.keys(), reverse=True)
    if ts_filter is not None:
        timesteps = [t for t in timesteps if t in ts_filter]
    if not timesteps:
        logger.error("[!] No matching timesteps.")
        return

    scores = []
    for ts in timesteps:
        p_data = activations[ts]["plus"].float()
        m_data = activations[ts]["minus"].float()
        n = min(len(p_data), len(m_data))
        if n < 2:
            logger.info(f" TS: {ts:04d} | skipped (< 2 samples)")
            continue

        p_flat = p_data[:n].flatten(1).numpy()
        m_flat = m_data[:n].flatten(1).numpy()
        X = np.concatenate([p_flat, m_flat], axis=0)
        y = np.concatenate([np.ones(n), np.zeros(n)])

        # PCA down to data rank to avoid massive dense matrices
        pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]))
        X_pca = pca.fit_transform(X)

        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(X_pca, y)

        lambda_max = getattr(lda, "explained_variance_ratio_", [0.0])[0]
        scores.append(lambda_max)
        logger.info(
            f" TS: {ts:04d} | Shape: {tuple(p_data.shape)} → "
            f"Flat: {X.shape[1]} | Lambda Max: {lambda_max:.5f}"
        )

    if not scores:
        logger.error("[!] No scores computed.")
        return

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, scores, marker="o", color="#9333ea", linewidth=2, markersize=7)
    plt.xlim(max(timesteps) + 50, min(timesteps) - 50)
    plt.title(f"LDA Separability (λ_max) — {concept}", fontsize=14, pad=12)
    plt.xlabel("Diffusion Timestep", fontsize=12)
    plt.ylabel("Top Eigenvalue (λ_max)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = out_dir / f"{concept}_lda_eigenvalue_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("-" * 61)
    logger.info(f" Saved plot → {plot_path}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
