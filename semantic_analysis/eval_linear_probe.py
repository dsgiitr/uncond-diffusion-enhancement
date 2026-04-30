#!/usr/bin/env python3
"""
eval_linear_probe.py
────────────────────
Trains a binary linear classifier (logistic regression via nn.Linear) on
flattened h-space activations for each timestep and reports test accuracy.

Produces:
  - <concept>_linear_probe_log.txt
  - <concept>_linear_probe_plot.png

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def parse_args():
    p = argparse.ArgumentParser("Linear Probe Evaluator")
    p.add_argument("--file", type=str, required=True, help="Path to .pt file")
    p.add_argument("--epochs", type=int, default=150,
                   help="Training epochs (full-batch GD)")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--split_ratio", type=float, default=0.8,
                   help="Train fraction (image-index safe)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--timesteps", type=str, default="",
                   help="Comma-separated timestep filter (empty = all)")
    return p.parse_args()


def parse_ts_filter(value: str):
    if not value or not value.strip():
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def train_probe(
    plus: torch.Tensor,
    minus: torch.Tensor,
    device: torch.device,
    split_ratio: float,
    epochs: int,
    lr: float,
) -> float:
    """Train a logistic regressor on GPU, return test accuracy."""
    n = min(len(plus), len(minus))
    if n < 2:
        raise ValueError("Need ≥ 2 samples per class.")

    # Balance
    plus = plus[:n]
    minus = minus[:n]

    # Flatten [N, C, H, W] → [N, D]
    p_flat = plus.flatten(1).to(device, dtype=torch.float32)
    m_flat = minus.flatten(1).to(device, dtype=torch.float32)
    D = p_flat.shape[1]

    # Image-index safe split
    idx = torch.randperm(n, device=device)
    n_train = int(n * split_ratio)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    if n_train == 0 or len(test_idx) == 0:
        raise ValueError("Not enough samples for train/test split.")

    X_train = torch.cat([p_flat[train_idx], m_flat[train_idx]])
    y_train = torch.cat([
        torch.ones(n_train, device=device),
        torch.zeros(n_train, device=device),
    ])
    X_test = torch.cat([p_flat[test_idx], m_flat[test_idx]])
    y_test = torch.cat([
        torch.ones(len(test_idx), device=device),
        torch.zeros(len(test_idx), device=device),
    ])

    # Shuffle train set
    shuf = torch.randperm(len(y_train), device=device)
    X_train, y_train = X_train[shuf], y_train[shuf]

    # Train
    model = nn.Linear(D, 1).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X_train).squeeze(), y_train)
        loss.backward()
        opt.step()

    # Test
    model.eval()
    with torch.no_grad():
        preds = (model(X_test).squeeze() > 0.0).float()
        acc = (preds == y_test).float().mean().item()
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ts_filter = parse_ts_filter(args.timesteps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_path = Path(args.file)
    concept = pt_path.stem
    out_dir = Path(args.output_dir) if args.output_dir else (pt_path.parent.parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"{concept}_linear_probe_log.txt"
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    if not pt_path.exists():
        logger.error(f"[!] File not found: {pt_path}")
        return

    logger.info(f"{'=' * 60}")
    logger.info(f" Linear Probe Analysis")
    logger.info(f"{'=' * 60}")
    logger.info(f" Concept     : {concept}")
    logger.info(f" Device      : {device}")
    logger.info(f" Epochs      : {args.epochs}  LR={args.lr}")
    logger.info(f" Split       : {args.split_ratio}")
    logger.info("-" * 61)

    data = torch.load(pt_path, map_location="cpu")
    activations = data["activations"]
    timesteps = sorted(activations.keys(), reverse=True)
    if ts_filter is not None:
        timesteps = [t for t in timesteps if t in ts_filter]
    if not timesteps:
        logger.error("[!] No matching timesteps.")
        return

    accs = []
    for ts in timesteps:
        acc = train_probe(
            activations[ts]["plus"], activations[ts]["minus"],
            device, args.split_ratio, args.epochs, args.lr,
        )
        accs.append(acc * 100.0)
        shape = tuple(activations[ts]["plus"].shape)
        logger.info(
            f" TS: {ts:04d} | Shape: {shape} → "
            f"Flat: {shape[1]*shape[2]*shape[3]} | Test Acc: {acc*100:.2f}%"
        )

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(timesteps, accs, marker="o", color="#2563eb", linewidth=2, markersize=7)
    plt.xlim(max(timesteps) + 50, min(timesteps) - 50)
    plt.title(f"Linear Probe Separability — {concept}", fontsize=14, pad=12)
    plt.xlabel("Diffusion Timestep", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = out_dir / f"{concept}_linear_probe_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("-" * 61)
    logger.info(f" Saved plot → {plot_path}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
