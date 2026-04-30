#!/usr/bin/env python3
"""
optimal_timestep_analysis.py
────────────────────────────
A unified script for optimal timestep analysis using transformation-based 
concept vectors. It combines:
  1. H-space activation extraction (using + and - transforms).
  2. Integrated Linear Probe (GPU), SVM Margin, and LDA Eigenvalue evaluators.
  3. Structured output storage (.pt) and automated plotting.

Usage example:
    python concept_extraction_pipeline/time-steps-evals/optimal_timestep_analysis.py \
        --concept sharp_vs_blur \
        --dataset_source local \
        --dataset_path celeba_hq_dataset \
        --num_samples 5000 \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# Import transforms
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from transformations import transform_sharp_blur
from transformations import transform_high_low_contrast
from transformations import transform_gray_oversat
from dataset_utils import load_image_dataset_for_profile, preprocess_pil_for_profile


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()


def parse_timesteps(value: str) -> List[int]:
    if not value.strip():
        # Default multiples of 50
        return list(range(0, 1000, 50))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


class ConceptPairDataset(Dataset):
    """
    Wraps a HF/local dataset. 
    Applies the specified plus/minus transforms to the underlying RGB image.
    """
    def __init__(self, image_dataset, plus_tx, minus_tx, dataset_profile: str, image_size: int):
        self.image_dataset = image_dataset
        self.plus_tx = plus_tx
        self.minus_tx = minus_tx
        self.dataset_profile = dataset_profile
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> dict:
        pil_img = self.image_dataset[idx]
        pil_img = preprocess_pil_for_profile(
            pil_img,
            image_size=self.image_size,
            dataset_profile=self.dataset_profile,
        )

        return {
            "plus": self.plus_tx(pil_img),
            "minus": self.minus_tx(pil_img),
        }


class MidBlockHook:
    def __init__(self, unet: UNet2DModel):
        self.h = None
        self._handle = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, out):
        self.h = out.detach()

    def remove(self):
        self._handle.remove()


def extract_concept_activations(
    dataset,
    model_id: str,
    scheduler_type: str,
    timesteps: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    save_dtype: str,
    ts_chunk_size: int = 0,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extracts plus/minus activations. Returns a dict mapping timestep -> {"plus": Tensor, "minus": Tensor}
    """
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler.from_pretrained(model_id)
    else:
        scheduler = DDIMScheduler.from_pretrained(model_id)
    
    # Pre-set scheduler explicitly to 1000 steps to permit valid arbitrary ts testing
    scheduler.set_timesteps(1000, device=device)

    T = len(timesteps)
    ts_chunk = ts_chunk_size if ts_chunk_size > 0 else T

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    dtype = torch.float16 if save_dtype == "float16" else torch.float32
    hook = MidBlockHook(unet)
    use_amp = (device.type == "cuda" and save_dtype == "float16")

    accum: Dict[int, Dict[str, list]] = {ts: {"plus": [], "minus": []} for ts in timesteps}
    g = torch.Generator(device=device).manual_seed(seed)

    print(f"\nExtracting {T} timesteps on {device} (ts_chunk={ts_chunk})...")
    print(f"Batch size={batch_size}, effective UNet batch={batch_size * 2 * ts_chunk} (due to plus/minus + chunks)")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Extraction", leave=True), start=1):
            plus_imgs = batch["plus"].to(device, non_blocking=True)
            minus_imgs = batch["minus"].to(device, non_blocking=True)
            B = plus_imgs.shape[0]

            # Concatenate plus/minus into a single [2B, 3, 256, 256] sequence
            combined_imgs = torch.cat([plus_imgs, minus_imgs], dim=0)

            for chunk_start in range(0, T, ts_chunk):
                chunk_ts = timesteps[chunk_start : chunk_start + ts_chunk]
                Tc = len(chunk_ts)

                # Broadcast across chunk [2B * Tc, 3, 256, 256]
                x_expanded = combined_imgs.repeat(Tc, 1, 1, 1)

                # Replicate timestep values [2B * Tc] 
                t_all = torch.cat([
                    torch.full((2 * B,), ts, device=device, dtype=torch.long)
                    for ts in chunk_ts
                ])

                eps_all = torch.randn(x_expanded.shape, generator=g, device=device)
                x_t = scheduler.add_noise(x_expanded, eps_all, t_all)

                # Correct pure t=0 cases
                for i, ts in enumerate(chunk_ts):
                    if ts == 0:
                        x_t[i * 2 * B : (i + 1) * 2 * B] = combined_imgs

                del x_expanded, eps_all

                try:
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            unet(x_t, t_all)
                    else:
                        unet(x_t, t_all)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"CUDA OOM with ts_chunk={ts_chunk}. "
                        f"Try --ts_chunk_size {max(1, ts_chunk // 2)} or reduce --batch_size."
                    )

                del x_t, t_all

                h_all = hook.h.to(dtype=dtype).cpu() # [2B * Tc, 512, 8, 8]

                for i, ts in enumerate(chunk_ts):
                    # Each slice represents a full 2B batch at this timestep
                    h_ts = h_all[i * 2 * B : (i + 1) * 2 * B]
                    accum[ts]["plus"].append(h_ts[:B])
                    accum[ts]["minus"].append(h_ts[B:])

                del h_all

    hook.remove()

    output = {}
    for ts in timesteps:
        output[ts] = {
            "plus": torch.cat(accum[ts]["plus"], dim=0),
            "minus": torch.cat(accum[ts]["minus"], dim=0),
        }

    return output


def evaluate_linear_probe(
    ts: int,
    plus_tensor: torch.Tensor,
    minus_tensor: torch.Tensor,
    device: torch.device,
    split_ratio: float,
    epochs: int,
    lr: float,
) -> float:
    """Trains a parallelized Linear Probe on GPU, returns testing categorical accuracy."""
    n = min(plus_tensor.shape[0], minus_tensor.shape[0])
    N = n

    p_flat = plus_tensor[:n].flatten(start_dim=1).to(device, dtype=torch.float32)
    m_flat = minus_tensor[:n].flatten(start_dim=1).to(device, dtype=torch.float32)
    feature_dim = p_flat.shape[1]

    # Image-index safe train/test split
    indices = torch.randperm(N, device=device)
    n_train = int(N * split_ratio)
    
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    X_train_plus  = p_flat[train_idx]
    X_train_minus = m_flat[train_idx]
    y_train_plus  = torch.ones((n_train,), device=device, dtype=torch.float32)
    y_train_minus = torch.zeros((n_train,), device=device, dtype=torch.float32)
    
    X_test_plus   = p_flat[test_idx]
    X_test_minus  = m_flat[test_idx]
    y_test_plus   = torch.ones((len(test_idx),), device=device, dtype=torch.float32)
    y_test_minus  = torch.zeros((len(test_idx),), device=device, dtype=torch.float32)

    X_train = torch.cat([X_train_plus, X_train_minus], dim=0)
    y_train = torch.cat([y_train_plus, y_train_minus], dim=0)
    
    X_test = torch.cat([X_test_plus, X_test_minus], dim=0)
    y_test = torch.cat([y_test_plus, y_test_minus], dim=0)

    shuf_idx = torch.randperm(len(y_train), device=device)
    X_train = X_train[shuf_idx]
    y_train = y_train[shuf_idx]

    model = nn.Linear(feature_dim, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).squeeze()
        test_preds = (test_logits > 0.0).float()
        correct = (test_preds == y_test).sum().item()
        acc = correct / len(y_test)
        
    return acc


def evaluate_svm_margin(
    plus_tensor: torch.Tensor,
    minus_tensor: torch.Tensor,
) -> float:
    """Calculates SVM Geometric Margin."""
    n = min(plus_tensor.shape[0], minus_tensor.shape[0])
    
    p_flat = plus_tensor[:n].flatten(start_dim=1).numpy()
    m_flat = minus_tensor[:n].flatten(start_dim=1).numpy()
    
    X = np.concatenate([p_flat, m_flat], axis=0)
    y = np.concatenate([np.ones(len(p_flat)), np.zeros(len(m_flat))])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    svm.fit(X_scaled, y)

    margin = 2.0 / (np.linalg.norm(svm.coef_) + 1e-8)
    return float(margin)


def evaluate_lda_eigenvalue(
    plus_tensor: torch.Tensor,
    minus_tensor: torch.Tensor,
) -> float:
    """Calculates Linear Discriminant Analysis Eigenvalue (Multivariate Fisher)."""
    n = min(plus_tensor.shape[0], minus_tensor.shape[0])
    
    p_flat = plus_tensor[:n].flatten(start_dim=1).numpy()
    m_flat = minus_tensor[:n].flatten(start_dim=1).numpy()
    
    X = np.concatenate([p_flat, m_flat], axis=0)
    y = np.concatenate([np.ones(len(p_flat)), np.zeros(len(m_flat))])
    
    # PCA to bound size and stabilize rank.
    pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(X_pca, y)
    
    lambda_max = getattr(lda, 'explained_variance_ratio_', [0.0])[0]
    return float(lambda_max)


def get_concept_transforms(concept: str, image_size: int = 256):
    """Factory mapping limited supported concept strings to transformation modules."""
    if concept == "sharp_vs_blur":
        return transform_sharp_blur.get_transforms(image_size=image_size)
    elif concept == "high_vs_low_contrast":
        return transform_high_low_contrast.get_transforms(image_size=image_size)
    elif concept == "gray_vs_oversat":
        return transform_gray_oversat.get_transforms(image_size=image_size)
    else:
        raise ValueError(
            f"Unsupported concept '{concept}'. "
            f"Please choose from: sharp_vs_blur, high_vs_low_contrast, gray_vs_oversat."
        )


def parse_args():
    p = argparse.ArgumentParser("Optimal Timestep Analysis")
    p.add_argument("--concept", type=str, required=True, 
                   help="Concept transform: sharp_vs_blur | high_vs_low_contrast | gray_vs_oversat")
    p.add_argument("--dataset_source", type=str, choices=["local", "hf"], default="local")
    p.add_argument("--dataset_path", type=str, default="celeba_hq_dataset")
    p.add_argument("--dataset_profile", type=str, choices=["celeba_hq", "lsun_church"], default="celeba_hq")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=10000, help="Random samples to pull from dataset.")
    
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--scheduler_type", type=str, choices=["ddim", "ddpm"], default="ddpm")
    p.add_argument("--timesteps", type=str, default="", help="Comma separated, e.g. 0,50,100")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ts_chunk_size", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    p.add_argument("--output_root", type=str, default="outputs/optimal-timestep-analysis")
    
    # Eval overrides
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split_ratio", type=float, default=0.8)
    return p.parse_args()


def main():
    args = parse_args()
    timesteps = parse_timesteps(args.timesteps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Base Dataset
    print(f"\nLoading dataset '{args.dataset_path}' via {args.dataset_source}...")
    dataset_dir = args.dataset_path if args.dataset_source == "local" else ""
    hf_dataset = args.dataset_path if args.dataset_source == "hf" else ""
    ds = load_image_dataset_for_profile(
        dataset_profile=args.dataset_profile,
        dataset_dir=dataset_dir,
        hf_dataset=hf_dataset,
        dataset_split=args.dataset_split,
        image_key="image",
    )

    if args.num_samples > 0 and args.num_samples < len(ds):
        # Deterministically sub-sample
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(ds), generator=rng).tolist()
        ds = Subset(ds, perm[:args.num_samples])
    print(f"Dataset has {len(ds)} images.")

    # Apply Transforms
    plus_tx, minus_tx = get_concept_transforms(args.concept, image_size=args.image_size)
    concept_ds = ConceptPairDataset(
        ds,
        plus_tx,
        minus_tx,
        dataset_profile=args.dataset_profile,
        image_size=args.image_size,
    )

    # Directories Setup
    concept_name = sanitize_name(args.concept)
    base_dir = Path(args.output_root) / concept_name
    data_dir = base_dir / "data"
    eval_dir = base_dir / "analysis"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Batch Extract Activations
    acts = extract_concept_activations(
        dataset=concept_ds,
        model_id=args.model_id,
        scheduler_type=args.scheduler_type,
        timesteps=timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        save_dtype=args.save_dtype,
        ts_chunk_size=args.ts_chunk_size,
    )

    pt_file = data_dir / f"{concept_name}_activations.pt"
    
    payload = {
        "config": vars(args),
        "activations": acts,
    }
    torch.save(payload, pt_file)
    print(f"\nSaved raw activations -> {pt_file}")

    # ── 2. Run Evaluators inline
    print("\nRunning Evaluation Metrics...")
    
    metrics: Dict[int, Dict[str, float]] = {ts: {} for ts in timesteps}
    
    for ts in sorted(timesteps, reverse=True):
        print(f"\nEvaluating t={ts}...")
        p_tensor = acts[ts]["plus"]
        m_tensor = acts[ts]["minus"]
        
        # Free memory safety
        n = min(p_tensor.shape[0], m_tensor.shape[0])
        if n < 2:
            print(f" t={ts} skipped (insufficient samples)")
            continue

        acc = evaluate_linear_probe(
            ts=ts, plus_tensor=p_tensor, minus_tensor=m_tensor, 
            device=device, split_ratio=args.split_ratio, 
            epochs=args.epochs, lr=args.lr
        )
        metrics[ts]["linear_probe_acc"] = acc * 100.0
        
        margin = evaluate_svm_margin(plus_tensor=p_tensor, minus_tensor=m_tensor)
        metrics[ts]["svm_margin"] = margin
        
        lda_val = evaluate_lda_eigenvalue(plus_tensor=p_tensor, minus_tensor=m_tensor)
        metrics[ts]["lda_lambda_max"] = lda_val
        
        print(f"  Linear Probe: {metrics[ts]['linear_probe_acc']:.2f}% | SVM Margin: {metrics[ts]['svm_margin']:.5f} | LDA Eigen: {metrics[ts]['lda_lambda_max']:.5f}")

    # Append results to .pt
    payload["metrics"] = metrics
    torch.save(payload, pt_file)

    # ── 3. Save Summary CSV & Plots
    out_csv = eval_dir / f"{concept_name}_summary_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "linear_probe_acc", "svm_margin", "lda_lambda_max"])
        for ts in sorted(timesteps, reverse=True):
            writer.writerow([
                ts,
                metrics[ts].get("linear_probe_acc", ""),
                metrics[ts].get("svm_margin", ""),
                metrics[ts].get("lda_lambda_max", ""),
            ])
    print(f"\nSaved CSV -> {out_csv}")

    # Extract axes
    plt_ts = sorted([ts for ts in timesteps if metrics[ts]])
    accs = [metrics[t]["linear_probe_acc"] for t in plt_ts]
    margins = [metrics[t]["svm_margin"] for t in plt_ts]
    ldas = [metrics[t]["lda_lambda_max"] for t in plt_ts]

    def _save_plot(y_vals, title, ylabel, filename, color):
        plt.figure(figsize=(8, 5))
        plt.plot(plt_ts, y_vals, marker="o", color=color, linewidth=2)
        plt.xlim(max(plt_ts) + 50, min(plt_ts) - 50)
        plt.title(title, fontsize=14, pad=10)
        plt.xlabel("Diffusion Timestep", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(eval_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    _save_plot(accs, f"Linear Separability: {concept_name}", "Test Accuracy (%)", f"{concept_name}_separability_plot.png", "blue")
    _save_plot(margins, f"SVM Geometric Margin Width: {concept_name}", "Margin Width (2 / ||w||)", f"{concept_name}_svm_margin_plot.png", "green")
    _save_plot(ldas, f"LDA Separability (λ_max): {concept_name}", "Top Eigenvalue (λ_max)", f"{concept_name}_lda_eigenvalue_plot.png", "purple")
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    
    def norm(col):
        c_min, c_max = min(col), max(col)
        if c_max - c_min == 0: return [0]*len(col)
        return [(v - c_min)/(c_max - c_min) for v in col]

    plt.plot(plt_ts, norm(accs), marker="o", label="Linear Probe (norm)", color="blue")
    plt.plot(plt_ts, norm(margins), marker="s", label="SVM Margin (norm)", color="green")
    plt.plot(plt_ts, norm(ldas), marker="^", label="LDA Eigenvalue (norm)", color="purple")
    
    plt.xlim(max(plt_ts) + 50, min(plt_ts) - 50)
    plt.title(f"Combined Min-Max Normalized Metrics: {concept_name}", fontsize=14, pad=10)
    plt.xlabel("Diffusion Timestep", fontsize=12)
    plt.ylabel("Normalized Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    combined_name = eval_dir / f"{concept_name}_combined_metrics_plot.png"
    plt.savefig(combined_name, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Plots -> {eval_dir}/")
    print("\nOptimal Timestep Analysis Complete! ✓")


if __name__ == "__main__":
    main()
