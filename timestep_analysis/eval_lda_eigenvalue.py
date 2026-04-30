#!/usr/bin/env python3
"""
eval_lda_eigenvalue.py
───────────────────────
Evaluates the linear separability of h-space concept representations 
using the top LDA eigenvalue (the correct multivariate Fisher).
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser("LDA Eigenvalue Evaluator")
    parser.add_argument("--file", type=str, required=True, help="Path to .pt file")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for plots/logs. Defaults to outputs/time-step-analysis"
    )
    parser.add_argument(
        "--timesteps", type=str, default="",
        help="Comma-separated timestep values to evaluate (default: all available)."
    )
    return parser.parse_args()


def parse_timesteps_arg(value: str):
    if not value or not value.strip():
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip()}

def main():
    args = parse_args()
    selected_timesteps = parse_timesteps_arg(args.timesteps)
    pt_path = Path(args.file)
    concept_name = pt_path.stem
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (base_dir / "outputs" / "time-step-analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = out_dir / f"{concept_name}_lda_eigenvalue_log.txt"
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ])
    logger = logging.getLogger()
    
    if not pt_path.exists():
        logger.error(f"[!] Target file '{pt_path}' does not exist.")
        return

    logger.info(f"={'=' * 60}")
    logger.info(f" LDA Eigenvalue Analysis (Multivariate Fisher)")
    logger.info(f"={'=' * 60}")
    logger.info(f" Concept     : {concept_name}")
    logger.info(f" Data file   : {pt_path}")
    logger.info("-" * 61)

    data = torch.load(pt_path, map_location="cpu")
    activations = data["activations"]
    timesteps = sorted(activations.keys(), reverse=True)
    if selected_timesteps is not None:
        timesteps = [ts for ts in timesteps if ts in selected_timesteps]
    if not timesteps:
        logger.error("[!] No matching timesteps to evaluate.")
        return
    
    scores = []
    
    for ts in timesteps:
        p_data = activations[ts]["plus"].float()
        m_data = activations[ts]["minus"].float()
        
        n = min(len(p_data), len(m_data))
        if n < 2:
            logger.info(f" TS: {ts:04d} | skipped (not enough balanced samples)")
            continue
        p_flat = p_data[:n].flatten(start_dim=1).numpy()
        m_flat = m_data[:n].flatten(start_dim=1).numpy()
        
        X = np.concatenate([p_flat, m_flat], axis=0)
        y = np.concatenate([np.ones(len(p_flat)), np.zeros(len(m_flat))])
        
        # Project down to the exact absolute data rank using PCA. 
        # This preserves 100% of the variance mathematically, but prevents the 
        # 32,768 x 32,768 dense matrix allocation which overflows bounds 
        # inside SciPy's LAPACK FORTRAN backend routines.
        pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # Using Ledoit-Wolf shrinkage to handle p >> n 
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        lda.fit(X_pca, y)
        
        lambda_max = getattr(lda, 'explained_variance_ratio_', [0.0])[0]
        
        scores.append(lambda_max)
        logger.info(f" TS: {ts:04d} | Shape: {tuple(p_data.shape)} -> Flat: {X.shape[1]} | Lambda Max: {lambda_max:>.5f}")

    # ---- Plotting ----
    if not scores:
        logger.error("[!] No scores computed.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, scores, marker="o", color="purple", linewidth=2)
    plt.xlim(max(timesteps) + 50, min(timesteps) - 50)
    plt.title(f"LDA Separability (λ_max): {concept_name}", fontsize=14, pad=10)
    plt.xlabel("Diffusion Timestep", fontsize=12)
    plt.ylabel("Top Eigenvalue (λ_max)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    out_plot_name = out_dir / f"{concept_name}_lda_eigenvalue_plot.png"
    plt.savefig(out_plot_name, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("-" * 61)
    logger.info(f" Saved plot to : {out_plot_name}")
    logger.info(f"({'=' * 60})\n")

if __name__ == "__main__":
    main()
