"""
Visualisation utilities for CKA analysis results.
"""

from __future__ import annotations

from typing import Dict, Optional


def plot_cka_bar(
    cka_scores: Dict[str, float],
    ref_layer: str = "mid_block",
    timestep: int = 20,
    pool_spatial: Optional[int] = 1,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Bar chart of CKA scores across encoder layers.

    Args:
        cka_scores:   ``{layer_name: cka_value, ...}``
        ref_layer:    Name of the reference layer (for the title).
        timestep:     Timestep used (for the title).
        pool_spatial: Pooling config (for the subtitle).
        save_path:    If given, save the figure to this path.
        figsize:      Matplotlib figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(cka_scores.keys())
    values = [cka_scores[n] for n in names]

    # Prettier labels: "down_block_0" → "Encoder 0"
    labels = [n.replace("down_block_", "Enc ") for n in names]

    fig, ax = plt.subplots(figsize=figsize)

    # Gradient colour map — deeper layers get warmer colours
    cmap = plt.cm.viridis
    colours = [cmap(v / max(max(values), 1e-6)) for v in values]

    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    pool_label = f"pool={pool_spatial}×{pool_spatial}" if pool_spatial else "no pooling"
    ax.set_title(
        f"CKA( encoder layer , {ref_layer} )   —   t = {timestep}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Encoder Block", fontsize=11)
    ax.set_ylabel("CKA Score", fontsize=11)
    ax.set_ylim(0, min(max(values) * 1.2, 1.05) if values else 1.0)
    ax.text(
        0.98, 0.95, pool_label,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, fontstyle="italic", color="gray",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot → {save_path}")

    plt.close(fig)
    return fig


def plot_cka_heatmap(
    cka_matrix: Dict[int, Dict[str, float]],
    ref_layer: str = "mid_block",
    pool_spatial: Optional[int] = 1,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Heatmap of CKA scores across timesteps × encoder layers.

    Args:
        cka_matrix: ``{timestep: {layer_name: cka_value, ...}, ...}``
        ref_layer:  Name of the reference layer (for the title).
        pool_spatial: Pooling config (for the subtitle).
        save_path:  If given, save the figure to this path.
        figsize:    Matplotlib figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    timesteps = sorted(cka_matrix.keys())
    layer_names = list(next(iter(cka_matrix.values())).keys())
    labels = [n.replace("down_block_", "Enc ") for n in layer_names]

    matrix = np.array([
        [cka_matrix[t][ln] for ln in layer_names]
        for t in timesteps
    ])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(timesteps)))
    ax.set_yticklabels([str(t) for t in timesteps], fontsize=10)

    ax.set_xlabel("Encoder Block", fontsize=11)
    ax.set_ylabel("Timestep", fontsize=11)

    pool_label = f"pool={pool_spatial}×{pool_spatial}" if pool_spatial else "no pooling"
    ax.set_title(
        f"CKA( encoder , {ref_layer} )  —  {pool_label}",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Annotate cells
    for i in range(len(timesteps)):
        for j in range(len(layer_names)):
            val = matrix[i, j]
            colour = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=colour)

    fig.colorbar(im, ax=ax, label="CKA", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved heatmap → {save_path}")

    plt.close(fig)
    return fig


def plot_cka_trajectory(
    cka_matrix: Dict[int, Dict[str, float]],
    ref_layer: str = "mid_block",
    pool_spatial: Optional[int] = 1,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """Line plot of CKA vs denoising timestep, one line per encoder layer.

    X-axis is *timestep value* (descending = more noise on the left,
    denoised on the right), showing how encoder-vs-hspace similarity
    evolves through the reverse diffusion process.

    Args:
        cka_matrix: ``{timestep: {layer_name: cka_value, ...}, ...}``
        ref_layer:  Name of the reference layer (for the title).
        pool_spatial: Pooling config (for the subtitle).
        save_path:  If given, save the figure to this path.
        figsize:    Matplotlib figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    timesteps = sorted(cka_matrix.keys(), reverse=True)  # high noise → low
    layer_names = list(next(iter(cka_matrix.values())).keys())
    labels = [n.replace("down_block_", "Enc ") for n in layer_names]

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette — one per encoder layer
    cmap = plt.cm.viridis
    n_layers = len(layer_names)
    colours = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for idx, (name, label) in enumerate(zip(layer_names, labels)):
        values = [cka_matrix[t][name] for t in timesteps]
        ax.plot(
            range(len(timesteps)), values,
            marker="o", markersize=5, linewidth=2,
            color=colours[idx], label=label,
        )

    ax.set_xticks(range(len(timesteps)))
    ax.set_xticklabels([str(t) for t in timesteps], fontsize=9, rotation=45)
    ax.set_xlabel("Timestep (high noise → denoised)", fontsize=11)
    ax.set_ylabel("CKA Score", fontsize=11)
    ax.set_ylim(0, 1.05)

    pool_label = f"pool={pool_spatial}×{pool_spatial}" if pool_spatial else "no pooling"
    ax.set_title(
        f"CKA Trajectory: encoder layers vs {ref_layer}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.text(
        0.98, 0.02, pool_label,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontstyle="italic", color="gray",
    )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved trajectory → {save_path}")

    plt.close(fig)
    return fig

