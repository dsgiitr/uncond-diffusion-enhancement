#!/usr/bin/env python3
"""
run_semantic_concept_experiment.py
──────────────────────────────────
End-to-end multi-timestep separability analysis for a semantic concept
(CelebA-HQ attribute).

Pipeline
--------
1. Load the local CelebA-HQ dataset (saved with `save_to_disk`).
2. Split samples into positive / negative groups by the requested concept
   (attribute column).
3. For EVERY requested timestep, add noise and run the UNet — capturing
   mid-block (bottleneck) activations.
4. Balance the two classes per timestep.
5. Save:
   - raw & balanced per-timestep tensors
   - consolidated `.pt` file (compatible with evaluator scripts)
6. Launch Linear Probe, SVM margin, and LDA eigenvalue evaluators.
7. Collect all metrics into a summary CSV.

GPU optimisations
-----------------
- Multiple timesteps are batched into a SINGLE UNet forward pass
  (controlled by --ts_chunk_size; 0 = all at once).
- torch.amp.autocast(\"cuda\") for fp16 inference.
- Persistent DataLoader workers to reduce process-spawning overhead.
- Pre-allocated CPU accumulators to avoid repeated concatenation.
- Non-blocking CUDA transfers + pinned memory.

Example
-------
    python "semantic vector extraction/run_semantic_concept_experiment.py" \\
        --concept Smiling \\
        --dataset_dir celeba_hq_dataset \\
        --max_samples 500 \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_name(name: str) -> str:
    """Convert an attribute name to a filesystem-safe lower-case slug."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()


def parse_timesteps(value: str) -> List[int]:
    if not value.strip():
        return [980, 880, 780, 680, 580, 480, 380, 280, 180, 80]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_scheduler(model_id: str, scheduler_type: str):
    if scheduler_type == "ddpm":
        return DDPMScheduler.from_pretrained(model_id)
    return DDIMScheduler.from_pretrained(model_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class ConceptImageDataset(Dataset):
    """Returns (image_tensor, binary_label) for a CelebA-HQ attribute column."""

    def __init__(self, hf_dataset, concept: str):
        self.ds = hf_dataset
        self.concept = concept

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        label = 1 if int(row[self.concept]) == 1 else 0
        return x, label


# ═══════════════════════════════════════════════════════════════════════════════
#  Mid-block hook
# ═══════════════════════════════════════════════════════════════════════════════

class MidBlockHook:
    """Persistent forward hook that captures unet.mid_block output."""

    def __init__(self, unet: UNet2DModel):
        self.h: Optional[torch.Tensor] = None
        self._handle = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, out):
        self.h = out.detach()

    def remove(self):
        self._handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
#  Core multi-timestep extraction  (GPU-optimised)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_multi_timestep(
    dataset,
    concept: str,
    model_id: str,
    scheduler_type: str,
    num_steps: int,
    timesteps: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    save_dtype: str,
    ts_chunk_size: int = 0,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract h-space activations for positive and negative samples at every
    requested timestep, using an optimised batched strategy.

    Returns
    -------
    {timestep_value: {"positive": Tensor, "negative": Tensor}}
    """
    # ── Model & scheduler ───────────────────────────────────────────────────
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    scheduler = get_scheduler(model_id, scheduler_type)
    scheduler.set_timesteps(num_steps, device=device)
    scheduler_ts = {int(t.item()) for t in scheduler.timesteps}

    for ts in timesteps:
        if ts not in scheduler_ts:
            raise ValueError(
                f"timestep {ts} not in scheduler timesteps. "
                f"Available: {sorted(scheduler_ts, reverse=True)}"
            )

    T = len(timesteps)
    ts_chunk = ts_chunk_size if ts_chunk_size > 0 else T

    # ── Dataset counting ────────────────────────────────────────────────────
    labels = dataset[concept]
    pos_count = sum(1 for x in labels if int(x) == 1)
    neg_count = len(labels) - pos_count

    ds = ConceptImageDataset(dataset, concept)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    dtype = torch.float16 if save_dtype == "float16" else torch.float32
    hook = MidBlockHook(unet)
    use_amp = device.type == "cuda"

    # Accumulators — list-of-tensors approach (concat once at end)
    accum: Dict[int, Dict[str, list]] = {
        ts: {"positive": [], "negative": []} for ts in timesteps
    }

    g = torch.Generator(device=device).manual_seed(seed)

    print(f"\n{'─' * 70}")
    print(f"  Extraction: {T} timesteps on {device}  (ts_chunk={ts_chunk})")
    print(f"  Batch size={batch_size}  |  Effective UNet batch={batch_size * ts_chunk}")
    print(f"  Class counts: positive={pos_count}  negative={neg_count}")
    print(f"{'─' * 70}")

    t0 = time.time()

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Extracting", leave=True), start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.shape[0]

        # Masks (computed once per batch, kept on CPU for indexing)
        pos_mask = (y == 1).cpu()
        neg_mask = ~pos_mask
        has_pos = pos_mask.any().item()
        has_neg = neg_mask.any().item()

        # ── Process timesteps in chunks ─────────────────────────────────────
        for chunk_start in range(0, T, ts_chunk):
            chunk_ts = timesteps[chunk_start : chunk_start + ts_chunk]
            Tc = len(chunk_ts)

            # Repeat images for each timestep: [B*Tc, C, H, W]
            x_expanded = x.repeat(Tc, 1, 1, 1)

            # Build per-sample timestep vector
            t_all = torch.cat([
                torch.full((B,), ts, device=device, dtype=torch.long)
                for ts in chunk_ts
            ])

            # Single noise generation call
            eps_all = torch.randn(x_expanded.shape, generator=g, device=device)

            # Add noise
            x_t = scheduler.add_noise(x_expanded, eps_all, t_all)

            # Handle ts==0 edge case
            for i, ts in enumerate(chunk_ts):
                if ts == 0:
                    x_t[i * B : (i + 1) * B] = x

            del x_expanded, eps_all

            # ── Single batched UNet forward pass ────────────────────────────
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
                    f"Try --ts_chunk_size {max(1, ts_chunk // 2)} "
                    f"or reduce --batch_size."
                )

            del x_t, t_all

            # Hook captured [B*Tc, C, H, W] — split & move to CPU
            h_all = hook.h.to(dtype=dtype).cpu()

            for i, ts in enumerate(chunk_ts):
                h_ts = h_all[i * B : (i + 1) * B]
                if has_pos:
                    accum[ts]["positive"].append(h_ts[pos_mask])
                if has_neg:
                    accum[ts]["negative"].append(h_ts[neg_mask])

            del h_all

        if batch_idx % 20 == 0 or batch_idx == len(loader):
            elapsed = time.time() - t0
            print(
                f"  batch {batch_idx:4d}/{len(loader)} | "
                f"{elapsed:.1f}s elapsed"
            )

    hook.remove()
    del unet
    torch.cuda.empty_cache()

    # ── Concatenate accumulators ────────────────────────────────────────────
    output: Dict[int, Dict[str, torch.Tensor]] = {}
    for ts in timesteps:
        output[ts] = {
            "positive": torch.cat(accum[ts]["positive"], dim=0),
            "negative": torch.cat(accum[ts]["negative"], dim=0),
        }

    total_time = time.time() - t0
    print(f"\n  Extraction complete in {total_time:.1f}s")
    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Balancing & saving
# ═══════════════════════════════════════════════════════════════════════════════

def balance_activations(
    acts_by_ts: Dict[int, Dict[str, torch.Tensor]], seed: int
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Down-sample majority class to min(pos, neg) per timestep."""
    rng = torch.Generator().manual_seed(seed)
    balanced = {}
    for ts, pm in acts_by_ts.items():
        pos = pm["positive"]
        neg = pm["negative"]
        n = min(len(pos), len(neg))
        if n < 2:
            raise ValueError(f"timestep {ts}: fewer than 2 samples in one class")
        p_idx = torch.randperm(len(pos), generator=rng)[:n]
        m_idx = torch.randperm(len(neg), generator=rng)[:n]
        balanced[ts] = {
            "positive": pos[p_idx].contiguous(),
            "negative": neg[m_idx].contiguous(),
        }
    return balanced


def save_per_timestep_tensors(
    base_dir: Path,
    acts_by_ts: Dict[int, Dict[str, torch.Tensor]],
    concept_slug: str,
):
    """Save individual .pt files per timestep."""
    for ts, pm in acts_by_ts.items():
        ts_dir = base_dir / f"t{ts:04d}"
        ts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pm["positive"], ts_dir / f"{concept_slug}_positive.pt")
        torch.save(pm["negative"], ts_dir / f"{concept_slug}_negative.pt")


def print_class_counts(
    acts_by_ts: Dict[int, Dict[str, torch.Tensor]], title: str
):
    print(f"\n{title}")
    print("-" * len(title))
    for ts in sorted(acts_by_ts.keys(), reverse=True):
        n_pos = int(acts_by_ts[ts]["positive"].shape[0])
        n_neg = int(acts_by_ts[ts]["negative"].shape[0])
        print(f"  t={ts:04d} | positive={n_pos} | negative={n_neg}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval_scripts(
    pt_file: Path,
    eval_dir: Path,
    timesteps: List[int],
    epochs: int,
    lr: float,
    split_ratio: float,
    seed: int,
):
    """Launch the three evaluation scripts as sub-processes."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts_arg = ",".join(str(t) for t in timesteps)
    script_root = Path(__file__).resolve().parent

    scripts = [
        (
            "eval_linear_probe.py",
            [
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--split_ratio", str(split_ratio),
                "--seed", str(seed),
            ],
        ),
        ("eval_svm.py", []),
        ("eval_lda.py", []),
    ]

    for script_name, extra_args in tqdm(scripts, desc="Evaluators", leave=True):
        cmd = [
            sys.executable,
            str(script_root / script_name),
            "--file", str(pt_file),
            "--output_dir", str(eval_dir),
            "--timesteps", ts_arg,
            *extra_args,
        ]
        print(f"  → {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def collect_summary_csv(eval_dir: Path, concept_name: str):
    """Merge log files from all three evaluators into a single CSV."""
    patterns = {
        "linear_probe_acc": (
            eval_dir / f"{concept_name}_linear_probe_log.txt",
            r"TS:\s*(\d+)\s*\|.*Test Acc:\s*([0-9.]+)%",
        ),
        "svm_margin": (
            eval_dir / f"{concept_name}_svm_margin_log.txt",
            r"TS:\s*(\d+)\s*\|.*Geometric Margin:\s*([0-9.]+)",
        ),
        "lda_lambda_max": (
            eval_dir / f"{concept_name}_lda_eigenvalue_log.txt",
            r"TS:\s*(\d+)\s*\|.*Lambda Max:\s*([0-9.]+)",
        ),
    }

    rows: Dict[int, Dict[str, float]] = {}
    for metric_name, (path, pattern) in patterns.items():
        if not path.exists():
            continue
        text = path.read_text()
        for ts_str, val_str in re.findall(pattern, text):
            ts = int(ts_str)
            rows.setdefault(ts, {})[metric_name] = float(val_str)

    out_csv = eval_dir / f"{concept_name}_summary_metrics.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "linear_probe_acc", "svm_margin", "lda_lambda_max"])
        for ts in sorted(rows.keys(), reverse=True):
            writer.writerow([
                ts,
                rows[ts].get("linear_probe_acc", ""),
                rows[ts].get("svm_margin", ""),
                rows[ts].get("lda_lambda_max", ""),
            ])

    print(f"\n  Saved summary CSV → {out_csv}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-timestep semantic concept separability analysis.\n"
            "Extracts bottleneck activations for a CelebA-HQ attribute "
            "and runs Linear Probe / SVM / LDA evaluators."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--concept", type=str, required=True,
                   help="CelebA-HQ attribute column (e.g. Smiling, Male, Young, ...)")
    p.add_argument("--dataset_dir", type=str, default="celeba_hq_dataset",
                   help="Path to local dataset (saved with save_to_disk)")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256",
                   help="HuggingFace model id for the UNet")
    p.add_argument("--scheduler_type", type=str, choices=["ddim", "ddpm"], default="ddim",
                   help="Noise scheduler type")
    p.add_argument("--num_steps", type=int, default=50,
                   help="Total scheduler inference steps")
    p.add_argument("--timesteps", type=str, default="980,880,780,680,580,480,380,280,180,80",
                   help="Comma-separated timestep values to analyse")
    p.add_argument("--batch_size", type=int, default=16,
                   help="DataLoader batch size")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Use first N samples (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16",
                   help="Dtype for saved activation tensors")
    p.add_argument("--device", type=str, default="",
                   help="Force device (auto if empty)")
    p.add_argument("--output_root", type=str,
                   default="semantic vector extraction/outputs",
                   help="Root directory for all outputs")
    # Eval hyper-parameters
    p.add_argument("--epochs", type=int, default=150,
                   help="Epochs for Linear Probe training")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for Linear Probe")
    p.add_argument("--split_ratio", type=float, default=0.8,
                   help="Train / test split ratio for Linear Probe")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip evaluation step (extraction only)")
    p.add_argument("--ts_chunk_size", type=int, default=0,
                   help="Timesteps per UNet call (0 = all). Reduce if OOM.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    timesteps = parse_timesteps(args.timesteps)
    device = resolve_device(args.device)

    # ── Load dataset ────────────────────────────────────────────────────────
    ds = load_from_disk(args.dataset_dir)
    if args.concept not in ds.column_names:
        raise ValueError(
            f"Concept '{args.concept}' not found. "
            f"Available columns: {ds.column_names}"
        )
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    concept_slug = sanitize_name(args.concept)

    # ── Output structure ────────────────────────────────────────────────────
    base_dir = Path(args.output_root) / concept_slug
    raw_dir = base_dir / "tensors" / "raw"
    balanced_dir = base_dir / "tensors" / "balanced"
    eval_dir = base_dir / "analysis"
    data_dir = base_dir / "data"
    for d in (raw_dir, balanced_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Semantic Concept Separability Experiment")
    print("=" * 70)
    print(f"  Concept        : {args.concept}  (slug: {concept_slug})")
    print(f"  Dataset        : {args.dataset_dir}  ({len(ds)} samples)")
    print(f"  Model          : {args.model_id}")
    print(f"  Scheduler      : {args.scheduler_type} ({args.num_steps} steps)")
    print(f"  Timesteps ({len(timesteps)}): {timesteps}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Device         : {device}")
    print(f"  Output root    : {base_dir}")
    print("=" * 70)

    # ── 1. Extract ──────────────────────────────────────────────────────────
    raw_acts = extract_multi_timestep(
        dataset=ds,
        concept=args.concept,
        model_id=args.model_id,
        scheduler_type=args.scheduler_type,
        num_steps=args.num_steps,
        timesteps=timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        save_dtype=args.save_dtype,
        ts_chunk_size=args.ts_chunk_size,
    )
    print_class_counts(raw_acts, "Raw extracted class counts")
    save_per_timestep_tensors(raw_dir, raw_acts, concept_slug)

    # ── 2. Balance ──────────────────────────────────────────────────────────
    balanced_acts = balance_activations(raw_acts, seed=args.seed)
    print_class_counts(balanced_acts, "Balanced class counts (used for eval)")
    save_per_timestep_tensors(balanced_dir, balanced_acts, concept_slug)

    # ── 3. Save consolidated .pt ────────────────────────────────────────────
    # NOTE: The evaluator scripts expect keys "plus" / "minus" to match the
    # original pipeline convention.  We store them under those names.
    payload = {
        "config": {
            "concept": args.concept,
            "model_id": args.model_id,
            "scheduler_type": args.scheduler_type,
            "num_steps": args.num_steps,
            "timesteps": timesteps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "save_dtype": args.save_dtype,
        },
        "activations": {
            ts: {
                "plus": balanced_acts[ts]["positive"],
                "minus": balanced_acts[ts]["negative"],
            }
            for ts in timesteps
        },
    }
    pt_file = data_dir / f"{concept_slug}.pt"
    torch.save(payload, pt_file)
    print(f"\n  Saved consolidated activations → {pt_file}")

    # ── 4. Evaluate ─────────────────────────────────────────────────────────
    if not args.skip_eval:
        run_eval_scripts(
            pt_file=pt_file,
            eval_dir=eval_dir,
            timesteps=timesteps,
            epochs=args.epochs,
            lr=args.lr,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
        collect_summary_csv(eval_dir, concept_slug)

    print("\n✓ Experiment complete.\n")


if __name__ == "__main__":
    main()
