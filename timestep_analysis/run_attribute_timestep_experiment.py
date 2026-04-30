#!/usr/bin/env python3
"""
Run a full attribute-timestep experiment:
1) Extract + / - activations for a single attribute across selected timesteps.
2) Balance class counts per timestep by pruning to min(count_pos, count_neg).
3) Save raw and balanced tensors for every timestep.
4) Build a consolidated .pt file compatible with time-step evaluators.
5) Run linear probe, SVM margin, and LDA eigenvalue evaluators.

Example:
python concept_extraction_pipeline/time-steps-evals/run_attribute_timestep_experiment.py \
  --attribute Smiling \
  --dataset_dir celeba_hq_dataset \
  --timesteps 980,880,780,680,580,480,380,280,180,80
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()


def parse_timesteps(value: str) -> List[int]:
    if not value.strip():
        return [980, 880, 780, 680, 580, 480, 380, 280, 180, 80]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


class AttributeImageDataset(Dataset):
    def __init__(self, hf_dataset, attribute: str):
        self.ds = hf_dataset
        self.attribute = attribute

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row["image"].convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        label = 1 if int(row[self.attribute]) == 1 else 0
        return x, label


class MidBlockHook:
    def __init__(self, unet: UNet2DModel):
        self.h = None
        self._handle = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, out):
        self.h = out.detach()

    def remove(self):
        self._handle.remove()


def get_scheduler(model_id: str, scheduler_type: str):
    if scheduler_type == "ddpm":
        return DDPMScheduler.from_pretrained(model_id)
    return DDIMScheduler.from_pretrained(model_id)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_multi_timestep(
    dataset,
    attribute: str,
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
):
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    scheduler = get_scheduler(model_id, scheduler_type)
    scheduler.set_timesteps(num_steps, device=device)
    scheduler_ts = {int(t.item()) for t in scheduler.timesteps}

    for ts in timesteps:
        if ts not in scheduler_ts:
            raise ValueError(f"timestep {ts} not found in scheduler timesteps")

    T = len(timesteps)
    # How many timesteps to batch into one UNet call.
    # Default (0) = all timesteps at once → 1 UNet call per dataloader batch.
    ts_chunk = ts_chunk_size if ts_chunk_size > 0 else T

    labels = dataset[attribute]
    pos_count = sum(1 for x in labels if int(x) == 1)
    neg_count = len(labels) - pos_count

    ds = AttributeImageDataset(dataset, attribute)
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

    accum: Dict[int, Dict[str, list]] = {ts: {"plus": [], "minus": []} for ts in timesteps}
    use_amp = device.type == "cuda"

    g = torch.Generator(device=device).manual_seed(seed)

    print(f"Extracting {T} timesteps on {device} (ts_chunk={ts_chunk})...")
    print(f"Batch size={batch_size}, effective UNet batch={batch_size * ts_chunk}")
    print(f"Class counts before balancing: pos={pos_count}, neg={neg_count}")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Batches", leave=True), start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B = x.shape[0]

            # Pre-compute masks once per batch
            pos_mask_cpu = (y == 1).cpu()
            neg_mask_cpu = ~pos_mask_cpu
            has_pos = pos_mask_cpu.any().item()
            has_neg = neg_mask_cpu.any().item()

            # Process timesteps in chunks (default: all at once)
            for chunk_start in range(0, T, ts_chunk):
                chunk_ts = timesteps[chunk_start : chunk_start + ts_chunk]
                Tc = len(chunk_ts)

                # Repeat images for each timestep in this chunk: [B*Tc, C, H, W]
                x_expanded = x.repeat(Tc, 1, 1, 1)

                # Build per-sample timestep tensor: [ts0]*B ++ [ts1]*B ++ ...
                t_all = torch.cat([
                    torch.full((B,), ts, device=device, dtype=torch.long)
                    for ts in chunk_ts
                ])

                # Generate ALL noise in one randn call (1 GPU sync instead of Tc)
                eps_all = torch.randn(x_expanded.shape, generator=g, device=device)

                # Add noise — scheduler handles per-sample timesteps natively
                x_t = scheduler.add_noise(x_expanded, eps_all, t_all)

                # Handle ts==0 (no noise should be added)
                for i, ts in enumerate(chunk_ts):
                    if ts == 0:
                        x_t[i * B : (i + 1) * B] = x

                del x_expanded, eps_all

                # ---- Single UNet forward pass for the entire chunk ----
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

                # Hook captured [B*Tc, ...] — split by timestep and move to CPU
                h_all = hook.h.to(dtype=dtype).cpu()

                for i, ts in enumerate(chunk_ts):
                    h_ts = h_all[i * B : (i + 1) * B]
                    if has_pos:
                        accum[ts]["plus"].append(h_ts[pos_mask_cpu])
                    if has_neg:
                        accum[ts]["minus"].append(h_ts[neg_mask_cpu])

                del h_all

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                print(f"batch {batch_idx:4d}/{len(loader)} done")

    hook.remove()

    output = {}
    for ts in timesteps:
        output[ts] = {
            "plus": torch.cat(accum[ts]["plus"], dim=0),
            "minus": torch.cat(accum[ts]["minus"], dim=0),
        }

    return output


def balance_activations(acts_by_ts: Dict[int, Dict[str, torch.Tensor]], seed: int):
    rng = torch.Generator().manual_seed(seed)
    balanced = {}

    for ts, pm in acts_by_ts.items():
        plus = pm["plus"]
        minus = pm["minus"]
        n = min(len(plus), len(minus))
        if n < 2:
            raise ValueError(f"timestep {ts}: not enough samples after balancing")

        p_idx = torch.randperm(len(plus), generator=rng)[:n]
        m_idx = torch.randperm(len(minus), generator=rng)[:n]

        balanced[ts] = {
            "plus": plus[p_idx].contiguous(),
            "minus": minus[m_idx].contiguous(),
        }

    return balanced


def save_per_timestep_tensors(base_dir: Path, acts_by_ts: Dict[int, Dict[str, torch.Tensor]], tag: str):
    for ts, pm in acts_by_ts.items():
        ts_dir = base_dir / f"t{ts:04d}"
        ts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pm["plus"], ts_dir / f"{tag}_positive.pt")
        torch.save(pm["minus"], ts_dir / f"{tag}_negative.pt")


def print_class_counts(acts_by_ts: Dict[int, Dict[str, torch.Tensor]], title: str):
    print(f"\n{title}")
    print("-" * len(title))
    for ts in sorted(acts_by_ts.keys(), reverse=True):
        n_pos = int(acts_by_ts[ts]["plus"].shape[0])
        n_neg = int(acts_by_ts[ts]["minus"].shape[0])
        print(f"t={ts:04d} | positive={n_pos} | negative={n_neg}")


def run_eval_scripts(pt_file: Path, eval_dir: Path, timesteps: List[int], epochs: int, lr: float, split_ratio: float, seed: int):
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts_arg = ",".join(str(t) for t in timesteps)

    scripts = [
        (
            "train_linear_probe.py",
            ["--epochs", str(epochs), "--lr", str(lr), "--split_ratio", str(split_ratio), "--seed", str(seed)],
        ),
        ("eval_svm_margin.py", []),
        ("eval_lda_eigenvalue.py", []),
    ]

    script_root = Path(__file__).resolve().parent

    for script_name, extra_args in tqdm(scripts, desc="Evaluators", leave=True):
        cmd = [
            sys.executable,
            str(script_root / script_name),
            "--file",
            str(pt_file),
            "--output_dir",
            str(eval_dir),
            "--timesteps",
            ts_arg,
            *extra_args,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def collect_summary_csv(eval_dir: Path, concept_name: str):
    patterns = {
        "linear_probe_acc": (eval_dir / f"{concept_name}_separability_log.txt", r"TS:\s*(\d+)\s*\|.*Test Acc:\s*([0-9.]+)%"),
        "svm_margin": (eval_dir / f"{concept_name}_svm_margin_log.txt", r"TS:\s*(\d+)\s*\|.*Geometric Margin:\s*([0-9.]+)"),
        "lda_lambda_max": (eval_dir / f"{concept_name}_lda_eigenvalue_log.txt", r"TS:\s*(\d+)\s*\|.*Lambda Max:\s*([0-9.]+)"),
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

    print(f"Saved summary metrics: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser("Run attribute timestep experiment")
    p.add_argument("--attribute", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, default="celeba_hq_dataset")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--scheduler_type", type=str, choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--timesteps", type=str, default="980,880,780,680,580,480,380,280,180,80")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dtype", type=str, choices=["float16", "float32"], default="float16")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs/attribute-time-step-analysis")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split_ratio", type=float, default=0.8)
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--ts_chunk_size", type=int, default=0,
                   help="Timesteps per UNet call (0=all at once). Reduce if OOM.")
    return p.parse_args()


def main():
    args = parse_args()
    timesteps = parse_timesteps(args.timesteps)
    device = resolve_device(args.device)

    ds = load_from_disk(args.dataset_dir)
    if args.attribute not in ds.column_names:
        raise ValueError(f"Attribute '{args.attribute}' not found in dataset columns")

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    concept_name = sanitize_name(args.attribute)
    base_dir = Path(args.output_root) / concept_name
    raw_dir = base_dir / "tensors" / "raw"
    balanced_dir = base_dir / "tensors" / "balanced"
    eval_dir = base_dir / "analysis"
    data_dir = base_dir / "data"

    raw_dir.mkdir(parents=True, exist_ok=True)
    balanced_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_acts = extract_multi_timestep(
        dataset=ds,
        attribute=args.attribute,
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
    save_per_timestep_tensors(raw_dir, raw_acts, concept_name)

    balanced_acts = balance_activations(raw_acts, seed=args.seed)
    print_class_counts(balanced_acts, "Balanced class counts used for training/eval")
    save_per_timestep_tensors(balanced_dir, balanced_acts, concept_name)

    payload = {
        "config": {
            "attribute": args.attribute,
            "model_id": args.model_id,
            "scheduler_type": args.scheduler_type,
            "num_steps": args.num_steps,
            "timesteps": timesteps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "save_dtype": args.save_dtype,
        },
        "activations": balanced_acts,
    }

    pt_file = data_dir / f"{concept_name}.pt"
    torch.save(payload, pt_file)
    print(f"Saved consolidated balanced activations: {pt_file}")

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
        collect_summary_csv(eval_dir, concept_name)

    print("Done.")


if __name__ == "__main__":
    main()
