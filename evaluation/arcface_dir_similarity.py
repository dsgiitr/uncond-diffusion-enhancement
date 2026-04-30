#!/usr/bin/env python3
"""Compute ArcFace cosine similarity between two image directories.

This script intentionally skips face detection/alignment and runs recognition
embeddings directly on resized images in large batches using onnxruntime-gpu.
It uses multithreading for I/O bounds and PyTorch (if available) for GPU matrix math.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Attempt to load PyTorch for GPU accelerated matrix math
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcFace cosine similarity for two datasets")
    parser.add_argument("--dir-a", type=Path, required=True, help="Path to first dataset directory")
    parser.add_argument("--dir-b", type=Path, required=True, help="Path to second dataset directory")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for recognizer.get_feat")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of CPU threads for image loading")
    parser.add_argument("--model-name", type=str, default="buffalo_l", help="InsightFace model pack name")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path.home() / ".insightface",
        help="InsightFace model cache root",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images in subdirectories",
    )
    return parser.parse_args()


def list_images(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    iterator: Iterable[Path]
    if recursive:
        iterator = folder.rglob("*")
    else:
        iterator = folder.glob("*")
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images


def load_for_arcface(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img)


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


def extract_embeddings(recognizer, image_paths: list[Path], batch_size: int, num_workers: int) -> tuple[np.ndarray, int]:
    all_feats: list[np.ndarray] = []
    skipped = 0

    # OPTIMIZATION 1: Use a ThreadPool to load/resize images in parallel
    # This prevents the GPU from starving while waiting for the CPU to read the hard drive
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            
            # Map the load function over the batch paths. Order is preserved automatically.
            batch_imgs = list(executor.map(load_for_arcface, batch_paths))
            
            # Filter out None values from corrupted or missing images
            valid_imgs = [img for img in batch_imgs if img is not None]
            skipped += len(batch_imgs) - len(valid_imgs)

            if not valid_imgs:
                continue

            feats = recognizer.get_feat(valid_imgs).astype(np.float32)
            all_feats.append(feats)

    if not all_feats:
        raise RuntimeError("No valid embeddings were extracted. Check your input directories.")

    embeddings = np.concatenate(all_feats, axis=0)
    embeddings = l2_normalize(embeddings, axis=1)
    return embeddings, skipped


def centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    center_a = l2_normalize(a.mean(axis=0, keepdims=True), axis=1)[0]
    center_b = l2_normalize(b.mean(axis=0, keepdims=True), axis=1)[0]
    return float(np.dot(center_a, center_b))


def mean_nearest_neighbor_cosine(query: np.ndarray, gallery: np.ndarray, chunk: int = 2048) -> float:
    # OPTIMIZATION 2: Push heavy matrix math to the GPU via PyTorch
    if HAS_TORCH_CUDA:
        # Move numpy arrays directly to the GPU
        query_gpu = torch.from_numpy(query).cuda()
        gallery_t_gpu = torch.from_numpy(gallery).cuda().T
        total = 0.0
        count = 0
        
        # Use torch.no_grad() to save VRAM during inference
        with torch.no_grad():
            for i in range(0, query_gpu.shape[0], chunk):
                # GPU Accelerated MatMul
                sims = torch.matmul(query_gpu[i : i + chunk], gallery_t_gpu)
                total += float(sims.max(dim=1)[0].sum().cpu())
                count += sims.shape[0]
        return total / max(count, 1)
        
    # Fallback if PyTorch is not installed (Uses standard CPU math)
    else:
        gallery_t = gallery.T
        total = 0.0
        count = 0
        for i in range(0, query.shape[0], chunk):
            sims = query[i : i + chunk] @ gallery_t
            total += float(sims.max(axis=1).sum())
            count += sims.shape[0]
        return total / max(count, 1)


def build_recognizer(model_name: str, model_root: Path):
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider not available. Install/activate onnxruntime-gpu and CUDA correctly."
        )

    app = FaceAnalysis(
        name=model_name,
        root=str(model_root),
        # FaceAnalysis asserts that detection is present at init time.
        # We still run recognition directly via recognizer.get_feat and never call detection.
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0)

    if "recognition" not in app.models:
        raise RuntimeError("Recognition model was not loaded. Check model name/root.")
    return app.models["recognition"]


def main() -> None:
    args = parse_args()

    images_a = list_images(args.dir_a, recursive=args.recursive)
    images_b = list_images(args.dir_b, recursive=args.recursive)

    if not images_a:
        raise RuntimeError(f"No images found in --dir-a: {args.dir_a}")
    if not images_b:
        raise RuntimeError(f"No images found in --dir-b: {args.dir_b}")

    recognizer = build_recognizer(args.model_name, args.model_root)

    print(f"Extracting embeddings for dir-a using {args.num_workers} threads...")
    emb_a, skipped_a = extract_embeddings(recognizer, images_a, args.batch_size, args.num_workers)
    
    print(f"Extracting embeddings for dir-b using {args.num_workers} threads...")
    emb_b, skipped_b = extract_embeddings(recognizer, images_b, args.batch_size, args.num_workers)

    math_backend = "PyTorch (GPU)" if HAS_TORCH_CUDA else "NumPy (CPU)"
    print(f"Calculating similarity scores using {math_backend}...")

    score_centroid = centroid_cosine(emb_a, emb_b)
    score_a_to_b = mean_nearest_neighbor_cosine(emb_a, emb_b)
    score_b_to_a = mean_nearest_neighbor_cosine(emb_b, emb_a)

    print("\n--- ArcFace Similarity Results ---")
    print(f"dir-a images: {len(images_a)} | valid embeddings: {emb_a.shape[0]} | skipped: {skipped_a}")
    print(f"dir-b images: {len(images_b)} | valid embeddings: {emb_b.shape[0]} | skipped: {skipped_b}")
    print(f"centroid_cosine:       {score_centroid:.6f}")
    print(f"mean_nn_cosine_a_to_b: {score_a_to_b:.6f}")
    print(f"mean_nn_cosine_b_to_a: {score_b_to_a:.6f}")


if __name__ == "__main__":
    main()