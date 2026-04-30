import argparse
import os
import glob
import random
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from PIL import Image

def read_batches_from_dir(dir_path: str, batch_size: int, num_samples: int = None):
    # Check if this is a HuggingFace Arrow dataset
    if os.path.exists(os.path.join(dir_path, "dataset_info.json")) or len(glob.glob(os.path.join(dir_path, "*.arrow"))) > 0:
        print(f"Detected HuggingFace Arrow dataset at {dir_path}. Loading stream...")
        from datasets import load_from_disk
        dataset = load_from_disk(dir_path)
        
        # Determine the split. Defaults to 'train' if it's a DatasetDict.
        if hasattr(dataset, "keys") and 'train' in dataset:
            split = dataset['train']
        elif hasattr(dataset, "keys") and len(dataset.keys()) > 0:
            split = dataset[list(dataset.keys())[0]]
        else:
            split = dataset
            
        if num_samples is not None and num_samples < len(split):
            print(f"Randomly selecting {num_samples} out of {len(split)} samples...")
            split = split.shuffle(seed=42).select(range(num_samples))
            
        batch = []
        for item in split:
            img = item.get('image') or item.get('img')
            if img is None: raise ValueError("Could not find image column in dataset.")
            img = img.convert("RGB")
            batch.append(np.array(img))
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
        if len(batch) > 0:
            yield np.stack(batch, axis=0)
        return

    # Fallback: standard image folder
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    files.sort()
    
    if not files:
        raise ValueError(f"No images or Arrow datasets found in {dir_path}")
        
    if num_samples is not None and num_samples < len(files):
        print(f"Randomly selecting {num_samples} out of {len(files)} files...")
        random.seed(42)
        random.shuffle(files)
        files = files[:num_samples]
        
    batch = []
    for f in files:
        img = Image.open(f).convert("RGB")
        batch.append(np.array(img))
        if len(batch) == batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if len(batch) > 0:
        yield np.stack(batch, axis=0)

def get_activations_and_stats(generator, model, device):
    import tqdm
    act = []
    for batch_np in tqdm.tqdm(generator, desc="Evaluating"):
        # Convert NHWC [0,255] to NCHW [0,1]
        batch_chw = np.transpose(batch_np, (0, 3, 1, 2))
        batch = torch.from_numpy(batch_chw).float() / 255.0
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        act.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))
        
    if not act:
        raise ValueError("No activations computed! Generator was empty.")

    act = np.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def main():
    parser = argparse.ArgumentParser(description="Calculate FID using PyTorch InceptionV3 (with HF Dataset support)")
    parser.add_argument("ref_path", type=str, help="Path to reference images directory, HuggingFace dataset, OR pre-computed .npz stats file")
    parser.add_argument("sample_dir", type=str, nargs="?", help="Path to sample images directory")
    parser.add_argument("--save-stats", type=str, help="Path to save computed reference stats as .npz")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generating activations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples to process")
    parser.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048], help="Inception features dimensionality")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, batch size: {args.batch_size}, dims: {args.dims}")

    # Load PyTorch Inception Model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # 1. Provide reference statistics
    if args.ref_path.endswith('.npz'):
        print(f"loading pre-computed reference statistics from {args.ref_path}...")
        obj = np.load(args.ref_path)
        mu1, sigma1 = obj["mu"], obj["sigma"]
    else:
        print(f"computing reference batch activations from {args.ref_path}...")
        ref_generator = read_batches_from_dir(args.ref_path, args.batch_size, args.num_samples)
        mu1, sigma1 = get_activations_and_stats(ref_generator, model, device)
        
        if args.save_stats:
            print(f"saving reference stats to {args.save_stats}...")
            np.savez(args.save_stats, mu=mu1, sigma=sigma1)
            print("Done saving.")

    # 2. Check if we have a sample directory
    if not args.sample_dir:
        if not args.save_stats:
            print("No sample directory or --save-stats provided. Nothing else to do.")
        return

    # 3. Compute sample statistics
    print(f"computing sample batch activations from {args.sample_dir}...")
    sample_generator = read_batches_from_dir(args.sample_dir, args.batch_size, args.num_samples)
    mu2, sigma2 = get_activations_and_stats(sample_generator, model, device)

    # 4. Compute Frechet Distance
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    print("\n" + "="*40)
    print(f"PyTorch FID Score: {fid_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
