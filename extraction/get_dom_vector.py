import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from datasets import load_from_disk
from tqdm.auto import tqdm
import numpy as np
import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.dataset_utils import (
    load_image_dataset_for_profile,
    normalize_dataset_profile,
    preprocess_pil_for_profile,
)

try:
    from transformations import (
        transform_sharp_blur,
        transform_gray_oversat,
        transform_high_low_contrast,
        transform_high_low_brightness,
        transform_warm_cool,
        transform_noisy_clean,
        transform_underexposed_exposed,
        transform_high_low_texture,
        transform_jpeg_uncompressed,
        transform_flat_dramatic_lighting,
        transform_hue_natural,
        transform_oversmoothed_natural,
    )
    TRANSFORMS_AVAILABLE = True
except ImportError:
    TRANSFORMS_AVAILABLE = False


class HSpaceHook:
    def __init__(self, unet: UNet2DModel):
        self.h: Optional[torch.Tensor] = None
        self._handle = unet.mid_block.register_forward_hook(self._fn)

    def _fn(self, module: nn.Module, inp, out):
        self.h = out.detach()

    def remove(self):
        self._handle.remove()


class CelebADataset(Dataset):
    """Dataset for CelebA-HQ categorical attributes (Smile, Male, etc.)"""
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


class PairedTransformDataset(Dataset):
    """Dataset for Transformation pairs (sharp_vs_blur, etc.)"""
    def __init__(
        self,
        image_dataset,
        plus_tx,
        minus_tx,
        dataset_profile: str = "celeba_hq",
        image_size: int = 256,
    ):
        self.ds = image_dataset
        self.plus_tx = plus_tx
        self.minus_tx = minus_tx
        self.dataset_profile = normalize_dataset_profile(dataset_profile)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        pil_img = self.ds[idx]
        pil_img = preprocess_pil_for_profile(
            pil_img,
            image_size=self.image_size,
            dataset_profile=self.dataset_profile,
        )
        # return two images for plus and minus
        x_plus = self.plus_tx(pil_img)
        x_minus = self.minus_tx(pil_img)
        return x_plus, x_minus


def resolve_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dom_vector(
    concept: str,
    timestep: int = 20,
    num_samples: int = 500,
    batch_size: int = 16,
    dataset_dir: str = "celeba_hq_dataset",
    dataset_profile: str = "celeba_hq",
    hf_dataset: str = "",
    dataset_split: str = "train",
    model_id: str = "google/ddpm-celebahq-256",
    scheduler_type: str = "ddim",
    device: Union[str, torch.device] = "",
    seed: int = 42,
    num_steps: int = 50,
    image_size: int = 256,
    num_workers: int = 4,
) -> torch.Tensor:
    """
    Directly calculate the Difference of Means (DoM) vector for a given concept.
    Does NOT save massive dictionaries; computes running means directly.
    """
    device = resolve_device(device) if isinstance(device, str) else device
    dataset_profile = normalize_dataset_profile(dataset_profile)

    print(
        f"Initializing DoM extraction for concept: '{concept}' at t={timestep} on {device} "
        f"(profile={dataset_profile})"
    )

    # Initialize Model
    unet = UNet2DModel.from_pretrained(model_id).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(model_id) if scheduler_type == "ddim" else DDPMScheduler.from_pretrained(model_id)
    scheduler.set_timesteps(num_steps, device=device)
    hook = HSpaceHook(unet)
    use_amp = device.type == "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Determine if Concept is a transform or an attribute
    transform_map = {
        "sharp_vs_blur": "transform_sharp_blur",
        "gray_vs_oversat": "transform_gray_oversat",
        "high_vs_low_contrast": "transform_high_low_contrast",
        "high_vs_low_brightness": "transform_high_low_brightness",
        "warm_vs_cool": "transform_warm_cool",
        "noisy_vs_clean": "transform_noisy_clean",
        "underexposed_vs_exposed": "transform_underexposed_exposed",
        "high_vs_low_texture": "transform_high_low_texture",
        "uncompressed_vs_jpeg": "transform_jpeg_uncompressed",
        "dramatic_vs_flat_lighting": "transform_flat_dramatic_lighting",
        "natural_vs_hue_shifted": "transform_hue_natural",
        "natural_vs_oversmoothed": "transform_oversmoothed_natural",
    }

    is_transform = concept in transform_map

    if is_transform and TRANSFORMS_AVAILABLE:
        # ── TRANSFORM MODE ───────────────────────────────────────────────────
        print(f"Using Transform mode for '{concept}'")
        if concept == "sharp_vs_blur":
            tx_plus, tx_minus = transform_sharp_blur.get_transforms()
        elif concept == "gray_vs_oversat":
            tx_plus, tx_minus = transform_gray_oversat.get_transforms()
        elif concept == "high_vs_low_contrast":
            tx_plus, tx_minus = transform_high_low_contrast.get_transforms()
        elif concept == "high_vs_low_brightness":
            tx_plus, tx_minus = transform_high_low_brightness.get_transforms()
        elif concept == "warm_vs_cool":
            tx_plus, tx_minus = transform_warm_cool.get_transforms()
        elif concept == "noisy_vs_clean":
            tx_plus, tx_minus = transform_noisy_clean.get_transforms()
        elif concept == "underexposed_vs_exposed":
            tx_plus, tx_minus = transform_underexposed_exposed.get_transforms()
        elif concept == "high_vs_low_texture":
            tx_plus, tx_minus = transform_high_low_texture.get_transforms()
        elif concept == "uncompressed_vs_jpeg":
            tx_plus, tx_minus = transform_jpeg_uncompressed.get_transforms()
        elif concept == "dramatic_vs_flat_lighting":
            tx_plus, tx_minus = transform_flat_dramatic_lighting.get_transforms()
        elif concept == "natural_vs_hue_shifted":
            tx_plus, tx_minus = transform_hue_natural.get_transforms()
        elif concept == "natural_vs_oversmoothed":
            tx_plus, tx_minus = transform_oversmoothed_natural.get_transforms()
            
        image_ds = load_image_dataset_for_profile(
            dataset_profile=dataset_profile,
            dataset_dir=dataset_dir,
            hf_dataset=hf_dataset,
            dataset_split=dataset_split,
            image_key="image",
        )

        if num_samples > 0 and num_samples < len(image_ds):
            image_ds = Subset(image_ds, list(range(num_samples)))

        dataset = PairedTransformDataset(
            image_ds,
            tx_plus,
            tx_minus,
            dataset_profile=dataset_profile,
            image_size=image_size,
        )
        print(f"Loaded {len(dataset)} transform samples")
    else:
        # ── ATTRIBUTE MODE ───────────────────────────────────────────────────
        if dataset_profile != "celeba_hq":
            raise ValueError(
                "Attribute-mode extraction requires CelebA-HQ labels. "
                "Use --dataset_profile celeba_hq or use transformation concepts for LSUN Church."
            )

        print(f"Using CelebA-HQ Attribute mode for '{concept}'")
        ds = load_from_disk(dataset_dir)
        if concept not in ds.column_names:
            raise ValueError(f"Concept '{concept}' not found in dataset columns: {ds.column_names}")
            
        print(f"Extracting true DoM over the ENTIRE dataset ({len(ds)} samples)...")
        dataset = CelebADataset(ds, concept)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Accumulators for direct streaming mean calculation
    sum_plus = None
    sum_minus = None
    count_plus = 0
    count_minus = 0

    t0 = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {concept}"):
            if is_transform:
                x_plus, x_minus = batch
                x_plus = x_plus.to(device, non_blocking=True)
                x_minus = x_minus.to(device, non_blocking=True)
                x_all = torch.cat([x_plus, x_minus], dim=0)
                B = x_plus.shape[0]
            else:
                x, y = batch
                x_all = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                B = x.shape[0]

            t_vec = torch.full((x_all.shape[0],), timestep, device=device, dtype=torch.long)
            eps = torch.randn(x_all.shape, generator=generator, device=device)
            
            if timestep > 0:
                x_t = scheduler.add_noise(x_all, eps, t_vec)
            else:
                x_t = x_all

            if use_amp:
                with torch.cuda.amp.autocast():
                    unet(x_t, t_vec)
            else:
                unet(x_t, t_vec)

            h = hook.h  # [B or 2B, 512, 8, 8]
            
            # Aggregate sums directly
            if is_transform:
                h_plus = h[:B]
                h_minus = h[B:]
                
                if sum_plus is None:
                    sum_plus = h_plus.sum(dim=0, dtype=torch.float64)
                    sum_minus = h_minus.sum(dim=0, dtype=torch.float64)
                else:
                    sum_plus += h_plus.sum(dim=0, dtype=torch.float64)
                    sum_minus += h_minus.sum(dim=0, dtype=torch.float64)
                    
                count_plus += B
                count_minus += B
                
            else:
                pos_mask = (y == 1)
                neg_mask = (y == 0)
                
                h_plus = h[pos_mask]
                h_minus = h[neg_mask]
                
                if sum_plus is None:
                    # Initialize with zeros in the same shape as h[0]
                    # Shape is typically [C, H, W] => [512, 8, 8]
                    sum_plus = torch.zeros(h.shape[1:], device=device, dtype=torch.float64)
                    sum_minus = torch.zeros(h.shape[1:], device=device, dtype=torch.float64)

                if h_plus.shape[0] > 0:
                    sum_plus += h_plus.sum(dim=0, dtype=torch.float64)
                    count_plus += h_plus.shape[0]
                    
                if h_minus.shape[0] > 0:
                    sum_minus += h_minus.sum(dim=0, dtype=torch.float64)
                    count_minus += h_minus.shape[0]

    hook.remove()
    del unet

    if count_plus == 0 or count_minus == 0:
        raise ValueError(f"Could not compute DoM! Missing classes: pos={count_plus}, neg={count_minus}")

    # Calculate Difference of Means
    mean_plus = sum_plus / count_plus
    mean_minus = sum_minus / count_minus
    
    dom_vector = (mean_plus - mean_minus).to(torch.float32) # [512, 8, 8]
    
    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"Total samples processed: pos={count_plus}, neg={count_minus}")
    print(f"DoM vector shape: {list(dom_vector.shape)}")
    
    return dom_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct DoM Vector Extractor")
    parser.add_argument("--concept", type=str, required=True, help="Concept (e.g. Male, Smiling, sharp_vs_blur)")
    parser.add_argument("--timestep", type=int, default=20, help="Timestep to extract from (e.g. 20 for t20)")
    parser.add_argument("--num_samples", type=int, default=500, help="Samples to process (Transforms ONLY. Attributes always use full dataset)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--image_size", type=int, default=256, help="Model image size")
    parser.add_argument("--dataset_profile", type=str, default="celeba_hq", choices=["celeba_hq", "lsun_church"], help="Dataset profile")
    parser.add_argument("--dataset_dir", type=str, default="celeba_hq_dataset", help="Local dataset path (HF disk dataset, LSUN LMDB, or image folder)")
    parser.add_argument("--hf_dataset", type=str, default="", help="Optional HF dataset id fallback")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split when loading HF datasets")
    parser.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256", help="Diffusion model id")
    parser.add_argument("--scheduler_type", type=str, default="ddim", choices=["ddim", "ddpm"], help="Scheduler type")
    parser.add_argument("--device", type=str, default="", help="Device override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=50, help="Scheduler inference steps")
    parser.add_argument("--output_dir", type=str, default="vectors", help="Directory to save the output files.")
    parser.add_argument("--output_subdir", type=str, default="", help="Optional nested subdirectory inside output_dir")
    parser.add_argument("--output_file", type=str, default="", help="Specific filename override. If not set, defaults to <output_dir>/<concept>_dom_t<timestep>.pt")
    
    args = parser.parse_args()
    
    dom_vec = get_dom_vector(
        concept=args.concept,
        timestep=args.timestep,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        dataset_dir=args.dataset_dir,
        dataset_profile=args.dataset_profile,
        hf_dataset=args.hf_dataset,
        dataset_split=args.dataset_split,
        model_id=args.model_id,
        scheduler_type=args.scheduler_type,
        device=args.device,
        seed=args.seed,
        num_steps=args.num_steps,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    
    if args.output_file:
        out_path = Path(args.output_file)
    else:
        out_dir = Path(args.output_dir)
        profile = normalize_dataset_profile(args.dataset_profile)

        if args.output_subdir:
            out_dir = out_dir / args.output_subdir
        elif profile == "lsun_church":
            out_dir = out_dir / "lsun_church"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.concept}_dom_t{args.timestep}.pt"
        
    torch.save(dom_vec, out_path)
    print(f"Saved DoM vector to {out_path}!")
