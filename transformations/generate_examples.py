import sys
import os
import argparse
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from dataset_utils import load_image_dataset_for_profile, preprocess_pil_for_profile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "concept_extraction_pipeline", "transformations")))

# Import all transformations
import transform_sharp_blur
import transform_gray_oversat
import transform_high_low_contrast
import transform_high_low_brightness
import transform_warm_cool
import transform_noisy_clean
import transform_underexposed_exposed
import transform_high_low_texture
import transform_jpeg_uncompressed
import transform_flat_dramatic_lighting
import transform_hue_natural
import transform_oversmoothed_natural

transform_map = {
    "sharp_vs_blur": transform_sharp_blur,
    "gray_vs_oversat": transform_gray_oversat,
    "high_vs_low_contrast": transform_high_low_contrast,
    "high_vs_low_brightness": transform_high_low_brightness,
    "warm_vs_cool": transform_warm_cool,
    "noisy_vs_clean": transform_noisy_clean,
    "underexposed_vs_exposed": transform_underexposed_exposed,
    "high_vs_low_texture": transform_high_low_texture,
    "uncompressed_vs_jpeg": transform_jpeg_uncompressed,
    "dramatic_vs_flat_lighting": transform_flat_dramatic_lighting,
    "natural_vs_hue_shifted": transform_hue_natural,
    "natural_vs_oversmoothed": transform_oversmoothed_natural,
}

def tensor_to_img(tensor):
    img = tensor.clone().detach()
    img = img * 0.5 + 0.5  # denormalize from [-1, 1] to [0, 1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate visual examples for transform concepts")
    parser.add_argument("--dataset_profile", type=str, default="celeba_hq", choices=["celeba_hq", "lsun_church"])
    parser.add_argument("--dataset_dir", type=str, default="", help="Local dataset path (HF disk / image folder / LSUN LMDB)")
    parser.add_argument("--hf_dataset", type=str, default="korexyz/celeba-hq-256x256")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading dataset for profile '{args.dataset_profile}' to extract visual examples...")
    image_ds = load_image_dataset_for_profile(
        dataset_profile=args.dataset_profile,
        dataset_dir=args.dataset_dir,
        hf_dataset=args.hf_dataset,
        dataset_split=args.dataset_split,
        image_key="image",
    )

    if len(image_ds) < 2:
        raise ValueError("Need at least 2 images in the selected dataset")
    
    # Pick 2 nice diverse examples
    idx1 = min(40, len(image_ds) - 1)
    idx2 = min(125, len(image_ds) - 1)
    img1 = preprocess_pil_for_profile(image_ds[idx1], image_size=args.image_size, dataset_profile=args.dataset_profile)
    img2 = preprocess_pil_for_profile(image_ds[idx2], image_size=args.image_size, dataset_profile=args.dataset_profile)
    
    out_dir = "concept_extraction_pipeline/transformations/examples"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving visual examples to {out_dir}/ ...")
    
    to_tensor = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    base_t1 = to_tensor(img1)
    base_t2 = to_tensor(img2)
    
    for concept_name, module in transform_map.items():
        print(f" -> Generating examples for {concept_name}...")
        tx_plus, tx_minus = module.get_transforms()
        
        # Apply transforms
        plus_t1 = tx_plus(img1)
        minus_t1 = tx_minus(img1)
        
        plus_t2 = tx_plus(img2)
        minus_t2 = tx_minus(img2)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Example 1
        axes[0, 0].imshow(tensor_to_img(base_t1))
        axes[0, 0].set_title("Base Input", fontsize=14)
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(tensor_to_img(plus_t1))
        axes[0, 1].set_title("Plus Transform (e.g. Natural/Clean)", fontsize=14)
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(tensor_to_img(minus_t1))
        axes[0, 2].set_title("Minus Transform (e.g. Degraded/Transformed)", fontsize=14)
        axes[0, 2].axis("off")
        
        # Example 2
        axes[1, 0].imshow(tensor_to_img(base_t2))
        axes[1, 0].set_title("Base Input", fontsize=14)
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(tensor_to_img(plus_t2))
        axes[1, 1].set_title("Plus Transform (e.g. Natural/Clean)", fontsize=14)
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(tensor_to_img(minus_t2))
        axes[1, 2].set_title("Minus Transform (e.g. Degraded/Transformed)", fontsize=14)
        axes[1, 2].axis("off")
        
        plt.suptitle(f"Transformation: {concept_name}", fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = os.path.join(out_dir, f"{concept_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    print("Done! All transformation visual layouts have been generated successfully.")

if __name__ == "__main__":
    main()
