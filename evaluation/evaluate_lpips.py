import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pyiqa
import os

def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS between two folders of images")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the reference image folder")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the generated image folder")
    args = parser.parse_args()

    folder1_path = Path(args.folder1)
    folder2_path = Path(args.folder2)
    
    if not folder1_path.exists() or not folder1_path.is_dir():
        print(f"Error: {folder1_path} is not a valid directory.")
        return
    if not folder2_path.exists() or not folder2_path.is_dir():
        print(f"Error: {folder2_path} is not a valid directory.")
        return

    # Initialize LPIPS metric using pyiqa
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing LPIPS metric on {device}...")
    lpips_metric = pyiqa.create_metric('lpips', device=device)

    # Valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    
    # We assume image names are the same in both folders!
    image_paths1 = [p for p in folder1_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions]
    
    if not image_paths1:
        print(f"No valid images found in {folder1_path}")
        return

    print(f"Evaluating LPIPS on images. Matching filenames between '{folder1_path}' and '{folder2_path}'")

    total_score = 0.0
    valid_count = 0
    
    for img1_path in tqdm(image_paths1, desc="Processing images"):
        img2_path = folder2_path / img1_path.name
        
        if not img2_path.exists():
            print(f"\nWarning: Match not found for {img1_path.name} in folder2. Skipping.")
            continue
            
        try:
            score = lpips_metric(str(img1_path), str(img2_path))
            total_score += score.item()
            valid_count += 1
        except Exception as e:
            print(f"\nError processing {img1_path.name}: {e}")

    if valid_count > 0:
        avg_score = total_score / valid_count
        print(f"\nCompleted evaluation.")
        print(f"Total Images Evaluated: {valid_count}")
        print(f"Average LPIPS Score: {avg_score:.4f} (Lower is better, meaning less perceptual difference)")
    else:
        print("\nNo images were successfully evaluated.")

if __name__ == "__main__":
    main()
