import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sharpness (Laplacian Variance) on a folder of images")
    parser.add_argument("--folder", type=str, required=True, help="Path to the image folder")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions]

    if not image_paths:
        print(f"No valid images found in {folder_path}")
        return

    print(f"Evaluating Sharpness (Laplacian Variance) on {len(image_paths)} images in '{folder_path}'")

    total_score = 0.0
    valid_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"\nWarning: Could not read image {img_path}")
                continue
            
            # Calculate gradient variance (sharpness)
            score = cv2.Laplacian(img, cv2.CV_64F).var()
            total_score += score
            valid_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    if valid_count > 0:
        avg_score = total_score / valid_count
        print(f"\nCompleted evaluation.")
        print(f"Total Images Evaluated: {valid_count}")
        print(f"Average Sharpness (Laplacian Variance): {avg_score:.4f} (Higher generally means sharper)")
    else:
        print("\nNo images were successfully evaluated.")

if __name__ == "__main__":
    main()
