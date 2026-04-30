import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mean Luminance (Mean L in LAB space) on a folder of images")
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

    print(f"Evaluating Mean Luminance on {len(image_paths)} images in '{folder_path}'")

    total_score = 0.0
    valid_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"\nWarning: Could not read image {img_path}")
                continue
            
            # Convert BGR to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # In OpenCV for 8-bit images, L channel is scaled to 0-255 (actual LAB L is 0-100)
            # We rescale back to the standard 0-100 LAB L-scale
            score = l_channel.astype(np.float32).mean() * (100.0 / 255.0)
            
            total_score += score
            valid_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    if valid_count > 0:
        avg_score = total_score / valid_count
        print(f"\nCompleted evaluation.")
        print(f"Total Images Evaluated: {valid_count}")
        print(f"Average Mean Luminance (LAB L-channel): {avg_score:.4f} (0-100 scale)")
    else:
        print("\nNo images were successfully evaluated.")

if __name__ == "__main__":
    main()
