import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Evaluate Saturation (Mean S in HSV) on a folder of images")
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

    print(f"Evaluating Saturation on {len(image_paths)} images in '{folder_path}'")

    total_score = 0.0
    valid_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"\nWarning: Could not read image {img_path}")
                continue
            
            # Convert BGR to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            
            # OpenCV S channel in HSV is 0-255 for 8-bit
            # Normalize to 0-1 range for a more interpretable score
            score = (s_channel.astype(np.float32) / 255.0).mean()
            
            total_score += score
            valid_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    if valid_count > 0:
        avg_score = total_score / valid_count
        print(f"\nCompleted evaluation.")
        print(f"Total Images Evaluated: {valid_count}")
        print(f"Average Saturation (Mean S-channel): {avg_score:.4f} (0-1 scale)")
    else:
        print("\nNo images were successfully evaluated.")

if __name__ == "__main__":
    main()
