import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pyiqa

def main():
    parser = argparse.ArgumentParser(description="Evaluate NIQE on a folder of images")
    parser.add_argument("--folder", type=str, required=True, help="Path to the image folder")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Initialize NIQE metric using pyiqa
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing NIQE metric on {device}...")
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # Valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = [p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions]

    if not image_paths:
        print(f"No valid images found in {folder_path}")
        return

    print(f"Evaluating NIQE on {len(image_paths)} images in '{folder_path}'")

    total_score = 0.0
    valid_count = 0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            score = niqe_metric(str(img_path))
            total_score += score.item()
            valid_count += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    if valid_count > 0:
        avg_score = total_score / valid_count
        print(f"\nCompleted evaluation.")
        print(f"Total Images Evaluated: {valid_count}")
        print(f"Average NIQE Score: {avg_score:.4f} (Lower is better for Natural Image Quality)")
    else:
        print("\nNo images were successfully evaluated.")

if __name__ == "__main__":
    main()
