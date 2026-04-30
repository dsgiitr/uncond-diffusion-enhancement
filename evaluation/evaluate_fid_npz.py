import argparse
import sys
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from pytorch_fid.utils import get_activations
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID from .npz files containing images (NHWC, [0, 255])")
    parser.add_argument("ref_batch", type=str, help="path to reference batch npz file")
    parser.add_argument("sample_batch", type=str, help="path to sample batch npz file")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generating activations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048], help="Inception features dimensionality")
    return parser.parse_args()

def calculate_activation_statistics(images, model, batch_size, dims, device):
    """
    Calculates the statistics used by FID (mean and covariance) from a list of images.
    images: numpy array of shape (N, H, W, C) in range [0, 255]
    """
    model.eval()
    act = np.empty((len(images), dims))

    # Convert NHWC [0,255] to NCHW [0,1]
    # pytorch_fid's Inception module expects NCHW tensors, range [0,1]
    # We'll construct a dataset/dataloader equivalent inline
    
    print(f"Calculating activations for {len(images)} images...")
    
    # Process in batches
    for i in tqdm(range(0, len(images), batch_size)):
        batch_images_np = images[i:i + batch_size]
        
        # Convert to torch tensor: NHWC -> NCHW
        # Numpy shape is (B, H, W, C), change to (B, C, H, W)
        batch_images_chw = np.transpose(batch_images_np, (0, 3, 1, 2))
        
        # Convert to float tensor and normalize to [0,1]
        batch = torch.from_numpy(batch_images_chw).float() / 255.0
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, collapse spatial dimensions if present
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

        act[i:i + batch_size] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def load_npz_images(path):
    print(f"Loading {path}...")
    try:
        # Load the npz file
        data = np.load(path)
        # Default behavior: assume the images are in the first array if 'arr_0' or 'images' exist
        if 'arr_0' in data:
            images = data['arr_0']
        elif 'images' in data:
            images = data['images']
        else:
            # Fallback to the first key
            images = data[list(data.keys())[0]]
            
        print(f"Loaded array shape: {images.shape}, dtype: {images.dtype}")
        
        # Ensure it's NHWC and [0, 255]
        if len(images.shape) != 4 or images.shape[-1] not in [1, 3]:
             raise ValueError(f"Expected NHWC format (e.g., NxHxWx3), got {images.shape}")
             
        if np.issubdtype(images.dtype, np.floating) and images.max() <= 1.0:
            print("Warning: Images appear to be in [0, 1] range. Multiplying by 255.")
            images = (images * 255).astype(np.uint8)
            
        return images
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)

    # 2. Load `.npz` arrays (NHWC format, [0, 255])
    ref_images = load_npz_images(args.ref_batch)
    sample_images = load_npz_images(args.sample_batch)

    # 3. Calculate statistics for reference batch
    print("\n--- Processing Reference Batch ---")
    m1, s1 = calculate_activation_statistics(ref_images, model, args.batch_size, args.dims, device)

    # 4. Calculate statistics for sample batch
    print("\n--- Processing Sample Batch ---")
    m2, s2 = calculate_activation_statistics(sample_images, model, args.batch_size, args.dims, device)

    # 5. Compute Frechet Distance
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    print("\n" + "="*40)
    print(f"FID Score: {fid_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
