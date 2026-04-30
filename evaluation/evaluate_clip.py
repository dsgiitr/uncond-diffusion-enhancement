import argparse
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import sys
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPModel, CLIPProcessor

# Setup path for local modules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config.config import DDPMConfig
from generation.hooks import HSpacePatcher
from generation.pipeline import (
    SCHEDULER_MAP,
    build_scheduler,
    generate_initial_noise,
    run_baseline,
    run_patched,
    run_cfg,
)
from diffusers import DDPMPipeline

def parse_args():
    parser = argparse.ArgumentParser("Generate CelebA-HQ and Evaluate CLIP Scores")
    parser.add_argument("--attribute", type=str, choices=["Male", "Smile"], default="Male", help="Attribute to evaluate")
    parser.add_argument("--prompt", type=str, default="A photo of a male face", help="Text prompt for CLIP evaluation")
    parser.add_argument("--n-samples", type=int, default=16, help="Total number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--v-scale-patched", type=float, default=2.0, help="Scale for patching")
    parser.add_argument("--v-scale-guided", type=float, default=2.0, help="Scale for guided")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Scale for the guidance")
    parser.add_argument("--test-v-path", type=str, default="", help="If you want to use a specific vector path")
    parser.add_argument("--patch-start", type=int, default=0)
    parser.add_argument("--patch-end", type=int, default=15)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="eval_outputs", help="Directory to save generated images")
    
    return parser.parse_args()


# Calculate CLIP Score
def get_clip_scores(processor, model, images, text, device):
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # Broadcast text_embeds (1, D) for the batch of images (B, D)
    cos_sim = (image_embeds * text_embeds).sum(dim=-1)
    
    return cos_sim.cpu().numpy() * 100.0


def main():
    args = parse_args()
    print("=" * 70)
    print("CLIP EVALUATION - DDPM ABLATIONS")
    print("=" * 70)
    
    # Create output directories
    run_name = f"{args.attribute}_vp{args.v_scale_patched}_vg{args.v_scale_guided}_cfg{args.cfg_scale}"
    out_dir = os.path.join(args.output_dir, run_name)
    baseline_dir = os.path.join(out_dir, "baseline")
    patched_dir = os.path.join(out_dir, "patched")
    guided_dir = os.path.join(out_dir, "guided")
    
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(patched_dir, exist_ok=True)
    os.makedirs(guided_dir, exist_ok=True)
    
    print(f"Saving generated images to: {out_dir}/")
    
    # Load CLIP Model
    print(f"Loading CLIP model '{args.clip_model}'...")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model.eval()

    # Load DDPM
    print("Loading DDPM model 'google/ddpm-celebahq-256'...")
    model_id = "google/ddpm-celebahq-256"
    pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(args.device)
    unet = pipe.unet
    scheduler = build_scheduler("ddim", pipe)
    
    # Setup Vector
    if args.test_v_path:
        v_path = args.test_v_path
    elif args.attribute == "Male":
        v_path = "vectors/semantic_celeba/Male_dir_trial_t20 (1).pt"
    else:
        v_path = "vectors/semantic_celeba/Smile_dir_trial_t20 (1).pt"
        
    print(f"Loading vector from: {v_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Direction vector not found at '{v_path}'.")
    
    v = torch.load(v_path, map_location=args.device, weights_only=False).to(torch.float16)
    
    patcher_patched = HSpacePatcher(v, scale=args.v_scale_patched)
    patcher_guided = HSpacePatcher(v, scale=args.v_scale_guided)
    
    target_layer_name = "mid_block"
    target_layer = getattr(unet, "mid_block")
    
    global_idx = 0
    pbar = tqdm(total=args.n_samples, desc="Generating & Evaluating")

    baseline_scores = []
    patched_scores = []
    guided_scores = []

    while global_idx < args.n_samples:
        current_batch_size = min(args.batch_size, args.n_samples - global_idx)
        current_seed = args.seed + global_idx
        
        # Initial noise
        x_T = generate_initial_noise(unet, current_batch_size, current_seed, args.device).to(torch.float16)
        
        # Generation
        common_args = dict(
            num_steps=args.steps,
            seed=current_seed,
            device=args.device,
            patch_mode="continuous",
            patch_start=args.patch_start,
            patch_end=args.patch_end
        )

        baseline_imgs = run_baseline(unet, scheduler, x_T, num_steps=args.steps, seed=current_seed, device=args.device)
        patched_imgs = run_patched(unet, scheduler, x_T, patcher_patched, target_layer, **common_args)
        cfg_imgs = run_cfg(unet, scheduler, x_T, patcher_guided, target_layer, cfg_scale=args.cfg_scale, **common_args)
        
        # Save generated images
        for i in range(current_batch_size):
            idx = global_idx + i
            baseline_imgs[i].save(os.path.join(baseline_dir, f"{idx:05d}.png"))
            patched_imgs[i].save(os.path.join(patched_dir, f"{idx:05d}.png"))
            cfg_imgs[i].save(os.path.join(guided_dir, f"{idx:05d}.png"))
        
        # Compute CLIP Scores
        print(f"\nComputing batch size {current_batch_size} CLIP Scores w/ prompt '{args.prompt}'...")
        b_scores = get_clip_scores(clip_processor, clip_model, baseline_imgs, args.prompt, args.device)
        baseline_scores.extend(b_scores.tolist())
        
        p_scores = get_clip_scores(clip_processor, clip_model, patched_imgs, args.prompt, args.device)
        patched_scores.extend(p_scores.tolist())
        
        c_scores = get_clip_scores(clip_processor, clip_model, cfg_imgs, args.prompt, args.device)
        guided_scores.extend(c_scores.tolist())
        
        global_idx += current_batch_size
        pbar.update(current_batch_size)
        
    pbar.close()
    
    # Calculate statistics
    b_mean, b_std = np.mean(baseline_scores), np.std(baseline_scores)
    p_mean, p_std = np.mean(patched_scores), np.std(patched_scores)
    g_mean, g_std = np.mean(guided_scores), np.std(guided_scores)
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY (CLIP SCORE)")
    print(f"Attribute : {args.attribute}")
    print(f"Prompt    : '{args.prompt}'")
    print(f"vp_scale: {args.v_scale_patched} | vg_scale: {args.v_scale_guided} | cfg: {args.cfg_scale}")
    print(f"Samples   : {args.n_samples}")
    print("-" * 50)
    print(f"Baseline Mean : {b_mean:.3f} | Std : {b_std:.3f}")
    print(f"Patched Mean  : {p_mean:.3f} | Std : {p_std:.3f}")
    print(f"Guidance Mean : {g_mean:.3f} | Std : {g_std:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
