#!/usr/bin/env python3
"""
extract_w_vector.py
───────────────────
Trains a Linear Probe using the entire dataset for a specific timestep
and extracts the trained model weights as the final operational Normal
Boundry (W) concept vector in the exact format required for diffusion
UNet patching [512, 8, 8].

It normalizes the vector to unit length so generic scale factors align properly.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser("Extract W Vector")
    parser.add_argument("--concept", type=str, required=True, help="Concept name (e.g. sharp_vs_blur)")
    parser.add_argument("--timestep", type=int, required=True, help="Specific timestep to extract the vector from.")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs for training full batch GD.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_file = Path("outputs") / f"{args.concept}.pt"
    if not input_file.exists():
        # Fallback if they fed the literal string instead of the stem
        input_file = Path(args.concept)
        if not input_file.exists():
            print(f"Error: Could not evaluate '{args.concept}'. Missing file.")
            return

    concept_name = input_file.stem
    data = torch.load(input_file, map_location="cpu")
    
    if args.timestep not in data["activations"]:
        print(f"Error: Timestep {args.timestep} not found. Available: {list(data['activations'].keys())}")
        return
        
    p_data = data["activations"][args.timestep]["plus"].to(device)
    m_data = data["activations"][args.timestep]["minus"].to(device)
    
    N = p_data.shape[0]
    p_flat = p_data.flatten(start_dim=1)
    m_flat = m_data.flatten(start_dim=1)
    
    X = torch.cat([p_flat, m_flat], dim=0)
    y = torch.cat([torch.ones(N, device=device), torch.zeros(N, device=device)], dim=0)
    
    model = nn.Linear(X.shape[1], 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    print(f"Training Logistic Linear Probe on target TS={args.timestep} using 100% data pooling...")
    
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = model(X).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        
    # Isolate W decision normal vector
    # NOTE: We do NOT unit-normalize. The raw learned weights carry the
    # natural scale of the concept boundary. Unit-normalizing a 32k-dim
    # vector crushes each element to ~0.005, making the patch invisible.
    # v_scale in the pipeline is designed to control the final strength.
    w_vector = model.weight.data.squeeze() # [32768]
    
    # De-flatten back to shape expected by UNet Mid-Block Hook 
    w_vector = w_vector.view(p_data.shape[1], p_data.shape[2], p_data.shape[3])
    
    out_dir = Path("vectors/weight-vectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{concept_name}.pt"
    
    torch.save(w_vector.cpu(), out_path)
    print(f"\nSuccess! Evaluated Classifier Weights (W Vector).")
    print(f"Final Tensor Shape: {tuple(w_vector.shape)}")
    print(f"Saved directly to : {out_path}")

if __name__ == "__main__":
    main()
