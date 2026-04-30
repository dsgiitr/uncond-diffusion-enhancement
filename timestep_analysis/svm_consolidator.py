#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def calculate_svm_metrics_gpu(p_tensor, m_tensor, device, epochs=150, lr=1e-2):
    """
    Trains a Linear SVM using Squared Hinge Loss on GPU.
    Uses lower weight decay to avoid margin inversion on noisy data.
    """
    n = min(p_tensor.shape[0], m_tensor.shape[0])
    if n < 10: # Minimum samples for meaningful eval
        return None
        
    p_flat = p_tensor[:n].flatten(start_dim=1).to(device, dtype=torch.float32)
    m_flat = m_tensor[:n].flatten(start_dim=1).to(device, dtype=torch.float32)
    
    X = torch.cat([p_flat, m_flat], dim=0)
    y = torch.cat([torch.ones(n, device=device), -torch.ones(n, device=device)], dim=0)
    
    # Normalize features
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True) + 1e-8
    X = (X - mean) / std
    
    # Train/Test split for accuracy
    indices = torch.randperm(len(X), device=device)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    feat_dim = X.shape[1]
    model = nn.Linear(feat_dim, 1).to(device)
    
    # Low weight decay to allow margin to reflect class distance
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X[train_idx]).squeeze()
        # Squared Hinge Loss: mean(max(0, 1 - y*f(x))^2)
        loss = torch.mean(torch.clamp(1 - y[train_idx] * outputs, min=0)**2)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        # Margin: 2 / ||w||
        w = model.weight.squeeze()
        margin = 2.0 / (torch.norm(w) + 1e-8)
        
        # Accuracy
        test_outputs = model(X[test_idx]).squeeze()
        preds = torch.sign(test_outputs)
        # map preds -1,1 to 0,1 or just compare with y
        correct = (preds == y[test_idx]).float().mean()
        
        return {
            "margin": float(margin.cpu().item()),
            "accuracy": float(correct.cpu().item())
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    concepts = ["gray_vs_oversat", "high_vs_low_contrast", "sharp_vs_blur"]
    base_dir = Path("outputs/optimal-timestep-analysis")
    all_results = {}

    for concept in concepts:
        print(f"\nProcessing concept: {concept}")
        pt_file = base_dir / concept / "data" / f"{concept}_activations.pt"
        if not pt_file.exists():
            print(f"Warning: {pt_file} not found. Skipping.")
            continue
            
        data = torch.load(pt_file, map_location="cpu")
        activations = data.get("activations", {})
        
        concept_data = {"margins": {}, "accuracies": {}}
        timesteps = sorted(activations.keys(), key=int)
        
        for ts in tqdm(timesteps, desc=f"GPU Eval: {concept}"):
            p_tensor = activations[ts]["plus"]
            m_tensor = activations[ts]["minus"]
            
            metrics = calculate_svm_metrics_gpu(p_tensor, m_tensor, device)
            if metrics:
                concept_data["margins"][str(ts)] = metrics["margin"]
                concept_data["accuracies"][str(ts)] = metrics["accuracy"]
        
        all_results[concept] = concept_data

    # Save to JSON
    output_path = Path("svm_scores.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nConsolidation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
