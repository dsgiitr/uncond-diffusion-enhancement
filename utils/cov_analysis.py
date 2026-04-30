import torch
from pathlib import Path

def analyze_covariance_similarity(concept="sharp_vs_blur", timesteps=[980, 780, 580, 380, 180]):
    data_path = Path(f"/opt/watchdog/users/glitch/adv_diffusion/final-battle/outputs/{concept}.pt")
    data = torch.load(data_path, map_location="cpu")
    activations = data["activations"]
    
    print(f"=== Covariance Structure Analysis for: {concept} ===")
    print(f"{'Timestep':<10} | {'Trace(+)':<12} | {'Trace(-)':<12} | {'Cosine Sim':<12} | {'Rel Diff':<12}")
    print("-" * 65)
    
    for ts in timesteps:
        p = activations[ts]["plus"].flatten(start_dim=1).float()
        m = activations[ts]["minus"].flatten(start_dim=1).float()
        
        N_p = p.shape[0]
        N_m = m.shape[0]
        
        p_c = p - p.mean(dim=0, keepdim=True)
        m_c = m - m.mean(dim=0, keepdim=True)
        
        # Computing Gram Matrices to simulate full 32k x 32k Covariances efficiently
        K_pp = torch.mm(p_c, p_c.t()) / (N_p - 1)
        K_mm = torch.mm(m_c, m_c.t()) / (N_m - 1)
        K_pm = torch.mm(p_c, m_c.t()) / ((N_p - 1)**0.5 * (N_m - 1)**0.5)
        
        trace_p = torch.trace(K_pp).item()
        trace_m = torch.trace(K_mm).item()
        
        norm_Sp = torch.norm(K_pp, p='fro').item()
        norm_Sm = torch.norm(K_mm, p='fro').item()
        
        trace_Sp_Sm = torch.norm(K_pm, p='fro').item() ** 2
        
        cos_sim = trace_Sp_Sm / (norm_Sp * norm_Sm)
        
        dist_sq = norm_Sp**2 + norm_Sm**2 - 2 * trace_Sp_Sm
        dist = max(0, dist_sq) ** 0.5
        rel_diff = dist / (norm_Sp + norm_Sm)
        
        print(f"{ts:<10} | {trace_p:<12.1f} | {trace_m:<12.1f} | {cos_sim:<12.4f} | {rel_diff:<12.4f}")

if __name__ == "__main__":
    analyze_covariance_similarity("high_vs_low_contrast")
