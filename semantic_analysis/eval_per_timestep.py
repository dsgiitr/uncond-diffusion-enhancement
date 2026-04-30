import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

base_dir = "/opt/watchdog/users/glitch/adv_diffusion/final-battle/semantic vector extraction/outputs/smiling/tensors/balanced"
out_dir = "/opt/watchdog/users/glitch/adv_diffusion/final-battle/semantic vector extraction/outputs/smiling/analysis"

t_dirs = glob.glob(os.path.join(base_dir, "t*"))

data_points = []

for d in t_dirs:
    ts_str = os.path.basename(d)[1:]
    try:
        ts = int(ts_str)
    except ValueError:
        continue
    
    pos_path = os.path.join(d, "smiling_positive.pt")
    neg_path = os.path.join(d, "smiling_negative.pt")
    
    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        continue
        
    p_data = torch.load(pos_path, map_location="cpu").float()
    m_data = torch.load(neg_path, map_location="cpu").float()
    
    n = min(len(p_data), len(m_data))
    MAX_SAMPLES = 500
    n = min(n, MAX_SAMPLES)
    if n < 2:
        continue
        
    p_flat = p_data[:n].flatten(1).numpy()
    m_flat = m_data[:n].flatten(1).numpy()
    X = np.concatenate([p_flat, m_flat], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    
    print(f"TS: {ts:04d} | Loaded {n} pos/neg | X shape: {X.shape}")
    
    # SVM
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    svm = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    svm.fit(X_s, y)
    margin = 2.0 / (np.linalg.norm(svm.coef_) + 1e-8)
    
    # LDA
    pca = PCA(n_components=min(X.shape[0] - 1, 150))
    X_pca = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X_pca, y)
    lambda_max = getattr(lda, "explained_variance_ratio_", [0.0])[0]
    
    print(f"  -> SVM Margin: {margin:.5f} | LDA lambda_max: {lambda_max:.5f}")
    data_points.append((ts, margin, lambda_max))

data_points.sort(key=lambda x: x[0], reverse=True)

timesteps = [x[0] for x in data_points]
svm_scores = [x[1] for x in data_points]
lda_scores = [x[2] for x in data_points]

fig, ax1 = plt.subplots(figsize=(9, 5))

color = '#16a34a'
ax1.set_xlabel('Diffusion Timestep', fontsize=12)
ax1.set_ylabel('SVM Margin Width', color=color, fontsize=12)
ax1.plot(timesteps, svm_scores, marker='o', color=color, linewidth=2, markersize=7, label='SVM Margin')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim(max(timesteps) + 50, min(timesteps) - 50)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()  
color = '#9333ea'
ax2.set_ylabel('LDA Top Eigenvalue (λ_max)', color=color, fontsize=12)  
ax2.plot(timesteps, lda_scores, marker='s', color=color, linewidth=2, markersize=7, label='LDA λ_max')
ax2.tick_params(axis='y', labelcolor=color)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title("Separability vs Timestep (Smiling) - Computed per-Timestep", fontsize=14, pad=12)
fig.tight_layout()  

os.makedirs(out_dir, exist_ok=True)
plot_path = os.path.join(out_dir, "balanced_tensors_eval.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {plot_path}")
