# Unconditional Diffusion Enhancement

This repository implements the full research pipeline for **Unconditional Diffusion Enhancement** — a method that enables semantic control over unconditional diffusion models (DDPMs) by extracting concept-specific direction vectors from the model's internal h-space (the bottleneck activation of the U-Net architecture) and applying them as targeted perturbations during the reverse generative process.
![alt text](<readme_images/Screenshot from 2026-05-01 00-07-28.png>)

Unlike classifier-free guidance (CFG), which requires conditional training, ADG operates entirely **post-hoc** on frozen, pre-trained unconditional models. The approach extracts a *Difference of Means (DoM)* vector that represents a semantic concept (e.g., "sharp vs. blur," "smiling vs. not smiling") and applies it as a directional patch to the mid-block activation during inference, enabling attribute-specific image manipulation without retraining.
![alt text](<readme_images/Screenshot from 2026-04-30 23-52-21.png>)

## Repository Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Only .py files are tracked
│
├── config/                        # Centralized configuration
│   └── config.py                  # DDPMConfig & ExtractionConfig dataclasses
│
├── data/                          # Dataset utilities
│   ├── dataset_utils.py           # Unified dataset loading (CelebA-HQ, LSUN Church)
│   └── build_dataset.py           # HuggingFace dataset builder with attribute labels
│
├── extraction/                    # Concept vector extraction
│   ├── get_dom_vector.py          # Core DoM vector extractor (streaming mean)
│   ├── main_extractor.py          # Batch extraction orchestrator
│   └── vector_extraction/
│       ├── extract_dom_vector.py  # DoM extraction with PCA/covariance analysis
│       └── extract_w_vector.py    # W-vector extraction variant
│
├── transformations/               # Image transformation modules
│   ├── transform_sharp_blur.py
│   ├── transform_gray_oversat.py
│   ├── transform_high_low_contrast.py
│   ├── transform_high_low_brightness.py
│   ├── transform_warm_cool.py
│   ├── transform_noisy_clean.py
│   ├── transform_underexposed_exposed.py
│   ├── transform_high_low_texture.py
│   ├── transform_jpeg_uncompressed.py
│   ├── transform_flat_dramatic_lighting.py
│   ├── transform_hue_natural.py
│   ├── transform_oversmoothed_natural.py
│   └── generate_examples.py       # Visual examples of all transformations
│
├── generation/                    # Unconditional DDPM generation pipeline
│   ├── main.py                    # CLI entry point: baseline + patched + CFG comparison
│   ├── pipeline.py                # Core: scheduler setup, noise generation, run_all()
│   ├── hooks.py                   # HSpacePatcher: forward hook for mid-block patching
│   ├── visualize.py               # 1×3 subplot comparison renderer
│   └── generate_triplet_dataset.py # Bulk triplet dataset generation
│
├── analysis/                      # Experimental analysis scripts
│   ├── taylor_decomposition.py    # Taylor residual decomposition (ε₀, Jv, R≥2)
│   ├── combined_entropy_taylor.py # Attention entropy + Taylor analysis (combined)
│   ├── norm_analysis.py           # Normalized L2 distance & noise norm experiments
│   └── cka/                       # CKA representational similarity
│       ├── cka_core.py            # Linear/RBF CKA computation
│       ├── compute_cka.py         # Standard CKA across layers
│       ├── compute_cka_generative.py          # CKA during generative passes
│       ├── compute_cka_generative_guided.py   # CKA with h-space guidance
│       ├── compute_global_means.py # Per-layer global mean computation
│       ├── data.py                # CelebA-HQ data loading for CKA
│       ├── hooks.py               # MultiLayerHook: capture all U-Net activations
│       ├── visualize.py           # CKA heatmap rendering
│       ├── test_cka.py            # Correctness tests for CKA computation
│       └── attention_entropy/
│           └── compute_attention_entropy.py
│
├── evaluation/                    # Quality & fidelity evaluation
│   ├── evaluate_clip.py           # CLIP score (semantic alignment with text prompts)
│   ├── evaluate_fid_folders.py    # FID from image folders
│   ├── evaluate_fid_npz.py        # FID from precomputed .npz statistics
│   ├── evaluate_fid_tf_folders.py # FID using TensorFlow InceptionV3
│   ├── evaluate_brisque.py        # BRISQUE (blind image quality)
│   ├── evaluate_contrast.py       # Contrast metrics
│   ├── evaluate_lpips.py          # LPIPS (perceptual similarity)
│   ├── evaluate_luminance.py      # Luminance analysis
│   ├── evaluate_niqe.py           # NIQE (naturalness)
│   ├── evaluate_saturation.py     # Color saturation metrics
│   ├── evaluate_sharpness.py      # Sharpness (Laplacian variance)
│   └── arcface_dir_similarity.py  # ArcFace identity preservation
│
├── semantic_analysis/             # Semantic concept separability
│   ├── run_semantic_concept_experiment.py  # Full experiment orchestrator
│   ├── extract_attribute_activations.py   # H-space activation caching
│   ├── eval_linear_probe.py       # Linear probe classifier
│   ├── eval_svm.py                # SVM separability
│   ├── eval_lda.py                # LDA discriminant analysis
│   └── eval_per_timestep.py       # Per-timestep separability sweep
│
├── timestep_analysis/             # Timestep-specific evaluations
│   ├── optimal_timestep_analysis.py       # Find optimal extraction timestep
│   ├── run_attribute_timestep_experiment.py # Multi-timestep experiment runner
│   ├── eval_lda_eigenvalue.py     # LDA eigenvalue analysis
│   ├── eval_svm_margin.py         # SVM margin analysis
│   ├── svm_consolidator.py        # Consolidate SVM results
│   └── train_linear_probe.py      # Train timestep-specific probes
│
└── utils/                         # Shared utilities
    ├── cov_analysis.py            # Covariance matrix analysis
    ├── visualize_samples.py       # Sample grid visualization
    ├── patch_env.py               # Environment patching for compatibility
    └── graphs.py                  # Plotting helpers
```

---

## Folder Details

### `config/`
Centralized configuration using Python `dataclass` objects. `DDPMConfig` holds model ID, scheduler type, inference steps, CFG scale, v-scale, patching mode, and generation parameters. `ExtractionConfig` holds dataset paths, concept definitions, and extraction hyperparameters.

### `data/`
Dataset loading utilities supporting **CelebA-HQ** (with categorical attribute labels like *Smiling*, *Male*, *Eyeglasses*) and **LSUN Church** (via LMDB). Handles normalization, resizing, and profile-specific preprocessing.

### `extraction/`
The core concept vector extraction pipeline. `get_dom_vector.py` computes the **Difference of Means (DoM)** vector via streaming mean computation — it registers a forward hook on the U-Net's `mid_block`, runs the dataset through the model at a specific timestep, and accumulates class-conditional means. Supports both **attribute mode** (CelebA-HQ binary labels) and **transformation mode** (paired image transformations like sharp/blur).

### `transformations/`
Twelve self-contained transformation modules, each implementing a `get_transforms() → (plus_tx, minus_tx)` interface. These define the positive and negative poles of a visual concept. Transformations include sharpness, contrast, brightness, color temperature, noise level, texture detail, JPEG compression, lighting style, hue shift, and smoothing.

### `generation/`
The unconditional DDPM generation pipeline. `main.py` is the CLI entry point that runs three generation modes from identical initial noise: **Baseline** (no patching), **Patched** (single-pass h-space patching), and **CFG** (dual-pass h-space classifier-free guidance). `pipeline.py` contains the core logic for scheduler construction, noise generation, and the three-mode execution. `hooks.py` implements `HSpacePatcher` which registers a PyTorch forward hook on the target layer.

### `analysis/`
Experimental analysis scripts for understanding the mechanism:

- **`taylor_decomposition.py`** — Decomposes the effect of h-space patching into first-order (Jacobian-vector product) and higher-order residual terms. Identifies *when* and *where* linearization of the decoder fails.
- **`combined_entropy_taylor.py`** — Combined attention entropy and Taylor residual analysis. Measures how attention distributions change under patching.
- **`norm_analysis.py`** — Tracks normalized L2 distance from the training distribution and predicted noise norms across guidance scales.
- **`cka/`** — CKA (Centered Kernel Alignment) pipeline for measuring representational similarity across layers and timesteps.

### `evaluation/`
Standardized evaluation metrics:

| Script | Metric | Purpose |
|--------|--------|---------|
| `evaluate_clip.py` | CLIP Score | Semantic alignment with text prompts |
| `evaluate_fid_*.py` | FID / sFID | Distributional quality (Fréchet Inception Distance) |
| `evaluate_brisque.py` | BRISQUE | Blind image quality assessment |
| `evaluate_niqe.py` | NIQE | Naturalness (no-reference) |
| `evaluate_lpips.py` | LPIPS | Perceptual similarity |
| `evaluate_sharpness.py` | Laplacian Variance | Edge detail preservation |
| `evaluate_contrast.py` | RMS Contrast | Dynamic range |
| `evaluate_luminance.py` | Mean Luminance | Brightness consistency |
| `evaluate_saturation.py` | Mean Saturation | Color vibrancy |
| `arcface_dir_similarity.py` | ArcFace Cosine Sim | Identity preservation |

### `semantic_analysis/`
Multi-timestep concept separability experiments. Extracts h-space activations at various timesteps and trains classifiers (Linear Probe, SVM, LDA) to measure how *separable* semantic concepts are in the representation space at each point in the diffusion trajectory.

### `timestep_analysis/`
Timestep-specific analysis to find the optimal extraction timestep for each concept. Evaluates LDA eigenvalues, SVM margins, and linear probe accuracy across the full timestep range.

### `utils/`
Shared utility functions: covariance analysis, sample visualization, environment patching, and plotting helpers.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Extract a Concept Vector

```bash
# Transformation-based (e.g., sharp vs. blur)
python extraction/get_dom_vector.py \
    --concept sharp_vs_blur \
    --timestep 20 \
    --num_samples 500 \
    --dataset_profile celeba_hq \
    --dataset_dir celeba_hq_dataset \
    --output_dir vectors

# Attribute-based (e.g., Smiling)
python extraction/get_dom_vector.py \
    --concept Smiling \
    --timestep 20 \
    --dataset_dir celeba_hq_dataset \
    --output_dir vectors
```

### 2. Generate Guided Images

```bash
python generation/main.py \
    --v-path vectors/sharp_vs_blur_dom_t20.pt \
    --v-scale 2.0 \
    --cfg-scale 5.0 \
    --steps 50 \
    --seed 42 \
    --output-dir outputs/sharp_guided
```

### 3. Run Evaluation

```bash
# CLIP score evaluation
python evaluation/evaluate_clip.py \
    --attribute Male \
    --prompt "A photo of a male face" \
    --n-samples 64

# FID evaluation
python evaluation/evaluate_fid_folders.py \
    --real-dir path/to/real/images \
    --gen-dir path/to/generated/images
```

### 4. Run Analysis Experiments

```bash
# Taylor decomposition
python analysis/taylor_decomposition.py

# Attention entropy + Taylor combined
python analysis/combined_entropy_taylor.py \
    --run_entropy --run_taylor \
    --batch_size 128 --taylor_batch_size 32
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- Key dependencies: `diffusers`, `transformers`, `datasets`, `pyiqa`, `scikit-learn`, `opencv-python`

See `requirements.txt` for the full list.
