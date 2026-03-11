# PUDM Extension — Point Cloud Upsampling with Swappable Generative Strategies

A clean, modular reimplementation of [PUDM](https://arxiv.org/abs/2404.06868) (CVPR 2024) that supports both **DDPM** and **Flow Matching** as generative strategies for point cloud upsampling.

## Project Structure

```
pudm_extension/
├── configs/              # Experiment configs (PU1K.json, PUGAN.json)
├── compile_ops.sh        # Build CUDA extensions
├── notebooks/            # Colab notebooks
├── src/
│   ├── data/             # Datasets (PU1K, PUGAN) and augmentation
│   ├── generative/       # Strategy pattern: DDPM, Flow Matching
│   │   ├── base.py       # Abstract GenerativeStrategy
│   │   ├── ddpm.py       # DDPMStrategy (T=1000)
│   │   └── flow_matching.py  # FlowMatchingStrategy (ODE)
│   ├── metrics/          # Chamfer Distance, Hausdorff Distance
│   ├── models/           # PointNet2 backbone with cross-attention
│   ├── ops/              # CUDA ops (pointnet2_ops, pointops)
│   ├── scripts/          # Train, sample, evaluate (strategy-agnostic)
│   └── utils/            # Config, seed, point cloud helpers
└── tests/
```

## Setup (Colab)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install pytorch3d (Colab)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 3. Compile CUDA extensions
bash compile_ops.sh
```

## Usage

All scripts accept `--strategy {ddpm,flow_matching}` to select the generative method.

### Training

```bash
# DDPM (baseline)
python -m src.scripts.train -c configs/PU1K.json --strategy ddpm

# Flow Matching
python -m src.scripts.train -c configs/PU1K.json --strategy flow_matching
```

### Sampling

```bash
python -m src.scripts.sample -c configs/PU1K.json --strategy ddpm --ckpt_iter 2000
```

### Evaluation

Evaluation runs automatically at the end of sampling and reports Chamfer Distance (CD) and Hausdorff Distance (HD).

### Single-File Inference

```bash
python -m src.scripts.example_sample \
    -c configs/PU1K.json \
    --strategy ddpm \
    --ckpt_path logs/checkpoint/pointnet_ema_2000.pkl \
    --input_xyz path/to/input.xyz
```

## Generative Strategies

| Strategy | Method | Sampling | Key Idea |
|----------|--------|----------|----------|
| `ddpm` | Denoising Diffusion | T-step reverse + DDIM | Predict noise ε at each step |
| `flow_matching` | Conditional Flow Matching | Euler ODE integration | Predict velocity v = z − x₀ |

Both strategies share the same PointNet2 backbone and condition encoder. The only difference is how noisy/interpolated samples are created during training and how denoising/integration proceeds during inference.

## Adding a New Strategy

1. Subclass `GenerativeStrategy` in `src/generative/`
2. Implement `compute_hyperparams()`, `training_loss()`, `sample()`, and `name`
3. Register in `src/generative/__init__.py` → `STRATEGIES` dict

## Credits

Based on [PUDM: Point Cloud Upsampling via Denoising Diffusion Model](https://github.com/hehaodele/PUDM) (CVPR 2024).
