# Diffusion Transformer (DiT) for Image Generation

A from-scratch implementation of a **Diffusion Transformer** trained on CIFAR-10, built entirely in PyTorch without any pretrained models or external diffusion libraries.

<img width="1444" height="1376" alt="image" src="https://github.com/user-attachments/assets/99bcec94-e903-412f-acf5-8b09cde39fbd" />


*Class-conditional samples at epoch 100. Rows (top to bottom): plane, car, bird, cat, deer, dog, frog, horse, ship, truck.*

---

## What this is

This project implements the full DiT (Diffusion Transformer) pipeline from the paper [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748) — every component written manually:

- **Gaussian diffusion** with a linear noise schedule (DDPM)
- **Transformer denoiser** with adaLN-Zero conditioning
- **Classifier-free guidance** (CFG) with 10% label dropout during training
- **DDIM sampling** for fast inference (25 steps vs 1000)
- **EMA** of model weights for stable generation
- Two model sizes trained and benchmarked head-to-head

---

## Results

### FID Score (10,000 samples, DDIM 100 steps)

| Model | Params | Epochs | FID |
|---|---|---|---|
| **DiT-small (ours)** | 16M | 200 | **22.18** |
| **DiT-large (ours)** | 58M | 100 | in progress |
| DDPM (Ho et al. 2020) | ~35M | ~800K steps | 3.17 |
| Improved DDPM (Nichol 2021) | 100M+ | ~800K steps | 2.90 |

Our 16M model achieves **FID 22.18** trained for 200 epochs (~78K gradient steps) on a single Apple M-series GPU — roughly 4× fewer parameters and 10× fewer training steps than the published baselines. The gap is expected; closing it is purely a compute and scale question.

### Loss curve

The model converges cleanly from ~0.14 (epoch 1) to ~0.03 (epoch 100), consistent with epsilon-prediction MSE on CIFAR-10 at this scale.

---

## Architecture

```
Input (3×32×32)
    │
    ▼
Patchify — Conv2d(3→D, k=4, s=4) → 64 patch tokens of dim D
    │
    + Learnable positional embedding (1, 64, D)
    │
    ▼
6–12× DiT Block
    ├── adaLN-Zero: conditioning vector c = t_emb + class_emb
    │       └── projects c → (shift, scale, gate) × 2
    ├── Multi-head self-attention (QKV projection)
    └── Feed-forward network (4× expansion, GELU)
    │
    ▼
Final adaLN → Linear(D → 48) → Unpatchify → (3×32×32) predicted noise
```

**Conditioning** — timestep `t` is encoded with sinusoidal embeddings → MLP, class label with a learned embedding. These are summed and injected into every transformer block via adaptive layer norm, scaled with zero-initialised gates so every block starts as an identity function.

**Classifier-free guidance** — during training, 10% of class labels are randomly replaced with a learned null token. At inference, both conditional and unconditional noise predictions are computed and blended:

```
ε_guided = ε_uncond + guidance_scale × (ε_cond − ε_uncond)
```

---

## Project structure

```
.
├── model.py          # DiT architecture — patchify, adaLN-Zero, attention, FFN
├── diffusion.py      # Noise schedule, forward diffusion, DDPM & DDIM samplers
├── train.py          # Training loop — EMA, CFG dropout, checkpointing, auto-resume
├── sample.py         # Generate DDPM and DDIM sample grids from a checkpoint
├── evaluate_fid.py   # FID computation against real CIFAR-10 (Inception-v3 features)
└── utils.py          # EMA class, image grid saving
```

---

## Usage

**Train**
```bash
# Small model (~16M params)
python train.py --model small

# Large model (~58M params)
python train.py --model large
```

Training auto-resumes from the latest checkpoint if one exists. Sample grids are saved every 10 epochs; checkpoints every 5.

**Generate samples**
```bash
python sample.py --model small --checkpoint checkpoints/dit_epoch0200.pt
```
Outputs both a full DDPM grid (1000 steps) and a fast DDIM grid (25 steps).

**Evaluate FID**
```bash
python evaluate_fid.py --checkpoint checkpoints/dit_epoch0200.pt
```

---

## Dependencies

```
torch
torchvision
numpy
Pillow
```

No HuggingFace diffusers, no timm, no pretrained weights.

---

## Key implementation details worth noting

- **adaLN-Zero init** — the linear layer that projects the conditioning vector to modulation parameters is zero-initialised. This means every DiT block starts as an identity function and learns to deviate from there, which significantly stabilises early training.
- **EMA weights** — a shadow copy of model parameters is maintained with decay=0.9999 and used exclusively for sampling. Raw model weights are only used for gradient updates.
- **DDIM** — at inference, 25 deterministic steps (η=0) replace the full 1000-step DDPM chain with minimal quality loss, making sample generation ~40× faster.
- **Schedule precision** — the noise schedule is computed in float32 throughout to stay compatible with Apple MPS (which doesn't support float64).
