"""
evaluate_fid.py — Compute FID between generated samples and real CIFAR-10.

FID (Fréchet Inception Distance) measures the distance between two
distributions in Inception-v3 feature space:

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r·Σ_g))

Lower is better. Reference scores on CIFAR-10:
    DDPM (Ho et al. 2020)          ~3.17
    Improved DDPM (Nichol 2021)    ~2.90
    ADM class-conditional          ~2.97

Usage:
    python evaluate_fid.py --checkpoint checkpoints/dit_epoch0100.pt
    python evaluate_fid.py --checkpoint checkpoints/dit_epoch0100.pt --num_samples 50000
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from model import DiT
from diffusion import GaussianDiffusion
from utils import EMA

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

IMG_SIZE     = 32
PATCH_SIZE   = 4
IN_CHANNELS  = 3
HIDDEN_SIZE  = 384
NUM_HEADS    = 6
DEPTH        = 6
NUM_CLASSES  = 10
TIMESTEPS    = 1000

GUIDANCE_SCALE = 3.0
DDIM_STEPS     = 100   # more steps → better quality for evaluation
BATCH_SIZE     = 100   # samples per generation batch

# Inception-v3 expects 299×299 inputs
INCEPTION_SIZE = 299


# ── Inception feature extractor ───────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """
    Wraps torchvision Inception-v3, returning the 2048-d global average pool
    features (before the classifier).  Uses a forward hook on the avgpool layer.
    """

    def __init__(self, device: str):
        super().__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=True)
        inception.eval()
        inception.to(device)
        self.inception = inception
        self.device = device
        self._features = None

        # Hook the avgpool output (2048-d) before the dropout/fc layers
        self.inception.avgpool.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output: (B, 2048, 1, 1)
        self._features = output.flatten(1)   # (B, 2048)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """
        images : (B, 3, H, W) float32 in [0, 1], will be resized to 299×299
        Returns (B, 2048) numpy array.
        """
        images = images.to(self.device)
        # Resize to Inception input size
        images = nn.functional.interpolate(
            images, size=(INCEPTION_SIZE, INCEPTION_SIZE),
            mode="bilinear", align_corners=False,
        )
        # Inception expects values normalised to [-1, 1]
        images = images * 2.0 - 1.0
        self.inception(images)
        return self._features.cpu().numpy()


# ── Statistics helpers ────────────────────────────────────────────────────────

def compute_statistics(features: np.ndarray):
    """Return (mean, covariance) of a (N, D) feature matrix."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric matrix square root of a PSD matrix A via
    eigendecomposition:  A = V·D·Vᵀ  →  sqrt(A) = V·sqrt(D)·Vᵀ
    Clips negative eigenvalues to 0 for numerical stability.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.clip(eigenvalues, 0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def compute_fid(mu_r, sigma_r, mu_g, sigma_g) -> float:
    """
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r·Σ_g))

    Tr(sqrt(Σ_r·Σ_g)) is computed as Tr(sqrt(sqrt(Σ_r)·Σ_g·sqrt(Σ_r)))
    which is numerically more stable.
    """
    diff = mu_r - mu_g
    mean_term = diff @ diff

    sqrt_sigma_r = matrix_sqrt(sigma_r)
    # M = sqrt(Σ_r) · Σ_g · sqrt(Σ_r)  — symmetric PSD
    M = sqrt_sigma_r @ sigma_g @ sqrt_sigma_r
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.clip(eigenvalues, 0, None)
    trace_sqrt = np.sum(np.sqrt(eigenvalues))

    fid = mean_term + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * trace_sqrt
    return float(fid)


# ── Real CIFAR-10 features ────────────────────────────────────────────────────

def get_real_features(extractor: InceptionFeatureExtractor, num_samples: int) -> np.ndarray:
    """Extract Inception features from real CIFAR-10 training images."""
    print(f"  Extracting features from {num_samples} real CIFAR-10 images…")
    transform = transforms.Compose([
        transforms.ToTensor(),   # [0, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=False)

    all_features = []
    collected = 0
    for imgs, _ in loader:
        feats = extractor.extract(imgs)
        all_features.append(feats)
        collected += len(feats)
        print(f"    real: {collected}/{num_samples}", end="\r")
        if collected >= num_samples:
            break

    print()
    features = np.concatenate(all_features, axis=0)[:num_samples]
    return features


# ── Generated sample features ─────────────────────────────────────────────────

def get_generated_features(
    model: DiT,
    diffusion: GaussianDiffusion,
    extractor: InceptionFeatureExtractor,
    num_samples: int,
) -> np.ndarray:
    """Generate samples with DDIM+CFG and extract their Inception features."""
    print(f"  Generating {num_samples} samples with DDIM ({DDIM_STEPS} steps)…")
    model.eval()

    all_features = []
    generated = 0

    while generated < num_samples:
        this_batch = min(BATCH_SIZE, num_samples - generated)
        # Balanced class labels across the batch
        classes = torch.arange(NUM_CLASSES, device=DEVICE).repeat(
            this_batch // NUM_CLASSES + 1
        )[:this_batch]

        shape = (this_batch, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
        samples = diffusion.ddim_sample(
            model, shape, classes,
            guidance_scale=GUIDANCE_SCALE,
            num_steps=DDIM_STEPS,
            eta=0.0,
            device=DEVICE,
        )

        # Rescale from [-1,1] to [0,1] for Inception
        samples = (samples.clamp(-1.0, 1.0) + 1.0) / 2.0
        feats = extractor.extract(samples)
        all_features.append(feats)
        generated += this_batch
        print(f"    generated: {generated}/{num_samples}", end="\r")

    print()
    return np.concatenate(all_features, axis=0)[:num_samples]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples for FID (default: 10000)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Samples: {args.num_samples}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model…")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model = DiT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
        hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, depth=DEPTH,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    ema = EMA(model)
    ema.load_state_dict(ckpt["ema_state_dict"], device=DEVICE)
    ema.copy_to(model)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    diffusion = GaussianDiffusion(timesteps=TIMESTEPS).to(DEVICE)

    # ── Inception extractor ───────────────────────────────────────────────────
    print("\nLoading Inception-v3…")
    extractor = InceptionFeatureExtractor(DEVICE)

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n[1/2] Real images")
    real_feats = get_real_features(extractor, args.num_samples)
    mu_r, sigma_r = compute_statistics(real_feats)

    print("\n[2/2] Generated images")
    gen_feats = get_generated_features(model, diffusion, extractor, args.num_samples)
    mu_g, sigma_g = compute_statistics(gen_feats)

    # ── Compute FID ───────────────────────────────────────────────────────────
    print("\nComputing FID…")
    fid = compute_fid(mu_r, sigma_r, mu_g, sigma_g)

    print("\n" + "─" * 40)
    print(f"  FID ({args.num_samples} samples): {fid:.2f}")
    print("─" * 40)
    print("\nReference scores on CIFAR-10:")
    print("  DDPM (Ho et al. 2020)        ~3.17")
    print("  Improved DDPM (Nichol 2021)  ~2.90")
    print("  ADM class-conditional        ~2.97")
    print("  DiT-S/4 on ImageNet 256px    ~68.4  (different task, for scale)")


if __name__ == "__main__":
    main()
