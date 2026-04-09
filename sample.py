"""
sample.py — Generate sample grids from a trained DiT checkpoint.

Usage
─────
    python sample.py --checkpoint checkpoints/dit_epoch0100.pt
    python sample.py --checkpoint checkpoints/dit_epoch0100.pt --output_dir my_samples

Outputs
───────
    <output_dir>/samples_ddpm.png   — 10×10 grid via full DDPM (1000 steps)
    <output_dir>/samples_ddim.png   — 10×10 grid via DDIM (25 steps, eta=0)

Both grids contain 10 samples per class (rows = classes 0-9).
The EMA weights stored in the checkpoint are used for both samplers.
"""

import os
import argparse

import torch

from model import DiT
from diffusion import GaussianDiffusion
from utils import EMA, save_image_grid

# ── Model presets ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "small": dict(hidden_size=384, num_heads=6,  depth=6),
    "large": dict(hidden_size=512, num_heads=8,  depth=12),
}

# ── Fixed config ──────────────────────────────────────────────────────────────
IMG_SIZE    = 32
PATCH_SIZE  = 4
IN_CHANNELS = 3
NUM_CLASSES = 10
TIMESTEPS   = 1000

# Sampling config
GUIDANCE_SCALE = 3.0
DDIM_STEPS = 25

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model(device: str, model_size: str = "small") -> DiT:
    cfg = MODEL_CONFIGS[model_size]
    return DiT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        **cfg,
    ).to(device)


def load_ema_model(checkpoint_path: str, device: str, model_size: str = "small") -> DiT:
    """
    Restore a DiT model with EMA weights from a checkpoint file.

    The checkpoint contains both the raw model weights ('model_state_dict')
    and the EMA shadow parameters ('ema_state_dict').  We load the raw weights
    first (needed to initialise the EMA object), then overwrite with the EMA
    shadow params so generation uses the smoothed parameters.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = build_model(device, model_size)
    model.load_state_dict(ckpt["model_state_dict"])

    # Restore and apply EMA weights
    ema = EMA(model)
    ema.load_state_dict(ckpt["ema_state_dict"], device=device)
    ema.copy_to(model)

    model.eval()
    print(
        f"  Loaded epoch {ckpt['epoch']:4d}  "
        f"train_loss={ckpt.get('loss', float('nan')):.4f}  "
        f"(EMA weights active)"
    )
    return model


def make_class_labels(num_per_class: int, device: str) -> torch.Tensor:
    """Return (N,) label tensor: [0]*k + [1]*k + … + [9]*k where k=num_per_class."""
    return torch.arange(NUM_CLASSES, device=device).repeat_interleave(num_per_class)


# ── Sampling routines ─────────────────────────────────────────────────────────

def sample_ddpm(
    model: DiT,
    diffusion: GaussianDiffusion,
    classes: torch.Tensor,
    device: str,
) -> torch.Tensor:
    B = classes.size(0)
    shape = (B, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    print(f"  DDPM sampling {B} images ({TIMESTEPS} steps) …")
    return diffusion.ddpm_sample(
        model, shape, classes,
        guidance_scale=GUIDANCE_SCALE,
        device=device,
    )


def sample_ddim(
    model: DiT,
    diffusion: GaussianDiffusion,
    classes: torch.Tensor,
    device: str,
) -> torch.Tensor:
    B = classes.size(0)
    shape = (B, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    print(f"  DDIM sampling {B} images ({DDIM_STEPS} steps, eta=0) …")
    return diffusion.ddim_sample(
        model, shape, classes,
        guidance_scale=GUIDANCE_SCALE,
        num_steps=DDIM_STEPS,
        eta=0.0,
        device=device,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sample from a trained DiT model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pt checkpoint file (e.g. checkpoints_small/dit_epoch0100.pt)",
    )
    parser.add_argument(
        "--model", choices=["small", "large"], default="small",
        help="Model size preset matching the checkpoint (default: small)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="samples",
        help="Directory to write PNG grids into (default: samples/)",
    )
    parser.add_argument(
        "--num_per_class", type=int, default=10,
        help="Number of samples to generate per class (default: 10)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────────
    model = load_ema_model(args.checkpoint, DEVICE, args.model)
    diffusion = GaussianDiffusion(timesteps=TIMESTEPS).to(DEVICE)
    classes = make_class_labels(args.num_per_class, DEVICE)

    # ── DDPM grid ─────────────────────────────────────────────────────────────
    ddpm_samples = sample_ddpm(model, diffusion, classes, DEVICE)
    ddpm_path = os.path.join(args.output_dir, "samples_ddpm.png")
    save_image_grid(ddpm_samples, ddpm_path, nrow=NUM_CLASSES)

    # ── DDIM grid ─────────────────────────────────────────────────────────────
    ddim_samples = sample_ddim(model, diffusion, classes, DEVICE)
    ddim_path = os.path.join(args.output_dir, "samples_ddim.png")
    save_image_grid(ddim_samples, ddim_path, nrow=NUM_CLASSES)

    print("\nDone.")
    print(f"  DDPM → {ddpm_path}")
    print(f"  DDIM → {ddim_path}")


if __name__ == "__main__":
    main()
