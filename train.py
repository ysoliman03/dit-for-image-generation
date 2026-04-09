"""
train.py — Training loop for DiT on CIFAR-10.

Usage
─────
    # Train the small model (16M params, original run)
    python train.py --model small

    # Train the large model (58M params, new run)
    python train.py --model large

Each model gets its own checkpoint and sample directories:
    checkpoints_small/   samples_small/
    checkpoints_large/   samples_large/

Auto-resumes from the latest checkpoint in the model's directory.
"""

import os
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DiT
from diffusion import GaussianDiffusion
from utils import EMA, save_image_grid

# ── Model presets ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "small": dict(hidden_size=384, num_heads=6,  depth=6,  lr=1e-4),  # ~16M params
    "large": dict(hidden_size=512, num_heads=8,  depth=12, lr=5e-5),  # ~58M params
}

# ── Fixed hyper-parameters ────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

IMG_SIZE      = 32
PATCH_SIZE    = 4
IN_CHANNELS   = 3
NUM_CLASSES   = 10

TIMESTEPS     = 1000
BATCH_SIZE    = 128
EPOCHS        = 200
LR            = 1e-4
WEIGHT_DECAY  = 0.0
EMA_DECAY     = 0.9999
CFG_DROP_PROB = 0.1

GUIDANCE_SCALE = 1.5
DDIM_STEPS     = 100

SAVE_EVERY   = 5
SAMPLE_EVERY = 10


# ── Sampling helper ───────────────────────────────────────────────────────────

def generate_sample_grid(model, ema, diffusion, epoch, device, out_dir, num_per_class=10):
    """Generate 100 samples (10 per class) with EMA weights via DDIM and save."""
    original_sd = copy.deepcopy(model.state_dict())
    ema.copy_to(model)
    model.eval()

    classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(num_per_class)
    shape = (len(classes), IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    samples = diffusion.ddim_sample(
        model, shape, classes,
        guidance_scale=GUIDANCE_SCALE,
        num_steps=DDIM_STEPS,
        eta=0.0,
        device=device,
    )

    path = os.path.join(out_dir, f"samples_epoch{epoch:04d}.png")
    save_image_grid(samples, path, nrow=NUM_CLASSES)

    model.load_state_dict(original_sd)
    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["small", "large"], default="small",
        help="Model size preset: 'small' (~16M) or 'large' (~58M)"
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model].copy()
    lr  = cfg.pop("lr")
    checkpoint_dir = f"checkpoints_{args.model}"
    sample_dir     = f"samples_{args.model}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir,     exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_loader = DataLoader(
        datasets.CIFAR10(root="./data", train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=False, drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DiT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES, **cfg,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : DiT-{args.model}  ({num_params:,} params)")
    print(f"Device : {DEVICE}")

    # ── Diffusion / EMA / optimiser ───────────────────────────────────────────
    diffusion = GaussianDiffusion(timesteps=TIMESTEPS).to(DEVICE)
    ema       = EMA(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Resume from latest checkpoint ─────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    ckpt_files  = sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else []
    resume_path = os.path.join(checkpoint_dir, ckpt_files[-1]) if ckpt_files else None

    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"], device=DEVICE)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = (start_epoch - 1) * (50000 // BATCH_SIZE)
        print(f"Resumed at epoch {start_epoch},  lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            B      = imgs.size(0)

            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE, dtype=torch.long)
            x_t, noise = diffusion.q_sample(imgs, t)

            # CFG dropout
            y = labels.clone()
            y[torch.rand(B, device=DEVICE) < CFG_DROP_PROB] = NUM_CLASSES

            noise_pred = model(x_t, t, y)
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss  += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"[{args.model}] Epoch [{epoch:3d}/{EPOCHS}]  "
                    f"Step [{global_step:6d}]  Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / num_batches
        print(f"── Epoch {epoch:3d} complete  avg_loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e} ──")

        scheduler.step()

        if epoch % SAMPLE_EVERY == 0:
            print(f"  Generating sample grid…")
            generate_sample_grid(model, ema, diffusion, epoch, DEVICE, sample_dir)

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"dit_epoch{epoch:04d}.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "ema_state_dict":       ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 avg_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
