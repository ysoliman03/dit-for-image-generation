"""
utils.py — EMA class and image saving utilities.
"""

import torch
import torchvision
import numpy as np
from PIL import Image


class EMA:
    """
    Exponential Moving Average of model parameters.
    Maintains a shadow copy of parameters updated with:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        # Initialise shadow params as copies of the current model params
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters — call once per optimiser step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def copy_to(self, model):
        """Overwrite model parameters with the EMA shadow values."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        """Return shadow params as a plain dict (moved to CPU for serialisation)."""
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device="cpu"):
        """Restore shadow params from a previously saved state_dict."""
        self.shadow = {k: v.to(device) for k, v in state_dict.items()}


def save_image_grid(images, path, nrow=10):
    """
    Save a grid of images as a PNG file.

    images : (N, C, H, W) float tensor in [-1, 1]
    path   : destination file path
    nrow   : number of images per row in the grid
    """
    images = images.clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0  # rescale to [0, 1]
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(path)
    print(f"  Saved image grid -> {path}")
