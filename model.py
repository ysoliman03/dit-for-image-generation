"""
model.py — Diffusion Transformer (DiT) for class-conditional image generation.

Architecture overview
─────────────────────
1. Patchify     : Conv2d(3→384, k=4, s=4) turns 32×32 image into 8×8=64 patch tokens
2. Pos-embed    : learnable (1, 64, 384) added to patch tokens
3. Conditioning : sinusoidal(t) → MLP → t_emb
                  nn.Embedding(11, 384) → class_emb   (index 10 = CFG null)
                  c = t_emb + class_emb
4. 6 × DiTBlock : adaLN-Zero (scale/shift from c, zero-init gate) + MHSA + FFN
5. Final layer  : adaLN → LayerNorm → Linear(384 → 48) → unpatchify → (B,3,32,32)
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ──────────────────────────────────────────────────────────────────────────────

def get_sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for a batch of integer timesteps.

    t   : (B,) int tensor
    dim : embedding dimensionality (must be even)
    Returns (B, dim) float32 tensor.
    """
    assert dim % 2 == 0, "Embedding dim must be even"
    half = dim // 2
    # Frequency bands: exp(-log(10000) * k / (half-1)) for k in [0, half)
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / (half - 1)
    )
    args = t[:, None].float() * freqs[None, :]   # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class TimestepEmbedder(nn.Module):
    """Timestep → sinusoidal encoding → 2-layer MLP → (B, hidden_size)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(get_sinusoidal_embedding(t, self.hidden_size))


# ──────────────────────────────────────────────────────────────────────────────
# Attention & FFN
# ──────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention with QKV projection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project and split into Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        )
        q, k, v = qkv.unbind(0)       # each (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        # Aggregate values and project
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear, 4× hidden expansion."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# DiT Block with adaLN-Zero
# ──────────────────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Single DiT transformer block.

    Conditioning vector c (B, D) is projected to 6 modulation scalars per token:
        [shift1, scale1, gate1, shift2, scale2, gate2]
    Applied as:
        x = x + gate1 * Attn( LayerNorm(x) * (1+scale1) + shift1 )
        x = x + gate2 * FFN(  LayerNorm(x) * (1+scale2) + shift2 )

    The adaLN projection is zero-initialised so every block starts as identity.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size)

        # adaLN-Zero: SiLU gate + linear projecting c → 6 modulation values
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init so all gates start at 0 → block is identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # shift, scale: (B, D) → broadcast over sequence dim N
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Unpack the 6 modulation params
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Attention branch
        x = x + gate1.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift1, scale1)
        )
        # FFN branch
        x = x + gate2.unsqueeze(1) * self.ff(
            self.modulate(self.norm2(x), shift2, scale2)
        )
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full DiT model
# ──────────────────────────────────────────────────────────────────────────────

class DiT(nn.Module):
    """
    Diffusion Transformer for 32×32 RGB class-conditional generation.

    Parameters
    ----------
    img_size    : input image side length (32)
    patch_size  : side of each square patch (4) → 8×8=64 tokens
    in_channels : image channels (3)
    hidden_size : token embedding dimension (384)
    num_heads   : attention heads per block (6)
    depth       : number of DiT blocks (6)
    num_classes : number of class labels (10); index num_classes is the CFG null class
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 384,
        num_heads: int = 6,
        depth: int = 6,
        num_classes: int = 10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        num_patches = (img_size // patch_size) ** 2  # 64

        # ── Patch embedding ────────────────────────────────────────────────────
        # Non-overlapping 4×4 convolution splits the image into patch tokens
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

        # ── Positional embedding (learnable) ───────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        # ── Conditioning ───────────────────────────────────────────────────────
        self.t_embedder = TimestepEmbedder(hidden_size)
        # num_classes + 1 to include the null / unconditional token at index 10
        self.class_embed = nn.Embedding(num_classes + 1, hidden_size)

        # ── Transformer blocks ─────────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )

        # ── Final output layer ─────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # adaLN for final layer (shift + scale only, no gate)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Projects each token to a flat patch: patch_size² × channels = 4²×3 = 48
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        # Zero-init final layer so predictions start near zero
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    # ── Spatial helpers ────────────────────────────────────────────────────────

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape flat patch tokens back to a spatial image.

        x : (B, N, patch_size² × C)  →  (B, C, H, W)
        where N = (H/p)*(W/p), p = patch_size.
        """
        B, N, _ = x.shape
        p = self.patch_size
        c = self.in_channels
        h = w = int(N ** 0.5)          # h = w = 8 for 32×32, p=4

        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, c, h, p, w, p)
        x = x.reshape(B, c, h * p, w * p)
        return x

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        x : (B, C, H, W)  noisy image in [-1, 1]
        t : (B,)           integer diffusion timestep
        y : (B,)           class label; use num_classes (10) for unconditional
        Returns (B, C, H, W) predicted noise.
        """
        # 1. Patchify: (B, C, H, W) → (B, N, hidden_size)
        x = self.patch_embed(x)           # (B, hidden_size, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_size)
        x = x + self.pos_embed            # add learnable position embedding

        # 2. Build conditioning vector: t_emb + class_emb  →  (B, hidden_size)
        c = self.t_embedder(t) + self.class_embed(y)

        # 3. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Final layer with adaLN
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)   # (B, D) each
        x = self.final_norm(x)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_proj(x)            # (B, N, p*p*C)

        # 5. Unpatchify → (B, C, H, W)
        return self.unpatchify(x)
