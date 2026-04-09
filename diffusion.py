"""
diffusion.py — Gaussian diffusion with linear noise schedule.

Implements:
  - forward diffusion  q(x_t | x_0)
  - DDPM reverse sampling with classifier-free guidance (CFG)
  - DDIM reverse sampling with CFG (deterministic, eta=0)

Noise schedule
──────────────
  β_t  linearly spaced in [β_start, β_end]
  α_t  = 1 − β_t                      (per-step alpha)
  ᾱ_t  = ∏ α_s  for s=0…t             (cumulative alpha)

Forward process
───────────────
  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ N(0,I)

DDPM reverse step  (epsilon prediction)
────────────────────────────────────────
  x_{t-1} = (1/√α_t) · (x_t − β_t/√(1−ᾱ_t) · ε_θ) + σ_t · z
  σ_t = √β_t,  z ~ N(0,I) for t>0, else z=0

DDIM reverse step  (eta=0, deterministic)
──────────────────────────────────────────
  x̂_0   = (x_t − √(1−ᾱ_t)·ε_θ) / √ᾱ_t
  x_{τ'} = √ᾱ_{τ'} · x̂_0 + √(1−ᾱ_{τ'}) · ε_θ
"""

import torch


class GaussianDiffusion:
    """
    Wraps the noise schedule and provides forward/reverse diffusion utilities.
    All tensors live in float64 for schedule precision; cast to float32 when
    interacting with model inputs/outputs.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps

        # ── Noise schedule ────────────────────────────────────────────────────
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precomputed helpers
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def to(self, device):
        """Move all schedule tensors to the given device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self

    # ── Forward diffusion ─────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Sample x_t given x_0 and timestep t.

        x0   : (B, C, H, W) clean image in [-1, 1]
        t    : (B,) integer timesteps
        Returns (x_t, noise) — both (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_t * noise
        return x_t, noise

    # ── DDPM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_sample(
        self,
        model,
        shape: tuple,
        classes: torch.Tensor,
        guidance_scale: float = 3.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Full DDPM ancestral sampling with classifier-free guidance.

        model          : DiT (in eval mode)
        shape          : (B, C, H, W) shape of the desired output
        classes        : (B,) class labels 0-9
        guidance_scale : CFG weight (1.0 = conditional only, no guidance)
        Returns (B, C, H, W) tensor clamped to [-1, 1].
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        # Null-class tokens for the unconditional branch (index 10)
        null_y = torch.full((B,), 10, dtype=torch.long, device=device)

        model.eval()
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            # ── CFG: combine conditional & unconditional predictions ──────────
            eps_cond = model(x, t_batch, classes)
            eps_uncond = model(x, t_batch, null_y)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # ── DDPM reverse step ─────────────────────────────────────────────
            alpha_t = self.alphas[t]
            beta_t = self.betas[t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t]

            # Posterior mean: (1/√α_t) * (x_t − β_t/√(1−ᾱ_t) * ε_θ)
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * eps
            )

            if t > 0:
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x)
                x = mean + sigma_t * z
            else:
                x = mean

        return x.clamp(-1.0, 1.0)

    # ── DDIM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple,
        classes: torch.Tensor,
        guidance_scale: float = 3.0,
        num_steps: int = 25,
        eta: float = 0.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        DDIM sampling with CFG.

        Uses a uniform sub-sequence of timesteps:
            [0, stride, 2*stride, …, T-stride]  where stride = T // num_steps
        e.g. with T=1000, num_steps=25 → [0, 40, 80, …, 960], reversed for sampling.

        eta=0.0 → fully deterministic (no stochastic noise injected).

        Returns (B, C, H, W) tensor clamped to [-1, 1].
        """
        B = shape[0]
        x = torch.randn(shape, device=device)

        # Build the timestep sub-sequence and reverse it (noisy → clean)
        stride = self.timesteps // num_steps                    # 40
        timesteps_asc = list(range(0, self.timesteps, stride))  # [0, 40, …, 960]
        timesteps_desc = list(reversed(timesteps_asc))          # [960, …, 40, 0]

        # Null class for CFG unconditional branch
        null_y = torch.full((B,), 10, dtype=torch.long, device=device)

        model.eval()
        for i, t in enumerate(timesteps_desc):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)

            # ── CFG ────────────────────────────────────────────────────────────
            eps_cond = model(x, t_batch, classes)
            eps_uncond = model(x, t_batch, null_y)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            alpha_cumprod_t = self.alpha_cumprod[t]

            # Predict x_0 from x_t and ε_θ, then clamp to valid range
            x0_pred = (
                x - torch.sqrt(1.0 - alpha_cumprod_t) * eps
            ) / torch.sqrt(alpha_cumprod_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # ᾱ at the previous (cleaner) timestep
            if i < len(timesteps_desc) - 1:
                t_prev = timesteps_desc[i + 1]
                alpha_cumprod_prev = self.alpha_cumprod[t_prev].float()
            else:
                # Final step: target is the clean image (ᾱ = 1 → no noise)
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            if eta == 0.0:
                # Deterministic DDIM update:
                # x_{τ'} = √ᾱ_{τ'} · x̂_0 + √(1 − ᾱ_{τ'}) · ε_θ
                x = (
                    torch.sqrt(alpha_cumprod_prev) * x0_pred
                    + torch.sqrt(1.0 - alpha_cumprod_prev) * eps
                )
            else:
                # Stochastic DDIM (general η)
                sigma_t = (
                    eta
                    * torch.sqrt(
                        (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
                    )
                    * torch.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_prev)
                )
                noise_dir = torch.sqrt(1.0 - alpha_cumprod_prev - sigma_t ** 2) * eps
                x = (
                    torch.sqrt(alpha_cumprod_prev) * x0_pred
                    + noise_dir
                    + sigma_t * torch.randn_like(x)
                )

        return x.clamp(-1.0, 1.0)
