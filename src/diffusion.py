"""DDPM diffusion process: forward noising and reverse sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, Config


class Diffusion(nn.Module):
    """Manages the DDPM schedule and provides forward / reverse helpers."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_steps = config.n_steps
        self.cfg_scale = config.cfg_scale

        # Linear beta schedule
        beta = torch.linspace(
            config.beta_start, config.beta_end, config.n_steps
        )
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

        # Register as buffers so they follow the model's device automatically
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    # ── Forward diffusion (training) ──────────────────────────────

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward-diffuse a clean image x0 to step t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a_cp = self.alpha_cumprod[t][:, None, None, None].sqrt()
        sqrt_1m_a_cp = (1.0 - self.alpha_cumprod[t])[:, None, None, None].sqrt()
        return sqrt_a_cp * x0 + sqrt_1m_a_cp * noise

    # ── Reverse diffusion (sampling) ──────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        cond_ids: torch.Tensor,
        uncond_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full reverse process from pure noise to clean image (CFG)."""
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i], device=x.device).long()

            # Classifier-Free Guidance: batch cond + uncond together
            x_combined = torch.cat([x, x], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            label_combined = torch.cat([cond_ids, uncond_ids], dim=0)

            noise_pred_all = model(x_combined, t_combined, label_combined)
            eps_cond, eps_uncond = noise_pred_all.chunk(2, dim=0)

            eps = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)

            # DDPM update
            a = self.alpha[i]
            a_cp = self.alpha_cumprod[i]
            beta_t = self.beta[i]

            x = (1.0 / a.sqrt()) * (
                x - ((1.0 - a) / (1.0 - a_cp).sqrt()) * eps
            )
            if i > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)

        return x