"""Conditional U-Net with FiLM conditioning and self-attention."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, Config


# ═══════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════


class SinusoidalPosEmbedding(nn.Module):
    """Sinusoidal position / time embedding (Transformer-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long tensor of timestep indices → (B, dim) float."""
        half = self.dim // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half, device=t.device).float() / half)
        )
        t_float = t.float().unsqueeze(-1)  # (B, 1)
        args = t_float * inv_freq.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual conv block with FiLM conditioning on time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.film_mlp = nn.Linear(time_emb_dim, out_ch * 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)

        # FiLM: inject global conditioning
        film = self.film_mlp(t_emb)[:, :, None, None]  # (B, 2*C, 1, 1)
        gamma, beta = film.chunk(2, dim=1)
        h = h * (1.0 + gamma) + beta

        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (flattened H×W)."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        return x + attn_out.transpose(1, 2).view(B, C, H, W)


# ═══════════════════════════════════════════════════════════════════
# Conditional U-Net
# ═══════════════════════════════════════════════════════════════════


class ConditionalUNet(nn.Module):
    """U-Net conditioned on text tokens (via embedding) and diffusion timestep."""

    def __init__(
        self,
        vocab_size: int,
        time_emb_dim: int = 512,
        base_channels: int = 128,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Condition embeddings
        self.label_emb = nn.Embedding(vocab_size + 1, time_emb_dim)
        self.pos_emb = SinusoidalPosEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        bc = base_channels
        self.inc = nn.Conv2d(4, bc, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(bc, bc * 2, time_emb_dim)
        self.down2 = ResidualBlock(bc * 2, bc * 4, time_emb_dim)

        # Bottleneck
        self.mid1 = ResidualBlock(bc * 4, bc * 4, time_emb_dim)
        self.attn = AttentionBlock(bc * 4, attn_heads)
        self.mid2 = ResidualBlock(bc * 4, bc * 4, time_emb_dim)

        # Decoder
        self.up1 = ResidualBlock(bc * 4 + bc * 2, bc * 2, time_emb_dim)
        self.up2 = ResidualBlock(bc * 2 + bc, bc, time_emb_dim)
        self.outc = nn.Conv2d(bc, 4, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """x: (B,4,H,W) noisy image, t: (B,) timestep, token_ids: (B, max_len)"""
        # ── Combine time + text embeddings ──
        mask = (token_ids != 0).float()  # (B, max_len)
        l_embs = self.label_emb(token_ids)  # (B, max_len, time_emb_dim)
        l_emb = (l_embs * mask.unsqueeze(-1)).sum(dim=1)  # sum over tokens

        t_emb = self.pos_emb(t)  # (B, time_emb_dim)
        emb = self.time_mlp(t_emb + l_emb)

        # ── Encoder ──
        x1 = self.inc(x)  # (B, bc, H, W)
        x2 = self.down1(x1, emb)  # (B, bc*2, H, W)
        x3 = self.down2(F.avg_pool2d(x2, 2), emb)  # (B, bc*4, H/2, W/2)

        # ── Bottleneck ──
        x3 = self.mid1(x3, emb)
        x3 = self.attn(x3)
        x3 = self.mid2(x3, emb)

        # ── Decoder ──
        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)  # skip connection
        x = self.up1(x, emb)
        x = torch.cat([x, x1], dim=1)  # skip connection
        x = self.up2(x, emb)

        return self.outc(x)