"""Texture generation / sampling from a trained checkpoint."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import Config, DEVICE
from .tokenizer import Tokenizer
from .networks import ConditionalUNet
from .diffusion import Diffusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("generate")


def load_checkpoint(path: str | Path) -> tuple:
    """Load model, tokenizer, and config from a checkpoint."""
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)

    # Reconstruct tokenizer
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab = ckpt["tokenizer_vocab"]
    tokenizer.vocab_size = len(tokenizer.vocab)
    tokenizer.max_len = ckpt.get("tokenizer_max_len", 10)

    # Config
    config = ckpt.get("config", Config())

    # Model
    model = ConditionalUNet(
        vocab_size=len(tokenizer),
        time_emb_dim=config.time_emb_dim,
        base_channels=config.base_channels,
        attn_heads=config.attn_heads,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(
        "Loaded checkpoint from epoch %d (loss %.5f)",
        ckpt.get("epoch", -1),
        ckpt.get("loss", float("nan")),
    )
    return model, tokenizer, config


@torch.no_grad()
def sample(
    prompt: str,
    checkpoint_path: str | Path,
    output_name: str | Path = "generated_texture.png",
    cfg_scale: float | None = None,
    seed: int = 42,
) -> None:
    """Generate a single texture from a text prompt."""
    model, tokenizer, config = load_checkpoint(checkpoint_path)

    if cfg_scale is not None:
        diffusion = Diffusion(config)
        diffusion.cfg_scale = cfg_scale
    else:
        diffusion = Diffusion(config)

    diffusion.to(DEVICE)

    # Tokenize
    cond_ids = tokenizer.encode_single(prompt).to(DEVICE)
    uncond_ids = tokenizer.unconditional(batch_size=1).to(DEVICE)

    torch.manual_seed(seed)
    logger.info("Generating: '%s' (cfg=%.1f, seed=%d)", prompt, diffusion.cfg_scale, seed)

    # Start from pure noise
    x = torch.randn((1, 4, 16, 16), device=DEVICE)

    x = diffusion.p_sample(model, x, cond_ids, uncond_ids)

    # Decode: [-1, 1] → [0, 255] RGBA
    x = (x.clamp(-1, 1) + 1) / 2
    x = x.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = Image.fromarray((x * 255).astype(np.uint8), "RGBA")

    # Upscale with nearest-neighbour for pixel-art look
    img = img.resize((config.output_upscale, config.output_upscale), Image.NEAREST)

    output_path = Path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info("Saved → %s", output_path)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    usage = (
        "Usage: python -m src.generate <prompt> [checkpoint] [output] [--cfg N] [--seed N]"
    )
    args = sys.argv[1:]

    if not args:
        print(usage)
        sys.exit(1)

    prompt = args[0]
    ckpt = args[1] if len(args) > 1 else "checkpoints/best_model.pth"
    output = args[2] if len(args) > 2 else "outputs/generated_texture.png"

    cfg = None
    seed = 42
    i = 3
    while i < len(args):
        if args[i] == "--cfg" and i + 1 < len(args):
            cfg = float(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        else:
            i += 1

    sample(prompt, ckpt, output, cfg, seed)