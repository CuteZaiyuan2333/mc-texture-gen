"""Device detection and training hyperparameters."""

from dataclasses import dataclass, field
import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect Intel XPU or fall back to CPU."""
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            name = torch.xpu.get_device_name(0)
            logger.info("Native Intel XPU detected: %s", name)
            return torch.device("xpu")
    except Exception:
        pass

    logger.info("XPU not found, falling back to CPU.")
    return torch.device("cpu")


DEVICE: torch.device = get_device()
XPU_AVAILABLE: bool = DEVICE.type == "xpu"


@dataclass
class Config:
    """Centralised hyperparameters for the diffusion model."""

    # ── Data ──────────────────────────────────────────
    data_root: str = "data"
    image_size: int = 16
    channels: int = 4  # RGBA

    # ── Tokenizer ─────────────────────────────────────
    max_token_len: int = 12

    # ── Model architecture ────────────────────────────
    time_emb_dim: int = 1024
    base_channels: int = 384
    attn_heads: int = 8

    # ── Diffusion ─────────────────────────────────────
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # ── Training ──────────────────────────────────────
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_workers: int = 0
    target_loss: float = 0.005
    save_every: int = 50
    log_every: int = 10
    cfg_dropout: float = 0.15  # classifier-free guidance dropout

    # ── Scheduler ─────────────────────────────────────
    scheduler_t0: int = 100
    scheduler_t_mult: int = 1

    # ── Generation ────────────────────────────────────
    cfg_scale: float = 4.0
    output_upscale: int = 256  # nearest-neighbour upscale for display

    # ── Paths ─────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    # ── Misc ──────────────────────────────────────────
    seed: int = 42
    use_compile: bool = False  # torch.compile (PyTorch >= 2.0)
    val_split: float = 0.05  # fraction of data for validation