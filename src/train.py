"""Training script for the Minecraft texture diffusion model.

Usage:
    python -m src.train                # fresh training
    python -m src.train --resume PATH  # resume from checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .config import Config, DEVICE, XPU_AVAILABLE
from .dataset import MCTextureDataset
from .tokenizer import Tokenizer
from .networks import ConditionalUNet
from .diffusion import Diffusion

# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("train")


# ═══════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════


def save_checkpoint(
    model: nn.Module,
    tokenizer: Tokenizer,
    config: Config,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    epoch: int,
    loss: float,
    best_val_loss: float,
    path: str | Path,
) -> None:
    """Save a full checkpoint (model + tokenizer + optimiser + config)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "tokenizer_max_len": tokenizer.max_len,
        "config": config,
        "epoch": epoch,
        "loss": loss,
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, str(path))
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint and restore model, optimizer, scheduler, scaler.
    
    Handles both old-format (no optimizer) and new-format checkpoints.
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        logger.info("Old checkpoint — optimizer state not restored")
    
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    else:
        logger.info("Old checkpoint — scheduler state not restored")
    
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    
    logger.info("Resumed from %s (epoch %d, loss %.5f)", path, ckpt["epoch"], ckpt["loss"])
    return ckpt


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def train(config: Config | None = None, resume_from: str | None = None) -> None:
    """Train the diffusion model, optionally resuming from a checkpoint."""
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)

    # ── Dataset ───────────────────────────────────────────────
    dataset = MCTextureDataset(root_dir=config.data_root)
    if len(dataset) == 0:
        logger.error("No textures found in %s — aborting.", config.data_root)
        return

    train_ds, val_ds = dataset.split(val_fraction=config.val_split, seed=config.seed)
    logger.info(
        "Split: %d train / %d val (%.0f%% val)",
        len(train_ds),
        len(val_ds),
        config.val_split * 100,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = Tokenizer(dataset.texts, max_len=config.max_token_len)
    logger.info("Tokenizer: %s", tokenizer)

    # ── Model & diffusion ─────────────────────────────────────
    model = ConditionalUNet(
        vocab_size=len(tokenizer),
        time_emb_dim=config.time_emb_dim,
        base_channels=config.base_channels,
        attn_heads=config.attn_heads,
    ).to(DEVICE)

    diffusion = Diffusion(config).to(DEVICE)

    # torch.compile (PyTorch ≥ 2.0)
    if config.use_compile and hasattr(torch, "compile"):
        logger.info("Enabling torch.compile …")
        model = torch.compile(model)  # type: ignore[assignment]

    # ── Optimizer & scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_t0, T_mult=config.scheduler_t_mult
    )

    # Mixed-precision scaler (XPU only; CPU AMP is less beneficial)
    scaler = torch.amp.GradScaler("xpu") if XPU_AVAILABLE else None

    # ── Resume or start fresh ─────────────────────────────────
    best_val_loss: float = float("inf")
    start_epoch: int = 0

    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, model, optimizer, scheduler, scaler, DEVICE)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        # Restore tokenizer from checkpoint
        tokenizer = Tokenizer(vocab=ckpt["tokenizer_vocab"], max_len=ckpt["tokenizer_max_len"])

    logger.info("Training on %s | %d params", DEVICE, sum(p.numel() for p in model.parameters()))

    # ── Training loop ─────────────────────────────────────────
    epoch = start_epoch

    while True:
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for images, texts in train_loader:
            images = images.to(DEVICE)
            token_ids = tokenizer.encode(texts).to(DEVICE)

            # CFG dropout
            if torch.rand(1).item() < config.cfg_dropout:
                token_ids = torch.zeros_like(token_ids)

            t = torch.randint(0, config.n_steps, (images.shape[0],), device=DEVICE)
            noise = torch.randn_like(images)
            noisy = diffusion.q_sample(images, t, noise)

            optimizer.zero_grad()
            if XPU_AVAILABLE:
                with torch.amp.autocast(device_type="xpu"):
                    pred = model(noisy, t, token_ids)
                    loss = nn.MSELoss()(pred, noise)
                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                pred = model(noisy, t, token_ids)
                loss = nn.MSELoss()(pred, noise)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(DEVICE)
                token_ids = tokenizer.encode(texts).to(DEVICE)
                t = torch.randint(0, config.n_steps, (images.shape[0],), device=DEVICE)
                noise = torch.randn_like(images)
                noisy = diffusion.q_sample(images, t, noise)
                pred = model(noisy, t, token_ids)
                val_loss += nn.MSELoss()(pred, noise).item()

        avg_val_loss = val_loss / len(val_loader)

        # Scheduler: step on epoch
        scheduler.step(epoch)

        # ---- Logging ----
        if epoch % config.log_every == 0:
            logger.info(
                "Epoch %4d | Train: %.5f | Val: %.5f | LR: %.2e",
                epoch,
                avg_train_loss,
                avg_val_loss,
                optimizer.param_groups[0]["lr"],
            )

        # ---- Checkpoint ----
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, tokenizer, config,
                optimizer, scheduler, scaler,
                epoch, avg_train_loss, best_val_loss,
                Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pth",
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model, tokenizer, config,
                optimizer, scheduler, scaler,
                epoch, avg_train_loss, best_val_loss,
                Path(config.checkpoint_dir) / "best_model.pth",
            )

        # ---- Early stopping ----
        if avg_train_loss < config.target_loss:
            logger.info("Target loss %.5f reached at epoch %d!", config.target_loss, epoch)
            save_checkpoint(
                model, tokenizer, config,
                optimizer, scheduler, scaler,
                epoch, avg_train_loss, best_val_loss,
                Path(config.checkpoint_dir) / "final_model.pth",
            )
            break

        epoch += 1


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Minecraft texture diffusion model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_from=args.resume)