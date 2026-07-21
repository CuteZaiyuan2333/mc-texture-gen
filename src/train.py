"""Training script for the Minecraft texture diffusion model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
# Helpers
# ═══════════════════════════════════════════════════════════════════


def save_checkpoint(
    model: nn.Module,
    tokenizer: Tokenizer,
    config: Config,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save a full checkpoint (model + tokenizer + config)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_vocab": tokenizer.vocab,
            "tokenizer_max_len": tokenizer.max_len,
            "config": config,
            "epoch": epoch,
            "loss": loss,
        },
        str(path),
    )
    logger.info("Checkpoint saved → %s", path)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def train(config: Config | None = None) -> None:
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

    logger.info("Training on %s | %d params", DEVICE, sum(p.numel() for p in model.parameters()))

    # ── Training loop ─────────────────────────────────────────
    best_val_loss = float("inf")
    epoch = 0

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

        # Scheduler: step on epoch (not loss — this was a bug)
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
                model,
                tokenizer,
                config,
                epoch,
                avg_train_loss,
                Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch:04d}.pth",
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model,
                tokenizer,
                config,
                epoch,
                avg_train_loss,
                Path(config.checkpoint_dir) / "best_model.pth",
            )

        # ---- Early stopping ----
        if avg_train_loss < config.target_loss:
            logger.info("Target loss %.5f reached at epoch %d!", config.target_loss, epoch)
            save_checkpoint(
                model,
                tokenizer,
                config,
                epoch,
                avg_train_loss,
                Path(config.checkpoint_dir) / "final_model.pth",
            )
            break

        epoch += 1


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()