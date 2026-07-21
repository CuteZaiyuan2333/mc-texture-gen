"""Minecraft texture dataset loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {"block", "item", "particle", "effect", "mob_effect"}


class MCTextureDataset(Dataset):
    """Scans a directory tree for 16×16 RGBA Minecraft textures and pairs
    each with a text label built from the folder hierarchy and filename."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.image_paths: List[Path] = []
        self.texts: List[str] = []

        logger.info("Scanning texture directory: %s", self.root_dir)

        self._scan()

        # Build transform once (was previously inside __getitem__)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Normalise RGBA channels to [-1, 1]
                transforms.Normalize(
                    (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
                ),
            ]
        )

        logger.info("Scan complete: %d valid 16×16 textures found.", len(self))

    # ── scanning ─────────────────────────────────────────────────

    def _scan(self) -> None:
        for png_path in self.root_dir.rglob("*.png"):
            if png_path.suffix.lower() != ".png":
                continue
            if png_path.suffix == ".mcmeta":  # belt-and-braces
                continue

            try:
                with Image.open(png_path) as img:
                    if img.size != (16, 16):
                        continue
            except Exception:
                continue

            self.image_paths.append(png_path)
            self.texts.append(self._build_label(png_path))

    def _build_label(self, path: Path) -> str:
        """Derive a text label: pack_name + category + clean_filename."""
        relative = path.relative_to(self.root_dir)
        parts = list(relative.parts)

        pack_name = parts[0] if parts else "vanilla"

        category = "unknown"
        for p in parts:
            if p.lower() in ALLOWED_CATEGORIES:
                category = p.lower()
                break

        clean_name = (
            path.stem.replace("_", " ").replace("-", " ")
        )
        return f"{pack_name} {category} {clean_name}".lower()

    # ── Dataset protocol ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img = Image.open(self.image_paths[idx]).convert("RGBA")

        # Horizontal flip is the only augmentation safe for pixel-art
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return self.transform(img), self.texts[idx]

    # ── helpers ──────────────────────────────────────────────────

    def split(
        self, val_fraction: float = 0.05, seed: int = 42
    ) -> Tuple[MCTextureDataset, MCTextureDataset]:
        """Split into train / val subsets (shallow wrapper around random_split)."""
        n_val = max(1, int(len(self) * val_fraction))
        n_train = len(self) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(self, [n_train, n_val], generator=generator)
        return train_ds, val_ds  # type: ignore[return-value]