"""Analyse the training-data directory: size distribution, categories, etc."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PIL import Image

# Match the categories used by MCTextureDataset
ALLOWED_CATEGORIES = {"block", "item", "particle", "effect", "mob_effect"}


def analyze(root: str = "training_data") -> None:
    root_path = Path(root)
    if not root_path.exists():
        print(f"❌ Directory not found: {root_path}")
        sys.exit(1)

    total_pngs = 0
    size_counter: Counter = Counter()
    category_counter: Counter = Counter()
    sample_paths: list[Path] = []

    for png_path in root_path.rglob("*.png"):
        # Skip .mcmeta files
        if png_path.suffix.lower() != ".png":
            continue

        total_pngs += 1

        if len(sample_paths) < 10:
            sample_paths.append(png_path)

        try:
            with Image.open(png_path) as img:
                size_counter[img.size] += 1
        except Exception:
            continue

        # Extract category from path
        for cat in ALLOWED_CATEGORIES:
            if cat in png_path.parts:
                category_counter[cat] += 1
                break

    print("=" * 50)
    print("  Dataset Analysis Report")
    print("=" * 50)
    print(f"  Total PNG files: {total_pngs}")

    print("\n  Size Distribution (top 5):")
    for size, count in size_counter.most_common(5):
        pct = (count / total_pngs * 100) if total_pngs else 0
        print(f"    {size[0]}×{size[1]}: {count} ({pct:.1f}%)")

    print("\n  Category Distribution:")
    for cat, count in category_counter.most_common():
        pct = (count / total_pngs * 100) if total_pngs else 0
        print(f"    {cat}: {count} ({pct:.1f}%)")

    print("\n  Path Structure Samples:")
    for p in sample_paths:
        print(f"    {p}")


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "training_data"
    analyze(root_dir)