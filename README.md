# 🎮 MC Texture Gen

> Diffusion-based Minecraft texture generator — generate 16×16 pixel-art textures from text prompts.

**mc-texture-gen** trains a Conditional U-Net with a DDPM (Denoising Diffusion Probabilistic Model) to produce Minecraft-style RGBA textures. Given a text prompt like *"vanilla block stone"* it generates a matching 16×16 pixel-art texture, upscaled with nearest-neighbour filtering for sharp display.

## ✨ Features

- **DDPM diffusion** — 1000-step denoising process with linear beta schedule
- **Conditional U-Net** — FiLM-conditioned residual blocks + self-attention bottleneck
- **Classifier-Free Guidance (CFG)** — controllable trade-off between fidelity and diversity
- **Intel XPU / CPU** — native Intel GPU support via `torch.xpu`; falls back to CPU automatically
- **Cosine warm-restart scheduler** — helps escape local minima during training
- **Mixed precision** — automatic FP16 on XPU via `torch.amp`
- **Checkpoint resume** — full checkpoint includes model weights, tokenizer vocab, and config
- **Validation split** — monitors overfitting with a held-out validation set

## 📁 Project Structure

```
mc-texture-gen/
├── src/                        # Core source code
│   ├── __init__.py             # Package exports
│   ├── config.py               # Device detection + hyperparameter dataclass
│   ├── dataset.py              # MCTextureDataset (scans directory tree)
│   ├── tokenizer.py            # Text tokenizer (word → token-id)
│   ├── networks.py             # ConditionalUNet + building blocks
│   ├── diffusion.py            # DDPM forward/reverse process
│   ├── train.py                # Training loop
│   └── generate.py             # Texture generation / sampling
├── scripts/
│   └── analyze_data.py         # Dataset statistics utility
├── checkpoints/                # Saved model checkpoints (.gitignored)
├── outputs/                    # Generated textures (.gitignored)
├── training_data/              # Your texture dataset (.gitignored)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

- **Python ≥ 3.10** required
- **PyTorch ≥ 2.0** — install the Intel XPU build from [Intel's PyTorch extension](https://pytorch.org/get-started/locally/) if you have an Intel GPU, or standard PyTorch for CPU-only

### 2. Prepare training data

Place your Minecraft textures in `training_data/` following this hierarchy:

```
training_data/
├── vanilla/
│   ├── block/
│   │   ├── stone.png         # 16×16 RGBA
│   │   ├── dirt.png
│   │   └── ...
│   ├── item/
│   │   ├── diamond.png
│   │   └── ...
│   └── particle/
│       └── ...
├── my_resource_pack/
│   ├── block/
│   └── item/
└── ...
```

**Requirements:**
- Must be **16×16 pixels** (non-square images are skipped)
- **PNG format** (RGBA recommended)
- Folder names are used to construct text labels: `{pack_name} {category} {filename}`

To check your dataset:

```bash
python scripts/analyze_data.py training_data
```

### 3. Train

```bash
python -m src.train
```

The training loop will:
- Build a vocabulary from all texture labels
- Split data into train/validation (95%/5% by default)
- Save checkpoints every 50 epochs to `checkpoints/`
- Save the best model (by validation loss) to `checkpoints/best_model.pth`
- Stop when training loss drops below the target threshold

### 4. Generate textures

```bash
# Basic usage
python -m src.generate "vanilla block stone" checkpoints/best_model.pth

# With custom CFG scale and seed
python -m src.generate "vanilla item diamond" checkpoints/best_model.pth outputs/diamond.png --cfg 7.0 --seed 123
```

## ⚙️ Configuration

All hyperparameters live in the `Config` dataclass in [src/config.py](src/config.py). You can override them programmatically:

```python
from src.config import Config
from src.train import train

config = Config(
    batch_size=128,
    learning_rate=2e-4,
    target_loss=0.003,
    cfg_scale=5.0,
    val_split=0.1,
)
train(config)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `n_steps` | 1000 | Number of diffusion steps |
| `time_emb_dim` | 512 | Time/text embedding dimension |
| `base_channels` | 128 | U-Net base channel count |
| `cfg_dropout` | 0.15 | CFG text dropout probability |
| `cfg_scale` | 4.0 | Classifier-Free Guidance scale |
| `target_loss` | 0.005 | Early-stopping MSE threshold |
| `val_split` | 0.05 | Validation set fraction |
| `use_compile` | False | Enable `torch.compile` (PyTorch ≥ 2.0) |

## 🧠 Architecture

### Diffusion Model (DDPM)

- **Forward process**: gradually adds Gaussian noise over 1000 steps
- **Reverse process**: the U-Net predicts the noise at each step, iteratively denoising
- **Linear beta schedule**: β ∈ [1e-4, 0.02]

### Conditional U-Net

```
Input (4×16×16) ──→ inc ──→ down1 ──→ down2 ──→ mid1 ──→ attn ──→ mid2 ──→ up1 ──→ up2 ──→ outc ──→ Output (4×16×16)
                      ↑         ↑          ↑                                            │         │
                      │         └──────────┼────────────────────────────────────────────┘         │
                      └────────────────────┼──────────────────────────────────────────────────────┘
                                           │
                              Time Emb + Text Emb (FiLM)
```

- **ResidualBlock** with FiLM conditioning — time embedding modulates features via learned scale/bias
- **AttentionBlock** at bottleneck — 4-head self-attention over spatial positions
- **Sinusoidal time embedding** — Transformer-style positional encoding
- **Skip connections** — encoder features concatenated to decoder

### Tokenizer

- Word-level tokenization from texture labels
- Index 0 reserved for padding / unconditional (CFG)
- Configurable `max_token_len` (default: 10)

## 📊 Expected Training

| Metric | Typical value |
|--------|--------------|
| Epochs to converge | 500–2000 |
| Final train loss | < 0.005 |
| Time per epoch (CPU) | ~30–60 s (depends on dataset size) |
| Time per epoch (XPU) | ~5–15 s |

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No textures found" | Check `training_data/` exists with 16×16 PNGs |
| CUDA/XPU not detected | Install Intel XPU PyTorch or use CPU fallback |
| Out of memory | Reduce `batch_size` in Config |
| Loss not decreasing | Try reducing `learning_rate` or increasing `time_emb_dim` |
| Generated textures blurry | Increase `cfg_scale` (e.g. 7.0–10.0) |

## 📄 License

MIT

## 🛠️ Tech Stack

- **PyTorch 2.0+** — deep learning framework (Intel XPU & CPU compatible)
- **torchvision** — image transforms
- **Pillow** — image I/O
- **NumPy** — array operations