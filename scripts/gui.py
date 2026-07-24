"""Tkinter GUI for interactively testing the Minecraft texture diffusion model.

Usage:
    python scripts/gui.py
    python scripts/gui.py checkpoints/best_model.pth  # pre-load a checkpoint
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageTk

# ── Ensure the project root is on sys.path ────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Config, DEVICE  # noqa: E402
from src.diffusion import Diffusion  # noqa: E402
from src.networks import ConditionalUNet  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load model, tokenizer, and config from a checkpoint."""
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)

    tokenizer = Tokenizer(vocab=ckpt["tokenizer_vocab"], max_len=ckpt.get("tokenizer_max_len", 10))
    config = ckpt.get("config", Config())

    model = ConditionalUNet(
        vocab_size=len(tokenizer),
        time_emb_dim=config.time_emb_dim,
        base_channels=config.base_channels,
        attn_heads=config.attn_heads,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = Diffusion(config).to(DEVICE)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "diffusion": diffusion,
        "epoch": ckpt.get("epoch", -1),
        "loss": ckpt.get("loss", float("nan")),
        "vocab_size": len(tokenizer),
    }


# ═══════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════


class TextureGenApp:
    def __init__(self, root: tk.Tk, initial_checkpoint: str | None = None) -> None:
        self.root = root
        self.root.title("Minecraft Texture Generator")
        self.root.geometry("720x620")
        self.root.resizable(False, False)

        self.ckpt_path: Path | None = None
        self.state: dict[str, Any] | None = None

        self._build_ui()

        if initial_checkpoint:
            self._load(Path(initial_checkpoint))

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top: checkpoint selection ─────────────────────────────
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Checkpoint:").pack(side=tk.LEFT, padx=(0, 4))
        self.ckpt_var = tk.StringVar(value="(none)")
        ttk.Entry(top, textvariable=self.ckpt_var, width=55, state="readonly").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load", command=self._load_current).pack(side=tk.LEFT, padx=4)

        # ── Info bar ──────────────────────────────────────────────
        info = ttk.LabelFrame(self.root, text="Checkpoint Info", padding=4)
        info.pack(fill=tk.X, padx=8, pady=4)

        self.info_text = tk.StringVar(value="No checkpoint loaded.")
        ttk.Label(info, textvariable=self.info_text, font=("Consolas", 9)).pack(
            anchor=tk.W, padx=4
        )

        # ── Prompt ────────────────────────────────────────────────
        prompt_frame = ttk.LabelFrame(self.root, text="Prompt", padding=4)
        prompt_frame.pack(fill=tk.X, padx=8, pady=4)

        self.prompt_var = tk.StringVar(value="")
        self.prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, width=60)
        self.prompt_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.prompt_entry.bind("<Return>", lambda _: self._generate())

        # Autocomplete / suggestion list
        self.suggest_listbox: tk.Listbox | None = None

        ttk.Button(prompt_frame, text="?", width=3, command=self._show_vocab).pack(
            side=tk.RIGHT, padx=2
        )

        # ── Settings ──────────────────────────────────────────────
        settings = ttk.LabelFrame(self.root, text="Settings", padding=4)
        settings.pack(fill=tk.X, padx=8, pady=4)

        # CFG scale
        ttk.Label(settings, text="CFG Scale:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.cfg_var = tk.DoubleVar(value=4.0)
        self.cfg_scale = ttk.Scale(
            settings, from_=1.0, to=10.0, variable=self.cfg_var, orient=tk.HORIZONTAL, length=200
        )
        self.cfg_scale.grid(row=0, column=1, padx=4)
        self.cfg_label = ttk.Label(settings, text="4.0", width=4)
        self.cfg_label.grid(row=0, column=2, padx=2)
        self.cfg_scale.configure(command=self._update_cfg_label)

        # Seed
        ttk.Label(settings, text="Seed:").grid(row=0, column=3, sticky=tk.W, padx=(20, 4))
        self.seed_var = tk.IntVar(value=42)
        self.seed_spin = ttk.Spinbox(
            settings, from_=0, to=999999, textvariable=self.seed_var, width=8
        )
        self.seed_spin.grid(row=0, column=4, padx=4)

        # Random seed button
        ttk.Button(settings, text="🎲", width=3, command=self._random_seed).grid(
            row=0, column=5, padx=2
        )

        # ── Generate button ───────────────────────────────────────
        btn_frame = ttk.Frame(self.root, padding=4)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)

        self.gen_btn = ttk.Button(btn_frame, text="Generate", command=self._generate)
        self.gen_btn.pack(side=tk.LEFT, padx=4)
        self.gen_btn.configure(state=tk.DISABLED)

        self.save_btn = ttk.Button(btn_frame, text="Save Image…", command=self._save_image)
        self.save_btn.pack(side=tk.LEFT, padx=4)
        self.save_btn.configure(state=tk.DISABLED)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btn_frame, textvariable=self.status_var, foreground="gray").pack(
            side=tk.RIGHT, padx=4
        )

        # ── Image display ─────────────────────────────────────────
        img_frame = ttk.LabelFrame(self.root, text="Generated Texture", padding=4)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Canvas with a white background for RGBA display
        self.canvas = tk.Canvas(img_frame, width=256, height=256, bg="white", highlightthickness=0)
        self.canvas.pack(expand=True)
        self.canvas_image: int | None = None

    # ── Callbacks ─────────────────────────────────────────────────

    def _update_cfg_label(self, value: str) -> None:
        self.cfg_label.configure(text=f"{float(value):.1f}")

    def _random_seed(self) -> None:
        import random

        self.seed_var.set(random.randint(0, 999999))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            initialdir=str(_PROJECT_ROOT / "checkpoints"),
            filetypes=[("PyTorch Checkpoint", "*.pth"), ("All Files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load_current(self) -> None:
        path = self.ckpt_var.get()
        if path and path != "(none)":
            self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Error", f"File not found:\n{path}")
            return

        self.status_var.set("Loading checkpoint…")
        self.root.update_idletasks()

        try:
            self.state = load_checkpoint(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.status_var.set("Load failed")
            return

        self.ckpt_path = path
        self.ckpt_var.set(str(path))
        self.diffusion = self.state["diffusion"]

        info = (
            f"Epoch: {self.state['epoch']}  |  "
            f"Loss: {self.state['loss']:.5f}  |  "
            f"Vocab: {self.state['vocab_size']} tokens  |  "
            f"Device: {DEVICE}"
        )
        self.info_text.set(info)
        self.gen_btn.configure(state=tk.NORMAL)
        self.status_var.set("Ready — enter a prompt and click Generate")
        self._clear_image()

    def _show_vocab(self) -> None:
        if self.state is None:
            messagebox.showinfo("Vocab", "Load a checkpoint first.")
            return

        vocab = self.state["tokenizer"].vocab
        words = sorted(vocab.keys())
        text = "\n".join(words[:100])
        if len(words) > 100:
            text += f"\n\n… and {len(words) - 100} more tokens"

        # Show in a popup
        top = tk.Toplevel(self.root)
        top.title(f"Vocabulary ({len(words)} tokens)")
        top.geometry("300x500")

        frame = ttk.Frame(top, padding=4)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Consolas", 9))
        for w in words:
            listbox.insert(tk.END, w)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=listbox.yview)

        # Double-click to insert token into prompt
        listbox.bind("<Double-Button-1>", lambda e: self._insert_token(listbox, top))

    def _insert_token(self, listbox: tk.Listbox, top: tk.Toplevel) -> None:
        sel = listbox.curselection()
        if sel:
            current = self.prompt_var.get().strip()
            token = listbox.get(sel[0])
            if current:
                self.prompt_var.set(f"{current} {token}")
            else:
                self.prompt_var.set(token)
            top.destroy()

    @torch.no_grad()
    def _generate(self) -> None:
        if self.state is None:
            return

        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Prompt", "Please enter a prompt.")
            return

        # Validate tokens
        tokenizer: Tokenizer = self.state["tokenizer"]
        unknown = [w for w in prompt.split() if w not in tokenizer.vocab]
        if unknown:
            if not messagebox.askyesno(
                "Unknown Tokens",
                f"These tokens are not in the vocabulary and will be ignored:\n\n"
                f"{', '.join(unknown)}\n\nContinue?",
            ):
                return

        self.gen_btn.configure(state=tk.DISABLED)
        self.status_var.set("Generating…")
        self.root.update_idletasks()

        try:
            self._run_generation(prompt)
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self.status_var.set("Generation failed")
        finally:
            self.gen_btn.configure(state=tk.NORMAL)
            self.save_btn.configure(state=tk.NORMAL)

    def _run_generation(self, prompt: str) -> None:
        model = self.state["model"]
        tokenizer: Tokenizer = self.state["tokenizer"]
        config: Config = self.state["config"]

        cfg = self.cfg_var.get()
        seed = self.seed_var.get()

        # Set CFG scale
        self.diffusion.cfg_scale = cfg

        cond_ids = tokenizer.encode_single(prompt).to(DEVICE)
        uncond_ids = tokenizer.unconditional(batch_size=1).to(DEVICE)

        torch.manual_seed(seed)

        x = torch.randn((1, config.channels, config.image_size, config.image_size), device=DEVICE)
        x = self.diffusion.p_sample(model, x, cond_ids, uncond_ids)

        # Decode: [-1, 1] → [0, 255]
        x = (x.clamp(-1, 1) + 1) / 2
        x = x.cpu().squeeze(0).permute(1, 2, 0).numpy()

        self.generated_image = Image.fromarray((x * 255).astype(np.uint8), "RGBA")
        self.generated_image_upscaled = self.generated_image.resize(
            (config.output_upscale, config.output_upscale), Image.NEAREST
        )

        self._display_image()

        self.status_var.set(
            f"Generated: '{prompt}' (CFG={cfg:.1f}, seed={seed})"
        )

    def _display_image(self) -> None:
        self._photo = ImageTk.PhotoImage(self.generated_image_upscaled)
        if self.canvas_image is not None:
            self.canvas.delete(self.canvas_image)
        self.canvas_image = self.canvas.create_image(
            128, 128, image=self._photo
        )

    def _clear_image(self) -> None:
        if self.canvas_image is not None:
            self.canvas.delete(self.canvas_image)
            self.canvas_image = None
        self.generated_image = None
        self.generated_image_upscaled = None
        self.save_btn.configure(state=tk.DISABLED)

    def _save_image(self) -> None:
        if self.generated_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save texture as",
            initialdir=str(_PROJECT_ROOT / "outputs"),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
        )
        if path:
            self.generated_image.save(path)
            self.status_var.set(f"Saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    app = TextureGenApp(root, initial_checkpoint=initial)
    root.mainloop()


if __name__ == "__main__":
    main()