"""Text tokenizer for Minecraft texture labels."""

from __future__ import annotations

import torch
from typing import Dict, List


class Tokenizer:
    """Converts texture label strings into padded token-id tensors.

    Vocab is built from all unique words appearing in the dataset labels.
    Index 0 is reserved for padding / unconditional (CFG).
    """

    PAD_IDX: int = 0

    def __init__(self, texts: List[str], max_len: int = 10) -> None:
        self.max_len = max_len
        self.vocab: Dict[str, int] = self._build_vocab(texts)
        self.vocab_size: int = len(self.vocab)

    # ── vocab construction ──────────────────────────────────────

    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        words: set[str] = set()
        for text in texts:
            for word in text.split():
                words.add(word)
        # sorted for determinism; index 1..N (0 = pad)
        return {w: i + 1 for i, w in enumerate(sorted(words))}

    # ── tokenization ────────────────────────────────────────────

    def encode(self, texts: List[str]) -> torch.LongTensor:
        """Encode a list of strings into a (B, max_len) LongTensor."""
        res = torch.zeros((len(texts), self.max_len), dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.split()
            for j, word in enumerate(words[: self.max_len]):
                res[i, j] = self.vocab.get(word, self.PAD_IDX)
        return res

    def encode_single(self, text: str) -> torch.LongTensor:
        """Encode a single string into (1, max_len)."""
        return self.encode([text])

    def unconditional(self, batch_size: int = 1) -> torch.LongTensor:
        """Return an all-zero token tensor for CFG unconditional pass."""
        return torch.zeros((batch_size, self.max_len), dtype=torch.long)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, max_len={self.max_len})"