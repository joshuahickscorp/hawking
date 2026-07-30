"""Tokenization and hashed bag-of-features for the small formal system."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import torch

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+|[^\sA-Za-z0-9_]")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def hash_bucket(token: str, dim: int) -> int:
    # FNV-1a 32-bit, stable across runs/processes.
    h = 2166136261
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def bag_vector(tokens: Iterable[str], dim: int = 4096, normalize: bool = True) -> torch.Tensor:
    v = torch.zeros(dim, dtype=torch.float32)
    n = 0
    for t in tokens:
        v[hash_bucket(t, dim)] += 1.0
        n += 1
    if normalize and n > 0:
        v = v / (v.norm() + 1e-8)
    return v


def bag_of_text(text: str, dim: int = 4096) -> torch.Tensor:
    return bag_vector(tokenize(text), dim=dim)


class LabelVocab:
    """Closed label vocabulary built from the train split only."""

    def __init__(self, labels: list[str], min_count: int = 1, max_size: int = 8000):
        counts = Counter(labels)
        kept = [lab for lab, c in counts.most_common(max_size) if c >= min_count]
        self._set_itos(kept)

    @classmethod
    def from_itos(cls, itos: list[str]) -> "LabelVocab":
        obj = object.__new__(cls)
        obj._set_itos(list(itos))
        return obj

    def _set_itos(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {lab: i for i, lab in enumerate(self.itos)}
        self.unk = len(self.itos)  # index for unknowns at eval

    def __len__(self) -> int:
        return len(self.itos) + 1  # +unk

    @property
    def n_known(self) -> int:
        return len(self.itos)

    def encode(self, label: str) -> int:
        return self.stoi.get(label, self.unk)

    def decode(self, idx: int) -> str | None:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return None

    def known(self, label: str) -> bool:
        return label in self.stoi
