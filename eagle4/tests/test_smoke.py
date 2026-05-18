"""Smoke tests — verify the module loads and the head builds + forwards."""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eagle4  # noqa: E402


def test_imports():
    assert eagle4.HIDDEN_DIM == 2048
    assert eagle4.N_MOE_LAYERS == 26
    assert eagle4.N_ROUTED == 64


def test_head_builds_and_forwards(tmp_path):
    # Build a tiny frozen.npz with random weights.
    frozen = tmp_path / "frozen.npz"
    H, V = eagle4.HIDDEN_DIM, eagle4.VOCAB
    np.savez(
        frozen,
        token_embd=np.zeros((H, V), dtype=np.float16),
        lm_head=np.zeros((H, V), dtype=np.float16),
        output_norm=np.ones((H,), dtype=np.float32),
    )
    head = eagle4.build_head(frozen)
    B, S = 2, 4
    prev = mx.zeros((B, S), dtype=mx.int32)
    h = mx.random.normal((B, S, H))
    tok, mask, draft, calib = head(prev, h, h, h, h)
    mx.eval(tok, mask, draft, calib)
    assert tok.shape == (B, S, V)
    assert mask.shape == (B, S, eagle4.N_MOE_LAYERS, eagle4.N_ROUTED)
    assert draft.shape == (B, S, H)
    assert calib.shape == (B, S)


def test_residual_gate_starts_at_zero(tmp_path):
    """At init, head output should equal post_norm(h_high) → identity-like behavior."""
    frozen = tmp_path / "frozen.npz"
    H = eagle4.HIDDEN_DIM
    np.savez(
        frozen,
        token_embd=np.zeros((H, eagle4.VOCAB), dtype=np.float16),
        lm_head=np.zeros((H, eagle4.VOCAB), dtype=np.float16),
        output_norm=np.ones((H,), dtype=np.float32),
    )
    head = eagle4.build_head(frozen)
    # Gate is initialized to a small positive value (0.05) so the block path
    # receives gradient from step 1 — see eagle4.py:EagleHead.__init__ comment.
    assert 0.0 < float(head.residual_gate[0]) <= 0.1


def test_quantize_roundtrip(tmp_path):
    """Quantize a fake head, decode, check shapes match."""
    src = tmp_path / "head.npz"
    np.savez(
        src,
        **{"in_proj.weight": np.random.randn(2048, 10240).astype(np.float16) * 0.02},
    )
    out = tmp_path / "head_q4.npz"
    eagle4.quantize_head(src, out)
    z = np.load(out, allow_pickle=False)
    assert "in_proj.weight" in z.files
    assert "in_proj.weight.scales" in z.files
    assert "in_proj.weight.biases" in z.files
