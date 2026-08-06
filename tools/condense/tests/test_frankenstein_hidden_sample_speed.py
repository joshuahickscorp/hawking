#!/usr/bin/env python3.12
"""Parity + speed for vectorized _hidden_sample equal-length fast path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import frankenstein_teacher_forced_executor as frank  # noqa: E402


def _loop_reference(hidden: np.ndarray, lengths: np.ndarray, width: int) -> dict:
    """Exact pre-vectorization loop (kept here as parity oracle)."""
    batch, seq, dim = hidden.shape
    width = min(width, dim)
    samples = np.zeros((batch, len(frank.SAMPLE_TOKEN_SLOTS), width), dtype=np.float32)
    means = np.zeros((batch, dim), dtype=np.float32)
    vars_ = np.zeros((batch, dim), dtype=np.float32)
    l2 = np.zeros((batch,), dtype=np.float32)
    absmax = np.zeros((batch,), dtype=np.float32)
    for b in range(batch):
        L = int(lengths[b]) if b < len(lengths) else seq
        L = max(1, min(L, seq))
        slice_ = hidden[b, :L]
        means[b] = np.mean(slice_, axis=0)
        vars_[b] = np.var(slice_, axis=0)
        l2[b] = float(np.sqrt(np.sum(slice_.astype(np.float64) ** 2)))
        absmax[b] = float(np.max(np.abs(slice_)))
        pos = frank._sample_positions(L)
        for i, slot in enumerate(frank.SAMPLE_TOKEN_SLOTS):
            samples[b, i] = slice_[pos[slot], :width]
    return {
        "samples": samples,
        "mean": means,
        "var": vars_,
        "l2": l2,
        "absmax": absmax,
        "sample_width": np.full((batch,), width, dtype=np.int32),
    }


def test_hidden_sample_equal_length_parity():
    rng = np.random.default_rng(0)
    B, S, H = 32, 64, 256
    hidden = rng.standard_normal((B, S, H)).astype(np.float32)
    lengths = np.full(B, S, dtype=np.int32)
    width = 64
    got = frank._hidden_sample(hidden, lengths, width)
    ref = _loop_reference(hidden, lengths, width)
    for key in ("samples", "mean", "var", "l2", "absmax", "sample_width"):
        assert got[key].shape == ref[key].shape, key
        # mean/var use float64 reduction then cast; allow tiny float noise
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=1e-5, err_msg=key)


def test_hidden_sample_variable_length_parity():
    rng = np.random.default_rng(1)
    B, S, H = 16, 48, 128
    hidden = rng.standard_normal((B, S, H)).astype(np.float32)
    lengths = rng.integers(1, S + 1, size=B, dtype=np.int32)
    width = 32
    got = frank._hidden_sample(hidden, lengths, width)
    ref = _loop_reference(hidden, lengths, width)
    for key in ("samples", "mean", "var", "l2", "absmax", "sample_width"):
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=1e-5, err_msg=key)


def test_hidden_sample_equal_length_not_slower_than_loop():
    """Equal-length fast path must not regress vs the historical row loop.

    On B=32,S=128,H=6144 this is only a modest win (~10%); download dominates
    the real pipeline. Guard against accidental pathological slowdowns only.
    """
    rng = np.random.default_rng(2)
    B, S, H = 32, 128, 6144
    hidden = rng.standard_normal((B, S, H)).astype(np.float32)
    lengths = np.full(B, S, dtype=np.int32)
    width = 64
    frank._hidden_sample(hidden, lengths, width)
    _loop_reference(hidden, lengths, width)
    t0 = time.perf_counter()
    for _ in range(8):
        frank._hidden_sample(hidden, lengths, width)
    t_vec = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(8):
        _loop_reference(hidden, lengths, width)
    t_loop = time.perf_counter() - t0
    assert t_vec < t_loop * 1.25, f"vec={t_vec:.4f}s loop={t_loop:.4f}s"
