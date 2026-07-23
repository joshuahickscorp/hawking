#!/usr/bin/env python3.12
"""The k-means distance matrix must not scale with the tensor.

GLM-5.2's embedding table is 951M weights: at dim=8 that is 119M subvectors, and a
whole [N,K] distance matrix against 128 centroids is a 61 GB single allocation. It
took the MPS allocator down and wired memory that was never released.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import gravity_forge as forge  # noqa: E402


def _sample(rows: int = 20_000, dim: int = 8):
    import torch

    rng = np.random.default_rng(0)
    values = rng.standard_normal((rows, dim)).astype(np.float32)
    return torch.from_numpy(values).to(forge._device())


def test_chunking_does_not_change_the_result(monkeypatch):
    values = _sample()
    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 1 << 30)  # one block
    whole_cb = forge._kmeans(values, 128, iters=6, seed=0)
    whole_idx = forge._assign(values, whole_cb)

    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 128 * 4 * 997)  # ~20 ragged blocks
    chunked_cb = forge._kmeans(values, 128, iters=6, seed=0)
    chunked_idx = forge._assign(values, chunked_cb)

    assert bool((whole_idx == chunked_idx).all()), "chunking changed the assignment"
    # Only float accumulation order differs, and far below this backend's own run-to-run
    # noise, which is ~1.4e-3.
    assert float((whole_cb - chunked_cb).abs().max()) < 1e-4


def test_distance_block_stays_within_budget():
    """The block count follows the codebook, never the tensor size."""
    budget = forge._DISTANCE_BUDGET_BYTES
    for rows, k in ((119_000_000, 128), (119_000_000, 256), (10_000, 128)):
        step = forge._chunk_rows(rows, k)
        assert step >= 1
        assert step * k * 4 <= budget or step == rows
        assert step <= rows


def test_assign_covers_every_row(monkeypatch):
    values = _sample(rows=5_000)
    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 128 * 4 * 333)
    cb = forge._kmeans(values, 64, iters=3, seed=0)
    idx = forge._assign(values, cb)
    assert tuple(idx.shape) == (5_000,), "a ragged final block dropped rows"
    assert int(idx.max()) < 64
