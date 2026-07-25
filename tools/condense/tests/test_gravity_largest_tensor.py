#!/usr/bin/env python3.12
"""Gate 5.1: the k-means working set must not scale with the tensor.

PREREGISTERED before the fix, per loop discipline. The gate is:

1. The chunked argmin must equal the unchunked argmin EXACTLY, on CPU and on the
   default device, across several geometries including a ragged final chunk.
2. Peak distance-matrix bytes must be bounded by the codebook and the budget,
   never by N.
3. Accumulation stays a single index_add_ over the whole tensor. Chunking the
   accumulation is what produced GPU address faults and a cb[nz] shape mismatch
   in the reverted attempt at 6046c103; that shape is asserted against here.
4. A tensor large enough to have blown the old path must fit the budget.

GLM-5.2's embedding and lm_head are 951,582,720 weights each, which at dim=8 is
118,947,840 subvectors. Against 128 centroids that is a 60.9 GB float32 matrix.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import gravity_forge as forge  # noqa: E402

GLM52_EMBED_WEIGHTS = 951_582_720
GLM52_EMBED_SUBVECTORS_AT_DIM8 = GLM52_EMBED_WEIGHTS // 8


def _tensor(rows: int, dim: int, device):
    import torch

    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.standard_normal((rows, dim)).astype(np.float32)).to(device)


@pytest.mark.parametrize("rows,dim,k", [(4096, 8, 64), (10_000, 8, 128), (3_331, 16, 32)])
def test_chunked_argmin_is_exactly_unchunked(monkeypatch, rows, dim, k):
    """Ragged final chunks included: 3331 rows never divides evenly."""
    import torch

    for device in ("cpu", forge._device()):
        values = _tensor(rows, dim, device)
        v2 = (values * values).sum(1, keepdim=True)
        cb = values[:k].clone()

        whole = forge._argmin_chunked(values, v2, cb, rows)
        chunked = forge._argmin_chunked(values, v2, cb, max(1, rows // 7))
        assert torch.equal(whole, chunked), f"chunking changed the argmin on {device}"


def test_distance_block_is_bounded_by_the_codebook_not_the_tensor():
    budget = forge._DISTANCE_BUDGET_BYTES
    huge = forge._chunk_rows(GLM52_EMBED_SUBVECTORS_AT_DIM8, 128)
    assert huge * 128 * 4 <= budget, "the distance block must fit the budget"
    assert huge < GLM52_EMBED_SUBVECTORS_AT_DIM8, "the embedding table must actually chunk"
    # the whole point: 128 centroids costs the same block whatever N is
    assert forge._chunk_rows(10_000_000, 128) == forge._chunk_rows(100_000_000, 128)
    # and a wider codebook costs a proportionally shorter block
    assert forge._chunk_rows(10_000_000, 256) <= forge._chunk_rows(10_000_000, 128)


def test_accumulation_is_not_chunked(monkeypatch):
    """The reverted attempt chunked index_add_ and raced its own count vector."""
    import torch

    values = _tensor(5_000, 8, forge._device())
    calls = {"n": 0}
    real = torch.Tensor.index_add_

    def counting(self, dim, index, source, **kwargs):
        calls["n"] += 1
        return real(self, dim, index, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_add_", counting)
    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 64 * 4 * 101)  # force ~50 blocks
    iters = 3
    forge._kmeans(values, 64, iters=iters, seed=0)
    # exactly two accumulations per iteration (sum and count), regardless of blocks
    assert calls["n"] == 2 * iters, (
        f"expected {2 * iters} accumulations, saw {calls['n']}: "
        "the accumulation is being chunked again"
    )


def test_kmeans_matches_across_block_sizes(monkeypatch):
    """Same centroids whether one block or many, on CPU where MPS noise cannot mask it."""
    import torch

    values = _tensor(8_000, 8, "cpu")
    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 1 << 30)
    whole = forge._kmeans(values, 64, iters=5, seed=0)
    monkeypatch.setattr(forge, "_DISTANCE_BUDGET_BYTES", 64 * 4 * 257)
    chunked = forge._kmeans(values, 64, iters=5, seed=0)
    assert torch.equal(whole, chunked), "chunking changed the CPU k-means result"
