"""Fast-packing invariants for gravity_forge._kmeans / _assign.

Two properties are load-bearing and easy to break silently:

  1. The sync-free centroid update (clamped counts + torch.where) must be arithmetically identical
     to the old boolean-mask update `cb[cnt>0] = new[cnt>0]/cnt[cnt>0]`, including the empty-cluster
     rule: an empty cluster KEEPS its previous centroid.
  2. Chunked assignment must be EXACT. Chunking is over vectors (independent rows) and each full
     chunk is reshaped into _BMM_GROUPS row groups to hit the fast MPS bmm kernel, so the answer
     must be bit-identical to the unchunked argmin at every chunk size.

Synthetic and small: runs on CPU or MPS in a couple of seconds.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf  # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _restore_knobs():
    chunk, groups = gf.ASSIGN_CHUNK_ELEMS, gf._BMM_GROUPS
    yield
    gf.ASSIGN_CHUNK_ELEMS, gf._BMM_GROUPS = chunk, groups


def _vecs(n=4096, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32)).to(gf._device())


def _masked_update(cb, new, cnt):
    """The ORIGINAL boolean-mask centroid update, kept here as the reference."""
    out = cb.clone()
    nz = cnt > 0
    out[nz] = new[nz] / cnt[nz].unsqueeze(1)
    return out


def _sync_free_update(cb, new, cnt):
    """The shipped update, lifted out of _kmeans."""
    c = cnt.unsqueeze(1)
    return torch.where(c > 0, new / c.clamp(min=1.0), cb)


# -- 1. sync-free centroid update ---------------------------------------------------------------
def test_sync_free_update_matches_masked_update():
    dev = gf._device()
    rng = np.random.default_rng(7)
    k, d = 16, 8
    cb = torch.from_numpy(rng.standard_normal((k, d)).astype(np.float32)).to(dev)
    new = torch.from_numpy(rng.standard_normal((k, d)).astype(np.float32) * 10).to(dev)
    cnt = torch.from_numpy(rng.integers(0, 30, size=k).astype(np.float32)).to(dev)
    cnt[3] = 0.0                                    # force at least one empty cluster
    cnt[11] = 0.0
    ref = _masked_update(cb, new, cnt)
    got = _sync_free_update(cb, new, cnt)
    assert torch.equal(ref, got), "clamped-count update must be bit-identical to the masked update"


def test_empty_clusters_keep_their_previous_centroid():
    dev = gf._device()
    k, d = 8, 4
    cb = torch.arange(k * d, dtype=torch.float32, device=dev).reshape(k, d)
    new = torch.zeros_like(cb)
    cnt = torch.zeros(k, device=dev)
    got = _sync_free_update(cb, new, cnt)
    assert torch.equal(got, cb), "every cluster empty => centroids unchanged, never divided by zero"
    assert torch.isfinite(got).all()


def test_kmeans_survives_a_k_larger_than_the_populated_clusters():
    """k > distinct points => many clusters go empty every iteration. Must stay finite."""
    v = torch.cat([_vecs(64, 4, seed=1)] * 8, 0)     # 512 rows, only 64 distinct
    cb = gf._kmeans(v, 128, iters=5, seed=0)
    assert cb.shape == (128, 4)
    assert torch.isfinite(cb).all()


# -- 2. chunked / grouped assignment exactness ---------------------------------------------------
def _unchunked(v, cb):
    return ((v * v).sum(1, keepdim=True) - 2.0 * (v @ cb.t().contiguous())
            + (cb * cb).sum(1)).argmin(1)


@pytest.mark.parametrize("chunk_elems", [1 << 24, 8192, 2048, 512])
def test_chunked_assign_is_exact(chunk_elems):
    v = _vecs(2048, 8, seed=2)
    cb = _vecs(64, 8, seed=3)
    ref = _unchunked(v, cb)
    gf.ASSIGN_CHUNK_ELEMS = chunk_elems
    assert torch.equal(gf._assign(v, cb), ref), f"chunk_elems={chunk_elems} changed the assignment"


@pytest.mark.parametrize("groups", [1, 2, 8, 32])
def test_bmm_row_grouping_is_exact(groups):
    """Row grouping is a pure reshape: the argmin must not move, at any group count."""
    v = _vecs(4096, 16, seed=4)
    cb = _vecs(256, 16, seed=5)
    ref = _unchunked(v, cb)
    gf._BMM_GROUPS = groups
    assert torch.equal(gf._assign(v, cb), ref), f"_BMM_GROUPS={groups} changed the assignment"


def test_ragged_row_count_takes_the_tail_path_and_stays_exact():
    """N not divisible by _BMM_GROUPS: the tail falls back to the plain 2D path, still exact."""
    v = _vecs(2049, 8, seed=6)                       # indivisible by 8
    cb = _vecs(32, 8, seed=7)
    ref = _unchunked(v, cb)
    gf.ASSIGN_CHUNK_ELEMS = 8192                     # 256 rows per chunk => several chunks + a tail
    got = gf._assign(v, cb)
    assert got.shape == (2049,)
    assert torch.equal(got, ref)


def test_large_k_does_not_materialize_the_full_distance_matrix():
    """k=8192 over 8192 vectors: [N,k] unchunked is 67M floats. Chunked it fits, and stays exact."""
    v = _vecs(8192, 8, seed=8)
    cb = _vecs(8192, 8, seed=9)
    gf.ASSIGN_CHUNK_ELEMS = 1 << 19                  # 64 rows per chunk: forces the chunked path
    got = gf._assign(v, cb)
    gf.ASSIGN_CHUNK_ELEMS = 1 << 30                  # one shot, for the reference
    assert torch.equal(got, gf._assign(v, cb))


def test_kmeans_runs_at_large_k_end_to_end():
    v = _vecs(8192, 8, seed=10)
    gf.ASSIGN_CHUNK_ELEMS = 1 << 18
    cb = gf._kmeans(v, 4096, iters=2, seed=0)
    assert cb.shape == (4096, 8) and torch.isfinite(cb).all()


# -- 3. the packers still behave -----------------------------------------------------------------
def test_pack_product_quant_unaffected_by_the_chunk_knob():
    """The knob is a memory ceiling, not a quality dial: rel error must not move with it.

    Tolerance is 1e-4, not 0: k-means accumulates centroids with index_add_, whose MPS atomics
    reorder run to run, so two packs of the SAME tensor at the SAME knob already differ by ~1e-5.
    Exactness of the chunking itself is asserted at the assignment level above, where it is real."""
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((128, 32)).astype(np.float32) @
         rng.standard_normal((32, 256)).astype(np.float32)) * 0.1
    gf.ASSIGN_CHUNK_ELEMS = 1 << 26
    a = gf.pack_product_quant(w, dim=16, subspaces=2, k=16, seed=0)
    gf.ASSIGN_CHUNK_ELEMS = 2048                     # 128 rows per chunk: real chunking, cheap
    b = gf.pack_product_quant(w, dim=16, subspaces=2, k=16, seed=0)
    assert a.physical_bytes == b.physical_bytes
    assert gf._rel_error(w, a.recon) == pytest.approx(gf._rel_error(w, b.recon), abs=1e-4)
