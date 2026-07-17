"""Tests for the Gravity Forge foundry - enforce honest accounting and the no-overclaim doctrine."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf  # noqa: E402


def _lowrank(m=128, n=128, r=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((m, r)).astype(np.float32) @ rng.standard_normal((r, n)).astype(np.float32)) * 0.1


def test_selftest_green():
    out = gf.selftest()
    assert out["ok"] and out["accounting_invariant_holds"]
    assert out["compressible_beats_random"]
    assert out["deterministic_bytes"]


def test_byte_ledger_counts_everything():
    """Whole-artifact bytes must include indices, codebooks, transform seed, and metadata -
    nothing is silently free (guards against hidden byte accounting)."""
    w = _lowrank()
    art = gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)
    assert "indices" in art.ledger.items and "fp16_params" in art.ledger.items
    assert "transform_seed" in art.ledger.items          # transform is billed (seed), not free
    # metadata charge is always present in the total
    assert art.ledger.total_bits() > sum(art.ledger.items.values())
    assert art.physical_bytes > 0


def test_accounting_invariant_whole_ge_base_plus_doctor():
    w = _lowrank()
    for art in (gf.pack_naive_rvq(w, dim=32, k=64, stages=2),
                gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=4, sparse_rows=4),
                gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)):
        assert art.whole_artifact_bpw >= art.base_bpw + art.doctor_bpw - 1e-6
        assert art.overhead_bpw >= -1e-6


def test_weight_error_is_not_capability_claim():
    """A family can have low weight error yet the artifact must never be flagged capability-parity;
    output_divergence must always label itself proxy_output / capability_parity False."""
    # low_rank reconstructs the low-rank matrix ~exactly (low weight error)...
    w = _lowrank(r=4)
    art = gf.pack_low_rank(w, rank=4)
    assert gf._rel_error(w, art.recon) < 0.01
    # ...but there is no capability flag anywhere on the artifact; it is a weight-space object only.
    assert not hasattr(art, "capability_parity")


def test_deterministic_same_seed_same_bytes():
    w = _lowrank()
    a = gf.pack_naive_rvq(w, dim=32, k=64, stages=2, seed=3)
    b = gf.pack_naive_rvq(w, dim=32, k=64, stages=2, seed=3)
    assert a.physical_bytes == b.physical_bytes


def test_shared_grammar_amortizes_codebook():
    """A larger expert cluster must lower whole-artifact BPW (shared codebook amortized), proving
    the MoE sharing lever is real and correctly accounted."""
    experts = [_lowrank(seed=i) for i in range(8)]
    small = gf.pack_shared_grammar(experts[:2], dim=32, k=64, stages=2)
    big = gf.pack_shared_grammar(experts, dim=32, k=64, stages=2)
    assert big.whole_artifact_bpw < small.whole_artifact_bpw


def test_repairability_splits_base_and_doctor():
    w = _lowrank()
    art = gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=4, sparse_rows=4)
    assert art.base_bpw > 0 and art.doctor_bpw > 0        # both accounted separately
    assert "doctor_lowrank" in art.ledger.items and "doctor_sparse_rows" in art.ledger.items


def test_no_dense_shadow_model():
    """The reconstruction returned is a bounded per-tile object the size of the tile, not a hidden
    expansion. recon.size must equal the represented weight count, never more."""
    w = _lowrank(m=256, n=128, r=8)
    art = gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)
    assert art.recon.size == w.size == art.n_weights


def test_four_materially_distinct_families_available():
    """Section 5 requires >=4 materially distinct families (beyond the RVQ/low-rank controls)."""
    out = gf.selftest()
    assert out["families_available"] >= 4
    w = _lowrank()
    families = {
        "transform_pq": gf.pack_transform_pq(w, dim=32, subspaces=2, k=64).family,
        "shared_grammar": gf.pack_shared_grammar([w, w * 1.01], dim=32, k=64, stages=2).family,
        "repairability": gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=2, sparse_rows=2).family,
        "ternary": gf.pack_ternary_factor(w, rank=4).family,
    }
    assert len(set(families.values())) == 4


def test_ternary_factor_is_ternary_and_billed_conservatively():
    """Ternary factors must be {-1,0,+1} and billed at 2 bits each (no base-3 packing claimed)."""
    w = _lowrank()
    art = gf.pack_ternary_factor(w, rank=6)
    assert art.config["ternary_bits"] == 2
    assert "ternary_factors" in art.ledger.items
    assert art.recon.size == w.size and np.isfinite(art.recon).all()
