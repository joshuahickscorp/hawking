"""Tests for the Hawking Mechanics Generation-M measurement module (B0/B1/M1).

Enforce the sealed laws: every op reports a fully-populated, fully-labelled 10-dim mechanical vector;
M1 (lookup-linear) equals B1 (bounded reconstruction) within tol because they execute the SAME PQ
artifact (the causal control); no execution grammar materializes a dense shadow (peak temporary is
bounded well under the full dense tensor); CPU and Metal M1 agree (Metal Quality Law); everything is
deterministic under a fixed seed. CPU (numpy) is authoritative; Metal checks skip if MPS is absent.
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
import mech_measure as mm  # noqa: E402

_VALID_LABELS = {"ANALYTICAL", "MEASURED", "ESTIMATED", "UNAVAILABLE"}


def _lowrank(m=256, n=128, r=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((m, r)).astype(np.float32)
            @ rng.standard_normal((r, n)).astype(np.float32) * 0.05).astype(np.float32)


def _artifact(seed=0, k=16, subspaces=8):
    w = _lowrank(seed=seed)
    return w, gf.pack_product_quant(w, dim=64, subspaces=subspaces, k=k, seed=0)


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------------------------
# Mechanical vector: fully populated + fully labelled, no dim hidden in another.
# --------------------------------------------------------------------------------------------
def test_mech_vector_populated_and_labelled():
    _, art = _artifact()
    codes = mm._codes(art)
    for mv in (mm.mech_b0_compact(codes, 1),
               mm.mech_b1(codes, 1, 64, mps=False), mm.mech_b1(codes, 1, 64, mps=True),
               mm.mech_m1(codes, 1, mps=False), mm.mech_m1(codes, 1, mps=True)):
        d = mv.to_dict()
        assert set(d["vector"].keys()) == set(mm.DIMS)
        assert set(d["labels"].keys()) == set(mm.DIMS)
        for dim in mm.DIMS:
            assert d["labels"][dim] in _VALID_LABELS, (dim, d["labels"][dim])
            assert d["vector"][dim] >= 0.0


def test_mech_vector_does_not_hide_arithmetic_in_lookups():
    """B0/B1 do dense-equivalent arithmetic (2*m*n FP32); M1 replaces most multiplies with lookups,
    so M1.F32 must be strictly less than B0.F32 while M1.Llookup is populated (accounted, not hidden)."""
    _, art = _artifact()
    codes = mm._codes(art)
    rows, cols = codes["rows"], codes["cols"]
    b0 = mm.mech_b0_compact(codes, 1)
    m1 = mm.mech_m1(codes, 1, mps=False)
    assert b0.F32 == pytest.approx(2.0 * rows * cols)
    assert m1.F32 < b0.F32                      # arithmetic genuinely reduced
    assert m1.Llookup > 0.0                     # and accounted as lookups, not folded into F32
    assert m1.labels["Llookup"] == "ANALYTICAL"


# --------------------------------------------------------------------------------------------
# Causal control: M1 == B1 within tol (same PQ artifact).
# --------------------------------------------------------------------------------------------
def test_m1_equals_b1_same_artifact():
    w, art = _artifact()
    rng = np.random.default_rng(3)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    y_b1 = mm.b1_reconstruct_matvec_np(art, x, tile_rows=64)
    y_m1 = mm.m1_lookup_linear_np(art, x)
    ag = mm.m1_vs_b1_agreement(y_m1, y_b1)
    assert ag["within_tol"], ag
    assert ag["rel_err_m1_vs_b1"] < 1e-4


def test_b0_b1_m1_all_execute_same_recon():
    """All three grammars execute recon @ x, so B0 (pq_execute) == B1 == M1 within float reorder."""
    w, art = _artifact()
    rng = np.random.default_rng(5)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    y_b0 = gf.pq_execute(art, x)
    y_b1 = mm.b1_reconstruct_matvec_np(art, x, tile_rows=64)
    y_m1 = mm.m1_lookup_linear_np(art, x)
    ref = art.recon @ x
    for y in (y_b0, y_b1, y_m1):
        assert mm._rel(y, ref) < 1e-4


def test_m1_batch_matches_b1_batch():
    w, art = _artifact()
    rng = np.random.default_rng(7)
    xb = rng.standard_normal((w.shape[1], 4)).astype(np.float32)
    y_b1 = mm.b1_reconstruct_matvec_np(art, xb, tile_rows=64)
    y_m1 = mm.m1_lookup_linear_np(art, xb)
    assert y_b1.shape == (w.shape[0], 4)
    assert mm._rel(y_m1, y_b1) < 1e-4


# --------------------------------------------------------------------------------------------
# No dense shadow: peak temporary bounded well under the full dense tensor.
# --------------------------------------------------------------------------------------------
def test_no_dense_shadow_bounded_temporary():
    _, art = _artifact()
    codes = mm._codes(art)
    rows, cols = codes["rows"], codes["cols"]
    full_dense = rows * cols * 4
    for mv in (mm.mech_b0_compact(codes, 1),
               mm.mech_b1(codes, 1, 64, mps=False), mm.mech_m1(codes, 1, mps=False)):
        assert 0 < mv.Ttemporary < 0.5 * full_dense, (mv.Ttemporary, full_dense)
    # M1 is the tightest: its peak temporary is far below B0's decoded-subspace tile
    assert mm.mech_m1(codes, 1, mps=False).Ttemporary < mm.mech_b0_compact(codes, 1).Ttemporary


def test_m1_reads_less_than_full_dense_bytes():
    """M1 gathers a bounded set of table entries; it must NOT stream the full dense matrix. Its Mread
    of decoded/gathered values is strictly below the dense-recon Mread of B0."""
    _, art = _artifact()
    codes = mm._codes(art)
    assert mm.mech_m1(codes, 1, mps=False).Mread < mm.mech_b0_compact(codes, 1).Mread


# --------------------------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------------------------
def test_deterministic_same_seed():
    """Determinism follows the frozen PQ doctrine: byte accounting is bit-identical across packs;
    reconstruction VALUES may drift within MPS near-tie reorder tolerance (the packer runs k-means on
    MPS). So assert byte-identical bytes + value agreement within tol, and bit-exact M1 given a FIXED
    artifact (the lookup grammar itself must be fully deterministic)."""
    w, art1 = _artifact(seed=0)
    _, art2 = _artifact(seed=0)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    assert art1.physical_bytes == art2.physical_bytes          # accounting bit-identical
    y1, y2 = mm.m1_lookup_linear_np(art1, x), mm.m1_lookup_linear_np(art2, x)
    assert mm._rel(y1, y2) < 1e-4                              # values within MPS reorder tol
    # the M1 grammar on a FIXED artifact is exactly reproducible (no MPS in the CPU lookup path)
    assert np.array_equal(mm.m1_lookup_linear_np(art1, x), mm.m1_lookup_linear_np(art1, x))


def test_quality_parity_matches_dense_definition():
    w, art = _artifact()
    rng = np.random.default_rng(13)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    y = gf.pq_execute(art, x)
    q = mm.quality_parity(w, y, x)
    assert 0.0 <= q["rel_error_vs_dense"]
    assert -1.0001 <= q["cosine_vs_dense"] <= 1.0001


def test_seal_roundtrip_self_sha256():
    obj = mm.seal({"a": 1, "b": [2, 3], "schema": "x"})
    import hashlib, json
    body = {k: v for k, v in obj.items() if k != "sha256"}
    calc = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()
    assert obj["sha256"] == calc


# --------------------------------------------------------------------------------------------
# Metal (MPS) parity - Metal Quality Law. Skips cleanly if MPS is unavailable.
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_m1_cpu_metal_parity():
    import torch
    w, art = _artifact()
    rng = np.random.default_rng(17)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    dev = mm._mps_device()
    y_cpu = mm.m1_lookup_linear_np(art, x)
    y_metal = mm.m1_lookup_linear_torch(art, torch.from_numpy(x).to(dev), dev=dev).detach().cpu().numpy()
    assert mm._rel(y_metal, y_cpu) <= 5e-3


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
def test_b1_cpu_metal_parity():
    import torch
    w, art = _artifact()
    rng = np.random.default_rng(19)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    dev = mm._mps_device()
    y_cpu = mm.b1_reconstruct_matvec_np(art, x, tile_rows=64)
    y_metal = mm.b1_reconstruct_matvec_torch(art, torch.from_numpy(x).to(dev),
                                             tile_rows=64, dev=dev).detach().cpu().numpy()
    assert mm._rel(y_metal, y_cpu) <= 5e-3


def test_module_selftest_green():
    out = mm.selftest()
    assert out["ok"]
    assert out["m1_vs_b1_within_tol"]
    assert out["no_dense_shadow"]
    assert out["m1_flops_lt_b0"]
    assert out["mech_vector_dims"] == 10
