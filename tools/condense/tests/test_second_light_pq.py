"""Second Light - tests for Product Quantization as a first-class Gravity Forge family.

Enforces the seven-verb PQ lifecycle, direct compact execution (no dense shadow), deterministic
byte accounting, billed protected islands, a budgeted PQ-aware Doctor, and the Metal Quality Law
(CPU/Metal parity). Style matches test_gravity_forge.py: numpy/MPS-tolerant, deterministic seeds,
judged on a genuinely low-rank synthetic (not a trivial toy)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf  # noqa: E402


def _lowrank(m=256, n=256, r=24, noise=0.01, seed=0):
    """A genuinely low-rank matrix (rank r of min(m,n)) plus a little full-rank noise, so PQ leaves a
    meaningful residual for islands and Doctor to work on. Not a 0.5B-style toy."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((m, r)).astype(np.float32) @ rng.standard_normal((r, n)).astype(np.float32)
    return (base * 0.1 + rng.standard_normal((m, n)).astype(np.float32) * noise).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Lifecycle: inspect / fit / pack / measure / repairability roundtrip.
# --------------------------------------------------------------------------------------------
def test_pq_lifecycle_roundtrip():
    w = _lowrank()
    fam = gf.PQFamily(dim=32, subspaces=4, k=32, seed=0)
    assert fam.VERBS == ("inspect", "fit", "pack", "measure", "execute", "validate", "repairability")

    ins = fam.inspect(w)
    assert ins["rows"] == 256 and ins["cols"] == 256
    assert ins["valid_subvector_dims"] == [4, 8, 16]              # all divide 256
    assert 1 <= ins["effective_rank_90"] <= 256
    assert "residual_kurtosis" in ins

    fit = fam.fit(w)
    assert len(fit["codebooks"]) == fit["S"]                       # one codebook per subspace
    assert fit["indices"].shape[1] == fit["S"]

    art = fam.pack(w)
    assert art.family == "product_quant" and art.recon.size == w.size == art.n_weights
    meas = fam.measure(w, art)
    assert meas["whole_artifact_bpw"] < 8.0 and meas["rel_error"] > 0.0
    assert meas["verdict"] in ("survives", "degraded", "collapse")

    rep = fam.repairability(w, art)
    assert 0.0 <= rep["rank4_capture"] <= 1.0 and 0.0 <= rep["sparse_row_capture"] <= 1.0
    assert rep["residual_rel_energy"] > 0.0


def test_pq_plain_and_rotated_both_available():
    """PQ must exist as its own geometry (plain, no Hadamard) AND as the rotated transform_pq variant."""
    w = _lowrank()
    plain = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=False)
    rotated = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=True)
    assert plain.family == "product_quant" and plain.config["rotate"] is False
    assert rotated.family == "transform_pq" and rotated.config["rotate"] is True
    # the plain variant does NOT carry a transform seed charge; the rotated one does.
    assert "transform_seed" not in plain.ledger.items
    assert "transform_seed" in rotated.ledger.items


# --------------------------------------------------------------------------------------------
# Direct compact execution: y = W_pq @ x from codebooks, no full dense reconstruction.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rotate", [False, True])
def test_pq_direct_execute_matches_dense(rotate):
    w = _lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=rotate)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    y = gf.pq_execute(art, x)
    y_dense = art.recon @ x                                        # the matvec execute computes compactly
    rel = float(np.linalg.norm(y - y_dense) / (np.linalg.norm(y_dense) + 1e-12))
    assert rel < 1e-4                                              # decode-exact vs the dense recon matvec
    val = gf.pq_validate(w, art, x)
    assert val["within_tol"] and val["no_dense_reconstruction"] is True


def test_pq_execute_batched():
    w = _lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((w.shape[1], 6)).astype(np.float32)
    Y = gf.pq_execute(art, X)
    assert Y.shape == (w.shape[0], 6)
    rel = float(np.linalg.norm(Y - art.recon @ X) / (np.linalg.norm(art.recon @ X) + 1e-12))
    assert rel < 1e-4


# --------------------------------------------------------------------------------------------
# Deterministic byte accounting: same seed => identical bytes, recon, and codes.
# --------------------------------------------------------------------------------------------
def test_pq_deterministic_bytes_and_recon():
    w = _lowrank()
    a = gf.pq_pack(w, dim=32, subspaces=4, k=32, seed=11)
    b = gf.pq_pack(w, dim=32, subspaces=4, k=32, seed=11)
    # Byte accounting is EXACTLY deterministic (the honesty contract), and so is the assignment.
    assert a.physical_bytes == b.physical_bytes
    assert np.array_equal(a.config["pq_codes"]["indices"], b.config["pq_codes"]["indices"])
    # Reconstructed floats may drift at the ~1e-6 level because MPS reductions are not bit-identical
    # run-to-run (documented: CPU is authoritative). The bytes and codes are what the ledger bills.
    assert np.allclose(a.recon, b.recon, atol=1e-5)


def test_pq_byte_ledger_counts_indices_and_codebooks():
    w = _lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    assert "indices" in art.ledger.items and "fp16_params" in art.ledger.items
    assert art.ledger.total_bits() > sum(art.ledger.items.values())   # metadata always charged
    assert art.whole_artifact_bpw >= art.base_bpw + art.doctor_bpw - 1e-6


# --------------------------------------------------------------------------------------------
# Protected islands: deterministic, evidence-based, and BILLED (no free islands).
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", list(gf._ISLAND_STRATEGIES))
def test_protected_island_selection_is_deterministic(strategy):
    w = _lowrank()
    resid = w - gf.pq_pack(w, dim=32, subspaces=4, k=32).recon
    act = np.abs(np.random.default_rng(5).standard_normal((10, w.shape[1]))).astype(np.float32)
    sens = np.abs(np.random.default_rng(6).standard_normal(w.shape)).astype(np.float32)
    a = gf.select_protected_islands(w, resid, strategy=strategy, budget_frac=0.05,
                                    activation=act, sensitivity=sens)
    b = gf.select_protected_islands(w, resid, strategy=strategy, budget_frac=0.05,
                                    activation=act, sensitivity=sens)
    assert np.array_equal(a["row_indices"], b["row_indices"])     # deterministic
    assert a["n_islands"] == int(np.ceil(0.05 * w.shape[0]))      # respects budget_frac
    assert np.all(np.diff(a["row_indices"]) > 0)                  # sorted, unique


def test_protected_islands_are_billed_and_increase_bpw():
    w = _lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    isl = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.05,
                                       strategy="residual_energy")
    assert "protected_islands" in isl.ledger.items                # islands billed inside the ledger
    assert isl.whole_artifact_bpw > base.whole_artifact_bpw       # strictly more bytes (no free island)
    # accounting invariant still holds with islands folded into the base representation.
    assert isl.whole_artifact_bpw >= isl.base_bpw + isl.doctor_bpw - 1e-6
    assert isl.overhead_bpw >= -1e-6
    # protected rows are reconstructed exactly.
    rows = isl.config["pq_codes"]["island_rows"]
    assert np.allclose(isl.recon[rows], w[rows], atol=1e-6)


def test_more_islands_cost_more_bytes():
    w = _lowrank()
    small = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.02)
    big = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.10)
    assert big.whole_artifact_bpw > small.whole_artifact_bpw


def test_protected_island_artifact_executes_with_exact_rows():
    w = _lowrank()
    isl = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.05)
    x = np.random.default_rng(7).standard_normal(w.shape[1]).astype(np.float32)
    y = gf.pq_execute(isl, x)
    rows = isl.config["pq_codes"]["island_rows"]
    # island rows must equal the ORIGINAL w @ x exactly (they are stored verbatim).
    assert np.allclose(y[rows], (w[rows] @ x), atol=1e-4)
    # and the whole execute matches the artifact's own reconstruction matvec.
    assert np.linalg.norm(y - isl.recon @ x) / (np.linalg.norm(isl.recon @ x) + 1e-12) < 1e-4


# --------------------------------------------------------------------------------------------
# PQ-aware Doctor: within a hard byte budget, and it reduces error.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("treatment", list(gf._DOCTOR_TREATMENTS))
def test_doctor_stays_within_budget(treatment):
    w = _lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    budget = 8000
    out = gf.doctor_pq(w, base, byte_budget=budget, strategy=treatment)
    assert out["treatment"] == treatment
    assert out["added_bytes"] <= budget and out["within_budget"]
    assert out["quality_delta"] >= -1e-6                          # never worsens the artifact


def test_doctor_reduces_error_for_repairing_treatments():
    """The three treatments that add explicit residual capacity must strictly reduce weight error."""
    w = _lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    for treatment in ("residual_codebook", "sparse_residual", "protected_island_expansion"):
        out = gf.doctor_pq(w, base, byte_budget=8000, strategy=treatment)
        assert out["quality_delta"] > 0.0, treatment
        assert out["err_after"] < out["err_before"], treatment


def test_doctor_respects_tiny_budget():
    """With a tiny budget the Doctor must still not exceed it (bounded, honest)."""
    w = _lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    out = gf.doctor_pq(w, base, byte_budget=200, strategy="sparse_residual")
    assert out["added_bytes"] <= 200


def test_doctor_rejects_unknown_treatment():
    w = _lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    with pytest.raises(ValueError):
        gf.doctor_pq(w, base, byte_budget=8000, strategy="not_a_treatment")


# --------------------------------------------------------------------------------------------
# Metal Quality Law: CPU (authoritative) vs MPS parity - no quality loss for speed.
# --------------------------------------------------------------------------------------------
def test_cpu_metal_parity_holds():
    w = _lowrank()
    par = gf.pq_cpu_metal_parity(w, dim=32, k=32, subspaces=4)
    assert par["authoritative"] == "cpu"
    assert par["within_tol"], par                                # bounded rel-error delta
    assert par["ranking_match"]                                  # same candidate ordering on both backends
    assert par["pass_match"]                                     # same survive/degrade/collapse verdict
    assert 0.0 <= par["assignment_agreement"] <= 1.0
    assert par["relerr_delta"] <= par["tol"]


def test_selftest_reports_second_light_signals():
    out = gf.selftest()
    assert out["pq_family_verbs"] == 7
    assert out["pq_execute_within_tol"]
    assert out["pq_deterministic_bytes"]
    assert out["pq_islands_increase_bpw"]
    assert out["pq_doctor_reduces_error"]
    assert out["pq_cpu_metal_within_tol"]
