"""Within-tensor low-rank NR: controls are falsifiable, accounting is load-bearing.

Capability is not asserted from reconstruction. A mutation that unbills scales,
unbills residual indices, or swaps the dense denominator must fail.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
import lowrank_nr as lr  # noqa: E402
from lowrank_nr import (  # noqa: E402
    AFFINE_BITS,
    AFFINE_GROUPS,
    BF16_BYTES,
    RESIDUAL_INDEX_BYTES,
    RESIDUAL_VALUE_BYTES,
    SCALE_BYTES,
    ZERO_BYTES,
    AccountingError,
    break_even_rank,
    capability_record,
    compare_matched,
    dense_payload_bytes,
    encode_affine,
    encode_broken_magnitude,
    encode_dense_bf16,
    encode_lowrank,
    encode_matched_residual,
    encode_pq,
    evaluator_handle,
    exact_rank_matrix,
    float_factor_payload_bytes,
    packed_bytes,
    parse_incumbent_spec,
    source_params_of,
    spectral_of,
    structure_verdict,
    synthetic_with_rank90,
)

SRC_PATH = Path(__file__).resolve().parent.parent / "tools" / "future" / "lowrank_nr.py"
SRC = SRC_PATH.read_text()


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# positive / negative / break-even / accounting controls
# ---------------------------------------------------------------------------

def test_positive_control_exact_rank_reconstructs_and_deficit_is_large():
    W = exact_rank_matrix(48, 32, 5, seed=1)
    arm = encode_lowrank(W, 5, 32, None, 32, None)
    assert arm.metrics["rel_fro"] < 1e-5, arm.metrics
    assert arm.metrics["cosine"] > 0.9999, arm.metrics
    bf = encode_lowrank(W, 5, 16, None, 16, None)
    assert bf.metrics["rel_fro"] < 1e-2, bf.metrics
    spec = spectral_of(W)
    assert spec["deficit_pct"] > 50.0, spec
    assert spec["structure_real"] is True
    assert "REAL" in structure_verdict(spec)
    assert "not of behaviour" in structure_verdict(spec)


def test_negative_control_random_is_not_a_win():
    W = _rng(2).standard_normal((64, 48), dtype=np.float32)
    spec = spectral_of(W)
    assert abs(spec["deficit_pct"]) < 5.0, spec
    assert spec["structure_real"] is False
    text = structure_verdict(spec)
    assert "ABSENT" in text, text
    assert "win" in text, text
    arm = encode_lowrank(W, 6, 16, None, 16, None)
    assert arm.metrics["rel_fro"] > 0.3, arm.metrics
    # Byte savings at r < break-even is arithmetic identity, not a quality win.
    assert arm.complete_bytes < encode_dense_bf16(W).complete_bytes
    assert "wins" not in text.lower()


def test_break_even_rank_stores_more_than_dense():
    m, n = 2048, 768
    be = break_even_rank([m, n])
    assert abs(be - m * n / (m + n)) < 1e-9
    r = int(math.ceil(be))
    assert r == 559
    assert float_factor_payload_bytes(m, n, r) > dense_payload_bytes(m, n)
    assert float_factor_payload_bytes(m, n, r - 1) < dense_payload_bytes(m, n)
    W = _rng(3).standard_normal((32, 24), dtype=np.float32)
    r_s = int(math.ceil(break_even_rank([32, 24])))
    fat = encode_lowrank(W, r_s, 16, None, 16, None)
    dense = encode_dense_bf16(W)
    assert fat.complete_bytes > dense.complete_bytes
    assert fat.stores_more_than_dense is True


def test_accounting_omitting_scales_drops_ebpw_and_does_not_change_reconstruction():
    W = _rng(4).standard_normal((32, 64), dtype=np.float32)
    full = encode_lowrank(W, 8, 4, 32, 2, 32, residual_k=16, include_scales=True)
    omit = encode_lowrank(W, 8, 4, 32, 2, 32, residual_k=16, include_scales=False)
    assert full.parts["A_scales"] > 0 and full.parts["A_zeros"] > 0
    assert full.parts["B_scales"] > 0 and full.parts["B_zeros"] > 0
    assert omit.parts["A_scales"] == 0 and omit.parts["A_zeros"] == 0
    assert omit.parts["B_scales"] == 0 and omit.parts["B_zeros"] == 0
    assert omit.complete_ebpw < full.complete_ebpw
    assert omit.complete_bytes == (
        full.complete_bytes
        - full.parts["A_scales"] - full.parts["A_zeros"]
        - full.parts["B_scales"] - full.parts["B_zeros"]
    )
    np.testing.assert_array_equal(omit.reconstruction, full.reconstruction)


def test_accounting_bills_residual_values_AND_indices():
    W = _rng(5).standard_normal((24, 32), dtype=np.float32)
    arm = encode_lowrank(W, 4, 4, 32, 2, 32, residual_k=11)
    assert arm.residual_k == 11
    assert arm.parts["residual_values"] == 11 * RESIDUAL_VALUE_BYTES
    assert arm.parts["residual_indices"] == 11 * RESIDUAL_INDEX_BYTES
    assert arm.parts["residual_indices"] > 0
    k0 = encode_lowrank(W, 4, 4, 32, 2, 32, residual_k=0)
    assert k0.parts["residual_values"] == 0 and k0.parts["residual_indices"] == 0
    assert arm.complete_bytes - k0.complete_bytes == 11 * (
        RESIDUAL_VALUE_BYTES + RESIDUAL_INDEX_BYTES
    )


def test_source_params_are_the_dense_organ_never_the_factors():
    W = _rng(6).standard_normal((40, 30), dtype=np.float32)
    arm = encode_lowrank(W, 5, 16, None, 16, None)
    assert arm.source_params == 40 * 30
    assert arm.source_params != 5 * (40 + 30)
    assert source_params_of(W.shape) == 40 * 30
    # Scales are overhead in the numerator, not source parameters.
    q = encode_lowrank(W, 5, 4, 32, 2, 32)
    assert q.source_params == arm.source_params
    assert q.parts["A_scales"] > 0


def test_parts_sum_to_complete_bytes_and_unknown_keys_are_refused():
    W = _rng(7).standard_normal((16, 32), dtype=np.float32)
    arm = encode_lowrank(W, 3, 4, 32, 2, 32, residual_k=4)
    assert sum(arm.parts.values()) == arm.complete_bytes
    with pytest.raises(AccountingError, match="undeclared"):
        lr.finalize_parts({**arm.parts, "hidden_codebook": 99}, arm.source_params, True)


def test_packed_payload_is_bits_not_one_byte_per_value():
    W = _rng(8).standard_normal((1, 128), dtype=np.float32)
    aff = encode_affine(W, 2, 128)
    assert aff.parts["dense_payload"] == packed_bytes(128, 2) == 32
    assert aff.parts["A_scales"] == SCALE_BYTES
    assert aff.parts["A_zeros"] == ZERO_BYTES
    # 4-bit of 64 values is 32 bytes, not 64.
    W2 = _rng(8).standard_normal((1, 64), dtype=np.float32)
    a4 = encode_affine(W2, 4, 64)
    assert a4.parts["dense_payload"] == packed_bytes(64, 4) == 32


def test_unequal_factor_precision_is_allowed_and_changes_the_bill():
    W = _rng(9).standard_normal((32, 64), dtype=np.float32)
    unequal = encode_lowrank(W, 8, 4, 64, 2, 128)
    equal = encode_lowrank(W, 8, 4, 64, 4, 64)
    assert unequal.factor_precisions["A"]["bits"] == 4
    assert unequal.factor_precisions["B"]["bits"] == 2
    assert unequal.complete_bytes != equal.complete_bytes
    assert "unequal" in " ".join(unequal.notes).lower() or (
        unequal.factor_precisions["A"] != unequal.factor_precisions["B"]
    )


def test_arm_letters_and_dense_reference():
    W = _rng(10).standard_normal((16, 32), dtype=np.float32)
    assert encode_dense_bf16(W).arm == "A"
    assert encode_lowrank(W, 4, 16, None, 16, None).arm == "B"
    assert encode_lowrank(W, 4, 4, 32, 2, 32).arm == "C"
    assert encode_lowrank(W, 4, 4, 32, 2, 32, residual_k=3).arm == "D"
    assert encode_affine(W, 2, 32).arm == "E"
    assert encode_pq(W, 4, 16).arm == "E"
    a = encode_dense_bf16(W)
    assert a.complete_bytes == 16 * 32 * BF16_BYTES
    assert abs(a.complete_ebpw - 16.0) < 1e-12


# ---------------------------------------------------------------------------
# matched EBPW
# ---------------------------------------------------------------------------

def test_matched_ebpw_D_lands_on_incumbent_E():
    W = synthetic_with_rank90(192, 128, r90=48, seed=11)
    E = encode_affine(W, 2, 128)
    D = encode_matched_residual(W, E, r=48, bits_a=2, group_a=128, bits_b=2, group_b=128)
    gap = E.complete_bytes - D.complete_bytes
    assert gap >= 0, (D.complete_bytes, E.complete_bytes)
    assert gap < (RESIDUAL_VALUE_BYTES + RESIDUAL_INDEX_BYTES)
    assert abs(D.complete_ebpw - E.complete_ebpw) < 0.02
    assert D.arm in ("C", "D")
    assert E.arm == "E"


def test_compare_matched_prints_both_arms_and_does_not_claim_capability():
    W = synthetic_with_rank90(96, 128, r90=32, seed=12)
    cmp = compare_matched(W, incumbent_spec="q2-g128")
    letters = {a.arm for a in cmp["arms"]}
    assert {"A", "B", "C", "D", "E"} <= letters, letters
    assert "D" in cmp["table"] and "E" in cmp["table"], cmp["table"]
    assert cmp["capability"]["status"] == "NOT_MEASURED"
    assert cmp["capability"]["capability_ok"] is None
    assert "not measured" in cmp["table"].lower()
    assert "conjunction" in cmp["table"].lower() or "AND" in cmp["table"]
    # Reconstruction is present and demoted.
    assert "cheap filter" in cmp["table"]
    assert "not of behaviour" in cmp["table"] or "ENERGY" in cmp["table"]


def test_evaluator_handle_matches_gravity_keys_and_refuses_nx():
    W = _rng(13).standard_normal((16, 32), dtype=np.float32)
    arm = encode_lowrank(W, 4, 4, 32, 2, 32, residual_k=2)
    h = evaluator_handle(arm)
    for k in ("complete_ebpw", "stored_bytes", "accounting", "capability_ok",
              "capability_status", "magnitude_ratio", "direction_similarity",
              "execution_complete"):
        assert k in h, k
    assert h["execution_complete"] is False
    assert h["nr_not_nx"] is True
    assert h["capability_ok"] is None
    assert h["accounting"]["parts"]["residual_indices"] == 2 * RESIDUAL_INDEX_BYTES
    assert "NX" not in h["representation_class"]


def test_parse_incumbent_refuses_unrunnable_specs():
    assert parse_incumbent_spec("q2-g128")["bits"] == 2
    assert parse_incumbent_spec("pq4k512")["codebook"] == 512
    with pytest.raises(ValueError):
        parse_incumbent_spec("outlier0.005-g128")
    with pytest.raises(ValueError):
        parse_incumbent_spec("q2-g256")
    with pytest.raises(ValueError):
        parse_incumbent_spec("q5-g64")


def test_capability_record_is_none_not_a_false_fail():
    cap = capability_record()
    assert cap["status"] == "NOT_MEASURED"
    assert cap["capability_ok"] is None
    assert "Metal" in cap["reason"] or "metal" in cap["reason"].lower()


def test_reconstruction_is_not_treated_as_the_gate():
    """High cosine on a truncated random matrix still does not pass capability."""
    W = _rng(14).standard_normal((32, 32), dtype=np.float32)
    arm = encode_lowrank(W, 24, 16, None, 16, None)
    assert arm.metrics["cosine"] > 0.9, arm.metrics
    cap = capability_record()
    h = evaluator_handle(arm, cap)
    assert h["capability_ok"] is not True
    assert h["capability_status"] == "NOT_MEASURED"


# ---------------------------------------------------------------------------
# source inspection: the load-bearing lines must exist
# ---------------------------------------------------------------------------

def test_source_bills_scales_zeros_and_residual_indices():
    """Registration of the names is not billing. These identifiers are the bill."""
    assert "A_scales" in SRC and "A_zeros" in SRC
    assert "residual_indices" in SRC
    assert "RESIDUAL_INDEX_BYTES" in SRC
    assert "include_scales" in SRC
    # Packed body uses bits, not a byte-per-value.
    packed = SRC[SRC.index("def packed_bytes"):SRC.index("def source_params_of")]
    assert "* int(bits) + 7) // 8" in packed
    assert "n_values * bits" not in packed.replace(" ", "") or "// 8" in packed
    # Denominator is m*n.
    assert "return m * n" in SRC
    i = SRC.index("def residual_parts")
    block = SRC[i:i + 600]
    assert "RESIDUAL_INDEX_BYTES" in block
    assert "RESIDUAL_VALUE_BYTES" in block


def test_source_does_not_assert_nx():
    from nrnx_discipline import violations
    v = violations(SRC, "tools/future/lowrank_nr.py")
    assert v == [], v


def test_o003_break_even_arithmetic_from_the_receipt():
    """The digits that were right: 2048x768 rank-495 is 0.886x dense, be=559."""
    fac = lr.factoring_report(2048, 768, 495)
    assert abs(fac["byte_ratio"] - 0.886) < 1e-3, fac
    assert fac["pays"] is True
    assert fac["factored_elems"] == 495 * (2048 + 768)
    assert fac["dense_elems"] == 2048 * 768
    W = synthetic_with_rank90(2048, 768, r90=495, seed=0)
    spec = spectral_of(W)
    assert spec["rank_90"] == pytest.approx(495, abs=8), spec
    assert spec["structure_real"] is True, spec


def test_real_o003_organ_if_present_has_the_receipt_deficit():
    snap = Path(lr.O003_ANATOMY_SNAP)
    if not snap.is_dir():
        pytest.skip("Qwen3-30B-A3B snapshot absent; organ assertion DID NOT RUN")
    try:
        W, _ = lr.load_o003_organ()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert tuple(W.shape) == (2048, 768)
    spec = spectral_of(W)
    # Receipt: ratio 0.7171, null 0.8289, deficit 13.491, rank_90 495.
    assert spec["rank_90"] == 495, spec
    assert abs(spec["ratio"] - 0.7171) < 0.01, spec
    assert spec["deficit_pct"] > 10.0, spec
    assert spec["structure_real"] is True


# ---------------------------------------------------------------------------
# gravity gate constants are the ones we refuse to proxy
# ---------------------------------------------------------------------------

def test_gate_constants_are_the_gravity_conjunction():
    import gravity_outlier_eval as G
    cap = capability_record()
    gate = cap.get("gate") or {}
    if "R4_MAX" in gate:
        assert gate["R4_MAX"] == G.R4_MAX
        assert gate["PPL_MAX"] == G.PPL_MAX
    # Conjunction: the measured collapsed arm (ppl 4.0847, r4 0.7204) fails.
    assert G.R4_MAX < 0.7204, "collapsed-generation r4 must fail the gate"
    assert G.PPL_MAX < 4.0847, "the measured collapsed arm's ppl must fail too"


def test_incumbent_groups_match_mx_quantize():
    assert AFFINE_GROUPS == (32, 64, 128)
    assert 2 in AFFINE_BITS and 4 in AFFINE_BITS


def test_broken_magnitude_arm_keeps_direction_and_destroys_magnitude():
    """The gauntlet's magnitude-destroyed control, without claiming the gate ran."""
    W = _rng(20).standard_normal((16, 32), dtype=np.float32)
    arm = encode_broken_magnitude(W, 0.01)
    assert arm.arm == "broken"
    assert arm.metrics["magnitude_ratio"] < 0.05, arm.metrics
    assert arm.metrics["cosine"] > 0.99, arm.metrics
    h = evaluator_handle(arm)
    assert h["capability_ok"] is None
    assert h["capability_status"] == "NOT_MEASURED"


def test_compare_includes_the_broken_arm():
    W = synthetic_with_rank90(64, 128, r90=20, seed=21)
    cmp = compare_matched(W, incumbent_spec="q2-g128")
    names = {a.arm for a in cmp["arms"]}
    assert "broken" in names
    broken = next(a for a in cmp["arms"] if a.arm == "broken")
    assert broken.metrics["magnitude_ratio"] < 0.05


def test_kimi_gravity_organ_structure_is_absent_if_present():
    """G004's O003 in gravity_outlier_eval is Kimi-VL, 2048x1408, not the Qwen organ.

    Layer-10 expert0 down_proj sits at ~3% deficit: ABSENT. A 9-tensor panel
    (L1/L14/L26 x down/gate/up) shows the prior does not transfer even inside
    this body: late gate_proj crosses the 10% bar. See
    test_kimi_late_gate_proj_structure_is_real.
    """
    snap = Path(lr.O003_GRAVITY_SNAP)
    if not snap.is_dir():
        pytest.skip("Kimi-VL-A3B snapshot absent; gravity-organ assertion DID NOT RUN")
    try:
        W, _ = lr.load_o003_gravity_organ()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert tuple(W.shape) == (2048, 1408)
    spec = spectral_of(W)
    assert spec["deficit_pct"] < lr.LOWRANK_LIVE_DEFICIT_PCT, spec
    assert spec["structure_real"] is False, spec
    text = structure_verdict(spec)
    assert "ABSENT" in text, text


def test_kimi_expert_depth_panel_shows_organ_dependence():
    snap = Path(lr.O003_GRAVITY_SNAP)
    if not snap.is_dir():
        pytest.skip("Kimi-VL-A3B snapshot absent; panel DID NOT RUN")
    try:
        panel = lr.expert_depth_panel()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert len(panel["rows"]) == 9, panel
    by = {(r["layer"], r["proj"]): r for r in panel["rows"]}
    # Layer-1 down is ABSENT; late gate is REAL. If this flips, the organ changed.
    assert by[(panel["layers_picked"][0], "down_proj")]["structure_real"] is False
    last = panel["layers_picked"][-1]
    assert by[(last, "gate_proj")]["structure_real"] is True
    assert panel["any_structure_real"] is True
    assert panel["capability"]["capability_ok"] is None


def test_kimi_late_gate_proj_structure_is_real():
    """Organ-dependence inside one expert: L26 gate_proj is REAL, down_proj is not."""
    snap = Path(lr.O003_GRAVITY_SNAP)
    if not snap.is_dir():
        pytest.skip("Kimi-VL-A3B snapshot absent; late-gate assertion DID NOT RUN")
    try:
        W, _ = lr.load_named_organ(
            lr.O003_GRAVITY_SNAP,
            "language_model.model.layers.26.mlp.experts.0.gate_proj.weight",
        )
    except FileNotFoundError as e:
        pytest.skip(str(e))
    spec = spectral_of(W)
    assert spec["deficit_pct"] >= lr.LOWRANK_LIVE_DEFICIT_PCT, spec
    assert spec["structure_real"] is True, spec
    assert "REAL" in structure_verdict(spec)
    # Float factoring still only barely pays; matched-EBPW vs affine is a
    # different question and is not decided here.
    assert spec["byte_ratio"] < 1.0, spec
