"""ONEBIT_FAMILIES: four structurally distinct families at matched executable bytes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from fractional_bit_canon import (  # noqa: E402
    SCALE_TRAP,
    TRIT_PACK_5IN8,
    codec_binary,
    codec_degenerate_absmax_b1,
    codec_ternary,
    codec_zero,
    score_pair,
)
from onebit_families import (  # noqa: E402
    B2_GROUP,
    B3_GROUP,
    B4_GROUP,
    B4_K,
    B6_D,
    B6_M,
    B8_GROUP,
    MATCH_TARGET_BPW,
    MATCH_WINDOW,
    OUT_PATH,
    SCHEMA,
    bill_shared_basis,
    codec_generated_walsh,
    codec_pq_aa,
    fit_shared_binary_bases,
    in_match_window,
    run_unit_instruments,
    walsh_signs,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        assert OUT_PATH.is_file(), (
            f"missing {OUT_PATH} — run python3 tools/headless/onebit_families.py"
        )
        RECEIPT_DOC = json.loads(OUT_PATH.read_text())
    return RECEIPT_DOC


# ---------------------------------------------------------------------------
# Unit instruments — no 27B, no capture.
# ---------------------------------------------------------------------------


def test_fitted_binary_is_not_deletion_and_g64_bills_1_25():
    rng = np.random.RandomState(0)
    W = rng.randn(128, 64).astype(np.float32)
    What, acc = codec_binary(W, g=64)
    assert np.count_nonzero(What) == What.size
    assert abs(acc["storage_bpw"] - 1.25) < 1e-12
    assert acc["active_fused_bpw"] == acc["storage_bpw"]
    assert acc["scales_counted"] is True
    deg, _ = codec_degenerate_absmax_b1(W, g=64)
    z, _ = codec_zero(W)
    assert not np.any(deg)
    assert np.array_equal(deg, z)


def test_b2_matched_group_bills_2_00():
    W = np.random.RandomState(1).randn(64, 64).astype(np.float32)
    What, acc = codec_binary(W, g=B2_GROUP)
    assert abs(acc["storage_bpw"] - (1.0 + 16.0 / B2_GROUP)) < 1e-12
    assert abs(acc["storage_bpw"] - 2.0) < 1e-12
    assert np.count_nonzero(What) == What.size


def test_b3_bills_5in8_plus_scale_not_one():
    W = np.random.RandomState(2).randn(64, 64).astype(np.float32)
    What, acc = codec_ternary(W, g=B3_GROUP)
    assert acc["scales_counted"] is True
    assert abs(acc["storage_bpw"] - (TRIT_PACK_5IN8 + 16.0 / B3_GROUP)) < 1e-12
    assert abs(acc["storage_bpw"] - 1.85) < 1e-12
    assert acc["storage_bpw"] != 1.0
    assert np.unique(np.abs(What) > 0).size >= 1


def test_b4_amortized_storage_is_2_00_over_two_layers():
    W = np.random.RandomState(3).randn(32, 64).astype(np.float32)
    acc = bill_shared_basis(W, K=B4_K, g=B4_GROUP, n_layers=2)
    expected = B4_K / 2.0 + (B4_K * 16.0) / B4_GROUP
    assert abs(expected - 2.0) < 1e-12
    assert abs(acc["storage_bpw"] - 2.0) < 1e-12
    assert acc["basis_count"] == B4_K
    assert acc["coefficient_bytes_per_layer"] > 0
    assert acc["active_basis_loads_bpw_cold"] > acc["active_fused_bpw_bases_resident"]
    assert acc["counterfactual_64_layer_storage_bpw"] < acc["storage_bpw"]
    # Bases approach 0 when amortized over 64 layers; coefficients do not.
    assert acc["counterfactual_64_layer_storage_bpw"] > 0.5


def test_b4_shared_basis_is_not_independent_signs():
    rng = np.random.RandomState(4)
    W0 = rng.randn(16, 32).astype(np.float32)
    W1 = rng.randn(16, 32).astype(np.float32)
    d = np.ones(32, dtype=np.float32)
    Whats, bases, _ = fit_shared_binary_bases([W0, W1], [d, d], K=1, g=32)
    assert len(bases) == 1
    assert bases[0].shape == W0.shape
    assert set(np.unique(bases[0])).issubset({-1.0, 1.0})
    # Independent sign(W0) and sign(W1) are not the same object as the shared basis.
    s0 = np.where(W0 >= 0, 1.0, -1.0)
    assert Whats[0].shape == W0.shape
    assert not np.array_equal(bases[0], s0) or not np.array_equal(bases[0], np.where(W1 >= 0, 1.0, -1.0))


def test_b6_route_replaces_a_fragment_and_counts_codebook_bytes():
    rng = np.random.RandomState(5)
    W = rng.randn(32, 16).astype(np.float32)
    energy = np.linspace(0.1, 2.0, 16).astype(np.float32)
    What, acc, codes, C = codec_pq_aa(
        W, d=4, M=8, energy=energy, seed=5, iters=4, sample_n=128
    )
    assert What.shape == W.shape
    assert acc["parent_weight_equivalents_per_route"] == 4
    assert acc["codebook_bytes"] == 8 * 4 * 16 / 8.0
    assert acc["extra_bits"] == 8 * 4 * 16
    assert acc["n_routes"] == 32 * (16 // 4)
    assert codes.max() < 8
    assert C.shape == (8, 4)
    # Codebook bytes are in the bill, not hidden behind a 2.00 headline.
    assert acc["storage_bpw"] > acc["code_bpw"]
    assert acc["extra_bits"] > 0


def test_b6_parent_weight_equivalents_formula():
    assert B6_D == 4
    assert B6_M == 256
    assert abs(math.log2(B6_M) / B6_D - 2.0) < 1e-12


def test_b8_counts_generator_runtime_and_cache_bytes():
    W = np.random.RandomState(6).randn(32, 64).astype(np.float32)
    What, acc = codec_generated_walsh(W, g=B8_GROUP, d=None)
    assert acc["code_bits"] == 0.0
    assert acc["generator_bytes"] > 0
    assert acc["cache_bytes_if_signs_materialised"] == W.size / 8.0
    assert acc["active_cached_signs_bpw"] == acc["storage_bpw"] + 1.0
    assert acc["no_information_hiding"] is True
    # On-the-fly form sits at 2 bpw of scales; generator identity is a header.
    assert abs(acc["storage_bpw"] - (16.0 / B8_GROUP)) < 1e-12
    assert in_match_window(acc["storage_bpw"])
    assert acc["generator_bits"] == acc["generator_bytes"] * 8
    assert What.shape == W.shape


def test_generated_signs_are_deterministic_and_not_stored_from_W():
    a = walsh_signs(8, 16, row0=0)
    b = walsh_signs(8, 16, row0=0)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({-1.0, 1.0})
    W = np.random.RandomState(7).randn(8, 16).astype(np.float32)
    assert not np.array_equal(a, np.where(W >= 0, 1.0, -1.0))


def test_scale_aware_metric_rejects_001W_while_cosine_is_one():
    rng = np.random.RandomState(8)
    Y = rng.randn(64, 32).astype(np.float32)
    sc = score_pair(Y, SCALE_TRAP * Y)
    assert abs(sc["cosine"] - 1.0) < 1e-5
    assert sc["gain"] < 0.05
    assert sc["scale_aware"] < 0.05
    # The null is stated; cosine beating the null is not a GO.
    assert sc["beats_null"] is True
    assert sc["rel_fro"] > 0.9


def test_matched_window_covers_the_four_realizable_budgets():
    b2 = 1.0 + 16.0 / B2_GROUP
    b3 = TRIT_PACK_5IN8 + 16.0 / B3_GROUP
    b4 = B4_K / 2.0 + (B4_K * 16.0) / B4_GROUP
    b6_code = math.log2(B6_M) / B6_D
    b8 = 16.0 / B8_GROUP
    assert abs(b2 - 2.0) < 1e-12
    assert abs(b3 - 1.85) < 1e-12
    assert abs(b4 - 2.0) < 1e-12
    assert abs(b6_code - 2.0) < 1e-12
    assert abs(b8 - 2.0) < 1e-12
    for bpw in (b2, b3, b4, b6_code, b8):
        assert in_match_window(bpw), bpw
    assert MATCH_WINDOW[0] <= MATCH_TARGET_BPW <= MATCH_WINDOW[1]


def test_unit_instruments_from_the_module():
    inst = run_unit_instruments()
    assert inst["not_an_activation_score"] is True
    assert inst["degenerate_absmax_b1_is_zero"] is True
    assert inst["g64_binary_storage_bpw_must_be_1.25"] is True
    assert inst["b2_g16_storage_bpw_must_be_2"] is True
    assert inst["b4_k2_g32_2layer_must_be_2"] is True
    assert inst["b8_signs_not_stored"] is True
    assert inst["b8_generator_bytes_counted"] is True
    assert inst["b8_cache_bytes_reported"] is True


# ---------------------------------------------------------------------------
# Receipt — written by onebit_families.py on real held-out activations.
# ---------------------------------------------------------------------------


def test_receipt_has_at_least_four_structurally_distinct_families():
    doc = receipt()
    assert doc["schema"] == SCHEMA
    fams = doc["families"]
    ids = [f["family_id"] for f in fams]
    assert len(set(ids)) >= 4, ids
    # The four required structural slots.
    for need in ("B2", "B3", "B4", "B6"):
        assert need in ids, ids
    assert doc["n_structurally_distinct_families"] >= 4
    assert doc["not_how_few_bits"] is True
    assert doc["did_not_load_second_27b"] is True
    assert doc["streamed_per_tensor"] is True


def test_receipt_matched_executable_bytes():
    doc = receipt()
    assert doc["matched_executable_bpw_target"] == MATCH_TARGET_BPW
    matched = [f for f in doc["families"] if f["in_matched_window"]]
    assert len(matched) >= 4
    bpws = [f["storage_bpw"] for f in matched]
    assert max(bpws) - min(bpws) <= (MATCH_WINDOW[1] - MATCH_WINDOW[0]) + 1e-9
    for f in matched:
        assert MATCH_WINDOW[0] <= f["storage_bpw"] <= MATCH_WINDOW[1]
        assert MATCH_WINDOW[0] <= f["active_fused_bpw"] <= MATCH_WINDOW[1] or f["family_id"] == "B8"


def test_each_family_has_storage_active_error_null_and_verdict():
    doc = receipt()
    for f in doc["families"]:
        assert "storage_bpw" in f and "active_fused_bpw" in f, f["family_id"]
        err = f["function_space_error"]
        assert err["null"] == "constant_mean_row_cosine"
        assert "mean" in err
        assert "mean_surplus_over_null" in err
        assert "mean_scale_aware" in err
        assert f["survival_verdict"]
        assert f["n_tensors"] >= 2
        assert f["all_beats_null"] is True or f["survival_verdict"] in (
            "FAILS",
            "DELETION",
            "SCALE_TRAP",
            "BEATS_NULL_BUT_UNHEALTHY",
            "SURVIVES_AT_MATCHED_BYTES",
            "SURVIVES_OFF_BUDGET",
        )
        for row in f["per_tensor"]:
            assert row["n_hold"] if "n_hold" in row else True
            assert "rel_fro" in row and "null" in row and "survival_verdict" in row
            assert row["null"] > 0.0


def test_receipt_used_real_held_out_activations_not_gaussian():
    doc = receipt()
    cap = doc["capture"]
    assert cap["not_gaussian"] is True
    assert cap["not_llama_server"] is True
    assert cap["n_hold"] >= 256
    assert cap["n_fit"] >= cap["n_hold"]
    assert cap["named_by"].endswith("FRACTIONAL_BIT_CANON.json")
    assert "capture_diverse2" in cap["path"]
    for block in doc["organ_blocks"]:
        for rec in block["per_layer"]:
            assert rec["n_hold"] >= 256
            assert rec["n_fit"] >= 256


def test_null_stated_and_scale_trap_rejects_001W():
    doc = receipt()
    null = doc["null"]
    assert null["name"] == "constant_mean_row_cosine"
    assert "0.01*W" in null["why_cosine_alone_is_illegal"] or "1.000000" in null["why_cosine_alone_is_illegal"]
    trap = doc["scale_trap_global"]
    assert trap["cosine_must_be_one"] is True
    assert trap["gain_rejects"] is True
    assert trap["instrument_ok"] is True
    assert abs(trap["cosine"] - 1.0) < 1e-5
    assert trap["gain"] < 0.05


def test_degenerate_absmax_is_an_instrument_not_a_family():
    doc = receipt()
    ids = [f["family_id"] for f in doc["families"]]
    assert "CTRL_ABSMAX1" not in ids
    assert "degenerate_absmax_b1" not in ids
    # The control is recorded per tensor and matches deletion.
    found = False
    for block in doc["organ_blocks"]:
        for rec in block["per_layer"]:
            deg = next(c for c in rec["controls"] if c["family_id"] == "CTRL_ABSMAX1")
            assert deg["matches_deletion"] is True
            assert deg["survival_verdict"] == "DELETION"
            found = True
    assert found


def test_verdict_does_not_say_1bit_is_impossible():
    doc = receipt()
    v = doc["verdict"]
    assert v["one_failed_scheme_is_not_1bit_impossible"] is True
    assert doc["one_failed_scheme_is_not_1bit_impossible"] is True
    # The recorded decision is about these families at this budget, never a
    # closed 1-bit question. A refusal that names the illegal claim is fine;
    # the verdict field itself must not be that claim.
    assert v["decision"] not in {
        "1-BIT IS IMPOSSIBLE",
        "1-bit is impossible",
        "ONE_BIT_IMPOSSIBLE",
        "IMPOSSIBLE",
    }
    assert "IMPOSSIBLE" not in v["decision"]
    assert "not a legal verdict" in v["global_claim_refused"].lower()


def test_g035_prior_is_column_svd_not_this_b4():
    doc = receipt()
    prior = doc["prior"]
    assert prior["g035_shared_beats_independent"] is False
    assert "column" in prior["g035_axis"].lower()
    assert "binary" in prior["b4_what_is_being_tested"].lower()
    assert "not g035" in prior["b4_what_is_being_tested"].lower()
    b4 = next(f for f in doc["families"] if f["family_id"] == "B4")
    extras = b4["accounting_extras"]
    assert extras.get("basis_count") == B4_K
    assert extras.get("coefficient_bytes_per_layer") is not None
    assert extras.get("active_basis_loads_bpw_cold") is not None


def test_b6_receipt_reports_parent_weight_equivalents_and_entropy():
    doc = receipt()
    b6 = next(f for f in doc["families"] if f["family_id"] == "B6")
    extras = b6["accounting_extras"]
    assert extras["parent_weight_equivalents_per_route"] == B6_D
    assert extras["route_entropy_bits"] is not None
    assert extras["reuse"] is not None
    large = doc.get("b6_large_fragment_coordinate_check")
    assert large is not None
    assert large["parent_weight_equivalents_per_route"] >= 32
    assert large.get("not_the_matched_row") is True
    # Large-fragment BPW is far below the match — this is the coordinate warning.
    assert large["storage_bpw"] < MATCH_WINDOW[0]


def test_b8_receipt_does_not_hide_generator_or_cache():
    doc = receipt()
    ids = [f["family_id"] for f in doc["families"]]
    if "B8" not in ids:
        return
    b8 = next(f for f in doc["families"] if f["family_id"] == "B8")
    extras = b8["accounting_extras"]
    assert extras["generator_bytes"] > 0
    assert extras["generator_runtime"]
    assert extras["cache_bytes_if_signs_materialised"] > 0


def test_each_family_scored_in_function_space_against_the_stated_null():
    doc = receipt()
    for f in doc["families"]:
        assert f["mean_null"] > 0.05
        # Cosine-only would let 0.01*W through; scale_aware is present.
        assert f["mean_scale_aware"] is not None
        for row in f["per_tensor"]:
            assert row["rel_fro"] >= 0.0
            assert -1.0 <= row["surplus_over_null"] <= 1.0
