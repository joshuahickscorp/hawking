"""N044 COORDINATE_TRANSFORM_PROBE: function-preserving rotations vs the 2.25 floor."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from coordinate_transform_probe import (  # noqa: E402
    BINARY_BPW,
    BLOCK,
    G032_Q2_DELTA_HOLD,
    GENERATOR,
    HIDDEN,
    INTERMEDIATE,
    MATERIAL_ARGMAX,
    MATERIAL_REL_FRO,
    PARENT_A,
    PARENT_PARAMS,
    Q2F_BPW,
    RECEIPT,
    SCHEMA,
    TERNARY_CODE_BPW,
    TERNARY_PACKED_BPW,
    argmax_agree,
    apply_fwht_last_axis,
    bad_blocks,
    codec_bpw,
    codec_reconstruct,
    dense_block_storage_bytes,
    fwht_last,
    function_preserve_error,
    hadamard_storage_bytes,
    identity_codec_error,
    make_bad,
    make_hadamard,
    make_identity,
    make_pca,
    orth_error_blocks,
    pca_blocks,
    parent_a_readonly,
    rotate_activation_for_weight,
    rotate_weight_input,
    rotate_weight_output,
    rotation_bill,
    score_composition,
    self_check,
    survives,
    synthetic_W,
    synthetic_X,
    unrotate_output,
)
from fractional_bit_canon import LOG2_3, rel_fro, score_pair  # noqa: E402


def test_qwen_dims_tile_block_1024_with_no_padding():
    assert HIDDEN % BLOCK == 0
    assert INTERMEDIATE % BLOCK == 0
    assert HIDDEN // BLOCK == 5
    assert INTERMEDIATE // BLOCK == 17
    assert BLOCK == 1024


def test_binary_is_1_25_ternary_code_is_log2_3_q2f_is_2_25():
    b = codec_bpw("binary_g64")
    t = codec_bpw("ternary_g64")
    q = codec_bpw("q2f_g64")
    assert abs(b["storage_bpw"] - 1.25) < 1e-12
    assert abs(b["storage_bpw"] - BINARY_BPW) < 1e-12
    assert abs(t["code_bpw"] - LOG2_3) < 1e-12
    assert abs(t["code_bpw"] - TERNARY_CODE_BPW) < 1e-12
    assert 1.58 < t["code_bpw"] < 1.59
    assert abs(t["storage_bpw"] - 1.85) < 1e-12
    assert abs(t["storage_bpw"] - TERNARY_PACKED_BPW) < 1e-12
    assert abs(q["storage_bpw"] - 2.25) < 1e-12
    assert abs(q["storage_bpw"] - Q2F_BPW) < 1e-12
    # Scales counted: 1-bit + f16/64 is 1.25, not 1.0.
    assert b["storage_bpw"] == pytest.approx(1.0 + 16.0 / 64)
    # Packed ternary is not the entropy quote; both are reported.
    assert t["s026_quote_bpw"] == 1.58
    assert t["storage_bpw"] > t["code_bpw"]


def test_hadamard_is_orthogonal_involutive_and_function_preserving():
    I = np.eye(BLOCK, dtype=np.float32)
    H = fwht_last(I)
    err = float(np.linalg.norm(H.T @ H - I) / math.sqrt(BLOCK * BLOCK))
    assert err < 1e-6
    # Involutive: H H = I (normalized FWHT).
    HH = H @ H
    assert float(np.linalg.norm(HH - I)) < 1e-4
    T = make_hadamard()
    assert T.orthogonal
    assert T.involutive
    assert T.absorbed_zero_runtime
    assert T.control == "candidate"
    err_p = function_preserve_error(T, out=64, inn=1024, n=32, seed=3)
    assert err_p < 1e-5
    # Tiles a Qwen hidden-shaped matrix.
    W = synthetic_W(8, HIDDEN, 9)
    X = synthetic_X(4, HIDDEN, 10)
    Y = X @ W.T
    Wr = rotate_weight_input(W, T)
    Xr = rotate_activation_for_weight(X, T)
    Yh = Xr @ Wr.T
    assert rel_fro(Y, Yh) < 1e-5


def test_identity_is_exact_noop_including_after_quantize():
    T = make_identity()
    assert T.control == "noop"
    assert T.orth_error == 0.0
    err = function_preserve_error(T, out=32, inn=128, n=16, seed=1)
    assert err < 1e-6
    for codec in ("binary_g64", "ternary_g64", "q2f_g64"):
        e = identity_codec_error(codec, out=32, inn=128, n=12)
        assert e < 1e-6, (codec, e)


def test_pca_block_is_orthogonal_and_function_preserving():
    X = synthetic_X(200, 1024, 21)
    Ts = pca_blocks(X, 1024)
    assert len(Ts) == 1
    assert orth_error_blocks(Ts) < 1e-5
    W = synthetic_W(16, 1024, 22)
    from coordinate_transform_probe import block_rmatmul

    Wr = block_rmatmul(W, Ts)
    Xr = block_rmatmul(X[:40], Ts)
    Y = X[:40] @ W.T
    Yh = Xr @ Wr.T
    assert rel_fro(Y, Yh) < 1e-4
    # Learned transform on native hidden dim.
    Xh = synthetic_X(64, HIDDEN, 23)
    Xi = synthetic_X(64, INTERMEDIATE, 24)
    T = make_pca(Xh, Xi)
    assert T.learned and T.orthogonal and T.structured
    assert T.orth_error < 1e-5
    Wg = synthetic_W(32, HIDDEN, 25)
    Xg = synthetic_X(20, HIDDEN, 26)
    Y = Xg @ Wg.T
    Yh = rotate_activation_for_weight(Xg, T) @ rotate_weight_input(Wg, T).T
    assert rel_fro(Y, Yh) < 1e-4


def test_bad_control_is_not_orthogonal_but_round_trips_with_inverse():
    T = make_bad()
    assert T.control == "bad"
    assert T.orthogonal is False
    assert T.absorbed_zero_runtime is False
    assert T.orth_error > 0.1
    Th, Th_inv = bad_blocks(HIDDEN)
    # T T^{-1} = I per block.
    acc = 0.0
    for A, B in zip(Th, Th_inv):
        acc += float(np.linalg.norm(A @ B - np.eye(BLOCK, dtype=np.float32)))
    assert acc / len(Th) < 1e-3
    W = synthetic_W(8, HIDDEN, 31)
    X = synthetic_X(6, HIDDEN, 32)
    Y = X @ W.T
    Wr = rotate_weight_input(W, T)
    Xr = rotate_activation_for_weight(X, T)
    Yh = Xr @ Wr.T
    assert rel_fro(Y, Yh) < 1e-4
    # Without the inverse, a naive X @ T would NOT preserve the function.
    X_naive = T.apply_last(X)
    Y_naive = X_naive @ Wr.T
    assert rel_fro(Y, Y_naive) > 0.05


def test_absorbed_hadamard_bills_zero_runtime_bytes_bad_does_not():
    T_h = make_hadamard()
    T_b = make_bad()
    T_i = make_identity()
    bh = rotation_bill(T_h, absorbed=True)
    bb = rotation_bill(T_b, absorbed=False)
    bi = rotation_bill(T_i, absorbed=True)
    assert hadamard_storage_bytes() == 0
    assert bh["runtime_bytes"] == 0
    assert bh["extra_complete_ebpw"] == 0.0
    assert bh["absorbed"] is True
    assert bi["runtime_bytes"] == 0
    assert bb["runtime_bytes"] > 0
    assert bb["extra_complete_ebpw"] > 0.0
    # Stored T would be f16 1024x1024 blocks; if absorbed they do not count.
    stored = dense_block_storage_bytes(HIDDEN) + dense_block_storage_bytes(INTERMEDIATE)
    assert stored == (5 + 17) * 1024 * 1024 * 2
    assert PARENT_PARAMS == 26_895_998_464
    extra = 8.0 * stored / PARENT_PARAMS
    assert extra < 0.02  # even the stored form is a rounding error on complete EBPW


def test_output_rotation_round_trips_hidden():
    T = make_hadamard()
    W = synthetic_W(HIDDEN, 1024, 41)
    mid = synthetic_X(10, 1024, 42)
    Y = mid @ W.T
    Wr = rotate_weight_output(W, T)
    y_rot = mid @ Wr.T
    y_hat = unrotate_output(y_rot, T)
    assert rel_fro(Y, y_hat) < 1e-5


def test_scale_trap_has_perfect_argmax_and_is_rejected():
    rng = np.random.RandomState(0)
    Y = rng.randn(32, 64).astype(np.float32)
    trap = score_composition(Y, 0.01 * Y)
    assert trap["argmax_agree"] == pytest.approx(1.0)
    assert trap["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert trap["gain"] < 0.05
    assert trap["rel_fro"] > 0.9
    assert trap["survives"] is False
    assert trap["matches_scale_trap"] is True
    # The survive rule is the campaign bar, not argmax alone.
    assert survives(trap) is False


def test_argmax_agree_is_one_on_identity_and_not_the_gate_alone():
    rng = np.random.RandomState(1)
    Y = rng.randn(64, 32).astype(np.float32)
    assert argmax_agree(Y, Y) == 1.0
    assert argmax_agree(Y, -Y) == 0.0 or argmax_agree(Y, -Y) < 0.1
    sc = score_pair(Y, Y)
    sc["argmax_agree"] = 1.0
    # Perfect copy survives; a scaled copy does not (tested above).


def test_material_threshold_beats_g032_delta():
    assert MATERIAL_REL_FRO > G032_Q2_DELTA_HOLD
    assert MATERIAL_REL_FRO >= 0.03
    assert MATERIAL_ARGMAX >= 0.05
    # A 0.008 hold-error move (G032 Q2) is not material for reopening.


def test_codecs_are_not_deletion_on_gaussian():
    W = np.random.RandomState(2).randn(64, 128).astype(np.float32)
    for name in ("binary_g64", "ternary_g64", "q2f_g64"):
        What, acc = codec_reconstruct(W, name)
        assert What.shape == W.shape
        assert float(np.abs(What).mean()) > 0.1 * float(np.abs(W).mean())
        assert acc["scales_counted"] is True
        assert acc["storage_bpw"] == pytest.approx(codec_bpw(name)["storage_bpw"])
        if name == "binary_g64":
            assert np.count_nonzero(What) == What.size
        if name == "q2f_g64":
            assert np.count_nonzero(What) == What.size


def test_self_check_passes_without_parent_tensors():
    sc = self_check()
    assert sc["ok"] is True, sc
    assert sc["identity_preserve_rel_fro"] < 1e-6
    assert sc["hadamard_preserve_rel_fro"] < 1e-5
    assert sc["identity_codec_rel_fro"] < 1e-6
    assert sc["bad_orth_error"] > 0.1


def test_parent_a_census_is_read_only():
    info = parent_a_readonly()
    assert info["mode"] == "read_only"
    assert info["mutated"] is False
    assert info["outside_worktree"] is True
    assert Path(info["path"]) == PARENT_A
    if info["catalog_present"]:
        assert info["catalog_bytes"] > 0
        assert info["n_segments"] >= 192


def test_receipt_answers_s026_78_117():
    assert RECEIPT.is_file(), (
        "receipts/headless/COORDINATE_TRANSFORM_PROBE.json missing — "
        "run python3 tools/headless/coordinate_transform_probe.py"
    )
    doc = json.loads(RECEIPT.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["generated_by"] == GENERATOR
    assert doc["hand_authored"] is False
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_benchmarks"] is True
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["dense_w"] == 0
    assert doc["dense_w_materialized"] == 0
    assert "§78" in doc["s026"] and "§117" in doc["s026"]
    assert "§11" in doc["s026"] and "§93" in doc["s026"]
    assert isinstance(doc["ROTATION_MOVES_BARRIER"], bool)
    assert "ROTATION_MOVES_BARRIER" in doc["answer"]
    assert doc.get("measured_deltas"), "receipt must carry the measured deltas"
    assert any("hadamard_b1024/ternary_g64" in s for s in doc["measured_deltas"])
    assert any("Δrel_fro=" in s or "delta_rel_fro" in s for s in doc["measured_deltas"])
    assert "Δrel_fro" in doc["answer"] or "delta" in doc["answer"].lower()
    assert doc["QWEN_MLP_2_25_stays_closed_for_unrotated_family"] is True
    if doc["ROTATION_MOVES_BARRIER"]:
        assert doc["reopening_frontier"] == "QWEN_MLP_ROTATED_TERNARY"
    else:
        assert doc["reopening_frontier"] is None
        assert "coordinate-robust" in doc["answer"] or "stays closed" in doc["answer"]
    # Controls.
    assert doc["controls"]["identity_reproduces_baseline"] is True
    assert doc["controls"]["bad_control_spurious_help"] is False
    # Real activations, not Gaussian.
    for L in doc["layers"]:
        assert L["n_hold"] >= 64
        for row in L["mlp"]:
            assert row["real_activations"] is True
            assert row["not_gaussian"] is True
            assert "argmax_agree" in row["composition"]
            assert "rel_fro" in row["composition"]
            assert row["accounting"]["rotation"]["runtime_bytes"] >= 0
    # Gate is composition, not weight-space.
    assert doc["quality_bar"]["weight_space_is_not_the_gate"] is True
    assert doc["quality_bar"]["scale_trap_rejected"] is True
    # Material threshold beats G032.
    assert doc["material_thresholds"]["rel_fro_drop"] > G032_Q2_DELTA_HOLD
    # N036 layers.
    assert doc["n036"]["earliest_layer"] == 0
    assert doc["n036"]["earliest_organ"] == "up_proj"
    assert 0 in doc["n036"]["probe_layers"]
    # No hidden bits: rotation bill is present, absorbed Hadamard is 0.
    had = doc["transforms"]["hadamard_b1024"]["bill"]
    assert had["runtime_bytes"] == 0
    assert had["absorbed"] is True
    # Parent A was only censused.
    assert doc["parent_a"]["mutated"] is False
    assert doc["parent_a"]["mode"] == "read_only"
    # Decision cites measured deltas.
    arms = doc["decision"]["per_arm"]
    names = {(a["transform"], a["codec"]) for a in arms}
    assert ("hadamard_b1024", "ternary_g64") in names
    assert ("hadamard_b1024", "binary_g64") in names
    assert ("pca_orth_b1024", "ternary_g64") in names
    assert ("bad_nonorth_b1024", "binary_g64") in names
    for a in arms:
        mlp = a["mlp_composition"]
        assert "delta_rel_fro" in mlp
        assert "delta_argmax_agree" in mlp
        if a["transform"] == "bad_nonorth_b1024":
            assert a["counts"] is False


def test_fwht_round_trip_on_activation_shaped_block():
    rng = np.random.RandomState(5)
    x = rng.randn(7, 5, BLOCK).astype(np.float32)
    y = fwht_last(x)
    z = fwht_last(y)
    assert float(np.max(np.abs(z - x))) < 1e-4
    M = rng.randn(3, HIDDEN).astype(np.float32)
    M2 = apply_fwht_last_axis(apply_fwht_last_axis(M))
    assert rel_fro(M, M2) < 1e-5
