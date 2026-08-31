"""Tests for DeltaNet generated transition coefficients.

Load-bearing refusals:

1. A verdict without multi-step results at 64 steps or more is REFUSED.
2. A train figure cannot be reported as held-out.
3. Synthetic / held-out-leak corpora are REFUSED (MLP guards, not a weaker copy).
4. Emit-W-then-ordinary-GEMV is REJECTED_DENSE_REMAT and scores removed == added.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.future import deltanet_generated_transition as dgt
from tools.future import deltanet_state_function as dsf
from tools.future import deltanet_teacher_corpus as dtc
from tools.future import mlp_teacher_corpus as mtc
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_held_out_split_refuses_shared_prompt_id():
    """NEGATIVE CONTROL: a prompt that sits on both sides must not emit."""
    rows = dtc.make_diverse_fixture_corpus(4, 3)
    split = dtc.split_by_prompt(rows)
    assert not (set(split["train_prompt_ids"]) & set(split["hold_prompt_ids"]))
    ok = dtc.emit_manifest(rows, split, allow_fixture=True, require_sizing=False)
    assert ok["accepted"] is True

    leaked = dict(split)
    leaked["train_prompt_ids"] = list(split["train_prompt_ids"]) + [
        split["hold_prompt_ids"][0]
    ]
    with pytest.raises(mtc.CorpusRefused) as caught:
        dtc.emit_manifest(rows, leaked, allow_fixture=True, require_sizing=False)
    assert "HELD_OUT_PROMPT_LEAK" in caught.value.codes
    assert "REFUSED" in str(caught.value)


def test_synthetic_row_refused():
    """NEGATIVE CONTROL: Gaussian X cannot close the corpus (NNS-001)."""
    rows = dtc.make_diverse_fixture_corpus(4, 3)
    split = dtc.split_by_prompt(rows)
    poisoned = list(rows)
    poisoned[3] = dtc.make_gaussian_row(rows[3])
    with pytest.raises(mtc.CorpusRefused) as caught:
        dtc.emit_manifest(poisoned, split, allow_fixture=True, require_sizing=False)
    assert "SYNTHETIC_ROW" in caught.value.codes
    assert "REFUSED" in str(caught.value)


def test_corpus_guards_are_the_mlp_guards_not_a_weaker_copy():
    assert dtc.CorpusRefused is mtc.CorpusRefused
    assert dtc.is_synthetic_row is mtc.is_synthetic_row
    assert dtc.split_by_prompt is mtc.split_by_prompt
    row = dtc.make_diverse_fixture_corpus(2, 3)[0]
    for key in (
        "pre_state_sha256",
        "post_state_sha256",
        "output_sha256",
        "prompt_id",
        "token_position",
        "capability_domain",
        "layer",
    ):
        assert row.get(key) not in (None, "")


def test_layer0_is_not_typical():
    reps = dtc.pick_representatives()
    assert reps["layer0_typical"] is False
    typical = next(p for p in reps["chosen"] if p["role"] == "typical")
    assert typical["layer"] != 0
    assert 0 in reps["chosen_layers"]
    for layer in reps["chosen_layers"]:
        assert dtc.is_linear_attention_layer(layer)


def test_verdict_refused_without_64_steps():
    """The module must refuse to report a verdict without 64-step results."""
    held = {"split": "hold", "source": "hold", "prompt_ids": [], "one_step_cosine": 0.999}
    with pytest.raises(dgt.VerdictRefuse) as caught:
        dgt.report_verdict(None, held)
    assert "64" in str(caught.value)
    assert "REFUSED" in str(caught.value)

    with pytest.raises(dgt.VerdictRefuse) as caught16:
        dgt.report_verdict(
            {
                "checkpoints": [1, 4, 16],
                "n_steps": 16,
                "state_relfro": {"1": 0.001, "4": 0.002, "16": 0.003},
                "state_relfro_dense": [0.001] * 16,
            },
            held,
        )
    assert "64" in str(caught16.value)

    # 256 without the 64 checkpoint is still a refusal: the contract names 64.
    with pytest.raises(dgt.VerdictRefuse):
        dgt.report_verdict(
            {
                "checkpoints": [1, 4, 16, 256],
                "n_steps": 256,
                "state_relfro_dense": [0.0] * 256,
            },
            held,
        )

    toy = dgt.fixture_multistep(n_steps=64)
    verdict = dgt.report_verdict(toy, held)
    assert verdict["max_step"] >= 64
    assert 64 in verdict["checkpoints"]
    assert verdict["held_out_split"] == "hold"
    assert verdict["verdict"] in {"PASS", "FAIL"}


def test_train_figure_cannot_be_reported_as_held_out():
    toy = dgt.fixture_multistep(n_steps=64)
    with pytest.raises(dgt.HeldOutRefuse) as caught:
        dgt.report_verdict(
            toy,
            {"split": "train", "source": "train", "prompt_ids": ["code:00"], "one_step_cosine": 0.999},
        )
    assert "held-out" in str(caught.value).lower() or "train" in str(caught.value).lower()
    assert "REFUSED" in str(caught.value)

    with pytest.raises(dgt.HeldOutRefuse):
        dgt.assert_held_out({"split": "fit", "prompt_ids": []})

    with pytest.raises(dgt.HeldOutRefuse):
        dgt.assert_held_out(
            {"split": "hold", "prompt_ids": ["code:00"]},
            split={"train_prompt_ids": ["code:00"], "hold_prompt_ids": ["code:15"]},
        )

    dgt.assert_held_out(
        {"split": "hold", "prompt_ids": ["code:15"]},
        split={"train_prompt_ids": ["code:00"], "hold_prompt_ids": ["code:15"]},
    )


def test_emit_w_then_gemv_is_rejected_dense_remat_and_scores_removed_equals_added():
    tag = dgt.consume_path_tag(emit_dense_w=True, ordinary_gemv=True)
    assert tag == dgt.REJECTED_DENSE_REMAT
    assert dgt.consume_path_tag(emit_dense_w=False, ordinary_gemv=False) == dgt.DIRECT_CONSUME

    rng = np.random.default_rng(0)
    T1 = rng.normal(size=(8, 32)).astype(np.float32)
    T2 = rng.normal(size=(16, 8)).astype(np.float32)
    d = np.ones(16, dtype=np.float32)
    e = np.zeros(8, dtype=np.float32)
    x = rng.normal(size=(32,)).astype(np.float32)
    y = dgt.native_consume(T1, T2, d, e, x)
    assert y.shape == (16,)
    with pytest.raises(dgt.DenseRematRefuse) as caught:
        dgt.emit_w_then_gemv(T1, T2, d, e, x)
    assert "REJECTED_DENSE_REMAT" in str(caught.value)

    scored = dgt.score_dense_remat()
    assert scored["dense_rematerialization"] == dgt.REJECTED_DENSE_REMAT
    assert int(scored["bytes_removed"]) == int(scored["bytes_added"]["total"])
    assert int(scored["net_bytes"]) == 0
    assert int(scored["bytes_removed"]) == dsf.QKVZ_ACTIVE_TARGET


def test_bytes_come_from_the_recorded_candidate_not_an_invented_model():
    cand = dgt.load_candidate()
    assert cand["id"] == dgt.CANDIDATE_ID
    billed = dgt.billed_bytes(cand)
    assert billed["bytes_removed"]["total"] == 2_139_096_960
    assert billed["bytes_removed"]["total"] == dsf.QKVZ_ACTIVE_TARGET
    assert billed["bytes_added"]["total"] == 4_548_560
    assert billed["bytes_added"]["generator"] == 2_924_624
    assert billed["bytes_added"]["embeddings"] == 49_152
    assert billed["bytes_added"]["residuals"] == 1_572_864
    assert billed["bytes_added"]["metadata"] == 1_920
    assert billed["bytes_added"]["generator"] + billed["bytes_added"]["embeddings"] + billed[
        "bytes_added"
    ]["residuals"] + billed["bytes_added"]["metadata"] == billed["bytes_added"]["total"]
    # Counted once at model scope, not per token and not per layer-copy of T1/T2.
    assert billed["bytes_added"]["generator"] == billed["bytes_added"]["total"] - (
        49_152 + 1_572_864 + 1_920
    )


def test_economics_scores_the_candidate_through_the_module():
    scored = dgt.score_candidate()
    assert scored["ok"] is True
    assert scored["organ"] == "deltanet"
    assert scored["consuming_primitive"] == "TiledProjection"
    assert scored["consuming_primitive"] in ATLAS_PRIMITIVES
    assert scored["bytes_removed"] == 2_139_096_960
    assert scored["bytes_added"]["total"] == 4_548_560
    assert scored["bytes_added"]["generator"] == 2_924_624
    assert scored["dense_rematerialization"] == dgt.DIRECT_CONSUME
    assert scored["state_update"]["latency_delta"] == 0
    assert scored["dispatch_change"]["delta"] == 48
    # FLOP save is recorded, not inverted into a negative extra-FLOP the
    # economics module would refuse.
    assert scored["extra_flops_per_token_recorded"] < 0
    assert scored["extra_flops_per_output_element"] == 0.0


def test_qkvz_coding_is_not_retried():
    floor = dgt.qkvz_coding_is_at_floor()
    assert floor["retried"] is False
    assert set(floor["not_worth_touching"]) >= {"q", "k", "v", "z"}
    assert floor["total_bytes_eliminated"] == 0
    assert all(int(v) == 4 for v in floor["bits"].values())


def test_multistep_crossing_and_one_step_are_not_a_verdict_alone():
    """Excellent one-step with later drift is a FAIL named at N."""
    n = 64
    dense = [0.0] * n
    dense[0] = 0.001  # one-step looks excellent
    dense[31] = 0.012  # crosses 1% at step 32
    dense[63] = 0.04
    ms = {
        "checkpoints": [1, 4, 16, 64],
        "n_steps": n,
        "state_relfro": {"1": 0.001, "4": 0.001, "16": 0.002, "64": 0.04},
        "state_relfro_dense": dense,
    }
    held = {"split": "hold", "source": "hold", "prompt_ids": [], "one_step_cosine": 0.999}
    verdict = dgt.report_verdict(ms, held)
    assert verdict["verdict"] == "FAIL"
    assert verdict["scar_id"] == "DELTANET_REPLACEMENT_UNSTABLE_AT_N"
    assert verdict["n_unstable"] == 32
    assert verdict["crossings"]["0.01"] == 32
    assert verdict["one_step_cosine"] == pytest.approx(0.999)
    assert verdict["mechanism"]
    assert "step 32" in verdict["mechanism"]


def test_native_consume_never_materializes_w():
    rng = np.random.default_rng(1)
    T1 = rng.normal(size=(4, 8)).astype(np.float32)
    T2 = rng.normal(size=(16, 4)).astype(np.float32)
    d = rng.normal(size=(16,)).astype(np.float32)
    e = rng.normal(size=(4,)).astype(np.float32)
    x = rng.normal(size=(8,)).astype(np.float32)
    y = dgt.native_consume(T1, T2, d, e, x)
    # Direct program: two skinny GEMVs + add + diagonal. Equivalent arithmetic
    # to (d[:,None] * T2 @ T1) @ x + d * (T2 @ e) — that identity is the
    # *check*, not the production path.
    z = T1 @ x + e
    expect = d * (T2 @ z)
    assert y == pytest.approx(expect, rel=1e-5, abs=1e-5)
    X = rng.normal(size=(3, 8)).astype(np.float32)
    Y = dgt.native_consume(T1, T2, d, e, X)
    assert Y.shape == (3, 16)


def test_pack_q4_roundtrip_is_the_billed_codec():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(64, 128)).astype(np.float32)
    codes, scales = dgt.pack_q4(W)
    rec = dgt.unpack_q4(codes, scales)
    assert rec.shape == W.shape
    # Signed-nibble Q4 is lossy; cosine must stay high on this isotropic draw.
    assert dgt._cosine_mat(rec, W) > 0.95


def test_build_emits_sealed_receipt():
    out = dgt.build(fit=False)
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "DELTANET_GENERATED_TRANSITION.json"
    assert doc["schema"] == dgt.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "DIAGNOSTIC_RELATIVE"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc
    assert doc["selftest"]["verdict_without_64_refused"] is True
    assert doc["selftest"]["train_as_held_out_refused"] is True
    assert doc["selftest"]["held_out_leak_refused"] is True
    assert doc["selftest"]["synthetic_refused"] is True
    assert doc["bytes"]["bytes_removed"]["total"] == 2_139_096_960
    assert doc["bytes"]["bytes_added"]["total"] == 4_548_560
    assert doc["dense_rematerialization"]["emit_w_then_ordinary_gemv"] == dgt.REJECTED_DENSE_REMAT
    assert doc["dense_rematerialization"]["emit_w_economics_removed_equals_added"] is True
    assert doc["qkvz_precision"]["retried"] is False
    assert doc["economics"]["consuming_primitive"] == "TiledProjection"
    assert doc["fusion_env"]["HAWKING_QWEN38_FUSE_DN_INPROJ"] == "1"
    assert doc["candidate_id"] == dgt.CANDIDATE_ID
    # A receipt without a 64-step fit must not smuggle a licensed PASS.
    if doc.get("verdict") is None:
        pass
    else:
        assert doc["verdict"]["max_step"] >= 64
        assert 64 in doc["verdict"]["checkpoints"]
        assert doc["verdict"]["held_out_split"] == "hold"


def test_selftest_proves_both_halves():
    result = dgt.selftest()
    assert result["verdict_without_64_refused"] is True
    assert result["train_as_held_out_refused"] is True
    assert result["held_out_leak_refused"] is True
    assert result["synthetic_refused"] is True
    assert result["dense_remat"] == dgt.REJECTED_DENSE_REMAT
    assert result["dense_remat_removed_equals_added"] is True
    assert result["qkvz_coding_retried"] is False
    assert result["bytes_added_total"] == 4_548_560
    assert result["bytes_removed_total"] == 2_139_096_960
    assert result["corpus"]["layer0_typical"] is False


def test_module_entrypoint_build():
    rc = dgt.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / dgt.RECEIPT).read_text())
    assert doc["schema"] == dgt.SCHEMA
    assert doc["seal_sha256"]
