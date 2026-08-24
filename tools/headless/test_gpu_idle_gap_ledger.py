"""N029 GPU idle-gap ledger: classified intervals, ranked, largest attacked."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gpu_idle_gap_ledger import (  # noqa: E402
    ABSENT,
    CAUSES,
    DERIVED,
    FRONTIER_DISPATCHES,
    INTRA_CB_ABSENT_REASON,
    MEASURED,
    RECEIPT,
    SCHEMA,
    aggregate_causes,
    attack_verdict,
    classify_token_intervals,
    separated,
    shader_evidence,
)

RECEIPT_DOC = None
KINDS = {MEASURED, DERIVED, ABSENT}


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_IDLE_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        from gpu_idle_gap_ledger import build, write_receipt  # noqa: WPS433

        RECEIPT_DOC = build(live=True)
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def _fake_token(encode=900_000, sync=400_000, sample=800, alloc=2_000, residual=500) -> dict:
    return {
        "role": "decode",
        "intervals": [
            {"cause": "allocation", "ns": alloc, "kind": MEASURED, "where": "tcb new"},
            {
                "cause": "command construction",
                "ns": encode,
                "kind": MEASURED,
                "where": "encode+submit",
            },
            {"cause": "sync", "ns": sync, "kind": MEASURED, "where": "wait-gpu"},
            {"cause": "sampler", "ns": sample, "kind": MEASURED, "where": "argmax u32"},
            {
                "cause": "state bookkeeping",
                "ns": 8_000,
                "kind": MEASURED,
                "where": "pos+tokenizer",
            },
            {"cause": "CPU sched", "ns": residual, "kind": MEASURED, "where": "gap"},
            {"cause": "Python", "ns": 0, "kind": MEASURED, "where": "none"},
            {"cause": "runtime lock", "ns": 0, "kind": MEASURED, "where": "none"},
            {"cause": "serialization", "ns": 0, "kind": MEASURED, "where": "none"},
            {
                "cause": "dependency",
                "ns": None,
                "kind": ABSENT,
                "absent_reason": INTRA_CB_ABSENT_REASON,
            },
        ],
    }


def test_causes_are_the_required_ten():
    assert CAUSES == (
        "dependency",
        "CPU sched",
        "command construction",
        "allocation",
        "serialization",
        "Python",
        "sampler",
        "state bookkeeping",
        "sync",
        "runtime lock",
    )


def test_classify_emits_every_cause_and_absent_is_never_zero():
    rows = classify_token_intervals(_fake_token())
    assert [r["cause"] for r in rows] == list(CAUSES)
    dep = next(r for r in rows if r["cause"] == "dependency")
    assert dep["kind"] == ABSENT
    assert dep["ns"] is None
    assert dep["ns"] != 0
    assert "atDispatchBoundary" in (dep.get("absent_reason") or "")
    py = next(r for r in rows if r["cause"] == "Python")
    assert py["kind"] == MEASURED
    assert py["ns"] == 0


def test_rank_puts_command_construction_first():
    ranked = aggregate_causes([_fake_token(), _fake_token()])
    measured = [r for r in ranked if r["kind"] == MEASURED]
    assert measured[0]["cause"] == "command construction"
    assert measured[0]["rank"] == 1
    assert measured[0]["total_idle_ns_per_token"] == 900_000
    assert measured[1]["cause"] == "sync"
    absent = [r for r in ranked if r["kind"] == ABSENT]
    assert len(absent) == 1
    assert absent[0]["cause"] == "dependency"
    assert absent[0]["total_idle_ns_per_token"] is None


def test_separation_helper_refuses_overlap():
    assert separated([1.0, 2.0], [3.0, 4.0]) is True
    assert separated([1.0, 3.0], [2.0, 4.0]) is False
    assert separated([], [1.0]) is False


def test_attack_verdict_overlap_is_not_separated_and_not_a_mean_delta():
    noop = {
        "complete_wall_ns_reps": [30e6, 31e6, 30.5e6, 30.2e6, 30.8e6, 30.1e6, 30.4e6],
        "new_token_ids": [1, 2, 3],
        "ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 400_000}],
    }
    serial = {
        "complete_wall_ns_reps": [29.8e6, 31.1e6, 30.4e6, 30.0e6, 30.9e6, 30.3e6, 30.2e6],
        "new_token_ids": [1, 2, 3],
        "ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 350_000}],
    }
    split = {
        "ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 20_000_000}],
    }
    v = attack_verdict(noop, serial, split)
    assert v["separated"] is False
    assert v["outcome"] == "NOT SEPARATED"
    assert "NOT SEPARATED" in v["note"]
    assert v["token_ids_unchanged"] is True
    assert v["bad_control_rejected"] is True
    assert v["dense_w_materialized"] == 0
    assert "mean" not in v["note"].lower() or "NOT SEPARATED" in v["note"]


def test_attack_verdict_refuses_identity_change():
    noop = {
        "complete_wall_ns_reps": [30e6] * 7,
        "new_token_ids": [1, 2, 3],
        "ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 1}],
    }
    serial = {
        "complete_wall_ns_reps": [20e6] * 7,
        "new_token_ids": [9, 9, 9],
        "ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 1}],
    }
    split = {"ranked_causes": [{"cause": "sync", "total_idle_ns_per_token": 2}]}
    v = attack_verdict(noop, serial, split)
    assert v["outcome"] == "REFUSED_IDENTITY"
    assert v["token_ids_unchanged"] is False


def test_shader_evidence_serial_encoder_is_wired():
    ev = shader_evidence()
    assert ev["decode_present"]
    assert ev["serial_token_encoder_wired"]
    assert ev["gpu_start_on_step_wall"]
    assert ev["encoder_count_on_tcb"]
    assert ev["no_new_kernels"]


def test_receipt_schema_causes_rank_and_no_second_27b():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert "N029" in doc["obligation"]
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert "NOETIC_PARENT_A" in doc["parent_immutable"]["path"]
    assert doc["parent_immutable"]["outside_worktree"] is True
    assert "27B" in doc["occupancy"]["note"] or "10+ GiB" in doc["occupancy"]["note"]
    assert doc["causes"] == list(CAUSES)
    assert doc["gpu_timestamp_authority"].startswith("completed MTLCommandBuffer")
    assert "CPU-wait proxy" in doc["gpu_timestamp_authority"]
    assert doc["dense_w_materialized"] == 0
    assert doc["causal_benchmark_law"]["dispatch_count"] == FRONTIER_DISPATCHES


def test_every_cause_ranked_and_absent_never_zero():
    doc = receipt()
    ranked = doc["ranked_by_idle_ns_per_token"]
    names = [r["cause"] for r in ranked]
    assert set(names) == set(CAUSES)
    measured = [r for r in ranked if r["kind"] == MEASURED]
    ranks = [r["rank"] for r in measured]
    assert ranks == list(range(1, len(measured) + 1))
    ns = [r["total_idle_ns_per_token"] for r in measured]
    assert ns == sorted(ns, reverse=True)
    for r in ranked:
        if r["kind"] == ABSENT:
            assert r["total_idle_ns_per_token"] is None
            assert r["total_idle_ns_per_token"] != 0
            assert r.get("absent_reason")
            assert any(
                s in r["absent_reason"].lower()
                for s in ("dispatch", "boundary", "command buffer", "sample")
            )
        else:
            assert r["kind"] in (MEASURED, DERIVED)
            assert r["total_idle_ns_per_token"] is not None
            assert r["total_idle_ns_per_token"] >= 0


def test_intervals_cover_causes_and_intra_cb_is_absent():
    doc = receipt()
    intra = doc["intra_cb_gpu_idle_ns"]
    assert intra["kind"] == ABSENT
    assert intra["value"] is None
    assert intra["value"] != 0
    assert "atDispatchBoundary" in intra["absent_reason"]
    tokens = doc["intervals_per_token"]
    assert tokens, "need at least one instrumented token"
    for tok in tokens:
        causes = [i["cause"] for i in tok["intervals"]]
        assert set(causes) == set(CAUSES)
        dep = next(i for i in tok["intervals"] if i["cause"] == "dependency")
        assert dep["kind"] == ABSENT
        assert dep["ns"] is None
        assert tok.get("gpu_start_s") is not None or tok.get("gpu_ns") is not None


def test_largest_attacked_or_measured_reason():
    doc = receipt()
    largest = doc["largest_cause"]
    assert largest is not None
    assert largest["cause"] in CAUSES
    assert largest["kind"] == MEASURED
    attack = doc["attack"]
    assert attack["target_cause"] == "command construction"
    assert attack["outcome"] in {
        "ATTACKED_AND_SEPARATED",
        "NOT SEPARATED",
        "ATTACKED_SLOWER",
        "REFUSED_IDENTITY",
        "NOT_MEASURED",
    }
    if attack["outcome"] == "NOT SEPARATED":
        assert "NOT SEPARATED" in attack["note"]
        assert attack["separated"] is False
    if attack["outcome"] != "NOT_MEASURED":
        assert attack["dense_w_materialized"] == 0
        assert isinstance(attack["token_ids_unchanged"], bool)
    assert doc["controls"]["reps"] >= 7
    assert "NOT SEPARATED" in doc["controls"]["report"]
    if attack["outcome"] != "NOT_MEASURED":
        assert attack["bad_control_rejected"] is True
        assert doc["arms"]["serial"].get("encoder_count") == 1
        assert doc["arms"]["noop"].get("encoder_count") == doc["arms"]["noop"].get("dispatches")


def test_kernel_autopsy_and_seven_reps():
    doc = receipt()
    assert doc["kernel_autopsy"]["any_new_kernel_defective"] is False
    assert doc["kernel_autopsy"]["ok"] is True
    noop = doc["arms"]["noop"]
    if noop.get("kind") == MEASURED:
        assert noop["n_reps"] >= 7
        assert noop["complete_wall_ns_min"] <= noop["complete_wall_ns_median"]
        assert noop["complete_wall_ns_median"] <= noop["complete_wall_ns_max"]
        assert noop["dense_w_materialized"] == 0
        serial = doc["arms"]["serial"]
        if serial.get("kind") == MEASURED and serial.get("new_token_ids"):
            assert serial["new_token_ids"] == noop.get("new_token_ids")


def test_labelled_quantities_never_fabricate_absent_zero():
    doc = receipt()
    for name in ("complete_token_wall_ns", "gpu_busy_ns", "wall_minus_gpu_ns", "intra_cb_gpu_idle_ns"):
        q = doc[name]
        assert q["kind"] in KINDS, name
        assert q["command"], name
        if q["kind"] == ABSENT:
            assert q["value"] is None, name
            assert q["value"] != 0
            assert q.get("absent_reason"), name
        else:
            assert isinstance(q["value"], (int, float)), name
