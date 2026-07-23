"""Tests for the mandatory post-parent global review."""
from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import post_parent_review as ppr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))


def synthetic_evidence(run_status="honest_boundary_sealed"):
    return {
        "schema": "hawking.foundry.parent_evidence.v1",
        "parent": {"id": "synth-9b", "label": "9B", "generation": "FT"},
        "run_status": run_status,
        "representation": {
            "winners": [{"name": "vq_d32_k65536", "rate_bpw": 0.5, "rel_error": 0.668}],
            "rate_response": [{"rate_bpw": 0.4, "rel_error": 0.71}, {"rate_bpw": 0.5, "rel_error": 0.668}],
        },
        "organ_sensitivity": {
            "organs": {"gate": {"sensitivity": "HIGH"}, "up": {"sensitivity": "HIGH"}, "down": {"sensitivity": "LOWER"}},
            "dominant_failure_organ": "gate",
            "inversion_confirmed": True,
        },
        "doctor": {"successes": [{"target": "down"}], "failures": [{"target": "gate"}]},
        "routing": {"calibration_tokens": 1000, "required_calibration_tokens": 1000},
        "activation": {"inter_expert_mean_pairwise_cosine": 0.0001},
        "quality": {
            "capability_gate": {"mean_symmetric_kl_max": 0.10, "next_token_argmax_agreement_min": 0.95},
            "result": {"selected_frontier": None},
            "failures": [{"probe": "real_forward", "domain": "logits"}],
            "probes": ["real_forward"],
            "domains_collapsing_first": ["logits"],
        },
        "runtime": {"timings": [{"stage": "pack", "seconds": 10}], "dominant_bottleneck": "expert streaming"},
        "resources": {"memory_floor_gib": 20},
        "source_format": {"lessons": [{"id": "packed_blocks"}], "decoder_requirements": ["packed block decode"]},
        "storage": {"lessons": [{"id": "release_after_harvest"}]},
        "assumptions": [
            {"id": "organ_inversion", "statement": "mlp1 sensitive", "verdict": "CONFIRMED", "evidence": "x"},
            {"id": "cache_64", "statement": "big cache helps", "verdict": "FALSIFIED", "evidence": "y"},
            {"id": "row_norm", "statement": "stratification helps", "verdict": "OPEN"},
        ],
        "methods": [{"id": "m1", "name": "row-norm stratification", "status": "UNTESTED", "transfer_breadth": 2.0}],
        "next_parent": {"storage": {"free_gib": 500, "required_gib": 120, "headroom_gib": 50}},
    }


def adapters(acks=None):
    return [
        {
            "id": "qwen3_moe",
            "consumes_parent_lessons": ["synth-9b"],
            "unverified_assumptions": ["row_norm"],
            "falsification_plan": "re-measure gate/up rel_err at matched rate; if down beats gate the inversion prior is dead",
            "subbit_closure_plan": {
                "rates": ["1/1", "1/2"],
                "method_families": ["quantization_aware_training", "distillation"],
            },
            "rebase_acks": acks or {},
        }
    ]


# ---------------------------------------------------------------- artifacts


def test_generate_writes_every_artifact(tmp_path):
    out = str(tmp_path / "review")
    written = ppr.generate(synthetic_evidence(), out, adapters())
    for name in ppr.REQUIRED_PER_PARENT:
        assert os.path.exists(os.path.join(out, "SYNTH_9B_" + name)), name
    for name in ppr.REQUIRED_GLOBAL:
        assert os.path.exists(os.path.join(out, name)), name
    assert len(written) == len(ppr.REQUIRED_PER_PARENT) + len(ppr.REQUIRED_GLOBAL)

    harvest = json.load(open(written["VULTURE_HARVEST.json"]))
    assert harvest["organ_sensitivity"]["dominant_failure_organ"] == "gate"
    assert harvest["provisional"] is False
    # every falsified prior travels forward as negative transfer
    ids = {n["id"] for n in harvest["negative_transfer_constraints"]}
    assert {"inter_expert_redundancy_zero", "entropy_coding_pq_indices", "aggressive_expert_cache"} <= ids

    review = json.load(open(written["GLOBAL_METHODOLOGY_REVIEW.json"]))
    assert [a["id"] for a in review["assumptions_falsified"]] == ["cache_64"]
    assert set(review["questions"]) == {k for k, _ in ppr._REVIEW_QUESTIONS}

    md = open(written["GLOBAL_METHODOLOGY_REVIEW.md"]).read()
    assert "dominant failure organ: gate" in md
    for ch in ("—", "–", "·"):
        assert ch not in md

    matrix = json.load(open(written["ADAPTER_REBASE_MATRIX.json"]))
    entry = matrix["adapters"]["qwen3_moe"]
    assert entry["sensitive_organ_prior"] == "gate"
    assert entry["cache_policy"]["expert_cache_cap_gib"] == 20
    assert entry["routing_calibration_plan"]["min_calibration_tokens"] == 1000
    assert any(c.startswith("DO_NOT_ATTEMPT:") for c in entry["candidate_ordering"])
    assert entry["rate_priors"]["not_a_selection"] is True

    resource = json.load(open(written["RESOURCE_REBASE.json"]))
    assert resource["cache_policy"]["expert_cache_cap_gib"] == 20

    promo = json.load(open(written["GRAVITY_METHOD_PROMOTION.json"]))
    assert all(m["potency"] <= 0 for m in promo["retired"])
    assert "gravity_potency" in promo["potency_backend"]

    ledger = [json.loads(x) for x in open(written["PROVIDER_ADAPTER_LESSON_LEDGER.jsonl"]) if x.strip()]
    assert ledger and all(r["parent_id"] == "synth-9b" for r in ledger)
    # idempotent: a second run appends nothing
    ppr.generate(synthetic_evidence(), out, adapters())
    again = [json.loads(x) for x in open(written["PROVIDER_ADAPTER_LESSON_LEDGER.jsonl"]) if x.strip()]
    assert len(again) == len(ledger)


def test_refuses_weakened_capability_gate():
    ev = synthetic_evidence()
    ev["quality"]["capability_gate"]["mean_symmetric_kl_max"] = 0.25
    with pytest.raises(ValueError, match="weakened capability gate"):
        ppr.validate_evidence(ev)


# ---------------------------------------------------------------- drift


def test_find_stale_adapters_flags_unrebased(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    d = matrix["adapters"]["qwen3_moe"]["prescription_digest"]

    stale = ppr.find_stale_adapters(ev["parent"], adapters(), matrix)
    assert stale[0]["adapter_id"] == "qwen3_moe"
    assert "no rebase_ack" in stale[0]["reasons"][0]

    acked = adapters({"synth-9b": {"prescription_digest": d}})
    assert ppr.find_stale_adapters(ev["parent"], acked, matrix) == []

    # priors change under the adapter -> the old ack goes stale by itself
    drifted = ppr.find_stale_adapters(ev["parent"], adapters({"synth-9b": {"prescription_digest": "deadbeef"}}), matrix)
    assert "digest drift" in drifted[0]["reasons"][0]


def test_stale_when_adapter_does_not_declare_falsification(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    d = matrix["adapters"]["qwen3_moe"]["prescription_digest"]
    a = adapters({"synth-9b": {"prescription_digest": d}})
    a[0]["falsification_plan"] = None
    a[0]["unverified_assumptions"] = None
    a[0]["consumes_parent_lessons"] = []
    reasons = ppr.find_stale_adapters(ev["parent"], a, matrix)[0]["reasons"]
    assert len(reasons) == 3


def test_shipped_registry_starts_stale_against_f0():
    reg = json.load(open(os.path.join(_HERE, "adapters", "tier_a_registry.json")))
    ev = ppr.load_evidence(os.path.join(_HERE, "evidence", "f0_gpt_oss_120b.json"))
    stale = ppr.find_stale_adapters(ev["parent"], reg["adapters"])
    assert len(stale) == len(reg["adapters"])


# ---------------------------------------------------------------- gate


def test_gate_blocks_next_parent_before_review(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / "review")
    state = ppr.build_gate_state(out, ev, adapters())
    ok, reasons = ppr.can_launch_next_parent(state)
    assert ok is False
    assert any("missing review artifacts" in r for r in reasons)
    assert any("stale adapters" in r for r in reasons)


def test_gate_allows_next_parent_after_review_and_acks(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / "review")
    written = ppr.generate(ev, out, adapters())
    matrix = json.load(open(written["ADAPTER_REBASE_MATRIX.json"]))
    d = matrix["adapters"]["qwen3_moe"]["prescription_digest"]
    state = ppr.build_gate_state(out, ev, adapters({"synth-9b": {"prescription_digest": d}}), matrix)
    ok, reasons = ppr.can_launch_next_parent(state)
    assert (ok, reasons) == (True, [])

    # the single heavy lease still overrides a satisfied review
    held = ppr.build_gate_state(out, ev, adapters({"synth-9b": {"prescription_digest": d}}), matrix, heavy_lease_held=True)
    ok2, reasons2 = ppr.can_launch_next_parent(held)
    assert ok2 is False and any("heavy lease" in r for r in reasons2)


def test_gate_blocks_on_provisional_review(tmp_path):
    ev = synthetic_evidence(run_status="in_flight")
    out = str(tmp_path / "review")
    written = ppr.generate(ev, out, adapters())
    assert json.load(open(written["VULTURE_HARVEST.json"]))["provisional"] is True
    matrix = json.load(open(written["ADAPTER_REBASE_MATRIX.json"]))
    d = matrix["adapters"]["qwen3_moe"]["prescription_digest"]
    state = ppr.build_gate_state(out, ev, adapters({"synth-9b": {"prescription_digest": d}}), matrix)
    ok, reasons = ppr.can_launch_next_parent(state)
    assert ok is False
    assert any("provisional" in r for r in reasons)
    assert any("not complete or honest_boundary_sealed" in r for r in reasons)


def test_download_may_run_concurrently_while_heavy_is_blocked(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / "review")  # review never run
    state = ppr.build_gate_state(out, ev, adapters(), storage=ev["next_parent"]["storage"])
    assert ppr.can_launch_next_parent(state)[0] is False
    assert ppr.can_start_next_download(state) == (True, [])


def test_download_blocked_only_by_storage(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / "review")
    ppr.generate(ev, out, adapters())
    tight = {"free_gib": 100, "required_gib": 120, "headroom_gib": 50}
    ok, reasons = ppr.can_start_next_download({"storage": tight})
    assert ok is False and "insufficient storage" in reasons[0]
    ok2, reasons2 = ppr.can_start_next_download({"storage": {"free_gib": 500, "required_gib": 1, "download_in_flight": True}})
    assert ok2 is False and "already in flight" in reasons2[0]


# ---------------------------------------------------------------- real F1 consumer


def test_f1_qwen_bundle_is_sealed_and_blocks_on_stale_adapters(tmp_path):
    """F1 sealed at honest_boundary_sealed. The review no longer blocks; adapters do."""
    ev = ppr.load_evidence(os.path.join(_HERE, "evidence", "f1_qwen3_235b.json"))
    assert ev["run_status"] == "honest_boundary_sealed"
    out = str(tmp_path / "review")
    written = ppr.generate(ev, out, adapters())
    harvest = json.load(open(written["VULTURE_HARVEST.json"]))
    assert harvest["provisional"] is False
    assert harvest["organ_sensitivity"]["dominant_failure_organ"] == "gate"
    cross = json.load(open(written["CROSS_PARENT_TRANSFER_MATRIX.json"]))
    assert cross["parents"]["qwen3-235b-a22b-instruct-2507"]["provisional"] is False
    # nothing passed the gate at any rate, and the over-ceiling point is never a seed
    assert harvest["capability_gate_result"]["selected_frontier"] is None
    matrix = json.load(open(written["ADAPTER_REBASE_MATRIX.json"]))
    priors = next(iter(matrix["adapters"].values()))["rate_priors"]
    assert priors["search_start_bpw"] <= 1.0
    assert [r["rate_bpw"] for r in priors["above_ceiling_excluded"]] == [1.007471652]
    # a sealed review alone satisfies the gate; the real Tier-A registry does not
    ok, _ = ppr.can_launch_next_parent(ppr.build_gate_state(out, ev, []))
    assert ok is True
    with open(os.path.join(_HERE, "adapters", "tier_a_registry.json")) as fh:
        registry = json.load(fh)["adapters"]
    ok, reasons = ppr.can_launch_next_parent(ppr.build_gate_state(out, ev, registry, matrix))
    assert ok is False and "stale adapters" in reasons[0]


def test_f0_then_f1_accumulate_in_cross_parent_matrix(tmp_path):
    out = str(tmp_path / "review")
    f0 = ppr.load_evidence(os.path.join(_HERE, "evidence", "f0_gpt_oss_120b.json"))
    f1 = ppr.load_evidence(os.path.join(_HERE, "evidence", "f1_qwen3_235b.json"))
    ppr.generate(f0, out, adapters())
    written = ppr.generate(f1, out, adapters())
    cross = json.load(open(written["CROSS_PARENT_TRANSFER_MATRIX.json"]))
    assert set(cross["parents"]) == {"gpt-oss-120b", "qwen3-235b-a22b-instruct-2507"}
    assert all(p["dominant_failure_organ"] == "gate" for p in cross["parents"].values())
    assert cross["parents"]["gpt-oss-120b"]["provisional"] is False
