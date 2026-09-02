"""Acceptance tests for the FLASH_* gates.

These tests do not weaken a criterion. They check that each assigned gate
has a receipt of a real symbol call, that numeric bars are compared
numerically, and that a failing bar is recorded BLOCKED.
"""
from __future__ import annotations

import json

import pytest

from tools.acceptance.flash.run_gates import (
    ASSIGNED,
    EBPW_THRESHOLD,
    FLASH_SPECIMEN,
    REPO,
    REQUIRED_INDEXED_PAYLOAD_BYTES,
    REQUIRED_SHARDS,
    REQUIRED_TENSORS,
    TPS_THRESHOLD,
    called_names_in_source,
    call_mix_report,
    symbol_call_names,
)

ACCEPT = REPO / "receipts" / "acceptance"


def _load(gate: str) -> dict:
    path = ACCEPT / f"{gate}.json"
    assert path.is_file(), f"missing acceptance receipt {path}"
    doc = json.loads(path.read_text())
    assert doc.get("schema") == "hawking.acceptance.gate.v1"
    assert doc.get("gate") == gate
    return doc


def test_every_assigned_gate_has_receipt():
    for gate in ASSIGNED:
        doc = _load(gate)
        assert doc["verdict"] in {"ACCEPTED", "BLOCKED"}
        assert doc.get("criterion_altered") is False
        assert doc.get("command"), f"{gate} has no command"
        assert doc.get("symbol", {}).get("kind") == "call"
        assert doc.get("evidence_tier") in {
            "STATIC",
            "FUNCTIONAL_SIM",
            "COST_MODEL",
            "CYCLE_APPROX",
            "HARDWARE_MEASURED",
        }
        assert "stdout" in doc
        assert doc.get("criterion", {}).get("quoted")


def test_runner_calls_each_gate_symbol_not_just_imports():
    names = called_names_in_source()
    required = symbol_call_names()
    for gate, want in required.items():
        missing = [item for item in want if item not in names]
        assert not missing, (
            f"{gate}: required Call nodes {want} missing {missing}; "
            f"found {sorted(names)}. An import is not a call."
        )


def test_ebpw_blocked_at_measured_3_139_against_threshold_1():
    """Re-call mix_report and compare numerically. Do not accept 3.139 as <= 1."""
    ran = call_mix_report()
    measured = float(ran["build"]["incumbent"]["complete_ebpw"])
    assert measured == pytest.approx(3.139300850311054)
    assert measured > EBPW_THRESHOLD
    assert not (measured <= EBPW_THRESHOLD)
    doc = _load("FLASH_COMPLETE_EBPW_LE_1")
    cmp_ = doc["comparison"]
    assert cmp_["op"] == "<="
    assert float(cmp_["threshold"]) == EBPW_THRESHOLD
    assert float(cmp_["measured"]) == pytest.approx(measured)
    assert cmp_["satisfied"] is False
    assert doc["verdict"] == "BLOCKED"
    assert "3.139" in (doc.get("blocker") or "")
    # Load-bearing: a receipt that claimed ACCEPTED while measured > 1 is a lie.
    assert not (doc["verdict"] == "ACCEPTED" and measured > EBPW_THRESHOLD)


def test_tps_blocked_because_accepted_tps_is_not_at_least_50():
    doc = _load("FLASH_ACCEPTED_TPS_GE_50")
    cmp_ = doc["comparison"]
    assert cmp_["op"] == ">="
    assert float(cmp_["threshold"]) == TPS_THRESHOLD
    measured = cmp_["measured"]
    assert measured is None or float(measured) < TPS_THRESHOLD
    assert cmp_["satisfied"] is False
    assert doc["verdict"] == "BLOCKED"
    assert doc["measured"]["accepted_tps"] is None
    assert "50" in (doc.get("blocker") or "")


def test_source_verified_census_matches_roadmap_numbers():
    doc = _load("FLASH_SOURCE_VERIFIED")
    measured = doc["measured"]
    assert int(measured["shard_count"]) == REQUIRED_SHARDS
    assert int(measured["tensor_count"]) == REQUIRED_TENSORS
    assert int(measured["source_parameter_bytes_indexed"]) == REQUIRED_INDEXED_PAYLOAD_BYTES
    assert doc["comparison"]["satisfied"] is True
    assert doc["verdict"] == "ACCEPTED"
    census = json.loads((ACCEPT / "FLASH_SOURCE_VERIFIED.census.json").read_text())
    assert int(census["shard_count"]) == REQUIRED_SHARDS
    assert int(census["tensor_count"]) == REQUIRED_TENSORS
    assert int(census["source_parameter_bytes_indexed"]) == REQUIRED_INDEXED_PAYLOAD_BYTES


def test_live_specimen_index_still_matches_the_same_numbers():
    index_path = FLASH_SPECIMEN / "model.safetensors.index.json"
    assert index_path.is_file(), f"sealed Flash specimen missing at {FLASH_SPECIMEN}"
    idx = json.loads(index_path.read_text())
    weight_map = idx["weight_map"]
    assert len(weight_map) == REQUIRED_TENSORS
    assert len(set(weight_map.values())) == REQUIRED_SHARDS
    assert int(idx["metadata"]["total_size"]) == REQUIRED_INDEXED_PAYLOAD_BYTES


def test_first_gravity_organ_has_doctor_healthy_frontier():
    doc = _load("FLASH_FIRST_GRAVITY_ORGAN")
    assert doc["verdict"] == "ACCEPTED"
    frontier = doc["measured"]["healthy_frontier"]
    assert len(frontier) >= 1
    organs = {row["organ"] for row in frontier}
    assert organs
    cycle = json.loads((ACCEPT / "FLASH_FIRST_GRAVITY_ORGAN.cycle.json").read_text())
    assert (cycle.get("doctor_seal") or {}).get("negative_control") == "zero_control_known_bad"
    healthy = [
        row
        for row in (cycle.get("pareto_frontier") or [])
        if (row.get("doctor") or {}).get("healthy") is True
    ]
    assert healthy


def test_dense_vs_nf_ab_ran_source_against_q4_and_nf4():
    doc = _load("FLASH_DENSE_VS_NF_AB")
    assert doc["verdict"] == "ACCEPTED"
    assert doc["evidence_tier"] == "FUNCTIONAL_SIM"
    ids = {c["id"] for c in doc["measured"]["candidates"]}
    assert "independent_q4_g64" in ids
    assert "independent_nf4_g64" in ids
    assert doc["measured"]["n_candidates"] >= 2
    assert doc["measured"]["native_kernel_execution_observed"] is False


def test_native_nf_kernel_blocked_on_metal_error():
    doc = _load("FLASH_NATIVE_NF_KERNEL")
    assert doc["verdict"] == "BLOCKED"
    assert doc["evidence_tier"] == "HARDWARE_MEASURED"
    assert doc["comparison"]["satisfied"] is False
    err = doc["measured"]["kernel_error"]
    assert err == "metal: no Metal-capable GPU"
    kernel = json.loads((ACCEPT / "FLASH_NATIVE_NF_KERNEL.kernel.json").read_text())
    assert kernel["status"] == "FAILED"
    assert kernel["error"]["message"] == "metal: no Metal-capable GPU"


def test_full_noetic_executable_blocked_as_scaffold_or_metadata_only():
    doc = _load("FLASH_FULL_NOETIC_EXECUTABLE")
    assert doc["verdict"] == "BLOCKED"
    assert doc["comparison"]["satisfied"] is False
    assert doc["measured"]["nx_status"] == "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"
    assert doc["measured"]["native_loader"] == "NOT_IMPLEMENTED"
    assert doc["measured"]["native_kernels"] == "PLAN_ONLY"
    assert doc["measured"]["nr_can_promote"] is False


def test_no_criterion_was_weakened():
    ebpw = _load("FLASH_COMPLETE_EBPW_LE_1")
    assert float(ebpw["comparison"]["threshold"]) == 1.0
    assert ebpw["comparison"]["op"] == "<="
    tps = _load("FLASH_ACCEPTED_TPS_GE_50")
    assert float(tps["comparison"]["threshold"]) == 50.0
    assert tps["comparison"]["op"] == ">="
    summary = json.loads((ACCEPT / "FLASH_ACCEPTANCE_SUMMARY.json").read_text())
    assert summary["criterion_altered"] is False
    assert summary["n_assigned"] == 7
    assert summary["n_accepted"] + summary["n_blocked"] == 7


def test_summary_counts_match_per_gate_verdicts():
    summary = json.loads((ACCEPT / "FLASH_ACCEPTANCE_SUMMARY.json").read_text())
    accepted = []
    blocked = []
    for gate in ASSIGNED:
        doc = _load(gate)
        if doc["verdict"] == "ACCEPTED":
            accepted.append(gate)
        else:
            blocked.append(gate)
    assert set(summary["accepted"]) == set(accepted)
    assert set(summary["blocked"]) == set(blocked)
    assert summary["n_accepted"] == len(accepted)
    assert summary["n_blocked"] == len(blocked)
