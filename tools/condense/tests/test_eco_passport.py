#!/usr/bin/env python3.12
"""Tests for the Hawking Passport identity/receipt graph (eco_passport)."""
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_passport as pp  # noqa: E402
from eco_common import EcoError  # noqa: E402


def _facets():
    return {
        "artifact": {"family": "qwen2.5-dense", "label": "14B"},
        "doctor_treatment": {"branch": "doctor_static", "treatment_bytes": 40_000_000},
        "physical_bytes": {
            "all_in_model_payload_bpw": 2.34,
            "all_in_model_payload_bytes": 4_200_000_000,
            "byte_breakdown": {"packed_2d_tensor_bytes": 4_150_000_000,
                               "doctor_correction_bytes": 40_000_000},
        },
        "capability_contract": {"ppl_rel_delta_max": 0.08, "capability_abs_delta_min": -0.05},
        "context_horizon": {"nominal": 32768, "layer": "context_system"},
        "session_state": {"continuum": "event_sourced", "layer": "agent_system"},
        "device_profile": {"name": "Studio-M3Ultra-96", "weight_budget_gb": 78.0},
        "client_compat": {"openai_chat": True, "mcp": True},
    }


def test_selftest_green():
    assert pp.selftest()["ok"] is True


def test_mint_and_verify_roundtrip():
    passport = pp.mint_passport(_facets(), parent_label="14B", rate_id="2", branch="doctor_static")
    ok, why = pp.verify_passport(passport)
    assert ok, why
    assert passport["passport_sha256"]
    # every dimension present + hashed
    for dim in pp.DIMENSIONS:
        assert dim in passport["facets"]
        assert passport["facets"][dim]["facet_sha256"]


def test_missing_dimension_refused():
    facets = _facets()
    del facets["session_state"]
    with pytest.raises(EcoError, match="missing dimensions"):
        pp.mint_passport(facets, parent_label="14B", rate_id="2", branch="x")


def test_physical_bytes_rejects_runtime_role():
    facets = _facets()
    facets["physical_bytes"]["byte_breakdown"]["kv_cache"] = 1_000_000
    with pytest.raises(EcoError, match="runtime role"):
        pp.mint_passport(facets, parent_label="14B", rate_id="2", branch="x")


def test_tamper_detected_on_verify():
    passport = pp.mint_passport(_facets(), parent_label="14B", rate_id="2", branch="x")
    passport["facets"]["physical_bytes"]["value"]["all_in_model_payload_bpw"] = 0.5
    ok, why = pp.verify_passport(passport)
    assert not ok
    assert any("content hash mismatch" in r or "self-seal" in r for r in why)


def test_identity_edge_content_addressed():
    parent = pp.mint_passport(_facets(), parent_label="14B", rate_id="2", branch="x")
    e1 = pp.identity_edge(parent, delta_kind="fork", delta={"tokens": 10},
                          model_identity="qwen2.5-14b", position_policy="yarn", kv_state_codec="int4")
    e2 = pp.identity_edge(parent, delta_kind="fork", delta={"tokens": 10},
                          model_identity="qwen2.5-14b", position_policy="yarn", kv_state_codec="int4")
    e3 = pp.identity_edge(parent, delta_kind="fork", delta={"tokens": 11},
                          model_identity="qwen2.5-14b", position_policy="yarn", kv_state_codec="int4")
    # same inputs -> same child identity; different delta -> different identity
    assert e1["child_identity_sha256"] == e2["child_identity_sha256"]
    assert e1["child_identity_sha256"] != e3["child_identity_sha256"]


def test_identity_edge_rejects_bad_parent():
    parent = pp.mint_passport(_facets(), parent_label="14B", rate_id="2", branch="x")
    parent["passport_sha256"] = "0" * 64  # break the seal
    with pytest.raises(EcoError, match="parent passport invalid"):
        pp.identity_edge(parent, delta_kind="fork", delta={}, model_identity="m",
                         position_policy="native", kv_state_codec="native")
