#!/usr/bin/env python3.12
"""Tests for the Press->Summon data-driven state machine (eco_pipeline)."""
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_pipeline as pipe  # noqa: E402
from eco_common import EcoError  # noqa: E402


def test_selftest_green():
    assert pipe.selftest()["ok"] is True


def test_canonical_order_matches_directive():
    assert list(pipe.CANONICAL_ORDER) == [
        "press", "doctor", "horizon", "context", "continuum", "lens",
        "bridge", "passport", "capsule", "summon"]


def test_spec_is_valid_topo_order():
    ok, why = pipe.validate_spec()
    assert ok, why


def test_every_passport_dimension_produced():
    import eco_passport
    produced = {s["passport_dimension"] for s in pipe.STAGES if s["passport_dimension"]}
    assert set(eco_passport.DIMENSIONS) <= produced


def test_spec_sha256_is_deterministic_content_address():
    # regression: no timestamp inside the seal, so the content address is stable and matches
    a = pipe.pipeline_spec()["spec_sha256"]
    b = pipe.pipeline_spec()["spec_sha256"]
    assert a == b
    assert pipe.new_state()["spec_sha256"] == a


def test_advance_blocks_on_unmet_requires():
    state = pipe.new_state()
    with pytest.raises(EcoError, match="unmet requires"):
        pipe.advance(state, "doctor", {"x": 1})


def test_runnable_progression():
    state = pipe.new_state()
    assert pipe.runnable(state) == ["press"]
    state = pipe.advance(state, "press", {"stage": "press"})
    assert pipe.runnable(state) == ["doctor"]
    state = pipe.advance(state, "doctor", {"stage": "doctor"})
    state = pipe.advance(state, "horizon", {"stage": "horizon"})
    state = pipe.advance(state, "context", {"stage": "context"})
    assert set(pipe.runnable(state)) >= {"continuum", "lens", "bridge"}


def test_rollback_reverts_only_dependents():
    state = pipe.new_state()
    for s in ("press", "doctor", "horizon", "context"):
        state = pipe.advance(state, s, {"stage": s})
    rolled = pipe.rollback(state, "horizon")
    assert rolled["stages"]["horizon"]["status"] == "pending"
    assert rolled["stages"]["context"]["status"] == "pending"  # dependent
    assert rolled["stages"]["press"]["status"] == "complete"   # not a dependent
    assert rolled["stages"]["doctor"]["status"] == "complete"


def test_offline_hydrate_stops_at_gap():
    # present press + horizon but NOT doctor -> hydration stops before horizon
    outputs = {"press": {"stage": "press"}, "horizon": {"stage": "horizon"}}
    hydrated = pipe.offline_hydrate(outputs)
    assert hydrated["stages"]["press"]["status"] == "complete"
    assert hydrated["stages"]["horizon"]["status"] == "pending"


def test_passport_validator_enforced():
    # advancing the passport stage requires all eight dimensions + self-seal
    import eco_passport
    state = pipe.new_state()
    for s in ("press", "doctor", "horizon", "context", "continuum", "bridge"):
        state = pipe.advance(state, s, {"stage": s})
    with pytest.raises(EcoError, match="all_eight_dimensions|self_seal"):
        pipe.advance(state, "passport", {"facets": {}})
    facets = {
        "artifact": {"a": 1}, "doctor_treatment": {"a": 1},
        "physical_bytes": {"all_in_model_payload_bpw": 2.0,
                           "all_in_model_payload_bytes": 10, "byte_breakdown": {"base": 10}},
        "capability_contract": {"a": 1}, "context_horizon": {"a": 1},
        "session_state": {"a": 1}, "device_profile": {"a": 1}, "client_compat": {"a": 1},
    }
    real = eco_passport.mint_passport(facets, parent_label="x", rate_id="2", branch="b")
    state = pipe.advance(state, "passport", real)
    assert state["stages"]["passport"]["status"] == "complete"
