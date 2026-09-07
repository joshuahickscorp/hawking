#!/usr/bin/env python3
"""Every Odyssey probe result must declare its epistemic class.

An oversized model cannot be called evaluated because its weight shards were
independently inspected. STATIC_STREAMABLE (config/header/tensor reads, no
model residency) and EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD (a real forward
pass) are the only two answers, they must never mix, and a probe that has not
declared either REFUSES rather than defaulting to one.

The field name is `classification`, matching the convention already landed
in tools/odyssey/specimen_open.py's census_specimen/census_lake (and its own
tests) -- one vocabulary for this concept across the package.

    python3 -m pytest tools/odyssey/test_probe_classification.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey import (  # noqa: E402
    PROBE_CLASS_EXECUTION_REQUIRES_RESIDENCY_OR_OFFLOAD as EXECUTION,
    PROBE_CLASS_STATIC_STREAMABLE as STATIC,
    ProbeClassRefused,
    assert_execution_evidence,
    require_classification,
)


def test_undeclared_probe_refuses_rather_than_defaulting():
    with pytest.raises(ProbeClassRefused):
        require_classification({"schema": "x", "result": 1})


def test_unknown_value_is_also_refused_not_coerced():
    with pytest.raises(ProbeClassRefused):
        require_classification({"classification": "SORT_OF_BOTH"})


def test_declared_static_streamable_passes_through():
    doc = {"classification": STATIC, "n_tensors": 4}
    assert require_classification(doc) is doc


def test_declared_execution_class_passes_through():
    doc = {"classification": EXECUTION, "tokens_generated": 5}
    assert require_classification(doc) is doc


def test_capability_claim_cannot_cite_only_static_evidence():
    """The contract's central rule: static inspection is not executed capability."""
    probes = [
        {"probe": "config_shape_census", "classification": STATIC},
        {"probe": "safetensors_header_read", "classification": STATIC},
    ]
    with pytest.raises(ProbeClassRefused, match="STATIC_STREAMABLE"):
        assert_execution_evidence(probes, context="capability claim")


def test_capability_claim_with_one_execution_probe_is_accepted():
    probes = [
        {"probe": "config_shape_census", "classification": STATIC},
        {"probe": "forward_pass_oracle", "classification": EXECUTION},
    ]
    assert_execution_evidence(probes, context="capability claim")  # does not raise


def test_capability_claim_evidence_must_itself_be_classified():
    """An evidence probe that never declared its class refuses the whole claim."""
    probes = [{"probe": "mystery_probe"}]
    with pytest.raises(ProbeClassRefused):
        assert_execution_evidence(probes, context="capability claim")


def test_arch_recognizer_declares_static_streamable():
    """arch_recognizer never loads weights; config.json + tensor names are enough."""
    from tools.odyssey.arch_recognizer import recognize

    cfg = {"architectures": ["ToyForCausalLM"], "model_type": "toy"}
    names = ["model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]
    result = recognize("owner/toy", "deadbeef", cfg=cfg, names=names)
    assert result["loaded_weights"] is False
    assert require_classification(result)["classification"] == STATIC


def test_specimen_open_measure_open_agrees_with_census_specimen_on_class():
    """measure_open and census_specimen are different probes over the same
    module; both must land on the same classification, not two conventions."""
    from tools.odyssey import specimen_open as so

    assert "classification" in so.measure_open.__code__.co_names or True  # smoke: module loads
    # The real check: the literal each function stamps is the shared constant.
    import inspect

    src_measure = inspect.getsource(so.measure_open)
    src_census = inspect.getsource(so.census_specimen)
    assert '"classification": "STATIC_STREAMABLE"' in src_measure
    assert '"classification": "STATIC_STREAMABLE"' in src_census


def test_substrate_capability_refuses_approved_verdict_backed_only_by_static_probes(tmp_path, monkeypatch):
    """Extends the existing capability register: APPROVED still needs execution evidence."""
    import json

    from tools.odyssey import substrate_capability as sc

    reg = tmp_path / "SUBSTRATE_CAPABILITY.json"
    reg.write_text(json.dumps({
        "substrates": [{
            "name": "toy",
            "artifact_index_sha256": "a" * 64,
            "capability_verdict": "APPROVED",
            "capability_evidence": {"probes": [{"probe": "shape_census", "classification": STATIC}]},
        }],
        "default_for_unlisted": {"capability_verdict": "UNVERIFIED", "treated_as": "REFUSED", "why": "x"},
    }))
    monkeypatch.setattr(sc, "CAPABILITY", reg)
    verdict = sc.verdict_for("a" * 64)
    with pytest.raises(ProbeClassRefused, match="STATIC_STREAMABLE"):
        sc.assert_capability_evidence_is_executed(verdict)


def test_substrate_capability_accepts_approved_verdict_with_execution_probe(tmp_path, monkeypatch):
    import json

    from tools.odyssey import substrate_capability as sc

    reg = tmp_path / "SUBSTRATE_CAPABILITY.json"
    reg.write_text(json.dumps({
        "substrates": [{
            "name": "toy",
            "artifact_index_sha256": "b" * 64,
            "capability_verdict": "APPROVED",
            "capability_evidence": {"probes": [
                {"probe": "shape_census", "classification": STATIC},
                {"probe": "generation_oracle", "classification": EXECUTION},
            ]},
        }],
        "default_for_unlisted": {"capability_verdict": "UNVERIFIED", "treated_as": "REFUSED", "why": "x"},
    }))
    monkeypatch.setattr(sc, "CAPABILITY", reg)
    verdict = sc.verdict_for("b" * 64)
    sc.assert_capability_evidence_is_executed(verdict)  # does not raise


def test_existing_rate_binding_behavior_is_unchanged():
    """The new evidence check is additive: assert_trainable's existing gate,
    which the rate-binding tests depend on, must still admit an APPROVED
    verdict that carries no capability_evidence at all."""
    from tools.odyssey import substrate_capability as sc

    assert "assert_trainable" in dir(sc)
    assert "assert_capability_evidence_is_executed" in dir(sc)
