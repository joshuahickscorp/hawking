"""Shared ArtifactIdentity + ExperimentReceipt contract.

Acceptance:
- >=3 named producers emit a valid ExperimentReceipt
- a real historical receipt still loads through the science_corpus adapter
- identity is stable under a pure path change and different under a content change
"""
from __future__ import annotations

import json

import pytest

from tools.future import experiment_receipt as er
from tools.future import science_corpus as sc


HISTORICAL_RECEIPT = "receipts/future/BA_DELTA_AB.json"


def _ident(**kwargs):
    content = kwargs.pop("content", {"payload": "same"})
    return er.artifact_identity(
        kind="experiment_receipt",
        producer="tools/future/test_experiment_receipt.py",
        content=content,
        machine="test-machine",
        commit="test-commit",
        **kwargs,
    )


def test_artifact_identity_stable_under_path_change_different_under_content():
    """Identity is not location. Same content, different path → same key."""
    a = _ident(location="/tmp/alpha/BA_DELTA_AB.json")
    b = _ident(location="/var/elsewhere/BA_DELTA_AB.json")
    assert a["identity_key"] == b["identity_key"]
    assert a["content_sha256"] == b["content_sha256"]
    assert a["location"] != b["location"]
    # Basename-equal paths must not be the identity either.
    assert a["identity_key"] != er.sha256_text("BA_DELTA_AB.json")

    changed = _ident(
        location="/tmp/alpha/BA_DELTA_AB.json",
        content={"payload": "different"},
    )
    assert changed["identity_key"] != a["identity_key"]
    assert changed["content_sha256"] != a["content_sha256"]


def test_artifact_identity_drops_path_from_inputs():
    ident = _ident(
        inputs=[
            {
                "role": "census",
                "content_sha256": "abc",
                "path": "/tmp/MLP_BYTE_CENSUS.json",
                "filename": "MLP_BYTE_CENSUS.json",
            }
        ]
    )
    assert ident["inputs"] == [{"role": "census", "content_sha256": "abc"}]
    assert "path" not in ident["inputs"][0]
    material = er.identity_material(
        kind="k",
        producer="p",
        inputs=ident["inputs"],
        machine="m",
        commit="c",
        content_sha256=ident["content_sha256"],
        location="/tmp/moved.json",
    )
    assert "location" not in material
    assert "path" not in material


def test_experiment_receipt_is_the_roadmap_envelope():
    ident = _ident(location="memory:unit")
    env = er.experiment_receipt(
        claim="a claim",
        verdict="ACCEPT",
        evidence_tier="STATIC",
        falsifier="the negative control fires and the claim still stands",
        identity=ident,
        scope="unit",
        facts=[{"claim": "fact"}],
        negative_controls=[{"id": "null-is-not-zero"}],
        failures=[],
        resource_usage={"gpu_authority": False},
        uncertainty=["fixture"],
    )
    for field in er.ROADMAP_ENVELOPE_FIELDS:
        assert field in env, field
    assert env["schema"] == er.RECEIPT_SCHEMA
    assert env["version"] == er.RECEIPT_VERSION
    assert env["verified_facts"] == env["facts"]
    assert env["resource_use"] == env["resource_usage"]
    er.validate_experiment_receipt(env)


def test_three_named_producers_emit_valid_experiment_receipt():
    emitted = er.emit_named_producers()
    assert len(emitted) >= 3
    for name in er.NAMED_PRODUCERS:
        assert name in emitted, name
        env = emitted[name]
        er.validate_experiment_receipt(env)
        assert env["identity"]["producer"] == name
        assert env["identity"]["identity_key"]
        assert env["identity"]["content_sha256"]
        assert env["falsifier"]
        assert env["evidence_tier"] in er.EVIDENCE_TIERS
        assert env["negative_controls"] is not None


def test_historical_ba_delta_ab_receipt_still_loads_through_adapter():
    """receipts/future/BA_DELTA_AB.json is the v1 shape. Do not rewrite it."""
    doc, meta = sc._read_json(HISTORICAL_RECEIPT)
    assert doc is not None, f"historical receipt missing: {meta}"
    assert doc.get("schema") == "hawking.future.ba_delta_ab.v1"
    assert "experiment_receipt" not in doc
    recs = sc.adapt_document(doc, source_receipt=HISTORICAL_RECEIPT)
    assert recs, "adapter emitted nothing for the historical receipt"
    kinds = {r["kind"] for r in recs}
    assert "experiment" in kinds
    experiment = next(r for r in recs if r["kind"] == "experiment")
    assert experiment["schema_family"] == "hawking.future.ba_delta_ab"
    assert experiment["key_fields"]["id"] == "ba_delta_ab"
    assert experiment["key_fields"]["lever"]
    restored = sc.round_trip(experiment)
    assert sc.key_fields_preserved(experiment, restored)


def test_new_producer_envelope_also_projects_without_dropping_old_fields():
    from tools.future import ba_delta_ab

    doc = ba_delta_ab.build()
    assert doc["schema"] == "hawking.future.ba_delta_ab.v1"
    assert doc["exact"]["dispatches_removed"] == 48.0
    env = er.extract_receipt(doc)
    recs = sc.adapt_document(doc, source_receipt="memory:ba_delta_ab.build")
    families = {r["schema_family"] for r in recs}
    assert "hawking.future.ba_delta_ab" in families
    assert "hawking.experiment.receipt" in families
    assert env["identity"]["identity_key"]
    # Moving the location on a rebuilt identity of the same producer payload
    # does not change the key.
    moved = er.artifact_identity(
        kind=env["identity"]["kind"],
        producer=env["identity"]["producer"],
        content={k: v for k, v in doc.items() if k not in ("artifact_identity", "experiment_receipt")},
        inputs=env["identity"]["inputs"],
        machine=env["identity"]["machine"],
        commit=env["identity"]["commit"],
        location="/tmp/moved/BA_DELTA_AB.json",
    )
    assert moved["identity_key"] == env["identity"]["identity_key"]


def test_older_scar_schema_still_loads_through_the_same_adapter_pattern():
    old = {
        "schema": "hawking.future.campaign_scars.v0",
        "evidence_class": "STATIC_ONLY",
        "scars": [
            {
                "scar_id": "OLD-SCAR-1",
                "what_was_wrong": "divided by the wrong denominator",
                "verdict": "FALSIFIED",
                "reopen_if": "numerator matches denominator events",
                "family": "DENOMINATOR",
            }
        ],
    }
    recs = sc.adapt_document(old, source_receipt="memory:v0-scars")
    assert len(recs) == 1
    assert recs[0]["schema_family"] == "hawking.future.campaign_scars"
    assert recs[0]["key_fields"]["id"] == "OLD-SCAR-1"


def test_unknown_verdict_is_refused():
    ident = _ident()
    with pytest.raises(er.ContractError, match="verdict"):
        er.experiment_receipt(
            claim="x",
            verdict="WIN",
            evidence_tier="STATIC",
            falsifier="y",
            identity=ident,
        )
