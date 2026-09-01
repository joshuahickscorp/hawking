from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import repatriation_effects as effects


def test_effects_bind_every_atlas_behavior_and_physical_candidate_scope():
    repo = Path(__file__).resolve().parents[2]
    document = effects.build_effects(repo_root=repo)

    assert document["schema"] == effects.SCHEMA
    assert effects.validate_effects(document)["passed"] is True
    atlas = json.loads(
        (repo / "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json").read_text()
    )
    queue = json.loads(
        (repo / "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json").read_text()
    )
    assert {row["behavior_id"] for row in document["entries"]} == {
        row["behavior_id"] for row in atlas["entries"]
    }
    assert {row["candidate_id"] for row in document["candidate_scope_bindings"]} == {
        row["candidate_id"] for row in queue["candidates"]
    }
    assert all(
        row["outcome"] in effects.CANDIDATE_OUTCOMES
        for row in document["candidate_scope_bindings"]
    )
    outcomes = {row["outcome"] for row in document["candidate_scope_bindings"]}
    assert "IMPLEMENTED_UNMEASURED" in outcomes
    assert "GENERIC_CANDIDATE" in outcomes
    assert document["transfer_policy"]["current_physical_law_count"] == 0


def test_effects_reject_generic_law_or_fingerprint_tampering():
    repo = Path(__file__).resolve().parents[2]
    document = effects.build_effects(repo_root=repo)

    tampered = copy.deepcopy(document)
    tampered["entries"][0]["scope"]["genericity"] = "VERIFIED"
    with pytest.raises(ValueError, match="no generic law"):
        effects.validate_effects(tampered)

    tampered = copy.deepcopy(document)
    tampered["entries"][0]["source_behavior"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        effects.validate_effects(tampered)
