from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import repatriation_audit as audit


def test_canonical_repatriation_audit_passes_all_structural_checks():
    repo = Path(__file__).resolve().parents[2]
    document = audit.build_audit(repo_root=repo)

    assert document["schema"] == audit.SCHEMA
    assert document["passed"] is True
    assert len(document["checks"]) == 14
    assert all(row["passed"] is True for row in document["checks"])
    assert "physical performance" in document["claim_boundary"]


def test_audit_emission_round_trips_and_detects_tampering(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    path = audit.emit_audit(repo_root=repo, output=tmp_path / "audit.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    result = audit.validate_audit(loaded)
    assert result["passed"] is True
    assert loaded["fingerprint"] == audit.build_audit(repo_root=repo)["fingerprint"]

    tampered = copy.deepcopy(loaded)
    tampered["checks"][0]["observed"] = {"tampered": True}
    with pytest.raises(ValueError, match="fingerprint"):
        audit.validate_audit(tampered)
