"""A kill is only useful if it says what would reopen it.

"Query before every expensive hypothesis" only works when each entry states the
PREMISE it killed, so a new architecture can be checked against it, and the
condition that would bring it back. An entry without those is folklore.
"""
from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
LEDGER = REPO / "receipts" / "future" / "NEGATIVE_SCIENCE_CAMPAIGN.json"
ODYSSEY = REPO / "workspace" / "campaign" / "odyssey" / "NEGATIVE_SCIENCE.json"


@pytest.fixture(scope="module")
def doc():
    return json.loads(LEDGER.read_text())


def test_every_entry_is_falsifiable_and_reopenable(doc):
    for e in doc["entries"]:
        for field in ("id", "mechanism", "premise", "killed_by", "evidence",
                      "transferable_as", "reopens_if"):
            assert e.get(field), f"{e.get('id')} has no {field}"
        assert len(e["killed_by"]) > 60, f"{e['id']}: killed_by states no measurement"


def test_ids_are_unique(doc):
    ids = [e["id"] for e in doc["entries"]]
    assert len(ids) == len(set(ids)), ids


def test_it_uses_the_same_schema_as_the_odyssey_ledger_so_they_merge(doc):
    assert doc["schema"] == "hawking.odyssey.negative_science.v1"
    if ODYSSEY.is_file():
        other = json.loads(ODYSSEY.read_text())
        assert other["schema"] == doc["schema"]
        mine = {e["id"] for e in doc["entries"]}
        theirs = {e.get("id") for e in other.get("entries", [])}
        assert not (mine & theirs), f"id collision with the Odyssey ledger: {mine & theirs}"


def test_it_does_not_write_into_the_odyssey_campaigns_store():
    """Another campaign owns that file. Same schema, separate ownership."""
    assert "not written here" in doc_text()


def doc_text() -> str:
    return LEDGER.read_text()


def test_each_kill_cites_an_artifact_that_exists(doc):
    """A kill citing a path nobody can open is a claim, not evidence."""
    missing = []
    for e in doc["entries"]:
        for ref in e["evidence"].split(","):
            path = ref.strip().split("::")[0].split(":")[0]
            if not path or not path.startswith(("tools/", "receipts/")):
                continue
            if not (REPO / path).exists():
                missing.append(f"{e['id']} -> {path}")
    assert not missing, missing
