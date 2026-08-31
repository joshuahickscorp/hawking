import json

from tools.future import evidence_snapshot as es
from tools.future._common import REPO, sha256_file


def test_build_captures_and_hashes():
    out = es.build()
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.evidence_snapshot.v1"
    assert doc["counts"]["captured"] > 0
    for row in doc["captured"]:
        p = REPO / row["snapshot_path"]
        assert p.exists()
        assert sha256_file(p) == row["sha256"]
        # The Codex handoff is sealed at the REPO ROOT, not under a subtree.
        # Pin the property that matters -- the source is inside the repo and is
        # a real relative path -- not the directories that happened to exist.
        assert not row["source_path"].startswith(("/", "../"))
        assert (REPO / row["source_path"]).exists()


def test_verify_is_green():
    es.build()
    assert es.verify() == 0


def test_verify_detects_a_tampered_snapshot():
    # Negative control: the manifest must actually be load-bearing.
    es.build()
    man = json.loads((REPO / "receipts/future/EVIDENCE_SNAPSHOT.json").read_text())
    victim = REPO / man["captured"][0]["snapshot_path"]
    original = victim.read_bytes()
    try:
        victim.write_bytes(original + b"\n// tampered\n")
        assert es.verify() == 1
    finally:
        victim.write_bytes(original)
    assert es.verify() == 0


def test_missing_wanted_files_are_reported_not_hidden():
    doc = json.loads((REPO / "receipts/future/EVIDENCE_SNAPSHOT.json").read_text())
    # Absence is a finding about the project's evidence, so it must be surfaced.
    assert "missing" in doc and isinstance(doc["missing"], list)
    assert doc["counts"]["wanted"] == doc["counts"]["captured"] + doc["counts"]["missing"]


def test_originals_are_untouched():
    doc = json.loads((REPO / "receipts/future/EVIDENCE_SNAPSHOT.json").read_text())
    for row in doc["captured"]:
        src = REPO / row["source_path"]
        assert src.exists(), f"snapshot must not move or delete {row['source_path']}"
        assert sha256_file(src) == row["sha256"]
