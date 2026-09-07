"""G030: the contract must refuse claims, not collect them."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
from nx_promotion import evaluate, CONDITIONS, MECHANICAL, check_accounting, check_no_hidden_parent


def _artifact(tmp_path, payload=b"x" * 1024):
    p = tmp_path / "candidate.nx"
    p.write_bytes(payload)
    return p


def _full_manifest(p):
    return dict({c: {"ok": True, "receipt": f"receipts/{c}.json"} for c in CONDITIONS},
                counted_bytes=p.stat().st_size, external_references=[],
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_a_complete_and_honest_manifest_promotes(tmp_path):
    p = _artifact(tmp_path)
    r = evaluate(_full_manifest(p), p)
    assert r["promotable"] is True, r["missing"]
    assert r["verdict"] == "NX"


def test_a_bare_boolean_is_refused(tmp_path):
    p = _artifact(tmp_path)
    m = _full_manifest(p)
    m["capability_qualification"] = True          # claim with no receipt
    r = evaluate(m, p)
    assert not r["promotable"]
    assert "capability_qualification" in r["missing"]
    assert r["conditions"]["capability_qualification"]["checked"] == "refused"


def test_uncounted_bytes_block_promotion(tmp_path):
    p = _artifact(tmp_path)
    m = _full_manifest(p); m["counted_bytes"] = p.stat().st_size - 64
    r = evaluate(m, p)
    assert not r["promotable"] and "complete_persistent_accounting" in r["missing"]
    assert "uncounted" in r["conditions"]["complete_persistent_accounting"]["why"]


def test_a_hidden_parent_blocks_promotion(tmp_path):
    p = _artifact(tmp_path)
    m = _full_manifest(p); m["external_references"] = ["/models/parent.safetensors"]
    r = evaluate(m, p)
    assert not r["promotable"] and "no_hidden_dense_parent" in r["missing"]


def test_a_missing_external_references_field_is_not_treated_as_none(tmp_path):
    """Absence of the field is not absence of a parent."""
    p = _artifact(tmp_path)
    m = _full_manifest(p); m.pop("external_references")
    assert not evaluate(m, p)["promotable"]


def test_a_wrong_hash_blocks_promotion(tmp_path):
    p = _artifact(tmp_path)
    m = _full_manifest(p); m["sha256"] = "0" * 64
    r = evaluate(m, p)
    assert not r["promotable"] and "immutable_artifact_identity" in r["missing"]


def test_all_ten_conditions_and_three_mechanical():
    assert len(CONDITIONS) == 10 and len(MECHANICAL) == 3
    assert set(MECHANICAL) <= set(CONDITIONS)


def test_nothing_promotes_on_an_empty_manifest(tmp_path):
    r = evaluate({}, _artifact(tmp_path))
    assert not r["promotable"] and len(r["missing"]) == 10
