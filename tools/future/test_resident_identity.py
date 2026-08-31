"""Tests for restart-survivable resident identity.

Negative controls that must actually fire:
- filling a field this host cannot evidence is REJECTED
- a full reload in a fresh process reconstructs identity from disk alone
- claiming zero known weaknesses while real blockers exist is refused
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tools.future import resident_identity as ri
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, REPO, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = ri.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_IDENTITY.json"
    assert doc["schema"] == "hawking.future.resident_identity.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["claim_class"] == "STATIC_ONLY"
    assert doc["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    assert doc["identity_validation"]["status"] == "ACCEPTED"
    assert doc["negative_control"]["invented_tps"]["fires"] is True
    assert doc["negative_control"]["zero_weaknesses"]["fires"] is True
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # A numeric hardware key anywhere in the receipt is a campaign failure.
        blob = json.dumps(doc)
        assert f'"{key}": ' not in blob or not isinstance(doc.get(key), (int, float))


def test_selftest_emits_sealed_receipt_and_keeps_the_guards():
    out = ri.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["negative_control"]["invented_tps"]["fires"] is True
    assert doc["negative_control"]["zero_weaknesses"]["fires"] is True


def test_every_named_field_is_persisted():
    ident = ri.collect()
    for field in ri.IDENTITY_FIELDS:
        assert field in ident, field
    assert ident["residency_status"] == ri.RESIDENCY_STATUS
    assert ident["gpu_authority"] is False
    assert ident["claim_class"] == "STATIC_ONLY"
    ri.accept(ident)
    doc = json.loads(ri.build().read_text())
    stored = doc["identity"]
    for field in ri.IDENTITY_FIELDS:
        assert field in stored, field
    assert doc["persisted_fields"] == list(ri.IDENTITY_FIELDS)
    assert doc["n_persisted_fields"] == len(ri.IDENTITY_FIELDS)


def test_unknown_slots_are_unknown_and_name_missing_evidence():
    ident = ri.collect()
    ebpw = ident["ebpw"]
    assert ebpw["value"] == ri.UNKNOWN
    assert ebpw["missing_evidence"]
    assert all(isinstance(x, str) and x.strip() for x in ebpw["missing_evidence"])
    assert isinstance(ebpw.get("flash"), dict)
    assert ebpw["flash"]["value"] == ri.UNKNOWN
    assert ebpw["flash"]["missing_evidence"]
    assert ebpw.get("copied_declared_catalog_number") is False

    tps = ident["tps_token_ns_evidence"]
    assert tps["value"] == ri.UNKNOWN
    assert tps["missing_evidence"]
    assert tps.get("copied_sealed_runtime_numbers") is False

    active = ident["active_bytes_evidence"]["value"]
    for key in (
        "actual_read_bytes_per_token",
        "transient_bytes_per_token",
        "active_representation_bytes_per_token",
    ):
        slot = active[key]
        assert slot["value"] == ri.UNKNOWN, key
        assert slot["missing_evidence"], key


def test_no_field_is_invented_hardware_number():
    ident = ri.collect()
    hits = ri._hardware_numeric_keys(ident)
    assert hits == []
    # Qualified physical EBPW / TPS must not be a plausible number.
    assert not isinstance(ident["ebpw"]["value"], (int, float))
    assert not isinstance(ident["tps_token_ns_evidence"]["value"], (int, float))


def test_invented_tps_is_rejected_naming_the_field():
    ident = ri.collect()
    ident["tps_token_ns_evidence"] = {
        "value": 24.4086,
        "missing_evidence": [],
        "evidence": ["invented"],
        "claim_class": "STATIC_ONLY",
    }
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any("tps_token_ns_evidence" in r for r in result["reasons"])
    assert result["named_refusal"].startswith("REJECTED:")
    with pytest.raises(ri.IdentityRejectedError) as caught:
        ri.accept(ident)
    assert "tps_token_ns_evidence" in str(caught.value)


def test_invented_ebpw_is_rejected_naming_the_field():
    ident = ri.collect()
    ident["ebpw"] = {
        "value": 3.1393,
        "missing_evidence": [],
        "evidence": ["invented"],
        "claim_class": "STATIC_ONLY",
    }
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any(r.startswith("ebpw:") for r in result["reasons"])
    with pytest.raises(ri.IdentityRejectedError) as caught:
        ri.accept(ident)
    assert "ebpw" in str(caught.value)


def test_invented_actual_read_bytes_is_rejected():
    ident = ri.collect()
    ident["active_bytes_evidence"]["value"]["actual_read_bytes_per_token"] = {
        "value": 9878901136,
        "missing_evidence": [],
        "evidence": ["invented"],
    }
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any("actual_read_bytes_per_token" in r for r in result["reasons"])


def test_zero_weaknesses_while_blockers_exist_is_rejected():
    ident = ri.collect()
    blockers = ri._blockers_evident(ident)
    assert blockers, "this host has real blockers; a guard nobody watched fail is not a guard"
    weaknesses = ri._as_list(ident["known_weaknesses"])
    assert weaknesses, "honest collect() must populate known_weaknesses from real blockers"
    ident["known_weaknesses"] = {
        "value": [],
        "missing_evidence": [],
        "evidence": [],
        "claim_class": "STATIC_ONLY",
    }
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any("known_weaknesses" in r for r in result["reasons"])
    with pytest.raises(ri.IdentityRejectedError) as caught:
        ri.accept(ident)
    assert "known_weaknesses" in str(caught.value)


def test_list_form_zero_weaknesses_is_also_rejected():
    ident = ri.collect()
    ident["known_weaknesses"] = []
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any("known_weaknesses" in r for r in result["reasons"])


def test_nonfinal_status_is_required_and_singularity_is_refused():
    ident = ri.collect()
    assert ident["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"
    ident["residency_status"] = "SINGULARITY"
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert any("residency_status" in r for r in result["reasons"])
    assert any("Singularity" in r for r in result["reasons"])


def test_missing_required_field_is_rejected_by_name():
    ident = ri.collect()
    del ident["machine_genome"]
    result = ri.validate(ident)
    assert result["status"] == "REJECTED"
    assert "missing required field machine_genome" in result["reasons"]


def test_known_weaknesses_cite_real_blockers():
    ident = ri.collect()
    weaknesses = ri._as_list(ident["known_weaknesses"])
    assert weaknesses
    ids = {w["id"] for w in weaknesses if isinstance(w, dict)}
    assert "W-NO-GPU-AUTHORITY" in ids
    assert "W-FLASH-NX-SCAFFOLD-ONLY" in ids
    assert "W-NONFINAL-27B" in ids
    statements = " ".join(str(w.get("statement")) for w in weaknesses if isinstance(w, dict))
    evidence = [e for w in weaknesses if isinstance(w, dict) for e in (w.get("evidence") or [])]
    assert evidence
    # Cope with either teacher-capture presence: if the receipt loaded, the
    # weakness is present; if not, another blocker still fills the list.
    if any("TEACHER" in i for i in ids):
        assert "256" in statements or "BLOCKED_NO_METAL" in statements or "Metal" in statements


def test_tokenizer_and_executable_cope_with_either_host_state():
    ident = ri.collect()
    tok = ident["tokenizer_identity"]
    exe = ident["executable_hash"]
    tok_val = tok.get("value")
    if tok_val == ri.UNKNOWN:
        assert tok.get("missing_evidence")
    else:
        assert isinstance(tok_val, dict)
        assert tok_val.get("sha256") and tok_val["sha256"] != ri.UNKNOWN
        assert len(tok_val["sha256"]) == 64
    exe_val = exe.get("value")
    if exe_val == ri.UNKNOWN:
        assert exe.get("missing_evidence")
    else:
        assert isinstance(exe_val, dict)
        by_role = exe_val.get("by_role") or {}
        assert by_role, "hashed executable must name at least one role"
        for digest in by_role.values():
            assert isinstance(digest, str) and len(digest) == 64


def test_load_recovers_identity_from_disk():
    path = ri.build()
    ident = ri.load(path)
    for field in ri.IDENTITY_FIELDS:
        assert field in ident
    assert ident["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"
    assert ident["ebpw"]["value"] == ri.UNKNOWN
    assert ident["tps_token_ns_evidence"]["value"] == ri.UNKNOWN


def test_fresh_process_reload_reconstructs_identity_from_disk_alone():
    """No in-memory state, no conversational input: a new interpreter, load() only."""
    written = ri.build()
    assert written.is_file()
    on_disk = json.loads(written.read_text())["identity"]
    script = """
import json
from tools.future.resident_identity import IDENTITY_FIELDS, UNKNOWN, load
ident = load()
payload = {
    "n_fields": sum(1 for f in IDENTITY_FIELDS if f in ident),
    "residency_status": ident.get("residency_status"),
    "ebpw": ident["ebpw"]["value"] if isinstance(ident.get("ebpw"), dict) else ident.get("ebpw"),
    "tps": ident["tps_token_ns_evidence"]["value"] if isinstance(ident.get("tps_token_ns_evidence"), dict) else ident.get("tps_token_ns_evidence"),
    "nx": ident["nx_id"]["value"] if isinstance(ident.get("nx_id"), dict) else ident.get("nx_id"),
    "weaknesses_n": len(ident["known_weaknesses"]["value"]) if isinstance(ident.get("known_weaknesses"), dict) and isinstance(ident["known_weaknesses"].get("value"), list) else -1,
}
print("RECOVERED")
print(json.dumps(payload, sort_keys=True, default=str))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RECOVERED" in proc.stdout
    lines = [ln for ln in proc.stdout.splitlines() if ln and ln != "RECOVERED"]
    payload = json.loads(lines[-1])
    assert payload["n_fields"] == len(ri.IDENTITY_FIELDS)
    assert payload["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"
    assert payload["ebpw"] == "UNKNOWN"
    assert payload["tps"] == "UNKNOWN"
    assert payload["weaknesses_n"] > 0
    # Same NX identity as the on-disk receipt, reconstructed without collect().
    disk_nx = on_disk["nx_id"]["value"] if isinstance(on_disk.get("nx_id"), dict) else on_disk.get("nx_id")
    assert payload["nx"] == disk_nx


def test_load_without_receipt_fails_closed():
    missing = RECEIPTS / "RESIDENT_IDENTITY.does-not-exist.json"
    with pytest.raises(ri.IdentityRejectedError) as caught:
        ri.load(missing)
    assert "not on disk" in str(caught.value)


def test_load_corrupt_receipt_fails_closed(tmp_path):
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{not json")
    with pytest.raises(ri.IdentityRejectedError):
        ri.load(bogus)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": ri.SCHEMA, "identity": {}}) + "\n")
    with pytest.raises(ri.IdentityRejectedError) as caught:
        ri.load(empty)
    assert "missing required field" in str(caught.value) or "residency_status" in str(caught.value)


def test_work_unit_and_resident_callable_are_complete():
    doc = json.loads(ri.build().read_text())
    wu = doc["work_unit"]
    assert wu["id"] == ri.WORK_UNIT_ID
    assert wu["output_receipt_path"] == "receipts/future/RESIDENT_IDENTITY.json"
    assert wu["verifier"] == "tools/future/test_resident_identity.py"
    callable_ = doc["resident_callable"]
    assert "--build" in callable_["entry_point"]
    assert callable_["receipt"] == "receipts/future/RESIDENT_IDENTITY.json"
    assert callable_["work_unit_emitted"]["id"] == ri.WORK_UNIT_ID
    assert callable_["fail_closed"]
    assert doc["nomenclature"]["no_era_vi"] is True
    assert doc["nomenclature"]["no_odyssey_iv"] is True


def test_children_and_codex_handoff_cope_with_either_state():
    ident = ri.collect()
    children = ident["children"]
    assert "value" in children
    assert children.get("status") in {"NONE_RECORDED_ON_DISK", "RECORDED"}
    assert "succession.py" in str(children.get("integration_point"))
    sources = ident["authority_sources"]["codex_accelerator_handoff"]
    assert sources["source"] in {"ON_DISK", "GIT_HEAD", "ABSENT"}
    # Either the handoff is used or its absence is recorded. Never encoded as a
    # test that the file must be missing.
    assert "path" in sources


# ---------------------------------------------------------------------------
# Launch binding. Finding the receipt is not binding it.
# ---------------------------------------------------------------------------


def test_receipt_status_is_not_null_by_accident():
    """The document carries status; consumers must not read a missing key as null."""
    doc = json.loads(ri.build().read_text())
    assert doc["status"] == "ACCEPTED"
    assert doc["identity_validation"]["status"] == "ACCEPTED"
    assert doc["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"
    bind = doc["binding"]
    assert bind["status"] == "ACCEPTED"
    assert bind["status"] is not None
    assert bind["residency_status"] == "CURRENT_NONFINAL_HCLI_WORKER"


def test_binding_pins_named_fields_or_names_the_missing_one():
    ident = ri.collect()
    result = ri.describe_binding(ident)
    assert set(ri.BIND_PIN_FIELDS) == {
        "nx_id",
        "sealed_model_id",
        "executable_hash",
        "artifact_root",
        "tokenizer",
        "qualification",
    }
    for name in result["pins_named"]:
        assert name in ri.BIND_PIN_FIELDS
        assert name in result["pins"]
    if result["bound"]:
        assert result["missing"] == []
        assert result["unbound_reason"] is None
        assert result["agrees_with_incumbent"] is True
        assert set(result["pins_named"]) == set(ri.BIND_PIN_FIELDS)
        assert result["pins"]["sealed_model_id"] == result["incumbent"]["id"]
        assert result["pins"]["nx_id"]["model_id"] == result["incumbent"]["id"]
        exe = result["pins"]["executable_hash"]["by_role"]
        assert exe and all(ri._sha256_ok(d) for d in exe.values())
        assert ri._sha256_ok(result["pins"]["tokenizer"]["sha256"])
        assert result["pins"]["artifact_root"]
        assert result["pins"]["qualification"]["role"]
        assert result["status"] == "ACCEPTED"
    else:
        assert result["missing"], "unbound must name the missing field"
        assert result["unbound_reason"]
        for field in result["missing"]:
            if field in {"incumbent", "identity_validation"}:
                continue
            assert field in result["unbound_reason"] or field in result["missing"]


def test_binding_refuses_when_executable_hash_is_unpinned():
    ident = ri.collect()
    ident["executable_hash"] = {
        "value": ri.UNKNOWN,
        "missing_evidence": ["found but does not pin an executable hash"],
        "evidence": [],
        "claim_class": ri.CLAIM_CLASS,
    }
    result = ri.describe_binding(ident)
    assert result["bound"] is False
    assert "executable_hash" in result["missing"]
    assert "executable_hash" in result["unbound_reason"]
    assert result["status"] == "ACCEPTED"


def test_binding_refuses_when_tokenizer_is_unpinned():
    ident = ri.collect()
    ident["tokenizer_identity"] = {
        "value": ri.UNKNOWN,
        "sha256": ri.UNKNOWN,
        "missing_evidence": ["tokenizer not hashed"],
        "evidence": [],
        "claim_class": ri.CLAIM_CLASS,
    }
    result = ri.describe_binding(ident)
    assert result["bound"] is False
    assert "tokenizer" in result["missing"]
    assert "tokenizer" in result["unbound_reason"]


def test_binding_refuses_incumbent_disagreement():
    ident = ri.collect()
    nx = ident["nx_id"]
    if isinstance(nx, dict) and isinstance(nx.get("value"), dict):
        nx = dict(nx)
        val = dict(nx["value"])
        val["model_id"] = "not-the-incumbent"
        nx["value"] = val
        ident["nx_id"] = nx
    else:
        ident["nx_id"] = {"value": {"model_id": "not-the-incumbent"}}
    result = ri.describe_binding(ident)
    assert result["bound"] is False
    assert result["agrees_with_incumbent"] is False
    assert "incumbent" in result["unbound_reason"]
    assert result["pins"]["sealed_model_id"] == "not-the-incumbent"


def test_launch_binding_from_disk_receipt_does_not_invent_identity():
    path = ri.build()
    doc = json.loads(path.read_text())
    block = ri.launch_binding(integration="tools/future/resident_identity.py")
    assert block["found"] is True
    assert block["schema"] == ri.SCHEMA
    assert block["status"] == "ACCEPTED"
    assert block["status"] is not None
    assert block["kind"] == "resident"
    if block["bound"]:
        assert block["pins"]["sealed_model_id"] == block["incumbent"]["id"]
        assert block["incumbent"]["id"] == ri.EXPECTED_INCUMBENT_ID
        assert set(block["pins_named"]) == set(ri.BIND_PIN_FIELDS)
        assert doc["binding"]["bound"] is True
    else:
        assert block["unbound_reason"]
        assert block["missing"]


def test_launch_binding_missing_receipt_stays_unbound(tmp_path):
    probe = {
        "found": False,
        "path_taken": "not_found",
        "resolved": None,
        "doc": None,
    }
    block = ri.launch_binding(probe=probe, integration="test")
    assert block["found"] is False
    assert block["bound"] is False
    assert block["status"] is None
    assert "receipt" in block["missing"]
    assert "not invented" in block["unbound_reason"]
