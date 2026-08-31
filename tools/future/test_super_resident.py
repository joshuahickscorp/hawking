"""SANDBOX_RESIDENT_FLOOR and the provider-neutral daemon.

Negative controls this file actually watches fail:
- Flash is rejected today with unmet clauses cited from real receipts
  (SCAFFOLD_ONLY NX, one accepted stateful token).
- The daemon drives a second stub provider through the identical contract.
- Protected-evidence eviction beats a provider that wants to stay loaded.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.future import super_resident as sr
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = sr.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SUPER_RESIDENT_FLOOR.json"
    assert doc["schema"] == sr.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["started_model_process"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/SUPER_RESIDENT_FLOOR.json"
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)


def test_selftest_emits_sealed_receipt():
    out = sr.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["evaluations"]["flash"]["clears_sandbox_floor"] is False


def test_floor_is_not_singularity():
    floor = sr.floor_definition()
    assert floor["name"] == "SANDBOX_RESIDENT_FLOOR"
    assert floor["distinct_from_singularity"] is True
    assert floor["not_a_promotion"] is True
    assert floor["clause_ids"] == list(sr.FLOOR_CLAUSE_IDS)
    required = (
        "self_contained_executable_identity",
        "source_independent_runtime",
        "no_forbidden_fallback",
        "stable_sessions",
        "restart",
        "capability_for_hcli_research",
        "sufficient_token_rate",
        "explicit_ebpw_evidence",
        "rollback",
    )
    assert floor["clause_ids"] == list(required)
    for clause in floor["clauses"]:
        assert "not_singularity" in clause


def test_flash_rejected_today_from_real_receipts():
    """Negative control: the floor REJECTS Flash with the unmet clauses cited live."""
    exec_doc, exec_src = sr.load_repo_json(sr.REL_FLASH_EXEC)
    nx_doc, nx_src = sr.load_repo_json(sr.REL_FLASH_NX)
    stateful, stateful_src = sr.load_repo_json(sr.REL_FLASH_STATEFUL)
    assert exec_doc is not None or nx_doc is not None or stateful is not None, (
        "Flash evidence (executable, NX, or stateful gate) was not locatable "
        "from this sparse worktree, pinned snapshot, main checkout, or git HEAD "
        "— cannot prove the SCAFFOLD_ONLY / one-token refusal"
    )

    flash = sr.evaluate_flash()
    assert flash["clears_sandbox_floor"] is False
    assert flash["clears_singularity"] is False
    unmet = flash["unmet_clauses"]
    assert "source_independent_runtime" in unmet
    assert "sufficient_token_rate" in unmet
    assert "stable_sessions" in unmet

    cited = json.dumps(flash["clauses"], sort_keys=True)
    if exec_doc is not None:
        assert exec_doc.get("status") == "SCAFFOLD_ONLY", exec_doc.get("status")
        assert exec_src
        assert "SCAFFOLD_ONLY" in cited
        assert exec_doc.get("qualification") is False
    if stateful is not None:
        accepted = sr._dot(stateful, "complete_stateful_session.accepted_generation_tokens")
        boundary = sr._dot(stateful, "first_physical_failure_boundary.status")
        assert accepted == 1 or boundary == "ONE_TOKEN_ACCEPTED", (accepted, boundary)
        assert stateful_src
        assert "accepted_generation_tokens" in cited or "ONE_TOKEN_ACCEPTED" in cited
        assert flash["one_accepted_stateful_token"] is True
    if nx_doc is not None:
        assert nx_doc.get("status") == "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"
        assert nx_src
        q = nx_doc.get("qualification") or {}
        assert q.get("resident_promotion") is not True
        assert q.get("accepted_multitoken_tps") is None
        assert q.get("complete_system_ebpw") is None

    # Codex handoff, when recoverable, must agree — and must not be required
    # if the pinned receipts already carry the same fields.
    handoff, _ = sr.load_repo_json(sr.REL_CODEX_HANDOFF)
    if isinstance(handoff, dict):
        nx_state = sr._dot(handoff, "current_flash_state.source_independent_nx") or {}
        gate = sr._dot(handoff, "current_flash_state.stateful_gate") or {}
        assert nx_state.get("status") == "SCAFFOLD_ONLY"
        assert nx_state.get("qualification") is False
        assert gate.get("accepted_tokens") == 1
        assert flash["source_independent_nx"]["status"] == "SCAFFOLD_ONLY"
        assert flash["stateful_gate"]["accepted_tokens"] == 1

    would = {
        row["clause"]: row.get("would_change")
        for row in flash["clauses"]
        if row["state"] not in sr.MET_STATES
    }
    assert would["source_independent_runtime"]
    assert "SCAFFOLD_ONLY" in would["source_independent_runtime"]
    assert would["sufficient_token_rate"]
    assert "accepted_generation_tokens > 1" in would["sufficient_token_rate"]


def test_qwen27_is_current_nonfinal_hcli_worker_not_singularity():
    ident, source = sr.load_repo_json(sr.REL_QWEN_IDENTITY)
    assert ident is not None, (
        f"{sr.REL_QWEN_IDENTITY} was not locatable — cannot evaluate the incumbent"
    )
    assert source
    qwen = sr.evaluate_qwen27()
    assert qwen["role"] == "CURRENT_NONFINAL_HCLI_WORKER"
    assert qwen["clears_singularity"] is False
    assert qwen["distinct_from_singularity"] is True
    assert qwen["resident_identity"] == ident.get("resident_identity") == "sealed-3.14"
    assert qwen["id"] == "qwen3.8-27b-sealed-3.14"
    # Floor clearance is identity-backed, not a promotion.
    assert qwen["clears_sandbox_floor"] is True
    assert "sufficient_token_rate" not in qwen["unmet_clauses"]
    cap_clause = next(c for c in qwen["clauses"] if c["clause"] == "capability_for_hcli_research")
    assert cap_clause["state"] == "MET_QUOTED"
    if qwen["evidence_recovery"]["capability"]["present"]:
        assert qwen["capability_identity_sufficient"] is False


def test_status_names_qwen_holder_without_starting_a_process():
    status = sr.sandbox_status()
    assert status["started_model_process"] is False
    assert status["took_gpu_lease"] is False
    assert status["floor_is_not_singularity"] is True
    assert status["holder"] == sr.QWEN_ID
    assert status["holder_role"] == sr.QWEN_ROLE
    assert status["flash"]["clears_sandbox_floor"] is False
    assert "SCAFFOLD_ONLY" in status["why"] or "one accepted" in status["why"]
    src = inspect.getsource(sr.sandbox_status)
    assert "HawkingNativeConnector" not in src
    assert "Popen" not in src
    assert "subprocess" not in src


def test_daemon_drives_second_stub_through_identical_contract():
    """Negative control: daemon logic is not hard-coded to one model."""
    alpha = sr.StubProvider(provider_id="stub-alpha", model_id="body-A", tool_calling=True)
    beta = sr.StubProvider(provider_id="stub-beta", model_id="body-B", tool_calling=False)
    daemon = sr.SandboxDaemon()
    drive_a = daemon.drive(alpha)
    drive_b = daemon.drive(beta)
    assert drive_a["ops"] == drive_b["ops"] == list(sr.PROVIDER_OPS)
    assert drive_a["ops_complete"] is True
    assert drive_b["ops_complete"] is True
    assert drive_a["provider_id"] == "stub-alpha"
    assert drive_b["provider_id"] == "stub-beta"
    assert drive_a["model_id"] != drive_b["model_id"]
    assert drive_a["results"]["generation"]["text"] != drive_b["results"]["generation"]["text"]
    assert drive_a["results"]["tool_calling"]["supported"] is True
    assert drive_b["results"]["tool_calling"]["supported"] is False
    src = inspect.getsource(sr.SandboxDaemon)
    lowered = src.lower()
    assert "qwen" not in lowered
    assert "flash" not in lowered
    assert "sealed-3.14" not in lowered
    proof = sr.prove_provider_neutral()
    assert proof["identical_ops"] is True
    assert proof["bodies_differ"] is True
    assert proof["daemon_source_names_no_body"] is True


def test_protected_eviction_beats_resident_convenience():
    clingy = sr.StubProvider(
        provider_id="stub-clingy",
        model_id="keep-me-loaded",
        prefer_keep_resident=True,
    )
    daemon = sr.SandboxDaemon()
    daemon.bind(clingy)
    clingy.load()
    clingy.session()
    evict = daemon.protected_evict(prefer_keep_resident=True)
    assert evict["resident_convenience_wins"] is False
    assert evict["unloaded"] is True
    assert evict["weights_dropped"] is True
    assert evict["device_released"] is True
    assert evict["prefer_keep_resident"] is True
    assert evict["priority"][-1] == "resident_convenience"
    assert clingy.state == "unloaded"
    with pytest.raises(sr.ProviderEvicted):
        daemon.generate_after_evict({"messages": []})
    with pytest.raises(sr.ResidentConvenienceError):
        daemon.protected_evict(reason="resident_convenience")
    proof = sr.prove_gpu_lease_subordination()
    assert proof["resident_convenience_wins"] is False
    assert proof["generate_after_evict_refused"] is True


def test_crash_does_not_silently_restart():
    provider = sr.StubProvider(provider_id="stub-crash", model_id="body-C")
    daemon = sr.SandboxDaemon()
    daemon.bind(provider)
    provider.load()
    crash = provider.crash_handling("boom")
    assert crash["silent_restart"] is False
    assert provider.state == "crashed"
    with pytest.raises(sr.ProviderEvicted):
        provider.generation({"messages": []})
    daemon.no_silent_restart = True
    sneaky = sr.StubProvider(provider_id="stub-sneaky", model_id="body-D")

    def _silent(event: str) -> dict:
        sneaky.crash = event
        sneaky.state = "loaded"
        return {"op": "crash_handling", "event": event, "silent_restart": True}

    sneaky.crash_handling = _silent  # type: ignore[method-assign]
    with pytest.raises(sr.SilentRestartRefused):
        daemon.drive(sneaky)


def test_provider_contract_names_every_required_op():
    contract = sr.provider_contract()
    required = (
        "load",
        "health",
        "session",
        "generation",
        "tool_calling",
        "pause",
        "resume",
        "unload",
        "capability_identity",
        "resource_identity",
        "crash_handling",
    )
    assert contract["ops"] == list(required)
    assert contract["provider_neutral"] is True
    assert contract["does_not_start_a_process"] is True
    assert contract["lease_priority"][0] == "protected_evidence_eviction"
    assert contract["lease_priority"][-1] == "resident_convenience"


def test_workunits_include_sleeping_flash_wakeup():
    status = sr.sandbox_status()
    flash = sr.evaluate_flash()
    units = sr.emit_workunits(status, flash)
    ids = [u["id"] for u in units]
    assert "future.super_resident.sandbox_floor" in ids
    wakeup = next(u for u in units if u["id"] == "future.super_resident.flash_floor_wakeup")
    assert wakeup["status"] == "sleeping"
    assert wakeup["classification"] == "SLEEPING"
    assert "SCAFFOLD_ONLY" in wakeup["blocked_reason"]
    assert wakeup["resource_class"] == "LIGHT_CONTROL"
    assert wakeup["effect_class"] == "READ_ONLY"
    for unit in units:
        assert unit["verifier"]
        assert unit["claim_boundary"]
        assert unit.get("may_promote") in (None, False)


def test_entry_point_status_seals_receipt_and_prints_holder():
    proc = subprocess.run(
        [sys.executable, str(Path(sr.__file__)), "--status"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "SANDBOX_RESIDENT_FLOOR" in out
    assert "CURRENT_NONFINAL_HCLI_WORKER" in out
    assert "flash.clears_sandbox_floor: False" in out
    assert "started_model_process: False" in out
    assert "SUPER_RESIDENT_FLOOR.json" in out
    assert "qwen" in out.lower() or "sealed-3.14" in out
    receipt = RECEIPTS / "SUPER_RESIDENT_FLOOR.json"
    assert receipt.is_file()
    doc = json.loads(receipt.read_text())
    assert doc["status"]["started_model_process"] is False


def test_hardware_claim_guard_still_armed():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"accepted_tps": 25.0})
    from tools.future._common import write_receipt

    with pytest.raises(HardwareClaimError):
        write_receipt(
            "SUPER_RESIDENT_FLOOR_SHOULD_NOT_EXIST.json",
            {"schema": "nope", "accepted_tps": 1.0},
            "tools/future/test_super_resident.py",
        )


def test_receipt_records_resident_callable_and_fail_closed():
    doc = json.loads(sr.build().read_text())
    rc = doc["resident_callable"]
    assert rc["entry_point"].endswith("super_resident.py --status")
    assert "future.super_resident.sandbox_floor" in rc["workunits_emitted"]
    assert rc["receipt"] == "receipts/future/SUPER_RESIDENT_FLOOR.json"
    assert rc["frontier_fed"]["id"] == "SANDBOX_RESIDENT_FLOOR"
    assert any("ProviderEvicted" in item for item in rc["fail_closed"])
    assert doc["status"]["holder_role"] == "CURRENT_NONFINAL_HCLI_WORKER"
    assert doc["evaluations"]["flash"]["clears_sandbox_floor"] is False
