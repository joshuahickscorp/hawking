from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_physical_release_state.py"
SPEC = importlib.util.spec_from_file_location("appendix_physical_release_state", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _observer(*, ready: bool = True) -> dict:
    return MODULE._stamp({
        "schema": "hawking.doctor_v5_post_120b_observer_state.v1",
        "final_interpretation_ready": ready,
        "source_deletion_permitted": False,
    }, "state_sha256")


def _cheap_report() -> dict:
    source = MODULE.appendix_cheap_gates.current_source_capsule()
    execution = MODULE.appendix_cheap_gates.execution_authority()
    command_manifest = MODULE.appendix_cheap_gates._command_manifest(
        MODULE.appendix_cheap_gates.GATES
    )
    rows = [
        {
            "id": gate_id,
            "command": command,
            "exit_code": 0,
            "passed": True,
        }
        for gate_id, command in MODULE.appendix_cheap_gates.GATES
    ]
    return MODULE.appendix_cheap_gates._stamp({
        "schema": MODULE.appendix_cheap_gates.SCHEMA,
        "source_commit": "a" * 40,
        "source_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": MODULE.canonical_sha256(command_manifest),
        "source_capsule": source,
        "source_capsule_sha256": source["capsule_sha256"],
        "source_capsule_after_sha256": source["capsule_sha256"],
        "source_capsule_stable_during_run": True,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution["authority_sha256"],
        "execution_authority_stable_during_run": True,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "mutates_active_corpus": False,
        "gate_count": len(rows),
        "passed_count": len(rows),
        "failed_count": 0,
        "gates": rows,
    })


def _release_cheap_report() -> dict:
    cheap = MODULE.appendix_cheap_gates
    source = cheap.current_source_capsule()
    execution = cheap.execution_authority()
    command_manifest = cheap._command_manifest(cheap.RELEASE_PACKET_GATES)
    rows = [
        {"id": gate_id, "command": command, "exit_code": 0, "passed": True}
        for gate_id, command in cheap.RELEASE_PACKET_GATES
    ]
    return cheap._stamp({
        "schema": cheap.RELEASE_PACKET_SCHEMA,
        "source_base_commit": "a" * 40,
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "gate_manifest": command_manifest,
        "gate_manifest_sha256": MODULE.canonical_sha256(command_manifest),
        "source_capsule": source,
        "source_capsule_sha256": source["capsule_sha256"],
        "source_capsule_after_sha256": source["capsule_sha256"],
        "source_capsule_stable_during_run": True,
        "execution_authority": execution,
        "execution_authority_sha256": execution["authority_sha256"],
        "execution_authority_after_sha256": execution["authority_sha256"],
        "execution_authority_stable_during_run": True,
        "uses_gpu": False,
        "reads_model_artifacts": False,
        "opens_or_hashes_active_corpus": False,
        "runs_cargo": False,
        "mutates_active_corpus": False,
        "mutates_runtime_defaults": False,
        "active_heavy_owner_count_before": 0,
        "active_heavy_owner_count_after": 0,
        "gate_count": len(rows),
        "passed_count": len(rows),
        "failed_count": 0,
        "gates": rows,
    })


def _inputs() -> dict:
    return {
        "observer": _observer(),
        "active_owners": [],
        "handoff_packet": MODULE.appendix_handoff.build_packet(),
        "cheap_gate_report": _cheap_report(),
        "release_packet_cheap_gate_report": _release_cheap_report(),
    }


def test_plan_is_validation_only_and_preserves_exact_release_order() -> None:
    plan = MODULE.build_plan()
    assert plan == MODULE.build_plan()
    assert plan["execution_capability"] is False
    assert plan["runtime_default_mutation_permitted"] is False
    assert plan["cheap_gate_contract"] == MODULE.appendix_cheap_gates.gate_contract()
    assert plan["physical_counter_collector_config_sha256"] == (
        MODULE.appendix_physical_counter_collector.build_config()["config_sha256"]
    )
    assert plan["physical_counter_authority_registry_sha256"] == (
        MODULE.appendix_physical_counter_authority.load_default_registry()["registry_sha256"]
    )
    assert plan["physical_counter_executor_capability_sha256"] == (
        MODULE.appendix_physical_counter_executor.execution_capability_contract()["capability_sha256"]
    )
    assert plan["physical_counter_request_builder_config_sha256"] == (
        MODULE.appendix_physical_counter_request.build_config()["config_sha256"]
    )
    assert plan["phase_order"] == list(MODULE.PHASES)
    physical = plan["release_sequence"][2]["requires"]
    assert any("stored device evidence before" in row for row in physical)
    assert any("B=1..8" in row for row in physical)


def test_closed_doctor_boundary_never_inspects_physical_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("aggregate physical gate was called behind a closed release boundary")

    monkeypatch.setattr(MODULE.appendix_physical_evidence_gate, "validate_gate", forbidden)
    values = _inputs()
    values["observer"] = _observer(ready=False)
    status = MODULE.build_status(**values, physical_packet={"gate_sha256": "a" * 64})
    assert status["release_boundary_open"] is False
    assert status["physical_evidence_evaluated"] is False
    assert status["admissible_phase_index"] == 0


def test_missing_packet_names_phase_and_per_b_counter_blockers() -> None:
    status = MODULE.build_status(**_inputs(), physical_packet=None)
    assert status["release_boundary_open"] is True
    assert status["appendix_audit_green"] is True
    assert status["physical_evidence_green"] is False
    blockers = "\n".join(status["blockers"])
    assert "phase-attributed" in blockers
    assert "per-B=1..8" in blockers


def test_missing_release_extension_cannot_make_appendix_audit_green() -> None:
    values = _inputs()
    values["release_packet_cheap_gate_report"] = None
    status = MODULE.build_status(**values, physical_packet=None)
    assert status["release_boundary_open"] is True
    assert status["appendix_audit_green"] is False
    assert status["admissible_phase_index"] == 1
    assert any(
        "release-extension cheap report" in blocker
        for blocker in status["blockers"]
    )


def test_stale_release_extension_command_cannot_make_appendix_audit_green() -> None:
    values = _inputs()
    report = values["release_packet_cheap_gate_report"]
    report["gates"][0]["command"] = ["python3.12", "stale-release-gate.py"]
    values["release_packet_cheap_gate_report"] = (
        MODULE.appendix_cheap_gates._stamp(report)
    )
    status = MODULE.build_status(**values, physical_packet=None)
    assert status["appendix_audit_green"] is False
    assert any("command differs" in blocker for blocker in status["blockers"])


def test_aggregate_phase_attribution_failure_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE.appendix_physical_evidence_gate,
        "validate_gate",
        lambda *_args, **_kwargs: ["B=4 counter phase marker is invalid or reused"],
    )
    status = MODULE.build_status(
        **_inputs(), physical_packet={"gate_sha256": "a" * 64},
        verify_counter_files=False,
    )
    assert status["physical_evidence_green"] is False
    assert any("B=4 counter phase marker" in row for row in status["blockers"])


def test_cas_advances_one_step_and_rejects_stale_or_skipped_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE.appendix_physical_evidence_gate,
        "validate_gate",
        lambda *_args, **_kwargs: [],
    )
    state = MODULE.initial_state()
    status = MODULE.build_status(
        **_inputs(), physical_packet={"gate_sha256": "a" * 64}, state=state,
        verify_counter_files=False,
    )
    assert status["admissible_phase_index"] == 4
    first = MODULE.transition(
        state, status, expected_state_sha256=state["state_sha256"],
        target_phase_index=1,
    )
    assert MODULE.validate_state(first) == []
    assert first["default_off"] is True and first["execution_capability"] is False
    with pytest.raises(ValueError, match="exactly one phase"):
        MODULE.transition(
            state, status, expected_state_sha256=state["state_sha256"],
            target_phase_index=2,
        )
    with pytest.raises(ValueError, match="CAS mismatch"):
        MODULE.transition(
            state, status, expected_state_sha256="0" * 64,
            target_phase_index=1,
        )


def test_state_tamper_or_activation_request_fails_closed() -> None:
    state = MODULE.initial_state()
    tampered = copy.deepcopy(state)
    tampered["activation_requested"] = True
    tampered = MODULE._stamp(tampered, "state_sha256")
    errors = MODULE.validate_state(tampered)
    assert any("default-off/no-execution" in error for error in errors)
    unstamped = copy.deepcopy(state)
    unstamped["phase"] = "sealed_default_off"
    assert any("self-hash" in error for error in MODULE.validate_state(unstamped))


def test_cas_rejects_tampered_status_and_binding_order_cannot_rebind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE.appendix_physical_evidence_gate,
        "validate_gate",
        lambda *_args, **_kwargs: [],
    )
    state = MODULE.initial_state()
    status = MODULE.build_status(
        **_inputs(), physical_packet={"gate_sha256": "a" * 64}, state=state,
        verify_counter_files=False,
    )
    tampered = copy.deepcopy(status)
    tampered["execution_capability"] = True
    with pytest.raises(ValueError, match="invalid release status"):
        MODULE.transition(
            state, tampered, expected_state_sha256=state["state_sha256"],
            target_phase_index=1,
        )

    missing_green_binding = copy.deepcopy(status)
    missing_green_binding["evidence_bindings"]["physical_evidence_gate_sha256"] = None
    missing_green_binding = MODULE._stamp(missing_green_binding, "status_sha256")
    with pytest.raises(ValueError, match="lacks physical_evidence_gate_sha256"):
        MODULE.transition(
            state, missing_green_binding,
            expected_state_sha256=state["state_sha256"], target_phase_index=1,
        )

    reordered = copy.deepcopy(status)
    reordered["evidence_bindings"] = dict(reversed(list(
        reordered["evidence_bindings"].items()
    )))
    reordered = MODULE._stamp(reordered, "status_sha256")
    successor = MODULE.transition(
        state, reordered, expected_state_sha256=state["state_sha256"],
        target_phase_index=1,
    )
    assert successor["evidence_bindings"]["observer_state_sha256"] == (
        status["evidence_bindings"]["observer_state_sha256"]
    )
    assert successor["evidence_bindings"]["appendix_handoff_packet_sha256"] is None
