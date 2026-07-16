#!/usr/bin/env python3.12
"""Validation-only CAS controller for the deferred Appendix physical release.

The controller cannot execute a probe, collect a counter, open a model, mutate
the Doctor corpus, or promote a runtime default.  It projects the highest phase
that already-recorded evidence authorizes and validates one-step, hash-chained
state transitions.  Physical semantics remain owned by
``appendix_physical_evidence_gate``; this module only sequences that green gate
behind the Doctor release boundary and the durable Appendix audits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import re
import sys
from typing import Any

import appendix_cheap_gates
import appendix_contract
import appendix_handoff
import appendix_physical_counter_collector
import appendix_physical_counter_authority
import appendix_physical_counter_executor
import appendix_physical_counter_request
import appendix_physical_evidence_gate
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports" / "appendix" / "physical_release"
OBSERVER = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "post_120b" / "observer_state.json"
HANDOFF = ROOT / "reports" / "appendix" / "appendix_handoff.json"
CHEAP_GATES = ROOT / "reports" / "appendix" / "appendix_cheap_gates.json"
RELEASE_PACKET_CHEAP_GATES = (
    ROOT / "reports" / "appendix" / "appendix_release_packet_cheap_gates.json"
)
PHYSICAL_PACKET = REPORT_ROOT / "physical_evidence_packet.json"
STATE = REPORT_ROOT / "release_state.json"
PLAN = REPORT_ROOT / "release_plan.json"

PLAN_SCHEMA = "hawking.appendix_physical_release_plan.v2"
STATUS_SCHEMA = "hawking.appendix_physical_release_status.v2"
STATE_SCHEMA = "hawking.appendix_physical_release_state.v2"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

PHASES = (
    "staged_default_off",
    "doctor_final_owner_free",
    "appendix_handoff_audited",
    "physical_evidence_green",
    "sealed_default_off",
)
BINDING_PHASES = {
    "observer_state_sha256": 1,
    "appendix_handoff_packet_sha256": 2,
    "appendix_cheap_gate_report_sha256": 2,
    "appendix_release_packet_cheap_gate_report_sha256": 2,
    "physical_evidence_gate_sha256": 3,
}


def canonical_sha256(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp(value: dict[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def build_plan() -> dict[str, Any]:
    """Return the immutable, non-executing release sequence."""
    aggregate = appendix_physical_evidence_gate.requirements()
    plan = {
        "schema": PLAN_SCHEMA,
        "mode": "validation-only-default-off",
        "execution_capability": False,
        "runtime_default_mutation_permitted": False,
        "source_or_corpus_mutation_permitted": False,
        "physical_packet_path": str(PHYSICAL_PACKET),
        "state_path": str(STATE),
        "phase_order": list(PHASES),
        "release_sequence": [
            {
                "phase": PHASES[1],
                "requires": [
                    "Doctor final_interpretation_ready=true with a valid observer self-hash",
                    "zero heavy owners at the same status evaluation",
                ],
            },
            {
                "phase": PHASES[2],
                "requires": [
                    "durable Appendix handoff packet verifies against current sources",
                    "main and release-extension cheap reports verify against their exact current command manifests and dirty-tree source-byte capsule",
                ],
            },
            {
                "phase": PHASES[3],
                "requires": [
                    "one held-lease prepare-release transaction binds one boundary to the source capsule, corpus freeze, pre-build verification, build, and post-build verification",
                    "complete hash-bound corpus with a truthful explicit negative/failure/partial census (including a valid zero census)",
                    "exact isolated critical-source capsule including local Python dependency closure",
                    "resolved Cargo/Rustc identities, unique boundary+capsule target, fresh compiler artifacts, and Cargo dep-info source closures",
                    "stored device evidence before exact stored-bound compact/hashed/computed evidence",
                    "immutable phase-attributed physical counters for every deferred device cell",
                    "non-skipping B=1..8 parity and directly attributed per-B curve counters",
                    "aggregate physical evidence gate has zero errors",
                ],
            },
            {
                "phase": PHASES[4],
                "requires": [
                    "one-step CAS from the green physical-evidence state",
                    "runtime remains default-off; seal grants no activation capability",
                ],
            },
        ],
        "aggregate_gate_requirements_sha256": canonical_sha256(aggregate),
        "cheap_gate_contract": appendix_cheap_gates.gate_contract(),
        "physical_counter_collector_config_sha256": (
            appendix_physical_counter_collector.build_config()["config_sha256"]
        ),
        "physical_counter_authority_registry_sha256": (
            appendix_physical_counter_authority.load_default_registry()["registry_sha256"]
        ),
        "physical_counter_executor_capability_sha256": (
            appendix_physical_counter_executor.execution_capability_contract()["capability_sha256"]
        ),
        "physical_counter_request_builder_config_sha256": (
            appendix_physical_counter_request.build_config()["config_sha256"]
        ),
        "rollback": {
            "strategy": "remove only the separate unactivated release state",
            "runtime_sources_mutated": False,
            "completed_evidence_mutation_permitted": False,
            "parent_source_deletion_permitted": False,
            "runtime_default_restore_needed": False,
        },
    }
    return _stamp(plan, "plan_sha256")


def initial_state(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = build_plan() if plan is None else plan
    return _stamp({
        "schema": STATE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "revision": 0,
        "phase_index": 0,
        "phase": PHASES[0],
        "previous_state_sha256": None,
        "evidence_bindings": {
            "observer_state_sha256": None,
            "appendix_handoff_packet_sha256": None,
            "appendix_cheap_gate_report_sha256": None,
            "appendix_release_packet_cheap_gate_report_sha256": None,
            "physical_evidence_gate_sha256": None,
        },
        "default_off": True,
        "activation_requested": False,
        "execution_capability": False,
        "runtime_default_mutation_permitted": False,
        "rollback": plan["rollback"],
    }, "state_sha256")


def validate_state(state: Any, plan: dict[str, Any] | None = None) -> list[str]:
    plan = build_plan() if plan is None else plan
    if not isinstance(state, dict):
        return ["release state must be an object"]
    expected = {
        "schema", "plan_sha256", "revision", "phase_index", "phase",
        "previous_state_sha256", "evidence_bindings", "default_off",
        "activation_requested", "execution_capability",
        "runtime_default_mutation_permitted", "rollback", "state_sha256",
    }
    errors: list[str] = []
    if set(state) != expected:
        errors.append("release state fields are incomplete or unexpected")
    if state.get("schema") != STATE_SCHEMA or state.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("release state schema/plan binding is invalid")
    unstamped = copy.deepcopy(state)
    claimed = unstamped.pop("state_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("release state self-hash mismatch")
    index = state.get("phase_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(PHASES):
        errors.append("release state phase index is invalid")
        index = 0
    elif state.get("phase") != PHASES[index]:
        errors.append("release state phase name/index mismatch")
    revision = state.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0 or revision != index:
        errors.append("release state revision must equal its one-step phase index")
    previous = state.get("previous_state_sha256")
    if (index == 0 and previous is not None) or (
        index > 0 and (not isinstance(previous, str) or not HEX64.fullmatch(previous))
    ):
        errors.append("release state previous CAS binding is invalid")
    bindings = state.get("evidence_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(BINDING_PHASES):
        errors.append("release state evidence bindings are incomplete")
    else:
        for name, threshold in BINDING_PHASES.items():
            value = bindings.get(name)
            if index >= threshold:
                if not isinstance(value, str) or not HEX64.fullmatch(value):
                    errors.append(f"release state {name} binding is absent")
            elif value is not None:
                errors.append(f"release state {name} is bound before its phase")
    if (
        state.get("default_off") is not True
        or state.get("activation_requested") is not False
        or state.get("execution_capability") is not False
        or state.get("runtime_default_mutation_permitted") is not False
    ):
        errors.append("release state weakened the default-off/no-execution boundary")
    if state.get("rollback") != plan.get("rollback"):
        errors.append("release state rollback contract differs from the plan")
    return errors


def validate_status(
    status: Any, *, state: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> list[str]:
    """Validate an untrusted status projection before it can authorize CAS."""
    plan = build_plan() if plan is None else plan
    if not isinstance(status, dict):
        return ["release status must be an object"]
    expected = {
        "schema", "plan_sha256", "current_state_sha256",
        "current_phase_index", "current_phase", "admissible_phase_index",
        "admissible_phase", "release_boundary_open", "appendix_audit_green",
        "physical_evidence_evaluated", "physical_evidence_green",
        "default_off", "execution_capability",
        "runtime_default_mutation_permitted", "evidence_bindings", "blockers",
        "status_sha256",
    }
    errors: list[str] = []
    if set(status) != expected:
        errors.append("release status fields are incomplete or unexpected")
    if status.get("schema") != STATUS_SCHEMA or status.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("release status schema/plan binding is invalid")
    unstamped = copy.deepcopy(status)
    claimed = unstamped.pop("status_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("release status self-hash mismatch")
    current_index = status.get("current_phase_index")
    if isinstance(current_index, bool) or not isinstance(current_index, int) or not 0 <= current_index < len(PHASES):
        errors.append("release status current phase index is invalid")
    elif status.get("current_phase") != PHASES[current_index]:
        errors.append("release status current phase name/index mismatch")
    admissible = status.get("admissible_phase_index")
    if isinstance(admissible, bool) or not isinstance(admissible, int) or not 0 <= admissible < len(PHASES):
        errors.append("release status admissible phase index is invalid")
    elif status.get("admissible_phase") != PHASES[admissible]:
        errors.append("release status admissible phase name/index mismatch")
    if state is not None:
        if status.get("current_state_sha256") != state.get("state_sha256"):
            errors.append("release status is not bound to the current state")
        if current_index != state.get("phase_index") or status.get("current_phase") != state.get("phase"):
            errors.append("release status current phase differs from state")
    bindings = status.get("evidence_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(BINDING_PHASES):
        errors.append("release status evidence bindings are incomplete")
    else:
        for name, value in bindings.items():
            if value is not None and (not isinstance(value, str) or not HEX64.fullmatch(value)):
                errors.append(f"release status {name} binding is invalid")
    release_open = status.get("release_boundary_open")
    audit_green = status.get("appendix_audit_green")
    physical_evaluated = status.get("physical_evidence_evaluated")
    physical_green = status.get("physical_evidence_green")
    if any(not isinstance(value, bool) for value in (
        release_open, audit_green, physical_evaluated, physical_green,
    )):
        errors.append("release status gate flags must be booleans")
    elif audit_green and not release_open:
        errors.append("release status audit cannot be green behind a closed Doctor boundary")
    elif physical_evaluated and not audit_green:
        errors.append("release status physical evidence cannot be evaluated before audit")
    elif physical_green and not physical_evaluated:
        errors.append("release status physical evidence cannot be green without evaluation")
    if isinstance(bindings, dict) and set(bindings) == set(BINDING_PHASES):
        required_flags = {
            "observer_state_sha256": release_open,
            "appendix_handoff_packet_sha256": audit_green,
            "appendix_cheap_gate_report_sha256": audit_green,
            "appendix_release_packet_cheap_gate_report_sha256": audit_green,
            "physical_evidence_gate_sha256": physical_green,
        }
        for name, required in required_flags.items():
            if required and (
                not isinstance(bindings.get(name), str)
                or not HEX64.fullmatch(bindings[name])
            ):
                errors.append(f"release status green gate lacks {name}")
    expected_admissible = 4 if physical_green else (2 if audit_green else (1 if release_open else 0))
    if admissible != expected_admissible:
        errors.append("release status admissible phase contradicts its gate flags")
    if (
        status.get("default_off") is not True
        or status.get("execution_capability") is not False
        or status.get("runtime_default_mutation_permitted") is not False
    ):
        errors.append("release status weakened the default-off/no-execution boundary")
    blockers = status.get("blockers")
    if not isinstance(blockers, list) or any(not isinstance(row, str) or not row for row in blockers):
        errors.append("release status blockers are malformed")
    return errors


def _observer_errors(observer: Any) -> list[str]:
    if not isinstance(observer, dict):
        return ["Doctor observer state is missing"]
    errors: list[str] = []
    if observer.get("schema") != "hawking.doctor_v5_post_120b_observer_state.v1":
        errors.append("Doctor observer schema is invalid")
    unstamped = copy.deepcopy(observer)
    claimed = unstamped.pop("state_sha256", None)
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed) or claimed != canonical_sha256(unstamped):
        errors.append("Doctor observer state self-hash mismatch")
    if observer.get("final_interpretation_ready") is not True:
        errors.append("Doctor final_interpretation_ready is false")
    if observer.get("source_deletion_permitted") is not False:
        errors.append("Doctor observer weakened source preservation")
    return errors


def build_status(
    *,
    observer: Any,
    active_owners: list[dict[str, Any]],
    handoff_packet: Any,
    cheap_gate_report: Any,
    release_packet_cheap_gate_report: Any,
    physical_packet: Any | None,
    state: Any | None = None,
    verify_counter_files: bool = True,
) -> dict[str, Any]:
    """Project readiness without executing or mutating any workload."""
    plan = build_plan()
    current = initial_state(plan) if state is None else state
    state_errors = validate_state(current, plan)
    doctor_errors = _observer_errors(observer)
    if active_owners:
        doctor_errors.append(f"{len(active_owners)} heavy owner(s) remain")

    handoff_errors = appendix_handoff.verify_packet(handoff_packet)
    cheap_errors = appendix_cheap_gates.verify_report(cheap_gate_report)
    release_cheap_errors = appendix_cheap_gates.verify_release_packet_report(
        release_packet_cheap_gate_report
    )
    cheap_contract = plan["cheap_gate_contract"]
    plan_binding_errors: list[str] = []
    if not isinstance(cheap_gate_report, dict) or (
        cheap_gate_report.get("gate_manifest_sha256")
        != cheap_contract["main_gate_manifest_sha256"]
    ):
        plan_binding_errors.append("main cheap report is not bound to the release plan command manifest")
    if not isinstance(release_packet_cheap_gate_report, dict) or (
        release_packet_cheap_gate_report.get("gate_manifest_sha256")
        != cheap_contract["release_packet_gate_manifest_sha256"]
    ):
        plan_binding_errors.append(
            "release-extension cheap report is not bound to the release plan command manifest"
        )
    for label, report in (
        ("main", cheap_gate_report),
        ("release-extension", release_packet_cheap_gate_report),
    ):
        if not isinstance(report, dict) or (
            report.get("source_capsule_sha256")
            != cheap_contract["source_capsule_sha256"]
        ):
            plan_binding_errors.append(
                f"{label} cheap report is not bound to the release plan source capsule"
            )
        if not isinstance(report, dict) or (
            report.get("execution_authority_sha256")
            != cheap_contract["execution_authority_sha256"]
        ):
            plan_binding_errors.append(
                f"{label} cheap report is not bound to the release plan execution authority"
            )
    audit_errors = [
        *handoff_errors,
        *(f"main cheap report: {error}" for error in cheap_errors),
        *(f"release-extension cheap report: {error}" for error in release_cheap_errors),
        *plan_binding_errors,
    ]

    release_open = not doctor_errors
    audit_green = release_open and not audit_errors
    physical_errors: list[str] = []
    physical_evaluated = False
    if audit_green:
        physical_evaluated = True
        if physical_packet is None:
            physical_errors = [
                "physical evidence packet is absent",
                "device phase-attributed immutable counter trials are absent",
                "spec per-B=1..8 independently attributed counter repeats are absent",
            ]
        else:
            physical_errors = appendix_physical_evidence_gate.validate_gate(
                physical_packet, verify_counter_files=verify_counter_files,
            )

    admissible = 0
    if release_open:
        admissible = 1
    if audit_green:
        admissible = 2
    if audit_green and physical_evaluated and not physical_errors:
        # A green aggregate packet authorizes the evidence phase and its
        # no-activation seal.  CAS still requires two separate transitions.
        admissible = 4

    blockers: list[str] = []
    blockers.extend(f"doctor_release: {error}" for error in doctor_errors)
    if release_open:
        blockers.extend(f"appendix_audit: {error}" for error in audit_errors)
    else:
        blockers.append("appendix_audit: dependency Doctor release boundary is closed")
    if audit_green:
        blockers.extend(f"physical_evidence: {error}" for error in physical_errors)
    else:
        blockers.append("physical_evidence: not inspected until Doctor release and Appendix audit pass")
    blockers.extend(f"state: {error}" for error in state_errors)

    bindings = {
        "observer_state_sha256": observer.get("state_sha256") if isinstance(observer, dict) else None,
        "appendix_handoff_packet_sha256": handoff_packet.get("packet_sha256") if isinstance(handoff_packet, dict) else None,
        "appendix_cheap_gate_report_sha256": cheap_gate_report.get("report_sha256") if isinstance(cheap_gate_report, dict) else None,
        "appendix_release_packet_cheap_gate_report_sha256": (
            release_packet_cheap_gate_report.get("report_sha256")
            if isinstance(release_packet_cheap_gate_report, dict) else None
        ),
        "physical_evidence_gate_sha256": physical_packet.get("gate_sha256") if isinstance(physical_packet, dict) else None,
    }
    return _stamp({
        "schema": STATUS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "current_state_sha256": current.get("state_sha256") if isinstance(current, dict) else None,
        "current_phase_index": current.get("phase_index") if isinstance(current, dict) else None,
        "current_phase": current.get("phase") if isinstance(current, dict) else None,
        "admissible_phase_index": admissible,
        "admissible_phase": PHASES[admissible],
        "release_boundary_open": release_open,
        "appendix_audit_green": audit_green,
        "physical_evidence_evaluated": physical_evaluated,
        "physical_evidence_green": physical_evaluated and not physical_errors,
        "default_off": True,
        "execution_capability": False,
        "runtime_default_mutation_permitted": False,
        "evidence_bindings": bindings,
        "blockers": blockers,
    }, "status_sha256")


def transition(
    state: dict[str, Any], status: dict[str, Any], *,
    expected_state_sha256: str, target_phase_index: int,
) -> dict[str, Any]:
    """Return one CAS successor; never writes or activates anything."""
    plan = build_plan()
    errors = validate_state(state, plan)
    if errors:
        raise ValueError("invalid prior release state: " + "; ".join(errors))
    status_errors = validate_status(status, state=state, plan=plan)
    if status_errors:
        raise ValueError("invalid release status: " + "; ".join(status_errors))
    if state["state_sha256"] != expected_state_sha256:
        raise ValueError("release state CAS mismatch")
    if status.get("current_state_sha256") != state["state_sha256"]:
        raise ValueError("status is not bound to the prior release state")
    if status.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("status is not bound to the current release plan")
    if target_phase_index != state["phase_index"] + 1:
        raise ValueError("release transitions must advance exactly one phase")
    if target_phase_index > status.get("admissible_phase_index", -1):
        raise ValueError("target release phase is not admissible")
    bindings = {
        name: status["evidence_bindings"][name]
        if target_phase_index >= threshold else None
        for name, threshold in BINDING_PHASES.items()
    }
    successor = _stamp({
        "schema": STATE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "revision": target_phase_index,
        "phase_index": target_phase_index,
        "phase": PHASES[target_phase_index],
        "previous_state_sha256": state["state_sha256"],
        "evidence_bindings": bindings,
        "default_off": True,
        "activation_requested": False,
        "execution_capability": False,
        "runtime_default_mutation_permitted": False,
        "rollback": plan["rollback"],
    }, "state_sha256")
    successor_errors = validate_state(successor, plan)
    if successor_errors:
        raise AssertionError("constructed invalid release state: " + "; ".join(successor_errors))
    return successor


def _load_optional(path: pathlib.Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def current_status() -> dict[str, Any]:
    observer = _load_optional(OBSERVER)
    handoff = _load_optional(HANDOFF)
    cheap = _load_optional(CHEAP_GATES)
    release_cheap = _load_optional(RELEASE_PACKET_CHEAP_GATES)
    state = _load_optional(STATE)
    owners = spec_reentry_scaffold.active_heavy_owners()
    # Do not even load the potentially large physical packet until the cheap
    # release/audit dependencies pass.  build_status independently rechecks the
    # same gates before invoking the aggregate validator.
    physical = None
    if (
        not owners and not _observer_errors(observer)
        and not appendix_handoff.verify_packet(handoff)
        and not appendix_cheap_gates.verify_report(cheap)
        and not appendix_cheap_gates.verify_release_packet_report(release_cheap)
    ):
        physical = _load_optional(PHYSICAL_PACKET)
    return build_status(
        observer=observer,
        active_owners=owners,
        handoff_packet=handoff,
        cheap_gate_report=cheap,
        release_packet_cheap_gate_report=release_cheap,
        physical_packet=physical,
        state=state,
    )


def _selftest() -> int:
    plan = build_plan()
    state = initial_state(plan)
    assert validate_state(state, plan) == []
    assert plan["execution_capability"] is False
    print("appendix_physical_release_state.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--write-plan", type=pathlib.Path)
    parser.add_argument("--write-status", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.plan or args.write_plan is not None:
        value = build_plan()
        if args.write_plan is not None:
            _atomic_json(args.write_plan, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.status or args.write_status is not None:
        value = current_status()
        if args.write_status is not None:
            _atomic_json(args.write_status, value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["physical_evidence_green"] else 75
    parser.error("choose --plan, --status, --write-plan, --write-status, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
