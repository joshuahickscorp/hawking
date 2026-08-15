"""Evidence-backed Bible V3 fidelity and traceability report.

The report intentionally separates three things which should never be blended:

* controller coverage — whether the Bible requirement has a wired code path;
* receipt completion — whether the evidence actually certifies it; and
* live execution — whether a long-running worker is operating it.

This prevents a large scaffold from being misreported as a finished Ascension
programme while still making every missing implementation or evidence edge
visible to the next controller invocation.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.ascension_execution_plan import audit_execution_sequence, execution_rows
from lab.operators.ascension_lifecycle import (
    CANONICAL_STATES,
    EXACT_CONTINUATION_OUTPUTS,
    FAMILY_RULES,
    LifecyclePaths,
    STAGE_SPECS,
)
from lab.receipts import seal


SCHEMA = "hawking.ascension.v3_fidelity_report.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_SCHEDULE_PATH = REPO_ROOT / "workspace" / "docs" / "plans" / "ascension" / "ASCENSION_MASTER_SCHEDULE.json"


@dataclass(frozen=True)
class RequirementTrace:
    section: str
    title: str
    states: tuple[str, ...]
    implementation_paths: tuple[str, ...]
    test_paths: tuple[str, ...]


REQUIREMENT_TRACE: tuple[RequirementTrace, ...] = (
    RequirementTrace("0", "Constitutional doctrine", ("V3_ADOPT",), ("lab/operators/ascension_lifecycle.py",), ("lab/tests/test_ascension_lifecycle.py",)),
    RequirementTrace("0.1", "No-timeline law", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py",), ("lab/tests/test_ascension_lifecycle.py",)),
    RequirementTrace("1", "Conjunctive launch contract", ("GLOBAL_LAUNCH_AUDIT", "EXTERNAL_REVIEW", "APPLE_RELEASE"), ("lab/operators/ascension_lifecycle.py",), ("lab/tests/test_ascension_lifecycle.py",)),
    RequirementTrace("1.1", "Complete-BPW accounting", ("MANAGER_30B_DENSITY", "MANAGER_80B_DENSITY", *tuple(state for state, _, _ in FAMILY_RULES)), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_parity_ladder.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_parity_ladder_scaffold.py")),
    RequirementTrace("1.2", "TG3 floor", ("MANAGER_30B_TG", "MANAGER_80B_TG"), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_tg_gauntlet.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_tg_gauntlet_scaffold.py")),
    RequirementTrace("1.3", "Capability cannot be traded for launch numbers", ("MANAGER_30B_AGENT", "MANAGER_80B_AGENT", *tuple(state for state, _, _ in FAMILY_RULES)), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_tournament_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("2", "Seed Archive and state reconciliation", ("V3_SEED_ARCHIVE",), ("lab/operators/ascension_lifecycle.py", "workspace/ops/ascension/after_proto_monitor.py"), ("lab/tests/test_ascension_lifecycle.py", "workspace/ops/ascension/tests/test_after_proto_monitor.py")),
    RequirementTrace("3", "Pre-sandbox Manager Tournament", ("MANAGER_30B_DENSITY", "MANAGER_30B_TG", "MANAGER_30B_AGENT", "MANAGER_80B_DENSITY", "MANAGER_80B_TG", "MANAGER_80B_AGENT", "MANAGER_TOURNAMENT"), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_source_admission.py", "lab/operators/ascension_campaign.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_source_admission.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("4", "Maximum-Grok build doctrine", ("V3_GROK_BUILD_FABRIC",), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_foundation_contracts.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("5", "Energy and resource efficiency", ("V3_GROK_BUILD_FABRIC",), ("workspace/ops/ascension/pressure_governor.py", "workspace/ops/ascension/bounded_process_runner.py", "lab/operators/ascension_foundation_contracts.py"), ("workspace/ops/ascension/tests/test_pressure_governor.py", "workspace/ops/ascension/tests/test_bounded_process_runner.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("6", "HCLI Agent OS", ("V3_AGENT_OS", "MANAGER_30B_AGENT", "MANAGER_80B_AGENT"), ("lab/hcli/option_c.py", "lab/hcli/residency.py", "lab/hcli/self_evolution.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_foundation_contracts.py"), ("lab/tests/test_option_c.py", "lab/tests/test_residency_modes.py", "lab/tests/test_self_evolution.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("7", "Evolutionary Gravity", ("V3_GRAVITY", "MANAGER_30B_DENSITY", "MANAGER_80B_DENSITY"), ("lab/operators/ascension_parity_ladder.py", "lab/operators/qwen30b_gravity_pack.py", "lab/operators/ascension_kernel_registry.py"), ("lab/tests/test_ascension_parity_ladder_scaffold.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("8", "Exact-model Apple compiler", ("V3_METAL_COMPILER",), ("lab/operators/ascension_parity_ladder.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_kernel_registry.py"), ("lab/tests/test_ascension_parity_ladder_scaffold.py", "lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("9", "Knowledge Plane", ("V3_KNOWLEDGE_PLANE",), ("lab/operators/research_registry.py", "lab/operators/ascension_graveyard.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_knowledge_contract.py"), ("lab/tests/test_research_registry.py", "lab/tests/test_ascension_graveyard.py", "lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("10", "Protected hierarchical verification", CANONICAL_STATES, ("lab/verification_authority.py", "lab/operators/ascension_lifecycle.py"), ("lab/tests/test_verification_authority.py", "lab/tests/test_ascension_lifecycle.py")),
    RequirementTrace("11", "Fluid acquisition and storage", ("V3_SEED_ARCHIVE", "MANAGER_30B_DENSITY", "MANAGER_80B_DENSITY"), ("lab/operators/credential_broker/lifecycle.py", "lab/operators/ascension_source_admission.py", "workspace/ops/ascension/garbage_ecosystem.py"), ("lab/tests/test_credential_broker.py", "lab/tests/test_ascension_source_admission.py", "workspace/ops/ascension/tests/test_garbage_ecosystem.py")),
    RequirementTrace("12", "Sandbox lifecycle and family matrix", ("SANDBOX_ACTIVATION", *tuple(state for state, _, _ in FAMILY_RULES)), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_sandbox.py", "lab/operators/ascension_family_workflow.py", "lab/operators/ascension_tournament_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_sandbox.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("13", "Family-specific starting doctrines", tuple(state for state, _, _ in FAMILY_RULES), ("lab/operators/ascension_parity_ladder.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_kernel_registry.py"), ("lab/tests/test_ascension_parity_ladder_scaffold.py", "lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("14", "Complete-token profiler", ("MANAGER_30B_TG", "MANAGER_80B_TG"), ("lab/operators/ascension_tg_gauntlet.py",), ("lab/tests/test_ascension_tg_gauntlet_scaffold.py",)),
    RequirementTrace("15", "Temporal Gravity gauntlet", ("MANAGER_30B_TG", "MANAGER_80B_TG", "TG2_TG1_FRONTIER"), ("lab/operators/ascension_tg_gauntlet.py",), ("lab/tests/test_ascension_tg_gauntlet_scaffold.py",)),
    RequirementTrace("16", "Notifications and review packets", ("MANAGER_TOURNAMENT", "EXTERNAL_REVIEW", "APPLE_RELEASE"), ("workspace/ops/ascension/notifications.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_tournament_workflow.py", "lab/operators/ascension_release_workflow.py"), ("workspace/ops/ascension/tests/test_notifications.py", "lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("17", "Restart-safe continuation", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_campaign.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("18", "Canonical execution sequence", CANONICAL_STATES, ("lab/operators/ascension_execution_plan.py", "lab/operators/ascension_campaign.py", "lab/operators/ascension_lifecycle.py"), ("lab/tests/test_ascension_campaign.py", "lab/tests/test_ascension_lifecycle.py")),
    RequirementTrace("19", "Required artifacts and tests", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py",), ("lab/tests/test_ascension_lifecycle.py",)),
    RequirementTrace("20", "Completion states", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py",), ("lab/tests/test_ascension_lifecycle.py",)),
    RequirementTrace("21", "Global launch review packet", ("GLOBAL_LAUNCH_AUDIT", "EXTERNAL_REVIEW", "APPLE_RELEASE"), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_release_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("22", "Final directive", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_campaign.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix A", "Manager Capability Contract test catalogue", ("MANAGER_30B_AGENT", "MANAGER_80B_AGENT"), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_manager_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix B", "Grok builder templates", ("V3_GROK_BUILD_FABRIC",), ("lab/operators/ascension_foundation_contracts.py", "lab/operators/ascension_campaign.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix C", "Agent OS performance and context engineering", ("V3_AGENT_OS",), ("lab/operators/ascension_foundation_contracts.py", "lab/hcli/option_c.py"), ("lab/tests/test_option_c.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix D", "Energy-optimal campaign scheduling", ("V3_GROK_BUILD_FABRIC",), ("lab/operators/ascension_foundation_contracts.py", "workspace/ops/ascension/pressure_governor.py", "workspace/ops/ascension/bounded_process_runner.py"), ("lab/tests/test_ascension_campaign.py", "workspace/ops/ascension/tests/test_pressure_governor.py", "workspace/ops/ascension/tests/test_bounded_process_runner.py")),
    RequirementTrace("Appendix E", "Per-family qualification manifest", tuple(state for state, _, _ in FAMILY_RULES), ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_family_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix F", "Evolutionary search algorithm", ("V3_GRAVITY",), ("lab/operators/ascension_kernel_registry.py", "lab/operators/ascension_lifecycle.py", "lab/operators/ascension_parity_ladder.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_parity_ladder_scaffold.py")),
    RequirementTrace("Appendix G", "Apple product launch hardening", ("GLOBAL_LAUNCH_AUDIT", "APPLE_RELEASE"), ("lab/operators/ascension_release_workflow.py", "lab/operators/ascension_lifecycle.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix H", "V3 self-review and relaunch contract", CANONICAL_STATES, ("lab/operators/ascension_campaign.py", "lab/operators/ascension_lifecycle.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
    RequirementTrace("Appendix I", "V3 launch checklist", CANONICAL_STATES, ("lab/operators/ascension_lifecycle.py", "lab/operators/ascension_release_workflow.py"), ("lab/tests/test_ascension_lifecycle.py", "lab/tests/test_ascension_campaign.py")),
)


# Bible §18/old master schedule has 34 concrete steps.  This map means each
# schedule row has a protected state transition, even where its implementation
# remains a pending worker contract rather than a claimed completed job.
SCHEDULE_STATE_MAP: dict[int, tuple[str, ...]] = {
    0: ("V3_SEED_ARCHIVE",),
    1: ("V3_AUTHORITY_FREEZE",),
    2: ("V3_AUTHORITY_FREEZE",),
    3: ("V3_GROK_BUILD_FABRIC",),
    4: ("V3_GROK_BUILD_FABRIC",),
    5: ("V3_KNOWLEDGE_PLANE",),
    6: ("V3_METAL_COMPILER",),
    7: ("V3_GRAVITY",),
    8: ("V3_GRAVITY",),
    9: ("V3_AGENT_OS",),
    10: ("V3_GRAVITY", "V3_METAL_COMPILER"),
    11: ("MANAGER_30B_DENSITY",),
    12: ("MANAGER_30B_DENSITY",),
    13: ("MANAGER_30B_TG",),
    14: ("MANAGER_30B_TG",),
    15: ("MANAGER_30B_TG",),
    16: ("MANAGER_30B_AGENT",),
    17: ("MANAGER_80B_DENSITY",),
    18: ("MANAGER_80B_DENSITY",),
    19: ("MANAGER_80B_DENSITY",),
    20: ("MANAGER_80B_TG",),
    21: ("MANAGER_80B_TG",),
    22: ("MANAGER_80B_TG",),
    23: ("MANAGER_80B_AGENT",),
    24: ("MANAGER_TOURNAMENT", "SANDBOX_ACTIVATION"),
    25: ("SANDBOX_ACTIVATION",),
    26: ("SANDBOX_ACTIVATION",),
    27: ("SANDBOX_ACTIVATION",),
    28: ("FAMILY_QWEN",),
    29: tuple(state for state, _, _ in FAMILY_RULES),
    30: tuple(state for state, _, _ in FAMILY_RULES),
    31: ("TG2_TG1_FRONTIER",),
    32: ("GLOBAL_LAUNCH_AUDIT", "EXTERNAL_REVIEW", "APPLE_RELEASE"),
    33: ("TG2_TG1_FRONTIER",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _required_artifact_filenames() -> set[str]:
    names = set(EXACT_CONTINUATION_OUTPUTS)
    names.update({"ASCENSION_V3_CONSTITUTION.json", "ASCENSION_V3_LAUNCH_GATE.py"})
    for stage in STAGE_SPECS:
        names.update(rule.filename for rule in stage.artifacts)
    return names


def _trace_row(trace: RequirementTrace, states: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    implementation_exists = [path for path in trace.implementation_paths if (REPO_ROOT / path).is_file()]
    tests_exist = [path for path in trace.test_paths if (REPO_ROOT / path).is_file()]
    represented_states = [state for state in trace.states if state in states]
    certified_states = [state for state in trace.states if states.get(state, {}).get("status") == "CERTIFIED"]
    return {
        "section": trace.section,
        "title": trace.title,
        "states": list(trace.states),
        "represented_states": represented_states,
        "certified_states": certified_states,
        "implementation_paths": list(trace.implementation_paths),
        "implementation_paths_present": implementation_exists,
        "test_paths": list(trace.test_paths),
        "test_paths_present": tests_exist,
        "controller_contract_covered": len(implementation_exists) == len(trace.implementation_paths)
        and len(represented_states) == len(trace.states),
        "evidence_complete": len(certified_states) == len(trace.states),
        "coverage_scope": (
            "implementation-path presence plus receipt-state linkage only; "
            "it is not a claim that the runtime behavior has been measured"
        ),
    }


def _schedule_rows(states: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        schedule = json.loads(MASTER_SCHEDULE_PATH.read_text(encoding="utf-8"))
        source_steps = schedule.get("steps") if isinstance(schedule, Mapping) else None
    except (OSError, json.JSONDecodeError):
        source_steps = None
    if not isinstance(source_steps, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in source_steps:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), int):
            continue
        identifier = int(item["id"])
        mapped_states = SCHEDULE_STATE_MAP.get(identifier, ())
        state_statuses = {state: states.get(state, {}).get("status", "ABSENT") for state in mapped_states}
        rows.append(
            {
                "id": identifier,
                "description": item.get("description"),
                "schedule_declared_status": item.get("status"),
                "mapped_states": list(mapped_states),
                "mapped_state_statuses": state_statuses,
                "controller_wired": bool(mapped_states) and all(state in states for state in mapped_states),
                "evidence_complete": bool(mapped_states)
                and all(value == "CERTIFIED" for value in state_statuses.values()),
            }
        )
    return rows


def _bible_heading_routing(
    bible_path: str | Path, requirements: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Route every source ``#`` heading to an explicit controller contract.

    A hierarchy route is deliberately not a behavior claim: for example, §6.4
    is owned by the §6 Agent OS contract until a dedicated receipt proves the
    KV/state behavior.  This allows the report to expose document drift without
    mislabelling scaffold presence as a measured implementation.
    """

    path = Path(bible_path).expanduser().resolve()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "source_path": str(path),
            "headings_found": 0,
            "mapped": 0,
            "controller_contract_covered": 0,
            "unmapped": [{"title": "<Bible unreadable>", "reason": type(exc).__name__}],
            "rows": [],
            "scope": "no heading routing could be evaluated because the Bible was unreadable",
        }

    traces = {str(row.get("section")): row for row in requirements}

    def route(key: str) -> str | None:
        if key in traces:
            return key
        if key.startswith("Appendix "):
            return None
        pieces = key.split(".")
        while len(pieces) > 1:
            pieces.pop()
            candidate = ".".join(pieces)
            if candidate in traces:
                return candidate
        return None

    rows: list[dict[str, Any]] = []
    for raw_title in re.findall(r"^# (.+)$", source, flags=re.MULTILINE):
        title = raw_title.strip()
        if title == "HAWKING ASCENSION BIBLE V3":
            continue
        numeric = re.match(r"^(\d+(?:\.\d+)?)(?:\.)?\s+", title)
        appendix = re.match(r"^(Appendix [A-I])\s+", title)
        key = numeric.group(1) if numeric else appendix.group(1) if appendix else None
        routed_key = route(key) if key else None
        trace = traces.get(routed_key) if routed_key else None
        rows.append(
            {
                "heading": title,
                "section_key": key,
                "controller_trace_section": routed_key,
                "controller_contract_covered": bool(trace and trace.get("controller_contract_covered") is True),
                "evidence_complete": bool(trace and trace.get("evidence_complete") is True),
                "scope": (
                    "hierarchical contract route; not runtime measurement"
                    if trace
                    else "no controller contract route"
                ),
            }
        )
    unmapped = [
        {"heading": row["heading"], "section_key": row["section_key"]}
        for row in rows
        if row["controller_trace_section"] is None
    ]
    return {
        "source_path": str(path),
        "headings_found": len(rows),
        "mapped": len(rows) - len(unmapped),
        "controller_contract_covered": sum(
            1 for row in rows if row["controller_contract_covered"]
        ),
        "unmapped": unmapped,
        "rows": rows,
        "scope": (
            "every Markdown # heading is routed to a direct or parent controller contract; "
            "routing coverage is not a claim that the behavior has been measured"
        ),
    }


def _markdown_report(
    *,
    report: Mapping[str, Any],
    bible_execution: Sequence[Mapping[str, Any]],
    requirements: Sequence[Mapping[str, Any]],
) -> str:
    """Human-readable companion to the canonical sealed JSON report.

    The Markdown is intentionally a derived view, not a receipt.  The sealed
    JSON remains the machine-verifiable source of truth.
    """

    fidelity = report.get("fidelity") if isinstance(report.get("fidelity"), Mapping) else {}
    states = fidelity.get("live_receipt_completion") if isinstance(fidelity.get("live_receipt_completion"), Mapping) else {}
    execution = fidelity.get("bible_execution_sequence") if isinstance(fidelity.get("bible_execution_sequence"), Mapping) else {}
    headings = fidelity.get("bible_heading_routing") if isinstance(fidelity.get("bible_heading_routing"), Mapping) else {}
    lines = [
        "# Ascension V3 Fidelity Report",
        "",
        "This is a derived human-readable view. The adjacent JSON report is sealed; controller wiring is not evidence completion.",
        "",
        "## Current assessment",
        "",
        f"- Overall: `{fidelity.get('overall_status')}`",
        f"- Bible §17.1 state machine: `{fidelity.get('canonical_state_machine', {}).get('covered')}/{fidelity.get('canonical_state_machine', {}).get('required')}` exact order = `{fidelity.get('canonical_state_machine', {}).get('exact_order')}`",
        f"- Bible §17.2 continuation outputs: `{fidelity.get('continuation_outputs', {}).get('covered')}/{fidelity.get('continuation_outputs', {}).get('required')}`",
        f"- Bible §18 lines wired: `{execution.get('covered')}/{execution.get('required')}` text matches = `{execution.get('text_matches_bible')}`",
        f"- Bible Markdown heading routes: `{headings.get('mapped')}/{headings.get('headings_found')}`; controller contracts present for `{headings.get('controller_contract_covered')}/{headings.get('headings_found')}`",
        f"- Direct section/appendix controller contracts: `{fidelity.get('bible_section_controller_coverage', {}).get('covered')}/{fidelity.get('bible_section_controller_coverage', {}).get('required')}`",
        f"- Required artifact contracts mapped: `{fidelity.get('required_artifact_contract', {}).get('mapped')}/{fidelity.get('required_artifact_contract', {}).get('required')}`",
        f"- Live receipt completion: `{states.get('certified_states')}/{states.get('required_states')}` — this is intentionally separate from wiring coverage.",
        "",
        "## Bible §18 execution lines",
        "",
        "| Step | Current state(s) | Receipt status |",
        "| ---: | --- | --- |",
    ]
    for row in bible_execution:
        statuses = row.get("state_statuses") if isinstance(row.get("state_statuses"), Mapping) else {}
        state_text = ", ".join(str(item) for item in row.get("states", ()))
        status_text = ", ".join(f"{key}={value}" for key, value in statuses.items())
        lines.append(f"| {row.get('id')} | `{state_text}` | `{status_text}` |")
    lines.extend(
        [
            "",
            "## Bible sections",
            "",
            "`wired` means a controller contract and test surface exist. It does not mean a runtime measurement has been earned.",
            "",
            "| Section | Wired | Evidence complete |",
            "| --- | --- | --- |",
        ]
    )
    for row in requirements:
        lines.append(
            f"| {row.get('section')} — {row.get('title')} | `{row.get('controller_contract_covered')}` | `{row.get('evidence_complete')}` |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "No timeline, plan, running daemon, candidate metadata record, or model self-report can advance a receipt-bound V3 state. Measured, sealed controller/human evidence is required.",
            "",
        ]
    )
    return "\n".join(lines)


def build_fidelity_report(
    paths: LifecyclePaths,
    *,
    bible: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    tournament: Mapping[str, Any],
    launch_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and persist a complete, non-inflated fidelity report."""

    state_map = {str(item["id"]): item for item in states if isinstance(item, Mapping) and item.get("id")}
    requirements = [_trace_row(trace, state_map) for trace in REQUIREMENT_TRACE]
    heading_routing = _bible_heading_routing(str(bible.get("path") or ""), requirements)
    legacy_schedule = _schedule_rows(state_map)
    bible_execution = execution_rows(state_map)
    bible_execution_audit = audit_execution_sequence(str(bible.get("path") or ""))
    required_artifacts = _required_artifact_filenames()
    represented = set(EXACT_CONTINUATION_OUTPUTS)
    represented.update({"ASCENSION_V3_CONSTITUTION.json", "ASCENSION_V3_LAUNCH_GATE.py"})
    for stage in STAGE_SPECS:
        represented.update(rule.filename for rule in stage.artifacts)
    controller_sections = sum(1 for row in requirements if row["controller_contract_covered"])
    certified_states = [state for state in states if state.get("status") == "CERTIFIED"]
    report = seal(
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "bible": {
                "path": bible.get("path"),
                "sha256": bible.get("sha256"),
                "state_machine_matches": bible.get("state_machine_matches"),
            },
            "fidelity": {
                "canonical_state_machine": {
                    "covered": len(state_map),
                    "required": len(CANONICAL_STATES),
                    "fraction": len(state_map) / len(CANONICAL_STATES),
                    "exact_order": list(state_map) == list(CANONICAL_STATES),
                },
                "continuation_outputs": {
                    "covered": len([name for name in EXACT_CONTINUATION_OUTPUTS if (paths.root / name).is_file()]),
                    "required": len(EXACT_CONTINUATION_OUTPUTS),
                },
                "required_artifact_contract": {
                    "mapped": len(required_artifacts & represented),
                    "required": len(required_artifacts),
                    "unmapped": sorted(required_artifacts - represented),
                },
                "bible_section_controller_coverage": {
                    "covered": controller_sections,
                    "required": len(REQUIREMENT_TRACE),
                    "fraction": controller_sections / len(REQUIREMENT_TRACE),
                    "scope": "direct controller contracts for top-level sections and appendices; subheadings are reported separately",
                },
                "bible_heading_routing": heading_routing,
                "bible_execution_sequence": {
                    "text_matches_bible": bible_execution_audit.get("matches") is True,
                    "covered": sum(
                        1
                        for row in bible_execution
                        if all(state in state_map for state in row.get("states", []))
                    ),
                    "required": 48,
                    "receipt_complete": sum(
                        1 for row in bible_execution if row.get("evidence_complete") is True
                    ),
                    "receipt_completion_is_not_wiring_coverage": True,
                },
                "legacy_master_schedule_transition_coverage": {
                    "covered": sum(1 for row in legacy_schedule if row["controller_wired"]),
                    "required": len(legacy_schedule),
                    "fraction": (sum(1 for row in legacy_schedule if row["controller_wired"]) / len(legacy_schedule)) if legacy_schedule else 0.0,
                    "not_the_authoritative_v3_execution_sequence": True,
                },
                "live_receipt_completion": {
                    "certified_states": len(certified_states),
                    "required_states": len(CANONICAL_STATES),
                    "fraction": len(certified_states) / len(CANONICAL_STATES),
                    "is_not_implementation_coverage": True,
                },
                "overall_status": (
                    "BIBLE_EXECUTION_DRIFT_BLOCKED"
                    if bible_execution_audit.get("matches") is not True
                    else "EVIDENCE_COMPLETE"
                    if len(certified_states) == len(CANONICAL_STATES)
                    else "CONTROLLER_WIRED_EVIDENCE_INCOMPLETE"
                ),
            },
            "requirements": requirements,
            "bible_execution_sequence": {
                "audit": bible_execution_audit,
                "steps": bible_execution,
            },
            "legacy_master_schedule": legacy_schedule,
            "runtime": {
                "tournament": dict(tournament),
                "launch_gate": dict(launch_gate),
            },
            "claim_boundary": {
                "controller_coverage_is_not_receipt_completion": True,
                "scaffolds_are_not_runtime_qualification": True,
                "does_not_substitute_plans_for_measurements": True,
            },
        }
    )
    _atomic_json(paths.root / "ASCENSION_V3_FIDELITY_REPORT.json", report)
    _atomic_text(
        paths.root / "ASCENSION_V3_FIDELITY.md",
        _markdown_report(
            report=report,
            bible_execution=bible_execution,
            requirements=requirements,
        ),
    )
    return report


__all__ = [
    "MASTER_SCHEDULE_PATH",
    "REQUIREMENT_TRACE",
    "SCHEMA",
    "SCHEDULE_STATE_MAP",
    "build_fidelity_report",
]
