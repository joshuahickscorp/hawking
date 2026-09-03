"""Canonical, auditable execution map for Bible V3 §18.

The repository contains an older 34-row planning schedule.  The V3 Bible is
the authority, however, and its canonical execution sequence has 48 numbered
steps (0 through 47).  This module keeps that sequence independently visible
to every controller.  It does *not* claim a step is complete: callers must
combine its mapping with the receipt-bound lifecycle state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BibleExecutionStep:
    """One exact §18 line and the receipt-bound state(s) that govern it."""

    identifier: int
    description: str
    states: tuple[str, ...]
    dispatch_class: str
    notes: str


_DESCRIPTIONS: tuple[str, ...] = (
    "Read V3 and adopt it as the sole canonical Ascension programme.",
    "Audit live state, processes, repo, storage, cloud, receipts and active lanes.",
    "Build and verify the V3 Seed Archive.",
    "Offload protected custom bodies and verify restoration.",
    "Freeze authority, benchmarks, hidden memberships, rollback and deletion policy.",
    "Install maximum-Grok lane scheduler and isolated worktree contracts.",
    "Install energy/resource telemetry and critical-path scheduling.",
    "Integrate HCLI Agent OS V3 into the live product path.",
    "Build the Ascension Knowledge Plane and import all prior evidence.",
    "Build/upgrade Evolutionary Gravity V3.",
    "Build/upgrade architecture fingerprinting, family plugins and exact-model Metal compiler.",
    "Freeze Qwen3-Coder-30B official source and Manager Capability Anchor.",
    "Evolve Qwen30 Gravity until complete_bpw <=1.5.",
    "Evolve Qwen30 runtime until TG3.",
    "Pass Qwen30 HCLI/Agent OS/residency/manager gates.",
    "Seal QWEN30_MANAGER_CANDIDATE.",
    "Freeze Qwen3-Coder-Next 80B official source and Manager Capability Anchor.",
    "Port every compatible Qwen30 mechanism.",
    "Build missing hybrid/DeltaNet/shared-expert semantics.",
    "Evolve Qwen80 Gravity until complete_bpw <=1.5.",
    "Evolve Qwen80 runtime until TG3.",
    "Pass Qwen80 HCLI/Agent OS/residency/manager gates.",
    "Seal QWEN80_MANAGER_CANDIDATE.",
    "Run the protected Manager Tournament.",
    "Seal ASCENSION_MANAGER.",
    "Offload and evict the alternate manager body.",
    "Activate the Ascension sandbox.",
    "Process Qwen family launch matrix.",
    "Process Llama family launch representative.",
    "Process Mistral/Mixtral family launch representative.",
    "Process DeepSeek family launch representative.",
    "Process GLM family launch representative.",
    "Process Kimi family launch representative.",
    "Process Gemma family launch representative.",
    "Process state-space/linear-attention hybrid launch representative.",
    "Verify generic HF reference intake.",
    "For every launch model, enforce <=1.5 complete BPW and TG3.",
    "Integrate every accepted clue into the cross-family Knowledge Plane.",
    "Run complete HCLI/Agent OS product gauntlet.",
    "Run storage/cleanup/recovery/foreground-user gauntlet.",
    "Run Apple packaging, install, update, restore and diagnostics.",
    "Generate the V3 Global Launch Audit.",
    "Submit the sealed review packet to Claude/Codex/human.",
    "Repair every rejected launch condition.",
    "Certify HAWKING_APPLE_V3_PRODUCTION_RELEASE_READY.",
    "Only after launch certification, continue system-wide TG2/TG1 frontier research.",
    "Restore Proto-Frankenstein later through the mature DeepSeek path.",
    "Begin CUDA only as a separate funded programme.",
)

# Multiple operational lines intentionally bind to one receipt state.  For
# example, the Manager Tournament state has three required sealed receipts:
# tournament, winner, and alternate offload.  The mapping retains those three
# Bible lines rather than collapsing them into a vague "tournament complete".
_STATE_MAP: tuple[tuple[str, ...], ...] = (
    ("V3_ADOPT",),
    ("V3_SEED_ARCHIVE",),
    ("V3_SEED_ARCHIVE",),
    ("V3_SEED_ARCHIVE",),
    ("V3_AUTHORITY_FREEZE",),
    ("V3_GROK_BUILD_FABRIC",),
    ("V3_GROK_BUILD_FABRIC",),
    ("V3_AGENT_OS",),
    ("V3_KNOWLEDGE_PLANE",),
    ("V3_GRAVITY",),
    ("V3_METAL_COMPILER",),
    ("MANAGER_30B_DENSITY",),
    ("MANAGER_30B_DENSITY",),
    ("MANAGER_30B_TG",),
    ("MANAGER_30B_AGENT",),
    ("MANAGER_30B_AGENT",),
    ("MANAGER_80B_DENSITY",),
    ("MANAGER_80B_DENSITY",),
    ("MANAGER_80B_DENSITY",),
    ("MANAGER_80B_DENSITY",),
    ("MANAGER_80B_TG",),
    ("MANAGER_80B_AGENT",),
    ("MANAGER_80B_AGENT",),
    ("MANAGER_TOURNAMENT",),
    ("MANAGER_TOURNAMENT",),
    ("MANAGER_TOURNAMENT",),
    ("SANDBOX_ACTIVATION",),
    ("FAMILY_QWEN",),
    ("FAMILY_LLAMA",),
    ("FAMILY_MISTRAL",),
    ("FAMILY_DEEPSEEK",),
    ("FAMILY_GLM",),
    ("FAMILY_KIMI",),
    ("FAMILY_GEMMA",),
    ("FAMILY_HYBRID",),
    ("GLOBAL_LAUNCH_AUDIT",),
    (
        "FAMILY_QWEN",
        "FAMILY_LLAMA",
        "FAMILY_MISTRAL",
        "FAMILY_DEEPSEEK",
        "FAMILY_GLM",
        "FAMILY_KIMI",
        "FAMILY_GEMMA",
        "FAMILY_HYBRID",
        "GLOBAL_LAUNCH_AUDIT",
    ),
    ("V3_KNOWLEDGE_PLANE", "GLOBAL_LAUNCH_AUDIT"),
    ("V3_AGENT_OS", "GLOBAL_LAUNCH_AUDIT"),
    ("GLOBAL_LAUNCH_AUDIT",),
    ("GLOBAL_LAUNCH_AUDIT",),
    ("GLOBAL_LAUNCH_AUDIT",),
    ("EXTERNAL_REVIEW",),
    ("EXTERNAL_REVIEW",),
    ("APPLE_RELEASE",),
    ("TG2_TG1_FRONTIER",),
    ("FAMILY_DEEPSEEK", "TG2_TG1_FRONTIER"),
    ("TG2_TG1_FRONTIER",),
)


def _dispatch_class(identifier: int) -> str:
    if identifier in {1, 11, 16, 35}:
        return "SAFE_CONTROLLER_AUDIT_OR_METADATA"
    if identifier in {46, 47}:
        return "POST_RELEASE_SEPARATE_PROGRAMME"
    return "CERTIFIED_EVIDENCE_GATED"


def _notes(identifier: int) -> str:
    if identifier in {11, 16}:
        return (
            "Source metadata may refresh through the credential broker, but a "
            "candidate metadata record is not the required controller-certified "
            "source/anchor evidence."
        )
    if identifier in {23, 24, 25}:
        return (
            "The tournament may be armed persistently; it cannot execute, select "
            "a winner, or evict an alternate until both candidate receipts qualify."
        )
    if identifier == 26:
        return "Sandbox activation requires the sealed winner and alternate-offload evidence."
    if identifier in {46, 47}:
        return "Explicitly outside the launch critical path and never auto-dispatched."
    return "Receipt-bound; a running controller, plan, or elapsed time cannot complete it."


BIBLE_EXECUTION_STEPS: tuple[BibleExecutionStep, ...] = tuple(
    BibleExecutionStep(
        identifier=index,
        description=description,
        states=_STATE_MAP[index],
        dispatch_class=_dispatch_class(index),
        notes=_notes(index),
    )
    for index, description in enumerate(_DESCRIPTIONS)
)

if len(BIBLE_EXECUTION_STEPS) != 48 or tuple(step.identifier for step in BIBLE_EXECUTION_STEPS) != tuple(range(48)):
    raise AssertionError("Bible §18 execution sequence must contain the exact 0..47 step range")


def extract_bible_execution_sequence(text: str) -> tuple[str, ...]:
    """Extract only the first V3 §18 code block, excluding legacy schedule prose."""

    section = re.search(
        r"^# 18\. Canonical execution sequence\s*\n\s*```text\s*\n(.*?)^```",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None:
        return ()
    rows: list[tuple[int, str]] = []
    for raw in section.group(1).splitlines():
        match = re.fullmatch(r"\s*(\d+)\.\s+(.*?)\s*", raw)
        if match is not None:
            rows.append((int(match.group(1)), match.group(2)))
    if tuple(identifier for identifier, _description in rows) != tuple(range(len(rows))):
        return ()
    return tuple(description for _identifier, description in rows)


def audit_execution_sequence(bible_path: str | Path) -> dict[str, object]:
    """Confirm the controller's direct §18 map still exactly matches the Bible."""

    path = Path(bible_path).expanduser().resolve()
    try:
        observed = extract_bible_execution_sequence(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "matches": False,
            "observed_count": 0,
            "issues": [f"cannot read Bible execution sequence: {exc}"],
        }
    expected = tuple(step.description for step in BIBLE_EXECUTION_STEPS)
    return {
        "path": str(path),
        "exists": True,
        "matches": observed == expected,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "issues": [] if observed == expected else ["Bible §18 differs from canonical execution map"],
    }


def execution_rows(states: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    """Derive an honest per-line status from lifecycle state, without mutation."""

    rows: list[dict[str, object]] = []
    for step in BIBLE_EXECUTION_STEPS:
        statuses = {
            state: str(states.get(state, {}).get("status") or "ABSENT") for state in step.states
        }
        evidence_complete = bool(statuses) and all(value == "CERTIFIED" for value in statuses.values())
        any_ready = any(value == "BLOCKED" for value in statuses.values())
        rows.append(
            {
                "id": step.identifier,
                "description": step.description,
                "states": list(step.states),
                "state_statuses": statuses,
                "dispatch_class": step.dispatch_class,
                "evidence_complete": evidence_complete,
                "admissible_for_evidence_intake": any_ready,
                "notes": step.notes,
            }
        )
    return rows


__all__ = [
    "BIBLE_EXECUTION_STEPS",
    "BibleExecutionStep",
    "audit_execution_sequence",
    "execution_rows",
    "extract_bible_execution_sequence",
]
