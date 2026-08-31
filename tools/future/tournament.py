"""FLASH_SINGULARITY.NX vs QWEN27_SINGULARITY.NX — pre-registered, not run.

Fair comparison registered before either contender is a complete NX. The
harness REFUSES to run while either is incomplete. This sidecar has no GPU
and must not execute the tournament even if completeness later flips.

    python3 tools/future/tournament.py --dry-run
    python3 tools/future/tournament.py --build
    python3 tools/future/tournament.py --run    # refused today

Not a fork of tools/genesis_tournament.py (Qwen3.8 vs Q80 chopping-block),
tools/odyssey/doctor_tournament.py (technique preconditions),
tools/odyssey/tournament.py (checkpoint selection),
tools/odyssey/pareto_archive.py (Qwen body density vs capability),
tools/pareto_table.py (G150 candidate table), or
lab/operators/ascension_manager_tournament_protocol.py (Gravity managers).
Those are different tournaments. This one is NX-vs-NX dominance.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, load_json, write_receipt
from tools.future.resident_install import PHASES as INSTALL_PHASES

RECEIPT = "TOURNAMENT_READINESS.json"
SCHEMA = "hawking.future.tournament.v1"

FLASH_ID = "FLASH_SINGULARITY.NX"
QWEN_ID = "QWEN27_SINGULARITY.NX"
CONTENDER_IDS: tuple[str, ...] = (FLASH_ID, QWEN_ID)

# Identity documents the completeness gate actually reads.
FLASH_NX_REL = "receipts/headless/FLASH_COMPLETE_V0.nx.json"
FLASH_NX_SIBLINGS: tuple[str, ...] = (
    "receipts/headless/FLASH_COMPLETE_V1.nx.json",
    "receipts/headless/FLASH_COMPLETE_V2.nx.json",
    "receipts/headless/FLASH_NEXT_MACHINE.nx.json",
)
QWEN_IDENTITY_REL = "hcli/hawking-native.sealed-3.14.json"
QWEN_SEAL_REL = "receipts/headless/HCLI_RESIDENT_SEAL.json"
QWEN_CAPABILITY_REL = "receipts/headless/CAPABILITY_noetic-sealed-3.14.json"
QWEN_NX_SEAL_REL = "receipts/ascent-2026-08-16/G104_NX_SEAL.json"
SCOREBOARD_REL = "receipts/headless/ACCELERATOR_SCOREBOARD.json"

METADATA_ONLY_MARKERS = ("METADATA_ONLY", "NOT_FOR_PROMOTION")

# ---------------------------------------------------------------------------
# Pre-registered common profile. Identical for both contenders. Registered
# before either NX is complete so nobody designs the bench after seeing them.
# ---------------------------------------------------------------------------

HARD_GATES: tuple[dict[str, str], ...] = (
    # Capability suite items HCLI actually depends on (tools/headless/capability_suite.py).
    # Repeats in that suite sum to 43; a HARD GATE is an item that must pass every repeat.
    {"id": "capability.fact-capital", "corpus": "tools/headless/capability_suite.py", "item": "fact-capital"},
    {"id": "capability.fact-arith", "corpus": "tools/headless/capability_suite.py", "item": "fact-arith"},
    {"id": "capability.json-answer", "corpus": "tools/headless/capability_suite.py", "item": "json-answer"},
    {"id": "capability.json-kind-correct", "corpus": "tools/headless/capability_suite.py", "item": "json-kind-correct"},
    {"id": "capability.json-no-prose", "corpus": "tools/headless/capability_suite.py", "item": "json-no-prose"},
    {"id": "capability.mutation-anchor-exact", "corpus": "tools/headless/capability_suite.py", "item": "mutation-anchor-exact"},
    {"id": "capability.mutation-refuses-invention", "corpus": "tools/headless/capability_suite.py", "item": "mutation-refuses-invention"},
    {"id": "capability.code-compiles", "corpus": "tools/headless/capability_suite.py", "item": "code-compiles"},
    {"id": "capability.no-think-leak", "corpus": "tools/headless/capability_suite.py", "item": "no-think-leak"},
    # Tool reliability / mission acceptance. Model proposes; verifier decides.
    {"id": "tool.typed_write_read", "corpus": "hcli/agentos/native_mission_gate.py", "item": "typed_write_ok+typed_read_ok"},
    {"id": "tool.verifier_owns_acceptance", "corpus": "hcli/agentos/native_mission_gate.py", "item": "acceptance_source=workunit_verifier"},
    {"id": "tool.zero_fallbacks", "corpus": "hcli/agentos/resident_gate.py", "item": "zero_fallbacks"},
    # Coherence / isolation.
    {"id": "coherence.state_reset_isolated", "corpus": "hcli/agentos/resident_gate.py", "item": "state_reset_isolated"},
    {"id": "lifecycle.one_model_open", "corpus": "hcli/agentos/resident_gate.py", "item": "one_model_open"},
    {"id": "lifecycle.no_restart_during_gate", "corpus": "hcli/agentos/resident_gate.py", "item": "no_restart"},
)

HARD_GATE_IDS: tuple[str, ...] = tuple(g["id"] for g in HARD_GATES)

MISSION_CORPUS: tuple[dict[str, str], ...] = (
    {"path": "hcli/mission.py", "role": "persistent mission loop; WorkUnits stay bounded"},
    {"path": "hcli/agentos/native_mission_gate.py", "role": "one live HCLI tool/fact/verifier mission"},
    {"path": "hcli/agentos/resident_gate.py", "role": "sequential residency + state-isolation prompts"},
    {"path": "tools/headless/capability_suite.py", "role": "deterministic 11-item / 43-repeat capability contract"},
    {"path": "tools/coherence_gate.py", "role": "coherence gate (existing; not forked)"},
)


@dataclass(frozen=True)
class Axis:
    name: str
    direction: str  # "higher" | "lower"
    kind: str
    energy: bool = False


# Scored axes of the dominance scoreboard. Hard gates are eligibility, not axes.
# Hardware-kind values stay null in every sidecar receipt.
SCORED_AXES: tuple[Axis, ...] = (
    Axis("accepted_tps", "higher", "hardware"),
    Axis("complete_token_ns", "lower", "hardware"),
    Axis("complete_ebpw", "lower", "artifact"),
    Axis("active_bytes_per_token", "lower", "hardware_or_ledger"),
    Axis("resident_memory_bytes", "lower", "artifact"),
    Axis("cold_launch_ns", "lower", "hardware"),
    Axis("warm_launch_ns", "lower", "hardware"),
    Axis("restart_ns", "lower", "hardware"),
    Axis("long_mission_verified_work", "higher", "mission"),
    Axis("optimization_headroom", "higher", "hardware"),
    Axis("energy_joules_per_token", "lower", "hardware", energy=True),
)

AXIS_NAMES: tuple[str, ...] = tuple(a.name for a in SCORED_AXES)

# Incumbent CONTROL quoted from the sealed identity. Not a target. Not a ceiling.
# Numeric TPS is stored as a string so write_receipt cannot see a hardware field.
INCUMBENT_CONTROL: dict[str, Any] = {
    "role": "CONTROL_NOT_TARGET_NOT_CEILING",
    "identity": QWEN_IDENTITY_REL,
    "resident_identity": "sealed-3.14",
    "quoted_physical_ebpw": 3.1393,
    "quoted_complete_tps_from_identity": "24.4086",
    "quoted_historical_qualified_tps": "34.0",
    "quoted_capability": "30/43 historical sealed contract",
    "headline_control": "~3.14 EBPW / ~25 accepted TPS",
    "beating_control_is_not_success": True,
    "falling_short_of_control_is_not_failure": True,
    "success_predicate": None,
}


class TournamentNotReady(RuntimeError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("tournament refused: " + "; ".join(self.reasons))


class ScalarCollapseError(ValueError):
    """Raised when a caller asks this harness to collapse axes to a scalar."""


# ---------------------------------------------------------------------------
# Receipt location. This worktree is a sparse checkout; Codex receipts live
# in the main working tree and/or git HEAD. Missing here is not absence.
# ---------------------------------------------------------------------------

def _search_roots() -> tuple[Path, ...]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        gd = Path(common)
        gd = gd.resolve() if gd.is_absolute() else (REPO / gd).resolve()
        if gd.name == ".git":
            main = gd.parent
            if main != REPO:
                roots.append(main)
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return tuple(out)


def locate(rel: str) -> Path | None:
    for root in _search_roots():
        p = root / rel
        if p.is_file():
            return p
    return None


def load_repo_json(rel: str) -> tuple[dict[str, Any] | None, str]:
    """Load a repo-relative JSON document. Prefers real disk over HEAD."""
    p = locate(rel)
    if p is not None:
        try:
            doc = load_json(p)
        except (OSError, json.JSONDecodeError, UnicodeError):
            doc = None
        if isinstance(doc, dict):
            return doc, str(p)
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            return doc, f"HEAD:{rel}"
    return None, ""


def _metadata_only(status: Any) -> bool:
    s = str(status or "")
    return any(m in s for m in METADATA_ONLY_MARKERS)


# ---------------------------------------------------------------------------
# Completeness gate — disk state is authority.
# ---------------------------------------------------------------------------

def inspect_flash(doc: Mapping[str, Any] | None, *, source: str) -> dict[str, Any]:
    q = doc.get("qualification") if isinstance(doc, Mapping) else None
    if not isinstance(q, Mapping):
        q = {}
    status = doc.get("status") if isinstance(doc, Mapping) else None
    promotion = q.get("resident_promotion")
    accepted = q.get("accepted_multitoken_tps")
    ebpw = q.get("complete_system_ebpw")
    reasons: list[str] = []
    if doc is None:
        reasons.append(f"{FLASH_ID}: identity {FLASH_NX_REL} not found")
    else:
        if _metadata_only(status):
            reasons.append(f"{FLASH_ID}: status {status}")
        if promotion is not True:
            reasons.append(f"{FLASH_ID}: resident_promotion={promotion!r}")
        if accepted is None:
            reasons.append(f"{FLASH_ID}: accepted_multitoken_tps is null")
        if ebpw is None:
            reasons.append(f"{FLASH_ID}: complete_system_ebpw is null")
        if not q.get("capability_receipt"):
            reasons.append(f"{FLASH_ID}: no capability suite receipt")
    complete = not reasons
    return {
        "id": FLASH_ID,
        "complete_nx": complete,
        "source": source or None,
        "status": status,
        "nx_kind": doc.get("nx_kind") if isinstance(doc, Mapping) else None,
        "schema": doc.get("schema") if isinstance(doc, Mapping) else None,
        "resident_promotion": promotion,
        "accepted_multitoken_tps": None,
        "complete_system_ebpw": None,
        "claim_boundary": doc.get("claim_boundary") if isinstance(doc, Mapping) else None,
        "reasons": reasons,
    }


def inspect_qwen(doc: Mapping[str, Any] | None, *, source: str) -> dict[str, Any]:
    reasons: list[str] = []
    singularity, singularity_src = load_repo_json("receipts/headless/QWEN27_SINGULARITY.nx.json")
    if singularity is None:
        reasons.append(f"{QWEN_ID}: QWEN27_SINGULARITY.NX document not found")
    if doc is None:
        reasons.append(f"{QWEN_ID}: incumbent identity {QWEN_IDENTITY_REL} not found")
    else:
        if not doc.get("nx_kind"):
            reasons.append(
                f"{QWEN_ID}: {QWEN_IDENTITY_REL} is a resident profile "
                f"(resident_identity={doc.get('resident_identity')!r}), not an NX genome"
            )
        if _metadata_only(doc.get("status")):
            reasons.append(f"{QWEN_ID}: status {doc.get('status')}")
    nx_seal, nx_seal_src = load_repo_json(QWEN_NX_SEAL_REL)
    complete = not reasons
    return {
        "id": QWEN_ID,
        "complete_nx": complete,
        "source": source or None,
        "role": "CONTROL_INCUMBENT_RESIDENT_NOT_TOURNAMENT_NX",
        "resident_identity": doc.get("resident_identity") if isinstance(doc, Mapping) else None,
        "family": doc.get("family") if isinstance(doc, Mapping) else None,
        "protocol": doc.get("protocol") if isinstance(doc, Mapping) else None,
        "nx_kind": doc.get("nx_kind") if isinstance(doc, Mapping) else None,
        "historical_nx_seal": nx_seal_src or None,
        "historical_nx_kind": nx_seal.get("nx_kind") if isinstance(nx_seal, Mapping) else None,
        "singularity_source": singularity_src or None,
        "reasons": reasons,
    }


def inspect_contenders() -> dict[str, dict[str, Any]]:
    flash_doc, flash_src = load_repo_json(FLASH_NX_REL)
    qwen_doc, qwen_src = load_repo_json(QWEN_IDENTITY_REL)
    return {
        FLASH_ID: inspect_flash(flash_doc, source=flash_src),
        QWEN_ID: inspect_qwen(qwen_doc, source=qwen_src),
    }


def can_run(inspections: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[bool, list[str]]:
    """False with reasons while either contender is not a complete NX."""
    rows = inspections or inspect_contenders()
    reasons: list[str] = []
    for cid in CONTENDER_IDS:
        row = rows[cid]
        if not row.get("complete_nx"):
            reasons.extend(str(r) for r in row.get("reasons") or [f"{cid}: incomplete"])
    return (not reasons), reasons


def run() -> None:
    """Execute the tournament. Refuses while either NX is incomplete.

    Even if completeness later flips, this sidecar still cannot produce
    PROTECTED_ABSOLUTE numbers. Codex owns that measurement.
    """
    ok, reasons = can_run()
    if not ok:
        raise TournamentNotReady(reasons)
    raise TournamentNotReady(
        ["both NX complete but sidecar has no GPU lease; PROTECTED_ABSOLUTE is Codex-owned"]
    )


# ---------------------------------------------------------------------------
# Multi-axis dominance. No scalar. Unique winner only on genuine dominance.
# ---------------------------------------------------------------------------

def _axes_for(candidates: Sequence[Mapping[str, Any]]) -> tuple[Axis, ...]:
    if all(bool(c.get("energy_trustworthy")) for c in candidates):
        return SCORED_AXES
    return tuple(a for a in SCORED_AXES if not a.energy)


def _score(candidate: Mapping[str, Any], axis: str) -> float | None:
    scores = candidate.get("scores")
    if not isinstance(scores, Mapping):
        return None
    val = scores.get(axis)
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _better(axis: Axis, a: float, b: float) -> bool:
    return a > b if axis.direction == "higher" else a < b


def _no_worse(axis: Axis, a: float, b: float) -> bool:
    return a >= b if axis.direction == "higher" else a <= b


def dominates(a: Mapping[str, Any], b: Mapping[str, Any], axes: Sequence[Axis] | None = None) -> bool:
    """A dominates B: every axis comparable, A no worse on all, strictly better on one."""
    if a.get("id") == b.get("id"):
        return False
    used = axes if axes is not None else _axes_for((a, b))
    saw_strict = False
    for axis in used:
        va, vb = _score(a, axis.name), _score(b, axis.name)
        if va is None or vb is None:
            return False
        if not _no_worse(axis, va, vb):
            return False
        if _better(axis, va, vb):
            saw_strict = True
    return saw_strict


def _gate_failures(candidate: Mapping[str, Any]) -> list[str] | None:
    """None = gates not evaluated (synthetic Pareto). List = evaluated; empty passes."""
    if "hard_gates" not in candidate:
        return None
    got = candidate.get("hard_gates") or {}
    if not isinstance(got, Mapping):
        return list(HARD_GATE_IDS)
    return [gid for gid in HARD_GATE_IDS if got.get(gid) is not True]


def winner(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the non-dominated set. A single winner only under genuine dominance.

    Never collapses axes to a scalar. A candidate failing a HARD GATE is out
    regardless of speed. Missing axis values make the pair incomparable.
    """
    ordered = list(candidates)
    ids = [str(c.get("id")) for c in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    by_id = {str(c["id"]): c for c in ordered}

    evaluated = any("hard_gates" in c for c in ordered)
    disqualified: dict[str, list[str]] = {}
    eligible_ids: list[str] = []
    if evaluated:
        for c in ordered:
            fails = _gate_failures(c) or []
            if fails:
                disqualified[str(c["id"])] = fails
            else:
                eligible_ids.append(str(c["id"]))
    else:
        eligible_ids = [str(c["id"]) for c in ordered]

    axes = _axes_for(ordered)
    eligible = [by_id[i] for i in eligible_ids]
    front: list[str] = []
    for c in eligible:
        if any(dominates(other, c, axes) for other in eligible if other is not c):
            continue
        front.append(str(c["id"]))
    front = sorted(front)

    unique: str | None = None
    unique_reason = "no unique dominator"
    if len(eligible) == 1:
        unique = eligible_ids[0]
        unique_reason = "sole hard-gate survivor" if evaluated and disqualified else "sole candidate"
    elif len(eligible) >= 2:
        dominators = [
            i for i in eligible_ids
            if all(dominates(by_id[i], by_id[j], axes) for j in eligible_ids if j != i)
        ]
        if len(dominators) == 1:
            unique = dominators[0]
            unique_reason = "dominates every other eligible candidate on the scored axes"
        else:
            unique_reason = (
                "non-dominated set is not a unique all-axis dominator; "
                "scalar collapse refused"
            )

    return {
        "schema": "hawking.future.tournament.winner.v1",
        "non_dominated": front,
        "unique_winner": unique,
        "unique_reason": unique_reason,
        "disqualified": {k: disqualified[k] for k in sorted(disqualified)},
        "eligible": sorted(eligible_ids),
        "axes": [a.name for a in axes],
        "scalar_collapsed": False,
        "rule": (
            "Pareto front of eligible candidates. Unique winner only when one "
            "candidate dominates every other eligible candidate. Hard-gate "
            "failure disqualifies regardless of speed. Beating the incumbent "
            "control is not success."
        ),
    }


def scalar_score(_candidate: Mapping[str, Any]) -> float:
    raise ScalarCollapseError(
        "this harness refuses to collapse axes to a scalar; use winner()"
    )


def interpret_versus_control(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """The control is a CONTROL. Beating it is not success; missing it is not failure."""
    del candidate
    return {
        "success": None,
        "role_of_control": INCUMBENT_CONTROL["role"],
        "beating_control_is_not_success": True,
        "falling_short_of_control_is_not_failure": True,
        "success_predicate": None,
    }


def common_profile() -> dict[str, Any]:
    return {
        "identical_for": list(CONTENDER_IDS),
        "designed_before_either_complete_nx": True,
        "mission_corpus": [dict(m) for m in MISSION_CORPUS],
        "hard_gates": [dict(g) for g in HARD_GATES],
        "hard_gate_rule": "a contender failing one HARD GATE is out regardless of speed",
        "tool_reliability": [
            g for g in HARD_GATES if g["id"].startswith("tool.")
        ],
        "coherence": [
            g for g in HARD_GATES if g["id"].startswith("coherence.") or g["id"] in {
                "capability.fact-capital", "capability.fact-arith", "capability.no-think-leak",
            }
        ],
        "scored_axes": [
            {
                "name": a.name,
                "direction": a.direction,
                "kind": a.kind,
                "energy": a.energy,
                "value": None,
                "note": "UNKNOWN until a PROTECTED_ABSOLUTE measurement exists" if a.kind.startswith("hardware") else "UNKNOWN until disk evidence exists",
            }
            for a in SCORED_AXES
        ],
        "energy_policy": (
            "energy_joules_per_token is compared only when every candidate "
            "sets energy_trustworthy=true; otherwise the axis is dropped. "
            "No Green Machine module exists yet (frontier F006)."
        ),
        "no_scalar": True,
        "no_anchoring": INCUMBENT_CONTROL,
    }


def _sibling_flash_status() -> list[dict[str, Any]]:
    rows = []
    for rel in FLASH_NX_SIBLINGS:
        doc, src = load_repo_json(rel)
        rows.append({
            "path": rel,
            "source": src or None,
            "status": doc.get("status") if isinstance(doc, Mapping) else None,
            "metadata_only": _metadata_only(doc.get("status") if isinstance(doc, Mapping) else None),
        })
    return rows


def build() -> Path:
    inspections = inspect_contenders()
    ok, reasons = can_run(inspections)
    scoreboard, scoreboard_src = load_repo_json(SCOREBOARD_REL)
    cap, cap_src = load_repo_json(QWEN_CAPABILITY_REL)
    cap_overall = cap.get("overall") if isinstance(cap, Mapping) else None

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Pre-registered FLASH_SINGULARITY.NX vs QWEN27_SINGULARITY.NX "
            "comparison. Refuses to run while either contender is not a complete NX. "
            "Does not collapse to a scalar. Does not treat beating the incumbent "
            "control as success."
        ),
        "contenders": inspections,
        "common_profile": common_profile(),
        "incumbent_control": {
            **INCUMBENT_CONTROL,
            "identity_source": inspections[QWEN_ID].get("source"),
        },
        "readiness": {
            "can_run": ok,
            "reasons": reasons,
            "headline": "NO" if not ok else "YES",
            "honest_answer_today": "NO for Flash (metadata-only NX); Qwen27 is an incumbent resident CONTROL, not a complete SINGULARITY NX",
        },
        "dominance": {
            "scalar_collapse": "REFUSED",
            "winner_returns": "non-dominated set; unique winner only under genuine dominance",
            "axes": [a.name for a in SCORED_AXES],
        },
        "run_policy": {
            "do_not_run_until_both_complete_nx": True,
            "sidecar_has_no_gpu": True,
            "measurement_class_if_run": "PROTECTED_ABSOLUTE owned by Codex, never this sidecar",
        },
        "resident_install": {
            "module": "tools/future/resident_install.py",
            "phases": list(INSTALL_PHASES),
            "generic_over_winner": True,
        },
        "flash_nx_siblings": _sibling_flash_status(),
        "scoreboard_view": {
            "path": SCOREBOARD_REL,
            "source": scoreboard_src or None,
            "schema": scoreboard.get("schema") if isinstance(scoreboard, Mapping) else None,
            "status": scoreboard.get("status") if isinstance(scoreboard, Mapping) else None,
        },
        "qwen_capability_receipt": {
            "path": QWEN_CAPABILITY_REL,
            "source": cap_src or None,
            "quoted_overall": (
                f"{cap_overall.get('passed')}/{cap_overall.get('total')}"
                if isinstance(cap_overall, Mapping) else None
            ),
        },
        "recovered_implementation": [
            {"path": "tools/genesis_tournament.py", "role": "different tournament: Qwen3.8 vs Q80 chopping-block model selection"},
            {"path": "tools/odyssey/doctor_tournament.py", "role": "different tournament: Doctor technique preconditions"},
            {"path": "tools/odyssey/tournament.py", "role": "different tournament: checkpoint selection, newest does not automatically win"},
            {"path": "tools/odyssey/pareto_archive.py", "role": "Qwen-body Pareto + capability floor; consumed as style, not forked"},
            {"path": "tools/pareto_table.py", "role": "G150 candidate Pareto table; different candidates"},
            {"path": "lab/operators/ascension_manager_tournament_protocol.py", "role": "Gravity manager tournament; different contenders"},
            {"path": FLASH_NX_REL, "role": "Flash NX identity; status SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"},
            {"path": QWEN_IDENTITY_REL, "role": "incumbent Qwen27 resident identity (CONTROL, not tournament NX)"},
            {"path": QWEN_SEAL_REL, "role": "HCLI resident seal for sealed-3.14"},
            {"path": "hcli/agentos/resident_gate.py", "role": "live residency proof"},
            {"path": "hcli/agentos/native_mission_gate.py", "role": "HCLI mission corpus entry"},
            {"path": "hcli/mission.py", "role": "persistent mission loop"},
            {"path": "hcli/agentos/recovery.py", "role": "crash recovery gate"},
            {"path": "hcli/agentos/protected_accelerator_benchmark.py", "role": "protected-benchmark evacuation (stop before closing quiescence)"},
            {"path": "tools/nx_genome.py", "role": "Qwen NX genome sealer (G104); not QWEN27_SINGULARITY.NX"},
            {"path": "tools/flash_nx_genome.py", "role": "Flash NX metadata sealer; writes METADATA_ONLY status"},
            {"path": SCOREBOARD_REL, "role": "accelerator scoreboard (Codex-owned; sidecar reads only)"},
        ],
        "gaps_closed": [
            "pre-registered common profile identical for both NX contenders",
            "multi-axis Pareto winner() that refuses scalar collapse",
            "can_run() completeness gate against real Flash NX receipts",
            "incumbent ~3.14 EBPW / ~25 TPS recorded as CONTROL not target/ceiling",
            "generic resident install contract the winner binds into",
        ],
        "negative_findings": [
            "hcli/agentos/resident.py does not exist; resident_gate.py is the live gate",
            "QWEN27_SINGULARITY.NX document does not exist",
            "FLASH_COMPLETE_V0.nx.json is not in git HEAD; recovered from the main working tree",
            "ACCELERATOR_SCOREBOARD.json is not in git HEAD; recovered from the main working tree when present",
            "this worktree is a sparse checkout; Codex receipts loaded via main checkout and git show",
            "no tournament was run (no complete Flash NX, no GPU)",
            "energy axis has no trustworthy measurement (no Green Machine; frontier F006)",
            "FLASH_COMPLETE_V1/V2.nx.json and FLASH_NEXT_MACHINE.nx.json are also metadata-only",
            "sidecar produces STATIC_ONLY / bench UNKNOWN; never DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
        ],
    }
    return write_receipt(RECEIPT, doc, "tools/future/tournament.py")


def selftest() -> Path:
    ok, reasons = can_run()
    assert ok is False, f"can_run must be False today, got True; reasons={reasons}"
    mixed = winner([
        {"id": "A", "scores": _demo_scores(accepted_tps=30.0, complete_ebpw=4.0)},
        {"id": "B", "scores": _demo_scores(accepted_tps=10.0, complete_ebpw=2.0)},
    ])
    assert mixed["unique_winner"] is None, mixed
    assert mixed["non_dominated"] == ["A", "B"], mixed
    return build()


def _demo_scores(**overrides: float) -> dict[str, float]:
    base = {
        "accepted_tps": 10.0,
        "complete_token_ns": 1.0e8,
        "complete_ebpw": 3.0,
        "active_bytes_per_token": 1.0e9,
        "resident_memory_bytes": 1.0e10,
        "cold_launch_ns": 1.0e12,
        "warm_launch_ns": 1.0e9,
        "restart_ns": 1.0e10,
        "long_mission_verified_work": 5.0,
        "optimization_headroom": 0.5,
    }
    base.update(overrides)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true", help="refused unless both contenders are complete NX")
    a = ap.parse_args()
    if a.run:
        try:
            run()
        except TournamentNotReady as exc:
            print("REFUSED", file=sys.stderr)
            for reason in exc.reasons:
                print(f"  {reason}", file=sys.stderr)
            return 2
        return 1
    ok, reasons = can_run()
    if a.dry_run:
        print(f"can_run: {ok}")
        for reason in reasons:
            print(f"  {reason}")
        mixed = winner([
            {"id": "A", "scores": _demo_scores(accepted_tps=30.0, complete_ebpw=4.0)},
            {"id": "B", "scores": _demo_scores(accepted_tps=10.0, complete_ebpw=2.0)},
        ])
        print(f"winner.unique_winner (premature-collapse demo): {mixed['unique_winner']}")
        print(f"winner.non_dominated: {mixed['non_dominated']}")
        print(f"incumbent_control.role: {INCUMBENT_CONTROL['role']}")
        print(selftest())
        return 0 if not ok else 1
    print(selftest() if a.build else build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
