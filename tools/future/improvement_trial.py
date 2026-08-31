#!/usr/bin/env python3
"""HCLI_10M_IMPROVEMENT_TRIAL — a 10-minute trial that can fail.

A previous autonomy trial PASSED its written criteria while sitting idle for
477 seconds with runnable work remaining. A trial that cannot fail is not a
trial. This harness's load-bearing deliverable is the six negative controls
that must each produce FAIL. If any control PASSES, the harness is broken
and this module reports BROKEN_HARNESS rather than a green trial.

PASS is IMPROVED KNOWLEDGE OR IMPROVED EXECUTABLE. A decisive falsification
passes. A correct reclassification passes. A valid mutation passes. A TPS
increase is not required and is not scored. The bar is whether it knew what
to do next, conjunctively with: no unjustified runnable idle, no repeated
scar, no conversational wait, durable state written, no degeneracy.
Killing nothing AND launching nothing is FAIL. A run that meets every
nominal condition and is degenerate on any axis is FAIL.

The resident is given the live frontier and NO task sequence. Evidence class
is STATIC_ONLY. No GPU lease. Cited organ rates are copied as strings from
landed receipts; none are re-measured here.

    python3 tools/future/improvement_trial.py --record
    python3 tools/future/improvement_trial.py --selftest
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import copy
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    _assert_no_hardware_claims,
    git,
    load_json,
    write_receipt,
)
from tools.future import autonomy_degeneracy as ad
from tools.future import negative_index as ni
from tools.future import work_events as we
from tools.future import workunit_species as wus


RECEIPT = "IMPROVEMENT_TRIAL.json"
VELOCITY_RECEIPT = "IMPROVEMENT_VELOCITY.json"
SCHEMA = "hawking.future.improvement_trial.v1"
VELOCITY_SCHEMA = "hawking.future.improvement_velocity.v1"
VERSION = 1
RECORDED_BY = "tools/future/improvement_trial.py"
TRIAL_ID = "HCLI_10M_IMPROVEMENT_TRIAL"
WINDOW_S = 10 * 60
IDLE_GAP_S = 60
OPEN_HANDLE_REPRO_S = 477
LOW_PAYOFF_MS = 0.02
MULTI_MS = 1.0
NS_PER_S = 1_000_000_000

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Canonical work-event kinds plus trial-specific kinds the judge reads.
TRIAL_KINDS = frozenset(
    {
        "STATE_RECOVERED",
        "CAUSAL_BUDGET_INSPECTED",
        "SCAR_QUERIED",
        "OPTIONS_RANKED",
        "FALSIFIER_GENERATED",
        "BRANCH_KILLED",
        "OPTION_TREE_UPDATED",
        "NEXT_LEFT_RUNNING",
        "HANDLE_WAIT",
        "CONVERSATIONAL_WAIT",
        "CONCLUSION",
        "DURABLE_STATE_WRITTEN",
        "IDLE_JUSTIFIED",
    }
)
CANONICAL_KINDS = we.CANONICAL_KINDS

AWAITING_PHRASES = (
    "all tasks complete, awaiting instructions",
    "awaiting instructions",
    "awaiting instruction",
    "what would you like me to do",
    "how can i help",
    "conversational wait",
    "start the next mission if more work is required",
)

DEAD_R_BOTTLENECK_FAMILIES = (
    "FACTORIZE_THE_FACTORS",
    "DICTIONARY_PROGRAM",
    "PRODUCT_DICTIONARY",
    "CONDITIONAL_PROGRAM",
    "GENERATED_BLOCK",
    "NONLINEAR_GENERATOR",
)

LIVE_RECEIPT_RELS = (
    "receipts/future/MLP_ALU_ROOFLINE.json",
    "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
    "receipts/future/FRONTIER_STATE.json",
    "receipts/future/ODYSSEY_LAUNCH_GATE.json",
    "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    "receipts/future/MLP_REGION_FALSIFIER.json",
    "receipts/future/ORGAN_BANDWIDTH.json",
    "receipts/future/TPS_FALSIFICATIONS.json",
    "receipts/future/PATH_TO_71.json",
    "receipts/future/MLP_SHARED_PROGRAM.json",
    "receipts/future/MLP_CODE_INFORMATION.json",
    "receipts/future/WALL_GPU_RECONCILIATION.json",
    "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    "receipts/future/REPRESENTATION_DECODE_FUSION.json",
)

CONTROL_NAMES = (
    "duplicate_workunits",
    "dead_scar_repetition",
    "low_payoff_distraction",
    "open_handle_wait",
    "stale_causal_model",
    "misleading_narrow_probe",
)

REQUIRED_ACTS = (
    "recover_live_state",
    "inspect_causal_budget",
    "query_scars",
    "identify_top_options",
    "generate_cheapest_falsifiers",
    "launch_non_conflicting",
    "ingest_completed",
    "kill_unsupported",
    "update_option_tree",
    "refill_novel",
    "leave_next_running",
)

PASS_CONDITIONS = REQUIRED_ACTS + (
    "improved_knowledge_or_executable",
    "no_unjustified_runnable_idle",
    "no_repeated_scar",
    "no_conversational_wait",
    "durable_state_written",
    "killed_or_launched",
    "no_duplicate_workunits",
    "no_low_payoff_distraction",
    "no_open_handle_wait",
    "no_stale_causal_model",
    "no_misleading_narrow_probe",
    "no_degeneracy",
)

AUTO_FAIL_IDS = (
    "duplicate_workunits",
    "dead_scar_repetition",
    "low_payoff_distraction",
    "open_handle_wait",
    "stale_causal_model",
    "misleading_narrow_probe",
    "unjustified_runnable_idle",
    "conversational_wait",
    "repeated_scar",
    "killed_nothing_launched_nothing",
    "degeneracy",
)

KIND_TO_ACT = {
    "STATE_RECOVERED": "recover_live_state",
    "CAUSAL_BUDGET_INSPECTED": "inspect_causal_budget",
    "SCAR_QUERIED": "query_scars",
    "OPTIONS_RANKED": "identify_top_options",
    "FALSIFIER_GENERATED": "generate_cheapest_falsifiers",
    "WORK_LAUNCHED": "launch_non_conflicting",
    "RESULT_INGESTED": "ingest_completed",
    "BRANCH_KILLED": "kill_unsupported",
    "OPTION_TREE_UPDATED": "update_option_tree",
    "WORK_REFILLED": "refill_novel",
    "NEXT_LEFT_RUNNING": "leave_next_running",
    "DURABLE_STATE_WRITTEN": "durable_state_written",
}


# ---------------------------------------------------------------------------
# Metabolism protocol (x1 owns the real module). Narrowest local stand-in.
# ---------------------------------------------------------------------------

class LocalTerminalState:
    LIVE = "LIVE"
    RUNNING = "RUNNING"
    LAUNCHED = "LAUNCHED"
    KILLED_ORACLE = "KILLED_ORACLE"
    KILLED_EXPERIMENT = "KILLED_EXPERIMENT"
    KILLED_SCAR = "KILLED_SCAR"
    SUPERSEDED = "SUPERSEDED"
    CLOSED = "CLOSED"


class LocalWorkUnitRole:
    FALSIFIER = "FALSIFIER"
    MUTATION = "MUTATION"
    RECLASSIFICATION = "RECLASSIFICATION"
    ORACLE = "ORACLE"
    INGEST = "INGEST"
    REFILL = "REFILL"
    EXPERIMENT = "EXPERIMENT"


@dataclass
class LocalOption:
    id: str
    family: str
    organ: str = ""
    payoff_ms: float = 0.0
    cost: str = "ONE_EXPERIMENT"
    terminal: str = LocalTerminalState.LIVE
    falsifier: str = ""
    evidence_cites: list[str] = field(default_factory=list)
    mechanism: str = ""
    role: str = LocalWorkUnitRole.FALSIFIER
    budget_generation: int = 0
    superseded_by: list[str] = field(default_factory=list)
    conflict_group: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "organ": self.organ,
            "payoff_ms": self.payoff_ms,
            "cost": self.cost,
            "terminal": _term(self.terminal),
            "falsifier": self.falsifier,
            "evidence_cites": list(self.evidence_cites),
            "mechanism": self.mechanism,
            "role": _term(self.role),
            "budget_generation": self.budget_generation,
            "superseded_by": list(self.superseded_by),
            "conflict_group": self.conflict_group,
            "notes": self.notes,
        }


@dataclass
class LocalOptionTree:
    options: dict[str, LocalOption] = field(default_factory=dict)
    generation: int = 0

    def add(self, option: LocalOption) -> None:
        self.options[option.id] = option

    def get(self, oid: str) -> LocalOption | None:
        return self.options.get(oid)

    def live(self) -> list[LocalOption]:
        live_states = {
            LocalTerminalState.LIVE,
            LocalTerminalState.RUNNING,
            LocalTerminalState.LAUNCHED,
        }
        return [o for o in self.options.values() if _term(o.terminal) in live_states]

    def live_mass_ms(self) -> float:
        return round(sum(float(o.payoff_ms or 0.0) for o in self.live()), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "n_options": len(self.options),
            "n_live": len(self.live()),
            "live_mass_ms": self.live_mass_ms(),
            "options": {k: v.to_dict() for k, v in self.options.items()},
        }


def _try_import_metabolism() -> Any | None:
    try:
        return importlib.import_module("tools.future.improvement_metabolism")
    except ImportError:
        return None


_METABOLISM = _try_import_metabolism()
METABOLISM_LANDED = _METABOLISM is not None

if METABOLISM_LANDED:
    TerminalState = getattr(_METABOLISM, "TerminalState", LocalTerminalState)
    WorkUnitRole = getattr(_METABOLISM, "WorkUnitRole", LocalWorkUnitRole)
    Option = getattr(_METABOLISM, "Option", LocalOption)
    OptionTree = getattr(_METABOLISM, "OptionTree", LocalOptionTree)
else:
    TerminalState = LocalTerminalState
    WorkUnitRole = LocalWorkUnitRole
    Option = LocalOption
    OptionTree = LocalOptionTree


def metabolism_seam() -> dict[str, Any]:
    return {
        "module": "tools/future/improvement_metabolism.py",
        "owned_by": "x1",
        "landed": METABOLISM_LANDED,
        "protocol_used": "imported" if METABOLISM_LANDED else "local_narrowest_protocol",
        "local_protocol_names": [
            "Option",
            "OptionTree",
            "WorkUnitRole",
            "TerminalState",
        ],
        "seam": (
            "This lane consumes OptionTree + TerminalState + WorkUnitRole. "
            "It does not author the causal option tree, frontier objects, or "
            "terminal-state vocabulary. When x1's module is absent the local "
            "protocol is the operational stand-in; when it lands, types are "
            "imported and local names remain the fallback."
        ),
        "this_lane_does_not_write": "tools/future/improvement_metabolism.py",
    }


def _term(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return str(value.value)
    return str(value)


def _option(
    *,
    id: str,
    family: str,
    organ: str = "",
    payoff_ms: float = 0.0,
    cost: str = "ONE_EXPERIMENT",
    terminal: str = LocalTerminalState.LIVE,
    falsifier: str = "",
    evidence_cites: Sequence[str] = (),
    mechanism: str = "",
    role: str = LocalWorkUnitRole.FALSIFIER,
    budget_generation: int = 0,
    superseded_by: Sequence[str] = (),
    conflict_group: str = "",
    notes: str = "",
) -> LocalOption:
    """Always construct the local option. Metabolism types are recorded, not required."""
    return LocalOption(
        id=id,
        family=family,
        organ=organ,
        payoff_ms=float(payoff_ms or 0.0),
        cost=cost,
        terminal=_term(terminal) or LocalTerminalState.LIVE,
        falsifier=falsifier,
        evidence_cites=list(evidence_cites),
        mechanism=mechanism,
        role=_term(role) or LocalWorkUnitRole.FALSIFIER,
        budget_generation=int(budget_generation or 0),
        superseded_by=list(superseded_by),
        conflict_group=conflict_group or organ or id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Receipt loading — missing-on-disk is not absence
# ---------------------------------------------------------------------------

def load_receipt(rel: str) -> tuple[dict[str, Any] | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            doc = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, f"unreadable:{rel}:{type(exc).__name__}"
        if isinstance(doc, dict):
            return doc, f"disk:{rel}"
        return None, f"not_object:{rel}"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            return None, f"git_unreadable:HEAD:{rel}"
        if isinstance(doc, dict):
            return doc, f"git:HEAD:{rel}"
    return None, f"absent_in_this_checkout:{rel}"


def load_live_receipts() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rel in LIVE_RECEIPT_RELS:
        doc, taken = load_receipt(rel)
        key = Path(rel).stem.lower()
        out[key] = {
            "rel": rel,
            "path_taken": taken,
            "present": doc is not None,
            "doc": doc or {},
        }
    return out


def _cite(value: Any) -> str:
    """Stringify a landed number so this sidecar never asserts a hardware field."""
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Clock and event log
# ---------------------------------------------------------------------------

class SimulatedClock:
    """Monotonic trial clock. The 10-minute window is judged on this, not wall sleep."""

    def __init__(self, start_ns: int = 0) -> None:
        self._ns = int(start_ns)

    def now_ns(self) -> int:
        return self._ns

    def now_s(self) -> int:
        return int(self._ns // NS_PER_S)

    def advance_s(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot run backwards")
        self._ns += int(seconds * NS_PER_S)


def _unit_identity(unit: Mapping[str, Any] | None) -> str:
    if not unit:
        return ""
    for key in ("id", "unit_id"):
        token = str(unit.get(key) or "").strip()
        if token:
            return token
    family = str(unit.get("hypothesis_family") or unit.get("family") or "").strip()
    return family


def _payload_lookup(event: Mapping[str, Any], key: str) -> Any:
    payload = event.get("payload")
    if isinstance(payload, Mapping) and key in payload:
        return payload[key]
    return event.get(key)


class EventLog:
    def __init__(self, clock: SimulatedClock | None = None) -> None:
        self.clock = clock or SimulatedClock()
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        cites: Sequence[str] | None = None,
        advance_s: float = 1.0,
    ) -> dict[str, Any]:
        if advance_s:
            self.clock.advance_s(advance_s)
        body = dict(payload or {})
        event: dict[str, Any] = {
            "seq": len(self.events) + 1,
            "t_s": self.clock.now_s(),
            "t_ns": self.clock.now_ns(),
            "kind": kind,
            "payload": body,
        }
        if cites is not None:
            event["cites"] = [str(c) for c in cites]
        if kind in CANONICAL_KINDS:
            built = we.make(
                kind,
                payload={k: v for k, v in body.items() if k != "cites"},
                cites=event.get("cites"),
            )
            event["payload"] = built.get("payload", body)
            if "cites" in built:
                event["cites"] = built["cites"]
        elif kind not in TRIAL_KINDS:
            raise ValueError(f"unknown trial event kind {kind!r}")
        self.events.append(event)
        return event


# ---------------------------------------------------------------------------
# Scar table from landed receipts (do not ingest the whole sparse-missing corpus)
# ---------------------------------------------------------------------------

def _scar(
    *,
    scar_id: str,
    source_path: str,
    family: str,
    organ: str = "mlp",
    verdict: str = "MEASURED_NEGATIVE",
    mechanism: str = "",
    claim: str = "",
    reopen: str = "",
    level: str = "GENERAL_PHYSICAL",
) -> ni.Scar:
    family_canon = ni.canon_family(family)
    return ni.Scar(
        scar_id=scar_id,
        source_path=source_path,
        source_origin="receipt",
        parse_status=ni.PARSED,
        organ=organ,
        organs=[organ] if organ else [ni.UNRECORDED],
        hypothesis_family=family_canon,
        verdict=verdict,
        refuse_eligible=True,
        failure_mechanism=mechanism or ni.UNRECORDED,
        claim_refuted=claim or ni.UNRECORDED,
        reopen_condition=reopen or ni.UNRECORDED,
        level=level,
    ).finalize()


def scars_from_live(live: Mapping[str, Mapping[str, Any]]) -> list[ni.Scar]:
    scars: list[ni.Scar] = []
    nonlinear = (live.get("mlp_nonlinear_program") or {}).get("doc") or {}
    for row in nonlinear.get("family_verdicts") or []:
        family = str(row.get("family") or "")
        if not family:
            continue
        scars.append(
            _scar(
                scar_id=f"mlp_nonlinear.{family}",
                source_path="receipts/future/MLP_NONLINEAR_PROGRAM.json",
                family=family,
                organ="mlp",
                verdict=str(row.get("status") or "MEASURED_NEGATIVE"),
                mechanism=str(row.get("mechanism") or "r-bottleneck"),
                claim=f"{family} is an r-bottleneck and does not replace F",
                reopen="full-width structured nonlinear that is not an r-bottleneck",
            )
        )
    region = (live.get("mlp_region_falsifier") or {}).get("doc") or {}
    if region.get("verdict") == "GRANULARITY_REFUTED" or region:
        scars.append(
            _scar(
                scar_id="mlp_region_granularity",
                source_path="receipts/future/MLP_REGION_FALSIFIER.json",
                family="region_granularity",
                organ="mlp",
                verdict="GRANULARITY_REFUTED",
                mechanism="fused regions / packing did not move GB/s",
                claim="MLP ~350 GB/s is fragmentation / region granularity",
                reopen="a different mechanism; ALU roofline names arithmetic, not packing",
            )
        )
        scars.append(
            _scar(
                scar_id="mlp_fused_regions_identical_arithmetic",
                source_path="receipts/future/MLP_REGION_FALSIFIER.json",
                family="reach_demonstrated_bandwidth_mlp",
                organ="mlp",
                verdict="GRANULARITY_REFUTED",
                mechanism="falsifier was fused regions with identical arithmetic",
                claim="contiguous fused-region MLP reaches demonstrated 497 GB/s",
            )
        )
    budget = (live.get("resident_71tps_causal_budget") or {}).get("doc") or {}
    for lever in budget.get("refuted_levers") or []:
        lid = str(lever.get("id") or "")
        if not lid:
            continue
        scars.append(
            _scar(
                scar_id=f"causal_budget.{lid}",
                source_path="receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
                family=lid,
                organ="mlp" if "mlp" in lid or lid.startswith("entropy") else "",
                verdict=str(lever.get("verdict") or "REFUTED"),
                mechanism=str(lever.get("evidence") or ""),
                claim=lid,
            )
        )
    tps_f = (live.get("tps_falsifications") or {}).get("doc") or {}
    for row in tps_f.get("falsifications") or []:
        fid = str(row.get("id") or "")
        if not fid:
            continue
        scars.append(
            _scar(
                scar_id=fid,
                source_path="receipts/future/TPS_FALSIFICATIONS.json",
                family=fid,
                organ="",
                verdict=str(row.get("verdict") or "FALSIFIED"),
                mechanism=str(row.get("evidence") or ""),
                claim=str(row.get("hypothesis") or fid),
            )
        )
    alu = (live.get("mlp_alu_roofline") or {}).get("doc") or {}
    for fam in alu.get("refuted_elsewhere") or []:
        scars.append(
            _scar(
                scar_id=f"alu_roofline.refuted.{fam}",
                source_path="receipts/future/MLP_ALU_ROOFLINE.json",
                family=str(fam),
                organ="mlp",
                verdict="REFUTED_ELSEWHERE",
                claim=str(fam),
            )
        )
    organ_bw = (live.get("organ_bandwidth") or {}).get("doc") or {}
    for finding in organ_bw.get("findings") or []:
        fid = str(finding.get("id") or "")
        if fid == "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE":
            scars.append(
                _scar(
                    scar_id=fid,
                    source_path="receipts/future/ORGAN_BANDWIDTH.json",
                    family="dispatch_count",
                    organ="",
                    verdict="REFUTED",
                    mechanism=str(finding.get("what") or ""),
                    claim="dispatch count sets organ GB/s",
                )
            )
    shared = (live.get("mlp_shared_program") or {}).get("doc") or {}
    if int(shared.get("n_survivors") or 0) == 0 and shared.get("candidates"):
        scars.append(
            _scar(
                scar_id="mlp_shared_program.all_17",
                source_path="receipts/future/MLP_SHARED_PROGRAM.json",
                family="shared_program",
                organ="mlp",
                verdict="MEASURED_NEGATIVE",
                mechanism="all shared-program candidates fail held-out",
                claim="a shared program replaces F at affordable rank",
                reopen="not these 17; function replacement remains an economics path",
            )
        )
    return scars


# ---------------------------------------------------------------------------
# Option tree from live frontier
# ---------------------------------------------------------------------------

def _budget_option(row: Mapping[str, Any], generation: int) -> LocalOption:
    oid = str(row.get("id") or "")
    status = str(row.get("status") or "OPEN").upper()
    terminal = {
        "RUNNING": LocalTerminalState.RUNNING,
        "QUEUED": LocalTerminalState.LIVE,
        "OPEN": LocalTerminalState.LIVE,
        "CLOSED": LocalTerminalState.CLOSED,
    }.get(status, LocalTerminalState.LIVE)
    organ = str(row.get("organ") or "")
    return _option(
        id=oid,
        family=oid,
        organ=organ,
        payoff_ms=float(row.get("ms_saved") or 0.0),
        cost=str(row.get("cost") or "ONE_EXPERIMENT"),
        terminal=terminal,
        falsifier=str(row.get("falsifier") or ""),
        evidence_cites=["receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"],
        mechanism=str(row.get("falsifier") or ""),
        budget_generation=generation,
        conflict_group=organ or oid,
        notes=str(row.get("why") or row.get("capability") or ""),
    )


def build_option_tree(live: Mapping[str, Mapping[str, Any]]) -> LocalOptionTree:
    tree = LocalOptionTree(generation=1)
    budget = (live.get("resident_71tps_causal_budget") or {}).get("doc") or {}
    for row in budget.get("experiments_ranked_by_gain") or []:
        if row.get("id"):
            tree.add(_budget_option(row, generation=1))

    nonlinear = (live.get("mlp_nonlinear_program") or {}).get("doc") or {}
    for family in nonlinear.get("families") or list(DEAD_R_BOTTLENECK_FAMILIES):
        tree.add(
            _option(
                id=f"mlp_r_bottleneck.{family}",
                family=family,
                organ="mlp",
                payoff_ms=0.0,
                cost="ALREADY_RUN",
                terminal=LocalTerminalState.LIVE,
                falsifier="held-out relative L2 on the sealed teacher corpus",
                evidence_cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
                mechanism="r-bottleneck",
                conflict_group="mlp_function_replacement",
                notes="landed MEASURED_NEGATIVE; trial must close it on the tree",
            )
        )

    alu = (live.get("mlp_alu_roofline") or {}).get("doc") or {}
    mlp = alu.get("mlp") or {}
    decode = (mlp.get("decode_tax") or {}).get("to_match_lm_head_497") or {}
    mlp_ms = 0.0
    existing = tree.get("reach_demonstrated_bandwidth_mlp")
    if existing:
        mlp_ms = float(existing.payoff_ms)
        existing.superseded_by.append("receipts/future/MLP_REGION_FALSIFIER.json")
        existing.superseded_by.append("receipts/future/MLP_ALU_ROOFLINE.json")
    tree.add(
        _option(
            id="mlp_decode_fma_cheapening",
            family="mlp_decode_fma_cheapening",
            organ="mlp",
            payoff_ms=mlp_ms,
            cost="ONE_EXPERIMENT",
            terminal=LocalTerminalState.LIVE,
            falsifier=(
                "STATIC plan of a cheaper decode (cited target decode-FMA per "
                "weight-byte 0.8835 vs production 1.3333). No GPU lease."
            ),
            evidence_cites=["receipts/future/MLP_ALU_ROOFLINE.json"],
            mechanism="arithmetic / decode FMA per weight-byte",
            conflict_group="mlp",
            notes=(
                f"cited production GB/s={_cite((mlp.get('production') or {}).get('effective_gb_s'))}; "
                f"cited stripped GB/s={_cite((mlp.get('arm_a_stripped') or {}).get('effective_gb_s'))}; "
                f"cited target decode-FMA/byte={_cite(decode.get('target_decode_fma_per_weight_byte'))}; "
                f"cited verdict={_cite(alu.get('verdict'))}"
            ),
        )
    )

    organ_bw = (live.get("organ_bandwidth") or {}).get("doc") or {}
    dn_organ = next(
        (o for o in (organ_bw.get("organs") or []) if o.get("organ") == "deltanet"),
        {},
    )
    dn_iso = ((alu.get("deltanet") or {}).get("production") or {})
    tree.add(
        _option(
            id="deltanet_organ_vs_isolated_kernel",
            family="deltanet_organ_vs_isolated_kernel",
            organ="deltanet",
            payoff_ms=float((tree.get("reach_demonstrated_bandwidth_deltanet") or _option(id="x", family="x")).payoff_ms or 0.0),
            cost="ONE_EXPERIMENT",
            terminal=LocalTerminalState.LIVE,
            falsifier=(
                "STATIC reconciliation of organ-trace GB/s against the isolated "
                "largest kernel. The isolated kernel is not the organ cost."
            ),
            evidence_cites=[
                "receipts/future/ORGAN_BANDWIDTH.json",
                "receipts/future/MLP_ALU_ROOFLINE.json",
            ],
            mechanism="unexplained organ remainder after the big kernel",
            conflict_group="deltanet",
            notes=(
                f"cited organ GB/s={_cite(dn_organ.get('effective_gb_s'))}; "
                f"cited isolated kernel GB/s={_cite(dn_iso.get('effective_gb_s'))}; "
                f"cited kernel={_cite((alu.get('deltanet') or {}).get('kernel'))}"
            ),
        )
    )
    tree.add(
        _option(
            id="deltanet_big_kernel_is_organ_cost",
            family="deltanet_big_kernel_is_organ_cost",
            organ="deltanet",
            payoff_ms=0.0,
            cost="ALREADY_MEASURED",
            terminal=LocalTerminalState.LIVE,
            falsifier="compare organ-trace 360 GB/s to isolated kernel 600.9 GB/s on the same receipt pair",
            evidence_cites=[
                "receipts/future/ORGAN_BANDWIDTH.json",
                "receipts/future/MLP_ALU_ROOFLINE.json",
            ],
            mechanism="largest kernel as the organ's cost",
            conflict_group="deltanet",
        )
    )
    tree.add(
        _option(
            id="dispatch_count_explains_organ_gap",
            family="dispatch_count",
            organ="",
            payoff_ms=0.0,
            cost="ALREADY_MEASURED",
            terminal=LocalTerminalState.LIVE,
            falsifier="least-squares t = a*bytes + b*dispatches over the four organs",
            evidence_cites=["receipts/future/ORGAN_BANDWIDTH.json"],
            mechanism="dispatch count",
            conflict_group="dispatch",
        )
    )

    gate = (live.get("odyssey_launch_gate") or {}).get("doc") or {}
    verdict = gate.get("verdict") or {}
    for unmet in verdict.get("unmet") or []:
        tree.add(
            _option(
                id=f"odyssey_unmet.{unmet}",
                family=str(unmet),
                organ="",
                payoff_ms=0.0,
                cost="CPU_STATIC",
                terminal=LocalTerminalState.LIVE,
                falsifier=f"close Odyssey I criterion {unmet}",
                evidence_cites=["receipts/future/ODYSSEY_LAUNCH_GATE.json"],
                mechanism="launch-gate criterion",
                role=LocalWorkUnitRole.REFILL,
                conflict_group="odyssey",
            )
        )
    return tree


def rank_options(tree: LocalOptionTree) -> list[LocalOption]:
    live = tree.live()
    live.sort(key=lambda o: (-float(o.payoff_ms or 0.0), o.id))
    return live


def cheapest_falsifiers(tree: LocalOptionTree, n: int = 4) -> list[LocalOption]:
    """Highest payoff first among LIVE options whose cost is one experiment or static."""
    cheap = [
        o
        for o in rank_options(tree)
        if _term(o.terminal) in {LocalTerminalState.LIVE, LocalTerminalState.RUNNING}
        and o.cost in {"ONE_EXPERIMENT", "CPU_STATIC", "ONE_FIT_PLUS_A_CAPABILITY_SCREEN"}
    ]
    return cheap[:n]


# ---------------------------------------------------------------------------
# Resident state machine
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    events: list[dict[str, Any]]
    tree: LocalOptionTree
    tree_before: dict[str, Any]
    killed: list[dict[str, Any]] = field(default_factory=list)
    launched: list[dict[str, Any]] = field(default_factory=list)
    refilled: list[str] = field(default_factory=list)
    next_running: list[str] = field(default_factory=list)
    ingested: list[str] = field(default_factory=list)
    scars_queried: list[str] = field(default_factory=list)
    experiments_avoided: list[dict[str, Any]] = field(default_factory=list)
    conclusions: list[dict[str, Any]] = field(default_factory=list)
    durable: dict[str, Any] | None = None
    window_s: int = WINDOW_S
    control: str | None = None
    live: bool = False
    receipts_loaded: list[dict[str, Any]] = field(default_factory=list)
    live_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> int:
        if not self.events:
            return 0
        return int(self.events[-1].get("t_s") or 0)


class Resident:
    def __init__(
        self,
        *,
        clock: SimulatedClock | None = None,
        tree: LocalOptionTree | None = None,
        scars: Sequence[ni.Scar] | None = None,
        window_s: int = WINDOW_S,
        control: str | None = None,
        live: bool = False,
    ) -> None:
        self.log = EventLog(clock or SimulatedClock())
        self.tree = tree or LocalOptionTree()
        self.scars = list(scars or [])
        self.window_s = window_s
        self.control = control
        self.live = live
        self.in_flight: dict[str, dict[str, Any]] = {}
        self.runnable: list[str] = []
        self.killed: list[dict[str, Any]] = []
        self.launched: list[dict[str, Any]] = []
        self.refilled: list[str] = []
        self.ingested: list[str] = []
        self.scars_queried: list[str] = []
        self.experiments_avoided: list[dict[str, Any]] = []
        self.conclusions: list[dict[str, Any]] = []
        self.durable: dict[str, Any] | None = None
        self.tree_before: dict[str, Any] = self.tree.to_dict()
        self.receipts_loaded: list[dict[str, Any]] = []
        self.live_summary: dict[str, Any] = {}
        self.budget_generation = 1

    def snapshot_before(self) -> None:
        self.tree_before = copy.deepcopy(self.tree.to_dict())

    def recover(self, live: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        loaded = []
        if live:
            for item in live.values():
                loaded.append(
                    {
                        "rel": item.get("rel"),
                        "path_taken": item.get("path_taken"),
                        "present": bool(item.get("present")),
                    }
                )
        self.receipts_loaded = loaded
        self.live_summary = live_state_summary(live or {})
        self.runnable = [o.id for o in self.tree.live()]
        self.log.emit(
            "STATE_RECOVERED",
            {
                "n_receipts": len(loaded),
                "n_live_options": len(self.tree.live()),
                "is_idle": False,
                "task_sequence": None,
            },
            cites=[r["rel"] for r in loaded if r.get("rel")],
        )
        if self.runnable:
            self.log.emit(
                "FRONTIER_HAS_WORK",
                {"unit_ids": list(self.runnable)},
                advance_s=0.0,
            )

    def inspect_budget(self, budget: Mapping[str, Any] | None = None) -> None:
        experiments = list((budget or {}).get("experiments_ranked_by_gain") or [])
        self.log.emit(
            "CAUSAL_BUDGET_INSPECTED",
            {
                "n_experiments": len(experiments),
                "generation": self.budget_generation,
                "ids": [str(e.get("id")) for e in experiments[:12]],
            },
            cites=["receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"],
        )

    def query_scars(self, families: Sequence[str] | None = None) -> None:
        targets = list(families or [o.family for o in self.tree.options.values()])
        hits: list[str] = []
        for family in targets:
            refusal = ni.refuse_if_dead(
                {"hypothesis_family": family, "organ": "mlp"},
                scars=self.scars,
            )
            self.scars_queried.append(family)
            if refusal:
                hits.append(str(refusal.get("scar_id") or family))
                self.experiments_avoided.append(
                    {
                        "family": family,
                        "scar_id": refusal.get("scar_id"),
                        "source_path": refusal.get("source_path"),
                    }
                )
        self.log.emit(
            "SCAR_QUERIED",
            {
                "n_families": len(targets),
                "n_hits": len(hits),
                "hits": hits[:24],
                "n_avoided": len(self.experiments_avoided),
            },
            cites=sorted(
                {str(s.source_path) for s in self.scars if getattr(s, "source_path", None)}
            ),
        )

    def rank(self) -> list[LocalOption]:
        ranked = rank_options(self.tree)
        self.log.emit(
            "OPTIONS_RANKED",
            {
                "top": [
                    {"id": o.id, "payoff_ms": o.payoff_ms, "family": o.family}
                    for o in ranked[:8]
                ],
                "n_live": len(ranked),
                "live_mass_ms": self.tree.live_mass_ms(),
            },
        )
        return ranked

    def generate_falsifiers(self) -> list[LocalOption]:
        items = cheapest_falsifiers(self.tree)
        for opt in items:
            self.log.emit(
                "FALSIFIER_GENERATED",
                {
                    "id": opt.id,
                    "family": opt.family,
                    "falsifier": opt.falsifier,
                    "payoff_ms": opt.payoff_ms,
                    "cost": opt.cost,
                },
                cites=opt.evidence_cites,
                advance_s=0.5,
            )
        if not items:
            self.log.emit("FALSIFIER_GENERATED", {"id": None, "n": 0}, advance_s=0.5)
        return items

    def ingest(self, rel: str, *, what: str = "landed_receipt") -> None:
        self.ingested.append(rel)
        self.budget_generation += 1
        self.tree.generation = self.budget_generation
        self.log.emit(
            "RESULT_INGESTED",
            {"what": what, "receipt": rel},
            cites=[rel],
        )

    def kill(
        self,
        option_id: str,
        *,
        warrant: str,
        cites: Sequence[str],
        reason: str,
    ) -> LocalOption | None:
        opt = self.tree.get(option_id)
        if opt is None:
            return None
        terminal = {
            "oracle": LocalTerminalState.KILLED_ORACLE,
            "experiment": LocalTerminalState.KILLED_EXPERIMENT,
            "scar": LocalTerminalState.KILLED_SCAR,
            "superseded": LocalTerminalState.SUPERSEDED,
        }.get(warrant, LocalTerminalState.KILLED_ORACLE)
        opt.terminal = terminal
        row = {
            "id": opt.id,
            "family": opt.family,
            "warrant": warrant,
            "terminal": _term(terminal),
            "reason": reason,
            "cites": list(cites),
        }
        self.killed.append(row)
        if option_id in self.runnable:
            self.runnable = [x for x in self.runnable if x != option_id]
        self.log.emit("BRANCH_KILLED", row, cites=cites)
        return opt

    def _refuse_launch(self, opt: LocalOption) -> dict[str, Any] | None:
        return ni.refuse_if_dead(
            {
                "hypothesis_family": opt.family,
                "organ": opt.organ,
                "id": opt.id,
            },
            scars=self.scars,
        )

    def launch(
        self,
        opt: LocalOption,
        *,
        unit_id: str | None = None,
        ignore_scar: bool = False,
        ignore_stale: bool = False,
        ignore_payoff: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        refusal = None if ignore_scar else self._refuse_launch(opt)
        if refusal:
            self.experiments_avoided.append(
                {
                    "family": opt.family,
                    "scar_id": refusal.get("scar_id"),
                    "source_path": refusal.get("source_path"),
                    "would_have_launched": opt.id,
                }
            )
            return None
        if not ignore_stale and opt.superseded_by:
            ingested_super = [c for c in opt.superseded_by if c in self.ingested]
            if ingested_super:
                # Launching a superseded option is the stale-causal-model defect.
                # The caller who wants the defect sets ignore_stale=True.
                pass
        uid = unit_id or f"WU.IMPROVEMENT.{opt.id}"
        try:
            unit = wus.emit_hcli_workunit(
                id=uid,
                role="science",
                description=opt.falsifier or opt.notes or opt.id,
                dependencies=(),
                resource_class="STATIC_ANALYSIS",
                verifier="tools.future.improvement_trial.ingest",
                provider="future.improvement_trial",
                classification="STATIC_ONLY",
                extras={
                    "hypothesis_family": opt.family,
                    "payoff_ms": opt.payoff_ms,
                    "organ": opt.organ,
                    "conflict_group": opt.conflict_group,
                    "budget_generation": opt.budget_generation,
                    **dict(extra or {}),
                },
            )
        except Exception:
            unit = {
                "id": uid,
                "role": "science",
                "description": opt.falsifier or opt.id,
                "resource_class": "STATIC_ANALYSIS",
                "classification": "STATIC_ONLY",
                "hypothesis_family": opt.family,
                "payoff_ms": opt.payoff_ms,
            }
        opt.terminal = LocalTerminalState.LAUNCHED
        public = {
            "id": unit.get("id") or uid,
            "family": opt.family,
            "organ": opt.organ,
            "payoff_ms": opt.payoff_ms,
            "resource_class": unit.get("resource_class"),
            "classification": unit.get("classification"),
            "conflict_group": opt.conflict_group,
            "budget_generation": opt.budget_generation,
            "falsifier": opt.falsifier,
            "mechanism": opt.mechanism,
            "superseded_by": list(opt.superseded_by),
        }
        self.launched.append(public)
        self.in_flight[public["id"]] = public
        if opt.id not in self.runnable:
            self.runnable.append(opt.id)
        self.log.emit(
            "WORK_GENERATED",
            {"candidate": {"id": public["id"], "hypothesis_family": opt.family}},
            advance_s=0.0,
        )
        self.log.emit("WORK_SCHEDULED", {"unit": {"id": public["id"]}}, advance_s=0.0)
        self.log.emit("WORK_LAUNCHED", {"unit": public})
        return public

    def launch_nonconflicting(
        self,
        candidates: Sequence[LocalOption],
        *,
        ignore_scar: bool = False,
        ignore_stale: bool = False,
    ) -> list[dict[str, Any]]:
        taken: set[str] = {
            str(u.get("conflict_group") or "") for u in self.launched
        }
        out: list[dict[str, Any]] = []
        ranked = sorted(candidates, key=lambda o: -float(o.payoff_ms or 0.0))
        for opt in ranked:
            if _term(opt.terminal) not in {
                LocalTerminalState.LIVE,
                LocalTerminalState.RUNNING,
            }:
                continue
            group = opt.conflict_group or opt.organ or opt.id
            if group in taken:
                continue
            launched = self.launch(
                opt, ignore_scar=ignore_scar, ignore_stale=ignore_stale
            )
            if launched:
                taken.add(group)
                out.append(launched)
        return out

    def refill(self, unit_ids: Sequence[str]) -> None:
        ids = [str(x) for x in unit_ids if str(x).strip()]
        if not ids:
            return
        self.refilled.extend(ids)
        for uid in ids:
            if uid not in self.runnable:
                self.runnable.append(uid)
        self.log.emit(
            "WORK_REFILLED",
            {"unit_ids": ids, "queue_depth": len(self.runnable)},
        )

    def leave_running(self) -> list[str]:
        running = [
            o.id
            for o in self.tree.options.values()
            if _term(o.terminal)
            in {LocalTerminalState.LAUNCHED, LocalTerminalState.RUNNING}
        ]
        running.extend(self.in_flight.keys())
        # unique, stable
        seen: set[str] = set()
        ordered: list[str] = []
        for item in running:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        self.log.emit(
            "NEXT_LEFT_RUNNING",
            {"unit_ids": ordered, "n": len(ordered)},
        )
        return ordered

    def update_tree(self) -> None:
        self.log.emit(
            "OPTION_TREE_UPDATED",
            {
                "generation": self.tree.generation,
                "n_live": len(self.tree.live()),
                "live_mass_ms": self.tree.live_mass_ms(),
                "n_killed": len(self.killed),
                "n_launched": len(self.launched),
            },
        )

    def wait_on_handle(self, handle_id: str, seconds: float, runnable: Sequence[str]) -> None:
        self.log.emit(
            "HANDLE_WAIT",
            {
                "handle_id": handle_id,
                "wait_s": seconds,
                "runnable_unit_ids": list(runnable),
            },
            advance_s=seconds,
        )

    def conversational(self, text: str) -> None:
        self.log.emit("CONVERSATIONAL_WAIT", {"text": text})

    def conclude(self, claim: str, probe: str, scope: str, supported: bool) -> None:
        row = {
            "claim": claim,
            "probe": probe,
            "scope": scope,
            "supported_by_probe": supported,
        }
        self.conclusions.append(row)
        self.log.emit("CONCLUSION", row)

    def write_durable(self) -> dict[str, Any]:
        doc = {
            "killed": list(self.killed),
            "launched": list(self.launched),
            "refilled": list(self.refilled),
            "next_running": [
                o.id
                for o in self.tree.options.values()
                if _term(o.terminal)
                in {LocalTerminalState.LAUNCHED, LocalTerminalState.RUNNING}
            ],
            "option_tree": self.tree.to_dict(),
            "ingested": list(self.ingested),
        }
        self.durable = doc
        self.log.emit("DURABLE_STATE_WRITTEN", {"n_keys": len(doc)})
        return doc

    def record(self) -> TrialRecord:
        next_running = [
            o.id
            for o in self.tree.options.values()
            if _term(o.terminal)
            in {LocalTerminalState.LAUNCHED, LocalTerminalState.RUNNING}
        ]
        next_running.extend(self.in_flight.keys())
        seen: set[str] = set()
        ordered: list[str] = []
        for item in next_running:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return TrialRecord(
            events=list(self.log.events),
            tree=self.tree,
            tree_before=self.tree_before,
            killed=list(self.killed),
            launched=list(self.launched),
            refilled=list(self.refilled),
            next_running=ordered,
            ingested=list(self.ingested),
            scars_queried=list(self.scars_queried),
            experiments_avoided=list(self.experiments_avoided),
            conclusions=list(self.conclusions),
            durable=self.durable,
            window_s=self.window_s,
            control=self.control,
            live=self.live,
            receipts_loaded=list(self.receipts_loaded),
            live_summary=dict(self.live_summary),
        )


def live_state_summary(live: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Cite landed numbers as strings. Do not re-measure."""
    alu = (live.get("mlp_alu_roofline") or {}).get("doc") or {}
    mlp = alu.get("mlp") or {}
    decode = (mlp.get("decode_tax") or {}).get("inner_loop") or {}
    target = (mlp.get("decode_tax") or {}).get("to_match_lm_head_497") or {}
    dn = alu.get("deltanet") or {}
    organ_bw = (live.get("organ_bandwidth") or {}).get("doc") or {}
    dn_organ = next(
        (o for o in (organ_bw.get("organs") or []) if o.get("organ") == "deltanet"),
        {},
    )
    gate = (live.get("odyssey_launch_gate") or {}).get("doc") or {}
    gv = gate.get("verdict") or {}
    nonlinear = (live.get("mlp_nonlinear_program") or {}).get("doc") or {}
    frontier = (live.get("frontier_state") or {}).get("doc") or {}
    return {
        "mlp_arithmetic_lever": {
            "source": "receipts/future/MLP_ALU_ROOFLINE.json",
            "cited_production_gb_s": _cite((mlp.get("production") or {}).get("effective_gb_s")),
            "cited_stripped_gb_s": _cite((mlp.get("arm_a_stripped") or {}).get("effective_gb_s")),
            "cited_decode_fma_per_weight_byte": _cite(decode.get("decode_fma_per_weight_byte")),
            "cited_target_decode_fma_per_weight_byte": _cite(
                target.get("target_decode_fma_per_weight_byte")
            ),
            "cited_verdict": _cite(alu.get("verdict") or mlp.get("verdict")),
            "do_not_promote_to_alu_bound": True,
        },
        "deltanet_unexplained_cost": {
            "organ_source": "receipts/future/ORGAN_BANDWIDTH.json",
            "kernel_source": "receipts/future/MLP_ALU_ROOFLINE.json",
            "cited_organ_gb_s": _cite(dn_organ.get("effective_gb_s")),
            "cited_isolated_kernel_gb_s": _cite((dn.get("production") or {}).get("effective_gb_s")),
            "cited_kernel": _cite(dn.get("kernel")),
            "cited_verdict": _cite(dn.get("verdict") or alu.get("verdict")),
        },
        "mlp_r_bottleneck_families": {
            "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
            "families": list(nonlinear.get("families") or list(DEAD_R_BOTTLENECK_FAMILIES)),
            "status": "MEASURED_NEGATIVE",
            "n": len(nonlinear.get("families") or DEAD_R_BOTTLENECK_FAMILIES),
        },
        "odyssey_gate": {
            "source": "receipts/future/ODYSSEY_LAUNCH_GATE.json",
            "n_met": gv.get("n_met"),
            "n_criteria": gv.get("n_criteria"),
            "n_unmet": gv.get("n_unmet"),
            "unmet": list(gv.get("unmet") or []),
            "allowed": gv.get("allowed"),
            "verdict": gv.get("verdict"),
        },
        "frontier": {
            "source": "receipts/future/FRONTIER_STATE.json",
            "n_frontiers": frontier.get("n_frontiers"),
            "n_next_work": frontier.get("n_next_work"),
            "n_sleeping": frontier.get("n_sleeping"),
            "is_idle": frontier.get("is_idle"),
        },
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "numbers_are_citations_not_measurements": True,
    }


# ---------------------------------------------------------------------------
# Judge — ignores self-report; TPS increase is not a condition
# ---------------------------------------------------------------------------

def _kinds(events: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(e.get("kind") or "") for e in events}


def _text_of(event: Mapping[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    bits = [str(event.get("kind") or "")]
    bits.extend(str(v) for v in (payload or {}).values() if isinstance(v, (str, int)))
    for key in ("text", "detail", "reason", "message"):
        if event.get(key):
            bits.append(str(event[key]))
    return " ".join(bits).lower()


def _met(cid: str, detail: str, cites: Sequence[str] | None = None) -> dict[str, Any]:
    return {"id": cid, "met": True, "detail": detail, "cites": list(cites or [])}


def _unmet(cid: str, detail: str, cites: Sequence[str] | None = None) -> dict[str, Any]:
    return {"id": cid, "met": False, "detail": detail, "cites": list(cites or [])}


def _launched_units(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "WORK_LAUNCHED":
            continue
        unit = _payload_lookup(event, "unit")
        if isinstance(unit, Mapping):
            out.append(dict(unit))
    return out


def eval_required_act(record: TrialRecord, act: str) -> dict[str, Any]:
    kinds = _kinds(record.events)
    want = [k for k, a in KIND_TO_ACT.items() if a == act]
    if act == "durable_state_written":
        if record.durable:
            return _met(act, "durable option tree snapshot present")
        if "DURABLE_STATE_WRITTEN" in kinds:
            return _met(act, "DURABLE_STATE_WRITTEN on the log")
        return _unmet(act, "no durable state was written")
    if act == "launch_non_conflicting":
        # The act is that the resident ran the launch step. Zero launches is
        # scored by killed_or_launched, not by hiding the step.
        if "WORK_LAUNCHED" in kinds:
            return _met(act, "at least one WORK_LAUNCHED")
        if "FALSIFIER_GENERATED" in kinds and "OPTIONS_RANKED" in kinds:
            return _met(act, "launch step ran; nothing eligible (scored by killed_or_launched)")
        return _unmet(act, "no launch step")
    if act == "kill_unsupported":
        if "BRANCH_KILLED" in kinds:
            return _met(act, f"killed {len(record.killed)} branch(es)")
        return _unmet(act, "no BRANCH_KILLED; evidence-warranted kill was required")
    if any(k in kinds for k in want):
        return _met(act, f"saw {want[0] if want else act}")
    return _unmet(act, f"missing act; expected one of {want}")


def eval_improved_knowledge(record: TrialRecord) -> dict[str, Any]:
    kills = [e for e in record.events if e.get("kind") == "BRANCH_KILLED"]
    reclass = [
        k
        for k in record.killed
        if k.get("warrant") in {"oracle", "superseded", "scar", "experiment"}
    ]
    mutations = [
        u
        for u in record.launched
        if str(u.get("role") or "") == LocalWorkUnitRole.MUTATION
        or "mutation" in str(u.get("family") or "").lower()
        or "decode_fma" in str(u.get("family") or "")
        or "cheapening" in str(u.get("family") or "")
    ]
    if kills or reclass:
        return _met(
            "improved_knowledge_or_executable",
            f"kills={len(kills)} reclassifications={len(reclass)} mutations={len(mutations)}",
            [c for k in record.killed for c in (k.get("cites") or [])][:8],
        )
    if mutations:
        return _met(
            "improved_knowledge_or_executable",
            f"valid mutation launched: {[m.get('id') for m in mutations]}",
        )
    return _unmet(
        "improved_knowledge_or_executable",
        "no falsification, reclassification, or valid mutation",
    )


def eval_killed_or_launched(record: TrialRecord) -> dict[str, Any]:
    if record.killed or record.launched:
        return _met(
            "killed_or_launched",
            f"killed={len(record.killed)} launched={len(record.launched)}",
        )
    return _unmet(
        "killed_or_launched",
        "killed nothing and launched nothing",
    )


def eval_no_duplicate_workunits(record: TrialRecord) -> dict[str, Any]:
    launched = _launched_units(record.events)
    seen: dict[str, int] = {}
    new_info_after: dict[str, bool] = {}
    ingested_seq = {
        int(e.get("seq") or 0)
        for e in record.events
        if e.get("kind") == "RESULT_INGESTED"
    }
    last_ingest = 0
    dups: list[str] = []
    for event in record.events:
        seq = int(event.get("seq") or 0)
        if event.get("kind") == "RESULT_INGESTED":
            last_ingest = seq
            continue
        if event.get("kind") != "WORK_LAUNCHED":
            continue
        unit = _payload_lookup(event, "unit") or {}
        ident = _unit_identity(unit if isinstance(unit, Mapping) else {})
        if not ident:
            continue
        if ident in seen:
            # duplicate unless a new ingest landed between launches
            if last_ingest <= seen[ident]:
                dups.append(ident)
                new_info_after[ident] = False
        seen[ident] = seq
    if dups:
        return _unmet(
            "no_duplicate_workunits",
            f"same unit relaunched with no new information: {dups}",
            dups,
        )
    return _met("no_duplicate_workunits", f"unique launches={len(seen)}")


def eval_no_repeated_scar(record: TrialRecord) -> dict[str, Any]:
    """Re-running a family the negative index closed is FAIL."""
    refused_families = {
        ni.canon_family(str(row.get("family") or ""))
        for row in record.experiments_avoided
        if row.get("family")
    }
    # Also: any scar-eligible family that was launched.
    repeats: list[str] = []
    launched_families: list[str] = []
    for unit in record.launched:
        family = str(unit.get("family") or unit.get("hypothesis_family") or "")
        launched_families.append(family)
        proposal = {
            "hypothesis_family": family,
            "organ": unit.get("organ") or "mlp",
        }
        # Reconstruct scars from the record's queried hits via experiments_avoided
        # plus any BRANCH_KILLED warrant=scar of the same family.
    killed_scar_families = {
        ni.canon_family(str(k.get("family") or ""))
        for k in record.killed
        if k.get("warrant") == "scar"
    }
    for unit in record.launched:
        family = str(unit.get("family") or unit.get("hypothesis_family") or "")
        canon = ni.canon_family(family)
        if canon in killed_scar_families or canon in refused_families:
            repeats.append(family)
        # Also: launch of a named r-bottleneck family
        if canon in {ni.canon_family(f) for f in DEAD_R_BOTTLENECK_FAMILIES}:
            if family not in repeats:
                repeats.append(family)
    if repeats:
        return _unmet(
            "no_repeated_scar",
            f"relaunched scar-dead families: {repeats}",
            repeats,
        )
    return _met("no_repeated_scar", "no scar-dead family was launched")


def eval_no_low_payoff_distraction(record: TrialRecord) -> dict[str, Any]:
    launched = record.launched
    if not launched:
        return _met("no_low_payoff_distraction", "nothing launched")
    live_before = (record.tree_before.get("options") or {})
    available_ms: list[tuple[str, float]] = []
    launched_ids = {str(u.get("id") or "") for u in launched}
    launched_option_ids = set()
    for uid in launched_ids:
        launched_option_ids.add(uid)
        if uid.startswith("WU.IMPROVEMENT."):
            launched_option_ids.add(uid[len("WU.IMPROVEMENT.") :])
    for oid, opt in live_before.items():
        term = _term((opt or {}).get("terminal"))
        if term not in {
            LocalTerminalState.LIVE,
            LocalTerminalState.RUNNING,
            LocalTerminalState.LAUNCHED,
        }:
            continue
        available_ms.append((oid, float((opt or {}).get("payoff_ms") or 0.0)))
    # also current tree live at start of launches
    for opt in record.tree.options.values():
        available_ms.append((opt.id, float(opt.payoff_ms or 0.0)))
    multi = [(i, m) for i, m in available_ms if m >= MULTI_MS]
    worked_low = [
        u
        for u in launched
        if float(u.get("payoff_ms") or 0.0) <= LOW_PAYOFF_MS + 1e-9
    ]
    unlaunched_multi = [
        (i, m)
        for i, m in multi
        if i not in launched_option_ids
        and f"WU.IMPROVEMENT.{i}" not in launched_ids
    ]
    if worked_low and unlaunched_multi:
        return _unmet(
            "no_low_payoff_distraction",
            (
                f"worked payoff_ms={[u.get('payoff_ms') for u in worked_low]} "
                f"while multi-ms available={unlaunched_multi[:4]}"
            ),
        )
    return _met("no_low_payoff_distraction", "highest available payoff class was worked")


def eval_no_open_handle_wait(record: TrialRecord) -> dict[str, Any]:
    events = list(record.events)
    window = int(record.window_s or WINDOW_S)
    waits: list[str] = []
    for event in events:
        if event.get("kind") != "HANDLE_WAIT":
            continue
        payload = event.get("payload") or {}
        wait_s = float(payload.get("wait_s") or 0.0)
        runnable = list(payload.get("runnable_unit_ids") or [])
        handle = str(payload.get("handle_id") or "")
        if runnable and wait_s >= OPEN_HANDLE_REPRO_S:
            waits.append(f"{handle}:{wait_s}s with runnable={runnable[:6]}")
        elif runnable and wait_s >= 0.5 * window:
            waits.append(f"{handle}:{wait_s}s (majority of {window}s) runnable={runnable[:6]}")
    # Silent gap while a handle is in flight and leftover work exists.
    in_flight: set[str] = set()
    leftover_named = False
    for prev, nxt in zip(events, events[1:]):
        if prev.get("kind") == "WORK_LAUNCHED":
            unit = _payload_lookup(prev, "unit") or {}
            ident = _unit_identity(unit if isinstance(unit, Mapping) else {})
            if ident:
                in_flight.add(ident)
        if prev.get("kind") == "FRONTIER_HAS_WORK":
            leftover_named = True
        if nxt.get("kind") in {"RESULT_INGESTED"}:
            pass
        t0 = int(prev.get("t_s") or 0)
        t1 = int(nxt.get("t_s") or 0)
        gap = t1 - t0
        if gap >= OPEN_HANDLE_REPRO_S and in_flight:
            has_other = leftover_named or any(
                e.get("kind") == "FRONTIER_HAS_WORK" for e in events
            )
            if has_other or len(record.tree.live()) > 1:
                waits.append(
                    f"silent {t0}->{t1}s ({gap}s) with in-flight={sorted(in_flight)[:4]}"
                )
    if waits:
        return _unmet("no_open_handle_wait", "; ".join(waits[:4]))
    return _met("no_open_handle_wait", "no majority-window handle block with other work runnable")


def eval_no_unjustified_idle(record: TrialRecord) -> dict[str, Any]:
    events = list(record.events)
    leftover_ids: list[str] = []
    for event in events:
        if event.get("kind") in {"FRONTIER_HAS_WORK", "OPTIONS_RANKED", "STATE_RECOVERED"}:
            payload = event.get("payload") or {}
            ids = payload.get("unit_ids") or [
                t.get("id") for t in (payload.get("top") or []) if isinstance(t, Mapping)
            ]
            leftover_ids = [str(x) for x in (ids or leftover_ids) if x]
    bad: list[str] = []
    for prev, nxt in zip(events, events[1:]):
        t0 = int(prev.get("t_s") or 0)
        t1 = int(nxt.get("t_s") or 0)
        gap = t1 - t0
        if gap < IDLE_GAP_S:
            continue
        kind0 = str(prev.get("kind") or "")
        if kind0 == "WORK_LAUNCHED":
            # performing that unit is not idle unless HANDLE_WAIT follows as the
            # only activity and other work is named. The open-handle evaluator
            # owns the 477s reproduction; here we still fail a silent sleep.
            continue
        if kind0 in {"IDLE_JUSTIFIED", "HANDLE_WAIT"}:
            if kind0 == "HANDLE_WAIT":
                payload = prev.get("payload") or {}
                if payload.get("runnable_unit_ids") and gap >= IDLE_GAP_S:
                    bad.append(f"{t0}->{t1}s ({gap}s) HANDLE_WAIT with leftover")
            continue
        if leftover_ids or record.tree.live():
            bad.append(f"{t0}->{t1}s ({gap}s) leftover={leftover_ids[:8]}")
    if bad:
        return _unmet("no_unjustified_runnable_idle", "; ".join(bad[:4]))
    return _met(
        "no_unjustified_runnable_idle",
        f"no unjustified gap >={IDLE_GAP_S}s while work remained",
    )


def eval_no_conversational_wait(record: TrialRecord) -> dict[str, Any]:
    hits: list[str] = []
    for event in record.events:
        if event.get("kind") in {"CONVERSATIONAL_WAIT", "awaiting_instructions"}:
            hits.append(f"seq:{event.get('seq')}")
            continue
        text = _text_of(event)
        if any(p in text for p in AWAITING_PHRASES):
            hits.append(f"seq:{event.get('seq')}:{text[:80]}")
    if hits:
        return _unmet("no_conversational_wait", f"conversational wait: {hits[:4]}")
    return _met("no_conversational_wait", "no awaiting-instructions phrasing")


def eval_no_stale_causal_model(record: TrialRecord) -> dict[str, Any]:
    ingested = set(record.ingested)
    for event in record.events:
        if event.get("kind") == "RESULT_INGESTED":
            cites = event.get("cites") or []
            ingested.update(str(c) for c in cites)
            rel = (_payload_lookup(event, "receipt") or "")
            if rel:
                ingested.add(str(rel))
    stale: list[str] = []
    for unit in record.launched:
        superseded = list(unit.get("superseded_by") or [])
        if not superseded:
            opt = record.tree.get(str(unit.get("family") or "")) or record.tree.get(
                str(unit.get("id") or "").removeprefix("WU.IMPROVEMENT.")
            )
            if opt:
                superseded = list(opt.superseded_by)
        hit = [c for c in superseded if c in ingested]
        if hit:
            stale.append(f"{unit.get('id')} after {hit}")
        # fused-regions MLP bandwidth after region falsifier ingested
        family = str(unit.get("family") or "")
        mech = str(unit.get("mechanism") or unit.get("falsifier") or "").lower()
        if family == "reach_demonstrated_bandwidth_mlp" or "fused" in mech or "identical arithmetic" in mech:
            if "receipts/future/MLP_REGION_FALSIFIER.json" in ingested:
                if f"{unit.get('id')} after" not in " ".join(stale):
                    stale.append(
                        f"{unit.get('id')} used fused-region falsifier after "
                        "MLP_REGION_FALSIFIER.json"
                    )
    if stale:
        return _unmet("no_stale_causal_model", "; ".join(stale[:4]))
    return _met("no_stale_causal_model", "no launch of a budget row superseded by ingested evidence")


def eval_no_degeneracy(record: TrialRecord) -> dict[str, Any]:
    """FAIL ON DEGENERACY EVEN WHEN EVERY CONDITION IS NOMINALLY MET."""
    try:
        report = ad.measure(record)
    except Exception as exc:  # noqa: BLE001 — fail closed
        return _unmet(
            "no_degeneracy",
            f"measure failed closed: {type(exc).__name__}: {exc}",
        )
    if report.get("verdict") == "FAIL":
        axes = list(report.get("degenerate_axes") or [])
        return _unmet(
            "no_degeneracy",
            f"degenerate on {axes}: {report.get('reason')}",
            axes,
        )
    n_axes = len(report.get("named_axes") or ())
    return _met(
        "no_degeneracy",
        f"all named axes non-degenerate (n_axes={n_axes})",
    )


def eval_no_misleading_narrow_probe(record: TrialRecord) -> dict[str, Any]:
    bad: list[str] = []
    for row in record.conclusions:
        claim = str(row.get("claim") or "").lower()
        probe = str(row.get("probe") or "").lower()
        scope = str(row.get("scope") or "").lower()
        supported = row.get("supported_by_probe")
        if supported is False:
            bad.append(f"claim={row.get('claim')!r} from probe={row.get('probe')!r}")
            continue
        broad = (
            "all organs" in scope
            or "alu_bound" in claim
            or "every organ" in claim
            or scope in {"all_organs", "whole_model", "global"}
        )
        narrow = (
            "one layer" in probe
            or "mixed" in probe
            or "single" in probe
            or "one organ" in probe
        )
        if broad and narrow:
            bad.append(
                f"broad claim {row.get('claim')!r} from narrow probe {row.get('probe')!r}"
            )
    # Also scan CONCLUSION events if conclusions list was not filled
    for event in record.events:
        if event.get("kind") != "CONCLUSION":
            continue
        payload = event.get("payload") or {}
        if payload.get("supported_by_probe") is False:
            desc = f"claim={payload.get('claim')!r} from probe={payload.get('probe')!r}"
            if desc not in bad:
                bad.append(desc)
    if bad:
        return _unmet("no_misleading_narrow_probe", "; ".join(bad[:4]))
    return _met("no_misleading_narrow_probe", "no over-claim from a narrower probe")


EVALUATORS: dict[str, Callable[[TrialRecord], dict[str, Any]]] = {
    "improved_knowledge_or_executable": eval_improved_knowledge,
    "killed_or_launched": eval_killed_or_launched,
    "no_duplicate_workunits": eval_no_duplicate_workunits,
    "no_repeated_scar": eval_no_repeated_scar,
    "no_low_payoff_distraction": eval_no_low_payoff_distraction,
    "no_open_handle_wait": eval_no_open_handle_wait,
    "no_unjustified_runnable_idle": eval_no_unjustified_idle,
    "no_conversational_wait": eval_no_conversational_wait,
    "no_stale_causal_model": eval_no_stale_causal_model,
    "no_misleading_narrow_probe": eval_no_misleading_narrow_probe,
    "no_degeneracy": eval_no_degeneracy,
}


def judge(record: TrialRecord) -> dict[str, Any]:
    """Mechanical conditions only. Self-report on the log is ignored. TPS is not scored."""
    conditions: list[dict[str, Any]] = []
    for act in REQUIRED_ACTS:
        conditions.append(eval_required_act(record, act))
    conditions.append(eval_required_act(record, "durable_state_written"))
    for cid in (
        "improved_knowledge_or_executable",
        "killed_or_launched",
        "no_duplicate_workunits",
        "no_repeated_scar",
        "no_low_payoff_distraction",
        "no_open_handle_wait",
        "no_unjustified_runnable_idle",
        "no_conversational_wait",
        "no_stale_causal_model",
        "no_misleading_narrow_probe",
        "no_degeneracy",
    ):
        conditions.append(EVALUATORS[cid](record))

    unmet = [c for c in conditions if not c["met"]]
    auto: list[dict[str, Any]] = []
    mapping = {
        "no_duplicate_workunits": "duplicate_workunits",
        "no_repeated_scar": "dead_scar_repetition",
        "no_low_payoff_distraction": "low_payoff_distraction",
        "no_open_handle_wait": "open_handle_wait",
        "no_stale_causal_model": "stale_causal_model",
        "no_misleading_narrow_probe": "misleading_narrow_probe",
        "no_unjustified_runnable_idle": "unjustified_runnable_idle",
        "no_conversational_wait": "conversational_wait",
        "killed_or_launched": "killed_nothing_launched_nothing",
        "no_degeneracy": "degeneracy",
    }
    for c in unmet:
        auto_id = mapping.get(c["id"])
        if auto_id:
            auto.append({"id": auto_id, "detail": c["detail"]})

    nominal_unmet = [c["id"] for c in unmet if c["id"] != "no_degeneracy"]
    degeneracy_unmet = any(c["id"] == "no_degeneracy" for c in unmet)
    nominal_conditions_met = not nominal_unmet
    failed_on_degeneracy = bool(degeneracy_unmet) and nominal_conditions_met

    if auto or unmet:
        verdict = "FAIL"
        if failed_on_degeneracy:
            reason = (
                "every nominal condition met; FAIL ON DEGENERACY: "
                + next(c["detail"] for c in unmet if c["id"] == "no_degeneracy")
            )
        else:
            reason = "; ".join(c["id"] for c in unmet) or "automatic failure"
    else:
        verdict = "PASS"
        reason = "improved knowledge or executable, and every conjunctive guard held"

    degen_report = None
    for c in conditions:
        if c["id"] == "no_degeneracy":
            degen_report = {
                "met": c["met"],
                "detail": c["detail"],
                "cites": list(c.get("cites") or []),
            }
            break

    tps_increase_required = False
    return {
        "trial": TRIAL_ID,
        "verdict": verdict,
        "reason": reason,
        "elapsed_s": record.elapsed_s,
        "window_s": record.window_s,
        "elapsed_is_not_a_pass": True,
        "nominal_conditions_met": nominal_conditions_met,
        "failed_on_degeneracy": failed_on_degeneracy,
        "degeneracy": degen_report,
        "tps_increase_required": tps_increase_required,
        "pass_is": "IMPROVED_KNOWLEDGE_OR_IMPROVED_EXECUTABLE",
        "conditions": conditions,
        "unmet": [c["id"] for c in unmet],
        "automatic_failures": auto,
        "killed": list(record.killed),
        "launched": list(record.launched),
        "n_killed": len(record.killed),
        "n_launched": len(record.launched),
        "control": record.control,
        "ignored_self_report": True,
    }


# ---------------------------------------------------------------------------
# Velocity — from the event log and receipt timestamps, not self-report
# ---------------------------------------------------------------------------

def _wall_s(events: Sequence[Mapping[str, Any]]) -> float:
    if not events:
        return 0.0
    t0 = int(events[0].get("t_ns") or int(events[0].get("t_s") or 0) * NS_PER_S)
    t1 = int(events[-1].get("t_ns") or int(events[-1].get("t_s") or 0) * NS_PER_S)
    return max((t1 - t0) / NS_PER_S, 0.0)


def verified_frontier_movement(record: TrialRecord) -> dict[str, Any]:
    unique_kills = sorted({str(k.get("id") or k.get("family")) for k in record.killed if k})
    unique_reclass = sorted(
        {
            str(k.get("id"))
            for k in record.killed
            if k.get("warrant") in {"oracle", "superseded"}
        }
    )
    seen_launch: set[str] = set()
    novel_launches: list[str] = []
    for unit in record.launched:
        ident = _unit_identity(unit)
        if ident and ident not in seen_launch:
            seen_launch.add(ident)
            # duplicates never count; first launch of a scar-dead family does not
            # count as verified movement either
            family = ni.canon_family(str(unit.get("family") or ""))
            if family in {ni.canon_family(f) for f in DEAD_R_BOTTLENECK_FAMILIES}:
                continue
            if unit.get("superseded_by") and any(
                s in set(record.ingested) for s in unit.get("superseded_by") or []
            ):
                continue
            novel_launches.append(ident)
    movement = len(unique_kills) + len(novel_launches)
    return {
        "unique_kills": unique_kills,
        "unique_reclassifications": unique_reclass,
        "novel_launches": novel_launches,
        "n": movement,
        "rule": (
            "verified frontier movement is unique warranted kills plus first "
            "launches of non-scar, non-superseded falsifiers; raw experiment "
            "count is not this number"
        ),
    }


def compute_velocity(record: TrialRecord) -> dict[str, Any]:
    events = list(record.events)
    wall = _wall_s(events)
    denom = wall if wall > 0 else 1e-9
    movement = verified_frontier_movement(record)

    receipt_to_next_launch_ns: list[dict[str, Any]] = []
    pending_receipt_ns: int | None = None
    pending_receipt: str | None = None
    for event in events:
        if event.get("kind") == "RESULT_INGESTED":
            pending_receipt_ns = int(event.get("t_ns") or 0)
            cites = event.get("cites") or []
            pending_receipt = str(cites[0] if cites else _payload_lookup(event, "receipt") or "")
        elif event.get("kind") == "WORK_LAUNCHED" and pending_receipt_ns is not None:
            dt = int(event.get("t_ns") or 0) - pending_receipt_ns
            receipt_to_next_launch_ns.append(
                {
                    "receipt": pending_receipt,
                    "dt_ns": dt,
                    "launch_id": _unit_identity(_payload_lookup(event, "unit") or {}),
                }
            )
            pending_receipt_ns = None

    families_considered = sorted(
        {o.family for o in record.tree.options.values()} | set(record.scars_queried)
    )
    oracle_killed = sorted(
        {str(k.get("family") or k.get("id")) for k in record.killed if k.get("warrant") in {"oracle", "superseded"}}
    )
    experimentally_killed = sorted(
        {
            str(k.get("family") or k.get("id"))
            for k in record.killed
            if k.get("warrant") in {"experiment", "scar"}
        }
    )
    still_live = sorted({o.family for o in record.tree.live()})

    idle_s = 0.0
    for prev, nxt in zip(events, events[1:]):
        gap = int(nxt.get("t_s") or 0) - int(prev.get("t_s") or 0)
        if gap >= IDLE_GAP_S and prev.get("kind") not in {"WORK_LAUNCHED"}:
            idle_s += gap

    mass_before = float((record.tree_before or {}).get("live_mass_ms") or 0.0)
    mass_after = record.tree.live_mass_ms()
    n_live_before = int((record.tree_before or {}).get("n_live") or 0)
    n_live_after = len(record.tree.live())

    headline = movement["n"] / denom
    raw_count = len(record.launched)
    return {
        "schema": VELOCITY_SCHEMA,
        "version": VERSION,
        "headline_objective": "VERIFIED_FRONTIER_MOVEMENT_PER_UNIT_WALL_TIME",
        "headline": headline,
        "verified_frontier_movement": movement["n"],
        "verified_frontier_movement_detail": movement,
        "wall_s": wall,
        "raw_experiment_count": raw_count,
        "raw_experiment_count_is_not_the_headline": True,
        "receipt_to_next_launch_ns": receipt_to_next_launch_ns,
        "branches_eliminated_per_unit_time": len(record.killed) / denom,
        "search_space_collapse": {
            "live_hypothesis_mass_ms_before": mass_before,
            "live_hypothesis_mass_ms_after": mass_after,
            "delta_ms": round(mass_before - mass_after, 6),
            "n_live_before": n_live_before,
            "n_live_after": n_live_after,
            "delta_n_live": n_live_before - n_live_after,
        },
        "families": {
            "considered": families_considered,
            "n_considered": len(families_considered),
            "oracle_killed": oracle_killed,
            "n_oracle_killed": len(oracle_killed),
            "experimentally_killed": experimentally_killed,
            "n_experimentally_killed": len(experimentally_killed),
            "still_live": still_live,
            "n_still_live": len(still_live),
        },
        "idle_runnable_seconds": idle_s,
        "repeated_scar_attempts": sum(
            1
            for u in record.launched
            if ni.canon_family(str(u.get("family") or ""))
            in {ni.canon_family(f) for f in DEAD_R_BOTTLENECK_FAMILIES}
        ),
        "experiments_avoided_by_prior_evidence": len(record.experiments_avoided),
        "experiments_avoided_detail": list(record.experiments_avoided)[:24],
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "derived_from": "event log timestamps (t_ns) and landed receipt cites, not self-report",
    }


def pad_with_duplicate_launches(record: TrialRecord, n: int = 50) -> TrialRecord:
    """Add n duplicate launches at the last timestamp. Count rises; movement must not."""
    padded = copy.deepcopy(record)
    if not record.launched:
        unit = {"id": "WU.IMPROVEMENT.pad", "family": "pad", "payoff_ms": 0.02}
    else:
        unit = dict(record.launched[0])
    last_ns = int(record.events[-1].get("t_ns") or 0) if record.events else 0
    last_s = int(record.events[-1].get("t_s") or 0) if record.events else 0
    seq = len(record.events)
    extra_events: list[dict[str, Any]] = []
    extra_launched: list[dict[str, Any]] = []
    for i in range(n):
        seq += 1
        extra_events.append(
            {
                "seq": seq,
                "t_s": last_s,
                "t_ns": last_ns,
                "kind": "WORK_LAUNCHED",
                "payload": {"unit": dict(unit)},
            }
        )
        extra_launched.append(dict(unit))
    padded.events = list(record.events) + extra_events
    padded.launched = list(record.launched) + extra_launched
    return padded


# ---------------------------------------------------------------------------
# Passing skeleton and six negative controls
# ---------------------------------------------------------------------------

def _fixture_scars() -> list[ni.Scar]:
    scars = [
        _scar(
            scar_id="fixture.factorize",
            source_path="receipts/future/MLP_NONLINEAR_PROGRAM.json",
            family="FACTORIZE_THE_FACTORS",
            mechanism="r-bottleneck",
            claim="factorize-the-factors replaces F",
            reopen="full-width structured nonlinear that is not an r-bottleneck",
        ),
        _scar(
            scar_id="fixture.region",
            source_path="receipts/future/MLP_REGION_FALSIFIER.json",
            family="region_granularity",
            verdict="GRANULARITY_REFUTED",
            mechanism="fused regions",
            claim="packing reaches 497 GB/s",
        ),
        _scar(
            scar_id="fixture.reach_mlp",
            source_path="receipts/future/MLP_REGION_FALSIFIER.json",
            family="reach_demonstrated_bandwidth_mlp",
            verdict="GRANULARITY_REFUTED",
            mechanism="identical arithmetic fused regions",
            claim="fused-region MLP is the remaining 4.79 ms",
        ),
    ]
    return scars


def _fixture_tree() -> LocalOptionTree:
    tree = LocalOptionTree(generation=1)
    tree.add(
        _option(
            id="mlp_decode_fma_cheapening",
            family="mlp_decode_fma_cheapening",
            organ="mlp",
            payoff_ms=4.79,
            falsifier="STATIC decode-FMA cheapening plan toward cited 0.8835",
            evidence_cites=["receipts/future/MLP_ALU_ROOFLINE.json"],
            mechanism="arithmetic",
            conflict_group="mlp",
        )
    )
    tree.add(
        _option(
            id="deltanet_organ_vs_isolated_kernel",
            family="deltanet_organ_vs_isolated_kernel",
            organ="deltanet",
            payoff_ms=2.273,
            falsifier="STATIC organ-vs-isolated-kernel reconciliation",
            evidence_cites=["receipts/future/ORGAN_BANDWIDTH.json"],
            mechanism="unexplained remainder",
            conflict_group="deltanet",
        )
    )
    tree.add(
        _option(
            id="reach_demonstrated_bandwidth_mlp",
            family="reach_demonstrated_bandwidth_mlp",
            organ="mlp",
            payoff_ms=4.79,
            terminal=LocalTerminalState.RUNNING,
            falsifier="one representative layer, contiguous, one/few fused regions, identical arithmetic",
            evidence_cites=["receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"],
            mechanism="fused regions identical arithmetic",
            superseded_by=["receipts/future/MLP_REGION_FALSIFIER.json"],
            conflict_group="mlp",
        )
    )
    tree.add(
        _option(
            id="mlp_r_bottleneck.FACTORIZE_THE_FACTORS",
            family="FACTORIZE_THE_FACTORS",
            organ="mlp",
            payoff_ms=0.0,
            falsifier="held-out relative L2",
            evidence_cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
            mechanism="r-bottleneck",
            conflict_group="mlp_function_replacement",
        )
    )
    tree.add(
        _option(
            id="host_gap_rounding",
            family="host_gap_rounding",
            organ="",
            payoff_ms=LOW_PAYOFF_MS,
            falsifier="round the 0.02 ms leftover",
            evidence_cites=["receipts/future/WALL_GPU_RECONCILIATION.json"],
            mechanism="host leftover",
            conflict_group="host",
        )
    )
    tree.add(
        _option(
            id="group_size_1024",
            family="group_size_1024",
            organ="mlp",
            payoff_ms=2.914,
            cost="ONE_FIT_PLUS_A_CAPABILITY_SCREEN",
            falsifier="held-out reconstruction plus organ error",
            evidence_cites=["receipts/future/MLP_AUXILIARY_INFORMATION.json"],
            conflict_group="mlp_aux",
        )
    )
    return tree


def _good_prefix(resident: Resident) -> None:
    resident.snapshot_before()
    resident.recover()
    resident.inspect_budget(
        {
            "experiments_ranked_by_gain": [
                {"id": o.id, "ms_saved": o.payoff_ms, "status": _term(o.terminal)}
                for o in resident.tree.options.values()
            ]
        }
    )
    resident.query_scars()
    resident.rank()
    resident.generate_falsifiers()
    resident.ingest("receipts/future/MLP_ALU_ROOFLINE.json")
    resident.ingest("receipts/future/MLP_REGION_FALSIFIER.json")
    resident.ingest("receipts/future/MLP_NONLINEAR_PROGRAM.json")
    resident.ingest("receipts/future/ORGAN_BANDWIDTH.json")


def _good_suffix(resident: Resident, *, launch: bool = True) -> TrialRecord:
    if launch:
        targets = [
            o
            for o in resident.tree.live()
            if o.id in {"mlp_decode_fma_cheapening", "deltanet_organ_vs_isolated_kernel"}
        ]
        resident.launch_nonconflicting(targets)
    resident.kill(
        "reach_demonstrated_bandwidth_mlp",
        warrant="oracle",
        cites=["receipts/future/MLP_REGION_FALSIFIER.json"],
        reason="fused-region / granularity falsifier is GRANULARITY_REFUTED",
    )
    resident.kill(
        "mlp_r_bottleneck.FACTORIZE_THE_FACTORS",
        warrant="scar",
        cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
        reason="r-bottleneck family MEASURED_NEGATIVE",
    )
    resident.update_tree()
    leftover = [o.id for o in resident.tree.live() if _term(o.terminal) == LocalTerminalState.LIVE]
    if leftover:
        resident.refill(leftover[:4])
    else:
        resident.refill(["group_size_1024"])
    resident.leave_running()
    resident.write_durable()
    return resident.record()


def passing_skeleton() -> TrialRecord:
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars(), control=None)
    _good_prefix(resident)
    return _good_suffix(resident)


def nominal_but_degenerate(*, n_ingests: int = 29) -> TrialRecord:
    """Every nominal condition still holds; one receipt is ingested `n_ingests` times.

    The 1h pathology in miniature: SPECIMEN_VERIFICATION.json looped until
    diversity is unmeasurable. The written acts, kills, and launches are
    untouched so the only unmet guard is no_degeneracy.
    """
    record = passing_skeleton()
    last = record.events[-1] if record.events else {
        "seq": 0, "t_s": 0, "t_ns": 0,
    }
    seq = int(last.get("seq") or 0)
    t_s = int(last.get("t_s") or 0)
    t_ns = int(last.get("t_ns") or 0)
    receipt = "receipts/future/SPECIMEN_VERIFICATION.json"
    extra: list[dict[str, Any]] = []
    for _ in range(int(n_ingests)):
        seq += 1
        extra.append(
            {
                "seq": seq,
                "t_s": t_s,
                "t_ns": t_ns,
                "kind": "RESULT_INGESTED",
                "payload": {"what": "loop", "receipt": receipt},
                "cites": [receipt],
            }
        )
        record.ingested.append(receipt)
    record.events = list(record.events) + extra
    return record


def _control_duplicate() -> TrialRecord:
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars(), control="duplicate_workunits")
    _good_prefix(resident)
    rec = _good_suffix(resident)
    # Relaunch the same unit with no new ingest in between.
    opt = resident.tree.get("mlp_decode_fma_cheapening")
    assert opt is not None
    opt.terminal = LocalTerminalState.LIVE
    resident.launch(opt, unit_id="WU.IMPROVEMENT.mlp_decode_fma_cheapening")
    return resident.record()


def _control_dead_scar() -> TrialRecord:
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars(), control="dead_scar_repetition")
    _good_prefix(resident)
    rec = _good_suffix(resident)
    dead = resident.tree.get("mlp_r_bottleneck.FACTORIZE_THE_FACTORS")
    assert dead is not None
    dead.terminal = LocalTerminalState.LIVE
    resident.launch(dead, ignore_scar=True, unit_id="WU.IMPROVEMENT.factorize_replay")
    return resident.record()


def _control_low_payoff() -> TrialRecord:
    resident = Resident(
        tree=_fixture_tree(), scars=_fixture_scars(), control="low_payoff_distraction"
    )
    _good_prefix(resident)
    # Work the 0.02 ms candidate; leave the multi-ms options sitting.
    low = resident.tree.get("host_gap_rounding")
    assert low is not None
    resident.launch(low)
    resident.kill(
        "reach_demonstrated_bandwidth_mlp",
        warrant="oracle",
        cites=["receipts/future/MLP_REGION_FALSIFIER.json"],
        reason="granularity refuted",
    )
    resident.kill(
        "mlp_r_bottleneck.FACTORIZE_THE_FACTORS",
        warrant="scar",
        cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
        reason="r-bottleneck",
    )
    resident.update_tree()
    resident.refill(["group_size_1024"])
    resident.leave_running()
    resident.write_durable()
    return resident.record()


def _control_open_handle() -> TrialRecord:
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars(), control="open_handle_wait")
    _good_prefix(resident)
    targets = [resident.tree.get("mlp_decode_fma_cheapening")]
    launched = resident.launch_nonconflicting([t for t in targets if t])
    handle = (launched[0]["id"] if launched else "WU.IMPROVEMENT.mlp_decode_fma_cheapening")
    other = [
        o.id
        for o in resident.tree.live()
        if o.id != "mlp_decode_fma_cheapening"
    ]
    resident.log.emit(
        "FRONTIER_HAS_WORK",
        {"unit_ids": other or ["deltanet_organ_vs_isolated_kernel", "group_size_1024"]},
        advance_s=0.0,
    )
    resident.wait_on_handle(handle, OPEN_HANDLE_REPRO_S, other or ["group_size_1024"])
    resident.kill(
        "reach_demonstrated_bandwidth_mlp",
        warrant="oracle",
        cites=["receipts/future/MLP_REGION_FALSIFIER.json"],
        reason="granularity refuted",
    )
    resident.kill(
        "mlp_r_bottleneck.FACTORIZE_THE_FACTORS",
        warrant="scar",
        cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
        reason="r-bottleneck",
    )
    resident.update_tree()
    resident.refill(other[:4] or ["group_size_1024"])
    resident.leave_running()
    resident.write_durable()
    return resident.record()


def _control_stale_causal() -> TrialRecord:
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars(), control="stale_causal_model")
    _good_prefix(resident)
    stale = resident.tree.get("reach_demonstrated_bandwidth_mlp")
    assert stale is not None
    stale.terminal = LocalTerminalState.LIVE
    resident.launch(stale, ignore_stale=True, ignore_scar=True)
    resident.kill(
        "mlp_r_bottleneck.FACTORIZE_THE_FACTORS",
        warrant="scar",
        cites=["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
        reason="r-bottleneck",
    )
    resident.update_tree()
    resident.refill(["group_size_1024"])
    resident.leave_running()
    resident.write_durable()
    return resident.record()


def _control_misleading_probe() -> TrialRecord:
    resident = Resident(
        tree=_fixture_tree(), scars=_fixture_scars(), control="misleading_narrow_probe"
    )
    _good_prefix(resident)
    rec = _good_suffix(resident)
    resident.conclude(
        claim="all organs are ALU_BOUND",
        probe="MLP_ALU_ROOFLINE one representative layer, verdict MIXED",
        scope="all_organs",
        supported=False,
    )
    return resident.record()


CONTROL_FACTORIES: dict[str, Callable[[], TrialRecord]] = {
    "duplicate_workunits": _control_duplicate,
    "dead_scar_repetition": _control_dead_scar,
    "low_payoff_distraction": _control_low_payoff,
    "open_handle_wait": _control_open_handle,
    "stale_causal_model": _control_stale_causal,
    "misleading_narrow_probe": _control_misleading_probe,
}


def negative_control(name: str) -> dict[str, Any]:
    if name not in CONTROL_FACTORIES:
        raise KeyError(name)
    record = CONTROL_FACTORIES[name]()
    judged = judge(record)
    return {
        "control": name,
        "verdict": judged["verdict"],
        "reason": judged["reason"],
        "unmet": judged["unmet"],
        "automatic_failures": judged["automatic_failures"],
        "elapsed_s": judged["elapsed_s"],
        "n_killed": judged["n_killed"],
        "n_launched": judged["n_launched"],
        "must_fail": True,
        "failed": judged["verdict"] == "FAIL",
    }


def run_negative_controls() -> dict[str, Any]:
    controls = [negative_control(name) for name in CONTROL_NAMES]
    n_pass = sum(1 for c in controls if c["verdict"] == "PASS")
    n_fail = sum(1 for c in controls if c["verdict"] == "FAIL")
    return {
        "n_controls": len(controls),
        "n_fail": n_fail,
        "n_pass": n_pass,
        "all_failed": n_fail == len(controls) and n_pass == 0,
        "controls": controls,
    }


def harness_verdict(control_rows: Sequence[Mapping[str, Any]]) -> str:
    """If any negative control PASSES, the harness is broken. Never report green."""
    for row in control_rows:
        if str(row.get("verdict") or "") == "PASS":
            return "BROKEN_HARNESS"
    return "OK"


def empty_kill_launch_record() -> TrialRecord:
    """Honest empty run: recovers, inspects, queries, ranks — kills nothing, launches nothing."""
    resident = Resident(tree=_fixture_tree(), scars=_fixture_scars())
    resident.snapshot_before()
    resident.recover()
    resident.inspect_budget({"experiments_ranked_by_gain": []})
    resident.query_scars()
    resident.rank()
    resident.generate_falsifiers()
    resident.ingest("receipts/future/MLP_ALU_ROOFLINE.json")
    resident.update_tree()
    resident.refill(["group_size_1024"])
    resident.leave_running()
    resident.write_durable()
    return resident.record()


# ---------------------------------------------------------------------------
# Live trial against the resident frontier as it actually stands
# ---------------------------------------------------------------------------

def _warranted_kills(tree: LocalOptionTree, live: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    alu_present = bool((live.get("mlp_alu_roofline") or {}).get("present"))
    region = (live.get("mlp_region_falsifier") or {}).get("doc") or {}
    budget = (live.get("resident_71tps_causal_budget") or {}).get("doc") or {}
    nonlinear = (live.get("mlp_nonlinear_program") or {}).get("doc") or {}
    organ_bw = (live.get("organ_bandwidth") or {}).get("doc") or {}
    out: list[dict[str, Any]] = []

    if region.get("verdict") == "GRANULARITY_REFUTED" or (
        "GRANULARITY_REFUTED" in json.dumps(region)
    ):
        if tree.get("reach_demonstrated_bandwidth_mlp"):
            out.append(
                {
                    "id": "reach_demonstrated_bandwidth_mlp",
                    "warrant": "oracle",
                    "cites": ["receipts/future/MLP_REGION_FALSIFIER.json"],
                    "reason": (
                        "causal-budget falsifier was fused regions with identical "
                        "arithmetic; MLP_REGION_FALSIFIER.json is GRANULARITY_REFUTED. "
                        "ALU_ROOFLINE names arithmetic, not packing. Do not rescue a "
                        "refuted layout lever."
                    ),
                }
            )
    refuted_ids = {str(r.get("id")) for r in budget.get("refuted_levers") or []}
    if "entropy_code_the_mlp_codes" in refuted_ids or "entropy_floor_of_mlp_codes" in [
        str(e.get("id")) for e in budget.get("experiments_ranked_by_gain") or []
    ]:
        if tree.get("entropy_floor_of_mlp_codes"):
            out.append(
                {
                    "id": "entropy_floor_of_mlp_codes",
                    "warrant": "oracle",
                    "cites": [
                        "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
                        "receipts/future/MLP_CODE_INFORMATION.json",
                    ],
                    "reason": "refuted_levers records AT_THE_FLOOR; OPEN on the ranked list is stale",
                }
            )
    for family in nonlinear.get("families") or list(DEAD_R_BOTTLENECK_FAMILIES):
        oid = f"mlp_r_bottleneck.{family}"
        if tree.get(oid):
            out.append(
                {
                    "id": oid,
                    "warrant": "experiment",
                    "cites": ["receipts/future/MLP_NONLINEAR_PROGRAM.json"],
                    "reason": f"{family} MEASURED_NEGATIVE: r-bottleneck, not a program for F",
                }
            )
    findings = {str(f.get("id")) for f in organ_bw.get("findings") or []}
    if "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE" in findings:
        if tree.get("dispatch_count_explains_organ_gap"):
            out.append(
                {
                    "id": "dispatch_count_explains_organ_gap",
                    "warrant": "oracle",
                    "cites": ["receipts/future/ORGAN_BANDWIDTH.json"],
                    "reason": "least-squares per-dispatch cost is negative; the model is refuted",
                }
            )
    if alu_present and organ_bw:
        if tree.get("deltanet_big_kernel_is_organ_cost"):
            out.append(
                {
                    "id": "deltanet_big_kernel_is_organ_cost",
                    "warrant": "oracle",
                    "cites": [
                        "receipts/future/ORGAN_BANDWIDTH.json",
                        "receipts/future/MLP_ALU_ROOFLINE.json",
                    ],
                    "reason": (
                        "cited organ GB/s is 360.0; cited isolated largest-kernel "
                        "GB/s is 600.9. The big kernel is not the organ's cost."
                    ),
                }
            )
    return out


def _live_launch_targets(tree: LocalOptionTree, scars: Sequence[ni.Scar]) -> list[LocalOption]:
    want_ids = (
        "mlp_decode_fma_cheapening",
        "deltanet_organ_vs_isolated_kernel",
    )
    out: list[LocalOption] = []
    for oid in want_ids:
        opt = tree.get(oid)
        if opt is None:
            continue
        if ni.refuse_if_dead({"hypothesis_family": opt.family, "organ": opt.organ}, scars=list(scars)):
            continue
        if _term(opt.terminal) in {LocalTerminalState.LIVE, LocalTerminalState.RUNNING}:
            out.append(opt)
    return out


def _refill_ids(live: Mapping[str, Mapping[str, Any]], tree: LocalOptionTree) -> list[str]:
    ids: list[str] = []
    frontier = (live.get("frontier_state") or {}).get("doc") or {}
    for unit in frontier.get("next_work") or []:
        if not isinstance(unit, Mapping):
            continue
        rc = str(unit.get("resource_class") or "")
        cl = str(unit.get("classification") or unit.get("candidate_status") or "")
        uid = str(unit.get("id") or "")
        if not uid:
            continue
        if rc in {"STATIC_ANALYSIS", "LIGHT_CONTROL", "TEST_AUTHORING"} or "STATIC" in cl:
            ids.append(uid)
        if len(ids) >= 6:
            break
    for o in tree.live():
        if o.role == LocalWorkUnitRole.REFILL or o.id.startswith("odyssey_unmet."):
            if o.id not in ids:
                ids.append(o.id)
        if o.id == "group_size_1024" and o.id not in ids:
            ids.append(o.id)
    return ids[:8]


def run_live_trial() -> TrialRecord:
    live = load_live_receipts()
    tree = build_option_tree(live)
    scars = scars_from_live(live)
    resident = Resident(tree=tree, scars=scars, live=True)
    resident.snapshot_before()
    resident.recover(live)
    budget = (live.get("resident_71tps_causal_budget") or {}).get("doc") or {}
    resident.inspect_budget(budget)
    resident.query_scars()
    resident.rank()
    resident.generate_falsifiers()

    # Ingest landed receipts that complete the picture (STATIC_ONLY — already on disk/git).
    for key, rel in (
        ("mlp_alu_roofline", "receipts/future/MLP_ALU_ROOFLINE.json"),
        ("mlp_region_falsifier", "receipts/future/MLP_REGION_FALSIFIER.json"),
        ("mlp_nonlinear_program", "receipts/future/MLP_NONLINEAR_PROGRAM.json"),
        ("organ_bandwidth", "receipts/future/ORGAN_BANDWIDTH.json"),
        ("resident_71tps_causal_budget", "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"),
        ("odyssey_launch_gate", "receipts/future/ODYSSEY_LAUNCH_GATE.json"),
        ("tps_falsifications", "receipts/future/TPS_FALSIFICATIONS.json"),
        ("path_to_71", "receipts/future/PATH_TO_71.json"),
        ("mlp_shared_program", "receipts/future/MLP_SHARED_PROGRAM.json"),
    ):
        if (live.get(key) or {}).get("present"):
            resident.ingest(rel)

    for kill in _warranted_kills(resident.tree, live):
        resident.kill(
            str(kill["id"]),
            warrant=str(kill["warrant"]),
            cites=list(kill["cites"]),
            reason=str(kill["reason"]),
        )

    resident.launch_nonconflicting(_live_launch_targets(resident.tree, scars))
    resident.update_tree()
    refill = _refill_ids(live, resident.tree)
    if refill:
        resident.refill(refill)
    resident.leave_running()
    resident.write_durable()
    return resident.record()


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def _public_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "id",
        "family",
        "organ",
        "payoff_ms",
        "resource_class",
        "classification",
        "conflict_group",
        "falsifier",
        "mechanism",
        "warrant",
        "reason",
        "terminal",
        "cites",
    )
    return {k: unit[k] for k in keep if k in unit}


def finalize_verdict(trial_verdict: str, controls: Mapping[str, Any]) -> str:
    rows = list(controls.get("controls") or [])
    integrity = harness_verdict(rows)
    if integrity == "BROKEN_HARNESS":
        return "BROKEN_HARNESS"
    return trial_verdict


def build(*, run_live: bool = True) -> Path:
    controls = run_negative_controls()
    integrity = harness_verdict(controls["controls"])
    live_record: TrialRecord | None = None
    live_judged: dict[str, Any] | None = None
    velocity: dict[str, Any] | None = None
    if run_live:
        live_record = run_live_trial()
        live_judged = judge(live_record)
        velocity = compute_velocity(live_record)
        # Proof artifact: stuffing duplicate launches cannot raise the headline.
        padded = pad_with_duplicate_launches(live_record, n=50)
        padded_v = compute_velocity(padded)
        velocity["count_cannot_raise_headline_proof"] = {
            "base_headline": velocity["headline"],
            "base_raw_experiment_count": velocity["raw_experiment_count"],
            "base_verified_frontier_movement": velocity["verified_frontier_movement"],
            "padded_duplicate_launches": 50,
            "padded_headline": padded_v["headline"],
            "padded_raw_experiment_count": padded_v["raw_experiment_count"],
            "padded_verified_frontier_movement": padded_v["verified_frontier_movement"],
            "headline_did_not_rise": padded_v["headline"] <= velocity["headline"] + 1e-15,
            "movement_unchanged": padded_v["verified_frontier_movement"]
            == velocity["verified_frontier_movement"],
            "count_did_rise": padded_v["raw_experiment_count"] > velocity["raw_experiment_count"],
        }

    trial_verdict = (live_judged or {}).get("verdict") or "FAIL"
    if not live_record:
        trial_verdict = "FAIL"
    final = finalize_verdict(trial_verdict, controls)
    if integrity == "BROKEN_HARNESS":
        # A control that PASSES means the judge cannot see the defect. Never green.
        final = "BROKEN_HARNESS"

    killed = [_public_unit(k) for k in (live_record.killed if live_record else [])]
    launched = [_public_unit(u) for u in (live_record.launched if live_record else [])]
    if live_record and not killed and not launched:
        final = "FAIL"
        trial_verdict = "FAIL"

    passing_judged = judge(passing_skeleton())
    count_proof_ok = True
    if velocity and velocity.get("count_cannot_raise_headline_proof"):
        count_proof_ok = bool(velocity["count_cannot_raise_headline_proof"]["headline_did_not_rise"])

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "trial": TRIAL_ID,
        "window_s": WINDOW_S,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "verdict": final,
        "trial_verdict_before_harness": trial_verdict,
        "harness_integrity": integrity,
        "pass_is": "IMPROVED_KNOWLEDGE_OR_IMPROVED_EXECUTABLE",
        "tps_increase_required": False,
        "timer_is_not_a_pass": True,
        "killed": killed,
        "launched": launched,
        "next_running": list(live_record.next_running) if live_record else [],
        "n_killed": len(killed),
        "n_launched": len(launched),
        "negative_controls": controls,
        "all_six_negative_controls_failed": bool(controls.get("all_failed")),
        "broken_harness_if_any_control_passes": True,
        "metabolism_integration": metabolism_seam(),
        "live_state": (live_record.live_summary if live_record else {}),
        "receipts_loaded": (live_record.receipts_loaded if live_record else []),
        "elapsed_s": (live_record.elapsed_s if live_record else 0),
        "conditions": (live_judged or {}).get("conditions") or [],
        "unmet": (live_judged or {}).get("unmet") or [],
        "automatic_failures": (live_judged or {}).get("automatic_failures") or [],
        "nominal_conditions_met": (live_judged or {}).get("nominal_conditions_met"),
        "failed_on_degeneracy": (live_judged or {}).get("failed_on_degeneracy"),
        "degeneracy": (live_judged or {}).get("degeneracy"),
        "reason": (live_judged or {}).get("reason") if final != "BROKEN_HARNESS" else (
            "a negative control PASSed; the harness cannot see the defect it claims to catch"
        ),
        "passing_skeleton_verdict": passing_judged["verdict"],
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "head": git("rev-parse", "HEAD"),
        "required_acts": list(REQUIRED_ACTS),
        "six_negative_controls": list(CONTROL_NAMES),
        "claim_boundary": (
            "STATIC_ONLY sidecar. No GPU lease and no hardware measurement. "
            "Cited GB/s and FMA/byte figures are copied as strings from landed "
            "receipts (MLP_ALU_ROOFLINE, ORGAN_BANDWIDTH, causal budget). They "
            "are not re-measured and are not claimed as this run's result. "
            "PASS is improved knowledge or improved executable, not a TPS "
            "increase. If this trial killed nothing and launched nothing the "
            "verdict is FAIL. If any of the six negative controls PASSes, the "
            "harness reports BROKEN_HARNESS rather than a green trial. "
            "FAIL ON DEGENERACY EVEN WHEN EVERY CONDITION IS NOMINALLY MET."
        ),
        "event_log": (live_record.events if live_record else []),
        "experiments_avoided_by_prior_evidence": (
            live_record.experiments_avoided if live_record else []
        ),
        "count_cannot_raise_headline_proof_ok": count_proof_ok,
    }
    _assert_no_hardware_claims(doc)
    trial_path = write_receipt(RECEIPT, doc, RECORDED_BY)

    vel_doc: dict[str, Any]
    if velocity is None:
        vel_doc = {
            "schema": VELOCITY_SCHEMA,
            "version": VERSION,
            "headline_objective": "VERIFIED_FRONTIER_MOVEMENT_PER_UNIT_WALL_TIME",
            "headline": 0.0,
            "raw_experiment_count_is_not_the_headline": True,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
            "note": "live trial was not run",
        }
    else:
        vel_doc = dict(velocity)
        vel_doc["recorded_by"] = RECORDED_BY
        vel_doc["trial"] = TRIAL_ID
        vel_doc["claim_boundary"] = (
            "Velocity is computed from event-log t_ns and landed receipt cites. "
            "It is not a self-reported rate and not a hardware measurement. "
            "The headline is verified frontier movement per unit wall time; "
            "raw experiment count cannot raise it."
        )
    _assert_no_hardware_claims(vel_doc)
    write_receipt(VELOCITY_RECEIPT, vel_doc, RECORDED_BY)
    return trial_path


def selftest() -> Path:
    return build(run_live=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args(argv)
    if args.controls_only:
        doc = run_negative_controls()
        for row in doc["controls"]:
            print(f"  {row['verdict']:4}  {row['control']}  {row['reason']}")
        print(f"{doc['n_fail']}/{doc['n_controls']} failed; all_failed={doc['all_failed']}")
        return 0 if doc["all_failed"] else 1
    path = build(run_live=True)
    doc = json.loads(path.read_text())
    print(f"wrote {path}")
    print(f"verdict={doc.get('verdict')} harness={doc.get('harness_integrity')}")
    print(f"killed={[k.get('id') for k in doc.get('killed') or []]}")
    print(f"launched={[u.get('id') for u in doc.get('launched') or []]}")
    if args.selftest or args.record:
        return 0 if doc.get("verdict") in {"PASS", "FAIL"} and doc.get("harness_integrity") == "OK" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
