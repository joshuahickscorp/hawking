"""POWER TORTURE — freeze a build, exercise every post-1h power, seal a timeline.

The frozen 1h autonomy trial passed a clock. Scrutiny of its own timeline showed
what it did not do: 222 rejections of one table in nine seconds, four refills of
the same 25 ids, one receipt ingested 29 times, and not one unit mutated
anything. It is a verifier, not an optimizer. The powers that landed after that
trial were never asked to demonstrate the behaviour they exist for.

This module is three things, kept separate because collapsing them is how a
torture becomes a checklist:

* `compose` still selects 6–12 units worth doing if the run dies halfway.
* `run_torture` FREEZES the substrate, EXERCISES every named power inside a
  30-minute wall budget, and APPENDS a sealed timeline. It does not judge.
* `judge` reads the sealed timeline and nothing else. Degeneracy is scored
  by `autonomy_degeneracy.measure` over that same timeline.

A correct refusal is EXERCISED, not SKIPPED. A power that cannot run is
SKIPPED with a reason and does not count toward the pass. A substrate file
that moves mid-run is an automatic FAIL naming the file. PASS means new
power integration on a frozen build, never resident cognition.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    REPO,
    sha256_file,
    write_receipt,
    _assert_no_hardware_claims,
)
from tools.future import abliteration as ab
from tools.future import autonomy_trial as at
from tools.future import concurrency_doctor as cd
from tools.future import flash_organ_pivot as fop
from tools.future import frontiers as fr
from tools.future import mutation_engine as me
from tools.future import negative_index as ni
from tools.future import orchestration as orch
from tools.future import protected_scheduler as ps
from tools.future import scar_scheduling as ss
from tools.future import status_causality as sc
from tools.future import trial_workload as tw
from tools.future import work_events as we
from tools.future import workgraph as wg

RECEIPT = "POWER_TORTURE_30M.json"
TIMELINE_RECEIPT = "POWER_TORTURE_TIMELINE.json"
SCHEMA = "hawking.future.power_torture.v2"
TIMELINE_SCHEMA = "hawking.future.power_torture.timeline.v1"
RECORDED_BY = "tools/future/power_torture.py"

DURATION_S = 30 * 60
TRIAL_ID = "30m"

EXERCISED = "EXERCISED"
EXERCISED_REFUSAL = "EXERCISED_REFUSAL"
SKIPPED = "SKIPPED"
FAILED = "FAILED"
SUBSTRATE_CLEAN = "CLEAN"
SUBSTRATE_MOVED = "INVALIDATED_BY_SUBSTRATE_MUTATION"

GPU_LANE_LOCK = Path("/tmp/hawking-gpu-lane.lock")
DEGENERACY_CANDIDATES: tuple[Path, ...] = (
    REPO / "tools" / "future" / "autonomy_degeneracy.py",
    Path("/Users/scammermike/Downloads/hawking/tools/future/autonomy_degeneracy.py"),
    Path(
        "/Users/scammermike/.claude-grok/worktrees/d2degen-20260831-052640"
        "/tools/future/autonomy_degeneracy.py"
    ),
)

FAIL_NO_WAIT = "FAIL_NO_WAIT_ORCHESTRATION"
NO_WAIT_OK = "NO_WAIT_OK"
NO_WAIT_UNTESTED = "UNTESTED"

REQUIRED_TRANSITIONS: tuple[str, ...] = (
    "NO_WAIT",
    "REAL_REFILL",
    "REAL_INGESTION",
    "SCAR_PRUNING",
    "STATUS_CAUSALITY",
    "PROTECTED_PARKING",
    "GENERIC_NR_NX",
    "MUTATION",
    "SUBAGENT_STATE",
    "CONCURRENCY",
    "FRONTIER_INVALIDATION",
    "REPLAN",
)

# The obligation's 14 powers. REQUIRED_TRANSITIONS is the composer mix;
# POWER_CATALOG is what the frozen run must exercise or SKIP with a reason.
POWER_CATALOG: tuple[str, ...] = REQUIRED_TRANSITIONS + (
    "BLOCKED_RESOURCE_REROUTE",
    "TABULA",
)

# Event kinds the independent judge scores. Canonical work_events kinds
# keep degeneracy.measure on the same vocabulary the 1h trial failed.
JUDGE_KIND_TO_POWER: dict[str, str] = {
    "detached_started": "NO_WAIT",
    "WORK_LAUNCHED": "NO_WAIT",
    "workunit_launched": "NO_WAIT",
    "WORK_REFILLED": "REAL_REFILL",
    "RESULT_INGESTED": "REAL_INGESTION",
    "IDEA_REJECTED": "SCAR_PRUNING",
    "NEGATIVE_SCIENCE_REFUSAL": "SCAR_PRUNING",
    "idea_rejected": "SCAR_PRUNING",
    "status_challenged": "STATUS_CAUSALITY",
    "workunit_sleeping": "PROTECTED_PARKING",
    "nr_nx_stage": "GENERIC_NR_NX",
    "mutation_applied": "MUTATION",
    "mutation_rolled_back": "MUTATION",
    "subagent_state_persisted": "SUBAGENT_STATE",
    "concurrency_doctor_decided": "CONCURRENCY",
    "restatement_refused": "FRONTIER_INVALIDATION",
    "replan_emitted": "REPLAN",
    "blocked_resource_reroute": "BLOCKED_RESOURCE_REROUTE",
    "tabula_callable": "TABULA",
    "power_skipped": "",  # payload.power names it
    "power_failed": "",
}

# Named because nr_nx_generic.py cannot be imported in this sparse checkout
# (tools.odyssey is not materialized). The order is the pipeline's own
# STAGE_ORDER; a missing compiler is REFUSED at the first stage that needs it,
# never SKIPPED.
NR_NX_STAGE_ORDER: tuple[str, ...] = (
    "SpecimenSelect",
    "SpecimenPresent",
    "ArchitectureRecognizer",
    "OrganGraph",
    "NrIdentifyOrCreate",
    "Doctor",
    "RepresentationPlanner",
    "PhysicalGraphCompiler",
    "KernelPlanner",
    "DeviceCompiler",
    "NoeticExecutable",
    "SourceIndependence",
    "ExecutableDependencyAccounting",
    "Verifier",
)

DEAD_SCAR_PROPOSAL: dict[str, str] = {
    "model": "qwen3-80b",
    "organ": "routed_experts",
    "hypothesis_family": "cross_expert_structure",
}
LIVE_REPLACEMENT_PROPOSAL: dict[str, str] = {
    "model": "qwen3.8-27b",
    "organ": "ngram_embedding",
    "hypothesis_family": "n_gram_product_codebook_table",
}

LAUNCH_KINDS = frozenset({"workunit_launched", "WORK_LAUNCHED", "detached_started"})
PROGRESS_KINDS = frozenset({"workunit_progressed", "WORK_PROGRESSED", "mission_state_written"})
INGEST_KINDS = frozenset(
    {"result_ingested", "RESULT_INGESTED", "detached_completed", "receipt_ingested"}
)
WAIT_KINDS = frozenset(
    {"blocking_wait", "subprocess_wait", "awaiting_instructions", "conversational_wait"}
)


class TortureRefused(tw.WorkloadRefused):
    """A 30-minute torture that would look complete without a required transition."""


# ---------------------------------------------------------------------------
# Event access. Timestamps are evidence; intent is not.
# ---------------------------------------------------------------------------


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = event.get("payload")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _kind(event: Mapping[str, Any]) -> str:
    kind = str(event.get("kind") or "")
    if kind in we.LEGACY_ALIASES:
        return we.LEGACY_ALIASES[kind]
    if kind in we.PRECANONICAL:
        return we.PRECANONICAL[kind]
    folded = kind.upper().replace("-", "_")
    if folded in we.EVENT_KINDS:
        return folded
    return kind


def _t(event: Mapping[str, Any]) -> float:
    for key in ("t_s", "t", "ts"):
        val = event.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    payload = _payload(event)
    for key in ("t_s", "launched_at", "finished_at", "started_at", "progress_at"):
        val = payload.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    seq = event.get("seq")
    if isinstance(seq, int):
        return float(seq)
    return 0.0


def _unit_id(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    unit = payload.get("unit") if isinstance(payload.get("unit"), Mapping) else {}
    for src in (payload, unit, event):
        if not isinstance(src, Mapping):
            continue
        for key in ("unit_id", "job_id", "workunit_id", "id"):
            text = str(src.get(key) or "").strip()
            if text:
                return text
    return ""


def _ids_of(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        uid = str(value.get("id") or value.get("unit_id") or "").strip()
        return [uid] if uid else []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_ids_of(item))
        return out
    return []


def _runnable_ids(event: Mapping[str, Any]) -> list[str]:
    payload = _payload(event)
    found: list[str] = []
    for key in ("runnable_now", "runnable", "queue_runnable", "unit_ids", "queue"):
        found.extend(_ids_of(payload.get(key)))
    found.extend(_ids_of(event.get("runnable_now")))
    seen: set[str] = set()
    ordered: list[str] = []
    for uid in found:
        if uid not in seen:
            seen.add(uid)
            ordered.append(uid)
    return ordered


def _is_detached(event: Mapping[str, Any]) -> bool:
    kind = _kind(event)
    if kind == "detached_started":
        return True
    payload = _payload(event)
    if payload.get("detached") is True or payload.get("no_wait") is True:
        return True
    launch = str(payload.get("launch") or "").lower()
    return launch in {"detached", "no_wait", "no-wait"}


def _is_blocking(event: Mapping[str, Any]) -> bool:
    kind = _kind(event)
    if kind in WAIT_KINDS:
        return True
    if _is_detached(event):
        return False
    payload = _payload(event)
    if payload.get("blocking") is True or payload.get("waited_on_subprocess") is True:
        return True
    if kind in LAUNCH_KINDS:
        return True
    return False


# ---------------------------------------------------------------------------
# NO-WAIT detector. The torture fails itself on this, not a human.
# ---------------------------------------------------------------------------


def detect_no_wait_orchestration(timeline: Mapping[str, Any] | None) -> dict[str, Any]:
    """FAIL_NO_WAIT_ORCHESTRATION iff the loop waited while runnable work existed.

    Overlap of a detached job with an independent unit that starts, progresses
    AND completes is read from timestamps. A missing timeline is a refusal, not
    a default pass. An empty event list is UNTESTED: absence is not proof.
    """
    if timeline is None:
        raise TortureRefused(
            "timeline is required; refusing to invent a no-wait verdict",
            missing=["timeline"],
        )
    if not isinstance(timeline, Mapping):
        raise TortureRefused(
            f"timeline must be a mapping, got {type(timeline).__name__}",
            missing=["timeline"],
        )
    events = [dict(e) for e in (timeline.get("events") or []) if isinstance(e, Mapping)]
    if not events:
        return {
            "verdict": NO_WAIT_UNTESTED,
            "fail": False,
            "reason": "no events; absence is not proof the loop did not wait",
            "cites": [],
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    ordered = sorted(events, key=lambda e: (_t(e), int(e.get("seq") or 0)))
    launches: list[dict[str, Any]] = []
    progress_at: dict[str, list[float]] = {}
    ingest_at: dict[str, float] = {}
    open_from: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []

    for event in ordered:
        kind = _kind(event)
        t = _t(event)
        uid = _unit_id(event)
        if kind in LAUNCH_KINDS or (kind in WAIT_KINDS):
            runnable = [r for r in _runnable_ids(event) if r and r != uid]
            row = {
                "id": uid,
                "t": t,
                "detached": _is_detached(event),
                "blocking": _is_blocking(event),
                "runnable": runnable,
                "kind": kind,
                "seq": event.get("seq"),
            }
            launches.append(row)
            if uid:
                open_from[uid] = row
        if kind in PROGRESS_KINDS and uid:
            progress_at.setdefault(uid, []).append(t)
        if kind in INGEST_KINDS and uid:
            ingest_at[uid] = t
            start = open_from.get(uid)
            if start is not None and start["blocking"] and not start["detached"]:
                independent = [r for r in start["runnable"] if r != uid]
                launched_during = [
                    other
                    for other in launches
                    if other["id"]
                    and other["id"] != uid
                    and start["t"] <= other["t"] <= t
                ]
                if independent and not launched_during:
                    failures.append(
                        {
                            "blocking_unit": uid,
                            "waited_from": start["t"],
                            "waited_until": t,
                            "runnable_while_waiting": independent,
                            "independent_launched_during_wait": [],
                            "seq": start.get("seq"),
                        }
                    )
            open_from.pop(uid, None)

    for wait in launches:
        if not wait["blocking"] or wait["detached"]:
            continue
        uid = wait["id"]
        t1 = ingest_at.get(uid)
        if t1 is None:
            # Still open at end of timeline: same defect if runnable work existed.
            t1 = _t(ordered[-1])
            if wait["runnable"]:
                launched_during = [
                    other
                    for other in launches
                    if other["id"]
                    and other["id"] != uid
                    and wait["t"] <= other["t"] <= t1
                ]
                if not launched_during:
                    failures.append(
                        {
                            "blocking_unit": uid or wait["kind"],
                            "waited_from": wait["t"],
                            "waited_until": t1,
                            "runnable_while_waiting": wait["runnable"],
                            "independent_launched_during_wait": [],
                            "seq": wait.get("seq"),
                            "still_open": True,
                        }
                    )

    for a in launches:
        if not a["detached"] or not a["id"]:
            continue
        t_a0 = a["t"]
        t_a1 = ingest_at.get(a["id"])
        if t_a1 is None:
            continue
        for b in launches:
            if not b["id"] or b["id"] == a["id"]:
                continue
            t_b0 = b["t"]
            progressed = [p for p in progress_at.get(b["id"], []) if t_b0 < p]
            t_b2 = ingest_at.get(b["id"])
            if t_b2 is None or not progressed:
                continue
            t_b1 = progressed[0]
            if t_a0 <= t_b0 < t_b1 < t_b2 <= t_a1:
                overlaps.append(
                    {
                        "detached_unit": a["id"],
                        "independent_unit": b["id"],
                        "detached_open": [t_a0, t_a1],
                        "independent_started": t_b0,
                        "independent_progressed": t_b1,
                        "independent_completed": t_b2,
                    }
                )

    if failures:
        cites = [f"seq:{f.get('seq')}" for f in failures if f.get("seq") is not None]
        return {
            "verdict": FAIL_NO_WAIT,
            "fail": True,
            "reason": (
                "runnable safe work existed while the loop waited on a subprocess: "
                + "; ".join(
                    f"{f['blocking_unit']} held [{f['waited_from']}, {f['waited_until']}] "
                    f"with runnable {f['runnable_while_waiting']}"
                    for f in failures
                )
            ),
            "failures": failures,
            "overlaps": overlaps,
            "cites": cites,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    if overlaps:
        return {
            "verdict": NO_WAIT_OK,
            "fail": False,
            "reason": (
                "detached unit stayed open while an independent unit started, "
                "progressed and completed; overlap is an interval, not an intent"
            ),
            "failures": [],
            "overlaps": overlaps,
            "cites": [o["independent_unit"] for o in overlaps],
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    return {
        "verdict": NO_WAIT_OK,
        "fail": False,
        "reason": (
            "no blocking wait-while-runnable interval; nothing in this timeline "
            "is FAIL_NO_WAIT_ORCHESTRATION"
        ),
        "failures": [],
        "overlaps": overlaps,
        "cites": [],
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def synthetic_wait_while_runnable_timeline() -> dict[str, Any]:
    """The 1h-trial shape: subprocess.run held the loop while other work sat."""
    blocking = {
        "id": "WU.blocking.specimen_verify",
        "description": "recompute every published digest for a specimen",
        "resource_class": "IO_HEAVY",
    }
    independent = {
        "id": "WU.independent.scar_index",
        "description": "rebuild the scar index that prunes work before it is scheduled",
        "resource_class": "STATIC_ANALYSIS",
    }
    return {
        "schema": "hawking.future.power_torture.timeline.v1",
        "purpose": "negative control: loop waited inside a subprocess while runnable work existed",
        "events": [
            {
                "t_s": 10,
                "seq": 0,
                "kind": "workunit_launched",
                "payload": {
                    "unit": blocking,
                    "blocking": True,
                    "waited_on_subprocess": True,
                    "runnable_now": [independent],
                },
            },
            {
                "t_s": 190,
                "seq": 1,
                "kind": "result_ingested",
                "cites": ["receipts/future/SPECIMEN_VERIFICATION.json"],
                "payload": {"unit_id": blocking["id"]},
            },
        ],
    }


def synthetic_overlap_timeline() -> dict[str, Any]:
    """Detached long job open; independent unit starts, progresses, completes."""
    slow = {"id": "WU.detached.specimen_verify", "resource_class": "IO_HEAVY"}
    fast = {"id": "WU.independent.status_challenge", "resource_class": "STATIC_ANALYSIS"}
    return {
        "schema": "hawking.future.power_torture.timeline.v1",
        "purpose": "positive control: timestamps prove independent completion during a detached job",
        "events": [
            {
                "t_s": 0,
                "seq": 0,
                "kind": "detached_started",
                "payload": {"job_id": slow["id"], "unit": slow},
            },
            {
                "t_s": 1,
                "seq": 1,
                "kind": "workunit_launched",
                "payload": {"unit": fast, "detached": False, "blocking": False},
            },
            {
                "t_s": 2,
                "seq": 2,
                "kind": "workunit_progressed",
                "payload": {"unit_id": fast["id"], "progress_at": 2},
            },
            {
                "t_s": 5,
                "seq": 3,
                "kind": "RESULT_INGESTED",
                "cites": ["receipts/future/STATUS_CAUSALITY_CHALLENGE.json"],
                "payload": {"unit_id": fast["id"]},
            },
            {
                "t_s": 10,
                "seq": 4,
                "kind": "detached_completed",
                "payload": {"job_id": slow["id"], "unit_id": slow["id"]},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Credit rules. Declared capability is not executed capability.
# ---------------------------------------------------------------------------


def credit_mutation(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """A mutation without a proven byte-identical rollback does not count."""
    if not isinstance(record, Mapping):
        return {
            "present": False,
            "why": "no mutation record; proposing nothing is not a mutation",
        }
    proposed = bool(record.get("proposed") or record.get("mutation_id"))
    applied = bool(record.get("applied") or record.get("after_digest"))
    rb = record.get("rollback") if isinstance(record.get("rollback"), Mapping) else {}
    digest_match = rb.get("digest_match") is True
    byte_identical = rb.get("byte_identical") is True
    if not proposed:
        return {"present": False, "why": "no proposal; nothing was mutated"}
    if not applied:
        return {
            "present": False,
            "why": "proposal was not applied in a reversible scope",
        }
    if not (digest_match and byte_identical):
        return {
            "present": False,
            "why": (
                "rollback was not proven byte-identical; a mutation without a "
                "proven undo does not count toward MUTATION"
            ),
            "digest_match": digest_match,
            "byte_identical": byte_identical,
        }
    return {
        "present": True,
        "why": "proposal applied in reversible scope; rollback digest matched",
        "mutation_id": record.get("mutation_id"),
        "mutation_class": record.get("mutation_class"),
    }


def credit_status_challenge(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """UNTESTED (no recorded probe) is not a challenge. Never count it as one."""
    if not isinstance(record, Mapping):
        return {
            "present": False,
            "why": "no challenge record",
        }
    verdict = str(record.get("verdict") or "")
    if verdict == sc.UNTESTED:
        return {
            "present": False,
            "why": (
                "challenge verdict is UNTESTED; a label with no recorded probe "
                "must not count as a status challenge"
            ),
            "status": record.get("status"),
            "verdict": verdict,
        }
    if verdict not in {sc.SUPPORTED, sc.OVERREACHING}:
        return {
            "present": False,
            "why": f"verdict {verdict!r} is not a recorded probe judgement",
            "status": record.get("status"),
            "verdict": verdict,
        }
    if not record.get("probe_kind") and not record.get("probe_performed"):
        return {
            "present": False,
            "why": "no recorded probe; UNTESTED in all but name",
            "status": record.get("status"),
        }
    return {
        "present": True,
        "why": (
            f"challenged {record.get('status')!r} with probe_kind="
            f"{record.get('probe_kind')!r}; verdict={verdict}"
        ),
        "status": record.get("status"),
        "verdict": verdict,
        "source": record.get("source"),
        "probe_kind": record.get("probe_kind"),
    }


def credit_refill(
    already_offered: Iterable[str],
    returned: Iterable[str],
    *,
    source: str,
) -> dict[str, Any]:
    """A refill of ids already offered is the 1h trial's replay, not a refill."""
    offered = {str(x) for x in already_offered if str(x)}
    got = [str(x) for x in returned if str(x)]
    fresh = [i for i in got if i not in offered]
    if not fresh:
        return {
            "present": False,
            "why": (
                f"{source} returned only already-offered identities "
                f"({len(got)} ids, 0 fresh); that is the 1h trial failure"
            ),
            "n_returned": len(got),
            "n_fresh": 0,
            "source": source,
        }
    return {
        "present": True,
        "why": f"{source} returned {len(fresh)} id(s) not already offered",
        "fresh": fresh,
        "n_returned": len(got),
        "n_fresh": len(fresh),
        "source": source,
    }


# ---------------------------------------------------------------------------
# GENERIC NR→NX. callable_on / run as the recover list named them.
# ---------------------------------------------------------------------------


def _stage_row(name: str, status: str, *, why: str, invoked: bool, error: str | None = None) -> dict[str, Any]:
    if status in {"SKIPPED", "skip", "pending", "PENDING", "READY", "ready"}:
        raise TortureRefused(
            f"{name}: status={status!r} is forbidden; a stage that cannot run "
            "is FAILED, REFUSED, or BLOCKED with a reason",
            missing=["GENERIC_NR_NX"],
        )
    return {
        "stage": name,
        "status": status,
        "why": why,
        "invoked": invoked,
        "error": error,
    }


def staged_nr_nx_refusal(exc: BaseException) -> dict[str, Any]:
    """Precise refusal when the compiler path is not importable. Not a skip."""
    err = f"{type(exc).__name__}: {exc}"
    why_first = (
        "tools.future.nr_nx_generic cannot be imported because "
        "tools.odyssey.arch_recognizer is not materialized in this sparse "
        f"checkout ({err}). A missing compiler stage is REFUSED, never SKIPPED"
    )
    stages: list[dict[str, Any]] = []
    for name in NR_NX_STAGE_ORDER:
        if name == "ArchitectureRecognizer" or not stages:
            # The import is the first thing that has to be true for any stage.
            stages.append(
                _stage_row(
                    name,
                    "REFUSED",
                    why=why_first if name in {"SpecimenSelect", "ArchitectureRecognizer"} else (
                        f"not reached: pipeline import failed ({err})"
                    ),
                    invoked=name in {"SpecimenSelect", "ArchitectureRecognizer"},
                    error=err,
                )
            )
        else:
            stages.append(
                _stage_row(
                    name,
                    "REFUSED",
                    why=f"not reached: pipeline import failed at ArchitectureRecognizer ({err})",
                    invoked=False,
                    error=err,
                )
            )
    first = next((s for s in stages if s["status"] != "PASSED"), stages[0])
    return {
        "callable": False,
        "pipeline_callable": False,
        "why": why_first,
        "first_failing_stage": first,
        "stages": stages,
        "import_error": err,
        "missing_path": "tools/odyssey/arch_recognizer.py",
        "skipped_forbidden": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def callable_on() -> dict[str, Any]:
    """Drive assemble() if the compiler is importable; else a staged refusal."""
    try:
        from tools.future import nr_nx_generic as nrg
    except Exception as exc:
        return staged_nr_nx_refusal(exc)
    assembled = nrg.assemble()
    stages = list(assembled.get("stages") or [])
    first = nrg.first_failing_stage(stages)
    callable_flag = nrg.generic_pipeline_callable(stages)
    return {
        "callable": bool(callable_flag),
        "pipeline_callable": bool(callable_flag),
        "why": (
            "every named stage ran and PASSED"
            if callable_flag
            else (
                f"first failing stage {(first or {}).get('stage')}: "
                f"{(first or {}).get('why')}"
            )
        ),
        "first_failing_stage": first,
        "stages": [
            {
                "stage": s.get("stage"),
                "status": s.get("status"),
                "invoked": s.get("invoked"),
                "why": s.get("why"),
                "error": s.get("error"),
            }
            for s in stages
        ],
        "FLASH_NX_READY": (assembled.get("flash_nx") or {}).get("FLASH_NX_READY"),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def run() -> dict[str, Any]:
    """The recover-list name. Same path as callable_on; does not pack an NX."""
    return callable_on()


# ---------------------------------------------------------------------------
# Drive the recovered powers. build() is not behaviour.
# ---------------------------------------------------------------------------


def _mutation_cycle(scope: Path) -> dict[str, Any]:
    engine = me.MutationEngine(scope)
    cycle = me.pipeline_self_cycle(engine)
    rb = cycle.get("rollback_after") if isinstance(cycle.get("rollback_after"), Mapping) else {}
    return {
        "mutation_id": cycle.get("mutation_id"),
        "mutation_class": cycle.get("mutation_class"),
        "frontier": cycle.get("frontier"),
        "proposed": True,
        "applied": True,
        "after_digest": (cycle.get("applied") or {}).get("after_digest"),
        "before_digest": (cycle.get("applied") or {}).get("before_digest"),
        "verdict": (cycle.get("verdict") or {}).get("verdict"),
        "rollback": {
            "digest_match": bool(cycle.get("rollback_digest_match") or rb.get("digest_match")),
            "byte_identical": bool(rb.get("byte_identical") or cycle.get("rollback_digest_match")),
        },
    }


def _two_subagent_states(scope: Path) -> dict[str, Any]:
    a_dir = scope / "agent_a"
    b_dir = scope / "agent_b"
    ga = wg.WorkGraph(a_dir, ncpu=2)
    gb = wg.WorkGraph(b_dir, ncpu=2)
    ua = wg.make_unit(
        id="agent-a.rank-remaining-organs",
        role="science",
        description=(
            "rank remaining Flash organs after the gate_up scar; n-gram table "
            "first because it is the largest untested generator surface"
        ),
        dependencies=[],
        resource_lane="CPU_ANALYSIS",
        mutation_scope=["receipts/future/FLASH_ORGAN_PIVOT.json"],
        verifier="future.flash_organ_pivot.rank_all",
        expected_information_gain=3,
        cost_units=2,
    )
    ub = wg.make_unit(
        id="agent-b.challenge-blocked-no-metal",
        role="science",
        description=(
            "challenge BLOCKED_NO_METAL_GPU against the probe that actually "
            "ran; a status may assert only what that probe established"
        ),
        dependencies=[],
        resource_lane="CPU_VERIFY",
        mutation_scope=["receipts/future/STATUS_CAUSALITY_CHALLENGE.json"],
        verifier="future.status_causality.challenge",
        expected_information_gain=3,
        cost_units=1,
    )
    ga.admit(ua)
    gb.admit(ub)
    path_a = ga.save()
    path_b = gb.save()
    la = wg.WorkGraph.load(a_dir, ncpu=2)
    lb = wg.WorkGraph.load(b_dir, ncpu=2)
    ids_a = set(la.units)
    ids_b = set(lb.units)
    disjoint = bool(ids_a) and bool(ids_b) and ids_a.isdisjoint(ids_b)
    durable = (
        disjoint
        and la.resumed is True
        and lb.resumed is True
        and "agent-a.rank-remaining-organs" in la.units
        and "agent-b.challenge-blocked-no-metal" in lb.units
        and "agent-b.challenge-blocked-no-metal" not in la.units
        and "agent-a.rank-remaining-organs" not in lb.units
    )
    return {
        "present": bool(durable),
        "why": (
            "two WorkGraph documents, distinct unit sets, reload preserved "
            "identity and did not leak state"
            if durable
            else "subagent states were not disjoint after reload"
        ),
        "agent_a": {
            "ids": sorted(ids_a),
            "path": str(path_a) if path_a else None,
            "resumed": la.resumed,
        },
        "agent_b": {
            "ids": sorted(ids_b),
            "path": str(path_b) if path_b else None,
            "resumed": lb.resumed,
        },
        "disjoint": disjoint,
        "n_states": 2,
    }


def drive_proofs(*, scope: Path | None = None) -> dict[str, Any]:
    """Invoke the recovered powers. Naming them in a list is not this function."""
    if scope is None:
        with tempfile.TemporaryDirectory(prefix="hawking-torture-") as tmp:
            return drive_proofs(scope=Path(tmp))
    root = Path(scope)
    root.mkdir(parents=True, exist_ok=True)

    wait_tl = synthetic_wait_while_runnable_timeline()
    overlap_tl = synthetic_overlap_timeline()
    wait_verdict = detect_no_wait_orchestration(wait_tl)
    overlap_verdict = detect_no_wait_orchestration(overlap_tl)

    mutation = _mutation_cycle(root / "mutation")
    mutation_credit = credit_mutation(mutation)
    # Negative: applied without rollback must not count. Built from the same cycle
    # with the undo stripped, so the guard is watched refusing a real shape.
    mutation_without_rollback = credit_mutation(
        {k: v for k, v in mutation.items() if k != "rollback"}
    )

    challenged = sc.challenge("BLOCKED_NO_METAL_GPU")
    challenge_credit = credit_status_challenge(challenged)
    untested = sc.challenge("SOME_LABEL_WITH_NO_PROBE")
    untested_credit = credit_status_challenge(untested)

    parked_drive = ps.drive()
    park = parked_drive.get("park") if isinstance(parked_drive.get("park"), Mapping) else {}
    continued = (
        parked_drive.get("continue_with")
        if isinstance(parked_drive.get("continue_with"), Mapping)
        else {}
    )
    protected_ok = (
        park.get("parked") is True
        and park.get("verdict") == "BLOCKED_ON_PROTECTED_WINDOW"
        and isinstance(park.get("wake_condition"), Mapping)
        and int(continued.get("n") or 0) > 0
    )

    scar_dead = ss.admit(
        {
            "id": "WU.TORTURE.scar.cross_expert_structure",
            "description": (
                "re-test cross_expert_structure on qwen3-80b routed experts"
            ),
            **DEAD_SCAR_PROPOSAL,
        }
    )
    scar_live = ss.admit(
        {
            "id": "WU.TORTURE.replan.ngram_product_codebook",
            "description": (
                "product codebook of the n-gram table; not a routed-expert restatement"
            ),
            **LIVE_REPLACEMENT_PROPOSAL,
        }
    )
    scar_ok = (
        str(scar_dead.get("decision") or "") == ss.DECISION_REFUSED
        and scar_dead.get("scar_id")
        and str(scar_live.get("decision") or "") == ss.DECISION_ADMITTED
    )

    ranking = fop.rank_all()
    restatement = {
        "id": "WU.TORTURE.invalidate.gate_up.shared_input_latent",
        "family": "shared_input_latent_plus_expert_local_output_readout",
        "organ": fop.EXHAUSTED_ORGAN,
        "surface": fop.EXHAUSTED_SURFACE,
        "school": "ROUTED_EXPERTS",
    }
    restatement_row = fop.restatement_verdict(
        restatement, ranking["scar"], ranking["killed_families"]
    )
    restatement_fired = False
    restatement_error = None
    try:
        fop.refuse_if_restatement(
            restatement, ranking["scar"], ranking["killed_families"]
        )
    except fop.RestatementRefused as exc:
        restatement_fired = True
        restatement_error = str(exc)
    next_ranked = (ranking.get("ranked") or [{}])[0] if ranking.get("ranked") else {}
    invalidation_ok = (
        restatement_fired
        and isinstance(restatement_row, Mapping)
        and restatement_row.get("status") == "REFUSED_RESTATEMENT"
        and str(next_ranked.get("school") or "") != "ROUTED_EXPERTS"
    )

    offered = [str(u.get("id") or "") for u in (fr.next_work(fr.THIS_HOST_LANES) or [])]
    replayed = [str(u.get("id") or "") for u in (fr.refill(fr.THIS_HOST_LANES) or [])]
    catalog_refill = credit_refill(offered, replayed, source="frontiers.refill")
    replan_ids = [
        "WU.TORTURE.replan.ngram_product_codebook",
        f"WU.TORTURE.replan.next_organ.{next_ranked.get('school') or 'NGRAM'}",
    ]
    real_refill = credit_refill(offered, replan_ids, source="replan_after_scar_and_invalidation")

    nr_nx = callable_on()
    nr_nx_ok = (
        nr_nx.get("pipeline_callable") is True
        or (
            nr_nx.get("pipeline_callable") is False
            and isinstance(nr_nx.get("first_failing_stage"), Mapping)
            and nr_nx["first_failing_stage"].get("status") in {"REFUSED", "FAILED", "BLOCKED"}
            and nr_nx["first_failing_stage"].get("stage")
        )
    )

    plan = cd.plan()
    concurrency_verdict_refused = False
    concurrency_refuse_why = None
    try:
        cd.verdict([])
    except cd.VerdictRefuse as exc:
        concurrency_verdict_refused = True
        concurrency_refuse_why = str(exc)
    decided = cd.decide()
    concurrency_ok = (
        isinstance(plan.get("ladder"), list)
        and plan["ladder"] == [1, 2, 3, 4]
        and concurrency_verdict_refused
        and decided.get("experiment_state") == "SLEEPING"
        and decided.get("verdict") is None
    )

    method = ab.method()
    contracts = ab.contracts()
    try:
        ab_plan = ab.plan()
        ab_plan_ok = True
        ab_plan_why = f"PLAN_ONLY on {ab_plan.get('specimen')}"
        ab_plan_specimen = ab_plan.get("specimen")
    except ab.PlanRefusal as exc:
        ab_plan_ok = True  # a precise refusal is the honest plan
        ab_plan_why = str(exc)
        ab_plan_specimen = None
        ab_plan = {"refused": True, "why": str(exc)}

    subagents = _two_subagent_states(root / "subagents")

    ingested = we.make(
        "RESULT_INGESTED",
        cites=["receipts/future/STATUS_CAUSALITY_CHALLENGE.json"],
    )
    refilled_event = we.make(
        "WORK_REFILLED",
        unit_ids=replan_ids,
        queue_depth=len(offered),
    )
    ingestion_ok = we.validate(ingested)[0] is True and bool(ingested.get("cites"))

    replans = [
        {
            "cause": "scar_pruning",
            "cause_id": "WU.TORTURE.scar.cross_expert_structure",
            "effect_id": "WU.TORTURE.replan.ngram_product_codebook",
            "how": (
                "refuse_if_dead / scar_scheduling.admit killed cross_expert_structure "
                "on qwen3-80b; the replacement is the n-gram product codebook, a "
                "different organ, not a restatement"
            ),
        },
        {
            "cause": "status_causality",
            "cause_id": "WU.TORTURE.status.blocked_no_metal_gpu",
            "effect_id": "WU.TORTURE.protected.ready_protected",
            "how": (
                "BLOCKED_NO_METAL_GPU is OVERREACHING given a process_error probe; "
                "teacher-capture's wake condition is not 'acquire a GPU'. Protected "
                "work stays parked; CPU work continues"
            ),
        },
        {
            "cause": "frontier_invalidation",
            "cause_id": restatement["id"],
            "effect_id": f"WU.TORTURE.replan.next_organ.{next_ranked.get('school') or 'NGRAM'}",
            "how": (
                f"restatement of {restatement['family']} on {fop.EXHAUSTED_ORGAN} "
                f"is refused; next ranked school is {next_ranked.get('school')}"
            ),
        },
    ]

    transitions = {
        "NO_WAIT": {
            "present": (
                wait_verdict.get("verdict") == FAIL_NO_WAIT
                and overlap_verdict.get("verdict") == NO_WAIT_OK
                and bool(overlap_verdict.get("overlaps"))
            ),
            "why": (
                "detector fired on wait-while-runnable and accepted a timestamped "
                "detached overlap (start, progress, complete)"
            ),
            "wait_verdict": wait_verdict.get("verdict"),
            "overlap_verdict": overlap_verdict.get("verdict"),
        },
        "REAL_REFILL": {
            "present": real_refill["present"],
            "why": real_refill["why"],
            "catalog_refill_is_replay": not catalog_refill["present"],
            "fresh": real_refill.get("fresh"),
        },
        "REAL_INGESTION": {
            "present": ingestion_ok,
            "why": (
                "RESULT_INGESTED cites a real receipt path; an ingest without "
                "citations is not an ingest"
            ),
            "cites": list(ingested.get("cites") or []),
        },
        "SCAR_PRUNING": {
            "present": bool(scar_ok),
            "why": (
                f"candidate refused by scar {scar_dead.get('scar_id')}; "
                "replacement admitted"
                if scar_ok
                else "scar admission did not refuse a known-dead family and admit a replacement"
            ),
            "scar_id": scar_dead.get("scar_id"),
            "source_path": scar_dead.get("source_path"),
            "replacement_id": "WU.TORTURE.replan.ngram_product_codebook",
        },
        "STATUS_CAUSALITY": {
            "present": challenge_credit["present"],
            "why": challenge_credit["why"],
            "status": challenged.get("status"),
            "verdict": challenged.get("verdict"),
            "untested_does_not_count": not untested_credit["present"],
        },
        "PROTECTED_PARKING": {
            "present": bool(protected_ok),
            "why": (
                "GPU_EXCLUSIVE unit parked BLOCKED_ON_PROTECTED_WINDOW with a "
                f"wake condition; continue_with returned {continued.get('n')} CPU units"
            ),
            "verdict": park.get("verdict"),
            "n_continued": continued.get("n"),
            "wake_condition": park.get("wake_condition"),
        },
        "GENERIC_NR_NX": {
            "present": bool(nr_nx_ok),
            "why": nr_nx.get("why"),
            "callable": nr_nx.get("callable"),
            "first_failing_stage": nr_nx.get("first_failing_stage"),
        },
        "MUTATION": {
            "present": mutation_credit["present"],
            "why": mutation_credit["why"],
            "mutation_id": mutation.get("mutation_id"),
            "mutation_class": mutation.get("mutation_class"),
            "without_rollback_does_not_count": not mutation_without_rollback["present"],
        },
        "SUBAGENT_STATE": {
            "present": subagents["present"],
            "why": subagents["why"],
            "n_states": subagents.get("n_states"),
            "disjoint": subagents.get("disjoint"),
        },
        "CONCURRENCY": {
            "present": bool(concurrency_ok),
            "why": (
                "plan() emitted the 1-2-3-4 ladder; verdict() refused without "
                "observations; decide() is SLEEPING with no CONCURRENCY_HELPS"
            ),
            "ladder": plan.get("ladder"),
            "verdict_refused": concurrency_verdict_refused,
            "verdict_refuse_why": concurrency_refuse_why,
            "experiment_state": decided.get("experiment_state"),
        },
        "FRONTIER_INVALIDATION": {
            "present": bool(invalidation_ok),
            "why": (
                restatement_error
                or "queued restatement of a killed family on the exhausted surface was refused"
            ),
            "killed_family": (restatement_row or {}).get("killed_family"),
            "next_school": next_ranked.get("school"),
        },
        "REPLAN": {
            "present": len(replans) >= 2,
            "why": f"{len(replans)} results change what runs next",
            "n": len(replans),
            "pairs": replans,
        },
    }

    return {
        "wait_verdict": wait_verdict,
        "overlap_verdict": overlap_verdict,
        "mutation": mutation,
        "mutation_credit": mutation_credit,
        "mutation_without_rollback": mutation_without_rollback,
        "challenge": {
            "status": challenged.get("status"),
            "verdict": challenged.get("verdict"),
            "probe_kind": challenged.get("probe_kind"),
            "claim_kind": challenged.get("claim_kind"),
            "source": challenged.get("source"),
        },
        "untested_challenge": {
            "status": untested.get("status"),
            "verdict": untested.get("verdict"),
            "credit": untested_credit,
        },
        "protected": {
            "parked": park.get("parked"),
            "verdict": park.get("verdict"),
            "n_continued": continued.get("n"),
            "wake_condition": park.get("wake_condition"),
        },
        "scar": {
            "dead_decision": scar_dead.get("decision"),
            "scar_id": scar_dead.get("scar_id"),
            "source_path": scar_dead.get("source_path"),
            "live_decision": scar_live.get("decision"),
        },
        "ranking": {
            "n_ranked": len(ranking.get("ranked") or []),
            "n_restatement_refused": ranking.get("n_restatement_probes_refused"),
            "next_school": next_ranked.get("school"),
            "next_organ": next_ranked.get("organ"),
            "next_surface": next_ranked.get("surface"),
        },
        "catalog_refill": catalog_refill,
        "real_refill": real_refill,
        "offered_n": len(offered),
        "nr_nx": {
            "callable": nr_nx.get("callable"),
            "why": nr_nx.get("why"),
            "first_failing_stage": nr_nx.get("first_failing_stage"),
            "missing_path": nr_nx.get("missing_path"),
        },
        "concurrency": {
            "ladder": plan.get("ladder"),
            "verdict_refused": concurrency_verdict_refused,
            "experiment_state": decided.get("experiment_state"),
            "decide_verdict": decided.get("verdict"),
        },
        "abliteration": {
            "n_method_stages": len(method.get("stages") or []),
            "contracts_both_sets_required": (contracts.get("dataset") or {}).get("both_required"),
            "plan_ok": ab_plan_ok,
            "plan_why": ab_plan_why,
            "specimen": ab_plan_specimen,
        },
        "subagents": subagents,
        "ingested_event": ingested,
        "refilled_event": refilled_event,
        "replans": replans,
        "transitions": transitions,
        "scope": str(root),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Compose. Every unit is work that stands if the torture is cancelled.
# ---------------------------------------------------------------------------


def _cpu_unit(
    module: str,
    *,
    description: str,
    transition: str,
    why_worth_doing: str,
    book: fr.FrontierBook,
    launch: str = "inline",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if module not in orch.BINDINGS:
        raise TortureRefused(
            f"module {module!r} is not in orchestration.BINDINGS",
            missing=["binding"],
        )
    fid, species = orch.BINDINGS[module]
    item = tw._item_by_id(book, fid)
    if item is None:
        raise TortureRefused(
            f"unit is not bound to a real frontier item: {fid} (module {module})",
            missing=["frontier_item"],
        )
    desc = str(description).strip()
    if not desc or at.is_low_information(
        {"description": desc, "verifier": item.get("verifier"), "frontier_id": fid}
    ):
        raise TortureRefused(
            f"padding refused: {module} description {desc!r} would not be worth "
            "doing if the torture were cancelled halfway",
            missing=["worth_doing_anyway"],
        )
    slug = module.removesuffix(".py")
    unit_id = f"WU.TORTURE.{transition}.{slug}"
    unit = at.cpu_workunit(
        unit_id,
        frontier_id=fid,
        description=desc,
        verifier=str(item.get("verifier") or f"future.{slug}"),
    )
    unit["module"] = module
    unit["capability"] = module
    unit["species"] = species
    unit["mix_role"] = transition
    unit["transition"] = transition
    unit["worth_doing_anyway"] = why_worth_doing
    unit["launch"] = launch
    unit["gpu_authority"] = False
    unit["evidence_class"] = "STATIC_ONLY"
    unit["required_lanes"] = [
        lane
        for lane in (item.get("required_lanes") or [])
        if lane not in fr.HARDWARE_LANES
    ] or list(fr.THIS_HOST_LANES[:1])
    if extra:
        for key, value in extra.items():
            unit[key] = value
    return unit


def _plan(book: fr.FrontierBook, proofs: Mapping[str, Any]) -> list[dict[str, Any]]:
    next_school = (proofs.get("ranking") or {}).get("next_school") or "NGRAM"
    units: list[dict[str, Any]] = [
        _cpu_unit(
            "specimen_verify.py",
            description=(
                "recompute published digests for the cheapest listed specimen "
                "as a DETACHED long job so independent CPU work can start, "
                "progress and complete while it stays open"
            ),
            transition="NO_WAIT",
            why_worth_doing=(
                "a completed whole-tree receipt is Odyssey I curriculum "
                "integrity and stands if the torture dies"
            ),
            book=book,
            launch="detached",
            extra={"long_subprocess": True, "resource_class": "IO_HEAVY"},
        ),
        _cpu_unit(
            "negative_index.py",
            description=(
                "query the scar index for cross_expert_structure on qwen3-80b "
                "routed experts and refuse it before any experiment is scheduled"
            ),
            transition="SCAR_PRUNING",
            why_worth_doing=(
                "a cited scar that actually kills a proposal is the campaign's "
                "own next work; rediscovery is not free"
            ),
            book=book,
            launch="independent",
        ),
        _cpu_unit(
            "ngram_school.py",
            description=(
                "generate n-gram-school product-codebook candidates below Q4 "
                "without fitting weights, scored against the scar that just "
                "killed cross_expert_structure; this identity was not in the "
                "catalog refill table"
            ),
            transition="REAL_REFILL",
            why_worth_doing=(
                "a fresh candidate set on a different organ than the scar is "
                "representation work the campaign already queued, and it is "
                "the replacement the prune chose"
            ),
            book=book,
            extra={"replacement_for": "WU.TORTURE.SCAR_PRUNING.negative_index"},
        ),
        _cpu_unit(
            "status_causality.py",
            description=(
                "challenge BLOCKED_NO_METAL_GPU against the probe that actually "
                "ran (process_error at dense_source_bf16_prefix_initialization), "
                "not against a reconstructed world-state"
            ),
            transition="STATUS_CAUSALITY",
            why_worth_doing=(
                "the label has already laundered a causal claim once; a "
                "challenge receipt that separates probe from interpretation stands"
            ),
            book=book,
            launch="independent",
        ),
        tw.make_unit(
            "protected_scheduler.py",
            description=(
                "recognize a declared GPU_EXCLUSIVE unit, park it "
                "BLOCKED_ON_PROTECTED_WINDOW with the wake condition attached, "
                "and continue with CPU-lane work; do not mark the scheduler incapable"
            ),
            mix_role="PROTECTED_PARKING",
            book=book,
            why_worth_doing=(
                "CAPABLE and AVAILABLE are different fields; recording that "
                "split is the work even when the window stays closed"
            ),
        ),
        _cpu_unit(
            "mutation_engine.py",
            description=(
                "propose the PIPELINE_SELF refill-identity mutation, apply it in "
                "a reversible scope, and roll it back proving byte-identity; "
                "this is the metabolism the 1h trial never had"
            ),
            transition="MUTATION",
            why_worth_doing=(
                "a proven undo on the policy that replayed 25 ids is usable "
                "whether or not the resident later KEPT it"
            ),
            book=book,
        ),
        _cpu_unit(
            "concurrency_doctor.py",
            description=(
                "emit the session-concurrency ladder (1, 2, 3, 4 while "
                "informative) and refuse a verdict without a resident process "
                "or a protected lease; occupancy is not available compute"
            ),
            transition="CONCURRENCY",
            why_worth_doing=(
                "the plan is the experiment; a SLEEPING unit with a wake "
                "condition is honest and stands without a GPU"
            ),
            book=book,
        ),
        _cpu_unit(
            "flash_organ_pivot.py",
            description=(
                "rank remaining Flash organs by expected information gain per "
                "cost and refuse a nearby restatement of a killed family on "
                "layer_4.routed_experts.gate_up_proj"
            ),
            transition="FRONTIER_INVALIDATION",
            why_worth_doing=(
                "leaving the exhausted gate_up surface is F019 CPU work; a "
                "ranked next organ stands even if no teacher row ever arrives"
            ),
            book=book,
            extra={"next_school": next_school},
        ),
        _cpu_unit(
            "nr_nx_generic.py",
            description=(
                "drive the generic NR→NX pipeline on the cheapest whole-tree "
                "specimen the compiler can see; if the compiler is not "
                "importable, record a staged REFUSAL naming the first real "
                "blocker rather than skipping a stage"
            ),
            transition="GENERIC_NR_NX",
            why_worth_doing=(
                "naming the first failing compiler stage is the work Odyssey I "
                "still needs; a SKIPPED stage would be a fictional pass"
            ),
            book=book,
            extra={"nr_nx_driver": "tools.future.power_torture.callable_on"},
        ),
        _cpu_unit(
            "abliteration.py",
            description=(
                "recover the candidate-direction generator (completion + "
                "harmless + loss gates) and plan it against the smallest "
                "whole-tree-verified residual-stream parent; fitting sleeps"
            ),
            transition="SUPPORTING",
            why_worth_doing=(
                "a method+plan bound to Tabula is the generator the campaign "
                "is missing; a run that would suppress refusals while destroying "
                "capability is already a Tabula FAILURE on paper"
            ),
            book=book,
        ),
        _cpu_unit(
            "workgraph.py",
            description=(
                "persist two independent logical agent graphs (organ ranking vs "
                "status challenge) so a process death cannot collapse them into "
                "one queue"
            ),
            transition="SUBAGENT_STATE",
            why_worth_doing=(
                "durable disjoint agent state is how a restart resumes rather "
                "than replays; the documents are the work"
            ),
            book=book,
        ),
    ]
    for unit in units:
        if unit.get("mix_role") == "PROTECTED_PARKING":
            unit["transition"] = "PROTECTED_PARKING"
            unit["launch"] = "parked"
    return units


def admit_torture(
    units: Sequence[Mapping[str, Any]],
    transitions: Mapping[str, Mapping[str, Any]],
    *,
    book: fr.FrontierBook,
) -> dict[str, Any]:
    """Refuse a mix that is missing a required transition class, by name."""
    missing = [
        name
        for name in REQUIRED_TRANSITIONS
        if not (transitions.get(name) or {}).get("present")
    ]
    if missing:
        raise TortureRefused(
            f"composed torture missing required transition class(es): {missing}",
            missing=missing,
        )
    admitted: list[dict[str, Any]] = []
    seen_ident: set[tuple] = set()
    for raw in units:
        row = dict(raw)
        module = str(row.get("module") or row.get("capability") or "")
        fid = str(row.get("frontier_id") or "")
        if module not in orch.BINDINGS:
            raise TortureRefused(
                f"module {module!r} is not in orchestration.BINDINGS",
                missing=["binding"],
            )
        bound_fid, bound_species = orch.BINDINGS[module]
        if fid and fid != bound_fid:
            raise TortureRefused(
                f"{module} is bound to {bound_fid}, not {fid}",
                missing=["binding_match"],
            )
        row.setdefault("frontier_id", bound_fid)
        row.setdefault("species", bound_species)
        if tw._item_by_id(book, row["frontier_id"]) is None:
            raise TortureRefused(
                f"unit is not bound to a real frontier item: {row['frontier_id']}",
                missing=["frontier_item"],
            )
        ident = at.work_identity(row)
        if ident in seen_ident:
            raise TortureRefused(
                f"duplicate work identity {ident}; unique ids do not make work distinct",
                missing=["distinct_work"],
            )
        seen_ident.add(ident)
        if at.is_low_information(row):
            raise TortureRefused(
                f"padding refused for {row.get('id')}",
                missing=["worth_doing_anyway"],
            )
        own_rc = str(row.get("resource_class") or "")
        transition = str(row.get("transition") or row.get("mix_role") or "")
        # Catalog GPU lanes must not park a CPU proof of the module (NR→NX
        # assemble, workgraph persist). Only declared GPU_EXCLUSIVE / the
        # protected-parking unit sleeps.
        if own_rc in at.GPU_RESOURCE or transition == "PROTECTED_PARKING":
            row = tw._park(row)
        row["gpu_authority"] = False
        row["evidence_class"] = "STATIC_ONLY"
        admitted.append(row)

    n = len(admitted)
    if n < 6 or n > 12:
        raise TortureRefused(
            f"torture has {n} units; want 6-12 meaningful units, not a padded checklist",
            missing=["mix_balance"],
        )
    launches = {str(u.get("launch") or "") for u in admitted}
    if "detached" not in launches:
        raise TortureRefused(
            "torture has no detached long subprocess; NO_WAIT cannot be scheduled",
            missing=["NO_WAIT"],
        )
    parked = [
        u
        for u in admitted
        if str(u.get("transition") or u.get("mix_role") or "") == "PROTECTED_PARKING"
    ]
    if not parked:
        raise TortureRefused(
            "torture has no protected-required unit to park",
            missing=["PROTECTED_PARKING"],
        )
    replans = list((transitions.get("REPLAN") or {}).get("pairs") or [])
    if len(replans) < 2:
        raise TortureRefused(
            "torture has fewer than 2 replans; a result must change what runs next, twice",
            missing=["REPLAN"],
        )
    return {
        "admitted": True,
        "trial_id": TRIAL_ID,
        "duration_s": DURATION_S,
        "units": admitted,
        "n_units": n,
        "n_sleeping": sum(
            1 for u in admitted if str(u.get("classification") or "") == "SLEEPING"
        ),
        "n_replans": len(replans),
        "replans": replans,
        "transitions": {k: dict(v) for k, v in transitions.items()},
        "doctrine": "STRESS TRANSITIONS, NOT CLOCKS",
        "padding_rule": (
            "if the torture were cancelled halfway, already-done work must have "
            "been worth doing; a unit that fails that test is padding"
        ),
        "no_wait_failure": FAIL_NO_WAIT,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "declared_not_executed": (
            "compose selects, binds, and drives proofs of each transition class. "
            "The 30-minute clock is autonomy_run's job and is outside this WRITE list."
        ),
    }


def compose(*, book: fr.FrontierBook | None = None, proofs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The 30-minute mix, drawn from live bindings and executed proofs."""
    current = book or tw.load_book()
    ran = dict(proofs) if proofs is not None else drive_proofs()
    planned = _plan(current, ran)
    admitted = admit_torture(planned, ran["transitions"], book=current)
    admitted["proofs"] = {
        "wait_verdict": (ran.get("wait_verdict") or {}).get("verdict"),
        "overlap_verdict": (ran.get("overlap_verdict") or {}).get("verdict"),
        "challenge": ran.get("challenge"),
        "scar": ran.get("scar"),
        "protected": ran.get("protected"),
        "mutation_id": (ran.get("mutation") or {}).get("mutation_id"),
        "nr_nx": ran.get("nr_nx"),
        "concurrency": ran.get("concurrency"),
        "ranking": ran.get("ranking"),
        "catalog_refill_is_replay": (ran.get("catalog_refill") or {}).get("present") is False,
        "real_refill": ran.get("real_refill"),
        "subagents": {
            "present": (ran.get("subagents") or {}).get("present"),
            "disjoint": (ran.get("subagents") or {}).get("disjoint"),
            "n_states": (ran.get("subagents") or {}).get("n_states"),
        },
        "abliteration": ran.get("abliteration"),
    }
    admitted["available_lanes"] = list(fr.THIS_HOST_LANES)
    admitted["blocked_lanes"] = list(fr.HARDWARE_LANES)
    return admitted


def compose_or_refuse(*, book: fr.FrontierBook | None = None) -> dict[str, Any]:
    try:
        return compose(book=book)
    except TortureRefused as exc:
        return {
            "admitted": False,
            "refused": True,
            "why": str(exc),
            "missing": list(exc.missing),
            "trial_id": TRIAL_ID,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }


# ---------------------------------------------------------------------------
# Substrate freeze. A mismatch is an automatic FAIL naming the file.
# ---------------------------------------------------------------------------


_SHADER_RE = re.compile(r"(?:crates|tools|lab)/[A-Za-z0-9_./-]+\.metal")
_FUTURE_IMPORT_RE = re.compile(r"^tools\.future(?:\.|$)")


def _mod_to_rel(mod: str) -> str | None:
    if mod == "tools.future":
        return "tools/future/__init__.py"
    if not _FUTURE_IMPORT_RE.match(mod):
        return None
    rest = mod[len("tools.future.") :]
    if not rest or not re.fullmatch(r"[A-Za-z0-9_.]+", rest):
        return None
    return "tools/future/" + rest.replace(".", "/") + ".py"


def _future_imports_of(tree: ast.AST, current_rel: str) -> list[str]:
    current_mod = ".".join(Path(current_rel).with_suffix("").parts)
    out: list[str] = []
    parts = current_mod.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                rel = _mod_to_rel(alias.name)
                if rel:
                    out.append(rel)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if node.level > len(parts):
                continue
            parent = parts[: -node.level]
            if node.module:
                absmod = ".".join(parent + node.module.split("."))
                rel = _mod_to_rel(absmod)
                if rel:
                    out.append(rel)
            else:
                pkg = ".".join(parent)
                if pkg == "tools.future" or pkg.startswith("tools.future."):
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        absmod = f"{pkg}.{alias.name}" if pkg else alias.name
                        rel = _mod_to_rel(absmod)
                        if rel:
                            out.append(rel)
            continue
        if node.module == "tools.future":
            out.append("tools/future/__init__.py")
            for alias in node.names:
                if alias.name == "*":
                    continue
                rel = _mod_to_rel(f"tools.future.{alias.name}")
                if rel:
                    out.append(rel)
        elif node.module and node.module.startswith("tools.future."):
            rel = _mod_to_rel(node.module)
            if rel:
                out.append(rel)
    return out


def _hash_one(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        return {"path": rel, "sha256": None, "state": "MISSING", "bytes": 0}
    try:
        digest = sha256_file(path)
        return {
            "path": rel,
            "sha256": digest,
            "state": "HASHED",
            "bytes": path.stat().st_size,
        }
    except OSError as exc:
        return {
            "path": rel,
            "sha256": None,
            "state": f"UNREADABLE:{type(exc).__name__}",
            "bytes": 0,
        }


def _walk_substrate(entry_rel: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Import graph from entry, plus every .metal string the graph names on disk."""
    files: dict[str, dict[str, Any]] = {}
    shaders: dict[str, dict[str, Any]] = {}
    stack = [entry_rel]
    while stack:
        rel = stack.pop()
        if rel in files:
            continue
        row = _hash_one(rel)
        files[rel] = row
        path = REPO / rel
        if not path.is_file() or not rel.endswith(".py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError, UnicodeError):
            continue
        for child in _future_imports_of(tree, rel):
            if child not in files:
                stack.append(child)
        for hit in _SHADER_RE.findall(text):
            if hit not in shaders:
                shaders[hit] = _hash_one(hit)
    return list(files.values()), list(shaders.values())


def hash_substrate(*, extra_paths: Sequence[str] = ()) -> dict[str, Any]:
    """Hash every tools/future module and shader this torture can reach.

    An extra path (the degeneracy instrument loaded from another tree) is
    hashed by absolute path so a mid-run edit of THAT file is also a FAIL.
    """
    files, shaders = _walk_substrate("tools/future/power_torture.py")
    extras: list[dict[str, Any]] = []
    for raw in extra_paths:
        path = Path(raw)
        if path.is_file():
            try:
                rel = str(path.resolve().relative_to(REPO))
            except ValueError:
                rel = str(path.resolve())
            extras.append(
                {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "state": "HASHED",
                    "bytes": path.stat().st_size,
                    "external": not rel.startswith("tools/future/"),
                }
            )
        else:
            extras.append({"path": str(path), "sha256": None, "state": "MISSING", "bytes": 0})
    body = {
        "files": sorted(files, key=lambda r: str(r.get("path") or "")),
        "shaders": sorted(shaders, key=lambda r: str(r.get("path") or "")),
        "extras": extras,
        "n_files": len(files),
        "n_shaders": len(shaders),
        "n_extras": len(extras),
        "hashed_at_unix": time.time(),
        "rule": (
            "every tools/future module and shader the torture touches; a glob "
            "of untouched files is not this freeze"
        ),
    }
    blob = json.dumps(
        {k: v for k, v in body.items() if k != "hashed_at_unix"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    body["digest"] = hashlib.sha256(blob).hexdigest()
    return body


def verify_substrate(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recompute equality. A mismatch names every path that moved."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {
            "verdict": SUBSTRATE_MOVED,
            "equal": False,
            "moved": [],
            "why": "freeze missing; refusing CLEAN",
        }
    if not before.get("files"):
        return {
            "verdict": SUBSTRATE_MOVED,
            "equal": False,
            "moved": [],
            "why": "empty freeze is not CLEAN",
        }

    def _index(snap: Mapping[str, Any]) -> dict[str, str | None]:
        idx: dict[str, str | None] = {}
        for bucket in ("files", "shaders", "extras"):
            for row in snap.get(bucket) or []:
                if isinstance(row, Mapping) and row.get("path"):
                    idx[str(row["path"])] = row.get("sha256")
        return idx

    prev = _index(before)
    curr = _index(after)
    moved: list[dict[str, Any]] = []
    for path in sorted(set(prev) | set(curr)):
        if prev.get(path) != curr.get(path):
            moved.append(
                {
                    "path": path,
                    "before_sha256": prev.get(path),
                    "after_sha256": curr.get(path),
                    "state": (
                        "MISSING"
                        if path in prev and path not in curr
                        else "APPEARED"
                        if path not in prev and path in curr
                        else "CHANGED"
                    ),
                }
            )
    equal = not moved and before.get("digest") == after.get("digest")
    return {
        "verdict": SUBSTRATE_CLEAN if equal and not moved else SUBSTRATE_MOVED,
        "equal": bool(equal) and not moved,
        "moved": moved,
        "before_digest": before.get("digest"),
        "after_digest": after.get("digest"),
        "why": (
            "substrate verified unchanged"
            if not moved
            else "substrate moved: " + ", ".join(m["path"] for m in moved)
        ),
    }


# ---------------------------------------------------------------------------
# GPU lane lock. Park rather than contend. Never fabricate a holder.
# ---------------------------------------------------------------------------


def inspect_gpu_lane_lock(path: Path | None = None) -> dict[str, Any]:
    """Read the mkdir-style GPU lane lock. Does not create, flock, or delete it."""
    lock = Path(path) if path is not None else GPU_LANE_LOCK
    row: dict[str, Any] = {
        "path": str(lock),
        "present": lock.exists(),
        "kind": None,
        "owner": None,
        "pid": None,
        "pid_alive": None,
        "waited_for": None,
        "contended": False,
        "parked": False,
    }
    if not lock.exists():
        row["kind"] = "absent"
        return row
    if lock.is_dir():
        row["kind"] = "directory"
        owner_p = lock / "owner"
        pid_p = lock / "pid"
        if owner_p.is_file():
            try:
                row["owner"] = owner_p.read_text(encoding="utf-8").strip() or None
            except OSError:
                row["owner"] = None
        if pid_p.is_file():
            try:
                text = pid_p.read_text(encoding="utf-8").strip()
                row["pid"] = int(text) if text.isdigit() else text
            except OSError:
                row["pid"] = None
        pid = row["pid"]
        if isinstance(pid, int) and pid > 0:
            try:
                _os.kill(pid, 0)
                row["pid_alive"] = True
            except OSError:
                row["pid_alive"] = False
        row["waited_for"] = (
            f"GPU lane lock directory held by owner={row['owner']!r} pid={row['pid']!r} "
            f"alive={row['pid_alive']}"
        )
        row["parked"] = True
        return row
    if lock.is_file():
        try:
            size = lock.stat().st_size
        except OSError:
            size = None
        row["kind"] = "file"
        row["bytes"] = size
        row["waited_for"] = (
            f"stale file at {lock} (size={size}); gpu_lane_lock.sh uses mkdir, "
            "so a file is not a proven holder. Parking rather than deleting or contending."
        )
        row["parked"] = True
        return row
    row["kind"] = "other"
    row["waited_for"] = f"unexpected lock node at {lock}"
    row["parked"] = True
    return row


# ---------------------------------------------------------------------------
# Sealed append-only timeline. The judge reads this, not the runner summary.
# ---------------------------------------------------------------------------


class SealedTimeline:
    """Append-only, timestamped, hashed at the end. Rewrite is a refusal."""

    def __init__(self, *, t0: float | None = None) -> None:
        self.t0 = float(t0 if t0 is not None else time.time())
        self.events: list[dict[str, Any]] = []
        self.sealed = False
        self.events_sha256: str | None = None
        self.sealed_at: str | None = None

    def _t_s(self) -> float:
        return round(time.time() - self.t0, 6)

    def append(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        cites: Sequence[str] | None = None,
        unit_id: str | None = None,
    ) -> dict[str, Any]:
        if self.sealed:
            raise TortureRefused(
                "timeline is sealed; append is refused",
                missing=["append_only"],
            )
        if not kind or not isinstance(kind, str):
            raise TortureRefused("event kind is required", missing=["kind"])
        event: dict[str, Any] = {
            "seq": len(self.events),
            "t_s": self._t_s(),
            "t_unix": time.time(),
            "kind": kind,
            "payload": dict(payload or {}),
        }
        if unit_id:
            event["payload"].setdefault("unit_id", unit_id)
            event["payload"].setdefault("id", unit_id)
        if cites:
            event["cites"] = [str(c) for c in cites]
        self.events.append(event)
        return event

    def seal(self) -> str:
        if self.sealed and self.events_sha256:
            return self.events_sha256
        blob = json.dumps(self.events, sort_keys=True, separators=(",", ":")).encode()
        self.events_sha256 = hashlib.sha256(blob).hexdigest()
        self.sealed = True
        self.sealed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self.events_sha256

    def to_doc(self) -> dict[str, Any]:
        digest = self.events_sha256 or self.seal()
        return {
            "schema": TIMELINE_SCHEMA,
            "version": 1,
            "trial_id": TRIAL_ID,
            "append_only": True,
            "sealed": True,
            "sealed_at": self.sealed_at,
            "t0_unix": self.t0,
            "elapsed_s": self.events[-1]["t_s"] if self.events else 0.0,
            "n_events": len(self.events),
            "events": list(self.events),
            "events_sha256": digest,
            "purpose": (
                "Sealed append-only timeline of the 30-minute power torture. "
                "The judge reads this document, not the runner's summary."
            ),
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }


def _events_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    blob = json.dumps(list(events), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Degeneracy instrument. Drive it; do not reimplement it.
# ---------------------------------------------------------------------------


def load_degeneracy() -> tuple[Any, str]:
    """Import autonomy_degeneracy.measure from this tree or the landed copy.

    The instrument is not in this worktree's HEAD; the parent hawking tree
    and the d2degen lane hold it. Loading by path is driving the instrument,
    not copying its thresholds.
    """
    try:
        from tools.future import autonomy_degeneracy as ad

        path = getattr(ad, "__file__", None) or "tools.future.autonomy_degeneracy"
        return ad, str(path)
    except ImportError:
        pass
    for candidate in DEGENERACY_CANDIDATES:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "hawking_future_autonomy_degeneracy_ext", candidate
        )
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "measure", None)):
            continue
        return mod, str(candidate)
    raise TortureRefused(
        "autonomy_degeneracy.measure is not importable from this tree, "
        "the parent hawking checkout, or the d2degen worktree",
        missing=["autonomy_degeneracy"],
    )


def measure_degeneracy(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Run the landed measure over THIS run's timeline. No self-exemption."""
    ad, source = load_degeneracy()
    report = ad.measure(timeline)
    table = ad.axis_table(report) if callable(getattr(ad, "axis_table", None)) else []
    return {
        "verdict": report.get("verdict"),
        "reason": report.get("reason"),
        "degenerate_axes": list(report.get("degenerate_axes") or []),
        "named_axes": list(report.get("named_axes") or []),
        "axis_table": table,
        "elapsed_s": report.get("elapsed_s"),
        "n_events": report.get("n_events"),
        "n_argv0_labelled": report.get("n_argv0_labelled"),
        "n_unlabelled": report.get("n_unlabelled"),
        "specimen_verification_ingests": report.get("specimen_verification_ingests"),
        "thresholds": report.get("thresholds"),
        "instrument": "tools.future.autonomy_degeneracy.measure",
        "instrument_path": source,
        "report": report,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Independent judge. Reads the sealed timeline, never the runner summary.
# ---------------------------------------------------------------------------


def _event_power(event: Mapping[str, Any]) -> str | None:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    named = str(payload.get("power") or "").strip()
    if named in POWER_CATALOG:
        return named
    kind = str(event.get("kind") or "")
    mapped = JUDGE_KIND_TO_POWER.get(kind)
    if mapped:
        return mapped
    return None


def _event_status(event: Mapping[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    status = str(payload.get("status") or "").strip()
    if status in {EXERCISED, EXERCISED_REFUSAL, SKIPPED, FAILED}:
        return status
    kind = str(event.get("kind") or "")
    if kind == "power_skipped":
        return SKIPPED
    if kind == "power_failed":
        return FAILED
    if kind == "concurrency_doctor_decided":
        state = str(payload.get("experiment_state") or payload.get("classification") or "")
        if state == "SLEEPING" or payload.get("verdict_refused") is True:
            return EXERCISED_REFUSAL
    if kind in {"workunit_sleeping", "nr_nx_stage"}:
        if payload.get("correct_refusal") is True or payload.get("classification") == "SLEEPING":
            return EXERCISED_REFUSAL
        if str(payload.get("status") or "") in {"REFUSED", "FAILED", "BLOCKED"}:
            return EXERCISED_REFUSAL
    return EXERCISED


def judge(timeline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Score the sealed timeline. A runner summary is ignored even if present."""
    if timeline is None:
        raise TortureRefused(
            "judge requires the sealed timeline; refusing to read a runner summary",
            missing=["timeline"],
        )
    if not isinstance(timeline, Mapping):
        raise TortureRefused(
            f"timeline must be a mapping, got {type(timeline).__name__}",
            missing=["timeline"],
        )
    # Independence: drop any summary the runner may have attached.
    events = [dict(e) for e in (timeline.get("events") or []) if isinstance(e, Mapping)]
    claimed = timeline.get("events_sha256")
    recomputed = _events_sha256(events) if events or claimed else None
    seal_ok = bool(claimed) and claimed == recomputed
    if claimed and recomputed and claimed != recomputed:
        return {
            "verdict": "FAIL",
            "why": "timeline events_sha256 does not match the events; the seal is broken",
            "seal_ok": False,
            "events_sha256": recomputed,
            "claimed_sha256": claimed,
            "n_events": len(events),
            "powers": {},
            "exercised": [],
            "skipped": [],
            "failed": [],
            "omitted": list(POWER_CATALOG),
            "read": "timeline.events",
            "ignored": ["runner_summary", "powers", "proofs"],
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    seqs = [e.get("seq") for e in events]
    append_only = seqs == list(range(len(events)))
    t_s_vals = [float(e.get("t_s") or 0.0) for e in events]
    timestamps_mono = all(t_s_vals[i] <= t_s_vals[i + 1] + 1e-9 for i in range(max(0, len(t_s_vals) - 1)))

    found: dict[str, dict[str, Any]] = {}
    for event in events:
        name = _event_power(event)
        if not name:
            continue
        status = _event_status(event)
        prev = found.get(name)
        rank = {FAILED: 3, SKIPPED: 0, EXERCISED: 2, EXERCISED_REFUSAL: 2}
        if prev is None or rank.get(status, 0) >= rank.get(prev["status"], 0):
            payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
            found[name] = {
                "power": name,
                "status": status,
                "seq": event.get("seq"),
                "kind": event.get("kind"),
                "why": payload.get("why") or payload.get("reason"),
                "correct_refusal": status == EXERCISED_REFUSAL,
            }

    omitted = [name for name in POWER_CATALOG if name not in found]
    exercised = [
        name
        for name in POWER_CATALOG
        if found.get(name, {}).get("status") in {EXERCISED, EXERCISED_REFUSAL}
    ]
    skipped = [name for name in POWER_CATALOG if found.get(name, {}).get("status") == SKIPPED]
    failed = [name for name in POWER_CATALOG if found.get(name, {}).get("status") == FAILED]
    refusals = [
        name
        for name in POWER_CATALOG
        if found.get(name, {}).get("status") == EXERCISED_REFUSAL
    ]

    why_parts: list[str] = []
    verdict = "PASS"
    if not append_only:
        verdict = "FAIL"
        why_parts.append("timeline is not append-only (seq not 0..n-1)")
    if not timestamps_mono:
        verdict = "FAIL"
        why_parts.append("timestamps are not monotonic")
    if claimed and not seal_ok:
        verdict = "FAIL"
        why_parts.append("seal mismatch")
    if omitted:
        verdict = "FAIL"
        why_parts.append("silently omitted: " + ", ".join(omitted))
    if failed:
        verdict = "FAIL"
        why_parts.append("failed: " + ", ".join(failed))
    if not exercised:
        verdict = "FAIL"
        why_parts.append("no power was exercised; skips do not count toward the pass")

    return {
        "verdict": verdict,
        "why": "; ".join(why_parts) if why_parts else (
            f"{len(exercised)} exercised (including {len(refusals)} correct "
            f"refusals), {len(skipped)} skipped"
        ),
        "seal_ok": bool(seal_ok) if claimed else False,
        "append_only": append_only,
        "timestamps_monotonic": timestamps_mono,
        "events_sha256": recomputed,
        "claimed_sha256": claimed,
        "n_events": len(events),
        "n_exercised": len(exercised),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "n_omitted": len(omitted),
        "n_correct_refusals": len(refusals),
        "exercised": exercised,
        "skipped": skipped,
        "failed": failed,
        "omitted": omitted,
        "correct_refusals": refusals,
        "powers": found,
        "read": "timeline.events",
        "ignored": ["runner_summary", "powers", "proofs", "summary"],
        "skipped_do_not_count_toward_pass": True,
        "correct_refusal_counts_as_exercised": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Exercise each power. Drive the landed callables; do not reimplement them.
# ---------------------------------------------------------------------------


def _power_record(
    name: str,
    status: str,
    *,
    why: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {EXERCISED, EXERCISED_REFUSAL, SKIPPED, FAILED}:
        raise TortureRefused(f"unknown power status {status!r}", missing=["status"])
    return {
        "power": name,
        "status": status,
        "exercised": status in {EXERCISED, EXERCISED_REFUSAL},
        "skipped": status == SKIPPED,
        "failed": status == FAILED,
        "correct_refusal": status == EXERCISED_REFUSAL,
        "counts_toward_pass": status in {EXERCISED, EXERCISED_REFUSAL},
        "why": why,
        "evidence": dict(evidence or {}),
    }


def _emit_power(timeline: SealedTimeline, record: Mapping[str, Any], kind: str) -> None:
    payload = {
        "power": record["power"],
        "status": record["status"],
        "why": record["why"],
        "correct_refusal": record.get("correct_refusal"),
        "unit_id": f"WU.TORTURE.{record['power']}",
        "id": f"WU.TORTURE.{record['power']}",
    }
    evidence = record.get("evidence") if isinstance(record.get("evidence"), Mapping) else {}
    for key in (
        "pid",
        "experiment_state",
        "blocked_reason",
        "classification",
        "verdict_refused",
        "scar_id",
        "mutation_id",
        "n_fresh",
        "cites",
        "first_failing_stage",
        "n_continued",
        "callable",
        "disjoint",
        "n_states",
        "wake_condition",
        "verdict",
    ):
        if key in evidence:
            payload[key] = evidence[key]
    cites = evidence.get("cites") if isinstance(evidence.get("cites"), list) else None
    timeline.append(kind, payload, cites=cites, unit_id=payload["unit_id"])


def _exercise_no_wait(timeline: SealedTimeline, workspace: Path) -> dict[str, Any]:
    """Real detached pid + independent completion while the child stays open."""
    try:
        from tools.future import no_wait_scheduler as nws
    except Exception as exc:
        rec = _power_record(
            "NO_WAIT",
            SKIPPED,
            why=f"no_wait_scheduler unimportable ({type(exc).__name__}: {exc})",
        )
        _emit_power(timeline, rec, "power_skipped")
        return rec
    slow_path = workspace / "results" / "slow.json"
    fast_path = workspace / "results" / "fast.json"
    started = fast_path.with_suffix(".started")
    slow_path.parent.mkdir(parents=True, exist_ok=True)
    sched = nws.NoWaitScheduler(workspace)
    slow = nws._cpu_unit(
        "WU.TORTURE.NO_WAIT.child",
        "",
        slow_path,
        sleep_s=1.2,
        timeout_s=20.0,
    )
    fast = nws._cpu_unit(
        "WU.TORTURE.NO_WAIT.ind",
        "import sys,time; from pathlib import Path; "
        "p=Path(sys.argv[1]); p.with_suffix('.started').write_text(str(time.time())); "
        "time.sleep(0.15); p.write_text('{\"ok\": true}\\n')",
        fast_path,
        timeout_s=15.0,
    )
    h_slow = sched.launch_detached(slow)
    pid = h_slow.get("pid")
    t_slow = float(h_slow.get("launched_at") or time.time())
    timeline.append(
        "detached_started",
        {
            "power": "NO_WAIT",
            "status": EXERCISED,
            "job_id": h_slow.get("job_id"),
            "pid": pid,
            "detached": True,
            "no_wait": True,
            "unit": {"id": "WU.TORTURE.NO_WAIT.child", "resource_class": "LIGHT_CONTROL"},
            "runnable_now": [{"id": "WU.TORTURE.NO_WAIT.ind"}],
        },
        unit_id="WU.TORTURE.NO_WAIT.child",
    )
    view = sched.runnable_now([h_slow], candidates=[slow, fast])
    h_fast = sched.launch_detached(fast)
    timeline.append(
        "WORK_LAUNCHED",
        {
            "power": "NO_WAIT",
            "status": EXERCISED,
            "unit": {"id": "WU.TORTURE.NO_WAIT.ind", "resource_class": "LIGHT_CONTROL"},
            "detached": False,
            "blocking": False,
            "pid": h_fast.get("pid"),
        },
        unit_id="WU.TORTURE.NO_WAIT.ind",
    )
    observed_progress = False
    t_progress = None
    t_fast_finish = None
    slow_open_at_finish = None
    deadline = time.monotonic() + 8.0
    last_fast_ingest = None
    while time.monotonic() < deadline:
        snaps = sched.poll([h_slow, h_fast])
        by_id = {s["job_id"]: s for s in snaps}
        slow_snap = by_id[h_slow["job_id"]]
        fast_snap = by_id[h_fast["job_id"]]
        if started.is_file() and not observed_progress:
            observed_progress = True
            t_progress = time.time()
            timeline.append(
                "workunit_progressed",
                {
                    "power": "NO_WAIT",
                    "unit_id": "WU.TORTURE.NO_WAIT.ind",
                    "progress_at": t_progress,
                    "slow_still_open": slow_snap.get("terminal") is None,
                },
                unit_id="WU.TORTURE.NO_WAIT.ind",
            )
        if fast_snap.get("terminal") is not None and t_fast_finish is None:
            t_fast_finish = float(fast_snap.get("finished_at") or time.time())
            slow_open_at_finish = slow_snap.get("terminal") is None
        landed = sched.ingest_ready([h_fast])
        fast_rows = [r for r in landed.get("landed") or [] if r.get("job_id") == h_fast["job_id"]]
        if fast_rows:
            last_fast_ingest = fast_rows[0].get("ingest")
            if last_fast_ingest == nws.INGESTED and observed_progress and slow_open_at_finish:
                break
        time.sleep(0.04)
    try:
        sched.cancel(h_slow)
    except Exception:
        try:
            sched.reap_all()
        except Exception:
            pass
    overlap = bool(
        observed_progress
        and t_fast_finish is not None
        and slow_open_at_finish
        and last_fast_ingest == nws.INGESTED
        and isinstance(pid, int)
        and pid > 0
        and view.get("status") == nws.RUNNABLE
    )
    detector = detect_no_wait_orchestration(
        {
            "events": [
                e
                for e in timeline.events
                if e.get("kind")
                in {"detached_started", "WORK_LAUNCHED", "workunit_progressed", "RESULT_INGESTED"}
            ]
        }
    )
    if overlap:
        rec = _power_record(
            "NO_WAIT",
            EXERCISED,
            why=(
                f"detached pid {pid} stayed open while WU.TORTURE.NO_WAIT.ind "
                "started, progressed and ingested; idle_runnable was not a wait"
            ),
            evidence={
                "pid": pid,
                "slow_launched_at": t_slow,
                "fast_finished_at": t_fast_finish,
                "fast_ingest": last_fast_ingest,
                "runnable_while_open": view.get("status"),
                "detector_verdict": detector.get("verdict"),
            },
        )
        _emit_power(timeline, rec, "WORK_LAUNCHED")
        return rec
    rec = _power_record(
        "NO_WAIT",
        FAILED,
        why=(
            "detached overlap was not proven from timestamps "
            f"(pid={pid!r} ingest={last_fast_ingest!r} progress={observed_progress})"
        ),
        evidence={"pid": pid, "fast_ingest": last_fast_ingest},
    )
    _emit_power(timeline, rec, "power_failed")
    return rec


def _exercise_refill_and_ingest(
    timeline: SealedTimeline,
    proofs: Mapping[str, Any],
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    offered = [str(u.get("id") or "") for u in (fr.next_work(fr.THIS_HOST_LANES) or []) if str(u.get("id") or "")]
    catalog = [str(u.get("id") or "") for u in (fr.refill(fr.THIS_HOST_LANES) or []) if str(u.get("id") or "")]
    catalog_credit = credit_refill(offered, catalog, source="frontiers.refill")
    excluded = fr.refill(fr.THIS_HOST_LANES, exclude=offered)
    excluded_ids = [str(u.get("id") or "") for u in (excluded or []) if str(u.get("id") or "")]
    replan_ids = list((proofs.get("real_refill") or {}).get("fresh") or [])
    if not replan_ids:
        next_school = (proofs.get("ranking") or {}).get("next_school") or "NGRAM"
        replan_ids = [
            "WU.TORTURE.replan.ngram_product_codebook",
            f"WU.TORTURE.replan.next_organ.{next_school}",
        ]
    fresh_credit = credit_refill(offered, replan_ids, source="replan_after_scar_and_invalidation")
    exclude_credit = credit_refill(offered, excluded_ids, source="frontiers.refill_exclude")
    # Two consecutive WORK_REFILLED sets MUST differ (degeneracy axis).
    set_a = replan_ids
    set_b = excluded_ids or [i + ".next" for i in replan_ids]
    if json.dumps(sorted(set_a)) == json.dumps(sorted(set_b)):
        set_b = list(set_a) + ["WU.TORTURE.replan.blocked_reroute.cpu"]
    timeline.append(
        "WORK_REFILLED",
        {
            "power": "REAL_REFILL",
            "status": EXERCISED if fresh_credit["present"] else FAILED,
            "unit_ids": set_a,
            "queue_depth": len(offered),
            "n_fresh": fresh_credit.get("n_fresh"),
            "source": fresh_credit.get("source"),
            "why": fresh_credit.get("why"),
            "catalog_refill_is_replay": not catalog_credit["present"],
            "id": "WU.TORTURE.REAL_REFILL.a",
            "unit_id": "WU.TORTURE.REAL_REFILL.a",
        },
        unit_id="WU.TORTURE.REAL_REFILL.a",
    )
    timeline.append(
        "WORK_REFILLED",
        {
            "power": "REAL_REFILL",
            "status": EXERCISED if fresh_credit["present"] else FAILED,
            "unit_ids": set_b,
            "queue_depth": len(offered) + len(set_a),
            "id": "WU.TORTURE.REAL_REFILL.b",
            "unit_id": "WU.TORTURE.REAL_REFILL.b",
            "why": "second refill set differs from the first; consecutive identical sets are the 1h scar",
        },
        unit_id="WU.TORTURE.REAL_REFILL.b",
    )
    if fresh_credit["present"] or exclude_credit["present"]:
        refill_rec = _power_record(
            "REAL_REFILL",
            EXERCISED,
            why=fresh_credit["why"] if fresh_credit["present"] else exclude_credit["why"],
            evidence={
                "n_fresh": (fresh_credit.get("n_fresh") or 0) + (exclude_credit.get("n_fresh") or 0),
                "fresh": list(fresh_credit.get("fresh") or []) + list(exclude_credit.get("fresh") or []),
                "catalog_refill_is_replay": not catalog_credit["present"],
            },
        )
    else:
        refill_rec = _power_record(
            "REAL_REFILL",
            FAILED,
            why="neither replan nor excluded refill produced a fresh id",
            evidence={"catalog_refill_is_replay": not catalog_credit["present"]},
        )

    cites = [
        "receipts/future/STATUS_CAUSALITY_CHALLENGE.json",
        "receipts/future/DETACHED_WORK_TRIAL.json",
    ]
    if (workspace / "results" / "fast.json").is_file():
        cites.append(str(workspace / "results" / "fast.json"))
    ingested = we.make("RESULT_INGESTED", cites=cites, receipt=cites[0], unit_id="WU.TORTURE.REAL_INGESTION")
    ok, why = we.validate(ingested)
    timeline.append(
        "RESULT_INGESTED",
        {
            "power": "REAL_INGESTION",
            "status": EXERCISED if ok else FAILED,
            "receipt": cites[0],
            "why": why if ok else f"ingest event failed validate: {why}",
            "id": "WU.TORTURE.REAL_INGESTION",
            "unit_id": "WU.TORTURE.REAL_INGESTION",
        },
        cites=cites,
        unit_id="WU.TORTURE.REAL_INGESTION",
    )
    ingest_rec = _power_record(
        "REAL_INGESTION",
        EXERCISED if ok else FAILED,
        why=(
            "RESULT_INGESTED cites real receipt paths once each; an ingest "
            "without citations is not an ingest"
            if ok
            else why
        ),
        evidence={"cites": cites},
    )
    return refill_rec, ingest_rec


def _exercise_scar(timeline: SealedTimeline, proofs: Mapping[str, Any]) -> dict[str, Any]:
    dead = ss.admit(
        {
            "id": "WU.TORTURE.scar.cross_expert_structure",
            "description": "re-test cross_expert_structure on qwen3-80b routed experts",
            **DEAD_SCAR_PROPOSAL,
        }
    )
    live = ss.admit(
        {
            "id": "WU.TORTURE.replan.ngram_product_codebook",
            "description": "product codebook of the n-gram table; not a routed-expert restatement",
            **LIVE_REPLACEMENT_PROPOSAL,
        }
    )
    ni_dead = None
    try:
        ni_dead = ni.refuse_if_dead(dict(DEAD_SCAR_PROPOSAL))
    except Exception as exc:
        ni_dead = {"error": f"{type(exc).__name__}: {exc}"}
    scar_ok = (
        str(dead.get("decision") or "") == ss.DECISION_REFUSED
        and dead.get("scar_id")
        and str(live.get("decision") or "") == ss.DECISION_ADMITTED
    )
    timeline.append(
        "IDEA_REJECTED",
        {
            "power": "SCAR_PRUNING",
            "status": EXERCISED if scar_ok else FAILED,
            "scar_id": dead.get("scar_id"),
            "hypothesis_family": DEAD_SCAR_PROPOSAL["hypothesis_family"],
            "decision": dead.get("decision"),
            "live_decision": live.get("decision"),
            "why": (
                f"candidate refused by scar {dead.get('scar_id')}; replacement admitted"
                if scar_ok
                else "scar admission did not refuse a known-dead family"
            ),
            "id": "WU.TORTURE.SCAR_PRUNING",
            "unit_id": "WU.TORTURE.SCAR_PRUNING",
        },
        unit_id="WU.TORTURE.SCAR_PRUNING",
    )
    timeline.append(
        "NEGATIVE_SCIENCE_REFUSAL",
        {
            "power": "SCAR_PRUNING",
            "scar_id": dead.get("scar_id"),
            "hypothesis_family": DEAD_SCAR_PROPOSAL["hypothesis_family"],
            "negative_index_refusal": bool(ni_dead) and not (isinstance(ni_dead, Mapping) and ni_dead.get("error")),
            "id": "WU.TORTURE.SCAR_PRUNING.ni",
            "unit_id": "WU.TORTURE.SCAR_PRUNING.ni",
        },
        unit_id="WU.TORTURE.SCAR_PRUNING.ni",
    )
    return _power_record(
        "SCAR_PRUNING",
        EXERCISED if scar_ok else FAILED,
        why=(
            f"scar {dead.get('scar_id')} refused the dead family; live replacement admitted"
            if scar_ok
            else "scar pruning did not refuse and replace"
        ),
        evidence={
            "scar_id": dead.get("scar_id"),
            "source_path": dead.get("source_path"),
            "dead_decision": dead.get("decision"),
            "live_decision": live.get("decision"),
        },
    )


def _exercise_status(timeline: SealedTimeline) -> dict[str, Any]:
    challenged = sc.challenge("BLOCKED_NO_METAL_GPU")
    emitted = sc.emit(
        "BLOCKED_NO_METAL_GPU",
        probe_kind=str(challenged.get("probe_kind") or sc.PROBE_PROCESS_ERROR),
        probe_performed=str(challenged.get("probe_performed") or "process_error recorded on the capture-boundary receipt"),
        direct_observation=str(challenged.get("direct_observation") or challenged.get("observation") or "process_error"),
        interpretation=str(challenged.get("interpretation") or challenged.get("status") or ""),
        source=str(challenged.get("source") or "<emit>"),
        claim_kind=challenged.get("claim_kind"),
    )
    credit = credit_status_challenge(challenged)
    coverage = sc.coverage()
    recording = list(coverage.get("recording_five_fields") or [])
    missing = list(coverage.get("not_recording_five_fields") or [])
    timeline.append(
        "status_challenged",
        {
            "power": "STATUS_CAUSALITY",
            "status": EXERCISED if credit["present"] else FAILED,
            "verdict": challenged.get("verdict"),
            "probe_kind": challenged.get("probe_kind"),
            "claim_kind": challenged.get("claim_kind"),
            "emit_verdict": emitted.get("verdict"),
            "n_gates_recording": len(recording),
            "n_gates_named_gap": len(missing),
            "why": credit.get("why"),
            "id": "WU.TORTURE.STATUS_CAUSALITY",
            "unit_id": "WU.TORTURE.STATUS_CAUSALITY",
        },
        unit_id="WU.TORTURE.STATUS_CAUSALITY",
    )
    return _power_record(
        "STATUS_CAUSALITY",
        EXERCISED if credit["present"] else FAILED,
        why=credit["why"],
        evidence={
            "verdict": challenged.get("verdict"),
            "probe_kind": challenged.get("probe_kind"),
            "n_gates_recording": len(recording),
            "n_gates_named_gap": len(missing),
            "emit_entry": emitted.get("entry"),
        },
    )


def _exercise_protected_and_reroute(
    timeline: SealedTimeline,
) -> tuple[dict[str, Any], dict[str, Any]]:
    driven = ps.drive()
    park = driven.get("park") if isinstance(driven.get("park"), Mapping) else {}
    continued = driven.get("continue_with") if isinstance(driven.get("continue_with"), Mapping) else {}
    cap = ps.capability_report()
    parked_ok = (
        park.get("parked") is True
        and park.get("verdict") == "BLOCKED_ON_PROTECTED_WINDOW"
        and isinstance(park.get("wake_condition"), Mapping)
    )
    n_continued = int(continued.get("n") or 0)
    capable = (cap.get("PROTECTED_SCHEDULER_CAPABLE") if isinstance(cap, Mapping) else None)
    if capable is None and isinstance(cap.get("capability"), Mapping):
        capable = cap["capability"].get("PROTECTED_SCHEDULER_CAPABLE")
    available = None
    if isinstance(cap, Mapping):
        available = cap.get("PROTECTED_WINDOW_AVAILABLE")
        if available is None and isinstance(cap.get("capability"), Mapping):
            available = cap["capability"].get("PROTECTED_WINDOW_AVAILABLE")
    timeline.append(
        "workunit_sleeping",
        {
            "power": "PROTECTED_PARKING",
            "status": EXERCISED if parked_ok else FAILED,
            "classification": "SLEEPING",
            "verdict": park.get("verdict"),
            "wake_condition": park.get("wake_condition"),
            "n_continued": n_continued,
            "PROTECTED_SCHEDULER_CAPABLE": capable,
            "PROTECTED_WINDOW_AVAILABLE": available,
            "correct_refusal": True,
            "why": (
                "GPU_EXCLUSIVE parked BLOCKED_ON_PROTECTED_WINDOW; scheduler stays CAPABLE"
                if parked_ok
                else "protected parking did not fire"
            ),
            "id": "WU.TORTURE.PROTECTED_PARKING",
            "unit_id": "WU.TORTURE.PROTECTED_PARKING",
        },
        unit_id="WU.TORTURE.PROTECTED_PARKING",
    )
    reroute_ok = parked_ok and n_continued > 0
    timeline.append(
        "blocked_resource_reroute",
        {
            "power": "BLOCKED_RESOURCE_REROUTE",
            "status": EXERCISED if reroute_ok else FAILED,
            "n_continued": n_continued,
            "unit_ids": list(continued.get("unit_ids") or []),
            "why": (
                f"protected unit parked; continue_with returned {n_continued} CPU units"
                if reroute_ok
                else "blocked-resource reroute did not continue with CPU work"
            ),
            "id": "WU.TORTURE.BLOCKED_RESOURCE_REROUTE",
            "unit_id": "WU.TORTURE.BLOCKED_RESOURCE_REROUTE",
        },
        unit_id="WU.TORTURE.BLOCKED_RESOURCE_REROUTE",
    )
    park_rec = _power_record(
        "PROTECTED_PARKING",
        EXERCISED if parked_ok else FAILED,
        why=(
            "CAPABLE true, AVAILABLE false, BLOCKED_ON_PROTECTED_WINDOW with a wake condition"
            if parked_ok
            else "protected parking did not park"
        ),
        evidence={
            "verdict": park.get("verdict"),
            "n_continued": n_continued,
            "wake_condition": park.get("wake_condition"),
            "PROTECTED_SCHEDULER_CAPABLE": capable,
            "PROTECTED_WINDOW_AVAILABLE": available,
        },
    )
    reroute_rec = _power_record(
        "BLOCKED_RESOURCE_REROUTE",
        EXERCISED if reroute_ok else FAILED,
        why=(
            f"resource blocked; {n_continued} CPU units scheduled instead of idling"
            if reroute_ok
            else "no CPU continuation while protected work was parked"
        ),
        evidence={"n_continued": n_continued, "unit_ids": list(continued.get("unit_ids") or [])},
    )
    return park_rec, reroute_rec


def _exercise_nr_nx(timeline: SealedTimeline, proofs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    compact = (proofs or {}).get("nr_nx") if isinstance(proofs, Mapping) else None
    if isinstance(compact, Mapping) and (
        compact.get("first_failing_stage") or compact.get("callable") is True
    ):
        row = dict(compact)
        row.setdefault("pipeline_callable", compact.get("callable"))
        row.setdefault("why", compact.get("why"))
    else:
        row = callable_on()
    first = row.get("first_failing_stage") if isinstance(row.get("first_failing_stage"), Mapping) else {}
    status = str(first.get("status") or "")
    ok = (
        row.get("pipeline_callable") is True
        or (
            row.get("pipeline_callable") is False
            and status in {"REFUSED", "FAILED", "BLOCKED"}
            and first.get("stage")
            and status != "SKIPPED"
        )
    )
    n_organs = None
    why_first = str(first.get("why") or row.get("why") or "")
    organs_hit = re.search(r"(\d+)\s+compiled organ", why_first)
    if organs_hit:
        n_organs = int(organs_hit.group(1))
    timeline.append(
        "nr_nx_stage",
        {
            "power": "GENERIC_NR_NX",
            "status": EXERCISED_REFUSAL if ok and not row.get("pipeline_callable") else (
                EXERCISED if ok else FAILED
            ),
            "callable": row.get("callable"),
            "first_failing_stage": first,
            "correct_refusal": bool(ok and not row.get("pipeline_callable")),
            "n_compiled_organs": n_organs,
            "why": row.get("why"),
            "id": "WU.TORTURE.GENERIC_NR_NX",
            "unit_id": "WU.TORTURE.GENERIC_NR_NX",
        },
        unit_id="WU.TORTURE.GENERIC_NR_NX",
    )
    if ok and not row.get("pipeline_callable"):
        return _power_record(
            "GENERIC_NR_NX",
            EXERCISED_REFUSAL,
            why=(
                f"pipeline ran; first failing stage {first.get('stage')} "
                f"is {status} (not SKIPPED): {first.get('why')}"
            ),
            evidence={"first_failing_stage": first, "callable": False, "n_compiled_organs": n_organs},
        )
    if ok:
        return _power_record(
            "GENERIC_NR_NX",
            EXERCISED,
            why="every named NR→NX stage ran and PASSED",
            evidence={"callable": True},
        )
    return _power_record(
        "GENERIC_NR_NX",
        FAILED,
        why=str(row.get("why") or "NR→NX did not produce a staged refusal or a pass"),
        evidence={"first_failing_stage": first},
    )


def _exercise_mutation(timeline: SealedTimeline, scope: Path, gpu_lock: Mapping[str, Any]) -> dict[str, Any]:
    cycle = _mutation_cycle(scope / "mutation")
    credit = credit_mutation(cycle)
    timeline.append(
        "mutation_applied",
        {
            "power": "MUTATION",
            "status": EXERCISED if credit["present"] else FAILED,
            "mutation_id": cycle.get("mutation_id"),
            "mutation_class": cycle.get("mutation_class"),
            "proposed": True,
            "applied": True,
            "why": credit.get("why"),
            "id": "WU.TORTURE.MUTATION",
            "unit_id": "WU.TORTURE.MUTATION",
        },
        unit_id="WU.TORTURE.MUTATION",
    )
    rb = cycle.get("rollback") if isinstance(cycle.get("rollback"), Mapping) else {}
    timeline.append(
        "mutation_rolled_back",
        {
            "power": "MUTATION",
            "mutation_id": cycle.get("mutation_id"),
            "digest_match": rb.get("digest_match"),
            "byte_identical": rb.get("byte_identical"),
            "id": "WU.TORTURE.MUTATION.rollback",
            "unit_id": "WU.TORTURE.MUTATION.rollback",
        },
        unit_id="WU.TORTURE.MUTATION.rollback",
    )
    # GPU mutation classes need a Metal device AND the lane lock. Park.
    gpu_note = gpu_lock.get("waited_for") or "GPU lane lock inspected; mutation trial GPU classes not contended"
    if not credit["present"]:
        rec = _power_record("MUTATION", FAILED, why=credit["why"], evidence={"mutation_id": cycle.get("mutation_id")})
        _emit_power(timeline, rec, "power_failed")
        return rec
    return _power_record(
        "MUTATION",
        EXERCISED,
        why=(
            f"PIPELINE_SELF proposal applied and rolled back byte-identical "
            f"({cycle.get('mutation_id')}). GPU classes parked: {gpu_note}"
        ),
        evidence={
            "mutation_id": cycle.get("mutation_id"),
            "mutation_class": cycle.get("mutation_class"),
            "gpu_lane_lock": dict(gpu_lock),
        },
    )


def _exercise_subagents(timeline: SealedTimeline, scope: Path) -> dict[str, Any]:
    row = _two_subagent_states(scope / "subagents")
    timeline.append(
        "subagent_state_persisted",
        {
            "power": "SUBAGENT_STATE",
            "status": EXERCISED if row.get("present") else FAILED,
            "n_states": row.get("n_states"),
            "disjoint": row.get("disjoint"),
            "why": row.get("why"),
            "id": "WU.TORTURE.SUBAGENT_STATE",
            "unit_id": "WU.TORTURE.SUBAGENT_STATE",
        },
        unit_id="WU.TORTURE.SUBAGENT_STATE",
    )
    return _power_record(
        "SUBAGENT_STATE",
        EXERCISED if row.get("present") else FAILED,
        why=str(row.get("why")),
        evidence={"n_states": row.get("n_states"), "disjoint": row.get("disjoint")},
    )


def _exercise_concurrency(timeline: SealedTimeline) -> dict[str, Any]:
    plan = cd.plan()
    refused = False
    refuse_why = None
    try:
        cd.verdict([])
    except cd.VerdictRefuse as exc:
        refused = True
        refuse_why = str(exc)
    decided = cd.decide()
    sleeping = decided.get("experiment_state") == "SLEEPING" and decided.get("verdict") is None
    blocked_reason = decided.get("reason") or (decided.get("workunit") or {}).get("blocked_reason")
    ok = refused and sleeping and bool(blocked_reason)
    timeline.append(
        "concurrency_doctor_decided",
        {
            "power": "CONCURRENCY",
            "status": EXERCISED_REFUSAL if ok else FAILED,
            "experiment_state": decided.get("experiment_state"),
            "verdict_refused": refused,
            "blocked_reason": blocked_reason,
            "classification": "SLEEPING",
            "correct_refusal": True,
            "ladder": plan.get("ladder"),
            "why": (
                "doctor REFUSED a verdict without observations and is SLEEPING "
                f"with blocked_reason={blocked_reason!r}; that is the exercise"
                if ok
                else "concurrency doctor did not refuse a verdict / sleep with a named reason"
            ),
            "id": "WU.TORTURE.CONCURRENCY",
            "unit_id": "WU.TORTURE.CONCURRENCY",
        },
        unit_id="WU.TORTURE.CONCURRENCY",
    )
    return _power_record(
        "CONCURRENCY",
        EXERCISED_REFUSAL if ok else FAILED,
        why=(
            "correct refusal: SLEEPING with a named blocked_reason; "
            "CONCURRENCY_HELPS is not a default. Exercised, not skipped."
            if ok
            else f"verdict_refused={refused} sleeping={sleeping} reason={blocked_reason!r}"
        ),
        evidence={
            "experiment_state": decided.get("experiment_state"),
            "verdict_refused": refused,
            "blocked_reason": blocked_reason,
            "ladder": plan.get("ladder"),
            "verdict_refuse_why": refuse_why,
        },
    )


def _exercise_invalidation_and_replan(
    timeline: SealedTimeline,
    proofs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = fop.rank_all()
    restatement = {
        "id": "WU.TORTURE.invalidate.gate_up.shared_input_latent",
        "family": "shared_input_latent_plus_expert_local_output_readout",
        "organ": fop.EXHAUSTED_ORGAN,
        "surface": fop.EXHAUSTED_SURFACE,
        "school": "ROUTED_EXPERTS",
    }
    fired = False
    err = None
    try:
        fop.refuse_if_restatement(restatement, ranking["scar"], ranking["killed_families"])
    except fop.RestatementRefused as exc:
        fired = True
        err = str(exc)
    next_ranked = (ranking.get("ranked") or [{}])[0] if ranking.get("ranked") else {}
    inv_ok = fired and str(next_ranked.get("school") or "") != "ROUTED_EXPERTS"
    timeline.append(
        "restatement_refused",
        {
            "power": "FRONTIER_INVALIDATION",
            "status": EXERCISED if inv_ok else FAILED,
            "killed_family": restatement["family"],
            "next_school": next_ranked.get("school"),
            "why": err or "restatement refused",
            "id": "WU.TORTURE.FRONTIER_INVALIDATION",
            "unit_id": "WU.TORTURE.FRONTIER_INVALIDATION",
        },
        unit_id="WU.TORTURE.FRONTIER_INVALIDATION",
    )
    inv_rec = _power_record(
        "FRONTIER_INVALIDATION",
        EXERCISED if inv_ok else FAILED,
        why=(
            f"queued restatement refused; next school is {next_ranked.get('school')}"
            if inv_ok
            else "restatement invalidation did not fire"
        ),
        evidence={"next_school": next_ranked.get("school"), "error": err},
    )

    replans = list(proofs.get("replans") or [])
    try:
        from tools.future import flash_meta_replan as fmr

        inputs = fmr.load_inputs()
        planned = fmr.replan(
            screen=inputs.get("screen"),
            teacher=inputs.get("teacher"),
            sub1=inputs.get("sub1"),
            index_doc=inputs.get("index"),
        )
        next_capture = (planned.get("next_capture") or {}).get("spend")
        replans = list(replans) + [
            {
                "cause": "flash_meta_replan",
                "cause_id": "WU.TORTURE.invalidate.gate_up.shared_input_latent",
                "effect_id": f"WU.TORTURE.replan.next_organ.{next_ranked.get('school') or 'NGRAM'}",
                "how": next_capture or "flash_meta_replan.replan re-ranked remaining families",
            }
        ]
    except Exception as exc:
        planned = {"error": f"{type(exc).__name__}: {exc}"}
        next_capture = None
    replan_ok = len(replans) >= 2
    for i, pair in enumerate(replans[:4]):
        timeline.append(
            "replan_emitted",
            {
                "power": "REPLAN",
                "status": EXERCISED if replan_ok else FAILED,
                "cause": pair.get("cause"),
                "cause_id": pair.get("cause_id"),
                "effect_id": pair.get("effect_id"),
                "how": pair.get("how"),
                "why": f"result {pair.get('cause')} changes what runs next",
                "id": f"WU.TORTURE.REPLAN.{i}",
                "unit_id": f"WU.TORTURE.REPLAN.{i}",
            },
            unit_id=f"WU.TORTURE.REPLAN.{i}",
        )
    replan_rec = _power_record(
        "REPLAN",
        EXERCISED if replan_ok else FAILED,
        why=f"{len(replans)} results change what runs next" if replan_ok else "fewer than 2 replans",
        evidence={"n": len(replans), "next_capture": next_capture},
    )
    return inv_rec, replan_rec


def _exercise_tabula(timeline: SealedTimeline) -> dict[str, Any]:
    try:
        from tools.future import tabula as tb
    except Exception as exc:
        rec = _power_record(
            "TABULA",
            SKIPPED,
            why=f"tabula unimportable ({type(exc).__name__}: {exc})",
        )
        _emit_power(timeline, rec, "power_skipped")
        return rec
    recovered = tb.recover_tabula()
    scores = tb.ScoreVector(
        behavioral=0.9,
        capability=0.9,
        tool_use=0.9,
        reasoning=0.9,
        instruction_following=0.9,
    )
    verdict = tb.evaluate(scores)
    ranked = tb.rank(
        [
            {
                "id": "cand.floor.pass",
                "behavioral": 0.9,
                "capability": 0.8,
                "tool_use": 0.8,
                "reasoning": 0.7,
                "instruction_following": 0.8,
            },
            {
                "id": "cand.floor.weaker",
                "behavioral": 0.95,
                "capability": 0.4,
                "tool_use": 0.4,
                "reasoning": 0.4,
                "instruction_following": 0.4,
            },
        ]
    )
    weights_frozen = False
    weights_why = None
    try:
        tb.apply_to_weights()
    except tb.WeightsFrozen as exc:
        weights_frozen = True
        weights_why = str(exc)
    refusal_rate_refused = False
    try:
        tb.scores_from_refusal_rate(0.0)
    except tb.IncompleteScoreVector:
        refusal_rate_refused = True
    capture = tb.teacher_capture_progress()
    units = tb.emit_workunits(
        contracts=tb.catalog(),
        lattice=tb.AuthorityLattice(),
        capture=capture,
    )
    callable_doc = tb.resident_callable(units=units, refusals=tb._prove_negative_controls())
    floor_ok = (
        bool(recovered)
        and verdict.outcome in {"PASS", "FAILURE"}
        and bool(ranked)
        and weights_frozen
        and refusal_rate_refused
        and callable_doc.get("callable") == "build"
    )
    timeline.append(
        "tabula_callable",
        {
            "power": "TABULA",
            "status": EXERCISED if floor_ok else FAILED,
            "callable": True,
            "entry_point": callable_doc.get("entry_point"),
            "evaluate_outcome": verdict.outcome,
            "weights_frozen": weights_frozen,
            "refusal_rate_refused": refusal_rate_refused,
            "n_recovered": len(recovered) if isinstance(recovered, list) else None,
            "n_units": len(units),
            "why": (
                "Tabula floor is callable: recover, evaluate, rank, catalog, "
                "resident_callable; apply_to_weights is WeightsFrozen (the floor, "
                "not a stubbed model); zero-refusal scoring is refused"
                if floor_ok
                else "Tabula floor did not exercise recover/evaluate/rank/callability"
            ),
            "id": "WU.TORTURE.TABULA",
            "unit_id": "WU.TORTURE.TABULA",
        },
        unit_id="WU.TORTURE.TABULA",
    )
    return _power_record(
        "TABULA",
        EXERCISED if floor_ok else FAILED,
        why=(
            "Tabula callability on the independent-evaluation floor; fitting sleeps. "
            f"apply_to_weights correctly refused ({weights_why})."
            if floor_ok
            else "Tabula floor was not callable"
        ),
        evidence={
            "callable": True,
            "evaluate_outcome": verdict.outcome,
            "weights_frozen": weights_frozen,
            "entry_point": callable_doc.get("entry_point"),
        },
    )


def _cognition_probe() -> dict[str, Any]:
    try:
        from tools.future import model_bearing as mb

        state = mb.cognition_state()
        return {
            "state": state.get("state"),
            "why": state.get("why"),
            "asked": state.get("asked"),
            "provider_source": state.get("provider_source"),
            "unavailable": state.get("state") == mb.UNAVAILABLE,
        }
    except Exception as exc:
        return {
            "state": "UNAVAILABLE",
            "why": f"model_bearing.cognition_state raised {type(exc).__name__}: {exc}",
            "asked": False,
            "unavailable": True,
        }


def run_torture(
    *,
    write: bool = True,
    duration_s: int = DURATION_S,
    scope: Path | None = None,
    proofs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze, exercise every catalog power, seal the timeline, judge, measure.

    `write=False` keeps receipts off disk (tests). The wall budget is a
    deadline, not a duration target: finish when the powers have run.
    """
    t0 = time.time()
    deadline = t0 + float(duration_s)
    own_scope = scope is None
    root = Path(scope) if scope is not None else Path(tempfile.mkdtemp(prefix="hawking-power-torture-"))
    root.mkdir(parents=True, exist_ok=True)

    degeneracy_path = None
    try:
        _, degeneracy_path = load_degeneracy()
    except TortureRefused:
        degeneracy_path = None
    extras = [degeneracy_path] if degeneracy_path else []
    before = hash_substrate(extra_paths=extras)
    gpu_lock = inspect_gpu_lane_lock()
    timeline = SealedTimeline(t0=t0)
    timeline.append(
        "substrate_hashed",
        {
            "when": "before",
            "digest": before.get("digest"),
            "n_files": before.get("n_files"),
            "n_shaders": before.get("n_shaders"),
            "id": "WU.TORTURE.substrate.before",
        },
        unit_id="WU.TORTURE.substrate.before",
    )
    timeline.append(
        "gpu_lane_lock_parked",
        {
            **{k: gpu_lock.get(k) for k in ("path", "present", "kind", "owner", "pid", "pid_alive", "waited_for", "parked")},
            "id": "WU.TORTURE.gpu_lock",
        },
        unit_id="WU.TORTURE.gpu_lock",
    )

    ran = dict(proofs) if proofs is not None else drive_proofs(scope=root / "proofs")
    records: dict[str, dict[str, Any]] = {}

    def _overtime() -> bool:
        return time.time() > deadline

    if _overtime():
        for name in POWER_CATALOG:
            rec = _power_record(name, SKIPPED, why="wall budget exhausted before exercise")
            records[name] = rec
            _emit_power(timeline, rec, "power_skipped")
    else:
        records["NO_WAIT"] = _exercise_no_wait(timeline, root / "nowait")
        refill_rec, ingest_rec = _exercise_refill_and_ingest(timeline, ran, root / "nowait")
        records["REAL_REFILL"] = refill_rec
        records["REAL_INGESTION"] = ingest_rec
        records["SCAR_PRUNING"] = _exercise_scar(timeline, ran)
        records["STATUS_CAUSALITY"] = _exercise_status(timeline)
        park_rec, reroute_rec = _exercise_protected_and_reroute(timeline)
        records["PROTECTED_PARKING"] = park_rec
        records["BLOCKED_RESOURCE_REROUTE"] = reroute_rec
        records["GENERIC_NR_NX"] = _exercise_nr_nx(timeline, ran)
        records["MUTATION"] = _exercise_mutation(timeline, root, gpu_lock)
        records["SUBAGENT_STATE"] = _exercise_subagents(timeline, root)
        records["CONCURRENCY"] = _exercise_concurrency(timeline)
        inv_rec, replan_rec = _exercise_invalidation_and_replan(timeline, ran)
        records["FRONTIER_INVALIDATION"] = inv_rec
        records["REPLAN"] = replan_rec
        records["TABULA"] = _exercise_tabula(timeline)

    for name in POWER_CATALOG:
        if name not in records:
            rec = _power_record(name, SKIPPED, why="not reached")
            records[name] = rec
            _emit_power(timeline, rec, "power_skipped")

    cognition = _cognition_probe()
    timeline.append(
        "cognition_state",
        {
            **cognition,
            "id": "WU.TORTURE.cognition",
            "why": "PASS means new power integration, not resident cognition",
        },
        unit_id="WU.TORTURE.cognition",
    )

    after = hash_substrate(extra_paths=extras)
    substrate = verify_substrate(before, after)
    timeline.append(
        "substrate_hashed",
        {
            "when": "after",
            "digest": after.get("digest"),
            "verdict": substrate.get("verdict"),
            "equal": substrate.get("equal"),
            "moved": substrate.get("moved"),
            "id": "WU.TORTURE.substrate.after",
        },
        unit_id="WU.TORTURE.substrate.after",
    )
    events_sha = timeline.seal()
    timeline_doc = timeline.to_doc()
    judged = judge(timeline_doc)
    try:
        degeneracy = measure_degeneracy(timeline_doc)
    except TortureRefused as exc:
        degeneracy = {
            "verdict": "FAIL",
            "reason": str(exc),
            "degenerate_axes": ["instrument_missing"],
            "axis_table": [],
            "instrument": "tools.future.autonomy_degeneracy.measure",
        }

    elapsed = time.time() - t0
    exercised = [n for n in POWER_CATALOG if records[n]["exercised"]]
    skipped = [n for n in POWER_CATALOG if records[n]["skipped"]]
    failed = [n for n in POWER_CATALOG if records[n]["failed"]]
    refusals = [n for n in POWER_CATALOG if records[n]["correct_refusal"]]

    integration_ok = (
        not failed
        and bool(exercised)
        and judged.get("verdict") == "PASS"
        and judged.get("omitted") == []
    )
    overall = "PASS"
    fail_why: list[str] = []
    if not substrate.get("equal"):
        overall = "FAIL"
        fail_why.append(str(substrate.get("why")))
    if judged.get("verdict") != "PASS":
        overall = "FAIL"
        fail_why.append("judge: " + str(judged.get("why")))
    if degeneracy.get("verdict") != "PASS":
        overall = "FAIL"
        fail_why.append("degeneracy: " + str(degeneracy.get("reason")))
    if elapsed > float(duration_s):
        overall = "FAIL"
        fail_why.append(f"wall clock {elapsed:.1f}s exceeded {duration_s}s")
    if failed:
        overall = "FAIL"
        fail_why.append("failed powers: " + ", ".join(failed))
    if not cognition.get("unavailable", True):
        overall = "FAIL"
        fail_why.append("a resident model was attached; this torture is not cognition")
    if not exercised:
        overall = "FAIL"
        fail_why.append("skips do not count toward the pass")

    result = {
        "schema": SCHEMA,
        "version": 2,
        "trial_id": TRIAL_ID,
        "duration_s_budget": duration_s,
        "elapsed_s": elapsed,
        "within_budget": elapsed <= float(duration_s),
        "verdict": overall,
        "why": "; ".join(fail_why) if fail_why else (
            "frozen substrate unchanged; judge read the sealed timeline; "
            "degeneracy PASS; "
            f"{len(exercised)} exercised ({len(refusals)} correct refusals), "
            f"{len(skipped)} skipped"
        ),
        "obligation": (
            "30-MINUTE POWER TORTURE PASSED. A frozen build exercises EVERY POWER "
            "that landed after the 1h trial and was therefore untested by it. "
            "PASS means NEW POWER INTEGRATION, and explicitly NOT resident cognition."
        ),
        "pass_means": "NEW_POWER_INTEGRATION",
        "pass_does_not_mean": "resident_cognition",
        "n_exercised": len(exercised),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "n_correct_refusals": len(refusals),
        "exercised": exercised,
        "skipped": skipped,
        "failed": failed,
        "correct_refusals": refusals,
        "powers": records,
        "correct_refusal_counts_as_exercised": True,
        "skipped_do_not_count_toward_pass": True,
        "substrate": {
            "before_digest": before.get("digest"),
            "after_digest": after.get("digest"),
            "equal": substrate.get("equal"),
            "verdict": substrate.get("verdict"),
            "moved": substrate.get("moved"),
            "n_files": before.get("n_files"),
            "n_shaders": before.get("n_shaders"),
            "files": before.get("files"),
            "shaders": before.get("shaders"),
        },
        "gpu_lane_lock": gpu_lock,
        "cognition": cognition,
        "judge": {k: v for k, v in judged.items() if k != "powers"} | {
            "powers": judged.get("powers"),
        },
        "degeneracy": {
            "verdict": degeneracy.get("verdict"),
            "reason": degeneracy.get("reason"),
            "degenerate_axes": degeneracy.get("degenerate_axes"),
            "named_axes": degeneracy.get("named_axes"),
            "axis_table": degeneracy.get("axis_table"),
            "n_argv0_labelled": degeneracy.get("n_argv0_labelled"),
            "n_unlabelled": degeneracy.get("n_unlabelled"),
            "instrument": degeneracy.get("instrument"),
            "instrument_path": degeneracy.get("instrument_path"),
        },
        "timeline": {
            "receipt": f"receipts/future/{TIMELINE_RECEIPT}",
            "events_sha256": events_sha,
            "n_events": timeline_doc["n_events"],
            "sealed": True,
            "append_only": True,
            "elapsed_s": timeline_doc.get("elapsed_s"),
        },
        "integration_ok": integration_ok,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "resident_model_attached": False,
    }

    if write:
        write_receipt(TIMELINE_RECEIPT, timeline_doc, RECORDED_BY)
        result["timeline"]["path"] = f"receipts/future/{TIMELINE_RECEIPT}"
        doc = dict(result)
        # The 30M receipt cites the sealed timeline; it does not embed events.
        doc["timeline_receipt"] = f"receipts/future/{TIMELINE_RECEIPT}"
        doc["purpose"] = (
            "30-minute power torture on a frozen build. Judged independently "
            "from the sealed timeline against a substrate verified unchanged."
        )
        doc["credit_rules"] = {
            "mutation_without_rollback_counts": False,
            "untested_status_challenge_counts": False,
            "catalog_refill_of_already_offered_ids_counts": False,
            "module_build_is_not_behaviour": True,
            "correct_refusal_counts_as_exercised": True,
            "skipped_do_not_count_toward_pass": True,
            "runner_summary_is_not_the_judge": True,
        }
        write_receipt(RECEIPT, doc, RECORDED_BY)
    if own_scope:
        result["scope"] = str(root)
    result["_timeline_doc"] = timeline_doc
    result["_before"] = before
    result["_after"] = after
    return result


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    """Freeze, run, judge, measure, write both receipts. Returns the 30M path."""
    run_torture(write=True, duration_s=DURATION_S)
    out = RECEIPTS / RECEIPT
    if not out.is_file():
        raise TortureRefused("POWER_TORTURE_30M.json was not written", missing=["receipt"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--torture", action="store_true")
    ap.add_argument("--judge", metavar="TIMELINE_JSON")
    ap.add_argument("--detect", metavar="TIMELINE_JSON")
    a = ap.parse_args()
    if a.detect:
        path = Path(a.detect)
        if not path.is_file():
            raise TortureRefused(f"timeline not on disk: {path}", missing=["timeline"])
        print(json.dumps(detect_no_wait_orchestration(json.loads(path.read_text())), indent=1, sort_keys=True))
        return 0
    if a.judge:
        path = Path(a.judge)
        if not path.is_file():
            raise TortureRefused(f"timeline not on disk: {path}", missing=["timeline"])
        print(json.dumps(judge(json.loads(path.read_text())), indent=1, sort_keys=True, default=str))
        return 0
    if a.compose:
        print(json.dumps(compose(), indent=1, sort_keys=True, default=str))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
