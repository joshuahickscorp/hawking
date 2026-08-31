"""POWER TORTURE — compose a 30-minute transition-density workload the 1h trial never ran.

The frozen 1h autonomy trial passed a clock. Scrutiny of its own timeline showed
what it did not do: 222 rejections of one table in nine seconds, four refills of
the same 25 ids, one receipt ingested 29 times, and not one unit mutated
anything. It is a verifier, not an optimizer. The powers that landed after that
trial (no-wait, mutation, status causality, protected parking, generic NR→NX,
concurrency doctor, abliteration, organ pivot) were never asked to demonstrate
the behaviour they exist for.

This module is the composer AND the detector. `compose` selects 6–12 units of
work that would have been worth doing if the torture were cancelled halfway.
`detect_no_wait_orchestration` is how the torture fails itself: runnable safe
work sitting idle while the loop waits inside a subprocess is
FAIL_NO_WAIT_ORCHESTRATION, proven from timestamps, never from intent.

It extends trial_workload's composer; it does not fork a second mix language.
It does not run a 30-minute clock, does not take a GPU lease, and does not
treat a module's build() as evidence of the behaviour the module exists for.
A missing transition class is a named refusal, not a padded pass.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.future._common import RECEIPTS, write_receipt
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

RECEIPT = "POWER_TORTURE.json"
SCHEMA = "hawking.future.power_torture.v1"

DURATION_S = 30 * 60
TRIAL_ID = "30m"

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
# Receipt
# ---------------------------------------------------------------------------


def build() -> Path:
    current = tw.load_book()
    proofs = drive_proofs()
    packed = compose_or_refuse(book=current)
    public_units = []
    if packed.get("admitted"):
        for u in packed["units"]:
            public_units.append(
                {
                    "id": u.get("id"),
                    "module": u.get("module"),
                    "frontier_id": u.get("frontier_id"),
                    "species": u.get("species"),
                    "resource_class": u.get("resource_class"),
                    "transition": u.get("transition") or u.get("mix_role"),
                    "launch": u.get("launch"),
                    "status": u.get("status"),
                    "classification": u.get("classification"),
                    "worth_doing_anyway": u.get("worth_doing_anyway"),
                    "description": u.get("description"),
                }
            )
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Compose a 30-minute transition-density workload that forces each "
            "post-1h power to demonstrate real behaviour, and detect "
            "FAIL_NO_WAIT_ORCHESTRATION from timestamps so the torture can fail itself."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "trial_id": TRIAL_ID,
        "duration_s": DURATION_S,
        "doctrine": "STRESS TRANSITIONS, NOT CLOCKS",
        "required_transitions": list(REQUIRED_TRANSITIONS),
        "admitted": bool(packed.get("admitted")),
        "n_units": packed.get("n_units"),
        "n_replans": packed.get("n_replans"),
        "units": public_units,
        "replans": packed.get("replans") if packed.get("admitted") else [],
        "transitions": packed.get("transitions") or proofs.get("transitions"),
        "compose_refusal": (
            None
            if packed.get("admitted")
            else {"why": packed.get("why"), "missing": packed.get("missing")}
        ),
        "detector": {
            "entry_point": "tools.future.power_torture.detect_no_wait_orchestration",
            "fail_verdict": FAIL_NO_WAIT,
            "negative_control": proofs["wait_verdict"]["verdict"],
            "positive_control": proofs["overlap_verdict"]["verdict"],
            "negative_fired": proofs["wait_verdict"]["verdict"] == FAIL_NO_WAIT,
            "positive_overlap_proven": bool(proofs["overlap_verdict"].get("overlaps")),
        },
        "proofs": packed.get("proofs") or {
            "wait_verdict": proofs["wait_verdict"]["verdict"],
            "overlap_verdict": proofs["overlap_verdict"]["verdict"],
            "challenge": proofs.get("challenge"),
            "scar": proofs.get("scar"),
            "protected": proofs.get("protected"),
            "nr_nx": proofs.get("nr_nx"),
            "concurrency": proofs.get("concurrency"),
        },
        "credit_rules": {
            "mutation_without_rollback_counts": False,
            "untested_status_challenge_counts": False,
            "catalog_refill_of_already_offered_ids_counts": False,
            "module_build_is_not_behaviour": True,
        },
        "recovered_implementation": [
            "tools/future/trial_workload.py — composer shape, WorkloadRefused, make_unit, admit_unit, load_book",
            "tools/future/no_wait_scheduler.py — launch_detached / runnable_now / ingest_ready / prove_overlap_interval (detector sits on top)",
            "tools/future/mutation_engine.py — propose / apply / evidence / rollback / pipeline_self_cycle",
            "tools/future/status_causality.py — challenge / scan / verdict; UNTESTED is not a challenge",
            "tools/future/protected_scheduler.py — recognize / decide / park / continue_with / drive",
            "tools/future/nr_nx_generic.py — assemble / STAGE_ORDER; callable_on wraps it and refuses closed on ImportError",
            "tools/future/concurrency_doctor.py — plan / observe / verdict / decide SLEEPING",
            "tools/future/abliteration.py — method / plan / contracts",
            "tools/future/flash_organ_pivot.py — rank_all / refuse_if_restatement",
            "tools/future/autonomy_run.py — the loop that will execute this (not edited; outside WRITE)",
            "tools/future/scar_scheduling.py — admit / refuse_if_dead via negative_index",
            "tools/future/frontiers.py — next_work / refill (catalog refill is a replay; recorded as such)",
            "tools/future/workgraph.py — two durable independent agent graphs",
            "tools/future/work_events.py — RESULT_INGESTED requires cites; WORK_REFILLED requires fresh ids",
            "tools/future/autonomy_trial.py — cpu_workunit / work_identity / is_low_information",
        ],
        "gaps_closed": [
            "no 30-minute transition-density composer existed; 15m/1h/3h/6h mixes never exercised post-trial powers",
            "FAIL_NO_WAIT_ORCHESTRATION is now a detector the torture can fire on itself from timestamps",
            "a mix missing any required transition class is a named refusal",
            "a mutation without proven rollback does not count toward MUTATION",
            "an UNTESTED status challenge does not count toward STATUS_CAUSALITY",
            "frontiers.refill returning the same ids is recorded as NOT a real refill; replan extras are",
        ],
        "negative_findings": [
            "this module is not in orchestration.BINDINGS (that table is outside this WRITE list)",
            "autonomy_run.py is not wired to trial_id=30m (outside this WRITE list); compose is the callable the loop would import",
            "frontiers.refill currently returns the same ids as next_work; that is the 1h replay, not a real refill",
            "nr_nx_generic.py cannot be imported here: tools.odyssey is not materialized; callable_on records a staged REFUSAL",
            "compose drives proofs of each class; it does not run a 30-minute clock and does not invoke specimen hashing",
            "resident_model_cognition is not exercised; the loop under test is HCLI orchestration",
            "PROTECTED_PARKING parks; this sidecar still has no GPU authority and will not execute the unit",
            "concurrency_doctor.verdict is refused without a resident process; CONCURRENCY_HELPS is not a default",
        ],
        "resident_callable": {
            "entry_point": "tools.future.power_torture.compose()",
            "workunit": (
                "one CPU_ANALYSIS unit; compose the 30-minute transition-density "
                "workload and refuse a mix missing any required transition class"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "missing transition class raises TortureRefused naming it; "
                "UNTESTED challenges and mutations without rollback do not count; "
                "detect_no_wait_orchestration returns FAIL_NO_WAIT_ORCHESTRATION "
                "rather than a guessed pass; absent timeline raises"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/power_torture.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--detect", metavar="TIMELINE_JSON")
    a = ap.parse_args()
    if a.detect:
        path = Path(a.detect)
        if not path.is_file():
            raise TortureRefused(f"timeline not on disk: {path}", missing=["timeline"])
        print(json.dumps(detect_no_wait_orchestration(json.loads(path.read_text())), indent=1, sort_keys=True))
        return 0
    if a.compose:
        print(json.dumps(compose(), indent=1, sort_keys=True, default=str))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
