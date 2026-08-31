#!/usr/bin/env python3
"""AUTONOMY EVIDENCE MUST BE NON-DEGENERATE.

Rejections that never change between runs, refills that re-offer an identical
set, a receipt ingested 29 times, and unlabelled units are all degenerate
evidence. A trial reports DISTINCT-VS-REPEATED FOR EVERY AXIS and FAILS ON
DEGENERACY EVEN WHEN EVERY CONDITION IS NOMINALLY MET.

This module is the reusable measure. It does not judge TPS, duration, or the
written acceptance conditions of any trial. It scores a timeline on every
axis the timeline carries and returns FAIL if any named axis is degenerate.

The proof is two receipts already on disk, opposite verdicts, no fixtures:

* receipts/future/AUTONOMY_TIMELINE_1h.json  — FAIL (four pathologies at once)
* receipts/future/DETACHED_WORK_TRIAL.json   — PASS (the non-degenerate case)

    python3 tools/future/autonomy_degeneracy.py --record
    python3 tools/future/autonomy_degeneracy.py --selftest
    python3 -m pytest tools/future/test_autonomy_degeneracy.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import (
    REPO,
    git,
    sha256_file,
    write_receipt,
    _assert_no_hardware_claims,
)


RECEIPT = "AUTONOMY_DEGENERACY.json"
SCHEMA = "hawking.future.autonomy_degeneracy.v1"
VERSION = 1
RECORDED_BY = "tools/future/autonomy_degeneracy.py"

TIMELINE_1H_REL = "receipts/future/AUTONOMY_TIMELINE_1h.json"
DETACHED_TRIAL_REL = "receipts/future/DETACHED_WORK_TRIAL.json"

# ---------------------------------------------------------------------------
# Thresholds. "Never repeats" is too strict; the line is the 1h pathology.
# ---------------------------------------------------------------------------

# A receipt may be read at recover, after a related experiment, and once more
# as confirmation. Five or more of the same path is a loop. The 1h timeline
# ingested SPECIMEN_VERIFICATION.json 31 times (the obligation named 29); an
# honest run ingesting each landed receipt once or twice stays under.
INGEST_MAX_REPEATS = 4

# Re-offering the identical id set is the 1h scar (four refills, the same
# frontier ids). A single id appearing in two *different* sets is not this.
# The detached trial: 49 refills, zero consecutive identical sets, 50 unique ids.
REFILL_IDENTICAL_CONSECUTIVE_MAX = 0

# Applied when n_rejections >= 12. 222 refusals of 29 scars is unique_ratio
# 0.131. An honest run refusing distinct dead families as they appear sits
# near 1.0. Repeating a scar twice is not the pathology.
REJECTION_MIN_UNIQUE_RATIO = 0.35
REJECTION_RATIO_MIN_N = 12

# Seven or more consecutive identical refusals is a stuck loop. The 1h table
# has a run of 18. A pair of the same scar in a row is not.
REJECTION_MAX_CONSECUTIVE_RUN = 6

# 222 idea_rejected, all in t_s 7-16 of a 3629 s run, never recur. A long
# trial whose every refusal lands in a 20 s opening window is replaying a
# table, not consulting scars as work arrives.
REJECTION_EARLY_CLUSTER_WINDOW_S = 20.0
REJECTION_EARLY_CLUSTER_MIN_N = 12
REJECTION_EARLY_CLUSTER_MIN_ELAPSED_S = 120.0
REJECTION_EARLY_CLUSTER_MAX_FRAC = 0.05

# Refills that stop in the opening minutes of a long trial. The 1h run's last
# refill is t_s 119; nothing in the remaining ~3500 s. A short trial (the
# detached 130 s run, the 10-minute improvement skeleton) is not this.
REFILL_STOPPED_EARLY_MIN_ELAPSED_S = 600.0
REFILL_STOPPED_EARLY_MAX_LAST_FRAC = 0.20

# An unlabelled or argv[0]-labelled unit cannot be distinguished from the
# others sharing that label. 32 launches called "python3" (the obligation
# named 29) made diversity unmeasurable. A real WorkUnit id is a label.
# argv0 may be recorded alongside; it must not BE the label.
LABELLING_ARGV0_OR_UNLABELLED_MAX = 0

# Duplicate WorkUnit ids are the duplicate_workunits control. Unique ids are
# the non-degenerate case: the 1h launches actually have unique ids; its
# degeneracy on launches is labelling, not identity.
WORKUNIT_IDS_MUST_BE_UNIQUE = True

# Relaunching a family the trial already killed as a scar is
# dead_scar_repetition. Zero is the line because one replay is the defect.
DEAD_SCAR_RELAUNCHES_MAX = 0

# Same decision content emitted eight or more times in a row, on a stream
# of at least 12 decisions, is a stuck policy. Distinct decisions
# interleaved with occasional repeats are not.
DECISION_MAX_CONSECUTIVE_RUN = 7
DECISION_RUN_MIN_N = 12
DECISION_MIN_UNIQUE_RATIO = 0.35
DECISION_RATIO_MIN_N = 12

THRESHOLDS: dict[str, Any] = {
    "ingest_max_repeats_per_receipt": INGEST_MAX_REPEATS,
    "refill_identical_consecutive_sets_max": REFILL_IDENTICAL_CONSECUTIVE_MAX,
    "rejection_min_unique_ratio": REJECTION_MIN_UNIQUE_RATIO,
    "rejection_ratio_min_n": REJECTION_RATIO_MIN_N,
    "rejection_max_consecutive_run": REJECTION_MAX_CONSECUTIVE_RUN,
    "rejection_early_cluster_window_s": REJECTION_EARLY_CLUSTER_WINDOW_S,
    "rejection_early_cluster_min_n": REJECTION_EARLY_CLUSTER_MIN_N,
    "rejection_early_cluster_min_elapsed_s": REJECTION_EARLY_CLUSTER_MIN_ELAPSED_S,
    "rejection_early_cluster_max_frac": REJECTION_EARLY_CLUSTER_MAX_FRAC,
    "refill_stopped_early_min_elapsed_s": REFILL_STOPPED_EARLY_MIN_ELAPSED_S,
    "refill_stopped_early_max_last_frac": REFILL_STOPPED_EARLY_MAX_LAST_FRAC,
    "labelling_argv0_or_unlabelled_max": LABELLING_ARGV0_OR_UNLABELLED_MAX,
    "workunit_ids_must_be_unique": WORKUNIT_IDS_MUST_BE_UNIQUE,
    "dead_scar_relaunches_max": DEAD_SCAR_RELAUNCHES_MAX,
    "decision_max_consecutive_run": DECISION_MAX_CONSECUTIVE_RUN,
    "decision_run_min_n": DECISION_RUN_MIN_N,
    "decision_min_unique_ratio": DECISION_MIN_UNIQUE_RATIO,
    "decision_ratio_min_n": DECISION_RATIO_MIN_N,
}

THRESHOLD_DEFENSE: dict[str, str] = {
    "ingest_max_repeats_per_receipt": (
        "A receipt legitimately read twice (recover, then confirm) is not "
        "the 29-times pathology. The line is 4: recover + related experiment "
        "+ confirmation still fits; 5+ of one path is a loop. The 1h timeline "
        "falls on the wrong side (SPECIMEN_VERIFICATION.json, 31 times). An "
        "honest run ingesting each landed receipt once or twice does not."
    ),
    "refill_identical_consecutive_sets_max": (
        "Re-offering the identical set is the defect named in the obligation. "
        "Zero consecutive identical sets is the line because one identical "
        "re-offer is already the 1h scar. Distinct overlapping sets (an id "
        "that stays on the frontier while others rotate) are allowed. The "
        "detached trial has 49 refills and zero consecutive identical sets."
    ),
    "rejection_min_unique_ratio": (
        "29 unique scars across 222 refusals is a deterministic table, not "
        "consultation. The line 0.35 at n>=12 lets a scar be refused two or "
        "three times as it reappears (ratio 0.33-0.5) and fails the 1h "
        "ratio 0.131. Never-repeats would fail an honest double-check."
    ),
    "rejection_max_consecutive_run": (
        "A run of 7+ identical refusals is a stuck cursor over one row of "
        "the table. The 1h max run is 18. A pair of the same scar is not."
    ),
    "rejection_early_cluster": (
        "All refusals landing inside a 20 s window at the start of a trial "
        "longer than 120 s, covering less than 5% of elapsed, with no later "
        "recurrence, is the t_s 7-16 table. The detached trial has zero "
        "idea_rejected and is not this."
    ),
    "refill_stopped_early": (
        "On a trial of at least 600 s, a last refill before 20% of elapsed "
        "means the frontier was asked once at the beginning and then the "
        "same set was not even re-asked. The 1h last refill is t_s 119 of "
        "3629 s. Short honest runs (detached ~130 s, the 10-minute "
        "improvement skeleton) are exempt."
    ),
    "labelling_argv0_or_unlabelled_max": (
        "Zero: an unlabelled or argv[0]-labelled unit is degenerate because "
        "the timeline cannot distinguish it from every other unit sharing "
        "that label. 32 launches labelled python3 (argv[0] of the shell) "
        "made the diversity of those launches unmeasurable. A WorkUnit id "
        "such as WU.DETACHED_TRIAL.ind.0004 is a label; the interpreter "
        "path recorded beside it is not."
    ),
    "workunit_ids_must_be_unique": (
        "A WorkUnit id names one piece of work. Relaunching the same id "
        "with no new information is the duplicate_workunits control. The "
        "1h launches actually have unique ids; they fail labelling, not "
        "this axis. The two mechanisms must not disagree on a duplicate run."
    ),
    "dead_scar_relaunches_max": (
        "Zero: relaunching a family the trial already killed as a scar is "
        "dead_scar_repetition. The 1h table-replay of refusals is caught by "
        "the rejection ratio; this line is the improvement-trial control so "
        "the two mechanisms agree on that run."
    ),
    "decisions": (
        "A decision stream of 12+ with unique ratio below 0.35, or a "
        "consecutive run of 8+ identical decisions, is a stuck policy. The "
        "1h stream after t_s 119 is empty next_work_left leftovers. The "
        "detached stream keeps unique independent unit ids through t_s 129."
    ),
}

NAMED_AXES = (
    "rejections",
    "refills",
    "ingestion",
    "launches",
    "workunit_ids",
    "decisions",
    "scars",
)

REJECTION_KINDS = frozenset(
    {
        "idea_rejected",
        "IDEA_REJECTED",
        "negative_science_refusal",
        "NEGATIVE_SCIENCE_REFUSAL",
    }
)
REFILL_KINDS = frozenset({"work_refilled", "WORK_REFILLED"})
INGEST_KINDS = frozenset(
    {
        "result_ingested",
        "RESULT_INGESTED",
        "receipt_ingested",
        "RECEIPT_INGESTED",
    }
)
LAUNCH_KINDS = frozenset(
    {
        "workunit_launched",
        "WORK_LAUNCHED",
        "CHILD_LAUNCHED",
        "INDEPENDENT_STARTED",
    }
)
SCAR_KILL_KINDS = frozenset({"BRANCH_KILLED", "branch_killed"})
DECISION_KINDS = frozenset(
    {
        "idea_rejected",
        "IDEA_REJECTED",
        "negative_science_refusal",
        "NEGATIVE_SCIENCE_REFUSAL",
        "work_refilled",
        "WORK_REFILLED",
        "NEXT_DECISION",
        "next_work_left",
        "NEXT_LEFT_RUNNING",
        "BRANCH_KILLED",
        "branch_killed",
        "OPTIONS_RANKED",
        "CONCURRENCY_BOUND_REFUSED",
        "RECEIPT_WAKEUP",
        "CHILD_TERMINAL",
        "FALSIFIER_GENERATED",
        "PRIORITY_ALTERED",
        "priority_altered",
    }
)
NAMED_KIND_UNION = (
    REJECTION_KINDS
    | REFILL_KINDS
    | INGEST_KINDS
    | LAUNCH_KINDS
    | SCAR_KILL_KINDS
    | DECISION_KINDS
)

_INTERPRETER_RE = re.compile(
    r"^(python\d*(\.\d+)*|pythonw\d*|pypy\d*|bash|sh|zsh|dash|ksh|"
    r"node|nodejs|ruby|perl|lua|osascript)$",
    re.IGNORECASE,
)
_GENERIC_LABELS = frozenset(
    {
        "",
        "python3",
        "python",
        "work",
        "unit",
        "todo",
        "none",
        "n/a",
        "na",
        "unknown",
        "placeholder",
        "busywork",
        "do work",
        "keep busy",
        "idle",
        "wait",
        "noop",
    }
)


def is_argv0_label(value: Any) -> bool:
    """True when `value` is an interpreter basename (argv[0] of a shell)."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    text = str(value).strip()
    if not text:
        return False
    base = Path(text).name
    return bool(_INTERPRETER_RE.match(base))


def _kind(event: Mapping[str, Any]) -> str:
    return str(event.get("kind") or "").strip()


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _t_s(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("t_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def distinct_vs_repeated(tokens: Sequence[str]) -> dict[str, Any]:
    """Unique count, total count, largest repeat run, consecutive identical."""
    total = len(tokens)
    unique = len(set(tokens)) if tokens else 0
    counts = Counter(tokens)
    max_item_count = max(counts.values()) if counts else 0
    most_repeated = counts.most_common(1)[0][0] if counts else None
    largest_run = 0
    run = 0
    prev: str | None = None
    cons_pairs = 0
    started = False
    for token in tokens:
        if started and token == prev:
            run += 1
            cons_pairs += 1
        else:
            run = 1
        largest_run = max(largest_run, run)
        prev = token
        started = True
    if total == 0:
        largest_run = 0
    unique_ratio = (unique / total) if total else 1.0
    return {
        "total": total,
        "unique": unique,
        "unique_ratio": round(unique_ratio, 6),
        "largest_repeat_run": largest_run,
        "consecutive_identical_pairs": cons_pairs,
        "consecutive_emissions_identical": bool(cons_pairs) if total >= 2 else False,
        "all_consecutive_identical": (cons_pairs == total - 1) if total >= 2 else False,
        "max_item_count": max_item_count,
        "most_repeated": most_repeated,
    }


def _row(
    axis: str,
    tokens: Sequence[str],
    *,
    degenerate: bool,
    reason: str,
    fail_eligible: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = distinct_vs_repeated(tokens)
    body.update(
        {
            "axis": axis,
            "degenerate": bool(degenerate),
            "reason": reason,
            "fail_eligible": bool(fail_eligible),
        }
    )
    if extra:
        for key, value in extra.items():
            if key not in body:
                body[key] = value
    return body


def _vacuous(axis: str, why: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return _row(axis, (), degenerate=False, reason=why, extra=extra)


# ---------------------------------------------------------------------------
# Source loading — a path, a receipt dict, an event list, or a TrialRecord.
# ---------------------------------------------------------------------------

def _load_doc(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    rel = str(path)
    try:
        rel = str(path.resolve().relative_to(REPO))
    except (OSError, ValueError):
        rel = str(path)
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return json.loads(blob)
    raise FileNotFoundError(f"timeline absent from disk and git: {path}")


def _events_of(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_absolute():
            path = REPO / path
        doc = _load_doc(path)
        events = list(doc.get("events") or doc.get("timeline") or [])
        meta = dict(doc)
        meta["_source_path"] = str(path)
        return events, meta
    if isinstance(source, Mapping):
        events = list(source.get("events") or source.get("timeline") or [])
        if events or "elapsed_s" in source or "timeline" in source:
            return events, dict(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return [dict(e) for e in source if isinstance(e, Mapping)], {}
    events = list(getattr(source, "events", None) or [])
    meta = {
        "elapsed_s": getattr(source, "elapsed_s", None),
        "killed": getattr(source, "killed", None),
        "launched": getattr(source, "launched", None),
        "ingested": getattr(source, "ingested", None),
        "refilled": getattr(source, "refilled", None),
        "experiments_avoided": getattr(source, "experiments_avoided", None),
        "scars_queried": getattr(source, "scars_queried", None),
    }
    return events, meta


def _elapsed_s(events: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> float:
    raw = meta.get("elapsed_s")
    try:
        if raw is not None and float(raw) > 0:
            return float(raw)
    except (TypeError, ValueError):
        pass
    if not events:
        return 0.0
    return max(_t_s(e) for e in events)


# ---------------------------------------------------------------------------
# Per-event identity extractors
# ---------------------------------------------------------------------------

def _rejection_id(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    for key in ("scar_id", "hypothesis_family", "idea", "family"):
        value = payload.get(key)
        if value:
            return str(value)
    return json.dumps(payload, sort_keys=True, default=str)[:160] or "rejection"


def _id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def _refill_token(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    ids = _id_list(payload.get("unit_ids") or payload.get("ids") or payload.get("novel_ids"))
    if not ids:
        cites = event.get("cites")
        if isinstance(cites, list) and cites:
            ids = [str(c) for c in cites if str(c).strip()]
    return json.dumps(sorted(set(ids)), separators=(",", ":"))


def _receipt_id(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    receipt = payload.get("receipt") or payload.get("path")
    if receipt:
        return str(receipt)
    cites = event.get("cites")
    if isinstance(cites, list):
        for item in cites:
            text = str(item)
            if text.endswith(".json") or "receipts/" in text:
                return text
        if cites:
            return str(cites[0])
    what = payload.get("what")
    return str(what) if what else "ingest"


def _unit_mapping(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload(event)
    unit = payload.get("unit")
    if isinstance(unit, Mapping):
        return dict(unit)
    return {}


def _unit_id(event: Mapping[str, Any]) -> str:
    unit = _unit_mapping(event)
    for key in ("id", "unit_id"):
        token = str(unit.get(key) or "").strip()
        if token:
            return token
    payload = _payload(event)
    for key in ("unit_id", "id", "job_id"):
        token = str(payload.get(key) or "").strip()
        if token:
            return token
    return ""


def _launch_fields(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload(event)
    unit = _unit_mapping(event)
    command = payload.get("command") or unit.get("command")
    capability = payload.get("capability") or unit.get("capability")
    label = payload.get("label") or unit.get("label") or unit.get("name")
    argv0 = payload.get("argv0") or unit.get("argv0")
    family = (
        unit.get("family")
        or unit.get("hypothesis_family")
        or payload.get("family")
        or payload.get("hypothesis_family")
    )
    generator = payload.get("generator") or unit.get("generator")
    description = unit.get("description") or payload.get("description")
    uid = _unit_id(event)
    return {
        "id": uid,
        "capability": capability,
        "label": label,
        "argv0": argv0,
        "command": command,
        "family": family,
        "generator": generator,
        "description": description,
    }


def _work_label(fields: Mapping[str, Any]) -> str:
    for key in ("capability", "label"):
        value = fields.get(key)
        if value:
            return str(value)
    generator = fields.get("generator")
    if isinstance(generator, Mapping):
        organ = generator.get("organ") or ""
        school = generator.get("school") or ""
        model = generator.get("model") or ""
        glued = "|".join(str(x) for x in (organ, school, model) if x)
        if glued:
            return f"generator:{glued}"
    family = fields.get("family")
    if family:
        return f"family:{family}"
    uid = str(fields.get("id") or "").strip()
    if uid:
        return f"id:{uid}"
    return "unlabelled"


def _is_unlabelled(fields: Mapping[str, Any]) -> bool:
    uid = str(fields.get("id") or "").strip()
    real_id = bool(uid) and uid.lower() not in _GENERIC_LABELS and not is_argv0_label(uid)
    family = str(fields.get("family") or "").strip()
    generator = fields.get("generator")
    has_generator = bool(generator) if not isinstance(generator, Mapping) else any(
        generator.get(k) for k in ("organ", "school", "model", "id")
    )
    description = str(fields.get("description") or "").strip()
    has_description = bool(description) and description.lower() not in _GENERIC_LABELS
    work = fields.get("capability") or fields.get("label")
    has_work = bool(work) and not is_argv0_label(work) and str(work).strip().lower() not in _GENERIC_LABELS
    return not (real_id or family or has_generator or has_description or has_work)


def _is_argv0_labelled(fields: Mapping[str, Any]) -> bool:
    for key in ("capability", "label"):
        if is_argv0_label(fields.get(key)):
            return True
    uid = str(fields.get("id") or "").strip()
    if uid and is_argv0_label(uid):
        return True
    return False


def _decision_token(event: Mapping[str, Any]) -> str:
    kind = _kind(event)
    payload = _payload(event)
    if kind in REJECTION_KINDS:
        return f"{kind}:{_rejection_id(event)}"
    if kind in REFILL_KINDS:
        return f"{kind}:{_refill_token(event)}"
    uid = _unit_id(event)
    if uid:
        return f"{kind}:{uid}"
    for key in ("id", "unit_id", "scar_id", "handle_id", "receipt"):
        value = payload.get(key)
        if value:
            return f"{kind}:{value}"
    ids = _id_list(payload.get("unit_ids") or payload.get("ids") or payload.get("hits"))
    if ids:
        return f"{kind}:{json.dumps(sorted(set(ids)), separators=(',', ':'))}"
    n = payload.get("n")
    if n is not None and not ids:
        return f"{kind}:n={n}"
    dumped = json.dumps(payload, sort_keys=True, default=str)
    return f"{kind}:{dumped[:160] if dumped != '{}' else 'empty'}"


def _kill_family(event: Mapping[str, Any]) -> tuple[str, str]:
    payload = _payload(event)
    family = str(payload.get("family") or payload.get("hypothesis_family") or "").strip()
    oid = str(payload.get("id") or "").strip()
    warrant = str(payload.get("warrant") or payload.get("terminal") or "").strip().lower()
    return family or oid, warrant


def _launch_family(event: Mapping[str, Any]) -> str:
    fields = _launch_fields(event)
    family = str(fields.get("family") or "").strip()
    if family:
        return family
    uid = str(fields.get("id") or "")
    if uid.startswith("WU.IMPROVEMENT."):
        return uid[len("WU.IMPROVEMENT.") :]
    return uid


def _payload_token(event: Mapping[str, Any]) -> str:
    kind = _kind(event)
    payload = _payload(event)
    for key in (
        "id",
        "unit_id",
        "receipt",
        "scar_id",
        "entry_id",
        "hypothesis_family",
        "job_id",
    ):
        value = payload.get(key)
        if value:
            return f"{kind}:{value}"
    unit = payload.get("unit")
    if isinstance(unit, Mapping) and unit.get("id"):
        return f"{kind}:{unit['id']}"
    ids = _id_list(payload.get("unit_ids") or payload.get("ids") or payload.get("units"))
    if ids:
        return f"{kind}:{json.dumps(sorted(set(ids)), separators=(',', ':'))}"
    dumped = json.dumps(payload, sort_keys=True, default=str)
    return f"{kind}:{dumped[:160]}"


# ---------------------------------------------------------------------------
# Named-axis evaluators
# ---------------------------------------------------------------------------

def _eval_rejections(
    events: Sequence[Mapping[str, Any]], elapsed_s: float
) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in REJECTION_KINDS]
    tokens = [_rejection_id(e) for e in rows]
    extra = {
        "n_events": len(rows),
        "t_s_min": min((_t_s(e) for e in rows), default=None),
        "t_s_max": max((_t_s(e) for e in rows), default=None),
        "early_cluster": False,
    }
    if not tokens:
        return _vacuous(
            "rejections",
            "no rejections; vacuous non-degeneracy (zero of this axis is not the table-replay pathology)",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    reasons: list[str] = []
    ts = [_t_s(e) for e in rows]
    clustered = False
    if (
        len(rows) >= REJECTION_EARLY_CLUSTER_MIN_N
        and elapsed_s >= REJECTION_EARLY_CLUSTER_MIN_ELAPSED_S
    ):
        span = max(ts) - min(ts)
        frac = (max(ts) / elapsed_s) if elapsed_s else 1.0
        clustered = (
            span <= REJECTION_EARLY_CLUSTER_WINDOW_S
            and frac <= REJECTION_EARLY_CLUSTER_MAX_FRAC
        )
        extra["early_cluster"] = clustered
        extra["cluster_span_s"] = round(span, 6)
        extra["cluster_last_frac"] = round(frac, 6)
        if clustered:
            reasons.append(
                f"all {len(rows)} rejections in t_s {min(ts):.0f}-{max(ts):.0f} "
                f"of elapsed {elapsed_s:.0f}s (span {span:.0f}s); a table replayed once"
            )
    if (
        len(tokens) >= REJECTION_RATIO_MIN_N
        and stats["unique_ratio"] < REJECTION_MIN_UNIQUE_RATIO
    ):
        reasons.append(
            f"unique_ratio {stats['unique_ratio']:.3f} < {REJECTION_MIN_UNIQUE_RATIO} "
            f"({stats['unique']}/{stats['total']})"
        )
    if stats["largest_repeat_run"] > REJECTION_MAX_CONSECUTIVE_RUN:
        reasons.append(
            f"largest_repeat_run {stats['largest_repeat_run']} > {REJECTION_MAX_CONSECUTIVE_RUN}"
        )
    degenerate = bool(reasons)
    reason = (
        "; ".join(reasons)
        if reasons
        else f"rejections unique={stats['unique']} total={stats['total']} under thresholds"
    )
    extra["early_cluster"] = clustered
    return _row("rejections", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_refills(
    events: Sequence[Mapping[str, Any]], elapsed_s: float
) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in REFILL_KINDS]
    tokens = [_refill_token(e) for e in rows]
    ts = [_t_s(e) for e in rows]
    last_t = max(ts) if ts else None
    extra: dict[str, Any] = {
        "n_events": len(rows),
        "last_t_s": last_t,
        "elapsed_after_last_s": (round(elapsed_s - last_t, 6) if last_t is not None else None),
        "stopped_early": False,
    }
    if not tokens:
        return _vacuous(
            "refills",
            "no refills; vacuous non-degeneracy (zero of this axis is not identical-set replay)",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    reasons: list[str] = []
    if stats["consecutive_identical_pairs"] > REFILL_IDENTICAL_CONSECUTIVE_MAX:
        reasons.append(
            f"{stats['consecutive_identical_pairs']} consecutive identical refill set(s); "
            f"largest_repeat_run={stats['largest_repeat_run']}"
        )
    if (
        last_t is not None
        and elapsed_s >= REFILL_STOPPED_EARLY_MIN_ELAPSED_S
        and (last_t / elapsed_s) <= REFILL_STOPPED_EARLY_MAX_LAST_FRAC
    ):
        extra["stopped_early"] = True
        reasons.append(
            f"last refill at t_s {last_t:.0f} of elapsed {elapsed_s:.0f}s "
            f"({extra['elapsed_after_last_s']}s of tail with no refill)"
        )
    degenerate = bool(reasons)
    reason = (
        "; ".join(reasons)
        if reasons
        else (
            f"refills unique_sets={stats['unique']} total={stats['total']} "
            f"consecutive_identical={stats['consecutive_identical_pairs']}"
        )
    )
    return _row("refills", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_ingestion(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in INGEST_KINDS]
    tokens = [_receipt_id(e) for e in rows]
    extra: dict[str, Any] = {"n_events": len(rows)}
    if not tokens:
        return _vacuous(
            "ingestion",
            "no ingestions; vacuous non-degeneracy (zero of this axis is not a 29-times loop)",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    extra["specimen_verification_ingests"] = sum(
        1 for t in tokens if "SPECIMEN_VERIFICATION" in t
    )
    degenerate = stats["max_item_count"] > INGEST_MAX_REPEATS
    if degenerate:
        reason = (
            f"{stats['most_repeated']} ingested {stats['max_item_count']} times "
            f"(line is {INGEST_MAX_REPEATS}; a receipt read twice is not this)"
        )
    else:
        reason = (
            f"ingestion unique={stats['unique']} total={stats['total']} "
            f"max_item_count={stats['max_item_count']} <= {INGEST_MAX_REPEATS}"
        )
    return _row("ingestion", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_launches(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in LAUNCH_KINDS]
    fields_list = [_launch_fields(e) for e in rows]
    tokens = [_work_label(f) for f in fields_list]
    n_argv0 = sum(1 for f in fields_list if _is_argv0_labelled(f))
    n_unlabelled = sum(1 for f in fields_list if _is_unlabelled(f))
    n_bad = n_argv0 + n_unlabelled
    extra = {
        "n_events": len(rows),
        "n_argv0_labelled": n_argv0,
        "n_unlabelled": n_unlabelled,
        "n_unlabelled_or_argv0": n_bad,
        "argv0_examples": sorted(
            {
                str(f.get("capability") or f.get("label") or f.get("id"))
                for f in fields_list
                if _is_argv0_labelled(f)
            }
        )[:8],
    }
    if not rows:
        return _vacuous(
            "launches",
            "no launches; vacuous non-degeneracy",
            extra,
        )
    degenerate = n_bad > LABELLING_ARGV0_OR_UNLABELLED_MAX
    if degenerate:
        reason = (
            f"{n_argv0} argv[0]-labelled and {n_unlabelled} unlabelled launch(es); "
            f"the timeline cannot distinguish them (line is "
            f"{LABELLING_ARGV0_OR_UNLABELLED_MAX})"
        )
    else:
        reason = (
            f"launches labelled; argv0={n_argv0} unlabelled={n_unlabelled} "
            f"unique_work_labels={len(set(tokens))}/{len(tokens)}"
        )
    return _row("launches", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_workunit_ids(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in LAUNCH_KINDS]
    tokens = [_unit_id(e) or f"missing:{i}" for i, e in enumerate(rows)]
    extra = {"n_events": len(rows), "n_missing_id": sum(1 for t in tokens if t.startswith("missing:"))}
    if not tokens:
        return _vacuous(
            "workunit_ids",
            "no launched WorkUnit ids; vacuous non-degeneracy",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    degenerate = bool(WORKUNIT_IDS_MUST_BE_UNIQUE) and stats["unique"] < stats["total"]
    if degenerate:
        reason = (
            f"duplicate WorkUnit ids: unique={stats['unique']} total={stats['total']} "
            f"most_repeated={stats['most_repeated']!r} x{stats['max_item_count']}"
        )
    else:
        reason = f"WorkUnit ids unique={stats['unique']} total={stats['total']}"
    return _row("workunit_ids", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_decisions(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [e for e in events if _kind(e) in DECISION_KINDS]
    tokens = [_decision_token(e) for e in rows]
    extra = {
        "n_events": len(rows),
        "t_s_min": min((_t_s(e) for e in rows), default=None),
        "t_s_max": max((_t_s(e) for e in rows), default=None),
    }
    if not tokens:
        return _vacuous(
            "decisions",
            "no decision events; vacuous non-degeneracy",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    reasons: list[str] = []
    if (
        len(tokens) >= DECISION_RATIO_MIN_N
        and stats["unique_ratio"] < DECISION_MIN_UNIQUE_RATIO
    ):
        reasons.append(
            f"unique_ratio {stats['unique_ratio']:.3f} < {DECISION_MIN_UNIQUE_RATIO} "
            f"({stats['unique']}/{stats['total']})"
        )
    if (
        len(tokens) >= DECISION_RUN_MIN_N
        and stats["largest_repeat_run"] > DECISION_MAX_CONSECUTIVE_RUN
    ):
        reasons.append(
            f"largest_repeat_run {stats['largest_repeat_run']} > {DECISION_MAX_CONSECUTIVE_RUN}"
        )
    degenerate = bool(reasons)
    reason = (
        "; ".join(reasons)
        if reasons
        else f"decisions unique={stats['unique']} total={stats['total']} under thresholds"
    )
    return _row("decisions", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_scars(
    events: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    rejected: list[str] = []
    killed_scar: list[str] = []
    relaunched: list[str] = []
    dead: set[str] = set()
    for row in meta.get("experiments_avoided") or ():
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("family") or "").strip()
        if family:
            dead.add(family)
    for event in events:
        kind = _kind(event)
        if kind in REJECTION_KINDS:
            ident = _rejection_id(event)
            rejected.append(ident)
            dead.add(ident)
            family = str(_payload(event).get("hypothesis_family") or "").strip()
            if family:
                dead.add(family)
        elif kind in SCAR_KILL_KINDS:
            family, warrant = _kill_family(event)
            if family and (
                "scar" in warrant or warrant in {"killed_scar", "scar"}
            ):
                killed_scar.append(family)
                dead.add(family)
        elif kind in LAUNCH_KINDS:
            family = _launch_family(event)
            if family and family in dead:
                relaunched.append(family)
    # Duck-typed TrialRecord extras, if the event log omitted a warrant.
    for row in meta.get("killed") or ():
        if not isinstance(row, Mapping):
            continue
        warrant = str(row.get("warrant") or "").lower()
        family = str(row.get("family") or row.get("id") or "").strip()
        if family and "scar" in warrant:
            if family not in killed_scar:
                killed_scar.append(family)
            dead.add(family)
    for row in meta.get("launched") or ():
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("family") or row.get("hypothesis_family") or "").strip()
        if family and family in dead and family not in relaunched:
            # only count if a launch event of this family exists; the event loop
            # already captured those. This branch is a belt for records whose
            # launched list is ahead of the event log (the dead-scar control
            # appends a launch after record()).
            relaunched.append(family)
    tokens = rejected + [f"kill:{f}" for f in killed_scar] + [f"relaunch:{f}" for f in relaunched]
    extra = {
        "n_rejections": len(rejected),
        "n_scar_kills": len(killed_scar),
        "n_dead_family_relaunches": len(relaunched),
        "relaunched_families": sorted(set(relaunched))[:12],
    }
    if not tokens:
        return _vacuous(
            "scars",
            "no scar-facing events; vacuous non-degeneracy",
            extra,
        )
    stats = distinct_vs_repeated(tokens)
    reasons: list[str] = []
    if len(relaunched) > DEAD_SCAR_RELAUNCHES_MAX:
        reasons.append(
            f"relaunched scar-dead families {sorted(set(relaunched))[:6]} "
            f"(n={len(relaunched)}; line is {DEAD_SCAR_RELAUNCHES_MAX})"
        )
    if (
        len(rejected) >= REJECTION_RATIO_MIN_N
        and distinct_vs_repeated(rejected)["unique_ratio"] < REJECTION_MIN_UNIQUE_RATIO
    ):
        rstats = distinct_vs_repeated(rejected)
        reasons.append(
            f"scar-refusal unique_ratio {rstats['unique_ratio']:.3f} < "
            f"{REJECTION_MIN_UNIQUE_RATIO} ({rstats['unique']}/{rstats['total']})"
        )
    degenerate = bool(reasons)
    reason = (
        "; ".join(reasons)
        if reasons
        else (
            f"scars unique={stats['unique']} total={stats['total']} "
            f"dead_relaunches={len(relaunched)}"
        )
    )
    return _row("scars", tokens, degenerate=degenerate, reason=reason, extra=extra)


def _eval_other_kinds(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        kind = _kind(event)
        if not kind or kind in NAMED_KIND_UNION:
            continue
        grouped[kind].append(event)
    rows: list[dict[str, Any]] = []
    for kind in sorted(grouped):
        tokens = [_payload_token(e) for e in grouped[kind]]
        rows.append(
            _row(
                f"kind:{kind}",
                tokens,
                degenerate=False,
                reason=(
                    "carried by the timeline; reported distinct-vs-repeated; "
                    "not a named fail axis (mission-state writes and similar "
                    "healthy repetition live here)"
                ),
                fail_eligible=False,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Public measure
# ---------------------------------------------------------------------------

def measure(source: Any) -> dict[str, Any]:
    """Per-axis distinct-vs-repeated. FAIL if any named axis is degenerate.

    `source` may be a filesystem path, a receipt dict, a list of events, or
    any object with an `.events` attribute (improvement_trial.TrialRecord).
    """
    events, meta = _events_of(source)
    elapsed = _elapsed_s(events, meta)
    named = [
        _eval_rejections(events, elapsed),
        _eval_refills(events, elapsed),
        _eval_ingestion(events),
        _eval_launches(events),
        _eval_workunit_ids(events),
        _eval_decisions(events),
        _eval_scars(events, meta),
    ]
    carried = _eval_other_kinds(events)
    axes = named + carried
    degenerate_axes = [row["axis"] for row in named if row["degenerate"]]
    if degenerate_axes:
        verdict = "FAIL"
        reason = "degenerate on " + ", ".join(degenerate_axes)
    else:
        verdict = "PASS"
        reason = "every named axis is non-degenerate under the stated thresholds"
    launches = next(r for r in named if r["axis"] == "launches")
    ingestion = next(r for r in named if r["axis"] == "ingestion")
    source_path = meta.get("_source_path")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "verdict": verdict,
        "reason": reason,
        "elapsed_s": elapsed,
        "n_events": len(events),
        "n_kinds": len({_kind(e) for e in events if _kind(e)}),
        "axes": axes,
        "named_axes": NAMED_AXES,
        "degenerate_axes": degenerate_axes,
        "table_not_a_score": True,
        "fails_on_degeneracy_even_when_nominal_conditions_met": True,
        "unlabelled_or_argv0_units": int(launches.get("n_unlabelled_or_argv0") or 0),
        "n_argv0_labelled": int(launches.get("n_argv0_labelled") or 0),
        "n_unlabelled": int(launches.get("n_unlabelled") or 0),
        "specimen_verification_ingests": int(
            ingestion.get("specimen_verification_ingests") or 0
        ),
        "thresholds": dict(THRESHOLDS),
        "source_path": source_path,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "score": None,
    }


def axis_table(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Contract projection: unique, total, largest repeat run, consecutive identical."""
    table: list[dict[str, Any]] = []
    for row in report.get("axes") or []:
        table.append(
            {
                "axis": row.get("axis"),
                "unique": row.get("unique"),
                "total": row.get("total"),
                "largest_repeat_run": row.get("largest_repeat_run"),
                "consecutive_emissions_identical": row.get(
                    "consecutive_emissions_identical"
                ),
                "degenerate": row.get("degenerate"),
                "fail_eligible": row.get("fail_eligible"),
                "reason": row.get("reason"),
            }
        )
    return table


def axis_by_name(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in report.get("axes") or []:
        if row.get("axis") == name:
            return dict(row)
    raise KeyError(name)


def replay_disk_timelines() -> dict[str, Any]:
    """The only proof that means anything: two real timelines, opposite verdicts."""
    one_h = measure(REPO / TIMELINE_1H_REL)
    detached = measure(REPO / DETACHED_TRIAL_REL)
    return {
        "autonomy_1h": one_h,
        "detached_work_trial": detached,
        "opposite_verdicts": one_h["verdict"] == "FAIL"
        and detached["verdict"] == "PASS",
        "fixtures": False,
        "paths": {
            "autonomy_1h": TIMELINE_1H_REL,
            "detached_work_trial": DETACHED_TRIAL_REL,
        },
    }


def agreement_with_improvement_guards(
    *,
    judge_unmet: Sequence[str],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Duplicate WorkUnits and dead-scar repetition must not disagree.

    If the improvement-trial judge already failed those guards, the matching
    degeneracy axis must also be degenerate. Degeneracy may be stricter.
    """
    unmet = set(judge_unmet)
    wu = axis_by_name(report, "workunit_ids")
    scars = axis_by_name(report, "scars")
    dup_judge = "no_duplicate_workunits" in unmet
    scar_judge = "no_repeated_scar" in unmet
    dup_agree = (not dup_judge) or bool(wu["degenerate"])
    scar_agree = (not scar_judge) or bool(scars["degenerate"])
    return {
        "duplicate_workunits": {
            "judge_unmet": dup_judge,
            "degeneracy_axis_degenerate": bool(wu["degenerate"]),
            "agree": dup_agree,
        },
        "dead_scar_repetition": {
            "judge_unmet": scar_judge,
            "degeneracy_axis_degenerate": bool(scars["degenerate"]),
            "agree": scar_agree,
        },
        "agree": dup_agree and scar_agree,
        "rule": (
            "if the judge flags duplicate WorkUnits, workunit_ids is degenerate; "
            "if the judge flags dead-scar repetition, scars is degenerate"
        ),
    }


def _provenance(rel: str) -> dict[str, Any]:
    path = REPO / rel
    present = path.is_file()
    digest = sha256_file(path) if present else None
    return {
        "rel": rel,
        "present": present,
        "sha256": digest,
        "n_bytes": path.stat().st_size if present else None,
    }


def _public_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Receipt projection: named-axis table plus compact extras, no event dump."""
    named = [row for row in report.get("axes") or [] if row.get("axis") in NAMED_AXES]
    extras = [
        {
            "axis": row.get("axis"),
            "unique": row.get("unique"),
            "total": row.get("total"),
            "largest_repeat_run": row.get("largest_repeat_run"),
            "consecutive_emissions_identical": row.get("consecutive_emissions_identical"),
        }
        for row in report.get("axes") or []
        if row.get("axis") not in NAMED_AXES
    ]
    keep = (
        "verdict",
        "reason",
        "elapsed_s",
        "n_events",
        "n_kinds",
        "degenerate_axes",
        "unlabelled_or_argv0_units",
        "n_argv0_labelled",
        "n_unlabelled",
        "specimen_verification_ingests",
        "table_not_a_score",
        "fails_on_degeneracy_even_when_nominal_conditions_met",
        "source_path",
    )
    out = {k: report.get(k) for k in keep}
    out["named_axis_table"] = named
    out["carried_kind_table"] = extras
    out["score"] = None
    return out


def build() -> Path:
    replay = replay_disk_timelines()
    one_h = replay["autonomy_1h"]
    detached = replay["detached_work_trial"]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "obligation": (
            "AUTONOMY EVIDENCE MUST BE NON-DEGENERATE. Rejections that never "
            "change between runs, refills that re-offer an identical set, a "
            "receipt ingested 29 times, and unlabelled units are all degenerate "
            "evidence. A trial reports DISTINCT-VS-REPEATED FOR EVERY AXIS and "
            "FAILS ON DEGENERACY EVEN WHEN EVERY CONDITION IS NOMINALLY MET."
        ),
        "table_not_a_score": True,
        "fails_on_degeneracy_even_when_nominal_conditions_met": True,
        "thresholds": dict(THRESHOLDS),
        "threshold_defense": dict(THRESHOLD_DEFENSE),
        "named_axes": list(NAMED_AXES),
        "replay": {
            "fixtures": False,
            "opposite_verdicts": replay["opposite_verdicts"],
            "autonomy_1h": {
                **_public_report(one_h),
                "provenance": _provenance(TIMELINE_1H_REL),
                "must_verdict": "FAIL",
            },
            "detached_work_trial": {
                **_public_report(detached),
                "provenance": _provenance(DETACHED_TRIAL_REL),
                "must_verdict": "PASS",
            },
        },
        "one_h_measured": {
            "verdict": one_h["verdict"],
            "degenerate_axes": one_h["degenerate_axes"],
            "n_argv0_labelled": one_h["n_argv0_labelled"],
            "n_unlabelled": one_h["n_unlabelled"],
            "specimen_verification_ingests": one_h["specimen_verification_ingests"],
            "rejections_total": axis_by_name(one_h, "rejections")["total"],
            "rejections_unique": axis_by_name(one_h, "rejections")["unique"],
            "refills_total": axis_by_name(one_h, "refills")["total"],
            "refills_consecutive_identical": axis_by_name(one_h, "refills")[
                "consecutive_emissions_identical"
            ],
        },
        "detached_measured": {
            "verdict": detached["verdict"],
            "degenerate_axes": detached["degenerate_axes"],
            "n_argv0_labelled": detached["n_argv0_labelled"],
            "n_unlabelled": detached["n_unlabelled"],
            "refills_total": axis_by_name(detached, "refills")["total"],
            "refills_unique": axis_by_name(detached, "refills")["unique"],
            "workunit_ids_unique": axis_by_name(detached, "workunit_ids")["unique"],
            "workunit_ids_total": axis_by_name(detached, "workunit_ids")["total"],
        },
        "claim_boundary": (
            "STATIC_ONLY sidecar. No GPU lease and no hardware measurement. "
            "Counts are event-log cardinalities from receipts already on disk. "
            "The 1h timeline is replayed and must FAIL; the detached-work trial "
            "timeline is replayed and must PASS. Thresholds are stated and "
            "defended in this receipt. A trial that meets every nominal "
            "condition and is degenerate on any named axis is FAIL."
        ),
        "does_not_edit": [
            "tools/future/detached_trial.py",
            "tools/future/autonomy_run.py",
        ],
        "api_for_callers": {
            "measure": "tools.future.autonomy_degeneracy.measure",
            "axis_table": "tools.future.autonomy_degeneracy.axis_table",
            "replay_disk_timelines": (
                "tools.future.autonomy_degeneracy.replay_disk_timelines"
            ),
            "note": (
                "detached_trial.py and autonomy_run.py were not edited. "
                "Call measure() on a timeline or TrialRecord."
            ),
        },
        "verdict": (
            "PASS"
            if replay["opposite_verdicts"]
            and one_h["verdict"] == "FAIL"
            and detached["verdict"] == "PASS"
            else "FAIL"
        ),
        "head": git("rev-parse", "HEAD"),
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--timeline", default=None, help="path to a timeline JSON")
    args = parser.parse_args(argv)
    if args.timeline:
        report = measure(args.timeline)
        print(json.dumps(
            {
                "verdict": report["verdict"],
                "reason": report["reason"],
                "degenerate_axes": report["degenerate_axes"],
                "table": axis_table(report),
            },
            indent=2,
        ))
        return 0 if report["verdict"] == "PASS" else 1
    path = build()
    doc = json.loads(path.read_text())
    print(f"wrote {path}")
    print(f"verdict={doc.get('verdict')}")
    print(f"1h={doc['replay']['autonomy_1h']['verdict']} "
          f"detached={doc['replay']['detached_work_trial']['verdict']}")
    if args.selftest or args.record:
        return 0 if doc.get("verdict") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
