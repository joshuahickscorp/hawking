"""CANONICAL WORK EVENTS — one contract so the driver and the judge score the same acts.

The 15m autonomy trial (e08529b84) emitted `receipt_ingested`; the judge scored
`result_ingested`. Seventy real ingestions of real receipts counted as zero, and
nothing errored. Separately, refilling work and reporting what work remains were
the same event (`next_work_left`), so a daemon that actively replaced completed
work was indistinguishable from one that merely had leftovers. A resident must
not get refill credit because static work happened to exist.

This module is the taxonomy both sides must speak. It does not run the loop, it
does not judge a trial, and it does not invent a hardware number. It refuses an
event that names a kind but lacks the payload that would make the claim
checkable. It parses retired names; it will not emit them.

    python3 tools/future/work_events.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import write_receipt

RECEIPT = "WORK_EVENT_CONTRACT.json"
SCHEMA = "hawking.future.work_events.v1"

# Canonical kinds, in the order the six acts happen. Required keys are the
# evidence that would let a stranger check the claim; without them the event
# is a label, not an observation.
EVENT_KINDS: dict[str, dict[str, Any]] = {
    "FRONTIER_HAS_WORK": {
        "claims": (
            "the frontier reports runnable items exist; a fact about the world; "
            "earns the daemon nothing"
        ),
        "required": ("unit_ids",),
        "nonempty": ("unit_ids",),
    },
    "WORK_GENERATED": {
        "claims": (
            "the daemon PRODUCED a candidate that did not exist before "
            "(a hypothesis, a derived unit)"
        ),
        "required": ("candidate",),
    },
    "WORK_REFILLED": {
        "claims": (
            "the daemon replaced completed or invalidated work by asking the "
            "frontier again"
        ),
        "required": ("unit_ids", "queue_depth"),
        "nonempty": ("unit_ids",),
    },
    "WORK_SCHEDULED": {
        "claims": "a valid WorkUnit was admitted to the queue",
        "required": ("unit",),
    },
    "WORK_LAUNCHED": {
        "claims": "execution actually started",
        "required": ("unit",),
    },
    "RESULT_INGESTED": {
        "claims": "a real receipt was read back and routed",
        "required": ("cites",),
        "nonempty": ("cites",),
    },
}

CANONICAL_KINDS: tuple[str, ...] = tuple(EVENT_KINDS)

# Compatibility PARSING only. Never teach anything to emit these.
LEGACY_ALIASES: dict[str, str] = {
    # 15m autonomy trial, e08529b84: the driver emitted receipt_ingested;
    # the judge scored result_ingested. Seventy real ingestions of real
    # receipts counted as zero, and nothing errored.
    "receipt_ingested": "RESULT_INGESTED",
}

# Precanonical spellings the driver and judge still speak. Parse-only: a later
# lane can switch both sides together. They are not LEGACY_ALIASES because
# teaching the driver to keep emitting them is the current state, not a
# regression against the 15m silent-drop. next_work_left is the leftover /
# world-fact half of the 1h refill conflation (d6eb11c79); it is NOT an alias
# of WORK_REFILLED — that was the bug.
PRECANONICAL: dict[str, str] = {
    "next_work_left": "FRONTIER_HAS_WORK",
    "workunit_launched": "WORK_LAUNCHED",
}

# The current driver records the depth it observed under this name. The
# canonical key is queue_depth; validate accepts either so an existing
# timeline that DID observe the depth is not rejected for a spelling.
_QUEUE_DEPTH_KEYS: tuple[str, ...] = ("queue_depth", "queue_remaining_when_asked")

_IDENTITY_KEYS: tuple[str, ...] = ("id", "unit_id", "hypothesis", "hypothesis_family")


def _payload_of(event: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if "payload" not in event:
        return {}, None
    raw = event["payload"]
    if not isinstance(raw, Mapping):
        return None, "payload is not an object"
    return dict(raw), None


def _lookup(event: Mapping[str, Any], key: str) -> Any:
    """Payload first, then the event root. cites lives at the root in the
    timeline schema the judge already scores; forcing it into payload would
    drop the field that currently makes ingest checkable."""
    payload, err = _payload_of(event)
    if err is None and payload is not None and key in payload:
        return payload[key]
    if key in event and key != "payload":
        return event[key]
    return _MISSING


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _ids(value: Any) -> tuple[list[str] | None, str | None]:
    if not isinstance(value, list):
        return None, "is not a list"
    out: list[str] = []
    for item in value:
        token = str(item).strip() if item is not None else ""
        if not token:
            return None, "contains an empty id"
        out.append(token)
    return out, None


def _nonneg_int(value: Any) -> bool:
    # bool is a subclass of int; a depth of True is not a depth.
    return type(value) is int and value >= 0


def _candidate_identifies(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return any(str(value.get(k) or "").strip() for k in _IDENTITY_KEYS)


def _unit_identifies(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(str(value.get("id") or "").strip())


def validate(event: Any) -> tuple[bool, str]:
    """Reject an event that names a kind but lacks the payload that would
    make its claim checkable. A legacy name is not a canonical kind: the
    caller must canonicalize() first, so a silent rename cannot pass again."""
    if not isinstance(event, Mapping):
        return False, "event is not an object"
    kind = event.get("kind")
    if not isinstance(kind, str) or not kind:
        return False, "event has no kind"
    if kind in LEGACY_ALIASES:
        return False, (
            f"kind {kind!r} is a legacy alias of {LEGACY_ALIASES[kind]}; "
            "canonicalize() first"
        )
    if kind in PRECANONICAL:
        return False, (
            f"kind {kind!r} is a precanonical spelling of {PRECANONICAL[kind]}; "
            "canonicalize() first"
        )
    folded = kind.upper().replace("-", "_")
    if kind not in EVENT_KINDS and folded in EVENT_KINDS:
        return False, (
            f"kind {kind!r} is a precanonical spelling of {folded}; "
            "canonicalize() first"
        )
    if kind not in EVENT_KINDS:
        return False, f"unknown kind {kind!r}"

    payload, payload_err = _payload_of(event)
    if payload_err:
        return False, f"{kind} {payload_err}"

    spec = EVENT_KINDS[kind]
    required: tuple[str, ...] = tuple(spec["required"])
    nonempty: tuple[str, ...] = tuple(spec.get("nonempty") or ())

    for key in required:
        if key == "queue_depth":
            depth = _MISSING
            for alias in _QUEUE_DEPTH_KEYS:
                got = _lookup(event, alias)
                if got is not _MISSING:
                    depth = got
                    break
            if depth is _MISSING:
                return False, (
                    f"{kind} missing queue_depth (the depth observed when "
                    "the frontier was asked)"
                )
            if not _nonneg_int(depth):
                return False, f"{kind} queue_depth must be a non-negative int"
            continue

        value = _lookup(event, key)
        if value is _MISSING:
            return False, f"{kind} missing {key}"

        if key == "candidate":
            if not _candidate_identifies(value):
                return False, (
                    f"{kind} candidate does not identify a produced unit"
                )
            continue
        if key == "unit":
            if not _unit_identifies(value):
                return False, f"{kind} unit has no id"
            continue
        if key in nonempty:
            ids, err = _ids(value)
            if err:
                return False, f"{kind} {key} {err}"
            if ids is None or not ids:
                if kind == "WORK_REFILLED":
                    return False, (
                        f"{kind} unit_ids is empty; a refill that added "
                        "nothing is not a refill"
                    )
                if kind == "FRONTIER_HAS_WORK":
                    return False, (
                        f"{kind} unit_ids is empty; the claim is that "
                        "runnable items exist"
                    )
                if kind == "RESULT_INGESTED":
                    return False, (
                        f"{kind} cites is empty; a real receipt was not named"
                    )
                return False, f"{kind} {key} is empty"

    return True, "ok"


def canonicalize(event: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a spoken or retired name to the canonical kind.

    A rewritten event carries `legacy_kind` so it is never mistaken for one
    that was emitted correctly. Already-canonical events are copied as-is.
    Unknown kinds raise rather than pass through as success-shaped."""
    if not isinstance(event, Mapping):
        raise TypeError("event is not an object")
    out = dict(event)
    if isinstance(out.get("payload"), Mapping):
        out["payload"] = dict(out["payload"])
    if isinstance(out.get("cites"), list):
        out["cites"] = list(out["cites"])
    kind = out.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("event has no kind")
    if kind in EVENT_KINDS:
        return out
    if kind in LEGACY_ALIASES:
        out["legacy_kind"] = kind
        out["kind"] = LEGACY_ALIASES[kind]
        return out
    if kind in PRECANONICAL:
        out["legacy_kind"] = kind
        out["kind"] = PRECANONICAL[kind]
        return out
    folded = kind.upper().replace("-", "_")
    if folded in EVENT_KINDS:
        out["legacy_kind"] = kind
        out["kind"] = folded
        return out
    raise ValueError(f"unknown kind {kind!r}")


def make(
    kind: str,
    *,
    payload: Mapping[str, Any] | None = None,
    cites: Sequence[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a canonical event. Refuses a legacy name and refuses a claim
    whose payload would not survive validate()."""
    if kind in LEGACY_ALIASES:
        raise ValueError(
            f"refusing to emit legacy kind {kind!r}; emit {LEGACY_ALIASES[kind]}"
        )
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    body = dict(payload or {})
    body.update(fields)
    event: dict[str, Any] = {"kind": kind, "payload": body}
    if cites is not None:
        event["cites"] = [str(x) for x in cites]
    ok, why = validate(event)
    if not ok:
        raise ValueError(why)
    return event


def example(kind: str) -> dict[str, Any]:
    """One checkable event of this kind. Used by watched refusals so a
    validator nobody has watched accept is as guarded as one nobody has
    watched reject."""
    if kind == "FRONTIER_HAS_WORK":
        return make(kind, unit_ids=["FT.TOOLS.frontiers-refill"])
    if kind == "WORK_GENERATED":
        return make(
            kind,
            candidate={
                "id": "WU.GEN.ngram.1",
                "hypothesis_family": "ngram_school",
            },
        )
    if kind == "WORK_REFILLED":
        return make(
            kind,
            unit_ids=["WU.AUTONOMY.freshness.4"],
            queue_depth=3,
        )
    if kind == "WORK_SCHEDULED":
        return make(kind, unit={"id": "WU.AUTONOMY.freshness.4", "status": "pending"})
    if kind == "WORK_LAUNCHED":
        return make(kind, unit={"id": "WU.AUTONOMY.freshness.4", "status": "running"})
    if kind == "RESULT_INGESTED":
        return make(
            kind,
            cites=["receipts/future/DERIVED_FRESHNESS.json", "WU.AUTONOMY.freshness.4"],
            receipt="receipts/future/DERIVED_FRESHNESS.json",
            unit_id="WU.AUTONOMY.freshness.4",
        )
    raise ValueError(f"no example for {kind!r}")


def watched_refusals() -> list[dict[str, Any]]:
    """Every judgement this module makes, watched failing.

    A validator nobody has watched reject is a validator that will silently
    drift into fiction."""
    rows: list[dict[str, Any]] = []
    for kind in CANONICAL_KINDS:
        ok, why = validate(example(kind))
        rows.append({"trial": f"{kind}.valid", "refused": (not ok), "why": why, "expect_ok": True})
        missing = {"kind": kind, "payload": {}}
        ok, why = validate(missing)
        rows.append(
            {
                "trial": f"{kind}.missing_payload",
                "refused": (not ok),
                "why": why,
                "expect_ok": False,
            }
        )
    ok, why = validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {"unit_ids": [], "queue_depth": 2},
        }
    )
    rows.append(
        {
            "trial": "WORK_REFILLED.empty_unit_ids",
            "refused": (not ok),
            "why": why,
            "expect_ok": False,
        }
    )
    ok, why = validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {"unit_ids": ["WU.1"]},
        }
    )
    rows.append(
        {
            "trial": "WORK_REFILLED.missing_queue_depth",
            "refused": (not ok),
            "why": why,
            "expect_ok": False,
        }
    )
    ok, why = validate(
        {
            "kind": "FRONTIER_HAS_WORK",
            "payload": {"unit_ids": ["FT.TOOLS.frontiers-refill"]},
        }
    )
    rows.append(
        {
            "trial": "FRONTIER_HAS_WORK.is_not_a_refill",
            "refused": (not ok),
            "why": why,
            "expect_ok": True,
        }
    )
    as_refill = {
        "kind": "WORK_REFILLED",
        "payload": {"unit_ids": ["FT.TOOLS.frontiers-refill"]},
    }
    ok, why = validate(as_refill)
    rows.append(
        {
            "trial": "same_unit_ids_without_depth_is_not_WORK_REFILLED",
            "refused": (not ok),
            "why": why,
            "expect_ok": False,
        }
    )
    ok, why = validate(
        {
            "kind": "receipt_ingested",
            "cites": ["receipts/future/DERIVED_FRESHNESS.json"],
            "payload": {"receipt": "receipts/future/DERIVED_FRESHNESS.json"},
        }
    )
    rows.append(
        {
            "trial": "legacy_name_rejected_until_canonicalize",
            "refused": (not ok),
            "why": why,
            "expect_ok": False,
        }
    )
    return rows


def scan_partition_for_legacy_emits() -> list[dict[str, Any]]:
    """AST-walk production modules for EMITS of a retired kind.

    "A string constant in any other production module is an emit" was too broad
    and made this scan permanently red for the one thing a migration REQUIRES.
    Of the five it flagged, exactly one was an emit:

        model_bearing_torture.py:1845   emit  -> migrated to RESULT_INGESTED
        model_bearing_torture.py:2694   fixture event -> migrated
        model_bearing_torture.py:1444   CONSUMER set: kind in {legacy, ...}
        autonomy_degeneracy.py:225      CONSUMER set, beside "RESULT_INGESTED"
        power_torture.py:175            CONSUMER set

    A consumer that still ACCEPTS the old name is how old timelines stay
    readable; flagging it as an emit asks the migration to break its own
    evidence. Constants inside a set/list/tuple/frozenset literal are reads, not
    emits, and are skipped - a genuine emit passes the kind as a bare argument.
    """
    here = Path(__file__).resolve()
    root = here.parent
    hits: list[dict[str, Any]] = []
    forbidden = frozenset(LEGACY_ALIASES)
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        if path.resolve() == here:
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError) as exc:
            hits.append(
                {
                    "file": f"tools/future/{path.name}",
                    "line": 0,
                    "kind": None,
                    "why": f"unreadable:{type(exc).__name__}",
                }
            )
            continue
        membership = {
            id(elt)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Set, ast.List, ast.Tuple))
            for elt in n.elts
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in membership:
                    continue  # a collection element is a read, not an emit
                if node.value in forbidden:
                    hits.append(
                        {
                            "file": f"tools/future/{path.name}",
                            "line": int(getattr(node, "lineno", 0) or 0),
                            "kind": node.value,
                            "why": "string constant equals a LEGACY_ALIASES key",
                        }
                    )
    return hits


def build() -> Path:
    watched = watched_refusals()
    drifted = [r for r in watched if r["refused"] == r["expect_ok"]]
    if drifted:
        raise ValueError(f"watched refusals drifted: {drifted}")
    emits = scan_partition_for_legacy_emits()
    if emits:
        raise ValueError(f"legacy kinds emitted in the partition: {emits}")
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "One canonical work-event contract so the autonomy driver and the "
            "autonomy judge score the same acts. FRONTIER_HAS_WORK is a fact "
            "about the world and earns nothing; WORK_REFILLED is the daemon "
            "asking the frontier again and requires unit ids plus the queue "
            "depth it observed."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "kinds": {
            kind: {
                "claims": spec["claims"],
                "required": list(spec["required"]),
                "nonempty": list(spec.get("nonempty") or ()),
            }
            for kind, spec in EVENT_KINDS.items()
        },
        "legacy_aliases": dict(LEGACY_ALIASES),
        "precanonical_spellings": dict(PRECANONICAL),
        "legacy_alias_runs": {
            "receipt_ingested": (
                "15m autonomy trial e08529b84: driver emitted receipt_ingested; "
                "judge scored result_ingested; seventy real ingestions counted "
                "as zero and nothing errored"
            ),
        },
        "refill_vs_world_fact": (
            "frontiers.refill is currently identical to frontiers.next_work. "
            "The distinction is the daemon's act, not the frontier function: "
            "WORK_REFILLED requires the unit ids the ask produced AND the "
            "queue depth at the moment it asked. FRONTIER_HAS_WORK is the "
            "same list of ids with no depth, and earns no refill credit. "
            "The 1h trial (d6eb11c79) previously used next_work_left for both."
        ),
        "queue_depth_accepted_keys": list(_QUEUE_DEPTH_KEYS),
        "judge_kind_today": {
            "eval_ingest_completed_result": "result_ingested -> RESULT_INGESTED",
            "eval_refill_work": (
                "work_refilled -> WORK_REFILLED (judge does not yet require "
                "queue_depth; this contract does)"
            ),
            "eval_launch_valid_workunit": "workunit_launched -> WORK_LAUNCHED",
            "eval_leave_next_work": "next_work_left -> FRONTIER_HAS_WORK",
        },
        "driver_kind_today": {
            "result_ingested": "RESULT_INGESTED (fixed after e08529b84)",
            "work_refilled": "WORK_REFILLED (payload key is queue_remaining_when_asked)",
            "workunit_launched": "WORK_LAUNCHED (also used for generate and admit)",
            "next_work_left": "FRONTIER_HAS_WORK",
        },
        "watched_refusals": watched,
        "legacy_emits_in_partition": emits,
        "recovered_implementation": [
            "tools/future/autonomy_run.py _emit is the live emitter: "
            "result_ingested, work_refilled, workunit_launched, next_work_left",
            "tools/future/autonomy_trial.py eval_ingest_completed_result scores "
            "result_ingested; eval_refill_work scores work_refilled; "
            "eval_launch_valid_workunit scores workunit_launched; "
            "eval_leave_next_work scores next_work_left",
            "tools/future/frontiers.py next_work / refill — refill currently "
            "returns next_work, so the world-fact and the refill act are the "
            "same function; the event contract is what splits them",
        ],
        "gaps_closed": [
            "one canonical taxonomy of six acts with required payload keys",
            "validate() rejects a kind that lacks the payload that would make its claim checkable",
            "WORK_REFILLED without unit ids or without observed queue depth is rejected",
            "WORK_REFILLED with an empty unit list is rejected",
            "LEGACY_ALIASES maps receipt_ingested -> RESULT_INGESTED and names the 15m trial",
            "canonicalize() rewrites a retired name and sets legacy_kind so a rewrite is never mistaken for a correct emit",
            "FRONTIER_HAS_WORK is a distinct kind from WORK_REFILLED and earns the daemon nothing",
        ],
        "negative_findings": [
            "this lane cannot switch autonomy_run.py or autonomy_trial.py onto the canonical names (WRITE list)",
            "frontiers.refill is still identical to frontiers.next_work",
            "the driver still emits precanonical spellings; canonicalize() parses them, make() will not emit them",
            "the judge still scores precanonical names and does not require queue_depth on work_refilled",
            "WORK_GENERATED and WORK_SCHEDULED are not yet distinct emits in the loop (generate and admit currently look like WORK_LAUNCHED)",
            "the seventy ingestions dropped by e08529b84 are not retroactively re-scored; the contract prevents the next silent drop",
        ],
        "resident_callable": {
            "entry_point": "tools.future.work_events.validate(event)",
            "workunit": (
                "one CPU_ANALYSIS unit; the contract both the driver and the "
                "judge must speak before a refill or an ingest counts"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "an event that names a kind but lacks the payload that would "
                "make its claim checkable is rejected; a legacy name is rejected "
                "until canonicalize(); make() refuses to emit a legacy name"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/work_events.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
