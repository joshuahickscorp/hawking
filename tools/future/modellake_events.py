#!/usr/bin/env python3
"""G101 wiring: MODEL_SEALED emits work, with no human in the loop.

modellake_scheduler_view.py can see the lake and the watcher. Its receipt
still says is_this_wired: False, because a declared trigger is not a
consumer. This module is that consumer.

A specimen is sealed because a lake manifest is on disk with a resolved
sha and a byte count - the same derivation the registry uses. A watcher
download_exit or already_complete line is a claim, and a claim is not a
seal. The watcher log must still exist: a missing log would read as
"nothing is downloading" rather than "the watcher is not running".

On a newly observed seal, the six S027 §22 triggers are emitted as
WorkUnits through frontiers._item. A unit belonging to a dead school is
refused rather than created. Seeing the same seal twice emits once.

    python3 tools/future/modellake_events.py --build
    python3 -m pytest tools/future/test_modellake_events.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt
from tools.future import frontiers as fr
from tools.future import modellake_scheduler_view as mv
from tools.future import specimen_load_cost as lc
from tools.future import specimen_registry as sr


RECORDED_BY = "tools/future/modellake_events.py"
RECEIPT_NAME = "MODELLAKE_EVENTS.json"
SCHEMA = "hawking.future.modellake_events.v1"
VERSION = 1

# S027 §22, in order. Imported so a drift in the view is a drift here.
SEAL_TRIGGERS: tuple[str, ...] = tuple(mv.SEAL_TRIGGERS)

WATCH_LOG_REL = Path("workspace/campaign/odyssey/downloads/modellake-watch.jsonl")

# Species every arrival unit carries. Distinct from catalog work.
SEAL_SPECIES = "modellake-seal"

# One frontier per trigger. Prefetch is MEMORY because residency, not
# compute, is the 143x lever specimen_load_cost measured.
TRIGGER_FRONTIER: dict[str, str] = {
    "fingerprint": "MODEL_REPRESENTATION",
    "role evaluation": "MODEL_CAPABILITY",
    "law and scar lookup": "ODYSSEY_TRANSFER",
    "initial economics": "MEMORY",
    "WorkGraph creation": "ODYSSEY_TRANSFER",
    "possible prefetch or load": "MEMORY",
}

TRIGGER_FAMILY: dict[str, str] = {
    "fingerprint": "modellake_seal_fingerprint",
    "role evaluation": "modellake_seal_role",
    "law and scar lookup": "modellake_seal_laws_scars",
    "initial economics": "modellake_seal_economics",
    "WorkGraph creation": "modellake_seal_workgraph",
    "possible prefetch or load": "modellake_seal_prefetch",
}


class LakeEventRefused(RuntimeError):
    """The watcher log is missing, or a required disk input is absent."""


def _discover_watch_log() -> Path:
    """The live JSONL lives on the primary checkout; a sparse worktree may not copy it."""
    declared = mv.WATCH_LOG
    if declared.is_file():
        return declared
    for root in fr._checkout_roots():
        cand = root / WATCH_LOG_REL
        if cand.is_file():
            return cand
    return declared


WATCH_LOG = _discover_watch_log()


def _require_watch_log() -> Path:
    if not WATCH_LOG.is_file():
        raise LakeEventRefused(
            f"{WATCH_LOG} is not on disk; the watcher's live state is the only "
            "source here and an empty view would read as 'nothing is "
            "downloading' rather than 'the watcher is not running'"
        )
    return WATCH_LOG


def _tail() -> list[dict[str, Any]]:
    """Same grain as the scheduler view: the append-only tail is live state.

    Unlike the view, a log that starts at byte 0 is not a fragment: dropping
    the first line of a tiny test log would refuse a perfectly parseable file.
    """
    path = _require_watch_log()
    size = os.path.getsize(path)
    start = max(0, size - mv.TAIL_BYTES)
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read().decode("utf-8", "ignore").splitlines()
    lines = raw[1:] if start > 0 and raw else raw
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    if not out:
        raise LakeEventRefused("the watcher log tail carries no parseable event")
    return out


def log_completion_claims(events: list[dict[str, Any]] | None = None) -> set[str]:
    """download_exit(0) and already_complete are claims. They are not seals."""
    ev = events if events is not None else _tail()
    out: set[str] = set()
    for e in ev:
        job = e.get("job")
        if not job:
            continue
        if e.get("event") == "already_complete":
            out.add(str(job))
        elif e.get("event") == "download_exit" and e.get("returncode") == 0:
            out.add(str(job))
    return out


def manifest_is_seal(specimen_id: str) -> bool:
    """A seal is a lake manifest with a resolved sha AND a byte count.

    Watcher cache files also sit under manifests/ (expected/files/sizes, no
    bytes). Those are not seals. The registry's own derivation is the same
    test; this function reads the file rather than trusting a lifecycle label.
    """
    if not specimen_id:
        raise LakeEventRefused("specimen_id is missing; a seal cannot be derived")
    man = sr._manifest(specimen_id)
    if not man:
        return False
    return bool(man.get("resolved_sha") and man.get("bytes"))


def sealed_from_disk() -> list[dict[str, Any]]:
    """Every currently sealed specimen, from the lake volume, not from a log."""
    rows = sr.registry()
    out = []
    for r in rows:
        if manifest_is_seal(r["id"]):
            out.append(r)
    return out


def _spec_slug(specimen_id: str) -> str:
    name = specimen_id.split("--")[-1]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "specimen"


def _unit_id(frontier: str, specimen_id: str, trigger: str) -> str:
    trig = re.sub(r"[^a-z0-9]+", "-", trigger.lower()).strip("-")
    return f"FT.{frontier}.seal.{_spec_slug(specimen_id)}.{trig}"


def _economics(specimen_id: str, costs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    table = costs if costs is not None else {r["id"]: r for r in lc.per_specimen()}
    row = table.get(specimen_id)
    if row is None:
        raise LakeEventRefused(
            f"{specimen_id} is sealed on disk but has no load-cost row; "
            "inventing a cost would be a guess wearing a receipt"
        )
    return row


def _trigger_kwargs(
    specimen: dict[str, Any],
    trigger: str,
    *,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if trigger not in SEAL_TRIGGERS:
        raise LakeEventRefused(
            f"unknown seal trigger {trigger!r}; S027 §22 names {list(SEAL_TRIGGERS)}"
        )
    sid = specimen.get("id")
    if not sid:
        raise LakeEventRefused("specimen id is missing; a unit cannot be addressed")
    frontier = TRIGGER_FRONTIER[trigger]
    man_path = str(sr.MANIFESTS / f"{sid}.json")
    arch = (specimen.get("architecture") or {}).get("model_type")
    model_type = arch or "UNKNOWN"
    title, detail = _trigger_copy(specimen, trigger, cost=cost, model_type=model_type)
    return {
        "id": _unit_id(frontier, sid, trigger),
        "frontier": frontier,
        "kind": "NEXT_WORK",
        "title": title,
        "detail": detail,
        "required_lanes": (fr.LANE_CPU, fr.LANE_ANALYSIS),
        "gain": fr.INFO_HIGH,
        "species": SEAL_SPECIES,
        "verifier": "tools.future.modellake_events.consume",
        "evidence": (
            man_path,
            "receipts/future/SPECIMEN_REGISTRY.json",
            "receipts/future/MODELLAKE_SCHEDULER_VIEW.json",
        ),
        "hypothesis_family": TRIGGER_FAMILY[trigger],
        "candidate_id": sid,
        "source_f": "S027-22",
    }


def _trigger_copy(
    specimen: dict[str, Any],
    trigger: str,
    *,
    cost: dict[str, Any] | None,
    model_type: str,
) -> tuple[str, str]:
    sid = specimen["id"]
    if trigger == "fingerprint":
        return (
            f"Fingerprint sealed specimen {sid} from config.json",
            f"config.json on disk names model_type={model_type}; "
            "do not open weights. A fingerprint is architecture and size, "
            "not a tensor experiment.",
        )
    if trigger == "role evaluation":
        return (
            f"Evaluate the role of sealed specimen {sid}",
            f"architecture {model_type} is the role signal. Distance from "
            "the incumbent is computed from that field, not from a human "
            "assignment.",
        )
    if trigger == "law and scar lookup":
        return (
            f"Look up transferable laws and scars for {sid} before any load",
            "S027 §20: the first question is whether to load at all. Query "
            "the law store and the negative index against this architecture "
            "before a byte is read.",
        )
    if trigger == "initial economics":
        if cost is None:
            raise LakeEventRefused(
                f"initial economics for {sid} has no measured cost row"
            )
        return (
            f"Compute initial load economics for {sid}",
            f"measured cold load is {cost['cold_load_seconds']} s "
            f"({cost['source_gb']} GB) at the quiet-cold rate; warm is "
            f"{cost['warm_load_seconds']} s. These are disk-read floors, "
            "not a full LOAD_SPECIMEN.",
        )
    if trigger == "WorkGraph creation":
        return (
            f"Create the arrival WorkGraph for sealed specimen {sid}",
            "the sealed model becomes schedulable material at once. Add a "
            "bounded WorkGraph to the running mission; do not restart Odyssey "
            "and do not ask whether to look at it.",
        )
    if trigger == "possible prefetch or load":
        if cost is None:
            raise LakeEventRefused(
                f"prefetch for {sid} has no measured cost row"
            )
        return (
            f"Consider prefetch or load of sealed specimen {sid}",
            f"cold load is {cost['cold_load_minutes']} minutes. S027 §8: "
            "do not wait until a model is needed to start loading it. "
            "Residency dominates download contention; this unit is the "
            "prefetch decision, not a GPU lease.",
        )
    raise LakeEventRefused(f"unhandled trigger {trigger!r}")


def emit_trigger(
    specimen: dict[str, Any],
    trigger: str,
    *,
    cost: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """One S027 trigger as a WorkUnit. Always constructed by frontiers._item."""
    kwargs = _trigger_kwargs(specimen, trigger, cost=cost)
    kwargs.update(overrides)
    return fr._item(**kwargs)


def emit_for_seal(
    specimen: dict[str, Any],
    ledger: set[str],
    *,
    costs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Emit the six triggers once per specimen id. A second sighting is a no-op."""
    sid = specimen.get("id")
    if not sid:
        raise LakeEventRefused("specimen id is missing")
    if not manifest_is_seal(sid):
        raise LakeEventRefused(
            f"{sid} has no sealing manifest; a claim is not a seal and "
            "COMPLETE_UNSEALED is not schedulable"
        )
    if sid in ledger:
        return []
    cost = _economics(sid, costs)
    units = [emit_trigger(specimen, t, cost=cost) for t in SEAL_TRIGGERS]
    ledger.add(sid)
    return units


def consume(ledger: set[str] | None = None) -> list[dict[str, Any]]:
    """Tail the watcher, detect seals from disk, emit new ones through _item."""
    events = _tail()
    claims = log_completion_claims(events)
    sealed = sealed_from_disk()
    costs = {r["id"]: r for r in lc.per_specimen()}
    seen = ledger if ledger is not None else set()
    out: list[dict[str, Any]] = []
    for row in sealed:
        units = emit_for_seal(row, seen, costs=costs)
        if not units:
            continue
        out.append({
            "specimen_id": row["id"],
            "lifecycle": row["lifecycle"],
            "model_type": (row.get("architecture") or {}).get("model_type"),
            "source_bytes": row.get("source_bytes"),
            "detection": "lake_manifest",
            "claimed_complete_in_log_tail": row["id"] in claims,
            "n_units": len(units),
            "unit_ids": [u["id"] for u in units],
            "units": units,
        })
    return out


def claims_without_seal(events: list[dict[str, Any]] | None = None) -> list[str]:
    """Log said complete; disk has no sealing manifest. Not an error, not a seal."""
    ev = events if events is not None else _tail()
    sealed_ids = {r["id"] for r in sealed_from_disk()}
    return sorted(job for job in log_completion_claims(ev) if job not in sealed_ids)


def build() -> dict[str, Any]:
    events = _tail()
    backlog = sr.seal_backlog()
    sealed = sealed_from_disk()
    ledger: set[str] = set()
    emitted = consume(ledger)
    # consume() re-tails; the ledger is local so this run is a snapshot of
    # first sight, not a durable queue.
    claims_only = claims_without_seal(events)
    return {
        "obligation": "G101",
        "authority": "S027 §2, §21, §22",
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "watch_log": str(WATCH_LOG),
        "watch_log_relative_if_in_repo": (
            str(WATCH_LOG.relative_to(REPO))
            if WATCH_LOG.is_relative_to(REPO) else
            "recovered_from_primary_checkout"
        ),
        "question": "When a model seals, does anything happen without a human noticing?",
        "answer": (
            "yes, now: a lake manifest appearing emits the six S027 §22 "
            "triggers as WorkUnits through frontiers._item, with no "
            "conversational boundary. Before this consumer, MODEL_SEALED "
            "triggered nothing."
        ),
        "is_this_wired": True,
        "no_conversational_boundary": True,
        "human_notified": False,
        "detection": {
            "authority": "a lake manifest with resolved_sha and bytes",
            "not_authority": (
                "a watcher download_exit or already_complete claim, a "
                "lifecycle label, or a human saying the model is ready"
            ),
            "n_sealed_on_disk": len(sealed),
            "n_complete_unsealed": backlog["n_complete_unsealed"],
            "n_sealed_registry": backlog["n_sealed"],
            "sealed_ids": sorted(r["id"] for r in sealed),
            "complete_unsealed_do_not_emit": True,
        },
        "seal_triggers": list(SEAL_TRIGGERS),
        "n_triggers_per_seal": len(SEAL_TRIGGERS),
        "through_frontiers_item": True,
        "dead_school_is_refused_not_created": True,
        "idempotent": True,
        "n_emitted_specimens": len(emitted),
        "n_emitted_units": sum(e["n_units"] for e in emitted),
        "emitted": emitted,
        "log_tail": {
            "n_parseable_events": len(events),
            "n_completion_claims": len(log_completion_claims(events)),
            "n_claims_without_seal": len(claims_only),
            "claims_without_seal": claims_only,
            "a_claim_is_not_a_seal": (
                "download_exit and already_complete mean the watcher thinks "
                "bytes landed. Schedulable material requires the manifest."
            ),
        },
        "what_the_view_still_says": (
            "modellake_scheduler_view.seal_contract is_this_wired is still "
            "False: that module is the reader, this module is the consumer, "
            "and the view is not edited by this lane."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({
        "is_this_wired": doc["is_this_wired"],
        "answer": doc["answer"],
        "n_emitted_specimens": doc["n_emitted_specimens"],
        "n_emitted_units": doc["n_emitted_units"],
        "detection": {
            k: doc["detection"][k]
            for k in ("n_sealed_on_disk", "n_complete_unsealed", "authority")
        },
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
