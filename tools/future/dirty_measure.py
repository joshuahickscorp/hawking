"""SELF_MEASURED_DIRTY — honest evidence from a contaminated machine.

A resident that lives on the machine contaminates every measurement it takes.
Those numbers are good for ranking, direction, cheap A/B and pruning. They
must NEVER promote. This module names that evidence class, binds it to a
stable contamination fingerprint, and structurally closes every promotion
path — including copying the value into a PROTECTED_ABSOLUTE shell.

Builds on tools/future/contamination.py (snapshot, class, paired A/B,
assert_promotable). Does not fork it. Does not take a GPU lease, start a
resident, or source hardware samples.

    python3 tools/future/dirty_measure.py --snapshot
    python3 tools/future/dirty_measure.py --build
    python3 tools/future/dirty_measure.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.future._common import HARDWARE_FIELDS
from tools.future.contamination import (
    CONTAMINATION_CLASSES,
    MIN_PAIRS,
    PromotionRefused,
    SYNTHETIC_PAIRS,
    assert_promotable as contamination_assert_promotable,
    classify_contamination,
    paired_ab_stats,
    snapshot as contamination_snapshot,
)

RECEIPT = "DIRTY_MEASUREMENT.json"
SCHEMA = "hawking.future.dirty_measure.v1"
VERSION = 1
RECORDED_BY = "tools/future/dirty_measure.py"

EVIDENCE_CLASS = "SELF_MEASURED_DIRTY"
MEASUREMENT_CLASS = "STATIC_ONLY"
PROTECTED = "PROTECTED_ABSOLUTE"
DIAGNOSTIC = "DIAGNOSTIC_RELATIVE"

LEGITIMATE_USES = ("rank", "direction", "cheap_paired_ab", "prune_dominated")
QUANTITY_KINDS = ("duration", "rate")
CONTAMINATION_FINGERPRINT_VERSION = 1

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

EVIDENCE_QUEUE = (
    REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
)
FRONTIER_PATH = REPO / "receipts" / "future" / "CLAUDE_GLOBAL_FRONTIER.json"

# write_receipt already refuses numeric HARDWARE_FIELDS. Dirty records also
# refuse latency/TPS aliases that are not in that set, so a ratio cannot be
# smuggled in under a friendlier name.
MAGNITUDE_FIELDS = frozenset(HARDWARE_FIELDS) | {
    "latency",
    "latency_ns",
    "latency_ms",
    "latency_us",
    "absolute_latency",
    "absolute_tps",
    "tokens_per_second",
    "ns_per_token",
    "ms_per_token",
    "wall_ms",
    "duration_ns",
    "duration_ms",
    "time_ms",
    "time_ns",
}

DIRTY_BINDING_KEY = "dirty_binding"
_VALUE_BUNDLE_KEYS = (
    "median_ratio",
    "ratio_q1",
    "ratio_q3",
    "ratio_iqr",
    "bootstrap_ci95",
)

# In-process registry: a field-copied value is still dirty.
_DIRTY_TOKENS: set[str] = set()
_DIRTY_VALUE_DIGESTS: set[str] = set()

# HCLI WorkUnit core, copied as a local interface. Swap for
# tools.future.workunit_species.emit_hcli_workunit / hcli.workunit.WorkUnit
# when resident_api.py lands. Do not import this-wave siblings.
HCLI_CORE_FIELDS = (
    "id",
    "role",
    "description",
    "dependencies",
    "status",
    "assigned_runtime",
    "attempts",
    "resource_class",
    "repairs",
    "failure_context",
    "preferred_backend",
    "assigned_backend",
    "backend_task_id",
    "verifier",
    "effect_class",
    "workspace",
    "verification",
    "repair_root",
    "repair_depth",
    "repair_reason",
    "repair_exhausted",
    "ready_at",
    "running_at",
    "finished_at",
    "classification",
    "provider",
    "content_hash",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. SELF_MEASURED_DIRTY guides ranking, direction, "
    "cheap A/B and prune on a busy machine. It cannot promote, cannot become "
    "PROTECTED_ABSOLUTE, and cannot emit an absolute latency or TPS."
)


class DirtyMagnitudeRefused(ValueError):
    """A dirty record tried to emit an absolute latency, TPS, or hardware field."""


class FrozenDirtyRecord(dict):
    """A SELF_MEASURED_DIRTY record. Mutation raises; copying yields a plain dict."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        super().__init__(data)
        self._frozen = True

    def __setitem__(self, key: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise PromotionRefused(
                f"SELF_MEASURED_DIRTY record is frozen; cannot set {key!r} "
                f"(evidence_class cannot be rebound to {PROTECTED})"
            )
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")

    def update(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_frozen", False):
            raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")
        super().update(*args, **kwargs)

    def clear(self) -> None:
        raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")

    def popitem(self) -> tuple[Any, Any]:
        raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise PromotionRefused("SELF_MEASURED_DIRTY record is frozen")


# ---------------------------------------------------------------------------
# fingerprints / envelope
# ---------------------------------------------------------------------------


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj)).hexdigest()


def contamination_fingerprint(
    snap: Mapping[str, Any],
    klass: Mapping[str, Any] | None = None,
) -> str:
    """Stable for the same machine state. No wall-clock, no jittery load average.

    Membership of competing PIDs, pressure level, contamination class, GPU
    occupancy integer and resident identity are the state. Exact RSS bytes
    are not: they wander inside a class without changing who is on the box.
    """
    if klass is None:
        klass = classify_contamination(snap)
    competing = [
        {"name": p.get("name"), "pid": p.get("pid")}
        for p in (snap.get("competing_workloads") or [])
        if isinstance(p, Mapping)
    ]
    competing.sort(key=lambda r: (r["pid"] is None, r["pid"] or 0, str(r["name"] or "")))
    mem = snap.get("memory_pressure") if isinstance(snap.get("memory_pressure"), Mapping) else {}
    thermal = snap.get("thermal_state")
    if isinstance(thermal, Mapping):
        thermal_status = thermal.get("status") or "UNKNOWN"
    elif thermal in (None,):
        thermal_status = "UNKNOWN"
    else:
        thermal_status = thermal
    probes = snap.get("probes") if isinstance(snap.get("probes"), Mapping) else {}
    gpu = probes.get("gpu_occupancy") if isinstance(probes.get("gpu_occupancy"), Mapping) else {}
    resident = snap.get("resident_local_model") if isinstance(snap.get("resident_local_model"), Mapping) else {}
    identity = snap.get("machine_identity") if isinstance(snap.get("machine_identity"), Mapping) else {}
    body = {
        "v": CONTAMINATION_FINGERPRINT_VERSION,
        "machine_identity_hash": identity.get("hash"),
        "contamination_class": klass.get("contamination_class"),
        "competing": competing,
        "pressure_level": mem.get("pressure_level"),
        "thermal_status": thermal_status,
        "gpu_device_utilization_pct": gpu.get("device_utilization_pct"),
        "resident_pid": resident.get("pid"),
        "resident_name": resident.get("name"),
    }
    return _sha(body)


def identify_resident_loaded(
    snap: Mapping[str, Any],
    *,
    declared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Which model, which pid. UNKNOWN when we cannot tell. Never a guess.

    This sidecar does not start a resident. `declared` is the local interface
    for a future resident_identity.py: pass {model, pid, process_name}.
    """
    neighbour = snap.get("resident_local_model")
    pid: Any = None
    process_name: Any = None
    model = "UNKNOWN"
    source = "UNKNOWN"
    status = "UNKNOWN"
    notes: list[str] = [
        "this sidecar did not start a resident and did not take a GPU lease"
    ]
    if isinstance(neighbour, Mapping):
        pid = neighbour.get("pid")
        process_name = neighbour.get("name")
        source = "largest_rss_neighbour"
        status = "PARTIAL"
        notes.append("process name is not a model identity; model stays UNKNOWN unless declared")
    if declared:
        if declared.get("model"):
            model = str(declared["model"])
        if declared.get("pid") is not None:
            pid = declared.get("pid")
        if declared.get("process_name"):
            process_name = declared.get("process_name")
        source = "declared"
        status = "OK" if model != "UNKNOWN" and pid is not None else "PARTIAL"
        notes.append("declared by caller; not invented here")
        competing_pids = {
            p.get("pid") for p in (snap.get("competing_workloads") or []) if isinstance(p, Mapping)
        }
        gpu_rows = ((snap.get("gpu_processes") or {}) if isinstance(snap.get("gpu_processes"), Mapping) else {}).get(
            "processes"
        ) or []
        gpu_pids = {p.get("pid") for p in gpu_rows if isinstance(p, Mapping)}
        if pid is not None and (competing_pids or gpu_pids) and pid not in competing_pids and pid not in gpu_pids:
            notes.append("declared pid not listed in competing_workloads; recorded, not dropped")
    return {
        "model": model,
        "pid": pid,
        "process_name": process_name,
        "source": source,
        "status": status,
        "note": "; ".join(notes),
        "did_not_start_resident": True,
    }


def _assert_no_magnitude(node: Any, path: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == DIRTY_BINDING_KEY:
                continue
            here = f"{path}.{key}" if path else str(key)
            if key in MAGNITUDE_FIELDS and isinstance(value, (int, float)):
                raise DirtyMagnitudeRefused(
                    f"{here} = {value!r}: SELF_MEASURED_DIRTY refuses absolute "
                    "latency/TPS/hardware fields; direction is a ratio, not a magnitude"
                )
            _assert_no_magnitude(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_magnitude(value, f"{path}[{i}]")


def _value_bundle(record: Mapping[str, Any]) -> dict[str, Any] | None:
    sources: list[Mapping[str, Any]] = [record]
    for key in ("ab_stats", "stats", "direction"):
        node = record.get(key)
        if isinstance(node, Mapping):
            sources.append(node)
    bundle: dict[str, Any] = {}
    for src in sources:
        for key in _VALUE_BUNDLE_KEYS:
            if key in src and key not in bundle:
                bundle[key] = src[key]
    return bundle or None


def _register_dirty(record: Mapping[str, Any]) -> str:
    bundle = _value_bundle(record) or {}
    token_body = {
        "evidence_class": EVIDENCE_CLASS,
        "fingerprint": record.get("contamination_fingerprint"),
        "bundle": bundle,
        "use": record.get("use"),
    }
    token = _sha(token_body)
    _DIRTY_TOKENS.add(token)
    if bundle:
        _DIRTY_VALUE_DIGESTS.add(_sha({"bundle": bundle}))
    return token


def _has_dirty_binding(node: Any) -> bool:
    if isinstance(node, Mapping):
        binding = node.get(DIRTY_BINDING_KEY)
        if isinstance(binding, Mapping) and binding.get("evidence_class") == EVIDENCE_CLASS:
            return True
        token = node.get("dirty_value_token")
        if isinstance(token, str) and token in _DIRTY_TOKENS:
            return True
        if node.get("evidence_class") == EVIDENCE_CLASS:
            return True
        return any(_has_dirty_binding(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_dirty_binding(v) for v in node)
    return False


def _values_match_dirty_emission(record: Mapping[str, Any]) -> bool:
    bundle = _value_bundle(record)
    if not bundle:
        return False
    return _sha({"bundle": bundle}) in _DIRTY_VALUE_DIGESTS


def is_dirty_sourced(record: Any) -> bool:
    """True if this mapping is a dirty record or a field-copy of one."""
    if not isinstance(record, Mapping):
        return False
    if record.get("evidence_class") == EVIDENCE_CLASS:
        return True
    if record.get("dirty_value_token") in _DIRTY_TOKENS:
        return True
    if _has_dirty_binding(record):
        return True
    if _values_match_dirty_emission(record):
        return True
    return False


def _binding(token: str, fingerprint: str, use: str) -> dict[str, Any]:
    return {
        "evidence_class": EVIDENCE_CLASS,
        "token": token,
        "contamination_fingerprint": fingerprint,
        "use": use,
        "rule": (
            f"{EVIDENCE_CLASS} values cannot be rebound as {PROTECTED}; "
            "copying this object does not change its evidence class"
        ),
    }


def _seal_record(body: dict[str, Any], *, use: str) -> FrozenDirtyRecord:
    _assert_no_magnitude(body)
    body = dict(body)
    body["evidence_class"] = EVIDENCE_CLASS
    body["measurement_class"] = MEASUREMENT_CLASS
    body["promotable"] = False
    body["may_promote"] = False
    body["offered_as"] = None
    body["use"] = use
    body["gpu_authority"] = False
    body["bench_state"] = "UNKNOWN"
    token = _register_dirty(body)
    body["dirty_value_token"] = token
    stats = body.get("ab_stats")
    if isinstance(stats, dict) and DIRTY_BINDING_KEY not in stats:
        stats[DIRTY_BINDING_KEY] = _binding(token, str(body.get("contamination_fingerprint") or ""), use)
    body[DIRTY_BINDING_KEY] = _binding(token, str(body.get("contamination_fingerprint") or ""), use)
    _assert_no_magnitude(body)
    return FrozenDirtyRecord(body)


def dirty_snapshot(
    *,
    probes: Mapping[str, Any] | None = None,
    declared_resident: Mapping[str, Any] | None = None,
    benchmark_ordinal: int | None = None,
) -> FrozenDirtyRecord:
    """SELF_MEASURED_DIRTY envelope of machine state. No measurement values."""
    snap = contamination_snapshot(benchmark_ordinal=benchmark_ordinal, probes=probes)
    klass = classify_contamination(snap)
    fingerprint = contamination_fingerprint(snap, klass)
    resident = identify_resident_loaded(snap, declared=declared_resident)
    thermal = snap.get("thermal_state")
    if thermal is None:
        thermal = "UNKNOWN"
    body = {
        "purpose": (
            "Envelope for measurements the resident takes of itself on a busy "
            "machine. The envelope is not a benchmark."
        ),
        "contamination_class": klass.get("contamination_class"),
        "contamination_reason": klass.get("contamination_reason"),
        "contamination_evidence": klass.get("contamination_evidence"),
        "contamination_fingerprint": fingerprint,
        "resident_loaded": resident,
        "gpu_processes": snap.get("gpu_processes"),
        "memory_pressure": snap.get("memory_pressure"),
        "thermal_state": thermal,
        "competing_workloads": snap.get("competing_workloads"),
        "machine_identity": snap.get("machine_identity"),
        "benchmark_ordinal": snap.get("benchmark_ordinal"),
        "snapshot": snap,
        "classification_rule": klass.get("rule"),
        "required_probes": klass.get("required_probes"),
        "legitimate_uses": list(LEGITIMATE_USES),
        "refuses": [PROTECTED, "absolute_latency", "absolute_tps", *sorted(HARDWARE_FIELDS)],
    }
    return _seal_record(body, use="envelope")


# ---------------------------------------------------------------------------
# direction is not magnitude
# ---------------------------------------------------------------------------


def _quantity_kind(kind: str) -> str:
    text = str(kind or "").strip().lower()
    if text not in QUANTITY_KINDS:
        raise ValueError(
            f"quantity_kind must be one of {QUANTITY_KINDS} (duration: lower is faster; "
            f"rate: higher is faster); got {kind!r}"
        )
    return text


def _direction_from_stats(
    stats: Mapping[str, Any],
    *,
    quantity_kind: str,
) -> dict[str, Any]:
    kind = _quantity_kind(quantity_kind)
    median = stats.get("median_ratio")
    ci = stats.get("bootstrap_ci95")
    n_kept = int(stats.get("n_kept") or 0)
    min_pairs = int(stats.get("min_pairs") or MIN_PAIRS)
    sample_ok = bool(stats.get("sufficient_for_decision"))
    ci_lo, ci_hi = (None, None)
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        ci_lo, ci_hi = float(ci[0]), float(ci[1])
    ci_excludes_one = (
        isinstance(ci_lo, float) and isinstance(ci_hi, float) and (ci_hi < 1.0 or ci_lo > 1.0)
    )
    if not sample_ok or not isinstance(median, (int, float)):
        arm = "UNDECIDED"
        faster = "UNDECIDED"
        sufficient = False
        reason = str(stats.get("reason") or "insufficient paired samples")
    else:
        if median > 1:
            larger = "B"
        elif median < 1:
            larger = "A"
        else:
            larger = "TIE"
        if larger == "TIE":
            faster = "TIE"
        elif kind == "duration":
            faster = "A" if larger == "B" else "B"  # larger duration is slower
        else:
            faster = larger  # larger rate is faster
        arm = larger
        iqr = stats.get("ratio_iqr")
        exact_tie = larger == "TIE" and (
            (ci_lo == 1.0 and ci_hi == 1.0) or (isinstance(iqr, (int, float)) and iqr == 0)
        )
        if exact_tie:
            sufficient = True
            reason = (
                f"n_kept={n_kept} >= min_pairs={min_pairs}; exact TIE "
                "(median_ratio=1 with no spread)"
            )
        elif not ci_excludes_one:
            sufficient = False
            reason = (
                f"n_kept={n_kept} >= min_pairs={min_pairs} but bootstrap CI "
                f"[{ci_lo}, {ci_hi}] includes 1.0; direction is leaning, not a decision"
            )
            if faster != "TIE":
                faster = "UNDECIDED"
                arm = "UNDECIDED"
        else:
            sufficient = True
            reason = (
                f"n_kept={n_kept} >= min_pairs={min_pairs}; CI excludes 1.0; "
                f"faster_arm={faster} for quantity_kind={kind}"
            )
    return {
        "quantity_kind": kind,
        "ratio_meaning": (
            "dimensionless pairwise B/A; this sidecar never emits the raw arm "
            "magnitudes. duration: lower is faster. rate: higher is faster."
        ),
        "arm_with_larger_value": arm,
        "faster_arm": faster,
        "median_ratio": median if isinstance(median, (int, float)) else None,
        "ratio_q1": stats.get("ratio_q1"),
        "ratio_q3": stats.get("ratio_q3"),
        "ratio_iqr": stats.get("ratio_iqr"),
        "bootstrap_ci95": list(ci) if isinstance(ci, (list, tuple)) else None,
        "ci_excludes_one": ci_excludes_one,
        "n_kept": n_kept,
        "n_pairs": stats.get("n_pairs"),
        "n_dropped": stats.get("n_dropped"),
        "min_pairs": min_pairs,
        "sufficient_for_decision": sufficient,
        "reason": reason,
        "sample_count_sufficient": sample_ok,
        "magnitude_refused": sorted(MAGNITUDE_FIELDS),
    }


def _require_envelope(envelope: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if envelope is None:
        return dirty_snapshot()
    if envelope.get("evidence_class") not in {EVIDENCE_CLASS, None}:
        raise PromotionRefused(
            f"envelope evidence_class {envelope.get('evidence_class')!r} is not {EVIDENCE_CLASS}"
        )
    if not envelope.get("contamination_fingerprint"):
        raise ValueError("envelope is missing contamination_fingerprint")
    return envelope


def cheap_paired_ab(
    samples: Iterable[Any],
    *,
    quantity_kind: str,
    envelope: Mapping[str, Any] | None = None,
    min_pairs: int | None = None,
    seed: int = 0,
) -> FrozenDirtyRecord:
    """Cheap paired A/B on a busy machine. Direction and robust ratio, no magnitude."""
    env = _require_envelope(envelope)
    stats = paired_ab_stats(samples, min_pairs=MIN_PAIRS if min_pairs is None else min_pairs, bootstrap_seed=seed)
    # paired_ab_stats is the landed contract; drop anything that looks like a mean.
    if "mean" in stats or "average" in stats:
        raise DirtyMagnitudeRefused("paired_ab_stats reported a mean; dirty A/B refuses it")
    direction = _direction_from_stats(stats, quantity_kind=quantity_kind)
    # Sample-count sufficiency is the A/B's own; direction may be stricter.
    sufficient = bool(stats.get("sufficient_for_decision"))
    reason = str(stats.get("reason") or direction["reason"])
    if sufficient and not direction["sufficient_for_decision"]:
        reason = direction["reason"]
    body = {
        "contamination_fingerprint": env.get("contamination_fingerprint"),
        "contamination_class": env.get("contamination_class"),
        "resident_loaded": env.get("resident_loaded"),
        "quantity_kind": direction["quantity_kind"],
        "faster_arm": direction["faster_arm"],
        "arm_with_larger_value": direction["arm_with_larger_value"],
        "median_ratio": direction["median_ratio"],
        "ratio_q1": direction["ratio_q1"],
        "ratio_q3": direction["ratio_q3"],
        "ratio_iqr": direction["ratio_iqr"],
        "bootstrap_ci95": direction["bootstrap_ci95"],
        "spread": {
            "q1": direction["ratio_q1"],
            "q3": direction["ratio_q3"],
            "iqr": direction["ratio_iqr"],
            "bootstrap_ci95": direction["bootstrap_ci95"],
        },
        "ab_stats": dict(stats),
        "direction": direction,
        "sufficient_for_decision": sufficient,
        "sufficient_for_direction": direction["sufficient_for_decision"],
        "reason": reason,
        "never_reports": ["mean", "average", "absolute_latency", "absolute_tps", *sorted(HARDWARE_FIELDS)],
    }
    return _seal_record(body, use="cheap_paired_ab")


def effect_direction(
    samples: Iterable[Any],
    *,
    quantity_kind: str,
    envelope: Mapping[str, Any] | None = None,
    min_pairs: int | None = None,
    seed: int = 0,
) -> FrozenDirtyRecord:
    """Which arm is faster, with robust ratio and spread. Not a magnitude."""
    rec = cheap_paired_ab(
        samples,
        quantity_kind=quantity_kind,
        envelope=envelope,
        min_pairs=min_pairs,
        seed=seed,
    )
    body = dict(rec)
    body["use"] = "direction"
    # re-seal to bind this use; token differs by use so a direction record and
    # an A/B record of the same pairs stay distinct while both remain dirty.
    body.pop("dirty_value_token", None)
    if isinstance(body.get("ab_stats"), dict):
        body["ab_stats"] = dict(body["ab_stats"])
        body["ab_stats"].pop(DIRTY_BINDING_KEY, None)
    body.pop(DIRTY_BINDING_KEY, None)
    return _seal_record(body, use="direction")


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("id", "candidate_id", "name"):
        if row.get(key):
            return str(row[key])
    return f"candidate[{index}]"


def _dirty_for_candidate(
    row: Mapping[str, Any],
    *,
    quantity_kind: str,
    envelope: Mapping[str, Any],
    min_pairs: int | None,
    seed: int,
) -> FrozenDirtyRecord:
    existing = row.get("dirty_record") or row.get("dirty")
    if isinstance(existing, Mapping):
        _assert_no_magnitude(existing)
        if existing.get("evidence_class") != EVIDENCE_CLASS:
            raise PromotionRefused(
                f"candidate {row!r} dirty_record is not {EVIDENCE_CLASS}; "
                "refusing to rank a record of another evidence class"
            )
        return existing if isinstance(existing, FrozenDirtyRecord) else _seal_record(dict(existing), use="rank")
    pairs = row.get("pairs")
    if pairs is None:
        raise ValueError(
            f"candidate {_candidate_id(row, 0)} needs dirty_record or pairs; "
            "dimensionless dirty A/B only"
        )
    return cheap_paired_ab(
        pairs,
        quantity_kind=quantity_kind,
        envelope=envelope,
        min_pairs=min_pairs,
        seed=seed,
    )


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    quantity_kind: str,
    envelope: Mapping[str, Any] | None = None,
    min_pairs: int | None = None,
    seed: int = 0,
) -> FrozenDirtyRecord:
    """Rank by dirty B/A vs a shared control. Too-noisy results are not used."""
    env = _require_envelope(envelope)
    kind = _quantity_kind(quantity_kind)
    _assert_no_magnitude({"candidates": list(candidates)})
    rows = list(candidates)
    prepared: list[dict[str, Any]] = []
    unranked: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for i, row in enumerate(rows):
        cid = _candidate_id(row, i)
        rec = _dirty_for_candidate(
            row, quantity_kind=kind, envelope=env, min_pairs=min_pairs, seed=seed
        )
        fp = str(rec.get("contamination_fingerprint") or "")
        if fp:
            fingerprints.add(fp)
        env_fp = str(env.get("contamination_fingerprint") or "")
        if env_fp and fp and fp != env_fp:
            unranked.append(
                {
                    "id": cid,
                    "reason": (
                        "contamination fingerprint does not match the envelope; "
                        "ranking would mix machine states"
                    ),
                }
            )
            continue
        if not rec.get("sufficient_for_decision"):
            unranked.append(
                {
                    "id": cid,
                    "reason": rec.get("reason") or "dirty result too noisy to rank",
                    "sufficient_for_decision": False,
                }
            )
            continue
        median = rec.get("median_ratio")
        if not isinstance(median, (int, float)):
            unranked.append({"id": cid, "reason": "median_ratio missing", "sufficient_for_decision": False})
            continue
        prepared.append(
            {
                "id": cid,
                "median_ratio": median,
                "ratio_iqr": rec.get("ratio_iqr"),
                "bootstrap_ci95": rec.get("bootstrap_ci95"),
                "faster_arm_vs_control": rec.get("faster_arm"),
                "dirty_value_token": rec.get("dirty_value_token"),
            }
        )
    mixed = len(fingerprints) > 1
    # duration: lower B/A is faster vs control. rate: higher B/A is faster.
    prepared.sort(key=lambda r: (r["median_ratio"], r["id"]), reverse=(kind == "rate"))
    n_rankable = len(prepared)
    min_rankable = 2
    if mixed:
        sufficient = False
        reason = (
            f"mixed contamination fingerprints ({len(fingerprints)}); "
            "refusing to rank across machine states"
        )
        ranked: list[dict[str, Any]] = []
    elif n_rankable < min_rankable:
        sufficient = False
        reason = (
            f"n_rankable={n_rankable} < {min_rankable}; a dirty result that is "
            "too noisy (or alone) is not used to rank"
        )
        ranked = []
    else:
        sufficient = True
        reason = (
            f"n_rankable={n_rankable} of n_candidates={len(rows)}; "
            f"ordered by median B/A ({kind})"
        )
        ranked = [{**row, "rank": i} for i, row in enumerate(prepared, start=1)]
    unranked.sort(key=lambda r: str(r.get("id") or ""))
    body = {
        "contamination_fingerprint": env.get("contamination_fingerprint"),
        "contamination_class": env.get("contamination_class"),
        "resident_loaded": env.get("resident_loaded"),
        "quantity_kind": kind,
        "ranked": ranked,
        "unranked": unranked,
        "n_candidates": len(rows),
        "n_rankable": n_rankable,
        "n_unranked": len(unranked),
        "sufficient_for_decision": sufficient,
        "reason": reason,
        "sort": "median_ratio ascending" if kind == "duration" else "median_ratio descending",
    }
    return _seal_record(body, use="rank")


def prune_dominated(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    quantity_kind: str,
    envelope: Mapping[str, Any] | None = None,
    min_pairs: int | None = None,
    seed: int = 0,
) -> FrozenDirtyRecord:
    """Prune a clearly-dominated candidate. CI must exclude 1.0. Not a promotion."""
    env = _require_envelope(envelope)
    kind = _quantity_kind(quantity_kind)
    _assert_no_magnitude({"comparisons": list(comparisons)})
    pruned: dict[str, dict[str, Any]] = {}
    kept_ids: set[str] = set()
    undecided: list[dict[str, Any]] = []
    for i, row in enumerate(comparisons):
        a_id = str(row.get("a") or row.get("id_a") or "")
        b_id = str(row.get("b") or row.get("id_b") or "")
        if not a_id or not b_id:
            raise ValueError(f"comparison[{i}] needs a and b candidate ids")
        kept_ids.add(a_id)
        kept_ids.add(b_id)
        rec = _dirty_for_candidate(
            row, quantity_kind=kind, envelope=env, min_pairs=min_pairs, seed=seed
        )
        env_fp = str(env.get("contamination_fingerprint") or "")
        fp = str(rec.get("contamination_fingerprint") or "")
        if env_fp and fp and fp != env_fp:
            undecided.append(
                {
                    "a": a_id,
                    "b": b_id,
                    "reason": "contamination fingerprint mismatch; not pruning across machine states",
                }
            )
            continue
        if not rec.get("sufficient_for_direction") and not (
            isinstance(rec.get("direction"), Mapping) and rec["direction"].get("sufficient_for_decision")
        ):
            undecided.append(
                {
                    "a": a_id,
                    "b": b_id,
                    "reason": rec.get("reason") or "not clearly dominated (noisy or CI includes 1.0)",
                    "faster_arm": rec.get("faster_arm"),
                }
            )
            continue
        faster = rec.get("faster_arm")
        # faster_arm names A/B of the pair, where the pair is (a, b) as (A, B).
        if faster == "A":
            dominated, winner = b_id, a_id
        elif faster == "B":
            dominated, winner = a_id, b_id
        elif faster == "TIE" and (
            rec.get("sufficient_for_direction")
            or (
                isinstance(rec.get("direction"), Mapping)
                and rec["direction"].get("sufficient_for_decision")
            )
        ):
            # Decided equality is not domination.
            continue
        else:
            undecided.append(
                {
                    "a": a_id,
                    "b": b_id,
                    "reason": "no faster arm; not clearly dominated",
                    "faster_arm": faster,
                }
            )
            continue
        pruned[dominated] = {
            "id": dominated,
            "dominated_by": winner,
            "status": "PRUNED_SELF_MEASURED_DIRTY",
            "not_a_protected_reject": True,
            "median_ratio": rec.get("median_ratio"),
            "bootstrap_ci95": rec.get("bootstrap_ci95"),
            "dirty_value_token": rec.get("dirty_value_token"),
            "reason": rec.get("reason"),
        }
    survivors = sorted(cid for cid in kept_ids if cid not in pruned)
    pruned_rows = [pruned[k] for k in sorted(pruned)]
    n_comparisons = len(comparisons)
    sufficient = bool(pruned_rows) or (
        n_comparisons > 0 and not undecided and not pruned_rows
    )
    # Honest: a prune decision is sufficient when every comparison was
    # decidable (clearly dominated or clearly not). Residual UNDECIDED
    # comparisons mean we did not finish pruning.
    if undecided:
        sufficient = False
        reason = (
            f"n_undecided={len(undecided)} of n_comparisons={n_comparisons}; "
            "a dirty result that is not clearly dominated is not pruned"
        )
    elif not n_comparisons:
        sufficient = False
        reason = "no comparisons; nothing to prune"
    else:
        sufficient = True
        reason = (
            f"n_pruned={len(pruned_rows)} n_survivors={len(survivors)} "
            f"of n_comparisons={n_comparisons}; clearly-dominated only"
        )
    body = {
        "contamination_fingerprint": env.get("contamination_fingerprint"),
        "contamination_class": env.get("contamination_class"),
        "resident_loaded": env.get("resident_loaded"),
        "quantity_kind": kind,
        "pruned": pruned_rows,
        "survivors": survivors,
        "undecided": undecided,
        "n_comparisons": n_comparisons,
        "n_pruned": len(pruned_rows),
        "n_survivors": len(survivors),
        "sufficient_for_decision": sufficient,
        "reason": reason,
        "status_vocabulary": {
            "PRUNED_SELF_MEASURED_DIRTY": "diagnostic prune on a busy machine; not PROTECTED_REJECT"
        },
    }
    return _seal_record(body, use="prune_dominated")


def use(kind: str, *args: Any, **kwargs: Any) -> FrozenDirtyRecord:
    """Dispatcher over the only legitimate uses of a dirty record."""
    table: dict[str, Callable[..., FrozenDirtyRecord]] = {
        "rank": rank_candidates,
        "direction": effect_direction,
        "cheap_paired_ab": cheap_paired_ab,
        "prune_dominated": prune_dominated,
    }
    if kind not in table:
        raise ValueError(
            f"unknown dirty use {kind!r}; legitimate uses are {list(LEGITIMATE_USES)}"
        )
    return table[kind](*args, **kwargs)


# ---------------------------------------------------------------------------
# structural refusal — a guard nobody has watched fail is not a guard
# ---------------------------------------------------------------------------


def _refusal_message(record: Mapping[str, Any] | None, extra: str) -> str:
    klass = None if record is None else record.get("evidence_class")
    mclass = None if record is None else record.get("measurement_class")
    return (
        f"{EVIDENCE_CLASS} cannot become {PROTECTED}. {extra} "
        f"(evidence_class={klass!r}, measurement_class={mclass!r})"
    )


def offer_for_promotion(record: Mapping[str, Any]) -> None:
    """The only promotion entry this module exposes. Always raises."""
    extra = "offered for promotion"
    if is_dirty_sourced(record):
        extra = "SELF_MEASURED_DIRTY record (or a field-copy of its values) offered for promotion"
    raise PromotionRefused(_refusal_message(record, extra))


def as_protected_absolute(record: Mapping[str, Any]) -> dict[str, Any]:
    """There is no conversion. Field copy is not a path."""
    extra = "as_protected_absolute is closed"
    if is_dirty_sourced(record):
        extra = "no conversion from SELF_MEASURED_DIRTY; field copy is not a path"
    raise PromotionRefused(_refusal_message(record, extra))


def copy_value_as(record: Mapping[str, Any], target_class: str) -> dict[str, Any]:
    """Refuse to copy a dirty value into another measurement class."""
    target = str(target_class or "")
    if target == PROTECTED or target == "QUALIFIED_PROTECTED":
        raise PromotionRefused(
            _refusal_message(
                record,
                f"copying its value as {target} is refused",
            )
        )
    if is_dirty_sourced(record) or record.get("evidence_class") == EVIDENCE_CLASS:
        raise PromotionRefused(
            _refusal_message(record, f"copying its value as {target} is refused")
        )
    raise PromotionRefused(
        f"this sidecar cannot rebind a record as {target!r}; copy_value_as is closed"
    )


def mint_protected_absolute(*_a: Any, **_k: Any) -> dict[str, Any]:
    """Trap. This sidecar has no GPU lease and cannot mint PROTECTED_ABSOLUTE."""
    raise PromotionRefused(
        f"this sidecar cannot mint {PROTECTED}; no GPU lease, no quiescence window, "
        "no field-copy constructor"
    )


def ingest_as_protected(record: Mapping[str, Any]) -> dict[str, Any]:
    """Would accept a mapping as a protected measurement. Closed, with reasons."""
    reasons: list[str] = []
    if record.get("evidence_class") == EVIDENCE_CLASS:
        reasons.append(f"evidence_class is {EVIDENCE_CLASS}")
    if _has_dirty_binding(record):
        reasons.append("dirty_binding travelled with the values")
    if record.get("dirty_value_token") in _DIRTY_TOKENS:
        reasons.append("dirty_value_token is registered as SELF_MEASURED_DIRTY")
    if _values_match_dirty_emission(record):
        reasons.append("dimensionless values match a SELF_MEASURED_DIRTY emission")
    if record.get("measurement_class") == PROTECTED:
        reasons.append(f"this sidecar has no GPU lease and cannot mint {PROTECTED}")
    if record.get("promotable") is True or record.get("may_promote") is True:
        reasons.append("a promotable/may_promote flag does not raise evidence class")
    if record.get("caveat") or record.get("with_caveat"):
        reasons.append("a caveat does not raise evidence class")
    reasons.append("ingest_as_protected is structurally closed")
    raise PromotionRefused(_refusal_message(record, "; ".join(reasons)))


def assert_promotable(record: Mapping[str, Any]) -> None:
    """Refuse dirty-sourced values even after field copy, then the landed gate."""
    if is_dirty_sourced(record):
        raise PromotionRefused(
            _refusal_message(
                record,
                "dirty-sourced values cannot promote, including after field copy",
            )
        )
    if record.get("measurement_class") == PROTECTED and record.get("gpu_authority") is not True:
        raise PromotionRefused(
            f"this sidecar cannot certify {PROTECTED} (gpu_authority is not true)"
        )
    contamination_assert_promotable(record)


def _must_raise(fn: Callable[..., Any], *args: Any, fragment: str, **kwargs: Any) -> dict[str, Any]:
    try:
        fn(*args, **kwargs)
    except PromotionRefused as exc:
        msg = str(exc)
        if fragment not in msg:
            raise AssertionError(f"gate raised but message {msg!r} lacked {fragment!r}") from exc
        return {"fired": True, "message": msg, "type": type(exc).__name__}
    except DirtyMagnitudeRefused as exc:
        msg = str(exc)
        if fragment not in msg:
            raise AssertionError(f"magnitude gate raised but message {msg!r} lacked {fragment!r}") from exc
        return {"fired": True, "message": msg, "type": type(exc).__name__}
    raise AssertionError(f"gate did not refuse {fragment!r}")


# ---------------------------------------------------------------------------
# WorkUnit — local HCLI-shaped dict; swap at the named integration point
# ---------------------------------------------------------------------------


def _workunit_hash(row: Mapping[str, Any]) -> str:
    body = {k: row.get(k) for k in HCLI_CORE_FIELDS if k != "content_hash"}
    body["command"] = row.get("command")
    body["output_receipt_path"] = row.get("output_receipt_path")
    return _sha(body)


def emit_workunit(*, sleeping_protected: bool = False) -> dict[str, Any]:
    """WorkUnit HCLI can schedule. Protected follow-up is SLEEPING, never synthetic."""
    if sleeping_protected:
        row: dict[str, Any] = {
            "id": "future.dirty-measure.protected-followup",
            "role": "science",
            "description": (
                "Wake when a QUIESCENT protected GPU lease exists and re-measure "
                "under PROTECTED_ABSOLUTE. SELF_MEASURED_DIRTY values are not inputs."
            ),
            "dependencies": ["future.dirty-measure.self-snapshot"],
            "status": "SLEEPING",
            "assigned_runtime": None,
            "attempts": 0,
            "resource_class": "GPU_EXCLUSIVE",
            "repairs": None,
            "failure_context": None,
            "preferred_backend": "metal",
            "assigned_backend": None,
            "backend_task_id": None,
            "verifier": "accelerator.physical.protected_complete_token",
            "effect_class": "READ_ONLY",
            "workspace": "repo-root",
            "verification": None,
            "repair_root": None,
            "repair_depth": 0,
            "repair_reason": None,
            "repair_exhausted": False,
            "ready_at": None,
            "running_at": None,
            "finished_at": None,
            "classification": "BLOCKED",
            "provider": "future.dirty_measure",
            "claim_boundary": CLAIM_BOUNDARY,
            "command": None,
            "output_receipt_path": None,
            "requires_quiescence": True,
            "blocked_reason": (
                "SELF_MEASURED_DIRTY cannot become PROTECTED_ABSOLUTE. "
                "queue_policy.protected_start_requires_machine_quiescence. "
                "Sleeps until hardware qualifies. A synthetic protected result is refused."
            ),
            "may_promote": False,
            "evidence_class": EVIDENCE_CLASS,
            "measurement_class": MEASUREMENT_CLASS,
            "species": "dirty_self_measurement",
        }
        row["content_hash"] = _workunit_hash(row)
        return row
    row = {
        "id": "future.dirty-measure.self-snapshot",
        "role": "science",
        "description": (
            "Take a SELF_MEASURED_DIRTY envelope of the current machine and seal "
            "receipts/future/DIRTY_MEASUREMENT.json. Rank/direction/A/B/prune use "
            "the envelope. Promotion is refused."
        ),
        "dependencies": [],
        "status": "pending",
        "assigned_runtime": None,
        "attempts": 0,
        "resource_class": "STATIC_ANALYSIS",
        "repairs": None,
        "failure_context": None,
        "preferred_backend": None,
        "assigned_backend": None,
        "backend_task_id": None,
        "verifier": "future.dirty_measure.not_promotable",
        "effect_class": "READ_ONLY",
        "workspace": "repo-root",
        "verification": None,
        "repair_root": None,
        "repair_depth": 0,
        "repair_reason": None,
        "repair_exhausted": False,
        "ready_at": None,
        "running_at": None,
        "finished_at": None,
        "classification": None,
        "provider": "future.dirty_measure",
        "claim_boundary": CLAIM_BOUNDARY,
        "command": ["python3", "tools/future/dirty_measure.py", "--snapshot"],
        "output_receipt_path": "receipts/future/DIRTY_MEASUREMENT.json",
        "requires_quiescence": False,
        "blocked_reason": None,
        "may_promote": False,
        "evidence_class": EVIDENCE_CLASS,
        "measurement_class": MEASUREMENT_CLASS,
        "species": "dirty_self_measurement",
        "frontier_fed": "DIRTY_RANKING (never the qualification promotion rung)",
    }
    row["content_hash"] = _workunit_hash(row)
    return row


def resident_callable() -> dict[str, Any]:
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/dirty_measure.py --snapshot",
        "module": "tools.future.dirty_measure",
        "functions": [
            "dirty_snapshot",
            "rank_candidates",
            "effect_direction",
            "cheap_paired_ab",
            "prune_dominated",
            "use",
        ],
        "workunit": emit_workunit(),
        "sleeping_protected_followup": emit_workunit(sleeping_protected=True),
        "receipt": "receipts/future/DIRTY_MEASUREMENT.json",
        "frontier_fed": {
            "id": "DIRTY_RANKING",
            "feeds": (
                "candidate ranking and prune on a busy machine; a diagnostic "
                "input to resident-optimizer ranking"
            ),
            "does_not_feed": "qualification funnel promotion rung",
            "related_frontier_entry": "F011",
            "queue_policy": "diagnostic_results_do_not_promote",
        },
        "fail_closed": {
            "promotion": (
                "offer_for_promotion / as_protected_absolute / copy_value_as / "
                "ingest_as_protected / mint_protected_absolute raise PromotionRefused"
            ),
            "field_copy": (
                "dirty_binding travels with ab_stats; value digests are registered; "
                "is_dirty_sourced remains true after copying numbers into a "
                f"{PROTECTED} shell"
            ),
            "magnitude": (
                "absolute latency/TPS raise DirtyMagnitudeRefused; "
                "write_receipt raises HardwareClaimError on HARDWARE_FIELDS"
            ),
            "noisy_rank": "insufficient dirty results are not used to rank",
            "protected_followup": (
                "SLEEPING WorkUnit until a QUIESCENT protected lease exists; "
                "never a synthetic PROTECTED_ABSOLUTE result"
            ),
        },
        "integration_point": (
            "emit_workunit is a local HCLI-shaped dict. Swap for "
            "tools.future.workunit_species.emit_hcli_workunit when resident_api.py "
            "lands. identify_resident_loaded(declared=...) is the swap for "
            "resident_identity.py."
        ),
    }


def queue_policy() -> dict[str, Any]:
    """Read the pinned qualification-queue policy. Missing is recorded, not assumed."""
    rel = "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
    path = REPO / rel
    out: dict[str, Any] = {
        "path": rel,
        "present": path.is_file(),
        "protected_start_requires_machine_quiescence": None,
        "diagnostic_results_do_not_promote": None,
    }
    if not path.is_file():
        out["reason"] = "pinned evidence copy not on disk in this checkout; not treated as absence of the policy"
        return out
    doc = load_json(path)
    policy = doc.get("queue_policy") if isinstance(doc.get("queue_policy"), Mapping) else {}
    out["protected_start_requires_machine_quiescence"] = policy.get(
        "protected_start_requires_machine_quiescence"
    )
    out["diagnostic_results_do_not_promote"] = policy.get("diagnostic_results_do_not_promote")
    return out


def _frontier_f011() -> dict[str, Any]:
    out: dict[str, Any] = {"id": "F011", "present": False}
    if not FRONTIER_PATH.is_file():
        out["reason"] = "CLAUDE_GLOBAL_FRONTIER.json not on disk in this checkout"
        return out
    entries = load_json(FRONTIER_PATH).get("entries") or []
    hit = next((e for e in entries if e.get("id") == "F011"), None)
    out["present"] = hit is not None
    if hit:
        out["title"] = hit.get("title")
        out["classification"] = hit.get("classification")
        out["integration_target"] = hit.get("integration_target")
    return out


# ---------------------------------------------------------------------------
# recovery / selftest / receipt
# ---------------------------------------------------------------------------


def _recovered() -> list[dict[str, Any]]:
    rows = [
        {
            "path": "tools/future/contamination.py",
            "on_disk_in_this_worktree": (REPO / "tools/future/contamination.py").is_file(),
            "role": (
                "LANDED machine-state snapshot, QUIESCENT/LIGHT/HEAVY/UNKNOWN class, "
                "paired A/B stats, assert_promotable -> PromotionRefused. This module "
                "imports it and does not fork it."
            ),
            "adequate_for_this_lane": False,
            "gap": (
                "No named SELF_MEASURED_DIRTY class, no stable contamination "
                "fingerprint, no legitimate-use APIs, no field-copy closure: a "
                "QUIESCENT PROTECTED_ABSOLUTE shell that copies dirty ab_stats "
                "numbers would pass contamination.assert_promotable."
            ),
        },
        {
            "path": "tools/future/evidence_snapshot.py",
            "on_disk_in_this_worktree": (REPO / "tools/future/evidence_snapshot.py").is_file(),
            "role": "Pinned Codex receipts the sidecar is allowed to read.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/future/lpc_dataset.py",
            "on_disk_in_this_worktree": (REPO / "tools/future/lpc_dataset.py").is_file(),
            "role": (
                "contamination_class is a required LPC field. Vocabulary there is "
                "PROTECTED_ABSOLUTE/DIAGNOSTIC_RELATIVE/STATIC_ONLY/UNKNOWN "
                "(measurement class), not QUIESCENT/LIGHT/HEAVY."
            ),
            "adequate_for_this_lane": False,
            "gap": "SELF_MEASURED_DIRTY must map to DIAGNOSTIC_RELATIVE/STATIC_ONLY, never PROTECTED_ABSOLUTE.",
        },
        {
            "path": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "on_disk_in_this_worktree": EVIDENCE_QUEUE.is_file(),
            "role": (
                "queue_policy.protected_start_requires_machine_quiescence and "
                "diagnostic_results_do_not_promote."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/future/qualification_pipeline.py",
            "on_disk_in_this_worktree": (REPO / "tools/future/qualification_pipeline.py").is_file(),
            "role": "Sequences quiescence assessment; never seizes the GPU. Consumes contamination.snapshot.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "hcli/agentos/benchmark_boundary.py",
            "on_disk_in_this_worktree": (REPO / "hcli/agentos/benchmark_boundary.py").is_file(),
            "role": "QUALIFIED_PROTECTED vs DIAGNOSTIC_CONTAMINATED. Different vocabulary.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/verify/perfgate.py",
            "on_disk_in_this_worktree": (REPO / "tools/verify/perfgate.py").is_file(),
            "role": "Paired ABAB, never mean alone, contamination_note on this machine.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/accelerator/bench.py",
            "on_disk_in_this_worktree": (REPO / "tools/accelerator/bench.py").is_file(),
            "role": "machine_quiescence; Codex surface. Not imported.",
            "adequate_for_this_lane": False,
        },
    ]
    rows.sort(key=lambda r: r["path"])
    return rows


def _public_return_protected_hits() -> list[str]:
    """AST: no public function *returns* a dict literal with measurement_class PROTECTED."""
    src_path = Path(__file__).resolve()
    tree = ast.parse(src_path.read_text())
    hits: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            for d in ast.walk(child.value):
                if not isinstance(d, ast.Dict):
                    continue
                for key, val in zip(d.keys, d.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "measurement_class"
                        and isinstance(val, ast.Constant)
                        and val.value == PROTECTED
                    ):
                        hits.append(node.name)
    return sorted(set(hits))


def selftest() -> dict[str, Any]:
    """Watch every promotion path fail. A guard nobody has watched fail is not a guard."""
    probes = {
        "processes": {
            "status": "OK",
            "method": "injected",
            "cpu_pct_available": True,
            "no_name_filter": True,
            "n_enumerated": 2,
            "all": [
                {"pid": 11, "name": "Python", "cpu_pct": 8.0, "rss_gib": 12.0, "state": "R"},
                {"pid": 12, "name": "idle", "cpu_pct": 0.1, "rss_gib": 0.1, "state": "S"},
            ],
            "reason": None,
        },
        "load": {"status": "OK", "load_1m": 4.0, "load_5m": 3.0, "load_15m": 2.0, "ncpu": 28},
        "memory": {
            "status": "OK",
            "pressure_level": 1,
            "pressure_name": "warn",
            "pages": {},
            "bytes": {},
        },
        "gpu_occupancy": {
            "status": "OK",
            "device_utilization_pct": 5,
            "renderer_utilization_pct": 4,
            "tiler_utilization_pct": 1,
        },
        "thermal": {"status": "UNKNOWN", "reason": "selftest"},
        "machine_identity": {"hash": "selftest-identity", "fields": {"hw.model": "selftest"}},
    }
    env = dirty_snapshot(
        probes=probes,
        declared_resident={"model": "selftest-resident", "pid": 11, "process_name": "Python"},
        benchmark_ordinal=0,
    )
    fp_a = env["contamination_fingerprint"]
    fp_b = contamination_fingerprint(env["snapshot"], classify_contamination(env["snapshot"]))
    if fp_a != fp_b:
        raise AssertionError("fingerprint not stable for the same snapshot")
    duration_pairs = [(10.0, 8.0)] * 7  # B faster for duration
    noisy_pairs = [(10.0, 10.1), (10.0, 9.9)]
    ab = cheap_paired_ab(duration_pairs, quantity_kind="duration", envelope=env)
    if ab["faster_arm"] != "B":
        raise AssertionError(f"expected B faster, got {ab['faster_arm']}")
    if ab["measurement_class"] != MEASUREMENT_CLASS:
        raise AssertionError("dirty A/B must be STATIC_ONLY")
    if ab["evidence_class"] != EVIDENCE_CLASS:
        raise AssertionError("dirty A/B must be SELF_MEASURED_DIRTY")
    if any(k in ab and isinstance(ab[k], (int, float)) for k in HARDWARE_FIELDS):
        raise AssertionError("dirty A/B leaked a hardware field")
    direction = effect_direction(duration_pairs, quantity_kind="duration", envelope=env)
    noisy = cheap_paired_ab(noisy_pairs, quantity_kind="duration", envelope=env)
    if noisy["sufficient_for_decision"]:
        raise AssertionError("noisy A/B should not be sufficient")
    ranked = rank_candidates(
        [
            {"id": "keep", "pairs": duration_pairs},
            {"id": "also", "pairs": [(10.0, 9.0)] * 7},
            {"id": "noisy", "pairs": noisy_pairs},
        ],
        quantity_kind="duration",
        envelope=env,
    )
    if not ranked["sufficient_for_decision"]:
        raise AssertionError("two sufficient dirty results should rank")
    if [r["id"] for r in ranked["ranked"]] != ["keep", "also"]:
        raise AssertionError(f"unexpected ranking {ranked['ranked']}")
    if any(u["id"] == "noisy" for u in ranked["unranked"]) is False:
        raise AssertionError("noisy candidate must not be used to rank")
    too_noisy_rank = rank_candidates(
        [{"id": "a", "pairs": noisy_pairs}, {"id": "b", "pairs": noisy_pairs}],
        quantity_kind="duration",
        envelope=env,
    )
    if too_noisy_rank["sufficient_for_decision"] or too_noisy_rank["ranked"]:
        raise AssertionError("noisy-only set must not rank")
    # Pair is (A, B). Duration [(8, 10)] means B is slower, so B is dominated.
    pruned = prune_dominated(
        [{"a": "keep", "b": "slow", "pairs": [(8.0, 10.0)] * MIN_PAIRS}],
        quantity_kind="duration",
        envelope=env,
    )
    if [p["id"] for p in pruned["pruned"]] != ["slow"]:
        raise AssertionError(f"expected slow pruned, got {pruned['pruned']}")
    if pruned["pruned"][0]["status"] != "PRUNED_SELF_MEASURED_DIRTY":
        raise AssertionError("prune status must not be PROTECTED_REJECT")

    stolen = {
        "measurement_class": PROTECTED,
        "contamination_class": "QUIESCENT",
        "evidence_class": PROTECTED,
        "promotable": True,
        "caveat": "pretty please",
        "ab_stats": {
            "median_ratio": ab["median_ratio"],
            "ratio_q1": ab["ratio_q1"],
            "ratio_q3": ab["ratio_q3"],
            "ratio_iqr": ab["ratio_iqr"],
            "bootstrap_ci95": ab["bootstrap_ci95"],
            "sufficient_for_decision": True,
            "n_kept": 7,
            "reason": "copied",
        },
        "median_ratio": ab["median_ratio"],
    }
    # The landed gate would accept this shell (QUIESCENT + PROTECTED + enough
    # pairs). That is the field-copy hole this module closes.
    try:
        contamination_assert_promotable(stolen)
        landed_would_accept = True
    except PromotionRefused:
        landed_would_accept = False

    offer = _must_raise(offer_for_promotion, ab, fragment=EVIDENCE_CLASS)
    flag = _must_raise(offer_for_promotion, {**dict(ab), "promotable": True, "offered_as": PROTECTED}, fragment=EVIDENCE_CLASS)
    convert = _must_raise(as_protected_absolute, ab, fragment="field copy")
    copy = _must_raise(copy_value_as, ab, PROTECTED, fragment="copying its value")
    ingest = _must_raise(ingest_as_protected, stolen, fragment="values match")
    mint = _must_raise(mint_protected_absolute, fragment="cannot mint")
    our_gate = _must_raise(assert_promotable, stolen, fragment="field copy")
    magnitude = _must_raise(
        _assert_no_magnitude,
        {"accepted_tps": ab["median_ratio"]},
        fragment="accepted_tps",
    )
    frozen = _must_raise(lambda: ab.__setitem__("evidence_class", PROTECTED), fragment="frozen")
    hits = _public_return_protected_hits()
    if hits:
        raise AssertionError(f"public functions return {PROTECTED} dicts: {hits}")

    sleeping = emit_workunit(sleeping_protected=True)
    if sleeping["status"] != "SLEEPING":
        raise AssertionError("protected follow-up must SLEEP, not synthesize a result")

    return {
        "fingerprint_stable": fp_a == fp_b,
        "faster_arm_duration": ab["faster_arm"],
        "noisy_insufficient": noisy["sufficient_for_decision"] is False,
        "rank_skips_noisy": True,
        "noisy_set_does_not_rank": True,
        "prune_status": pruned["pruned"][0]["status"],
        "landed_contamination_gate_would_accept_field_copy": landed_would_accept,
        "offer_for_promotion_raises": offer,
        "flag_does_not_promote": flag,
        "as_protected_absolute_raises": convert,
        "copy_value_as_raises": copy,
        "ingest_field_copy_raises": ingest,
        "mint_raises": mint,
        "assert_promotable_closes_field_copy": our_gate,
        "magnitude_refused": magnitude,
        "frozen_rebind_raises": frozen,
        "no_public_return_of_protected_absolute": True,
        "sleeping_protected_followup": sleeping["status"],
        "synthetic_pairs_are_fixtures_not_hardware": list(SYNTHETIC_PAIRS) != duration_pairs,
        "contamination_classes": list(CONTAMINATION_CLASSES),
    }


def build(*, probes: Mapping[str, Any] | None = None) -> Path:
    env = dirty_snapshot(probes=probes, benchmark_ordinal=None)
    gate = selftest()
    policy = queue_policy()
    # A fixture A/B bound to the LIVE envelope, using dimensionless pairs.
    live_ab = cheap_paired_ab(
        [(10.0, 8.0)] * MIN_PAIRS,
        quantity_kind="duration",
        envelope=env,
    )
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Named evidence class SELF_MEASURED_DIRTY: the resident's own "
            "measurements on a contaminated machine. Good for rank, direction, "
            "cheap A/B and prune. Structurally unable to become PROTECTED_ABSOLUTE."
        ),
        "measurement_class": MEASUREMENT_CLASS,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "vocabulary": {
            "SELF_MEASURED_DIRTY": (
                "evidence the resident took while it was the contamination. "
                "Guides. Never promotes."
            ),
            "DIAGNOSTIC_RELATIVE": "landed sibling class; guides; never promotes",
            "PROTECTED_ABSOLUTE": "protected-lease measurement. Decides. This sidecar cannot produce it.",
            "STATIC_ONLY": "everything this sidecar emits. Bench UNKNOWN.",
            "contamination_classes": list(CONTAMINATION_CLASSES),
            "legitimate_uses": list(LEGITIMATE_USES),
        },
        "envelope": dict(env),
        "resident_loaded": env.get("resident_loaded"),
        "gpu_processes": env.get("gpu_processes"),
        "memory_pressure": env.get("memory_pressure"),
        "thermal_state": env.get("thermal_state"),
        "competing_workloads": env.get("competing_workloads"),
        "contamination_fingerprint": env.get("contamination_fingerprint"),
        "contamination_class": env.get("contamination_class"),
        "legitimate_uses": {
            "rank": "rank_candidates(candidates, quantity_kind=, envelope=)",
            "direction": "effect_direction(pairs, quantity_kind=, envelope=)",
            "cheap_paired_ab": "cheap_paired_ab(pairs, quantity_kind=, envelope=)",
            "prune_dominated": "prune_dominated(comparisons, quantity_kind=, envelope=)",
            "dispatcher": "use(kind, ...)",
        },
        "direction_is_not_magnitude": {
            "reports": ["faster_arm", "median_ratio", "ratio_iqr", "bootstrap_ci95"],
            "refuses": sorted(MAGNITUDE_FIELDS),
            "write_receipt_backstop": sorted(HARDWARE_FIELDS),
            "fixture_ab": {
                "faster_arm": live_ab.get("faster_arm"),
                "median_ratio": live_ab.get("median_ratio"),
                "sufficient_for_decision": live_ab.get("sufficient_for_decision"),
                "reason": live_ab.get("reason"),
                "note": "dimensionless fixture pairs, not a hardware sample",
            },
        },
        "honest_sufficiency": {
            "rank": "a dirty result with insufficient pairs is unranked; n_rankable<2 refuses the ranking",
            "direction": "CI must exclude 1.0; otherwise UNDECIDED",
            "prune": "only clearly-dominated (CI excludes 1.0); never PROTECTED_REJECT",
            "min_pairs": MIN_PAIRS,
        },
        "structural_refusal": {
            "offer_for_promotion": "always raises PromotionRefused",
            "as_protected_absolute": "always raises; field copy is not a path",
            "copy_value_as": f"raises when target is {PROTECTED}",
            "ingest_as_protected": "raises on dirty_binding, registered token, value digest, flags, caveats",
            "mint_protected_absolute": "trap; always raises",
            "frozen_record": "assignment of evidence_class raises PromotionRefused",
            "landed_gate_hole_closed": (
                "contamination.assert_promotable accepts a QUIESCENT "
                f"{PROTECTED} shell that copies dirty ratios; "
                "dirty_measure.assert_promotable refuses that shell. "
                f"selftest.landed_contamination_gate_would_accept_field_copy="
                f"{gate['landed_contamination_gate_would_accept_field_copy']}"
            ),
            "selftest": gate,
        },
        "queue_policy": policy,
        "frontier_entry": _frontier_f011(),
        "recovered_implementation": _recovered(),
        "gaps_closed": [
            "Named evidence class SELF_MEASURED_DIRTY carrying resident_loaded "
            "(model, pid), gpu_processes, memory_pressure, thermal_state "
            "(UNKNOWN when unexposed), competing_workloads, and a contamination "
            "fingerprint stable for the same machine state.",
            "Explicit APIs for the only legitimate uses: rank_candidates, "
            "effect_direction, cheap_paired_ab, prune_dominated.",
            "Structural refusal: FrozenDirtyRecord, offer_for_promotion, "
            "as_protected_absolute, copy_value_as, ingest_as_protected, "
            "mint_protected_absolute. Field-copy of dirty ratios into a "
            f"{PROTECTED} shell is refused even when the landed contamination "
            "gate would accept it.",
            "Direction is not magnitude: dirty A/B reports faster_arm and "
            "robust ratio with spread; absolute latency/TPS raise "
            "DirtyMagnitudeRefused and write_receipt still refuses HARDWARE_FIELDS.",
            "Honest sufficiency: sufficient_for_decision with a reason; noisy "
            "dirty results are not used to rank.",
        ],
        "negative_findings": [
            "contamination.assert_promotable does not close field-copy of dirty "
            f"ratios into a QUIESCENT {PROTECTED} shell; this module does.",
            "PID-level GPU attribution is still unavailable without a protected "
            "lease (inherited from contamination.snapshot).",
            "Thermal state is UNKNOWN on this host without sudo unless sysctl "
            "exposes it (inherited).",
            "This sidecar does not source paired samples from hardware; fixture "
            "pairs in the receipt are dimensionless.",
            "Resident model identity is UNKNOWN unless the caller declares it; "
            "process name is not a model guess. resident_identity.py is a this-wave "
            "sibling and was not imported.",
            "lpc_dataset.contamination_class uses measurement-class vocabulary; "
            "SELF_MEASURED_DIRTY must never be stored there as PROTECTED_ABSOLUTE.",
            "No Era VI, no Odyssey IV. FPGA remains Accelerator/Physical Compiler/Fusion.",
        ],
        "resident_callable": resident_callable(),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", action="store_true", help="take a live dirty envelope and write the receipt")
    ap.add_argument("--build", action="store_true", help="same as --snapshot")
    ap.add_argument("--selftest", action="store_true", help="run the gate self-test and print JSON")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
