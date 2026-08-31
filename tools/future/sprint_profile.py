"""SPRINT_WALL_ATTRIBUTION — name the tallest denominator so the next attempt attacks cost.

If a 24-hour sprint misses, the required response is not a motivational
status. It is a wall-attribution profile. UNKNOWN is a first-class bucket
and is never folded into "other". Zero is a claim; absence is UNKNOWN. A
profile whose buckets sum neatly to the wall clock guessed.

Does not measure hardware, does not take a GPU lease, and does not invent
seconds for a qualitative scar. `tallest()` refuses to name a winner when
UNKNOWN exceeds the largest attributed bucket — the answer then is measure
first.

    python3 tools/future/sprint_profile.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
from datetime import datetime
from typing import Any, Mapping, Sequence

from tools.future import autonomy_run as ar
from tools.future import autonomy_scars as asc
from tools.future import autonomy_trial as at
from tools.future._common import git, write_receipt
from tools.future.frontiers import load_optional
from tools.future.repro_science import FailClosed

RECEIPT = "ODYSSEY_LAUNCH_SPRINT_PROFILE.json"
SCHEMA = "hawking.future.sprint_profile.v1"
TIMELINE_REL = "receipts/future/AUTONOMY_TIMELINE_1h.json"
RUN_RECEIPT_REL = "receipts/future/AUTONOMY_RUN.json"
SCARS_RECEIPT_REL = "receipts/future/AUTONOMY_SCARS.json"

TARGET_S = 24 * 3600
TRIAL_1H = "1h"

BUCKET_IDS: tuple[str, ...] = (
    "scheduler",
    "resident_reasoning",
    "compilation",
    "gpu",
    "model_loading",
    "verification",
    "source_io",
    "process_wait",
    "trial_reruns",
    "human_authority",
    "infrastructure_defect",
)

MEASURED = "measured"
INFERRED = "inferred"
UNKNOWN = "unknown"
DERIVATIONS = (MEASURED, INFERRED, UNKNOWN)


def _fail(fault: str, reason: str) -> None:
    raise FailClosed(fault, reason)


def _as_seconds(value: Any, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("seconds_not_numeric", f"{what} is {value!r}, not a duration")
    if value < 0:
        _fail("negative_seconds", f"{what} is {value!r}; a duration cannot be negative")
    return int(value)


def make_bucket(
    bucket_id: str,
    *,
    attributed_seconds: int | float | None,
    derivation: str,
    evidence: Sequence[Any],
    why: str,
) -> dict[str, Any]:
    """One named bucket. UNKNOWN must not carry a number; a number is a claim."""
    if bucket_id not in BUCKET_IDS:
        _fail("unknown_bucket", f"{bucket_id!r} is not a sprint-wall bucket")
    if derivation not in DERIVATIONS:
        _fail("unknown_derivation", f"{derivation!r} is not measured/inferred/unknown")
    if derivation == UNKNOWN:
        if attributed_seconds is not None:
            _fail(
                "unknown_must_not_carry_seconds",
                f"{bucket_id}: UNKNOWN with {attributed_seconds!r}s would launder a guess "
                "as a measurement (zero is a claim; absence is UNKNOWN)",
            )
        seconds: int | None = None
    else:
        if attributed_seconds is None:
            _fail(
                "attributed_without_seconds",
                f"{bucket_id}: {derivation} requires a duration, or it is UNKNOWN",
            )
        seconds = _as_seconds(attributed_seconds, what=f"{bucket_id}.attributed_seconds")
    return {
        "id": bucket_id,
        "attributed_seconds": seconds,
        "derivation": derivation,
        "evidence": list(evidence),
        "why": why,
    }


def unknown_bucket(bucket_id: str, why: str, evidence: Sequence[Any] = ()) -> dict[str, Any]:
    return make_bucket(
        bucket_id,
        attributed_seconds=None,
        derivation=UNKNOWN,
        evidence=evidence,
        why=why,
    )


def is_attributed(bucket: Mapping[str, Any]) -> bool:
    """A duration with a derivation that is not UNKNOWN. Zero counts; None does not."""
    if str(bucket.get("derivation") or "") == UNKNOWN:
        return False
    sec = bucket.get("attributed_seconds")
    if sec is None or isinstance(sec, bool) or not isinstance(sec, (int, float)):
        return False
    return True


def attributed_sum(buckets: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(b["attributed_seconds"]) for b in buckets if is_attributed(b))


def remainder_of(
    *,
    elapsed_s: int | None,
    buckets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Elapsed minus attributed. Never distributed into the named buckets.

    A missing elapsed clock cannot be subtracted from. The 24h target is a
    budget, not a substitute elapsed, and is not used here.
    """
    attributed = attributed_sum(buckets)
    if elapsed_s is None:
        return {
            "seconds": None,
            "derivation": UNKNOWN,
            "attributed_sum_s": attributed,
            "distributed_into_buckets": False,
            "why": (
                "sprint elapsed is unmeasured; remainder is not (24h target - attributed) "
                "because the target is a budget, not a clock"
            ),
        }
    elapsed = _as_seconds(elapsed_s, what="elapsed_s")
    rem = elapsed - attributed
    if rem < 0:
        _fail(
            "attributed_exceeds_elapsed",
            f"attributed {attributed}s exceeds elapsed {elapsed}s; that is double-counting, "
            "not a remainder to clamp",
        )
    return {
        "seconds": rem,
        "derivation": MEASURED,
        "attributed_sum_s": attributed,
        "elapsed_s": elapsed,
        "distributed_into_buckets": False,
        "why": "elapsed - sum(attributed); the unnamed remainder is not folded into 'other'",
    }


def tallest(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Largest ATTRIBUTED bucket, or a refusal when UNKNOWN dominates.

    Naming a winner while UNKNOWN is larger is a ranking of what we happened
    to measure, not of the wall. The honest answer is then 'measure first'.
    """
    buckets = list(profile.get("buckets") or [])
    if not buckets:
        _fail("empty_profile", "tallest() has no buckets to rank")
    remainder = profile.get("unattributed_remainder")
    if not isinstance(remainder, Mapping):
        _fail("missing_remainder", "tallest() requires an unattributed remainder, not a silent 0")

    attributed = [b for b in buckets if is_attributed(b)]
    unknown_ids = [str(b.get("id")) for b in buckets if not is_attributed(b)]
    rem_s = remainder.get("seconds")
    rem_unknown = rem_s is None or str(remainder.get("derivation") or "") == UNKNOWN

    if not attributed:
        return {
            "named": False,
            "winner": None,
            "reason": "no attributed bucket; UNKNOWN is the whole profile; measure first",
            "unknown_bucket_ids": unknown_ids,
            "unknown_mass_s": rem_s,
        }

    winner = max(attributed, key=lambda b: int(b["attributed_seconds"]))
    win_s = int(winner["attributed_seconds"])
    leader = {
        "id": winner["id"],
        "attributed_seconds": win_s,
        "derivation": winner["derivation"],
    }

    if rem_unknown:
        return {
            "named": False,
            "winner": None,
            "reason": (
                "UNKNOWN is unquantified (sprint elapsed unmeasured or remainder absent), "
                f"so it cannot be shown not to exceed {winner['id']} at {win_s}s; measure first"
            ),
            "leader_if_unknown_were_smaller": leader,
            "unknown_bucket_ids": unknown_ids,
            "unknown_mass_s": None,
        }

    mass = _as_seconds(rem_s, what="unattributed_remainder.seconds")
    if mass >= win_s:
        return {
            "named": False,
            "winner": None,
            "reason": (
                f"UNKNOWN ({mass}s) exceeds-or-ties tallest attributed "
                f"{winner['id']} ({win_s}s); measure first"
            ),
            "leader_if_unknown_were_smaller": leader,
            "unknown_bucket_ids": unknown_ids,
            "unknown_mass_s": mass,
        }
    return {
        "named": True,
        "winner": winner["id"],
        "attributed_seconds": win_s,
        "derivation": winner["derivation"],
        "unknown_mass_s": mass,
        "unknown_bucket_ids": unknown_ids,
        "reason": (
            f"{winner['id']} at {win_s}s is strictly larger than the unattributed remainder "
            f"({mass}s)"
        ),
    }


def _parse_ts(raw: str, *, what: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        _fail("missing_timestamp", f"{what} is empty")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        _fail("malformed_timestamp", f"{what}={text!r}: {exc}")


def invalidated_seconds(row: Mapping[str, Any]) -> dict[str, Any]:
    """Subtract recorded start/kill. A missing stamp is a refusal, not now()."""
    started_raw = str(row.get("started") or "")
    killed_raw = str(row.get("killed") or "")
    if not started_raw or not killed_raw:
        _fail(
            "incomplete_invalidated_interval",
            "INVALIDATED_BY_SUBSTRATE_MUTATION row missing started or killed; "
            "refusing to guess 'still running' or 'now'",
        )
    started = _parse_ts(started_raw, what="invalidated.started")
    killed = _parse_ts(killed_raw, what="invalidated.killed")
    delta = (killed - started).total_seconds()
    if delta < 0:
        _fail(
            "inverted_invalidated_interval",
            f"killed {killed_raw} precedes started {started_raw}",
        )
    return {
        "started": started_raw,
        "killed": killed_raw,
        "elapsed_s": int(delta),
        "verdict": str(row.get("verdict") or ""),
        "trial": str(row.get("trial") or ""),
        "why": str(row.get("why") or ""),
        "source": "tools.future.autonomy_run.INVALIDATED_RUNS",
        "derivation": MEASURED,
        "measurement": "isoformat timestamps subtracted; not estimated",
    }


def _blob_digest(blob: str) -> str:
    return hashlib.sha256(blob.encode()).hexdigest()


def timeline_snapshots() -> list[dict[str, Any]]:
    """Unique JSON snapshots of the 1h timeline, oldest commit first.

    The live file was rewritten. HEAD currently PASSES verify(); the first
    1h trial that FAILED is a historical snapshot. Using only the live file
    would invert the causal claim.
    """
    rel = TIMELINE_REL
    hashes = [
        ln.strip()
        for ln in git("log", "--format=%H", "--reverse", "--", rel).splitlines()
        if ln.strip()
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for commit in hashes:
        blob = git("show", f"{commit}:{rel}")
        if not blob:
            continue
        digest = _blob_digest(blob)
        if digest in seen:
            continue
        seen.add(digest)
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError as exc:
            out.append({
                "commit": commit,
                "path_taken": f"git:{commit}:{rel}",
                "unreadable": type(exc).__name__,
            })
            continue
        if not isinstance(doc, dict):
            continue
        out.append({
            "commit": commit,
            "path_taken": f"git:{commit}:{rel}",
            "digest": digest,
            "doc": doc,
        })
    live, live_taken = load_optional(rel)
    if isinstance(live, dict):
        blob = json.dumps(live, sort_keys=True, separators=(",", ":"))
        digest = _blob_digest(blob)
        if digest not in seen:
            out.append({
                "commit": None,
                "path_taken": live_taken,
                "digest": digest,
                "doc": live,
            })
    return out


def judge_1h(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Invoke the trial judge. A commit message is not a verdict."""
    return at.verify(TRIAL_1H, doc)


def failed_1h_trial(snapshots: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Earliest 1h timeline that the judge actually FAILs.

    One filename is one trial overwritten, not N additive intervals. Later
    PASS rewrites are not extra rerun cost and are not the failed first trial.
    """
    rows = list(snapshots) if snapshots is not None else timeline_snapshots()
    for row in rows:
        doc = row.get("doc")
        if not isinstance(doc, Mapping):
            continue
        verdict = judge_1h(doc)
        if verdict.get("verdict") != "FAIL":
            continue
        elapsed = verdict.get("elapsed_s")
        if elapsed is None:
            elapsed = doc.get("elapsed_s")
        if elapsed is None:
            continue
        return {
            "path_taken": row.get("path_taken"),
            "commit": row.get("commit"),
            "elapsed_s": _as_seconds(elapsed, what="failed_1h.elapsed_s"),
            "duration_s": verdict.get("duration_s"),
            "verdict": verdict.get("verdict"),
            "unmet": list(verdict.get("unmet") or []),
            "elapsed_meets_duration": verdict.get("elapsed_meets_duration"),
            "elapsed_is_not_a_pass": verdict.get("elapsed_is_not_a_pass"),
            "reason": verdict.get("reason"),
            "n_events": verdict.get("n_events"),
            "judge": "tools.future.autonomy_trial.verify('1h', timeline)",
            "source": TIMELINE_REL,
            "derivation": MEASURED,
            "measurement": "timeline.elapsed_s judged FAIL by autonomy_trial.verify; not a commit subject",
        }
    return None


def live_1h_verdict(snapshots: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Whatever the newest snapshot judges as. Status labels stay hypotheses."""
    rows = list(snapshots) if snapshots is not None else timeline_snapshots()
    if not rows:
        return None
    for row in reversed(rows):
        doc = row.get("doc")
        if not isinstance(doc, Mapping):
            continue
        verdict = judge_1h(doc)
        return {
            "path_taken": row.get("path_taken"),
            "commit": row.get("commit"),
            "verdict": verdict.get("verdict"),
            "unmet": list(verdict.get("unmet") or []),
            "elapsed_s": verdict.get("elapsed_s"),
            "reason": verdict.get("reason"),
            "judge": "tools.future.autonomy_trial.verify('1h', timeline)",
        }
    return None


def trial_reruns_bucket(
    *,
    failed: Mapping[str, Any] | None,
    invalidated: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    total = 0
    if failed is not None:
        evidence.append(dict(failed))
        total += int(failed["elapsed_s"])
    for row in invalidated:
        evidence.append(dict(row))
        total += int(row["elapsed_s"])
    if not evidence:
        return unknown_bucket(
            "trial_reruns",
            why=(
                "no judged-FAIL 1h timeline and no INVALIDATED_BY_SUBSTRATE_MUTATION "
                "interval recovered; refusing to treat absence as 0s"
            ),
            evidence=[{"path": TIMELINE_REL, "path_taken": "absent"},
                      {"path": "tools.future.autonomy_run.INVALIDATED_RUNS", "n": 0}],
        )
    return make_bucket(
        "trial_reruns",
        attributed_seconds=total,
        derivation=MEASURED,
        evidence=evidence,
        why=(
            "first 1h trial judged FAIL plus every INVALIDATED_BY_SUBSTRATE_MUTATION "
            "interval; both recovered from disk/git and judged/subtracted, not estimated"
        ),
    )


def infrastructure_defect_bucket(scars: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Four autonomy scars are evidence. None of them quantify wall seconds."""
    rows = [dict(s) for s in scars]
    if not rows:
        return unknown_bucket(
            "infrastructure_defect",
            why="no autonomy scars recovered; absence is UNKNOWN, not 0s of defect",
            evidence=[{"path": SCARS_RECEIPT_REL, "n_scars": 0}],
        )
    evidence = [
        {
            "id": s.get("id"),
            "family": s.get("family"),
            "verdict": s.get("verdict"),
            "cost": s.get("cost"),
            "law": s.get("law"),
            "source": "tools.future.autonomy_scars.scars()",
        }
        for s in rows
    ]
    return unknown_bucket(
        "infrastructure_defect",
        why=(
            f"{len(rows)} scars record qualitative cost and the symptom that hid it; "
            "none carry a duration. Inventing seconds would be estimation, which this "
            "bucket is forbidden from doing"
        ),
        evidence=evidence,
    )


def _nested_why(bucket_id: str) -> str:
    return (
        f"{bucket_id} intervals exist inside the 1h trial timelines, but those "
        "intervals are already counted wholesale as trial_reruns; splitting them "
        "out here would double-count. Independent sprint-level evidence for this "
        "bucket was not recovered"
    )


def _static_unknown_buckets() -> list[dict[str, Any]]:
    """Buckets this sidecar cannot measure. Parked/unavailable is not zero."""
    return [
        unknown_bucket(
            "scheduler",
            why=_nested_why("scheduler"),
            evidence=[{"recovered": "tools/future/autonomy_run.py loop is the scheduler under test"}],
        ),
        unknown_bucket(
            "resident_reasoning",
            why=(
                "autonomy_run records resident_model_cognition UNAVAILABLE: the loop "
                "under test is HCLI orchestration, not model cognition. UNAVAILABLE "
                "is not 0s of reasoning"
            ),
            evidence=[{"source": "tools.future.autonomy_run.run state_recovered payload"}],
        ),
        unknown_bucket(
            "compilation",
            why=(
                "turnaround.py records compile as UNKNOWN (cargo_build_forbidden: "
                "would contend with the live campaign). A CPU proxy of compile is refused"
            ),
            evidence=[{"source": "tools.future.turnaround.GPU_OR_BUILD_PHASES", "phase": "compile"}],
        ),
        unknown_bucket(
            "gpu",
            why=(
                "sidecar has no GPU authority and takes no lease. Blocked lanes park "
                "SLEEPING; parked is not a measurement that GPU time was 0s — the live "
                "Codex campaign may be using the GPU, and this partition cannot see it"
            ),
            evidence=[{"source": "tools.future.frontiers.HARDWARE_LANES", "gpu_authority": False}],
        ),
        unknown_bucket(
            "model_loading",
            why="no model-load interval receipt in this partition; load_ns on the scoreboard is null",
            evidence=[{"source": "tools.future.turnaround.SCOREBOARD_DEVELOPMENT_PHASES", "load_ns": None}],
        ),
        unknown_bucket(
            "verification",
            why=_nested_why("verification"),
            evidence=[{"source": TIMELINE_REL, "note": "specimen_verify spans sit inside the 1h trial"}],
        ),
        unknown_bucket(
            "source_io",
            why=_nested_why("source_io"),
            evidence=[{"source": TIMELINE_REL, "note": "whole-tree specimen hashing is nested I/O"}],
        ),
        unknown_bucket(
            "process_wait",
            why=_nested_why("process_wait"),
            evidence=[{"source": TIMELINE_REL, "note": "process_failed / window-timeout sit inside the 1h trial"}],
        ),
        unknown_bucket(
            "human_authority",
            why="no human-gate / authority-wait interval receipt; conversational wait was not observed, and 'not observed' is not 0s",
            evidence=[{"source": "tools.future.autonomy_trial.eval_never_conversational_wait"}],
        ),
    ]


def attribute(
    *,
    snapshots: Sequence[Mapping[str, Any]] | None = None,
    invalidated_rows: Sequence[Mapping[str, Any]] | None = None,
    scars: Sequence[Mapping[str, Any]] | None = None,
    elapsed_s: int | None = None,
    elapsed_why: str | None = None,
) -> dict[str, Any]:
    """Attribute the sprint wall. Missing inputs become UNKNOWN, never zero.

    snapshots/invalidated_rows/scars = None means recover from disk/git.
    An empty list is an explicit absence (tests use this as a negative control).
    elapsed_s = None is the live case: no sprint clock exists on disk.
    """
    if snapshots is None:
        recovered_snaps: Sequence[Mapping[str, Any]] = timeline_snapshots()
        snapshots_path = "git log + load_optional " + TIMELINE_REL
    else:
        recovered_snaps = snapshots
        snapshots_path = "supplied"

    failed = failed_1h_trial(recovered_snaps)
    live = live_1h_verdict(recovered_snaps)

    if invalidated_rows is None:
        inv_src = [dict(r) for r in ar.INVALIDATED_RUNS]
        inv_path = "tools.future.autonomy_run.INVALIDATED_RUNS"
        run_doc, run_taken = load_optional(RUN_RECEIPT_REL)
        if isinstance(run_doc, dict) and run_doc.get("invalidated_runs"):
            # Same intervals; receipt is corroboration, not a second copy.
            inv_path = inv_path + f" + {run_taken}"
    else:
        inv_src = [dict(r) for r in invalidated_rows]
        inv_path = "supplied"

    invalidated: list[dict[str, Any]] = []
    for row in inv_src:
        invalidated.append(invalidated_seconds(row))

    if scars is None:
        scar_rows = asc.scars()
        scar_path = "tools.future.autonomy_scars.scars()"
    else:
        scar_rows = [dict(s) for s in scars]
        scar_path = "supplied"

    reruns = trial_reruns_bucket(failed=failed, invalidated=invalidated)
    defect = infrastructure_defect_bucket(scar_rows)

    by_id = {b["id"]: b for b in _static_unknown_buckets()}
    by_id["trial_reruns"] = reruns
    by_id["infrastructure_defect"] = defect
    buckets = [by_id[i] for i in BUCKET_IDS]

    if elapsed_s is None and elapsed_why is None:
        elapsed_why = (
            "no sprint start/stop receipt on disk or in git; the 24h figure is the "
            "declared target, not a measured elapsed"
        )

    remainder = remainder_of(elapsed_s=elapsed_s, buckets=buckets)
    ranking = tallest({
        "buckets": buckets,
        "unattributed_remainder": remainder,
    })

    return {
        "schema": SCHEMA,
        "sprint": {
            "target_s": TARGET_S,
            "target_is": "declared 24-hour sprint duration; not a measured elapsed",
            "elapsed_s": elapsed_s,
            "elapsed_derivation": UNKNOWN if elapsed_s is None else MEASURED,
            "why_elapsed_unknown": elapsed_why if elapsed_s is None else None,
        },
        "buckets": buckets,
        "bucket_ids": list(BUCKET_IDS),
        "attributed_sum_s": attributed_sum(buckets),
        "unattributed_remainder": remainder,
        "tallest": ranking,
        "recovery": {
            "timeline_rel": TIMELINE_REL,
            "snapshots_from": snapshots_path,
            "n_snapshots": len(list(recovered_snaps)),
            "failed_1h": None if failed is None else {
                k: failed[k] for k in (
                    "path_taken", "commit", "elapsed_s", "verdict", "unmet",
                    "reason", "judge",
                ) if k in failed
            },
            "live_1h": live,
            "invalidated_from": inv_path,
            "n_invalidated": len(invalidated),
            "scars_from": scar_path,
            "n_scars": len(scar_rows),
        },
    }


def build() -> Any:
    profile = attribute()
    ranking = profile["tallest"]
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "If the 24-hour Odyssey-launch sprint misses, name the tallest "
            "denominator so the next attempt attacks actual cost. UNKNOWN is "
            "first-class. Buckets are not required to sum to the wall clock."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "sprint": profile["sprint"],
        "buckets": profile["buckets"],
        "bucket_ids": profile["bucket_ids"],
        "attributed_sum_s": profile["attributed_sum_s"],
        "unattributed_remainder": profile["unattributed_remainder"],
        "tallest": ranking,
        "recovery": profile["recovery"],
        "invariants": [
            "a bucket with no evidence is UNKNOWN, never 0",
            "UNKNOWN is not folded into 'other'",
            "buckets are not required to sum to the wall clock",
            "unattributed remainder is reported, never distributed",
            "tallest() refuses when UNKNOWN exceeds the largest attributed bucket",
            "the first 1h trial is judged by autonomy_trial.verify, not by a commit subject",
            "scar costs stay qualitative until a duration exists on disk",
        ],
        "recovered_implementation": [
            "tools/future/autonomy_trial.py — verify() is the judge; this module invokes it",
            "tools/future/autonomy_run.py — INVALIDATED_RUNS timestamps (started/killed)",
            "tools/future/autonomy_scars.py — four orchestrator scars with qualitative cost",
            "tools/future/turnaround.py — compile/GPU/load phases already UNKNOWN; not remeasured",
            "tools/future/frontiers.py — load_optional (sparse-checkout coping) and HARDWARE_LANES",
            "tools/future/_common.py — sealed receipts, hardware-claim block, read-only git",
            f"{TIMELINE_REL} — real 1h timeline; historical FAIL snapshot recovered via git log",
            f"{SCARS_RECEIPT_REL} — corroborates autonomy_scars.scars()",
            f"{RUN_RECEIPT_REL} — corroborates INVALIDATED_RUNS",
        ],
        "gaps_closed": [
            "no sprint-wall attribution profile existed; a miss had no named denominator",
            "trial_reruns derived from a judged-FAIL 1h timeline plus the invalidated rerun",
            "infrastructure_defect cites four scars without inventing seconds",
            "UNKNOWN is a first-class bucket; remainder is not distributed to make 100%",
            "tallest() refuses when UNKNOWN dominates, so 'measure first' is the ranking",
        ],
        "negative_findings": [
            "sprint elapsed is unmeasured: no start/stop receipt, so remainder seconds stay UNKNOWN",
            "9 of 11 buckets have no independent quantified duration",
            "infrastructure_defect has evidence (four scars) and still no seconds — qualitative cost is not a duration",
            "the live AUTONOMY_TIMELINE_1h.json currently PASSES verify(); using it as the failed first trial would invert the causal claim. The FAIL snapshot is historical (git log, judged)",
            "compilation, GPU, and model_loading cannot be filled here: STATIC_ONLY, no GPU lease, cargo build forbidden",
            "nested 1h-trial phases (scheduler/verification/source_io/process_wait) were not split out of trial_reruns (double-count refusal)",
        ],
        "what_you_could_not_establish": [
            "whether the 24h target has elapsed, is in flight, or has not started",
            "wall seconds of each infrastructure scar",
            "GPU / compile / load time of the live Codex campaign (wrong partition, no authority)",
            "a unique tallest denominator — UNKNOWN currently dominates",
        ],
        "next_workunits": [
            "write a sprint start/stop receipt so elapsed (and therefore UNKNOWN mass) can be compared",
            "keep scar costs qualitative until a measured duration exists; do not estimate one",
            "do not rerun a 1h trial while editing a subprocess it invokes (INVALIDATED_BY_SUBSTRATE_MUTATION)",
        ],
        "resident_callable": {
            "entry_point": "tools.future.sprint_profile.attribute() / tallest()",
            "workunit": (
                "one CPU_ANALYSIS unit; recover 1h timeline history, invoke "
                "autonomy_trial.verify, subtract invalidated timestamps, cite scars"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.LATENCY.cpu-turnaround",
            "fails_closed": (
                "absent timeline/scars/timestamps become UNKNOWN, never 0; "
                "UNKNOWN-with-a-number raises; attributed-without-seconds raises; "
                "tallest() refuses rather than ranking a partial measurement; "
                "inverted or incomplete invalidated intervals raise"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/sprint_profile.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
