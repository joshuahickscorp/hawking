#!/usr/bin/env python3
"""G101: ModelLake as scheduler state, not a downloader Claude checks on.

S027 §2, §21, §22. The watcher already writes everything a scheduler needs -
active jobs, remaining bytes, free space, network throughput - into a JSONL it
appends to every few seconds. Nothing consumed it. This reads the live tail and
joins it to the specimen registry so a newly sealed model becomes schedulable
material without a conversational boundary.

ETAs ARE COMPUTED FROM THE MEASURED NETWORK RATE, not from a nominal bandwidth.
When no recent network sample exists the ETA is None and says so, because a
scheduler that plans against an invented rate schedules against fiction.

    python3 tools/future/modellake_scheduler_view.py --build
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402
import specimen_registry as sr  # noqa: E402

RECORDED_BY = "tools/future/modellake_scheduler_view.py"
RECEIPT_NAME = "MODELLAKE_SCHEDULER_VIEW.json"

WATCH_LOG = REPO / "workspace/campaign/odyssey/downloads/modellake-watch.jsonl"
# The log is hundreds of megabytes and append-only; only the tail is live state.
TAIL_BYTES = 400_000
# A sample older than this is not "current" and must not be presented as such.
STALE_SECONDS = 900.0

# S027 §22. What MODEL_SEALED must wake, in order.
SEAL_TRIGGERS = (
    "fingerprint",
    "role evaluation",
    "law and scar lookup",
    "initial economics",
    "WorkGraph creation",
    "possible prefetch or load",
)


class LakeViewRefused(RuntimeError):
    """The watcher log is missing or carries no usable sample."""


def _tail() -> list[dict[str, Any]]:
    if not WATCH_LOG.is_file():
        raise LakeViewRefused(
            f"{WATCH_LOG} is not on disk; the watcher's live state is the only "
            "source here and an empty view would read as 'nothing is "
            "downloading' rather than 'the watcher is not running'"
        )
    size = os.path.getsize(WATCH_LOG)
    with open(WATCH_LOG, "rb") as f:
        f.seek(max(0, size - TAIL_BYTES))
        raw = f.read().decode("utf-8", "ignore").splitlines()
    out = []
    for line in raw[1:]:  # first line is probably a fragment
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not out:
        raise LakeViewRefused("the watcher log tail carries no parseable event")
    return out


def _age_s(ts: str | None, now: float | None = None) -> float | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None
    ref = now if now is not None else datetime.now(timezone.utc).timestamp()
    return ref - t


def live(now: float | None = None) -> dict[str, Any]:
    ev = _tail()
    samples = [e for e in ev if e.get("event") == "watcher_sample"]
    nets = [e for e in ev if e.get("event") == "network_sample"]
    if not samples:
        raise LakeViewRefused("no watcher_sample in the log tail")
    s = samples[-1]
    age = _age_s(s.get("ts"), now)
    rates = [float(n["rx_bytes_per_sec"]) for n in nets[-12:]
             if n.get("rx_bytes_per_sec")]
    rate = statistics.median(rates) if rates else None
    return {
        "sample_ts": s.get("ts"),
        "sample_age_s": round(age, 1) if age is not None else None,
        "stale": (age is not None and age > STALE_SECONDS),
        "stale_threshold_s": STALE_SECONDS,
        "active_jobs": list(s.get("active_jobs") or []),
        "active_remaining_bytes": s.get("active_remaining_bytes"),
        "free_bytes": s.get("free_bytes"),
        "p0_done": s.get("p0_done"),
        "rx_bytes_per_sec_median": round(rate) if rate else None,
        "rx_samples_used": len(rates),
        "rate_is_none_because": (
            None if rate else
            "no network_sample in the log tail; an ETA computed from a nominal "
            "bandwidth would be fiction"
        ),
    }


def eta(now: float | None = None) -> dict[str, Any]:
    """Time to drain the active queue, from the MEASURED rate only."""
    lv = live(now)
    rem = lv["active_remaining_bytes"]
    rate = lv["rx_bytes_per_sec_median"]
    if not rem or not rate:
        return {
            "seconds": None, "hours": None,
            "why": "remaining bytes or measured rate is unavailable",
        }
    secs = rem / rate
    return {
        "remaining_bytes": rem,
        "rx_bytes_per_sec_median": rate,
        "seconds": round(secs),
        "hours": round(secs / 3600.0, 2),
        "is_an_estimate_because": (
            "the rate is a median of recent samples and will not hold: it "
            "varies with the host, the file mix and disk contention. This is a "
            "planning number, not a promise."
        ),
        "covers_active_jobs_only": lv["active_jobs"],
        "does_not_cover": (
            "queued specimens the watcher has not admitted yet, so the real "
            "time to a full library is longer than this"
        ),
    }


def arrivals(now: float | None = None) -> dict[str, Any]:
    """Join the live queue to the registry: what is coming, and what it is."""
    lv = live(now)
    reg = {r["id"]: r for r in sr.registry()}
    rows = []
    for job in lv["active_jobs"]:
        r = reg.get(job)
        rows.append({
            "id": job,
            "known_to_registry": r is not None,
            "lifecycle": r["lifecycle"] if r else None,
            "model_type": r["architecture"]["model_type"] if r else None,
            "already_sealed": bool(r and r["lifecycle"] in
                                   ("SEALED_SOURCE", "FINGERPRINTED")),
        })
    return {
        "active": rows,
        "n_active": len(rows),
        "n_unknown_to_registry": sum(1 for r in rows if not r["known_to_registry"]),
        "unknown_means": (
            "the download has not written a directory the registry can see "
            "yet, which is normal early in a job and is NOT an error"
        ),
    }


def seal_contract() -> dict[str, Any]:
    """S027 §22: MODEL_SEALED wakes real work, with no conversational boundary."""
    return {
        "event": "MODEL_SEALED(model_id)",
        "must_trigger": list(SEAL_TRIGGERS),
        "no_conversational_boundary": (
            "the sealed model becomes schedulable material at once; nobody is "
            "asked whether to look at it"
        ),
        "is_this_wired": False,
        "honest_status": (
            "DECLARED, NOT WIRED. The watcher emits download_exit and "
            "already_complete events and the registry derives SEALED_SOURCE "
            "from a manifest, so both halves exist - but nothing today turns "
            "the first into the second and then into a WorkGraph. Claiming "
            "otherwise would be the fake completion this campaign forbids."
        ),
        "what_wiring_it_needs": (
            "a consumer that tails this log, detects a manifest appearing, and "
            "emits the six triggers as WorkUnits through the frontier layer - "
            "the same path the causal pattern library already uses"
        ),
    }


def build() -> dict[str, Any]:
    lv = live()
    return {
        "obligation": "G101",
        "authority": "S027 §2, §21, §22",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "watch_log": str(WATCH_LOG.relative_to(REPO)),
        "live": lv,
        "eta": eta(),
        "arrivals": arrivals(),
        "seal_contract": seal_contract(),
        "registry_join": {
            "n_specimens_known": len(sr.registry()),
            "n_schedulable": len(sr.schedulable()),
        },
        "the_watcher_already_wrote_all_of_this": (
            "active jobs, remaining bytes, free space and network throughput "
            "were being appended every few seconds and nothing consumed them. "
            "The gap S027 §2 names was never missing data; it was a missing "
            "reader."
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
    print(json.dumps({k: doc[k] for k in
                      ("live", "eta", "arrivals", "seal_contract")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
