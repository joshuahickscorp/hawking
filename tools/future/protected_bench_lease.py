#!/usr/bin/env python3
"""G087: pause, measure, resume - encoded, not re-asked.

S025 §41. The user already decided this policy for G075: SIGSTOP the ModelLake
downloads, take the absolute measurement, resume. Asking again every time is the
ceremony the steer says to remove.

S025 §42 keeps the distinction that makes it worth having:

    contaminated window -> paired RELATIVE experiments are fine
    protected window    -> canonical ABSOLUTES and promotion

so science does not stop because ModelLake is streaming. Ratios run now;
quiescence is spent only on numbers that will be promoted.

THE RESUME IS THE DANGEROUS PART. A lease that stops five downloads and dies
leaves them stopped, so resume runs in a finally and the lease REFUSES to report
success unless every stopped pid is confirmed running again.

    python3 tools/future/protected_bench_lease.py --plan
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/protected_bench_lease.py"
RECEIPT_NAME = "PROTECTED_BENCH_LEASE.json"

CONTAMINATOR_PATTERN = "bin/hf download"
QUIESCENCE_LOADAVG_MAX = 4.0

ABSOLUTE = "PROTECTED_ABSOLUTE"
RELATIVE = "PAIRED_RELATIVE"


class LeaseRefused(RuntimeError):
    """The window could not be protected, or could not be given back."""


class ResumeFailed(RuntimeError):
    """Processes were stopped and are not running again. Never swallowed."""


def contaminators(pattern: str = CONTAMINATOR_PATTERN) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def _state(pid: int) -> str:
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                         capture_output=True, text=True)
    return out.stdout.strip()


def _stopped(pid: int) -> bool:
    return _state(pid).startswith("T")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def loadavg() -> float:
    return os.getloadavg()[0]


def what_a_contaminated_window_may_still_do() -> dict[str, Any]:
    """S025 §42. Do not stop all science because ModelLake is active."""
    return {
        "allowed": RELATIVE,
        "why": (
            "absolutes move with load while ratios do not - isolated qkvz read "
            "601 GB/s in one run and 347 in another while its ARM A ratio held at "
            "1.57x both times. A matched pair is valid under contention."
        ),
        "reserved_for_protected": ABSOLUTE,
        "reserved_why": "canonical absolutes and any number used for promotion",
    }


def acquire(pattern: str = CONTAMINATOR_PATTERN) -> dict[str, Any]:
    pids = contaminators(pattern)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass
    time.sleep(2.0)
    stopped = [p for p in pids if not _alive(p) or _stopped(p)]
    missed = [p for p in pids if p not in stopped]
    return {
        "pattern": pattern,
        "pids": pids,
        "stopped": stopped,
        "not_stopped": missed,
        "loadavg_after_stop": round(loadavg(), 2),
        "quiescent": loadavg() <= QUIESCENCE_LOADAVG_MAX,
        "quiescence_bar": QUIESCENCE_LOADAVG_MAX,
    }


def release(pids: Sequence[int]) -> dict[str, Any]:
    """Give the machine back. Raises if anything is left stopped."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            pass
    time.sleep(1.0)
    still = [p for p in pids if _alive(p) and _stopped(p)]
    if still:
        raise ResumeFailed(
            f"pids still stopped after SIGCONT: {still}. A lease that stops work "
            "and does not give it back is worse than one that never ran."
        )
    return {"resumed": list(pids), "still_stopped": [], "verified": True}


def measure(fn, *, pattern: str = CONTAMINATOR_PATTERN) -> dict[str, Any]:
    """Run fn() inside a protected window. Resume is in a finally, always."""
    lease = acquire(pattern)
    result: Any = None
    error: str | None = None
    try:
        if not lease["quiescent"]:
            raise LeaseRefused(
                f"loadavg {lease['loadavg_after_stop']} is above the "
                f"{QUIESCENCE_LOADAVG_MAX} bar after stopping "
                f"{len(lease['stopped'])} contaminators; this window is not "
                "protected and an absolute taken in it would be wrong"
            )
        result = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        released = release(lease["pids"])
    return {
        "lease": lease,
        "released": released,
        "measured": result,
        "error": error,
        "evidence_class": ABSOLUTE if error is None else "REFUSED",
    }


def build(run: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "hawking.future.protected_bench_lease.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY" if run is None else run["evidence_class"],
        "gpu_authority": False,
        "policy": {
            "steps": ["stop contaminators", "verify quiescence", "measure",
                      "resume", "verify resumed"],
            "resume_is_in_a_finally": True,
            "refuses_if_not_quiescent": True,
            "does_not_ask_each_time": (
                "the user decided this for G075; re-asking is the ceremony S025 "
                "§41 says to remove"
            ),
        },
        "contaminated_window": what_a_contaminated_window_may_still_do(),
        "run": run,
        "claim_boundary": (
            "The lease governs WHEN an absolute may be taken, not what it means. "
            "A protected window makes an absolute eligible; it does not make a "
            "dirty-source build clean or a single run repeatable."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if args.plan:
        print(json.dumps({
            "contaminators": contaminators(),
            "loadavg": round(loadavg(), 2),
            "would_be_quiescent_bar": QUIESCENCE_LOADAVG_MAX,
        }, indent=1))
        return 0
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=False, lane="protected-lease"),
        ))
        return 0
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
