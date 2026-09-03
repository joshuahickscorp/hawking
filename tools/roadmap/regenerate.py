"""One command for the whole roadmap regeneration, in the only correct order.

The cycle is four steps and the order is not obvious, because two of them write
the same file and one of them reads what another writes:

    1. python3 -m tools.roadmap --build        graph from HEAD blobs; also patches
                                               three pointer keys into the state
    2. python3 -m tools.roadmap.recompile      PART I-III + APPENDIX + COMPRESSION
                                               and the whole state, FROM the graph
    3. python3 -m tools.roadmap.saturation --emit   the delta receipt, from the graph
    4. python3 -m tools.roadmap.emit_revised   H-ROADMAP-REVISED.md, fingerprinted
                                               against HEAD and the state's sha256

Run recompile before build and the state renders from the PREVIOUS graph. That is
not hypothetical: it happened, and the regenerated frontier listed a gate as the
number one thing to do minutes after that gate closed, saying its acceptance
criterion had never been run. Step 4 must come last because it fingerprints the
state, so anything that rewrites the state afterwards makes the emitted roadmap
instantly stale against its own detector.

Every step is run with its output visible and its exit code checked. A `>/dev/null`
in a shell chain swallowed a KeyError here once, leaving the human-readable
roadmap newer than the machine-readable authority it is derived from.

    python3 -m tools.roadmap.regenerate            run the cycle
    python3 -m tools.roadmap.regenerate --check    verify freshness, change nothing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: (label, argv). Order is load-bearing; see the module docstring.
STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph", ("-m", "tools.roadmap", "--build")),
    ("documents+state", ("-m", "tools.roadmap.recompile",)),
    ("saturation", ("-m", "tools.roadmap.saturation", "--emit")),
    ("revised roadmap", ("-m", "tools.roadmap.emit_revised",)),
)

CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("saturation", ("-m", "tools.roadmap.saturation", "--check")),
)


def _run(label: str, args: tuple[str, ...], *, quiet: bool) -> float:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *args], cwd=REPO,
        capture_output=quiet, text=True,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        if quiet:
            sys.stderr.write(proc.stdout or "")
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(
            f"regeneration FAILED at step {label!r} (exit {proc.returncode}). "
            "The remaining steps were not run, so the documents and the state may "
            "now disagree; fix the cause and re-run the whole cycle."
        )
    return elapsed


def regenerate(*, quiet: bool = False) -> dict[str, float]:
    timings: dict[str, float] = {}
    for label, args in STEPS:
        print(f"[{label}] {' '.join(args)}", flush=True)
        timings[label] = _run(label, args, quiet=quiet)
        print(f"[{label}] {timings[label]:.1f}s", flush=True)
    return timings


def check() -> int:
    """Verify freshness without writing. Non-zero when regeneration is needed."""
    from tools.roadmap import emit_revised
    failures = []
    for label, args in CHECKS:
        proc = subprocess.run([sys.executable, *args], cwd=REPO,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(f"{label}: {(proc.stdout or proc.stderr).strip().splitlines()[-1:]}")
    if emit_revised.check() != 0:
        failures.append("revised roadmap: STALE against current authority")
    for line in failures:
        print(f"STALE {line}")
    if not failures:
        print("FRESH: graph, state, documents, saturation receipt and revised roadmap agree")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report staleness and change nothing")
    ap.add_argument("--quiet", action="store_true",
                    help="capture step output; it is still printed on failure")
    args = ap.parse_args()
    if args.check:
        return check()
    started = time.perf_counter()
    timings = regenerate(quiet=args.quiet)
    total = time.perf_counter() - started
    widest = max(len(k) for k in timings)
    print()
    for label, secs in timings.items():
        print(f"    {label:{widest}}  {secs:6.1f}s")
    print(f"    {'TOTAL':{widest}}  {total:6.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
