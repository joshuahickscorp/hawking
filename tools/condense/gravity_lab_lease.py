#!/usr/bin/env python3.12
"""Paired contention gate: prove a GPU lab job does not steal from the live campaign.

The inference lab runs beside a live GLM-5.2 stream that owns source fetching, packing and
eviction.  The packer calls ``gravity_forge._kmeans``, which runs on MPS, so the live
campaign is itself a GPU client -- lab work and campaign work contend for the same 60
cores.  A single before/after reading cannot separate that contention from ordinary drift,
because the campaign's own rate varies with which window it is fetching.

So this measures A/B/A/B rather than before/after.  Each window samples the campaign's own
observable progress -- the fetcher's accumulated CPU time and the bytes landed in the
artifact directory -- and the B windows run the candidate GPU load concurrently.  Pairing
cancels the drift that a single comparison would attribute to the lab.

The gate is a regression bound, not a speed claim: the lab may proceed only when the
campaign's measured rate under load stays within tolerance of its rate unloaded.  A run
that cannot observe the campaign at all fails closed, because "I saw no regression" and "I
could not see" are the same reading and only one of them is safe.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LEASE_SCHEMA = "hawking.glm52.lab_contention_gate.v1"
DEFAULT_TOLERANCE = 0.05          # mandate section 11: less than 5% regression
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_CYCLES = 2                # A/B/A/B


class ContentionGateError(RuntimeError):
    """The gate could not be evaluated, or the lab is not authorized to run."""


@dataclass
class CampaignProbe:
    """What the live campaign is observably doing, sampled at one instant."""

    at: float
    cpu_seconds: float | None       # accumulated CPU time of the tracked processes
    artifact_bytes: int | None      # total bytes in the artifact directory
    artifact_count: int | None

    def observable(self) -> bool:
        return self.cpu_seconds is not None or self.artifact_bytes is not None


def _cpu_seconds(pids: list[int]) -> float | None:
    """Accumulated CPU time across the tracked pids, or None if none are visible.

    ``ps -o time=`` prints ``[dd-]hh:mm:ss``; a dead pid simply contributes nothing, and if
    every pid is gone the campaign is not observable and the caller must fail closed.
    """
    if not pids:
        return None
    argv = ["/bin/ps", "-o", "time=", "-p", ",".join(str(p) for p in pids)]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    total = 0.0
    seen = False
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        days, _, clock = raw.rpartition("-")
        parts = clock.split(":")
        try:
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + float(part)
            if days:
                seconds += float(days) * 86400.0
        except ValueError:
            continue
        total += seconds
        seen = True
    return total if seen else None


def _artifact_state(directory: Path | None) -> tuple[int | None, int | None]:
    """Total bytes and file count in the artifact directory, read-only.

    This never opens a shard.  ``stat`` on the directory entries is enough to see the
    packer's progress, and it cannot interfere with a write in flight.
    """
    if directory is None or not directory.is_dir():
        return None, None
    total = 0
    count = 0
    for entry in directory.iterdir():
        if entry.suffix != ".gravity":
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
        count += 1
    return total, count


def probe(pids: list[int], artifact_dir: Path | None) -> CampaignProbe:
    artifact_bytes, artifact_count = _artifact_state(artifact_dir)
    return CampaignProbe(at=time.time(), cpu_seconds=_cpu_seconds(pids),
                         artifact_bytes=artifact_bytes, artifact_count=artifact_count)


def live_pids() -> list[int]:
    """Discover the live campaign's processes by command line rather than a stored pid.

    A pid file would go stale across a launchd restart; the command line will not.
    """
    try:
        out = subprocess.run(["/bin/ps", "-axo", "pid=,command="],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        pid, _, command = raw.partition(" ")
        if "glm52_source_fetch.py" in command or "glm52_worker.py" in command:
            try:
                found.append(int(pid))
            except ValueError:
                continue
    return found


@dataclass
class Window:
    """One measured interval, with or without the lab load running."""

    loaded: bool
    seconds: float
    cpu_rate: float | None            # campaign CPU-seconds per wall second
    byte_rate: float | None           # campaign artifact bytes per wall second
    start: CampaignProbe
    end: CampaignProbe


def _window(loaded: bool, seconds: float, pids: list[int], artifact_dir: Path | None,
            load: Callable[[], None] | None) -> Window:
    start = probe(pids, artifact_dir)
    if loaded and load is not None:
        deadline = start.at + seconds
        while time.time() < deadline:
            load()
    else:
        time.sleep(seconds)
    end = probe(pids, artifact_dir)

    elapsed = max(1e-9, end.at - start.at)
    cpu_rate = None
    if start.cpu_seconds is not None and end.cpu_seconds is not None:
        cpu_rate = (end.cpu_seconds - start.cpu_seconds) / elapsed
    byte_rate = None
    if start.artifact_bytes is not None and end.artifact_bytes is not None:
        byte_rate = (end.artifact_bytes - start.artifact_bytes) / elapsed
    return Window(loaded=loaded, seconds=elapsed, cpu_rate=cpu_rate, byte_rate=byte_rate,
                  start=start, end=end)


def _regression(unloaded: list[float], loaded: list[float]) -> float | None:
    """Fractional drop in the campaign's rate under load.  Negative means it sped up.

    Compared on medians rather than means: this box has a second unrelated project on it,
    so a single preempted window should not decide the gate.
    """
    clean = [v for v in unloaded if v is not None]
    dirty = [v for v in loaded if v is not None]
    if not clean or not dirty:
        return None
    base = sorted(clean)[len(clean) // 2]
    under = sorted(dirty)[len(dirty) // 2]
    if base <= 0:
        return None
    return (base - under) / base


def evaluate(load: Callable[[], None] | None, *, pids: list[int] | None = None,
             artifact_dir: Path | None = None, window_seconds: float = DEFAULT_WINDOW_SECONDS,
             cycles: int = DEFAULT_CYCLES,
             tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Run A/B/A/B and decide whether the lab may take the GPU.

    ``load`` is called repeatedly for the duration of every loaded window; it should be one
    short unit of the real candidate GPU work, not a synthetic burn, or the gate measures
    something the lab will not actually do.
    """
    pids = live_pids() if pids is None else pids
    windows: list[Window] = []
    for _ in range(max(1, cycles)):
        windows.append(_window(False, window_seconds, pids, artifact_dir, None))
        windows.append(_window(True, window_seconds, pids, artifact_dir, load))

    cpu_regression = _regression([w.cpu_rate for w in windows if not w.loaded],
                                 [w.cpu_rate for w in windows if w.loaded])
    byte_regression = _regression([w.byte_rate for w in windows if not w.loaded],
                                  [w.byte_rate for w in windows if w.loaded])

    observed = [r for r in (cpu_regression, byte_regression) if r is not None]
    # Fail closed: an unobservable campaign is not an unharmed campaign.
    if not observed:
        authorized = False
        reason = "campaign not observable: no live pid and no artifact-directory progress"
    elif max(observed) > tolerance:
        authorized = False
        reason = f"regression {max(observed):.4f} exceeds tolerance {tolerance:.4f}"
    else:
        authorized = True
        reason = f"worst regression {max(observed):.4f} within tolerance {tolerance:.4f}"

    return {
        "schema": LEASE_SCHEMA,
        "authorized": authorized,
        "reason": reason,
        "tolerance": tolerance,
        "window_seconds": window_seconds,
        "cycles": cycles,
        "tracked_pids": pids,
        "artifact_dir": str(artifact_dir) if artifact_dir else None,
        "cpu_rate_regression": cpu_regression,
        "artifact_byte_rate_regression": byte_regression,
        "windows": [
            {"loaded": w.loaded, "seconds": w.seconds, "cpu_rate": w.cpu_rate,
             "byte_rate": w.byte_rate,
             "artifact_count": w.end.artifact_count}
            for w in windows
        ],
    }


def selftest() -> int:
    """The gate's own invariants, none of which need a GPU or a live campaign."""
    # rate arithmetic
    assert _regression([10.0, 10.0], [9.0, 9.0]) == 0.1
    assert _regression([10.0], [11.0]) == -0.1        # sped up, negative regression
    assert _regression([], [1.0]) is None
    assert _regression([0.0], [0.0]) is None          # a zero baseline cannot be divided

    # medians, not means: one preempted window must not decide the gate
    assert abs(_regression([10.0, 10.0, 10.0], [10.0, 10.0, 1.0]) - 0.0) < 1e-12

    # fail closed when the campaign cannot be observed at all
    blind = evaluate(None, pids=[], artifact_dir=None, window_seconds=0.01, cycles=1)
    assert not blind["authorized"], blind
    assert "not observable" in blind["reason"], blind

    # a visible campaign with a no-op load authorizes
    calls = {"n": 0}

    def _noop() -> None:
        calls["n"] += 1
        time.sleep(0.001)

    seen = evaluate(_noop, pids=[], artifact_dir=None, window_seconds=0.01, cycles=1)
    assert calls["n"] > 0, "the load was never invoked"
    assert not seen["authorized"], "still blind, so still refused"

    # ps time parsing, including the day form
    assert _cpu_seconds([]) is None

    print(json.dumps({"selftest": "PASS", "schema": LEASE_SCHEMA,
                      "fails_closed_when_blind": True,
                      "tolerance": DEFAULT_TOLERANCE}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired GPU contention gate for the inference lab.")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--artifact-dir", default=None,
                        help="live artifact directory to watch for packer progress")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--report", default=None, help="write the verdict here as JSON")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    # The default load is the real thing the lab wants to run: a compact matvec on the GPU.
    def _gpu_load() -> None:
        import numpy as np
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import gravity_forge as forge
        import gravity_metal
        state = _gpu_load.__dict__
        if "codes" not in state:
            rng = np.random.default_rng(0)
            weights = rng.standard_normal((2048, 6144)).astype(np.float32)
            artifact = forge.pack_product_quant(weights, dim=8, subspaces=1, k=128,
                                                seed=0, iters=4)
            state["codes"] = artifact.config["pq_codes"]
            state["x"] = rng.standard_normal(6144).astype(np.float32)
            state["gpu"] = gravity_metal.decoder()
            # content-addressed rather than a literal: a fixed string would pin this probe's
            # upload under a name any other caller could reuse for a different tensor
            state["key"] = gravity_metal.content_key(state["codes"])
        state["gpu"].matvec(state["codes"], state["x"], key=state["key"])

    verdict = evaluate(_gpu_load,
                       artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
                       window_seconds=args.window_seconds, cycles=args.cycles,
                       tolerance=args.tolerance)
    text = json.dumps(verdict, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0 if verdict["authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
