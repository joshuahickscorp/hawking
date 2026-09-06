"""Campaign resource guard (S010 §5, §6).

Built from the 2026-09-06 kernel panic, not from intuition. That panic was a
watchdog timeout caused by the VM compressor reaching 100% of its segment limit
with 100 swapfiles, while free memory stood at 924 pages (14.4 MB).

TWO SIGNALS THE PANIC PROVED USELESS, both of which a naive guard would use:

  * `memoryPressure` reads FALSE in the panic record itself, with 14 MB free and
    the compressor at its cap.
  * `sysctl vm.swapusage used` is a boot HIGH-WATER MARK, not a live reading. It
    reports 0.00M after a reboot regardless of what happened before.

A guard built on either would have watched the crash happen and reported OK. So
this reads only what actually moved: free pages, compressor pages, swapfile
count.

States (S010 §6): OK continue; WARN checkpoint + diagnose + release; STOP no new
heavy work until pressure falls and the cause is classified.
"""
from __future__ import annotations

import re
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

PAGE = 16384
GB = 1 << 30

SWAPFILE_CAP = 100          # macOS segment cap; the panic hit exactly this
SWAPFILE_WARN = 40
SWAPFILE_STOP = 60

COMPRESSOR_WARN_GB = 16.0   # S010: 20 GB is the ceiling, warn before it
COMPRESSOR_STOP_GB = 20.0

FREE_WARN_GB = 4.0          # the panic reached 0.014 GB
FREE_STOP_GB = 1.5


@dataclass
class Snapshot:
    free_gb: float
    compressor_gb: float
    swapfiles: int
    wired_gb: float
    state: str
    reasons: tuple

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _vm_stat() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    d = {}
    for line in out.splitlines():
        m = re.match(r'"?([^":]+)"?:\s+(\d+)', line.strip())
        if m:
            d[m.group(1).strip()] = int(m.group(2))
    return d


def _swapfiles() -> int:
    try:
        return len([p for p in Path("/private/var/vm").iterdir()
                    if p.name.startswith("swapfile")])
    except Exception:
        return 0


def classify(free_gb: float, compressor_gb: float, swapfiles: int) -> tuple:
    """Pure, so the panic's recorded numbers can be replayed through it."""
    reasons = []
    state = "OK"

    def esc(new, why):
        nonlocal state
        reasons.append(why)
        if new == "STOP" or state == "STOP":
            state = "STOP"
        else:
            state = new

    if free_gb < FREE_STOP_GB:
        esc("STOP", f"free {free_gb:.2f} GB < {FREE_STOP_GB} GB")
    elif free_gb < FREE_WARN_GB:
        esc("WARN", f"free {free_gb:.2f} GB < {FREE_WARN_GB} GB")
    if compressor_gb >= COMPRESSOR_STOP_GB:
        esc("STOP", f"compressor {compressor_gb:.1f} GB >= {COMPRESSOR_STOP_GB} GB ceiling")
    elif compressor_gb >= COMPRESSOR_WARN_GB:
        esc("WARN", f"compressor {compressor_gb:.1f} GB approaching {COMPRESSOR_STOP_GB} GB")
    if swapfiles >= SWAPFILE_STOP:
        esc("STOP", f"{swapfiles} swapfiles >= {SWAPFILE_STOP} (cap {SWAPFILE_CAP})")
    elif swapfiles >= SWAPFILE_WARN:
        esc("WARN", f"{swapfiles} swapfiles >= {SWAPFILE_WARN} (cap {SWAPFILE_CAP})")
    return state, tuple(reasons)


def sample() -> Snapshot:
    v = _vm_stat()
    free_pages = v.get("Pages free", 0) + v.get("Pages speculative", 0)
    comp = v.get("Pages occupied by compressor", 0)
    wired = v.get("Pages wired down", 0)
    sf = _swapfiles()
    free_gb, comp_gb = free_pages * PAGE / GB, comp * PAGE / GB
    state, why = classify(free_gb, comp_gb, sf)
    return Snapshot(round(free_gb, 2), round(comp_gb, 2), sf,
                    round(wired * PAGE / GB, 2), state, why)


class Abort(RuntimeError):
    """Raised in the worker when the watcher sees STOP mid-experiment."""


@dataclass
class Watch:
    """Live state a long experiment polls between chunks."""
    state: str = "OK"
    reasons: tuple = ()
    samples: int = 0
    worst: str = "OK"

    def check(self) -> None:
        """Call between chunks of expensive work. Raises Abort on STOP.

        S010 §5: the OS must never be the first component to discover Hawking
        exceeded its budget. An admission check alone cannot do that -- the
        2026-09-06 panic developed DURING an experiment whose free RAM was fine
        when it started.
        """
        if self.state == "STOP":
            raise Abort(f"campaign guard STOP mid-experiment: {'; '.join(self.reasons)}")


@contextmanager
def watch(interval_s: float = 5.0):
    """Sample in the background for the duration of the block."""
    w = Watch()
    stop = threading.Event()

    def loop():
        order = {"OK": 0, "WARN": 1, "STOP": 2}
        while not stop.wait(interval_s):
            try:
                s = sample()
            except Exception:
                continue
            w.state, w.reasons, w.samples = s.state, s.reasons, w.samples + 1
            if order[s.state] > order[w.worst]:
                w.worst = s.state
    t = threading.Thread(target=loop, daemon=True, name="campaign-guard")
    t.start()
    try:
        yield w
    finally:
        stop.set()
        t.join(timeout=interval_s + 1.0)


def require_ok(what: str) -> Snapshot:
    """Call BEFORE an expensive experiment. Raises on STOP."""
    s = sample()
    if s.state == "STOP":
        raise RuntimeError(f"campaign guard STOP before {what}: {'; '.join(s.reasons)}")
    return s


def _selftest() -> None:
    # NEGATIVE CONTROL: the real numbers from the 2026-09-06 panic. If the guard
    # does not say STOP on the state that actually crashed the machine, it is
    # not a guard.
    panic_free = 924 * PAGE / GB
    panic_comp = 3_518_434 * PAGE / GB
    st, why = classify(panic_free, panic_comp, 100)
    assert st == "STOP", f"guard did NOT fire on the real panic state: {st} {why}"
    assert len(why) >= 3, f"expected all three signals to fire, got {why}"

    # Positive control: a healthy machine must not be blocked.
    st_ok, _ = classify(60.0, 2.0, 3)
    assert st_ok == "OK", f"guard blocks a healthy machine: {st_ok}"

    # Each signal alone must be able to STOP -- no signal is decorative.
    assert classify(0.5, 1.0, 0)[0] == "STOP", "free-memory signal is dead"
    assert classify(60.0, 25.0, 0)[0] == "STOP", "compressor signal is dead"
    assert classify(60.0, 1.0, 70)[0] == "STOP", "swapfile signal is dead"

    # WARN must be reachable and distinct from STOP.
    assert classify(60.0, 17.0, 0)[0] == "WARN", "WARN unreachable on compressor"
    assert classify(3.0, 1.0, 0)[0] == "WARN", "WARN unreachable on free"
    # the watcher must actually raise when it sees STOP
    w = Watch(state="STOP", reasons=("synthetic",))
    try:
        w.check()
        raise AssertionError("Watch.check did NOT raise on STOP")
    except Abort:
        pass
    Watch(state="WARN", reasons=("x",)).check()   # WARN must NOT abort
    Watch().check()                                # OK must NOT abort

    # The watcher must PROPAGATE state, not merely tick. Mutating the
    # propagation line to only increment `samples` passed an earlier version of
    # this check -- a watcher that samples forever and never reports STOP would
    # have shipped. Inject a STOP-returning sample and require it to arrive.
    import time as _t
    _real_sample = globals()["sample"]
    globals()["sample"] = lambda: Snapshot(0.01, 60.0, 100, 20.0, "STOP", ("injected",))
    try:
        with watch(interval_s=0.05) as live:
            _t.sleep(0.2)
        assert live.samples >= 1, f"watcher never sampled ({live.samples})"
        assert live.state == "STOP", f"watcher did not PROPAGATE state: {live.state!r}"
        assert live.worst == "STOP", f"watcher did not record worst: {live.worst!r}"
        assert live.reasons, "watcher propagated no reasons"
        try:
            live.check()
            raise AssertionError("a propagated STOP did not abort")
        except Abort:
            pass
    finally:
        globals()["sample"] = _real_sample

    print("  selftest: PASS (fires STOP on the real panic state, allows a healthy one,"
          " all 3 signals live, WARN reachable, watcher samples and aborts)")


if __name__ == "__main__":
    _selftest()
    s = sample()
    print(f"\n  free {s.free_gb} GB | compressor {s.compressor_gb} GB | "
          f"swapfiles {s.swapfiles} | wired {s.wired_gb} GB")
    print(f"  STATE: {s.state}" + (f"  ({'; '.join(s.reasons)})" if s.reasons else ""))
