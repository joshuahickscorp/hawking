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
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any
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
    residents: int = 0          # S010 §5: resident COUNT, not just host memory
    resident_rss_gb: float = 0.0
    expected_gb: float = 0.0    # what the experiment says it will take
    headroom_gb: float = 0.0    # free - expected, the number that decides

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


def _parse_residents(out: str) -> tuple:
    """Count resident bodies and total RSS from `ps` output.

    Split out from the ps call so it can be tested with a fixture. The first
    version asserted only `residents >= 0`, which a function returning a
    constant zero satisfies -- mutation caught it.
    """
    n, rss = 0, 0
    for line in out.splitlines()[1:]:
        if not ("resident-body" in line or "--supervise" in line or "hawkingd" in line):
            continue
        # `ps` output is data, and this process's own matcher appears in it --
        # already walked into once this campaign.
        #
        # The filter must be token-aware. A bare `"awk" in line` matches
        # "hAWKingd", so the substring form excluded the very daemon this
        # function exists to count, and would have under-reported residents on
        # a real host. Caught by the fixture, not by reading it.
        if (" ps -Ao" in line or "/bin/sh -c" in line
                or " grep " in line or line.rstrip().endswith(" grep")
                or " awk " in line or "| awk" in line):
            continue
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1].isdigit():
            n += 1
            rss += int(parts[1])
    return n, rss * 1024 / GB


def _residents() -> tuple:
    """Count resident bodies and their total RSS.

    S010 §5 asks for resident count and RSS before launch, not only host
    totals: two 9.9 GB bodies on a 96 GB host is a different situation from one,
    and the host figure alone cannot tell them apart.
    """
    try:
        out = subprocess.run(["ps", "-Ao", "pid,rss,command"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0, 0.0
    return _parse_residents(out)


# macOS moved the swap store. On Darwin 27 /private/var/vm is EMPTY and the real
# swapfiles live under /System/Volumes/VM. Reading only the old path made
# _swapfiles() return 0 unconditionally, so SWAPFILE_WARN 40 and SWAPFILE_STOP 60
# were unreachable and this guard's swapfile axis was structurally dead -- in the
# guard written BECAUSE the panic hit exactly SWAPFILE_CAP 100.
SWAP_DIRS = ("/System/Volumes/VM", "/private/var/vm")


def _swapfiles() -> int:
    """Count swapfiles across every known store.

    Returns -1, never 0, when no store could be read at all. A count of zero is a
    real and reassuring measurement; an unreadable store is not, and the two must
    not share an encoding -- that equivalence is what kept this axis silent.
    """
    seen, readable = 0, False
    for d in SWAP_DIRS:
        try:
            seen += len([p for p in Path(d).iterdir() if p.name.startswith("swapfile")])
            readable = True
        except Exception:
            continue
    return seen if readable else -1


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
    if swapfiles < 0:
        return "STOP", ("swapfile store unreadable: the swap axis of this guard is "
                        "BLIND, and a blind guard must not report OK",)
    if swapfiles >= SWAPFILE_STOP:
        esc("STOP", f"{swapfiles} swapfiles >= {SWAPFILE_STOP} (cap {SWAPFILE_CAP})")
    elif swapfiles >= SWAPFILE_WARN:
        esc("WARN", f"{swapfiles} swapfiles >= {SWAPFILE_WARN} (cap {SWAPFILE_CAP})")
    return state, tuple(reasons)


def sample(expected_gb: float = 0.0) -> Snapshot:
    v = _vm_stat()
    free_pages = v.get("Pages free", 0) + v.get("Pages speculative", 0)
    comp = v.get("Pages occupied by compressor", 0)
    wired = v.get("Pages wired down", 0)
    sf = _swapfiles()
    free_gb, comp_gb = free_pages * PAGE / GB, comp * PAGE / GB
    n_res, rss_gb = _residents()
    headroom = free_gb - expected_gb
    state, why = classify(free_gb, comp_gb, sf)
    if expected_gb and headroom < FREE_STOP_GB:
        state = "STOP"
        why = why + (f"expected {expected_gb:.1f} GB leaves {headroom:.1f} GB headroom, "
                     f"below {FREE_STOP_GB} GB",)
    elif expected_gb and headroom < FREE_WARN_GB and state == "OK":
        state = "WARN"
        why = why + (f"expected {expected_gb:.1f} GB leaves only {headroom:.1f} GB",)
    return Snapshot(round(free_gb, 2), round(comp_gb, 2), sf,
                    round(wired * PAGE / GB, 2), state, why,
                    residents=n_res, resident_rss_gb=round(rss_gb, 2),
                    expected_gb=expected_gb, headroom_gb=round(headroom, 2))


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
def resource_cost(interval_s: float = 2.0, label: str = ""):
    """Bracket an experiment and yield its measured resource cost.

    G026's metric is progress per wall per RESOURCE per intervention, and the
    resource term was unmeasurable: watch() tracks the WORST STATE seen but
    records no delta and no peak, so nothing could say what a run actually cost.

    Peak is reported as None, never as the entry reading, when the sampler got
    no observations -- a run shorter than one interval has an UNKNOWN peak, and
    quoting its starting value as the peak would understate every fast
    experiment while looking like a measurement.
    """
    start = sample()
    t0 = time.time()
    peak_comp, peak_rss, low_free, n = start.compressor_gb, start.resident_rss_gb, start.free_gb, 0
    stop = threading.Event()

    def loop():
        nonlocal peak_comp, peak_rss, low_free, n
        while not stop.wait(interval_s):
            try:
                s2 = sample()
            except Exception:
                continue
            n += 1
            peak_comp = max(peak_comp, s2.compressor_gb)
            peak_rss = max(peak_rss, s2.resident_rss_gb)
            low_free = min(low_free, s2.free_gb)
    t = threading.Thread(target=loop, daemon=True, name="campaign-resource")
    t.start()
    cost: dict = {"label": label}
    try:
        yield cost
    finally:
        stop.set()
        t.join(timeout=interval_s * 2)
        end = sample()
        cost.update({
            "wall_s": round(time.time() - t0, 2),
            "samples": n,
            "free_gb_start": start.free_gb, "free_gb_end": end.free_gb,
            "free_gb_low": low_free if n else None,
            "compressor_gb_start": start.compressor_gb,
            "compressor_gb_peak": peak_comp if n else None,
            "resident_rss_gb_peak": peak_rss if n else None,
            "swapfiles_start": start.swapfiles, "swapfiles_end": end.swapfiles,
            "swapfiles_delta": end.swapfiles - start.swapfiles,
            "peak_unknown_reason": (None if n else
                f"the sampler observed nothing in {round(time.time() - t0, 2)}s at a "
                f"{interval_s}s interval, so the peak is UNKNOWN; the entry reading is "
                f"not a peak and is not reported as one"),
        })


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


def require_ok(what: str, expected_gb: float = 0.0) -> Snapshot:
    """Call BEFORE an expensive experiment. Raises on STOP.

    `expected_gb` is what the experiment expects to need. Supplying it turns the
    check from "is the host healthy now" into "will it still be healthy after
    this runs", which is the question the 2026-09-06 panic answered the hard way.
    """
    s = sample(expected_gb)
    if s.state == "STOP":
        raise RuntimeError(f"campaign guard STOP before {what}: {'; '.join(s.reasons)}")
    return s


def checkpoint_and_release(what: str, checkpoint: Any = None,
                           release: Any = None) -> dict:
    """S010 §5 on approaching the ceiling: checkpoint, release, VERIFY release.

    Verifying is the part that is easy to skip and the only part that proves
    anything: a release that freed nothing looks identical to one that worked
    unless the freed bytes are read back off the machine.
    """
    before = sample()
    saved = None
    if checkpoint is not None:
        saved = checkpoint()
    released = None
    if release is not None:
        released = release()
    after = sample()
    freed = after.free_gb - before.free_gb
    return {
        "what": what,
        "checkpointed": saved is not None,
        "checkpoint": saved,
        "released": released is not None,
        "free_gb_before": before.free_gb,
        "free_gb_after": after.free_gb,
        "freed_gb": round(freed, 2),
        "release_verified": bool(freed > 0.5),
        "state_before": before.state,
        "state_after": after.state,
    }


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

    # S010 §5: expected experiment memory must be able to STOP a launch that
    # the host would survive but the EXPERIMENT would not.
    s_ok = sample(expected_gb=0.0)
    assert s_ok.state in ("OK", "WARN", "STOP")
    huge = sample(expected_gb=s_ok.free_gb + 50.0)
    assert huge.state == "STOP", f"a 50GB-over-budget experiment was allowed: {huge}"
    assert huge.headroom_gb < 0, huge.headroom_gb
    assert any("headroom" in r or "expected" in r for r in huge.reasons), huge.reasons

    # Resident accounting, against a fixture -- two real bodies plus the
    # matcher line that `ps` shows for the query itself.
    FIX = (
        "  PID    RSS COMMAND\n"
        " 1001 10380000 /path/hawkingd --supervise /w/state.json\n"
        " 1002 10380000 hcli-resident-body --role resident-body\n"
        " 1003     900 /bin/sh -c ps -Ao pid,rss,command | grep resident-body\n"
        " 1005     800 ps -Ao pid,rss,command hawkingd\n"
        " 1006     700 awk /resident-body/ {print}\n"
        " 1004  120000 /usr/bin/python3 -m unrelated.thing\n")
    n, rss = _parse_residents(FIX)
    assert n == 2, f"resident count wrong: {n} (matcher or unrelated line counted?)"
    assert 19.0 < rss < 20.5, f"resident RSS wrong: {rss:.2f} GB"
    assert _parse_residents("  PID    RSS COMMAND\n")[0] == 0
    # each exclusion clause independently: a bare ps, and a bare awk
    assert _parse_residents("H\n 9 1 ps -Ao pid,rss,command hawkingd\n")[0] == 0
    assert _parse_residents("H\n 9 1 awk /resident-body/ {print}\n")[0] == 0
    assert _parse_residents("H\n 9 1 grep resident-body /var/log/x\n")[0] == 0

    # checkpoint_and_release must VERIFY, not assume
    rep = checkpoint_and_release("selftest", checkpoint=lambda: {"mark": 1},
                                 release=lambda: True)
    assert rep["checkpointed"] is True and rep["released"] is True
    assert "release_verified" in rep and "freed_gb" in rep
    assert rep["release_verified"] is (rep["freed_gb"] > 0.5), rep

    print("  selftest: PASS (fires STOP on the real panic state, allows a healthy one,"
          " all 3 signals live, WARN reachable, watcher samples and aborts,"
          " expected-memory STOPs an over-budget launch, release is verified)")


if __name__ == "__main__":
    _selftest()
    s = sample()
    print(f"\n  free {s.free_gb} GB | compressor {s.compressor_gb} GB | "
          f"swapfiles {s.swapfiles} | wired {s.wired_gb} GB")
    print(f"  STATE: {s.state}" + (f"  ({'; '.join(s.reasons)})" if s.reasons else ""))
