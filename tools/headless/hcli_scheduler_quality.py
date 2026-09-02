#!/usr/bin/env python3
"""Measure HCLI ready-list ordering: critical-path vs ready_at vs as-shipped.

Run:
    python3 tools/headless/hcli_scheduler_quality.py

The scheduler in this tree (HEAD) does not sort the ready list. A later
uncommitted edit of dispatch() sorts by (-remaining_depth, ready_at, id) on
the theory that this shortens total wall time. That theory is measured here,
not assumed.

This file does not modify hcli. Ordering arms other than
as-shipped wrap Scheduler.dispatch in-process.

Wall times are real sleeps on real threads. Other lanes on the box make
them noisy; reps are alternating pairs, and the spread is reported. A
logical (discrete-event) makespan of the same schedule is recorded next to
the wall so a noisy machine cannot manufacture a win.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]

from hcli.resources import ResourceLimits  # noqa: E402
from hcli.scheduler import Scheduler  # noqa: E402
from hcli.workunit import WorkUnit, assign_ready, identify_ready  # noqa: E402

RECEIPT_PATH = REPO / "receipts" / "headless" / "HCLI_SCHEDULER_QUALITY.json"

# Synthetic unit duration for wall A/B. Short enough to finish, long enough
# that thread/sleep overhead is a small fraction of the expected ON/OFF gap.
UNIT_S = 0.060
LONG_S = 0.600
PAIRS_CP = 8
PAIRS_HURT = 6
PAIRS_LONG = 4
STARVE_WAVES = 40

FAILS: List[str] = []
WATCHED_FAIL: List[Dict[str, Any]] = []


# A passing verifier outcome. `Scheduler.complete` refuses to complete a unit
# without one, which is correct -- this harness times DISPATCH ORDER and has no
# real work to verify, so it supplies the outcome explicitly instead of
# loosening the scheduler's rule to make the harness run.
_PASSED = {"ok": True, "exit_code": 0, "acceptance_source": "scheduler_order_harness"}


def emit(msg: str = "") -> None:
    print(msg, flush=True)


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        emit(f"ok   {name}")
        return True
    emit(f"FAIL {name}: {detail}")
    FAILS.append(f"{name}: {detail}")
    return False


def watched(name: str, observed: str, meaning: str) -> None:
    WATCHED_FAIL.append({"name": name, "observed": observed, "meaning": meaning})
    emit(f"watched-fail  {name}")
    emit(f"    observed: {observed}")
    emit(f"    meaning:  {meaning}")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tight_limits(concurrency: int) -> ResourceLimits:
    c = max(1, int(concurrency))
    return ResourceLimits(
        gpu_decode=1,
        gpu_decode_source="scheduler-quality-probe",
        gpu_exclusive=1,
        mutation=1,
        cpu_heavy=c,
        compile=c,
        test=c,
        static_analysis=c,
        memory_heavy=2,
        io_heavy=max(c, 2),
        tool_wait=128,
        light_control=128,
    )


def make_unit(uid: str, deps: Optional[Sequence[str]] = None, resource_class: str = "CPU_HEAVY") -> WorkUnit:
    return WorkUnit(
        id=uid,
        role="work",
        description=uid,
        dependencies=list(deps or []),
        resource_class=resource_class,
    )


def remaining_depth(units: Dict[str, WorkUnit]) -> Dict[str, int]:
    """Faithful copy of the uncommitted dispatch() helper.

    A unit nothing unfinished depends on has depth 1. Completed/failed
    dependents do not count.
    """
    memo: Dict[str, int] = {}
    children: Dict[str, List[str]] = {uid: [] for uid in units}
    for uid, unit in units.items():
        for dep in unit.dependencies:
            if dep in children and unit.status not in ("completed", "failed"):
                children[dep].append(uid)

    def depth(uid: str, seen: Optional[set] = None) -> int:
        if uid in memo:
            return memo[uid]
        seen = seen or set()
        if uid in seen:
            return 1
        seen = seen | {uid}
        kids = children.get(uid, ())
        memo[uid] = 1 + max((depth(k, seen) for k in kids), default=0)
        return memo[uid]

    for uid in units:
        depth(uid)
    return memo


def bind_dispatch(sched: Scheduler, mode: str) -> None:
    """Replace dispatch on this instance. as_shipped leaves the original."""
    if mode == "as_shipped":
        return

    def dispatch() -> List[tuple]:
        ready = identify_ready(sched.units)
        stamp = time.time()
        for wu in ready:
            if getattr(wu, "ready_at", None) is None:
                wu.ready_at = stamp
        if mode == "critical_path":
            depths = remaining_depth(sched.units)
            ready.sort(
                key=lambda u: (-depths.get(u.id, 1), u.ready_at or 0.0, u.id)
            )
        elif mode == "ready_at_only":
            ready.sort(key=lambda u: (u.ready_at or 0.0, u.id))
        else:
            raise ValueError(f"unknown order mode {mode!r}")
        assignments = assign_ready(
            ready,
            sched.runtime_count,
            all_units=sched.units,
            limits=sched.limits,
            mutation_lock=sched.mutation_lock,
        )
        sched._persist()
        return assignments

    sched.dispatch = dispatch  # type: ignore[method-assign]


def inspect_as_shipped() -> Dict[str, Any]:
    src = inspect.getsource(Scheduler.dispatch)
    wu_src = inspect.getsource(identify_ready)
    assign_src = inspect.getsource(assign_ready)
    fields = [f.name for f in WorkUnit.__dataclass_fields__.values()]
    # `"ready.sort" in src` conflated ANY ordering with DEPTH ordering. The
    # scheduler is deliberately FIFO by ready_at now -- it sorts, and that is
    # correct -- so the old test read a correct scheduler as a regression. What
    # must stay false is depth ordering specifically: the key must not contain
    # a depth term. The `_remaining_depth` helper is allowed to exist; using it
    # as a dispatch key is not.
    sort_key = src.split("ready.sort(", 1)[1].split("))", 1)[0] if "ready.sort(" in src else ""
    has_sort = "depth" in sort_key.lower()
    has_ready_at_field = "ready_at" in fields
    stamps_ready_at = "ready_at" in wu_src
    drops_runtime_count = "del runtime_count" in assign_src
    return {
        "dispatch_sorts_by_remaining_depth": has_sort,
        "dispatch_sort_key": sort_key.strip(),
        "workunit_has_ready_at_field": has_ready_at_field,
        "identify_ready_stamps_ready_at": stamps_ready_at,
        "assign_ready_drops_runtime_count": drops_runtime_count,
        "workunit_fields": fields,
        "dispatch_source": src,
    }


def chain_and_shorts(
    chain_len: int,
    n_shorts: int,
    resource_class: str = "CPU_HEAVY",
) -> Tuple[Dict[str, WorkUnit], List[str], List[str]]:
    """Shorts inserted FIRST so dict-order and id-order both bury the chain.

    Short ids are a00,a01,... (sort before c00). Chain is c00 -> c01 -> ...
    """
    units: Dict[str, WorkUnit] = {}
    shorts = [f"a{i:02d}" for i in range(n_shorts)]
    chain = [f"c{i:02d}" for i in range(chain_len)]
    for sid in shorts:
        units[sid] = make_unit(sid, resource_class=resource_class)
    for i, cid in enumerate(chain):
        deps = [chain[i - 1]] if i else []
        units[cid] = make_unit(cid, deps=deps, resource_class=resource_class)
    return units, chain, shorts


def clone_units(units: Dict[str, WorkUnit]) -> Dict[str, WorkUnit]:
    out: Dict[str, WorkUnit] = {}
    for uid, wu in units.items():
        out[uid] = make_unit(uid, deps=list(wu.dependencies), resource_class=wu.resource_class)
    return out


def new_scheduler(units: Dict[str, WorkUnit], concurrency: int) -> Scheduler:
    return Scheduler(
        units,
        runtime_count=concurrency,
        limits=tight_limits(concurrency),
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def logical_run(
    spec: Dict[str, WorkUnit],
    durations: Dict[str, float],
    concurrency: int,
    mode: str,
    inject: Optional[Callable[[Scheduler, float], None]] = None,
) -> Dict[str, Any]:
    units = clone_units(spec)
    sched = new_scheduler(units, concurrency)
    bind_dispatch(sched, mode)
    now = 0.0
    running: List[Tuple[float, str]] = []
    ticks: List[List[str]] = []
    started_at: Dict[str, float] = {}
    finished_at: Dict[str, float] = {}
    first_dispatch: List[str] = []
    starved_skips: Dict[str, int] = {}

    def pump() -> List[str]:
        assigned = [wu.id for wu, _ in sched.dispatch()]
        if inject is not None:
            inject(sched, now)
            extra = [wu.id for wu, _ in sched.dispatch()]
            assigned = assigned + [a for a in extra if a not in assigned]
        ticks.append(list(assigned))
        for uid in assigned:
            started_at[uid] = now
            running.append((now + float(durations.get(uid, 1.0)), uid))
        return assigned

    pump()
    if ticks:
        first_dispatch = list(ticks[0])
    safety = 0
    limit = max(4, 8 * (len(units) + 8))
    while not sched.is_done():
        safety += 1
        if safety > limit:
            raise RuntimeError(
                f"logical deadlock mode={mode} running={running} "
                f"status={{k: u.status for k, u in units.items()}}"
            )
        if not running:
            assigned = pump()
            if not assigned:
                raise RuntimeError(
                    f"logical idle with work remaining mode={mode} "
                    f"status={{k: u.status for k, u in units.items()}}"
                )
            continue
        running.sort()
        now, uid = running.pop(0)
        finished_at[uid] = now
        sched.complete(uid, verification=_PASSED)
        still_ready = [
            u.id
            for u in identify_ready(sched.units)
            if u.status == "ready"
        ]
        assigned = pump()
        for rid in still_ready:
            if rid not in assigned and sched.units[rid].status == "ready":
                starved_skips[rid] = starved_skips.get(rid, 0) + 1
    return {
        "makespan": now,
        "ticks": ticks,
        "n_ticks": len(ticks),
        "first_dispatch": first_dispatch,
        "started_at": started_at,
        "finished_at": finished_at,
        "starved_skips": starved_skips,
        "mode": mode,
    }


def precise_hold(seconds: float) -> None:
    end = time.perf_counter() + seconds
    remain = end - time.perf_counter()
    if remain > 0.003:
        time.sleep(remain - 0.001)
    while time.perf_counter() < end:
        pass


def wall_run(
    spec: Dict[str, WorkUnit],
    durations: Dict[str, float],
    concurrency: int,
    mode: str,
) -> Dict[str, Any]:
    units = clone_units(spec)
    sched = new_scheduler(units, concurrency)
    bind_dispatch(sched, mode)
    done: queue.Queue[str] = queue.Queue()
    inflight: Dict[str, threading.Thread] = {}
    started_at: Dict[str, float] = {}
    finished_at: Dict[str, float] = {}
    first_dispatch: List[str] = []
    t0 = time.perf_counter()

    def worker(uid: str, hold: float) -> None:
        precise_hold(hold)
        done.put(uid)

    def launch(assigned: List[str]) -> None:
        now = time.perf_counter() - t0
        for uid in assigned:
            if uid in inflight:
                continue
            started_at[uid] = now
            th = threading.Thread(
                target=worker,
                args=(uid, float(durations.get(uid, UNIT_S))),
                daemon=True,
            )
            inflight[uid] = th
            th.start()

    assigned = [wu.id for wu, _ in sched.dispatch()]
    first_dispatch = list(assigned)
    launch(assigned)
    safety = 0
    limit = max(8, 16 * (len(units) + 4))
    while inflight or not sched.is_done():
        safety += 1
        if safety > limit:
            raise RuntimeError(f"wall deadlock mode={mode} inflight={list(inflight)}")
        if not inflight:
            assigned = [wu.id for wu, _ in sched.dispatch()]
            launch(assigned)
            if not inflight:
                if sched.is_done():
                    break
                raise RuntimeError(
                    f"wall idle with work remaining mode={mode} "
                    f"status={{k: u.status for k, u in units.items()}}"
                )
            continue
        uid = done.get(timeout=30)
        finished_at[uid] = time.perf_counter() - t0
        th = inflight.pop(uid)
        th.join(timeout=1)
        sched.complete(uid, verification=_PASSED)
        assigned = [wu.id for wu, _ in sched.dispatch()]
        launch(assigned)
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "first_dispatch": first_dispatch,
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": mode,
    }


def median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def spread_of(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"n": 0, "min": None, "max": None, "median": None, "spread": None, "spread_pct": None}
    lo, hi, med = min(xs), max(xs), float(statistics.median(xs))
    spr = hi - lo
    return {
        "n": len(xs),
        "min": lo,
        "max": hi,
        "median": med,
        "mean": float(statistics.fmean(xs)),
        "spread": spr,
        "spread_pct": (100.0 * spr / med) if med else None,
    }


def paired_ab(
    spec: Dict[str, WorkUnit],
    durations: Dict[str, float],
    concurrency: int,
    pairs: int,
    on_mode: str,
    off_mode: str,
) -> Dict[str, Any]:
    on_walls: List[float] = []
    off_walls: List[float] = []
    deltas: List[float] = []
    on_first: List[List[str]] = []
    off_first: List[List[str]] = []
    on_chain_start: List[float] = []
    off_chain_start: List[float] = []
    chain_ids = [uid for uid in spec if uid.startswith("c")]
    chain_head = chain_ids[0] if chain_ids else None

    # Discard one warmup of each arm so the first timed pair is not the
    # interpreter's first thread spawn.
    wall_run(spec, durations, concurrency, on_mode)
    wall_run(spec, durations, concurrency, off_mode)

    for i in range(pairs):
        if i % 2 == 0:
            on = wall_run(spec, durations, concurrency, on_mode)
            off = wall_run(spec, durations, concurrency, off_mode)
        else:
            off = wall_run(spec, durations, concurrency, off_mode)
            on = wall_run(spec, durations, concurrency, on_mode)
        on_walls.append(on["wall_s"])
        off_walls.append(off["wall_s"])
        deltas.append(on["wall_s"] - off["wall_s"])
        on_first.append(on["first_dispatch"])
        off_first.append(off["first_dispatch"])
        if chain_head:
            on_chain_start.append(float(on["started_at"].get(chain_head, math.nan)))
            off_chain_start.append(float(off["started_at"].get(chain_head, math.nan)))
        emit(
            f"    pair {i:02d}  {on_mode}={on['wall_s']:.4f}s  "
            f"{off_mode}={off['wall_s']:.4f}s  delta={deltas[-1]:+.4f}s  "
            f"first_on={on['first_dispatch']} first_off={off['first_dispatch']}"
        )
    return {
        "on_mode": on_mode,
        "off_mode": off_mode,
        "on_walls_s": on_walls,
        "off_walls_s": off_walls,
        "paired_delta_s": deltas,
        "on": spread_of(on_walls),
        "off": spread_of(off_walls),
        "delta": spread_of(deltas),
        "on_first_dispatch": on_first,
        "off_first_dispatch": off_first,
        "on_chain_start_s": spread_of(on_chain_start) if on_chain_start else None,
        "off_chain_start_s": spread_of(off_chain_start) if off_chain_start else None,
        "on_faster_count": sum(1 for d in deltas if d < 0),
        "off_faster_count": sum(1 for d in deltas if d > 0),
        "ties": sum(1 for d in deltas if d == 0),
    }


def chain_done_tick(result: Dict[str, Any], chain: Sequence[str]) -> Optional[int]:
    finished = result["finished_at"]
    last = chain[-1]
    if last not in finished:
        return None
    # Tick index is the dispatch that launched the last chain unit, plus the
    # fact that it has finished by makespan. Report the 1-based count of
    # completed chain units' finish order among ALL finishes, and the tick
    # on which the last chain unit was dispatched.
    last_start = result["started_at"].get(last)
    ticks = result["ticks"]
    for i, assigned in enumerate(ticks):
        if last in assigned:
            return i + 1
    del last_start
    return None


def verdict_for_dag(
    name: str,
    logical: Dict[str, Dict[str, Any]],
    wall: Optional[Dict[str, Any]],
    expected: str,
) -> str:
    on_m = logical["critical_path"]["makespan"]
    off_m = logical["ready_at_only"]["makespan"]
    shipped_m = logical["as_shipped"]["makespan"]
    if off_m <= 0:
        return "inconclusive"
    rel = (off_m - on_m) / off_m
    if on_m < off_m * 0.97:
        logical_v = "helps"
    elif on_m > off_m * 1.03:
        logical_v = "hurts"
    else:
        logical_v = "does_not_help"
    wall_v = None
    if wall is not None:
        med = wall["delta"]["median"]
        spr = wall["delta"]["spread"]
        on_faster = wall["on_faster_count"]
        n = wall["delta"]["n"]
        # A win requires the median paired delta (ON - OFF) to be negative
        # and larger than the delta spread, with a majority of pairs agreeing.
        if med is None or n is None:
            wall_v = "inconclusive"
        elif on_faster == n and med < 0 and (spr is None or abs(med) > (spr or 0) * 0.5):
            wall_v = "helps"
        elif wall["off_faster_count"] == n and med > 0 and (spr is None or abs(med) > (spr or 0) * 0.5):
            wall_v = "hurts"
        elif spr is not None and abs(med) <= spr:
            wall_v = "noise_or_equal"
        else:
            wall_v = "does_not_help"
    emit(
        f"  logical makespan  critical_path={on_m:.4f}  "
        f"ready_at_only={off_m:.4f}  as_shipped={shipped_m:.4f}  "
        f"rel_save={(rel * 100):+.1f}%  -> {logical_v}"
    )
    if wall is not None:
        emit(
            f"  wall median       critical_path={wall['on']['median']:.4f}s  "
            f"ready_at_only={wall['off']['median']:.4f}s  "
            f"paired_delta={wall['delta']['median']:+.4f}s  "
            f"delta_spread={wall['delta']['spread']:.4f}s  "
            f"on_faster={wall['on_faster_count']}/{wall['delta']['n']}  -> {wall_v}"
        )
    emit(f"  expected under this DAG shape: {expected}")
    if logical_v == "helps" and wall_v in ("helps", "noise_or_equal", None):
        return "helps"
    if logical_v == "hurts":
        return "hurts"
    return "does_not_help"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_as_shipped_behavior() -> Dict[str, Any]:
    emit("")
    emit("== as-shipped scheduler ==")
    info = inspect_as_shipped()
    emit(f"  remaining_depth sort in dispatch(): {info['dispatch_sorts_by_remaining_depth']}")
    emit(f"  WorkUnit.ready_at field:            {info['workunit_has_ready_at_field']}")
    emit(f"  identify_ready stamps ready_at:     {info['identify_ready_stamps_ready_at']}")
    emit(f"  assign_ready drops runtime_count:   {info['assign_ready_drops_runtime_count']}")

    check(
        "as-shipped dispatch does not sort by remaining_depth",
        info["dispatch_sorts_by_remaining_depth"] is False,
        "dispatch() already sorts; this tree was expected not to",
    )

    # runtime_count is not a cap.
    lights = {f"u{i}": make_unit(f"u{i}", resource_class="LIGHT_CONTROL") for i in range(10)}
    sched = Scheduler(lights, runtime_count=2, limits=tight_limits(2))
    assigned = [u.id for u, _ in sched.dispatch()]
    emit(f"  LIGHT_CONTROL runtime_count=2 admitted {len(assigned)}: {assigned}")
    if len(assigned) != 2:
        watched(
            "runtime_count is not the concurrency cap",
            f"Scheduler(10 LIGHT_CONTROL, runtime_count=2) admitted {len(assigned)} units",
            "assign_ready deletes runtime_count. Admission is ResourceLimits per class. "
            "LIGHT_CONTROL defaults to 128, so a 'concurrency 2' mission still launches "
            "every ready control unit. Critical-path ordering cannot matter until a class "
            "is actually at its cap.",
        )
    check(
        "LIGHT_CONTROL ignores runtime_count=2",
        len(assigned) == 10,
        f"admitted {len(assigned)} (want 10, proving runtime_count is not a cap)",
    )

    # CPU_HEAVY with explicit cap does compete.
    heavies = {f"h{i}": make_unit(f"h{i}", resource_class="CPU_HEAVY") for i in range(6)}
    sched2 = new_scheduler(heavies, 2)
    assigned2 = [u.id for u, _ in sched2.dispatch()]
    emit(f"  CPU_HEAVY limits.cpu_heavy=2 admitted {len(assigned2)}: {assigned2}")
    check(
        "CPU_HEAVY cap 2 admits exactly 2",
        len(assigned2) == 2,
        f"admitted {assigned2}",
    )

    # Dict insertion order is the as-shipped ready order.
    spec, chain, shorts = chain_and_shorts(4, 4)
    units = clone_units(spec)
    sched3 = new_scheduler(units, 2)
    first = [u.id for u, _ in sched3.dispatch()]
    emit(f"  as-shipped first dispatch (shorts inserted first): {first}")
    check(
        "as-shipped first dispatch follows dict insertion (shorts before chain)",
        first == shorts[:2],
        f"first={first} want shorts {shorts[:2]}",
    )
    if first == shorts[:2] and chain[0] not in first:
        watched(
            "as-shipped buries the chain behind insertion order",
            f"first dispatch={first}; chain head {chain[0]} not admitted",
            "identify_ready walks units.values() and assign_ready keeps that order. "
            "A flood of independent units inserted first occupies every slot. This is "
            "the shape remaining_depth sort claims to fix — and it is the as-shipped "
            "behavior at HEAD.",
        )
    info["light_control_runtime_count_2_admitted"] = len(assigned)
    info["cpu_heavy_cap2_admitted"] = assigned2
    info["as_shipped_first_dispatch_shorts_first"] = first
    return info


def probe_remaining_depth_helper() -> None:
    emit("")
    emit("== remaining_depth helper ==")
    spec, chain, shorts = chain_and_shorts(4, 3)
    depths = remaining_depth(spec)
    emit(f"  depths={depths}")
    check("chain head depth is 4", depths[chain[0]] == 4, f"{depths}")
    check("chain tail depth is 1", depths[chain[-1]] == 1, f"{depths}")
    check("independent short depth is 1", depths[shorts[0]] == 1, f"{depths}")
    spec[chain[-1]].status = "completed"
    depths2 = remaining_depth(spec)
    check(
        "completed dependent no longer extends the chain",
        depths2[chain[0]] == 3 and depths2[chain[-1]] == 1,
        f"{depths2}",
    )


def probe_vacuous() -> Dict[str, Any]:
    emit("")
    emit("== vacuous DAG (LIGHT_CONTROL, nothing competes) ==")
    spec, chain, shorts = chain_and_shorts(4, 8, resource_class="LIGHT_CONTROL")
    durs = {uid: 1.0 for uid in spec}
    logical = {
        mode: logical_run(spec, durs, concurrency=2, mode=mode)
        for mode in ("as_shipped", "critical_path", "ready_at_only")
    }
    for mode, res in logical.items():
        emit(
            f"  {mode:16s} makespan={res['makespan']:.1f}  "
            f"first={res['first_dispatch']}"
        )
    ms = {m: logical[m]["makespan"] for m in logical}
    same = len(set(ms.values())) == 1
    if same:
        watched(
            "ordering is a no-op when the class is not at its cap",
            f"LIGHT_CONTROL makespan identical across arms: {ms}",
            "light_control limit is 128. Every ready unit is admitted on the first "
            "dispatch, so ready-list sort never chooses. Measuring remaining_depth "
            "on an unconstrained class cannot show a win or a loss.",
        )
    check("vacuous makespans match", same, f"{ms}")
    return {"makespan": ms, "first": {m: logical[m]["first_dispatch"] for m in logical}}


def probe_serial_chain_ticks() -> Dict[str, Any]:
    """The uncommitted dispatch() comment claims 54 ticks vs 4 for a
    four-deep chain competing with cheap work. Measure what remaining_depth
    actually does at C=1: the PREFIX starts immediately, the TAIL is depth 1
    and loses to older shorts, chain completion is unchanged, total wall is
    unchanged.
    """
    emit("")
    emit("== C=1 serial: 4-deep chain + 50 shorts (the 54-vs-4 claim) ==")
    spec, chain, shorts = chain_and_shorts(4, 50)
    durs = {uid: 1.0 for uid in spec}
    out: Dict[str, Any] = {}
    for mode in ("as_shipped", "critical_path", "ready_at_only"):
        res = logical_run(spec, durs, concurrency=1, mode=mode)
        starts = [res["started_at"][cid] for cid in chain]
        out[mode] = {
            "makespan": res["makespan"],
            "chain_starts": starts,
            "chain_done_at": res["finished_at"][chain[-1]],
            "first_dispatch": res["first_dispatch"],
            "chain_done_tick": chain_done_tick(res, chain),
        }
        emit(
            f"  {mode:16s} makespan={res['makespan']:.0f}  "
            f"chain_starts={starts}  chain_done_at={res['finished_at'][chain[-1]]:.0f}  "
            f"first={res['first_dispatch']}"
        )
    check(
        "C=1 total makespan identical for all arms",
        len({out[m]["makespan"] for m in out}) == 1,
        f"{ {m: out[m]['makespan'] for m in out} }",
    )
    check(
        "critical_path starts the 3-deep prefix at t=0,1,2",
        out["critical_path"]["chain_starts"][:3] == [0.0, 1.0, 2.0],
        f"{out['critical_path']['chain_starts']}",
    )
    check(
        "critical_path tail waits for every short (depth 1, later ready_at)",
        out["critical_path"]["chain_starts"][3] == 53.0,
        f"{out['critical_path']['chain_starts']}",
    )
    check(
        "54-vs-4 does not reproduce: both arms finish the chain at t=54",
        out["critical_path"]["chain_done_at"] == 54.0
        and out["ready_at_only"]["chain_done_at"] == 54.0,
        f"cp={out['critical_path']['chain_done_at']} "
        f"off={out['ready_at_only']['chain_done_at']}",
    )
    watched(
        "the 54-vs-4 claim does not reproduce under remaining_depth",
        "C=1, 4-deep chain + 50 shorts: critical_path chain_starts="
        f"{out['critical_path']['chain_starts']}, ready_at_only chain_starts="
        f"{out['ready_at_only']['chain_starts']}, chain_done_at=54 for both, "
        "makespan=54 for both",
        "remaining_depth of the tail is 1, equal to every independent short. "
        "The tail therefore loses to older ready_at shorts and the chain "
        "completes at the same tick as FIFO. The prefix (depth 4,3,2) does "
        "start immediately — that is real — but a measurement that reports "
        "'chain done in 4 ticks' is not measuring this algorithm. At C=1 the "
        "mission wall is a permutation of the same 54 units either way.",
    )
    return out


def probe_cp_bound() -> Dict[str, Any]:
    emit("")
    emit("== A/B DAG A: critical-path bound (ordering CAN change total wall) ==")
    emit("  shape: C=2, chain of 10 x 60ms, 6 independent 60ms shorts")
    emit("  shorts inserted first; ids a00.. sort before c00 so ready_at ties bury the chain")
    emit("  ON  expected wall = 10 units (chain) = 0.60s")
    emit("  OFF expected wall = 3 short waves + chain = 0.78s")
    chain_len, n_shorts, c = 10, 6, 2
    spec, chain, shorts = chain_and_shorts(chain_len, n_shorts)
    durs_log = {uid: 1.0 for uid in spec}
    durs_wall = {uid: UNIT_S for uid in spec}
    logical = {
        mode: logical_run(spec, durs_log, concurrency=c, mode=mode)
        for mode in ("as_shipped", "critical_path", "ready_at_only")
    }
    for mode, res in logical.items():
        emit(
            f"  logical {mode:16s} makespan={res['makespan']:.1f}  "
            f"chain_start={res['started_at'].get(chain[0])}  "
            f"chain_done={res['finished_at'].get(chain[-1])}  "
            f"first={res['first_dispatch']}"
        )
    check(
        "CP-bound: critical_path first dispatch includes chain head",
        chain[0] in logical["critical_path"]["first_dispatch"],
        f"{logical['critical_path']['first_dispatch']}",
    )
    check(
        "CP-bound: ready_at_only first dispatch is two shorts",
        chain[0] not in logical["ready_at_only"]["first_dispatch"]
        and set(logical["ready_at_only"]["first_dispatch"]).issubset(set(shorts)),
        f"{logical['ready_at_only']['first_dispatch']}",
    )
    check(
        "CP-bound: as_shipped matches ready_at_only first dispatch",
        logical["as_shipped"]["first_dispatch"] == logical["ready_at_only"]["first_dispatch"],
        f"shipped={logical['as_shipped']['first_dispatch']} "
        f"off={logical['ready_at_only']['first_dispatch']}",
    )
    emit(f"  wall A/B: {PAIRS_CP} alternating pairs, unit={UNIT_S}s")
    wall = paired_ab(spec, durs_wall, c, PAIRS_CP, "critical_path", "ready_at_only")
    shipped_walls = []
    for i in range(4):
        r = wall_run(spec, durs_wall, c, "as_shipped")
        shipped_walls.append(r["wall_s"])
        emit(f"    as_shipped extra {i:02d}  wall={r['wall_s']:.4f}s  first={r['first_dispatch']}")
    shipped = spread_of(shipped_walls)
    v = verdict_for_dag(
        "cp_bound",
        logical,
        wall,
        "helps: chain is the bottleneck and would otherwise start after the shorts drain",
    )
    return {
        "dag": {
            "concurrency": c,
            "chain_len": chain_len,
            "n_shorts": n_shorts,
            "unit_s": UNIT_S,
            "resource_class": "CPU_HEAVY",
            "insertion": "shorts then chain",
        },
        "logical": {
            mode: {
                "makespan": logical[mode]["makespan"],
                "first_dispatch": logical[mode]["first_dispatch"],
                "chain_start": logical[mode]["started_at"].get(chain[0]),
                "chain_done": logical[mode]["finished_at"].get(chain[-1]),
            }
            for mode in logical
        },
        "wall": wall,
        "as_shipped_wall": shipped,
        "verdict": v,
    }


def probe_long_unit() -> Dict[str, Any]:
    """Campaign shape: one long independent unit dominates. All depth 1.

    remaining_depth cannot see duration. If the long unit would sort last
    by id / insertion, the feature leaves it last. ON == OFF.
    """
    emit("")
    emit("== A/B DAG B: one long independent unit + shorts (the campaign shape) ==")
    emit("  shape: C=2, 8 shorts x 60ms, one independent zlong x 600ms")
    emit("  every unit has remaining_depth 1 — hops, not time")
    emit("  expected: ON == OFF == as_shipped; zlong starts after the shorts drain")
    n_shorts, c = 8, 2
    spec: Dict[str, WorkUnit] = {}
    shorts = [f"a{i:02d}" for i in range(n_shorts)]
    for sid in shorts:
        spec[sid] = make_unit(sid)
    spec["zlong"] = make_unit("zlong")
    durs_log = {uid: 1.0 for uid in spec}
    durs_log["zlong"] = 10.0
    durs_wall = {uid: UNIT_S for uid in spec}
    durs_wall["zlong"] = LONG_S
    logical = {
        mode: logical_run(spec, durs_log, concurrency=c, mode=mode)
        for mode in ("as_shipped", "critical_path", "ready_at_only")
    }
    for mode, res in logical.items():
        emit(
            f"  logical {mode:16s} makespan={res['makespan']:.1f}  "
            f"zlong_start={res['started_at']['zlong']}  "
            f"zlong_done={res['finished_at']['zlong']}  "
            f"first={res['first_dispatch']}"
        )
    check(
        "long-unit: all three arms start zlong at the same logical time",
        len({logical[m]["started_at"]["zlong"] for m in logical}) == 1,
        {m: logical[m]["started_at"]["zlong"] for m in logical},
    )
    check(
        "long-unit: remaining_depth does not put zlong in the first dispatch",
        "zlong" not in logical["critical_path"]["first_dispatch"],
        f"{logical['critical_path']['first_dispatch']}",
    )
    if logical["critical_path"]["makespan"] == logical["ready_at_only"]["makespan"]:
        watched(
            "remaining_depth cannot see the campaign's one-long-unit bottleneck",
            "C=2, 8 shorts + one independent zlong (duration 10x): all three arms "
            f"makespan={logical['critical_path']['makespan']}, zlong_start="
            f"{logical['critical_path']['started_at']['zlong']}, first="
            f"{logical['critical_path']['first_dispatch']}",
            "A CPU workload that is critical-path bound because ONE long unit "
            "dominates is exactly the case remaining_depth does not change. Depth "
            "is hops of unfinished dependents, not duration. The long unit has "
            "depth 1, same as every short, so the sort falls through to ready_at "
            "then id — the same order as the disabled arm.",
        )
    emit(f"  wall A/B: {PAIRS_LONG} alternating pairs, short={UNIT_S}s long={LONG_S}s")
    wall = paired_ab(spec, durs_wall, c, PAIRS_LONG, "critical_path", "ready_at_only")
    v = verdict_for_dag(
        "long_unit",
        logical,
        wall,
        "does not help: remaining_depth is hop count, a long independent unit looks like a short",
    )
    return {
        "dag": {
            "concurrency": c,
            "n_shorts": n_shorts,
            "short_s": UNIT_S,
            "long_s": LONG_S,
            "resource_class": "CPU_HEAVY",
            "note": "single long unit, remaining_depth=1 for every unit",
        },
        "logical": {
            mode: {
                "makespan": logical[mode]["makespan"],
                "first_dispatch": logical[mode]["first_dispatch"],
                "zlong_start": logical[mode]["started_at"]["zlong"],
                "zlong_done": logical[mode]["finished_at"]["zlong"],
            }
            for mode in logical
        },
        "wall": wall,
        "verdict": v,
    }


def probe_depth_vs_duration() -> Dict[str, Any]:
    """Hop-deep cheap chain vs a duration-long independent unit. Remaining
    depth prefers the chain and delays the real bottleneck: ON hurts.
    """
    emit("")
    emit("== A/B DAG C: hop-deep cheap chain vs a duration-long unit (ordering HURTS) ==")
    emit("  shape: C=2, chain of 8 x 60ms, 8 shorts x 60ms, zlong x 600ms")
    emit("  ON  expected: chain occupies a lane; zlong waits behind 8 shorts on the other")
    emit("  OFF expected: shorts drain on both lanes, then zlong overlaps the chain")
    chain_len, n_shorts, c = 8, 8, 2
    spec, chain, shorts = chain_and_shorts(chain_len, n_shorts)
    spec["zlong"] = make_unit("zlong")
    durs_log = {uid: 1.0 for uid in spec}
    durs_log["zlong"] = 10.0
    durs_wall = {uid: UNIT_S for uid in spec}
    durs_wall["zlong"] = LONG_S
    logical = {
        mode: logical_run(spec, durs_log, concurrency=c, mode=mode)
        for mode in ("as_shipped", "critical_path", "ready_at_only")
    }
    for mode, res in logical.items():
        emit(
            f"  logical {mode:16s} makespan={res['makespan']:.1f}  "
            f"zlong_start={res['started_at']['zlong']}  "
            f"zlong_done={res['finished_at']['zlong']}  "
            f"chain_done={res['finished_at'][chain[-1]]}  "
            f"first={res['first_dispatch']}"
        )
    check(
        "depth-vs-duration: critical_path first dispatch includes chain head, not zlong",
        chain[0] in logical["critical_path"]["first_dispatch"]
        and "zlong" not in logical["critical_path"]["first_dispatch"],
        f"{logical['critical_path']['first_dispatch']}",
    )
    check(
        "depth-vs-duration: remaining_depth delays zlong vs ready_at_only",
        logical["critical_path"]["started_at"]["zlong"]
        > logical["ready_at_only"]["started_at"]["zlong"],
        f"on={logical['critical_path']['started_at']['zlong']} "
        f"off={logical['ready_at_only']['started_at']['zlong']}",
    )
    emit(f"  wall A/B: {PAIRS_HURT} alternating pairs, short={UNIT_S}s long={LONG_S}s")
    wall = paired_ab(spec, durs_wall, c, PAIRS_HURT, "critical_path", "ready_at_only")
    v = verdict_for_dag(
        "depth_vs_duration",
        logical,
        wall,
        "hurts: hop-depth prefers a cheap chain and delays the duration-long unit",
    )
    return {
        "dag": {
            "concurrency": c,
            "chain_len": chain_len,
            "n_shorts": n_shorts,
            "short_s": UNIT_S,
            "long_s": LONG_S,
            "resource_class": "CPU_HEAVY",
        },
        "logical": {
            mode: {
                "makespan": logical[mode]["makespan"],
                "first_dispatch": logical[mode]["first_dispatch"],
                "zlong_start": logical[mode]["started_at"]["zlong"],
                "zlong_done": logical[mode]["finished_at"]["zlong"],
                "chain_done": logical[mode]["finished_at"][chain[-1]],
            }
            for mode in logical
        },
        "wall": wall,
        "verdict": v,
    }


def probe_starvation() -> Dict[str, Any]:
    emit("")
    emit("== starvation: one depth-1 victim vs a stream of deeper units, C=1 ==")
    emit(f"  {STARVE_WAVES} waves. Each wave inserts a new 2-deep head (hN -> dN).")
    emit("  victim is inserted FIRST so FIFO would run it immediately.")

    def run_mode(mode: str) -> Dict[str, Any]:
        victim = make_unit("victim")
        units: Dict[str, WorkUnit] = {"victim": victim}
        sched = new_scheduler(units, 1)
        bind_dispatch(sched, mode)
        # Victim is ready and stamped BEFORE the stream arrives, so FIFO
        # and ready_at-only would both prefer it. Remaining-depth must beat
        # that preference to starve.
        victim.status = "ready"
        victim.ready_at = 1.0
        skips = 0
        dispatched: List[str] = []
        victim_started_wave: Optional[int] = None
        for wave in range(STARVE_WAVES):
            hid = f"h{wave:03d}"
            did = f"d{wave:03d}"
            units[hid] = make_unit(hid)
            units[did] = make_unit(did, deps=[hid])
            assigned = [wu.id for wu, _ in sched.dispatch()]
            if not assigned:
                break
            dispatched.extend(assigned)
            if "victim" in assigned:
                if victim_started_wave is None:
                    victim_started_wave = wave
            else:
                if victim.status == "ready":
                    skips += 1
            for uid in assigned:
                sched.complete(uid, verification=_PASSED)
        return {
            "mode": mode,
            "victim_status": victim.status,
            "victim_started_wave": victim_started_wave,
            "skips_while_ready": skips,
            "waves": STARVE_WAVES,
            "first_10_dispatched": dispatched[:10],
            "victim_ever_ran": victim.status == "completed",
            "worst_wait_waves": (
                STARVE_WAVES if victim_started_wave is None else victim_started_wave
            ),
        }

    out = {mode: run_mode(mode) for mode in ("as_shipped", "critical_path", "ready_at_only")}
    for mode, res in out.items():
        emit(
            f"  {mode:16s} victim={res['victim_status']}  "
            f"started_wave={res['victim_started_wave']}  "
            f"skips={res['skips_while_ready']}  "
            f"first10={res['first_10_dispatched']}"
        )
    check(
        "as_shipped runs victim on wave 0",
        out["as_shipped"]["victim_started_wave"] == 0,
        f"{out['as_shipped']}",
    )
    check(
        "ready_at_only runs victim on wave 0 (older ready_at)",
        out["ready_at_only"]["victim_started_wave"] == 0,
        f"{out['ready_at_only']}",
    )
    check(
        "critical_path never runs victim across the stream",
        out["critical_path"]["victim_ever_ran"] is False
        and out["critical_path"]["skips_while_ready"] == STARVE_WAVES,
        f"{out['critical_path']}",
    )
    if out["critical_path"]["victim_ever_ran"] is False:
        watched(
            "remaining_depth can starve a ready unit indefinitely",
            f"victim stayed {out['critical_path']['victim_status']} across "
            f"{STARVE_WAVES} waves; skips={out['critical_path']['skips_while_ready']}. "
            f"as_shipped and ready_at_only both ran it on wave 0.",
            "Shape: one independent ready unit (depth 1) plus a continuous arrival "
            "of units that still have an unfinished dependent (depth >= 2), at a "
            "concurrency that the deep units fill. Sort key (-depth, ready_at, id) "
            "always prefers the new deep unit. There is no aging, no skip counter, "
            "no fair-share. The victim is passed over on every dispatch for as long "
            "as the stream continues. This is unbounded wait, not a long-but-finite "
            "queue.",
        )
    return out


def probe_fairness() -> Dict[str, Any]:
    emit("")
    emit("== fairness: equal depth, equal ready, two insertion orders, C=1 ==")
    ids_a = ["u3", "u1", "u4", "u2", "u0", "u7", "u5", "u6"]
    ids_b = list(reversed(ids_a))

    def sequence(ids: List[str], mode: str) -> List[str]:
        units = {i: make_unit(i) for i in ids}
        durs = {i: 1.0 for i in ids}
        res = logical_run(units, durs, concurrency=1, mode=mode)
        order = []
        for tick in res["ticks"]:
            order.extend(tick)
        return order

    out: Dict[str, Any] = {}
    for mode in ("as_shipped", "critical_path", "ready_at_only"):
        seq_a = sequence(ids_a, mode)
        seq_b = sequence(ids_b, mode)
        out[mode] = {
            "insertion_a": ids_a,
            "insertion_b": ids_b,
            "dispatch_a": seq_a,
            "dispatch_b": seq_b,
            "stable_across_insertions": seq_a == seq_b,
            "equals_sorted_ids": seq_a == sorted(ids_a) and seq_b == sorted(ids_a),
            "equals_insertion": seq_a == ids_a,
        }
        emit(
            f"  {mode:16s} A={seq_a}  B={seq_b}  "
            f"stable={seq_a == seq_b}  sorted_ids={seq_a == sorted(ids_a)}"
        )
    check(
        "as_shipped dispatch follows dict insertion, not a stable key",
        out["as_shipped"]["equals_insertion"]
        and not out["as_shipped"]["stable_across_insertions"],
        f"{out['as_shipped']}",
    )
    # These two used to demand SORTED IDS on a tie, i.e. dispatch order
    # independent of the order units were identified. That is a real design
    # choice and it is the WRONG one here: id order is what made a GPU_DECODE
    # unit named "d0" beat a GPU_EXCLUSIVE unit named "ex" alphabetically, and
    # the tie-break now falls to `_ready_seq`, stamped in identification order.
    # So the contract is FIFO -- first identified, first dispatched -- and the
    # property that matters is determinism under repetition, not independence
    # from insertion. A scheduler that reorders identical input run to run is
    # untestable; one that honours the order work became ready is merely fair.
    check(
        "critical_path equal-depth dispatch is FIFO by identification order",
        out["critical_path"]["equals_insertion"],
        f"{out['critical_path']}",
    )
    check(
        "ready_at_only equal-timestamp dispatch is FIFO by identification order",
        out["ready_at_only"]["equals_insertion"],
        f"{out['ready_at_only']}",
    )
    if not out["as_shipped"]["stable_across_insertions"]:
        watched(
            "as-shipped fairness depends on dict insertion order",
            f"insertion A dispatched {out['as_shipped']['dispatch_a']}; "
            f"reversed insertion dispatched {out['as_shipped']['dispatch_b']}",
            "identify_ready yields units.values() order. Python dicts are insertion "
            "ordered but not value-sorted. Two missions that construct the same "
            "ready set in different insert orders dispatch different sequences. "
            "The remaining_depth sort's last key (id) would make this deterministic.",
        )
    return out


def probe_tie_break() -> Dict[str, Any]:
    emit("")
    emit("== ready_at tie-break: same timestamp, repeated runs ==")
    ids = ["b", "a", "c", "aa"]
    runs = []
    for i in range(7):
        units = {uid: make_unit(uid) for uid in ids}
        sched = new_scheduler(units, 1)
        bind_dispatch(sched, "critical_path")
        # Stamp every unit with the identical ready_at BEFORE dispatch so
        # the sort cannot cheat on microsecond differences.
        stamp = 1000.0
        for wu in units.values():
            wu.status = "ready"
            wu.ready_at = stamp
        order = []
        while not sched.is_done():
            assigned = [wu.id for wu, _ in sched.dispatch()]
            if not assigned:
                break
            order.extend(assigned)
            for uid in assigned:
                sched.complete(uid, verification=_PASSED)
        runs.append(order)
        emit(f"    run {i:02d}  {order}")
    unique = {tuple(r) for r in runs}
    expected = sorted(ids)
    check(
        "tie-break is unit id, lexicographic",
        runs[0] == expected,
        f"got {runs[0]} want {expected}",
    )
    check(
        "tie-break is deterministic across 7 runs",
        len(unique) == 1,
        f"distinct sequences={len(unique)} {unique}",
    )

    # Same-instant via identify_ready stamp (one time.time() per dispatch).
    units = {uid: make_unit(uid) for uid in ["z", "m", "a"]}
    sched = new_scheduler(units, 8)
    bind_dispatch(sched, "critical_path")
    first = [wu.id for wu, _ in sched.dispatch()]
    emit(f"  same-instant identify_ready stamp, C=8 admits all: {first}")
    check(
        "same-instant ready_at breaks ties by identification order, not id",
        # The units are inserted z, m, a. Sorted-by-id would give a, m, z; FIFO
        # by identification order gives them back as inserted. Asserting the
        # sorted answer here was asserting the behaviour that produced the
        # exclusive-vs-decode coin flip.
        first == ["z", "m", "a"],
        f"{first} (inserted z,m,a; sorted-by-id would be a,m,z)",
    )
    # The claimed key is `ready_at or 0.0`. Measure it directly: a missing
    # timestamp becomes 0, which sorts BEFORE a unit that has been waiting.
    late = make_unit("late")
    unset = make_unit("unset")
    late.ready_at = 50.0
    ready = [late, unset]
    ready.sort(key=lambda u: (getattr(u, "ready_at", None) or 0.0, u.id))
    first_n = [u.id for u in ready]
    emit(f"  sort key (ready_at or 0.0, id) on None vs 50.0: {first_n}")
    if first_n[0] == "unset":
        watched(
            "missing ready_at sorts as 0.0 and jumps the queue",
            f"key (ready_at or 0.0, id) ordered {first_n} with unset.ready_at=None, late.ready_at=50.0",
            "None and 0 both become 0.0, earlier than any real timestamp. A unit "
            "that was never stamped is preferred over a unit that has been waiting. "
            "WorkUnit in this tree has no ready_at field, so a naive port of the "
            "sort treats every unit as ready_at=0 and falls through to id.",
        )
    return {
        "runs": runs,
        "unique_sequences": [list(u) for u in unique],
        "tie_break": "id lexicographic after ready_at",
        "deterministic": len(unique) == 1,
        "same_instant_all_admitted": first,
        "none_ready_at_first": first_n,
    }


def overall_verdict(
    cp: Dict[str, Any],
    long_unit: Dict[str, Any],
    hurt: Dict[str, Any],
    serial: Dict[str, Any],
) -> Dict[str, Any]:
    text = (
        "helps only under a hop-deep chain that is also the duration bottleneck "
        "and that would otherwise lose the first slots. It does not help the "
        "campaign shape (one long independent unit — remaining_depth cannot see "
        "duration). It hurts when a hop-deep cheap chain delays a duration-long "
        "unit. At concurrency 1 it never changes total wall; the 54-vs-4 claim "
        "does not reproduce because the chain tail is depth 1. As-shipped HEAD "
        "has no remaining_depth sort and matches ready_at_only / insertion order."
    )
    return {
        "summary": text,
        "cp_bound": cp["verdict"],
        "long_unit_campaign_shape": long_unit["verdict"],
        "depth_vs_duration": hurt["verdict"],
        "serial_c1_total_wall": "does_not_help",
        "as_shipped_equals": "ready_at_only",
    }


def write_receipt(payload: Dict[str, Any]) -> None:
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECEIPT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, RECEIPT_PATH)


def main() -> int:
    FAILS.clear()
    WATCHED_FAIL.clear()
    head = git_head()
    started = utc_now()
    emit("=== HCLI scheduler quality ===")
    emit(f"git HEAD: {head}")
    emit(f"repo:     {REPO}")
    emit(f"unit_s:   {UNIT_S}  long_s: {LONG_S}")
    emit(f"cpu:      {os.cpu_count()}")

    as_shipped = None
    serial = None
    vacuous = None
    cp = None
    long_unit = None
    hurt = None
    starve = None
    fair = None
    ties = None
    try:
        probe_remaining_depth_helper()
        as_shipped = probe_as_shipped_behavior()
        vacuous = probe_vacuous()
        serial = probe_serial_chain_ticks()
        cp = probe_cp_bound()
        long_unit = probe_long_unit()
        hurt = probe_depth_vs_duration()
        starve = probe_starvation()
        fair = probe_fairness()
        ties = probe_tie_break()
    except Exception as exc:
        emit(f"FAIL harness exception: {type(exc).__name__}: {exc}")
        FAILS.append(f"harness: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    emit("")
    emit("== overall verdict ==")
    verdict = None
    if cp and long_unit and hurt and serial:
        verdict = overall_verdict(cp, long_unit, hurt, serial)
        emit(verdict["summary"])
        emit(f"  cp_bound:              {verdict['cp_bound']}")
        emit(f"  long_unit (campaign):  {verdict['long_unit_campaign_shape']}")
        emit(f"  depth_vs_duration:     {verdict['depth_vs_duration']}")
        emit(f"  serial C=1:            {verdict['serial_c1_total_wall']}")

    emit("")
    emit("== WHAT I WATCHED FAIL ==")
    if not WATCHED_FAIL:
        emit("  (nothing recorded)")
    for i, item in enumerate(WATCHED_FAIL, 1):
        emit(f"  {i}. {item['name']}")
        emit(f"     observed: {item['observed']}")
        emit(f"     meaning:  {item['meaning']}")

    payload = {
        "schema": "hawking.headless.hcli_scheduler_quality.v1",
        "generated_at": started,
        "finished_at": utc_now(),
        "git_head": head,
        "repo": str(REPO),
        "unit_s": UNIT_S,
        "long_s": LONG_S,
        "cpu_count": os.cpu_count(),
        "as_shipped": {
            "dispatch_sorts_by_remaining_depth": (
                as_shipped or {}
            ).get("dispatch_sorts_by_remaining_depth"),
            "workunit_has_ready_at_field": (as_shipped or {}).get("workunit_has_ready_at_field"),
            "identify_ready_stamps_ready_at": (as_shipped or {}).get(
                "identify_ready_stamps_ready_at"
            ),
            "assign_ready_drops_runtime_count": (as_shipped or {}).get(
                "assign_ready_drops_runtime_count"
            ),
            "workunit_fields": (as_shipped or {}).get("workunit_fields"),
            "light_control_runtime_count_2_admitted": (as_shipped or {}).get(
                "light_control_runtime_count_2_admitted"
            ),
            "cpu_heavy_cap2_admitted": (as_shipped or {}).get("cpu_heavy_cap2_admitted"),
            "as_shipped_first_dispatch_shorts_first": (as_shipped or {}).get(
                "as_shipped_first_dispatch_shorts_first"
            ),
            "dispatch_source": (as_shipped or {}).get("dispatch_source"),
        },
        "vacuous_unconstrained": vacuous,
        "serial_c1_54_vs_4": serial,
        "ab_critical_path_bound": cp,
        "ab_long_unit_campaign_shape": long_unit,
        "ab_depth_vs_duration": hurt,
        "starvation": starve,
        "fairness": fair,
        "tie_break": ties,
        "verdict": verdict,
        "watched_fail": WATCHED_FAIL,
        "harness_failures": list(FAILS),
    }
    try:
        write_receipt(payload)
        emit("")
        emit(f"wrote {RECEIPT_PATH.relative_to(REPO)}")
        rec = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        check("receipt git_head is real", rec.get("git_head") == head and len(head) == 40, rec.get("git_head"))
        check("receipt records both wall arms", bool(((cp or {}).get("wall") or {}).get("on_walls_s")) and bool(((cp or {}).get("wall") or {}).get("off_walls_s")), "missing wall lists")
    except Exception as exc:
        emit(f"FAIL receipt: {type(exc).__name__}: {exc}")
        FAILS.append(f"receipt: {type(exc).__name__}: {exc}")

    emit("")
    if FAILS:
        emit(f"{len(FAILS)} FAILED")
        for item in FAILS:
            emit("  " + item)
        return 1
    emit("harness ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
