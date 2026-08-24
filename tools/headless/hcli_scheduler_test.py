#!/usr/bin/env python3
"""Protected HCLI scheduler checks. Plain python3 + assert. No GPU, no model.

Run:
    python3 tools/headless/hcli_scheduler_test.py
    pytest tools/headless/hcli_scheduler_test.py -q

Every check below was watched FAILING against the pre-change skeleton.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Scheduler.complete now refuses to complete a WorkUnit without a passing
# deterministic verifier outcome. These checks are about scheduling, resource
# classes and durability -- not about completing unverified work -- so they
# supply a passing outcome rather than dropping the new gate.
PASSING_VERIFICATION = {"ok": True, "verifier": "headless-test-fixture"}

REPO = Path(__file__).resolve().parents[2]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


def make_unit(uid: str, resource_class: str, deps=None, role: str = "work"):
    from hcli.workunit import WorkUnit

    kwargs = dict(
        id=uid,
        role=role,
        description=uid,
        dependencies=list(deps or []),
    )
    try:
        return WorkUnit(**kwargs, resource_class=resource_class)
    except TypeError:
        wu = WorkUnit(**kwargs)
        wu.resource_class = resource_class
        return wu


def make_scheduler(units, runtime_count, workspace=None, **extra):
    from hcli.scheduler import Scheduler

    try:
        return Scheduler(
            units, runtime_count, workspace=workspace, **extra
        )
    except TypeError:
        return Scheduler(units, runtime_count)


def running_ids(units):
    return sorted(u.id for u in units.values() if u.status == "running")


def _env_decode_limit(value: str):
    saved = os.environ.get("ACTIVE_DECODE_LIMIT")
    os.environ["ACTIVE_DECODE_LIMIT"] = value
    return saved


def _restore_decode_limit(saved):
    if saved is None:
        os.environ.pop("ACTIVE_DECODE_LIMIT", None)
    else:
        os.environ["ACTIVE_DECODE_LIMIT"] = saved


def check_decode_limit():
    saved = _env_decode_limit("2")
    try:
        units = {
            f"g{i}": make_unit(f"g{i}", "GPU_DECODE") for i in range(6)
        }
        with tempfile.TemporaryDirectory() as tmp:
            sched = make_scheduler(units, runtime_count=6, workspace=tmp)
            observed_max = 0
            sched.dispatch()
            observed_max = max(observed_max, len(running_ids(units)))
            for _ in range(6):
                live = [u for u in units.values() if u.status == "running"]
                if not live:
                    break
                sched.complete(live[0].id, verification=PASSING_VERIFICATION)
                sched.dispatch()
                observed_max = max(observed_max, len(running_ids(units)))
            check(
                "decode limit is respected",
                observed_max <= 2 and observed_max >= 1,
                f"observed max running GPU_DECODE units = {observed_max}, ACTIVE_DECODE_LIMIT=2",
            )
    finally:
        _restore_decode_limit(saved)


def check_cpu_not_blocked_by_gpu():
    saved = _env_decode_limit("2")
    try:
        units = {
            "g0": make_unit("g0", "GPU_DECODE"),
            "g1": make_unit("g1", "GPU_DECODE"),
            "c0": make_unit("c0", "COMPILE", role="compile"),
            "t0": make_unit("t0", "TEST", role="test"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            # runtime_count matches the saturated decode cap so a naive
            # assign_ready(ready, runtime_count) would refuse the CPU units.
            sched = make_scheduler(units, runtime_count=2, workspace=tmp)
            sched.dispatch()
            gpu_running = {
                uid for uid in ("g0", "g1") if units[uid].status == "running"
            }
            cpu_running = [
                uid
                for uid in ("c0", "t0")
                if units[uid].status == "running"
            ]
            statuses = {k: v.status for k, v in units.items()}
            check(
                "CPU work is not blocked by a saturated GPU",
                len(gpu_running) == 2 and len(cpu_running) >= 1,
                f"gpu_running={sorted(gpu_running)} cpu_running={cpu_running} "
                f"statuses={statuses}",
            )
    finally:
        _restore_decode_limit(saved)


def check_gpu_exclusive_excludes():
    saved = _env_decode_limit("2")
    try:
        exclusive = make_unit("ex", "GPU_EXCLUSIVE")
        decode = make_unit("d0", "GPU_DECODE")
        units = {"ex": exclusive, "d0": decode}
        with tempfile.TemporaryDirectory() as tmp:
            sched = make_scheduler(units, runtime_count=4, workspace=tmp)
            sched.dispatch()
            # If decode slipped in first, complete it and dispatch exclusive alone.
            if exclusive.status != "running" and decode.status == "running":
                sched.complete("d0", verification=PASSING_VERIFICATION)
                # put decode back as ready so we can observe exclusion
                decode.status = "pending"
                decode.assigned_runtime = None
                sched.dispatch()
            check(
                "GPU_EXCLUSIVE is running before we test exclusion",
                exclusive.status == "running",
                f"ex={exclusive.status} d0={decode.status}",
            )
            # decode must not be admitted while exclusive holds the GPU
            if decode.status != "running":
                sched.dispatch()
            check(
                "GPU_EXCLUSIVE excludes GPU_DECODE",
                exclusive.status == "running" and decode.status != "running",
                f"ex={exclusive.status} d0={decode.status}",
            )
            sched.complete("ex", verification=PASSING_VERIFICATION)
            sched.dispatch()
            check(
                "GPU_DECODE resumes after GPU_EXCLUSIVE completes",
                decode.status == "running",
                f"d0={decode.status} after exclusive complete",
            )
    finally:
        _restore_decode_limit(saved)


def check_mutation_single_writer():
    units = {
        "m1": make_unit("m1", "MUTATION", role="mutation"),
        "m2": make_unit("m2", "MUTATION", role="mutation"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        sched = make_scheduler(units, runtime_count=4, workspace=tmp)
        sched.dispatch()
        running = running_ids(units)
        check(
            "mutation is single-writer",
            running == ["m1"] or running == ["m2"],
            f"running={running} (want exactly one MUTATION)",
        )
        holder = running[0] if running else None
        if holder:
            sched.complete(holder, verification=PASSING_VERIFICATION)
            sched.dispatch()
            other = "m2" if holder == "m1" else "m1"
            check(
                "second MUTATION runs after the first completes",
                units[other].status == "running",
                f"after {holder} completed, {other}={units[other].status}",
            )
        else:
            check(
                "second MUTATION runs after the first completes",
                False,
                "no first MUTATION was running",
            )


def check_crash_safe_mutation_lock():
    from hcli.resources import MutationLock, process_start_token

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        lock_dir = ws / ".hcli"
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / "mutation.lock"

        lock_path.write_text(
            json.dumps(
                {
                    "pid": 99999999,
                    "start_time": "ghost-process-not-running",
                    "acquired_at": 0,
                    "unit_id": "ghost",
                }
            ),
            encoding="utf-8",
        )
        dead_units = {"m_dead": make_unit("m_dead", "MUTATION")}
        s_dead = make_scheduler(dead_units, runtime_count=1, workspace=ws)
        s_dead.dispatch()
        check(
            "crash-safe lock: dead pid is broken",
            dead_units["m_dead"].status == "running",
            f"status={dead_units['m_dead'].status}; lock was pid=99999999",
        )
        if dead_units["m_dead"].status == "running":
            s_dead.complete("m_dead", verification=PASSING_VERIFICATION)

        live_pid = os.getpid()
        token = process_start_token(live_pid)
        lock_path.write_text(
            json.dumps(
                {
                    "pid": live_pid,
                    "start_time": token,
                    "acquired_at": time.time(),
                    "unit_id": "foreign-live",
                }
            ),
            encoding="utf-8",
        )
        # A later scheduler must not steal a live holder's lock.
        live_units = {"m_live": make_unit("m_live", "MUTATION")}
        s_live = make_scheduler(live_units, runtime_count=1, workspace=ws)
        broke = MutationLock(ws).try_break_stale()
        s_live.dispatch()
        check(
            "crash-safe lock: live pid is NOT broken",
            broke is False and live_units["m_live"].status != "running",
            f"try_break_stale={broke} status={live_units['m_live'].status} "
            f"pid={live_pid}",
        )


def check_no_barrier():
    units = {
        "A1": make_unit("A1", "LIGHT_CONTROL"),
        "A2": make_unit("A2", "LIGHT_CONTROL", deps=["A1"]),
        "B1": make_unit("B1", "LIGHT_CONTROL"),
        "B2": make_unit("B2", "LIGHT_CONTROL", deps=["B1"]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        sched = make_scheduler(units, runtime_count=4, workspace=tmp)
        first = sched.dispatch()
        check(
            "no-barrier setup: A1 and B1 both running",
            units["A1"].status == "running" and units["B1"].status == "running",
            f"first_assignments={[(u.id, idx) for u, idx in first]} "
            f"statuses={{k: v.status for k, v in units.items()}}",
        )
        sched.complete("A1", verification=PASSING_VERIFICATION)
        t0 = time.monotonic()
        second = sched.dispatch()
        elapsed = time.monotonic() - t0
        assigned = {u.id for u, _ in second}
        check(
            "no barrier: A2 dispatches immediately without waiting for B1",
            units["A2"].status == "running"
            and units["B1"].status == "running"
            and units["B2"].status != "running"
            and "A2" in assigned
            and elapsed < 0.5,
            f"A2={units['A2'].status} B1={units['B1'].status} "
            f"B2={units['B2'].status} assigned={sorted(assigned)} elapsed={elapsed:.4f}s",
        )


def check_durable_restart():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {
            "a": make_unit("a", "LIGHT_CONTROL"),
            "b": make_unit("b", "LIGHT_CONTROL", deps=["a"]),
            "c": make_unit("c", "LIGHT_CONTROL"),
        }
        sched = make_scheduler(units, runtime_count=2, workspace=ws)
        sched.dispatch()
        # Leave one unit running so a restart has to recover it.
        running_before = running_ids(units)
        if "a" in running_before:
            sched.complete("a", verification=PASSING_VERIFICATION)
        left_running = running_ids(units)
        check(
            "durable restart setup left a running unit",
            len(left_running) >= 1,
            f"running={left_running}",
        )
        crashed_id = left_running[0] if left_running else None
        dag_path = ws / ".hcli" / "dag.json"
        check(
            "durable restart wrote dag.json before crash",
            dag_path.is_file(),
            f"missing {dag_path}",
        )
        del sched

        from hcli.scheduler import Scheduler

        restarted = Scheduler.from_workspace(ws, runtime_count=2)
        check(
            "durable restart: completed status survived",
            restarted.units["a"].status == "completed",
            f"a={restarted.units['a'].status}",
        )
        recovered = restarted.units[crashed_id] if crashed_id else None
        # `interrupted` is a first-class status, not a synonym for failed: a
        # crash is not a verifier failure, so it must NOT consume the retry
        # budget and must NOT grow the repair tree (workunit.is_ready and
        # make_repair_unit both special-case it). This assertion used to demand
        # "failed" or "ready" with attempts>=1, which contradicted that design.
        # What actually matters is that the unit is not completed and is still
        # dispatchable, so check dispatchability rather than the label.
        from hcli.workunit import is_ready

        check(
            "durable restart: running unit did NOT come back completed",
            recovered is not None and recovered.status != "completed",
            f"crashed_id={crashed_id} recovered="
            f"{None if recovered is None else (recovered.status, recovered.attempts)}",
        )
        check(
            "durable restart: the interrupted unit is still dispatchable",
            recovered is not None and is_ready(recovered, restarted.units),
            f"status={None if recovered is None else recovered.status} "
            f"not re-dispatchable after crash recovery",
        )
        # `attempts` counts DISPATCHES and is incremented when a unit is
        # assigned a slot, so a unit that ran once legitimately carries 1. What
        # makes the crash free is that re-dispatching an INTERRUPTED unit skips
        # that increment (workunit.assign_slots) -- so the retry budget, which
        # is only consulted for status=="failed", is never spent by a crash.
        from hcli.workunit import CLASSIFICATION_INTERRUPTED, assign_ready

        check(
            "durable restart: the crash is classified INTERRUPTED, not failed",
            recovered is not None
            and recovered.classification == CLASSIFICATION_INTERRUPTED,
            f"classification={None if recovered is None else recovered.classification}",
        )
        before = recovered.attempts if recovered else None
        redispatched = assign_ready([recovered], 2, restarted.units) if recovered else []
        check(
            "durable restart: re-dispatch after a crash does not spend an attempt",
            recovered is not None and recovered.attempts == before,
            f"attempts {before} -> {None if recovered is None else recovered.attempts} "
            f"(redispatched={len(redispatched)})",
        )


def check_atomic_write():
    from hcli.dag_store import DagCorruptError, DagStore, atomic_write_json

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": make_unit("a", "LIGHT_CONTROL")}
        store = DagStore(ws)

        import os as _os

        calls = []
        real_replace = _os.replace

        def spy(src, dst, *args, **kwargs):
            calls.append((str(src), str(dst)))
            return real_replace(src, dst, *args, **kwargs)

        _os.replace = spy
        try:
            store.save(units)
        finally:
            _os.replace = real_replace

        dag_path = ws / ".hcli" / "dag.json"
        check(
            "atomic write: dag.json exists after save",
            dag_path.is_file(),
            f"missing {dag_path}",
        )
        used_replace = bool(calls)
        tmp_used = False
        dst_ok = False
        if calls:
            src, dst = calls[-1]
            tmp_used = ".tmp" in Path(src).name or src.endswith(".tmp")
            dst_ok = Path(dst).name == "dag.json"
        check(
            "atomic write: temp file + os.replace",
            used_replace and tmp_used and dst_ok,
            f"replace_calls={calls}",
        )

        # A corrupted DAG must not be loadable as if it were valid.
        dag_path.write_text("{this is not json", encoding="utf-8")
        raised = None
        loaded = None
        try:
            loaded = store.load()
        except DagCorruptError as exc:
            raised = exc
        except Exception as exc:  # noqa: BLE001 — any raise is a refusal to treat as valid
            raised = exc
        check(
            "atomic write: corrupted DAG is not loadable as valid",
            raised is not None and loaded is None,
            f"raised={type(raised).__name__ if raised else None} loaded={loaded!r}",
        )

        # Public helper is itself a temp+replace write.
        target = ws / ".hcli" / "other.json"
        calls2 = []
        _os.replace = lambda src, dst, *a, **k: (
            calls2.append((str(src), str(dst))) or real_replace(src, dst, *a, **k)
        )
        try:
            atomic_write_json(target, {"ok": True})
        finally:
            _os.replace = real_replace
        check(
            "atomic_write_json helper uses replace",
            bool(calls2) and ".tmp" in Path(calls2[-1][0]).name,
            f"calls={calls2}",
        )


def check_no_progress():
    from hcli.scheduler import NO_PROGRESS

    units = {
        "u1": make_unit("u1", "LIGHT_CONTROL"),
        "u2": make_unit("u2", "LIGHT_CONTROL"),
        "u3": make_unit("u3", "LIGHT_CONTROL"),
        "v1": make_unit("v1", "LIGHT_CONTROL"),
        "v2": make_unit("v2", "LIGHT_CONTROL"),
        "v3": make_unit("v3", "LIGHT_CONTROL"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        threshold = 3
        sched = make_scheduler(
            units,
            runtime_count=6,
            workspace=tmp,
            no_progress_threshold=threshold,
        )
        sched.dispatch()
        for uid in ("u1", "u2"):
            if units[uid].status != "running":
                sched.dispatch()
            sched.complete(uid, fingerprint="fa1e64e89916fb4265c6", verification=PASSING_VERIFICATION)
        raised = None
        try:
            if units["u3"].status != "running":
                sched.dispatch()
            sched.complete("u3", fingerprint="fa1e64e89916fb4265c6", verification=PASSING_VERIFICATION)
        except NO_PROGRESS as exc:
            raised = exc
        check(
            "no-progress: unchanged fingerprint raises NO_PROGRESS",
            raised is not None,
            f"raised={raised!r} after {threshold} identical fingerprints",
        )

        quiet = make_scheduler(
            {
                "v1": units["v1"],
                "v2": units["v2"],
                "v3": units["v3"],
            },
            runtime_count=6,
            workspace=tmp,
            no_progress_threshold=threshold,
        )
        quiet.dispatch()
        raised_quiet = None
        try:
            quiet.complete("v1", fingerprint="alpha", verification=PASSING_VERIFICATION)
            quiet.complete("v2", fingerprint="beta", verification=PASSING_VERIFICATION)
            quiet.complete("v3", fingerprint="gamma", verification=PASSING_VERIFICATION)
        except NO_PROGRESS as exc:
            raised_quiet = exc
        check(
            "no-progress: changing fingerprint does NOT raise",
            raised_quiet is None,
            f"raised={raised_quiet!r} on a changing fingerprint stream",
        )


def check_repair_not_repetition():
    units = {"orig": make_unit("orig", "LIGHT_CONTROL", role="implementation")}
    with tempfile.TemporaryDirectory() as tmp:
        sched = make_scheduler(units, runtime_count=1, workspace=tmp)
        sched.dispatch()
        check(
            "repair setup: original is running",
            units["orig"].status == "running",
            f"status={units['orig'].status}",
        )
        orig_attempts = units["orig"].attempts
        sched.fail("orig", context={"error": "boom", "stderr": "assert 1 == 2"})
        others = [u for u in sched.units.values() if u.id != "orig"]
        repair_ok = False
        if others:
            repair = others[0]
            field = getattr(repair, "repairs", None)
            repair_ok = repair.id != "orig" and (
                field == "orig"
                or field == ["orig"]
                or (isinstance(field, list) and "orig" in field)
            )
        check(
            "repair, not repetition: a distinguishable repair unit is created",
            repair_ok,
            f"units={list(sched.units)} others="
            f"{[(u.id, getattr(u, 'repairs', None), getattr(u, 'failure_context', None)) for u in others]}",
        )
        if others:
            ctx = getattr(others[0], "failure_context", None) or {}
            check(
                "repair carries failure context",
                bool(ctx) and ("boom" in str(ctx) or "orig" in str(ctx)),
                f"failure_context={ctx!r}",
            )
        else:
            check("repair carries failure context", False, "no repair unit")
        check(
            "original remains failed rather than only bumping attempts",
            units["orig"].status == "failed" and units["orig"].attempts == orig_attempts,
            f"status={units['orig'].status} attempts={units['orig'].attempts} "
            f"(was {orig_attempts})",
        )


CHECKS = [
    ("decode_limit", check_decode_limit),
    ("cpu_not_blocked_by_gpu", check_cpu_not_blocked_by_gpu),
    ("gpu_exclusive_excludes", check_gpu_exclusive_excludes),
    ("mutation_single_writer", check_mutation_single_writer),
    ("crash_safe_mutation_lock", check_crash_safe_mutation_lock),
    ("no_barrier", check_no_barrier),
    ("durable_restart", check_durable_restart),
    ("atomic_write", check_atomic_write),
    ("no_progress", check_no_progress),
    ("repair_not_repetition", check_repair_not_repetition),
]


def main() -> int:
    FAILS.clear()
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli scheduler checks passed")
    return 0


def test_hcli_scheduler():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} scheduler checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
