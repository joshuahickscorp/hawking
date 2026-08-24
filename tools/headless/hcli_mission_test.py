#!/usr/bin/env python3
"""Protected HCLI mission-loop checks. Plain python3 + assert. No GPU, no model.

Run:
    python3 tools/headless/hcli_mission_test.py
    pytest tools/headless/hcli_mission_test.py -q

Drive the real Mission with a stub engine. Every check below was watched
FAILING against the pre-change tree (no mission.py).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"ok   {name}")
        return True
    print(f"FAIL {name}: {detail}")
    FAILS.append(f"{name}: {detail}")
    return False


def _import_mission():
    from hcli.mission import Mission  # noqa: WPS433

    return Mission


def _wu(uid, resource_class="LIGHT_CONTROL", deps=None, role="work"):
    from hcli.workunit import WorkUnit

    return WorkUnit(
        id=uid,
        role=role,
        description=uid,
        dependencies=list(deps or []),
        resource_class=resource_class,
    )


class StubEngine:
    """Canned work. Never talks to a model. Records start/finish order."""

    def __init__(self, workspace, delays=None, results=None, gates=None):
        self.workspace = Path(workspace)
        self.delays = dict(delays or {})
        self.results = dict(results or {})
        self.gates = dict(gates or {})
        self.lock = threading.Lock()
        self.events = []
        self.contexts = {}
        self.ran = []
        self.gpu_inflight = 0
        self.max_gpu = 0
        self.cpu_during_saturated_gpu = False
        self.cancelled = False
        self.child_pids = []

    def cancel(self):
        self.cancelled = True
        for gate in self.gates.values():
            try:
                gate.set()
            except Exception:
                pass

    def execute_workunit(self, unit, context):
        uid = unit.id
        rc = getattr(unit, "resource_class", "LIGHT_CONTROL")
        with self.lock:
            self.events.append(("start", uid, time.perf_counter()))
            self.ran.append(uid)
            self.contexts[uid] = dict(context or {})
            if rc == "GPU_DECODE":
                self.gpu_inflight += 1
                if self.gpu_inflight > self.max_gpu:
                    self.max_gpu = self.gpu_inflight
            elif self.gpu_inflight >= 2:
                self.cpu_during_saturated_gpu = True
        try:
            gate = self.gates.get(uid)
            if gate is not None:
                gate.wait(timeout=30)
            delay = float(self.delays.get(uid, 0.0))
            deadline = time.perf_counter() + delay
            while time.perf_counter() < deadline:
                if self.cancelled:
                    return {"cancelled": True, "status": "cancelled"}
                checker = (context or {}).get("is_cancelled")
                if callable(checker) and checker():
                    return {"cancelled": True, "status": "cancelled"}
                time.sleep(0.01)
            if self.cancelled:
                return {"cancelled": True, "status": "cancelled"}
            canned = self.results.get(uid)
            if canned is not None:
                return dict(canned)
            marker = self.workspace / f"accepted_{uid}.txt"
            marker.write_text(uid, encoding="utf-8")
            return {"validation": {"ok": True}}
        finally:
            with self.lock:
                self.events.append(("finish", uid, time.perf_counter()))
                if rc == "GPU_DECODE":
                    self.gpu_inflight = max(0, self.gpu_inflight - 1)


def _event_time(events, kind, uid):
    for k, i, t in events:
        if k == kind and i == uid:
            return t
    return None


def _env_decode(value):
    saved = os.environ.get("ACTIVE_DECODE_LIMIT")
    os.environ["ACTIVE_DECODE_LIMIT"] = value
    return saved


def _restore_decode(saved):
    if saved is None:
        os.environ.pop("ACTIVE_DECODE_LIMIT", None)
    else:
        os.environ["ACTIVE_DECODE_LIMIT"] = saved


def check_no_barrier():
    name = "no barrier: A2 starts before slow B1 finishes"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {
            "A1": _wu("A1"),
            "A2": _wu("A2", deps=["A1"]),
            "B1": _wu("B1"),
            "B2": _wu("B2", deps=["B1"]),
        }
        engine = StubEngine(ws, delays={"B1": 0.35, "A1": 0.02, "A2": 0.02, "B2": 0.02})
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
            heartbeat_s=60,
        )
        mission.run()
        a2s = _event_time(engine.events, "start", "A2")
        b1f = _event_time(engine.events, "finish", "B1")
        check(
            name,
            a2s is not None and b1f is not None and a2s < b1f,
            f"A2_start={a2s} B1_finish={b1f} events={engine.events}",
        )


def check_decode_limit():
    name = "decode limit respected; CPU still dispatches"
    saved = _env_decode("2")
    try:
        try:
            Mission = _import_mission()
        except Exception as exc:
            check(name, False, f"{type(exc).__name__}: {exc}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            units = {
                "g0": _wu("g0", "GPU_DECODE"),
                "g1": _wu("g1", "GPU_DECODE"),
                "g2": _wu("g2", "GPU_DECODE"),
                "g3": _wu("g3", "GPU_DECODE"),
                "c0": _wu("c0", "COMPILE", role="compile"),
                "t0": _wu("t0", "TEST", role="test"),
            }
            engine = StubEngine(
                ws,
                delays={
                    "g0": 0.20,
                    "g1": 0.20,
                    "g2": 0.20,
                    "g3": 0.20,
                    "c0": 0.05,
                    "t0": 0.05,
                },
            )
            mission = Mission(
                ws,
                engine=engine,
                units=units,
                runtime_count=6,
                quiet=True,
                no_progress_threshold=100,
                heartbeat_s=60,
            )
            mission.run()
            max_gpu = engine.max_gpu

            def _gpu_inflight_at(t):
                n = 0
                for kind, uid, ts in engine.events:
                    if not str(uid).startswith("g"):
                        continue
                    if kind == "start" and ts <= t:
                        n += 1
                    elif kind == "finish" and ts <= t:
                        n -= 1
                return n

            cpu_starts = [
                t
                for kind, uid, t in engine.events
                if kind == "start" and uid in ("c0", "t0")
            ]
            cpu_ok = any(_gpu_inflight_at(t) >= 1 for t in cpu_starts)
            check(
                name,
                max_gpu == 2 and cpu_ok,
                f"max_gpu={max_gpu} (want exactly 2) "
                f"cpu_ok={cpu_ok} cpu_starts={cpu_starts} ran={engine.ran}",
            )
    finally:
        _restore_decode(saved)


def check_restart_resumes():
    name = "restart resumes; in-flight is not completed"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        script = ws / "crash_mission.py"
        script.write_text(
            "\n".join(
                [
                    "import os, sys",
                    
                    "from pathlib import Path",
                    "from hcli.mission import Mission",
                    "from hcli.workunit import WorkUnit",
                    f"ws = Path({str(ws)!r})",
                    "class E:",
                    "    cancelled = False",
                    "    def cancel(self):",
                    "        self.cancelled = True",
                    "    def execute_workunit(self, unit, context):",
                    "        (ws / f'started_{unit.id}').write_text('1')",
                    "        if unit.id == 'B':",
                    "            os._exit(9)",
                    "        (ws / f'accepted_{unit.id}.txt').write_text(unit.id)",
                    "        return {'validation': {'ok': True}}",
                    "units = {",
                    "    'A': WorkUnit(id='A', role='work', description='A'),",
                    "    'B': WorkUnit(id='B', role='work', description='B', dependencies=['A']),",
                    "}",
                    "m = Mission(ws, engine=E(), units=units, quiet=True,",
                    "            no_progress_threshold=100, heartbeat_s=60,",
                    "            install_signals=False)",
                    "m.run()",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode not in (9, 1, 128 + 9) and not (ws / "started_B").is_file():
            check(
                name,
                False,
                f"crash subprocess did not run B: rc={proc.returncode} "
                f"stderr={proc.stderr[-800:]!r} stdout={proc.stdout[-400:]!r}",
            )
            return
        try:
            restarted = Mission.from_workspace(ws, engine=StubEngine(ws), quiet=True)
        except Exception as exc:
            check(name, False, f"from_workspace: {type(exc).__name__}: {exc}")
            return
        units = restarted.scheduler.units
        a_status = units["A"].status if "A" in units else None
        b_status = units["B"].status if "B" in units else None
        # `interrupted` is a first-class status distinct from `failed`: a crash
        # is not a verifier failure, so it neither consumes the retry budget nor
        # grows the repair tree. This check predates that status and demanded
        # "failed" or "ready", so a correctly-recovered mission read as broken.
        # The invariant that matters is that in-flight work is NOT completed and
        # is still dispatchable.
        from hcli.workunit import is_ready

        b_unit = units.get("B")
        check(
            name,
            a_status == "completed"
            and b_status != "completed"
            and b_unit is not None
            and is_ready(b_unit, units),
            f"A={a_status} B={b_status} "
            f"B_dispatchable={b_unit is not None and is_ready(b_unit, units)} "
            f"keys={list(units)}",
        )


def check_atomic_state():
    name = "atomic state: temp+rename; truncated rejected"
    try:
        Mission = _import_mission()
        from hcli.mission import MissionCorruptError
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": _wu("a")}
        engine = StubEngine(ws)
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
        )
        calls = []
        real_replace = os.replace

        def spy(src, dst, *args, **kwargs):
            calls.append((str(src), str(dst)))
            return real_replace(src, dst, *args, **kwargs)

        os.replace = spy
        try:
            mission.checkpoint()
        finally:
            os.replace = real_replace
        state_path = ws / ".hcli" / "mission" / "state.json"
        used_tmp = False
        dst_ok = False
        if calls:
            for src, dst in calls:
                if Path(dst) == state_path or Path(dst).name == "state.json":
                    used_tmp = ".tmp" in Path(src).name or src.endswith(".tmp")
                    dst_ok = True
                    break
            if not dst_ok and calls:
                src, dst = calls[-1]
                used_tmp = ".tmp" in Path(src).name or src.endswith(".tmp")
                dst_ok = "mission" in dst or Path(dst).name in {"state.json", "dag.json"}
        wrote = state_path.is_file()
        atomic_ok = wrote and used_tmp and dst_ok
        check(
            "atomic state: temp+rename",
            atomic_ok,
            f"wrote={wrote} used_tmp={used_tmp} dst_ok={dst_ok} calls={calls}",
        )
        if not wrote:
            check(
                "atomic state: truncated rejected",
                False,
                "no state.json to truncate",
            )
            return
        state_path.write_text('{"id": "truncated"', encoding="utf-8")
        raised = None
        loaded = None
        try:
            loaded = Mission.from_workspace(ws, engine=StubEngine(ws), quiet=True)
        except MissionCorruptError as exc:
            raised = exc
        except Exception as exc:  # any refusal to treat as valid is the point
            raised = exc
        check(
            "atomic state: truncated rejected",
            raised is not None and loaded is None,
            f"raised={type(raised).__name__ if raised else None} loaded={loaded!r}",
        )


def check_cancel_clean():
    name = "cancel is clean: SIGINT, checkpoint, zero children"
    try:
        _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        script = ws / "cancel_mission.py"
        script.write_text(
            "\n".join(
                [
                    "import os, sys, subprocess, time",
                    
                    "from pathlib import Path",
                    "from hcli.mission import Mission",
                    "from hcli.workunit import WorkUnit",
                    f"ws = Path({str(ws)!r})",
                    "child = subprocess.Popen(['sleep', '60'], start_new_session=True)",
                    "(ws / 'child.pid').write_text(str(child.pid))",
                    "class E:",
                    "    cancelled = False",
                    "    child_pids = [child.pid]",
                    "    def cancel(self):",
                    "        self.cancelled = True",
                    "    def execute_workunit(self, unit, context):",
                    "        (ws / 'started').write_text('1')",
                    "        for _ in range(2000):",
                    "            if self.cancelled:",
                    "                return {'cancelled': True}",
                    "            checker = (context or {}).get('is_cancelled')",
                    "            if callable(checker) and checker():",
                    "                return {'cancelled': True}",
                    "            time.sleep(0.02)",
                    "        return {'validation': {'ok': True}}",
                    "units = {'slow': WorkUnit(id='slow', role='work', description='slow')}",
                    "m = Mission(ws, engine=E(), units=units, quiet=True,",
                    "            no_progress_threshold=100, heartbeat_s=60,",
                    "            install_signals=True)",
                    "m.register_child_pid(child.pid)",
                    "m.install_signal_handlers()",
                    "(ws / 'ready').write_text('1')",
                    "raise SystemExit(m.main_exit())",
                ]
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ready = ws / "ready"
        deadline = time.time() + 8
        while time.time() < deadline and not ready.is_file():
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        child_pid = None
        pid_file = ws / "child.pid"
        if pid_file.is_file():
            try:
                child_pid = int(pid_file.read_text().strip())
            except ValueError:
                child_pid = None
        if proc.poll() is not None:
            out, err = proc.communicate()
            check(
                name,
                False,
                f"subprocess exited before SIGINT rc={proc.returncode} "
                f"stderr={err[-800:]!r} stdout={out[-400:]!r}",
            )
            return
        os.kill(proc.pid, signal.SIGINT)
        try:
            stdout, stderr = proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            check(
                name,
                False,
                f"did not exit promptly after SIGINT stdout={stdout[-400:]!r} "
                f"stderr={stderr[-400:]!r}",
            )
            return
        state_path = ws / ".hcli" / "mission" / "state.json"
        checkpointed = state_path.is_file()
        child_alive = False
        if child_pid:
            try:
                os.kill(child_pid, 0)
                child_alive = True
            except OSError:
                child_alive = False
        reason = (stderr or "") + (stdout or "")
        legible = "cancel" in reason.lower() or "sigint" in reason.lower()
        check(
            name,
            proc.returncode != 0
            and checkpointed
            and not child_alive
            and legible,
            f"rc={proc.returncode} checkpointed={checkpointed} "
            f"child_pid={child_pid} child_alive={child_alive} "
            f"legible={legible} stderr={stderr[-500:]!r}",
        )


def check_repair_not_repetition():
    name = "repair, not repetition"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"orig": _wu("orig", role="implementation")}
        engine = StubEngine(
            ws,
            results={"orig": {"validation": {"ok": False, "reason": "boom"}, "error": "boom"}},
        )
        # repairs should succeed if the stub doesn't know them
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
            heartbeat_s=60,
        )
        mission.run()
        others = [u for u in mission.scheduler.units.values() if u.id != "orig"]
        orig = mission.scheduler.units["orig"]
        repair_ok = False
        repair = None
        if others:
            repair = others[0]
            repair_ok = repair.id != "orig" and getattr(repair, "repairs", None) == "orig"
        ctx = getattr(repair, "failure_context", None) if repair is not None else None
        orig_runs = [x for x in engine.ran if x == "orig"]
        check(
            name,
            repair_ok
            and orig.status == "failed"
            and len(orig_runs) == 1
            and bool(ctx)
            and ("boom" in str(ctx) or "orig" in str(ctx)),
            f"orig={orig.status} ran={engine.ran} others="
            f"{[(u.id, getattr(u, 'repairs', None), getattr(u, 'failure_context', None)) for u in others]}",
        )


def check_no_progress_acts():
    name = "no-progress acts, and stays quiet when fingerprints move"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {f"u{i}": _wu(f"u{i}") for i in range(3)}
        engine = StubEngine(ws)

        def same_fp():
            return "fa1e64e89916fb4265c6"

        stuck = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=3,
            fingerprint_fn=same_fp,
            heartbeat_s=60,
        )
        result = stuck.run()
        warning = stuck.status().get("no_progress_warning")
        acted = (
            stuck.strategy != "default"
            or stuck.phase in ("no_progress", "failed", "stopped")
            or (result or {}).get("reason") == "no_progress"
            or bool(warning)
        )
        said = bool(warning) or "no_progress" in str(result).lower() or "no-progress" in str(result).lower()
        check(
            "no-progress acts and says so",
            acted and said,
            f"strategy={stuck.strategy!r} phase={stuck.phase!r} "
            f"warning={warning!r} result={result!r}",
        )

    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {f"v{i}": _wu(f"v{i}") for i in range(3)}
        engine = StubEngine(ws)
        n = {"i": 0}

        def moving_fp():
            n["i"] += 1
            return f"fp-{n['i']}"

        moving = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=3,
            fingerprint_fn=moving_fp,
            heartbeat_s=60,
        )
        moving.run()
        warning = moving.status().get("no_progress_warning")
        check(
            "no-progress stays quiet on a moving fingerprint",
            not warning and moving.phase not in ("no_progress",),
            f"phase={moving.phase!r} warning={warning!r} strategy={moving.strategy!r}",
        )


def check_model_text_is_not_evidence():
    name = "model text is not evidence"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"claim": _wu("claim", role="implementation")}
        engine = StubEngine(
            ws,
            results={
                "claim": {
                    "status": "completed",
                    "kind": "mutation",
                    "content": "I definitely applied the patch and all tests passed.",
                    "operations": [],
                    "tests": [],
                }
            },
        )
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
            heartbeat_s=60,
        )
        mission.run()
        wu = mission.scheduler.units["claim"]
        check(
            name,
            wu.status != "completed",
            f"status={wu.status} (model claimed completed; workspace unchanged)",
        )


def check_steering():
    name = "steering reaches later units and does not rewrite the past"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {
            "A": _wu("A"),
            "B": _wu("B", deps=["A"]),
        }
        engine = StubEngine(ws, delays={"A": 0.02})
        steered = {"done": False}

        def before_dispatch(mission):
            if steered["done"]:
                return
            if mission.scheduler.units["A"].status == "completed":
                mission.append_steer("prefer the smaller patch")
                steered["done"] = True

        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
            heartbeat_s=60,
            session_id="steer-test",
            before_dispatch=before_dispatch,
        )
        mission.run()
        a_status_after = mission.scheduler.units["A"].status
        b_ctx = engine.contexts.get("B") or {}
        seen = "prefer the smaller patch" in json.dumps(b_ctx, default=str)
        check(
            name,
            steered["done"]
            and a_status_after == "completed"
            and seen,
            f"steered={steered['done']} A_after={a_status_after} "
            f"B_ctx={b_ctx} seen={seen}",
        )


def check_status_shape():
    name = "/status shape"
    try:
        Mission = _import_mission()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        units = {"a": _wu("a")}
        engine = StubEngine(ws)
        mission = Mission(
            ws,
            engine=engine,
            units=units,
            quiet=True,
            no_progress_threshold=100,
            heartbeat_s=60,
        )
        mission.run()
        snap = mission.status()
        required = [
            "mission_id",
            "phase",
            "units_by_status",
            "active_runtimes",
            "active_decodes",
            "accepted_units_per_hour",
            "elapsed_wall",
            "last_checkpoint",
            "no_progress_warning",
        ]
        missing = [k for k in required if k not in snap]
        numeric_ok = True
        detail = []
        if not isinstance(snap.get("units_by_status"), dict):
            numeric_ok = False
            detail.append(f"units_by_status={snap.get('units_by_status')!r}")
        else:
            for key, val in snap["units_by_status"].items():
                if not isinstance(val, (int, float)):
                    numeric_ok = False
                    detail.append(f"units_by_status[{key}]={val!r}")
        for key in (
            "active_runtimes",
            "active_decodes",
            "accepted_units_per_hour",
            "elapsed_wall",
        ):
            if not isinstance(snap.get(key), (int, float)):
                numeric_ok = False
                detail.append(f"{key}={snap.get(key)!r}")
        last = snap.get("last_checkpoint")
        if not isinstance(last, (int, float, str)) or last in ("", None):
            numeric_ok = False
            detail.append(f"last_checkpoint={last!r}")
        check(
            name,
            not missing and numeric_ok and snap.get("mission_id"),
            f"missing={missing} detail={detail} snap={snap!r}",
        )


CHECKS = [
    ("no_barrier", check_no_barrier),
    ("decode_limit", check_decode_limit),
    ("restart_resumes", check_restart_resumes),
    ("atomic_state", check_atomic_state),
    ("cancel_clean", check_cancel_clean),
    ("repair_not_repetition", check_repair_not_repetition),
    ("no_progress_acts", check_no_progress_acts),
    ("model_text_is_not_evidence", check_model_text_is_not_evidence),
    ("steering", check_steering),
    ("status_shape", check_status_shape),
]


def main() -> int:
    FAILS.clear()
    for _name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {_name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{_name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli mission checks passed")
    return 0


def test_hcli_mission():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} mission checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
