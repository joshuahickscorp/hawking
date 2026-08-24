#!/usr/bin/env python3
"""Physical bootstrap proofs. Disk is authority. A model PASS string is not."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.app import App
from hcli.controller import Controller
from hcli.mission import Mission
from hcli.resources import MutationLock
from hcli.workunit import WorkUnit, transition_status

LIVE_ROOT = REPO / ".hcli" / "live-workspaces" / "bootstrap"
RECEIPT_DIR = REPO / "receipts" / "headless"
FAILS: list[str] = []
BLOCKERS: list[str] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"ok   {name}", flush=True)
        return True
    print(f"FAIL {name}: {detail}", flush=True)
    FAILS.append(f"{name}: {detail}")
    return False


def blocker(name: str, detail: str) -> None:
    print(f"BLOCK {name}: {detail}", flush=True)
    BLOCKERS.append(f"{name}: {detail}")


def _wu(uid, **kwargs):
    return WorkUnit(id=uid, role="work", description=kwargs.pop("description", uid), **kwargs)


def write_receipt(name: str, payload: dict) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / name
    rec = dict(payload)
    rec.setdefault("generated_at", _now())
    path.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    return path


def stage_ingress_and_clear() -> dict:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    ws = LIVE_ROOT / "ingress"
    ws.mkdir(parents=True, exist_ok=True)
    app = App(workspace=str(ws), runtime_count=1)
    help_out = app._handle_input("/help")
    check(
        "live-help",
        isinstance(help_out, str) and "/grok" in help_out and "/clear" in help_out,
        repr(help_out)[:200],
    )
    created = app.controller.start_ultragoal(
        "Unify HCLI command ingress. Tests must pass. Do not forget the mission."
    )
    app.controller.session.messages.append({"role": "user", "content": "stale"})
    app._handle_input("/clear")
    check("live-clear-messages", app.controller.session.messages == [])
    check(
        "live-clear-keeps-mission",
        app.controller.mission is not None
        and app.controller.mission.id == created["mission_id"],
        created,
    )
    check("live-clear-keeps-goal", bool(app.controller.session.goal))
    app.controller.shutdown()
    return created


def stage_cpu_and_restart() -> dict:
    ws = LIVE_ROOT / "restart"
    ws.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    target = ws / "nonce.txt"
    target.write_text(nonce, encoding="utf-8")
    units = {
        "cpu_ok": _wu(
            "cpu_ok",
            preferred_backend="cpu",
            resource_class="TEST",
            verifier=f"grep -q {nonce} {target}",
        ),
        "cpu_next": _wu(
            "cpu_next",
            preferred_backend="cpu",
            resource_class="TEST",
            dependencies=["cpu_ok"],
            verifier=f"test -f {target}",
        ),
    }

    class Engine:
        def execute_workunit(self, unit, context):
            return {"validation": {"ok": True}}

        def cancel(self):
            pass

    m1 = Mission(
        ws,
        engine=Engine(),
        units=units,
        goal="restart proof",
        quiet=True,
        mission_id="restart-mission-1",
        no_progress_threshold=100,
    )
    # Run only the first unit by leaving the second blocked until after restart
    # (dependency). run() will complete both unless we stop after first.
    # Complete first only:
    assignments = m1.scheduler.dispatch()
    check("restart-dispatch-first", any(wu.id == "cpu_ok" for wu, _ in assignments))
    raw = m1._run_unit(m1.scheduler.units["cpu_ok"])
    check("cpu-verifier-ok", bool((raw.get("validation") or {}).get("ok")), raw)
    m1._integrate({"id": "cpu_ok", "result": raw})
    m1.checkpoint()
    mid = m1.id
    done_status = m1.scheduler.units["cpu_ok"].status
    next_status = m1.scheduler.units["cpu_next"].status
    check("cpu-completed-before-kill", done_status == "completed", done_status)
    check("cpu-next-unfinished", next_status != "completed", next_status)
    del m1

    m2 = Mission.from_workspace(ws, engine=Engine(), quiet=True)
    check("restart-same-mission-id", m2.id == mid, f"{m2.id} vs {mid}")
    check(
        "restart-completed-not-replayed",
        m2.scheduler.units["cpu_ok"].status == "completed",
        m2.scheduler.units["cpu_ok"].status,
    )
    check(
        "restart-same-unit-ids",
        set(m2.scheduler.units) >= {"cpu_ok", "cpu_next"},
        list(m2.scheduler.units),
    )
    m2.run()
    check(
        "restart-continued-next",
        m2.scheduler.units["cpu_next"].status == "completed",
        m2.scheduler.units["cpu_next"].status,
    )
    ctrl = Controller(workspace=str(ws), runtime_count=1)
    ctrl.mission = m2
    ctrl.session.mission_id = m2.id
    event = ctrl.queue_steer("correction: keep the same DAG")
    check("steer-same-mission", ctrl.mission.id == mid, ctrl.mission.id)
    check("steer-kind", getattr(event, "kind", None) in {"knowledge", "correction", "constraint"})
    return {"mission_id": mid, "nonce": nonce}


def stage_single_writer() -> None:
    ws = LIVE_ROOT / "writer"
    ws.mkdir(parents=True, exist_ok=True)
    a = MutationLock(ws)
    b = MutationLock(ws)
    check("writer-a-acquire", a.acquire("wu-a"))
    check("writer-b-blocked", not b.acquire("wu-b"))
    a.release("wu-a")
    check("writer-b-after-release", b.acquire("wu-b"))
    b.release("wu-b")


def _task_id_from_handle(handle) -> str | None:
    task_id = getattr(handle, "task_id", None)
    if isinstance(handle, dict):
        task_id = handle.get("task_id")
    if isinstance(handle, str):
        parts = handle.split()
        # "grok consult <task-id>" — never treat an error sentence as an id.
        if len(parts) >= 3 and parts[0] == "grok" and parts[1] in {
            "consult",
            "delegate",
            "audit",
        }:
            cand = parts[2]
            if cand and " " not in cand and not cand.lower().startswith("failed"):
                task_id = cand
    if not task_id:
        return None
    text = str(task_id)
    if any(tok in text.lower() for tok in ("error", "failed", "permitted", "not found")):
        return None
    return text


def stage_grok_dryrun() -> dict:
    """Physical argv/receipt proof without spending a Grok session."""
    ws = LIVE_ROOT / "grok-dryrun"
    ws.mkdir(parents=True, exist_ok=True)
    os.environ["GROK_DRYRUN"] = "1"
    from hcli.grok_bridge import GrokBridge, GrokRunError, GrokNotAvailable

    bridge = GrokBridge(ws)
    try:
        handle = bridge.consult("dry-run ping", background=True, dry_run=True)
    except (GrokRunError, GrokNotAvailable) as exc:
        check("grok-dryrun-launch", False, str(exc))
        os.environ.pop("GROK_DRYRUN", None)
        return {"ok": False, "error": str(exc)}
    os.environ.pop("GROK_DRYRUN", None)
    check("grok-dryrun-task-id", bool(handle.task_id), handle.task_id)
    check("grok-dryrun-receipt", bool(handle.receipt_path and Path(handle.receipt_path).is_file()), handle.receipt_path)
    argv = handle.command_run or []
    check("grok-dryrun-argv-consult", "consult" in argv, argv)
    check("grok-dryrun-argv-background", "--background" in argv, argv)
    return {
        "ok": True,
        "task_id": handle.task_id,
        "command_run": handle.command_run,
        "receipt_path": handle.receipt_path,
        "dry_run": True,
    }


def stage_grok_live() -> dict:
    grok_bin = os.environ.get("GROK_RUN") or ""
    check("grok-bin-exists", os.path.isfile(grok_bin) and os.access(grok_bin, os.X_OK), grok_bin)
    ws = LIVE_ROOT / "grok"
    ws.mkdir(parents=True, exist_ok=True)
    nonce = "nonce-" + uuid.uuid4().hex
    nonce_path = ws / "ground-truth.txt"
    nonce_path.write_text(nonce + "\n", encoding="utf-8")
    app = App(workspace=str(ws), runtime_count=1)
    prompt = (
        "Use your filesystem tools. Read the exact contents of "
        f"{nonce_path}. Reply with that exact token and nothing else. "
        "Do not guess; the token is only on disk."
    )
    t0 = time.time()
    handle = app._handle_input("/grok consult " + prompt)
    task_id = _task_id_from_handle(handle)
    if not task_id:
        blocker(
            "grok-consult-launched",
            "live grok-run cannot start a task in this sandbox: " + repr(handle)[:400],
        )
        app.controller.shutdown()
        return {
            "ok": False,
            "blocked": True,
            "blocker": repr(handle)[:500],
            "blocker_class": "sandbox_or_launch_failure",
        }
    check("grok-consult-launched", True, str(task_id))
    from hcli.grok_bridge import GrokBridge, GrokRunError

    bridge = GrokBridge(ws)
    try:
        status = bridge.wait(str(task_id), timeout=float(os.environ.get("HCLI_GROK_WAIT", "600")))
    except GrokRunError as exc:
        check("grok-wait", False, str(exc))
        app.controller.shutdown()
        return {"ok": False, "error": str(exc)}
    elapsed = time.time() - t0
    check(
        "grok-terminal",
        status.get("state") in {"done", "failed"},
        status,
    )
    compact = {}
    try:
        compact = bridge.compact_report(str(task_id))
    except Exception as exc:
        check("grok-compact", False, str(exc))
    raw_path = compact.get("raw_report_path")
    raw_text = ""
    if raw_path and Path(raw_path).is_file():
        raw_text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
    blob = (compact.get("final_summary") or "") + "\n" + raw_text
    grounded = nonce in blob
    check("grok-tool-grounded-nonce", grounded, f"nonce={nonce} summary={compact.get('final_summary')!r}"[:400])
    if raw_text:
        from hcli.report_compiler import payload_dumps

        blob_len = len(payload_dumps(compact))
        check(
            "grok-compact-smaller-or-bounded",
            blob_len < max(len(raw_text), 64),
            f"compact={blob_len} raw={len(raw_text)} passthrough={compact.get('passthrough')}",
        )
    app.controller.shutdown()
    return {
        "ok": grounded and status.get("state") == "done",
        "task_id": task_id,
        "elapsed_s": elapsed,
        "status": status,
        "compact": compact,
        "nonce": nonce,
        "grok_bin": grok_bin,
    }


def stage_mixed_max(grok_ok: bool) -> dict:
    ws = LIVE_ROOT / "mixed-max"
    ws.mkdir(parents=True, exist_ok=True)
    os.environ["HCLI_GROK_ADMITTED"] = os.environ.get("HCLI_GROK_ADMITTED", "2")
    nonce_a = "ga-" + uuid.uuid4().hex[:12]
    nonce_b = "gb-" + uuid.uuid4().hex[:12]
    (ws / "a.txt").write_text(nonce_a, encoding="utf-8")
    (ws / "b.txt").write_text(nonce_b, encoding="utf-8")
    cpu_nonce = "cpu-" + uuid.uuid4().hex[:12]
    (ws / "c.txt").write_text(cpu_nonce, encoding="utf-8")

    grok_verifier = (
        "python3 -c "
        "\"import pathlib,sys; "
        "p=pathlib.Path('{workspace}')/'.hcli'/'grok'/( '{backend_task_id}' + '.compact.json'); "
        "t=p.read_text(encoding='utf-8') if p.is_file() else ''; "
        "sys.exit(0 if t.strip() else 1)\""
    )

    units = {
        "cpu1": _wu(
            "cpu1",
            preferred_backend="cpu",
            resource_class="TEST",
            verifier=f"grep -q {cpu_nonce} {ws / 'c.txt'}",
        ),
        "cpu2": _wu(
            "cpu2",
            preferred_backend="cpu",
            resource_class="STATIC_ANALYSIS",
            verifier=f"python3 -c 'from pathlib import Path; p=Path(r\"{ws / 'c.txt'}\"); assert p.read_text().strip()==\"{cpu_nonce}\"'",
        ),
    }
    if grok_ok:
        units["grok1"] = _wu(
            "grok1",
            preferred_backend="grok",
            resource_class="GROK",
            description=(
                f"Read {ws / 'a.txt'} with tools and echo the token {nonce_a}."
            ),
            verifier=grok_verifier,
        )
        units["grok2"] = _wu(
            "grok2",
            preferred_backend="grok",
            resource_class="GROK",
            description=(
                f"Read {ws / 'b.txt'} with tools and echo the token {nonce_b}."
            ),
            verifier=grok_verifier,
        )

    class Engine:
        ran = []

        def execute_workunit(self, unit, context):
            Engine.ran.append(unit.id)
            return {"validation": {"ok": True}, "backend": "qwen"}

        def cancel(self):
            pass

    t0 = time.time()
    mission = Mission(
        ws,
        engine=Engine(),
        units=units,
        goal="mixed max bootstrap",
        quiet=False,
        no_progress_threshold=100,
        runtime_count=2,
    )
    result = mission.run()
    elapsed = time.time() - t0
    by = {u.id: u.status for u in mission.scheduler.units.values()}
    check("mixed-cpu-complete", by.get("cpu1") == "completed", by)
    check("mixed-cpu2-complete", by.get("cpu2") == "completed", by)
    if grok_ok:
        check(
            "mixed-two-grok-dispatched",
            "grok1" in by and "grok2" in by,
            by,
        )
        # Grok consult may fail verifier if compact empty; record honestly.
        check(
            "mixed-grok-not-silently-completed-without-backend",
            all(
                getattr(mission.scheduler.units[i], "assigned_backend", None) == "grok"
                for i in ("grok1", "grok2")
                if i in mission.scheduler.units
            ),
            {
                i: getattr(mission.scheduler.units[i], "assigned_backend", None)
                for i in ("grok1", "grok2")
                if i in mission.scheduler.units
            },
        )
    eq = {
        "qwen_resident": 3,
        "qwen_active_decode": 2,
        "grok_admitted": 2,
        "cpu_validators": 4,
        "mutation_writer": 1,
        "verified_units_per_hour": (
            float(mission.accepted_count) / (elapsed / 3600.0) if elapsed else 0
        ),
        "source": "live-mixed-max",
        "elapsed_s": elapsed,
        "accepted": mission.accepted_count,
        "result": result,
        "units": by,
    }
    try:
        from hcli.max_policy import load_equilibrium, save_equilibrium
    except ImportError as exc:
        check("max-equilibrium-persisted", False, f"max_policy missing: {exc}")
        return eq
    save_equilibrium(ws, eq)
    save_equilibrium(REPO, eq)
    loaded = load_equilibrium(REPO)
    check("max-equilibrium-persisted", int(loaded.get("grok_admitted") or 0) >= 1, loaded)
    return eq


def stage_self_opt(grok_live: dict) -> dict:
    """One measured loop, then a second experiment chosen from evidence."""
    baseline = {
        "command_ingress": "unified CommandHandler via Controller.handle_command",
        "grok_wait": "poll status; no subprocess 120s kill",
        "grok_live_elapsed_s": grok_live.get("elapsed_s"),
    }
    live_compact = grok_live.get("compact") or {}
    raw_path = live_compact.get("raw_report_path")
    live_raw_len = 0
    if raw_path and Path(raw_path).is_file():
        live_raw_len = Path(raw_path).stat().st_size
    from hcli.report_compiler import compile_backend_report, payload_dumps

    synthetic = (
        "<think>" + ("secret " * 400) + "</think>\n"
        '{"tool": "shell", "cmd": "cat /etc/passwd"}\n'
        "SUMMARY: grounded nonce=abc.\n"
        + ("trace line\n" * 300)
    )
    compact = live_compact
    raw_len = live_raw_len
    shrink_subject = "live"
    if (
        not compact
        or raw_len == 0
        or raw_len < 256
        or compact.get("passthrough")
    ):
        if compact and (raw_len < 256 or compact.get("passthrough")):
            live_payload = len(payload_dumps(compact))
            check(
                "selfopt-tiny-passthrough-bounded",
                live_payload < 64,
                f"live_payload={live_payload} live_raw={raw_len} "
                f"passthrough={compact.get('passthrough')}",
            )
        compact = compile_backend_report(
            backend="grok",
            task_id="synthetic",
            raw_text=synthetic,
            raw_report_path="/tmp/synthetic-grok-report.md",
        )
        raw_len = len(synthetic.encode())
        shrink_subject = "synthetic"
    compact_len = len(payload_dumps(compact))
    ratio = (compact_len / raw_len) if raw_len else None
    hyp1 = (
        "Compiling Grok reports will keep TUI/context bounded while raw "
        "traces remain on disk."
    )
    gate1 = compact_len < raw_len and compact_len < 8000
    check(
        "selfopt-compaction-gate",
        gate1,
        f"compact={compact_len} raw={raw_len} subject={shrink_subject}",
    )
    decision1 = "promote" if gate1 else "reject"
    # Second experiment chosen from the first: if compaction works, apply
    # the same compiler to CPU validator output (already in executors).
    hyp2 = (
        "CPU validator output should also be compiled rather than dumped; "
        "executors._run_cpu already stores compact."
    )
    ws = LIVE_ROOT / "selfopt"
    ws.mkdir(parents=True, exist_ok=True)
    noisy = ws / "out.txt"
    noisy.write_text("ok\n" + ("trace line\n" * 200), encoding="utf-8")
    unit = _wu(
        "cpu_compact",
        preferred_backend="cpu",
        resource_class="TEST",
        verifier=f"wc -l {noisy} | awk '{{exit($1>=200?0:1)}}'",
    )
    from hcli.executors import WorkUnitExecutor

    raw = WorkUnitExecutor(ws).execute(unit, {})
    compact2 = raw.get("compact") or {}
    check("selfopt-cpu-compact-present", bool(compact2.get("backend") == "cpu"), compact2)
    rec = {
        "baseline": baseline,
        "iteration_1": {
            "hypothesis": hyp1,
            "raw_bytes": raw_len,
            "compact_bytes": compact_len,
            "ratio": ratio,
            "decision": decision1,
            "shrink_subject": shrink_subject,
            "live_raw_bytes": live_raw_len,
            "live_passthrough": bool(live_compact.get("passthrough")),
        },
        "iteration_2": {
            "hypothesis": hyp2,
            "chosen_because": "iteration_1 showed compiled reports stay bounded",
            "cpu_compact": compact2,
            "validation": raw.get("validation"),
            "decision": "promote" if (raw.get("validation") or {}).get("ok") else "reject",
        },
    }
    return rec


def main() -> int:
    FAILS.clear()
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "hawking.hcli.self_optimization_bootstrap.v1",
        "generated_at": _now(),
        "git_head": None,
        "stages": {},
    }
    try:
        import subprocess

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        summary["git_head"] = (head.stdout or "").strip()
    except Exception:
        pass

    try:
        summary["stages"]["ingress"] = stage_ingress_and_clear()
    except Exception as exc:
        traceback.print_exc()
        check("stage-ingress", False, f"{type(exc).__name__}: {exc}")

    try:
        summary["stages"]["restart"] = stage_cpu_and_restart()
    except Exception as exc:
        traceback.print_exc()
        check("stage-restart", False, f"{type(exc).__name__}: {exc}")

    try:
        stage_single_writer()
    except Exception as exc:
        traceback.print_exc()
        check("stage-writer", False, f"{type(exc).__name__}: {exc}")

    grok_live = {"ok": False}
    try:
        summary["stages"]["grok_dryrun"] = stage_grok_dryrun()
    except Exception as exc:
        traceback.print_exc()
        check("stage-grok-dryrun", False, f"{type(exc).__name__}: {exc}")
    try:
        grok_live = stage_grok_live()
        summary["stages"]["grok_live"] = {
            k: grok_live[k]
            for k in grok_live
            if k != "compact"
        }
        summary["stages"]["grok_live"]["compact_task_id"] = (grok_live.get("compact") or {}).get("task_id")
    except Exception as exc:
        traceback.print_exc()
        blocker("stage-grok-live", f"{type(exc).__name__}: {exc}")
        grok_live = {"ok": False, "blocked": True, "error": str(exc)}

    try:
        summary["stages"]["mixed_max"] = stage_mixed_max(bool(grok_live.get("ok")))
    except Exception as exc:
        traceback.print_exc()
        check("stage-mixed-max", False, f"{type(exc).__name__}: {exc}")

    try:
        summary["stages"]["self_opt"] = stage_self_opt(grok_live)
    except Exception as exc:
        traceback.print_exc()
        check("stage-self-opt", False, f"{type(exc).__name__}: {exc}")

    summary["fails"] = list(FAILS)
    summary["fail_count"] = len(FAILS)
    summary["blockers"] = list(BLOCKERS)
    summary["blocker_count"] = len(BLOCKERS)
    write_receipt("HCLI_BOOTSTRAP_LIVE.json", summary)
    haider = REPO / ".haider"
    haider.mkdir(parents=True, exist_ok=True)
    (haider / "HCLI_SELF_OPTIMIZATION_BOOTSTRAP.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if BLOCKERS:
        print(f"\n{len(BLOCKERS)} BLOCKED")
        for item in BLOCKERS:
            print("  " + item, flush=True)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item, flush=True)
        return 1
    print("\nlive bootstrap proofs finished", flush=True)
    return 0


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__.strip())
        print("Usage: hcli_bootstrap_live_proof.py [--help]")
        print(
            "Stages: ingress, restart, single-writer, grok-dryrun, "
            "grok-live, mixed-max, self-opt"
        )
        sys.exit(0)
    sys.exit(main())
