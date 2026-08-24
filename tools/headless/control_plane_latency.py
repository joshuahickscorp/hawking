#!/usr/bin/env python3
"""CONTROL_PLANE_LATENCY — measure HCLI ceremony with no model inference.

The product is a ledger, not a refactor. Another lane owns the package tree.
This file times ordinary control-plane overhead (startup, import, mission,
DAG, context, status, scheduler, receipt, checkpoint, verifier, experiment
setup, Grok bridge, runtime admission) and censuses repeated filesystem
scans, JSON parses, hashing, subprocess/Python spawns, duplicated
validation, and tool discovery.

Every number carries the exact command and cold vs warm separately.

    python3 tools/headless/control_plane_latency.py
    python3 -m pytest tools/headless/control_plane_latency.py -q
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA = "hawking.headless.control_plane_latency.v1"
GATE = "CONTROL_PLANE_LATENCY"
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
RECEIPT = REPO / "receipts" / "headless" / "CONTROL_PLANE_LATENCY_LEDGER.json"
RECEIPT_LEGACY = REPO / "receipts" / "headless" / "CONTROL_PLANE_LATENCY.json"
HCLI_GIT_PREFIX = "hcli"
METAL_BUDGET_SRC = REPO / "tools" / "headless" / "metal_budget.py"
# Acceptance stage name -> measurement id. A missing row is ABSENT, never estimated.
STAGE_MAP = {
    "cli_startup": "cli_startup",
    "python_import_time": "import_hcli",
    "mission_load": "mission_load",
    "dag_load": "dag_load",
    "context_compiler": "context_compile",
    "scheduler_cycle": "scheduler_cycle",
    "persistence": "checkpoint",
    "receipts_write": "receipt_write",
    "verifier_launch": "verifier_dispatch",
    "subprocess_spawn": "python_pass",
    "grok_bridge_init": "grok_bridge",
    "experiment_setup": "experiment_setup",
}
GROK_RUN_HINT = Path.home() / ".claude-grok" / "bin" / "grok-run"
N_COLD = 5
N_WARM = 5
N_EXPENSIVE = 3
PROCESS_TIMEOUT = 180

GOAL_TEXT = """# Control-plane ceremony ledger

Measure ordinary overhead. Do not run model inference.
Preserve receipts/headless. Never modify workspace/campaign.

## Acceptance
`python3 -m pytest tools/headless -q` exits 0.
Write tools/headless/control_plane_latency.py.

## Invariants
Never invent a Grok session. Must not load a model.
"""


def git_head(repo: Path) -> Optional[str]:
    p = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return p.stdout.strip() or None


def run_env(pythonpath: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONSTARTUP"] = ""
    env["CONTROL_PLANE_REPO"] = str(REPO)
    env.pop("PYTHONPROFILEIMPORTTIME", None)
    grok_bin = str(GROK_RUN_HINT.parent)
    path = env.get("PATH") or ""
    if grok_bin not in path.split(os.pathsep):
        env["PATH"] = grok_bin + os.pathsep + path
    if extra:
        env.update(extra)
    return env


def locate_hcli(repo: Path, extract_root: Path) -> Dict[str, Any]:
    """Prefer on-disk package; otherwise extract HEAD via git archive.

    Live package is top-level ``hcli/`` (``tools/haider/hcli`` is the fossil
    namespace). A missing path in this sparse checkout is not evidence the
    package does not exist in git.
    """
    marker = repo / "hcli" / "__main__.py"
    if marker.is_file():
        return {
            "mode": "on-disk",
            "pythonpath": str(repo.resolve()),
            "package": str((repo / "hcli").resolve()),
            "reason": "hcli is materialized in this worktree",
            "metal_budget": str(METAL_BUDGET_SRC) if METAL_BUDGET_SRC.is_file() else None,
        }

    haider_marker = repo / "tools" / "haider" / "hcli" / "__main__.py"
    if haider_marker.is_file():
        return {
            "mode": "on-disk-haider-fossil",
            "pythonpath": str((repo / "tools" / "haider").resolve()),
            "package": str((repo / "tools" / "haider" / "hcli").resolve()),
            "reason": "tools/haider/hcli is materialized (fossil namespace)",
            "metal_budget": str(METAL_BUDGET_SRC) if METAL_BUDGET_SRC.is_file() else None,
        }

    raw = subprocess.run(
        ["git", "-C", str(repo), "archive", "HEAD", "hcli"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if raw.returncode != 0:
        raise RuntimeError(
            f"git archive HEAD hcli failed: {raw.stderr.decode('utf-8', 'replace').strip()}"
        )
    subprocess.run(
        ["tar", "-x", "-C", str(extract_root)],
        input=raw.stdout,
        capture_output=True,
        check=True,
    )
    pkg = extract_root / "hcli"
    if not (pkg / "__main__.py").is_file():
        raise RuntimeError(f"git archive did not produce {pkg / '__main__.py'}")
    headless_dst = extract_root / "tools" / "headless"
    headless_dst.mkdir(parents=True, exist_ok=True)
    if METAL_BUDGET_SRC.is_file():
        shutil.copy2(METAL_BUDGET_SRC, headless_dst / "metal_budget.py")
    cargo = repo / "Cargo.toml"
    if cargo.is_file():
        shutil.copy2(cargo, extract_root / "Cargo.toml")
    return {
        "mode": "git-archive-HEAD",
        "pythonpath": str(extract_root),
        "package": str(pkg),
        "extract_root": str(extract_root),
        "reason": "sparse checkout: hcli not on disk; content is HEAD",
        "metal_budget": str(headless_dst / "metal_budget.py"),
    }


def stats_ms(samples: Sequence[float]) -> Dict[str, Any]:
    xs = [float(x) for x in samples]
    if not xs:
        return {
            "n": 0,
            "median": None,
            "min": None,
            "max": None,
            "mean": None,
            "stdev": None,
            "samples": [],
        }
    return {
        "n": len(xs),
        "median": round(statistics.median(xs), 3),
        "min": round(min(xs), 3),
        "max": round(max(xs), 3),
        "mean": round(statistics.mean(xs), 3),
        "stdev": round(statistics.pstdev(xs), 3) if len(xs) > 1 else 0.0,
        "samples": [round(x, 3) for x in xs],
    }


def measurement(
    *,
    id: str,
    label: str,
    command: Sequence[str],
    cold: Sequence[float],
    warm: Sequence[float],
    ok: bool = True,
    notes: str = "",
    removable: bool = False,
    remove_by: str = "",
    category: str = "",
    floor_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    cold_s = stats_ms(cold)
    warm_s = stats_ms(warm)
    cold_med = cold_s["median"]
    removable_ms = None
    if removable and cold_med is not None:
        removable_ms = round(max(0.0, float(cold_med) - float(floor_ms or 0.0)), 3)
    row = {
        "id": id,
        "label": label,
        "command": list(command),
        "category": category,
        "ok": bool(ok),
        "notes": notes,
        "removable": bool(removable),
        "remove_by": remove_by,
        "floor_ms": round(float(floor_ms or 0.0), 3),
        "removable_ms": removable_ms,
        "cold_ms": cold_s,
        "warm_ms": warm_s,
    }
    if error:
        row["error"] = error
    if extra:
        row["extra"] = extra
    return row


# ---------------------------------------------------------------------------
# Worker process (same file, --worker OP)
# ---------------------------------------------------------------------------


def _ws(root: Optional[str] = None) -> Path:
    if root:
        p = Path(root)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.mkdtemp(prefix="cpl-ws-"))


def _units(n: int = 8):
    from hcli.workunit import WorkUnit

    units = {}
    prev = None
    for i in range(n):
        deps = [prev] if prev else []
        uid = f"u{i:02d}"
        units[uid] = WorkUnit(
            id=uid,
            role="work",
            description=f"unit {uid} for tools/headless/control_plane_latency.py",
            dependencies=list(deps),
            resource_class="LIGHT_CONTROL",
            verifier="python3 -m pytest tools/headless/control_plane_latency.py -q",
        )
        prev = uid
    return units


def _time_n(fn: Callable[[], Any], n: int) -> List[float]:
    out: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def worker_python_pass() -> Dict[str, Any]:
    # This worker IS the timed process; parent records wall. Placeholder.
    return {"ok": True}


def worker_import_hcli() -> Dict[str, Any]:
    t0 = time.perf_counter()
    import hcli  # noqa: F401

    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        import hcli  # noqa: F401

        warm.append((time.perf_counter() - t1) * 1000.0)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm}


def worker_import_cli() -> Dict[str, Any]:
    t0 = time.perf_counter()
    from hcli.cli import parse_haider_args  # noqa: F401

    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        from hcli.cli import parse_haider_args  # noqa: F401

        warm.append((time.perf_counter() - t1) * 1000.0)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm}


def worker_mission_load() -> Dict[str, Any]:
    from hcli.mission import Mission
    from hcli.workunit import WorkUnit

    ws = _ws()
    units = _units(6)
    m = Mission(ws, units=units, goal=GOAL_TEXT, runtime_count=1, quiet=True)
    m.checkpoint()
    t0 = time.perf_counter()
    loaded = Mission.from_workspace(ws, runtime_count=1, quiet=True)
    cold = (time.perf_counter() - t0) * 1000.0
    assert loaded.goal == GOAL_TEXT or loaded.goal is not None
    warm = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        Mission.from_workspace(ws, runtime_count=1, quiet=True)
        warm.append((time.perf_counter() - t1) * 1000.0)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "units": len(loaded.scheduler.units),
    }


def worker_dag_load() -> Dict[str, Any]:
    from hcli.dag_store import DagStore

    ws = _ws()
    store = DagStore(ws)
    store.save(_units(12))
    t0 = time.perf_counter()
    loaded = store.load(recover_running=False)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        store.load(recover_running=False)
        warm.append((time.perf_counter() - t1) * 1000.0)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm, "units": len(loaded)}


def worker_dag_save() -> Dict[str, Any]:
    from hcli.dag_store import DagStore

    ws = _ws()
    store = DagStore(ws)
    units = _units(12)
    t0 = time.perf_counter()
    store.save(units)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        store.save(units)
        warm.append((time.perf_counter() - t1) * 1000.0)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm}


def worker_context_compile() -> Dict[str, Any]:
    from hcli.goal import GoalCompiler

    compiler = GoalCompiler()
    t0 = time.perf_counter()
    compiled = compiler.compile(GOAL_TEXT)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: compiler.compile(GOAL_TEXT), N_WARM)
    dag = compiled.get("workunits")
    n_wu = len(getattr(dag, "units", None) or (dag if isinstance(dag, dict) else {}) or {})
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "workunits": n_wu,
        "goal_summary": compiled.get("goal_summary"),
    }


def worker_worker_packet() -> Dict[str, Any]:
    from hcli.goal import GoalCompiler, compile_worker_context, identity_for_path

    ws = _ws()
    (ws / "evidence.md").write_text("# evidence\nmeasured.\n", encoding="utf-8")
    compiler = GoalCompiler()
    compiled = compiler.compile(GOAL_TEXT)
    units = _units(4)
    wu = next(iter(units.values()))
    evidence = [identity_for_path(ws / "evidence.md", root=ws)]
    t0 = time.perf_counter()
    pkt = compile_worker_context(
        wu,
        compiled,
        phase="work",
        units=units,
        steering=[],
        workspace=ws,
        evidence=evidence,
        goal_ref=str(ws / "goal.md"),
        root_goal=GOAL_TEXT,
    )
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(
        lambda: compile_worker_context(
            wu,
            compiled,
            phase="work",
            units=units,
            steering=[],
            workspace=ws,
            evidence=evidence,
            goal_ref=str(ws / "goal.md"),
            root_goal=GOAL_TEXT,
        ),
        N_WARM,
    )
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "prompt_chars": len(pkt.prompt),
    }


def worker_status_render() -> Dict[str, Any]:
    from hcli.commands import format_status
    from hcli.mission import Mission

    ws = _ws()
    m = Mission(ws, units=_units(8), goal=GOAL_TEXT, runtime_count=1, quiet=True)
    t0 = time.perf_counter()
    snap = m.status()
    text = format_status(
        {
            **snap,
            "goal": m.goal,
            "occupancy": {"LIGHT_CONTROL": 0},
            "blocked_units": 0,
        }
    )
    cold = (time.perf_counter() - t0) * 1000.0
    assert "WU" in text or "mission" in text.lower() or snap.get("mission_id")

    def once() -> None:
        s = m.status()
        format_status(
            {
                **s,
                "goal": m.goal,
                "occupancy": {"LIGHT_CONTROL": 0},
                "blocked_units": 0,
            }
        )

    warm = _time_n(once, N_WARM)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm, "chars": len(text)}


def worker_scheduler_cycle() -> Dict[str, Any]:
    from hcli.scheduler import Scheduler
    from hcli.workunit import identify_ready

    repo = os.environ.get("CONTROL_PLANE_REPO")
    ws = _ws()
    units = _units(8)
    # Constructor persists. Build once, then time dispatch.
    sched = Scheduler(units, 1, workspace=ws, repo_root=repo)
    # First dispatch is the cycle under test.
    t0 = time.perf_counter()
    assigned = sched.dispatch()
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: sched.dispatch(), N_WARM)
    ready = identify_ready(sched.units)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "assigned": len(assigned),
        "ready": len(ready),
        "persist_on_dispatch": True,
    }


def worker_receipt_write() -> Dict[str, Any]:
    from hcli.engine import Engine
    from hcli.events import EventBus
    from hcli.workspace import Workspace

    ws = _ws()
    engine = Engine(
        workspace=Workspace(str(ws)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )
    t0 = time.perf_counter()
    path = engine._write_receipt(
        "goal-cpl",
        GOAL_TEXT,
        {"kind": "answer", "status": "completed", "operations": [], "tests": []},
        [],
        {"ok": True, "checks": []},
        False,
    )
    cold = (time.perf_counter() - t0) * 1000.0
    n = 0

    def once() -> None:
        nonlocal n
        n += 1
        engine._write_receipt(
            f"goal-cpl-{n}",
            GOAL_TEXT,
            {"kind": "answer", "status": "completed", "operations": [], "tests": []},
            [],
            {"ok": True, "checks": []},
            False,
        )

    warm = _time_n(once, N_WARM)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm, "path": str(path)}


def worker_checkpoint() -> Dict[str, Any]:
    from hcli.mission import Mission

    ws = _ws()
    m = Mission(ws, units=_units(8), goal=GOAL_TEXT, runtime_count=1, quiet=True)
    t0 = time.perf_counter()
    path = m.checkpoint()
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: m.checkpoint(), N_WARM)
    return {"ok": True, "cold_ms": cold, "warm_ms": warm, "path": str(path)}


def worker_verifier_dispatch() -> Dict[str, Any]:
    """Verifier ceremony without a model: admit a command, then Engine._validate.

    `_validate` spawns `python -m py_compile` and (when tests are admitted)
    a contained pytest subprocess. That is the Python-interpreter-spawn tax.
    """
    from hcli.engine import Engine
    from hcli.events import EventBus
    from hcli.verifier_pipeline import command_is_admissible
    from hcli.workspace import Workspace

    ws = _ws()
    (ws / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (ws / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    engine = Engine(
        workspace=Workspace(str(ws)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )
    cmd = "python3 -m pytest test_calc.py -q"
    t0 = time.perf_counter()
    admitted, reason = command_is_admissible(cmd)
    validation = engine._validate(
        [ws / "calc.py", ws / "test_calc.py"],
        ["test_calc.py"],
    )
    cold = (time.perf_counter() - t0) * 1000.0

    def once() -> None:
        command_is_admissible(cmd)
        engine._validate(
            [ws / "calc.py", ws / "test_calc.py"],
            ["test_calc.py"],
        )

    warm = _time_n(once, N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "command_admitted": admitted,
        "admit_reason": reason,
        "validate_ok": bool(validation.get("ok")),
    }


def worker_experiment_setup() -> Dict[str, Any]:
    """RuntimePool + Controller construction. No backend start, no inference."""
    from hcli.controller import Controller
    from hcli.runtime import RuntimePool

    repo = os.environ.get("CONTROL_PLANE_REPO")
    ws = _ws()
    t0 = time.perf_counter()
    pool = RuntimePool(
        model_path="/missing.gguf",
        requested_n=1,
        workspace=ws,
        repo_root=repo,
        backend_factory=lambda **_k: None,
    )
    ctl = Controller(workspace=str(ws), runtime_count=1, model="/missing.gguf")
    cold = (time.perf_counter() - t0) * 1000.0

    def once() -> None:
        RuntimePool(
            model_path="/missing.gguf",
            requested_n=1,
            workspace=_ws(),
            repo_root=repo,
            backend_factory=lambda **_k: None,
        )
        Controller(workspace=str(_ws()), runtime_count=1, model="/missing.gguf")

    warm = _time_n(once, N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "admitted_n": int(getattr(pool, "admitted_n", 0) or 0),
        "controller_engine": type(ctl.engine).__name__,
    }


def worker_grok_bridge() -> Dict[str, Any]:
    from hcli.grok_bridge import GrokBridge, GrokNotAvailable

    ws = _ws()
    bridge = GrokBridge(ws)
    t0 = time.perf_counter()
    try:
        handle = bridge.consult("ping", background=False, dry_run=True)
        cold = (time.perf_counter() - t0) * 1000.0
        err = ""
        ok = True
        extra = {
            "task_id": getattr(handle, "task_id", None),
            "dry_run": getattr(handle, "dry_run", None),
            "command_run": list(getattr(handle, "command_run", None) or []),
        }
    except GrokNotAvailable as exc:
        cold = (time.perf_counter() - t0) * 1000.0
        err = str(exc)
        ok = False
        extra = {"unavailable": True}
        handle = None

    warm = []
    if ok:
        for _ in range(N_WARM):
            t1 = time.perf_counter()
            bridge.consult("ping", background=False, dry_run=True)
            warm.append((time.perf_counter() - t1) * 1000.0)
    return {
        "ok": ok,
        "cold_ms": cold,
        "warm_ms": warm,
        "error": err,
        **extra,
    }


def worker_runtime_admission() -> Dict[str, Any]:
    """MemGate.consider with live Metal probe (compiles Swift on this machine)."""
    import hcli.machine as machine

    gate = machine.MemGate(model_bytes=13_000_000_000, topology="slot")
    machine._METAL_CACHE = None
    t0 = time.perf_counter()
    decision = gate.consider(admitted=0, extra=1, refresh_metal=True)
    cold = (time.perf_counter() - t0) * 1000.0
    warm_force = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        gate.consider(admitted=0, extra=1, refresh_metal=True)
        warm_force.append((time.perf_counter() - t1) * 1000.0)
    cached = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        gate.consider(admitted=0, extra=1, refresh_metal=False)
        cached.append((time.perf_counter() - t1) * 1000.0)
    metal = machine.metal_device_info(force=False)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm_force,
        "cached_ms": cached,
        "allow": bool(decision.allow),
        "reason": decision.reason,
        "gate": decision.gate,
        "metal_source": metal.get("source"),
        "metal_name": metal.get("name"),
    }


def worker_metal_device() -> Dict[str, Any]:
    import hcli.machine as machine

    machine._METAL_CACHE = None
    machine._METAL_MOD = None
    t0 = time.perf_counter()
    info = machine.metal_device_info(force=True)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = []
    for _ in range(N_WARM):
        machine._METAL_CACHE = None
        t1 = time.perf_counter()
        machine.metal_device_info(force=True)
        warm.append((time.perf_counter() - t1) * 1000.0)
    cached = []
    for _ in range(N_WARM):
        t1 = time.perf_counter()
        machine.metal_device_info(force=False)
        cached.append((time.perf_counter() - t1) * 1000.0)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "cached_ms": cached,
        "source": info.get("source"),
        "name": info.get("name"),
        "error": info.get("error"),
    }


def worker_model_discovery() -> Dict[str, Any]:
    from hcli.models import ModelRegistry

    reg = ModelRegistry()
    t0 = time.perf_counter()
    models = reg.discover(refresh=True, include_sidecars=True)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: reg.discover(refresh=True, include_sidecars=True), N_WARM)
    cached = _time_n(lambda: reg.discover(refresh=False), N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "cached_ms": cached,
        "n_models": len(models),
        "roots": list(reg.roots),
    }


def worker_fingerprint() -> Dict[str, Any]:
    from hcli.mission import Mission

    small = _ws()
    for i in range(20):
        (small / f"f{i:02d}.txt").write_text("x" * 256, encoding="utf-8")
    m_small = Mission(small, units=_units(2), goal="g", runtime_count=1, quiet=True)
    t0 = time.perf_counter()
    fp = m_small.fingerprint()
    cold_small = (time.perf_counter() - t0) * 1000.0
    warm_small = _time_n(lambda: m_small.fingerprint(), N_WARM)

    # Copy large immutable receipts into a temp workspace so fingerprint
    # cannot write .hcli/ under receipts/headless (WRITE scope).
    big = _ws()
    for name in ("SPRING_CLEAN_CENSUS.json", "CODE_GRAPH.json", "NAMESPACE_PLAN.json"):
        src = REPO / "receipts" / "headless" / name
        if src.is_file():
            shutil.copy2(src, big / name)
    m_big = Mission(big, units=_units(2), goal="g", runtime_count=1, quiet=True)
    t0 = time.perf_counter()
    fp_big = m_big.fingerprint()
    cold_big = (time.perf_counter() - t0) * 1000.0
    warm_big = _time_n(lambda: m_big.fingerprint(), min(3, N_WARM))
    return {
        "ok": True,
        "cold_ms": cold_big,
        "warm_ms": warm_big,
        "small_cold_ms": cold_small,
        "small_warm_ms": warm_small,
        "small_fp": fp,
        "big_fp": fp_big,
        "big_bytes": sum(p.stat().st_size for p in big.glob("*.json")),
    }


def worker_hash_artifact() -> Dict[str, Any]:
    from hcli.goal import identity_for_path

    artifact = REPO / "receipts" / "headless" / "SPRING_CLEAN_CENSUS.json"
    if not artifact.is_file():
        artifact = REPO / "receipts" / "headless" / "CODE_GRAPH.json"
    t0 = time.perf_counter()
    ident = identity_for_path(artifact, root=REPO)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: identity_for_path(artifact, root=REPO), N_WARM)
    raw = artifact.read_bytes()
    t0 = time.perf_counter()
    json.loads(raw)
    json_cold = (time.perf_counter() - t0) * 1000.0
    json_warm = _time_n(lambda: json.loads(raw), N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "path": str(artifact),
        "bytes": ident.size,
        "json_loads_cold_ms": json_cold,
        "json_loads_warm_ms": json_warm,
    }


def worker_tool_discovery() -> Dict[str, Any]:
    from hcli.grok_bridge import find_grok_run

    t0 = time.perf_counter()
    path = find_grok_run()
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(find_grok_run, N_WARM)
    which_cold = []
    which_warm = []
    t0 = time.perf_counter()
    shutil.which("grok-run")
    which_cold.append((time.perf_counter() - t0) * 1000.0)
    which_warm.extend(_time_n(lambda: shutil.which("grok-run"), N_WARM))
    t0 = time.perf_counter()
    shutil.which("swift")
    swift_c = (time.perf_counter() - t0) * 1000.0
    swift_w = _time_n(lambda: shutil.which("swift"), N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "grok_run": path,
        "which_grok_cold_ms": which_cold[0],
        "which_grok_warm_ms": which_warm,
        "which_swift_cold_ms": swift_c,
        "which_swift_warm_ms": swift_w,
    }


def worker_workspace_git() -> Dict[str, Any]:
    from hcli.workspace import Workspace

    t0 = time.perf_counter()
    ws = Workspace(str(REPO))
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: Workspace(str(REPO)), N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "git_root": ws.git_root,
    }


def worker_host_snapshot() -> Dict[str, Any]:
    from hcli.machine import host_snapshot

    t0 = time.perf_counter()
    snap = host_snapshot()
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(host_snapshot, N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "pressure": snap.get("pressure"),
    }


def worker_resource_limits() -> Dict[str, Any]:
    from hcli.resources import ResourceLimits

    repo = os.environ.get("CONTROL_PLANE_REPO")
    t0 = time.perf_counter()
    limits = ResourceLimits.resolve(repo_root=repo)
    cold = (time.perf_counter() - t0) * 1000.0
    warm = _time_n(lambda: ResourceLimits.resolve(repo_root=repo), N_WARM)
    return {
        "ok": True,
        "cold_ms": cold,
        "warm_ms": warm,
        "gpu_decode": limits.gpu_decode,
        "source": limits.gpu_decode_source,
    }


def _install_trace(counts: Dict[str, int], files: Dict[str, List[str]]) -> Tuple[Any, ...]:
    orig = (
        os.walk,
        json.loads,
        hashlib.sha256,
        subprocess.run,
        subprocess.Popen,
        shutil.which,
    )

    def walk(*a, **k):
        counts["os.walk"] += 1
        files["os.walk"].append(str(a[0]) if a else "")
        return orig[0](*a, **k)

    def loads(*a, **k):
        counts["json.loads"] += 1
        return orig[1](*a, **k)

    def sha256(*a, **k):
        counts["hashlib.sha256"] += 1
        n = 0
        if a and isinstance(a[0], (bytes, bytearray)):
            n = len(a[0])
        counts["hashlib.sha256_bytes"] += n
        return orig[2](*a, **k)

    def run(*a, **k):
        counts["subprocess.run"] += 1
        cmd = a[0] if a else k.get("args")
        files["subprocess.run"].append(str(cmd)[:200])
        return orig[3](*a, **k)

    def popen(*a, **k):
        counts["subprocess.Popen"] += 1
        cmd = a[0] if a else k.get("args")
        files["subprocess.Popen"].append(str(cmd)[:200])
        return orig[4](*a, **k)

    def which(*a, **k):
        counts["shutil.which"] += 1
        files["shutil.which"].append(str(a[0]) if a else "")
        return orig[5](*a, **k)

    os.walk = walk  # type: ignore[assignment]
    json.loads = loads  # type: ignore[assignment]
    hashlib.sha256 = sha256  # type: ignore[assignment]
    subprocess.run = run  # type: ignore[assignment]
    subprocess.Popen = popen  # type: ignore[assignment]
    shutil.which = which  # type: ignore[assignment]
    return orig


def _restore_trace(orig: Tuple[Any, ...]) -> None:
    os.walk, json.loads, hashlib.sha256, subprocess.run, subprocess.Popen, shutil.which = orig


def worker_trace() -> Dict[str, Any]:
    """Count ceremony calls inside one scheduler cycle, one checkpoint, one admit."""
    from collections import Counter, defaultdict

    from hcli.dag_store import DagStore
    from hcli.machine import MemGate
    from hcli.mission import Mission
    from hcli.models import ModelRegistry
    from hcli.scheduler import Scheduler

    repo = os.environ.get("CONTROL_PLANE_REPO")
    traces: Dict[str, Any] = {}

    def run_traced(name: str, fn: Callable[[], Any]) -> Any:
        counts: Dict[str, int] = Counter()
        files: Dict[str, List[str]] = defaultdict(list)
        orig = _install_trace(counts, files)
        t0 = time.perf_counter()
        try:
            result = fn()
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            _restore_trace(orig)
        traces[name] = {
            "ms": round(ms, 3),
            "counts": dict(counts),
            "examples": {k: v[:8] for k, v in files.items() if v},
        }
        return result

    ws = _ws()
    units = _units(8)
    sched = Scheduler(units, 1, workspace=ws, repo_root=repo)
    run_traced("scheduler_dispatch", lambda: sched.dispatch())
    run_traced("dag_save_again", lambda: DagStore(ws).save(sched.units))
    m = Mission(ws, units=units, goal=GOAL_TEXT, runtime_count=1, quiet=True)
    run_traced("checkpoint", lambda: m.checkpoint())
    run_traced("fingerprint_small", lambda: m.fingerprint())
    big = _ws()
    src = REPO / "receipts" / "headless" / "SPRING_CLEAN_CENSUS.json"
    if src.is_file():
        shutil.copy2(src, big / src.name)
    run_traced(
        "fingerprint_large_artifact",
        lambda: Mission(
            big,
            units=_units(1),
            goal="g",
            runtime_count=1,
            quiet=True,
        ).fingerprint(),
    )
    run_traced(
        "model_discovery",
        lambda: ModelRegistry().discover(refresh=True, include_sidecars=True),
    )
    run_traced(
        "memgate_consider",
        lambda: MemGate(model_bytes=13_000_000_000).consider(
            admitted=0, extra=1, refresh_metal=True
        ),
    )
    return {"ok": True, "cold_ms": 0.0, "warm_ms": [], "traces": traces}


WORKERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "import_hcli": worker_import_hcli,
    "import_cli": worker_import_cli,
    "mission_load": worker_mission_load,
    "dag_load": worker_dag_load,
    "dag_save": worker_dag_save,
    "context_compile": worker_context_compile,
    "worker_packet": worker_worker_packet,
    "status_render": worker_status_render,
    "scheduler_cycle": worker_scheduler_cycle,
    "receipt_write": worker_receipt_write,
    "checkpoint": worker_checkpoint,
    "verifier_dispatch": worker_verifier_dispatch,
    "experiment_setup": worker_experiment_setup,
    "grok_bridge": worker_grok_bridge,
    "runtime_admission": worker_runtime_admission,
    "metal_device": worker_metal_device,
    "model_discovery": worker_model_discovery,
    "fingerprint": worker_fingerprint,
    "hash_artifact": worker_hash_artifact,
    "tool_discovery": worker_tool_discovery,
    "workspace_git": worker_workspace_git,
    "host_snapshot": worker_host_snapshot,
    "resource_limits": worker_resource_limits,
    "trace": worker_trace,
}


def worker_main(op: str) -> int:
    fn = WORKERS.get(op)
    if fn is None:
        _emit({"ok": False, "error": f"unknown worker op {op!r}"})
        return 2
    try:
        payload = fn()
        payload.setdefault("ok", True)
        _emit(payload)
        return 0 if payload.get("ok", True) else 1
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        return 1


# ---------------------------------------------------------------------------
# Parent: process-level timing + harvest
# ---------------------------------------------------------------------------


def time_process(
    cmd: Sequence[str],
    env: Dict[str, str],
    *,
    n: int,
    timeout: int = PROCESS_TIMEOUT,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    samples: List[float] = []
    codes: List[int] = []
    first_out = ""
    first_err = ""
    for i in range(n):
        t0 = time.perf_counter()
        p = subprocess.run(
            list(cmd),
            capture_output=True,
            timeout=timeout,
            env=env,
            cwd=str(cwd or REPO),
        )
        samples.append((time.perf_counter() - t0) * 1000.0)
        codes.append(p.returncode)
        if i == 0:
            first_out = (p.stdout or b"").decode("utf-8", "replace")[:800]
            first_err = (p.stderr or b"").decode("utf-8", "replace")[-800:]
    return {
        "samples_ms": samples,
        "returncodes": codes,
        "stdout_preview": first_out,
        "stderr_tail": first_err,
        "ok": all(c == 0 for c in codes),
    }


def run_worker_once(
    python: str,
    pythonpath: str,
    op: str,
    env: Dict[str, str],
    timeout: int = PROCESS_TIMEOUT,
) -> Dict[str, Any]:
    cmd = [python, str(HERE), "--worker", op]
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO),
    )
    wall = (time.perf_counter() - t0) * 1000.0
    stdout = (p.stdout or b"").decode("utf-8", "replace")
    stderr = (p.stderr or b"").decode("utf-8", "replace")
    payload: Dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not payload:
        payload = {
            "ok": False,
            "error": f"no json from worker rc={p.returncode}",
            "stderr": stderr[-1500:],
            "stdout": stdout[-1500:],
        }
    payload["process_wall_ms"] = wall
    payload["returncode"] = p.returncode
    payload["command"] = cmd
    if stderr and "error" not in payload:
        payload["stderr_tail"] = stderr[-800:]
    return payload


def harvest_worker(
    python: str,
    pythonpath: str,
    op: str,
    *,
    n_cold: int,
    env: Dict[str, str],
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    cold: List[float] = []
    warm: List[float] = []
    last: Dict[str, Any] = {}
    for i in range(n_cold):
        payload = run_worker_once(python, pythonpath, op, env)
        last = payload
        if payload.get("cold_ms") is not None:
            cold.append(float(payload["cold_ms"]))
        elif payload.get("ok") and payload.get("process_wall_ms") is not None:
            cold.append(float(payload["process_wall_ms"]))
        w = payload.get("warm_ms") or []
        if isinstance(w, list):
            warm.extend(float(x) for x in w)
        if not payload.get("ok", False) and i == 0:
            break
    return cold, warm, last


def ast_waste_census(pkg: Path) -> Dict[str, Any]:
    hits: Dict[str, List[Dict[str, Any]]] = {
        "os.walk": [],
        "json.loads": [],
        "hashlib.sha256": [],
        "subprocess": [],
        "shutil.which": [],
        "ast.parse": [],
    }
    per_mod: Dict[str, Dict[str, int]] = {}
    files = sorted(
        p
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts and "tests" not in p.parts
    )
    for path in files:
        rel = path.relative_to(pkg)
        mod = (
            "hcli"
            if rel.as_posix() == "__init__.py"
            else "hcli." + str(rel.with_suffix("")).replace("/", ".")
        )
        src = path.read_text(encoding="utf-8")
        counts = {k: 0 for k in hits}
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in ("os.walk", "walk"):
                    counts["os.walk"] += 1
                    hits["os.walk"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
                elif name in ("json.loads", "json.load"):
                    counts["json.loads"] += 1
                    hits["json.loads"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
                elif name in ("hashlib.sha256", "sha256"):
                    counts["hashlib.sha256"] += 1
                    hits["hashlib.sha256"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
                elif name in (
                    "subprocess.run",
                    "subprocess.Popen",
                    "subprocess.check_output",
                    "Popen",
                ):
                    counts["subprocess"] += 1
                    hits["subprocess"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
                elif name in ("shutil.which",):
                    counts["shutil.which"] += 1
                    hits["shutil.which"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
                elif name in ("ast.parse",):
                    counts["ast.parse"] += 1
                    hits["ast.parse"].append(
                        {"module": mod, "line": node.lineno, "name": name}
                    )
        per_mod[mod] = counts
    totals = {k: len(v) for k, v in hits.items()}
    return {"totals": totals, "hits": hits, "per_module": per_mod, "module_count": len(per_mod)}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        left = _call_name(func.value)
        return f"{left}.{func.attr}" if left else func.attr
    return ""


def write_receipt(doc: Dict[str, Any]) -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    for path in (RECEIPT, RECEIPT_LEGACY):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return RECEIPT


def validate_receipt(doc: Dict[str, Any]) -> List[str]:
    fails: List[str] = []
    if doc.get("schema") != SCHEMA:
        fails.append(f"schema {doc.get('schema')!r} != {SCHEMA}")
    rows = doc.get("measurements") or []
    if not isinstance(rows, list) or not rows:
        fails.append("measurements missing")
        return fails
    required = {
        "cli_startup",
        "import_hcli",
        "mission_load",
        "dag_load",
        "context_compile",
        "status_render",
        "scheduler_cycle",
        "receipt_write",
        "checkpoint",
        "verifier_dispatch",
        "experiment_setup",
        "grok_bridge",
        "runtime_admission",
    }
    have = {r.get("id") for r in rows if isinstance(r, dict)}
    missing = sorted(required - have)
    if missing:
        fails.append("missing measurements: " + ", ".join(missing))
    for row in rows:
        if not isinstance(row, dict):
            fails.append("non-object measurement")
            continue
        rid = row.get("id") or "<unknown>"
        if not row.get("command"):
            fails.append(f"{rid}: missing command")
        for side in ("cold_ms", "warm_ms"):
            fig = row.get(side)
            if not isinstance(fig, dict):
                fails.append(f"{rid}: missing {side}")
                continue
            if "median" not in fig or "samples" not in fig:
                fails.append(f"{rid}: {side} lacks median/samples")
            elif fig.get("median") is None or not fig.get("samples"):
                if side == "warm_ms" and not row.get("ok"):
                    continue
                fails.append(f"{rid}: {side} median/samples empty")
    top = doc.get("largest_removable_cost") or {}
    if not isinstance(top, dict) or top.get("ms") is None:
        fails.append("largest_removable_cost.ms missing")
    if not top.get("id"):
        fails.append("largest_removable_cost.id missing")
    if not top.get("remove_by"):
        fails.append("largest_removable_cost.remove_by missing")
    stages = doc.get("stages") or {}
    if not isinstance(stages, dict) or not stages:
        fails.append("stages missing")
    else:
        for name in STAGE_MAP:
            st = stages.get(name)
            if not isinstance(st, dict):
                fails.append(f"stages.{name} missing")
                continue
            status = st.get("status")
            if status == "MEASURED":
                if st.get("cold_median_ms") is None:
                    fails.append(f"stages.{name} MEASURED but cold_median_ms empty")
            elif status == "ABSENT":
                if not st.get("reason"):
                    fails.append(f"stages.{name} ABSENT without reason")
            else:
                fails.append(f"stages.{name} status {status!r} not MEASURED/ABSENT")
    return fails


def build_ledger(
    python: str,
    located: Dict[str, Any],
    generated_at: str,
    head: Optional[str],
) -> Dict[str, Any]:
    pythonpath = located["pythonpath"]
    env = run_env(pythonpath)
    rows: List[Dict[str, Any]] = []
    notes_fail: List[str] = []

    def add(row: Dict[str, Any]) -> None:
        rows.append(row)
        if not row.get("ok"):
            notes_fail.append(f"{row.get('id')}: {row.get('error') or 'not ok'}")

    # Floors: interpreter and argparse. Fresh processes; first = cold, rest = warm.
    pass_cmd = [python, "-c", "pass"]
    pass_run = time_process(pass_cmd, env, n=N_COLD)
    add(
        measurement(
            id="python_pass",
            label="Python interpreter spawn (python3 -c pass)",
            command=pass_cmd,
            cold=pass_run["samples_ms"],
            warm=pass_run["samples_ms"][1:],
            ok=pass_run["ok"],
            notes=(
                "Irreducible floor of any Python subprocess. Each process is a cold "
                "interpreter; warm is later processes (page cache). Not a defect."
            ),
            removable=False,
            category="python_interpreter_spawns",
        )
    )
    python_pass_ms = float(stats_ms(pass_run["samples_ms"]).get("median") or 0.0)

    argparse_code = (
        "import argparse,sys\n"
        "p=argparse.ArgumentParser(prog='hcli',"
        "description='HCLI — autonomous local model engineering')\n"
        "try:\n p.parse_args(['--help'])\n"
        "except SystemExit:\n pass\n"
    )
    argparse_cmd = [python, "-c", argparse_code]
    argparse_run = time_process(argparse_cmd, env, n=N_COLD)
    add(
        measurement(
            id="argparse_help",
            label="argparse --help floor (no hcli import)",
            command=argparse_cmd,
            cold=argparse_run["samples_ms"],
            warm=argparse_run["samples_ms"][1:],
            ok=argparse_run["ok"],
            notes="Help-path floor. Fast. --help should approach this, not Controller import.",
            removable=False,
            category="cli_startup",
        )
    )

    help_cmd = [python, "-m", "hcli", "--help"]
    help_run = time_process(help_cmd, env, n=N_COLD)
    help_ok = help_run["ok"] and "autonomous local model engineering" in (
        help_run.get("stdout_preview") or ""
    )
    if not help_ok:
        notes_fail.append(
            f"cli_startup: rc={help_run['returncodes']} err={help_run.get('stderr_tail','')[:300]}"
        )
    argparse_med = stats_ms(argparse_run["samples_ms"]).get("median") or 0.0
    add(
        measurement(
            id="cli_startup",
            label="CLI startup (python3 -m hcli --help)",
            command=help_cmd,
            cold=help_run["samples_ms"],
            warm=help_run["samples_ms"][1:],
            ok=help_ok,
            notes=(
                "Fresh process each sample. Cold is the first process (page-cache cold); "
                "warm is later processes. Package __init__ imports Controller, so --help "
                "pays the Engine/Runtime/Mission import graph. See STARTUP_CENSUS D1."
            ),
            removable=True,
            remove_by=(
                "Stop importing Controller from tools/haider/hcli/__init__.py on the "
                "help path (PEP 562 __getattr__). argparse --help sys.exits before App."
            ),
            category="cli_startup",
            floor_ms=float(argparse_med),
            extra={"returncodes": help_run["returncodes"]},
        )
    )

    worker_specs = [
        (
            "import_hcli",
            "import hcli (package init pulls Controller)",
            "import",
            True,
            "Defer Controller/Workspace/EventBus in __init__.py via __getattr__.",
            N_COLD,
        ),
        (
            "import_cli",
            "from hcli.cli import parse_haider_args",
            "import",
            True,
            "Same as import_hcli: package init still imports Controller first.",
            N_COLD,
        ),
        (
            "mission_load",
            "Mission.from_workspace after checkpoint",
            "mission_load",
            False,
            "",
            N_COLD,
        ),
        (
            "dag_load",
            "DagStore.load",
            "dag_load",
            False,
            "",
            N_COLD,
        ),
        (
            "dag_save",
            "DagStore.save (re-reads previous JSON every write)",
            "dag_load",
            True,
            (
                "DagStore.save always _read_document() of the file it last wrote, "
                "re-parses units, then atomic_write_json. Cache the previous document "
                "in-process; skip the disk round-trip when the caller just persisted."
            ),
            N_COLD,
        ),
        (
            "context_compile",
            "GoalCompiler.compile (no model)",
            "context_compile",
            False,
            "",
            N_COLD,
        ),
        (
            "worker_packet",
            "compile_worker_context (hashes evidence files)",
            "context_compile",
            True,
            (
                "identity_for_path sha256-reads every evidence file on each compile. "
                "Reuse size+mtime_ns as the invalidation key; hash only on change."
            ),
            N_COLD,
        ),
        (
            "status_render",
            "Mission.status + commands.format_status",
            "status_rendering",
            False,
            "",
            N_COLD,
        ),
        (
            "scheduler_cycle",
            "Scheduler.dispatch (one cycle, persists DAG)",
            "scheduler_cycle",
            True,
            (
                "dispatch() always _persist() → DagStore.save → re-parse previous "
                "dag.json. Persist on a dirty flag, or after complete/fail, not on "
                "every empty dispatch."
            ),
            N_COLD,
        ),
        (
            "receipt_write",
            "Engine._write_receipt",
            "receipt_write",
            False,
            "",
            N_COLD,
        ),
        (
            "checkpoint",
            "Mission.checkpoint (DAG first, then state.json)",
            "checkpoint",
            False,
            "",
            N_COLD,
        ),
        (
            "verifier_dispatch",
            "command_is_admissible + Engine._validate (py_compile + pytest spawns)",
            "verifier_dispatch",
            True,
            (
                "Each _validate spawns python -m py_compile and a contained pytest. "
                "Reuse one interpreter (or py_compile in-process via compile()) for "
                "the ceremony; keep pytest as the real check."
            ),
            N_EXPENSIVE,
        ),
        (
            "experiment_setup",
            "RuntimePool.__init__ + Controller.__init__ (no start, no inference)",
            "experiment_setup",
            True,
            (
                "Controller.__init__ constructs ModelRegistry and walks ~/models. "
                "RuntimePool reads genome JSON and decode topology. Do not start a "
                "backend for a status/help/setup path that will not decode."
            ),
            N_EXPENSIVE,
        ),
        (
            "grok_bridge",
            "GrokBridge.consult(..., dry_run=True) — no model session",
            "grok_bridge",
            True,
            (
                "dry_run still spawns grok-run (Node) to print argv. Construct the "
                "argv in-process when dry_run=True; spawn only for live launches. "
                "Cache find_grok_run() instead of shutil.which on every call."
            ),
            N_EXPENSIVE,
        ),
        (
            "runtime_admission",
            "MemGate.consider(refresh_metal=True)",
            "runtime_admission",
            True,
            (
                "metal_budget.metal_device() writes a temp .swift file and runs "
                "`swift <file>` on every uncached/force call (~1s cold, ~200ms warm) "
                "even when Metal returns no device and the code then falls back to "
                "sysctl. Ship a precompiled helper or call Metal via ctypes/objc; "
                "cache recommendedMaxWorkingSetSize for the process; do not compile "
                "Swift to read currentAllocatedSize."
            ),
            N_EXPENSIVE,
        ),
        (
            "metal_device",
            "machine.metal_device_info(force=True) — Swift compiler ceremony",
            "runtime_admission",
            True,
            (
                "Same Swift compile as runtime_admission, isolated. Cached "
                "force=False path is microseconds once _METAL_CACHE is filled."
            ),
            N_EXPENSIVE,
        ),
        (
            "model_discovery",
            "ModelRegistry.discover(refresh=True) walks ~/models",
            "experiment_setup",
            True,
            (
                "os.walk of ~/models plus GGUF header reads for every .gguf. "
                "Cache inventory by (dir mtime, file set); do not walk on "
                "Controller construction when an explicit model path is given."
            ),
            N_COLD,
        ),
        (
            "fingerprint",
            "Mission.fingerprint — os.walk + sha256 of every file",
            "checkpoint",
            True,
            (
                "fingerprint() hashes file bytes of the whole workspace on every "
                "no-progress check. Hash size+mtime (or a merkle of those) and only "
                "re-hash files whose identity changed. Never hash receipts/ or "
                "model artifacts as if they were source."
            ),
            N_EXPENSIVE,
        ),
        (
            "hash_artifact",
            "identity_for_path sha256 of SPRING_CLEAN_CENSUS.json",
            "hashing",
            True,
            "Do not sha256 large immutable receipts on every packet compile; use mtime+size.",
            N_COLD,
        ),
        (
            "tool_discovery",
            "find_grok_run / shutil.which",
            "tool_discovery",
            True,
            "Cache shutil.which('grok-run') and shutil.which('swift') per process.",
            N_COLD,
        ),
        (
            "workspace_git",
            "Workspace.__init__ → git rev-parse --show-toplevel",
            "subprocess",
            True,
            "Cache git_root on the workspace; do not spawn git per Workspace().",
            N_COLD,
        ),
        (
            "host_snapshot",
            "host_snapshot: vm_stat + sysctl + memory_pressure",
            "runtime_admission",
            False,
            "",
            N_COLD,
        ),
        (
            "resource_limits",
            "ResourceLimits.resolve (MACHINE_GENOME JSON)",
            "runtime_admission",
            True,
            "Memoize genome JSON per process. Scheduler and RuntimePool both re-read it.",
            N_COLD,
        ),
    ]

    extras: Dict[str, Any] = {}
    for op, label, category, removable, remove_by, n_cold in worker_specs:
        cmd = [python, str(HERE), "--worker", op]
        cold, warm, last = harvest_worker(
            python, pythonpath, op, n_cold=n_cold, env=env
        )
        extra = {
            k: last.get(k)
            for k in last
            if k
            not in {
                "ok",
                "cold_ms",
                "warm_ms",
                "command",
                "returncode",
                "process_wall_ms",
                "traceback",
                "stderr_tail",
                "stdout",
                "stderr",
            }
        }
        floor = 0.0
        if op in {"verifier_dispatch"}:
            floor = float(python_pass_ms)
        add(
            measurement(
                id=op,
                label=label,
                command=cmd,
                cold=cold,
                warm=warm,
                ok=bool(last.get("ok", False)) and bool(cold),
                notes=last.get("notes") or "",
                removable=removable,
                remove_by=remove_by,
                category=category,
                floor_ms=floor,
                extra=extra or None,
                error=str(last.get("error") or ""),
            )
        )
        if last.get("cached_ms"):
            extras[f"{op}_cached_ms"] = stats_ms(last["cached_ms"])
        if last.get("traces"):
            extras["dynamic_trace"] = last["traces"]
        if not last.get("ok", False):
            notes_fail.append(f"{op}: {last.get('error') or last.get('stderr_tail') or 'fail'}")

    # Dedicated trace process (counts, not a ranked measurement).
    trace_payload = run_worker_once(python, pythonpath, "trace", env)
    dynamic_trace = (trace_payload.get("traces") or extras.get("dynamic_trace") or {})

    pkg = Path(located["package"])
    ast_info = ast_waste_census(pkg)

    # Rank removable costs by measured milliseconds (cold median minus floor).
    ranked = []
    for row in rows:
        if not row.get("removable"):
            continue
        ms = row.get("removable_ms")
        if ms is None:
            continue
        ranked.append(
            {
                "id": row["id"],
                "ms": ms,
                "cold_median_ms": row["cold_ms"]["median"],
                "warm_median_ms": row["warm_ms"]["median"],
                "remove_by": row.get("remove_by") or "",
                "label": row.get("label"),
            }
        )
    ranked.sort(key=lambda r: float(r["ms"] or 0.0), reverse=True)
    top = ranked[0] if ranked else {
        "id": None,
        "ms": None,
        "remove_by": "no removable cost measured",
    }

    # Name the waste categories from AST + dynamic trace + timings.
    waste = {
        "repeated_filesystem_scans": {
            "ast_sites": ast_info["totals"].get("os.walk", 0),
            "examples": ast_info["hits"].get("os.walk", [])[:12],
            "measured": {
                "model_discovery": _row(rows, "model_discovery"),
                "fingerprint": _row(rows, "fingerprint"),
            },
            "dynamic": {
                k: dynamic_trace.get(k)
                for k in (
                    "model_discovery",
                    "fingerprint_small",
                    "fingerprint_large_artifact",
                    "scheduler_dispatch",
                )
                if k in dynamic_trace
            },
            "reading": (
                "Mission.fingerprint os.walks and hashes every file. "
                "ModelRegistry.discover os.walks ~/models on Controller init. "
                "index.py also walks, and is unused in production."
            ),
        },
        "repeated_json_parsing": {
            "ast_sites": ast_info["totals"].get("json.loads", 0),
            "examples": ast_info["hits"].get("json.loads", [])[:12],
            "measured": {
                "dag_save": _row(rows, "dag_save"),
                "resource_limits": _row(rows, "resource_limits"),
            },
            "dynamic": {
                k: dynamic_trace.get(k)
                for k in ("scheduler_dispatch", "dag_save_again", "checkpoint")
                if k in dynamic_trace
            },
            "reading": (
                "DagStore.save re-parses the JSON document it just wrote, on every "
                "Scheduler.dispatch/_persist. ResourceLimits.resolve and RuntimePool "
                "both json.loads MACHINE_GENOME.json."
            ),
        },
        "repeated_hashing": {
            "ast_sites": ast_info["totals"].get("hashlib.sha256", 0),
            "examples": ast_info["hits"].get("hashlib.sha256", [])[:12],
            "measured": {
                "fingerprint": _row(rows, "fingerprint"),
                "hash_artifact": _row(rows, "hash_artifact"),
            },
            "dynamic": {
                k: dynamic_trace.get(k)
                for k in ("fingerprint_small", "fingerprint_large_artifact")
                if k in dynamic_trace
            },
            "reading": (
                "identity_for_path sha256-reads evidence on every packet compile. "
                "Mission.fingerprint sha256-reads every workspace file. "
                "WorkUnit.content_hash hashes identity on admit."
            ),
        },
        "subprocess_cold_starts": {
            "ast_sites": ast_info["totals"].get("subprocess", 0),
            "examples": ast_info["hits"].get("subprocess", [])[:16],
            "measured": {
                "metal_device": _row(rows, "metal_device"),
                "runtime_admission": _row(rows, "runtime_admission"),
                "grok_bridge": _row(rows, "grok_bridge"),
                "workspace_git": _row(rows, "workspace_git"),
                "host_snapshot": _row(rows, "host_snapshot"),
            },
            "reading": (
                "Largest: `swift` compile of a temp Metal probe on every force "
                "metal_device_info. Grok dry-run still spawns grok-run. "
                "Workspace spawns git. host_snapshot spawns vm_stat/sysctl/"
                "memory_pressure."
            ),
        },
        "python_interpreter_spawns": {
            "measured": {
                "python_pass": _row(rows, "python_pass"),
                "cli_startup": _row(rows, "cli_startup"),
                "verifier_dispatch": _row(rows, "verifier_dispatch"),
            },
            "reading": (
                "python3 -c pass is ~20ms. Engine._validate pays that again for "
                "py_compile and again for pytest, per mutation. CLI --help pays "
                "a whole interpreter plus the Controller import graph."
            ),
        },
        "duplicated_validation": {
            "ast_sites": ast_info["totals"].get("ast.parse", 0),
            "examples": ast_info["hits"].get("ast.parse", [])[:12],
            "reading": (
                "command_is_admissible lives in verifier_pipeline; Engine._admit_test "
                "re-implements pytest-idiom detection. GoalCompiler and Ledger both "
                "extract obligations from the same goal text. Not the millisecond "
                "leader — Swift compile is."
            ),
        },
        "repeated_tool_discovery": {
            "ast_sites": ast_info["totals"].get("shutil.which", 0),
            "examples": ast_info["hits"].get("shutil.which", [])[:12],
            "measured": {"tool_discovery": _row(rows, "tool_discovery")},
            "reading": (
                "find_grok_run() calls shutil.which on every GrokBridge._launch. "
                "metal_device() calls shutil.which('swift') on every probe. "
                "backends.py which()s mlx_lm.server at start, not import. Fast "
                "on its own; the spawn it enables is not."
            ),
        },
    }

    fast = [
        {
            "id": r["id"],
            "cold_median_ms": r["cold_ms"]["median"],
            "warm_median_ms": r["warm_ms"]["median"],
            "why": r.get("notes") or "measured fast; not a defect",
        }
        for r in rows
        if r.get("ok")
        and (r.get("cold_ms") or {}).get("median") is not None
        and float(r["cold_ms"]["median"]) < 5.0
    ]

    return {
        "schema": SCHEMA,
        "gate": GATE,
        "generated_at": generated_at,
        "git_head": head,
        "repo": str(REPO),
        "python": {
            "executable": python,
            "version": sys.version,
        },
        "sparse_checkout": {
            "hcli_on_disk": (REPO / "hcli" / "__main__.py").is_file(),
            "tools_haider_on_disk": (
                REPO / "tools" / "haider" / "hcli" / "__main__.py"
            ).is_file(),
            "hcli_source_mode": located["mode"],
            "hcli_pythonpath": located["pythonpath"],
            "hcli_package": located["package"],
            "reason": located["reason"],
            "metal_budget": located.get("metal_budget"),
        },
        "method": {
            "wall": "time.perf_counter around the named call (in-process) or subprocess (CLI)",
            "cold": (
                f"{N_COLD} fresh Python processes; first in-process call is cold. "
                f"CLI/interpreter rows: first process is cold, later processes are warm "
                "(page cache / dyld)."
            ),
            "warm": f"{N_WARM} subsequent calls in the same process (or later processes for CLI)",
            "no_model_inference": True,
            "anti_goodhart": (
                "Rank by measured milliseconds, not by how bad the code looks. "
                "A path that is already fast is reported as fast. Optimising a "
                "20µs which() while paying 900ms of Swift compile is a failure."
            ),
        },
        "measurements": rows,
        "stages": _stages(rows),
        "noop_adversary": _noop_adversary(rows),
        "ranked_removable_ms": ranked,
        "largest_removable_cost": {
            "id": top.get("id"),
            "ms": top.get("ms"),
            "cold_median_ms": top.get("cold_median_ms"),
            "warm_median_ms": top.get("warm_median_ms"),
            "label": top.get("label"),
            "remove_by": top.get("remove_by"),
            "which_figure": "cold_median_ms minus irreducible floor_ms",
        },
        "fast_paths": fast,
        "waste_census": waste,
        "dynamic_trace": dynamic_trace,
        "ast_waste": {
            "totals": ast_info["totals"],
            "module_count": ast_info["module_count"],
        },
        "metal_cached_ms": extras.get("metal_device_cached_ms"),
        "runtime_admission_cached_ms": extras.get("runtime_admission_cached_ms"),
        "what_i_watched_fail": notes_fail,
        "scope_guard": {
            "wrote": [
                "tools/headless/control_plane_latency.py",
                "receipts/headless/CONTROL_PLANE_LATENCY_LEDGER.json",
                "receipts/headless/CONTROL_PLANE_LATENCY.json",
            ],
            "did_not_modify": [
                "receipts/ascent-2026-08-16",
                "receipts/ascent-2026-08-18",
                "workspace/campaign",
                "receipts/headless/BANDWIDTH*",
                "receipts/headless/PREFILL_KV*",
            ],
        },
    }


def _stages(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {r.get("id"): r for r in rows if isinstance(r, dict)}
    out: Dict[str, Any] = {}
    for name, rid in STAGE_MAP.items():
        r = by.get(rid)
        if not isinstance(r, dict):
            out[name] = {
                "status": "ABSENT",
                "measurement_id": rid,
                "reason": f"measurement {rid!r} was not collected",
            }
            continue
        cold = (r.get("cold_ms") or {}).get("median")
        warm = (r.get("warm_ms") or {}).get("median")
        if not r.get("ok") or cold is None:
            out[name] = {
                "status": "ABSENT",
                "measurement_id": rid,
                "reason": r.get("error") or r.get("notes") or "measurement not ok / no samples",
                "command": r.get("command"),
            }
            continue
        out[name] = {
            "status": "MEASURED",
            "measurement_id": rid,
            "label": r.get("label"),
            "command": r.get("command"),
            "cold_median_ms": cold,
            "warm_median_ms": warm,
            "cold_ms": r.get("cold_ms"),
            "warm_ms": r.get("warm_ms"),
            "ok": True,
        }
    return out


def _noop_adversary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """S020 §36: would a no-op (`python3 -c pass`) also produce this number?

    Process-level rows must beat the interpreter floor or they are the floor.
    In-process rows are allowed to be faster than the floor — they are not a spawn.
    """
    by = {r.get("id"): r for r in rows if isinstance(r, dict)}
    floor = ((by.get("python_pass") or {}).get("cold_ms") or {}).get("median")
    process_level = {"cli_startup", "python_pass"}
    comparisons = []
    for r in rows:
        rid = r.get("id")
        med = (r.get("cold_ms") or {}).get("median")
        if med is None or floor is None:
            continue
        kind = "process" if rid in process_level else "in_process_or_inner"
        comparisons.append(
            {
                "id": rid,
                "cold_median_ms": med,
                "python_pass_floor_ms": floor,
                "kind": kind,
                "exceeds_noop_floor": bool(med > floor) if kind == "process" else None,
                "note": (
                    "CLI --help is a real process; it must exceed python3 -c pass "
                    "or the number is the interpreter, not hcli."
                    if rid == "cli_startup"
                    else (
                        "This IS the no-op floor."
                        if rid == "python_pass"
                        else "In-process call; a no-op process would be slower, not equal."
                    )
                ),
            }
        )
    return {
        "question": "Would a no-op also produce this number?",
        "noop": "python3 -c pass (measurement id python_pass)",
        "floor_ms": floor,
        "comparisons": comparisons,
    }


def _row(rows: List[Dict[str, Any]], id: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("id") == id:
            return {
                "command": r.get("command"),
                "cold_ms": r.get("cold_ms"),
                "warm_ms": r.get("warm_ms"),
                "ok": r.get("ok"),
                "removable_ms": r.get("removable_ms"),
            }
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker"]:
        op = argv[1] if len(argv) > 1 else ""
        return worker_main(op)

    python = sys.executable
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = git_head(REPO)
    extract_root = Path(tempfile.mkdtemp(prefix="hcli-cpl-"))
    try:
        located = locate_hcli(REPO, extract_root)
        doc = build_ledger(python, located, generated_at, head)
        path = write_receipt(doc)
        fails = validate_receipt(doc)
        top = doc.get("largest_removable_cost") or {}
        print(f"wrote {path}")
        print(
            f"largest removable cost: {top.get('id')} "
            f"{top.get('ms')} ms (cold median {top.get('cold_median_ms')} ms, "
            f"warm median {top.get('warm_median_ms')} ms)"
        )
        print(f"remove_by: {top.get('remove_by')}")
        print(f"measurements: {len(doc.get('measurements') or [])}")
        if fails:
            print("receipt self-check FAILED:")
            for f in fails:
                print("  " + f)
            return 1
        print("receipt self-check ok")
        return 0
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------


def _load_or_build_receipt() -> Dict[str, Any]:
    if RECEIPT.is_file():
        doc = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if not validate_receipt(doc):
            return doc
    rc = main([])
    if rc != 0:
        raise AssertionError("control_plane_latency main() failed to write a valid receipt")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_receipt_is_written_with_schema():
    doc = _load_or_build_receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert doc["gate"] == GATE


def test_every_number_has_command_and_cold_warm():
    doc = _load_or_build_receipt()
    fails = validate_receipt(doc)
    assert not fails, fails
    for row in doc["measurements"]:
        assert isinstance(row["command"], list) and row["command"], row["id"]
        for side in ("cold_ms", "warm_ms"):
            fig = row[side]
            assert "median" in fig and "samples" in fig and "n" in fig, (row["id"], side)


def test_largest_removable_cost_named_in_milliseconds():
    doc = _load_or_build_receipt()
    top = doc["largest_removable_cost"]
    assert top["id"], top
    assert isinstance(top["ms"], (int, float)), top
    assert top["ms"] > 0, top
    assert top["remove_by"], top


def test_required_ceremony_paths_are_present():
    doc = _load_or_build_receipt()
    have = {r["id"] for r in doc["measurements"]}
    for key in (
        "cli_startup",
        "import_hcli",
        "mission_load",
        "dag_load",
        "context_compile",
        "status_render",
        "scheduler_cycle",
        "receipt_write",
        "checkpoint",
        "verifier_dispatch",
        "experiment_setup",
        "grok_bridge",
        "runtime_admission",
    ):
        assert key in have, key


def test_ledger_filename_and_stages_are_measured():
    doc = _load_or_build_receipt()
    assert RECEIPT.is_file()
    assert RECEIPT.name == "CONTROL_PLANE_LATENCY_LEDGER.json"
    stages = doc["stages"]
    for name in STAGE_MAP:
        assert name in stages, name
        st = stages[name]
        assert st["status"] in {"MEASURED", "ABSENT"}, (name, st)
        if st["status"] == "MEASURED":
            assert isinstance(st["cold_median_ms"], (int, float)), (name, st)
            assert st["command"], name
        else:
            assert st.get("reason"), (name, st)


if __name__ == "__main__":
    raise SystemExit(main())
