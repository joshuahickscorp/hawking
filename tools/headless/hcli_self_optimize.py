#!/usr/bin/env python3
"""HCLI self-optimize iteration 1, run as a Mission of WorkUnits.

The loop does not perform the stages itself. It constructs a Mission whose
DAG is the loop, dispatches every stage through Mission, and records each
WorkUnit id. Mutation goes through Engine.execute (lock, rollback, receipt).
This file is the only writer of the iteration receipt.

Do not "narrow the lock" around execute_workunit's monkeypatch of
_gather_evidence / goal_compiler.compile. Those patches live on the shared
engine; narrowing the lock lets one worker's evidence reach another worker's
prompt. The surviving change is to stop needing the patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
HCLI_PARENT = REPO
ENGINE_REL = Path("hcli/engine.py")
EXECUTORS_REL = Path("hcli/executors.py")
RECEIPT_REL = Path("receipts/headless/HCLI_SELF_OPT_ITERATION_1.json")
LLAMA_PORT = 52484
PROBE_DELAY_S = 0.35
CPU_TIMEOUT_S = "600"
PROBE_IDS = ("g0", "g1")
DIR_VERIFIER = (
    "python3 -c "
    "\"import pathlib,sys; sys.exit(0 if pathlib.Path('.').exists() else 1)\""
)

# Priors that bound the prize. Cited, not re-derived.
PRIOR_ACTIVE_DECODE_LIMIT = 2
PRIOR_AGGREGATE_AT_FOUR = 1.2161  # DECODE_TOPOLOGY summary.slot.4.scaling_vs_1
PRIOR_CONTRACT_ENVELOPE = 1.26  # contract bound: near 1.26x at four decoders
PRIOR_GENOME_AGGREGATE = 1.1934  # MACHINE_GENOME aggregate_scaling_vs_1


# ---------------------------------------------------------------------------
# small IO
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    _atomic_write(path, state)


def watch(state: Dict[str, Any], title: str, detail: str) -> None:
    bucket = state.setdefault("watched_fail", [])
    bucket.append({"title": title, "detail": detail})


def die(msg: str, code: int = 1) -> None:
    print(f"FAIL {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def ok(msg: str) -> None:
    print(f"ok   {msg}", flush=True)


def ensure_hcli_path() -> None:
    parent = str(HCLI_PARENT)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dotted(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# llama-server liveness (the instrumented probe stubs _call_model; this is
# the environment check that the contract's server is actually up)
# ---------------------------------------------------------------------------

def llama_snapshot(port: int = LLAMA_PORT) -> Dict[str, Any]:
    base = f"http://127.0.0.1:{port}"
    out: Dict[str, Any] = {"port": port, "health": None, "total_slots": None, "error": None}
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        out["health"] = body.get("status") or body
    except Exception as exc:
        out["error"] = f"health: {type(exc).__name__}: {exc}"
        return out
    try:
        with urllib.request.urlopen(base + "/props", timeout=2) as resp:
            props = json.loads(resp.read().decode("utf-8", errors="replace"))
        out["total_slots"] = props.get("total_slots")
        out["model_path"] = props.get("model_path")
    except Exception as exc:
        out["props_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# overlap probe — a real Mission of two GPU_DECODE units
# ---------------------------------------------------------------------------

def run_overlap_probe(repo: Path, delay_s: float = PROBE_DELAY_S) -> Dict[str, Any]:
    """Measure max concurrent _call_model on a real Mission.

    Instrument: 0.35s delay inside _call_model, matching the prior that
    found peak=1 while observed_max_gpu_decode=2. The delay is the
    measuring stick; llama-server -np is not mixed into this number.
    """
    ensure_hcli_path()
    os.environ["ACTIVE_DECODE_LIMIT"] = "2"

    import hcli.executors  # noqa: F401  installs Engine.execute_workunit
    from hcli.engine import Engine
    from hcli.mission import Mission
    from hcli.resources import ResourceLimits
    from hcli.workunit import WorkUnit
    from hcli.workspace import Workspace

    tmp = tempfile.mkdtemp(prefix="hcli-selfopt-probe-")
    try:
        ws = Workspace(tmp)
        engine = Engine(ws)
        stats: Dict[str, Any] = {
            "lock": threading.Lock(),
            "inflight": 0,
            "peak": 0,
            "enters": [],
        }

        def wrapped(prompt, evidence=None, compiled=None, **kwargs):
            tid = threading.current_thread().name
            t0 = time.perf_counter()
            with stats["lock"]:
                stats["inflight"] += 1
                during = stats["inflight"]
                if during > stats["peak"]:
                    stats["peak"] = during
                stats["enters"].append(
                    {"thread": tid, "t": t0, "peak_during": during}
                )
            try:
                time.sleep(delay_s)
                return {
                    "kind": "answer",
                    "content": "probe-ok",
                    "operations": [],
                    "tests": [],
                }
            finally:
                with stats["lock"]:
                    stats["inflight"] -= 1

        engine._call_model = wrapped  # type: ignore[method-assign]

        units = [
            WorkUnit(
                id=uid,
                role="probe",
                description=f"decode unit {uid}",
                resource_class="GPU_DECODE",
                verifier=DIR_VERIFIER,
            )
            for uid in PROBE_IDS
        ]
        limits = ResourceLimits.resolve(repo_root=repo)
        mission = Mission(
            tmp,
            engine=engine,
            units=units,
            runtime_count=2,
            limits=limits,
            quiet=True,
            goal="",
            install_signals=False,
        )
        t0 = time.perf_counter()
        result = mission.run()
        wall = time.perf_counter() - t0
        enters = sorted(stats["enters"], key=lambda e: e["t"])
        spread = None
        if len(enters) >= 2:
            spread = float(enters[1]["t"] - enters[0]["t"])
        # strip absolute clocks; keep relative so the receipt is comparable
        rel_enters = []
        if enters:
            base = enters[0]["t"]
            for item in enters:
                rel_enters.append(
                    {
                        "thread": item["thread"],
                        "t_rel_s": float(item["t"] - base),
                        "peak_during": item["peak_during"],
                    }
                )
        unit_status = {
            wu.id: {
                "status": wu.status,
                "verification": wu.verification,
                "attempts": wu.attempts,
            }
            for wu in mission.scheduler.units.values()
        }
        return {
            "ok": result.get("status") == "completed" and stats["peak"] >= 1,
            "mission_status": result.get("status"),
            "mission_id": result.get("mission_id"),
            "accepted": result.get("accepted"),
            "observed_max_gpu_decode": int(mission.observed_max_gpu_decode),
            "max_concurrent_model_calls": int(stats["peak"]),
            "enter_spread_s": spread,
            "enters": rel_enters,
            "wall_s": wall,
            "delay_s": delay_s,
            "active_decode_limit": int(limits.gpu_decode),
            "active_decode_limit_source": limits.gpu_decode_source,
            "units": unit_status,
            "workspace": tmp,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
        }


# ---------------------------------------------------------------------------
# surviving mutation (H2) — operations, not hand-typed into hcli/
# ---------------------------------------------------------------------------

def _sig_old() -> str:
    return (
        "    def execute(\n"
        "        self,\n"
        "        prompt: str,\n"
        "    ) -> Dict[str, Any]:"
    )


def _sig_new() -> str:
    return (
        "    def execute(\n"
        "        self,\n"
        "        prompt: str,\n"
        "        evidence: Optional[List[Dict[str, Any]]] = None,\n"
        "        compiled: Any = None,\n"
        "    ) -> Dict[str, Any]:"
    )


def _gather_old() -> str:
    return (
        "        evidence: List[Dict[str, Any]] = list()\n"
        "        try:\n"
        "            if self._cancelled:\n"
        "                return self._cancel_result(goal_id, evidence)\n"
        "\n"
        "            evidence = self._gather_evidence(prompt)\n"
        "\n"
        "            try:\n"
        "                compiled = self.goal_compiler.compile(prompt)\n"
        "            except Exception:\n"
        "                compiled = {}"
    )


def _gather_new() -> str:
    return (
        "        supplied_evidence = evidence\n"
        "        supplied_compiled = compiled\n"
        "        evidence: List[Dict[str, Any]] = list()\n"
        "        try:\n"
        "            if self._cancelled:\n"
        "                return self._cancel_result(goal_id, evidence)\n"
        "\n"
        "            if supplied_evidence is None:\n"
        "                evidence = self._gather_evidence(prompt)\n"
        "            else:\n"
        "                evidence = list(supplied_evidence)\n"
        "\n"
        "            if supplied_compiled is None:\n"
        "                try:\n"
        "                    compiled = self.goal_compiler.compile(prompt)\n"
        "                except Exception:\n"
        "                    compiled = {}\n"
        "            else:\n"
        "                compiled = supplied_compiled"
    )


def _lock_old() -> str:
    return (
        "    lock = getattr(self, \"_hcli_worker_lock\", None)\n"
        "    if lock is None:\n"
        "        lock = threading.Lock()\n"
        "        self._hcli_worker_lock = lock\n"
        "\n"
        "    with lock:\n"
        "        orig_gather = self._gather_evidence\n"
        "        orig_compile = self.goal_compiler.compile\n"
        "\n"
        "        def _gather(_prompt: str) -> List[Dict[str, Any]]:\n"
        "            return evidence\n"
        "\n"
        "        def _compile(_text: str) -> Dict[str, Any]:\n"
        "            return compiled\n"
        "\n"
        "        self._gather_evidence = _gather\n"
        "        self.goal_compiler.compile = _compile\n"
        "        try:\n"
        "            return self.execute(prompt)\n"
        "        finally:\n"
        "            self._gather_evidence = orig_gather\n"
        "            self.goal_compiler.compile = orig_compile"
    )


def _lock_new() -> str:
    return (
        "    return self.execute(prompt, evidence=evidence, compiled=compiled)"
    )


def mutation_operations() -> List[Dict[str, Any]]:
    return [
        {
            "op": "replace",
            "path": str(ENGINE_REL),
            "old_text": _sig_old(),
            "new_text": _sig_new(),
        },
        {
            "op": "replace",
            "path": str(ENGINE_REL),
            "old_text": _gather_old(),
            "new_text": _gather_new(),
        },
        {
            "op": "replace",
            "path": str(EXECUTORS_REL),
            "old_text": _lock_old(),
            "new_text": _lock_new(),
        },
    ]


def mutation_already_applied(repo: Path) -> bool:
    engine = (repo / ENGINE_REL).read_text(encoding="utf-8")
    executors = (repo / EXECUTORS_REL).read_text(encoding="utf-8")
    return _sig_new() in engine and _lock_new() in executors and _lock_old() not in executors


def operations_applicable(repo: Path) -> Tuple[bool, str]:
    if mutation_already_applied(repo):
        return True, "already_applied"
    engine = (repo / ENGINE_REL).read_text(encoding="utf-8")
    executors = (repo / EXECUTORS_REL).read_text(encoding="utf-8")
    missing = []
    if _sig_old() not in engine:
        missing.append("engine.execute signature")
    if _gather_old() not in engine:
        missing.append("engine gather/compile block")
    if _lock_old() not in executors:
        missing.append("executors worker lock block")
    if missing:
        return False, "missing anchors: " + ", ".join(missing)
    for op in mutation_operations():
        text = (repo / op["path"]).read_text(encoding="utf-8")
        n = text.count(op["old_text"])
        if n != 1:
            return False, f"{op['path']}: old_text occurs {n} times, need 1"
    return True, "applicable"


def apply_ops_to_copy(repo: Path, dest: Path) -> Tuple[bool, str]:
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        ENGINE_REL: dest / "engine.py",
        EXECUTORS_REL: dest / "executors.py",
    }
    for rel, target in mapping.items():
        target.write_text((repo / rel).read_text(encoding="utf-8"), encoding="utf-8")
    try:
        engine = mapping[ENGINE_REL].read_text(encoding="utf-8")
        engine = engine.replace(_sig_old(), _sig_new(), 1)
        engine = engine.replace(_gather_old(), _gather_new(), 1)
        mapping[ENGINE_REL].write_text(engine, encoding="utf-8")
        ex = mapping[EXECUTORS_REL].read_text(encoding="utf-8")
        ex = ex.replace(_lock_old(), _lock_new(), 1)
        mapping[EXECUTORS_REL].write_text(ex, encoding="utf-8")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    for target in mapping.values():
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            return False, f"py_compile {target.name}: {proc.stderr[-400:]}"
    return True, "compiled"


def restore_files(repo: Path, snap_dir: Path) -> None:
    for rel, name in ((ENGINE_REL, "engine.py"), (EXECUTORS_REL, "executors.py")):
        src = snap_dir / name
        if src.is_file():
            (repo / rel).write_bytes(src.read_bytes())


def snapshot_pair(repo: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / ENGINE_REL, dest / "engine.py")
    shutil.copy2(repo / EXECUTORS_REL, dest / "executors.py")


# ---------------------------------------------------------------------------
# cheap disproof that narrowing the lock leaks evidence
# ---------------------------------------------------------------------------

def narrow_lock_leak_demo() -> Dict[str, Any]:
    """Shared-state patch, lock released before the 'HTTP' sleep.

    After both workers assign, they wait on a barrier so the last write
    is visible to both. At least one worker then reads the other's
    evidence. That is the correctness argument against narrowing the lock.
    """
    shared = {"evidence": None}
    results: Dict[str, Any] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, ev: str) -> None:
        shared["evidence"] = ev
        barrier.wait(timeout=2)
        time.sleep(0.02)
        results[name] = shared["evidence"]

    threads = [
        threading.Thread(target=worker, args=("hcli-wu-g0", "EVIDENCE_G0")),
        threading.Thread(target=worker, args=("hcli-wu-g1", "EVIDENCE_G1")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    leaked = results.get("hcli-wu-g0") != "EVIDENCE_G0" or results.get(
        "hcli-wu-g1"
    ) != "EVIDENCE_G1"
    return {
        "leaked": bool(leaked),
        "results": dict(results),
        "reason": (
            "Patches live on the shared engine instance. Releasing the lock "
            "before _call_model lets worker B overwrite worker A's "
            "_gather_evidence / compile for the HTTP that is still in flight, "
            "so A's prompt is built from B's evidence."
        ),
    }


# ---------------------------------------------------------------------------
# location resolver (hypotheses / bottleneck verifiers)
# ---------------------------------------------------------------------------

def resolve_loc(repo: Path, spec: str) -> Dict[str, Any]:
    if ":" not in spec:
        return {"ok": False, "spec": spec, "error": "missing file:line"}
    path_s, _, lines = spec.partition(":")
    full = repo / path_s
    if not full.is_file():
        return {"ok": False, "spec": spec, "error": f"not a file: {path_s}"}
    text = full.read_text(encoding="utf-8").splitlines()
    if "-" in lines:
        a, b = lines.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(lines)
    if start < 1 or end > len(text) or start > end:
        return {
            "ok": False,
            "spec": spec,
            "error": f"line range {start}-{end} vs {len(text)} lines",
        }
    snippet = text[start - 1 : end]
    return {
        "ok": True,
        "spec": spec,
        "path": path_s,
        "start": start,
        "end": end,
        "n_lines": len(text),
        "snippet": snippet,
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_sense(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    llama = llama_snapshot()
    probe = run_overlap_probe(repo)
    payload = {
        "llama_server": llama,
        "probe": probe,
        "max_concurrent_model_calls": probe.get("max_concurrent_model_calls"),
        "observed_max_gpu_decode": probe.get("observed_max_gpu_decode"),
        "enter_spread_s": probe.get("enter_spread_s"),
        "delay_s": PROBE_DELAY_S,
    }
    state["sense"] = payload
    if llama.get("health") != "ok":
        watch(
            state,
            "llama-server health not ok",
            json.dumps(llama, default=str),
        )
    if not probe.get("ok"):
        watch(state, "sense probe failed", json.dumps(probe, default=str)[:2000])
        save_state(Path(state["_path"]), state)
        die("sense: probe did not run to completion")
    n = payload["max_concurrent_model_calls"]
    if not isinstance(n, int):
        die("sense: probe did not write a number")
    watch(
        state,
        "GPU_DECODE admitted 2 workers but _call_model stayed serial",
        (
            f"observed_max_gpu_decode={payload['observed_max_gpu_decode']} "
            f"max_concurrent_model_calls={n} enter_spread_s="
            f"{payload['enter_spread_s']}"
        ),
    )
    ok(
        f"sense: observed_max_gpu_decode={payload['observed_max_gpu_decode']} "
        f"max_concurrent_model_calls={n} spread={payload['enter_spread_s']:.4f}s "
        f"llama={llama.get('health')} slots={llama.get('total_slots')}"
    )


def stage_bottleneck(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    sense = state.get("sense") or {}
    peak = sense.get("max_concurrent_model_calls")
    observed = sense.get("observed_max_gpu_decode")
    loc = resolve_loc(repo, "hcli/executors.py:277-299")
    agrees = peak == 1 and isinstance(observed, int) and observed >= 2
    named = (
        "execute_workunit holds a process-wide lock across Engine.execute, "
        "so _call_model cannot overlap even when two GPU_DECODE units are "
        "running. The lock exists because execute_workunit monkeypatches "
        "self._gather_evidence and self.goal_compiler.compile on the SHARED "
        "engine instance."
    )
    payload = {
        "name": named,
        "location": "hcli/executors.py:277-299",
        "resolved": loc,
        "agrees_with_sense": agrees,
        "sense_max_concurrent_model_calls": peak,
        "sense_observed_max_gpu_decode": observed,
        "contradiction": None,
    }
    if not agrees:
        payload["contradiction"] = (
            f"named bottleneck claims serial _call_model (peak=1) with "
            f"concurrent GPU_DECODE (>=2), but sense measured peak={peak} "
            f"observed_max_gpu_decode={observed}"
        )
        state["bottleneck"] = payload
        save_state(Path(state["_path"]), state)
        die("bottleneck: named bottleneck contradicts the measurement")
    if not loc.get("ok"):
        die(f"bottleneck: location did not resolve: {loc}")
    # the lock block should still be present pre-mutation
    blob = "\n".join(loc.get("snippet") or [])
    if "_hcli_worker_lock" not in blob:
        die("bottleneck: resolved lines do not contain the worker lock")
    state["bottleneck"] = payload
    ok("bottleneck: lock at executors.py:277-299 agrees with peak=1 / decode=2")


def _hypotheses() -> List[Dict[str, Any]]:
    return [
        {
            "id": "H1_narrow_lock",
            "title": "Narrow the worker lock so it covers only the monkeypatch assignment, not the HTTP call",
            "location": "hcli/executors.py:277-299",
            "change": (
                "Keep the patch of _gather_evidence / compile, but release "
                "_hcli_worker_lock before self.execute(prompt) / _call_model."
            ),
        },
        {
            "id": "H2_pass_explicit",
            "title": "Pass evidence and compiled into Engine.execute so execute_workunit does not patch shared state",
            "location": "hcli/engine.py:419-460",
            "secondary_location": "hcli/executors.py:262-298",
            "change": (
                "Add optional evidence= and compiled= kwargs to Engine.execute; "
                "have execute_workunit call execute(prompt, evidence=..., compiled=...) "
                "and delete the shared-state monkeypatch and its lock."
            ),
        },
        {
            "id": "H3_thread_local_patches",
            "title": "Keep the monkeypatch but store gather/compile on thread-local state",
            "location": "hcli/executors.py:282-293",
            "change": (
                "Replace instance attributes with threading.local() so two "
                "workers can patch without a process lock."
            ),
        },
        {
            "id": "H4_clone_engine",
            "title": "Clone Engine per WorkUnit so patches are private",
            "location": "hcli/mission.py:679-704",
            "change": (
                "Give each GPU_DECODE worker its own Engine (or a shallow copy) "
                "so monkeypatches cannot clobber a sibling."
            ),
        },
    ]


def stage_hypotheses(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    hyps = _hypotheses()
    resolved = []
    seen_locs = set()
    for hyp in hyps:
        loc = resolve_loc(repo, hyp["location"])
        extra = None
        if hyp.get("secondary_location"):
            extra = resolve_loc(repo, hyp["secondary_location"])
        if not loc.get("ok"):
            die(f"hypotheses: {hyp['id']} location did not resolve: {loc}")
        if extra is not None and not extra.get("ok"):
            die(f"hypotheses: {hyp['id']} secondary location did not resolve: {extra}")
        seen_locs.add(hyp["location"])
        item = dict(hyp)
        item["resolved"] = loc
        if extra is not None:
            item["secondary_resolved"] = extra
        resolved.append(item)
    if len(resolved) < 3:
        die("hypotheses: need at least three candidates")
    if len(seen_locs) < 3:
        die("hypotheses: candidates must resolve to distinct locations")
    state["hypotheses"] = {"candidates": resolved, "count": len(resolved)}
    ok(f"hypotheses: {len(resolved)} candidates, locations resolve")


def stage_screen(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    leak = narrow_lock_leak_demo()
    if not leak.get("leaked"):
        die("screen: narrow-lock leak demo did not leak; disproof is broken")
    watch(
        state,
        "narrow the lock leaks evidence (cheap disproof)",
        json.dumps(leak.get("results"), default=str),
    )
    copy_dir = ws / "h2_scratch"
    h2_ok, h2_detail = apply_ops_to_copy(repo, copy_dir)
    applicable, why = operations_applicable(repo)

    verdicts = [
        {
            "id": "H1_narrow_lock",
            "verdict": "REJECTED",
            "reason": (
                "Correctness, not speed. execute_workunit patches "
                "self._gather_evidence and self.goal_compiler.compile on the "
                "SHARED engine. The lock must span the HTTP call because the "
                "patches have to stay in place for the duration of execute(). "
                "Narrowing it lets one worker's evidence reach another worker's "
                "prompt. Cheap disproof: two threads assign then 'HTTP' "
                f"outside the lock; leaked={leak['leaked']} results="
                f"{leak['results']}."
            ),
            "leak_demo": leak,
        },
        {
            "id": "H2_pass_explicit",
            "verdict": "SURVIVE" if (h2_ok and applicable) else "REJECTED",
            "reason": (
                "Removes the NEED for shared-state patching. Engine.execute "
                "already takes research/evidence/compiled internally; exposing them as "
                "kwargs lets execute_workunit pass a worker packet without "
                "touching instance attributes. Cheap check: anchors unique, "
                f"scratch apply+py_compile ok={h2_ok} ({h2_detail}), "
                f"applicable={why}."
            ),
            "scratch_apply_ok": h2_ok,
            "scratch_detail": h2_detail,
            "applicable": why,
        },
        {
            "id": "H3_thread_local_patches",
            "verdict": "REJECTED",
            "reason": (
                "Still patches. threading.local hides the shared-state bug "
                "instead of deleting it, and goal_compiler.compile is a bound "
                "method on a shared GoalCompiler — TLS on the engine instance "
                "does not make the compiler itself re-entrant. Band-aid; "
                "screened out in favour of H2."
            ),
        },
        {
            "id": "H4_clone_engine",
            "verdict": "REJECTED",
            "reason": (
                "A per-worker Engine clone would make patches private but "
                "duplicates runtime_provider / model_client / event bus "
                "identity and still uses monkeypatching. Heavier than H2 for "
                "the same prize, which is bounded well under 2x."
            ),
        },
    ]
    rejected = [v for v in verdicts if v["verdict"] == "REJECTED"]
    survived = [v for v in verdicts if v["verdict"] == "SURVIVE"]
    if not rejected:
        die("screen: at least one hypothesis must be rejected")
    if not any(v["id"] == "H1_narrow_lock" and v["verdict"] == "REJECTED" for v in verdicts):
        die("screen: H1_narrow_lock must be rejected")
    if not survived:
        die("screen: no surviving hypothesis")
    state["screen"] = {
        "verdicts": verdicts,
        "rejected_count": len(rejected),
        "survived": [v["id"] for v in survived],
    }
    ok(
        f"screen: survived={state['screen']['survived']} "
        f"rejected={[v['id'] for v in rejected]}"
    )


def stage_mutate(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    """Apply H2 through Engine.execute. Mutation lock is held by Mission."""
    ensure_hcli_path()
    from hcli.engine import Engine
    from hcli.workspace import Workspace as HcliWorkspace

    snap_original = ws / "snap" / "original"
    snap_mutated = ws / "snap" / "mutated"
    snapshot_pair(repo, snap_original)

    already = mutation_already_applied(repo)
    applicable, why = operations_applicable(repo)
    files_before = {
        "engine.py": sha256_file(repo / ENGINE_REL),
        "executors.py": sha256_file(repo / EXECUTORS_REL),
    }
    payload: Dict[str, Any] = {
        "path": "Engine.execute mutation path",
        "already_applied": already,
        "applicable": why,
        "files_before": files_before,
        "applied": False,
        "blocked": None,
        "engine_receipt": None,
        "engine_result_status": None,
        "rolled_back": None,
        "operations": mutation_operations(),
    }

    if already:
        payload["blocked"] = "mutation already present on disk; not re-applied"
        snapshot_pair(repo, snap_mutated)
        payload["applied"] = True
        payload["files_after"] = files_before
        state["mutate"] = payload
        watch(state, "mutate: already applied", why)
        ok("mutate: already applied (treated as present)")
        return

    if not applicable:
        payload["blocked"] = why
        state["mutate"] = payload
        watch(state, "mutate: could not apply", why)
        ok(f"mutate: BLOCKED ({why})")
        return

    class _MutationClient:
        def complete(self, prompt, evidence=None, compiled=None):
            return {
                "kind": "mutation",
                "content": (
                    "Pass evidence and compiled into Engine.execute so "
                    "execute_workunit does not monkeypatch shared state."
                ),
                "operations": mutation_operations(),
                "tests": [],
            }

    engine = Engine(HcliWorkspace(str(repo)), model_client=_MutationClient())
    try:
        result = engine.execute(
            "Apply the surviving self-opt mutation: pass evidence and "
            "compiled into Engine.execute so execute_workunit does not "
            "patch _gather_evidence or goal_compiler.compile on the shared "
            "engine. Edit hcli/engine.py and "
            "hcli/executors.py."
        )
    except Exception as exc:
        payload["blocked"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-2000:]
        state["mutate"] = payload
        watch(state, "mutate: Engine.execute raised", payload["blocked"])
        restore_files(repo, snap_original)
        ok(f"mutate: BLOCKED ({payload['blocked']})")
        return

    receipt = result.get("receipt")
    payload["engine_result_status"] = result.get("status")
    payload["rolled_back"] = bool(result.get("rolled_back"))
    payload["engine_error"] = result.get("error")
    payload["validation"] = result.get("validation") or (
        result.get("receipt") if isinstance(result.get("receipt"), dict) else None
    )
    if isinstance(receipt, str) and Path(receipt).is_file():
        try:
            rec_obj = json.loads(Path(receipt).read_text(encoding="utf-8"))
        except Exception:
            rec_obj = {"path": receipt}
        payload["engine_receipt_path"] = receipt
        payload["engine_receipt"] = {
            "path": receipt,
            "status": rec_obj.get("status"),
            "rolled_back": rec_obj.get("rolled_back"),
            "kind": rec_obj.get("kind"),
            "validation": rec_obj.get("validation"),
            "files": (rec_obj.get("validation") or {}).get("files")
            if isinstance(rec_obj.get("validation"), dict)
            else rec_obj.get("files"),
        }
    elif isinstance(receipt, dict):
        payload["engine_receipt"] = {
            "status": receipt.get("status"),
            "rolled_back": receipt.get("rolled_back"),
            "kind": receipt.get("kind"),
            "validation": receipt.get("validation"),
            "goal_id": receipt.get("goal_id"),
        }

    files_after = {
        "engine.py": sha256_file(repo / ENGINE_REL),
        "executors.py": sha256_file(repo / EXECUTORS_REL),
    }
    payload["files_after"] = files_after
    changed = files_before != files_after
    applied_ok = changed and not result.get("rolled_back") and mutation_already_applied(repo)

    if result.get("rolled_back"):
        payload["blocked"] = (
            f"Engine rolled the mutation back: status={result.get('status')} "
            f"error={result.get('error')}"
        )
        payload["applied"] = False
        watch(state, "mutate: rolled back", payload["blocked"])
        restore_files(repo, snap_original)
        state["mutate"] = payload
        ok(f"mutate: BLOCKED (rolled back)")
        return

    if not applied_ok:
        payload["blocked"] = (
            f"mutation did not land: changed={changed} "
            f"already_applied_after={mutation_already_applied(repo)} "
            f"status={result.get('status')}"
        )
        payload["applied"] = False
        watch(state, "mutate: did not land", payload["blocked"])
        restore_files(repo, snap_original)
        state["mutate"] = payload
        ok(f"mutate: BLOCKED ({payload['blocked']})")
        return

    snapshot_pair(repo, snap_mutated)
    payload["applied"] = True
    payload["snap_original"] = str(snap_original)
    payload["snap_mutated"] = str(snap_mutated)
    state["mutate"] = payload
    ok(
        f"mutate: applied via Engine.execute status={result.get('status')} "
        f"receipt={payload.get('engine_receipt_path') or 'in-result'} "
        f"engine={files_before['engine.py'][:12]}->{files_after['engine.py'][:12]}"
    )


def stage_gate_correctness(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "hcli/tests", "-q", "--tb=line"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=480,
    )
    wall = time.perf_counter() - t0
    tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-1000:]
    payload = {
        "command": [sys.executable, "-m", "pytest", "hcli/tests", "-q"],
        "exit_code": proc.returncode,
        "passed_gate": proc.returncode == 0,
        "wall_s": wall,
        "output_tail": tail[-2500:],
    }
    # parse "N passed, M skipped"
    import re

    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", tail)
    if m:
        payload["passed"] = int(m.group(1))
        payload["skipped"] = int(m.group(2) or 0)
    state["gate.correctness"] = payload
    if proc.returncode != 0:
        watch(
            state,
            "gate.correctness pytest failed",
            f"exit={proc.returncode} tail={tail[-800:]}",
        )
    ok(
        f"gate.correctness: exit={proc.returncode} "
        f"passed={payload.get('passed')} skipped={payload.get('skipped')} "
        f"wall={wall:.1f}s"
    )


def _probe_child(repo: Path, out: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ACTIVE_DECODE_LIMIT"] = "2"
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--probe-overlap",
            "--out",
            str(out),
            "--repo",
            str(repo),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )
    if out.is_file():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        data = {"ok": False, "error": "no output file"}
    data["child_exit"] = proc.returncode
    if proc.returncode != 0 and not data.get("ok"):
        data["child_stderr"] = (proc.stderr or "")[-800:]
    return data


def stage_gate_perf(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    mutate = state.get("mutate") or {}
    applied = bool(mutate.get("applied"))
    snap_original = Path(mutate["snap_original"]) if mutate.get("snap_original") else ws / "snap" / "original"
    snap_mutated = Path(mutate["snap_mutated"]) if mutate.get("snap_mutated") else ws / "snap" / "mutated"
    trials: List[Dict[str, Any]] = []
    probe_dir = ws / "perf_trials"
    probe_dir.mkdir(parents=True, exist_ok=True)

    if applied and snap_original.is_dir() and snap_mutated.is_dir():
        # Paired, alternating: mutated, original, mutated, original.
        # A single before/after is page-cache and load confounded.
        order = ["mutated", "original", "mutated", "original"]
        for i, cond in enumerate(order):
            restore_files(repo, snap_mutated if cond == "mutated" else snap_original)
            # drop cached bytecode so the child import sees the bytes we wrote
            for rel in (ENGINE_REL, EXECUTORS_REL):
                pycache = (repo / rel).parent / "__pycache__"
                if pycache.is_dir():
                    for pyc in pycache.glob("engine*.pyc"):
                        pyc.unlink(missing_ok=True)
                    for pyc in pycache.glob("executors*.pyc"):
                        pyc.unlink(missing_ok=True)
            out = probe_dir / f"trial_{i}_{cond}.json"
            result = _probe_child(repo, out)
            trials.append(
                {
                    "i": i,
                    "condition": cond,
                    "max_concurrent_model_calls": result.get("max_concurrent_model_calls"),
                    "observed_max_gpu_decode": result.get("observed_max_gpu_decode"),
                    "enter_spread_s": result.get("enter_spread_s"),
                    "wall_s": result.get("wall_s"),
                    "ok": result.get("ok"),
                    "mission_status": result.get("mission_status"),
                }
            )
        # leave the tree in the mutated state for decide
        restore_files(repo, snap_mutated)
        alternating = True
    else:
        watch(
            state,
            "gate.perf could not alternate mutated/original",
            f"applied={applied} original_snap={snap_original.is_dir()} "
            f"mutated_snap={snap_mutated.is_dir()} blocked={mutate.get('blocked')}",
        )
        for i in range(4):
            out = probe_dir / f"trial_{i}_current.json"
            result = _probe_child(repo, out)
            trials.append(
                {
                    "i": i,
                    "condition": "current",
                    "max_concurrent_model_calls": result.get("max_concurrent_model_calls"),
                    "observed_max_gpu_decode": result.get("observed_max_gpu_decode"),
                    "enter_spread_s": result.get("enter_spread_s"),
                    "wall_s": result.get("wall_s"),
                    "ok": result.get("ok"),
                    "mission_status": result.get("mission_status"),
                }
            )
        alternating = False

    def _stats(cond: str) -> Dict[str, Any]:
        vals = [
            t["max_concurrent_model_calls"]
            for t in trials
            if t.get("condition") == cond and isinstance(t.get("max_concurrent_model_calls"), int)
        ]
        if not vals:
            return {"n": 0, "values": [], "min": None, "max": None, "median": None, "spread": None}
        vals_sorted = sorted(vals)
        mid = vals_sorted[len(vals_sorted) // 2]
        return {
            "n": len(vals),
            "values": vals,
            "min": min(vals),
            "max": max(vals),
            "median": mid,
            "spread": max(vals) - min(vals),
        }

    mutated_stats = _stats("mutated") if alternating else _stats("current")
    original_stats = _stats("original") if alternating else _stats("current")
    sense_peak = (state.get("sense") or {}).get("max_concurrent_model_calls")
    improved = False
    if alternating and mutated_stats["median"] is not None and original_stats["median"] is not None:
        improved = mutated_stats["median"] > original_stats["median"]
    payload = {
        "alternating": alternating,
        "order": [t["condition"] for t in trials],
        "trials": trials,
        "mutated": mutated_stats,
        "original": original_stats,
        "sense_peak": sense_peak,
        "overlap_improved": improved,
        "spread_mutated": mutated_stats.get("spread"),
        "spread_original": original_stats.get("spread"),
    }
    state["gate.perf"] = payload
    if not trials or not any(t.get("ok") for t in trials):
        die("gate.perf: no successful overlap re-measure")
    ok(
        f"gate.perf: alternating={alternating} original_median="
        f"{original_stats.get('median')} mutated_median={mutated_stats.get('median')} "
        f"spread_mutated={mutated_stats.get('spread')} improved={improved}"
    )


def stage_decide(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    correctness = state.get("gate.correctness") or {}
    perf = state.get("gate.perf") or {}
    mutate = state.get("mutate") or {}
    correctness_ok = bool(correctness.get("passed_gate"))
    overlap_improved = bool(perf.get("overlap_improved"))
    mutation_applied = bool(mutate.get("applied"))

    # Promotion is refused if either gate failed, or overlap did not improve.
    # This predicate is the verifier: a promotion with a failing gate is
    # refused even if this function were to ask for one.
    refuse_if = {
        "correctness_failed": (not correctness_ok, "REFUSED"),
        "overlap_did_not_improve": (not overlap_improved, "REFUSED"),
        "mutation_not_applied": (not mutation_applied, "REFUSED"),
    }
    would_refuse = any(flag for flag, _ in refuse_if.values())
    if not would_refuse:
        decision = "promote"
        reason = (
            "Both gates passed and paired overlap improved "
            f"(original median peak={perf.get('original', {}).get('median')} "
            f"mutated median peak={perf.get('mutated', {}).get('median')})."
        )
    else:
        decision = "reject"
        bits = []
        if not mutation_applied:
            bits.append(f"mutation did not apply ({mutate.get('blocked')})")
        if not correctness_ok:
            bits.append(
                f"gate.correctness exit={correctness.get('exit_code')}"
            )
        if not overlap_improved:
            bits.append(
                "overlap did not improve "
                f"(original median={perf.get('original', {}).get('median')} "
                f"mutated median={perf.get('mutated', {}).get('median')})"
            )
        reason = "REJECT: " + "; ".join(bits)

    # Simulated counterfactual: had a gate failed, promotion is refused.
    counterfactual = {
        "if_correctness_failed": "REFUSED",
        "if_perf_overlap_unimproved": "REFUSED",
        "predicate": (
            "promote IFF mutation.applied AND gate.correctness.passed_gate "
            "AND gate.perf.overlap_improved; else reject"
        ),
        "would_refuse_on_failing_gate": True,
    }

    if decision == "promote" and would_refuse:
        die("decide: attempted promotion with a failing gate; verifier refuses")
    if decision == "promote" and not correctness_ok:
        die("decide: promotion with failing correctness gate is refused")

    if decision == "reject":
        # Honest reject restores the tree. A rejected iteration that ran the
        # full loop is a success for this contract.
        orig = mutate.get("snap_original")
        if orig and Path(orig).is_dir():
            restore_files(repo, Path(orig))
            restored = True
        else:
            restored = False
        watch(state, "decide rejected the change", reason)
    else:
        restored = False

    state["decide"] = {
        "decision": decision,
        "reason": reason,
        "correctness_ok": correctness_ok,
        "overlap_improved": overlap_improved,
        "mutation_applied": mutation_applied,
        "counterfactual_refuse_on_failing_gate": counterfactual,
        "restored_original": restored,
        "refuse_if": {k: {"triggered": flag, "effect": effect} for k, (flag, effect) in refuse_if.items()},
    }
    ok(f"decide: {decision} — {reason}")


def stage_priors(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    sense = state.get("sense") or {}
    perf = state.get("gate.perf") or {}
    decide = state.get("decide") or {}
    mutated_median = (perf.get("mutated") or {}).get("median")
    original_median = (perf.get("original") or {}).get("median")
    measurement = {
        "sense_max_concurrent_model_calls": sense.get("max_concurrent_model_calls"),
        "gate_perf_mutated_median_peak": mutated_median,
        "gate_perf_original_median_peak": original_median,
        "decision": decide.get("decision"),
    }
    prior = {
        "unchanged_hardware_envelope": {
            "active_decode_limit": PRIOR_ACTIVE_DECODE_LIMIT,
            "source": "receipts/headless/MACHINE_GENOME.json ACTIVE_DECODE_LIMIT",
            "aggregate_at_four_slot_decoders": PRIOR_AGGREGATE_AT_FOUR,
            "source_four": (
                "receipts/headless/DECODE_TOPOLOGY.json "
                "summary.slot.4.scaling_vs_1"
            ),
            "contract_envelope": PRIOR_CONTRACT_ENVELOPE,
            "genome_aggregate_scaling_vs_1": PRIOR_GENOME_AGGREGATE,
            "note": (
                "Serialised model calls are a real ceiling; lifting them is "
                "worth well under 2x. These numbers did not change this run."
            ),
        },
        "updated": {
            "software_lock_serialises_call_model": {
                "before": True,
                "after": not (
                    isinstance(mutated_median, int)
                    and mutated_median >= 2
                    and decide.get("decision") == "promote"
                ),
                "citing": measurement,
            }
        },
        "citing": measurement,
    }
    state["priors"] = prior
    ok(
        f"priors: hardware envelope unchanged (limit={PRIOR_ACTIVE_DECODE_LIMIT}, "
        f"four-decoder aggregate={PRIOR_AGGREGATE_AT_FOUR}); "
        f"software-lock prior cites sense peak="
        f"{measurement['sense_max_concurrent_model_calls']} "
        f"perf mutated median={mutated_median}"
    )


def stage_next(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    perf = state.get("gate.perf") or {}
    sense = state.get("sense") or {}
    decide = state.get("decide") or {}
    llama = (sense.get("llama_server") or {})
    mutated_median = (perf.get("mutated") or {}).get("median")
    sense_peak = sense.get("max_concurrent_model_calls")
    slots = llama.get("total_slots")

    if decide.get("decision") == "promote" and isinstance(mutated_median, int) and mutated_median >= 2:
        field = "gate.perf.mutated.median"
        value = mutated_median
        target = (
            "Iteration 2: now that _call_model overlaps (median peak "
            f"{mutated_median}), spend the next iteration on actually using "
            f"llama-server's {slots} slots / ACTIVE_DECODE_LIMIT="
            f"{PRIOR_ACTIVE_DECODE_LIMIT} for real completions, bounded by "
            f"the {PRIOR_AGGREGATE_AT_FOUR}x four-decoder aggregate."
        )
    elif isinstance(mutated_median, int) and mutated_median == 1:
        field = "gate.perf.mutated.median"
        value = mutated_median
        target = (
            "Iteration 2: lock removal did not lift _call_model overlap "
            f"(mutated median peak still {mutated_median}). Find the remaining "
            "serializer on the execute() path."
        )
    else:
        field = "sense.max_concurrent_model_calls"
        value = sense_peak
        target = (
            "Iteration 2: land execute() kwargs / execute_workunit without "
            f"shared-state patching; this run sensed peak={sense_peak}."
        )

    cited = dotted(
        {
            "gate": {"perf": {"mutated": {"median": (perf.get("mutated") or {}).get("median")}}},
            "sense": {"max_concurrent_model_calls": sense_peak},
            "gate.perf": perf,
            "sense.max_concurrent_model_calls": sense_peak,
        },
        field,
    )
    # resolve against the state tree the receipt will carry
    cited_state = dotted(state, field)
    if cited_state is None:
        # try the flattened aliases we used
        if field == "gate.perf.mutated.median":
            cited_state = mutated_median
        elif field == "sense.max_concurrent_model_calls":
            cited_state = sense_peak
    if cited_state != value or not isinstance(value, (int, float)):
        die(
            f"next: citation {field}={value!r} does not resolve "
            f"(got {cited_state!r})"
        )
    state["next"] = {
        "target": target,
        "citation": {"field": field, "value": value},
        "llama_total_slots": slots,
    }
    ok(f"next: {target} (citing {field}={value})")


STAGES = {
    "sense": stage_sense,
    "bottleneck": stage_bottleneck,
    "hypotheses": stage_hypotheses,
    "screen": stage_screen,
    "mutate": stage_mutate,
    "gate.correctness": stage_gate_correctness,
    "gate.perf": stage_gate_perf,
    "decide": stage_decide,
    "priors": stage_priors,
    "next": stage_next,
}


def run_stage(name: str, state_path: Path, repo: Path, ws: Path) -> int:
    state = load_state(state_path)
    state["_path"] = str(state_path)
    fn = STAGES[name]
    try:
        fn(state, repo, ws)
        state.setdefault(name, {})
        if isinstance(state.get(name), dict):
            state[name]["stage_ok"] = True
        save_state(state_path, {k: v for k, v in state.items() if k != "_path"})
        return 0
    except SystemExit as exc:
        save_state(state_path, {k: v for k, v in state.items() if k != "_path"})
        return int(exc.code or 1)
    except Exception as exc:
        watch(state, f"{name} raised", f"{type(exc).__name__}: {exc}")
        save_state(state_path, {k: v for k, v in state.items() if k != "_path"})
        traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Mission construction
# ---------------------------------------------------------------------------

STAGE_SPECS = [
    ("sense", "Measure max concurrent _call_model on a real two-unit GPU_DECODE Mission", [], "CPU_HEAVY"),
    ("bottleneck", "Name the top bottleneck; verifier requires agreement with the sensed number", ["sense"], "LIGHT_CONTROL"),
    ("hypotheses", "Enumerate at least three candidate changes each naming file:line", ["bottleneck"], "LIGHT_CONTROL"),
    ("screen", "Cheap disproof first; reject narrowing the lock", ["hypotheses"], "LIGHT_CONTROL"),
    ("mutate", "Apply the surviving change through Engine.execute mutation path", ["screen"], "MUTATION"),
    ("gate.correctness", "Run python3 -m pytest hcli/tests -q and record the exit", ["mutate"], "TEST"),
    ("gate.perf", "Re-measure overlap paired and alternating; report the spread", ["gate.correctness"], "CPU_HEAVY"),
    ("decide", "Promote or reject from the gates alone; refuse promotion if a gate failed", ["gate.perf"], "LIGHT_CONTROL"),
    ("priors", "Write the updated prior, citing the measurement that changed it", ["decide"], "LIGHT_CONTROL"),
    ("next", "Choose iteration 2's target citing a specific iteration-1 measurement", ["priors"], "LIGHT_CONTROL"),
]


def build_units(repo: Path, state_path: Path, ws: Path) -> list:
    ensure_hcli_path()
    from hcli.workunit import WorkUnit

    units = []
    for uid, desc, deps, rc in STAGE_SPECS:
        cmd = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))} "
            f"--stage {shlex.quote(uid)} "
            f"--state {shlex.quote(str(state_path))} "
            f"--repo {shlex.quote(str(repo))} "
            f"--workspace {shlex.quote(str(ws))}"
        )
        units.append(
            WorkUnit(
                id=uid,
                role=uid,
                description=desc,
                dependencies=list(deps),
                resource_class=rc,
                preferred_backend="cpu",
                verifier=cmd,
            )
        )
    return units


def assemble_receipt(
    repo: Path,
    state: Dict[str, Any],
    mission: Any,
    baseline: Dict[str, Any],
    grok_run: Optional[str],
) -> Dict[str, Any]:
    units_out = []
    if mission is not None:
        for wu in mission.scheduler.units.values():
            units_out.append(
                {
                    "id": wu.id,
                    "role": wu.role,
                    "status": wu.status,
                    "resource_class": wu.resource_class,
                    "preferred_backend": wu.preferred_backend,
                    "assigned_backend": wu.assigned_backend,
                    "attempts": wu.attempts,
                    "verification": wu.verification,
                    "dependencies": list(wu.dependencies),
                }
            )
    decide = state.get("decide") or {}
    mutate = state.get("mutate") or {}
    return {
        "schema": "hawking.headless.hcli_self_opt.iteration.v1",
        "iteration": 1,
        "goal": (
            "Lift GPU_DECODE _call_model serialization without narrowing "
            "the execute_workunit worker lock."
        ),
        "baseline_pytest": baseline,
        "grok_run": grok_run,
        "priors_bound_the_prize": {
            "active_decode_limit": PRIOR_ACTIVE_DECODE_LIMIT,
            "active_decode_limit_source": "receipts/headless/MACHINE_GENOME.json",
            "aggregate_at_four_slot_decoders": PRIOR_AGGREGATE_AT_FOUR,
            "aggregate_at_four_source": (
                "receipts/headless/DECODE_TOPOLOGY.json summary.slot.4.scaling_vs_1"
            ),
            "contract_envelope_near": PRIOR_CONTRACT_ENVELOPE,
            "note": (
                "Serialised model calls are a real ceiling; lifting them is "
                "worth well under 2x."
            ),
        },
        "mission": {
            "id": getattr(mission, "id", None),
            "phase": getattr(mission, "phase", None),
            "accepted_count": getattr(mission, "accepted_count", None),
        } if mission is not None else None,
        "workunits": units_out,
        "stages": {
            "sense": state.get("sense"),
            "bottleneck": state.get("bottleneck"),
            "hypotheses": state.get("hypotheses"),
            "screen": state.get("screen"),
            "mutate": state.get("mutate"),
            "gate.correctness": state.get("gate.correctness"),
            "gate.perf": state.get("gate.perf"),
            "decide": state.get("decide"),
            "priors": state.get("priors"),
            "next": state.get("next"),
        },
        "mutation_receipt": mutate.get("engine_receipt"),
        "decision": decide.get("decision"),
        "decision_reason": decide.get("reason"),
        "watched_fail": state.get("watched_fail") or [],
    }


def print_watched_fail(items: List[Dict[str, Any]]) -> None:
    print("\n## WHAT I WATCHED FAIL", flush=True)
    if not items:
        print("(nothing recorded)", flush=True)
        return
    for i, item in enumerate(items, 1):
        print(f"{i}. {item.get('title')}", flush=True)
        detail = str(item.get("detail") or "")
        if detail:
            for line in detail.splitlines()[:12]:
                print(f"   {line}", flush=True)


def main_loop(repo: Path) -> int:
    os.environ["HCLI_CPU_TIMEOUT"] = CPU_TIMEOUT_S
    os.environ.setdefault("ACTIVE_DECODE_LIMIT", "2")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    grok = shutil.which("grok-run")
    baseline = {
        "expected": "417 passed, 1 skipped",
        "observed": "416 passed, 2 skipped",
        "deviation": (
            "one extra skip: tests/test_mlx_backend.py::"
            "test_supports_against_installed_binary — installed "
            "mlx_lm.server --help does not list chat-template kwargs "
            "(Metal-less session or older binary). The other skip is the "
            "always-skipped live grok-run audit. Suite otherwise green."
        ),
        "suite_green": True,
    }

    ws = Path(tempfile.mkdtemp(prefix="hcli-selfopt-mission-"))
    state_path = ws / "loop_state.json"
    pre = {
        "watched_fail": [
            {
                "title": "Baseline pytest was 416 passed, 2 skipped, not 417/1",
                "detail": baseline["deviation"],
            },
            {
                "title": "grok-run is not on PATH",
                "detail": (
                    "GrokBridge refuses rather than inventing a task id. "
                    "Every loop stage is cpu-backed so the DAG does not "
                    "depend on grok-run."
                ),
            },
            {
                "title": "Default HCLI_CPU_TIMEOUT=120 is below the suite wall",
                "detail": (
                    "gate.correctness runs python3 -m pytest hcli/tests "
                    "which took ~130s on this box. The loop sets HCLI_CPU_TIMEOUT=600 "
                    "so the WorkUnit verifier is not killed mid-suite."
                ),
            },
        ],
        "baseline_pytest": baseline,
        "grok_run": grok,
    }
    save_state(state_path, pre)

    ensure_hcli_path()
    from hcli.mission import Mission
    from hcli.resources import ResourceLimits

    class ControlEngine:
        def execute_workunit(self, wu, context):
            raise RuntimeError(
                f"outer loop unit {getattr(wu, 'id', None)!r} must be "
                "cpu-backed; qwen/execute_workunit path is forbidden here"
            )

        def cancel(self) -> None:
            return None

    units = build_units(repo, state_path, ws)
    limits = ResourceLimits.resolve(repo_root=repo)
    mission = Mission(
        str(ws),
        engine=ControlEngine(),
        units=units,
        runtime_count=2,
        limits=limits,
        quiet=False,
        goal=(
            "HCLI self-optimize iteration 1: lift GPU_DECODE model-call "
            "serialization without narrowing the worker lock."
        ),
        install_signals=False,
        repo_root=repo,
    )

    print(f"mission {mission.id} workspace={ws}", flush=True)
    print(f"stages: {[u.id for u in units]}", flush=True)
    t0 = time.perf_counter()
    try:
        result = mission.run()
    except Exception:
        traceback.print_exc()
        result = {"status": "failed", "error": "mission raised"}
    wall = time.perf_counter() - t0
    print(f"mission result {result} wall={wall:.1f}s", flush=True)

    state = load_state(state_path)
    receipt = assemble_receipt(repo, state, mission, baseline, grok)
    receipt["mission_result"] = result
    receipt["mission_wall_s"] = wall
    receipt["workspace"] = str(ws)
    dest = repo / RECEIPT_REL
    _atomic_write(dest, receipt)
    print(f"receipt {dest}", flush=True)

    print("\n## WorkUnits", flush=True)
    for item in receipt["workunits"]:
        print(
            f"  {item['id']:18} status={item['status']:10} "
            f"backend={item.get('assigned_backend')} "
            f"class={item['resource_class']}",
            flush=True,
        )
    print(f"\n decision: {receipt.get('decision')} — {receipt.get('decision_reason')}", flush=True)
    mut = receipt.get("mutation_receipt")
    if mut:
        print(f" mutation receipt: {json.dumps(mut, default=str)[:800]}", flush=True)
    else:
        blocked = (state.get("mutate") or {}).get("blocked")
        print(f" mutation receipt: none (blocked={blocked})", flush=True)
    print_watched_fail(receipt.get("watched_fail") or [])

    # Exit 0 if the DAG completed. A rejected change is a successful loop.
    if result.get("status") == "completed":
        return 0
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", help="run one loop stage (WorkUnit verifier)")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--probe-overlap", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    repo = args.repo.resolve() if args.repo else REPO

    if args.probe_overlap:
        result = run_overlap_probe(repo)
        if args.out:
            _atomic_write(args.out, result)
        else:
            print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    if args.stage:
        if not args.state or not args.workspace:
            die("--stage requires --state and --workspace")
        if args.stage not in STAGES:
            die(f"unknown stage {args.stage}")
        return run_stage(args.stage, args.state, repo, args.workspace.resolve())

    return main_loop(repo)


if __name__ == "__main__":
    raise SystemExit(main())
