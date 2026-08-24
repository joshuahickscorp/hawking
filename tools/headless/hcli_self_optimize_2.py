#!/usr/bin/env python3
"""HCLI self-optimize iteration 2, remasured.

The first iteration-2 receipt PROMOTED on a throughput number that did
not measure the mutation. cond=="mutated" set width=2 and cond=="original"
set width=1, then fan_completions(width) POSTed straight at the live
llama-server. Completions never entered the RuntimePool whose _admit the
mutation changes (Controller.ensure_runtime_pool passing observed_overlap
so the cap lifts 1→2). The 25.3 vs 22.6 tok/s gap was llama-server's own
--parallel 2 handling of two HTTP requests; a no-op mutation produces
the same number. admitted_n was a FakeBackend side probe.

This remasurement (SELF_OPT_EXECUTION, settled):

* Routes every gated completion through Controller.ensure_runtime_pool
  → RuntimePool.complete → an attach-only backend on :52484. The live
  27B is never spawned and never killed. RuntimePool.__init__ is patched
  only to inject that backend and a lenient MemGate; observed_overlap
  and workspace still come from Controller, which is the mutation.
* Offers the same load (two concurrent pool.complete calls) under both
  the mutated and the original Controller. Admission, not the caller,
  is what is allowed to change the number.
* Promotes only if Engine validation.ok is True. tests=[] → NO_EVIDENCE
  → REFUSED, even if tok/s moved.
* Runs a real failing-gate trial (pytest assert False) and records that
  compute_decision returns REFUSED; would_refuse_on_failing_gate is
  evidenced, not hardcoded.

Default invocation is the PROMOTION-control harness (G021). It does not
re-run the ten-WorkUnit mission and does not write hcli/. Four controls
run through Controller.ensure_runtime_pool on a git-archive scratch copy
of HEAD hcli, so a missing worktree checkout is not a hole:

1. NO-OP candidate — mutation changes bytes, not admission. Must not win.
2. BAD candidate — requested_n=0, genuinely worse. Gate must REFUSE.
3. Paired / interleaved trials — not a baseline block then a candidate block.
4. Failing-gate refusal, physically exercised — pytest assert False fed
   into compute_decision; would_refuse_on_failing_gate is computed.

`--mission-loop` restores the iteration-2 Mission DAG. Do not spawn a
second llama-server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
CONTROLLER_REL = Path("hcli/controller.py")
RUNTIME_REL = Path("hcli/runtime.py")
RECEIPT_REL = Path("receipts/headless/HCLI_SELF_OPT_ITERATION_2_REMEASURED.json")
PROMOTION_RECEIPT_REL = Path("receipts/headless/SELF_OPT_CANDIDATE_PROMOTED.json")
MUTATION_TEST = "tools/headless/hcli_self_optimize_2.py"
PROMOTION_TRIALS = 4  # interleaved candidate/baseline pairs, not two blocks
OFFERED_STREAMS = 2
LLAMA_PORT = 52484
PROBE_DELAY_S = 0.35
CPU_TIMEOUT_S = "600"
PROBE_IDS = ("g0", "g1")
N_PREDICT = 96
WARMUP_PREDICT = 16
THROUGHPUT_PROMPT = "Count upward by ones starting from one: 1 2 3 4 5"
DIR_VERIFIER = (
    "python3 -c "
    "\"import pathlib,sys; sys.exit(0 if pathlib.Path('.').exists() else 1)\""
)

# Priors that bound the prize. Cited, not re-derived.
PRIOR_ACTIVE_DECODE_LIMIT = 2
PRIOR_AGGREGATE_AT_FOUR = 1.2161  # DECODE_TOPOLOGY summary.slot.4.scaling_vs_1
PRIOR_CONTRACT_ENVELOPE = 1.26
PRIOR_GENOME_AGGREGATE = 1.1934
PRIOR_RECOMMENDED_WS_GIB = 77.76
PRIOR_PER_RUNTIME_GIB = 19.79
PRIOR_TWO_SERVER_TPS = 3.986
PRIOR_ONE_SERVER_TPS = 33.47


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


def _resolved_sys_path_entry(p: str) -> Optional[str]:
    if p is None or p == "":
        return p
    try:
        return str(Path(p).resolve())
    except Exception:
        return os.path.abspath(p)


def _hcli_import_parent() -> str:
    """Directory that must supply `hcli` for this process.

    Probe children pin this to the git-archive scratch copy via
    HCLI_SELFOPT_HCLI_PARENT. The invoking worktree is the wrong parent:
    when that worktree has hcli/ on disk, inserting it at sys.path[0]
    binds HEAD (already H1) and the scratch variant is ignored.
    """
    pinned = os.environ.get("HCLI_SELFOPT_HCLI_PARENT")
    if pinned:
        return str(Path(pinned).resolve())
    return str(HCLI_PARENT.resolve())


def ensure_hcli_path() -> None:
    """Pin the chosen hcli parent at sys.path[0].

    Insert-if-absent is not enough: a later insert of the invoking
    worktree can sit in front of the scratch copy. Always move the
    chosen parent to the front.
    """
    parent = _hcli_import_parent()
    kept = []
    for p in sys.path:
        if _resolved_sys_path_entry(p) == parent:
            continue
        kept.append(p)
    sys.path[:] = kept
    sys.path.insert(0, parent)


def pin_hcli_import_root(root: Path) -> Path:
    """Force the next `import hcli` to load the package under root.

    G021 divergence: `_install_fake_pool_patches` called `ensure_hcli_path()`,
    which inserted the invoking worktree at sys.path[0]. Then
    `from hcli.runtime import RuntimePool` bound that tree's module. `hcli`
    stayed in sys.modules, so `from hcli.controller import Controller` used
    HEAD — already H1 — while the wiring marker was read from the stripped
    scratch file. Baseline therefore reported observed_overlap_ctor=2 with
    controller_has_h1_wiring=false.

    A sparse hole does not have hcli/ at the worktree, so the same child
    fell through to scratch and the variants actually ran. That is why the
    lane PROMOTE (baseline admitted_n [1,1]) did not reproduce on a
    checkout where hcli/ was present (both arms [2,2]).

    The editable-install meta-path finder was tested and refuted: scratch
    first, sys.modules purged, resolves to scratch. This pin is the
    worktree-directory case that test never exercised.
    """
    root_s = str(Path(root).resolve())
    marker = Path(root_s) / "hcli" / "controller.py"
    if not marker.is_file():
        die(f"pin_hcli_import_root: {marker} is not a file")
    os.environ["HCLI_SELFOPT_HCLI_PARENT"] = root_s
    for mod in list(sys.modules):
        if mod == "hcli" or mod.startswith("hcli."):
            del sys.modules[mod]
    kept = []
    for p in sys.path:
        resolved = _resolved_sys_path_entry(p)
        if resolved == root_s:
            continue
        if resolved:
            other = Path(resolved)
            if (other / "hcli" / "__init__.py").is_file() or (
                other / "hcli" / "controller.py"
            ).is_file():
                # Another tree that could supply hcli/. Drop it so a
                # materialized worktree (or a git-archive shadow used in
                # tests) cannot win over scratch.
                continue
        kept.append(p)
    sys.path[:] = kept
    sys.path.insert(0, root_s)
    return Path(root_s)


def _hcli_loaded_from(root: Path) -> Dict[str, Any]:
    """Identity of the imported hcli package. Empty if not imported yet."""
    root_s = str(Path(root).resolve())
    mod = sys.modules.get("hcli")
    ctrl = sys.modules.get("hcli.controller")
    runtime = sys.modules.get("hcli.runtime")
    hcli_file = getattr(mod, "__file__", None)
    controller_file = getattr(ctrl, "__file__", None)
    runtime_file = getattr(runtime, "__file__", None)

    def _under(path: Optional[str]) -> bool:
        if not path:
            return False
        try:
            return Path(path).resolve().is_relative_to(Path(root_s))
        except Exception:
            return str(Path(path).resolve()).startswith(root_s + os.sep)

    return {
        "hcli_file": hcli_file,
        "controller_file": controller_file,
        "runtime_file": runtime_file,
        "import_root": root_s,
        "import_root_is_scratch": bool(
            _under(hcli_file) and _under(controller_file) and _under(runtime_file)
        ),
    }


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


def _http_json(url: str, timeout: float = 2.0) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", "replace")), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# llama-server liveness — attach only, never spawn
# ---------------------------------------------------------------------------

def llama_snapshot(port: int = LLAMA_PORT) -> Dict[str, Any]:
    base = f"http://127.0.0.1:{port}"
    out: Dict[str, Any] = {
        "port": port,
        "health": None,
        "total_slots": None,
        "error": None,
        "slots_processing": None,
    }
    body, err = _http_json(base + "/health", timeout=2)
    if err:
        out["error"] = f"health: {err}"
        return out
    if isinstance(body, dict):
        out["health"] = body.get("status") or body
    else:
        out["health"] = body
    props, perr = _http_json(base + "/props", timeout=2)
    if perr:
        out["props_error"] = perr
    elif isinstance(props, dict):
        out["total_slots"] = props.get("total_slots")
        out["model_path"] = props.get("model_path")
    slots, serr = _http_json(base + "/slots", timeout=2)
    if not serr and isinstance(slots, list):
        out["slot_count"] = len(slots)
        out["slots_processing"] = sum(
            1 for item in slots if isinstance(item, dict) and item.get("is_processing")
        )
    return out


def llama_completion(
    port: int,
    n_predict: int,
    prompt: str = THROUGHPUT_PROMPT,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - t0,
        }
    wall = time.perf_counter() - t0
    timings = body.get("timings") if isinstance(body, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    predicted_n = timings.get("predicted_n")
    try:
        predicted_n = int(predicted_n) if predicted_n is not None else None
    except (TypeError, ValueError):
        predicted_n = None
    pred_tps = timings.get("predicted_per_second")
    try:
        pred_tps = float(pred_tps) if pred_tps is not None else None
    except (TypeError, ValueError):
        pred_tps = None
    delivered = None
    if predicted_n is not None and wall > 0:
        delivered = predicted_n / wall
    return {
        "ok": True,
        "wall_s": wall,
        "predicted_n": predicted_n,
        "predicted_per_second": pred_tps,
        "prompt_n": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "delivered_tps": delivered,
        "content_preview": str((body or {}).get("content") or "")[:80],
    }


def fan_completions(n_streams: int, n_predict: int, port: int = LLAMA_PORT) -> Dict[str, Any]:
    """Identical total work: n_streams sequences of n_predict tokens.

    n_streams=1 is one request. Callers who want two serial sequences
    invoke this twice. n_streams=2 posts both at once against the live
    2-slot server.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * n_streams

    def run(i: int) -> None:
        results[i] = llama_completion(port, n_predict)

    threads = [
        threading.Thread(target=run, args=(i,), name=f"tps-{i}")
        for i in range(n_streams)
    ]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    ok_rows = [row for row in results if row and row.get("ok")]
    tokens = sum(int(row.get("predicted_n") or 0) for row in ok_rows)
    aggregate = (tokens / wall) if wall > 0 else None
    return {
        "n_streams": n_streams,
        "n_predict": n_predict,
        "batch_wall_s": wall,
        "ok": len(ok_rows),
        "tokens": tokens,
        "aggregate_tps": aggregate,
        "per_stream_predicted_tps": [
            (row or {}).get("predicted_per_second") for row in results
        ],
        "per_stream_wall_s": [(row or {}).get("wall_s") for row in results],
        "failures": [row for row in results if not (row and row.get("ok"))],
        "streams": results,
    }


# ---------------------------------------------------------------------------
# overlap probe — a real Mission of two GPU_DECODE units
# ---------------------------------------------------------------------------

def run_overlap_probe(repo: Path, delay_s: float = PROBE_DELAY_S) -> Dict[str, Any]:
    """Measure max concurrent _call_model on a real Mission on THIS tree."""
    ensure_hcli_path()
    os.environ["ACTIVE_DECODE_LIMIT"] = "2"

    import hcli.executors  # noqa: F401  installs Engine.execute_workunit
    from hcli.engine import Engine
    from hcli.mission import Mission
    from hcli.resources import ResourceLimits
    from hcli.workunit import WorkUnit
    from hcli.workspace import Workspace

    tmp = tempfile.mkdtemp(prefix="hcli-selfopt2-probe-")
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
# admission probe — FakeBackend, never a real llama-server
# ---------------------------------------------------------------------------

class FakeBackend:
    def __init__(self, model_path, port, n_slots=1, index=0, **_kwargs):
        self.model_path = model_path
        self.port = int(port) if port is not None else 0
        self.n_slots = n_slots
        self.index = index
        self.process = None
        self.pid = None
        self.start_time = None

    def spawn(self, **kwargs):
        if kwargs.get("port") is not None:
            self.port = int(kwargs["port"])
        if kwargs.get("n_slots") is not None:
            self.n_slots = int(kwargs["n_slots"])

    def ready(self, timeout):
        return True

    def identity(self):
        return {"backend": "fake", "port": self.port, "n_slots": self.n_slots}

    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def supports(self, feature):
        return True

    def complete(self, payload, timeout=None):
        ensure_hcli_path()
        from hcli.backends import CompletionResult

        return CompletionResult(
            raw={"ok": True, "payload": payload},
            finish_reason="stop",
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )

    def stop(self):
        return {"pid": None, "gone": True, "unreaped": []}


class AttachLiveBackend:
    """Attach to the live llama-server on LLAMA_PORT.

    spawn() never launches a process. stop() never kills one. complete()
    POSTs to /completion on the resident --parallel 2 server. RuntimePool
    still plans n_slots from _admit; that is the mutation's effect.
    """

    def __init__(self, model_path, port, n_slots=1, index=0, **_kwargs):
        self.model_path = model_path
        self.port = int(LLAMA_PORT)
        self.n_slots = max(1, int(n_slots))
        self.index = index
        self.process = None
        self.pid = None
        self.start_time = None
        self.allocated_port = port
        self._stopped = False
        self.completions = 0

    def spawn(self, **kwargs):
        if kwargs.get("n_slots") is not None:
            self.n_slots = max(1, int(kwargs["n_slots"]))
        self.port = int(LLAMA_PORT)
        self.process = None
        self.pid = None

    def ready(self, timeout):
        body, err = _http_json(f"http://127.0.0.1:{self.port}/health", timeout=min(2.0, float(timeout or 2)))
        if err:
            return False
        if isinstance(body, dict):
            return body.get("status") == "ok"
        return True

    def identity(self):
        return {
            "backend": "attach-live",
            "port": self.port,
            "n_slots": self.n_slots,
            "pid": None,
            "spawned": False,
        }

    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def supports(self, feature):
        return True

    def complete(self, payload, timeout=None):
        ensure_hcli_path()
        from hcli.backends import CompletionResult

        limit = float(timeout if timeout is not None else 180.0)
        body = dict(payload or {})
        req = urllib.request.Request(
            f"{self.endpoint()}/completion",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=limit) as resp:
            raw = json.loads(resp.read().decode("utf-8", "replace"))
        timings = raw.get("timings") if isinstance(raw, dict) else {}
        if not isinstance(timings, dict):
            timings = {}
        self.completions += 1
        return CompletionResult(
            raw=raw,
            finish_reason="stop" if isinstance(raw, dict) and raw.get("stop") else None,
            text=str((raw or {}).get("content") or "")[:80] if isinstance(raw, dict) else None,
            prompt_tokens=timings.get("prompt_n"),
            completion_tokens=timings.get("predicted_n"),
            total_tokens=None,
        )

    def stop(self):
        self._stopped = True
        return {
            "pid": None,
            "gone": True,
            "unreaped": [],
            "attached": True,
            "killed_live_server": False,
            "completions": self.completions,
        }


def _lenient_gate(topology: str = "slot"):
    ensure_hcli_path()
    from hcli.machine import GIB, MemGate

    # Swap on this box is currently ~16 GiB because the live 27B server is
    # resident. The admit probe is isolating _overlap_admit_cap, not the
    # host swap gate, so the ceiling is raised past that load.
    return MemGate(
        reserve_bytes=1,
        swap_ceiling_bytes=64 * GIB,
        model_bytes=100,
        per_runtime_overhead_bytes=100,
        headroom_frac=0.1,
        metal_info={
            "recommendedMaxWorkingSetSize": 80 * GIB,
            "currentAllocatedSize": 0,
            "source": "selfopt2-inject",
        },
        topology=topology,
    )


def _dummy_model(root: Path) -> str:
    path = root / "dummy.gguf"
    path.write_bytes(b"x" * 64)
    return str(path)


def measure_admit(workspace: Path, store_n: Optional[int] = None) -> Dict[str, Any]:
    """How many runtimes RuntimePool._admit actually plans, FakeBackend only."""
    ensure_hcli_path()
    os.environ.setdefault("HCLI_DISABLE_SIGNAL_HOOKS", "1")
    os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
    os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
    os.environ.pop("HCLI_MAX_RUNTIMES", None)
    os.environ.pop("HCLI_OBSERVED_MODEL_OVERLAP", None)
    os.environ.pop("HCLI_DECODE_TOPOLOGY", None)

    from hcli.runtime import (
        DEFAULT_OVERLAP_ADMIT_CAP,
        RuntimePool,
        load_observed_overlap,
        store_observed_overlap,
    )

    workspace.mkdir(parents=True, exist_ok=True)
    if store_n is not None:
        store_observed_overlap(workspace, int(store_n))
    model = _dummy_model(workspace)
    pool = RuntimePool(
        model,
        requested_n=2,
        workspace=workspace,
        backend_factory=FakeBackend,
        mem_gate=_lenient_gate("slot"),
        topology="slot",
        repo_root=workspace,
    )
    try:
        pool.start()
        return {
            "ok": True,
            "admitted_n": int(pool.admitted_n),
            "overlap_admit_cap": int(pool.overlap_admit_cap),
            "requested_n": 2,
            "stored": store_n,
            "loaded": load_observed_overlap(workspace),
            "default_cap": int(DEFAULT_OVERLAP_ADMIT_CAP),
            "narrowed": getattr(pool, "admission_narrowed", None),
            "refusal_reason": pool.refusal_reason,
            "n_slots_backend": getattr(pool.runtimes[0].backend, "n_slots", None)
            if pool.runtimes
            else None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
    finally:
        try:
            pool.stop()
        except Exception:
            pass


def _install_attach_pool_patches() -> None:
    """Inject attach-live backend + lenient MemGate into RuntimePool.__init__.

    Controller.ensure_runtime_pool does not take backend_factory. Without
    this patch it would spawn a second 27B. The patch must NOT set
    observed_overlap or workspace — those are the mutation.

    Import RuntimePool from the pinned scratch parent, not the invoking
    worktree. Patching a different module than Controller later imports
    is the G021 miss: the baseline arm then is not a baseline.
    """
    ensure_hcli_path()
    from hcli.runtime import RuntimePool

    if getattr(RuntimePool.__init__, "_selfopt2_attach_patched", False):
        return

    orig_init = RuntimePool.__init__

    def wrapped_init(
        self,
        model_path,
        requested_n=1,
        workspace=None,
        *,
        backend_factory=None,
        mem_gate=None,
        reserve_bytes=None,
        swap_ceiling_bytes=None,
        topology=None,
        repo_root=None,
        observed_overlap=None,
    ):
        if backend_factory is None:
            backend_factory = AttachLiveBackend
        if mem_gate is None:
            mem_gate = _lenient_gate("slot")
        if topology is None:
            topology = "slot"
        return orig_init(
            self,
            model_path,
            requested_n,
            workspace,
            backend_factory=backend_factory,
            mem_gate=mem_gate,
            reserve_bytes=reserve_bytes,
            swap_ceiling_bytes=swap_ceiling_bytes,
            topology=topology,
            repo_root=repo_root,
            observed_overlap=observed_overlap,
        )

    wrapped_init._selfopt2_attach_patched = True  # type: ignore[attr-defined]
    RuntimePool.__init__ = wrapped_init  # type: ignore[method-assign]


def _install_fake_pool_patches() -> None:
    """Inject FakeBackend + lenient MemGate into RuntimePool.__init__.

    Same contract as the attach-live patch: do NOT set observed_overlap or
    workspace. Those come from Controller, which is the mutation. Used by
    the promotion-control harness so it can run without a live llama-server
    and without spawning one.

    Must run after pin_hcli_import_root / HCLI_SELFOPT_HCLI_PARENT so
    RuntimePool and Controller come from the same scratch copy. The
    previous `ensure_hcli_path()` insert of the invoking worktree patched
    one object and constructed another when hcli/ was on disk.
    """
    ensure_hcli_path()
    from hcli.runtime import RuntimePool

    if getattr(RuntimePool.__init__, "_selfopt2_fake_patched", False):
        return

    orig_init = RuntimePool.__init__

    def wrapped_init(
        self,
        model_path,
        requested_n=1,
        workspace=None,
        *,
        backend_factory=None,
        mem_gate=None,
        reserve_bytes=None,
        swap_ceiling_bytes=None,
        topology=None,
        repo_root=None,
        observed_overlap=None,
    ):
        if backend_factory is None:
            backend_factory = FakeBackend
        if mem_gate is None:
            mem_gate = _lenient_gate("slot")
        if topology is None:
            topology = "slot"
        return orig_init(
            self,
            model_path,
            requested_n,
            workspace,
            backend_factory=backend_factory,
            mem_gate=mem_gate,
            reserve_bytes=reserve_bytes,
            swap_ceiling_bytes=swap_ceiling_bytes,
            topology=topology,
            repo_root=repo_root,
            observed_overlap=observed_overlap,
        )

    wrapped_init._selfopt2_fake_patched = True  # type: ignore[attr-defined]
    RuntimePool.__init__ = wrapped_init  # type: ignore[method-assign]


def pool_fan_completions(pool: Any, n_streams: int, n_predict: int) -> Dict[str, Any]:
    """Same offered load always: n_streams concurrent RuntimePool.complete calls."""
    results: List[Optional[Dict[str, Any]]] = [None] * n_streams

    def run(i: int) -> None:
        payload = {
            "prompt": THROUGHPUT_PROMPT,
            "n_predict": int(n_predict),
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": False,
        }
        t0 = time.perf_counter()
        try:
            cr = pool.complete(payload, timeout=180.0)
            wall = time.perf_counter() - t0
            raw = cr.raw if getattr(cr, "raw", None) is not None else {}
            timings = raw.get("timings") if isinstance(raw, dict) else {}
            if not isinstance(timings, dict):
                timings = {}
            predicted_n = timings.get("predicted_n")
            if predicted_n is None:
                predicted_n = getattr(cr, "completion_tokens", None)
            try:
                predicted_n = int(predicted_n) if predicted_n is not None else 0
            except (TypeError, ValueError):
                predicted_n = 0
            pred_tps = timings.get("predicted_per_second")
            try:
                pred_tps = float(pred_tps) if pred_tps is not None else None
            except (TypeError, ValueError):
                pred_tps = None
            results[i] = {
                "ok": True,
                "wall_s": wall,
                "predicted_n": predicted_n,
                "predicted_per_second": pred_tps,
                "delivered_tps": (predicted_n / wall) if wall > 0 else None,
                "runtime_index": getattr(cr, "runtime_index", None),
                "via": "RuntimePool.complete",
            }
        except Exception as exc:
            results[i] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - t0,
                "via": "RuntimePool.complete",
            }

    threads = [
        threading.Thread(target=run, args=(i,), name=f"pool-tps-{i}")
        for i in range(n_streams)
    ]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    ok_rows = [row for row in results if row and row.get("ok")]
    tokens = sum(int(row.get("predicted_n") or 0) for row in ok_rows)
    aggregate = (tokens / wall) if wall > 0 else None
    return {
        "n_streams": n_streams,
        "n_predict": n_predict,
        "batch_wall_s": wall,
        "ok": len(ok_rows),
        "tokens": tokens,
        "aggregate_tps": aggregate,
        "runtime_indexes": [(row or {}).get("runtime_index") for row in results],
        "per_stream_predicted_tps": [
            (row or {}).get("predicted_per_second") for row in results
        ],
        "per_stream_wall_s": [(row or {}).get("wall_s") for row in results],
        "failures": [row for row in results if not (row and row.get("ok"))],
        "streams": results,
        "via": "RuntimePool.complete",
    }


# ---------------------------------------------------------------------------
# surviving mutation (H1) — operations, not hand-typed into hcli/
# ---------------------------------------------------------------------------

def _import_old() -> str:
    return "from .runtime import RuntimePool\n"


def _import_new() -> str:
    return "from .runtime import RuntimePool, load_observed_overlap\n"


def _pool_old() -> str:
    return (
        "        if self.runtime_pool is None:\n"
        "            pool = RuntimePool(\n"
        "                model_path,\n"
        "                requested_n=self.runtime_count,\n"
        "            )\n"
    )


def _pool_new() -> str:
    return (
        "        if self.runtime_pool is None:\n"
        "            pool = RuntimePool(\n"
        "                model_path,\n"
        "                requested_n=self.runtime_count,\n"
        "                workspace=self.workspace_root,\n"
        "                repo_root=self.workspace_root,\n"
        "                observed_overlap=load_observed_overlap(self.workspace_root),\n"
        "            )\n"
    )


def _pool_noop() -> str:
    """Identity mutation: bytes change, RuntimePool constructor does not."""
    return (
        "        if self.runtime_pool is None:\n"
        "            # no-op candidate: comment only, constructor unchanged\n"
        "            pool = RuntimePool(\n"
        "                model_path,\n"
        "                requested_n=self.runtime_count,\n"
        "            )\n"
    )


def _pool_bad() -> str:
    """Genuinely worse: admit nothing even when the caller asked for 2."""
    return (
        "        if self.runtime_pool is None:\n"
        "            pool = RuntimePool(\n"
        "                model_path,\n"
        "                requested_n=0,\n"
        "            )\n"
    )


def mutation_operations() -> List[Dict[str, Any]]:
    return [
        {
            "op": "replace",
            "path": str(CONTROLLER_REL),
            "old_text": _import_old(),
            "new_text": _import_new(),
        },
        {
            "op": "replace",
            "path": str(CONTROLLER_REL),
            "old_text": _pool_old(),
            "new_text": _pool_new(),
        },
    ]


def mutation_already_applied(repo: Path) -> bool:
    text = (repo / CONTROLLER_REL).read_text(encoding="utf-8")
    return _pool_new() in text and _import_new() in text and _pool_old() not in text


def operations_applicable(repo: Path) -> Tuple[bool, str]:
    if mutation_already_applied(repo):
        return True, "already_applied"
    text = (repo / CONTROLLER_REL).read_text(encoding="utf-8")
    missing = []
    if _import_old() not in text:
        missing.append("controller RuntimePool import")
    if _pool_old() not in text:
        missing.append("controller RuntimePool constructor")
    if missing:
        return False, "missing anchors: " + ", ".join(missing)
    for op in mutation_operations():
        blob = (repo / op["path"]).read_text(encoding="utf-8")
        n = blob.count(op["old_text"])
        if n != 1:
            return False, f"{op['path']}: old_text occurs {n} times, need 1"
    return True, "applicable"


def apply_ops_to_copy(repo: Path, dest: Path) -> Tuple[bool, str]:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "controller.py"
    target.write_text((repo / CONTROLLER_REL).read_text(encoding="utf-8"), encoding="utf-8")
    try:
        text = target.read_text(encoding="utf-8")
        text = text.replace(_import_old(), _import_new(), 1)
        text = text.replace(_pool_old(), _pool_new(), 1)
        target.write_text(text, encoding="utf-8")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(target)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        return False, f"py_compile controller.py: {proc.stderr[-400:]}"
    if "observed_overlap=load_observed_overlap" not in target.read_text(encoding="utf-8"):
        return False, "scratch apply did not land observed_overlap="
    return True, "compiled"


def restore_files(repo: Path, snap_dir: Path) -> None:
    src = snap_dir / "controller.py"
    if src.is_file():
        (repo / CONTROLLER_REL).write_bytes(src.read_bytes())


def snapshot_pair(repo: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / CONTROLLER_REL, dest / "controller.py")


def clear_controller_pyc(repo: Path) -> None:
    pycache = (repo / CONTROLLER_REL).parent / "__pycache__"
    if not pycache.is_dir():
        return
    for pyc in pycache.glob("controller*.pyc"):
        pyc.unlink(missing_ok=True)


def revert_mutation_on_disk(repo: Path) -> Tuple[bool, str]:
    """Undo H1 so Engine.execute can apply it with a real test."""
    path = repo / CONTROLLER_REL
    if not path.is_file():
        return False, f"missing {CONTROLLER_REL}"
    text = path.read_text(encoding="utf-8")
    if _pool_old() in text and _import_old() in text and _pool_new() not in text:
        return True, "already_original"
    if _pool_new() not in text:
        return False, "mutated pool constructor not found; cannot revert"
    if _import_new() not in text:
        return False, "mutated import not found; cannot revert"
    text = text.replace(_import_new(), _import_old(), 1)
    text = text.replace(_pool_new(), _pool_old(), 1)
    path.write_text(text, encoding="utf-8")
    clear_controller_pyc(repo)
    if mutation_already_applied(repo):
        return False, "revert wrote bytes but mutation_already_applied is still True"
    if _pool_old() not in path.read_text(encoding="utf-8"):
        return False, "revert did not restore original constructor"
    return True, "reverted"


def mutation_test_command() -> str:
    return shlex.join(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            MUTATION_TEST,
        ]
    )


def mutation_validation_ok(mutate: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    receipt = mutate.get("engine_receipt")
    validation: Any = None
    if isinstance(receipt, dict):
        validation = receipt.get("validation")
    if validation is None:
        validation = mutate.get("validation")
    if not isinstance(validation, dict):
        return False, "NO_EVIDENCE"
    if validation.get("ok") is True:
        return True, None
    reason = validation.get("reason")
    if not reason:
        reason = "validation.ok is not true"
    return False, str(reason)


def test_h1_controller_wires_observed_overlap_into_runtimepool() -> None:
    """Fails before H1, passes after. Engine.execute uses this as evidence.

    Collected only when pytest is pointed at this file (the mutation's
    tests= list). Not part of tools/haider/hcli/tests.
    """
    repo = Path(__file__).resolve().parents[2]
    text = (repo / CONTROLLER_REL).read_text(encoding="utf-8")
    assert "from .runtime import RuntimePool, load_observed_overlap" in text
    assert "workspace=self.workspace_root" in text
    assert "repo_root=self.workspace_root" in text
    assert "observed_overlap=load_observed_overlap(self.workspace_root)" in text


# ---------------------------------------------------------------------------
# location resolver
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
    unmeasured = measure_admit(ws / "admit_unmeasured", store_n=None)
    stored = None
    peak = probe.get("max_concurrent_model_calls")
    if isinstance(peak, int) and peak >= 1:
        stored = measure_admit(ws / "admit_stored", store_n=peak)
    payload = {
        "llama_server": llama,
        "probe": probe,
        "max_concurrent_model_calls": peak,
        "observed_max_gpu_decode": probe.get("observed_max_gpu_decode"),
        "enter_spread_s": probe.get("enter_spread_s"),
        "delay_s": PROBE_DELAY_S,
        "unmeasured_admission": unmeasured,
        "stored_admission": stored,
        "unmeasured_admitted_n": unmeasured.get("admitted_n"),
        "stored_admitted_n": None if stored is None else stored.get("admitted_n"),
    }
    state["sense"] = payload
    if llama.get("health") != "ok":
        watch(state, "llama-server health not ok", json.dumps(llama, default=str))
        save_state(Path(state["_path"]), state)
        die("sense: live llama-server on :52484 is required for gate.perf")
    slots = llama.get("total_slots")
    if slots != 2:
        watch(
            state,
            "llama-server total_slots is not 2",
            json.dumps(llama, default=str),
        )
    if not probe.get("ok"):
        watch(state, "sense probe failed", json.dumps(probe, default=str)[:2000])
        save_state(Path(state["_path"]), state)
        die("sense: overlap probe did not run to completion")
    if not isinstance(peak, int):
        die("sense: probe did not write a number")
    if not unmeasured.get("ok"):
        die(f"sense: unmeasured admit probe failed: {unmeasured.get('error')}")
    watch(
        state,
        "current-tree overlap vs unmeasured RuntimePool admission",
        (
            f"observed_max_gpu_decode={payload['observed_max_gpu_decode']} "
            f"max_concurrent_model_calls={peak} enter_spread_s="
            f"{payload['enter_spread_s']} unmeasured_admitted_n="
            f"{unmeasured.get('admitted_n')} stored_admitted_n="
            f"{payload['stored_admitted_n']} llama_slots={slots}"
        ),
    )
    ok(
        f"sense: observed_max_gpu_decode={payload['observed_max_gpu_decode']} "
        f"max_concurrent_model_calls={peak} spread={payload['enter_spread_s']:.4f}s "
        f"unmeasured_admitted_n={unmeasured.get('admitted_n')} "
        f"stored_admitted_n={payload['stored_admitted_n']} "
        f"llama={llama.get('health')} slots={slots}"
    )


def stage_bottleneck(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    sense = state.get("sense") or {}
    peak = sense.get("max_concurrent_model_calls")
    observed = sense.get("observed_max_gpu_decode")
    admitted = sense.get("unmeasured_admitted_n")
    loc = resolve_loc(repo, "hcli/runtime.py:651-710")
    agrees = (
        isinstance(peak, int)
        and peak >= 2
        and isinstance(observed, int)
        and observed >= 2
        and admitted == 1
    )
    named = (
        "RuntimePool._admit narrows requested_n to _overlap_admit_cap, which "
        "is 1 unless constructor observed_overlap, HCLI_OBSERVED_MODEL_OVERLAP, "
        "or workspace .hcli/model_overlap.json says otherwise. Iteration 1 "
        "made _call_model overlap, but unmeasured admission is still 1, so "
        "the pool still plans a single slot and does not spend llama-server's "
        "2 slots / ACTIVE_DECODE_LIMIT=2 on real completions. Extra process "
        "runtimes are the 19.79 GiB cost; extra SLOT runtimes share weights."
    )
    payload = {
        "name": named,
        "location": "hcli/runtime.py:651-710",
        "resolved": loc,
        "agrees_with_sense": agrees,
        "sense_max_concurrent_model_calls": peak,
        "sense_observed_max_gpu_decode": observed,
        "unmeasured_admitted_n": admitted,
        "contradiction": None,
    }
    if not agrees:
        payload["contradiction"] = (
            "named bottleneck claims overlap>=2 with unmeasured admit=1, "
            f"but sense measured peak={peak} observed_max_gpu_decode="
            f"{observed} unmeasured_admitted_n={admitted}"
        )
        state["bottleneck"] = payload
        save_state(Path(state["_path"]), state)
        die("bottleneck: named bottleneck contradicts the measurement")
    if not loc.get("ok"):
        die(f"bottleneck: location did not resolve: {loc}")
    blob = "\n".join(loc.get("snippet") or [])
    if "_overlap_admit_cap" not in blob or "DEFAULT_OVERLAP_ADMIT_CAP" not in (
        (repo / RUNTIME_REL).read_text(encoding="utf-8")
    ):
        die("bottleneck: resolved lines do not contain the overlap admit cap")
    state["bottleneck"] = payload
    ok(
        f"bottleneck: _overlap_admit_cap default 1 agrees with peak={peak} "
        f"/ decode={observed} / unmeasured_admitted_n={admitted}"
    )


def _hypotheses() -> List[Dict[str, Any]]:
    return [
        {
            "id": "H1_highwater",
            "title": (
                "Raise admission via the measured high-water path: Controller "
                "passes workspace and observed_overlap=load_observed_overlap"
            ),
            "location": "hcli/controller.py:605-609",
            "secondary_location": "hcli/runtime.py:651-671",
            "change": (
                "ensure_runtime_pool currently constructs RuntimePool without "
                "workspace, so Engine._enter_model_call's store_observed_overlap"
                "(self.root) is not the file _overlap_admit_cap reads. Pass "
                "workspace=self.workspace_root, repo_root=self.workspace_root, "
                "and observed_overlap=load_observed_overlap(self.workspace_root) "
                "so a measured peak of 2 lifts admission to 2 on the next start."
            ),
        },
        {
            "id": "H2_default_cap",
            "title": "Hard-code DEFAULT_OVERLAP_ADMIT_CAP = 2",
            "location": "hcli/runtime.py:50",
            "change": (
                "Change the measured default of 1 to 2 so every unmeasured "
                "pool admits two runtimes without a high-water file."
            ),
        },
        {
            "id": "H3_process_second_server",
            "title": "Force PROCESS topology and spawn a second 27B llama-server",
            "location": "hcli/runtime.py:1069-1072",
            "change": (
                "Set topology=process and requested_n=2 so two independent "
                "llama-server processes decode at once."
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
    sense = state.get("sense") or {}
    unmeasured = sense.get("unmeasured_admission") or {}
    stored = sense.get("stored_admission") or {}
    if unmeasured.get("admitted_n") != 1:
        die(
            "screen: unmeasured admit probe did not admit 1 "
            f"(got {unmeasured.get('admitted_n')})"
        )
    if stored.get("admitted_n") != 2:
        die(
            "screen: high-water store of sense peak did not admit 2 "
            f"(got {stored.get('admitted_n')}); cheap check of H1 is broken"
        )
    copy_dir = ws / "h1_scratch"
    h1_ok, h1_detail = apply_ops_to_copy(repo, copy_dir)
    applicable, why = operations_applicable(repo)

    default_reason = (
        "REJECTED as correctness against the unmeasured invariant. "
        "test_unmeasured_overlap_admits_one_runtime requires "
        "DEFAULT_OVERLAP_ADMIT_CAP=1: a mission that has never overlapped "
        "must not reserve a second slot (or a 19.79 GiB process). Cheap "
        f"disproof already in hand: unmeasured admitted_n="
        f"{unmeasured.get('admitted_n')} with default_cap="
        f"{unmeasured.get('default_cap')}. Raising the default would make "
        "that probe admit 2 without evidence."
    )
    process_reason = (
        "REJECTED on measured tok/s, not theory. A native run with two "
        f"27B model servers resident delivered {PRIOR_TWO_SERVER_TPS} tok/s "
        f"against {PRIOR_ONE_SERVER_TPS} with one — an "
        f"{PRIOR_ONE_SERVER_TPS / PRIOR_TWO_SERVER_TPS:.1f}x collapse. "
        "This contract forbids spawning a second llama-server. PROCESS "
        "topology is how you buy that collapse. SLOT topology on the live "
        "--parallel 2 server is the only honest way to spend overlap=2."
    )

    verdicts = [
        {
            "id": "H1_highwater",
            "verdict": "SURVIVE" if (h1_ok and applicable) else "REJECTED",
            "reason": (
                "Raises admission via the measured high-water path the "
                "runtime already implements. Cheap check: unmeasured "
                f"admitted_n={unmeasured.get('admitted_n')}; after "
                f"store_observed_overlap(peak) admitted_n="
                f"{stored.get('admitted_n')}. Controller currently drops "
                "workspace on the floor, so Engine's store never becomes "
                "the pool's cap. Scratch apply+py_compile "
                f"ok={h1_ok} ({h1_detail}), applicable={why}."
            ),
            "scratch_apply_ok": h1_ok,
            "scratch_detail": h1_detail,
            "applicable": why,
            "unmeasured_admitted_n": unmeasured.get("admitted_n"),
            "stored_admitted_n": stored.get("admitted_n"),
        },
        {
            "id": "H2_default_cap",
            "verdict": "REJECTED",
            "reason": default_reason,
        },
        {
            "id": "H3_process_second_server",
            "verdict": "REJECTED",
            "reason": process_reason,
            "two_server_tps": PRIOR_TWO_SERVER_TPS,
            "one_server_tps": PRIOR_ONE_SERVER_TPS,
        },
    ]
    rejected = [v for v in verdicts if v["verdict"] == "REJECTED"]
    survived = [v for v in verdicts if v["verdict"] == "SURVIVE"]
    if not rejected:
        die("screen: at least one hypothesis must be rejected")
    if not any(v["id"] == "H2_default_cap" and v["verdict"] == "REJECTED" for v in verdicts):
        die("screen: H2_default_cap must be rejected")
    if not any(
        v["id"] == "H3_process_second_server" and v["verdict"] == "REJECTED"
        for v in verdicts
    ):
        die("screen: H3_process_second_server must be rejected")
    if not survived:
        die("screen: no surviving hypothesis")
    watch(state, "H2_default_cap rejected", default_reason)
    watch(state, "H3_process_second_server rejected", process_reason)
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
    """Apply H1 through Engine.execute. Mutation lock is held by Mission."""
    ensure_hcli_path()
    from hcli.engine import Engine
    from hcli.workspace import Workspace as HcliWorkspace

    snap_original = ws / "snap" / "original"
    snap_mutated = ws / "snap" / "mutated"

    already = mutation_already_applied(repo)
    if already:
        snapshot_pair(repo, snap_mutated)
        reverted, why_rev = revert_mutation_on_disk(repo)
        if not reverted:
            payload_block: Dict[str, Any] = {
                "path": "Engine.execute mutation path",
                "already_applied": True,
                "applied": False,
                "blocked": f"HEAD already has H1 but revert failed: {why_rev}",
                "engine_receipt": None,
            }
            state["mutate"] = payload_block
            watch(state, "mutate: could not revert already-applied H1", why_rev)
            ok(f"mutate: BLOCKED (revert failed: {why_rev})")
            return
        watch(
            state,
            "mutate: reverted already-applied H1 so Engine.execute can apply with a test",
            why_rev,
        )
        snapshot_pair(repo, snap_original)
    else:
        snapshot_pair(repo, snap_original)

    applicable, why = operations_applicable(repo)
    files_before = {"controller.py": sha256_file(repo / CONTROLLER_REL)}
    payload: Dict[str, Any] = {
        "path": "Engine.execute mutation path",
        "already_applied": already,
        "reverted_before_execute": already,
        "applicable": why,
        "files_before": files_before,
        "applied": False,
        "blocked": None,
        "engine_receipt": None,
        "engine_result_status": None,
        "rolled_back": None,
        "operations": mutation_operations(),
        "tests": [mutation_test_command()],
    }

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
                    "Wire Controller.ensure_runtime_pool to the measured "
                    "high-water path so RuntimePool._admit can lift from 1 to 2."
                ),
                "operations": mutation_operations(),
                "tests": [mutation_test_command()],
            }

    engine = Engine(HcliWorkspace(str(repo)), model_client=_MutationClient())
    try:
        result = engine.execute(
            "Apply the surviving self-opt mutation: Controller.ensure_runtime_pool "
            "must pass workspace, repo_root, and observed_overlap="
            "load_observed_overlap(workspace) into RuntimePool so a measured "
            "model-call overlap of 2 lifts admission. Edit "
            "hcli/controller.py only. Do not spawn a second "
            "llama-server and do not change DEFAULT_OVERLAP_ADMIT_CAP."
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

    files_after = {"controller.py": sha256_file(repo / CONTROLLER_REL)}
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
        ok("mutate: BLOCKED (rolled back)")
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
    val_ok, val_reason = mutation_validation_ok(payload)
    payload["validation_ok"] = val_ok
    payload["validation_reason"] = val_reason
    if not val_ok:
        watch(
            state,
            "mutate: Engine receipt is not validation.ok; decide must not promote",
            f"status={result.get('status')} reason={val_reason}",
        )
    state["mutate"] = payload
    ok(
        f"mutate: applied via Engine.execute status={result.get('status')} "
        f"validation.ok={val_ok} reason={val_reason} "
        f"receipt={payload.get('engine_receipt_path') or 'in-result'} "
        f"controller={files_before['controller.py'][:12]}->"
        f"{files_after['controller.py'][:12]}"
    )


def stage_gate_correctness(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    t0 = time.perf_counter()
    env = dict(os.environ)
    # Live 27B decode leaves ~16 GiB of swap; MemGate's default 2 GiB
    # ceiling then refuses FakeBackend pools and 9 RuntimePool tests
    # fail for a reason that is not the mutation. Isolate the suite
    # from that host load so the gate measures the change.
    env.setdefault("HCLI_SWAP_CEILING_GIB", "64")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tools/haider/hcli/tests", "-q", "--tb=line"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=480,
        env=env,
    )
    wall = time.perf_counter() - t0
    tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-1000:]
    payload = {
        "command": [sys.executable, "-m", "pytest", "tools/haider/hcli/tests", "-q"],
        "exit_code": proc.returncode,
        "passed_gate": proc.returncode == 0,
        "wall_s": wall,
        "output_tail": tail[-2500:],
        "hcli_swap_ceiling_gib": env.get("HCLI_SWAP_CEILING_GIB"),
    }
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


def _throughput_child(repo: Path, out: Path, width: int, n_predict: int) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo)
    env["HCLI_SELFOPT_HCLI_PARENT"] = str(repo)
    env["ACTIVE_DECODE_LIMIT"] = "2"
    env["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    env.setdefault("HCLI_SWAP_CEILING_GIB", "64")
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--probe-throughput",
            "--width",
            str(width),
            "--n-predict",
            str(n_predict),
            "--out",
            str(out),
            "--repo",
            str(repo),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=240,
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


def run_throughput_probe(
    width: int,
    n_predict: int,
    repo: Optional[Path] = None,
) -> Dict[str, Any]:
    """Completions go through Controller.ensure_runtime_pool → RuntimePool.

    Offered load is always OFFERED_STREAMS concurrent pool.complete calls.
    `width` is accepted for CLI compatibility and ignored as a fan-out
    knob — that was the original defect. Admission comes from the
    Controller on disk (mutated vs original), which is swapped by the
    parent before this child starts.

    The high-water file is written into the Controller workspace in both
    conditions. Mutated Controller passes it into RuntimePool; original
    Controller drops it. A no-op mutation therefore cannot change
    admitted_n.
    """
    del width  # not a fan-out knob; offered load is OFFERED_STREAMS
    llama = llama_snapshot()
    if llama.get("health") != "ok":
        return {"ok": False, "error": "llama-server not ok", "llama": llama}

    model = llama.get("model_path") or os.environ.get("HCLI_MODEL_PATH")
    if not model:
        return {"ok": False, "error": f"no live model_path: {model!r}", "llama": llama}

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    os.environ["ACTIVE_DECODE_LIMIT"] = "2"
    os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
    os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
    os.environ["HCLI_SWAP_CEILING_GIB"] = "64"
    os.environ["HCLI_MODEL_PATH"] = str(model)
    os.environ.pop("HCLI_OBSERVED_MODEL_OVERLAP", None)
    os.environ.pop("HCLI_MAX_RUNTIMES", None)
    os.environ.pop("HCLI_DECODE_TOPOLOGY", None)

    controller_root = Path(repo).resolve() if repo is not None else REPO
    pin_hcli_import_root(controller_root)
    _install_attach_pool_patches()
    from hcli.controller import Controller
    from hcli.runtime import load_observed_overlap, store_observed_overlap

    tmp = tempfile.mkdtemp(prefix="hcli-selfopt2-ctrl-")
    isolate = tempfile.mkdtemp(prefix="hcli-selfopt2-isolate-")
    # tempfile dirs live under the host temp dir, which on this box already
    # has a .hcli/model_overlap.json (prior pools wrote the high-water into
    # /var/folders/.../T). resolve_workspace(None) walks parents and would
    # load that file, so original Controller would admit 2 even though it
    # never passes observed_overlap — hiding the mutation. Pin ambient
    # lookup to an empty isolate that is NOT the Controller workspace.
    os.environ["HCLI_WORKSPACE"] = isolate
    os.chdir(isolate)
    store_observed_overlap(tmp, 2)
    controller_src = (controller_root / CONTROLLER_REL).read_text(encoding="utf-8")
    has_wiring = "observed_overlap=load_observed_overlap(self.workspace_root)" in controller_src

    ctrl = None
    pool = None
    try:
        ctrl = Controller(tmp, runtime_count=2, model=str(model))
        pool = ctrl.ensure_runtime_pool()
        backend = pool.runtimes[0].backend if pool.runtimes else None
        backend_name = type(backend).__name__ if backend is not None else None
        if backend_name != "AttachLiveBackend":
            return {
                "ok": False,
                "error": f"expected AttachLiveBackend, got {backend_name}",
                "admitted_n": getattr(pool, "admitted_n", None),
            }
        if getattr(backend, "pid", None) is not None:
            return {
                "ok": False,
                "error": "attach backend has a pid; refusing to risk the live server",
                "pid": backend.pid,
            }
        batch = pool_fan_completions(pool, OFFERED_STREAMS, n_predict)
        tokens = int(batch.get("tokens") or 0)
        wall = float(batch.get("batch_wall_s") or 0)
        ok_n = int(batch.get("ok") or 0)
        aggregate = batch.get("aggregate_tps")
        return {
            "ok": ok_n >= OFFERED_STREAMS and aggregate is not None,
            "path": (
                "Controller.ensure_runtime_pool -> RuntimePool.complete -> "
                "AttachLiveBackend.complete -> http://127.0.0.1:52484/completion"
            ),
            "through_controller": True,
            "through_runtime_pool": True,
            "spawned_second_server": False,
            "attached_port": LLAMA_PORT,
            "controller_has_h1_wiring": has_wiring,
            "controller_class": f"{type(ctrl).__module__}.{type(ctrl).__name__}",
            "pool_class": f"{type(pool).__module__}.{type(pool).__name__}",
            "backend_class": backend_name,
            "backend_pid": getattr(backend, "pid", None),
            "backend_n_slots": getattr(backend, "n_slots", None),
            "observed_overlap_ctor": getattr(pool, "observed_overlap", None),
            "overlap_admit_cap": getattr(pool, "overlap_admit_cap", None),
            "admitted_n": int(pool.admitted_n),
            "n_runtimes": len(pool.runtimes),
            "runtime_indexes": batch.get("runtime_indexes"),
            "loaded_overlap_on_controller_ws": load_observed_overlap(tmp),
            "loaded_overlap_on_pool_ws": load_observed_overlap(pool.workspace),
            "loaded_overlap_on_isolate": load_observed_overlap(isolate),
            "workspace_pool": str(pool.workspace),
            "workspace_controller": tmp,
            "workspace_isolate": isolate,
            "offered_streams": OFFERED_STREAMS,
            "mode": "pool_complete_two_concurrent",
            "n_predict": n_predict,
            "tokens": tokens,
            "wall_s": wall,
            "aggregate_tps": aggregate,
            "ok_streams": ok_n,
            "llama": llama,
            "streams": batch.get("streams"),
            "via": "RuntimePool.complete",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "controller_has_h1_wiring": has_wiring,
            "llama": llama,
        }
    finally:
        if pool is not None:
            try:
                pool.stop()
            except Exception:
                pass
        if ctrl is not None and getattr(ctrl, "runtime_pool", None) is not None:
            try:
                ctrl.runtime_pool.stop()
            except Exception:
                pass


def stage_gate_perf(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    llama = llama_snapshot()
    if llama.get("health") != "ok":
        die(f"gate.perf: llama-server not ok: {llama}")
    if llama.get("total_slots") != 2:
        watch(
            state,
            "gate.perf llama-server slots != 2",
            json.dumps(llama, default=str),
        )

    warmup = llama_completion(LLAMA_PORT, WARMUP_PREDICT)
    if not warmup.get("ok"):
        watch(state, "gate.perf warmup failed", json.dumps(warmup, default=str))
        die(f"gate.perf: warmup completion failed: {warmup.get('error')}")

    mutate = state.get("mutate") or {}
    applied = bool(mutate.get("applied"))
    snap_original = (
        Path(mutate["snap_original"])
        if mutate.get("snap_original")
        else ws / "snap" / "original"
    )
    snap_mutated = (
        Path(mutate["snap_mutated"])
        if mutate.get("snap_mutated")
        else ws / "snap" / "mutated"
    )
    trials: List[Dict[str, Any]] = []
    probe_dir = ws / "perf_trials"
    probe_dir.mkdir(parents=True, exist_ok=True)

    if applied and snap_original.is_dir() and snap_mutated.is_dir():
        order = ["mutated", "original", "mutated", "original"]
        alternating = True
    else:
        watch(
            state,
            "gate.perf could not alternate mutated/original source",
            f"applied={applied} original_snap={snap_original.is_dir()} "
            f"mutated_snap={snap_mutated.is_dir()} blocked={mutate.get('blocked')}",
        )
        order = ["current", "current", "current", "current"]
        alternating = False

    llama_before = llama_snapshot()
    for i, cond in enumerate(order):
        if cond == "mutated" and snap_mutated.is_dir():
            restore_files(repo, snap_mutated)
        elif cond == "original" and snap_original.is_dir():
            restore_files(repo, snap_original)
        clear_controller_pyc(repo)
        out = probe_dir / f"trial_{i}_{cond}.json"
        result = _throughput_child(
            repo, out, width=OFFERED_STREAMS, n_predict=N_PREDICT
        )
        trials.append(
            {
                "i": i,
                "condition": cond,
                "offered_streams": OFFERED_STREAMS,
                "width": result.get("admitted_n"),
                "aggregate_tps": result.get("aggregate_tps"),
                "tokens": result.get("tokens"),
                "wall_s": result.get("wall_s"),
                "admitted_n": result.get("admitted_n"),
                "overlap_admit_cap": result.get("overlap_admit_cap"),
                "observed_overlap_ctor": result.get("observed_overlap_ctor"),
                "workspace_pool": result.get("workspace_pool"),
                "loaded_overlap_on_pool_ws": result.get("loaded_overlap_on_pool_ws"),
                "n_runtimes": result.get("n_runtimes"),
                "runtime_indexes": result.get("runtime_indexes"),
                "backend_class": result.get("backend_class"),
                "backend_n_slots": result.get("backend_n_slots"),
                "backend_pid": result.get("backend_pid"),
                "through_controller": result.get("through_controller"),
                "through_runtime_pool": result.get("through_runtime_pool"),
                "controller_has_h1_wiring": result.get("controller_has_h1_wiring"),
                "path": result.get("path"),
                "via": result.get("via"),
                "ok": result.get("ok"),
                "mode": result.get("mode"),
                "ok_streams": result.get("ok_streams"),
                "child_exit": result.get("child_exit"),
                "error": result.get("error"),
            }
        )
    llama_after = llama_snapshot()
    if applied and snap_mutated.is_dir():
        restore_files(repo, snap_mutated)
        clear_controller_pyc(repo)

    if llama_before.get("model_path") != llama_after.get("model_path"):
        watch(
            state,
            "llama-server identity changed during gate.perf",
            json.dumps({"before": llama_before, "after": llama_after}, default=str),
        )

    def _stats(cond: str) -> Dict[str, Any]:
        vals = [
            float(t["aggregate_tps"])
            for t in trials
            if t.get("condition") == cond and isinstance(t.get("aggregate_tps"), (int, float))
        ]
        if not vals:
            return {
                "n": 0,
                "values": [],
                "min": None,
                "max": None,
                "median": None,
                "spread": None,
            }
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
    spread = max(
        mutated_stats.get("spread") or 0,
        original_stats.get("spread") or 0,
        0,
    )
    improved = False
    if (
        alternating
        and mutated_stats["median"] is not None
        and original_stats["median"] is not None
    ):
        improved = float(mutated_stats["median"]) > float(original_stats["median"]) + float(
            spread
        )
    def _admit_vals(cond: str) -> List[int]:
        out: List[int] = []
        for trial in trials:
            if trial.get("condition") != cond:
                continue
            n = trial.get("admitted_n")
            if isinstance(n, int):
                out.append(n)
        return out

    mutated_admits = _admit_vals("mutated") if alternating else _admit_vals("current")
    original_admits = _admit_vals("original") if alternating else _admit_vals("current")
    admission_differs = bool(mutated_admits) and bool(original_admits) and (
        set(mutated_admits) != set(original_admits)
    )
    through_pool = all(
        t.get("through_runtime_pool") and t.get("through_controller") and t.get("via") == "RuntimePool.complete"
        for t in trials
        if t.get("ok")
    )
    if not through_pool:
        watch(
            state,
            "gate.perf completions did not all go through Controller/RuntimePool",
            json.dumps(
                [
                    {
                        "i": t.get("i"),
                        "via": t.get("via"),
                        "through_controller": t.get("through_controller"),
                        "through_runtime_pool": t.get("through_runtime_pool"),
                        "backend_class": t.get("backend_class"),
                    }
                    for t in trials
                ],
                default=str,
            ),
        )
    if alternating and not admission_differs:
        watch(
            state,
            "gate.perf admitted_n did not change when the mutation was reverted",
            (
                f"mutated admitted_n={mutated_admits} original admitted_n="
                f"{original_admits}. If these match, the measurement is still "
                "insensitive to Controller.ensure_runtime_pool."
            ),
        )
    payload = {
        "alternating": alternating,
        "order": [t["condition"] for t in trials],
        "n_predict": N_PREDICT,
        "offered_streams": OFFERED_STREAMS,
        "warmup": {
            "n_predict": WARMUP_PREDICT,
            "ok": warmup.get("ok"),
            "wall_s": warmup.get("wall_s"),
            "predicted_per_second": warmup.get("predicted_per_second"),
            "delivered_tps": warmup.get("delivered_tps"),
        },
        "metric": (
            "aggregate_tps = total predicted_n / wall_s for two concurrent "
            "RuntimePool.complete calls (same offered load in both conditions). "
            "Admission is Controller.ensure_runtime_pool -> RuntimePool._admit."
        ),
        "code_path": {
            "entry": "Controller.ensure_runtime_pool",
            "pool": "RuntimePool.complete",
            "backend": "AttachLiveBackend.complete",
            "endpoint": f"http://127.0.0.1:{LLAMA_PORT}/completion",
            "offered_load": (
                f"{OFFERED_STREAMS} concurrent pool.complete calls under both "
                "mutated and original Controller"
            ),
            "does_not_call": [
                "fan_completions",
                "llama_completion as the gated measurement",
                "urllib /completion bypassing RuntimePool",
            ],
            "through_controller": through_pool,
            "through_runtime_pool": through_pool,
        },
        "trials": trials,
        "mutated": mutated_stats,
        "original": original_stats,
        "mutated_admitted_n": mutated_admits,
        "original_admitted_n": original_admits,
        "admission_differs": admission_differs,
        "spread": spread,
        "spread_mutated": mutated_stats.get("spread"),
        "spread_original": original_stats.get("spread"),
        "throughput_improved": improved,
        "improvement_predicate": (
            "mutated.median > original.median + max(spread_mutated, spread_original, 0); "
            "same offered load; only Controller wiring (and therefore admission) differs"
        ),
        "llama_before": llama_before,
        "llama_after": llama_after,
        "spawned_second_server": False,
    }
    state["gate.perf"] = payload
    if not trials or not any(t.get("ok") for t in trials):
        die("gate.perf: no successful throughput re-measure")
    if not through_pool:
        die("gate.perf: completions did not route through the mutated RuntimePool")
    ok(
        f"gate.perf: alternating={alternating} original_median_tps="
        f"{original_stats.get('median')} mutated_median_tps="
        f"{mutated_stats.get('median')} spread={spread} improved={improved} "
        f"admitted mutated={mutated_admits} original={original_admits} "
        f"admission_differs={admission_differs} via=RuntimePool.complete"
    )


def compute_decision(
    *,
    correctness_ok: bool,
    throughput_improved: bool,
    mutation_applied: bool,
    validation_ok: bool,
    validation_reason: Optional[str],
    orig_med: Any,
    mut_med: Any,
    spread: Any,
    mutation_blocked: Optional[str] = None,
    correctness_exit: Any = None,
    admission_differs: Optional[bool] = None,
    metric_name: str = "tps",
) -> Dict[str, Any]:
    """Promote IFF every gate is actually green. NO_EVIDENCE is not green."""
    refuse_if = {
        "correctness_failed": not correctness_ok,
        "throughput_did_not_improve_beyond_spread": not throughput_improved,
        "mutation_not_applied": not mutation_applied,
        "mutation_receipt_not_validated": not validation_ok,
        "admission_did_not_differ": admission_differs is False,
    }
    would_refuse = any(refuse_if.values())
    if not would_refuse:
        decision = "promote"
        verdict = "PROMOTE"
        reason = (
            "Both gates passed, mutation receipt validation.ok=true, and paired "
            "score through Controller.ensure_runtime_pool improved beyond spread "
            f"(original median {metric_name}={orig_med} mutated median "
            f"{metric_name}={mut_med} spread={spread})."
        )
    else:
        decision = "reject"
        verdict = "REFUSED"
        bits = []
        if not mutation_applied:
            bits.append(f"mutation did not apply ({mutation_blocked})")
        if not validation_ok:
            bits.append(
                "mutation receipt is not validation.ok "
                f"(reason={validation_reason or 'NO_EVIDENCE'})"
            )
        if not correctness_ok:
            bits.append(f"gate.correctness exit={correctness_exit}")
        if not throughput_improved:
            bits.append(
                "score through Controller.ensure_runtime_pool did not improve beyond spread "
                f"(original median {metric_name}={orig_med} mutated median "
                f"{metric_name}={mut_med} spread={spread})"
            )
        if admission_differs is False:
            bits.append(
                "admitted_n did not change when the mutation was reverted "
                "(gate is insensitive or mutation is a no-op on the measured path)"
            )
        reason = "REJECT: " + "; ".join(bits)
    return {
        "decision": decision,
        "verdict": verdict,
        "reason": reason,
        "correctness_ok": correctness_ok,
        "throughput_improved": throughput_improved,
        "mutation_applied": mutation_applied,
        "validation_ok": validation_ok,
        "validation_reason": validation_reason,
        "would_refuse": would_refuse,
        "refuse_if": {
            k: {"triggered": flag, "effect": "REFUSED"} for k, flag in refuse_if.items()
        },
    }


def run_failing_gate_trial(ws: Path) -> Dict[str, Any]:
    """Force a real pytest failure and show compute_decision returns REFUSED.

    Also inject a NO_EVIDENCE mutation receipt against otherwise-green gates.
    Neither trial hardcodes the verdict.
    """
    probe = Path(ws) / "failing_gate_trial"
    probe.mkdir(parents=True, exist_ok=True)
    test_path = probe / "test_forced_fail.py"
    test_path.write_text(
        "def test_forced_fail():\n"
        "    assert False, 'injected failing-gate trial'\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    failing = compute_decision(
        correctness_ok=proc.returncode == 0,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=True,
        validation_reason=None,
        orig_med=20.0,
        mut_med=25.0,
        spread=0.1,
        correctness_exit=proc.returncode,
    )
    no_evidence = compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=False,
        validation_reason="NO_EVIDENCE",
        orig_med=20.0,
        mut_med=25.0,
        spread=0.1,
    )
    return {
        "hardcoded": False,
        "method": (
            "ran pytest on a temp test that asserts False; fed that exit into "
            "compute_decision. separately invoked compute_decision with "
            "validation_ok=False reason=NO_EVIDENCE against otherwise-green gates."
        ),
        "pytest_command": [sys.executable, "-m", "pytest", str(test_path), "-q"],
        "pytest_exit_code": proc.returncode,
        "pytest_passed_gate": proc.returncode == 0,
        "pytest_output_tail": ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-800:],
        "decision_on_failing_correctness": failing,
        "decision_on_no_evidence": no_evidence,
        "would_refuse_on_failing_gate": failing.get("verdict") == "REFUSED",
        "would_refuse_on_no_evidence": no_evidence.get("verdict") == "REFUSED",
        "evidenced": (
            proc.returncode != 0
            and failing.get("verdict") == "REFUSED"
            and no_evidence.get("verdict") == "REFUSED"
        ),
    }


# ---------------------------------------------------------------------------
# G021 promotion controls — mutation caused the win, or the harness is lying
# ---------------------------------------------------------------------------

def _git_show(repo: Path, rel: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        die(f"git show HEAD:{rel} failed: {(proc.stderr or proc.stdout)[-400:]}")
    return proc.stdout


def materialize_hcli_from_head(repo: Path, dest: Path) -> Dict[str, Any]:
    """Checkout HEAD hcli into dest without touching the sparse worktree.

    This worktree often does not materialize hcli/. A missing path here is
    not evidence it is absent from git. git archive reads HEAD directly.
    """
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "-C", str(repo), "archive", "HEAD", "hcli"],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        die(
            "git archive HEAD hcli failed: "
            + (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")[-400:]
        )
    import tarfile
    import io

    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tar:
        try:
            tar.extractall(path=dest, filter="data")
        except TypeError:
            tar.extractall(path=dest)
    controller = dest / CONTROLLER_REL
    if not controller.is_file():
        die(f"git archive did not produce {CONTROLLER_REL}")
    return {
        "dest": str(dest),
        "source": "git archive HEAD hcli",
        "worktree_hcli_present": (repo / "hcli").is_dir(),
        "controller_sha256": sha256_file(controller),
        "verified_against": "HEAD",
    }


def revert_h1_text(text: str) -> str:
    if _pool_old() in text and _import_old() in text and _pool_new() not in text:
        return text
    if _pool_new() not in text or _import_new() not in text:
        die("HEAD controller.py does not contain the H1 anchors; cannot build variants")
    text = text.replace(_import_new(), _import_old(), 1)
    text = text.replace(_pool_new(), _pool_old(), 1)
    if _pool_old() not in text or _pool_new() in text:
        die("revert_h1_text did not restore the original constructor")
    return text


def apply_h1_text(text: str) -> str:
    if _pool_new() in text and _import_new() in text:
        return text
    if _pool_old() not in text or _import_old() not in text:
        die("original controller text missing RuntimePool constructor anchors")
    text = text.replace(_import_old(), _import_new(), 1)
    text = text.replace(_pool_old(), _pool_new(), 1)
    if _pool_new() not in text:
        die("apply_h1_text did not land observed_overlap=")
    return text


def controller_variants(head_text: str) -> Dict[str, str]:
    """Four controller bodies: original, H1, NO-OP, BAD. All from the same HEAD blob."""
    if _pool_new() in head_text:
        h1 = head_text
        original = revert_h1_text(head_text)
    else:
        original = head_text
        h1 = apply_h1_text(head_text)
    noop = original.replace(_pool_old(), _pool_noop(), 1)
    bad = original.replace(_pool_old(), _pool_bad(), 1)
    if noop == original:
        die("NO-OP candidate is byte-identical to original; it is not a mutation")
    if bad == original:
        die("BAD candidate is byte-identical to original; it is not a mutation")
    if "requested_n=0" not in bad:
        die("BAD candidate did not land requested_n=0")
    if "no-op candidate" not in noop:
        die("NO-OP candidate did not land its comment")
    if _pool_new() not in h1:
        die("H1 variant missing observed_overlap wiring")
    return {"original": original, "h1": h1, "noop": noop, "bad": bad}


def h1_wiring_present(text: str) -> bool:
    return (
        "from .runtime import RuntimePool, load_observed_overlap" in text
        and "workspace=self.workspace_root" in text
        and "repo_root=self.workspace_root" in text
        and "observed_overlap=load_observed_overlap(self.workspace_root)" in text
    )


def run_controller_admit_probe(repo: Optional[Path] = None) -> Dict[str, Any]:
    """Admission + FakeBackend complete through Controller.ensure_runtime_pool.

    Offered load is always OFFERED_STREAMS concurrent pool.complete calls.
    Admission is whatever Controller on disk passes into RuntimePool. A
    no-op mutation cannot change admitted_n.
    """
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    os.environ["ACTIVE_DECODE_LIMIT"] = "2"
    os.environ["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
    os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = "4"
    os.environ["HCLI_SWAP_CEILING_GIB"] = "64"
    os.environ.pop("HCLI_OBSERVED_MODEL_OVERLAP", None)
    os.environ.pop("HCLI_MAX_RUNTIMES", None)
    os.environ.pop("HCLI_DECODE_TOPOLOGY", None)

    controller_root = Path(repo).resolve() if repo is not None else REPO
    pin_hcli_import_root(controller_root)
    _install_fake_pool_patches()
    from hcli.controller import Controller
    from hcli.runtime import load_observed_overlap, store_observed_overlap

    imported = _hcli_loaded_from(controller_root)
    if not imported.get("import_root_is_scratch"):
        return {
            "ok": False,
            "error": (
                "hcli import did not bind the scratch copy: "
                f"{imported}. Baseline would not be a baseline."
            ),
            "import_identity": imported,
        }

    tmp = Path(tempfile.mkdtemp(prefix="hcli-selfopt2-admit-ws-"))
    isolate = Path(tempfile.mkdtemp(prefix="hcli-selfopt2-admit-iso-"))
    model = Path(_dummy_model(tmp))
    os.environ["HCLI_WORKSPACE"] = str(isolate)
    os.environ["HCLI_MODEL_PATH"] = str(model)
    os.chdir(isolate)
    store_observed_overlap(tmp, 2)
    controller_src = (controller_root / CONTROLLER_REL).read_text(encoding="utf-8")
    has_wiring = h1_wiring_present(controller_src)

    ctrl = None
    pool = None
    try:
        ctrl = Controller(tmp, runtime_count=2, model=str(model))
        pool = ctrl.ensure_runtime_pool()
        backend = pool.runtimes[0].backend if pool.runtimes else None
        backend_name = type(backend).__name__ if backend is not None else None
        if pool.runtimes and backend_name != "FakeBackend":
            return {
                "ok": False,
                "error": f"expected FakeBackend, got {backend_name}",
                "admitted_n": getattr(pool, "admitted_n", None),
            }
        # Snapshot overlap files at admit time, before pool.complete may
        # store a high-water mark onto pool.workspace (the isolate, when
        # the constructor does not pass workspace). Post-complete loaded=2
        # on the isolate is not the source of admitted_n.
        pool_ws = getattr(pool, "workspace", None)
        loaded_controller_at_admit = load_observed_overlap(tmp)
        loaded_pool_at_admit = (
            load_observed_overlap(pool_ws) if pool_ws is not None else None
        )
        loaded_isolate_at_admit = load_observed_overlap(isolate)
        batch = {"ok": 0, "tokens": 0, "batch_wall_s": 0.0, "aggregate_tps": None,
                 "runtime_indexes": [], "via": "RuntimePool.complete"}
        if pool.runtimes:
            batch = pool_fan_completions(pool, OFFERED_STREAMS, 1)
        return {
            "ok": True,
            "path": (
                "Controller.ensure_runtime_pool -> RuntimePool._admit / "
                "RuntimePool.complete -> FakeBackend.complete"
            ),
            "through_controller": True,
            "through_runtime_pool": True,
            "spawned_second_server": False,
            "controller_has_h1_wiring": has_wiring,
            "controller_has_noop_comment": "no-op candidate" in controller_src,
            "controller_has_requested_n_zero": "requested_n=0" in controller_src,
            "controller_class": f"{type(ctrl).__module__}.{type(ctrl).__name__}",
            "pool_class": f"{type(pool).__module__}.{type(pool).__name__}",
            "backend_class": backend_name,
            "backend_n_slots": getattr(backend, "n_slots", None) if backend else None,
            "observed_overlap_ctor": getattr(pool, "observed_overlap", None),
            "overlap_admit_cap": getattr(pool, "overlap_admit_cap", None),
            "admitted_n": int(pool.admitted_n),
            "requested_n": int(getattr(pool, "requested_n", -1)),
            "n_runtimes": len(pool.runtimes),
            "runtime_indexes": batch.get("runtime_indexes"),
            "stored": 2,
            "loaded": loaded_controller_at_admit,
            "loaded_overlap_on_controller_ws": loaded_controller_at_admit,
            "loaded_overlap_on_pool_ws": loaded_pool_at_admit,
            "loaded_overlap_on_isolate": loaded_isolate_at_admit,
            "workspace_pool": str(pool_ws) if pool_ws is not None else None,
            "workspace_controller": str(tmp),
            "workspace_isolate": str(isolate),
            "offered_streams": OFFERED_STREAMS,
            "complete_ok": batch.get("ok"),
            "complete_via": batch.get("via"),
            "via": "Controller.ensure_runtime_pool",
            "import_identity": imported,
            "hcli_file": imported.get("hcli_file"),
            "controller_file": imported.get("controller_file"),
            "runtime_file": imported.get("runtime_file"),
            "import_root_is_scratch": imported.get("import_root_is_scratch"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
            "controller_has_h1_wiring": has_wiring,
            "import_identity": imported,
        }
    finally:
        if pool is not None:
            try:
                pool.stop()
            except Exception:
                pass
        if ctrl is not None and getattr(ctrl, "runtime_pool", None) is not None:
            try:
                ctrl.runtime_pool.stop()
            except Exception:
                pass


def _admit_child(hcli_parent: Path, out: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(hcli_parent)
    env["HCLI_SELFOPT_HCLI_PARENT"] = str(hcli_parent)
    env["ACTIVE_DECODE_LIMIT"] = "2"
    env["HCLI_ACTIVE_DECODE_LIMIT"] = "2"
    env["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    env.setdefault("HCLI_SWAP_CEILING_GIB", "64")
    env.pop("HCLI_OBSERVED_MODEL_OVERLAP", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            str(SCRIPT),
            "--probe-controller-admit",
            "--out",
            str(out),
            "--repo",
            str(hcli_parent),
        ],
        cwd=str(hcli_parent),
        capture_output=True,
        text=True,
        timeout=120,
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


def _trial_stats(trials: List[Dict[str, Any]], cond: str, key: str = "admitted_n") -> Dict[str, Any]:
    vals = [
        float(t[key])
        for t in trials
        if t.get("condition") == cond and isinstance(t.get(key), (int, float))
    ]
    if not vals:
        return {"n": 0, "values": [], "min": None, "max": None, "median": None, "spread": None}
    return {
        "n": len(vals),
        "values": vals,
        "min": min(vals),
        "max": max(vals),
        "median": sorted(vals)[len(vals) // 2],
        "spread": max(vals) - min(vals),
    }


def run_interleaved_pair(
    scratch: Path,
    *,
    candidate_name: str,
    candidate_text: str,
    baseline_text: str,
    n_trials: int = PROMOTION_TRIALS,
) -> Dict[str, Any]:
    """Paired interleaved trials: C, B, C, B — never a block then a block."""
    controller = scratch / CONTROLLER_REL
    probe_dir = scratch / "perf_trials" / candidate_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    order = (["candidate", "baseline"] * ((n_trials + 1) // 2))[:n_trials]
    trials: List[Dict[str, Any]] = []
    for i, cond in enumerate(order):
        controller.write_text(
            candidate_text if cond == "candidate" else baseline_text,
            encoding="utf-8",
        )
        clear_controller_pyc(scratch)
        out = probe_dir / f"trial_{i}_{cond}.json"
        result = _admit_child(scratch, out)
        if result.get("ok") and result.get("import_root_is_scratch") is False:
            die(
                "admit child imported hcli from "
                f"{result.get('hcli_file')!r} not scratch {scratch}. "
                "Baseline is not a baseline."
            )
        trials.append(
            {
                "i": i,
                "condition": cond,
                "admitted_n": result.get("admitted_n"),
                "requested_n": result.get("requested_n"),
                "overlap_admit_cap": result.get("overlap_admit_cap"),
                "observed_overlap_ctor": result.get("observed_overlap_ctor"),
                "n_runtimes": result.get("n_runtimes"),
                "backend_class": result.get("backend_class"),
                "through_controller": result.get("through_controller"),
                "through_runtime_pool": result.get("through_runtime_pool"),
                "controller_has_h1_wiring": result.get("controller_has_h1_wiring"),
                "controller_has_noop_comment": result.get("controller_has_noop_comment"),
                "controller_has_requested_n_zero": result.get("controller_has_requested_n_zero"),
                "path": result.get("path"),
                "via": result.get("via"),
                "complete_via": result.get("complete_via"),
                "ok": result.get("ok"),
                "child_exit": result.get("child_exit"),
                "error": result.get("error"),
                "stored": result.get("stored"),
                "loaded": result.get("loaded"),
                "loaded_overlap_on_controller_ws": result.get("loaded_overlap_on_controller_ws"),
                "loaded_overlap_on_pool_ws": result.get("loaded_overlap_on_pool_ws"),
                "loaded_overlap_on_isolate": result.get("loaded_overlap_on_isolate"),
                "workspace_pool": result.get("workspace_pool"),
                "workspace_controller": result.get("workspace_controller"),
                "workspace_isolate": result.get("workspace_isolate"),
                "hcli_file": result.get("hcli_file"),
                "controller_file": result.get("controller_file"),
                "runtime_file": result.get("runtime_file"),
                "import_root_is_scratch": result.get("import_root_is_scratch"),
            }
        )
    cand = _trial_stats(trials, "candidate")
    base = _trial_stats(trials, "baseline")
    spread = max(cand.get("spread") or 0, base.get("spread") or 0, 0)
    improved = (
        cand["median"] is not None
        and base["median"] is not None
        and float(cand["median"]) > float(base["median"]) + float(spread)
    )
    cand_admits = [
        int(t["admitted_n"])
        for t in trials
        if t.get("condition") == "candidate" and isinstance(t.get("admitted_n"), int)
    ]
    base_admits = [
        int(t["admitted_n"])
        for t in trials
        if t.get("condition") == "baseline" and isinstance(t.get("admitted_n"), int)
    ]
    admission_differs = bool(cand_admits) and bool(base_admits) and set(cand_admits) != set(base_admits)
    through = all(
        t.get("through_controller")
        and t.get("through_runtime_pool")
        and t.get("via") == "Controller.ensure_runtime_pool"
        for t in trials
        if t.get("ok")
    )
    decided = compute_decision(
        correctness_ok=True,
        throughput_improved=improved,
        mutation_applied=candidate_text != baseline_text,
        validation_ok=True,
        validation_reason=None,
        orig_med=base.get("median"),
        mut_med=cand.get("median"),
        spread=spread,
        admission_differs=admission_differs,
        metric_name="admitted_n",
    )
    return {
        "candidate": candidate_name,
        "alternating": True,
        "order": [t["condition"] for t in trials],
        "block_design": False,
        "trials": trials,
        "candidate_stats": cand,
        "baseline_stats": base,
        "candidate_admitted_n": cand_admits,
        "baseline_admitted_n": base_admits,
        "admission_differs": admission_differs,
        "spread": spread,
        "score": "admitted_n through Controller.ensure_runtime_pool",
        "improved": improved,
        "is_win": bool(improved and decided.get("decision") == "promote"),
        "through_mutated_mechanism": through,
        "decision": decided,
    }


def run_h1_wiring_test(scratch: Path, controller_text: str) -> Dict[str, Any]:
    """Physical pytest against a candidate's controller.py. Evidence, not assertion."""
    tests = scratch / "tools" / "headless"
    tests.mkdir(parents=True, exist_ok=True)
    (scratch / CONTROLLER_REL).write_text(controller_text, encoding="utf-8")
    test_path = tests / "test_h1_wiring_evidence.py"
    test_path.write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_h1_controller_wires_observed_overlap_into_runtimepool():\n"
        "    text = (Path(__file__).resolve().parents[2] / 'hcli' / 'controller.py')"
        ".read_text(encoding='utf-8')\n"
        "    assert 'from .runtime import RuntimePool, load_observed_overlap' in text\n"
        "    assert 'workspace=self.workspace_root' in text\n"
        "    assert 'repo_root=self.workspace_root' in text\n"
        "    assert 'observed_overlap=load_observed_overlap(self.workspace_root)' in text\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=line"],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = proc.returncode == 0
    return {
        "ok": passed,
        "exit_code": proc.returncode,
        "output_tail": ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-600:],
        "test_path": str(test_path),
        "wiring_present": h1_wiring_present(controller_text),
    }


def _trial_import_ok(trials: Optional[List[Dict[str, Any]]]) -> bool:
    rows = trials or []
    if not rows:
        return False
    return all(t.get("import_root_is_scratch") is True for t in rows if t.get("ok"))


def build_root_cause(
    *,
    materialize: Dict[str, Any],
    variants_meta: Dict[str, Any],
    h1: Dict[str, Any],
    noop: Dict[str, Any],
    bad: Dict[str, Any],
) -> Dict[str, Any]:
    """The G021 divergence, measured rather than asserted."""
    h1_trials = h1.get("trials") or []
    baseline = [t for t in h1_trials if t.get("condition") == "baseline"]
    candidate = [t for t in h1_trials if t.get("condition") == "candidate"]
    base_ctor = [t.get("observed_overlap_ctor") for t in baseline]
    cand_ctor = [t.get("observed_overlap_ctor") for t in candidate]
    base_wiring = [t.get("controller_has_h1_wiring") for t in baseline]
    base_imported = [t.get("import_root_is_scratch") for t in baseline]
    base_files = [t.get("controller_file") for t in baseline]
    base_stored = [t.get("stored") for t in baseline]
    base_loaded = [t.get("loaded") for t in baseline]
    base_loaded_pool = [t.get("loaded_overlap_on_pool_ws") for t in baseline]
    return {
        "id": "G021_SCRATCH_IMPORT_SHADOW",
        "unaccounted_overlap_2": (
            "Baseline observed_overlap_ctor=2 with controller_has_h1_wiring="
            "false was not an unaccounted RuntimePool load. "
            "observed_overlap on the pool is set only from the constructor "
            "argument (runtime.py stores None when the arg is None; the "
            "workspace fallback fills overlap_admit_cap, not "
            "self.observed_overlap). ctor=2 therefore means the executed "
            "Controller passed load_observed_overlap(workspace). That is "
            "HEAD's ensure_runtime_pool (controller.py:611). The wiring "
            "marker is read from the scratch file the parent just wrote, "
            "which had H1 stripped. Two different objects: disk variant vs "
            "imported module."
        ),
        "why_the_imported_module_was_HEAD": (
            "run_controller_admit_probe called _install_fake_pool_patches, "
            "which called ensure_hcli_path(), which inserted the invoking "
            "worktree (Path(__file__).parents[2]) at sys.path[0]. "
            "from hcli.runtime import RuntimePool then bound that tree's "
            "package. hcli stayed in sys.modules, so "
            "from hcli.controller import Controller ignored the scratch "
            "copy even though PYTHONPATH and a later sys.path.insert(0, "
            "scratch) named it. The FakeBackend patch therefore attached "
            "to whichever RuntimePool the first import found — the worktree "
            "when hcli/ was on disk, the scratch copy when hcli/ was a "
            "sparse hole."
        ),
        "why_the_lane_promoted": (
            "The lane worktree had hcli/ absent (sparse hole, materialized "
            "only via git archive into scratch). ensure_hcli_path inserted "
            "a directory that did not contain hcli/, so import fell through "
            "to scratch. Variants ran. Synthetic original (no workspace, no "
            "observed_overlap) admitted 1; H1 admitted 2. stored/loaded on "
            "the isolate were null because original does not pass workspace "
            "and HCLI_WORKSPACE is the empty isolate. That 1→2 is a real "
            "delta against a constructor HEAD no longer has."
        ),
        "why_main_checkout_refused": (
            "The main checkout had hcli/ present. Both arms imported HEAD. "
            "HEAD already passes observed_overlap=load_observed_overlap("
            "self.workspace_root) at controller.py:611 and RuntimePool "
            "already falls back to load_observed_overlap(self.workspace) at "
            "runtime.py:637 when the constructor arg is None. Both arms "
            "admitted 2. REFUSED was the correct tree-level verdict, reached "
            "for the wrong reason: the baseline arm was not a baseline."
        ),
        "refuted": {
            "editable_install_meta_path_finder": (
                "Direct test: git archive HEAD hcli to a temp dir, purge "
                "sys.modules of hcli*, sys.path.insert(0, scratch), import "
                "hcli — resolves to the scratch copy. The finder does not "
                "win. The divergence is a sys.path directory that actually "
                "contains hcli/, inserted in front of scratch by "
                "ensure_hcli_path."
            )
        },
        "h1_at_head": bool(variants_meta.get("h1_equals_head")),
        "h1_observed_overlap_arg_redundant_with_runtime_fallback": (
            "Once workspace is the Controller workspace, RuntimePool."
            "_overlap_admit_cap loads the high-water file itself. Passing "
            "observed_overlap=load_observed_overlap(...) changes where the "
            "value is named, not whether it is used. HEAD already passes "
            "workspace=self.workspace_root, so the extra kwarg is redundant "
            "at HEAD."
        ),
        "h1_workspace_arg_is_load_bearing_vs_synthetic_original": (
            "The synthetic original constructor passes neither workspace "
            "nor observed_overlap. resolve_workspace(None) then uses "
            "HCLI_WORKSPACE (the empty isolate). The fallback cannot see "
            "the high-water file stored on the Controller workspace. That "
            "is why stripped original admits 1 after the import pin, and "
            "why the lane's PROMOTE measured a before-state that is not "
            "HEAD."
        ),
        "no_remaining_admitted_n_candidate_at_head": (
            "The only production RuntimePool() site is Controller."
            "ensure_runtime_pool, already H1. requested_n=2 with overlap=2 "
            "already admits 2. H2 (DEFAULT_OVERLAP_ADMIT_CAP=2) remains "
            "rejected as correctness against the unmeasured invariant. "
            "There is no honest admitted_n promotion left at HEAD."
        ),
        "fix": (
            "pin_hcli_import_root(scratch) before importing hcli. Purge "
            "sys.modules, drop every sys.path entry that contains hcli/, "
            "insert scratch at [0], set HCLI_SELFOPT_HCLI_PARENT so later "
            "ensure_hcli_path calls cannot put the worktree back in front. "
            "Die if the imported controller_file is not under scratch."
        ),
        "measured_this_run": {
            "worktree_hcli_present": bool(materialize.get("worktree_hcli_present")),
            "h1_equals_head": bool(variants_meta.get("h1_equals_head")),
            "baseline_observed_overlap_ctor": base_ctor,
            "candidate_observed_overlap_ctor": cand_ctor,
            "baseline_controller_has_h1_wiring": base_wiring,
            "baseline_import_root_is_scratch": base_imported,
            "baseline_controller_file": base_files,
            "baseline_stored": base_stored,
            "baseline_loaded_controller_ws": base_loaded,
            "baseline_loaded_pool_ws": base_loaded_pool,
            "h1_candidate_admitted_n": h1.get("candidate_admitted_n"),
            "h1_baseline_admitted_n": h1.get("baseline_admitted_n"),
            "noop_candidate_admitted_n": noop.get("candidate_admitted_n"),
            "noop_baseline_admitted_n": noop.get("baseline_admitted_n"),
            "bad_candidate_admitted_n": bad.get("candidate_admitted_n"),
            "bad_baseline_admitted_n": bad.get("baseline_admitted_n"),
            "all_trials_imported_scratch": (
                _trial_import_ok(h1_trials)
                and _trial_import_ok(noop.get("trials"))
                and _trial_import_ok(bad.get("trials"))
            ),
        },
        "four_controls_unchanged": {
            "noop_must_not_win": True,
            "bad_must_be_refused": True,
            "paired_interleaved_not_blocks": True,
            "failing_gate_physically_exercised": True,
        },
    }


def assemble_promotion_receipt(
    *,
    repo: Path,
    materialize: Dict[str, Any],
    variants_meta: Dict[str, Any],
    noop: Dict[str, Any],
    bad: Dict[str, Any],
    h1: Dict[str, Any],
    failing: Dict[str, Any],
    wiring: Dict[str, Any],
    llama: Dict[str, Any],
) -> Dict[str, Any]:
    noop_win = bool(noop.get("is_win"))
    bad_verdict = (bad.get("decision") or {}).get("verdict")
    h1_verdict = (h1.get("decision") or {}).get("verdict")
    would_refuse = failing.get("would_refuse_on_failing_gate")
    root_cause = build_root_cause(
        materialize=materialize,
        variants_meta=variants_meta,
        h1=h1,
        noop=noop,
        bad=bad,
    )
    imports_ok = bool(root_cause["measured_this_run"]["all_trials_imported_scratch"])
    harness_ok = (
        not noop_win
        and bad_verdict == "REFUSED"
        and bool(would_refuse) is True
        and bool(h1.get("through_mutated_mechanism"))
        and bool(noop.get("through_mutated_mechanism"))
        and bool(bad.get("through_mutated_mechanism"))
        and bool((h1.get("order") or ["x"])[0] != (h1.get("order") or ["x", "x"])[1])
        and bool(wiring.get("h1", {}).get("ok"))
        and not bool(wiring.get("noop", {}).get("ok"))
        and not bool(wiring.get("bad", {}).get("ok"))
        and not bool(wiring.get("original", {}).get("ok"))
        and imports_ok
    )
    if noop_win:
        decision = "reject"
        verdict = "HARNESS_INVALID"
        reason = (
            "NO-OP candidate scored as a win; the harness is measuring something "
            "other than the mutation."
        )
    elif not imports_ok:
        decision = "reject"
        verdict = "HARNESS_INVALID"
        reason = (
            "A trial imported hcli from somewhere other than the git-archive "
            "scratch copy. Baseline is not a baseline."
        )
    elif not harness_ok:
        decision = "reject"
        verdict = "REFUSED"
        reason = (
            "A required control did not hold: "
            f"noop_win={noop_win} bad_verdict={bad_verdict} "
            f"would_refuse_on_failing_gate={would_refuse} "
            f"h1_through={h1.get('through_mutated_mechanism')}"
        )
    elif not wiring.get("h1", {}).get("ok"):
        decision = "reject"
        verdict = "REFUSED"
        reason = "H1 wiring test did not pass; promotion would be NO_EVIDENCE"
    elif variants_meta.get("h1_equals_head"):
        # The 1→2 delta against synthetic original is real and is recorded
        # on candidate_h1. It is not a tree change: HEAD already is H1.
        decision = "reject"
        verdict = "REFUSED"
        reason = (
            "H1 is already HEAD (h1_equals_head=true). The 1→2 admitted_n "
            "delta is against a synthetic pre-H1 constructor that is not "
            "the tree; applying H1 is a no-op. RuntimePool already loads "
            "the high-water file when workspace is passed "
            "(runtime.py:637), so the extra observed_overlap= constructor "
            "kwarg is redundant at HEAD. No remaining admitted_n candidate "
            "at HEAD (only production RuntimePool() site already has the "
            "wiring; requested_n=2 with overlap=2 already admits 2)."
        )
    else:
        # H1's own compute_decision (admission 1→2 through the mechanism).
        decided = h1.get("decision") or {}
        decision = decided.get("decision") or "reject"
        verdict = decided.get("verdict") or "REFUSED"
        reason = decided.get("reason") or "no decision"
        if decision == "promote" and not wiring.get("h1", {}).get("ok"):
            decision = "reject"
            verdict = "REFUSED"
            reason = "attempted promotion on a mutation receipt that is not validation.ok"
    return {
        "schema": "hawking.headless.hcli_self_opt.candidate_promoted.v1",
        "goal": (
            "Prove the mutation caused the win: baseline and candidate both "
            "execute through Controller.ensure_runtime_pool, a NO-OP must not "
            "win, a BAD candidate must be refused, trials are interleaved, and "
            "a failing gate is physically exercised."
        ),
        "root_cause": root_cause,
        "sparse_checkout": {
            "worktree_hcli_present": bool(materialize.get("worktree_hcli_present")),
            "hcli_source": materialize.get("source"),
            "verified_against": "HEAD",
            "controller_verified_via": "git archive HEAD hcli + git show HEAD:hcli/controller.py",
            "repo_hcli_written": False,
        },
        "already_present_before_this_change": {
            "interleaved_gate_perf": (
                "stage_gate_perf already alternates mutated/original/mutated/original "
                "through Controller.ensure_runtime_pool"
            ),
            "failing_gate_computed": (
                "run_failing_gate_trial already feeds a real pytest assert False "
                "into compute_decision; would_refuse_on_failing_gate is "
                "failing.get('verdict') == 'REFUSED', never hardcoded True"
            ),
            "no_evidence_refused": (
                "compute_decision refuses validation_ok=False reason=NO_EVIDENCE"
            ),
            "missing_before_this_change": [
                "NO-OP candidate physically scored through the same mechanism",
                "BAD candidate physically scored through the same mechanism",
            ],
        },
        "mechanism": {
            "entry": "Controller.ensure_runtime_pool",
            "pool": "RuntimePool._admit / RuntimePool.complete",
            "backend": "FakeBackend (no live llama-server, no second 27B)",
            "score": "admitted_n with high-water overlap=2 stored on the Controller workspace",
            "why_not_tok_s": (
                "llama-server on :52484 is not required for this proof. Prior "
                "iteration-2 remasurement already refused a tok/s win "
                "(24.063 vs 24.010, spread 0.052). Scoring tok/s of FakeBackend "
                "would let a NO-OP win on noise — the original defect."
            ),
            "does_not_call": [
                "fan_completions",
                "llama_completion as the gated measurement",
                "urllib /completion bypassing RuntimePool",
            ],
        },
        "materialize": materialize,
        "variants": variants_meta,
        "controls": {
            "noop": {
                "id": "NO-OP",
                "ran": True,
                "mutation": (
                    "comment-only insert above the original RuntimePool constructor; "
                    "requested_n=self.runtime_count unchanged"
                ),
                "must_not_win": True,
                "is_win": noop_win,
                "admission_differs": noop.get("admission_differs"),
                "candidate_admitted_n": noop.get("candidate_admitted_n"),
                "baseline_admitted_n": noop.get("baseline_admitted_n"),
                "spread": noop.get("spread"),
                "order": noop.get("order"),
                "through_mutated_mechanism": noop.get("through_mutated_mechanism"),
                "decision": noop.get("decision"),
                "wiring_test": wiring.get("noop"),
            },
            "bad": {
                "id": "BAD",
                "ran": True,
                "mutation": "requested_n=0 in Controller.ensure_runtime_pool",
                "must_be_refused": True,
                "is_win": bool(bad.get("is_win")),
                "admission_differs": bad.get("admission_differs"),
                "candidate_admitted_n": bad.get("candidate_admitted_n"),
                "baseline_admitted_n": bad.get("baseline_admitted_n"),
                "spread": bad.get("spread"),
                "order": bad.get("order"),
                "through_mutated_mechanism": bad.get("through_mutated_mechanism"),
                "decision": bad.get("decision"),
                "wiring_test": wiring.get("bad"),
            },
            "paired_interleaved": {
                "id": "PAIRED_INTERLEAVED",
                "ran": True,
                "block_design": False,
                "h1_order": h1.get("order"),
                "noop_order": noop.get("order"),
                "bad_order": bad.get("order"),
                "h1_spread": h1.get("spread"),
                "noop_spread": noop.get("spread"),
                "bad_spread": bad.get("spread"),
                "h1": {
                    "candidate_admitted_n": h1.get("candidate_admitted_n"),
                    "baseline_admitted_n": h1.get("baseline_admitted_n"),
                    "admission_differs": h1.get("admission_differs"),
                    "improved": h1.get("improved"),
                    "through_mutated_mechanism": h1.get("through_mutated_mechanism"),
                },
            },
            "failing_gate": {
                "id": "FAILING_GATE",
                "ran": True,
                "hardcoded": False,
                "would_refuse_on_failing_gate": failing.get("would_refuse_on_failing_gate"),
                "would_refuse_on_no_evidence": failing.get("would_refuse_on_no_evidence"),
                "evidenced": failing.get("evidenced"),
                "pytest_exit_code": failing.get("pytest_exit_code"),
                "method": failing.get("method"),
                "decision_on_failing_correctness": failing.get("decision_on_failing_correctness"),
                "decision_on_no_evidence": failing.get("decision_on_no_evidence"),
            },
        },
        "candidate_h1": {
            "ran": True,
            "is_win": bool(h1.get("is_win")),
            "admission_differs": h1.get("admission_differs"),
            "candidate_admitted_n": h1.get("candidate_admitted_n"),
            "baseline_admitted_n": h1.get("baseline_admitted_n"),
            "spread": h1.get("spread"),
            "order": h1.get("order"),
            "through_mutated_mechanism": h1.get("through_mutated_mechanism"),
            "decision": h1.get("decision"),
            "wiring_test": wiring.get("h1"),
            "trials": h1.get("trials"),
        },
        "mutation_receipt": {
            "kind": "controller_variant_from_HEAD",
            "path": str(CONTROLLER_REL),
            "validation": {
                "ok": bool(wiring.get("h1", {}).get("ok")),
                "reason": None if wiring.get("h1", {}).get("ok") else "NO_EVIDENCE",
                "h1_wiring_pytest_exit": wiring.get("h1", {}).get("exit_code"),
                "h1_wiring_present": wiring.get("h1", {}).get("wiring_present"),
                "noop_wiring_pytest_exit": wiring.get("noop", {}).get("exit_code"),
                "bad_wiring_pytest_exit": wiring.get("bad", {}).get("exit_code"),
                "note": (
                    "Evidence is a pytest file physically run against each "
                    "candidate's controller.py. tests=[] would be NO_EVIDENCE "
                    "and is refused. Repo hcli/ was not written."
                ),
            },
            "files": {
                "hcli/controller.py": {
                    "original_sha256": variants_meta.get("original_sha256"),
                    "h1_sha256": variants_meta.get("h1_sha256"),
                    "noop_sha256": variants_meta.get("noop_sha256"),
                    "bad_sha256": variants_meta.get("bad_sha256"),
                }
            },
        },
        "llama_server": llama,
        "decision": decision,
        "decision_verdict": verdict,
        "decision_reason": reason,
        "harness_ok": harness_ok,
        "would_refuse_on_failing_gate": would_refuse,
        "failing_gate_trial": failing,
        "trials": {
            "noop": noop.get("trials"),
            "bad": bad.get("trials"),
            "h1": h1.get("trials"),
        },
    }


def print_four_controls(receipt: Dict[str, Any]) -> None:
    controls = receipt.get("controls") or {}
    noop = controls.get("noop") or {}
    bad = controls.get("bad") or {}
    paired = controls.get("paired_interleaved") or {}
    fail = controls.get("failing_gate") or {}
    print("\n## FOUR CONTROLS", flush=True)
    print(
        f"1. NO-OP candidate: ran={noop.get('ran')} is_win={noop.get('is_win')} "
        f"admitted_n candidate={noop.get('candidate_admitted_n')} "
        f"baseline={noop.get('baseline_admitted_n')} "
        f"admission_differs={noop.get('admission_differs')} "
        f"through_mechanism={noop.get('through_mutated_mechanism')} "
        f"verdict={(noop.get('decision') or {}).get('verdict')}",
        flush=True,
    )
    print(
        f"2. BAD candidate: ran={bad.get('ran')} is_win={bad.get('is_win')} "
        f"admitted_n candidate={bad.get('candidate_admitted_n')} "
        f"baseline={bad.get('baseline_admitted_n')} "
        f"through_mechanism={bad.get('through_mutated_mechanism')} "
        f"verdict={(bad.get('decision') or {}).get('verdict')}",
        flush=True,
    )
    print(
        f"3. Paired/interleaved: ran={paired.get('ran')} "
        f"block_design={paired.get('block_design')} "
        f"h1_order={paired.get('h1_order')} "
        f"h1_spread={paired.get('h1_spread')} "
        f"h1_admitted candidate={(paired.get('h1') or {}).get('candidate_admitted_n')} "
        f"baseline={(paired.get('h1') or {}).get('baseline_admitted_n')}",
        flush=True,
    )
    print(
        f"4. Failing-gate: ran={fail.get('ran')} hardcoded={fail.get('hardcoded')} "
        f"pytest_exit={fail.get('pytest_exit_code')} "
        f"would_refuse_on_failing_gate={fail.get('would_refuse_on_failing_gate')} "
        f"would_refuse_on_no_evidence={fail.get('would_refuse_on_no_evidence')} "
        f"evidenced={fail.get('evidenced')}",
        flush=True,
    )
    print(
        f"\n decision: {receipt.get('decision')} ({receipt.get('decision_verdict')}) "
        f"— {receipt.get('decision_reason')}",
        flush=True,
    )
    if noop.get("is_win"):
        print(
            "FINDING: NO-OP scored as a win; the harness is not measuring the mutation.",
            flush=True,
        )
    cause = receipt.get("root_cause") or {}
    measured = cause.get("measured_this_run") or {}
    print("\n## ROOT CAUSE", flush=True)
    print(f"id: {cause.get('id')}", flush=True)
    print(f"h1_equals_head: {cause.get('h1_at_head')}", flush=True)
    print(
        f"import_root_is_scratch (all trials): {measured.get('all_trials_imported_scratch')}",
        flush=True,
    )
    print(
        f"H1 admitted candidate={measured.get('h1_candidate_admitted_n')} "
        f"baseline={measured.get('h1_baseline_admitted_n')} "
        f"baseline_ctor={measured.get('baseline_observed_overlap_ctor')} "
        f"baseline_wiring={measured.get('baseline_controller_has_h1_wiring')}",
        flush=True,
    )
    print(
        "four controls unchanged: "
        f"{cause.get('four_controls_unchanged')}",
        flush=True,
    )


def run_promotion_controls(repo: Path) -> Dict[str, Any]:
    """Run the four G021 controls. Does not write repo hcli/."""
    os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("ACTIVE_DECODE_LIMIT", "2")

    scratch = Path(tempfile.mkdtemp(prefix="hcli-selfopt2-promo-"))
    materialize = materialize_hcli_from_head(repo, scratch)
    head_text = (scratch / CONTROLLER_REL).read_text(encoding="utf-8")
    head_via_show = _git_show(repo, str(CONTROLLER_REL))
    if head_text != head_via_show:
        die("git archive controller.py != git show HEAD:hcli/controller.py")
    variants = controller_variants(head_text)
    variants_meta = {
        "original_sha256": hashlib.sha256(variants["original"].encode("utf-8")).hexdigest(),
        "h1_sha256": hashlib.sha256(variants["h1"].encode("utf-8")).hexdigest(),
        "noop_sha256": hashlib.sha256(variants["noop"].encode("utf-8")).hexdigest(),
        "bad_sha256": hashlib.sha256(variants["bad"].encode("utf-8")).hexdigest(),
        "h1_equals_head": variants["h1"] == head_text,
        "noop_equals_original": variants["noop"] == variants["original"],
        "bad_equals_original": variants["bad"] == variants["original"],
    }

    print(f"promotion scratch {scratch} source={materialize['source']}", flush=True)
    print("running NO-OP interleaved trials", flush=True)
    noop = run_interleaved_pair(
        scratch,
        candidate_name="noop",
        candidate_text=variants["noop"],
        baseline_text=variants["original"],
    )
    print(
        f"ok   NO-OP is_win={noop['is_win']} admitted={noop['candidate_admitted_n']} "
        f"vs {noop['baseline_admitted_n']} verdict={noop['decision']['verdict']}",
        flush=True,
    )
    print("running BAD interleaved trials", flush=True)
    bad = run_interleaved_pair(
        scratch,
        candidate_name="bad",
        candidate_text=variants["bad"],
        baseline_text=variants["original"],
    )
    print(
        f"ok   BAD is_win={bad['is_win']} admitted={bad['candidate_admitted_n']} "
        f"vs {bad['baseline_admitted_n']} verdict={bad['decision']['verdict']}",
        flush=True,
    )
    print("running H1 interleaved trials", flush=True)
    h1 = run_interleaved_pair(
        scratch,
        candidate_name="h1",
        candidate_text=variants["h1"],
        baseline_text=variants["original"],
    )
    print(
        f"ok   H1 is_win={h1['is_win']} admitted={h1['candidate_admitted_n']} "
        f"vs {h1['baseline_admitted_n']} verdict={h1['decision']['verdict']}",
        flush=True,
    )

    print("running H1/NO-OP/BAD wiring pytest", flush=True)
    wiring = {
        "h1": run_h1_wiring_test(scratch, variants["h1"]),
        "noop": run_h1_wiring_test(scratch, variants["noop"]),
        "bad": run_h1_wiring_test(scratch, variants["bad"]),
        "original": run_h1_wiring_test(scratch, variants["original"]),
    }
    print(
        f"ok   wiring pytest h1={wiring['h1']['exit_code']} "
        f"noop={wiring['noop']['exit_code']} bad={wiring['bad']['exit_code']} "
        f"original={wiring['original']['exit_code']}",
        flush=True,
    )

    fail_ws = scratch / "failing_gate"
    fail_ws.mkdir(parents=True, exist_ok=True)
    print("running failing-gate trial", flush=True)
    failing = run_failing_gate_trial(fail_ws)
    print(
        f"ok   failing-gate pytest_exit={failing.get('pytest_exit_code')} "
        f"would_refuse_on_failing_gate={failing.get('would_refuse_on_failing_gate')} "
        f"hardcoded={failing.get('hardcoded')}",
        flush=True,
    )

    llama = llama_snapshot()
    receipt = assemble_promotion_receipt(
        repo=repo,
        materialize=materialize,
        variants_meta=variants_meta,
        noop=noop,
        bad=bad,
        h1=h1,
        failing=failing,
        wiring=wiring,
        llama=llama,
    )
    receipt["workspace"] = str(scratch)
    dest = repo / PROMOTION_RECEIPT_REL
    _atomic_write(dest, receipt)
    receipt["receipt_path"] = str(dest)
    print(f"receipt {dest}", flush=True)
    print_four_controls(receipt)
    return receipt


def run_promotion_controls_main(repo: Path) -> int:
    receipt = run_promotion_controls(repo)
    controls = receipt.get("controls") or {}
    shown = all(
        (controls.get(name) or {}).get("ran") is True
        for name in ("noop", "bad", "paired_interleaved", "failing_gate")
    )
    if not shown:
        print("FAIL four controls were not all recorded as ran", file=sys.stderr, flush=True)
        return 1
    if receipt.get("would_refuse_on_failing_gate") is not True:
        print("FAIL would_refuse_on_failing_gate was not computed True", file=sys.stderr, flush=True)
        return 1
    if (controls.get("noop") or {}).get("is_win"):
        # Report the finding; do not disguise it. Non-zero so a green run
        # cannot mean "the no-op won".
        print("FAIL NO-OP scored as a win", file=sys.stderr, flush=True)
        return 1
    if ((controls.get("bad") or {}).get("decision") or {}).get("verdict") != "REFUSED":
        print("FAIL BAD candidate was not REFUSED", file=sys.stderr, flush=True)
        return 1
    return 0


def stage_decide(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    correctness = state.get("gate.correctness") or {}
    perf = state.get("gate.perf") or {}
    mutate = state.get("mutate") or {}
    correctness_ok = bool(correctness.get("passed_gate"))
    throughput_improved = bool(perf.get("throughput_improved"))
    mutation_applied = bool(mutate.get("applied"))
    validation_ok, validation_reason = mutation_validation_ok(mutate)
    orig_med = (perf.get("original") or {}).get("median")
    mut_med = (perf.get("mutated") or {}).get("median")
    spread = perf.get("spread")
    admission_differs = perf.get("admission_differs")

    decided = compute_decision(
        correctness_ok=correctness_ok,
        throughput_improved=throughput_improved,
        mutation_applied=mutation_applied,
        validation_ok=validation_ok,
        validation_reason=validation_reason,
        orig_med=orig_med,
        mut_med=mut_med,
        spread=spread,
        mutation_blocked=mutate.get("blocked"),
        correctness_exit=correctness.get("exit_code"),
        admission_differs=admission_differs if isinstance(admission_differs, bool) else None,
    )
    trial = run_failing_gate_trial(ws)
    if not trial.get("evidenced"):
        watch(
            state,
            "failing-gate trial did not evidence REFUSED",
            json.dumps(trial, default=str)[:2000],
        )
        save_state(Path(state["_path"]), state)
        die("decide: failing-gate trial did not come back REFUSED")
    if decided["decision"] == "promote" and decided["would_refuse"]:
        die("decide: attempted promotion with a failing gate; verifier refuses")
    if decided["decision"] == "promote" and not correctness_ok:
        die("decide: promotion with failing correctness gate is refused")
    if decided["decision"] == "promote" and not validation_ok:
        die(
            "decide: attempted promotion on a mutation receipt that is not "
            "validation.ok; verifier refuses"
        )

    decision = decided["decision"]
    reason = decided["reason"]
    if decision == "reject":
        orig = mutate.get("snap_original")
        if orig and Path(orig).is_dir():
            restore_files(repo, Path(orig))
            clear_controller_pyc(repo)
            restored = True
        else:
            restored = False
        watch(state, "decide rejected the change", reason)
    else:
        restored = False

    state["decide"] = {
        "decision": decision,
        "verdict": decided["verdict"],
        "reason": reason,
        "correctness_ok": correctness_ok,
        "throughput_improved": throughput_improved,
        "mutation_applied": mutation_applied,
        "validation_ok": validation_ok,
        "validation_reason": validation_reason,
        "admission_differs": admission_differs,
        "failing_gate_trial": trial,
        "counterfactual_refuse_on_failing_gate": {
            "if_correctness_failed": "REFUSED",
            "if_perf_throughput_unimproved": "REFUSED",
            "if_mutation_receipt_no_evidence": "REFUSED",
            "predicate": (
                "promote IFF mutation.applied AND mutation_receipt.validation.ok "
                "AND gate.correctness.passed_gate AND gate.perf.throughput_improved "
                "(mutated.median > original.median + spread, measured through "
                "RuntimePool.complete); else reject/REFUSED"
            ),
            "would_refuse_on_failing_gate": trial.get("would_refuse_on_failing_gate"),
            "would_refuse_on_no_evidence": trial.get("would_refuse_on_no_evidence"),
            "hardcoded": False,
            "evidenced_by": "failing_gate_trial",
        },
        "restored_original": restored,
        "refuse_if": decided["refuse_if"],
    }
    ok(
        f"decide: {decision} ({decided['verdict']}) — {reason} "
        f"[failing-gate trial REFUSED={trial.get('would_refuse_on_failing_gate')} "
        f"NO_EVIDENCE REFUSED={trial.get('would_refuse_on_no_evidence')}]"
    )


def stage_priors(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    sense = state.get("sense") or {}
    perf = state.get("gate.perf") or {}
    decide = state.get("decide") or {}
    mutated_median = (perf.get("mutated") or {}).get("median")
    original_median = (perf.get("original") or {}).get("median")
    measurement = {
        "sense_max_concurrent_model_calls": sense.get("max_concurrent_model_calls"),
        "sense_unmeasured_admitted_n": sense.get("unmeasured_admitted_n"),
        "sense_stored_admitted_n": sense.get("stored_admitted_n"),
        "gate_perf_mutated_median_tps": mutated_median,
        "gate_perf_original_median_tps": original_median,
        "gate_perf_spread": perf.get("spread"),
        "decision": decide.get("decision"),
    }
    prior = {
        "unchanged_hardware_envelope": {
            "active_decode_limit": PRIOR_ACTIVE_DECODE_LIMIT,
            "source": "receipts/headless/MACHINE_GENOME.json ACTIVE_DECODE_LIMIT",
            "aggregate_at_four_slot_decoders": PRIOR_AGGREGATE_AT_FOUR,
            "source_four": (
                "receipts/headless/DECODE_TOPOLOGY.json summary.slot.4.scaling_vs_1"
            ),
            "contract_envelope": PRIOR_CONTRACT_ENVELOPE,
            "genome_aggregate_scaling_vs_1": PRIOR_GENOME_AGGREGATE,
            "recommendedMaxWorkingSetSize_gib": PRIOR_RECOMMENDED_WS_GIB,
            "per_runtime_gib": PRIOR_PER_RUNTIME_GIB,
            "two_server_tps": PRIOR_TWO_SERVER_TPS,
            "one_server_tps": PRIOR_ONE_SERVER_TPS,
            "note": (
                "Decode concurrency tops out near 1.2161x aggregate at four "
                "decoders. A second runtime costs ~19.79 GiB of Metal working "
                "set. Two resident 27B servers collapsed tok/s from 33.47 to "
                "3.986. These numbers did not change this run."
            ),
        },
        "updated": {
            "unmeasured_runtimepool_admit_cap_is_1_while_call_model_overlaps": {
                "before": True,
                "after": decide.get("decision") == "promote",
                "citing": measurement,
            }
        },
        "citing": measurement,
    }
    state["priors"] = prior
    ok(
        f"priors: hardware envelope unchanged (limit={PRIOR_ACTIVE_DECODE_LIMIT}, "
        f"four-decoder aggregate={PRIOR_AGGREGATE_AT_FOUR}, "
        f"recommendedMaxWorkingSetSize={PRIOR_RECOMMENDED_WS_GIB}GiB); "
        f"citing sense peak={measurement['sense_max_concurrent_model_calls']} "
        f"unmeasured_admitted_n={measurement['sense_unmeasured_admitted_n']} "
        f"perf mutated median tps={mutated_median}"
    )


def stage_next(state: Dict[str, Any], repo: Path, ws: Path) -> None:
    perf = state.get("gate.perf") or {}
    sense = state.get("sense") or {}
    decide = state.get("decide") or {}
    llama = sense.get("llama_server") or {}
    mutated_median = (perf.get("mutated") or {}).get("median")
    original_median = (perf.get("original") or {}).get("median")
    spread = perf.get("spread")
    sense_peak = sense.get("max_concurrent_model_calls")
    slots = llama.get("total_slots")

    if isinstance(mutated_median, (int, float)):
        field = "gate.perf.mutated.median"
        value = mutated_median
        if decide.get("decision") == "promote":
            target = (
                "Iteration 3: admission-2 RuntimePool.complete delivered median "
                f"{mutated_median} tok/s against admit-1 {original_median} "
                f"(spread {spread}), same offered load of two concurrent "
                "pool.complete calls, past the measured spread. Next: spend "
                "that width on prefix-cache locality / KV split cost rather "
                "than a third runtime — the four-decoder aggregate ceiling "
                f"is still {PRIOR_AGGREGATE_AT_FOUR}x."
            )
        else:
            target = (
                "Iteration 3: admit-2 RuntimePool.complete median tps="
                f"{mutated_median} did not beat admit-1 median tps="
                f"{original_median} beyond spread={spread}. Do not chase a "
                "second runtime (19.79 GiB) or a second 27B server "
                f"({PRIOR_TWO_SERVER_TPS} vs {PRIOR_ONE_SERVER_TPS} tok/s). "
                "Next: the remaining prize is per-stream prompt-eval / "
                "prefix-cache, not decode width — citing this run's mutated "
                "median tps."
            )
    else:
        field = "sense.max_concurrent_model_calls"
        value = sense_peak
        target = (
            "Iteration 3: throughput probe did not land a median tps; "
            f"this run sensed peak={sense_peak}. Stay on real completions "
            "against the live --parallel 2 server, still without spawning."
        )

    cited_state = dotted(state, field)
    if cited_state is None:
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
        "original_median_tps": original_median,
        "spread": spread,
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
    (
        "sense",
        "Measure current-tree _call_model overlap AND unmeasured RuntimePool admission",
        [],
        "CPU_HEAVY",
    ),
    (
        "bottleneck",
        "Name the admit-cap bottleneck; verifier requires agreement with the sensed numbers",
        ["sense"],
        "LIGHT_CONTROL",
    ),
    (
        "hypotheses",
        "Enumerate high-water admission raise plus alternatives, each naming file:line",
        ["bottleneck"],
        "LIGHT_CONTROL",
    ),
    (
        "screen",
        "Cheap disproof first; reject DEFAULT=2 and a second 27B server",
        ["hypotheses"],
        "LIGHT_CONTROL",
    ),
    (
        "mutate",
        "Apply the surviving high-water wiring through Engine.execute",
        ["screen"],
        "MUTATION",
    ),
    (
        "gate.correctness",
        "Run python3 -m pytest tools/haider/hcli/tests -q and record the exit",
        ["mutate"],
        "TEST",
    ),
    (
        "gate.perf",
        "Paired alternating RuntimePool.complete tok/s through Controller.ensure_runtime_pool; same offered load; no second server",
        ["gate.correctness"],
        "CPU_HEAVY",
    ),
    (
        "decide",
        "Promote only if validation.ok AND both gates pass AND throughput through the pool improved beyond spread",
        ["gate.perf"],
        "LIGHT_CONTROL",
    ),
    (
        "priors",
        "Write the updated prior, citing the measurement that changed it",
        ["decide"],
        "LIGHT_CONTROL",
    ),
    (
        "next",
        "Choose iteration 3's target citing a specific iteration-2 measurement",
        ["priors"],
        "LIGHT_CONTROL",
    ),
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
        "iteration": 2,
        "remeasurement": {
            "of": "receipts/headless/HCLI_SELF_OPT_ITERATION_2.json",
            "defect": (
                "gate.perf fanned HTTP at llama-server; completions never entered "
                "the mutated RuntimePool. would_refuse_on_failing_gate was hardcoded. "
                "Engine receipts with validation.ok=false reason=NO_EVIDENCE were promoted."
            ),
            "code_path": (
                "Controller.ensure_runtime_pool -> RuntimePool.complete -> "
                "AttachLiveBackend.complete -> http://127.0.0.1:52484/completion"
            ),
        },
        "goal": (
            "Raise RuntimePool admission via the measured high-water path "
            "so llama-server's 2 slots / ACTIVE_DECODE_LIMIT=2 serve real "
            "completions, without spawning a second 27B process."
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
            "recommendedMaxWorkingSetSize_gib": PRIOR_RECOMMENDED_WS_GIB,
            "per_runtime_gib": PRIOR_PER_RUNTIME_GIB,
            "two_server_tps": PRIOR_TWO_SERVER_TPS,
            "one_server_tps": PRIOR_ONE_SERVER_TPS,
            "note": (
                "The honest ceiling is well under 2x. A measured no-improvement "
                "is the expected outcome as often as not; rejecting is a success."
            ),
        },
        "mission": {
            "id": getattr(mission, "id", None),
            "phase": getattr(mission, "phase", None),
            "accepted_count": getattr(mission, "accepted_count", None),
        }
        if mission is not None
        else None,
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
        "mutation_validation_ok": mutate.get("validation_ok"),
        "decision": decide.get("decision"),
        "decision_verdict": decide.get("verdict"),
        "decision_reason": decide.get("reason"),
        "failing_gate_trial": decide.get("failing_gate_trial"),
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
    os.environ["HCLI_DISABLE_SIGNAL_HOOKS"] = "1"

    grok = shutil.which("grok-run")
    baseline = {
        "expected": "464 passed, 1 skipped",
        "observed": None,
        "deviation": None,
        "suite_green": True,
        "note": (
            "Contract baseline is 464 passed, 1 skipped with "
            "HCLI_SWAP_CEILING_GIB=64. gate.correctness re-measures."
        ),
    }

    ws = Path(tempfile.mkdtemp(prefix="hcli-selfopt2-mission-"))
    state_path = ws / "loop_state.json"
    pre = {
        "watched_fail": [
            {
                "title": "Prior iteration 2 promoted a bypassing benchmark",
                "detail": (
                    "HCLI_SELF_OPT_ITERATION_2.json reported 25.313 vs 22.575 tok/s "
                    "from fan_completions(width) hitting llama-server directly. "
                    "Completions never entered RuntimePool. Mutation receipts were "
                    "status=unverified validation.ok=false reason=NO_EVIDENCE. "
                    "would_refuse_on_failing_gate was hardcoded True."
                ),
            },
            {
                "title": "grok-run is not required and may be absent",
                "detail": (
                    f"which(grok-run)={grok!r}. Every loop stage is cpu-backed "
                    "so the DAG does not depend on grok-run."
                ),
            },
            {
                "title": "Default HCLI_CPU_TIMEOUT=120 is below the suite wall",
                "detail": (
                    "gate.correctness runs python3 -m pytest tools/haider/hcli/tests "
                    "which took ~142s on this box. The loop sets HCLI_CPU_TIMEOUT=600 "
                    "so the WorkUnit verifier is not killed mid-suite."
                ),
            },
            {
                "title": "A second 27B llama-server is forbidden",
                "detail": (
                    f"Native run: {PRIOR_TWO_SERVER_TPS} tok/s with two model "
                    f"servers resident vs {PRIOR_ONE_SERVER_TPS} with one. "
                    "Completions attach to the live server on :52484."
                ),
            },
            {
                "title": "Host swap (~16 GiB) trips the default 2 GiB MemGate ceiling",
                "detail": (
                    "The live 27B server leaves ~16 GiB of swap in use. A first "
                    "admit probe that used the default swap ceiling reported "
                    "admitted_n=0 even though overlap_admit_cap was 1/2. The "
                    "probe now injects a 64 GiB swap ceiling so it isolates "
                    "_overlap_admit_cap rather than the host swap gate."
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
            "HCLI self-optimize iteration 2: raise RuntimePool admission via "
            "measured overlap so llama-server's 2 slots serve real completions, "
            "without spawning a second 27B process."
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
    parser.add_argument("--probe-throughput", action="store_true")
    parser.add_argument(
        "--probe-controller-admit",
        action="store_true",
        help="child: admission through Controller.ensure_runtime_pool + FakeBackend",
    )
    parser.add_argument(
        "--promotion-controls",
        action="store_true",
        help="run the four G021 promotion controls (default with no other action)",
    )
    parser.add_argument(
        "--mission-loop",
        action="store_true",
        help="run the iteration-2 ten-WorkUnit Mission (needs live llama-server)",
    )
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=N_PREDICT)
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

    if args.probe_throughput:
        result = run_throughput_probe(
            width=int(args.width),
            n_predict=int(args.n_predict),
            repo=repo,
        )
        if args.out:
            _atomic_write(args.out, result)
        else:
            print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    if args.probe_controller_admit:
        result = run_controller_admit_probe(repo=repo)
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

    if args.mission_loop:
        return main_loop(repo)

    return run_promotion_controls_main(repo)


if __name__ == "__main__":
    raise SystemExit(main())
