from __future__ import annotations

import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .backends import (
    CompletionResult,
    allocate_port,
    is_mlx_model_dir,
    is_remote_endpoint,
    OpenAICompatibleBackend,
    terminate_pid,
)
from .hawking_native import config_for_model_path, is_hawking_native_path
from .context_budget import (
    ContextBudget,
    apply_observed_slot,
    probe_server_context,
    resolve as resolve_context_budget,
)
from .machine import (
    AdmissionDecision,
    MemGate,
    default_repo_root,
    host_snapshot,
    resolve_decode_topology,
    resolve_runtime_limits,
    slot_allocation_decision,
)
from .persist import atomic_write_text as _atomic_write
from .resources import pid_is_alive, process_start_token

OWNERSHIP_SCHEMA = "hcli.runtime_pool.v1"
OWNERSHIP_NAME = "runtime_pool.json"
OVERLAP_SCHEMA = "hcli.model_overlap.v1"
OVERLAP_NAME = "model_overlap.json"
# Measured: a real Mission with two independent GPU_DECODE units dispatched
# concurrently (observed_max_gpu_decode=2) still issued model calls one at a
# time (max in-flight = 1). Extra resident processes on this box cost ~19.79 GiB
# each and receive no requests. Admit that many until a later run stores a
# higher observed max.
DEFAULT_OVERLAP_ADMIT_CAP = 1
_OVERLAP_LOCK = threading.Lock()

TOPOLOGY_KEYS = (
    "model_path",
    "artifact_identity",
    "pid",
    "port",
    "ctx_size",
    "parallel",
    "per_slot_context",
    "active_sequences",
    "kv_configuration",
)



_LIVE_POOLS: List[weakref.ReferenceType] = []
_HOOKS_INSTALLED = False
_HOOKS_LOCK = threading.Lock()


def _atexit_stop() -> None:
    for ref in list(_LIVE_POOLS):
        pool = ref()
        if pool is None:
            continue
        try:
            pool.stop()
        except Exception:
            pass


def _install_hooks() -> None:
    global _HOOKS_INSTALLED
    with _HOOKS_LOCK:
        if _HOOKS_INSTALLED:
            return
        atexit.register(_atexit_stop)
        _HOOKS_INSTALLED = True
        if os.environ.get("HCLI_DISABLE_SIGNAL_HOOKS") == "1":
            return
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)
            except Exception:
                prev = signal.SIG_DFL

            def handler(signum, frame, _prev=prev):
                _atexit_stop()
                if callable(_prev):
                    _prev(signum, frame)
                elif _prev == signal.SIG_DFL:
                    try:
                        signal.signal(signum, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)
                    except OSError:
                        pass

            try:
                signal.signal(sig, handler)
            except Exception:
                pass


def _reap_orphans_once() -> None:
    """Reclaim resident bodies no live process owns, before starting another.

    atexit handles a clean exit and demonstrably works. It CANNOT run on
    SIGKILL, and a daemon meets SIGKILL routinely - OOM kill, crash, kill -9,
    power loss. Measured: SIGKILLing the CLI mid-run left TWO orphaned bodies
    holding 22.22 GB, still present 75 seconds later and not self-exiting.
    Over an unattended overnight run that is fatal on its own.

    So self-heal on startup rather than trusting the exit path - the same shape
    the ModelLake reconciliation pass uses, and for the same reason: the
    cleanup fails precisely when the process that should have done it is gone.
    Never touches a body a live resident state file claims.
    """
    try:
        from .processes import reap_orphaned_bodies

        result = reap_orphaned_bodies()
        if result.get("reaped"):
            gib = result.get("bytes_held", 0) / 1024 ** 3
            print(
                f"[hcli] reclaimed {len(result['reaped'])} orphaned resident "
                f"body/bodies holding {gib:.2f}G",
                file=sys.stderr,
            )
    except Exception:
        # Best-effort by design: reclaiming memory must never be the reason a
        # runtime fails to start. The reporting line is INSIDE the guard because
        # a NameError in it escaped and killed the caller during testing.
        return


def _register_live(pool: "RuntimePool") -> None:
    _reap_orphans_once()
    _LIVE_POOLS.append(weakref.ref(pool))
    _install_hooks()


def topology_field(value: Any, reason: Optional[str] = None) -> Dict[str, Any]:
    """One topology receipt field. Unobservable values are null with a reason."""
    if value is None or value == "":
        return {
            "value": None,
            "reason": reason or "unobserved",
        }
    return {"value": value, "reason": None}


def _flag_value(argv: Optional[List[str]], *names: str) -> Optional[str]:
    if not argv:
        return None
    name_set = set(names)
    for i, tok in enumerate(argv):
        if tok in name_set:
            if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                return str(argv[i + 1])
            return ""
        for name in names:
            prefix = name + "="
            if tok.startswith(prefix):
                return tok[len(prefix) :]
    return None


def _flag_present(argv: Optional[List[str]], *names: str) -> bool:
    if not argv:
        return False
    for tok in argv:
        if tok in names:
            return True
        for name in names:
            if tok.startswith(name + "="):
                return True
    return False


def _ps_argv(pid: int) -> Optional[List[str]]:
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-www", "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        text = (out.stdout or "").strip()
        if not text:
            return None
        return shlex.split(text)
    except Exception:
        return None


def _http_json(url: str, timeout: float = 0.25) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", "replace")), None
    except Exception as exc:
        return None, f"{type(exc).__name__}"


def _kv_from_argv(argv: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not argv:
        return None
    kv: Dict[str, Any] = {}
    valued = {
        "--cache-type-k": "cache_type_k",
        "-ctk": "cache_type_k",
        "--cache-type-v": "cache_type_v",
        "-ctv": "cache_type_v",
        "--cache-ram": "cache_ram",
        "--cache-reuse": "cache_reuse",
    }
    for flag, key in valued.items():
        raw = _flag_value(argv, flag)
        if raw is not None and raw != "":
            kv[key] = raw
    if _flag_present(argv, "--kv-unified"):
        kv["kv_unified"] = True
    if _flag_present(argv, "--no-kv-offload"):
        kv["kv_offload"] = False
    elif _flag_present(argv, "--kv-offload"):
        kv["kv_offload"] = True
    return kv or None


def _kv_from_props(props: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(props, dict):
        return None
    kv: Dict[str, Any] = {}
    settings = props.get("default_generation_settings")
    blobs: List[Dict[str, Any]] = [props]
    if isinstance(settings, dict):
        blobs.append(settings)
    keys = (
        "cache_type_k",
        "cache_type_v",
        "cache-type-k",
        "cache-type-v",
        "kv_unified",
        "cache_ram",
        "cache_reuse",
    )
    for blob in blobs:
        for key in keys:
            if key in blob and blob[key] is not None:
                kv[key.replace("-", "_")] = blob[key]
    return kv or None


def observe_runtime_topology(
    runtime: "Runtime",
    pool: Optional["RuntimePool"] = None,
) -> Dict[str, Any]:
    """Record only what this runtime can observe. Unobservable -> null + reason."""
    backend = getattr(runtime, "backend", None)
    ident: Dict[str, Any] = {}
    if backend is not None and hasattr(backend, "identity"):
        try:
            got = backend.identity()
            if isinstance(got, dict):
                ident = got
        except Exception:
            ident = {}

    argv: Optional[List[str]] = None
    if backend is not None and callable(getattr(backend, "command", None)):
        try:
            argv = list(
                backend.command(
                    port=getattr(backend, "port", None) or runtime.port,
                    n_slots=getattr(backend, "n_slots", None),
                )
            )
        except Exception:
            argv = None

    pid = runtime.pid or getattr(backend, "pid", None) or ident.get("pid")
    ps_argv = _ps_argv(int(pid)) if pid else None

    model_path = (
        runtime.model
        or ident.get("model_path")
        or (pool.model_path if pool is not None else None)
        or getattr(backend, "model_path", None)
    )
    if model_path and not is_remote_endpoint(str(model_path)):
        model_path = os.path.realpath(os.path.expanduser(str(model_path)))
    model_reason = None if model_path else "runtime has no model path"

    artifact = ident.get("model_identity")
    artifact_reason = None
    if not artifact:
        if model_path and os.path.isfile(model_path):
            try:
                artifact = f"{model_path}:{os.path.getsize(model_path)}"
            except OSError:
                artifact = None
                artifact_reason = "model file unreadable"
        else:
            artifact_reason = (
                "backend.identity() had no model_identity and the model "
                "file was not readable"
            )

    port = runtime.port or getattr(backend, "port", None) or ident.get("port")
    port_reason = None if port is not None else "runtime has no port"

    pid_reason = None if pid is not None else "process has no pid"

    ctx = getattr(backend, "ctx_size", None) if backend is not None else None
    if ctx is None:
        ctx = ident.get("context")
    if ctx is None:
        raw = _flag_value(argv, "--ctx-size", "-c") or _flag_value(
            ps_argv, "--ctx-size", "-c"
        )
        if raw not in (None, ""):
            try:
                ctx = int(raw)
            except (TypeError, ValueError):
                ctx = None
    try:
        ctx_i = int(ctx) if ctx is not None else None
    except (TypeError, ValueError):
        ctx_i = None
    ctx_reason = (
        None
        if ctx_i is not None
        else "backend did not expose --ctx-size and process argv had none"
    )

    parallel = getattr(backend, "n_slots", None) if backend is not None else None
    if parallel is None:
        parallel = ident.get("n_slots")
    if parallel is None:
        raw = _flag_value(argv, "--parallel", "-np") or _flag_value(
            ps_argv, "--parallel", "-np"
        )
        if raw not in (None, ""):
            try:
                parallel = int(raw)
            except (TypeError, ValueError):
                parallel = None
    try:
        parallel_i = int(parallel) if parallel is not None else None
    except (TypeError, ValueError):
        parallel_i = None
    parallel_reason = (
        None
        if parallel_i is not None
        else "backend did not expose --parallel and process argv had none"
    )

    per_slot = None
    per_slot_reason = None
    if ctx_i is not None and parallel_i is not None:
        per_slot = int(ctx_i) // max(1, int(parallel_i))
    else:
        per_slot_reason = (
            "need observed --ctx-size and --parallel to derive per-slot context"
        )

    active = getattr(runtime, "in_flight", None)
    active_reason = None
    if active is None:
        active_reason = "pool did not track in_flight for this runtime"

    kv = _kv_from_argv(argv) or _kv_from_argv(ps_argv)
    kv_reason = None
    if kv is None and port is not None:
        props, _err = _http_json(f"http://127.0.0.1:{int(port)}/props")
        kv = _kv_from_props(props)
    if kv is None:
        kv_reason = (
            "KV configuration was not on the spawn argv "
            "(--cache-type-k/--cache-type-v/--kv-unified) and live /props "
            "was not observed"
        )

    return {
        "model_path": topology_field(model_path, model_reason),
        "artifact_identity": topology_field(artifact, artifact_reason),
        "pid": topology_field(int(pid) if pid is not None else None, pid_reason),
        "port": topology_field(
            int(port) if port is not None else None, port_reason
        ),
        "ctx_size": topology_field(ctx_i, ctx_reason),
        "parallel": topology_field(parallel_i, parallel_reason),
        "per_slot_context": topology_field(per_slot, per_slot_reason),
        "active_sequences": topology_field(active, active_reason),
        "kv_configuration": topology_field(kv, kv_reason),
    }


def load_observed_overlap(workspace: Union[str, Path]) -> Optional[int]:
    path = Path(workspace) / ".hcli" / OVERLAP_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("max_in_flight")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    return n


def store_observed_overlap(workspace: Union[str, Path], n: int) -> None:
    """Persist the high-water mark. Never decreases a previously stored max."""
    try:
        value = int(n)
    except (TypeError, ValueError):
        return
    if value < 1:
        return
    root = Path(workspace)
    path = root / ".hcli" / OVERLAP_NAME
    with _OVERLAP_LOCK:
        existing = load_observed_overlap(root)
        value = max(value, existing or 0)
        if value < 1:
            return
        record = {
            "schema": OVERLAP_SCHEMA,
            "max_in_flight": value,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            _atomic_write(path, json.dumps(record, indent=2))
        except OSError:
            pass


def resolve_workspace(workspace: Optional[Union[str, Path]] = None) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    env = os.environ.get("HCLI_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    cur = Path.cwd().resolve()
    for path in (cur, *cur.parents):
        if (path / ".hcli").exists() or (path / ".haider").exists():
            return path
    return cur


@dataclass
class Runtime:
    index: int
    pid: Optional[int] = None
    port: Optional[int] = None
    model: str = ""
    process: Optional[subprocess.Popen] = None
    active: bool = False
    backend: Any = None
    slot: int = 0
    in_flight: int = 0
    owns_process: bool = True
    start_time: Optional[str] = None
    topology: Optional[Dict[str, Any]] = None

    def topology_receipt(self) -> Dict[str, Any]:
        if self.topology:
            return dict(self.topology)
        return observe_runtime_topology(self)

    def stop(self) -> None:
        if self.backend is not None and self.owns_process:
            try:
                self.backend.stop()
            except Exception:
                pass
            self.pid = None
            self.active = False
            self.process = None
            return
        if not self.owns_process:
            self.active = False
            return
        proc = self.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        if self.pid and pid_is_alive(self.pid):
            terminate_pid(int(self.pid))
        self.pid = None
        self.active = False
        self.process = None


class RuntimePool:
    """Resident runtimes and a separately enforced active-decode cap.

    Concurrency is not a throughput lever on this machine. RESIDENT_RUNTIME_LIMIT
    is how many may stay warm; ACTIVE_DECODE_LIMIT is how many may decode at
    once. Extra resident runtimes earn their keep through warm state, prefix
    locality, and non-decode work — not aggregate tok/s.

    Topology default is SLOT when receipts/headless/DECODE_TOPOLOGY.json exists:
    one llama-server with N slots, because process topology collapses past its
    peak while slot degrades gracefully. PROCESS (N servers x 1 slot) remains
    available via HCLI_DECODE_TOPOLOGY=process.
    """

    def __init__(
        self,
        model_path: str,
        requested_n: int = 1,
        workspace: Optional[Union[str, Path]] = None,
        *,
        backend_factory: Optional[Callable[..., Any]] = None,
        mem_gate: Optional[MemGate] = None,
        reserve_bytes: Optional[int] = None,
        swap_ceiling_bytes: Optional[int] = None,
        topology: Optional[str] = None,
        repo_root: Optional[Union[str, Path]] = None,
        observed_overlap: Optional[int] = None,
    ) -> None:
        raw_model_path = str(model_path or "").strip()
        self.model_path = (
            OpenAICompatibleBackend(raw_model_path).selection()
            if is_remote_endpoint(raw_model_path)
            else os.path.realpath(os.path.expanduser(raw_model_path))
        )
        self.requested_n = int(requested_n)
        self.workspace = resolve_workspace(workspace)
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else default_repo_root()
        )
        self.backend_factory = backend_factory
        self.observed_overlap = (
            int(observed_overlap) if observed_overlap is not None else None
        )

        model_bytes = 0
        if is_remote_endpoint(self.model_path):
            model_bytes = 0
        elif is_hawking_native_path(self.model_path):
            try:
                inventory = config_for_model_path(self.model_path).identity().get(
                    "artifact_inventory", {}
                )
                model_bytes = int(inventory.get("artifact_bytes") or 0)
            except (OSError, TypeError, ValueError):
                model_bytes = 0
        elif os.path.isfile(self.model_path):
            try:
                model_bytes = os.path.getsize(self.model_path)
            except OSError:
                model_bytes = 0

        limits = resolve_runtime_limits(
            repo_root=self.repo_root,
            start_dir=self.workspace,
            model_path=self.model_path,
            model_bytes=model_bytes or None,
        )
        self.limit_resolution = limits
        self.genome_reports = list(limits.genome_reports)
        self.resident_limit = int(limits.resident_limit)
        self.resident_source = limits.resident_source
        self.active_decode_limit = int(limits.active_decode_limit)
        self.active_source = limits.active_source
        self.allocation_decision: Optional[Dict[str, Any]] = None

        if topology in {"slot", "process"}:
            self.topology = topology
            self.topology_source = "constructor"
        else:
            self.topology, self.topology_source = resolve_decode_topology(
                self.repo_root
            )
        if mem_gate is not None:
            self.mem_gate = mem_gate
            self.mem_gate.topology = self.topology
            if reserve_bytes is not None:
                self.mem_gate.reserve_bytes = reserve_bytes
            if not self.mem_gate.model_bytes and model_bytes:
                self.mem_gate.model_bytes = model_bytes
        else:
            self.mem_gate = MemGate(
                reserve_bytes=reserve_bytes,
                swap_ceiling_bytes=swap_ceiling_bytes,
                model_bytes=model_bytes,
                topology=self.topology,
            )

        self._apply_context_budget(self._resolve_context_budget(self.requested_n))
        self.admitted_n = 0
        self.runtimes: List[Runtime] = []
        self.admission_records: List[Dict[str, Any]] = []
        self.refusal_reason: Optional[str] = None
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.max_in_flight_observed = 0
        self._in_flight = 0
        self.overlap_admit_cap = self._overlap_admit_cap()
        self._prefix_owner: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._decode_sema = threading.Semaphore(max(1, self.active_decode_limit))
        self._stopped = False
        self._pool_start_time = process_start_token(os.getpid())
        _register_live(self)
        self.reap_orphans()

    def _n_parallel_for_budget(self, n: int) -> int:
        if self.topology == "slot":
            return max(1, int(n))
        return 1

    def _resolve_context_budget(self, n: int) -> ContextBudget:
        return resolve_context_budget(
            model_path=self.model_path,
            n_parallel=self._n_parallel_for_budget(n),
            repo_root=str(self.repo_root),
        )

    def _apply_context_budget(self, budget: ContextBudget) -> None:
        self.context_budget = budget
        self.ctx_size = int(budget.total_ctx)
        self.per_request_ctx = int(budget.per_request_ctx)

    def _reconcile_spawned_budget(self, backend: Any) -> None:
        port = getattr(backend, "port", None)
        if port is None:
            return
        props = probe_server_context(int(port))
        if not props or not props.get("per_slot_n_ctx"):
            return
        observed = int(props["per_slot_n_ctx"])
        self._apply_context_budget(
            apply_observed_slot(self.context_budget, observed, props=props)
        )

    def _overlap_admit_cap(self) -> int:
        """How many runtimes in-flight model calls have been observed to need.

        Constructor `observed_overlap` wins, then HCLI_OBSERVED_MODEL_OVERLAP,
        then the workspace high-water file, then the measured default of 1.
        """
        if self.observed_overlap is not None:
            try:
                return max(1, int(self.observed_overlap))
            except (TypeError, ValueError):
                pass
        raw = os.environ.get("HCLI_OBSERVED_MODEL_OVERLAP")
        if raw is not None and str(raw).strip() != "":
            try:
                return max(1, int(str(raw).strip()))
            except ValueError:
                pass
        loaded = load_observed_overlap(self.workspace)
        if loaded is not None:
            return max(1, int(loaded))
        return DEFAULT_OVERLAP_ADMIT_CAP

    def _admit(self, n: int) -> int:
        """Plan how many logical runtimes to try. Not a memory gate.

        Width follows measured model-call overlap, not the caller's requested
        constant. A mission that never has two completions in flight does not
        get extra 19.79 GiB processes (or extra KV slots) sitting idle.
        """
        want = max(0, int(n))
        want = min(want, int(self.resident_limit))
        raw = os.environ.get("HCLI_MAX_RUNTIMES")
        if raw:
            try:
                want = min(want, max(0, int(raw)))
            except ValueError:
                pass
        cap = self._overlap_admit_cap()
        self.overlap_admit_cap = cap
        if cap < want:
            # Say so. Narrowing admission from what the caller asked for is a
            # real decision with a real consequence -- each runtime this drops
            # is ~19.79 GiB of Metal working set NOT reserved -- and a decision
            # nobody can see is the shape this campaign keeps finding: a caller
            # asks for 5, gets 1, and nothing in the record says why.
            self.admission_narrowed = {
                "requested": int(n),
                "admitted": int(cap),
                "reason": "observed model-call overlap",
                "source": (
                    "constructor" if self.observed_overlap is not None
                    else "HCLI_OBSERVED_MODEL_OVERLAP"
                    if os.environ.get("HCLI_OBSERVED_MODEL_OVERLAP")
                    else "workspace high-water file"
                    if load_observed_overlap(self.workspace) is not None
                    else "measured default"
                ),
            }
        want = min(want, cap)
        return want

    def _ownership_path(self) -> Path:
        return self.workspace / ".hcli" / OWNERSHIP_NAME

    def _read_ownership(self) -> Optional[Dict[str, Any]]:
        path = self._ownership_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_ownership(self) -> None:
        children = []
        seen = set()
        for runtime in self.runtimes:
            if not runtime.pid or runtime.pid in seen:
                continue
            seen.add(runtime.pid)
            ident = {}
            if runtime.backend is not None and hasattr(runtime.backend, "identity"):
                try:
                    ident = runtime.backend.identity()
                except Exception:
                    ident = {}
            children.append(
                {
                    "pid": runtime.pid,
                    "start_time": runtime.start_time,
                    "port": runtime.port,
                    "model": runtime.model or self.model_path,
                    "started_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "backend": ident.get("backend") or "llama_server",
                    "model_identity": ident.get("model_identity"),
                    "topology": runtime.topology_receipt()
                    if runtime.topology or runtime.backend is not None
                    else None,
                }
            )
        record = {
            "schema": OWNERSHIP_SCHEMA,
            "pool_pid": os.getpid(),
            "pool_start_time": self._pool_start_time,
            "topology": self.topology,
            "model": self.model_path,
            "children": children,
        }
        _atomic_write(self._ownership_path(), json.dumps(record, indent=2))

    def _clear_ownership(self) -> None:
        path = self._ownership_path()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            try:
                _atomic_write(
                    path,
                    json.dumps(
                        {
                            "schema": OWNERSHIP_SCHEMA,
                            "pool_pid": os.getpid(),
                            "pool_start_time": self._pool_start_time,
                            "children": [],
                        },
                        indent=2,
                    ),
                )
            except OSError:
                pass

    def _owner_is_live(self, record: Dict[str, Any]) -> bool:
        try:
            pid = int(record.get("pool_pid"))
        except (TypeError, ValueError):
            return False
        if pid == os.getpid() and self.runtimes:
            return True
        if not pid_is_alive(pid):
            return False
        recorded = record.get("pool_start_time")
        if not recorded:
            return True
        live = process_start_token(pid)
        if live is None:
            return True
        return str(live) == str(recorded)

    def reap_orphans(self) -> List[Dict[str, Any]]:
        """Reap children recorded by a previous, now-dead pool.

        Match pid AND start time. Never kill a process this pool did not spawn.
        A bare `pkill llama-server` is forbidden.
        """
        record = self._read_ownership()
        if not record:
            return []
        if self._owner_is_live(record):
            return []
        reports: List[Dict[str, Any]] = []
        remaining = []
        for child in record.get("children") or []:
            if not isinstance(child, dict):
                continue
            try:
                pid = int(child.get("pid"))
            except (TypeError, ValueError):
                continue
            recorded_start = child.get("start_time")
            if not pid_is_alive(pid):
                reports.append({"pid": pid, "action": "already_dead"})
                continue
            if not recorded_start:
                remaining.append(child)
                reports.append(
                    {
                        "pid": pid,
                        "action": "skipped",
                        "reason": "no recorded start_time; refusing to kill",
                    }
                )
                continue
            live_start = process_start_token(pid)
            if live_start is None:
                remaining.append(child)
                reports.append(
                    {
                        "pid": pid,
                        "action": "skipped",
                        "reason": "could not read live start_time",
                    }
                )
                continue
            if str(live_start) != str(recorded_start):
                reports.append(
                    {
                        "pid": pid,
                        "action": "skipped",
                        "reason": "pid reused; start_time mismatch",
                    }
                )
                continue
            killed = terminate_pid(pid)
            killed["action"] = "reaped" if killed.get("gone") else "unreaped"
            reports.append(killed)
            if not killed.get("gone"):
                remaining.append(child)
        if remaining:
            record["children"] = remaining
            try:
                _atomic_write(self._ownership_path(), json.dumps(record, indent=2))
            except OSError:
                pass
        else:
            self._clear_ownership()
        return reports

    def _make_backend(self, index: int, n_slots: int, port: int) -> Any:
        if self.backend_factory is not None:
            backend = self.backend_factory(
                model_path=self.model_path,
                port=port,
                n_slots=n_slots,
                index=index,
            )
        else:
            from .runtime_iface import make_backend_for_model

            backend = make_backend_for_model(
                self.model_path,
                port=port,
                n_slots=n_slots,
                ctx_size=self.ctx_size,
                index=index,
            )
        spawn = getattr(backend, "spawn", None)
        if callable(spawn):
            spawn(port=port, n_slots=n_slots)
        return backend

    def _plan_count(self) -> int:
        want = self._admit(self.requested_n)
        if want <= 0:
            self.refusal_reason = "requested or resident limit is 0"
            return 0
        n = 0
        last_refuse: Optional[AdmissionDecision] = None
        for _ in range(want):
            decision = self.mem_gate.consider(admitted=n, extra=1, refresh_metal=False)
            if not decision.allow:
                last_refuse = decision
                break
            n += 1
        if n <= 0:
            self.refusal_reason = (
                last_refuse.reason if last_refuse is not None else "memgate refused"
            )
            self.admission_records.append(
                {
                    "index": 0,
                    "admitted": False,
                    "reason": self.refusal_reason,
                    "gate": last_refuse.gate if last_refuse else "memgate",
                    "details": last_refuse.details if last_refuse else {},
                    "marginal_free_ram_cost_bytes": 0,
                }
            )
            return 0
        if last_refuse is not None:
            self.refusal_reason = last_refuse.reason
        return n

    def _wait_ready(self, backend: Any, index: int, timeout: float) -> None:
        ready = getattr(backend, "ready", None)
        if callable(ready):
            ok = ready(timeout)
            if not ok:
                proc = getattr(backend, "process", None)
                code = proc.returncode if proc is not None else None
                tail = ""
                log_tail = getattr(backend, "log_tail", None)
                if callable(log_tail):
                    try:
                        tail = log_tail()
                    except Exception:
                        tail = ""
                detail = (
                    f"runtime {index} exited before readiness (code={code})"
                    if proc is not None
                    and getattr(proc, "poll", lambda: None)() is not None
                    else f"Runtime {index} not ready after {timeout}s"
                )
                if tail:
                    detail = f"{detail}\n--- llama-server log ---\n{tail}"
                raise RuntimeError(detail)
            return
        raise RuntimeError(f"runtime {index} backend has no ready()")

    def _runtime_from_backend(
        self,
        index: int,
        backend: Any,
        *,
        slot: int,
        owns_process: bool,
    ) -> Runtime:
        proc = getattr(backend, "process", None)
        pid = getattr(backend, "pid", None) or (proc.pid if proc is not None else None)
        port = getattr(backend, "port", None)
        start_time = getattr(backend, "start_time", None) or (
            process_start_token(int(pid)) if pid else None
        )
        runtime = Runtime(
            index=index,
            pid=pid,
            port=port,
            model=self.model_path,
            process=proc,
            active=True,
            backend=backend,
            slot=slot,
            owns_process=owns_process,
            start_time=start_time,
        )
        runtime.topology = observe_runtime_topology(runtime, pool=self)
        return runtime

    def _record_admission(
        self,
        index: int,
        runtime: Runtime,
        before: Dict[str, Any],
        after: Dict[str, Any],
        kind: str,
        gpu_charged_bytes: int,
    ) -> None:
        marginal = int(before.get("free_bytes") or 0) - int(after.get("free_bytes") or 0)
        rss = 0
        if runtime.pid:
            try:
                out = subprocess.run(
                    ["ps", "-p", str(runtime.pid), "-o", "rss="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                text = (out.stdout or "").strip()
                if text:
                    rss = int(text.split()[0]) * 1024
            except Exception:
                rss = 0
        self.admission_records.append(
            {
                "index": index,
                "admitted": True,
                "kind": kind,
                "pid": runtime.pid,
                "port": runtime.port,
                "slot": runtime.slot,
                "rss_bytes": rss,
                "marginal_free_ram_cost_bytes": max(0, marginal),
                "gpu_charged_bytes": int(gpu_charged_bytes),
                "free_after_bytes": int(after.get("free_bytes") or 0),
                "swap_after_bytes": int(after.get("swap_used_bytes") or 0),
                "reason": "admitted",
                "topology": runtime.topology_receipt(),
                "allocation_decision": self.allocation_decision,
            }
        )

    def start(self) -> None:
        def runtime_is_live(runtime: Runtime) -> bool:
            if not runtime.active:
                return False
            process = runtime.process
            if process is not None:
                return process.poll() is None
            if runtime.pid is not None:
                return pid_is_alive(runtime.pid)
            backend = runtime.backend
            ready = getattr(backend, "ready", None) if backend is not None else None
            if callable(ready):
                try:
                    return bool(ready(0.0))
                except Exception:
                    return False
            return False

        if self.runtimes and all(runtime_is_live(runtime) for runtime in self.runtimes):
            return

        self.stop()
        self._stopped = False
        self.admission_records = []
        self.refusal_reason = None
        self.prefix_hits = 0
        self.prefix_misses = 0
        self.max_in_flight_observed = 0
        self._in_flight = 0
        self._prefix_owner = {}
        self.reap_orphans()

        if self.backend_factory is None and not (
            os.path.isfile(self.model_path)
            or is_mlx_model_dir(self.model_path)
            or is_hawking_native_path(self.model_path)
            or is_remote_endpoint(self.model_path)
        ):
            raise FileNotFoundError(self.model_path)

        planned = self._plan_count()
        self.allocation_decision = slot_allocation_decision(
            planned_slots=planned,
            topology=self.topology,
            active_decode_limit=self.active_decode_limit,
            requested_n=self.requested_n,
        )
        if planned <= 0:
            self.admitted_n = 0
            self.runtimes = []
            return

        self._apply_context_budget(self._resolve_context_budget(planned))
        timeout = float(os.environ.get("HCLI_READY_TIMEOUT", "300"))
        try:
            if self.topology == "slot":
                self._start_slot(planned, timeout)
            else:
                self._start_process(planned, timeout)
        except BaseException:
            self.stop()
            raise

        self.admitted_n = len(self.runtimes)
        if self.runtimes:
            self._write_ownership()

    def _start_slot(self, planned: int, timeout: float) -> None:
        before = host_snapshot()
        port = allocate_port()
        backend = self._make_backend(0, planned, port)
        try:
            self._wait_ready(backend, 0, timeout)
        except BaseException:
            try:
                backend.stop()
            except Exception:
                pass
            raise
        self._reconcile_spawned_budget(backend)
        after = host_snapshot()
        for slot in range(planned):
            runtime = self._runtime_from_backend(
                slot, backend, slot=slot, owns_process=(slot == 0)
            )
            self.runtimes.append(runtime)
            if slot == 0:
                self._record_admission(
                    slot,
                    runtime,
                    before,
                    after,
                    kind="slot-process",
                    gpu_charged_bytes=self.mem_gate.gpu_cost_bytes(0, extra=1),
                )
            else:
                # Extra slots share the process mmap and a single Metal working
                # set for the weights; host marginal is the post-spawn delta
                # already attributed to slot 0.
                self.admission_records.append(
                    {
                        "index": slot,
                        "admitted": True,
                        "kind": "slot",
                        "pid": runtime.pid,
                        "port": runtime.port,
                        "slot": slot,
                        "rss_bytes": self.admission_records[0].get("rss_bytes", 0)
                        if self.admission_records
                        else 0,
                        "marginal_free_ram_cost_bytes": 0,
                        "gpu_charged_bytes": self.mem_gate.per_runtime_overhead_bytes,
                        "free_after_bytes": int(after.get("free_bytes") or 0),
                        "swap_after_bytes": int(after.get("swap_used_bytes") or 0),
                        "reason": "slot shares process weights (mmap + one MTL working set)",
                        "topology": runtime.topology_receipt(),
                        "allocation_decision": self.allocation_decision,
                    }
                )
            post = self.mem_gate.consider(
                admitted=slot + 1, extra=1, snapshot=after, refresh_metal=False
            )
            if slot + 1 < planned and not post.allow:
                self.refusal_reason = post.reason
                # Already spawned `planned` slots in one process; keep them.
                break

    def _start_process(self, planned: int, timeout: float) -> None:
        for index in range(planned):
            decision = self.mem_gate.consider(
                admitted=len(self.runtimes), extra=1, refresh_metal=False
            )
            if not decision.allow:
                self.refusal_reason = decision.reason
                self.admission_records.append(
                    {
                        "index": index,
                        "admitted": False,
                        "reason": decision.reason,
                        "gate": decision.gate,
                        "details": decision.details,
                        "marginal_free_ram_cost_bytes": 0,
                    }
                )
                break
            before = host_snapshot()
            port = allocate_port()
            backend = self._make_backend(index, 1, port)
            try:
                self._wait_ready(backend, index, timeout)
            except BaseException:
                try:
                    backend.stop()
                except Exception:
                    pass
                raise
            if index == 0:
                self._reconcile_spawned_budget(backend)
            after = host_snapshot()
            runtime = self._runtime_from_backend(
                index, backend, slot=0, owns_process=True
            )
            self.runtimes.append(runtime)
            self._record_admission(
                index,
                runtime,
                before,
                after,
                kind="process",
                gpu_charged_bytes=self.mem_gate.gpu_cost_bytes(index, extra=1)
                - self.mem_gate.gpu_cost_bytes(index, extra=0),
            )
            self._write_ownership()
            post = self.mem_gate.consider(
                admitted=len(self.runtimes), extra=1, snapshot=after, refresh_metal=False
            )
            if not post.allow:
                self.refusal_reason = post.reason
                break

    def _pick_locked(self, prefix_key: Optional[str]) -> Runtime:
        live = [r for r in self.runtimes if r.active]
        if not live:
            raise RuntimeError("RuntimePool has no admitted runtimes")
        if prefix_key:
            owner = self._prefix_owner.get(prefix_key)
            if owner is not None:
                for runtime in live:
                    if runtime.index == owner:
                        self.prefix_hits += 1
                        return runtime
            self.prefix_misses += 1
        return min(live, key=lambda r: (r.in_flight, r.index))

    def complete(
        self,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        prefix_key: Optional[str] = None,
    ) -> CompletionResult:
        if not self.runtimes:
            raise RuntimeError("RuntimePool has no admitted runtimes")
        self._decode_sema.acquire()
        runtime: Optional[Runtime] = None
        try:
            with self._lock:
                self._in_flight += 1
                grew = False
                if self._in_flight > self.max_in_flight_observed:
                    self.max_in_flight_observed = self._in_flight
                    grew = True
                runtime = self._pick_locked(prefix_key)
                runtime.in_flight += 1
            if grew:
                store_observed_overlap(
                    self.workspace, self.max_in_flight_observed
                )
            backend = runtime.backend
            if backend is None or not hasattr(backend, "complete"):
                raise RuntimeError("runtime has no backend.complete")
            old_handle = (runtime.pid, id(runtime.process), runtime.start_time)
            try:
                result = backend.complete(payload, timeout=timeout)
            finally:
                # A resident backend may replace its child after a protocol
                # failure. Refresh the pool's handle before Engine snapshots
                # provenance or a later stop/reap would still point at the
                # dead generation.
                process = getattr(backend, "process", None)
                if hasattr(backend, "process"):
                    runtime.process = process
                if hasattr(backend, "pid"):
                    runtime.pid = getattr(backend, "pid", None) or (
                        process.pid if process is not None else None
                    )
                if hasattr(backend, "port"):
                    runtime.port = getattr(backend, "port", None)
                if hasattr(backend, "start_time"):
                    runtime.start_time = getattr(backend, "start_time", None) or (
                        process_start_token(runtime.pid) if runtime.pid else None
                    )
                new_handle = (runtime.pid, id(runtime.process), runtime.start_time)
                if new_handle != old_handle and (hasattr(backend, "pid") or hasattr(backend, "process")):
                    runtime.topology = observe_runtime_topology(runtime, pool=self)
                    try:
                        self._write_ownership()
                    except OSError:
                        # Ownership persistence must not turn a completed
                        # inference into a failed one; the next pool start
                        # still performs its normal liveness checks.
                        pass
            if not isinstance(result, CompletionResult):
                result = CompletionResult(raw=result)
            result.runtime_index = runtime.index
            if prefix_key:
                with self._lock:
                    self._prefix_owner[prefix_key] = runtime.index
            return result
        finally:
            if runtime is not None:
                with self._lock:
                    runtime.in_flight = max(0, runtime.in_flight - 1)
                    self._in_flight = max(0, self._in_flight - 1)
            self._decode_sema.release()

    def stop(self) -> Dict[str, Any]:
        reaped: List[int] = []
        unreaped: List[int] = []
        seen_backends = set()
        for runtime in list(self.runtimes):
            backend = runtime.backend
            pid = runtime.pid
            if backend is not None and id(backend) not in seen_backends:
                seen_backends.add(id(backend))
                try:
                    report = backend.stop()
                except Exception:
                    report = {}
                gone = True if not report else bool(report.get("gone", True))
                leftover = list(report.get("unreaped") or [])
                if pid and gone and not pid_is_alive(pid):
                    reaped.append(int(pid))
                elif pid and pid_is_alive(pid):
                    killed = terminate_pid(int(pid))
                    if killed.get("gone"):
                        reaped.append(int(pid))
                    else:
                        unreaped.append(int(pid))
                unreaped.extend(int(p) for p in leftover if p)
            elif runtime.owns_process:
                try:
                    runtime.stop()
                except Exception:
                    pass
                if pid and not pid_is_alive(pid):
                    reaped.append(int(pid))
                elif pid and pid_is_alive(pid):
                    unreaped.append(int(pid))
            runtime.active = False
            runtime.pid = None
            runtime.process = None
        self.runtimes.clear()
        self.admitted_n = 0
        self._stopped = True
        if unreaped:
            # Keep ownership of anything we could not reap so a later pool can.
            still = []
            for pid in unreaped:
                still.append(
                    {
                        "pid": pid,
                        "start_time": process_start_token(pid),
                        "model": self.model_path,
                    }
                )
            try:
                _atomic_write(
                    self._ownership_path(),
                    json.dumps(
                        {
                            "schema": OWNERSHIP_SCHEMA,
                            "pool_pid": os.getpid(),
                            "pool_start_time": "stopped-unreaped",
                            "children": still,
                        },
                        indent=2,
                    ),
                )
            except OSError:
                pass
        else:
            self._clear_ownership()
        return {"reaped": reaped, "unreaped": unreaped}


__all__ = [
    "Runtime",
    "RuntimePool",
    "allocate_port",
    "CompletionResult",
    "observe_runtime_topology",
    "topology_field",
    "TOPOLOGY_KEYS",
    "load_observed_overlap",
    "store_observed_overlap",
    "DEFAULT_OVERLAP_ADMIT_CAP",
    "OVERLAP_NAME",
]
