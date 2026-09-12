"""Wire S010's decision core to a real resident lifecycle.

``tools/future/resident_adoption.py`` is a pure function of a record and an
observation. Nothing in the execution path called it. This module is that
caller: observe the live process, prove it can GENERATE, then adopt — or
reap and spawn. The orphan reaper stays; uncertainty always reaps.

A leaked resident holds ~22 GB. Adopting a listener that cannot generate is
how an orphaned mlx_lm.server held 127.0.0.1:9999 with its model directory
already gone.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from .backends import OpenAICompatibleBackend, terminate_pid
from .persist import atomic_write_json
from .resources import pid_is_alive, process_start_token

def _decision_core_path() -> Path:
    """Locate the pure adoption core from a checkout or stamped install.

    The HCLI package is copied into ``~/.local/share/hcli/build-*`` by
    ``install-shims``.  In that form ``Path(__file__).parent.parent`` is the
    build directory, not the repository, so deriving ``tools/future`` from it
    breaks an ordinary ``hcli web`` launch.  Keep the source module as the
    single canonical owner, but resolve it through the install stamp when the
    package is detached from the checkout.
    """
    candidates = []
    configured = os.environ.get("HCLI_SOURCE_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser() / "tools" / "future" /
                          "resident_adoption.py")

    package_root = Path(__file__).resolve().parent.parent
    candidates.append(package_root / "tools" / "future" / "resident_adoption.py")

    stamp_path = package_root / "install.json"
    source: Optional[Path] = None
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        raw_source = str(stamp.get("source", "")).strip()
        if raw_source:
            source = Path(raw_source).expanduser()
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        source = None
    if source is not None:
        candidates.append(source / "tools" / "future" / "resident_adoption.py")
        # install_shims stamps the package directory, while the source core
        # lives beside that package at the repository root.
        if source.name == "hcli":
            candidates.append(source.parent / "tools" / "future" /
                              "resident_adoption.py")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise ImportError(f"decision core is missing; searched: {searched}")


_CORE_PATH = _decision_core_path()
_spec = importlib.util.spec_from_file_location(
    "hawking_resident_adoption_core", _CORE_PATH
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"decision core is missing: {_CORE_PATH}")
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)

ADOPTED = "ADOPTED"
SPAWNED = "SPAWNED"
REAPED_THEN_SPAWNED = "REAPED_THEN_SPAWNED"

_claim_threads: dict[str, int] = {}
_claim_threads_lock = threading.Lock()


class CompetingOwnerError(RuntimeError):
    """A live owner already holds the claim. We did not adopt and did not reap."""


@dataclass(frozen=True)
class ResidentHandle:
    pid: int
    proc_start: str
    endpoint: str
    model_hash: str
    config_hash: str
    owner_generation: int


def prove_unix_jsonl(path: str, timeout: float = 5.0) -> bool:
    """One-token JSONL generate over a Unix socket. Bind is not readiness."""
    import socket as _socket

    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.settimeout(max(1.0, float(timeout)))
    try:
        sock.connect(path)
        sock.sendall(b'{"id":"prove","prompt":"1","max_new_tokens":1}\n')
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                return False
            buf += chunk
        rec = json.loads(buf.split(b"\n", 1)[0].decode("utf-8", "replace"))
        if not isinstance(rec, dict):
            return False
        if rec.get("status") in {"ok", "ready"}:
            return True
        return rec.get("id") == "prove" and "error" not in rec
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def prove_endpoint_capability(endpoint: str, timeout: float = 5.0) -> bool:
    """Reachability is not readiness. Delegate to OpenAICompatibleBackend.ready.

    That path runs ``_prove_capability``: a one-token generate for loopback
    endpoints, cached per connector. Do not replace this with a listener check.
    Unix JSONL sockets (hawking-native after stdin EOF) get a one-token ping.
    """
    url = str(endpoint or "").strip()
    if url.startswith("unix:"):
        return prove_unix_jsonl(url[5:], timeout)
    if not url.startswith(("http://", "https://")):
        return False
    try:
        backend = OpenAICompatibleBackend(url)
    except (ValueError, TypeError):
        return False
    backend.spawn()
    try:
        return bool(backend.ready(timeout))
    except Exception:
        return False


def adopt_or_spawn(
    identity: Any,
    spawn_fn: Callable[[], Any],
    ownership_path: Union[str, os.PathLike[str]],
    *,
    capability_timeout: float = 5.0,
    now: Optional[float] = None,
    prove_fn: Optional[Callable[[str, float], bool]] = None,
) -> tuple[ResidentHandle, str]:
    """Adopt a proven resident, or spawn one after a safe reap.

    Returns ``(handle, "ADOPTED"|"SPAWNED"|"REAPED_THEN_SPAWNED")``.
    A live competing owner raises ``CompetingOwnerError`` rather than
    adopting the same body or killing it out from under the winner.

    ``prove_fn(endpoint, timeout)`` defaults to ``prove_endpoint_capability``
    (OpenAICompatibleBackend.ready → ``_prove_capability``). The serving
    layer passes its own prove so a just-spawned native/fake backend that
    already passed ``ready()`` is not required to speak HTTP.
    """
    model_hash, config_hash = _identity_hashes(identity)
    path = Path(ownership_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clock = time.time() if now is None else float(now)
    prove = prove_fn or prove_endpoint_capability

    transfer_interrupted = False
    if not _acquire_claim(path):
        holder = _read_claim(path)
        holder_pid = _claim_owner_pid(holder)
        if holder_pid == os.getpid() and _claim_held_by_this_thread(path):
            pass
        elif holder_pid > 0 and pid_is_alive(holder_pid) and holder_pid != os.getpid():
            raise CompetingOwnerError(
                f"claim held by live pid {holder_pid}"
            )
        elif holder_pid == os.getpid() and not _claim_held_by_this_thread(path):
            raise CompetingOwnerError(
                "claim held by another adopter in this process"
            )
        else:
            # Claim held, owner gone. If the record's owner is that same
            # dead pid, the previous pool finished the transfer and then
            # died — that is the adoptable case. If the claim owner and
            # the record owner differ, the transfer was interrupted.
            rec_preview = _core.load(path)
            transfer_interrupted = not (
                rec_preview is not None and rec_preview.owner_pid == holder_pid
            )
            _release_claim(path)
            if not _acquire_claim(path):
                raise CompetingOwnerError("could not obtain exclusive claim")

    rec = _core.load(path)
    if rec is not None and _foreign_owner_live(rec):
        _release_claim(path)
        raise CompetingOwnerError(
            f"already owned by live pid {rec.owner_pid}"
        )

    obs = _observe(rec, model_hash, config_hash, capability_timeout, prove)
    decision, why = _core.decide(rec, obs, clock)
    if transfer_interrupted:
        decision, why = _core.REAP, (
            "owner crashed during transfer; claim held, owner gone"
        )

    if decision == _core.ADOPT and rec is not None:
        gen = _next_generation(rec)
        handle = ResidentHandle(
            pid=int(rec.pid),
            proc_start=str(rec.proc_start),
            endpoint=str(rec.endpoint),
            model_hash=str(rec.model_hash),
            config_hash=str(rec.config_hash),
            owner_generation=gen,
        )
        _write_record(path, handle, gen, ready=True)
        return handle, ADOPTED

    had_record = rec is not None
    _safe_reap(rec)
    handle = _spawn_ready(
        spawn_fn,
        model_hash,
        config_hash,
        _next_generation(rec),
        capability_timeout,
        path,
        prove,
    )
    _write_record(path, handle, handle.owner_generation, ready=True)
    if had_record:
        return handle, REAPED_THEN_SPAWNED
    return handle, SPAWNED


def _identity_hashes(identity: Any) -> tuple[str, str]:
    if isinstance(identity, Mapping):
        model = identity.get("model_hash")
        config = identity.get("config_hash")
    else:
        model = getattr(identity, "model_hash", None)
        config = getattr(identity, "config_hash", None)
    model_s = str(model or "")
    config_s = str(config or "")
    if not model_s or not config_s:
        raise ValueError("identity requires non-empty model_hash and config_hash")
    return model_s, config_s


def _observe(
    rec: Any,
    model_hash: str,
    config_hash: str,
    capability_timeout: float,
    prove: Callable[[str, float], bool],
) -> dict[str, Any]:
    """Honest observation. Copied record fields are a lie and would adopt corpses."""
    if not isinstance(rec, _core.Record):
        return {
            "pid_alive": False,
            "proc_start": None,
            "model_hash": model_hash,
            "config_hash": config_hash,
            "responsive": False,
            "guard_state": "OK",
            "owned_by_live_other": False,
        }
    pid_alive = pid_is_alive(int(rec.pid))
    live_start = process_start_token(int(rec.pid)) if pid_alive else None
    same_incarnation = bool(pid_alive and live_start == rec.proc_start)
    # Uncertainty about start identity is a mismatch, never a copy of the record.
    obs_start = live_start
    responsive = False
    if same_incarnation:
        try:
            responsive = bool(prove(rec.endpoint, capability_timeout))
        except Exception:
            responsive = False
    return {
        "pid_alive": pid_alive,
        "proc_start": obs_start,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "responsive": responsive,
        "guard_state": "OK",
        "owned_by_live_other": _foreign_owner_live(rec),
    }


def _foreign_owner_live(rec: Any) -> bool:
    try:
        owner = int(rec.owner_pid)
    except (TypeError, ValueError, AttributeError):
        return False
    if owner <= 0 or owner == os.getpid():
        return False
    return pid_is_alive(owner)


def _next_generation(rec: Any) -> int:
    if not isinstance(rec, _core.Record):
        return 1
    try:
        current = int(rec.owner_generation)
    except (TypeError, ValueError):
        return 1
    return current + 1 if current >= 0 else 1


def _acquire_claim(path: Path) -> bool:
    ok = bool(_core.claim(path, 0))
    if ok:
        with _claim_threads_lock:
            _claim_threads[str(path)] = threading.get_ident()
    return ok


def _claim_held_by_this_thread(path: Path) -> bool:
    with _claim_threads_lock:
        return _claim_threads.get(str(path)) == threading.get_ident()


def release_ownership(path: Union[str, os.PathLike[str]]) -> None:
    """Drop the claim and the record. A clean stop leaves nothing to adopt."""
    dest = Path(path)
    _release_claim(dest)
    try:
        dest.unlink()
    except FileNotFoundError:
        pass


def _release_claim(path: Path) -> None:
    _core.release(path)
    with _claim_threads_lock:
        _claim_threads.pop(str(path), None)


def _read_claim(path: Path) -> Optional[dict[str, Any]]:
    lock = path.with_suffix(".claim")
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _claim_owner_pid(holder: Optional[dict[str, Any]]) -> int:
    if not holder:
        return 0
    try:
        return int(holder.get("owner_pid") or 0)
    except (TypeError, ValueError):
        return 0


def _safe_reap(rec: Any) -> None:
    """Kill only the recorded incarnation. A reused pid is a different process."""
    if not isinstance(rec, _core.Record):
        return
    try:
        pid = int(rec.pid)
    except (TypeError, ValueError):
        return
    if pid <= 0 or not pid_is_alive(pid):
        return
    live_start = process_start_token(pid)
    if live_start is None or live_start != rec.proc_start:
        return
    terminate_pid(pid)


def _spawn_ready(
    spawn_fn: Callable[[], Any],
    model_hash: str,
    config_hash: str,
    generation: int,
    capability_timeout: float,
    path: Path,
    prove: Callable[[str, float], bool],
) -> ResidentHandle:
    try:
        spawned = spawn_fn()
    except Exception:
        _release_claim(path)
        raise
    handle = _as_handle(spawned, model_hash, config_hash, generation)
    if handle.model_hash != model_hash or handle.config_hash != config_hash:
        _safe_reap_pid(handle.pid, handle.proc_start)
        _release_claim(path)
        raise RuntimeError("spawn_fn returned a different identity")
    if not handle.proc_start:
        # A record with no start token cannot defend against PID reuse later.
        _safe_reap_pid(handle.pid, handle.proc_start)
        _release_claim(path)
        raise RuntimeError("spawned resident has no process start identity")
    try:
        capable = bool(prove(handle.endpoint, capability_timeout))
    except Exception:
        capable = False
    if not capable:
        _safe_reap_pid(handle.pid, handle.proc_start)
        _release_claim(path)
        raise RuntimeError("spawned resident cannot generate")
    return handle


def _safe_reap_pid(pid: int, proc_start: str) -> None:
    if pid <= 0 or not pid_is_alive(pid):
        return
    live_start = process_start_token(pid)
    if live_start is None:
        return
    if proc_start and live_start != proc_start:
        return
    terminate_pid(pid)


def _as_handle(
    spawned: Any, model_hash: str, config_hash: str, generation: int
) -> ResidentHandle:
    if isinstance(spawned, ResidentHandle):
        return ResidentHandle(
            pid=int(spawned.pid),
            proc_start=str(spawned.proc_start or "")
            or (process_start_token(int(spawned.pid)) or ""),
            endpoint=str(spawned.endpoint),
            model_hash=str(spawned.model_hash or model_hash),
            config_hash=str(spawned.config_hash or config_hash),
            owner_generation=int(generation),
        )
    if isinstance(spawned, Mapping):
        pid = int(spawned["pid"])
        endpoint = str(spawned["endpoint"])
        proc_start = str(spawned.get("proc_start") or process_start_token(pid) or "")
        return ResidentHandle(
            pid=pid,
            proc_start=proc_start,
            endpoint=endpoint,
            model_hash=str(spawned.get("model_hash") or model_hash),
            config_hash=str(spawned.get("config_hash") or config_hash),
            owner_generation=int(generation),
        )
    pid = int(getattr(spawned, "pid"))
    endpoint = str(getattr(spawned, "endpoint"))
    proc_start = str(
        getattr(spawned, "proc_start", None) or process_start_token(pid) or ""
    )
    return ResidentHandle(
        pid=pid,
        proc_start=proc_start,
        endpoint=endpoint,
        model_hash=str(getattr(spawned, "model_hash", None) or model_hash),
        config_hash=str(getattr(spawned, "config_hash", None) or config_hash),
        owner_generation=int(generation),
    )


def _rss_bytes(pid: int) -> int:
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        text = (out.stdout or "").strip()
        if text:
            return int(text.split()[0]) * 1024
    except Exception:
        pass
    return 0


def _write_record(
    path: Path, handle: ResidentHandle, generation: int, *, ready: bool
) -> None:
    if generation < 1:
        raise RuntimeError("ownership generation is not monotone")
    rec = {
        "schema": _core.SCHEMA,
        "pid": int(handle.pid),
        "proc_start": str(handle.proc_start),
        "model_hash": str(handle.model_hash),
        "config_hash": str(handle.config_hash),
        "owner_generation": int(generation),
        "owner_pid": os.getpid(),
        "heartbeat_epoch": time.time(),
        "rss_bytes": _rss_bytes(handle.pid),
        "endpoint": str(handle.endpoint),
        "ready": bool(ready),
    }
    atomic_write_json(path, rec)


__all__ = [
    "ADOPTED",
    "SPAWNED",
    "REAPED_THEN_SPAWNED",
    "CompetingOwnerError",
    "ResidentHandle",
    "adopt_or_spawn",
    "prove_endpoint_capability",
    "release_ownership",
]
