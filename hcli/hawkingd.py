"""`hawkingd` -- the long-lived Hawking daemon.

`hcli` is the client; hawkingd is what keeps running after the client exits.
The split matters for more than tidiness: `ps` should describe the process, and
"python3 -m hcli.agentos.resident --supervise <state>" describes the
interpreter that happens to host it. The name is deliberately model-neutral. A
supervisor may hold several bodies, in parallel or one after another, so
naming the daemon for whichever model is loaded today would be wrong tomorrow.

This module exists rather than pointing the shim straight at
``hcli.agentos.resident`` because ``hcli.agentos.__init__`` already imports
that module, so ``-m hcli.agentos.resident`` re-executes an
already-imported module and Python warns about it on every single launch:

    RuntimeWarning: 'hcli.agentos.resident' found in sys.modules after import
    of package 'hcli.agentos', but prior to execution of ...

Ownership of a running daemon is decided by pid plus process start token
(``hcli.resources.process_start_token``), never by argv, so an incumbent
launched under the older name stays owned across this change.

The HTTP surface is also owned here.  ``hawkingd serve`` is the canonical
long-lived launch path; ``hcli serve`` remains a compatibility alias that
lands in the same implementation.  A process-wide advisory lease prevents a
second local body/server from being started on another port and loading a
second model into unified memory.  The model provider may still be a child
process (MLX-VLM currently is), but it is created and stopped by this daemon
and is never an independent Hawking launch surface.
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import time
import sys
from pathlib import Path
from typing import List, Optional

from hcli.agentos.resident import daemon_main


DEFAULT_DAEMON_LOCK = Path.home() / ".hcli" / "hawkingd.lock"


class DaemonAlreadyRunning(RuntimeError):
    """The machine-wide Hawking server lease is held by another process."""


class DaemonLease:
    """A crash-safe advisory lease for one long-lived Hawking surface.

    ``flock`` is the authority.  The JSON in the file is only human-readable
    evidence for the refusal message and status inspection; stale text cannot
    block a new owner after the old file descriptor disappears.
    """

    def __init__(self, handle, path: Path) -> None:
        self.handle = handle
        self.path = path
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self.handle.close()
            except OSError:
                pass

    def __enter__(self) -> "DaemonLease":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()


def acquire_daemon_lease(role: str, metadata: Optional[dict] = None) -> DaemonLease:
    """Claim the one local long-lived Hawking execution surface.

    The lock is deliberately outside the repository so a second checkout or
    a stale installed snapshot cannot start another GPU resident.  Tests and
    controlled isolated runs may override it with ``HCLI_DAEMON_LOCK``.
    """
    raw_path = os.environ.get("HCLI_DAEMON_LOCK")
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_DAEMON_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        try:
            handle.seek(0)
            incumbent = handle.read().strip()
        except OSError:
            incumbent = ""
        handle.close()
        detail = f"; incumbent={incumbent}" if incumbent else ""
        raise DaemonAlreadyRunning(
            f"hawkingd already owns the local execution surface at {path}{detail}"
        ) from exc

    payload = {
        "schema": "hawking.hawkingd_lease.v1",
        "role": str(role),
        "pid": os.getpid(),
        "started_at": time.time(),
        **(dict(metadata or {})),
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    handle.flush()
    os.fsync(handle.fileno())
    lease = DaemonLease(handle, path)
    atexit.register(lease.release)
    return lease


def daemon_lease_metadata() -> dict:
    """Read the incumbent hint for clients that want to reuse the live surface.

    The file is not ownership authority; the held ``flock`` is.  Callers must
    still health-check the advertised endpoint before using this hint.
    """
    raw_path = os.environ.get("HCLI_DAEMON_LOCK")
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_DAEMON_LOCK
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Keep one name in `ps` and one owner for the browser/OpenAI surface.  The
    # resident supervisor's existing --supervise/--worker protocol remains
    # untouched and continues to use its own durable state/ownership checks.
    if raw and raw[0] == "serve":
        from hcli.serve import main as serve_main

        return serve_main(raw[1:])
    return daemon_main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
