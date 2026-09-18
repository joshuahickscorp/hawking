"""An OpenAI-compatible endpoint over the PERSISTENT Hawking resident.

WHY THIS EXISTS. Open WebUI (already installed on this machine) speaks
`/v1/models` + `/v1/chat/completions` and nothing else. Hawking has two prior
OpenAI surfaces and neither can drive a browser chat:

  * `crates/hawking-serve` wants `model-*.gravity` shards. Every body on disk
    is .hq30uq4 / .f32v2 / .hgrafv01, so it has nothing to serve.
  * `tools/hawking_resident/serve_sealed.py` is correct and seal-verified, and its
    own docstring says the binary RELOADS THE MODEL PER CALL. That is a
    plumbing surface, not a chat one -- a ten-gigabyte load per message.

`make_backend_for_model` already returns a backend whose `complete(payload)`
takes an OpenAI-shaped dict, and for the native profile that backend holds a
HawkingNativeConnector wrapping a PERSISTENT ResidentProcess: one load, then
every turn is a pipe round trip. So this server is thin on purpose -- HTTP in,
`backend.complete` out. It adds no second runtime, no second model resolution
and no second state universe.

STREAMING IS SHAPE, NOT TIMING, AND IS LABELLED AS SUCH. The connector returns
a completed string; there is no incremental hook to subscribe to yet. With
`stream: true` this emits ONE content delta followed by `[DONE]` -- a valid SSE
conversation that Open WebUI renders correctly -- and stamps
`hawking.streaming: "single-chunk"` on the non-stream path plus `/health` so
nobody reads the SSE frames as evidence of token-by-token decode. Pretending
otherwise would be a false affordance, and those are bugs.

GREEDY PROFILES REFUSE SAMPLERS RATHER THAN IGNORING THEM. sealed-3.14 decodes
argmax. A request asking for temperature 0.8 is answered with a 400 that names
the accepted values, because silently serving a different sampler than the
caller asked for is how a benchmark ends up measuring something else. That rule
is `serve_sealed.py`'s and it is kept here.

SAMPLER REFUSAL IS ENFORCED BEFORE THE BACKEND IS TOUCHED. The refusal above is
only honest if it happens on the request path, not after a decode. `_reject_
unsupported_sampler` is therefore called at the top of the chat-completions
handler, before `backend.complete`, so a greedy profile answers 400 without
spending a resident round trip.

TUNNEL WIRE STATUS IS REPORTED, NOT INFERRED. P18_TUNNEL_WIRE_STATUS_V5 is the
observable answer to "is the resident tunnel actually carrying bytes right
now?" and it is derived from the same counters the request path already
maintains, so a caller never has to guess from a 200 alone. `_tunnel_wire_status`
returns a dict with `state` in {"idle", "active", "stalled"}, the cumulative
`requests` and `bytes_in`/`bytes_out` totals, and `last_activity_age_seconds`
measured against the monotonic clock. `state` is "stalled" when a request is in
flight and no byte has moved for longer than `TUNNEL_WIRE_STALL_SECONDS`, which
is the only condition under which a browser chat should be told the tunnel is
not making progress. The status is stamped on `/health` under
`hawking.tunnel_wire` so the regression test can name the keys rather than
restate them.

REQUEST BODIES ARE BOUNDED BEFORE THEY ARE PARSED. A chat surface that reads
`Content-Length` and hands the bytes straight to `json.loads` lets any caller
pin an unbounded amount of memory in the resident process, and a malformed
length is a crash rather than a 400. `_read_json_body` therefore enforces
`MAX_REQUEST_BODY_BYTES` (and a hard cap on the declared length) and answers
413/400 with a JSON error body instead of raising. The limit is a constant so
the regression test can name it rather than restate a magic number.

REQUEST BODIES ARE ALSO BOUNDED IN TIME. A declared length is a promise, not a
delivery: a caller that opens a socket, sends a valid `Content-Length` and then
stalls holds a resident thread forever, because `rfile.read(n)` blocks until n
bytes arrive or the peer closes. `_read_json_body` therefore reads the body in
bounded chunks under a wall-clock deadline (`REQUEST_BODY_READ_TIMEOUT_SECONDS`)
and answers 408 with a JSON error body when the deadline expires, so a stalled
peer is a bounded 408 rather than an unbounded resident thread.ll-clock deadline (`MAX_REQUEST_BODY_SECONDS`) and
answers 408 with a JSON error body when the deadline expires, so a slow or
half-open request costs one timeout instead of one thread. The deadline is a
constant for the same reason the byte cap is: the regression test names it
rather than restating a magic number.

REQUEST BODIES ARE ALSO BOUNDED IN SIZE ON THE WIRE, NOT JUST IN THE HEADER. A
`Content-Length` is a claim; a caller that omits it, or lies about it, must not
be able to stream an unbounded body into the resident. `_read_json_body`
therefore counts the bytes it actually reads and answers 413 with a JSON error
body the moment the running total exceeds `MAX_REQUEST_BODY_BYTES`, so the cap
is enforced against delivered bytes rather than against a declared number.

REQUEST BODIES ARE ALSO BOUNDED IN TIME ON THE WIRE, NOT JUST IN THE HEADER.
The wall-clock deadline above only starts once the first byte of the body
arrives; a caller that sends a complete, valid header block and then never
sends a body byte at all leaves `rfile.read` blocked on the first chunk with
the deadline never armed, so the resident thread is held forever. The header
read is therefore bounded by the same wall-clock budget: `_read_json_body`
arms `MAX_REQUEST_BODY_SECONDS` before it reads the request line and headers,
and answers 408 with a JSON error body when that budget expires before the
body read begins, so a header-only or half-open request costs one timeout
instead of one thread.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import nullcontext
import hashlib
import hmac
import re
import json
import os
import signal
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import uuid
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .cli import (
    DEFAULT_ENDPOINT_BASE_URL,
    DEFAULT_ENDPOINT_HOST,
    DEFAULT_ENDPOINT_MODEL,
    DEFAULT_ENDPOINT_PORT,
)
from .context_budget import resolve
from .resources import process_start_token
from .mutation import execute_mutation_proposal
from .auto_mode import AUTO_POLICY_REVISION, REMOTE_AUTO_ROSTER
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:  # the catalog import is deferred so a menu is never
    from .catalog import Body as CatalogBody  # built just to import this module

DEFAULT_PORT = DEFAULT_ENDPOINT_PORT
DEFAULT_HOST = DEFAULT_ENDPOINT_HOST
DEFAULT_BASE_URL = DEFAULT_ENDPOINT_BASE_URL
# The default must be an admitted Gravity body.  The former default pointed at
# the legacy sealed Qwen3.8/Ascension profile, which is archived and no longer
# belongs on the live operational surface.
# A role, not a provider or a checkpoint.  When no native role has earned its
# binding yet, hawkingd still owns the control/API port and returns a structured
# withholding rather than silently loading a retired vendor resident.
DEFAULT_MODEL = DEFAULT_ENDPOINT_MODEL
REVIEW_ONLY_MODEL_ID = "KIMI_P0_OPERATIONAL"
# The visible cloud roster is deliberately the small Hawking Auto pool.  The
# provider catalog remains searchable through Hawking, but the Web client does
# not mirror every provider entry into its primary selector.
WEB_AUTO_MODEL_IDS = REMOTE_AUTO_ROSTER
# The launcher uses the daemon's advertised endpoint contract to distinguish
# the current H-Web owner from an older healthy-but-stale process.  Keep this
# list beside the route owner so a new Web capability cannot be promoted while
# the daemon is still serving an older Python module.
HAWKING_WEB_ENDPOINTS = (
    "/hawking/review",
    "/hawking/web/release",
    "/hawking/web/permissions",
    "/hawking/web/events",
    "/hawking/web/events/ack",
    "/hawking/web/session",
    "/hawking/web/build/session",
    "/hawking/web/conversations",
    "/hawking/web/models",
    "/hawking/web/chat",
    "/hawking/web/tunnel",
    "/hawking/web/goal",
    "/hawking/web/ultragoal",
    "/hawking/web/attachments",
    "/hawking/web/artifacts",
    "/hawking/web/artifacts/config",
    "/hawking/web/memory/search",
    "/hawking/web/markdown",
)
WEB_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: Small on purpose. If window resolution ever fails again, the effect
#: is eager compaction that someone notices -- not a plausible constant
#: that silently replaces the measurement.
_WINDOW_UNKNOWN = 2048

# A request labelled ``read_only_research`` does not inherit every tool whose
# registry metadata happens to say READ_ONLY or RESEARCH. Some of those doors
# intentionally start work or write forensic/measurement state. Keep this
# small, explicit and useful for the delegated work Hawking currently admits:
# source navigation, receipt/history inspection, and public read-only research.
# Generic shells, tests, benchmarks, model actions,
# promotion and all canonical-state writers stay outside this request mode.
READ_ONLY_RESEARCH_TOOL_NAMES = frozenset({
    "fs.read", "filesystem.read", "fs.list", "filesystem.list",
    "fs.search", "filesystem.search",
    "receipt.read", "receipt.inspect", "roadmap.read", "roadmap.inspect",
    "campaign.state", "odyssey.priors",
    # Arbitrary web.fetch validates a host before urllib resolves it again;
    # keep that DNS-rebinding-sensitive transport outside constrained workers.
    # Search and the named public-source APIs below retain fixed providers.
    "web.search", "github.fetch", "github.search",
    "huggingface.resolve", "huggingface.manifest", "huggingface.fetch_file",
    "huggingface.history",
    "tests.list", "benchmark.inspect",
})

# H-Web main chat has a narrower but richer read-only projection. Keep this
# separate from the historical resident-worker fence: existing worker callers
# intentionally exclude runtime/catalog/context doors, while the operator
# surface needs them to answer Hawking questions from current canonical state.
HWEB_READ_ONLY_TOOL_NAMES = READ_ONLY_RESEARCH_TOOL_NAMES | frozenset({
    "source.owner", "source.outline", "source.symbol", "source.references",
    "git.status", "git.log", "git.diff",
    "processes.list", "processes.summary",
    "permissions.status", "context.recall", "tools.catalog",
    "workspace.identity", "runtime.status", "capabilities.status",
    "artifact.read", "artifact.search", "memory.search", "goals.list",
})

# This is deliberately a daemon contract revision rather than a browser asset
# revision.  A healthy old daemon can otherwise serve new H-Web JavaScript
# while silently retaining the pre-tool-projection chat executor.
HWEB_CAPABILITY_PROJECTION_REVISION = "hweb-main-chat-read-tools-v25-slot-recovery"
# A running daemon must advertise this independently of the read projection:
# the browser can otherwise receive a new launcher while its daemon still has
# no server-owned builder-session handoff or mutation fence.
HWEB_BUILDER_SESSION_REVISION = "hweb-builder-session-v9-repo-edit-projection"
# Auto policy is a daemon-side contract too: reusing a healthy old daemon
# after changing its cognition topology would silently leave the operator on
# stale conservative scheduling.
HAWKING_AUTO_ORCHESTRATION_REVISION = "hawking-auto-aggressive-v2-measured-diversity-unbounded-14"
# Build is intentionally finite per provider turn.  A normal interactive
# repair gets enough room for orientation -> red -> edit -> green -> status /
# diff; a durable Goal/WorkUnit gets the larger bounded lane.  These values
# are policy, not provider authority, and are surfaced in the receipt.
HWEB_BUILD_TOOL_BUDGET = 24
HWEB_BUILD_GOAL_TOOL_BUDGET = 32
# Request bodies are small control packets, never bulk artifacts. Keep the
# parser's allocation and a stalled peer's thread occupancy bounded before any
# route-specific code examines a request.
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BODY_SECONDS = 5.0


def _read_json_body(handler: BaseHTTPRequestHandler) -> Optional[Dict[str, Any]]:
    """Read one bounded JSON object, returning a response instead of raising.

    The deadline is armed before the first body read. A valid header that
    advertises bytes but never delivers them is therefore a normal 408 rather
    than a thread pinned forever in ``rfile.read``. Header ``readline`` calls
    are bounded by :meth:`HawkingOpenAIHandler.handle_one_request` using the
    same named timeout.
    """
    raw_length = handler.headers.get("Content-Length")
    try:
        length = int(raw_length or "0")
    except (TypeError, ValueError):
        handler._send_body_error(400, "invalid Content-Length")
        return None
    if length < 0:
        handler._send_body_error(400, "invalid Content-Length")
        return None
    if length > MAX_REQUEST_BODY_BYTES:
        handler._send_body_error(413, "request body exceeds the configured limit")
        return None

    deadline = time.monotonic() + MAX_REQUEST_BODY_SECONDS
    connection = getattr(handler, "connection", None)
    old_timeout: Optional[float] = None
    can_timeout = connection is not None and hasattr(connection, "settimeout")
    if can_timeout:
        try:
            old_timeout = connection.gettimeout()
        except OSError:
            can_timeout = False
    chunks: list[bytes] = []
    remaining = length
    try:
        while remaining:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                handler._send_body_error(408, "request body timed out")
                return None
            if can_timeout:
                connection.settimeout(remaining_seconds)
            try:
                chunk = handler.rfile.read(min(64 * 1024, remaining))
            except (socket.timeout, TimeoutError, OSError):
                handler._send_body_error(408, "request body timed out")
                return None
            if not chunk:
                handler._send_body_error(400, "request body ended before Content-Length")
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        if can_timeout:
            try:
                connection.settimeout(old_timeout)
            except OSError:
                pass
    try:
        body = json.loads(b"".join(chunks) or b"{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        handler._send_body_error(400, "request body must be valid JSON")
        return None
    if not isinstance(body, dict):
        handler._send_body_error(400, "request body must be a JSON object")
        return None
    return body


class HawkingOpenAIHandler(BaseHTTPRequestHandler):
    """Shared bounded request parser for the public HTTP surface."""

    protocol_version = "HTTP/1.1"

    def handle_one_request(self) -> None:
        """Apply the body deadline to request-line and header reads too."""
        connection = getattr(self, "connection", None)
        old_timeout: Optional[float] = None
        can_timeout = connection is not None and hasattr(connection, "settimeout")
        if can_timeout:
            try:
                old_timeout = connection.gettimeout()
                connection.settimeout(MAX_REQUEST_BODY_SECONDS)
            except OSError:
                can_timeout = False
        try:
            super().handle_one_request()
        except (socket.timeout, TimeoutError, OSError):
            try:
                self._send_body_error(408, "request headers timed out")
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True
        finally:
            if can_timeout:
                try:
                    connection.settimeout(old_timeout)
                except OSError:
                    pass

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_body_error(self, code: int, message: str) -> None:
        self.close_connection = True
        self._send(code, {"error": {"message": message, "type": "invalid_request"}})

    def _read_json_body(self) -> Optional[Dict[str, Any]]:
        return _read_json_body(self)

    def do_POST(self) -> None:  # noqa: N802 - testable safe default
        if self._read_json_body() is not None:
            self._send(404, {"error": {"message": "route not found"}})


# Retain the historical name used by socket-level regression probes without
# exporting a closure-bound production handler.
_Handler = HawkingOpenAIHandler


def _hweb_build_tool_budget(
    body: Mapping[str, Any], *, is_web_builder: bool = False,
    is_goal: bool = False,
) -> Tuple[int, str]:
    """Choose the bounded Build lane without creating a second scheduler."""
    if is_goal:
        return HWEB_BUILD_GOAL_TOOL_BUDGET, "goal_workunit"
    if is_web_builder:
        campaign = bool(
            str(body.get("hawking_goal_id") or "").strip()
            or str(body.get("hawking_workunit_id") or "").strip()
            or body.get("hawking_build_campaign") is True
        )
        return (
            HWEB_BUILD_GOAL_TOOL_BUDGET if campaign else HWEB_BUILD_TOOL_BUDGET,
            "builder_campaign" if campaign else "interactive_build",
        )
    try:
        value = int(os.environ.get("HAWKING_REMOTE_MAX_TOOL_CALLS", "16"))
    except (TypeError, ValueError):
        value = 16
    return max(0, min(32, value)), "ordinary_worker"


def _build_trace_digest(trace: Any) -> str:
    """Digest only source/mutation/test/Git evidence used by Build."""
    rows = []
    for row in trace if isinstance(trace, (list, tuple)) else []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("tool") or "") not in {
            "repo.edit", "tests.run", "git.status", "git.diff",
        }:
            continue
        compact = {
            key: row.get(key)
            for key in ("tool", "arguments", "dispatched", "ok", "result", "error")
            if key in row
        }
        encoded = json.dumps(compact, sort_keys=True, ensure_ascii=False, default=str)
        rows.append(encoded[:16000])
    payload = "\n".join(rows).encode("utf-8", "replace")
    return hashlib.sha256(payload).hexdigest()[:24]


def _build_tool_budget_state(
    ceiling: int, trace: Any, completion: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    used = sum(
        bool(row.get("dispatched"))
        for row in (trace if isinstance(trace, (list, tuple)) else [])
        if isinstance(row, Mapping)
    )
    complete = bool(isinstance(completion, Mapping) and completion.get("complete"))
    unmet = [
        str(item) for item in (completion.get("unmet") or [])
    ] if isinstance(completion, Mapping) else []
    return {
        "schema": "hawking.web.build-budget.v1",
        "state": "COMPLETE" if complete else "CHECKPOINT",
        "ceiling": max(0, int(ceiling)),
        "used": used,
        "remaining": max(0, int(ceiling) - used),
        "source_revision": _build_trace_digest(trace),
        "unmet": unmet[:12],
        "checkpoint_required": not complete,
    }


class WorkerRequestMode(str, Enum):
    """Authority and tool-loop policy for one bounded resident request.

    The model body is not the authority. A caller selects the narrowest mode
    that can answer its question, and the server removes mutation capability
    before the body receives a tool contract. This deliberately survives a
    future worker identity change: KIMI exercises the same contract Flash will
    inherit rather than acquiring a KIMI-specific permission path.
    """

    DEFAULT = "default"
    REASON_ONLY = "reason_only"
    READ_ONLY_RESEARCH = "read_only_research"
    BUILDER_INTERACTIVE = "builder_interactive"


class ResidentUnavailable(RuntimeError):
    """The loaded provider cannot serve without a forbidden recovery action."""


def worker_request_mode(raw: object) -> WorkerRequestMode:
    """Parse the opt-in Hawking worker policy without passing it to a provider."""
    if raw is None:
        return WorkerRequestMode.DEFAULT
    if not isinstance(raw, str):
        raise ValueError("hawking_worker_mode must be a string")
    normalized = raw.strip().lower()
    try:
        return WorkerRequestMode(normalized)
    except ValueError as exc:
        accepted = ", ".join(mode.value for mode in WorkerRequestMode)
        raise ValueError(
            f"unknown hawking_worker_mode {raw!r}; use one of: {accepted}"
        ) from exc


def worker_tool_allowlist(
        mode: WorkerRequestMode,
) -> Optional[frozenset[str]]:
    """Return the server-owned callable-tool fence for one request mode."""
    if mode is WorkerRequestMode.REASON_ONLY:
        return frozenset()
    if mode is WorkerRequestMode.READ_ONLY_RESEARCH:
        return READ_ONLY_RESEARCH_TOOL_NAMES
    if mode is WorkerRequestMode.BUILDER_INTERACTIVE:
        # This is an authority fence, not a model hint.  The builder doors are
        # separately injected below and remain outside the ordinary registry.
        return HWEB_READ_ONLY_TOOL_NAMES
    return None


class DaemonDelegationSupervisor:
    """Own bounded HAWKING delegation workers as children of ``hawkingd``.

    A delegated mission is allowed to outlive an ``hawking run`` client, but it
    must not become an orphaned Python root.  The daemon accepts only a
    prewritten delegation workspace under this checkout's durable mission
    root; it does not accept arbitrary argv, cwd, or shell text over HTTP.
    """

    def __init__(self, repo_root: str | os.PathLike[str]) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = (self.repo_root / ".hawking" / "delegations").resolve()
        self._lock = threading.Lock()
        self._children: Dict[str, subprocess.Popen[Any]] = {}

    def _resolve_workspace(self, raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("workspace must be a non-empty path")
        workspace = Path(raw).expanduser().resolve()
        try:
            workspace.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError(
                f"delegation workspace must be below {self.workspace_root}"
            ) from exc
        if not (workspace / ".hawking" / "mission" / "delegation_spec.json").is_file():
            raise ValueError("workspace has no delegation_spec.json")
        return workspace

    def _reap(self) -> None:
        self._children = {
            key: proc for key, proc in self._children.items()
            if proc.poll() is None
        }

    @staticmethod
    def _signal_child_group(
        proc: subprocess.Popen[Any],
        signum: int,
    ) -> None:
        """Signal a still-live executor's dedicated group, then its PID.

        The direct leader must still be live before Hawkingd uses its launch
        PID as a process-group ID.  Once it exits, that numeric group ID can
        eventually be recycled; the witnessed executor's SIGTERM hook owns
        descendant cleanup while that identity is unambiguous.
        """

        if proc.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signum)
                return
            except ProcessLookupError:
                return
            except (OSError, PermissionError):
                pass
        if proc.poll() is None:
            try:
                if signum == signal.SIGKILL:
                    proc.kill()
                else:
                    proc.terminate()
            except OSError:
                pass

    def start(self, raw_workspace: object) -> Dict[str, Any]:
        workspace = self._resolve_workspace(raw_workspace)
        key = str(workspace)
        with self._lock:
            self._reap()
            existing = self._children.get(key)
            if existing is not None:
                return {"started": False, "pid": existing.pid, "workspace": key,
                        "owner": "hawkingd"}
            # The hidden executor verb consumes this single-use opaque token
            # before it can run a mission.  It is passed only through the
            # daemon child's environment; a direct CLI invocation with merely
            # a workspace path is therefore not another executor surface.
            from .delegate import (
                DAEMON_EXEC_CAPABILITY_ENV,
                DaemonSupervisorWitness,
                issue_daemon_executor_capability,
                revoke_daemon_executor_capability,
            )

            # A dedicated child session preserves ordinary parent/child
            # ownership while giving this executor a distinct process group.
            # Bind this particular daemon incarnation into its one-use
            # capability so its watchdog can stop rather than letting a
            # recycled PID impersonate the old parent. No start token means
            # we cannot prove that ownership relation, so refuse the launch
            # before a capability or child exists.
            supervisor_start_token = process_start_token(os.getpid())
            if not supervisor_start_token:
                raise RuntimeError(
                    "cannot establish Hawkingd process-incarnation witness for delegated child"
                )
            revoke_daemon_executor_capability(workspace)
            capability = issue_daemon_executor_capability(
                workspace,
                supervisor=DaemonSupervisorWitness(
                    pid=os.getpid(),
                    start_token=supervisor_start_token,
                ),
            )
            child_env = dict(os.environ)
            child_env[DAEMON_EXEC_CAPABILITY_ENV] = capability
            log = workspace / ".hawking" / "mission" / "delegate_exec.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log.open("ab") as handle:
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "hawking", "__delegate_exec", key],
                        cwd=str(self.repo_root), stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=handle,
                        # A session does not change PPID: this remains a direct
                        # Hawkingd child and is reaped by it. It does give the
                        # watchdog one executor-only group to end without ever
                        # signalling the daemon's own group.
                        start_new_session=True,
                        env=child_env,
                    )
            except BaseException:
                revoke_daemon_executor_capability(workspace)
                raise
            self._children[key] = proc
            return {"started": True, "pid": proc.pid, "workspace": key,
                    "owner": "hawkingd"}

    def close(self) -> None:
        with self._lock:
            children = list(self._children.items())
            self._children.clear()
        for key, proc in children:
            self._signal_child_group(proc, signal.SIGTERM)
        deadline = time.monotonic() + 2.0
        for key, proc in children:
            remaining = max(0.0, deadline - time.monotonic())
            if proc.poll() is None:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
            # If the direct executor remains live after grace, its launch PID
            # is still a proven group identity and can be escalated safely.
            # If it already exited, do not signal a potentially recycled PGID:
            # the witnessed executor's SIGTERM hook was responsible for
            # atomically ending ordinary same-group descendants first.
            if proc.poll() is None:
                self._signal_child_group(proc, signal.SIGKILL)
            # Usually the child has already atomically consumed this file.
            # If it died before exec, clear its unused token so a future
            # daemon-supervised retry can get a fresh capability.
            try:
                from .delegate import revoke_daemon_executor_capability

                revoke_daemon_executor_capability(key)
            except Exception:
                pass

#: Values that mean "do not sample", per parameter. A greedy profile accepts
#: these and refuses everything else. `True == 1` in Python, so membership is
#: checked with an explicit type guard rather than a bare `in`.
GREEDY_OK: Dict[str, Tuple[Any, ...]] = {
    "temperature": (0, 0.0),
    "top_p": (1, 1.0),
    "top_k": (0, 1),
    "n": (1,),
    "presence_penalty": (0, 0.0),
    "frequency_penalty": (0, 0.0),
}


def _acceptable(key: str, value: Any) -> bool:
    if value is None:
        return True
    for ok in GREEDY_OK[key]:
        if type(value) is bool or type(ok) is bool:
            if type(value) is type(ok) and value == ok:
                return True
        elif value == ok:
            return True
    return False


def profile_is_greedy(model: str) -> bool:
    """True when the artifact's own profile says it does not sample.

    Read from the admitted profile rather than assuming sampler semantics.
    Retired source specimens never reach the serving path.
    """
    try:
        path = Path(os.path.expanduser(str(model)))
        if path.suffix != ".json" or not path.is_file():
            return False
        gen = json.loads(path.read_text(encoding="utf-8")).get("generation") or {}
    except Exception:
        return False
    if gen.get("do_sample") is False:
        return True
    return gen.get("temperature") in (0, 0.0) and gen.get("top_k") in (1,)


def sampler_refusal(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bad = sorted(k for k in GREEDY_OK if k in body and not _acceptable(k, body[k]))
    if not bad:
        return None
    accepted = ", ".join(f"{k}={GREEDY_OK[k][-1]!r}" for k in bad)
    return {"error": {
        "message": (
            f"this resident decodes GREEDY ARGMAX, so {bad} would ask for a sampler "
            f"it does not run. Refused rather than ignored, because silently serving "
            f"a different sampler than you asked for is how a measurement ends up "
            f"describing something else. Accepted here: {accepted} (or leave them "
            f"unset). In Open WebUI these live under Chat Controls > Advanced Params."
        ),
        "type": "unsupported_parameter",
        "param": bad[0],
    }}


def models_payload(identity: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": identity,
        "object": "model",
        # Open WebUI sorts and labels on these two; a missing `created` renders
        # the model as "Invalid Date" in the picker.
        "created": int(time.time()),
        "owned_by": "hawking",
    }
    if extra:
        row.update(extra)
    return {"object": "list", "data": [row]}


def _text_of(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text:
        return text
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


# These are tokenizer/template control markers, never user-visible answer
# content. Keep the list explicit: broad markup stripping would hide a model
# answer or a tool result and make an integration failure look repaired.
_VISIBLE_PROTOCOL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
)


def visible_text(text: Any) -> str:
    """Remove only known chat-template markers from a rendered answer."""
    if not isinstance(text, str):
        return ""
    for marker in _VISIBLE_PROTOCOL_TOKENS:
        text = text.replace(marker, "")
    return text


def trim_repetition_collapse(text: Any, *, min_cycles: int = 6,
                             max_period_tokens: int = 8):
    """Detect a greedy repetition collapse and trim it to the coherent prefix.

    Returns ``(clean_text, collapsed)``. Model-general: any body decoded
    greedily without a tuned anti-repetition kernel can fall into a short-cycle
    attractor and run to the token cap (measured on ascension-qwen38, 2026-09-08:
    a coherent answer then " (S) (S) (S) ..." for ~2000 tokens). A collapse is a
    short unit repeated many times consecutively at the TAIL; thresholds are
    conservative so ordinary prose -- even with a repeated word or phrase --
    never trips it.
    """
    if not isinstance(text, str) or len(text) < 80:
        return text, False
    toks = text.split(" ")
    n = len(toks)
    onset = None
    for period in range(1, max_period_tokens + 1):
        if n < period * min_cycles:
            continue
        cycle = toks[n - period:]
        index = n - period
        reps = 1
        while index - period >= 0 and toks[index - period:index] == cycle:
            reps += 1
            index -= period
        if reps >= min_cycles and (onset is None or index < onset):
            onset = index
    if onset is not None:
        kept = " ".join(toks[:onset]).rstrip()
        return (kept if kept else text), bool(kept and kept != text)
    # char-level fallback: a short substring repeated for a long trailing run --
    # a single-token loop with no spaces, e.g. "aaaa..." or "abababab...".
    for period in range(1, 9):
        if len(text) < period * 40:
            continue
        unit = text[-period:]
        cut = len(text)
        reps = 0
        while cut - period >= 0 and text[cut - period:cut] == unit:
            reps += 1
            cut -= period
        if reps >= 40:
            kept = text[:cut].rstrip()
            return (kept if kept else text), bool(kept and kept != text)
    return text, False


def _content_and_flags(result: Any):
    """Content, finish_reason and degraded flags for one completion, with the
    repetition-collapse guard applied. ONE place, so the stream and non-stream
    paths cannot disagree about what the model actually produced."""
    text = visible_text(_text_of(result))
    finish = getattr(result, "finish_reason", None) or "stop"
    degraded = list(getattr(result, "degraded", None) or [])
    trimmed, collapsed = trim_repetition_collapse(text)
    if collapsed:
        text = trimmed
        if "repetition_collapse" not in degraded:
            degraded.append("repetition_collapse")
    return text, finish, degraded


def _coalesce_system(messages: Any) -> list:
    """Exactly one system message, at position 0.

    Two independent seams each PREPEND a system message: the durable session
    working-set and repo-context injection. So a returning conversation arrives
    as [repo_system, durable_system, user, ...] -- and the sealed artifact
    template rejects any system message that is not at the beginning ("System
    message must be at the beginning"), which surfaced as an error completion
    (no usable response) on every second+ turn. Merge every system-role message
    into one block at the front, order preserved. Model-general: a single
    leading system block is what chat templates expect.
    """
    rows = [dict(m) for m in (messages or [])]
    systems = [m for m in rows if m.get("role") == "system"]
    if len(systems) <= 1:
        return rows
    blocks = [str(m.get("content") or "") for m in systems]
    merged = {"role": "system",
              "content": "\n\n".join(b for b in blocks if b.strip())}
    rest = [m for m in rows if m.get("role") != "system"]
    return [merged, *rest]


def chat_payload(result: Any, identity: str, *, request_id: str) -> Dict[str, Any]:
    text, finish, degraded = _content_and_flags(result)
    usage = {
        "prompt_tokens": getattr(result, "prompt_tokens", None) or 0,
        "completion_tokens": getattr(result, "completion_tokens", None) or 0,
        "total_tokens": getattr(result, "total_tokens", None) or 0,
    }
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": identity,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": usage,
        # Named so a reader of this response cannot mistake the SSE shape on the
        # other path for evidence of incremental decode.
        "hawking": {
            "streaming": "single-chunk",
            "degraded": degraded,
        },
    }
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("timings"), dict):
        # Provider timings are for the final cognition call. Tool-loop work is
        # accounted separately in hawking.tools_used and is intentionally not
        # collapsed into a misleading single-token rate.
        body["timings"] = dict(raw["timings"])
        body["hawking"]["timings_scope"] = "final_model_call_only"
    return body


def stream_frames(result: Any, identity: str, *, request_id: str):
    """SSE frames for one completed answer: role, one content delta, stop, DONE."""
    created = int(time.time())

    def frame(delta: Dict[str, Any], finish: Optional[str]) -> bytes:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": identity,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\n"

    yield frame({"role": "assistant"}, None)
    text, finish, _degraded = _content_and_flags(result)
    if text:
        yield frame({"content": text}, None)
    yield frame({}, finish)
    yield b"data: [DONE]\n\n"


def _gateway_label(value: Any, *, fallback: str = "unknown") -> str:
    """Keep caller labels useful in receipts without accepting free-form secrets."""
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:@/-]+", "-", text)
    return (text[:120] or fallback)


def _openrouter_model_id(
    model: Any,
    *,
    provider_hint: Any = None,
) -> Optional[str]:
    """Resolve an explicit or OpenAI-style model name to an OpenRouter id.

    Hawking selectors remain explicit.  A slash-bearing model is accepted as
    the generic OpenAI edge convention (``openai/gpt-*``), but an existing
    local path always stays on the native route.
    """
    value = str(model or "").strip()
    hint = str(provider_hint or "").strip().lower()
    lowered = value.lower()
    if lowered.startswith("openrouter:"):
        return value.split(":", 1)[1].strip() or None
    if hint in {"openrouter", "openrouter.ai"}:
        return value or None
    if hint in {"hawking", "hawking-native", "native", "local"}:
        return None
    if lowered.startswith(("hawking:", "native:")):
        return None
    if value and "/" in value:
        try:
            if Path(value).expanduser().exists():
                return None
        except OSError:
            pass
        if not value.startswith(("/", "./", "../")):
            return value
    return None


def _is_hawking_auto_model(value: Any) -> bool:
    """Recognize Hawking Auto before slash-bearing provider resolution."""
    from .auto_mode import is_auto_model

    return is_auto_model(value)


def _configured_openrouter_rows(policy: Any) -> List[Dict[str, Any]]:
    """Build non-network catalog rows from explicit Hawking configuration."""
    configured = list(getattr(policy, "allowed_models", ()) or ())
    configured.extend(
        item.strip()
        for item in str(os.environ.get("HAWKING_OPENROUTER_MODELS") or "").split(",")
        if item.strip()
    )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for model in configured:
        model_id = str(model).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        rows.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "hawking",
            "hawking": {
                "provider": "openrouter",
                "catalog_source": "hawking_configuration",
                "remote_model_id": model_id,
            },
        })
    return rows


def _review_only_device_model_rows() -> List[Dict[str, Any]]:
    """Expose the existing Kimi P0 identity without reviving its legacy runtime.

    Kimi P0 remains a frozen MLX artifact on disk, not a canonical hawkingd
    resident. It is useful as the one honest review identity in the client
    surface, but it must stay visibly withheld and must not become executable
    merely because Open WebUI can display it.
    """
    try:
        from .catalog import GRAVITY_REGISTRY, GRAVITY_STATUSES
        document = json.loads(Path(GRAVITY_REGISTRY).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    entries = document.get("artifacts") if isinstance(document, Mapping) else None
    if not isinstance(entries, list):
        return []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("id") or "").strip() != REVIEW_ONLY_MODEL_ID:
            continue
        if str(entry.get("kind") or "").strip() != "mlx":
            continue
        if str(entry.get("status") or "").strip() not in GRAVITY_STATUSES:
            continue
        raw_path = str(entry.get("path") or "").strip()
        model_path = Path(os.path.realpath(os.path.expanduser(raw_path)))
        if not (
            model_path.is_dir()
            and (model_path / "config.json").is_file()
            and (model_path / "model.safetensors.index.json").is_file()
        ):
            continue
        return [{
            "id": REVIEW_ONLY_MODEL_ID,
            "name": "Hawking",
            "description": (
                "Kimi P0 local review identity. The legacy MLX runtime is "
                "present on device but withheld from canonical hawkingd execution."
            ),
            "object": "model",
            "created": 0,
            "owned_by": "hawking",
            "hawking": {
                "model_class": "local-review",
                "source": "gravity",
                "backend": "mlx",
                "device_present": True,
                "availability": "WITHHELD",
                "review_only": True,
                "runtime": "legacy-mlx-withheld",
                "supported_actions": [],
            },
        }]
    return []


def _visible_openrouter_model_ids() -> set[str]:
    """Return the small default provider surface exposed to model pickers.

    Hawking can still search the live provider catalog explicitly, but the
    normal models endpoint is a control surface, not a 450-row provider
    mirror. An operator may deliberately replace the default surface with
    HAWKING_OPENROUTER_VISIBLE_MODELS; Auto's routing roster remains owned
    by auto_mode.REMOTE_AUTO_ROSTER regardless of that display preference.
    """
    configured = {
        item.strip()
        for item in str(
            os.environ.get("HAWKING_OPENROUTER_VISIBLE_MODELS") or ""
        ).split(",")
        if item.strip()
    }
    if configured:
        return configured
    return set()


def _catalog_search_blob(row: Mapping[str, Any]) -> str:
    """Build a bounded, provider-safe search document for one model row."""
    fields = (
        row.get("id"),
        row.get("name"),
        row.get("description"),
        row.get("canonical_slug"),
        row.get("owned_by"),
    )
    return " ".join(str(value) for value in fields if value).casefold()


def _select_openrouter_catalog_rows(
    configured_rows: Iterable[Mapping[str, Any]],
    catalog_rows: Iterable[Mapping[str, Any]],
    *,
    search: str = "",
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """Expose a bounded Hawking roster, or matching live rows on search."""
    configured = [dict(row) for row in configured_rows if isinstance(row, Mapping)]
    catalog = [dict(row) for row in catalog_rows if isinstance(row, Mapping)]
    term = str(search or "").strip().casefold()
    if term:
        candidates = [
            row for row in (configured + catalog)
            if term in _catalog_search_blob(row)
        ]
    else:
        visible = _visible_openrouter_model_ids()
        candidates = [
            row for row in (configured + catalog)
            if str(row.get("id") or "") in visible
        ]

    merged: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        model_id = str(row.get("id") or "").strip()
        if not model_id:
            continue
        # The live catalog row is later than the configuration placeholder, so
        # provider metadata wins without making configuration a second source
        # of truth for actual availability.
        if model_id not in merged or "hawking" in row:
            merged[model_id] = row
    rows = list(merged.values())
    return rows[:max(1, int(limit))] if term else rows


def _openrouter_catalog_rows(models: Any) -> List[Dict[str, Any]]:
    """Normalize provider catalog entries while retaining provider metadata."""
    rows: List[Dict[str, Any]] = []
    for item in models if isinstance(models, list) else []:
        if not isinstance(item, Mapping) or not item.get("id"):
            continue
        row = json.loads(json.dumps(dict(item), default=str))
        original_owned_by = row.get("owned_by")
        row.setdefault("object", "model")
        row.setdefault("created", int(time.time()))
        row["owned_by"] = "hawking"
        hawking = row.get("hawking")
        if not isinstance(hawking, dict):
            hawking = {}
        hawking.update({
            "provider": "openrouter",
            "catalog_source": "openrouter",
            "remote_model_id": str(row["id"]),
        })
        if original_owned_by is not None:
            hawking.setdefault("remote_owned_by", original_owned_by)
        row["hawking"] = hawking
        rows.append(row)
    return rows


def _web_model_label(model_id: Any, raw_name: Any = None) -> str:
    """Return a short human label without exposing provider choreography."""
    if str(model_id) == REVIEW_ONLY_MODEL_ID:
        return "Kimi P0 · present · withheld"
    name = str(raw_name or "").strip()
    if name:
        # OpenRouter commonly prefixes names with ``Vendor:``.  The model id
        # remains the machine identity; the UI only needs the readable name.
        if ":" in name:
            name = name.split(":", 1)[1].strip()
        return name[:120]
    known = {
        "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        "moonshotai/kimi-k3": "Kimi K3",
        "z-ai/glm-5.3": "GLM 5.3 · crossover trial",
        "qwen/qwen3.8-2.4t-a95b": "Qwen3.8 2.4T A95B · escalation",
        "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra · diversity",
        REVIEW_ONLY_MODEL_ID: "Kimi P0 · present · withheld",
    }
    return known.get(str(model_id), str(model_id))[:120]


def _web_model_scope(model_id: Any) -> str:
    value = str(model_id or "").strip()
    if value in {REVIEW_ONLY_MODEL_ID, "hawking/auto", "hawking-auto"}:
        return "local" if value == REVIEW_ONLY_MODEL_ID else "auto"
    return "cloud"


def _web_session_id(value: Any) -> str:
    """Validate the opaque Web session handle before it becomes a filename."""
    text = str(value or "").strip()
    if not text:
        return f"web-{uuid.uuid4().hex[:20]}"
    if not WEB_SESSION_ID_RE.fullmatch(text):
        raise ValueError("session_id contains unsupported characters")
    return text


def _goal_diff_scope(
    root: str | os.PathLike[str], goal: Mapping[str, Any]
) -> Tuple[Path, List[Tuple[str, Path]]]:
    """Resolve a Goal's focused files to one bounded Git working scope.

    H-Web Goals may target a linked worktree while their durable Goal record is
    stored in the repository root. Running ``git diff`` from that root would
    accidentally expose unrelated dirty main-checkout changes. Prefer an
    explicitly recorded worktree, then derive the conventional
    ``.worktrees/<name>/...`` scope from focused paths. Every returned file is
    proven to remain inside the selected scope before any Git command runs.
    """
    repo_root = Path(root).expanduser().resolve()
    raw_paths = [str(item).strip() for item in (goal.get("focus_files") or [])]
    raw_paths = [item for item in raw_paths if item]

    def inside(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
        except ValueError:
            return False
        return True

    scope = repo_root
    explicit = goal.get("worktree_path")
    if not explicit and isinstance(goal.get("worktree"), Mapping):
        explicit = goal["worktree"].get("root")
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        worktrees_root = (repo_root / ".worktrees").resolve()
        if inside(candidate, worktrees_root) and candidate.is_dir():
            scope = candidate

    if scope == repo_root:
        for raw in raw_paths:
            match = re.match(r"^\.worktrees/([^/]+)/(.+)$", raw)
            if not match:
                continue
            candidate = (repo_root / ".worktrees" / match.group(1)).resolve()
            if inside(candidate, (repo_root / ".worktrees").resolve()) and candidate.is_dir():
                scope = candidate
                break

    resolved: List[Tuple[str, Path]] = []
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            absolute = candidate.resolve()
            if not inside(absolute, scope):
                continue
            relative = absolute.relative_to(scope)
        else:
            parts = candidate.parts
            if scope != repo_root and len(parts) >= 3 and parts[0] == ".worktrees":
                if parts[1] != scope.name:
                    continue
                relative = Path(*parts[2:])
            else:
                relative = candidate
            absolute = (scope / relative).resolve()
            if not inside(absolute, scope):
                continue
        relative_text = relative.as_posix()
        if relative_text and all(item[0] != relative_text for item in resolved):
            resolved.append((relative_text, absolute))
    return scope, resolved


def _goal_diff_payload(
    root: str | os.PathLike[str], goal: Mapping[str, Any], goal_id: str
) -> Dict[str, Any]:
    """Return bounded tracked and untracked diff evidence for one Goal."""
    scope, focused = _goal_diff_scope(root, goal)
    rel_paths = [relative for relative, _absolute in focused]
    display_root = Path(root).expanduser().resolve()
    display_paths: List[str] = []
    for relative, absolute in focused:
        try:
            display_paths.append(absolute.resolve().relative_to(display_root).as_posix())
        except ValueError:
            display_paths.append(relative)

    def run_git(args: List[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(scope),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    scoped = ["--", *rel_paths] if rel_paths else []
    status = run_git(["status", "--short", *scoped])
    stat = run_git(["diff", "--stat", *scoped])
    diff = run_git(["diff", *scoped])
    stat_parts = [stat.stdout]
    diff_parts = [diff.stdout]
    returncode = diff.returncode

    # ``git diff`` intentionally omits untracked files. Include only the
    # Goal's focused untracked files via no-index so the UI can inspect the
    # actual harmless disposable write without leaking main-checkout state.
    for relative, absolute in focused:
        tracked = run_git(["ls-files", "--error-unmatch", "--", relative])
        if tracked.returncode == 0 or not absolute.is_file():
            continue
        no_index_stat = run_git(
            ["diff", "--no-index", "--stat", "--", os.devnull, relative]
        )
        no_index_diff = run_git(["diff", "--no-index", "--", os.devnull, relative])
        if no_index_stat.stdout:
            stat_parts.append(no_index_stat.stdout)
        if no_index_diff.stdout:
            diff_parts.append(no_index_diff.stdout)
        returncode = no_index_diff.returncode

    return {
        "goal_id": goal_id,
        "worktree_root": str(scope),
        "paths": display_paths,
        "status": status.stdout[-12000:],
        "stat": "".join(stat_parts)[-12000:],
        "diff": "".join(diff_parts)[-40000:],
        "returncode": returncode,
        "claim_boundary": "bounded local Git diff in the Goal worktree; not a commit or deployment",
    }


def _web_session_message_rows(session: Any) -> List[Dict[str, Any]]:
    """Return one de-duplicated durable message stream for H-Web."""
    rows: List[Dict[str, Any]] = []
    try:
        from .session import SessionStore
        rows.extend(SessionStore(str(session.workspace)).load_history(session.id))
    except Exception:
        pass
    rows.extend(list(getattr(session, "messages", []) or []))
    unique: Dict[int, Dict[str, Any]] = {}
    fallback: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("role") or "").strip().lower()
        content = row.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        try:
            seq = int(row.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0
        item = {"role": role, "content": content}
        if row.get("message_id"):
            item["message_id"] = str(row.get("message_id"))[:160]
        attachments = row.get("attachments")
        if isinstance(attachments, list) and attachments:
            item["attachments"] = [
                dict(value) for value in attachments[:8]
                if isinstance(value, Mapping) and value.get("artifact_id")
            ]
        if seq > 0:
            unique[seq] = item
        else:
            fallback.append(item)
    ordered = [unique[key] for key in sorted(unique)] + fallback
    return ordered[-64:]


def _web_attachment_projection(session: Any, attachments: Any) -> str:
    """Project attached markdown as explicitly marked user data for cognition."""
    if not isinstance(attachments, list) or not attachments:
        return ""
    try:
        from .web_artifacts import projection
        return projection(
            str(getattr(session, "workspace", "")),
            [item for item in attachments if isinstance(item, Mapping)],
        )
    except Exception as exc:
        # The durable message remains readable even if its artifact is
        # unavailable.  Do not silently turn a missing attachment into a
        # confident model answer.
        ids = [
            str(item.get("artifact_id") or "")[:96]
            for item in attachments
            if isinstance(item, Mapping) and item.get("artifact_id")
        ]
        return (
            "\n\n[HAWKING ATTACHMENT UNAVAILABLE — user data was not loaded; "
            f"artifact_ids={', '.join(ids) or 'unknown'}; reason={type(exc).__name__}]\n"
        )


def _web_memory_projection(
    session: Any,
    query: str,
    *,
    exclude_artifact_ids: Optional[set[str]] = None,
) -> str:
    """Project strong local H-NOTES matches into one provider turn.

    Retrieval is deliberately local and lexical for the base H-Web build. A
    weak/no match returns no prompt material, and every returned excerpt is
    marked as user-owned source data rather than instructions or authority.
    """
    if not str(query or "").strip():
        return ""
    try:
        from .web_artifacts import memory_projection
        return memory_projection(
            str(getattr(session, "workspace", "")),
            str(query),
            session_id=str(getattr(session, "id", "") or ""),
            exclude_artifact_ids=exclude_artifact_ids,
        )
    except Exception:
        # Retrieval is an optional context enhancement. Failing closed keeps
        # the provider request truthful and never exposes storage diagnostics.
        return ""


def _web_conversation_messages(session: Any) -> List[Dict[str, Any]]:
    """Project Hawking's durable hot/cold session into provider chat messages."""
    rows = _web_session_message_rows(session)
    projected: List[Dict[str, Any]] = []
    for row in rows:
        role = str(row.get("role") or "").strip().lower()
        content = row.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        projected.append({
            "role": role,
            "content": content + _web_attachment_projection(
                session,
                # Assistant-side markdown exports are durable result files,
                # not new prompt material on the next turn.  Only user
                # attachments are projected into provider context.
                row.get("attachments") if role == "user" else [],
            ),
        })
    return projected[-64:]


def _web_public_messages(session: Any) -> List[Dict[str, Any]]:
    """Return a bounded presentation view; internal sequence fields stay internal."""
    public: List[Dict[str, Any]] = []
    for row in _web_session_message_rows(session):
        role = str(row.get("role") or "").strip().lower()
        content = row.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        item: Dict[str, Any] = {"role": role, "content": content}
        if row.get("message_id"):
            item["message_id"] = str(row.get("message_id"))[:160]
        attachments = row.get("attachments")
        if isinstance(attachments, list) and attachments:
            item["attachments"] = [
                dict(value) for value in attachments[:8]
                if isinstance(value, Mapping) and value.get("artifact_id")
            ]
        public.append(item)
    return public[-64:]


def _remote_receipt(result: Any, provider: Any) -> Any:
    raw = getattr(result, "raw", None)
    if isinstance(raw, Mapping):
        for key in ("remote_cognition", "hawking_cognition"):
            if isinstance(raw.get(key), Mapping):
                return dict(raw[key])
    receipt = getattr(provider, "last_receipt", None)
    return receipt() if callable(receipt) else receipt


def _remote_tool_calls(result: Any) -> List[Dict[str, Any]]:
    raw = getattr(result, "raw", None)
    if not isinstance(raw, Mapping):
        return []
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return []
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        return [dict(call) for call in calls if isinstance(call, Mapping)]
    function = message.get("function_call")
    if isinstance(function, Mapping):
        return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "function": dict(function)}]
    return []


def _remote_usage(result: Any) -> Dict[str, Any]:
    usage = dict(getattr(result, "usage", None) or {})
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total = usage.get("total_tokens")
    if total is None:
        total = int(prompt or 0) + int(completion or 0)
    usage.update({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    })
    return usage


def _remote_chat_payload(
    result: Any,
    *,
    model: str,
    request_id: str,
    provider: Any,
    client_label: str,
    auto_route: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    text = visible_text(_text_of(result))
    raw = getattr(result, "raw", None)
    finish = getattr(result, "finish_reason", None) or "stop"
    if isinstance(raw, Mapping):
        choices = raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish = choices[0].get("finish_reason") or finish
    tool_calls = _remote_tool_calls(result)
    message: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    receipt = _remote_receipt(result, provider)
    usage = _remote_usage(result)
    remote_hawking: Dict[str, Any] = {
        "gateway": "hawkingd",
        "runtime": "remote",
        "provider": "openrouter",
        "client": client_label,
        "remote_cognition": receipt,
        "remote_request_id": usage.get("remote_request_id"),
    }
    if isinstance(auto_route, Mapping):
        remote_hawking["auto_route"] = dict(auto_route)
    if isinstance(raw, Mapping):
        if isinstance(raw.get("hawking_tool_trace"), list):
            projected_trace: List[Dict[str, Any]] = []
            for original in raw["hawking_tool_trace"]:
                if not isinstance(original, Mapping):
                    continue
                row = dict(original)
                result_value = row.get("result")
                if isinstance(result_value, Mapping):
                    result_value = result_value.get("value")
                if isinstance(result_value, Mapping):
                    tool_name = str(row.get("tool") or "")
                    row["observation"] = json.dumps(
                        result_value, sort_keys=True, default=str
                    )[:2000]
                    if tool_name == "repo.edit":
                        validation = result_value.get("validation")
                        nested_execution = result_value.get("execution")
                        if not isinstance(nested_execution, Mapping):
                            nested_execution = {}
                        row.update({
                            "verdict": result_value.get("status"),
                            "applied": result_value.get("applied"),
                            "rolled_back": result_value.get("rolled_back"),
                            "reason": (
                                result_value.get("reason")
                                or nested_execution.get("reason")
                            ),
                            "paths": result_value.get("paths"),
                            "bundle_complete": result_value.get("bundle_complete"),
                            "engine_test_validation": validation,
                            "engine_test_passed": bool(
                                isinstance(validation, Mapping)
                                and validation.get("ok") is True
                            ),
                            "engine_test_paths": result_value.get("tests") or [],
                            "bundle_operations": result_value.get("bundle_operations") or [],
                        })
                    elif tool_name == "tests.run":
                        row["verified"] = result_value.get("verified")
                        row["returncode"] = result_value.get("returncode")
                projected_trace.append(row)
            remote_hawking["tools_used"] = projected_trace
        if isinstance(raw.get("hawking_completion"), Mapping):
            remote_hawking["completion"] = dict(raw["hawking_completion"])
        if isinstance(raw.get("hawking_tool_admission"), Mapping):
            # Keep the non-secret admission facts visible to the owner and
            # H-Web. This is diagnostic provenance only; it never grants a
            # tool and never contains credentials or request headers.
            remote_hawking["tool_admission"] = dict(raw["hawking_tool_admission"])
        if isinstance(raw.get("hawking_proposal_bridge"), list):
            # A bounded non-secret statement of the proposal -> executor
            # handoff.  WorkUnit ownership uses this to distinguish a parser
            # miss from an unavailable executor or a rejected mutation.
            remote_hawking["proposal_bridge"] = [
                dict(item) for item in raw["hawking_proposal_bridge"]
                if isinstance(item, Mapping)
            ]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": usage,
        "hawking": remote_hawking,
    }


def _responses_messages(body: Mapping[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    source = body.get("input")
    if source is None:
        source = body.get("messages")
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
    elif isinstance(source, list):
        pending_text: List[str] = []
        for item in source:
            if isinstance(item, Mapping) and item.get("role"):
                messages.append(dict(item))
                continue
            if isinstance(item, Mapping):
                item_type = str(item.get("type") or "").lower()
                if item_type in {"input_text", "text"} and isinstance(item.get("text"), str):
                    pending_text.append(str(item["text"]))
                    continue
                if item_type in {"message", "input_message"}:
                    role = str(item.get("role") or "user")
                    content = item.get("content")
                    if content is not None:
                        messages.append({"role": role, "content": content})
                        continue
            if isinstance(item, str):
                pending_text.append(item)
        if pending_text:
            messages.append({"role": "user", "content": "\n".join(pending_text)})
    if not messages:
        raise ValueError("responses input is required")
    return messages


def _responses_request_body(body: Mapping[str, Any], model: str) -> Dict[str, Any]:
    text = body.get("text")
    response_format = body.get("response_format")
    if response_format is None and isinstance(text, Mapping):
        fmt = text.get("format")
        if isinstance(fmt, Mapping):
            if fmt.get("type") == "json_schema" and fmt.get("schema") is not None:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.get("name") or "response",
                        "schema": fmt.get("schema"),
                        "strict": fmt.get("strict", True),
                    },
                }
            else:
                response_format = dict(fmt)
    request: Dict[str, Any] = {
        "model": model,
        "messages": _responses_messages(body),
    }
    for key in (
        "temperature", "top_p", "tools", "tool_choice", "parallel_tool_calls",
        "provider", "stream_options", "reasoning", "user", "service_tier",
        "modalities", "audio", "transforms", "models",
    ):
        if key in body:
            request[key] = body[key]
    if body.get("max_output_tokens") is not None:
        request["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        request["max_tokens"] = body["max_tokens"]
    if response_format is not None:
        request["response_format"] = response_format
    return request


def _remote_response_payload(
    result: Any,
    *,
    model: str,
    response_id: str,
    provider: Any,
    client_label: str,
    auto_route: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    text = visible_text(_text_of(result))
    usage = _remote_usage(result)
    output: List[Dict[str, Any]] = [{
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }]
    for call in _remote_tool_calls(result):
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        output.append({
            "type": "function_call",
            "id": str(call.get("id") or f"fc_{uuid.uuid4().hex[:12]}"),
            "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "name": str(function.get("name") or ""),
            "arguments": str(function.get("arguments") or "{}"),
            "status": "completed",
        })
    receipt = _remote_receipt(result, provider)
    hawking: Dict[str, Any] = {
        "gateway": "hawkingd",
        "runtime": "remote",
        "provider": "openrouter",
        "client": client_label,
        "remote_cognition": receipt,
    }
    if isinstance(auto_route, Mapping):
        hawking["auto_route"] = dict(auto_route)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "hawking": hawking,
    }


class Resident:
    """The one loaded body, and the ability to become a different one.

    Open WebUI sends `model` on every request and reads `/v1/models` for its
    dropdown, so honouring both is all switching needs -- no reconnection, no
    second endpoint, no restart. That is why this lives behind the OpenAI shape
    rather than beside it.

    ONE BODY AT A TIME, AND THE OLD ONE STOPS FIRST. These are 8-150 GB
    artifacts on a 103 GB machine; spawning the new resident before stopping the
    old one would page the box into the ground and violate the campaign's swap
    ceiling. So a switch is stop-then-start, serialised under a lock, and a
    request that arrives mid-switch waits rather than racing a half-loaded body.

    A BODY THAT CANNOT FIT IS REFUSED, NOT ATTEMPTED. Qwen2.5-72B is 145 GB and
    this machine has 103 GB. Starting it would thrash for many minutes and then
    fail; the refusal names the two numbers instead.
    """

    def __init__(self, body: "CatalogBody", backend: Any, *, resolved_action: Any = None,
                 ready_timeout: float = 900.0, write: bool = False,
                 provider_owned: bool = False, backend_factory: Any = None):
        if resolved_action is not None:
            from .catalog import ResolvedAction

            if not isinstance(resolved_action, ResolvedAction):
                raise TypeError("Resident requires a ResolvedAction value")
            if resolved_action.action not in {"serve", "web"}:
                raise PermissionError(
                    f"resolved action {resolved_action.action!r} cannot own a resident"
                )
        if write:
            from .catalog import require_write_session_authority

            require_write_session_authority(body)
        self.body = body
        self.backend = backend
        self.resolved_action = resolved_action
        self.ready_timeout = ready_timeout
        self.write_authority = bool(write)
        self.state = "ready"
        self.error: Optional[str] = None
        self._lock = threading.RLock()
        # A Popen-shaped object is not proof that Hawking owns it.  Only the
        # daemon's load path may opt an initial backend into recovery; every
        # later backend is marked here after this Resident successfully built
        # it.  Matching both object identities prevents an observed foreign
        # adapter from being stopped/restarted merely because it happens to
        # expose spawn/ready/stop/poll methods.
        self._owned_backend: Optional[Any] = backend if provider_owned else None
        self._owned_process: Optional[Any] = (
            getattr(backend, "process", None) if provider_owned else None
        )
        # The canonical front door supplies a native-only factory.  Legacy
        # compatibility callers retain the historical generic factory, but a
        # role-routed resident must keep that native fence during switches and
        # restoration as well as its initial load.
        self._backend_factory = backend_factory
        # Tool capability belongs to the BODY, not to the server. Qualifying
        # once at startup left a 0.6B's verdict standing after a switch to a
        # different body -- the new model inheriting an old model's unverified
        # action capability, which is the precise failure S034 s15 names.
        self.tools_verdict: Dict[str, Any] = {"qualified": False,
                                              "reason": "not yet qualified"}

    @property
    def identity(self) -> str:
        return self.body.name

    @property
    def greedy(self) -> bool:
        return profile_is_greedy(self.body.path)

    @property
    def native_tools(self) -> bool:
        """Does this body's own artifact template declare tools?

        The sealed artifact's chat_template.jinja has a `tools` slot and states
        the call format the body was trained on. When it does, the template is
        the authority on the tool contract and a hand-written system message is
        both redundant and worse -- it is what made the body invent `shell`.
        """
        return self.body.kind == "noetic_native"

    def catalog(self) -> List[Dict[str, Any]]:
        from .catalog import catalog as _catalog
        return [b.to_openai(loaded=(b.name == self.body.name)) for b in _catalog()]

    def admits(self, body: "CatalogBody") -> Optional[str]:
        if not body.bytes:
            return None
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None
        if body.bytes > total * 0.92:
            return (f"{body.name} is {body.bytes / 1e9:.0f} GB and this machine has "
                    f"{total / 1e9:.0f} GB. Loading it would page the box into swap "
                    f"rather than run, so it is refused here. Bodies that fit are "
                    f"listed by `hawking use`.")
        return None

    def _owned_provider_process_exited(self) -> bool:
        """Whether a daemon-owned local provider has already exited.

        A ``Resident`` may remain alive longer than a provider child.
        Treat only a directly-owned ``Popen``-like child as restartable here:
        remote and native backends are deliberately not inferred dead from a
        missing process attribute.
        """
        process = getattr(self.backend, "process", None)
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return False
        try:
            return poll() is not None
        except Exception:
            # Health observation must not turn an unfamiliar backend into a
            # restart target. Its normal completion path remains authoritative.
            return False

    def _can_restart_owned_provider(self) -> bool:
        """True only for a local child whose lifecycle this Resident owns."""
        process = getattr(self.backend, "process", None)
        return (
            self.backend is self._owned_backend
            and process is self._owned_process
            and callable(getattr(process, "poll", None))
            and callable(getattr(self.backend, "spawn", None))
            and callable(getattr(self.backend, "stop", None))
            and callable(getattr(self.backend, "ready", None))
        )

    def _recover_exited_owned_provider(self) -> bool:
        """Serially recreate a dead provider under this same ``hawkingd``.

        This is intentionally a single same-body switch: no standalone
        fallback, no second model, and no unbounded retry loop.
        """
        if (not self._can_restart_owned_provider()
                or not self._owned_provider_process_exited()):
            return False
        self.switch(
            self.resolved_action if self.resolved_action is not None else self.body.name,
            _force_reload=True,
        )
        return True

    @staticmethod
    def _stop_for_switch(backend: Any, *, role: str) -> Dict[str, Any]:
        """Stop one provider and require an explicit exit proof.

        Switching is a memory-safety boundary, not best-effort cleanup. Every
        production RuntimeBackend reports ``{"gone": bool}``; accepting an
        exception, ``None``, or an ambiguous report here could spawn a second
        multi-gigabyte provider while the first still owns memory.
        """
        try:
            report = backend.stop()
        except Exception as exc:
            raise RuntimeError(
                f"{role} stop failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(report, dict) or report.get("gone") is not True:
            raise RuntimeError(
                f"{role} stop did not prove exit: {report!r}")
        return report

    def switch(self, selection: Any, *, research: bool = False,
               _force_reload: bool = False) -> Dict[str, Any]:
        """Serially replace the loaded body.

        ``research`` is intentionally an admission door, not a catalog mode:
        research bodies may be selected by a daemon-local WorkUnit but never
        appear in the normal OpenAI/Open WebUI selector.  The old provider is
        always stopped before a new body is spawned because this machine has
        already demonstrated that concurrent resident loads are destructive.
        """
        from .catalog import ResolvedAction, resolve, resolve_action, revalidate_action
        with self._lock:
            # Keep the normal resolver call shape stable for existing callers
            # and lightweight test doubles.  Research is the only path that
            # needs the expanded ModelLake catalog.
            if research:
                target = resolve(str(selection), research=True)
                target_action = None
            elif isinstance(selection, ResolvedAction):
                if selection.action not in {"serve", "web"}:
                    raise PermissionError(
                        f"resolved action {selection.action!r} cannot control a resident surface"
                    )
                target_action = revalidate_action(selection)
                target = target_action.body
            else:
                target_action = resolve_action(str(selection), "serve")
                target = target_action.body if target_action is not None else None
            if target is None:
                raise LookupError(
                    f"no {'research ' if research else ''}body named {selection!r}. "
                    "`hawking use` lists normal admitted bodies.")
            if self.write_authority:
                from .catalog import require_write_session_authority

                require_write_session_authority(target)
            live_same_resident = (
                self.state == "ready"
                and not _force_reload
                and not (
                    self._can_restart_owned_provider()
                    and self._owned_provider_process_exited()
                )
            )
            if (
                not research
                and self.resolved_action is not None
                and target_action is not None
                and target_action.has_same_binding(self.resolved_action)
                and live_same_resident
            ):
                # A verified Web action may take over an already loaded serve
                # action when the artifact/grant binding is identical.  The
                # action remains in the published contract, while the binding
                # deliberately describes the physical resident that need not
                # be reloaded for this transport-only transition.
                self.body = target
                self.resolved_action = target_action
                return {
                    "switched": False,
                    "resident": self.body.name,
                    "resolved_action": target_action.to_wire(),
                }
            if research and target.name == self.body.name and live_same_resident:
                return {"switched": False, "resident": self.body.name}
            refusal = self.admits(target)
            if refusal:
                raise MemoryError(refusal)
            factory = self._backend_factory
            if factory is None:
                from .runtime_iface import make_backend_for_model
                factory = make_backend_for_model
            previous_body = self.body
            previous = previous_body.name
            previous_resolved_action = self.resolved_action
            previous_tools_verdict = dict(self.tools_verdict)
            candidate = None
            previous_state = self.state
            # Do not construct a candidate unless the old provider's exit is
            # proven. This boundary is deliberately outside the restoration
            # path below: an old provider that may still be alive must never
            # trigger a "restore" load alongside itself.
            try:
                self._stop_for_switch(self.backend, role="current provider")
            except RuntimeError as stop_exc:
                self.state = previous_state
                self.error = str(stop_exc)
                raise
            self.state = "switching"
            try:
                candidate = factory(target.path)
                candidate.spawn()
                if not candidate.ready(self.ready_timeout):
                    raise RuntimeError(
                        f"{target.name} did not become ready within "
                        f"{self.ready_timeout:.0f}s")
                self.backend = candidate
                self.body = target
                self.resolved_action = target_action
                self._owned_backend = candidate
                self._owned_process = getattr(candidate, "process", None)
                self.state = "ready"
                self.error = None
                # A research-only ModelLake admission earns only an execution
                # path.  Running the normal resident tool battery here can
                # force a large specimen through an unrelated conversational
                # gate before an activation capture can start, while holding
                # the resident switch lock.  Its result is intentionally
                # unqualified until a later explicit body-specific gate.
                if research:
                    self.tools_verdict = {
                        "qualified": False,
                        "reason": "research body: tool contract not qualified",
                    }
                else:
                    self.qualify_tools()
                result = {"switched": True, "from": previous,
                        "resident": target.name, "research": bool(research),
                        "tools": dict(self.tools_verdict)}
                if target_action is not None:
                    result["resolved_action"] = target_action.to_wire()
                return result
            except Exception as exc:
                # A failed candidate must be stopped before a previous-body
                # restoration is even considered.  Starting restoration while
                # an unknown candidate remains alive violates the one-body
                # memory invariant.  Anything short of an explicit
                # `gone=True` is a failed cleanup; do not reinterpret it as a
                # harmless status.
                candidate_cleanup_error = None
                if candidate is not None:
                    try:
                        self._stop_for_switch(candidate, role="candidate provider")
                    except Exception as stop_exc:
                        candidate_cleanup_error = str(stop_exc)
                if candidate_cleanup_error is not None:
                    self.state = "failed"
                    self._owned_backend = None
                    self._owned_process = None
                    self.error = (
                        f"{type(exc).__name__}: {exc}; "
                        f"{candidate_cleanup_error}; previous body not restored"
                    )
                    raise
                # A failed research/runtime admission must not strand the
                # normal Resident after we deliberately serialized the load.
                # Restore through the same daemon-owned factory; never launch
                # a second standalone provider as a recovery shortcut.
                restored = None
                try:
                    restored = factory(previous_body.path)
                    restored.spawn()
                    if restored.ready(self.ready_timeout):
                        self.backend = restored
                        self.body = previous_body
                        self.resolved_action = previous_resolved_action
                        self._owned_backend = restored
                        self._owned_process = getattr(restored, "process", None)
                        self.tools_verdict = previous_tools_verdict
                        self.state = "ready"
                        self.error = (
                            f"{type(exc).__name__}: {exc}; restored {previous}")
                    else:
                        raise RuntimeError("previous body did not become ready")
                except Exception as restore_exc:
                    restoration_cleanup_error = None
                    if restored is not None:
                        try:
                            self._stop_for_switch(
                                restored, role="restoration provider")
                        except Exception as cleanup_exc:
                            restoration_cleanup_error = str(cleanup_exc)
                    self.state = "failed"
                    self._owned_backend = None
                    self._owned_process = None
                    self.error = (
                        f"{type(exc).__name__}: {exc}; restoration failed: "
                        f"{type(restore_exc).__name__}: {restore_exc}"
                        + (f"; {restoration_cleanup_error}"
                           if restoration_cleanup_error else ""))
                raise

    def qualify_tools(self, prefix: Any = None, registry: Any = None,
                      write: Optional[bool] = None) -> Dict[str, Any]:
        """Can THIS body emit a typed action, under its real conditions?

        Same contract the traffic uses: a native body is probed with its own
        template `tools` slot populated, because that is what it will see.
        """
        from .chat_tools import openai_schemas, qualify, system_block
        self.qualify_prefix = prefix if prefix is not None else getattr(
            self, "qualify_prefix", None)
        self.qualify_registry = registry if registry is not None else getattr(
            self, "qualify_registry", None)
        if write is not None:
            if write:
                from .catalog import require_write_session_authority

                require_write_session_authority(self.body)
            self.write_authority = bool(write)
        schemas = (openai_schemas(self.qualify_registry,
                                  write=self.write_authority)
                   if self.native_tools and self.qualify_registry else None)
        prefix_messages = list(self.qualify_prefix or [])
        if not self.native_tools and self.qualify_registry is not None:
            from .chat_tools import prepend_system
            prefix_messages = prepend_system(
                prefix_messages, system_block(registry=self.qualify_registry,
                                              write=self.write_authority))

        def _complete(convo):
            payload: Dict[str, Any] = {"messages": convo, "max_tokens": 96}
            if schemas:
                payload["tools"] = schemas
            return _text_of(self.backend.complete(payload))

        self.tools_verdict = qualify(
            _complete, prefix=prefix_messages, registry=self.qualify_registry,
            write=self.write_authority)
        return self.tools_verdict

    def complete(self, payload: Dict[str, Any], timeout: Optional[float] = None,
                 *, model_lock: Optional[str] = None,
                 allow_owned_recovery: bool = True) -> Any:
        wanted = str(payload.get("model") or "").strip()
        with self._lock:
            if model_lock is not None:
                locked = str(model_lock).strip()
                if not locked or locked != self.body.name:
                    raise PermissionError(
                        "restricted worker request may only use the resident "
                        "that was loaded when its request began")
                if wanted and wanted != locked:
                    raise PermissionError(
                        "restricted worker request may not select another resident")
            if self.resolved_action is not None:
                # An OpenAI request that omits `model` still runs the loaded
                # action. Re-resolve it on every normal completion so a
                # revoked or changed binding cannot keep answering merely
                # because its human-facing name is unchanged.
                from .catalog import resolve_action
                try:
                    target_action = resolve_action(
                        wanted or self.body.name, self.resolved_action.action
                    )
                except (LookupError, PermissionError, ValueError):
                    target_action = None
                if target_action is None:
                    raise LookupError(
                        f"no body named {wanted or self.body.name!r} -- the loaded resident is "
                        f"{self.body.name}. `hawking use` lists what is available.")
                if (not allow_owned_recovery
                        and not target_action.has_same_binding(self.resolved_action)):
                    raise PermissionError(
                        "constrained worker request cannot reload a changed resident binding")
                if not allow_owned_recovery:
                    # Revalidation is still mandatory, but a constrained
                    # worker must not enter ``switch`` even for a same binding:
                    # its later liveness check owns the one permitted live
                    # observation before the backend call.
                    self.body = target_action.body
                    self.resolved_action = target_action
                else:
                    # A same binding is a no-op load; a normal request may
                    # strictly stop-then-start for a changed binding.
                    self.switch(target_action)
            elif wanted and wanted != self.body.name:
                # Research/legacy residents have no normal admitted action
                # record. Preserve explicit selection without presenting it as
                # a normal catalog authorization path.
                from .catalog import resolve_action
                try:
                    target_action = resolve_action(wanted, "serve")
                except (LookupError, PermissionError, ValueError):
                    target_action = None
                if target_action is None:
                    raise LookupError(
                        f"no body named {wanted!r} -- the loaded resident is "
                        f"{self.body.name}. `hawking use` lists what is available.")
                if not allow_owned_recovery:
                    raise PermissionError(
                        "constrained worker request cannot reload another resident")
                self.switch(target_action)
            if (not allow_owned_recovery
                    and self._can_restart_owned_provider()
                    and self._owned_provider_process_exited()):
                raise ResidentUnavailable(
                    "resident provider exited; this constrained worker request "
                    "cannot start recovery")
            # A constrained request is allowed to observe the current
            # resident, never to revive it. Do not merely pre-check liveness:
            # a child can exit between two polls, so calling the recovery
            # helper at all would leave a start race in the supposedly
            # fail-closed path.
            forced_reload_used = (
                self._recover_exited_owned_provider()
                if allow_owned_recovery else False
            )
            # The `model` field is OURS -- a catalog name that selects which body
            # is loaded. It is an edge/catalog selector, not an instruction to
            # a backend to download a model. The admitted backend serves the
            # body selected by Hawking's catalog and needs no provider lookup.
            inner = {k: v for k, v in payload.items() if k != "model"}

            def invoke() -> Any:
                def _complete_backend() -> Any:
                    return self.backend.complete(inner, timeout) if timeout is not None \
                        else self.backend.complete(inner)

                if allow_owned_recovery:
                    return _complete_backend()
                # Resident-level recovery is not enough for the native
                # connector: its JSONL transport historically restarted an
                # owned child after a broken pipe, and one-shot mode launches
                # a process on every completion. Constrained workers get an
                # already-live resident only; the connector fence makes that
                # true below this adapter boundary as well.
                from .hawking_native import (
                    HawkingNativeUnavailable,
                    suppress_native_runtime_spawning,
                )
                try:
                    with suppress_native_runtime_spawning():
                        return _complete_backend()
                except HawkingNativeUnavailable as exc:
                    raise ResidentUnavailable(str(exc)) from exc

            try:
                return invoke()
            except urllib.error.URLError:
                # A local child can be alive while its loopback listener has
                # already failed. Rebuild it once through the daemon-owned
                # switch path, then let the retry's actual error remain truth.
                if (not allow_owned_recovery or forced_reload_used
                        or not self._can_restart_owned_provider()):
                    raise
                self.switch(
                    self.resolved_action if self.resolved_action is not None else self.body.name,
                    _force_reload=True,
                )
                return invoke()


def make_handler(backend: Any, identity: str, *, greedy: bool,
                 health: Dict[str, Any], repo: Any = None,
                 registry: Any = None, stores: Optional[Dict[str, Any]] = None,
                 webui_manager: Any = None,
                 delegation_supervisor: Any = None,
                 endpoint_base: Optional[str] = None,
                 canonical_router: Any = None):
    stores = stores or {}
    gateway_workspace = Path(
        stores.get("state_root")
        or (getattr(repo, "root", None) if repo is not None else None)
        or os.getcwd()
    ).resolve()
    gateway_state: Dict[str, Any] = {
        "policy": None,
        "remote_admission": None,
        "last_remote_receipt": None,
        "remote_catalog_ids": set(),
    }
    gateway_state_lock = threading.Lock()
    # All remote Goal mutations share one Hawking-side admission lock.  The
    # model may be ephemeral, but the repository transaction is not; this is
    # the same single-writer boundary used by the existing local Goal path.
    remote_mutation_lock = threading.Lock()
    # H-Web uses the existing SessionStore/Session owner for transcript and
    # routing continuity.  The handles below are lazy so unit-test handlers
    # that never open H-Web do not create state in the checkout.
    web_session_store: Any = None
    web_session_lock = threading.RLock()
    # H-Web assets have a local current/candidate/previous release identity.
    # This is deliberately a static-asset owner only; it cannot create another
    # HTTP server or execute a second backend writer.
    from .web_release import WebReleaseStore
    web_releases = WebReleaseStore(
        gateway_workspace,
        Path(__file__).resolve().parents[1] / "artifacts" / "hawking-review.html",
    )

    def _web_store() -> Any:
        nonlocal web_session_store
        if web_session_store is None:
            from .session import SessionStore
            web_session_store = SessionStore(str(gateway_workspace))
        return web_session_store

    def _web_enabled_models() -> set[str]:
        """Read the Hawking-owned cloud selector state, never provider state."""
        from .config import Config

        value = Config(str(gateway_workspace)).value(
            "hawking_web_enabled_models",
            "HAWKING_WEB_ENABLED_MODELS",
            None,
        )
        if value is None:
            return set(WEB_AUTO_MODEL_IDS)
        if isinstance(value, (list, tuple, set)):
            raw = value
        else:
            raw = str(value).split(",")
        return {
            str(item).strip()
            for item in raw
            if str(item).strip() and "/" in str(item).strip()
        }

    def _web_save_enabled_models(models: set[str]) -> set[str]:
        from .config import Config

        config = Config(str(gateway_workspace))
        data = config.load()
        data["hawking_web_enabled_models"] = sorted(
            item for item in models if "/" in str(item)
        )
        config.save_project(data)
        return set(data["hawking_web_enabled_models"])

    def _web_model_rows(search: str = "") -> List[Dict[str, Any]]:
        """Build the small H-Web catalog; live provider metadata is opt-in."""
        term = str(search or "").strip()[:120]
        enabled = _web_enabled_models()
        known = set(WEB_AUTO_MODEL_IDS) | enabled
        rows: List[Dict[str, Any]] = []
        if term:
            # Search is the only path that reaches the provider catalog.  The
            # browser still calls Hawking, and Hawking returns only bounded
            # matches with pricing/capability metadata attached.
            try:
                from .remote_cognition import cached_openrouter_catalog
                catalog = _openrouter_catalog_rows(
                    cached_openrouter_catalog(gateway_policy())
                )
            except Exception:
                catalog = []
            rows.extend(
                row for row in catalog
                if term.casefold() in _catalog_search_blob(row).casefold()
            )
            # Keep configured roster entries discoverable if the provider
            # catalog is temporarily unavailable.
            rows.extend(
                {"id": item, "object": "model", "owned_by": "hawking"}
                for item in sorted(known)
                if term.casefold() in item.casefold()
            )
        else:
            rows.extend(
                {"id": item, "object": "model", "owned_by": "hawking"}
                for item in sorted(known)
            )
        local_rows = _review_only_device_model_rows()
        if not term or any(
                term.casefold() in _catalog_search_blob(row).casefold()
                for row in local_rows):
            rows.extend(local_rows)

        merged: Dict[str, Dict[str, Any]] = {}
        for original in rows:
            if not isinstance(original, Mapping) or not original.get("id"):
                continue
            model_id = str(original["id"])
            row = dict(original)
            row["name"] = _web_model_label(model_id, row.get("name"))
            scope = _web_model_scope(model_id)
            row["hawking_web"] = {
                "scope": scope,
                "enabled": model_id in enabled,
                "selectable": scope != "local"
                or not bool((row.get("hawking") or {}).get("review_only")),
            }
            if scope == "cloud":
                row.setdefault("hawking", {})
                if isinstance(row["hawking"], dict):
                    row["hawking"].setdefault("provider", "openrouter")
                    row["hawking"].setdefault("remote_model_id", model_id)
            # Live metadata wins over a configured placeholder.
            if model_id not in merged or "pricing" in row:
                merged[model_id] = row
        return list(merged.values())[:40]

    def _web_session(raw_id: Any = None, *, create: bool = True) -> Any:
        from .session import Session

        sid = _web_session_id(raw_id)
        with web_session_lock:
            session = _web_store().load(sid)
            if session is None and create:
                session = Session(session_id=sid, model="hawking/auto")
                session.ui = {
                    "surface": "hawking-web",
                    "route_scope": "auto",
                    "auto_scope": "cloud",
                    "active_goal_id": "",
                    "goal_ids": [],
                }
                _web_store().save(session)
            if session is not None:
                # SessionStore keeps the durable transcript in the existing
                # Hawking workspace.  The owner is attached in memory only so
                # the shared hot/cold projection can load archived turns;
                # it is intentionally not serialized as a second path.
                session.workspace = str(gateway_workspace)
                if not isinstance(session.ui, dict):
                    session.ui = {}
                changed = False
                if session.ui.get("surface") != "hawking-web":
                    session.ui["surface"] = "hawking-web"
                    changed = True
                if session.ui.get("route_scope") == "auto":
                    # Auto is a model choice inside a source page now.  Migrate
                    # the old top-level Auto page to its existing safe default
                    # (Cloud) while still accepting the legacy value on input.
                    session.ui["route_scope"] = "cloud"
                    changed = True
                elif session.ui.get("route_scope") not in {"local", "cloud"}:
                    session.ui["route_scope"] = "local" if _web_model_scope(session.model) == "local" else "cloud"
                    changed = True
                if session.ui.get("auto_scope") not in {"cloud", "local", "both"}:
                    session.ui["auto_scope"] = "cloud"
                    changed = True
                goal_ids = _web_goal_ids(session.ui)
                if session.ui.get("goal_ids") != goal_ids:
                    session.ui["goal_ids"] = goal_ids
                    changed = True
                if "active_goal_id" not in session.ui:
                    session.ui["active_goal_id"] = ""
                    changed = True
                if changed:
                    _web_store().save(session)
            return session

    def _mint_web_builder_session(workspace: Path) -> Dict[str, str]:
        """Create one short-lived, local browser handoff for BUILD authority.

        The token is deliberately not the authority itself: only a hash is
        durable, and redemption turns it into an HttpOnly local cookie.  The
        browser cannot manufacture this Session.ui state through its ordinary
        JSON APIs.
        """
        # A local shell can enter the same macOS volume through a differently
        # cased component (`downloads` versus `Downloads`) or a symlinked
        # checkout alias.  String equality turned that legitimate invocation
        # into a false workspace-authority rejection.  Compare filesystem
        # identity instead; a missing or genuinely different path still fails
        # closed.
        try:
            same_workspace = workspace.samefile(gateway_workspace)
        except OSError:
            same_workspace = False
        if not same_workspace:
            raise ValueError("builder workspace does not match this Hawking daemon")
        session = _web_session(None)
        token = secrets.token_urlsafe(24)
        now = int(time.time())
        if not isinstance(session.ui, dict):
            session.ui = {}
        session.ui["authority_profile"] = "build"
        session.ui["build_handoff"] = {
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "expires_at": now + 300,
            "consumed": False,
            "issued_at": now,
        }
        _web_store().save(session)
        return {"session_id": str(session.id), "token": token}

    def _delete_web_session(raw_id: Any) -> Dict[str, Any]:
        """Delete one Web-owned chat record while preserving detached Goals."""
        sid = _web_session_id(raw_id)
        with web_session_lock:
            store = _web_store()
            session = store.load(sid)
            if session is None:
                raise ValueError("Hawking Web conversation was not found")
            ui = getattr(session, "ui", {}) or {}
            if not (
                sid.startswith("web-")
                or (isinstance(ui, Mapping) and ui.get("surface") == "hawking-web")
            ):
                raise ValueError("only Hawking Web conversations can be deleted here")
            record = Path(store.dir) / f"{sid}.json"
            history = store.history_path(sid)
            for path in (record, history):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise ValueError(f"conversation could not be deleted: {type(exc).__name__}") from exc
            return {
                "schema": "hawking.web.conversation.delete.v1",
                "session_id": sid,
                "deleted": True,
                "preserved_goal_ids": _web_goal_ids(ui),
            }

    def _web_attachment_refs(values: Any) -> List[Dict[str, Any]]:
        """Resolve browser-provided IDs through the canonical artifact owner."""
        from .web_artifacts import normalize_refs

        return normalize_refs(gateway_workspace, values)

    def _web_goal_ids(ui: Any) -> List[str]:
        """Return durable Goal handles carried by one Hawking Web session.

        These are references into the canonical Goal/WorkUnit owner, not a
        second Goal store.  A bounded, de-duplicated list keeps a long-lived
        browser session portable without allowing arbitrary UI state to grow
        without limit.
        """
        value = ui if isinstance(ui, Mapping) else {}
        raw = value.get("goal_ids")
        candidates: List[Any] = list(raw) if isinstance(raw, list) else []
        active = value.get("active_goal_id")
        if active:
            candidates.append(active)
        result: List[str] = []
        for item in candidates:
            goal_id = str(item or "").strip()
            if not goal_id or not re.fullmatch(r"GOAL-[A-Za-z0-9_-]{1,100}", goal_id):
                continue
            if goal_id not in result:
                result.append(goal_id)
        return result[-32:]

    def _web_goal_snapshot(goal_id: Any) -> Optional[Dict[str, Any]]:
        goal_id = str(goal_id or "").strip()
        if not goal_id:
            return None
        try:
            from .goal_surface import load_goal, load_result
            from .workunit_owner import WorkunitOwner
            goal = load_goal(str(gateway_workspace), goal_id)
            if not goal:
                return {"goal_id": goal_id, "error": "goal_not_found"}
            owner = WorkunitOwner(gateway_workspace)
            ids = []
            for item in (
                *(goal.get("workunit_ids") or [] if isinstance(goal.get("workunit_ids"), list) else []),
                goal.get("workunit_id"), goal.get("active_workunit_id"),
            ):
                wid = str(item or "").strip()
                if wid and wid not in ids:
                    ids.append(wid)
            if not ids:
                ids = [goal_id]
            workunits = []
            for wid in ids[-32:]:
                try:
                    row = owner.status(wid)
                except Exception as exc:
                    row = {"workunit_id": wid, "state": "UNKNOWN", "error": type(exc).__name__}
                packet_path = (
                    Path(gateway_workspace) / "receipts" / "future" / "workunits"
                    / f"{wid}_RESULT_PACKET.json"
                )
                if packet_path.is_file():
                    try:
                        packet = json.loads(packet_path.read_text(encoding="utf-8"))
                        if isinstance(packet, Mapping):
                            row["result_packet"] = {
                                "path": str(packet_path),
                                "status": packet.get("status"),
                                "summary": packet.get("summary"),
                                "files_changed": packet.get("files_changed") or [],
                                "tests": packet.get("tests") or {},
                                "evidence": list(packet.get("evidence") or [])[-12:],
                            }
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        pass
                workunits.append(row)
            active_id = str(goal.get("active_workunit_id") or "").strip()
            workunit = next(
                (row for row in workunits if str(row.get("workunit_id") or "") == active_id),
                workunits[-1],
            )
            return {
                "goal": goal,
                "workunit": workunit,
                "workunits": workunits,
                "active_workunit_id": active_id or str(workunit.get("workunit_id") or ""),
                "result": load_result(str(gateway_workspace), goal_id),
            }
        except Exception as exc:
            return {"goal_id": goal_id, "error": type(exc).__name__}

    def _web_session_snapshot(session: Any) -> Dict[str, Any]:
        if session is None:
            return {"session_id": None, "messages": []}
        ui = dict(getattr(session, "ui", {}) or {})
        goal_ids = _web_goal_ids(ui)
        goal_id = str(ui.get("active_goal_id") or "").strip()
        goals = [snapshot for item in goal_ids
                 if (snapshot := _web_goal_snapshot(item)) is not None]
        active_goal = next(
            (item for item in goals
             if str((item.get("goal") or {}).get("goal_id") or item.get("goal_id") or "") == goal_id),
            None,
        )
        compatibility = {
            "schema": "hawking.web.session.v1",
            "session_id": session.id,
            "model": session.model or "hawking/auto",
            "route_scope": ui.get("route_scope") or _web_model_scope(session.model),
            "auto_scope": ui.get("auto_scope") or "cloud",
            "messages": _web_public_messages(session),
            "turns": len(_web_conversation_messages(session)),
            "active_goal_id": goal_id or None,
            "active_goal": active_goal,
            "goal_ids": goal_ids,
            "goals": goals,
            "ui": ui,
        }
        # The existing fields remain for the deployed minimal client, while
        # the typed projection is now the live contract.  Both are computed
        # from the durable Session/Goal/Workunit owners above; there is no Web
        # mirror to reconcile after a refresh.
        from .web_projection import project_snapshot
        contract = project_snapshot(gateway_workspace, session, goals)
        return {
            **compatibility,
            "contract": contract,
            "contract_schema": contract["schema"],
            "contract_version": contract["version"],
            "state_revision": contract["state_revision"],
        }

    def _web_event_replay(session: Any, after_sequence: int, limit: int) -> Dict[str, Any]:
        """Project only canonical durable state into a reconnectable event feed."""
        from .web_projection import WebEventJournal

        snapshot = _web_session_snapshot(session)
        contract = snapshot["contract"]
        journal = WebEventJournal(gateway_workspace, str(session.id))
        journal.sync(contract)
        replay = journal.replay(after_sequence, limit)
        return {
            "schema": contract["schema"],
            "session_id": session.id,
            "state_revision": contract["state_revision"],
            "replay": replay,
            # A full projection is present only for an explicit gap recovery;
            # browser ACKs are client cursors, not Hawking state.
            "snapshot": contract if replay["snapshot_required"] else None,
        }

    def _web_record_assistant(body: Mapping[str, Any], text: str,
                              metadata: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        raw_id = body.get("hawking_web_session_id")
        if not raw_id:
            return None
        with web_session_lock:
            session = _web_session(raw_id)
            if session is None:
                return None
            session.append_message("assistant", visible_text(text), kind="conversation")
            if body.get("model"):
                session.model = str(body.get("model"))[:160]
            if not isinstance(session.ui, dict):
                session.ui = {}
            if body.get("hawking_web_route_scope"):
                session.ui["route_scope"] = str(body.get("hawking_web_route_scope"))[:20]
            if body.get("hawking_auto_scope") in {"cloud", "local", "both"}:
                session.ui["auto_scope"] = str(body.get("hawking_auto_scope"))
            if isinstance(metadata, Mapping):
                route = metadata.get("auto_route")
                if isinstance(route, Mapping):
                    session.ui["last_auto_route"] = {
                        key: str(route.get(key))[:160]
                        for key in ("selected_model", "policy", "reason")
                        if route.get(key) is not None
                    }
                    if isinstance(route.get("cognition_plan"), Mapping):
                        session.ui["last_cognition_plan"] = dict(route["cognition_plan"])
                if metadata.get("remote_cognition") is not None:
                    session.ui["last_remote"] = {
                        "provider": "openrouter",
                        "model": str(session.model or "")[:160],
                    }
            _web_store().save(session)
            return {
                "id": session.id,
                "turns": len(_web_conversation_messages(session)),
                "model": session.model,
                "route_scope": session.ui.get("route_scope"),
                "assistant_seq": int(session.messages[-1].get("seq") or 0)
                if session.messages and isinstance(session.messages[-1], Mapping)
                and session.messages[-1].get("role") == "assistant" else None,
            }

    def _web_attach_to_last_assistant(session_id: Any,
                                      attachment: Mapping[str, Any]) -> int:
        """Attach one Hawking-owned output artifact to the latest assistant turn."""
        sid = _web_session_id(session_id)
        with web_session_lock:
            session = _web_session(sid, create=False)
            if session is None:
                raise ValueError("Hawking Web session was not found")
            for row in reversed(session.messages):
                if not isinstance(row, dict) or row.get("role") != "assistant":
                    continue
                rows = row.setdefault("attachments", [])
                if not isinstance(rows, list):
                    rows = []
                    row["attachments"] = rows
                artifact_id = str(attachment.get("artifact_id") or "")
                if artifact_id and not any(
                    isinstance(item, Mapping)
                    and item.get("artifact_id") == artifact_id
                    for item in rows
                ):
                    rows.append(dict(attachment))
                    row["attachments"] = rows[-8:]
                _web_store().save(session)
                return int(row.get("seq") or 0)
        raise ValueError("Hawking Web has no assistant turn to export")

    def _web_record_goal(session_id: Any, goal_id: Any) -> None:
        if not session_id or not goal_id:
            return
        with web_session_lock:
            session = _web_session(session_id)
            if session is None:
                return
            if not isinstance(session.ui, dict):
                session.ui = {}
            goal_id = str(goal_id).strip()
            goal_ids = _web_goal_ids(session.ui)
            if goal_id and goal_id not in goal_ids:
                goal_ids.append(goal_id)
            session.ui["goal_ids"] = goal_ids[-32:]
            session.ui["active_goal_id"] = goal_id[:100]
            _web_store().save(session)

    def gateway_policy() -> Any:
        # Configuration is read once per server, while credentials and the
        # catalog remain lazy inside the provider.  This gives one spend and
        # concurrency boundary to every generic client on this listener.
        with gateway_state_lock:
            if gateway_state["policy"] is None:
                from .remote_cognition import OpenRouterGatewayPolicy
                from .auto_orchestration import RemoteWorkerAdmission
                gateway_state["policy"] = OpenRouterGatewayPolicy.from_environment(
                    gateway_workspace
                )
                gateway_state["remote_admission"] = RemoteWorkerAdmission(
                    max_workers=gateway_state["policy"].concurrency,
                    # Premium-model count is not a second hidden ceiling. An
                    # explicit finite OpenRouter concurrency policy, when
                    # configured, is the single operator gate for the route.
                    max_premium_workers=None,
                )
            return gateway_state["policy"]

    def gateway_admission() -> Any:
        gateway_policy()
        return gateway_state["remote_admission"]

    def acquire_remote_lease(body: Mapping[str, Any], model: str) -> Any:
        """Admit one real gateway request through Hawking's worker gate.

        The provider only sees the sanitized GenerationRequest.  Worker and
        Goal identity remain Hawking-side admission metadata, so a shared
        OpenRouter credential never becomes worker identity or authority.
        """
        from .auto_orchestration import WorkerAdmissionError
        from .remote_cognition import RemoteCognitionError

        try:
            return gateway_admission().acquire(
                model=model,
                goal_id=str(body.get("hawking_goal_id") or "").strip(),
                workunit_id=str(body.get("hawking_workunit_id") or "").strip(),
                worker_attempt_id=str(body.get("hawking_worker_attempt_id") or "").strip(),
                worker_id=str(body.get("hawking_worker_id") or "").strip() or None,
            )
        except WorkerAdmissionError as exc:
            raise RemoteCognitionError(
                "REMOTE_CONCURRENCY_LIMIT",
                str(exc),
                provider="openrouter",
                model_id=model,
                recoverable=True,
            ) from exc

    def remember_remote_receipt(provider: Any) -> None:
        """Retain one sanitized provider receipt for Hawking's control surface."""
        receipt = getattr(provider, "last_receipt", None)
        receipt = receipt() if callable(receipt) else receipt
        if isinstance(receipt, Mapping):
            with gateway_state_lock:
                gateway_state["last_remote_receipt"] = dict(receipt)

    class Handler(HawkingOpenAIHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The bounded WorkUnit owner may have timed out while the
                # upstream route was recovering.  There is no client left to
                # receive an error; do not turn that normal cancellation edge
                # into a daemon traceback or a second misleading 502.
                self.close_connection = True
                return

        def _send_bytes(
            self, code: int, body: bytes, content_type: str,
            *, download_name: Optional[str] = None,
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if download_name:
                encoded_name = urllib.parse.quote(
                    str(download_name), safe="._-"
                )
                self.send_header(
                    "Content-Disposition",
                    f"attachment; filename*=UTF-8''{encoded_name}",
                )
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_OPTIONS(self) -> None:  # noqa: N802 - CORS preflight from a browser
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _identity(self) -> str:
            return getattr(backend, "identity", None) or identity

        def _local_only(self) -> bool:
            return self.client_address[0] in {"127.0.0.1", "::1"}

        def _cookie_value(self, name: str) -> str:
            for row in (self.headers.get("Cookie") or "").split(";"):
                key, separator, value = row.strip().partition("=")
                if separator and key == name:
                    return value
            return ""

        def _trusted_builder_session(self) -> Any:
            """Return the build session selected by an HttpOnly handoff cookie.

            A persisted session id alone is intentionally insufficient.  This
            prevents a copied transcript id or share URL from acquiring the
            mutation ceiling owned by a local `h build` browser handoff.
            """
            raw = self._cookie_value("hawking_build_session")
            session_id, separator, proof = raw.partition(".")
            if not separator or not WEB_SESSION_ID_RE.fullmatch(session_id) or not proof:
                return None
            session = _web_session(session_id, create=False)
            if session is None or not isinstance(getattr(session, "ui", None), dict):
                return None
            handoff = session.ui.get("build_handoff")
            if (not isinstance(handoff, Mapping)
                    or handoff.get("revoked_at") is not None
                    or int(handoff.get("expires_at") or 0) < int(time.time())):
                return None
            expected = str(handoff.get("browser_proof_sha256") or "")
            if not expected or not hmac.compare_digest(
                    expected, hashlib.sha256(proof.encode("utf-8")).hexdigest()):
                return None
            return session

        def _send_build_handoff(self, session_id: str, token: str) -> None:
            """Redeem a single-use local handoff and establish the browser binding."""
            if not self._local_only():
                return self._send(403, {"error": {"message": "builder handoff is local-only"}})
            session = _web_session(session_id, create=False)
            handoff = (getattr(session, "ui", {}) or {}).get("build_handoff") if session else None
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if not isinstance(handoff, Mapping) or bool(handoff.get("consumed")) or int(handoff.get("expires_at") or 0) < int(time.time()):
                return self._send(410, {"error": {"message": "builder handoff expired or already used"}})
            if not hmac.compare_digest(str(handoff.get("token_sha256") or ""), token_hash):
                return self._send(403, {"error": {"message": "builder handoff denied"}})
            proof = secrets.token_urlsafe(24)
            session.ui["build_handoff"] = {
                **dict(handoff),
                "consumed": True,
                "browser_proof_sha256": hashlib.sha256(proof.encode("utf-8")).hexdigest(),
                "bound_at": int(time.time()),
            }
            _web_store().save(session)
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie",
                f"hawking_build_session={session_id}.{proof}; Path=/hawking/web; HttpOnly; SameSite=Strict",
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json_body(self) -> Optional[Dict[str, Any]]:
            if hasattr(self, "_hawking_json_body"):
                return self._hawking_json_body
            body = self._read_json_body()
            self._hawking_json_body = body
            return body

        def _gateway_access_allowed(self) -> bool:
            """Keep the remote-capable edge local unless explicitly exposed."""
            if self._local_only():
                return True
            expected = os.environ.get("HAWKING_ACCESS_TOKEN")
            supplied = self.headers.get("Authorization") or ""
            if not expected or not supplied.lower().startswith("bearer "):
                return False
            return hmac.compare_digest(supplied[7:].strip(), expected)

        def _client_label(self) -> str:
            return _gateway_label(
                self.headers.get("X-Hawking-Client")
                or self.headers.get("X-Client-Name"),
                fallback="openai-client",
            )

        def _remote_auto_route(self, body: Mapping[str, Any]) -> Dict[str, Any]:
            """Select Auto from the live roster under the existing gateway policy."""
            from .auto_mode import AutoRouteUnavailable, choose_remote_route
            from .remote_cognition import RemoteCognitionError

            policy = gateway_policy()
            with gateway_state_lock:
                catalog_ids = set(gateway_state.get("remote_catalog_ids") or ())
            try:
                route = choose_remote_route(
                    body,
                    allowed_model_ids=policy.allowed_models,
                    # A catalog request is the strongest availability proof. If
                    # no client has asked for it yet, the exact qualified roster
                    # remains the bounded fallback and provider admission still
                    # supplies the final policy check.
                    available_model_ids=catalog_ids or None,
                )
            except AutoRouteUnavailable as exc:
                raise RemoteCognitionError(
                    "AUTO_UNAVAILABLE",
                    str(exc),
                    provider="openrouter",
                    model_id="hawking-auto",
                    recoverable=False,
                ) from exc
            payload = route.to_dict()
            payload["scope"] = str(body.get("hawking_auto_scope") or "cloud").strip().lower()
            return payload

        def _remote_error(self, exc: BaseException) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
            from .remote_cognition import (
                RemoteAuthError,
                RemoteCancelledError,
                RemoteCognitionError,
                RemoteCostError,
                RemoteRateLimitError,
            )
            headers: Dict[str, str] = {}
            if isinstance(exc, RemoteRateLimitError):
                if exc.retry_after_s is not None:
                    headers["Retry-After"] = str(max(0, int(exc.retry_after_s)))
                return 429, headers, {"type": "rate_limit_error", "code": exc.code}
            if isinstance(exc, RemoteAuthError):
                return 503, headers, {"type": "provider_auth_error", "code": exc.code}
            if isinstance(exc, RemoteCostError):
                return 402, headers, {"type": "cost_policy_error", "code": exc.code}
            if isinstance(exc, RemoteCancelledError):
                return 499, headers, {"type": "request_cancelled", "code": exc.code}
            if isinstance(exc, RemoteCognitionError):
                if exc.code in {"MODEL_NOT_ALLOWED", "REQUEST_TOKEN_LIMIT_EXCEEDED"}:
                    return 403, headers, {"type": "policy_error", "code": exc.code}
                if exc.code == "AUTO_UNAVAILABLE":
                    return 503, headers, {"type": "auto_unavailable", "code": exc.code}
                if exc.code in {"MODEL_REQUIRED", "MODEL_ID_MISMATCH"}:
                    return 400, headers, {"type": "invalid_request_error", "code": exc.code}
                if "TIMEOUT" in exc.code:
                    return 504, headers, {"type": "upstream_timeout", "code": exc.code}
                return 502, headers, {"type": "provider_error", "code": exc.code}
            if isinstance(exc, (ValueError, TypeError)):
                return 400, headers, {"type": "invalid_request_error"}
            return 502, headers, {"type": "gateway_error"}

        def _send_remote_error(self, exc: BaseException) -> None:
            status, headers, details = self._remote_error(exc)
            from .remote_cognition import RemoteCognitionError
            message = str(exc)[:1200]
            if isinstance(exc, RemoteCognitionError):
                message = str(exc)
            body = {"error": {"message": message, **details}}
            encoded = json.dumps(body).encode()
            try:
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(encoded)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True
                return

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path.rstrip("/") or "/"
            # Hawking control belongs under /api.  Keep the old /v1/goal
            # aliases readable while existing local browser tabs migrate, but
            # never advertise them as part of the canonical OpenAI surface.
            if path.startswith("/api/goal"):
                path = "/v1" + path[len("/api"):]
            if path == "/api/status":
                path = "/health"
            if path == "/hawking/web/read":
                # `h web` intentionally drops any older builder binding in
                # this browser.  A command spelling is an explicit authority
                # choice; a stale local cookie must not keep write doors open.
                # Revoke the server-side binding too: retaining a copied
                # browser cookie must not defeat an explicit downgrade.
                bound_builder = self._trusted_builder_session()
                if bound_builder is not None:
                    handoff = bound_builder.ui.get("build_handoff")
                    if isinstance(handoff, Mapping):
                        bound_builder.ui["build_handoff"] = {
                            **dict(handoff),
                            "revoked_at": int(time.time()),
                        }
                    bound_builder.ui["authority_profile"] = "read"
                    _web_store().save(bound_builder)
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header(
                    "Set-Cookie",
                    "hawking_build_session=; Path=/hawking/web; HttpOnly; SameSite=Strict; Max-Age=0",
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            handoff = re.fullmatch(
                r"/hawking/web/build/([A-Za-z0-9_-]{1,64})/([A-Za-z0-9_-]{16,128})",
                path,
            )
            if handoff:
                return self._send_build_handoff(handoff.group(1), handoff.group(2))
            if path == "/api/runtime/roles":
                if canonical_router is None:
                    return self._send(404, {"error": {
                        "message": "canonical runtime routing is unavailable"}})
                return self._send(200, canonical_router.status())
            review_host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
            if path == "/hawking/review" or (
                path == "/" and review_host in {"hawking.web", "hawking.localhost"}
            ):
                # The review surface is a Hawking-owned static client. Keep it
                # outside the installed Open WebUI package so the artifact can
                # be reviewed through the same local daemon without changing
                # the inherited client or its persisted data.  The host-only
                # alias keeps the user-facing local review URL short without
                # changing the daemon's JSON root for API clients.
                try:
                    body, release = web_releases.current_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Hawking-Web-Release", str(release["current"]["release_id"]))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                    self.wfile.flush()
                    return
                except (OSError, ValueError) as exc:
                    return self._send(404, {"error": {
                        "message": f"Hawking review surface is unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/release":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "H-Web release state is local-only"}})
                try:
                    return self._send(200, web_releases.bootstrap())
                except (OSError, ValueError) as exc:
                    return self._send(503, {"error": {"message": f"H-Web release unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/permissions":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Hawking machine authority is local-only"}})
                try:
                    from .permissions import PermissionsService
                    values = urllib.parse.parse_qs(parsed_url.query)
                    broad = (values.get("broad") or [""])[0].strip().lower() in {"1", "true", "yes"}
                    return self._send(200, PermissionsService(gateway_workspace).machine_ready(broad=broad))
                except Exception as exc:
                    return self._send(503, {"error": {"message": f"Hawking machine authority unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/events":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Hawking Web events are local-only"}})
                values = urllib.parse.parse_qs(parsed_url.query)
                raw_id = (values.get("id") or [None])[0]
                try:
                    after = int((values.get("after") or ["0"])[0])
                    limit = int((values.get("limit") or ["64"])[0])
                    if after < 0 or limit < 1:
                        raise ValueError("after must be non-negative and limit must be positive")
                    session = _web_session(raw_id)
                    return self._send(200, _web_event_replay(session, after, min(64, limit)))
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {"message": f"event projection unavailable: {type(exc).__name__}"}})
            if path == "/hawking/recovery":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "recovery status is local-only"}})
                try:
                    from .recovery import recovery_status
                    return self._send(200, recovery_status(gateway_workspace))
                except Exception as exc:
                    return self._send(503, {"error": {"message": f"recovery unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/artifacts/config":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web artifact configuration is local-only"}})
                from .web_artifacts import (
                    LONG_PASTE_THRESHOLD_BYTES, LONG_PASTE_THRESHOLD_TOKENS,
                    MAX_WEB_ARTIFACT_BYTES, MAX_WEB_ARTIFACTS_PER_TURN,
                )
                return self._send(200, {
                    "schema": "hawking.web.artifacts.config.v1",
                    "owner": "hawkingd",
                    "long_paste_threshold_bytes": LONG_PASTE_THRESHOLD_BYTES,
                    "long_paste_threshold_tokens": LONG_PASTE_THRESHOLD_TOKENS,
                    "max_artifact_bytes": MAX_WEB_ARTIFACT_BYTES,
                    "max_artifacts_per_turn": MAX_WEB_ARTIFACTS_PER_TURN,
                })
            if path == "/hawking/web/memory/search":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web memory search is local-only"}})
                try:
                    from .web_artifacts import memory_search, workspace_id
                    values = urllib.parse.parse_qs(parsed_url.query)
                    query = str((values.get("query") or values.get("q") or [""])[0])
                    if not query.strip():
                        raise ValueError("query is required")
                    limit = int((values.get("limit") or ["8"])[0])
                    session_id = (values.get("session_id") or [None])[0]
                    return self._send(200, {
                        "schema": "hawking.web.memory.search.v1",
                        "owner": "hawkingd",
                        "workspace_id": workspace_id(gateway_workspace),
                        "query": query[:240],
                        "results": memory_search(
                            gateway_workspace, query, session_id=session_id, limit=limit
                        ),
                    })
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"Hawking memory search unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/attachments":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web artifact access is local-only"}})
                try:
                    from .web_artifacts import read as read_web_artifact
                    values = urllib.parse.parse_qs(parsed_url.query)
                    artifact_id = (values.get("id") or [""])[0]
                    if not artifact_id:
                        raise ValueError("attachment id is required")
                    item = read_web_artifact(gateway_workspace, artifact_id)
                    if (values.get("content") or [""])[0].lower() in {"1", "true", "yes"}:
                        download = (values.get("download") or [""])[0].lower() in {
                            "1", "true", "yes"
                        }
                        return self._send_bytes(
                            200,
                            item["bytes"],
                            str(item["attachment"].get("media_type") or "text/plain")
                            + "; charset=utf-8",
                            download_name=(item["attachment"].get("filename")
                                           if download else None),
                        )
                    return self._send(200, {"attachment": item["attachment"]})
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(404, {"error": {
                        "message": f"attachment unavailable: {type(exc).__name__}"}})
            if path.startswith("/s/"):
                # Share links are the only public-looking object surface. The
                # token is checked by the capability store; no /v1 or /api
                # route is proxied through this path.
                try:
                    from .share_bridge import (
                        ShareError, html_share, manifest_share, parse_share_get,
                        snapshot_share,
                    )
                    token, view = parse_share_get(path)
                    root = stores.get("state_root") or (
                        getattr(repo, "root", None) if repo is not None else None
                    ) or os.getcwd()
                    if view == "html":
                        return self._send_bytes(200, html_share(root, token), "text/html; charset=utf-8")
                    if view == "manifest":
                        return self._send(200, manifest_share(root, token))
                    return self._send(200, snapshot_share(root, token))
                except ShareError as exc:
                    return self._send(exc.status, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(500, {"error": {"message": f"share unavailable: {type(exc).__name__}"}})
            if path == "/hawking/web/session":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web session access is local-only"}})
                values = urllib.parse.parse_qs(parsed_url.query)
                bound = self._trusted_builder_session()
                # A build browser always resumes the session that the daemon
                # bound during the local handoff.  localStorage remains a
                # convenience cache, never an authority selector.
                raw_id = bound.id if bound is not None else (values.get("id") or [None])[0]
                try:
                    session = _web_session(raw_id)
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                return self._send(200, _web_session_snapshot(session))
            if path == "/hawking/web/conversations":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web session access is local-only"}})
                items: List[Dict[str, Any]] = []
                try:
                    for record_path in sorted(
                        Path(_web_store().dir).glob("*.json"),
                        key=lambda item: item.stat().st_mtime,
                        reverse=True,
                    ):
                        session = _web_store().load(record_path.stem)
                        if session is None:
                            continue
                        ui = getattr(session, "ui", {}) or {}
                        # The existing SessionStore is shared by Hawking's
                        # other surfaces.  Only sessions explicitly owned by
                        # this client belong in the Web chat rail; otherwise
                        # legacy/operator prompts look like one new chat per
                        # message even though the Web transcript is durable.
                        if not (
                            session.id.startswith("web-")
                            or ui.get("surface") == "hawking-web"
                        ):
                            continue
                        messages = _web_conversation_messages(session)
                        goal_ids = _web_goal_ids(ui)
                        if not messages and not goal_ids:
                            continue
                        first = next(
                            (row.get("content") for row in messages
                             if row.get("role") == "user"),
                            "New conversation",
                        )
                        items.append({
                            "session_id": session.id,
                            "title": " ".join(str(first).split())[:96],
                            "updated_at": getattr(session, "created_at", ""),
                            "turns": len(messages),
                            "goal_ids": goal_ids,
                            "goal_count": len(goal_ids),
                            "active_goal_id": (
                                ui.get("active_goal_id")
                                or None
                            ),
                        })
                except OSError:
                    items = []
                return self._send(200, {
                    "schema": "hawking.web.conversations.v1",
                    "conversations": items[:40],
                })
            if path == "/hawking/web/models":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web model state is local-only"}})
                values = urllib.parse.parse_qs(parsed_url.query)
                search = (values.get("search") or values.get("q") or [""])[0]
                try:
                    rows = _web_model_rows(search)
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"model search unavailable: {type(exc).__name__}"}})
                return self._send(200, {
                    "schema": "hawking.web.models.v1",
                    "models": rows,
                    "enabled_models": sorted(_web_enabled_models()),
                    "search": str(search or "")[:120] or None,
                    "search_owner": "hawkingd",
                })
            if path == "/hawking/assets/hawking.svg":
                asset = Path(__file__).with_name("assets") / "hawking.svg"
                try:
                    return self._send_bytes(
                        200, asset.read_bytes(), "image/svg+xml; charset=utf-8"
                    )
                except OSError:
                    return self._send(404, {"error": {
                        "message": "Hawking branding asset is unavailable"}})
            if path in ("/v1/models", "/models"):
                if not self._gateway_access_allowed():
                    return self._send(401, {"error": {
                        "message": "Hawking remote gateway access requires a bearer token",
                        "type": "authentication_error",
                    }})
                if canonical_router is not None:
                    # Logical role names are scheduling classifications, not
                    # selectable models. Keep them in the internal router, but
                    # expose only the existing frozen Kimi P0 review identity
                    # on this deliberately minimal client surface.
                    native_rows = _review_only_device_model_rows()
                else:
                    # Every admitted body, so a source specimen cannot become
                    # executable merely because Open WebUI displayed its name.
                    rows = getattr(backend, "catalog", None)
                    native_rows = rows() if callable(rows) else models_payload(identity)["data"]
                    review_rows = _review_only_device_model_rows()
                    native_rows = review_rows or native_rows
                asset_base = (
                    endpoint_base.rsplit("/v1", 1)[0]
                    if endpoint_base and "/v1" in endpoint_base
                    else ""
                )
                if asset_base:
                    for row in native_rows:
                        if (
                            isinstance(row, dict)
                            and row.get("id") == REVIEW_ONLY_MODEL_ID
                        ):
                            row["info"] = {
                                "meta": {
                                    "profile_image_url": (
                                        asset_base + "/hawking/assets/hawking.svg"
                                    ),
                                },
                            }
                native_rows = list(native_rows or [])
                policy = gateway_policy()
                configured_rows = _configured_openrouter_rows(policy)
                catalog_rows: List[Dict[str, Any]] = []
                search_values = urllib.parse.parse_qs(parsed_url.query)
                search_term = str(
                    (search_values.get("search") or search_values.get("q") or [""])[0]
                ).strip()[:120]
                # Resolve only a boolean here.  The credential itself stays
                # inside the provider/keychain boundary and never enters the
                # catalog response or Hawking state.
                from .remote_cognition import openrouter_credential_available
                has_credential = openrouter_credential_available()
                catalog_error = None
                if has_credential and search_term:
                    try:
                        from .remote_cognition import cached_openrouter_catalog
                        catalog_rows = _openrouter_catalog_rows(
                            cached_openrouter_catalog(
                                policy,
                                force=(search_values.get("refresh") == ["true"]),
                            )
                        )
                        with gateway_state_lock:
                            gateway_state["remote_catalog_ids"] = {
                                str(row["id"]) for row in catalog_rows
                                if isinstance(row, Mapping) and row.get("id")
                            }
                    except Exception as exc:
                        catalog_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                remote_rows = _select_openrouter_catalog_rows(
                    configured_rows,
                    catalog_rows,
                    search=search_term,
                )
                merged: Dict[str, Dict[str, Any]] = {}
                for row in native_rows:
                    if isinstance(row, Mapping) and row.get("id"):
                        merged.setdefault(str(row["id"]), dict(row))
                # Provider metadata wins over a configuration-only placeholder
                # for the same model id, while native ids remain authoritative.
                for row in remote_rows:
                    if isinstance(row, Mapping) and row.get("id"):
                        model_id = str(row["id"])
                        if model_id not in merged or "hawking" in row:
                            merged[model_id] = dict(row)
                auto_row = merged.get("hawking-auto")
                if isinstance(auto_row, dict):
                    auto_row.setdefault("name", "Hawking Auto")
                    auto_row.setdefault(
                        "description",
                        "Hawking selects a qualified cloud or native route for this request.",
                    )
                    auto_meta = auto_row.setdefault("hawking", {})
                    with gateway_state_lock:
                        live_remote_ids = set(
                            gateway_state.get("remote_catalog_ids") or ()
                        )
                    if not live_remote_ids:
                        live_remote_ids = {
                            str(row.get("id")) for row in remote_rows
                            if isinstance(row, Mapping) and row.get("id")
                        }
                    from .auto_mode import REMOTE_AUTO_ROSTER
                    live_roster = [
                        model_id for model_id in REMOTE_AUTO_ROSTER
                        if model_id in live_remote_ids
                    ]
                    if has_credential and live_roster:
                        auto_meta.update({
                            "availability": "AVAILABLE",
                            "native_only": False,
                            "remote_capable": True,
                            "runtime": "hawking-auto",
                            "cloud": {
                                "provider": "openrouter",
                                "qualified_roster": live_roster,
                            },
                            "selection_policy": AUTO_POLICY_REVISION,
                        })
                payload = {"object": "list", "data": list(merged.values())}
                payload["hawking_catalog"] = {
                    "surface": "hawking-web",
                    "primary_model": REVIEW_ONLY_MODEL_ID,
                    "default_remote_models": sorted(_visible_openrouter_model_ids()),
                    "search": search_term or None,
                    "search_available": True,
                    "hidden_provider_catalog": not bool(search_term),
                    "search_limit": 40,
                }
                if catalog_error:
                    payload["hawking_catalog"].update({
                        "provider": "openrouter",
                        "status": "unavailable",
                        "error": catalog_error,
                    })
                return self._send(200, payload)
            if path in ("/health", "/"):
                live = dict(health)
                live["hweb_capability_projection_revision"] = (
                    HWEB_CAPABILITY_PROJECTION_REVISION
                )
                live["hweb_builder_session_revision"] = HWEB_BUILDER_SESSION_REVISION
                live["hawking_auto_orchestration_revision"] = HAWKING_AUTO_ORCHESTRATION_REVISION
                with gateway_state_lock:
                    receipt = gateway_state.get("last_remote_receipt")
                if isinstance(receipt, Mapping):
                    live["remote_cognition"] = dict(receipt)
                if canonical_router is not None:
                    live["canonical_runtime"] = canonical_router.status()
                if webui_manager is not None:
                    try:
                        live["client_surfaces"] = webui_manager.snapshot()
                    except Exception as exc:  # health must remain observable
                        live["client_surfaces"] = []
                        live["client_surfaces_error"] = (
                            f"{type(exc).__name__}: {exc}")
                if hasattr(backend, "body"):
                    live.update(resident=backend.identity,
                                model=backend.body.path,
                                state=backend.state,
                                sampling=("greedy-argmax" if backend.greedy
                                          else "profile-default"))
                    verdict = getattr(backend, "tools_verdict", None)
                    if verdict is not None and "tools" in live:
                        from .chat_tools import prompt_menu_names
                        live["tools"] = {
                            "qualified": bool(verdict.get("qualified")),
                            "names": (prompt_menu_names(
                                registry,
                                write=stores.get("engine") is not None)
                                      if verdict.get("qualified") else []),
                            "reason": verdict.get("reason"),
                            # The reply that failed. A verdict without the
                            # evidence behind it cannot be argued with -- and
                            # this gate has already produced one false negative
                            # against sealed-3.14.
                            "reply_excerpt": verdict.get("reply_excerpt"),
                        }
                    if backend.error:
                        live["error"] = backend.error
                    resolved_action = getattr(backend, "resolved_action", None)
                    if resolved_action is not None:
                        live["resolved_action"] = resolved_action.to_wire()
                        live["resident_binding"] = resolved_action.binding
                return self._send(200, live)
            if path.startswith("/v1/workers") or path.startswith("/api/workers"):
                # This is a view over the existing durable WorkUnit owner, not
                # a second scheduler.  Remote cognition is identified by
                # Hawking-owned worker ids and lifecycle fields; no provider
                # PID is invented for an OpenRouter process.
                root = Path(
                    stores.get("state_root")
                    or (getattr(repo, "root", None) if repo is not None else None)
                    or os.getcwd()
                ).resolve()
                workers: List[Dict[str, Any]] = []
                workunit_root = root / ".hawking" / "workunits"
                for state_path in sorted(workunit_root.glob("*.json"))[-100:]:
                    try:
                        value = json.loads(state_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(value, dict):
                        workers.append(value)
                return self._send(200, {
                    "schema": "hawking.worker_fabric_view.v1",
                    "workers": workers,
                    "count": len(workers),
                    "claim_boundary": (
                        "Hawking-owned WorkUnit lifecycle view; remote providers "
                        "have no local process identity."
                    ),
                })
            if path == "/v1/auto/plan":
                values = urllib.parse.parse_qs(parsed_url.query)
                objective = (values.get("objective") or values.get("q") or [""])[0]
                try:
                    from .auto_orchestration import build_cognition_plan
                    plan = build_cognition_plan({
                        "model": "hawking/auto",
                        "messages": [{"role": "user", "content": str(objective)}],
                    }, allowed_models=gateway_policy().allowed_models)
                    return self._send(200, plan)
                except Exception as exc:
                    return self._send(400, {"error": {"message": f"Auto plan unavailable: {type(exc).__name__}: {exc}"}})
            if path == "/v1/goal/tree":
                values = urllib.parse.parse_qs(parsed_url.query)
                gid = (values.get("id") or [""])[0]
                root = Path(stores.get("state_root") or (
                    getattr(repo, "root", None) if repo is not None else None
                ) or os.getcwd()).resolve()
                from .goal_surface import load_goal, load_result
                from .workunit_owner import WorkunitOwner
                goal = load_goal(root, gid)
                if not goal:
                    return self._send(404, {"error": {"message": f"no goal {gid}"}})
                owner = WorkunitOwner(root)
                records = []
                for state_path in sorted((root / ".hawking" / "workunits").glob("*.json")):
                    try:
                        record = owner.load(state_path.stem)
                    except Exception:
                        continue
                    if record.goal_id != gid:
                        continue
                    records.append({
                        "workunit_id": record.workunit_id,
                        "worker_id": record.worker_id,
                        "worker_attempt_id": getattr(record, "worker_attempt_id", ""),
                        "context_packet_digest": getattr(record, "context_packet_digest", ""),
                        "candidate_manifest_id": getattr(record, "candidate_manifest_id", ""),
                        "execution_binding_digest": getattr(record, "execution_binding_digest", ""),
                        "model": record.worker_model,
                        "role": getattr(record, "role", "worker"),
                        "task_class": getattr(record, "task_class", "general"),
                        "state": record.state,
                        "worker_status": record.worker_status,
                        "parent_workunit_id": record.parent_workunit_id,
                        "child_worker_ids": list(record.child_worker_ids),
                        "cost_usd": float(record.worker_cost_usd),
                        "last_turn": record.last_turn,
                        "plan_id": getattr(record, "plan_id", ""),
                        "worker_packet_path": getattr(record, "worker_packet_path", ""),
                    })
                return self._send(200, {
                    "schema": "hawking.goal.worker_tree.v1",
                    "goal": goal,
                    "workunits": records[:64],
                    "result": load_result(root, gid),
                    "claim_boundary": "Hawking-owned worker tree; no hidden reasoning or provider credentials",
                })
            if path.startswith("/v1/goal/status"):
                # One operator read joins Goal, owner heartbeat, and result
                # without making the Web client reconstruct Hawking state.
                from urllib.parse import urlparse, parse_qs
                from hawking.goal_surface import load_goal, load_result
                from hawking.workunit_owner import WorkunitOwner
                qs = parse_qs(urlparse(self.path).query)
                gid = (qs.get("id") or [""])[0]
                compact_view = (qs.get("view") or [""])[0].strip().lower() == "compact"
                root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                if not gid:
                    return self._send(400, {"error": {"message": "id required"}})
                goal = load_goal(root, gid)
                if not goal:
                    return self._send(404, {"error": {"message": f"no goal {gid}"}})
                wid = str(goal.get("workunit_id") or gid)
                try:
                    owner = WorkunitOwner(root).status(wid)
                except Exception as exc:
                    owner = {"error": f"{type(exc).__name__}: {exc}"}
                if compact_view:
                    goal = dict(goal)
                    goal["objective"] = str(goal.get("objective") or "")[:800]
                    owner = dict(owner)
                    owner["objective"] = str(owner.get("objective") or "")[:800]
                return self._send(200, {
                    "goal": goal,
                    "workunit": owner,
                    "result": load_result(root, gid),
                    "view": "compact" if compact_view else "full",
                })
            if path.startswith("/v1/goal/diff"):
                from urllib.parse import urlparse, parse_qs
                from hawking.goal_surface import load_goal
                qs = parse_qs(urlparse(self.path).query)
                gid = (qs.get("id") or [""])[0]
                root = Path(stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()).resolve()
                if not gid:
                    return self._send(400, {"error": {"message": "id required"}})
                goal = load_goal(root, gid)
                if not goal:
                    return self._send(404, {"error": {"message": f"no goal {gid}"}})
                return self._send(200, _goal_diff_payload(root, goal, gid))
            if path == "/v1/goal/ui" or path.startswith("/v1/goal/ui"):
                from hawking.goal_surface import OPERATOR_UI_HTML
                if not self._local_only():
                    return self._send(403, {"error": {"message": "goal UI is local-only"}})
                body = OPERATOR_UI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                return
            if path.startswith("/v1/goal/result"):
                from urllib.parse import urlparse, parse_qs
                from hawking.goal_surface import load_result
                qs = parse_qs(urlparse(self.path).query)
                gid = (qs.get("id") or [""])[0]
                root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                if not gid:
                    return self._send(400, {"error": {"message": "id required"}})
                doc = load_result(root, gid)
                if not doc:
                    return self._send(404, {"error": {"message": f"no result for {gid}"}})
                return self._send(200, doc)
            from urllib.parse import urlparse, parse_qs
            if urlparse(self.path).path.rstrip("/") in {"/v1/goal", "/v1/goals"}:
                from hawking.goal_surface import list_goals, load_goal
                qs = parse_qs(urlparse(self.path).query)
                gid = (qs.get("id") or [""])[0]
                root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                if gid:
                    doc = load_goal(root, gid)
                    if not doc:
                        return self._send(404, {"error": {"message": f"no goal {gid}"}})
                    return self._send(200, doc)
                items = list_goals(root, limit=20)
                return self._send(200, {"goals": items, "count": len(items)})
            if path == "/hawkingd/webui":
                if webui_manager is None:
                    return self._send(404, {"error": {
                        "message": "daemon WebUI supervision is unavailable"}})
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "WebUI supervision is local-only"}})
                return self._send(200, {
                    "owner": "hawkingd",
                    "client_surfaces": webui_manager.snapshot(),
                })
            self._send(404, {"error": {"message": f"no route {self.path}"}})

        def _remote_request(
            self,
            body: Mapping[str, Any],
            model: str,
            *,
            responses: bool = False,
        ) -> Any:
            from .providers import GenerationRequest

            policy = gateway_policy()
            request_body = (
                _responses_request_body(body, model)
                if responses
                else {**dict(body), "model": model}
            )
            if not responses:
                messages = request_body.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty array")
            typed_mutation_contract = False
            raw_completion_contract = request_body.get("hawking_completion_contract")
            if isinstance(raw_completion_contract, Mapping):
                required_contract_tools = raw_completion_contract.get(
                    "required_successful_tools"
                ) or raw_completion_contract.get("required_tools") or []
                if isinstance(required_contract_tools, str):
                    required_contract_tools = [required_contract_tools]
                typed_mutation_contract = "repo.edit" in {
                    str(name).strip() for name in required_contract_tools
                }
            # Discrete WorkUnits have a shorter owner turn budget than the
            # gateway's ordinary retry envelope.  Keep one bounded upstream
            # attempt per mutation turn so a provider 5xx/timeout returns to
            # Hawking quickly for fresh-route recovery instead of outliving
            # the owner and surfacing as a late BrokenPipe/502.
            fast_failover = typed_mutation_contract
            max_tokens = request_body.get("max_tokens")
            policy.validate_request(model, int(max_tokens) if max_tokens is not None else None)
            if not policy.allow_model_fallbacks:
                # OpenRouter's optional ``models`` list can substitute a
                # different model.  Exact model provenance is Hawking's
                # default; only an explicit server policy may enable it.
                request_body.pop("models", None)
            # Caller metadata is intentionally not forwarded.  Client identity
            # is a bounded label in a Hawking-owned receipt, while spend and
            # privacy policy come from the server configuration.
            request_body.pop("metadata", None)
            request_body.pop("hawking_provider", None)
            request_body.pop("hawking_worker_mode", None)
            request_body.pop("hawking_worker_id", None)
            request_body.pop("hawking_worker_attempt_id", None)
            request_body.pop("hawking_workunit_id", None)
            request_body.pop("hawking_goal_id", None)
            request_body.pop("hawking_goal_mode", None)
            request_body.pop("hawking_budget_phase", None)
            request_body.pop("hawking_mutation_bundle", None)
            request_body.pop("hawking_completion_contract", None)
            request_body.pop("hawking_web_session_id", None)
            request_body.pop("hawking_web_route_scope", None)
            request_body.pop("hawking_web", None)
            request_body.pop("hawking_capability_intent", None)
            requested_worker_budget = request_body.pop(
                "hawking_worker_budget_usd", None
            )
            provider_session_cold = bool(
                request_body.pop("hawking_provider_session_cold", False)
            )
            request_body["transport_options"] = policy.transport_options(
                model, request_body.get("provider")
            )
            if fast_failover:
                request_body["transport_options"]["hawking_fast_failover"] = True
                # An upstream HTTP 200 that omits the required tool payload is
                # a provider semantic miss, not a gateway outage. Preserve it
                # as a typed response so the WorkUnit can classify/recover
                # without triggering the generic HTTP-502 retry path.
                request_body["transport_options"]["hawking_semantic_miss_as_response"] = True
            if typed_mutation_contract and not str(model).strip().lower().startswith("deepseek/"):
                # Some preferred OpenRouter routes accept the schema but ignore
                # required tool choice and answer in prose.  For a declared
                # mutation contract, require a parameter-capable route and let
                # OpenRouter choose one rather than pinning the incompatible
                # preference. Ordinary chat and read-only workers retain the
                # configured provider preference.
                provider_options = request_body["transport_options"].get("provider")
                if isinstance(provider_options, dict):
                    provider_options.pop("order", None)
                    provider_options["require_parameters"] = True
                    provider_options["allow_fallbacks"] = True
                    # Provider-specific thinking workarounds remain scoped to
                    # non-DeepSeek routes; the active Goal is pinned to the
                    # direct DeepSeek route and must retain its named choice.
                    if str(model).strip().lower().startswith("qwen/"):
                        provider_options["chat_template_kwargs"] = {"enable_thinking": False}
            metadata = {
                "gateway": "hawkingd",
                "client": self._client_label(),
                "project": _gateway_label(
                    self.headers.get("X-Hawking-Project"), fallback="unscoped"
                ),
            }
            if requested_worker_budget is not None:
                try:
                    requested = max(0.0, float(requested_worker_budget))
                except (TypeError, ValueError):
                    requested = 0.0
                configured = max(0.0, float(policy.cost_policy.max_cost_usd))
                metadata["cost_authorization"] = {
                    "max_usd": min(requested, configured)
                }
            request_body["metadata"] = metadata
            request_body["request_id"] = f"gateway-{uuid.uuid4().hex[:24]}"
            return GenerationRequest.from_mapping(request_body)

        def _remote_provider(self, model: str) -> Any:
            from .remote_cognition import OpenRouterProvider, _keychain_resolver_from_environment

            policy = gateway_policy()
            return OpenRouterProvider(
                model,
                workspace=gateway_workspace,
                endpoint=policy.endpoint,
                cost_policy=policy.cost_policy,
                key_resolver=_keychain_resolver_from_environment(),
                timeout=policy.timeout_s,
                max_attempts=policy.max_attempts,
                catalog_ttl_s=policy.catalog_ttl_s,
            )

        def _sse(self, payload: Any, *, event: Optional[str] = None) -> None:
            if event:
                self.wfile.write(f"event: {event}\n".encode())
            if payload == "[DONE]":
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.wfile.write(b"data: " + encoded + b"\n\n")
            self.wfile.flush()

        def _chat_stream_chunk(
            self,
            *,
            request_id: str,
            model: str,
            delta: Dict[str, Any],
            finish_reason: Optional[str] = None,
            usage: Optional[Mapping[str, Any]] = None,
            hawking: Optional[Mapping[str, Any]] = None,
        ) -> Dict[str, Any]:
            chunk: Dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }
            if usage is not None:
                chunk["usage"] = dict(usage)
            if hawking is not None:
                chunk["hawking"] = dict(hawking)
            return chunk

        def _begin_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True

        def _remote_hawking_completion(
            self,
            body: Mapping[str, Any],
            provider: Any,
            request: Any,
            policy: Any,
            *,
            stream: bool = False,
            mutation_lease: Optional[str] = None,
        ) -> Any:
            """Run an explicit remote worker through Hawking's tool registry.

            Generic OpenRouter requests retain provider tool-call passthrough.
            Only a Hawking worker mode or durable discrete Goal opts into this
            loop, where the registry, mutation lock, Goal bundle, and bounded
            completion contract remain Hawking-owned.
            """
            if registry is None:
                return provider.generate(request, timeout=policy.timeout_s)
            from .chat_tools import (
                BUILDER_TOOLS,
                LOCAL_TOOLS,
                _declared_schema,
                prompt_menu_names,
                run_builder_tool,
                run_local_tool,
                session_menu,
                validate_mutation_bundle,
            )

            try:
                mode = worker_request_mode(body.get("hawking_worker_mode"))
            except ValueError:
                mode = WorkerRequestMode.DEFAULT
            goal_id = str(body.get("hawking_goal_id") or "").strip()
            goal_mode = str(body.get("hawking_goal_mode") or "").strip()
            is_goal = bool(goal_id and goal_mode == "discrete_goal")
            is_research = bool(goal_id and goal_mode == "worker_research")
            is_web_builder = bool(
                body.get("hawking_web")
                and body.get("hawking_web_build_authorized") is True
                and mode is WorkerRequestMode.BUILDER_INTERACTIVE
            )
            is_hawking_worker = bool(
                is_goal
                or is_research
                or mode in {
                    WorkerRequestMode.READ_ONLY_RESEARCH,
                    WorkerRequestMode.REASON_ONLY,
                    WorkerRequestMode.BUILDER_INTERACTIVE,
                }
            )
            write = bool(
                (is_goal or is_web_builder)
                and mode is WorkerRequestMode.DEFAULT
                and stores.get("engine") is not None
            )
            if is_web_builder and stores.get("engine") is not None:
                write = True
            if (
                (is_goal or is_web_builder)
                and mode in {
                    WorkerRequestMode.DEFAULT,
                    WorkerRequestMode.BUILDER_INTERACTIVE,
                }
                and stores.get("engine") is None
                and stores.get("state_root")
                and str(health.get("authority") or "") == "remote_goal_write"
            ):
                # The waiting-native daemon still admits locally submitted
                # remote Goals, but an older construction path left the
                # transactional repo engine out of the request-local store.
                # Bind the existing canonical engine lazily; never grant this
                # to generic or unscoped requests.
                from .engine import Engine
                from .workspace import Workspace

                stores["engine"] = Engine(Workspace(str(stores["state_root"])))
                write = True
            requested = (
                HWEB_READ_ONLY_TOOL_NAMES
                if body.get("hawking_web")
                and mode is WorkerRequestMode.READ_ONLY_RESEARCH
                else worker_tool_allowlist(mode)
            )
            if is_goal and requested is None:
                # A durable Goal has the same authority boundary as a local
                # write session, but it must not receive the entire registry
                # as a provider schema list. That made a normal Build turn
                # spend its budget on 100+ irrelevant doors and could starve
                # the typed mutation door. Keep the canonical H-Web read
                # envelope plus the builder doors callable; tools.catalog
                # remains the explicit escape hatch for an admitted name.
                requested = frozenset(HWEB_READ_ONLY_TOOL_NAMES) | frozenset(BUILDER_TOOLS)
                allowed = prompt_menu_names(
                    registry, write=write, allowed_names=requested,
                )
            else:
                allowed = prompt_menu_names(
                    registry,
                    write=write,
                    allowed_names=requested,
                )
            extra_schemas: List[Dict[str, Any]] = []
            # Builder doors intentionally remain outside the ordinary registry,
            # but an explicit BUILD mutation turn must still project the
            # canonical repo.edit schema into the provider request.  Previously
            # it was placed in `extra_schemas` (an internal context hint) while
            # the actual request was generated from `allowed_names`, so the
            # model could not call the mutation owner.
            if write and (is_web_builder or is_goal):
                requested = frozenset(set(requested or ()) | {
                    "repo.edit", "tests.run",
                })
                # `repo.edit` is a typed BUILD door, not a registry tool, so
                # the normal registry schema projection cannot discover it.
                # Keep it on the same provider boundary as every other
                # admitted function instead of exposing only its name in the
                # allowlist.
                extra_schemas.append({
                    "type": "function",
                    "function": {
                        "name": "repo.edit",
                        "description": BUILDER_TOOLS["repo.edit"],
                        "parameters": _declared_schema("repo.edit"),
                    },
                })
                if "repo.edit" not in allowed:
                    allowed.append("repo.edit")
                if "tests.run" not in allowed:
                    allowed.append("tests.run")
            mutation_bundle = body.get("hawking_mutation_bundle")
            mutation_plan = (
                mutation_bundle.get("mutation_plan")
                if isinstance(mutation_bundle, Mapping)
                and isinstance(mutation_bundle.get("mutation_plan"), Mapping)
                else {}
            )
            if write and (is_goal or is_web_builder):
                if "repo.edit" not in allowed:
                    allowed.append("repo.edit")
            worker_tool_names = (
                "hawking.worker.spawn",
                "hawking.worker.kill",
                "hawking.worker.status",
            )
            child_models = (
                "deepseek/deepseek-v4.1-flash",
                "qwen/qwen3.8-flash",
            )
            if is_hawking_worker and not is_web_builder:
                allowed.extend(
                    name for name in worker_tool_names if name not in allowed
                )
                extra_schemas.extend([
                    {
                        "type": "function",
                        "function": {
                            "name": "hawking.worker.spawn",
                            "description": (
                                "Spawn one bounded read-only Hawking child worker "
                                "for a materially separable objective. The child "
                                "uses the existing Goal/WorkUnit owner and may not "
                                "mutate the repository."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "objective": {"type": "string", "maxLength": 2000},
                                    "model": {"type": "string", "enum": list(child_models)},
                                    "budget_usd": {"type": "number", "minimum": 0, "maximum": 0.5},
                                    "acceptance": {
                                        "type": "array",
                                        "items": {"type": "string", "maxLength": 500},
                                        "maxItems": 8,
                                    },
                                },
                                "required": ["objective"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "hawking.worker.kill",
                            "description": (
                                "Kill the current ephemeral worker or one of its "
                                "direct children. Hawking preserves the checkpoint."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "worker_id": {"type": "string", "maxLength": 100},
                                    "reason": {"type": "string", "maxLength": 300},
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "hawking.worker.status",
                            "description": (
                                "Read the Hawking-owned lifecycle status of the "
                                "current worker or one direct child."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "worker_id": {"type": "string", "maxLength": 100},
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                ])
            if (
                is_goal
                and str(mutation_plan.get("kind") or "") in {
                    "macos_receipt_regression",
                    "document_capture_ocr",
                }
            ):
                # This continuation has already consumed a separately
                # evidenced native WorkUnit. Expose only the four remaining
                # contract doors; a broad menu made the resident repeatedly
                # re-read the tree instead of crossing the mutation boundary.
                required_mutation_tools = {
                    "repo.edit", "tests.run", "git.status", "git.diff",
                }
                allowed = [name for name in allowed if name in required_mutation_tools]
                if write and "repo.edit" not in allowed:
                    allowed.append("repo.edit")
                for name in sorted(required_mutation_tools - set(allowed)):
                    if name != "repo.edit" and active_registry.get(name) is not None:
                        allowed.append(name)
            completion_contract = None
            if is_research:
                # Preserve the owner's compact first-evidence door when the
                # daemon reconstructs the research contract.  Losing this
                # field here lets a provider spend every bounded turn in
                # prose before Hawking observes the admitted source.  Read
                # the already-persisted WorkUnit contract only; do not reread
                # or widen the parent objective.
                research_first_tool = ""
                research_focus_paths = []
                workunit_id = str(body.get("hawking_workunit_id") or "").strip()
                if workunit_id and Path(workunit_id).name == workunit_id:
                    try:
                        # ``owner`` is initialized later in this request
                        # handler for registry dispatch.  This early contract
                        # seam needs only the fixed, Hawking-owned path; keep
                        # it bounded to a filename and the current workspace.
                        contract_path = (
                            gateway_workspace / ".hawking" / "goals"
                            / f"{workunit_id}.contract.json"
                        ).resolve()
                        contract_path.relative_to(gateway_workspace)
                        contract_doc = json.loads(contract_path.read_text(encoding="utf-8"))
                        focus_files = [
                            str(item).strip().replace("\\", "/")
                            for item in (contract_doc.get("focus_files") or [])
                            if str(item).strip()
                        ]
                        focus_tests = [
                            str(item).strip().replace("\\", "/").split("::", 1)[0]
                            for item in (contract_doc.get("focus_tests") or [])
                            if str(item).strip()
                        ]
                        # This is a bounded diagnostic lane, not a repository
                        # inventory.  Requiring every declared focus path kept
                        # its terminal JSON door closed long after the two-read
                        # budget had been reached, inflating provider context
                        # and producing truncation/semantic failures.  Admit a
                        # source and test anchor at most; the complete focus
                        # list remains visible in the WorkUnit prompt.
                        research_focus_paths = list(dict.fromkeys(
                            focus_files + focus_tests
                        ))[:2]
                        if research_focus_paths:
                            research_first_tool = "fs.read"
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        research_first_tool = ""
                completion_contract = {
                    "required_any_successful_tools": [
                        "fs.read", "filesystem.read", "fs.search", "filesystem.search",
                        "receipt.read", "receipt.inspect", "web.search", "github.search",
                        "hawking.worker.spawn",
                    ],
                    "required_first_tool": research_first_tool,
                    "required_focus_paths": research_focus_paths,
                    "require_structured_result": True,
                    "structured_required_keys": [
                        "observations", "sources", "recommendations",
                        "unresolved_questions", "child_worker_ids", "status",
                    ],
                    "structured_status_values": ["COMPLETE", "STRUCTURED_RESULT_READY"],
                    "allow_read_only_prose_result": True,
                    "max_continuation_turns": 4,
                    # Two grounded reads are sufficient for a bounded
                    # blocker-analysis packet. A ten-read menu bloated the
                    # terminal request until providers returned truncated
                    # prose, which the semantic boundary correctly surfaced
                    # as a 502 instead of useful read evidence.
                    "read_only_tool_call_limit": 2,
                    "reminder": (
                        "This Hawking child is still open. Return a compact structured "
                        "research packet after at least one real read-only observation, "
                        "or use Hawking worker delegation; do not mutate the worktree."
                    ),
                }
            if is_goal:
                learning_phase = str(
                    body.get("hawking_budget_phase") or ""
                ).strip().lower() == "tool_learning"
                if learning_phase:
                    completion_contract = {
                        "required_any_successful_tools": [
                            "tools.catalog", "fs.search", "fs.read", "git.status",
                        ],
                        "require_structured_result": True,
                        "read_only_tool_call_limit": 8,
                        "max_continuation_turns": 4,
                        "reminder": (
                            "This is Hawking's separate tool-learning phase. Use "
                            "the full admitted permissioned surface efficiently: "
                            "batch one targeted observation, use tools.catalog only "
                            "when a signature is unknown, and return a compact "
                            "structured evidence packet. Do not repeat broad "
                            "archaeology; the build phase follows this gate or the "
                            "learning allowance cap."
                        ),
                    }
                    # Provider mutation is proposal data, never an authority
                    # surface.  Hawking executes the validated proposal via
                    # the existing repo.edit transaction.
                    completion_contract["mutation_proposal_mode"] = True
                else:
                    completion_contract = {
                        "required_successful_tools": [
                            "repo.edit", "tests.run", "git.status", "git.diff",
                        ],
                        "mutation_proposal_mode": True,
                        # At most two focused observations (source plus a
                        # test/search anchor) are useful. The remote adapter
                        # turns an explicit read-only proposal into the final
                        # bounded observation, then closes the tool menu for
                        # patch synthesis.
                        "read_only_tool_call_limit": 2,
                        "require_final_tests_pass": True,
                        # Keep a malformed proposal to one compact repair.
                        # Further retries use a fresh durable WorkUnit packet
                        # rather than replaying a growing provider context.
                        "max_continuation_turns": 1,
                        "reminder": (
                            "This discrete Goal is still open. Do not call repo.edit "
                            "directly. Return exactly one JSON object with status "
                            "MUTATION_PROPOSED, objective, operations, expected_base_hashes, "
                            "required_tests, canonical_root, and completion_contract. "
                            "Hawking will validate and execute it, then provide grounded "
                            "test/status/diff evidence. Prose is not acceptance evidence."
                        ),
                    }
                    # Some admitted source/qualification tranches intentionally
                    # carry no focused test path.  They still require the
                    # engine's deterministic mutation validation plus status
                    # and diff evidence, but an unconditional tests.run door
                    # would make those valid WorkUnits impossible to close.
                    mutation_bundle = body.get("hawking_mutation_bundle")
                    bundle_tests = (
                        mutation_bundle.get("focus_tests")
                        or mutation_bundle.get("required_tests")
                        or []
                    ) if isinstance(mutation_bundle, Mapping) else []
                    if isinstance(mutation_bundle, Mapping):
                        test_paths = mutation_bundle.get("test_mutation_paths") or bundle_tests
                        completion_contract["required_regression_test_paths"] = [
                            str(path).strip().replace("\\", "/").split("::", 1)[0]
                            for path in test_paths
                            if str(path).strip()
                        ][:4]
                        completion_contract["existing_regression_test_paths"] = [
                            path for path in completion_contract["required_regression_test_paths"]
                            if (gateway_workspace / path).is_file()
                        ]
                        completion_contract["require_regression_test_mutation"] = bool(
                            mutation_bundle.get("require_regression_test_mutation")
                        )
                    if not bundle_tests:
                        completion_contract["required_successful_tools"] = [
                            "repo.edit", "git.status", "git.diff",
                        ]
                        completion_contract["require_final_tests_pass"] = False
                        completion_contract["verification_mode"] = (
                            "engine_validation_and_status_diff_no_focused_tests"
                        )

            if is_web_builder:
                # The public `h build` handoff is a real WorkUnit boundary,
                # not a model prompt convention.  Require the same grounded
                # mutation/verification graph as a discrete Goal so a prose
                # answer cannot close the Build lane.
                completion_contract = {
                    "required_successful_tools": [
                        "repo.edit", "tests.run", "git.status", "git.diff",
                    ],
                    "require_final_tests_pass": True,
                    # Permit a short bounded orientation, then the remote
                    # loop forces the next owed typed BUILD door instead of
                    # spending the whole session on repository archaeology.
                    "read_only_tool_call_limit": 8,
                    "max_continuation_turns": 4,
                    "reminder": (
                        "This Hawking BUILD turn is still open. Complete one "
                        "bounded repair through repo.edit, tests.run, git.status, "
                        "and git.diff. Provider prose is not acceptance evidence."
                    ),
                }

            web_session_id = str(body.get("hawking_web_session_id") or "").strip()
            web_session = None
            bundle_state: Dict[str, Any] = {}
            if is_web_builder and web_session_id:
                try:
                    web_session = _web_session(web_session_id, create=False)
                    prior_build = (
                        (getattr(web_session, "ui", {}) or {}).get("build_state")
                        if web_session is not None else None
                    )
                    if isinstance(prior_build, Mapping):
                        prior_bundle = prior_build.get("pending_mutation_bundle")
                        if isinstance(prior_bundle, Mapping):
                            bundle_state.update(dict(prior_bundle))
                except Exception:
                    # A missing browser record is a bounded continuation
                    # failure, never a reason to invent state or widen access.
                    web_session = None

            from .tool_registry import READ_ONLY, ToolResult
            from .workunit_owner import WorkunitOwner

            owner = WorkunitOwner(gateway_workspace)
            current_worker_id = str(body.get("hawking_worker_id") or "").strip()
            web_goal_ids: List[str] = []
            if body.get("hawking_web") and web_session_id:
                try:
                    web_session = _web_session(web_session_id, create=False)
                    web_goal_ids = _web_goal_ids(
                        getattr(web_session, "ui", {}) if web_session is not None else {}
                    )
                except Exception:
                    web_goal_ids = []
            runtime_snapshot = {
                "health": health,
                "session_id": web_session_id,
                "requested_model": body.get("model"),
                "route_scope": body.get("hawking_web_route_scope"),
                "auto_scope": body.get("hawking_auto_scope"),
            }

            def goal_browser_authority() -> Dict[str, Any]:
                """Recover browser authority only from the bound WorkUnit contract.

                Request JSON is untrusted transport data.  A client cannot gain
                browser action authority merely by naming a Goal or sending a
                browser-shaped field: all worker/session/attempt identifiers
                must agree with the durable Hawking record and its contract.
                """
                if not (is_goal or is_research):
                    return {}
                workunit_id = str(body.get("hawking_workunit_id") or "").strip()
                if not workunit_id or not current_worker_id:
                    return {}
                try:
                    record = owner.load(workunit_id)
                    if (
                        record.goal_id != goal_id
                        or record.worker_id != current_worker_id
                        or record.session_id != str(body.get("session_id") or "")
                    ):
                        return {}
                    requested_attempt = str(body.get("hawking_worker_attempt_id") or "").strip()
                    if requested_attempt and requested_attempt != record.worker_attempt_id:
                        return {}
                    contract_path = Path(record.contract_path).expanduser().resolve()
                    contract_path.relative_to(gateway_workspace)
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    authority = contract.get("authority")
                    browser = authority.get("browser") if isinstance(authority, Mapping) else None
                    if not isinstance(browser, Mapping) or not browser.get("observe"):
                        return {}
                    from .goal_surface import browser_session_id_for_goal

                    expected = browser_session_id_for_goal(goal_id)
                    if str(browser.get("browser_session_id") or "") != expected:
                        return {}
                    return dict(browser)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    return {}

            def goal_macos_authority() -> Dict[str, Any]:
                """Recover semantic desktop authority from the bound contract only."""
                if not is_goal:
                    return {}
                workunit_id = str(body.get("hawking_workunit_id") or "").strip()
                if not workunit_id or not current_worker_id:
                    return {}
                try:
                    record = owner.load(workunit_id)
                    if (
                        record.goal_id != goal_id
                        or record.worker_id != current_worker_id
                        or record.session_id != str(body.get("session_id") or "")
                    ):
                        return {}
                    requested_attempt = str(body.get("hawking_worker_attempt_id") or "").strip()
                    if requested_attempt and requested_attempt != record.worker_attempt_id:
                        return {}
                    contract_path = Path(record.contract_path).expanduser().resolve()
                    contract_path.relative_to(gateway_workspace)
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    authority = contract.get("authority")
                    macos = authority.get("macos") if isinstance(authority, Mapping) else None
                    if not isinstance(macos, Mapping) or not macos.get("observe"):
                        return {}
                    from .goal_surface import normalize_macos_authority

                    normalized = normalize_macos_authority(macos, goal_id=goal_id)
                    if not normalized or str(normalized.get("goal_id") or "") != goal_id:
                        return {}
                    # These identities are recovered from the durable WorkUnit,
                    # never accepted from model arguments.
                    normalized.update({
                        "goal_id": goal_id,
                        "workunit_id": record.workunit_id,
                        "worker_id": record.worker_id,
                    })
                    return normalized
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    return {}

            def goal_prior_tool_evidence() -> Dict[str, Any]:
                """Rehydrate prior successful tools from the bound WorkUnit.

                A WorkUnit may need several provider turns: native action first,
                then repository mutation and validation.  The completion
                contract must accumulate Hawking-owned observations without
                trusting model prose or caller-supplied claims.
                """
                if not is_goal:
                    return {"successful_tools": [], "verified_tools": []}
                workunit_id = str(body.get("hawking_workunit_id") or "").strip()
                successful: set[str] = set()
                verified: set[str] = set()
                try:
                    record = owner.load(workunit_id)
                    for item in record.evidence or []:
                        if not isinstance(item, Mapping):
                            continue
                        successful.update(
                            str(name).strip()
                            for name in (item.get("successful_tools") or [])
                            if str(name).strip()
                        )
                        verified.update(
                            str(name).strip()
                            for name in (item.get("verified_tools") or [])
                            if str(name).strip()
                        )
                    macos_root = gateway_workspace / "receipts" / "future" / "macos"
                    for receipt_path in macos_root.glob("*.json"):
                        try:
                            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        except (OSError, UnicodeError, ValueError, TypeError):
                            continue
                        if receipt.get("workunit_id") != workunit_id:
                            continue
                        operation = str(receipt.get("operation") or "")
                        outcome = str(receipt.get("outcome") or "")
                        if operation == "acquire" and outcome == "ADMITTED":
                            successful.add("macos.lease.acquire")
                        elif operation == "release" and outcome == "RELEASED":
                            successful.add("macos.lease.release")
                        elif operation == "preempt" and outcome == "PREEMPTED":
                            successful.add("macos.lease.preempt")
                        elif operation == "press" and outcome == "VERIFIED":
                            successful.add("macos.press")
                            verified.add("macos.press")
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
                return {
                    "successful_tools": sorted(successful),
                    "verified_tools": sorted(verified),
                }

            browser_authority = goal_browser_authority()
            macos_authority = goal_macos_authority()

            def goal_declares_browser_acceptance() -> bool:
                """Avoid inheriting a sibling capability's completion gate.

                A parent Goal may carry browser authority for an already
                accepted browser tranche while a child WorkUnit is scoped to a
                different surface (for example, macOS semantic actions).
                Authority is still recovered from the durable contract, but
                the child must only receive browser tools and browser proof
                obligations when its own acceptance contract asks for them.
                """
                if not is_goal:
                    return False
                required_names = []
                if isinstance(completion_contract, Mapping):
                    for key in (
                        "required_successful_tools",
                        "required_tools",
                        "required_verified_tools",
                        "required_any_successful_tools",
                    ):
                        value = completion_contract.get(key) or []
                        if isinstance(value, str):
                            value = [value]
                        if isinstance(value, (list, tuple, set, frozenset)):
                            required_names.extend(str(item).strip() for item in value)
                if any(name.startswith("browser.") for name in required_names):
                    return True
                workunit_id = str(body.get("hawking_workunit_id") or "").strip()
                if not workunit_id:
                    return False
                try:
                    record = owner.load(workunit_id)
                    contract_path = Path(record.contract_path).expanduser().resolve()
                    contract_path.relative_to(gateway_workspace)
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    acceptance = contract.get("acceptance") or []
                    if isinstance(acceptance, str):
                        acceptance = [acceptance]
                    return any("browser" in str(item).lower() for item in acceptance)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    return False

            browser_acceptance = goal_declares_browser_acceptance()
            # A semantic-action child may inherit the parent's browser
            # authority in its provenance packet, but that must never turn the
            # browser tranche into a required gate (or an exposed tool set).
            # A future mixed-surface child must opt into both explicitly.
            if (
                macos_authority.get("actions")
                and not bool(body.get("hawking_browser_acceptance"))
            ):
                browser_acceptance = False
            # Browser authority can be present on the parent for provenance;
            # it is not automatically a child capability.
            if browser_authority and not browser_acceptance:
                browser_authority = {}
            active_registry = registry
            if browser_authority or macos_authority:
                # The daemon's standard Goal registry deliberately omits
                # EXTERNAL_WRITE.  Rebuild the same canonical registry with
                # that one additional class only after the durable contract
                # has admitted this named Goal capability.
                from .tool_registry import EXTERNAL_WRITE, default_tool_registry

                active_registry = default_tool_registry(
                    gateway_workspace,
                    repo_root=registry.context.repo_root,
                    permissions=set(registry.context.permissions) | {EXTERNAL_WRITE},
                    authority={
                        "browser": browser_authority,
                        "macos": macos_authority,
                    },
                )
                browser_action_names = {
                    "browser.click", "browser.type", "browser.select",
                    "browser.key", "browser.scroll",
                }
                for name in sorted(
                    item["name"]
                    for item in active_registry.discover()
                    if isinstance(item, Mapping) and str(item.get("name") or "").startswith("browser.")
                ):
                    if name in browser_action_names and not browser_authority.get("actions"):
                        continue
                    if browser_authority and name not in allowed:
                        allowed.append(name)
                macos_action_names = {
                    "macos.lease.acquire", "macos.lease.release",
                    "macos.lease.preempt", "macos.press",
                }
                for name in sorted(
                    item["name"]
                    for item in active_registry.discover()
                    if isinstance(item, Mapping) and str(item.get("name") or "").startswith("macos.")
                ):
                    if name in macos_action_names and not macos_authority.get("actions"):
                        continue
                    if macos_authority and name not in allowed:
                        allowed.append(name)
                if not browser_authority:
                    # Keep an inherited parent surface out of a child packet;
                    # dispatch_tool also refuses these names, but omitting
                    # them prevents the model from spending calls on a
                    # capability that this WorkUnit did not request.
                    allowed[:] = [
                        name for name in allowed
                        if not str(name).startswith("browser.")
                    ]
                # Browser verification is evidence-bearing, not a decorative
                # optional tool.  A scoped action Goal cannot seal after an
                # initial failed predicate or a bare click; it must reload the
                # authorized page and obtain a later positive browser.verify.
                if is_goal and browser_acceptance and isinstance(completion_contract, Mapping):
                    completion_contract = dict(completion_contract)
                    required_browser = {
                        "browser.open", "browser.observe", "browser.find",
                        "browser.goto", "browser.verify",
                    }
                    if browser_authority.get("actions"):
                        required_browser.add("browser.click")
                    existing_required = completion_contract.get("required_successful_tools") or []
                    completion_contract["required_successful_tools"] = sorted(
                        {str(name).strip() for name in existing_required if str(name).strip()}
                        | required_browser
                    )
                    completion_contract["required_verified_tools"] = sorted(
                        {
                            *(
                                str(name).strip()
                                for name in (completion_contract.get("required_verified_tools") or [])
                                if str(name).strip()
                            ),
                            "browser.verify",
                        }
                    )
                    reminder = str(completion_contract.get("reminder") or "").strip()
                    completion_contract["reminder"] = (
                        reminder
                        + " Browser action acceptance additionally requires browser.open, "
                        "browser.observe, browser.find, browser.click, browser.goto, and "
                        "a later browser.verify whose verified value is true."
                    )[:1600]
                if is_goal and macos_authority and macos_authority.get("actions") and isinstance(completion_contract, Mapping):
                    completion_contract = dict(completion_contract)
                    required_macos = {
                        "macos.observe", "macos.find", "macos.lease.acquire",
                        "macos.press", "macos.verify", "macos.lease.release",
                    }
                    existing_required = completion_contract.get("required_successful_tools") or []
                    completion_contract["required_successful_tools"] = sorted(
                        {str(name).strip() for name in existing_required if str(name).strip()}
                        | required_macos
                    )
                    existing_verified = completion_contract.get("required_verified_tools") or []
                    completion_contract["required_verified_tools"] = sorted(
                        {str(name).strip() for name in existing_verified if str(name).strip()}
                        | {"macos.press", "macos.verify"}
                    )
                    reminder = str(completion_contract.get("reminder") or "").strip()
                    completion_contract["reminder"] = (
                        reminder
                        + " MacOS action acceptance additionally requires macos.observe, "
                        "macos.find, macos.lease.acquire, one exact macos.press with a "
                        "verified post-state, macos.verify, and macos.lease.release."
                    )[:1600]

            # A typed mutation WorkUnit does not need the full H-Web discovery
            # envelope.  Keep the provider request small and deterministic:
            # the bounded read/search doors plus the declared mutation and
            # verification doors.  Hawking still validates every invocation
            # through the same registry and completion contract; this only
            # removes unrelated schemas that can make a provider route spend
            # its entire bounded turn choosing among irrelevant capabilities.
            if (
                is_goal
                and isinstance(completion_contract, Mapping)
                and "repo.edit" in {
                    str(name).strip()
                    for name in (completion_contract.get("required_successful_tools") or [])
                    if str(name).strip()
                }
                and not browser_authority
                and not macos_authority
            ):
                bounded_goal_tools = {
                    "fs.search", "filesystem.search", "fs.read", "filesystem.read",
                    "source.search", "source.read", "repo.edit", "tests.run",
                    "git.status", "git.diff",
                }
                allowed = [name for name in allowed if name in bounded_goal_tools]
                for name in ("repo.edit", "tests.run"):
                    if name in bounded_goal_tools and name not in allowed:
                        allowed.append(name)

            def worker_view(record: Any) -> Dict[str, Any]:
                view = {
                    "worker_id": record.worker_id,
                    "workunit_id": record.workunit_id,
                    "goal_id": record.goal_id,
                    "goal_mode": record.goal_mode,
                    "model": record.worker_model,
                    "role": getattr(record, "role", "worker"),
                    "task_class": getattr(record, "task_class", "general"),
                    "worker_policy": getattr(record, "worker_policy", "auto"),
                    "plan_id": getattr(record, "plan_id", ""),
                    "worker_packet_path": getattr(record, "worker_packet_path", ""),
                    "state": record.state,
                    "worker_status": record.worker_status,
                    "worker_kill_reason": record.worker_kill_reason,
                    "parent_worker_id": record.parent_worker_id,
                    "parent_workunit_id": record.parent_workunit_id,
                    "child_worker_ids": list(record.child_worker_ids),
                    "last_heartbeat_at": record.worker_last_heartbeat_at,
                    "last_turn": record.last_turn,
                    "last_outcome": record.last_outcome,
                    "cost_usd": float(record.worker_cost_usd),
                    "provider_request_ids": list(record.provider_request_ids)[-8:],
                    "exact_next_action": str(record.exact_next_action or "")[:500],
                }
                if record.goal_id:
                    try:
                        from hawking.goal_surface import load_result
                        result = load_result(gateway_workspace, record.goal_id)
                        if isinstance(result, Mapping):
                            view["result"] = {
                                "status": result.get("status"),
                                "summary": result.get("summary"),
                                "output_excerpt": result.get("output_excerpt"),
                                "completion": result.get("completion"),
                            }
                    except Exception:
                        pass
                return view

            def result_for(name: str, *, ok: bool, value: Any = None,
                           error: Optional[str] = None) -> Any:
                return ToolResult(
                    tool=name,
                    invocation_id=f"hawking-{uuid.uuid4().hex[:12]}",
                    ok=ok,
                    value=value,
                    error=error,
                    mutation=READ_ONLY,
                    provenance={"owner": "hawkingd", "fabric": "workunit"},
                )

            def current_record() -> Any:
                if not current_worker_id:
                    raise ValueError("worker identity is unavailable for this delegation call")
                record = owner.find_by_worker_id(current_worker_id)
                if record is None:
                    raise ValueError("current Hawking worker is not present in the owner ledger")
                return record

            def target_record(arguments: Mapping[str, Any], *, allow_child: bool = True) -> Any:
                parent = current_record()
                target = str(arguments.get("worker_id") or current_worker_id).strip()
                record = owner.find_by_worker_id(target)
                if record is None:
                    try:
                        record = owner.load(target)
                    except Exception:
                        record = None
                if record is None:
                    raise ValueError("unknown Hawking worker")
                if record.worker_id != parent.worker_id:
                    permitted = allow_child and record.worker_id in set(parent.child_worker_ids)
                    if not permitted:
                        raise ValueError("worker action is limited to the current worker or a direct child")
                return record

            def worker_dispatch(name: str, arguments: Mapping[str, Any]) -> Any:
                try:
                    if name == "hawking.worker.status":
                        return result_for(name, ok=True, value=worker_view(target_record(arguments)))
                    if name == "hawking.worker.kill":
                        target = target_record(arguments)
                        reason = str(arguments.get("reason") or "child_requested").strip()[:300]
                        cancelled = owner.cancel(target.workunit_id)
                        if target.goal_id:
                            from hawking.goal_surface import update_goal
                            update_goal(
                                gateway_workspace,
                                target.goal_id,
                                status="CANCELLED",
                                phase="CANCELLED",
                                last_operator_action="worker_kill",
                                worker_kill_reason=reason,
                            )
                        return result_for(name, ok=True, value={
                            "action": "kill",
                            "reason": reason,
                            "worker": worker_view(type("Worker", (), cancelled)()),
                        })
                    if name == "hawking.worker.spawn":
                        parent = current_record()
                        objective = str(arguments.get("objective") or "").strip()
                        if not objective:
                            raise ValueError("worker.spawn requires objective")
                        if len(objective) > 2000:
                            raise ValueError("worker.spawn objective exceeds 2000 characters")
                        seen_workunits = {parent.workunit_id}
                        cursor = parent
                        while cursor.parent_workunit_id:
                            if cursor.parent_workunit_id in seen_workunits:
                                raise ValueError("worker parent chain is cyclic")
                            seen_workunits.add(cursor.parent_workunit_id)
                            try:
                                cursor = owner.load(cursor.parent_workunit_id)
                            except Exception:
                                break
                        # No fixed fan-out/depth/count ceiling: real process,
                        # provider, budget, dependency, and cycle guards are
                        # the admission boundaries.  An explicitly configured
                        # finite gateway concurrency remains enforceable by
                        # the shared RemoteWorkerAdmission lease.
                        child_model = str(
                            arguments.get("model") or "deepseek/deepseek-v4.1-flash"
                        ).strip()
                        if child_model.lower().startswith("openrouter:"):
                            child_model = child_model.split(":", 1)[1].strip()
                        if child_model not in child_models:
                            raise ValueError("worker.spawn model is outside Hawking's bounded remote roster")
                        child_mode = str(arguments.get("goal_mode") or "worker_research").strip()
                        if child_mode != "worker_research":
                            raise ValueError("worker.spawn only creates read-only worker_research children")
                        configured = max(0.0, float(policy.cost_policy.max_cost_usd))
                        if configured <= 0.0:
                            raise ValueError("worker.spawn is blocked because Hawking has no positive remote cost authority")
                        fabric_limit = configured
                        if policy.cost_policy.mission_limit_usd is not None:
                            fabric_limit = min(
                                fabric_limit,
                                max(0.0, float(policy.cost_policy.mission_limit_usd)),
                            )
                        requested_budget = arguments.get("budget_usd", min(0.50, configured))
                        try:
                            budget = float(requested_budget)
                        except (TypeError, ValueError):
                            raise ValueError("worker.spawn budget_usd must be numeric")
                        if budget <= 0.0 or budget > min(0.50, configured):
                            raise ValueError(
                                f"worker.spawn budget must be >0 and <= {min(0.50, configured):.6f}"
                            )
                        ledger_spent = sum(max(0.0, float(row.worker_cost_usd or 0.0)) for row in rows)
                        provider_spent = max(0.0, float(policy.cost_policy.spent_usd or 0.0))
                        reserved = 0.0
                        for row in rows:
                            if row.state not in {"QUEUED", "RUNNING", "CHECKPOINTING", "COMPACTING"}:
                                continue
                            try:
                                contract_doc = json.loads(Path(row.contract_path).read_text(encoding="utf-8"))
                            except (OSError, ValueError, json.JSONDecodeError):
                                contract_doc = {}
                            try:
                                authorized = max(0.0, float(contract_doc.get("budget_authorized_usd") or 0.0))
                            except (TypeError, ValueError):
                                authorized = 0.0
                            reserved += max(0.0, authorized - max(0.0, float(row.worker_cost_usd or 0.0)))
                        if max(ledger_spent, provider_spent) + reserved + budget > fabric_limit:
                            raise ValueError("worker.spawn would exceed Hawking's remaining fabric budget")
                        acceptance = arguments.get("acceptance")
                        if not isinstance(acceptance, list) or not acceptance:
                            acceptance = [
                                "return a structured evidence packet",
                                "do not mutate the worktree",
                            ]
                        from hawking.goal_surface import create_goal, dispatch_goal, update_goal
                        child = create_goal(
                            gateway_workspace,
                            objective=objective,
                            acceptance=[str(item)[:500] for item in acceptance[:8]],
                            authority_boundary=(
                                "read-only research; may delegate bounded read-only children; "
                                "no source, receipt, git, test, or production mutation"
                            ),
                            flash_resource_rule=(
                                "Flash/Pulsar protected; remote child roster only; "
                                "Hawking cost gate remains authoritative"
                            ),
                            completion_contract=(
                                "structured research packet after real read-only evidence"
                            ),
                            resident=child_model,
                            goal_mode="worker_research",
                            budget_usd=budget,
                            parent_worker_id=parent.worker_id,
                            parent_workunit_id=parent.workunit_id,
                            source="hawking_worker_spawn",
                        )
                        dispatched = dispatch_goal(
                            gateway_workspace, child, model=child_model, background=True
                        )
                        child_owner = dispatched.get("owner") or {}
                        child_wid = str(dispatched.get("workunit_id") or child.get("workunit_id") or "")
                        child_worker_id = str(child_owner.get("worker_id") or "")
                        if not child_worker_id:
                            raise ValueError("child owner did not return a Hawking worker id")
                        parent = owner.load(parent.workunit_id)
                        if child_worker_id not in parent.child_worker_ids:
                            parent.child_worker_ids.append(child_worker_id)
                            owner.save(parent)
                        update_goal(
                            gateway_workspace,
                            child["goal_id"],
                            parent_worker_id=parent.worker_id,
                            parent_workunit_id=parent.workunit_id,
                            spawned_by_worker_id=parent.worker_id,
                            budget_authorized_usd=budget,
                        )
                        return result_for(name, ok=True, value={
                            "action": "spawn",
                            "worker_id": child_worker_id,
                            "workunit_id": child_wid,
                            "goal_id": child["goal_id"],
                            "model": child_model,
                            "goal_mode": "worker_research",
                            "budget_usd": budget,
                            "state": str(child_owner.get("state") or "QUEUED"),
                            "parent_worker_id": parent.worker_id,
                        })
                    return result_for(name, ok=False, error="unknown Hawking worker action")
                except Exception as exc:
                    return result_for(
                        name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )

            def dispatch_tool(name: str, arguments: Mapping[str, Any]) -> Any:
                if name == "repo.edit":
                    # Provider cognition may run concurrently.  Serialize only
                    # the Hawking-owned effect, not the network/model turn;
                    # otherwise one slow provider request stalls every other
                    # WorkUnit at the daemon boundary.
                    with remote_mutation_lock:
                        return run_builder_tool(
                            name,
                            dict(arguments),
                            engine=stores.get("engine"),
                            require_tests=True,
                            mutation_bundle=(
                                mutation_bundle if isinstance(mutation_bundle, Mapping) else None
                            ),
                            bundle_state=bundle_state,
                        )
                if name in worker_tool_names:
                    return worker_dispatch(name, arguments)
                if name in LOCAL_TOOLS:
                    scoped = dict(arguments)
                    # The browser transport is not allowed to widen the Goal
                    # projection. H-Web session membership is the only source
                    # of detached Goal ids for main-chat cognition.
                    if name == "goals.list" and body.get("hawking_web"):
                        scoped["goal_ids"] = list(web_goal_ids)
                        scoped["session_id"] = web_session_id
                    if name in {"artifact.search", "memory.search"} and body.get("hawking_web"):
                        scoped["session_id"] = web_session_id
                    # Keep the post-mutation verification observation together
                    # with the other repository observations when a provider
                    # calls these direct doors.  Read/search cognition itself
                    # remains fully parallel.
                    critical = name in {"tests.run", "git.status", "git.diff"}
                    guard = remote_mutation_lock if critical else nullcontext()
                    with guard:
                        local_result = run_local_tool(
                            name,
                            scoped,
                            cache=stores.get("cache"),
                            knowledge=stores.get("knowledge"),
                            workspace=gateway_workspace,
                            runtime=runtime_snapshot,
                        )
                    if name == "capabilities.status" and getattr(local_result, "ok", False):
                        value = getattr(local_result, "value", None)
                        if isinstance(value, Mapping):
                            value = dict(value)
                            value["builder_doors"] = {
                                "repo.edit": {
                                    "canonical": True,
                                    "state": (
                                        "available_on_explicit_mutation_turn"
                                        if is_web_builder
                                        else "unavailable_in_read_turn"
                                    ),
                                    "owner": "hawking.chat_tools.run_builder_tool",
                                    "gate": "red_before_green",
                                },
                            }
                            local_result.value = value
                    return local_result
                if name.startswith("browser."):
                    if not browser_authority:
                        return result_for(name, ok=False, error="browser capability is not admitted for this Goal")
                    scoped = dict(arguments)
                    expected_session = str(browser_authority["browser_session_id"])
                    supplied_session = str(scoped.get("browser_session_id") or "").strip()
                    if supplied_session and supplied_session != expected_session:
                        return result_for(name, ok=False, error="browser session is owned by another Goal")
                    scoped["browser_session_id"] = expected_session
                    if name in {"browser.open", "browser.goto"} and scoped.get("url"):
                        from urllib.parse import urlsplit

                        parsed = urlsplit(str(scoped["url"]))
                        origin = (
                            "file://" if parsed.scheme == "file"
                            else f"{parsed.scheme}://{parsed.netloc}"
                        )
                        if origin not in set(browser_authority.get("origins") or []):
                            return result_for(
                                name,
                                ok=False,
                                error="browser navigation origin is outside this Goal authority",
                            )
                    return active_registry.invoke(name, scoped)
                if name.startswith("macos."):
                    if not macos_authority:
                        return result_for(name, ok=False, error="macOS capability is not admitted for this Goal")
                    scoped = dict(arguments)
                    return active_registry.invoke(name, scoped)
                return active_registry.invoke(name, arguments)

            tool_budget_ceiling, tool_budget_scope = _hweb_build_tool_budget(
                body, is_web_builder=is_web_builder, is_goal=is_goal,
            )

            def persist_build_state(
                    trace: Any, completion: Optional[Mapping[str, Any]] = None,
                    *, calls_used: Optional[int] = None) -> None:
                """Persist the bounded Build checkpoint in the H-Web session."""
                if not is_web_builder or web_session is None:
                    return
                state = _build_tool_budget_state(
                    tool_budget_ceiling, trace, completion,
                )
                if calls_used is not None:
                    state["used"] = max(0, int(calls_used))
                    state["remaining"] = max(
                        0, tool_budget_ceiling - state["used"]
                    )
                state["scope"] = tool_budget_scope
                state["turn_id"] = str(request.request_id)
                state["pending_mutation_bundle"] = (
                    dict(bundle_state) if state["state"] != "COMPLETE" else {}
                )
                ui = dict(getattr(web_session, "ui", {}) or {})
                ui["build_state"] = state
                web_session.ui = ui
                _web_store().save(web_session)

            def execute_provider_mutation_proposal(proposal: Any) -> Dict[str, Any]:
                """Compile provider data, then emit one Hawking evidence chain."""
                authority_lease = str(
                    body.get("hawking_mutation_lease")
                    or mutation_lease
                    or ""
                ).strip()
                raw_bundle = body.get("hawking_mutation_bundle")
                raw_scope = (
                    raw_bundle.get("allowed_files")
                    or raw_bundle.get("allowed_paths")
                    or raw_bundle.get("focus_files")
                    or []
                ) if isinstance(raw_bundle, Mapping) else []
                allowed_scope = (
                    list(raw_scope) if isinstance(raw_scope, (list, tuple, set, frozenset))
                    else [str(raw_scope)] if raw_scope else []
                )
                bundle_tests = [
                    str(item).strip()
                    for item in (
                        raw_bundle.get("focus_tests")
                        or raw_bundle.get("required_tests")
                        or []
                    )
                    if str(item).strip()
                ] if isinstance(raw_bundle, Mapping) else []
                execution_proposal = proposal
                if isinstance(proposal, Mapping):
                    # Compact workers return semantic intent, not a canonical
                    # mutation bundle. Compile the anchored intent before the
                    # Goal completeness gate so the gate can see the locally
                    # reconstructed source operation. Previously the compact
                    # object reached ``validate_mutation_bundle`` as
                    # ``{anchor, op, body}``, which has no ``operations``
                    # member and was rejected as MISSING_SOURCE_MUTATION even
                    # though Hawking had already resolved the anchor.
                    if str(
                        proposal.get("s") or proposal.get("status") or ""
                    ).strip().upper() == "MUTATE" and proposal.get("anchor"):
                        from .mutation import normalize_mutation_result
                        try:
                            compiled = normalize_mutation_result(
                                {
                                    **dict(proposal),
                                    "authority": {
                                        "capabilities": list(allowed) if write else [],
                                        "mutation_lease": authority_lease if write else None,
                                    },
                                },
                                canonical_root=str(gateway_workspace),
                                goal_id=str(body.get("hawking_goal_id") or ""),
                                workunit_id=str(body.get("hawking_workunit_id") or ""),
                            )
                        except Exception as exc:
                            return {
                                "status": "rejected",
                                "reason": "COMPACT_INTENT_REJECTED",
                                "detail": type(exc).__name__ + ": " + str(exc)[:300],
                                "applied": False,
                            }
                        execution_proposal = compiled.to_dict()
                    # The workspace is Hawking authority, not provider data.
                    # Some otherwise valid typed proposals echo a synthetic
                    # root from their prompt and are rejected as
                    # WRONG_WORKSPACE before the engine can validate the real
                    # paths. Bind the canonical root to the admitted gateway
                    # workspace; scope, anchors, digests, and rollback remain
                    # fail-closed below.
                    proposal_map = dict(
                        execution_proposal
                        if isinstance(execution_proposal, Mapping)
                        else proposal
                    )
                    proposal_map["canonical_root"] = str(gateway_workspace)
                    execution_proposal = proposal_map
                if bundle_tests and isinstance(execution_proposal, Mapping):
                    # Bind the already-admitted focused tests before the
                    # canonical engine transaction. A provider may omit them,
                    # but it cannot replace or widen this Goal-owned list.
                    # Preserve the canonical-root normalization above. Using
                    # the raw provider mapping here reintroduced a stale or
                    # synthetic root whenever focused tests were present,
                    # causing valid Auto proposals to fail WRONG_WORKSPACE.
                    proposal_map = dict(
                        execution_proposal
                        if isinstance(execution_proposal, Mapping)
                        else proposal
                    )
                    # Focused tests are Goal-owned admission data, not a
                    # provider-editable field. Always bind the admitted path
                    # list so shell-shaped provider text (for example
                    # ``python -m pytest ...``) cannot replace the canonical
                    # test identities or widen the verification scope.
                    proposal_map["required_tests"] = list(bundle_tests)
                    execution_proposal = proposal_map
                if isinstance(raw_bundle, Mapping) and isinstance(
                    execution_proposal, Mapping
                ) and any(
                    bool(raw_bundle.get(key))
                    for key in (
                        "require_source_mutation",
                        "require_regression_test_mutation",
                        "require_focused_test_invocation",
                    )
                ):
                    # Direct builder calls already pass through this
                    # completeness gate. Typed provider proposals must cross
                    # the same boundary before the engine, otherwise a
                    # source-only proposal can be executed while the Goal's
                    # required regression operation is silently absent.
                    proposal_operations = (
                        execution_proposal.get("operations")
                        or execution_proposal.get("edits")
                        or execution_proposal.get("operation")
                        or []
                    )
                    if isinstance(proposal_operations, Mapping):
                        proposal_operations = [proposal_operations]
                    proposal_tests = (
                        execution_proposal.get("required_tests")
                        or execution_proposal.get("tests")
                        or []
                    )
                    bundle_verdict = validate_mutation_bundle(
                        proposal_operations,
                        proposal_tests if isinstance(proposal_tests, list) else [proposal_tests],
                        bundle=raw_bundle,
                    )
                    if not bundle_verdict.get("bundle_complete"):
                        return {
                            "status": "rejected",
                            "reason": (
                                ";".join(bundle_verdict.get("missing_requirements") or [])
                                or "MUTATION_BUNDLE_INCOMPLETE"
                            ),
                            "applied": False,
                            "bundle_complete": False,
                            "missing_requirements": list(
                                bundle_verdict.get("missing_requirements") or []
                            ),
                            "test_mutation_paths": list(
                                bundle_verdict.get("test_mutation_paths") or []
                            ),
                            "allowed_files": list(
                                bundle_verdict.get("allowed_files") or []
                            ),
                            "preserved_operations": list(
                                bundle_verdict.get("source_operations") or []
                            ),
                            "proposal": dict(execution_proposal),
                        }
                # The provider request is intentionally outside this lock.  A
                # proposal is the point where model data becomes an effect;
                # serialize that short Hawking-owned transaction and its
                # required verification without serializing model inference.
                with remote_mutation_lock:
                    execution = execute_mutation_proposal(
                        execution_proposal,
                        engine=stores.get("engine"),
                        authority={
                            "workspace_root": str(gateway_workspace),
                            "goal_id": str(body.get("hawking_goal_id") or ""),
                            "workunit_id": str(body.get("hawking_workunit_id") or ""),
                            "capabilities": list(allowed) if write else [],
                            "mutation_lease": authority_lease if write else None,
                            "allowed_paths": allowed_scope,
                        },
                    )
                if not isinstance(execution, dict):
                    return {"status": "rejected", "reason": "INVALID_EXECUTION_PACKET", "applied": False}
                status = str(execution.get("status") or "").lower()
                inner = execution.get("execution")
                # Keep projection tolerant of the canonical engine packet
                # shape.  Older adapters surfaced the engine status/applied
                # flags only under ``execution``; treating that packet as
                # rejected here makes a real APPLIED intent invisible to the
                # WorkUnit completion contract.
                inner_status = (
                    str(inner.get("status") or "").lower()
                    if isinstance(inner, Mapping) else ""
                )
                effective_status = status or inner_status
                effective_applied = execution.get("applied")
                if effective_applied is None and isinstance(inner, Mapping):
                    effective_applied = inner.get("applied")
                proposal_doc = execution.get("proposal")
                if not isinstance(proposal_doc, Mapping):
                    proposal_doc = {}
                required_tests = [
                    str(item).strip()
                    for item in (proposal_doc.get("required_tests") or [])
                    if str(item).strip()
                ]
                if not required_tests and isinstance(raw_bundle, Mapping):
                    # The provider may omit the verification list while the
                    # admitted WorkUnit bundle already binds focused tests.
                    # Use that Hawking-owned declaration as the fallback; the
                    # provider never gets to widen it or skip tests.run.
                    required_tests = [
                        str(item).strip()
                        for item in (
                            bundle_tests
                            or []
                        )
                        if str(item).strip()
                    ]
                post_trace: list[Dict[str, Any]] = []

                def result_dict(value: Any) -> Dict[str, Any]:
                    if hasattr(value, "to_dict"):
                        try:
                            raw = value.to_dict()
                            return dict(raw) if isinstance(raw, Mapping) else {"ok": False, "value": raw}
                        except Exception as exc:
                            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    return {
                        "ok": bool(getattr(value, "ok", False)),
                        "value": getattr(value, "value", None),
                        "error": getattr(value, "error", None),
                    }

                if effective_status in {"accepted", "unproven"} and effective_applied is not False:
                    # Engine.apply_typed_mutation is the canonical repo.edit
                    # owner. Project its result under the ordinary receipt
                    # name so old WorkUnit completion readers remain valid.
                    repo_value = dict(inner) if isinstance(inner, Mapping) else {}
                    repo_value.update({
                        "status": effective_status,
                        "applied": True if effective_applied is None else effective_applied,
                        "rolled_back": execution.get(
                            "rolled_back",
                            inner.get("rolled_back", False)
                            if isinstance(inner, Mapping) else False,
                        ),
                        "bundle_complete": True,
                        "bundle_operations": list(proposal_doc.get("operations") or []),
                        "tests": required_tests,
                    })
                    post_trace.append({
                        "tool": "repo.edit",
                        "dispatched": True,
                        "ok": True,
                        "result": {"ok": True, "value": repo_value},
                    })

                    # The engine already performs its proving validation. Run
                    # the canonical registry tests.run as a separate durable
                    # observation so the WorkUnit contract records the exact
                    # repo.edit -> tests.run -> git.status -> git.diff chain.
                    if required_tests:
                        try:
                            test_result = active_registry.invoke(
                                "tests.run",
                                {
                                    "runner": "pytest",
                                    "root": ".",
                                    "paths": required_tests,
                                    "timeout_s": 900.0,
                                },
                            )
                            test_observation = result_dict(test_result)
                            post_trace.append({
                                "tool": "tests.run",
                                "dispatched": True,
                                "ok": bool(test_observation.get("ok")),
                                "result": test_observation,
                            })
                        except Exception as exc:
                            post_trace.append({
                                "tool": "tests.run",
                                "dispatched": False,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    for name, args in (
                        ("git.status", {"path": str(gateway_workspace)}),
                        ("git.diff", {"path": str(gateway_workspace)}),
                    ):
                        try:
                            observed = result_dict(active_registry.invoke(name, args))
                            post_trace.append({
                                "tool": name,
                                "dispatched": True,
                                "ok": bool(observed.get("ok")),
                                "result": observed,
                            })
                        except Exception as exc:
                            post_trace.append({
                                "tool": name,
                                "dispatched": False,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                if post_trace:
                    execution["post_mutation_trace"] = post_trace
                    execution["verification_receipts"] = [
                        {"tool": row.get("tool"), "ok": row.get("ok")}
                        for row in post_trace
                    ]
                return execution

            context: Dict[str, Any] = {
                "tool_registry": active_registry,
                "hawking_tools": True,
                "tool_write_authority": write,
                "mutation_lock_held": write,
                "allowed_tool_names": allowed,
                "extra_tool_schemas": extra_schemas,
                "extra_tool_names": list(worker_tool_names) if is_hawking_worker and not is_web_builder else [],
                "mutation_plan": dict(mutation_plan),
                "tool_dispatch": dispatch_tool,
                "mutation_proposal_executor": (
                    execute_provider_mutation_proposal
                    if write and stores.get("engine") is not None else None
                ),
                # Explicit H-Web BUILD/Goal turns get the bounded engineering
                # lane; ordinary chat and native-action lanes keep their
                # existing limits.  The same ceiling is recorded below.
                "max_tool_calls": (
                    # A discrete Goal is advanced through a bounded tool slice
                    # per owner turn. The first call may be the mandatory
                    # Hawking-owned grounding read; leaving only one call here
                    # starves the provider before it can inspect, propose, or
                    # emit verification evidence. The WorkUnit driver still
                    # persists each receipt and rotates the request boundary.
                    max(2, min(tool_budget_ceiling, 8))
                    if (is_goal and not is_web_builder) else (
                        tool_budget_ceiling
                        if is_web_builder else (
                        12
                        if macos_authority.get("actions")
                        else int(os.environ.get("HAWKING_REMOTE_MAX_TOOL_CALLS", "16"))
                        )
                    )
                ),
                "tool_budget": {
                    "ceiling": tool_budget_ceiling,
                    "scope": tool_budget_scope,
                },
                "timeout": policy.timeout_s,
                "completion_contract": completion_contract,
                "prior_tool_evidence": goal_prior_tool_evidence(),
                # A Goal proposal must be grounded by one real Hawking-owned
                # read before the provider serializes its mutation packet.
                # Without this, providers commonly put fs.read inside the
                # proposal's operations array; the mutation compiler correctly
                # drops it, but the WorkUnit then dead-letters without ever
                # seeing the focused source.
                "require_initial_read": bool(is_goal),
                "fresh_provider_session": bool(
                    body.get("hawking_provider_session_cold")
                ),
            }
            capability_trace = {
                "turn_id": str(request.request_id),
                "session_id": str(body.get("hawking_web_session_id") or ""),
                "workspace_id": str(body.get("hawking_web_workspace_id") or ""),
                "requested_model": str(body.get("model") or ""),
                "resolved_model": str(getattr(provider, "model_id", "") or ""),
                "provider": "openrouter",
                "effect_ceiling": "MUTATE" if write else "READ",
                "capability_intent": str(body.get("hawking_capability_intent") or ""),
                "projection_owner": "hawkingd.remote_hawking_completion",
                "tool_budget_ceiling": tool_budget_ceiling,
                "tool_budget_scope": tool_budget_scope,
            }
            if write:
                # Do not hold the repository mutation lock across provider
                # inference.  The old scope made one 120s upstream timeout
                # block every other admitted WorkUnit and presented as a
                # daemon/HTTP-200 failure.  Direct mutation and proposal
                # execution acquire the same lock at their effect boundary.
                schemas, projected = provider._workunit_tool_schemas(context)
                if not schemas:
                    return provider.generate(request, timeout=policy.timeout_s)
                if stream:
                    events = provider.generate_stream_with_workunit_tools(
                        request, context, tuple(schemas), projected
                    )
                    def _stream_with_build_state():
                        for item in events:
                            if (
                                is_web_builder
                                and isinstance(item, Mapping)
                                and item.get("type") == "done"
                            ):
                                completion = item.get("hawking_completion")
                                trace = item.get("tool_trace") or []
                                persist_build_state(
                                    trace,
                                    completion if isinstance(completion, Mapping) else None,
                                    calls_used=item.get("calls_used"),
                                )
                            yield item
                    return _stream_with_build_state()
                result, _trace = provider._generate_with_workunit_tools(
                    request, context, tuple(schemas), projected
                )
                raw = getattr(result, "raw", None)
                completion = raw.get("hawking_completion") if isinstance(raw, Mapping) else None
                persist_build_state(
                    _trace,
                    completion if isinstance(completion, Mapping) else None,
                    calls_used=(
                        completion.get("calls_used")
                        if isinstance(completion, Mapping) else None
                    ),
                )
                if isinstance(getattr(result, "raw", None), dict):
                    result.raw["hawking_capability_trace"] = {
                        **capability_trace,
                        "resolved_tools": list(projected),
                        "provider_tool_count": len(schemas),
                        "tool_budget": {
                            "ceiling": tool_budget_ceiling,
                            "scope": tool_budget_scope,
                        },
                    }
                return result
            schemas, projected = provider._workunit_tool_schemas(context)
            if not schemas:
                return provider.generate(request, timeout=policy.timeout_s)
            if stream:
                events = provider.generate_stream_with_workunit_tools(
                    request, context, tuple(schemas), projected
                )
                def _stream_with_build_state():
                    for item in events:
                        if (
                            is_web_builder
                            and isinstance(item, Mapping)
                            and item.get("type") == "done"
                        ):
                            completion = item.get("hawking_completion")
                            trace = item.get("tool_trace") or []
                            persist_build_state(
                                trace,
                                completion if isinstance(completion, Mapping) else None,
                                calls_used=item.get("calls_used"),
                            )
                        yield item
                return _stream_with_build_state()
            result, _trace = provider._generate_with_workunit_tools(
                request, context, tuple(schemas), projected
            )
            raw = getattr(result, "raw", None)
            completion = raw.get("hawking_completion") if isinstance(raw, Mapping) else None
            persist_build_state(
                _trace,
                completion if isinstance(completion, Mapping) else None,
                calls_used=(
                    completion.get("calls_used")
                    if isinstance(completion, Mapping) else None
                ),
            )
            if isinstance(getattr(result, "raw", None), dict):
                result.raw["hawking_capability_trace"] = {
                    **capability_trace,
                    "resolved_tools": list(projected),
                    "provider_tool_count": len(schemas),
                    "tool_budget": {
                        "ceiling": tool_budget_ceiling,
                        "scope": tool_budget_scope,
                    },
                }
            return result

        def _remote_chat(
            self,
            body: Mapping[str, Any],
            model: str,
            *,
            auto_route: Optional[Mapping[str, Any]] = None,
        ) -> None:
            policy = gateway_policy()
            lease = acquire_remote_lease(body, model)
            provider = None
            request = None
            try:
                request = self._remote_request(body, model)
                provider = self._remote_provider(model)
                worker_mode = body.get("hawking_worker_mode")
                goal_mode = body.get("hawking_goal_mode")
                use_hawking_loop = bool(
                    goal_mode == "discrete_goal"
                    or worker_mode in {
                        "read_only_research", "reason_only", "builder_interactive",
                    }
                )
                if not body.get("stream"):
                    result = (
                        self._remote_hawking_completion(
                            body, provider, request, policy,
                            mutation_lease=getattr(lease, "lease_id", None),
                        )
                        if use_hawking_loop else
                        provider.generate(request, timeout=policy.timeout_s)
                    )
                    body_out = _remote_chat_payload(
                        result,
                        model=(str(auto_route.get("requested_model"))
                               if isinstance(auto_route, Mapping) else model),
                        request_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        provider=provider,
                        client_label=self._client_label(),
                        auto_route=auto_route,
                    )
                    session_meta = _web_record_assistant(
                        body,
                        body_out["choices"][0]["message"].get("content") or "",
                        body_out.get("hawking"),
                    )
                    if session_meta is not None:
                        body_out["hawking"]["session"] = session_meta
                    return self._send(200, body_out)

                # H-Web keeps the provider's real SSE transport even when a
                # turn calls Hawking.  The stream-aware loop forwards provider
                # deltas, pauses only to execute admitted tool calls, then
                # continues the same bounded conversation with the provider.
                if use_hawking_loop:
                    stream_events = self._remote_hawking_completion(
                        body, provider, request, policy, stream=True,
                        mutation_lease=getattr(lease, "lease_id", None),
                    )
                    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    response_model = (
                        str(auto_route.get("requested_model"))
                        if isinstance(auto_route, Mapping) else model
                    )
                    include_usage = bool(
                        isinstance(body.get("stream_options"), Mapping)
                        and body["stream_options"].get("include_usage")
                    )
                    self._begin_sse()
                    text_parts: List[str] = []
                    done: Dict[str, Any] = {}
                    tool_events: List[Dict[str, Any]] = []
                    try:
                        self._sse(self._chat_stream_chunk(
                            request_id=request_id,
                            model=response_model,
                            delta={"role": "assistant"},
                        ))
                        for item in stream_events:
                            if item.get("type") == "delta":
                                delta: Dict[str, Any] = {}
                                text = visible_text(item.get("text") or "")
                                if text:
                                    delta["content"] = text
                                    text_parts.append(text)
                                if item.get("tool_calls"):
                                    delta["tool_calls"] = item["tool_calls"]
                                if delta:
                                    self._sse(self._chat_stream_chunk(
                                        request_id=request_id,
                                        model=response_model,
                                        delta=delta,
                                    ))
                            elif item.get("type") == "tool" and isinstance(item.get("trace"), Mapping):
                                tool_events.append(dict(item["trace"]))
                            elif item.get("type") == "done":
                                done = dict(item)
                        receipt = done.get("receipt")
                        hawking_meta: Dict[str, Any] = {
                            "gateway": "hawkingd",
                            "runtime": "remote",
                            "provider": "openrouter",
                            "client": self._client_label(),
                            "streaming": "hawking_tool_loop",
                            "remote_cognition": receipt,
                            "tools_used": [
                                {
                                    "tool": str(row.get("tool") or ""),
                                    "ok": bool(row.get("ok")),
                                    "dispatched": bool(row.get("dispatched")),
                                }
                                for row in (done.get("tool_trace") or tool_events)
                                if isinstance(row, Mapping)
                            ],
                            "tool_admission": dict(done.get("tool_admission") or {}),
                            **({"auto_route": dict(auto_route)}
                               if isinstance(auto_route, Mapping) else {}),
                        }
                        capability_trace = {
                            "turn_id": str(request.request_id),
                            "session_id": str(body.get("hawking_web_session_id") or ""),
                            "workspace_id": str(body.get("hawking_web_workspace_id") or ""),
                            "requested_model": str(body.get("model") or ""),
                            "resolved_model": str(getattr(provider, "model_id", "") or ""),
                            "provider": "openrouter",
                            "effect_ceiling": "READ",
                            "capability_intent": str(body.get("hawking_capability_intent") or ""),
                            "projection_owner": "hawkingd.remote_hawking_completion",
                            "provider_tool_count": int(
                                (done.get("tool_admission") or {}).get("tool_count") or 0
                            ),
                        }
                        hawking_meta["capability_trace"] = capability_trace
                        session_meta = _web_record_assistant(
                            body, "".join(text_parts), hawking_meta
                        )
                        if session_meta is not None:
                            hawking_meta["session"] = session_meta
                        self._sse(self._chat_stream_chunk(
                            request_id=request_id,
                            model=response_model,
                            delta={},
                            finish_reason=str(done.get("finish_reason") or "stop"),
                            usage=(done.get("usage") if include_usage else None),
                            hawking=hawking_meta,
                        ))
                        self._sse("[DONE]")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        provider.cancel(request.request_id)
                    return

                request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                response_model = (
                    str(auto_route.get("requested_model"))
                    if isinstance(auto_route, Mapping) else model
                )
                include_usage = bool(
                    isinstance(body.get("stream_options"), Mapping)
                    and body["stream_options"].get("include_usage")
                )
                self._begin_sse()
                self._sse(self._chat_stream_chunk(
                    request_id=request_id,
                    model=response_model,
                    delta={"role": "assistant"},
                ))
                done: Dict[str, Any] = {}
                text_parts: List[str] = []
                try:
                    for item in provider.generate_stream(request, timeout=policy.timeout_s):
                        if item.get("type") == "delta":
                            delta: Dict[str, Any] = {}
                            if item.get("text"):
                                content = visible_text(item["text"])
                                delta["content"] = content
                                text_parts.append(content)
                            if item.get("tool_calls"):
                                delta["tool_calls"] = item["tool_calls"]
                            if delta:
                                self._sse(self._chat_stream_chunk(
                                    request_id=request_id,
                                    model=response_model,
                                    delta=delta,
                                ))
                        elif item.get("type") == "done":
                            done = dict(item)
                    usage = _remote_usage(type("StreamResult", (), {
                        "usage": done.get("usage") or {},
                    })())
                    receipt = done.get("receipt")
                    hawking_meta: Dict[str, Any] = {
                        "gateway": "hawkingd",
                        "runtime": "remote",
                        "provider": "openrouter",
                        "client": self._client_label(),
                        "remote_cognition": receipt,
                        **({"auto_route": dict(auto_route)}
                           if isinstance(auto_route, Mapping) else {}),
                    }
                    session_meta = _web_record_assistant(
                        body, "".join(text_parts), hawking_meta
                    )
                    if session_meta is not None:
                        hawking_meta["session"] = session_meta
                    self._sse(self._chat_stream_chunk(
                        request_id=request_id,
                        model=response_model,
                        delta={},
                        finish_reason=str(done.get("finish_reason") or "stop"),
                        usage=(usage if include_usage else None),
                        hawking=hawking_meta,
                    ))
                    self._sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    if request is not None and provider is not None:
                        provider.cancel(request.request_id)
                except Exception as exc:
                    if request is not None and provider is not None:
                        provider.cancel(request.request_id)
                    status, _headers, details = self._remote_error(exc)
                    self._sse({"error": {
                        "message": str(exc)[:1200],
                        "type": details.get("type", "gateway_error"),
                        "code": details.get("code"),
                        "status": status,
                    }})
                    self._sse("[DONE]")
            except Exception as exc:
                if self.wfile and not getattr(self, "close_connection", False):
                    return self._send_remote_error(exc)
                raise
            finally:
                if provider is not None:
                    remember_remote_receipt(provider)
                gateway_admission().release(lease.lease_id)

        def _remote_responses(
            self,
            body: Mapping[str, Any],
            model: str,
            *,
            auto_route: Optional[Mapping[str, Any]] = None,
        ) -> None:
            from .providers import GenerationResponse
            policy = gateway_policy()
            lease = acquire_remote_lease(body, model)
            provider = None
            request = None
            try:
                request = self._remote_request(body, model, responses=True)
                provider = self._remote_provider(model)
                response_id = f"resp_{uuid.uuid4().hex[:24]}"
                if not body.get("stream"):
                    result = provider.generate(request, timeout=policy.timeout_s)
                    return self._send(200, _remote_response_payload(
                        result,
                        model=(str(auto_route.get("requested_model"))
                               if isinstance(auto_route, Mapping) else model),
                        response_id=response_id,
                        provider=provider,
                        client_label=self._client_label(),
                        auto_route=auto_route,
                    ))

                self._begin_sse()
                self._sse({
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": (str(auto_route.get("requested_model"))
                                  if isinstance(auto_route, Mapping) else model),
                        "output": [],
                    },
                }, event="response.created")
                item_id = f"msg_{uuid.uuid4().hex[:16]}"
                text_parts: List[str] = []
                done: Dict[str, Any] = {}
                try:
                    for item in provider.generate_stream(request, timeout=policy.timeout_s):
                        if item.get("type") == "delta":
                            text = visible_text(item.get("text") or "")
                            if text:
                                text_parts.append(text)
                                self._sse({
                                    "type": "response.output_text.delta",
                                    "delta": text,
                                    "item_id": item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                }, event="response.output_text.delta")
                            for call in item.get("tool_calls") or []:
                                function = call.get("function") if isinstance(call, Mapping) else {}
                                arguments = function.get("arguments") if isinstance(function, Mapping) else ""
                                if arguments:
                                    self._sse({
                                        "type": "response.function_call_arguments.delta",
                                        "delta": str(arguments),
                                        "item_id": str(call.get("id") or ""),
                                        "output_index": 0,
                                    }, event="response.function_call_arguments.delta")
                        elif item.get("type") == "done":
                            done = dict(item)
                    stream_result = GenerationResponse(
                        text="".join(text_parts),
                        usage=dict(done.get("usage") or {}),
                        raw={"remote_cognition": done.get("receipt")},
                        provider="openrouter",
                        request_id=request.request_id,
                        finish_reason=str(done.get("finish_reason") or "stop"),
                    )
                    completed = _remote_response_payload(
                        stream_result,
                        model=(str(auto_route.get("requested_model"))
                               if isinstance(auto_route, Mapping) else model),
                        response_id=response_id,
                        provider=provider,
                        client_label=self._client_label(),
                        auto_route=auto_route,
                    )
                    self._sse({"type": "response.completed", "response": completed},
                              event="response.completed")
                    self._sse("[DONE]")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    if request is not None and provider is not None:
                        provider.cancel(request.request_id)
                except Exception as exc:
                    if request is not None and provider is not None:
                        provider.cancel(request.request_id)
                    status, _headers, details = self._remote_error(exc)
                    self._sse({"error": {
                        "message": str(exc)[:1200],
                        "type": details.get("type", "gateway_error"),
                        "code": details.get("code"),
                        "status": status,
                    }})
                    self._sse("[DONE]")
            except Exception as exc:
                if self.wfile and not getattr(self, "close_connection", False):
                    return self._send_remote_error(exc)
                raise
            finally:
                if provider is not None:
                    remember_remote_receipt(provider)
                gateway_admission().release(lease.lease_id)

        def do_POST(self) -> None:  # noqa: N802
            route = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
            preloaded_body = self._json_body()
            if preloaded_body is None:
                return
            web_goal_auto_route: Optional[Dict[str, Any]] = None
            if route == "/hawking/web/artifacts":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web artifact actions are local-only"}})
                try:
                    from .web_artifacts import (
                        artifact_path, pin_artifact, read as read_web_artifact,
                        save_as_note,
                    )
                    body = self._json_body()
                    action = str(body.get("action") or "").strip().lower()
                    artifact_id = str(body.get("artifact_id") or "").strip()
                    if not artifact_id:
                        raise ValueError("artifact_id is required")
                    if action == "pin":
                        result = pin_artifact(gateway_workspace, artifact_id)
                        return self._send(200, {
                            "schema": "hawking.web.artifact.action.v1",
                            "action": "pin",
                            "artifact": result,
                        })
                    if action in {"note", "save", "save_as_note"}:
                        result = save_as_note(
                            gateway_workspace, artifact_id, body.get("filename")
                        )
                        return self._send(200, {
                            "schema": "hawking.web.artifact.action.v1",
                            "action": "save_as_note",
                            "artifact": result,
                        })
                    if action == "reveal":
                        # The browser can request a Finder reveal, but it can
                        # never provide the path. Hawking resolves the checked
                        # content-addressed artifact before invoking macOS.
                        path = artifact_path(gateway_workspace, artifact_id)
                        subprocess.Popen(
                            ["open", "-R", str(path)],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        return self._send(200, {
                            "schema": "hawking.web.artifact.action.v1",
                            "action": "reveal",
                            "artifact_id": artifact_id,
                            "revealed": True,
                        })
                    if action == "read":
                        item = read_web_artifact(gateway_workspace, artifact_id)
                        return self._send(200, {"artifact": item["attachment"]})
                    raise ValueError("artifact action must be pin, save_as_note, reveal, or read")
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"artifact action unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/markdown":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web markdown export is local-only"}})
                try:
                    from .web_artifacts import MAX_WEB_ARTIFACT_BYTES, admit_bytes
                    body = self._json_body()
                    session_id = _web_session_id(body.get("session_id"))
                    content = body.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("markdown content is required")
                    payload = content.encode("utf-8")
                    if len(payload) > MAX_WEB_ARTIFACT_BYTES:
                        raise ValueError(
                            f"markdown export exceeds the {MAX_WEB_ARTIFACT_BYTES // (1024 * 1024)} MiB limit"
                        )
                    attachment = admit_bytes(
                        gateway_workspace,
                        filename=body.get("filename") or "hawking-plan.md",
                        payload=payload,
                        source_type="goal_result",
                        session_id=session_id,
                    )
                    attachment["kind"] = "markdown_export"
                    message_seq = _web_attach_to_last_assistant(
                        session_id, attachment
                    )
                    query = urllib.parse.urlencode({
                        "id": attachment["artifact_id"],
                        "content": "1",
                        "download": "1",
                    })
                    return self._send(201, {
                        "schema": "hawking.web.markdown.v1",
                        "session_id": session_id,
                        "message_seq": message_seq,
                        "attachment": attachment,
                        "download_url": f"/hawking/web/attachments?{query}",
                    })
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"markdown export unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/attachments":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web artifact admission is local-only"}})
                try:
                    # The JSON/base64 envelope is deliberately bounded before
                    # decoding so a browser cannot turn the upload door into
                    # an unbounded memory allocation.
                    from .web_artifacts import MAX_WEB_ARTIFACT_BYTES, admit_bytes
                    length = int(self.headers.get("Content-Length") or 0)
                    max_envelope = MAX_WEB_ARTIFACT_BYTES * 2 + 64 * 1024
                    if length > max_envelope:
                        return self._send(413, {"error": {
                            "message": "attachment upload envelope is too large"}})
                    body = self._json_body()
                    if "path" in body:
                        raise ValueError("attachment admission accepts bytes, not filesystem paths")
                    encoded = body.get("content_base64")
                    if not isinstance(encoded, str) or not encoded:
                        raise ValueError("content_base64 is required")
                    if len(encoded) > max_envelope:
                        return self._send(413, {"error": {
                            "message": "attachment upload envelope is too large"}})
                    try:
                        payload = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise ValueError("content_base64 is invalid") from exc
                    attachment = admit_bytes(
                        gateway_workspace,
                        filename=body.get("filename"),
                        payload=payload,
                        source_type=str(body.get("source_type") or "file_attachment"),
                        media_type=(
                            str(body.get("media_type"))
                            if body.get("media_type") else None
                        ),
                        session_id=(
                            str(body.get("session_id"))
                            if body.get("session_id") else None
                        ),
                        message_id=(
                            str(body.get("message_id"))
                            if body.get("message_id") else None
                        ),
                    )
                    return self._send(201, {"attachment": attachment})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"attachment admission unavailable: {type(exc).__name__}"}})
            if route.startswith("/s/"):
                try:
                    from .share_bridge import (
                        ShareError, act_share, parse_share_post, tunnel_action,
                    )
                    token, action = parse_share_post(route)
                    body = self._json_body()
                    root = stores.get("state_root") or (
                        getattr(repo, "root", None) if repo is not None else None
                    ) or os.getcwd()
                    if action.startswith("tunnel/"):
                        return self._send(200, tunnel_action(
                            root,
                            token,
                            action=action.split("/", 1)[1],
                            body=body,
                        ))
                    result = act_share(
                        root,
                        token,
                        action=action,
                        capability=str(body.get("capability") or ""),
                        message=str(body.get("message") or ""),
                        idempotency_key=str(body.get("idempotency_key") or ""),
                    )
                    return self._send(200, result)
                except ShareError as exc:
                    return self._send(exc.status, {"error": {"message": str(exc)}})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(500, {"error": {"message": f"share action unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/share":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web share creation is local-only"}})
                try:
                    from .share_bridge import ShareError, create_share, revoke_share
                    body = self._json_body()
                    root = stores.get("state_root") or (
                        getattr(repo, "root", None) if repo is not None else None
                    ) or os.getcwd()
                    action = str(body.get("action") or "create").strip().lower()
                    if action == "revoke":
                        return self._send(200, revoke_share(root, str(body.get("share_id") or "")))
                    session_id = body.get("session_id") or body.get("hawking_web_session_id")
                    session = _web_session(session_id) if session_id else None
                    ui = getattr(session, "ui", {}) if session is not None else {}
                    active_goal = str(
                        body.get("target_id") or (ui or {}).get("active_goal_id") or ""
                    ).strip()
                    target_type = str(body.get("target_type") or ("goal" if active_goal else "conversation")).strip().lower()
                    target_id = active_goal if target_type == "goal" else str(
                        body.get("target_id") or (session.id if session is not None else "")
                    ).strip()
                    if not target_id:
                        raise ShareError("a current conversation or Goal is required", 400)
                    permissions = body.get("permissions") or ["read"]
                    host = str(self.headers.get("Host") or "hawking.localhost:8014").split(",", 1)[0].strip()
                    if not re.fullmatch(r"[A-Za-z0-9.:-]+", host):
                        host = "hawking.localhost:8014"
                    return self._send(201, create_share(
                        root,
                        target_type=target_type,
                        target_id=target_id,
                        permissions=permissions,
                        ttl_seconds=body.get("ttl_seconds", 3600),
                        max_uses=body.get("max_uses", 0),
                        local_base=f"http://{host}",
                    ))
                except ShareError as exc:
                    return self._send(exc.status, {"error": {"message": str(exc)}})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(500, {"error": {"message": f"share creation unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/tunnel":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web Tunnel host control is local-only"}})
                try:
                    from .share_bridge import (
                        ShareError, create_tunnel_invitation, disconnect_tunnel,
                    )
                    body = self._json_body()
                    root = stores.get("state_root") or (
                        getattr(repo, "root", None) if repo is not None else None
                    ) or os.getcwd()
                    action = str(body.get("action") or "invite").strip().lower()
                    share_id = str(body.get("share_id") or "").strip()
                    capability = str(body.get("capability") or "")
                    if action == "invite":
                        return self._send(201, create_tunnel_invitation(
                            root,
                            share_id,
                            capability=capability,
                            ttl_seconds=body.get("ttl_seconds", 300),
                            direct_available=bool(body.get("direct_available", True)),
                            relay_available=bool(body.get("relay_available", True)),
                        ))
                    if action == "disconnect":
                        return self._send(200, disconnect_tunnel(
                            root,
                            share_id,
                            reconnect_secret=str(body.get("reconnect_secret") or ""),
                            request_id=str(body.get("request_id") or ""),
                        ))
                    raise ShareError("Tunnel host action must be invite or disconnect", 400)
                except ShareError as exc:
                    return self._send(exc.status, {"error": {"message": str(exc)}})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(500, {"error": {"message": f"Tunnel host control unavailable: {type(exc).__name__}"}})
            if route in {"/v1/auto/plan", "/api/auto/plan"}:
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Auto planning is local-only"}})
                try:
                    from .auto_orchestration import build_cognition_plan
                    body = self._json_body()
                    objective = str(body.get("objective") or "").strip()
                    if not objective:
                        raise ValueError("objective is required")
                    request = dict(body)
                    request["messages"] = [{"role": "user", "content": objective}]
                    plan = build_cognition_plan(
                        request,
                        allowed_models=body.get("allowed_models") or gateway_policy().allowed_models,
                        pinned_model=body.get("pinned_model"),
                        budget_usd=body.get("budget_usd"),
                        goal_id=str(body.get("goal_id") or ""),
                    )
                    return self._send(200, plan)
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(400, {"error": {"message": f"Auto plan unavailable: {type(exc).__name__}: {exc}"}})
            if route in {
                "/v1/goals/synthesize", "/api/goals/synthesize",
                "/v1/goals/consolidate", "/api/goals/consolidate",
            }:
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Goal coordination is local-only"}})
                try:
                    from .goal_surface import build_goal_consolidation, build_goal_synthesis
                    body = self._json_body()
                    ids = body.get("goal_ids") or body.get("ids")
                    if not isinstance(ids, list):
                        raise ValueError("goal_ids must be a list")
                    root = stores.get("state_root") or (
                        getattr(repo, "root", None) if repo is not None else None
                    ) or os.getcwd()
                    if route.endswith("/synthesize"):
                        return self._send(200, build_goal_synthesis(root, ids))
                    return self._send(200, build_goal_consolidation(
                        root,
                        ids,
                        objective=str(body.get("objective") or ""),
                        commit=bool(body.get("commit", False)),
                    ))
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(409, {"error": {"message": f"Goal coordination unavailable: {type(exc).__name__}: {exc}"}})
            if route == "/hawking/web/release":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "H-Web release control is local-only"}})
                try:
                    body = self._json_body()
                    action = str(body.get("action") or "").strip().lower()
                    if action == "stage":
                        return self._send(200, web_releases.stage())
                    if action == "activate":
                        return self._send(200, web_releases.activate())
                    if action == "rollback":
                        return self._send(200, web_releases.rollback())
                    raise ValueError("release action must be stage, activate, or rollback")
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(409, {"error": {"message": f"release action unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/events/ack":
                # ACK is intentionally not persisted in browser or daemon
                # state.  Reconnect resumes from the supplied monotonic cursor
                # against the canonical journal, so a lost tab cannot corrupt
                # an execution record.
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Hawking Web event acknowledgement is local-only"}})
                try:
                    body = self._json_body()
                    session = _web_session(body.get("session_id"))
                    sequence = int(body.get("sequence") or 0)
                    if sequence < 0:
                        raise ValueError("sequence must be non-negative")
                    return self._send(200, {"schema": "hawking.web.contract.v1", "session_id": session.id, "acknowledged_sequence": sequence})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
            if route == "/hawking/recovery":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "recovery control is local-only"}})
                try:
                    from .recovery import reconcile_recovery
                    body = self._json_body()
                    goal_id = str(body.get("goal_id") or "").strip() or None
                    return self._send(200, reconcile_recovery(gateway_workspace, goal_id=goal_id))
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(409, {"error": {"message": f"recovery reconciliation unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/session":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web session access is local-only"}})
                try:
                    body = self._json_body()
                    action = str(body.get("action") or "update").strip().lower()
                    if action == "delete":
                        return self._send(200, _delete_web_session(body.get("session_id")))
                    if action == "new":
                        session = _web_session(None)
                    else:
                        session = _web_session(body.get("session_id"))
                    if session is None:
                        raise ValueError("session could not be opened")
                    if action not in {"new", "update", "select"}:
                        raise ValueError("session action must be new, update, or select")
                    model = body.get("model")
                    if model is not None:
                        model = str(model).strip()[:160]
                        if model != "hawking/auto" and model != REVIEW_ONLY_MODEL_ID \
                                and "/" not in model:
                            raise ValueError("session model must be Auto or an admitted Hawking model id")
                        session.model = model
                    if not isinstance(session.ui, dict):
                        session.ui = {}
                    if body.get("route_scope") is not None:
                        scope = str(body.get("route_scope") or "auto").strip().lower()
                        if scope not in {"auto", "local", "cloud"}:
                            raise ValueError("route_scope must be auto, local, or cloud")
                        session.ui["route_scope"] = scope
                    if body.get("auto_scope") is not None:
                        auto_scope = str(body.get("auto_scope") or "cloud").strip().lower()
                        if auto_scope not in {"cloud", "local", "both"}:
                            raise ValueError("auto_scope must be cloud, local, or both")
                        session.ui["auto_scope"] = auto_scope
                    if "active_goal_id" in body:
                        requested_goal = str(body.get("active_goal_id") or "").strip()
                        goal_ids = _web_goal_ids(session.ui)
                        if requested_goal:
                            if not re.fullmatch(
                                r"GOAL-[A-Za-z0-9_-]{1,100}", requested_goal
                            ):
                                raise ValueError("active_goal_id is not a valid Hawking Goal id")
                            if requested_goal not in goal_ids:
                                from .goal_surface import load_goal
                                if not load_goal(str(gateway_workspace), requested_goal):
                                    raise ValueError("active_goal_id is not a known Hawking Goal")
                                goal_ids.append(requested_goal)
                            session.ui["goal_ids"] = goal_ids[-32:]
                        session.ui["active_goal_id"] = requested_goal[:100]
                    _web_store().save(session)
                    return self._send(200, _web_session_snapshot(session))
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
            if route == "/hawking/web/build/session":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking builder-session creation is local-only"}})
                try:
                    body = self._json_body()
                    workspace = Path(str(body.get("workspace") or "")).expanduser().resolve()
                    minted = _mint_web_builder_session(workspace)
                    # The token is returned only to the local launcher for an
                    # immediate browser navigation.  It is never written into
                    # a receipt, transcript, process list, or log.
                    return self._send(200, {
                        "schema": "hawking.web.builder_handoff.v1",
                        "handoff_url": "/hawking/web/build/"
                        f"{minted['session_id']}/{minted['token']}",
                    })
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
            if route == "/hawking/web/models":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web model state is local-only"}})
                try:
                    body = self._json_body()
                    action = str(body.get("action") or "").strip().lower()
                    model_id = str(body.get("model_id") or body.get("model") or "").strip()
                    if action not in {"add", "remove"}:
                        raise ValueError("model action must be add or remove")
                    if not model_id or "/" not in model_id:
                        raise ValueError("model_id must be a provider/model id")
                    # A search result or Hawking's bounded default roster is
                    # the admission proof for a selector mutation.  The
                    # provider credential remains inside the catalog call.
                    known = {
                        str(row.get("id")) for row in _web_model_rows(model_id)
                        if isinstance(row, Mapping) and row.get("id")
                    }
                    if model_id not in known:
                        raise ValueError("model_id is not present in Hawking model search")
                    enabled = _web_enabled_models()
                    if action == "add":
                        enabled.add(model_id)
                    else:
                        enabled.discard(model_id)
                    enabled = _web_save_enabled_models(enabled)
                    return self._send(200, {
                        "schema": "hawking.web.models.v1",
                        "models": _web_model_rows(""),
                        "enabled_models": sorted(enabled),
                        "changed": model_id,
                        "action": action,
                        "owner": "hawkingd",
                    })
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {
                        "message": f"model state unavailable: {type(exc).__name__}"}})
            if route == "/hawking/web/chat":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web chat access is local-only"}})
                try:
                    incoming = self._json_body()
                    bound_builder = self._trusted_builder_session()
                    sid = _web_session_id(incoming.get("session_id"))
                    session = (
                        bound_builder
                        if bound_builder is not None and bound_builder.id == sid
                        else _web_session(sid)
                    )
                    message = str(incoming.get("message") or "").strip()
                    if not message:
                        raise ValueError("message is required")
                    model = str(incoming.get("model") or "hawking/auto").strip()
                    if model != "hawking/auto" and model != REVIEW_ONLY_MODEL_ID \
                            and "/" not in model:
                        raise ValueError("model must be Auto or an admitted Hawking model id")
                    scope = str(
                        incoming.get("route_scope")
                        or (getattr(session, "ui", {}) or {}).get("route_scope")
                        or _web_model_scope(model)
                    ).strip().lower()
                    if scope not in {"auto", "local", "cloud"}:
                        raise ValueError("route_scope must be auto, local, or cloud")
                    auto_scope = str(
                        incoming.get("auto_scope")
                        or (getattr(session, "ui", {}) or {}).get("auto_scope")
                        or "cloud"
                    ).strip().lower()
                    if auto_scope not in {"cloud", "local", "both"}:
                        raise ValueError("auto_scope must be cloud, local, or both")
                    if model == REVIEW_ONLY_MODEL_ID:
                        raise ValueError(
                            "no admitted local model is available; Kimi P0 is present but withheld"
                        )
                    from .chat_tools import (
                        explicit_tool_observation_required,
                        goal_state_query_required,
                    )
                    classified_tool_intent = explicit_tool_observation_required([
                        {"role": "user", "content": message}
                    ]) or goal_state_query_required(message) or bool(re.search(
                        r"\b(?:inspect|identify|find|search|read|list|show|status|where|which)\b"
                        r".{0,100}\b(?:repo(?:sitory)?|source|file|git|goal|worker|runtime|"
                        r"permission|artifact|note|hawking)\b",
                        message,
                        re.I | re.S,
                    ))
                    attachment_refs = _web_attachment_refs(
                        incoming.get("attachment_ids", incoming.get("attachments"))
                    )
                    message_id = str(
                        incoming.get("message_id") or f"webmsg-{uuid.uuid4().hex[:24]}"
                    ).strip()[:160]
                    memory_projection = _web_memory_projection(
                        session,
                        message,
                        exclude_artifact_ids={
                            str(item.get("artifact_id"))
                            for item in attachment_refs
                            if isinstance(item, Mapping) and item.get("artifact_id")
                        },
                    )
                    history = _web_conversation_messages(session)
                    # The H-Web cloud stream is intentionally a direct
                    # provider stream, so it does not pass through the
                    # resident's generic tool loop.  Operator questions about
                    # detached Goals still need a Hawking-owned fact boundary:
                    # project a bounded canonical snapshot for this turn only,
                    # without persisting a second copy or claiming that the
                    # provider can inspect Goal files itself.
                    goal_projection = ""
                    try:
                        from .chat_tools import goal_state_query_required, run_local_tool
                        if goal_state_query_required(message):
                            state_root = stores.get("state_root") or (
                                getattr(repo, "root", None) if repo is not None else None
                            ) or os.getcwd()
                            observed = run_local_tool(
                                "goals.list",
                                {
                                    "limit": 12,
                                    "goal_ids": _web_goal_ids(session.ui),
                                    "session_id": sid,
                                },
                                workspace=state_root,
                            )
                            if getattr(observed, "ok", False):
                                goal_projection = (
                                    "[HAWKING CANONICAL DETACHED-GOAL PROJECTION]\n"
                                    "Use this bounded projection as factual context. "
                                    "Do not claim that Goal state is inaccessible.\n"
                                    + json.dumps(
                                        getattr(observed, "value", {}),
                                        ensure_ascii=False, sort_keys=True,
                                    )
                                )
                    except Exception:
                        # A projection failure is fail-closed: the provider
                        # must not receive guessed Goal state or secret-bearing
                        # error details.
                        goal_projection = ""
                    session.append_message(
                        "user", message, kind="conversation",
                        attachments=attachment_refs,
                        message_id=message_id,
                    )
                    session.model = model
                    if not isinstance(session.ui, dict):
                        session.ui = {}
                    session.ui["route_scope"] = scope
                    session.ui["auto_scope"] = auto_scope
                    _web_store().save(session)
                    from .web_artifacts import workspace_id as _web_workspace_id
                    preloaded_body = {
                        key: value for key, value in incoming.items()
                        if key not in {
                            "message", "session_id", "route_scope",
                            "attachment_ids", "attachments", "message_id",
                        }
                    }
                    build_authorized = bool(
                        bound_builder is not None
                        and bound_builder.id == session.id
                        and (getattr(session, "ui", {}) or {}).get("authority_profile") == "build"
                    )
                    # Builder authority is a session ceiling.  Turn routing is
                    # still deliberately narrow: ordinary questions receive
                    # the canonical READ menu; an explicit source mutation
                    # request gains only the existing typed builder doors.
                    mutation_intent = bool(re.search(
                        r"\b(?:fix|implement|edit|change|add|remove|repair|write|refactor|create)\b"
                        r".{0,140}\b(?:source|code|file|test|repo(?:sitory)?|fixture|bug)\b",
                        message, re.I | re.S,
                    ))
                    prior_build_state = (
                        (getattr(session, "ui", {}) or {}).get("build_state")
                        if session is not None else None
                    )
                    build_checkpoint_pending = (
                        isinstance(prior_build_state, Mapping)
                        and str(prior_build_state.get("state") or "").upper()
                        not in {"", "COMPLETE"}
                    )
                    # A continuation prompt may be as small as “continue”.
                    # The durable Build checkpoint, not a keyword, carries the
                    # authority to re-enter the same bounded lane.
                    build_turn = build_authorized and (
                        mutation_intent or build_checkpoint_pending
                    )
                    preloaded_body.update({
                        "model": model,
                        "messages": [
                            *history,
                            *([{"role": "system", "content": goal_projection}]
                              if goal_projection else []),
                            *([{"role": "system", "content": memory_projection}]
                              if memory_projection else []),
                            {
                                "role": "user",
                                "content": message + _web_attachment_projection(
                                    session, attachment_refs
                                ),
                            },
                        ],
                        "stream": bool(incoming.get("stream", True)),
                        "hawking_web_session_id": sid,
                        "hawking_web_route_scope": scope,
                        "hawking_auto_scope": auto_scope,
                        "hawking_web": True,
                        # A workspace-backed H-Web chat always has Hawking's
                        # bounded READ envelope.  Intent may affect which
                        # tool a model chooses, but it must never decide
                        # whether a model has any legal route to current
                        # source, artifact, memory, or runtime evidence.
                        "hawking_worker_mode": (
                            "builder_interactive" if build_turn else "read_only_research"
                        ),
                        "hawking_web_build_authorized": build_turn,
                        "hawking_capability_intent": (
                            "explicit_read" if classified_tool_intent
                            else "workspace_default_read"
                        ),
                        "hawking_web_workspace_id": _web_workspace_id(gateway_workspace),
                    })
                    if build_turn:
                        build_state = prior_build_state
                        if isinstance(build_state, Mapping) and str(
                                build_state.get("state") or "").upper() != "COMPLETE":
                            # This is a compact checkpoint, not a replayable
                            # command.  The worker must re-observe source/test/
                            # Git state before attempting another mutation.
                            checkpoint_view = {
                                key: build_state.get(key)
                                for key in (
                                    "state", "scope", "ceiling", "used",
                                    "remaining", "source_revision", "unmet",
                                )
                                if key in build_state
                            }
                            preloaded_body["messages"] = [
                                {
                                    "role": "system",
                                    "content": (
                                        "HAWKING BUILD CHECKPOINT (durable): "
                                        + json.dumps(
                                            checkpoint_view,
                                            sort_keys=True,
                                            ensure_ascii=False,
                                        )
                                        + " Reconcile current source and Git evidence "
                                        "before continuing; never blindly replay a prior edit."
                                    ),
                                },
                                *preloaded_body["messages"],
                            ]
                    if preloaded_body.get("hawking_worker_mode") is None:
                        preloaded_body.pop("hawking_worker_mode", None)
                    if preloaded_body.get("stream"):
                        stream_options = preloaded_body.get("stream_options")
                        stream_options = dict(stream_options) if isinstance(stream_options, Mapping) else {}
                        stream_options["include_usage"] = True
                        preloaded_body["stream_options"] = stream_options
                    route = "/v1/chat/completions"
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
            if route == "/hawking/web/ultragoal":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web UltraGoal access is local-only"}})
                try:
                    body = self._json_body()
                    objective = str(body.get("objective") or "").strip()
                    if not objective:
                        raise ValueError("objective is required")
                    # Reuse Hawking's existing compiler IR.  This dry-run is
                    # intentionally non-persistent and non-dispatching: it
                    # proves phase decomposition without creating a second
                    # Odyssey/Mission or spending remote budget.
                    from .goal import GoalCompiler
                    compiled = GoalCompiler().compile(objective)
                    dag = compiled.get("workunits")
                    units = getattr(dag, "units", {}) if dag is not None else {}
                    plan = []
                    for unit_id, unit in units.items():
                        plan.append({
                            "id": str(unit_id),
                            "role": str(getattr(unit, "role", "") or ""),
                            "description": str(getattr(unit, "description", "") or "")[:1200],
                            "dependencies": [str(item) for item in (getattr(unit, "dependencies", None) or [])],
                            "resource_class": str(getattr(unit, "resource_class", "") or ""),
                            "status": "planned",
                        })
                    return self._send(200, {
                        "schema": "hawking.web.ultragoal.v1",
                        "dry_run": True,
                        "goal_summary": str(compiled.get("goal_summary") or "")[:400],
                        "phase_plan": {
                            "phases": ["PLAN", "WORKUNIT_GRAPH", "TERMINAL_REVIEW"],
                            "workunit_count": len(plan),
                        },
                        "workunits": plan,
                        "execution": {
                            "workers": "not started",
                            "terminal": "dry_run",
                            "remote_cost_usd": 0.0,
                        },
                        "claim_boundary": (
                            "Compiler decomposition only. No Goal, worker, mutation, "
                            "or remote request was created."
                        ),
                    })
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
            if route == "/hawking/web/goal":
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "Hawking Web Goal access is local-only"}})
                try:
                    preloaded_body = self._json_body()
                    objective = str(preloaded_body.get("objective") or "").strip()
                    if not objective:
                        raise ValueError("objective is required")
                    attachment_refs = _web_attachment_refs(
                        preloaded_body.get(
                            "attachment_ids", preloaded_body.get("attachments")
                        )
                    )
                    requested = str(
                        preloaded_body.get("model")
                        or preloaded_body.get("resident")
                        or "hawking/auto"
                    ).strip()
                    auto_scope = str(preloaded_body.get("auto_scope") or "cloud").strip().lower()
                    if auto_scope not in {"cloud", "local", "both"}:
                        raise ValueError("auto_scope must be cloud, local, or both")
                    if _is_hawking_auto_model(requested):
                        if auto_scope == "local":
                            # Keep the logical Auto request intact so
                            # dispatch can use an actually admitted native
                            # route, or fail closed when local remains withheld.
                            requested = "hawking-auto"
                        else:
                            web_goal_auto_route = self._remote_auto_route({
                                "model": "hawking/auto",
                                "messages": [{"role": "user", "content": objective}],
                                "hawking_auto_scope": auto_scope,
                            })
                            requested = str(web_goal_auto_route["selected_model"])
                    preloaded_body.update({
                        "model": requested,
                        "resident": requested,
                        "hawking_auto_scope": auto_scope,
                        "attachments": attachment_refs,
                        "source": "h_web",
                    })
                    route = "/v1/goal"
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send_remote_error(exc)
            if route.startswith("/api/goal"):
                route = "/v1" + route[len("/api"):]
            if route == "/api/runtime/switch":
                route = "/v1/switch"
            if route == "/v1/goal" or route.startswith("/v1/goal?"):
                route = "/v1/goal"
            if route == "/v1/goal/workunit":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "Goal WorkUnit dispatch is local-only"}})
                try:
                    body = preloaded_body if preloaded_body is not None else self._json_body()
                    if not isinstance(body, Mapping):
                        raise ValueError("body must be object")
                    parent_goal_id = str(body.get("parent_goal_id") or body.get("goal_id") or "").strip()
                    if not parent_goal_id:
                        raise ValueError("parent_goal_id is required")
                    from hawking.goal_surface import dispatch_child_workunit
                    root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                    dispatched = dispatch_child_workunit(
                        root,
                        parent_goal_id,
                        objective=str(body.get("objective") or ""),
                        acceptance=(
                            [str(item) for item in body.get("acceptance") if str(item).strip()]
                            if isinstance(body.get("acceptance"), list) else []
                        ),
                        focus_files=(
                            [str(item) for item in body.get("focus_files") if str(item).strip()]
                            if isinstance(body.get("focus_files"), list) else None
                        ),
                        focus_tests=(
                            [str(item) for item in body.get("focus_tests") if str(item).strip()]
                            if isinstance(body.get("focus_tests"), list) else None
                        ),
                        budget_usd=(
                            float(body.get("budget_usd"))
                            if body.get("budget_usd") is not None else 1.0
                        ),
                        macos_authority=(
                            body.get("macos_authority")
                            if isinstance(body.get("macos_authority"), Mapping) else None
                        ),
                        goal_mode=str(body.get("goal_mode") or "discrete_goal").strip(),
                        model_override=str(body.get("model_override") or "").strip() or None,
                        background=True,
                    )
                    return self._send(201, dispatched)
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(503, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            if route == "/v1/goal":
                if not self._local_only():
                    return self._send(403, {"error": {"message": "goal create is local-only"}})
                if preloaded_body is not None:
                    body = preloaded_body
                else:
                    try:
                        length = int(self.headers.get("Content-Length") or 0)
                    except ValueError:
                        length = 0
                    raw = self.rfile.read(length) if length else b"{}"
                    try:
                        body = json.loads(raw.decode("utf-8") or "{}")
                    except json.JSONDecodeError as exc:
                        return self._send(400, {"error": {"message": f"invalid json: {exc}"}})
                    if not isinstance(body, dict):
                        return self._send(400, {"error": {"message": "body must be object"}})
                requested_goal_worker = str(
                    body.get("model") or body.get("resident") or ""
                ).strip()
                remote_goal_worker = bool(
                    requested_goal_worker.lower().startswith("openrouter:")
                    or (
                        "/" in requested_goal_worker
                        and not requested_goal_worker.startswith(("/", "."))
                        and not requested_goal_worker.endswith((".json", ".safetensors"))
                    )
                )
                if (
                    canonical_router is not None
                    and getattr(backend, "canonical_unbound", False)
                    and not remote_goal_worker
                ):
                    return self._send(503, {"error": {
                        "message": (
                            "cannot dispatch a Goal while hawkingd has no qualified "
                            "native resident; inspect /api/runtime/roles"
                        ),
                        "type": "runtime_role_unavailable",
                    }})
                try:
                    from hawking.goal_surface import create_goal, dispatch_goal
                    root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                    doc = create_goal(
                        root,
                        objective=str(body.get("objective") or ""),
                        attachments=(
                            body.get("attachments")
                            if isinstance(body.get("attachments"), list) else None
                        ),
                        acceptance=body.get("acceptance") if isinstance(body.get("acceptance"), list) else None,
                        authority_boundary=str(body.get("authority_boundary") or "staging worktree only; no production mutation"),
                        flash_resource_rule=str(body.get("flash_resource_rule") or "Flash protected-lane; native resident only; no unqualified provider fallback"),
                        completion_contract=str(body.get("completion_contract") or "RUNNING to COMPLETE only when acceptance met; emit canonical result"),
                        resident=str(
                            body.get("resident") or body.get("model") or "pulsar"
                        ),
                        workunit_id=(str(body["workunit_id"]) if body.get("workunit_id") else None),
                        focus_tests=(body.get("focus_tests") if isinstance(body.get("focus_tests"), list) else None),
                        focus_files=(body.get("focus_files") if isinstance(body.get("focus_files"), list) else None),
                        goal_mode=str(body.get("goal_mode") or "discrete_goal"),
                        budget_usd=(
                            float(body["budget_usd"])
                            if body.get("budget_usd") is not None else None
                        ),
                        tool_learning_budget_usd=(
                            float(body["tool_learning_budget_usd"])
                            if body.get("tool_learning_budget_usd") is not None else None
                        ),
                        parent_worker_id=str(body.get("parent_worker_id") or ""),
                        parent_workunit_id=str(body.get("parent_workunit_id") or ""),
                        worker_policy=str(
                            body.get("worker_policy")
                            or ("auto" if web_goal_auto_route is not None else "pinned")
                        ),
                        auto_scope=str(body.get("auto_scope") or "cloud"),
                        allowed_models=(
                            body.get("allowed_models")
                            if isinstance(body.get("allowed_models"), list) else None
                        ),
                        cognition_plan=(
                            web_goal_auto_route.get("cognition_plan")
                            if isinstance(web_goal_auto_route, Mapping)
                            and isinstance(web_goal_auto_route.get("cognition_plan"), Mapping)
                            else None
                        ),
                        parent_goal_ids=(
                            body.get("parent_goal_ids")
                            if isinstance(body.get("parent_goal_ids"), list) else None
                        ),
                        browser_authority=(
                            body.get("browser_authority")
                            if isinstance(body.get("browser_authority"), Mapping) else None
                        ),
                        macos_authority=(
                            body.get("macos_authority")
                            if isinstance(body.get("macos_authority"), Mapping) else None
                        ),
                        source=str(body.get("source") or "h_web"),
                    )
                    dispatched = dispatch_goal(
                        root,
                        doc,
                        model=str(
                            body.get("model") or body.get("resident") or "pulsar"
                        ),
                    )
                    if body.get("hawking_web_session_id"):
                        _web_record_goal(
                            body.get("hawking_web_session_id"),
                            (dispatched.get("goal") or {}).get("goal_id")
                            if isinstance(dispatched, Mapping) else None,
                        )
                    if web_goal_auto_route is not None:
                        dispatched = {
                            **dispatched,
                            "auto_route": dict(web_goal_auto_route),
                        }
                    return self._send(201, dispatched)
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    try:
                        from hawking.goal_surface import update_goal
                        if isinstance(locals().get("doc"), dict):
                            update_goal(
                                root,
                                str(doc.get("goal_id") or ""),
                                status="BLOCKED",
                                phase="DISPATCH_FAILED",
                                blocker=f"{type(exc).__name__}: {exc}",
                            )
                    except Exception:
                        pass
                    return self._send(503, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            if route in ("/v1/workers/action", "/v1/worker/action"):
                if not self._local_only():
                    return self._send(403, {"error": {"message": "worker control is local-only"}})
                try:
                    body = self._json_body()
                    action = str(body.get("action") or "").strip().lower()
                    if action == "spawn":
                        raise ValueError("worker spawn is model-mediated; use hawking.worker.spawn")
                    from hawking.workunit_owner import WorkunitOwner
                    root = Path(
                        stores.get("state_root")
                        or (getattr(repo, "root", None) if repo is not None else None)
                        or os.getcwd()
                    ).resolve()
                    owner = WorkunitOwner(root)
                    worker_id = str(body.get("worker_id") or "").strip()
                    workunit_id = str(body.get("workunit_id") or "").strip()
                    record = owner.find_by_worker_id(worker_id) if worker_id else None
                    if record is None and workunit_id:
                        record = owner.load(workunit_id)
                    if record is None:
                        raise ValueError("worker_id or workunit_id must identify a Hawking worker")
                    if action == "status":
                        return self._send(200, owner.status(record.workunit_id))
                    if action == "kill":
                        value = owner.cancel(record.workunit_id)
                        updated = owner.load(record.workunit_id)
                        updated.worker_kill_reason = str(
                            body.get("reason") or "operator_worker_action"
                        )[:300]
                        owner.save(updated)
                        return self._send(200, updated.to_dict())
                    if action == "pause":
                        return self._send(200, owner.pause(record.workunit_id, reason="operator_worker_action"))
                    if action == "resume":
                        return self._send(200, owner.resume(record.workunit_id))
                    raise ValueError("worker action must be status, kill, pause, or resume")
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(409, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            if route in ("/v1/goal/control", "/v1/goal/action"):
                if not self._local_only():
                    return self._send(403, {"error": {"message": "goal control is local-only"}})
                try:
                    body = self._json_body()
                    gid = str(body.get("id") or body.get("goal_id") or "").strip()
                    action = str(body.get("action") or "").strip().lower()
                    if not gid or not action:
                        raise ValueError("id and action are required")
                    from hawking.goal_surface import load_goal, update_goal
                    from hawking.workunit_owner import WorkunitOwner
                    root = stores.get("state_root") or (getattr(repo, "root", None) if repo is not None else None) or os.getcwd()
                    goal = load_goal(root, gid)
                    if not goal:
                        return self._send(404, {"error": {"message": f"no goal {gid}"}})
                    wid = str(goal.get("workunit_id") or gid)
                    owner = WorkunitOwner(root)
                    if action == "steer":
                        note = str(body.get("message") or body.get("steer") or "").strip()
                        if not note:
                            raise ValueError("steer requires message")
                        rec = owner.load(wid)
                        rec.exact_next_action = note[:2000]
                        rec.current_phase = "STEERED"
                        rec.evidence.append({"at": time.time(), "kind": "operator_steer", "note": note[:2000]})
                        owner.save(rec)
                        updated = update_goal(root, gid, phase="STEERED", last_operator_action="steer") or goal
                        return self._send(200, {"goal": updated, "workunit": owner.status(wid)})
                    if action == "pause":
                        rec = owner.pause(wid, reason=str(body.get("reason") or "operator"))
                        updated = update_goal(root, gid, status="PAUSED", phase="PAUSED", last_operator_action="pause") or goal
                        return self._send(200, {"goal": updated, "workunit": rec})
                    if action == "resume":
                        resumed = owner.resume(wid)
                        updated = update_goal(root, gid, status="RUNNING", phase="RESUMED", last_operator_action="resume") or goal
                        return self._send(200, {"goal": updated, "workunit": resumed.get("owner"), "job": resumed.get("background")})
                    if action == "cancel":
                        rec = owner.cancel(wid)
                        updated = update_goal(root, gid, status="CANCELLED", phase="CANCELLED", last_operator_action="cancel") or goal
                        return self._send(200, {"goal": updated, "workunit": rec})
                    if action == "checkpoint":
                        rec = owner.status(wid)
                        updated = update_goal(root, gid, phase="CHECKPOINTED", last_operator_action="checkpoint") or goal
                        return self._send(200, {"goal": updated, "workunit": rec, "checkpoint": rec.get("checkpoint_path")})
                    if action == "result":
                        from hawking.goal_surface import load_result
                        from urllib.parse import quote
                        result = load_result(root, gid)
                        return self._send(200 if result else 404, {"goal": goal, "result": result})
                    if action == "diff":
                        return self._send(200, _goal_diff_payload(root, goal, gid))
                    raise ValueError(f"unsupported goal action: {action}")
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(409, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            if route == "/hawkingd/delegations/start":
                if delegation_supervisor is None:
                    return self._send(404, {"error": {
                        "message": "daemon delegation supervision is unavailable"}})
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "delegation supervision is local-only"}})
                try:
                    body = self._json_body()
                    return self._send(200, delegation_supervisor.start(body.get("workspace")))
                except PermissionError as exc:
                    return self._send(403, {"error": {"message": str(exc)}})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(502, {"error": {
                        "message": f"{type(exc).__name__}: {exc}"}})
            if route in ("/hawkingd/webui", "/hawkingd/webui/stop"):
                if webui_manager is None:
                    return self._send(404, {"error": {
                        "message": "daemon WebUI supervision is unavailable"}})
                if not self._local_only():
                    return self._send(403, {"error": {
                        "message": "WebUI supervision is local-only"}})
                try:
                    body = self._json_body()
                    port = int(body.get("port") or 0)
                    if route.endswith("/stop"):
                        stopped = webui_manager.stop(
                            port,
                            requester_pid=int(body.get("requester_pid") or 0),
                        )
                        return self._send(200 if stopped else 404,
                                          {"stopped": bool(stopped), "port": port})
                    if endpoint_base is None:
                        raise RuntimeError("daemon endpoint identity is unavailable")
                    child = webui_manager.start(
                        port, endpoint_base, int(body.get("requester_pid") or 0)
                    )
                    return self._send(200, child)
                except PermissionError as exc:
                    return self._send(403, {"error": {"message": str(exc)}})
                except (ValueError, json.JSONDecodeError) as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(502, {"error": {
                        "message": f"{type(exc).__name__}: {exc}"}})
            if route in ("/v1/switch", "/switch", "/hawkingd/research/switch"):
                # The control door `hawking use` knocks on. Switching is also
                # reachable by naming a model in a chat request; this exists so
                # a person can pay the load cost deliberately instead of
                # discovering it inside their first message.
                try:
                    body = self._json_body()
                except Exception as exc:
                    return self._send(400, {"error": {"message": f"bad json: {exc}"}})
                research = route == "/hawkingd/research/switch"
                if canonical_router is not None:
                    # The canonical port is not a raw-provider or research
                    # switchboard.  Public callers select a logical role or an
                    # admitted native artifact; the router supplies the exact
                    # action binding and excludes MLX/remote fallbacks.
                    if research:
                        return self._send(403, {"error": {
                            "message": "research switching is not exposed on the canonical public runtime",
                            "type": "canonical_runtime_only",
                        }})
                    if body.get("resolved_action") is not None:
                        return self._send(403, {"error": {
                            "message": "canonical runtime switching accepts model roles or admitted native artifact IDs, not caller-supplied action bindings",
                            "type": "canonical_runtime_only",
                        }})
                    from .canonical_runtime import RuntimeRoleUnavailable
                    try:
                        canonical_route = canonical_router.route(
                            body.get("model"), body)
                    except RuntimeRoleUnavailable as exc:
                        return self._send(503, {"error": {
                            "message": str(exc),
                            "type": "runtime_role_unavailable",
                            "hawking": exc.to_dict(),
                        }})
                    if getattr(backend, "canonical_unbound", False):
                        return self._send(503, {"error": {
                            "message": (
                                f"{canonical_route.role}: a native artifact is now routable, "
                                "but this waiting daemon has no loaded resident; restart hawkingd "
                                "on the same canonical port to activate it"
                            ),
                            "type": "runtime_activation_required",
                            "hawking": canonical_route.to_dict(),
                        }})
                    switch = getattr(backend, "switch", None)
                    if not callable(switch):
                        return self._send(409, {"error": {
                            "message": "this surface serves a single fixed body"}})
                    try:
                        result = switch(canonical_route.action)
                    except LookupError as exc:
                        return self._send(404, {"error": {"message": str(exc)}})
                    except PermissionError as exc:
                        return self._send(403, {"error": {"message": str(exc)}})
                    except MemoryError as exc:
                        return self._send(507, {"error": {"message": str(exc)}})
                    except Exception as exc:
                        return self._send(502, {"error": {
                            "message": f"{type(exc).__name__}: {exc}"}})
                    return self._send(200, {**result, "route": canonical_route.to_dict()})
                if research and not self._local_only():
                    return self._send(403, {"error": {
                        "message": "research switching is local-only"}})
                switch = getattr(backend, "switch", None)
                if not callable(switch):
                    return self._send(409, {"error": {
                        "message": "this surface serves a single fixed body"}})
                try:
                    raw_action = body.get("resolved_action")
                    if raw_action is not None:
                        if research:
                            raise ValueError(
                                "research switching does not accept a normal resolved action"
                            )
                        if not isinstance(raw_action, dict):
                            raise ValueError("resolved_action must be an object")
                        from .catalog import ResolvedAction
                        selection = ResolvedAction.from_wire(raw_action)
                    else:
                        selection = str(body.get("model") or "")
                    return self._send(200, switch(selection, research=research))
                except ValueError as exc:
                    return self._send(400, {"error": {"message": str(exc)}})
                except LookupError as exc:
                    return self._send(404, {"error": {"message": str(exc)}})
                except PermissionError as exc:
                    return self._send(403, {"error": {"message": str(exc)}})
                except MemoryError as exc:
                    return self._send(507, {"error": {"message": str(exc)}})
                except Exception as exc:
                    return self._send(502, {"error": {
                        "message": f"{type(exc).__name__}: {exc}"}})
            if route not in (
                "/v1/chat/completions", "/chat/completions",
                "/v1/responses", "/responses",
            ):
                return self._send(404, {"error": {
                    "message": f"no route {self.path}; this server serves "
                               f"/v1/models, /v1/chat/completions, /v1/responses and /v1/switch"}})
            if route in {
                "/v1/chat/completions", "/chat/completions",
                "/v1/responses", "/responses",
            } and not self._gateway_access_allowed():
                return self._send(401, {"error": {
                    "message": "Hawking remote gateway access requires a bearer token",
                    "type": "authentication_error",
                }})
            if preloaded_body is not None:
                body = preloaded_body
            else:
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except Exception as exc:
                    return self._send(400, {"error": {"message": f"bad json: {exc}"}})
                if not isinstance(body, dict):
                    return self._send(400, {"error": {"message": "body must be an object"}})
            provider_hint = (
                body.get("hawking_provider")
                or self.headers.get("X-Hawking-Provider")
            )
            hint = str(provider_hint or "").strip().lower()
            if hint and hint not in {
                "openrouter", "openrouter.ai", "hawking", "hawking-native",
                "native", "local",
            }:
                return self._send(400, {"error": {
                    "message": f"unsupported Hawking provider hint {provider_hint!r}",
                    "type": "invalid_request_error",
                }})
            requested_model = str(body.get("model") or "").strip()
            auto_route: Optional[Dict[str, Any]] = None
            remote_model = None
            auto_requested = _is_hawking_auto_model(requested_model)
            native_auto_route = None
            auto_scope = str(body.get("hawking_auto_scope") or "cloud").strip().lower()
            if auto_scope not in {"cloud", "local", "both"}:
                return self._send(400, {"error": {
                    "message": "hawking_auto_scope must be cloud, local, or both",
                    "type": "invalid_request_error",
                }})
            if auto_requested and hint not in {"openrouter", "openrouter.ai"}:
                # ``cloud`` is the current safe default. ``local`` and
                # ``both`` opt into a native role when one is actually
                # admitted; a withheld native role never becomes a fake local
                # model and local-only Auto fails closed.
                if auto_scope in {"local", "both"} and canonical_router is not None:
                    from .canonical_runtime import AUTO_ROLE, RuntimeRoleUnavailable
                    try:
                        native_auto_route = canonical_router.route(AUTO_ROLE, body)
                    except RuntimeRoleUnavailable:
                        native_auto_route = None
                if auto_scope == "local" and native_auto_route is None:
                    return self._send(503, {"error": {
                        "message": "Hawking Auto local scope has no admitted native route",
                        "type": "auto_local_unavailable",
                    }})
                if native_auto_route is None:
                    try:
                        auto_route = self._remote_auto_route(body)
                    except Exception as exc:
                        return self._send_remote_error(exc)
                    remote_model = str(auto_route["selected_model"])
                else:
                    # Let the canonical block below perform the ordinary
                    # native resident binding and activation checks.
                    requested_model = "hawking-auto"
            else:
                remote_model = _openrouter_model_id(
                    requested_model,
                    provider_hint=provider_hint,
                )
            if auto_requested and remote_model is None:
                requested_model = "hawking-auto"
            if route in ("/v1/responses", "/responses") and remote_model is None:
                if not native_auto_route:
                    return self._send(400, {"error": {
                        "message": "the Responses endpoint requires an explicit OpenRouter model id",
                        "type": "invalid_request_error",
                    }})
            if remote_model is not None or hint in {"openrouter", "openrouter.ai"}:
                if not remote_model:
                    return self._send(400, {"error": {
                        "message": "an OpenRouter model id is required",
                        "type": "invalid_request_error",
                    }})
                try:
                    if route in ("/v1/responses", "/responses"):
                        return self._remote_responses(
                            body, remote_model, auto_route=auto_route
                        )
                    return self._remote_chat(
                        body, remote_model, auto_route=auto_route
                    )
                except Exception as exc:
                    # Streaming failures are emitted as SSE after headers; this
                    # path covers admission/provider construction failures.
                    return self._send_remote_error(exc)
            try:
                worker_mode = worker_request_mode(body.get("hawking_worker_mode"))
            except ValueError as exc:
                return self._send(400, {"error": {"message": str(exc)}})
            restricted_worker = worker_mode is not WorkerRequestMode.DEFAULT
            is_web_builder = bool(
                body.get("hawking_web")
                and body.get("hawking_web_build_authorized") is True
                and worker_mode is WorkerRequestMode.BUILDER_INTERACTIVE
            )
            request_resident = str(getattr(backend, "identity", identity) or identity)
            canonical_route = None
            if canonical_router is not None:
                from .canonical_runtime import RuntimeRoleUnavailable
                try:
                    canonical_route = canonical_router.route(requested_model, body)
                except RuntimeRoleUnavailable as exc:
                    return self._send(503, {"error": {
                        "message": str(exc),
                        "type": "runtime_role_unavailable",
                        "hawking": exc.to_dict(),
                    }})
                if getattr(backend, "canonical_unbound", False):
                    return self._send(503, {"error": {
                        "message": (
                            f"{canonical_route.role}: a native artifact is routable, but "
                            "this waiting daemon has no loaded resident; restart hawkingd on "
                            "the same canonical port to activate it"
                        ),
                        "type": "runtime_activation_required",
                        "hawking": canonical_route.to_dict(),
                    }})
                # Resident.complete owns stop-then-start and binding
                # revalidation.  It receives only the native artifact selected
                # by the role router; the public logical alias never becomes a
                # raw provider selector.
                requested_model = canonical_route.action.name
                body = {**body, "model": requested_model}
            if (restricted_worker and requested_model
                    and requested_model != request_resident):
                # Model selection is an action: Resident.complete() would
                # otherwise stop/spawn a new provider before it ever reached
                # the restricted request's tool policy.  Refuse before
                # session handling or any provider call, and repeat the lock
                # inside Resident.complete() to close concurrent switches.
                return self._send(409, {"error": {
                    "message": (
                        "restricted worker request may not select another "
                        f"resident; loaded={request_resident!r}"),
                    "type": "model_switch_denied",
                }})
            # A worker policy is server-side authority. It must be resolved
            # before durable session state is updated, otherwise a read-only
            # research turn could advertise a write-capable session merely
            # because this daemon also serves supervised build work.
            write_authority = (
                stores.get("engine") is not None
                and worker_mode is WorkerRequestMode.DEFAULT
            )
            # A discrete Goal carries a server-owned mutation-bundle contract.
            # Resolve it from the durable Goal contract rather than trusting a
            # model-visible prompt or arbitrary provider field.  The owner also
            # sends the bundle as a convenience for detached Web callers; the
            # on-disk contract remains authoritative.
            goal_mutation_bundle = None
            goal_id = str(body.get("hawking_goal_id") or "").strip()
            goal_mode = str(body.get("hawking_goal_mode") or "").strip()
            if goal_id and goal_mode == "discrete_goal":
                state_root = stores.get("state_root") or (
                    getattr(repo, "root", None) if repo is not None else None
                ) or os.getcwd()
                if re.fullmatch(r"GOAL-[A-Za-z0-9_-]+", goal_id):
                    contract_path = (Path(state_root) / ".hawking" / "goals"
                                     / f"{goal_id}.contract.json")
                    try:
                        contract_doc = json.loads(
                            contract_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        contract_doc = {}
                    if isinstance(contract_doc, dict) and (
                            contract_doc.get("goal_mode") == "discrete_goal"):
                        candidate_bundle = contract_doc.get("mutation_bundle")
                        if isinstance(candidate_bundle, dict):
                            goal_mutation_bundle = dict(candidate_bundle)
            live_greedy = getattr(backend, "greedy", greedy)
            if live_greedy:
                refusal = sampler_refusal(body)
                if refusal is not None:
                    return self._send(400, refusal)
            messages = body.get("messages") or []
            if not messages:
                return self._send(400, {"error": {
                    "message": "no messages: send {'messages':[{'role':'user',"
                               "'content':'...'}]}"}})
            # DURABLE SESSION. Open WebUI sends no id and resends the whole
            # conversation, so the first user turn's digest is the key: stable
            # for the life of that conversation, no protocol change needed.
            session = None
            if stores.get("state_root") and not restricted_worker:
                from .chat_state import ChatSession, session_key, working_set
                root = stores["state_root"]
                session = ChatSession.load(
                    root, session_key(messages, root,
                                      explicit=body.get("session_id")
                                      or body.get("chat_id")))
                session.resident = getattr(backend, "identity", identity)
                # The session inherits the surface's authority. A session record
                # defaulting to "read" while the surface was started --write left
                # the working set and the response both claiming read on a build
                # session -- true of a fresh record, wrong for this surface.
                session.authority = "write" if write_authority else "read"
                session.turns += 1
                # RESUME BEFORE ANYTHING ELSE. A checkpoint means a previous
                # invocation yielded mid-objective; the resuming turn must see
                # where it stands and whether the repository moved under it.
                from .chat_state import compact, resume, resume_block
                revived = resume(session)
                durable = working_set(session)
                revival = resume_block(revived) if (
                    revived and session.turns <= 1) else ""
                if revival:
                    durable = (durable + "\n\n" + revival) if durable else revival
                if durable:
                    # AFTER the tool contract and repo identity, BEFORE the
                    # conversation: it changes as the plan advances, so it must
                    # not sit in front of the material the prefix cache reuses.
                    messages = [*messages[:1], {"role": "system",
                                                "content": durable},
                                *messages[1:]] if (
                        messages and messages[0].get("role") == "system") else [
                        {"role": "system", "content": durable}, *messages]
            if repo is not None and not restricted_worker:
                # The folder HAWKING was opened in, in front of the conversation.
                # Stable block first so the resident's prefix cache keeps it
                # across turns -- see hawking/repo_context.py for why the obvious
                # order would cost 20s per follow-up instead of 0.5s.
                from .repo_context import inject
                messages = inject(messages, repo)
                body = {**body, "messages": messages}
            if session is not None:
                # COMPACT WHEN THE CONVERSATION OUTGROWS ITS SHARE. Turns leave
                # the working set into the session archive; the invariant
                # leading region is untouched so the prefix cache still hits.
                # `context_window`, NOT `prompt_window`. Using the ceiling here
                # was a REGRESSION and it took the campaign down: cycle 82 got
                # "prompt is 8807 tokens and native max_seq_len is 8192; no
                # generation token fits", repeatedly, as a 502.
                #
                # The double-reserve diagnosis was right and the fix was wrong.
                # `usable_input_tokens` (5632) was reserving generation room a
                # second time AND, unknowingly, absorbing a ~2x error in the
                # CHARS_PER_TOKEN=4 estimate: 0.55 * 8192 = 4505 ESTIMATED tokens
                # measured 8807 REAL ones, because code and JSON tokenise far
                # worse than the prose that heuristic was tuned on. Removing the
                # slack exposed the estimator, and the estimator is the thing
                # that is actually wrong.
                #
                # Until the budget is computed from a real token count, the
                # conservative window is the correct one. `prompt_window` stays
                # in /health as information; it is not an admission bound.
                window = int(health.get("context_window") or 8192)
                messages, compaction = compact(messages, session, window=window)
                if compaction.compacted:
                    body = {**body, "messages": messages}
                    checkpoint_note = (
                        f"compacted {compaction.evicted_turns} turns "
                        f"({compaction.before_tokens}->{compaction.after_tokens} tok)")
                    from .chat_state import checkpoint as _ckpt
                    _ckpt(session, resident=session.resident, note=checkpoint_note)
            # ONE leading system message. Repo context and the session
            # working-set each prepend one; the sealed template rejects a
            # second system message that is not at position 0.
            messages = _coalesce_system(messages)
            body = {**body, "messages": messages}
            payload = {k: v for k, v in body.items() if k != "stream"}
            # ``tool_choice`` belongs to Hawking's tool-loop policy, not to a
            # provider which may silently ignore it. A bounded analytical
            # request can explicitly opt out without globally disabling a
            # resident's earned tool contract.
            requested_tool_choice = payload.pop("tool_choice", None)
            # Callers cannot inject a second tool contract into a provider.
            # HAWKING constructs the one native schema list it is prepared to
            # dispatch, and restricted/reason-only requests construct none.
            # Cover both OpenAI's current and legacy function fields: leaving
            # either shape in the payload would let a bridge render an
            # unowned declaration while Hawking reported zero offered tools.
            caller_tool_schemas_withheld = any(
                key in payload for key in ("tools", "functions", "function_call")
            )
            for key in ("tools", "functions", "function_call"):
                payload.pop(key, None)
            # Per-request worker authority also owns provider-specific prompt
            # knobs. A constrained worker must not inherit a caller-selected
            # template contract or spend policy through an adapter-specific
            # field Hawking does not interpret.
            if restricted_worker:
                payload.pop("chat_template_kwargs", None)
            # Hawking-specific authority metadata never crosses the provider
            # boundary. A provider may silently ignore it, which would turn a
            # policy claim into an unsafe prompt convention.
            payload.pop("hawking_worker_mode", None)
            payload.pop("hawking_goal_id", None)
            payload.pop("hawking_goal_mode", None)
            payload.pop("hawking_mutation_bundle", None)
            tools_disabled_by_request = (
                worker_mode is WorkerRequestMode.REASON_ONLY
                or requested_tool_choice == "none"
                or (isinstance(requested_tool_choice, dict)
                    and requested_tool_choice.get("type") == "none")
            )
            payload.setdefault(
                "model", request_resident if restricted_worker else identity,
            )

            def complete_backend(request: Dict[str, Any]) -> Any:
                if restricted_worker:
                    # build_server supplies a Resident, but enforce the native
                    # no-spawn/no-telemetry context at the HTTP boundary too:
                    # direct test/embedding callers must not get a weaker
                    # constrained mode merely by passing the adapter itself.
                    from .hawking_native import (
                        HawkingNativeUnavailable,
                        suppress_native_runtime_spawning,
                    )
                    try:
                        with suppress_native_runtime_spawning():
                            if isinstance(backend, Resident):
                                return backend.complete(
                                    request, model_lock=request_resident,
                                    allow_owned_recovery=False)
                            return backend.complete(request)
                    except HawkingNativeUnavailable as exc:
                        raise ResidentUnavailable(str(exc)) from exc
                return backend.complete(request)

            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            trace: list = []
            tool_contract: Optional[Dict[str, Any]] = None
            live_tools = getattr(backend, "tools_verdict", None)
            tools_ok = bool((live_tools or {}).get("qualified")) if live_tools is not None else True
            try:
                if registry is not None and tools_ok and not tools_disabled_by_request:
                    # The chat can look things up. One backend, one registry --
                    # the same one `hawking research-gate` uses, with its
                    # read/research permission set, so a browser session cannot
                    # reach a tool that writes.
                    from .chat_tools import (openai_schemas, prepend_system,
                                              prompt_menu_names, run_with_tools,
                                              session_menu, system_block)

                    # Durable Goal workers must use the typed provider schema
                    # path even when the resident's generic interactive
                    # adapter is textual-only.  The textual contract is what
                    # allowed DeepSeek to stop after archaeology without ever
                    # receiving the admitted repo.edit door. Ordinary chat
                    # keeps the backend's existing native/textual choice.
                    native = bool(getattr(backend, "native_tools", False)) or bool(
                        body.get("hawking_goal_id")
                    )
                    writing = bool(write_authority or is_web_builder)
                    build_budget_ceiling, build_budget_scope = _hweb_build_tool_budget(
                        body, is_web_builder=is_web_builder,
                        is_goal=bool(body.get("hawking_goal_id")),
                    )
                    allowed_names = worker_tool_allowlist(worker_mode)
                    if writing and is_web_builder:
                        allowed_names = frozenset(set(allowed_names or ()) | {
                            "repo.edit", "tests.run",
                        })
                    selected_names = prompt_menu_names(
                        registry, write=writing, allowed_names=allowed_names)
                    schemas = (openai_schemas(registry, selected_names,
                                              write=writing,
                                              allowed_names=allowed_names)
                               if native else None)

                    last_completion = None

                    def _complete(convo):
                        nonlocal last_completion
                        inner = {**payload, "messages": convo}
                        if schemas:
                            inner["tools"] = schemas
                        last_completion = complete_backend(inner)
                        return _text_of(last_completion)

                    contract_text = system_block(
                        registry=registry, write=writing,
                        allowed_names=allowed_names)
                    with_contract = (messages if native else prepend_system(
                        messages, contract_text))
                    if is_web_builder:
                        with_contract = [{
                            "role": "system",
                            "content": (
                                f"Hawking BUILD turn budget: {build_budget_ceiling} "
                                "tool rounds maximum. "
                                "Prioritize source binding, the declared red test, "
                                "repo.edit, then tests.run, git.status, and git.diff; "
                                "report remaining budget before further archaeology."
                            ),
                        }, *with_contract]
                    serialized = (json.dumps(schemas, sort_keys=True)
                                  if schemas is not None else contract_text)
                    tool_contract = {
                        "qualified": True,
                        "worker_mode": worker_mode.value,
                        "mutation_authority": ("granted" if writing else "denied"),
                        "offered_count": len(session_menu(
                            registry, write=writing, allowed_names=allowed_names)),
                        "selected_count": len(selected_names),
                        "serialized_count": (len(schemas)
                                              if schemas is not None else len(selected_names)),
                        "serialized_format": ("openai_tools_argument"
                                              if native else "textual_json_contract"),
                        "serialized_chars": len(serialized),
                        "caller_tool_schemas_withheld": caller_tool_schemas_withheld,
                        "initial_message_count": len(with_contract),
                        "tool_loop_entered": True,
                        "parser": "json_object+xml_function+bounded_narrative",
                        "tool_budget_ceiling": (
                            build_budget_ceiling if is_web_builder else int(
                                os.environ.get("HAWKING_REMOTE_MAX_TOOL_CALLS", "16")
                            )
                        ),
                        "tool_budget_scope": build_budget_scope,
                    }
                    # SEAL_B76 / U001-2: exact estimated prefill category breakdown.
                    try:
                        from .chat_tools import prefill_token_categories
                        from .chat_state import CHARS_PER_TOKEN
                        tool_contract["prefill_token_categories"] = prefill_token_categories(
                            with_contract,
                            system_contract=("" if native else contract_text),
                            chars_per_token=float(CHARS_PER_TOKEN),
                        )
                        from .chat_tools import stable_prefix_digest
                        _dig = stable_prefix_digest(
                            "" if native else contract_text,
                            chars_per_token=float(CHARS_PER_TOKEN),
                        )
                        tool_contract["stable_prefix_digest"] = _dig
                    except Exception as _prefill_exc:  # fail-open measurement only
                        tool_contract["prefill_token_categories_error"] = str(_prefill_exc)[:240]
                    # ADMISSION GUARD for every completion INSIDE the tool
                    # loop, not just the one before it. `compact()` above only
                    # ever runs once, on the INCOMING messages -- measured
                    # live, once batched tool calls started executing (see
                    # chat_tools.parse_calls), a cycle grew the conversation
                    # to 10203 tokens against the resident's native
                    # max_seq_len=8192 with no check in between, and the
                    # backend raised a bare 502. Same window, same headroom
                    # share compact() already uses -- not a new policy.
                    from .chat_state import CHARS_PER_TOKEN, CONTEXT_SHARE
                    # Same regression, same fix: the conservative window until
                    # the estimate is replaced by a real count.
                    window_tokens = int(health.get("context_window") or 8192)
                    answer, trace = run_with_tools(
                        _complete, with_contract, registry, native=native,
                        max_calls=(build_budget_ceiling if is_web_builder else 12),
                        # PasteCache is disk-backed. A constrained worker may
                        # read a large observation, but must not turn it into
                        # durable state merely because it needs compaction.
                        cache=(None if restricted_worker else stores.get("cache")),
                        # No constrained tool is allowed to use context.recall,
                        # but withholding the store as well keeps a future
                        # local-tool addition from silently restoring durable
                        # memory access to this request class.
                        knowledge=(None if restricted_worker
                                   else stores.get("knowledge")),
                        engine=(stores.get("engine") if writing else None),
                        allowed_names=allowed_names,
                        suppress_native_helpers=restricted_worker,
                        mutation_bundle=(goal_mutation_bundle if writing else None),
                        bundle_state=(session.pending_mutation_bundle
                                      if session is not None and writing else None),
                        max_prompt_chars=int(
                            window_tokens * CONTEXT_SHARE * CHARS_PER_TOKEN))
                    # Keep the final provider result when the tool loop was
                    # used.  The old wrapper retained only `answer`, which
                    # made a live provider body look as if it used zero tokens
                    # and had no timings even though the child returned both.
                    # The tool trace is still reported separately; these
                    # provider fields describe the final cognition call only,
                    # not the aggregate cost of all tool-loop calls.
                    result = last_completion or type("R", (), {
                        "text": answer, "finish_reason": "stop", "degraded": [],
                        "prompt_tokens": None, "completion_tokens": None,
                        "total_tokens": None, "raw": {}})()
                    result.text = answer
                    result.finish_reason = getattr(result, "finish_reason", None) or "stop"
                    tool_contract.update({
                        "actual_invocations": sum(
                            bool(row.get("dispatched")) for row in trace),
                        "trace_entries": len(trace),
                        "tool_calls_used": sum(
                            bool(row.get("dispatched")) for row in trace
                        ),
                        "tool_calls_remaining": max(
                            0,
                            int(tool_contract.get("tool_budget_ceiling") or 0)
                            - sum(bool(row.get("dispatched")) for row in trace),
                        ),
                    })
                    if session is not None and is_web_builder:
                        from .chat_state import checkpoint as _build_checkpoint
                        successful = {
                            str(row.get("tool") or "")
                            for row in trace
                            if isinstance(row, Mapping)
                            and row.get("dispatched")
                            and row.get("ok")
                        }
                        tests_passed = any(
                            row.get("tool") == "tests.run"
                            and row.get("dispatched")
                            and isinstance(row.get("result"), Mapping)
                            and isinstance(row["result"].get("value"), Mapping)
                            and row["result"]["value"].get("verified") is True
                            and row["result"]["value"].get("returncode") == 0
                            for row in trace
                            if isinstance(row, Mapping)
                        )
                        complete = bool(
                            {"repo.edit", "git.status", "git.diff"}.issubset(successful)
                            and tests_passed
                        )
                        session.tool_budget = _build_tool_budget_state(
                            build_budget_ceiling,
                            trace,
                            {"complete": complete},
                        )
                        session.tool_budget["scope"] = build_budget_scope
                        _build_checkpoint(
                            session,
                            next_action=(
                                "none; accepted Build evidence is complete"
                                if complete else
                                "reconcile source, then complete the missing Build evidence"
                            ),
                            resident=getattr(backend, "identity", identity),
                            source_revision=session.tool_budget.get("source_revision", ""),
                            tool_budget=session.tool_budget,
                            state="COMPLETE" if complete else "CHECKPOINT",
                            note="Hawking Build tool contract observation",
                        )
                    if goal_mutation_bundle is not None:
                        tool_contract["mutation_bundle"] = {
                            "schema": goal_mutation_bundle.get("schema"),
                            "required": [
                                key for key, enabled in (
                                    ("source_mutation", goal_mutation_bundle.get(
                                        "require_source_mutation")),
                                    ("regression_test_mutation", goal_mutation_bundle.get(
                                        "require_regression_test_mutation")),
                                    ("focused_test_invocation", goal_mutation_bundle.get(
                                        "require_focused_test_invocation")),
                                ) if enabled
                            ],
                            "bundle_state_pending": bool(
                                session is not None and session.pending_mutation_bundle),
                        }
                    if session is not None:
                        from .chat_state import record_tool_trace
                        tool_trace_path = record_tool_trace(
                            session, trace, tool_contract=tool_contract,
                            answer=visible_text(answer),
                            resident=getattr(backend, "identity", identity),
                        )
                else:
                    if registry is not None:
                        from .chat_tools import session_menu
                        allowed_names = worker_tool_allowlist(worker_mode)
                        tool_contract = {
                            "qualified": tools_ok,
                            "worker_mode": worker_mode.value,
                            "mutation_authority": (
                                "granted" if write_authority else "denied"
                            ),
                            "offered_count": len(session_menu(
                                registry, write=write_authority,
                                allowed_names=allowed_names)),
                            "selected_count": 0,
                            "serialized_count": 0,
                            "serialized_format": (
                                "disabled_by_worker_mode"
                                if worker_mode is WorkerRequestMode.REASON_ONLY
                                else ("disabled_by_request"
                                      if tools_disabled_by_request
                                      else "withheld_until_qualification")
                            ),
                            "serialized_chars": 0,
                            "caller_tool_schemas_withheld": caller_tool_schemas_withheld,
                            "tool_loop_entered": False,
                            "actual_invocations": 0,
                            "trace_entries": 0,
                        }
                    result = complete_backend(payload)
            except LookupError as exc:
                return self._send(404, {"error": {
                    "message": str(exc), "type": "model_not_found"}})
            except MemoryError as exc:
                return self._send(507, {"error": {
                    "message": str(exc), "type": "insufficient_storage"}})
            except PermissionError as exc:
                return self._send(403, {"error": {
                    "message": str(exc), "type": "worker_policy_denied"}})
            except ResidentUnavailable as exc:
                return self._send(503, {"error": {
                    "message": str(exc), "type": "resident_unavailable"}})
            except Exception as exc:
                return self._send(502, {"error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "resident_error"}})
            if not body.get("stream"):
                answered = getattr(backend, "identity", identity)
                body_out = chat_payload(result, answered, request_id=request_id)
                # Runtime identity is independent evidence from the daemon,
                # not a claim generated by the resident body. Expose the
                # machine-owned state alongside tool trace so liveness
                # consumers can derive reachability without trusting prose.
                provider = getattr(getattr(backend, "backend", None),
                                   "process", None)
                provider_pid = getattr(provider, "pid", None)
                body_out["hawking"]["runtime"] = {
                    "source": "hawkingd:/health",
                    "status": health.get("status"),
                    "daemon_pid": (health.get("owner") or {}).get("pid"),
                    "provider_pid": provider_pid,
                    "resident": answered,
                    "model": health.get("model"),
                    "state": getattr(backend, "state", None),
                    "repo": health.get("repo"),
                    "identity_match": answered == health.get("resident"),
                    "tools_qualified": bool(
                        (getattr(backend, "tools_verdict", None) or {}).get(
                            "qualified")),
                }
                if canonical_route is not None:
                    body_out["hawking"]["route"] = canonical_route.to_dict()
                # Always expose the empty trace too. An absent field made “no
                # action parsed” indistinguishable from “telemetry was never
                # wired.” `dispatched` is the mechanical liveness fact.
                body_out["hawking"]["tools_used"] = trace
                goal_response = bool(
                    body.get("hawking_goal_id")
                    and str(body.get("hawking_goal_mode") or "") == "discrete_goal"
                )
                if goal_response and isinstance(
                    body.get("hawking_completion_contract"), Mapping
                ):
                    # Durable Goal owners consume this compact projection, not
                    # provider prose.  The interactive builder loop already
                    # recorded the canonical trace, but older responses only
                    # exposed ``tools_used`` and therefore made a real applied
                    # mutation look like an empty completion contract.
                    successful: set[str] = set()
                    verified: set[str] = set()
                    for row in trace:
                        if not isinstance(row, Mapping) or row.get("dispatched") is not True:
                            continue
                        tool = str(row.get("tool") or "").strip()
                        if not tool:
                            continue
                        if tool == "repo.edit":
                            if (
                                str(row.get("verdict") or "").lower()
                                not in {"accepted", "unproven"}
                                or row.get("applied") is not True
                                or row.get("rolled_back") is True
                            ):
                                continue
                        if tool == "tests.run":
                            if row.get("verified") is not True or row.get("returncode") != 0:
                                continue
                            verified.add(tool)
                        successful.add(tool)
                    response_contract = body.get("hawking_completion_contract")
                    required = [
                        str(name).strip()
                        for name in (response_contract.get("required_successful_tools") or [])
                        if str(name).strip()
                    ]
                    unmet = [name for name in required if name not in successful]
                    required_verified = [
                        str(name).strip()
                        for name in (response_contract.get("required_verified_tools") or [])
                        if str(name).strip()
                    ]
                    unmet.extend(
                        f"{name}:verified"
                        for name in required_verified
                        if name not in verified
                    )
                    body_out["hawking"]["completion"] = {
                        "complete": not unmet,
                        "unmet": list(dict.fromkeys(unmet)),
                        "successful_tools": sorted(successful),
                        "verified_tools": sorted(verified),
                        "required_successful_tools": required,
                        "source": "hawkingd.chat_tools.trace",
                    }
                if tool_contract is not None:
                    body_out["hawking"]["tool_contract"] = tool_contract
                if session is not None:
                    from .chat_state import remember_turn
                    remember_turn(session, messages, visible_text(_text_of(result)),
                                  knowledge=stores.get("knowledge"))
                    session.save()
                    body_out["hawking"]["session"] = {
                        "id": session.id, "turns": session.turns,
                        "active_plan": session.active_plan,
                        "authority": session.authority,
                        "checkpoint": session.last_checkpoint or None}
                    if locals().get("tool_trace_path"):
                        body_out["hawking"]["session"]["tool_trace"] = tool_trace_path
                    if locals().get("compaction") is not None and compaction.compacted:
                        body_out["hawking"]["compaction"] = compaction.to_dict()
                return self._send(200, body_out)
            # Open WebUI normally uses the streaming path. Persist its durable
            # session before writing the SSE frames just as the JSON path does;
            # otherwise every successful browser answer renders and then
            # disappears from HAWKING's objective/plan state on refresh or daemon
            # restart. The transcript remains the client's; this saves only the
            # compact session working set.
            if session is not None:
                from .chat_state import remember_turn
                remember_turn(session, messages, visible_text(_text_of(result)),
                              knowledge=stores.get("knowledge"))
                session.save()
            # An SSE body has no Content-Length, so under HTTP/1.1 keep-alive a
            # client cannot tell where it ends and waits until it times out --
            # which is exactly what a browser chat looks like when it hangs
            # forever on a reply that was in fact delivered. Close the
            # connection to frame the end of the stream.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True
            for chunk in stream_frames(
                    result, getattr(backend, "identity", identity),
                    request_id=request_id):
                self.wfile.write(chunk)
                self.wfile.flush()

    return Handler


def _context_window(model: str) -> int:
    """The loaded body's usable input budget, from the existing authority.

    `context_budget.resolve` is the repo's single authority on window
    arithmetic. It is KEYWORD-ONLY, and calling it positionally raised a
    TypeError that a bare `except` swallowed -- returning a fallback that
    happened to equal the real value for this profile, so the broken path was
    invisible. The fallback is now deliberately NOT a plausible real number, so
    the next such failure shows up instead of hiding.
    """
    try:
        budget = resolve(model_path=str(model))
    except Exception:
        return _WINDOW_UNKNOWN
    for attr in ("usable_input_tokens", "per_request_ctx", "model_ceiling"):
        value = getattr(budget, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return _WINDOW_UNKNOWN


def _prompt_window(model: str) -> int:
    """The window CONTEXT_SHARE is a share OF: the body's ceiling.

    `_context_window` reports `usable_input_tokens`, which `context_budget`
    already computed as ceiling - generation_reserve - framing_reserve. That is
    the right answer to "how much input may I send" and the wrong input to
    CONTEXT_SHARE, whose whole job is to leave room for the answer -- room
    subtracted once already. Multiplying the two reserves the same tokens
    twice: 5632 * 0.55 = 3097 where the design intends 8192 * 0.55 = 4505.

    The self-development campaign ran 74 cycles against a body holding 31% less
    context than the arithmetic allows, on a path whose failures were all shaped
    like running out of room. Taking the reserve once still leaves it: 4505
    prompt + 2048 generation + 512 framing = 7065 of 8192.
    """
    try:
        budget = resolve(model_path=str(model))
    except Exception:
        return _WINDOW_UNKNOWN
    for attr in ("model_ceiling", "total_ctx", "per_request_ctx"):
        value = getattr(budget, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return _context_window(model)


def build_server(model: str, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 ready_timeout: float = 600.0, repo: Any = None,
                 registry: Any = None, write: bool = False,
                 webui_manager: Any = None,
                 delegation_supervisor: Any = None,
                 resolved_action: Any = None,
                 canonical_router: Any = None):
    """Spawn the resident, wait for it, and return (httpd, identity, health)."""
    from .catalog import ResolvedAction, resolve_action, revalidate_action
    from .runtime_iface import (
        make_backend_for_model,
        make_canonical_backend_for_model,
    )

    if resolved_action is not None:
        if not isinstance(resolved_action, ResolvedAction):
            raise TypeError("build_server requires a ResolvedAction value")
        selected = revalidate_action(resolved_action)
        if model and os.path.realpath(os.path.expanduser(model)) != selected.path:
            raise ValueError("--model does not match the resolved action path")
    else:
        selected = resolve_action(model, "serve")
    if selected is None or selected.body is None:
        raise LookupError(
            f"{model!r} has no admitted Hawking execution binding; "
            "source specimens must be qualified before serve")
    if selected.action not in {"serve", "web"}:
        raise PermissionError(
            f"resolved action {selected.action!r} cannot start a resident surface"
        )
    body = selected.body
    if canonical_router is not None and selected.kind != "noetic_native":
        raise PermissionError(
            "canonical hawkingd may start only an admitted Hawking-native/Noetic artifact"
        )
    if write:
        from .catalog import require_write_session_authority

        require_write_session_authority(body)

    backend_factory = (
        make_canonical_backend_for_model
        if canonical_router is not None else make_backend_for_model
    )
    backend = backend_factory(selected.path)
    backend.spawn()
    # Own teardown during the load phase too. The normal daemon handler is
    # installed after build_server returns, but a model can spend minutes in
    # ready(); TERM in that window previously killed only hawkingd and left its
    # provider child alive. Convert a load-time signal into a controlled stop,
    # then restore the caller's handlers before publishing the server.
    import signal as _signal
    previous_handlers: Dict[Any, Any] = {}

    def _interrupt_load(_signum, _frame):  # noqa: ANN001
        try:
            backend.stop()
        finally:
            raise SystemExit(0)

    for _sig in (_signal.SIGTERM, _signal.SIGHUP, _signal.SIGINT):
        try:
            previous_handlers[_sig] = _signal.getsignal(_sig)
            _signal.signal(_sig, _interrupt_load)
        except (ValueError, OSError):
            pass
    try:
        ready = backend.ready(ready_timeout)
    finally:
        for _sig, _handler in previous_handlers.items():
            try:
                _signal.signal(_sig, _handler)
            except (ValueError, OSError):
                pass
    if not ready:
        try:
            tail = getattr(backend, "log_tail", lambda *_: "(no log)")()
        finally:
            try:
                backend.stop()
            except Exception:
                pass
        raise RuntimeError(
            f"the resident did not become ready within {ready_timeout:.0f}s. "
            f"Its log tail is the evidence: "
            f"{tail!s:.400}"
        )
    resident = Resident(
        body, backend, resolved_action=selected, ready_timeout=ready_timeout,
        write=write, provider_owned=True, backend_factory=backend_factory,
    )
    identity = resident.identity
    greedy = resident.greedy
    health = {
        "status": "ok",
        "owner": {
            "daemon": "hawkingd",
            "pid": os.getpid(),
            "single_surface": True,
            "provider_children": "daemon-owned",
        },
        "resident": identity,
        "model": selected.path,
        "resolved_action": selected.to_wire(),
        "resident_binding": selected.binding,
        "sampling": "greedy-argmax" if greedy else "profile-default",
        "streaming": "single-chunk (the connector returns a completed string; "
                     "SSE shape is provided so browser clients render, not as "
                     "evidence of incremental decode)",
        "endpoints": [
            "/", "/health", "/api/status", "/api/runtime/roles",
            "/api/runtime/switch", "/api/goal", "/api/goal/status",
            "/api/goal/control", "/api/goal/result", "/api/goal/diff",
            "/api/goal/ui", "/v1/models", "/v1/chat/completions",
            "/v1/workers", "/v1/workers/action", *HAWKING_WEB_ENDPOINTS,
        ],
        "client_supervision": "hawkingd-owned Open WebUI children",
        "switchable": True,
        # The body's real window, so compaction triggers on the actual ceiling
        # rather than a constant. resolve() is the repo's existing authority on
        # this and already reads native profiles and GGUF headers.
        "context_window": _context_window(body.path),
        # The ceiling, for callers that must take the answer's reserve
        # THEMSELVES. context_window has already had it taken; a share applied
        # to that number reserves the same tokens twice.
        "prompt_window": _prompt_window(body.path),
    }
    if canonical_router is not None:
        health["canonical_runtime"] = canonical_router.status()
    if repo is not None:
        health["repo"] = {"name": repo.name, "root": str(repo.root),
                          "git": repo.is_git, "branch": repo.branch}
    if registry is not None:
        # CAPABILITY IS EARNED PER BODY. A body that cannot emit a typed action
        # must not be handed a tool contract: it answers in prose that sounds
        # like it searched, and a reader cannot tell that from an answer that
        # did. One cheap probe at load, and again after every switch.
        from .chat_tools import prompt_menu_names

        prefix = None
        if repo is not None:
            from .repo_context import inject
            prefix = inject([], repo)
        verdict = resident.qualify_tools(prefix, registry, write=write)
        health["tools"] = {
            "qualified": bool(verdict.get("qualified")),
            "names": (prompt_menu_names(registry, write=write)
                      if verdict.get("qualified") else []),
            "reason": verdict.get("reason"),
            "reply_excerpt": verdict.get("reply_excerpt"),
        }

    stores: Dict[str, Any] = {}
    if registry is not None:
        # DISK-FIRST OBSERVATIONS AND BIDIRECTIONAL EXPANSION, from stores that
        # already exist: PasteCache is a content-addressed text store whose
        # store() had no caller outside its own tests, and KnowledgeStore is a
        # bounded hot index over a gzip cold archive with a real recall() valve.
        # Neither is a new architecture; both were built and left unwired.
        workspace = str(repo.root) if repo is not None else os.getcwd()
        try:
            from .paste_cache import PasteCache
            stores["cache"] = PasteCache(workspace)
        except Exception:
            stores["cache"] = None
        try:
            from .knowledge import KnowledgeStore
            stores["knowledge"] = KnowledgeStore(workspace)
        except Exception:
            stores["knowledge"] = None
        health["observation_store"] = stores.get("cache") is not None
        health["knowledge_store"] = stores.get("knowledge") is not None
        stores["state_root"] = workspace
        health["sessions"] = True
        if write:
            # HANDS, ONLY WHEN ASKED FOR. The engine is what makes repo.edit
            # reachable; a read session simply has none, so the door is absent
            # from the menu rather than refused by persuasion.
            from .engine import Engine
            from .workspace import Workspace
            stores["engine"] = Engine(Workspace(workspace),
                                      runtime_provider=lambda: resident)
        health["authority"] = "write" if write else "read"
    # The body's own capability surface, so the model and the operator can both
    # see what it can do -- named as capability (INSPECT, BUILD, ...) rather
    # than as package, and probed against live machinery rather than asserted.
    try:
        from .capabilities import capability_map
        health["capabilities"] = capability_map(
            getattr(repo, "root", None) or os.getcwd()
        )["can"]
    except Exception:
        pass
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(resident, identity, greedy=greedy,
                                   health=health, repo=repo, registry=registry,
                                   stores=stores, webui_manager=webui_manager,
                                   delegation_supervisor=delegation_supervisor,
                                   endpoint_base=f"http://{host}:{port}/v1",
                                   canonical_router=canonical_router))
    httpd.backend = resident  # type: ignore[attr-defined]
    return httpd, identity, health


class _CanonicalWaitingBackend:
    """A real Hawking front door while native role qualification is pending.

    It owns no provider, cannot create one, and exists so clients can retain
    the canonical :8014 configuration while inspecting the exact missing role
    evidence.  Once a native artifact is admitted, restart hawkingd on the
    same port to load it through the normal resident lifecycle.
    """

    canonical_unbound = True
    native_tools = False
    greedy = False
    state = "waiting_for_native_runtime"
    tools_verdict = {
        "qualified": False,
        "reason": "no qualified native resident is loaded",
    }

    def __init__(self, router: Any, waiting_for: Any) -> None:
        self.router = router
        self.waiting_for = waiting_for
        self.error = str(waiting_for)

    @property
    def identity(self) -> str:
        return "hawkingd-awaiting-native"

    def catalog(self) -> List[Dict[str, Any]]:
        return self.router.public_models()

    def complete(self, _payload: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        del timeout
        raise ResidentUnavailable(
            "no qualified native Hawking resident is loaded; inspect "
            "/api/runtime/roles for the missing admission evidence"
        )

    def stop(self) -> Dict[str, bool]:
        return {"gone": True}


def build_canonical_server(
    model: str = DEFAULT_MODEL,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ready_timeout: float = 600.0,
    repo: Any = None,
    registry: Any = None,
    write: bool = False,
    webui_manager: Any = None,
    delegation_supervisor: Any = None,
    remote_goal_write: bool = False,
):
    """Build the one native-only hawkingd surface, bound or explicitly waiting."""
    from .canonical_runtime import CanonicalRoleRouter, RuntimeRoleUnavailable

    router = CanonicalRoleRouter()
    try:
        route = router.route(model, {"objective": "routine local work"})
    except RuntimeRoleUnavailable as waiting_for:
        if write and not remote_goal_write:
            raise PermissionError(
                "cannot grant write authority while canonical Hawking has no "
                "qualified native resident"
            ) from waiting_for
        backend = _CanonicalWaitingBackend(router, waiting_for)
        health = {
            "status": "waiting_for_native_runtime",
            "owner": {
                "daemon": "hawkingd",
                "pid": os.getpid(),
                "single_surface": True,
                "provider_children": "none while native role is withheld",
            },
            "resident": None,
            "model": None,
            "requested_model": model,
            "state": backend.state,
            "error": str(waiting_for),
            "endpoints": [
                "/", "/health", "/api/status", "/api/runtime/roles",
                "/api/runtime/switch", "/api/goal", "/api/goal/status",
                "/api/goal/control", "/api/goal/result", "/api/goal/diff",
                "/api/goal/ui", "/v1/models", "/v1/chat/completions",
                "/v1/workers", "/v1/workers/action", *HAWKING_WEB_ENDPOINTS,
            ],
            "canonical_runtime": router.status(),
            "switchable": False,
            "authority": "remote_goal_write" if remote_goal_write else "read",
        }
        stores: Dict[str, Any] = {}
        if remote_goal_write:
            # A remote Goal may use Hawking's existing transactional mutation
            # engine while the native resident role is still withheld.  This
            # does not make the waiting backend a model resident: the engine
            # only supplies repo.edit/test validation for a locally admitted
            # Goal, and generic requests remain unable to select write tools.
            workspace_root = str(
                getattr(repo, "root", None) or os.getcwd()
            )
            stores["state_root"] = workspace_root
            from .engine import Engine
            from .workspace import Workspace
            stores["engine"] = Engine(Workspace(workspace_root))
        httpd = ThreadingHTTPServer(
            (host, port),
            make_handler(
                backend, backend.identity, greedy=False, health=health,
                repo=repo, registry=registry, webui_manager=webui_manager,
                stores=stores,
                delegation_supervisor=delegation_supervisor,
                endpoint_base=f"http://{host}:{port}/v1",
                canonical_router=router,
            ),
        )
        httpd.backend = backend  # type: ignore[attr-defined]
        return httpd, backend.identity, health
    return build_server(
        route.action.path,
        host=host,
        port=port,
        ready_timeout=ready_timeout,
        repo=repo,
        registry=registry,
        write=write,
        webui_manager=webui_manager,
        delegation_supervisor=delegation_supervisor,
        resolved_action=route.action,
        canonical_router=router,
    )


def _stop_owned_backend_children(root: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Stop a serving backend and safely clean up only its verified children."""
    from .backends import terminate_pid
    from .resources import pid_is_alive, process_identity_status

    candidates = [root, getattr(root, "backend", None)]
    owned_pids: Dict[int, Optional[str]] = {}
    skipped: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate is None:
            continue
        process = getattr(candidate, "process", None)
        pid = getattr(candidate, "pid", None)
        if pid is None and process is not None:
            pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            recorded_start = getattr(candidate, "start_time", None)
            if recorded_start or pid not in owned_pids:
                owned_pids[pid] = recorded_start
    try:
        if root is not None:
            root.stop()
    except Exception:
        pass
    for pid, recorded_start in owned_pids.items():
        try:
            if not pid_is_alive(pid):
                continue
            identity_status = process_identity_status(pid, recorded_start)
            if identity_status == "matches":
                terminate_pid(pid, term_timeout=1.0, kill_timeout=1.0)
            else:
                skipped.append(
                    {"pid": pid, "action": "skipped", "identity": identity_status}
                )
        except (OSError, ProcessLookupError):
            pass
    return {"skipped": skipped}


def main(argv: Optional[list] = None) -> int:
    # This process is `hawkingd` in production. Enable its supervised Rust
    # Gravity discovery and process-observer children; short-lived HAWKING clients
    # keep compatibility fallbacks and never create competing resident state.
    os.environ.setdefault("HAWKING_NATIVE_GRAVITY", "1")
    os.environ.setdefault("HAWKING_NATIVE_PROCESS_SERVER", "1")
    ap = argparse.ArgumentParser(
        prog="hawking serve",
        description="OpenAI-compatible endpoint over the persistent Hawking resident.")
    # POSITIONAL, because `hawking serve Qwen3-14B` is what a person types. --model
    # stays as an alias so existing scripts keep working.
    ap.add_argument("model", nargs="?", default=None,
                    help="which body to load: a name from `hawking use`, or a path")
    ap.add_argument("--model", dest="model_flag", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--resolved-action-json", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--no-repo", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-tools", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true",
                    help="grant repo-scoped write authority to this session")
    ap.add_argument(
        "--remote-goal-write",
        action="store_true",
        help=(
            "grant repo-scoped mutation authority only to locally submitted "
            "remote Hawking Goals while no native role is loaded"
        ),
    )
    a = ap.parse_args(list(argv or []))
    a.model = a.model or a.model_flag
    selected_action = None
    if a.resolved_action_json:
        from .catalog import resolved_action_from_json
        try:
            selected_action = resolved_action_from_json(a.resolved_action_json)
            if selected_action.action not in {"serve", "web"}:
                raise ValueError(
                    f"resolved action {selected_action.action!r} cannot start a resident surface"
                )
            if a.model and os.path.realpath(os.path.expanduser(a.model)) != selected_action.path:
                raise ValueError("--model does not match the resolved action path")
            # The canonical front door re-resolves this immutable ID through
            # the native role/artifact router.  Do not hand a caller-supplied
            # path through as a provider selector.
            a.model = selected_action.name
        except (LookupError, PermissionError, ValueError) as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2

    # One machine, one long-lived Hawking execution surface.  The native backend may
    # own a provider child, but a second `serve` on a
    # different port would still load another body into UMA and was the direct
    # cause of the recent multi-instance pressure event.  The lease is held by
    # the daemon process for its entire lifetime; `hawking serve` and
    # `hawkingd serve` therefore share the same guard.
    from .hawkingd import DaemonAlreadyRunning, acquire_daemon_lease

    try:
        _daemon_lease = acquire_daemon_lease(
            "serve",
            {"model": a.model, "host": a.host, "port": a.port},
        )
    except DaemonAlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        return 17

    model = a.model or DEFAULT_MODEL
    # `ps` should describe the daemon, not the interpreter. The name is
    # deliberately model-neutral -- hawkingd may hold any body over its life --
    # with the current body in brackets for identification. Best-effort: absent
    # setproctitle, the process simply keeps its default name.
    # On macOS, setproctitle rewrites the useful argv display but also changes
    # the native process-name field of our dedicated Mach-O launcher back to
    # generic Python. Preserve the native hawkingd identity in production;
    # retain the title fallback for source/development launches.
    if Path(sys.executable).name != "hawkingd":
        try:
            import setproctitle as _spt
            _label = (
                Path(model).stem
                if ("/" in model or model.endswith(".json"))
                else model
            )
            _spt.setproctitle(f"hawkingd serve :{a.port} [{_label}]")
        except Exception:
            pass
    repo = None
    if not a.no_repo:
        from .repo_context import RepoContext, prewarm_native_gravity_index
        repo = RepoContext.detect(os.getcwd())
        if repo is not None:
            # Do not charge the first model/tool request for a bounded active
            # code index. The child is still owned by this hawkingd process;
            # this overlaps its initial build with unavoidable resident
            # startup and does not add a second daemon authority.
            prewarm_native_gravity_index(repo.root)
    registry = None
    if not a.no_tools:
        from .chat_tools import build_registry
        root = str(repo.root) if repo is not None else os.getcwd()
        registry = build_registry(
            root,
            root,
            write=bool(a.write or a.remote_goal_write),
        )
    from .web import OwnedWebUIManager
    webui_manager = OwnedWebUIManager()
    delegation_supervisor = DaemonDelegationSupervisor(
        repo.root if repo is not None else os.getcwd()
    )
    httpd, identity, health = build_canonical_server(
        model, host=a.host, port=a.port, ready_timeout=a.ready_timeout,
        repo=repo, registry=registry,
        write=bool(getattr(a, "write", False) or getattr(a, "remote_goal_write", False)),
        remote_goal_write=bool(getattr(a, "remote_goal_write", False)),
        webui_manager=webui_manager,
        delegation_supervisor=delegation_supervisor)
    print(json.dumps({"listening": f"http://{a.host}:{a.port}",
                      "openai_base_url": f"http://{a.host}:{a.port}/v1",
                      **health}), flush=True)
    # A MODEL SERVER MUST NOT SURVIVE ITS SURFACE. `hawking stop` sends SIGTERM,
    # whose default handler exits WITHOUT running the finally below, so every
    # stop and every switch orphaned a historical spawned model server. Measured: 11
    # orphans accumulated across one session, oldest 1h28m, holding ~7 GB of
    # swap-backed state -- swap fell from 9.7 GB to 2.5 GB when they were
    # reaped. Handle the signal so shutdown is shutdown.
    import signal as _signal
    _runtime_stopped = False

    def _stop_runtime() -> None:
        """Stop the resident and prove its provider child is gone.

        The normal backend stop path owns the child, but a provider can still
        survive a teardown race and become reparented to launchd. Capture the
        owned PID before stopping, then apply the same TERM/KILL/reap helper if
        it remains alive. Never scan by process name: only PIDs reachable from
        this daemon's backend object are in scope.
        """
        nonlocal _runtime_stopped
        if _runtime_stopped:
            return
        _runtime_stopped = True
        root = getattr(httpd, "backend", None)
        setattr(httpd, "provider_cleanup", _stop_owned_backend_children(root))

    def _shutdown(signum, _frame):  # noqa: ANN001
        _stop_runtime()
        raise SystemExit(0)

    for _sig in (_signal.SIGTERM, _signal.SIGHUP, _signal.SIGINT):
        try:
            _signal.signal(_sig, _shutdown)
        except (ValueError, OSError):
            pass
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        _stop_runtime()
        delegation_supervisor.close()
        webui_manager.close()
        _daemon_lease.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
