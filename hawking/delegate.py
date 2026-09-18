"""Claude -> HAWKING delegation surface: run / status / steer / result / abort.

This module is an ADAPTER. It owns no scheduling, no locking, no retry and no
persistence of its own. Everything durable here is one of:

* ``hawking.mission`` — ``Mission``, ``mission_dir``, ``mission_state_path``,
  ``Mission.checkpoint``, ``Mission.from_workspace``, ``Mission.cancel``
* ``hawking.steering`` — ``SteeringQueue`` / ``SteerEvent``
* ``hawking.resources`` — ``MutationLock`` (which is what carries ``pid_is_alive``
  and the process START TOKEN, so a recycled pid cannot impersonate a holder)
* ``hawking.persist`` — ``atomic_write_json`` / ``atomic_write_text``
* ``hawking.verifier_pipeline`` — ``run_pipeline`` / ``Verdict`` / ``ModelCaller``
* ``hawking.mutation`` — ``content_fingerprint``
* ``hawking.backends`` — ``_post_json`` for the real OpenAI-compatible endpoint

THE CLASSIFICATION LAW lives in exactly two functions and every path routes
through them:

* ``classify_claim(text, artifact)`` is the ONLY door into ``verified_facts``.
  ``artifact`` is a required positional parameter with no default, so a caller
  cannot promote a claim by forgetting an argument. A claim with no deterministic
  artifact — including anything a model said, at any level of confidence — is a
  hypothesis.
* ``verifier_outcome(record)`` is the ONLY judge of whether a verifier passed.
  ``TRUE`` with a nonzero exit code is ``failed``, not ``passed``.

``decide_verdict`` consumes both and can never return ACCEPT while a required
verifier is ``failed``.

Every verb is DISK-FIRST: ``status``, ``result``, ``steer`` and ``abort`` read
durable state and never require a live in-process ``Mission``. Only ``abort``
loads state through ``Mission`` machinery, and only to reuse ``cancel``.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import signal
import stat
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .mission import Mission, mission_dir, mission_state_path
from .paths import find_repo_root
from .persist import atomic_write_json
from .resources import MutationLock, process_start_token
from .steering import STEER_KINDS, SteeringQueue

SPEC_FILENAME = "delegation_spec.json"
ENVELOPE_FILENAME = "delegation_envelope.json"
CANCEL_FILENAME = "delegation_cancel.json"
DELEGATIONS_DIRNAME = "delegations"
SPEC_VERSION = 1
ENVELOPE_VERSION = 1
EXEC_VERB = "__delegate_exec"
DAEMON_EXEC_CAPABILITY_FILENAME = "daemon_delegate_executor_capability.json"
DAEMON_EXEC_CAPABILITY_ENV = "HAWKING_DAEMON_DELEGATION_CAPABILITY"
DAEMON_EXEC_CAPABILITY_SCHEMA = "hawking.daemon_delegate_capability.v1"

VERIFY_MODES = ("standard", "strict")
# Delegated work defaults to the Hawking daemon rather than its browser-facing
# Open WebUI peer.  Only hawkingd owns the restricted worker-mode enforcement
# and daemon-supervised executor tree; :8080 historically returned 405 for
# delegation posts and, if replaced by a generic OpenAI surface, could ignore
# the authority field entirely.
DEFAULT_ENDPOINT = "http://127.0.0.1:8011/v1/chat/completions"
# A delegated HAWKING worker has no authority to mutate through the model
# surface.  ``reason_only`` is the ordinary verifier-pipeline path: model
# calls propose and interpret, while the bounded shell runner is the separate,
# audited executor.  Read/search tool access is an explicit opt-in for the
# narrow delegation cases that actually need it.
SAFE_WORKER_MODES = ("reason_only", "read_only_research")
DEFAULT_WORKER_MODE = "reason_only"


def _is_hawkingd_endpoint(endpoint: str) -> bool:
    """Whether an endpoint is the local daemon surface that enforces workers.

    A compatible-looking OpenAI server is not sufficient here: only Hawkingd
    owns the process tree and applies the restricted worker policy before a
    provider sees a request.  Keep this predicate deliberately narrow so an
    ambient ``HAWKING_ENDPOINT`` cannot turn durable delegation into an orphaned
    client-side worker.
    """

    parsed = urllib.parse.urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and (parsed.hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}
        and port == 8011
    )


def _unsafe_unmanaged_delegation_enabled() -> bool:
    """Explicit, development-only escape hatch for legacy local tests.

    It is intentionally not exposed by the CLI and cannot be selected through
    ``HAWKING_ENDPOINT`` alone.  Production/ordinary delegation must remain
    owned by the daemon.
    """

    return os.environ.get("HAWKING_ALLOW_UNSUPERVISED_DELEGATION") == "1"
MAX_FIELD_CHARS = 4000
COMMAND_TIMEOUT_S = 600.0
ABORT_JOIN_TIMEOUT_S = 3.0


class DelegationError(RuntimeError):
    """Base for adapter refusals."""


class MissionNotFound(DelegationError):
    """No delegation spec at the resolved workspace."""


class DelegationBusy(DelegationError):
    """A live writer already holds this workspace."""


@dataclass(frozen=True)
class DaemonSupervisorWitness:
    """PID-reuse-safe identity of the daemon that spawned one executor."""

    pid: int
    start_token: str


def _daemon_executor_capability_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(Path(workspace)) / DAEMON_EXEC_CAPABILITY_FILENAME


def issue_daemon_executor_capability(
    workspace: Union[str, Path],
    *,
    supervisor: DaemonSupervisorWitness | None = None,
) -> str:
    """Issue one opaque, one-use executor capability for a spawned child.

    Hawkingd keeps the raw token only in the child's environment.  The durable
    mission directory receives its SHA-256, so invoking the hidden executor
    verb directly cannot impersonate daemon ownership by merely naming a
    workspace.  This is process-ownership fencing, not a new authorization
    system or a replacement for server-side worker enforcement.
    """

    ws = Path(workspace).resolve()
    path = _daemon_executor_capability_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise DelegationError(f"daemon executor capability is already pending for {ws}")
    token = secrets.token_urlsafe(32)
    document = {
        "schema": DAEMON_EXEC_CAPABILITY_SCHEMA,
        "workspace": str(ws),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "issued_at": time.time(),
        "supervisor_pid": None if supervisor is None else supervisor.pid,
        "supervisor_start_token": None if supervisor is None else supervisor.start_token,
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DelegationError(f"daemon executor capability is already pending for {ws}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
    return token


def revoke_daemon_executor_capability(workspace: Union[str, Path]) -> None:
    """Remove an unconsumed daemon-issued capability after a failed spawn."""

    path = _daemon_executor_capability_path(workspace)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise DelegationError("daemon executor capability path is not a regular file")
    path.unlink()


def _consume_daemon_executor_capability(
    workspace: Union[str, Path],
) -> DaemonSupervisorWitness | None:
    """Atomically consume the daemon-only capability before executing a mission."""

    raw = os.environ.pop(DAEMON_EXEC_CAPABILITY_ENV, "")
    if _unsafe_unmanaged_delegation_enabled() and not raw:
        # This deliberately noisy development escape remains separate from an
        # ambient endpoint choice.  It is useful only for explicit local
        # compatibility tests; normal CLI execution cannot reach this branch.
        return None
    if not raw:
        raise DelegationError("__delegate_exec is reserved for a Hawkingd-supervised child")
    ws = Path(workspace).resolve()
    path = _daemon_executor_capability_path(ws)
    claimed = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.claimed")
    try:
        os.rename(path, claimed)
    except FileNotFoundError as exc:
        raise DelegationError("daemon executor capability is absent, stale, or already consumed") from exc
    try:
        metadata = claimed.lstat()
        if not stat.S_ISREG(metadata.st_mode) or claimed.is_symlink():
            raise DelegationError("daemon executor capability is not a regular file")
        document, defect = _read_json(claimed)
        if not isinstance(document, dict) or defect:
            raise DelegationError("daemon executor capability is malformed")
        expected = document.get("token_sha256")
        observed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if (
            document.get("schema") != DAEMON_EXEC_CAPABILITY_SCHEMA
            or document.get("workspace") != str(ws)
            or not isinstance(expected, str)
            or not hmac.compare_digest(expected, observed)
        ):
            raise DelegationError("daemon executor capability does not bind this child workspace")
        raw_pid = document.get("supervisor_pid")
        raw_start = document.get("supervisor_start_token")
        if raw_pid is None and raw_start is None:
            return None
        if (
            isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
            or raw_pid <= 0
            or not isinstance(raw_start, str)
            or not raw_start
        ):
            raise DelegationError("daemon executor capability has an invalid supervisor witness")
        return DaemonSupervisorWitness(raw_pid, raw_start)
    finally:
        claimed.unlink(missing_ok=True)


def _supervisor_witness_is_live(witness: DaemonSupervisorWitness) -> bool:
    return process_start_token(witness.pid) == witness.start_token


class _VerifierProcessRegistry:
    """The foreground verifier processes owned by one executor.

    ``shell=True`` starts a shell which may in turn start a verifier process.
    The registry makes launch atomic with daemon-loss cleanup. The supervised
    executor itself has a dedicated POSIX process group, which contains ordinary
    verifier descendants even after a shell exits. This is deliberately
    per-executor state rather than another durable scheduler or supervisor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[Any] = set()
        self._closed = False
        # Keep this sticky rather than deriving it from ``_processes``.  A
        # shell leader can finish and unregister while an ordinary descendant
        # from that verifier remains in the executor group.  Final cleanup
        # must still know that this executor crossed the only boundary that
        # can create such a descendant.
        self._launched_verifier = False

    def spawn(self, factory: Callable[[], Any]) -> Any:
        """Create and register a verifier atomically with daemon-loss cleanup."""

        with self._lock:
            if self._closed:
                raise DelegationError("Hawkingd supervisor was lost before verifier launch")
            process = factory()
            self._processes.add(process)
            self._launched_verifier = True
            return process

    def unregister(self, process: Any) -> None:
        with self._lock:
            self._processes.discard(process)

    def has_launched_verifier(self) -> bool:
        """Whether this executor ever created a same-group verifier leader.

        This is intentionally an audit/control bit, not a claim that a child
        remains live.  A completed shell can leave an ordinary descendant that
        is no longer present in the foreground registry.
        """

        with self._lock:
            return self._launched_verifier

    def terminate_all(self) -> None:
        """Prevent new launches and end registered shell leaders before exit.

        A shell registered by a Hawkingd-supervised executor inherits the
        executor's dedicated process group.  Its PID is therefore *not* a
        private process-group ID and must not be passed to ``killpg``.  The
        watchdog follows this bounded leader cleanup with an executor-group
        kill, which is the operation that reaches ordinary descendants.
        """

        with self._lock:
            self._closed = True
            processes = tuple(self._processes)
        for process in processes:
            _terminate_verifier_process(process, group=False)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if all(process.poll() is not None for process in processes):
                return
            time.sleep(0.02)
        for process in processes:
            _terminate_verifier_process(process, force=True, group=False)

    def terminate_supervised_executor_group(self) -> None:
        """Fail closed on verifier timeout: the executor group owns descendants.

        A supervised shell inherits the dedicated executor group rather than
        becoming a separate session. Once a verifier times out, terminating
        only its shell could leave a descendant behind, so the safe result is
        to end the whole delegated executor. In production ``SIGKILL`` never
        returns here; the exception is a conservative fallback for an
        unsupported or mocked process-group operation.
        """

        _terminate_owned_supervised_executor_group()
        raise DelegationError(
            "supervised verifier timeout requires POSIX executor-group ownership"
        )


def _terminate_owned_supervised_executor_group() -> None:
    """End the current daemon-owned executor group, including stale children.

    Hawkingd launches the executor in a fresh session, so its PID is also the
    executor-only process-group ID.  This helper is intentionally called from
    the executor itself: unlike a later parent-side reaper, it cannot mistake a
    recycled PID for its former group.  A successful POSIX ``SIGKILL`` does not
    return; the normal return path exists only for test doubles.
    """

    if os.name != "posix":
        raise DelegationError(
            "supervised executor cleanup requires POSIX executor-group ownership"
        )
    try:
        if os.getpgrp() != os.getpid():
            raise DelegationError(
                "supervised executor lacks an executor-only process group for cleanup"
            )
        os.killpg(os.getpid(), signal.SIGKILL)
    except ProcessLookupError as exc:
        raise DelegationError(
            "supervised executor process group disappeared before cleanup"
        ) from exc
    except (OSError, PermissionError) as exc:
        raise DelegationError(
            "cannot terminate supervised executor group during cleanup"
        ) from exc


def _install_supervised_executor_sigterm_cleanup() -> Any:
    """Install the witnessed executor's minimal SIGTERM containment hook.

    Python's default ``SIGTERM`` action bypasses ``exec_main``'s ``finally``.
    That is unsafe once a host-owned verifier may have created an ordinary
    same-group descendant: there is a small launch/bookkeeping interval in
    which even a sticky registry bit cannot yet describe the child.  The
    executor itself still owns the only unambiguous group identity, so the
    handler immediately ends that group.  It deliberately takes no lock,
    performs no disk I/O, and does not signal a parent-side PID/PGID.

    This is installed only after the daemon witness and dedicated-group proof
    in ``_start_supervisor_watchdog``.  A successful group kill never returns;
    ``os._exit`` is a fail-closed fallback for an unsupported or mocked signal
    operation, where returning into an interrupted mission could leave a
    descendant running.
    """

    if os.name != "posix":
        raise DelegationError(
            "supervised executor SIGTERM cleanup requires POSIX process-group ownership"
        )

    def _cleanup(_signum: int, _frame: Any) -> None:
        try:
            _terminate_owned_supervised_executor_group()
        except BaseException:
            os._exit(75)
        # A test double can make the helper return; never resume a mission
        # after the containment path has been selected.
        os._exit(75)

    try:
        return signal.signal(signal.SIGTERM, _cleanup)
    except (OSError, ValueError) as exc:
        raise DelegationError(
            "cannot install Hawkingd-supervised executor SIGTERM cleanup"
        ) from exc


def _terminate_verifier_process(
    process: Any,
    *,
    force: bool = False,
    group: bool = True,
    include_exited_leader: bool = False,
) -> None:
    """Terminate one shell-led verifier, optionally including its own group.

    Only direct ``shell_runner`` use creates a shell-owned session, so only
    that path may use the shell PID as a process-group ID.  A supervised shell
    shares the executor group and is terminated by PID here; daemon-loss or
    timeout cleanup owns the executor group separately.
    """

    if process.poll() is not None and not (group and include_exited_leader):
        return
    signal_to_send = signal.SIGKILL if force else signal.SIGTERM
    if group and os.name == "posix":
        try:
            # Direct shell_runner creates a new session, so process.pid is its
            # own PGID. Supervised callers pass group=False above.
            os.killpg(process.pid, signal_to_send)
            return
        except ProcessLookupError:
            return
        except (OSError, PermissionError):
            # Preserve a bounded best effort on a platform where group signals
            # are unavailable; this path cannot claim descendant coverage.
            pass
    if process.poll() is None:
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except OSError:
            pass


def _start_supervisor_watchdog(
    witness: DaemonSupervisorWitness | None,
    *,
    on_supervisor_lost: Callable[[], None] | None = None,
) -> threading.Event | None:
    """End an executor and its registered verifier groups if Hawkingd dies."""

    if witness is None:
        return None
    if not _supervisor_witness_is_live(witness):
        raise DelegationError("Hawkingd supervisor died before its delegated child began")
    if os.name == "posix":
        try:
            if os.getpgrp() != os.getpid():
                raise DelegationError(
                    "Hawkingd-supervised executor lacks its dedicated process group"
                )
        except OSError as exc:
            raise DelegationError(
                "cannot verify Hawkingd-supervised executor process-group ownership"
            ) from exc
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.25):
            if not _supervisor_witness_is_live(witness):
                # The mutation lock is PID/start-token checked and will become
                # stale safely. Stop registered shell leaders first; the
                # executor-group kill below then reaches their ordinary
                # same-group descendants without treating a shell PID as a
                # separate process-group ID.
                if on_supervisor_lost is not None:
                    try:
                        on_supervisor_lost()
                    except BaseException:
                        # Parent loss is terminal even if a best-effort child
                        # cleanup itself encounters a platform failure.
                        pass
                if os.name == "posix":
                    try:
                        # DaemonDelegationSupervisor starts this executor in
                        # its own session. This reaches ordinary descendants
                        # which may outlive a foreground shell, rather than
                        # only the executor PID itself.
                        os.killpg(os.getpid(), signal.SIGKILL)
                    except (OSError, PermissionError):
                        pass
                os._exit(75)

    threading.Thread(target=watch, name="hawking-daemon-parent-watch", daemon=True).start()
    return stop


def _safe_worker_mode(raw: object) -> str:
    """Return the one server-enforced mode a delegation may request.

    This is client-side contract validation, not the permission boundary: the
    Hawking daemon parses and enforces ``hawking_worker_mode`` again before a
    provider call.  Refusing ``default`` here prevents a historical or hand
    edited delegation spec from silently restoring broad server authority.
    """
    if raw is None:
        mode = DEFAULT_WORKER_MODE
    elif isinstance(raw, str):
        mode = raw.strip().lower()
    else:
        mode = ""
    if mode not in SAFE_WORKER_MODES:
        allowed = ", ".join(SAFE_WORKER_MODES)
        raise DelegationError(
            f"worker_mode must be one of {allowed}, got {raw!r}"
        )
    return mode


# ---------------------------------------------------------------------------
# Paths. mission_dir() is the authority; nothing here invents a layout.
# ---------------------------------------------------------------------------


def delegations_root(root: Optional[Union[str, Path]] = None) -> Path:
    return Path(root or os.getcwd()) / ".hawking" / DELEGATIONS_DIRNAME


def spec_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / SPEC_FILENAME


def envelope_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / ENVELOPE_FILENAME


def cancel_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / CANCEL_FILENAME


def resolve_workspace(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Path:
    """A mission handle is either a workspace path or a mission id.

    The filesystem is the registry: a mission id names a directory under
    ``<root>/.hawking/delegations/``. No index file, so nothing can go stale.
    """
    token = os.fspath(mission)
    candidates = [Path(token), delegations_root(root) / token]
    for ws in candidates:
        if spec_path(ws).is_file():
            return ws
    raise MissionNotFound(
        f"no {SPEC_FILENAME} for mission {token!r}; looked in "
        + ", ".join(str(mission_dir(c)) for c in candidates)
    )


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """(value, defect). Never raises, never guesses a value."""
    if not path.is_file():
        return None, f"missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable {path.name}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# THE CLASSIFICATION LAW
# ---------------------------------------------------------------------------

_ARTIFACT_KINDS = ("receipt", "command")


@dataclass(frozen=True)
class Artifact:
    """A deterministic support. Either a path that exists, or a command run."""

    kind: str
    ref: str
    exit_code: Optional[int] = None
    output: Optional[str] = None
    observed_at: Optional[float] = None
    # What exit code makes this command SUPPORT its claim. Defaults to 0, so a
    # command that FAILED cannot back an unqualified success claim. A refutation
    # sets this to the failing code, because there the failure IS the evidence.
    expected_exit: Optional[int] = 0

    def to_dict(self) -> Dict[str, Any]:
        text, clipped = _clip(self.output)
        return {
            "kind": self.kind,
            "ref": self.ref,
            "exit_code": self.exit_code,
            "expected_exit": self.expected_exit,
            "output": text,
            "output_truncated": clipped,
            "observed_at": self.observed_at,
        }


def artifact_defect(artifact: Optional[Artifact]) -> Optional[str]:
    """Why this artifact cannot support a verified fact. None = it can."""
    if artifact is None:
        return "no deterministic artifact supplied"
    if not isinstance(artifact, Artifact):
        return f"not an Artifact record: {type(artifact).__name__}"
    if artifact.kind not in _ARTIFACT_KINDS:
        return f"unknown artifact kind {artifact.kind!r}"
    ref = (artifact.ref or "").strip()
    if not ref:
        return f"{artifact.kind} artifact has an empty ref"
    if artifact.kind == "receipt":
        if not Path(ref).exists():
            return f"receipt path does not exist on disk: {ref}"
        return None
    if not isinstance(artifact.exit_code, int) or isinstance(artifact.exit_code, bool):
        return "command artifact has no recorded integer exit code"
    # A FAILED command is evidence of failure, not of the claim. The adversarial
    # audit found this promotion latent -- unreachable through build_envelope
    # today, but live the moment a new caller passes a nonzero code -- and the
    # steer requires the distinction be hard to violate ACCIDENTALLY. The escape
    # hatch is explicit, never a default: a refutation declares the failing code
    # as expected, so `verifier for X did not pass (exit_code=1)` stays verifiable.
    # This mirrors the AKB rule already in this repo: "evidence_class Measured but
    # source records pass: false -- a failed run is evidence of failure".
    expected = artifact.expected_exit
    if not isinstance(expected, int) or isinstance(expected, bool):
        return "command artifact has no integer expected_exit"
    if artifact.exit_code != expected:
        return (f"command exited {artifact.exit_code}, expected {expected}: a command "
                f"that did not do what was expected is evidence of that, not of the claim")
    return None


def classify_claim(text: str, artifact: Optional[Artifact]) -> Dict[str, Any]:
    """THE ONE DOOR into ``verified_facts``.

    ``artifact`` is positional and has NO default: promotion cannot happen by
    omission. Model prose reaches this function with ``artifact=None`` and
    stays a hypothesis no matter how it is worded.
    """
    claim = str(text)
    defect = artifact_defect(artifact)
    if defect is None:
        assert isinstance(artifact, Artifact)  # narrowed by artifact_defect
        return {
            "claim": claim,
            "class": "verified",
            "artifact": artifact.to_dict(),
            "reason": None,
        }
    return {
        "claim": claim,
        "class": "hypothesis",
        "artifact": artifact.to_dict() if isinstance(artifact, Artifact) else None,
        "reason": defect,
    }


def verifier_outcome(record: Dict[str, Any]) -> str:
    """THE ONE JUDGE: 'passed' | 'failed' | 'unresolved'.

    A model verdict of TRUE against a nonzero exit code is ``failed``. This is
    the function the adversarial mutation test (H) attacks; if mutating it does
    not break test A, the classifier is decoration.
    """
    verdict = str(record.get("verdict") or "").strip().upper()
    code = record.get("exit_code")
    if verdict == "TRUE" and isinstance(code, int) and code == 0:
        return "passed"
    if verdict == "UNVERIFIABLE":
        return "unresolved"
    return "failed"


def decide_verdict(
    *,
    verified_facts: Sequence[Dict[str, Any]],
    verifier_records: Sequence[Dict[str, Any]],
    blocker: Optional[str],
    defects: Sequence[str],
    remaining: Sequence[str],
) -> str:
    """ACCEPT | BLOCKED | INCONCLUSIVE. Never ACCEPT over a failed verifier."""
    if defects:
        return "INCONCLUSIVE"
    failed = [r for r in verifier_records if verifier_outcome(r) == "failed"]
    required_failed = [r for r in failed if r.get("required")]
    if required_failed:
        return "BLOCKED"
    if blocker:
        return "BLOCKED"
    if failed:
        # Non-required verifier failed: not a blocker, but not a clean accept.
        return "INCONCLUSIVE"
    if remaining:
        return "INCONCLUSIVE"
    if not verified_facts:
        return "INCONCLUSIVE"
    return "ACCEPT"


# Authority tiers. An actual command run beats a recorded number; newer durable
# disk state beats a stale receipt.
_AUTHORITY_TIER = {"command": 3, "disk": 2, "receipt": 1, "recorded": 0}


def pick_authority(
    candidates: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Resolve conflicting evidence for one value.

    Each candidate is ``{"value":..., "source": command|disk|receipt|recorded,
    "observed_at": float|None}``. Higher tier wins outright; within a tier the
    newer observation wins. ``None`` values never win.
    """
    usable = [c for c in candidates if c.get("value") is not None]
    if not usable:
        return None
    return max(
        usable,
        key=lambda c: (
            _AUTHORITY_TIER.get(str(c.get("source")), 0),
            float(c.get("observed_at") or 0.0),
        ),
    )


def _clip(text: Any, limit: int = MAX_FIELD_CHARS) -> Tuple[Optional[str], bool]:
    if text is None:
        return None, False
    body = text if isinstance(text, str) else str(text)
    if len(body) <= limit:
        return body, False
    return body[:limit] + f"\n[...truncated {len(body) - limit} chars]", True


# ---------------------------------------------------------------------------
# run — writes the spec BEFORE any work, returns a mission id, does not block
# ---------------------------------------------------------------------------


def build_spec(
    objective: str,
    *,
    mission_id: str,
    constraints: Sequence[str] = (),
    allowed_resources: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
    verification: str = "standard",
    budget: Optional[Dict[str, Any]] = None,
    output_contract: Sequence[str] = (),
    endpoint: Optional[str] = None,
    worker_mode: str = DEFAULT_WORKER_MODE,
) -> Dict[str, Any]:
    mode = (verification or "standard").strip().lower()
    if mode not in VERIFY_MODES:
        raise DelegationError(
            f"verification must be one of {VERIFY_MODES}, got {verification!r}"
        )
    if not (objective or "").strip():
        raise DelegationError("run requires a non-empty objective")
    return {
        "spec_version": SPEC_VERSION,
        "mission_id": mission_id,
        "objective": objective.strip(),
        "constraints": [str(c) for c in constraints],
        "allowed_resources": [str(r) for r in allowed_resources],
        "protected_paths": [str(p) for p in protected_paths],
        "verification": mode,
        "budget": dict(budget) if budget else None,
        "output_contract": [str(p) for p in output_contract],
        "endpoint": endpoint or os.environ.get("HAWKING_ENDPOINT", DEFAULT_ENDPOINT),
        "worker_mode": _safe_worker_mode(worker_mode),
        "created_at": time.time(),
    }


def run(
    objective: str,
    *,
    root: Optional[Union[str, Path]] = None,
    workspace: Optional[Union[str, Path]] = None,
    spawn: bool = True,
    **spec_kwargs: Any,
) -> Dict[str, Any]:
    """Start a delegated mission. Returns immediately with a mission id."""
    mission_id = str(spec_kwargs.pop("mission_id", "") or uuid.uuid4().hex[:12])
    ws = Path(workspace) if workspace is not None else delegations_root(root) / mission_id
    ws.mkdir(parents=True, exist_ok=True)

    # Exclusivity is MutationLock's job. run() does not become the writer; it
    # only refuses to start a second one over a live holder.
    lock = MutationLock(ws)
    if not lock.try_break_stale():
        record = lock.read() or {}
        raise DelegationBusy(
            f"workspace {ws} already has a live writer (pid={record.get('pid')}, "
            f"unit={record.get('unit_id')})"
        )

    spec = build_spec(objective, mission_id=mission_id, **spec_kwargs)
    spec["fingerprint_at_start"] = _fingerprint(spec["output_contract"], ws)
    mission_dir(ws).mkdir(parents=True, exist_ok=True)
    atomic_write_json(spec_path(ws), spec)

    # units={} on purpose. Mission(goal=...) would compile the objective into
    # `implement`/`validate` scheduler units, and this adapter never schedules
    # a unit — obligations are settled through verifier_pipeline. Left to
    # default they sit `pending` forever and get reported as real outstanding
    # work in every status and every envelope, including accepted ones.
    mission = Mission(
        ws, goal=spec["objective"], mission_id=mission_id, quiet=True, units={}
    )
    mission.phase = "delegated"
    mission.checkpoint()

    pid = None
    if spawn:
        pid = _spawn_executor(ws)
        if pid is not None:
            mission.register_child_pid(pid)
    return {
        "mission_id": mission_id,
        "workspace": str(ws),
        "spec": str(spec_path(ws)),
        "state": str(mission_state_path(ws)),
        "writer_pid": pid,
        "spawned": bool(pid),
    }


def _spawn_executor(workspace: Path) -> Optional[int]:
    """Ask the live Hawking daemon to own production delegation workers.

    A resident endpoint on :8011 is the production Hawking surface.  Falling
    back to a detached client-owned Python process there would recreate the
    very orphan-provider pattern the daemon exists to eliminate.  Older
    daemon images therefore leave the mission queued until a controlled
    restart installs the supervision endpoint.  A non-daemon endpoint leaves
    the mission queued unless the process owner explicitly enables the
    development-only unmanaged escape hatch.
    """
    spec, _ = _read_json(spec_path(workspace))
    endpoint = str((spec or {}).get("endpoint") or "") if isinstance(spec, dict) else ""
    if _is_hawkingd_endpoint(endpoint):
        parsed = urllib.parse.urlsplit(endpoint)
        base = urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, "", "", ""))
        request = urllib.request.Request(
            base.rstrip("/") + "/hawkingd/delegations/start",
            data=json.dumps({"workspace": str(workspace.resolve())}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            pid = payload.get("pid") if isinstance(payload, dict) else None
            if isinstance(pid, int) and pid > 0:
                return pid
            return None
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            # Fail closed for the live daemon surface.  A later controlled
            # daemon restart will expose the endpoint and adopt this mission;
            # silently detaching it from hawking would violate process ownership.
            return None
    if not _unsafe_unmanaged_delegation_enabled():
        # A custom HAWKING_ENDPOINT is configuration, not authorization to evade
        # Hawkingd's supervision and server-side worker fence.  Leave the
        # mission durable and queued so an operator can correct the endpoint;
        # never recreate the historical detached-worker fallback.
        return None
    repo = Path(__file__).resolve().parent.parent
    log = mission_dir(workspace) / "delegate_exec.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    capability = issue_daemon_executor_capability(workspace)
    child_env = dict(os.environ)
    child_env[DAEMON_EXEC_CAPABILITY_ENV] = capability
    try:
        handle = open(log, "ab")
    except OSError:
        revoke_daemon_executor_capability(workspace)
        return None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hawking", EXEC_VERB, str(workspace)],
            cwd=str(repo),
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
    except OSError:
        revoke_daemon_executor_capability(workspace)
        return None
    finally:
        handle.close()
    return proc.pid


def _fingerprint(paths: Sequence[str], workspace: Path) -> Optional[str]:
    if not paths:
        return None
    from .mutation import content_fingerprint

    try:
        return content_fingerprint(list(paths), root=str(workspace))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# status / steer / abort — disk-first
# ---------------------------------------------------------------------------


def status(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Live snapshot from DURABLE state only. No in-process Mission needed."""
    ws = resolve_workspace(mission, root)
    spec, spec_defect = _read_json(spec_path(ws))
    state, state_defect = _read_json(mission_state_path(ws))
    envelope, _ = _read_json(envelope_path(ws))
    cancel, _ = _read_json(cancel_path(ws))
    spec = spec if isinstance(spec, dict) else {}
    state = state if isinstance(state, dict) else {}

    lock = MutationLock(ws)
    record = lock.read()
    writer = None
    if record:
        writer = {
            "pid": record.get("pid"),
            "unit_id": record.get("unit_id"),
            "acquired_at": record.get("acquired_at"),
            "alive": lock.holder_is_live(record),
        }

    units = state.get("units") if isinstance(state.get("units"), dict) else {}
    by_status: Dict[str, int] = {}
    for payload in units.values():
        if isinstance(payload, dict):
            key = str(payload.get("status") or "unknown")
            by_status[key] = by_status.get(key, 0) + 1

    return {
        "mission_id": spec.get("mission_id") or state.get("id"),
        "workspace": str(ws),
        "objective": spec.get("objective"),
        "verification": spec.get("verification"),
        "created_at": spec.get("created_at"),
        "phase": state.get("phase"),
        "last_checkpoint": state.get("last_checkpoint"),
        "units_by_status": by_status,
        "writer": writer,
        "steers_pending": len(_pending_steers(ws, state)),
        "cancel_requested": bool(cancel),
        "cancel_reason": (cancel or {}).get("reason") or state.get("cancel_reason"),
        "envelope_present": isinstance(envelope, dict),
        "verdict": (envelope or {}).get("verdict"),
        "blocker": (envelope or {}).get("blocker"),
        "defects": [d for d in (spec_defect, state_defect) if d],
    }


def _steering_queue(ws: Path, state: Optional[Dict[str, Any]] = None) -> SteeringQueue:
    if state is None:
        state, _ = _read_json(mission_state_path(ws))
    state = state if isinstance(state, dict) else {}
    spec, _ = _read_json(spec_path(ws))
    spec = spec if isinstance(spec, dict) else {}
    session = state.get("session_id") or spec.get("mission_id") or state.get("id")
    if not session:
        raise MissionNotFound(f"no session id recorded for {ws}")
    return SteeringQueue(str(ws), str(session))


def _pending_steers(ws: Path, state: Optional[Dict[str, Any]] = None) -> List[Any]:
    try:
        return _steering_queue(ws, state).pending()
    except Exception:
        return []


def steer(
    mission: Union[str, Path],
    text: str,
    kind: str = "knowledge",
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Queue a steer on DISK. Applies to future work; never rewrites history.

    Deliberately does NOT reload a ``Mission``: ``Mission.from_workspace`` fails
    every unit it finds ``running``, which would corrupt a live mission just
    because someone typed a steer. The executor owns the ledger and consumes
    the queue between obligations.
    """
    token = (kind or "knowledge").strip().lower()
    if token not in STEER_KINDS:
        raise DelegationError(f"kind must be one of {STEER_KINDS}, got {kind!r}")
    if not (text or "").strip():
        raise DelegationError("steer requires non-empty text")
    ws = resolve_workspace(mission, root)
    event = _steering_queue(ws).enqueue(text.strip(), kind=token)
    return {
        "mission_id": status(ws)["mission_id"],
        "steer_id": event.id,
        "kind": event.kind,
        "queued_at": event.timestamp,
        "applies_to": "future work only",
    }


def abort(
    mission: Union[str, Path],
    reason: str = "aborted by operator",
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Stop a delegation and leave lock + state recoverable."""
    ws = resolve_workspace(mission, root)
    reason = (reason or "aborted by operator").strip()
    atomic_write_json(cancel_path(ws), {"reason": reason, "at": time.time()})

    lock = MutationLock(ws)
    record = lock.read() or {}
    signalled = None
    pid = record.get("pid")
    try:
        pid = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid = None
    if pid and pid != os.getpid() and lock.holder_is_live(record):
        try:
            os.kill(pid, signal.SIGTERM)
            signalled = pid
        except OSError:
            signalled = None

    if signalled:
        # Bounded wait so the reported lock_free is usually true rather than
        # merely eventually true. A holder that outlives it is reported
        # honestly as lock_free=False; the next run() breaks it as stale.
        deadline = time.time() + ABORT_JOIN_TIMEOUT_S
        while time.time() < deadline and lock.holder_is_live(lock.read()):
            time.sleep(0.05)

    lock.release(record.get("unit_id"))
    lock_free = lock.try_break_stale()

    # Same contract as Mission.checkpoint(): DAG first, then state.json,
    # both stamped with one checkpoint_id. Patching state alone used to
    # leave a cancelled generation that dag.json could not name.
    state, defect = _read_json(mission_state_path(ws))
    if isinstance(state, dict):
        try:
            loaded = Mission.from_workspace(ws, quiet=True)
            loaded.cancel(reason)
            loaded.phase = "cancelled"
            loaded.checkpoint()
        except Exception as exc:
            defect = defect or f"checkpoint failed: {type(exc).__name__}: {exc}"

    envelope = build_envelope(ws, aborted_reason=reason)
    atomic_write_json(envelope_path(ws), envelope)
    return {
        "mission_id": envelope.get("mission_id"),
        "workspace": str(ws),
        "reason": reason,
        "signalled_pid": signalled,
        "lock_free": bool(lock_free),
        "verdict": envelope.get("verdict"),
        "state_defect": defect,
    }


# ---------------------------------------------------------------------------
# result — the envelope
# ---------------------------------------------------------------------------


def result(
    mission: Union[str, Path], root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Return the envelope. Rebuilt from durable state when work is unfinished."""
    ws = resolve_workspace(mission, root)
    stored, _ = _read_json(envelope_path(ws))
    if isinstance(stored, dict) and stored.get("envelope_version") == ENVELOPE_VERSION:
        return stored
    return build_envelope(ws)


def _verifier_records(
    pipeline: Optional[Dict[str, Any]], mode: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize pipeline output. Missing fields are DEFECTS, never guesses."""
    records: List[Dict[str, Any]] = []
    defects: List[str] = []
    if pipeline is None:
        return records, defects
    if not isinstance(pipeline, dict):
        return records, [f"pipeline result is {type(pipeline).__name__}, not an object"]
    raw = pipeline.get("verdicts")
    if raw is None:
        return records, ["pipeline result has no 'verdicts'"]
    if not isinstance(raw, list):
        return records, ["pipeline 'verdicts' is not a list"]
    statements = {}
    for ob in pipeline.get("obligations") or []:
        if isinstance(ob, dict) and ob.get("id"):
            statements[str(ob["id"])] = ob
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            defects.append(f"verdict[{index}] is not an object")
            continue
        oid = str(item.get("obligation_id") or "")
        verdict = item.get("verdict")
        if verdict is None:
            defects.append(f"verdict[{index}] ({oid or 'unknown'}) has no verdict field")
        ob = statements.get(oid, {})
        consequential = ob.get("consequential")
        records.append(
            {
                "obligation_id": oid or None,
                "statement": ob.get("statement") or oid or None,
                "verdict": verdict,
                "command": item.get("command"),
                "exit_code": item.get("exit_code"),
                "output": item.get("output"),
                "evidence": item.get("evidence"),
                # strict: every verifier is required. standard: only the
                # obligations the planner marked consequential.
                "required": True if mode == "strict" else bool(consequential),
            }
        )
    return records, defects


def _tests_field(
    records: Sequence[Dict[str, Any]], spec: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Reconcile any recorded test count against an actual run. Run wins."""
    from .verifier_pipeline import _pytest_passed_count

    candidates: List[Dict[str, Any]] = []
    recorded = (spec.get("budget") or {}).get("expected_tests_passed")
    if recorded is not None:
        candidates.append(
            {"value": recorded, "source": "recorded", "observed_at": spec.get("created_at")}
        )
    command_ref = None
    for rec in records:
        command = str(rec.get("command") or "")
        if "pytest" not in command:
            continue
        counted = _pytest_passed_count(str(rec.get("output") or ""))
        if counted <= 0:
            continue
        command_ref = command
        candidates.append(
            {"value": counted, "source": "command", "observed_at": time.time()}
        )
    winner = pick_authority(candidates)
    if winner is None:
        return None
    return {
        "passed": winner["value"],
        "authority": winner["source"],
        "command": command_ref,
        "candidates": candidates,
    }


def build_envelope(
    workspace: Union[str, Path],
    *,
    pipeline: Optional[Dict[str, Any]] = None,
    blocker: Optional[str] = None,
    aborted_reason: Optional[str] = None,
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Compose the delegation envelope. Every claim routes through classify_claim."""
    ws = Path(workspace)
    spec, spec_defect = _read_json(spec_path(ws))
    state, state_defect = _read_json(mission_state_path(ws))
    spec_ok = isinstance(spec, dict)
    spec = spec if spec_ok else {}
    state = state if isinstance(state, dict) else {}

    defects: List[str] = [d for d in (spec_defect, state_defect) if d]
    if spec_defect is None and not spec_ok:
        defects.append("delegation spec is not an object")

    if pipeline is None:
        stored, _ = _read_json(mission_dir(ws) / "pipeline_result.json")
        pipeline = stored if isinstance(stored, dict) else None

    mode = str(spec.get("verification") or "standard").lower()
    records, record_defects = _verifier_records(pipeline, mode)
    defects.extend(record_defects)

    verified: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    refutations: List[Dict[str, Any]] = []
    remaining: List[str] = []

    def file_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """classify_claim's answer decides the bucket. Nothing else does."""
        (verified if entry["class"] == "verified" else hypotheses).append(entry)
        return entry

    for rec in records:
        statement = rec.get("statement") or rec.get("obligation_id") or "(unnamed claim)"
        command = str(rec.get("command") or "").strip()
        art = (
            Artifact(
                kind="command",
                ref=command,
                exit_code=rec.get("exit_code"),
                output=rec.get("output"),
                observed_at=state.get("last_checkpoint"),
            )
            if command
            else None
        )
        outcome = verifier_outcome(rec)
        if outcome == "passed":
            file_entry(classify_claim(statement, art))
            continue
        if outcome == "failed":
            # The artifact supports the REFUTATION, not the statement. Here the
            # nonzero exit IS the evidence, so it is declared expected -- the one
            # place that is legitimate, and it is stated rather than defaulted.
            ref_art = (
                replace(art, expected_exit=art.exit_code)
                if isinstance(art, Artifact) and isinstance(art.exit_code, int)
                and not isinstance(art.exit_code, bool)
                else art
            )
            refutations.append(
                classify_claim(
                    f"verifier for {statement!r} did not pass "
                    f"(verdict={rec.get('verdict')}, exit_code={rec.get('exit_code')})",
                    ref_art,
                )
            )
            entry = file_entry(classify_claim(statement, None))
            entry["reason"] = (
                f"verifier FAILED: verdict={rec.get('verdict')} "
                f"exit_code={rec.get('exit_code')}"
            )
            continue
        entry = file_entry(classify_claim(statement, None))
        entry["reason"] = "verifier UNVERIFIABLE: no command settled it"
        remaining.append(statement)

    # Model prose. There is no wording that promotes this.
    # isinstance, not `pipeline or {}`: a truthy non-dict (a list, a bare
    # string) reached .get() and raised, throwing away the defect
    # _verifier_records had already recorded for exactly that input.
    answer = pipeline.get("answer") if isinstance(pipeline, dict) else None
    if answer:
        file_entry(classify_claim(str(answer), None))

    # Artifacts on disk. Presence is NOT verification (adversarial test B).
    artifacts: List[Dict[str, Any]] = []
    for declared in spec.get("output_contract") or []:
        path = Path(declared)
        if not path.is_absolute():
            path = ws / declared
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "verified": any(
                    str(path) in str(v.get("artifact", {}).get("ref") or "")
                    for v in verified
                ),
                "note": "presence on disk is not verification",
            }
        )

    cancel, _ = _read_json(cancel_path(ws))
    if aborted_reason is None and isinstance(cancel, dict):
        aborted_reason = cancel.get("reason")

    # BEFORE decide_verdict, not after. These used to be appended below the
    # verdict, which left decide_verdict's `if remaining: INCONCLUSIVE` rule
    # dead for them: an envelope could carry verdict ACCEPT and a
    # remaining_uncertainty naming work that had never run.
    for uid, payload in (state.get("units") or {}).items():
        if isinstance(payload, dict) and payload.get("status") in ("pending", "ready"):
            remaining.append(f"work unit {uid} still {payload.get('status')}")

    effective_blocker = blocker
    if effective_blocker is None and aborted_reason:
        effective_blocker = f"aborted: {aborted_reason}"
    if effective_blocker is None:
        # A BLOCKED verdict must always be able to say what blocked it.
        failed_required = [
            str(r.get("obligation_id") or r.get("statement") or "?")
            for r in records
            if r.get("required") and verifier_outcome(r) == "failed"
        ]
        if failed_required:
            effective_blocker = "required verifier(s) failed: " + ", ".join(
                failed_required
            )

    verdict = (
        "ABORTED"
        if aborted_reason
        else decide_verdict(
            verified_facts=verified,
            verifier_records=records,
            blocker=effective_blocker,
            defects=defects,
            remaining=remaining,
        )
    )

    fingerprint_now = _fingerprint(spec.get("output_contract") or [], ws)
    fingerprint_start = spec.get("fingerprint_at_start")
    if fingerprint_now is None or fingerprint_start is None:
        changed_files = None
    else:
        changed_files = {
            "fingerprint_at_start": fingerprint_start,
            "fingerprint_now": fingerprint_now,
            "changed": fingerprint_now != fingerprint_start,
            "paths": list(spec.get("output_contract") or []),
        }

    lock = MutationLock(ws)
    lock_record = lock.read()
    receipts = [
        str(p)
        for p in (spec_path(ws), mission_state_path(ws), envelope_path(ws))
        if p.is_file()
    ]

    truncated = any(
        (entry.get("artifact") or {}).get("output_truncated")
        for entry in list(verified) + list(refutations)
    )

    claim_source = pick_authority(
        [
            {
                "value": spec.get("objective"),
                "source": "receipt",
                "observed_at": spec.get("created_at"),
            },
            {
                "value": state.get("goal"),
                "source": "disk",
                "observed_at": state.get("last_checkpoint"),
            },
        ]
    )

    return {
        "envelope_version": ENVELOPE_VERSION,
        "mission_id": spec.get("mission_id") or state.get("id"),
        "workspace": str(ws),
        "state": state.get("phase"),
        "verdict": verdict,
        "claim": (claim_source or {}).get("value"),
        "verified_facts": verified,
        "hypotheses": hypotheses,
        "physical_measurements": None,
        "artifacts": artifacts,
        "changed_files": changed_files,
        "tests": _tests_field(records, spec),
        "negative_controls": None,
        "mutation_controls": None,
        "failed_attempts_that_change_interpretation": None,
        "refutations": refutations,
        "resource_usage": {
            "writer_pid": (lock_record or {}).get("pid"),
            "writer_alive": lock.holder_is_live(lock_record) if lock_record else False,
            "started_at": started_at or spec.get("created_at"),
            "elapsed_s": (
                time.time() - float(spec["created_at"])
                if spec.get("created_at")
                else None
            ),
            "allowed_resources": spec.get("allowed_resources") or None,
        },
        "blocker": effective_blocker,
        "remaining_uncertainty": remaining or None,
        "recommended_next_action": _next_action(verdict, effective_blocker, remaining),
        "receipt_paths": receipts,
        "defects": defects or None,
        "truncated": bool(truncated),
        "built_at": time.time(),
    }


def _next_action(
    verdict: str, blocker: Optional[str], remaining: Sequence[str]
) -> str:
    if verdict == "ACCEPT":
        return "consume the verified_facts; re-run the named commands to re-check"
    if verdict == "ABORTED":
        return "inspect the state file, then start a fresh mission if still needed"
    if verdict == "BLOCKED":
        return f"clear the blocker, then steer or re-run: {blocker or 'unnamed blocker'}"
    if remaining:
        return "steer the mission with what is missing, or re-run with --verify strict"
    return "treat nothing here as established; supply a deterministic verifier"


# ---------------------------------------------------------------------------
# The executor. Model access is behind ONE seam so tests stay offline.
# ---------------------------------------------------------------------------


_DELEGATE_MAX_TOKENS = int(os.environ.get("HAWKING_DELEGATE_MAX_TOKENS", "4096"))


def _delegate_timeout() -> float:
    """Seconds to wait on one model call.

    180 was hardcoded in three places, which is fine against a warm server and
    wrong against a local resident: the sealed 27B is invoked as a subprocess that
    RELOADS THE MODEL PER CALL, and a real mission timed out at 722 s having
    exceeded 180 on a single generate. Read at call time, not at import, so a
    harness can raise it for a slow resident without editing this file.
    """
    try:
        return max(1.0, float(os.environ.get("HAWKING_DELEGATE_TIMEOUT_S", "") or 180.0))
    except ValueError:
        return 180.0


class ModelRunaway(RuntimeError):
    """The model produced no content because it spent its budget reasoning."""


def default_caller(
    endpoint: str,
    model: Optional[str] = None,
    *,
    worker_mode: str = DEFAULT_WORKER_MODE,
) -> Callable[..., Any]:
    """Real model seam: an OpenAI-compatible endpoint via backends._post_json.

    The default is the Hawking daemon's ``:8011`` surface, which owns both the
    worker-policy enforcement and the delegated executor tree. An unavailable
    daemon produces a BLOCKED envelope naming the connection error, never a
    fake ACCEPT.

    Delegated calls always carry a safe, server-enforced worker mode.  Ordinary
    verifier-pipeline cognition is ``reason_only``; callers that genuinely
    need the daemon's narrow source/receipt search surface must opt into
    ``read_only_research`` in the durable delegation contract.
    """
    from .backends import (
        _post_json,
        completion_from_openai,
        make_structured_output_contract,
    )

    selected_worker_mode = _safe_worker_mode(worker_mode)
    if not _is_hawkingd_endpoint(endpoint) and not _unsafe_unmanaged_delegation_enabled():
        raise DelegationError(
            "delegated model calls require the local Hawkingd :8011 endpoint; "
            "HAWKING_ENDPOINT alone cannot select an unmanaged OpenAI-compatible server"
        )

    def caller(prompt: str, *, schema: Optional[dict] = None) -> Any:
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            # The daemon owns the enforcement.  It strips this control before
            # it reaches a provider, fences tool execution, and refuses a
            # restricted model switch before any resident lifecycle action.
            "hawking_worker_mode": selected_worker_mode,
            # THINKING OFF, and this is not a preference. Measured against the
            # resident Qwen3.8-27B on mlx_lm.server: with thinking on, the plan
            # call came back finish_reason="length" having spent the whole budget
            # in a SEPARATE `message.reasoning` field and never emitting
            # `content` at all -- which arrived here as "" and surfaced as the
            # misleading `PlanError: planner returned no obligations`. With
            # enable_thinking=False the same prompt returns clean content and
            # zero reasoning. The rest of HAWKING already does this
            # (backends.py MlxServerBackend.chat_template_args, and
            # config.enable_thinking defaults False); this seam bypassed the
            # backend via _post_json and so bypassed the setting too.
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": _DELEGATE_MAX_TOKENS,
        }
        if model:
            payload["model"] = model
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "delegation", "schema": schema},
            }

        def _complete(body, timeout=None):
            raw = _post_json(endpoint, body, float(timeout or _delegate_timeout()), "delegate")
            return completion_from_openai(raw, [])

        # STRUCTURED OUTPUT GOES THROUGH THE EXISTING CONTRACT, not a bare post.
        # mlx_lm.server has NO response_format and NO grammar (backends.py says so
        # in its capability table), so a raw post just sends a field the server
        # ignores and the model answers in prose. Measured: the resident 27B
        # returned a markdown essay with LaTeX, `json.loads` raised, and the
        # planner reported the misleading `PlanError: planner returned no
        # obligations`. make_structured_output_contract is the machinery this repo
        # already has for exactly that backend: strip the field, inject the schema
        # instruction, VALIDATE every reply, retry a bounded number of times with
        # the rejection reason appended, then raise StructuredOutputExhausted --
        # never a silent pass.
        if schema:
            contract = make_structured_output_contract(None, schema)
            if contract is not None:
                completion = contract.enforce(_complete, payload, _delegate_timeout())
                parsed = (completion.raw or {}).get("_structured") \
                    if isinstance(completion.raw, dict) else None
                if parsed is not None:
                    return parsed
                try:
                    return json.loads(completion.text or "")
                except json.JSONDecodeError:
                    return completion.text or ""

        data = _post_json(endpoint, payload, _delegate_timeout(), "delegate")
        completion = completion_from_openai(data, [])
        text = completion.text or ""

        # A runaway must be NAMED, not returned as an empty string. An empty
        # content with a populated reasoning field means the model spent its
        # budget thinking; silently passing "" downstream is what turned a
        # diagnosable runaway into an unrelated-looking planner error.
        if not text.strip():
            choices = (data or {}).get("choices") or [{}]
            msg = (choices[0] or {}).get("message") or {}
            reasoning = msg.get("reasoning") or ""
            finish = (choices[0] or {}).get("finish_reason")
            if reasoning or finish == "length":
                raise ModelRunaway(
                    f"model returned no content (finish_reason={finish!r}); "
                    f"{len(reasoning)} chars landed in `message.reasoning`. The "
                    f"budget was spent thinking. chat_template_kwargs."
                    f"enable_thinking=False is already set here, so this server "
                    f"is ignoring it -- check that it supports the field."
                )
        if schema:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text

    return caller


# ---------------------------------------------------------------- G076 roots
# Three roots, named, so a mission never has to hardcode an absolute path and a
# repo-relative verifier resolves the same way every time (S032 §19).
#
#   REPO_ROOT      the hawking checkout. Where `tools/`, `crates/` and
#                  `receipts/` live. READ-ONLY to a verifier.
#   MISSION_ROOT   <workspace>/.hawking/<mission> -- the mission's own state.
#   WORKSPACE_ROOT the delegation workspace, and the verifier's cwd.
#
# THE ISOLATION IS NOT WEAKENED, which S031 §21 forbids: the cwd is still the
# workspace, the repo is still not copied in, and nothing gains write access.
# What changes is that a verifier can now NAME the repo deterministically.
ROOT_ENV = ("HAWKING_REPO_ROOT", "HAWKING_MISSION_ROOT", "HAWKING_WORKSPACE_ROOT")

# Verbs that cannot write a file they are handed. DELIBERATELY SHORT, and the
# absences are the design: `python3`, `python`, `pytest`, `awk`, `sed`, `sort`
# and `find` are NOT here, because every one of them can write a path given to
# it -- `python3 -c "open(p,\'w\')"`, `awk \'{print > p}\'`, `sed \'s/a/b/w p\'`,
# `sort -o p`, `find -delete`. One of those was in this list until the negative
# control below wrote the protected file and failed.
READ_ONLY_VERBS = frozenset({
    "cat", "head", "tail", "less", "more", "wc", "grep", "egrep", "fgrep", "rg",
    "cut", "tr", "diff", "cmp", "md5", "shasum", "sha256sum", "stat", "ls",
    "file", "jq", "echo", "true", "false", "basename", "dirname", "realpath",
    "readlink", "test", "[",
})

# Anything here can MUTATE regardless of verb, so a command carrying one may not
# name a protected path at all.
WRITE_TOKENS = (
    # ">" and ">>" are NOT here: _guard_verdict checks the redirection TARGET,
    # which is the difference between reading a receipt into a file and writing
    # over one.
    "|&", "tee", "rm ", "mv ", "cp ", "install ", "truncate",
    "dd ", "chmod", "chown", "chgrp", "ln ", "mkdir", "rmdir", "touch",
    # NOT a bare "-i " or "-o ": those match `grep -i` and `grep -o` and would
    # refuse ordinary reads. Name the in-place writers instead.
    "sed -i", "perl -i", "ruby -i", "--in-place", "-delete", "-exec",
    "sort -o", "shred", "unlink",
    "git add", "git commit", "git checkout", "git rm", "git mv", "git clean",
    "git reset", "git stash", "git push",
)

def _names_protected(command: str, guarded: Sequence[str]) -> Optional[str]:
    """The protected path this text names, or None."""
    for path in guarded:
        if str(path) in command:
            return str(path)
    return None


_SEPARATORS = ("|", "&&", "||", ";", "&")


def _segments(command: str) -> List[str]:
    """Split a command into pipeline/list segments, keeping each one's text.

    Crude on purpose: this is not a shell parser and does not claim to be. It
    exists so the guard can ask its question PER SEGMENT rather than over the
    whole line, which is the difference between refusing
    `cat receipts/x.json | python3 -c ...` and allowing it.
    """
    parts, buf, i, quote = [], [], 0, ""
    while i < len(command):
        ch = command[i]
        # QUOTE AWARE. The first version was not, and it split inside a quoted
        # argument: a real mission proposed
        # `python3 -c "import os; p = ...; path = os.path.join(...)"` and the guard
        # reported "verb 'path' is not in the read-only set" -- a verb scraped out
        # of Python source. The verdict was still correct (python3 is not provably
        # read-only, so it failed closed), but a reason nobody can act on is a bug
        # in its own right.
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if command.startswith("&&", i) or command.startswith("||", i):
            parts.append("".join(buf)); buf = []; i += 2
            continue
        if ch in ("|", ";", "&"):
            parts.append("".join(buf)); buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _segment_is_read_only(segment: str) -> Tuple[bool, str]:
    """Can this ONE segment write the path it names? FAILS CLOSED."""
    import shlex

    # Redirection is NOT checked here -- _guard_verdict checks the TARGET, so
    # `cat receipts/x > copy.json` is a read and `cat y > receipts/x` is not.
    low = segment.lower()
    for tok in WRITE_TOKENS:
        if tok in low:
            return False, f"segment carries write operator {tok.strip()!r}"
    try:
        parts = shlex.split(segment.split(">")[0])
    except ValueError as exc:
        return False, f"segment does not parse ({exc}); refused rather than guessed"
    if not parts:
        return False, "empty segment"
    verb = pathlib.PurePath(parts[0]).name
    if verb not in READ_ONLY_VERBS:
        return False, f"verb {verb!r} is not in the read-only set"
    return True, ""


def _redirect_targets(command: str) -> List[str]:
    """Every path a `>`/`>>` redirection writes to."""
    import re as _re
    return [m.group(2) for m in _re.finditer(r"(>>|>\|?)\s*([^\s;|&]+)", command)]


def _background_operator(command: str) -> bool:
    """Whether unquoted shell syntax asks the verifier runner to detach work.

    ``&&`` and ``|&`` are foreground control operators. A remaining unquoted
    ampersand backgrounds a shell job, which would make its lifetime impossible
    to bind to the foreground verifier process group after the shell returns.
    Such commands add no value to a bounded verifier and are refused.
    """

    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            index += 1
            continue
        if char != "&":
            index += 1
            continue
        previous = command[index - 1] if index else ""
        following = command[index + 1] if index + 1 < len(command) else ""
        if previous == "|":
            index += 1
            continue
        if following == "&":
            index += 2
            continue
        return True
    return False


def _guard_verdict(command: str, guarded: Sequence[str]) -> Optional[str]:
    """None to allow, or the reason to refuse. THE RULE IS ONE SENTENCE:

        a protected path may not be WRITTEN. Reading it is what it is for.

    Three ways a command can write one, and each is checked where it lives:

    1. REDIRECTION TARGET -- `> receipts/x` or `>> receipts/x`. Checked over the
       whole line, because `>` binds to the line and not to a pipeline segment.
    2. A WRITING VERB HANDED THE PATH -- `tee receipts/x`, `rm receipts/x`,
       `sed -i ... receipts/x`. Checked per segment.
    3. A VERB NOBODY VOUCHED FOR HANDED THE PATH -- `python3 receipts/x`,
       `awk ... receipts/x`. Refused because it CANNOT BE SHOWN not to write, not
       because it will. Fails closed.

    WHAT IS DELIBERATELY ALLOWED, and an earlier version of this guard got it
    wrong in both directions: `cat receipts/x > copy.json` and
    `cat receipts/x | tee copy.json` both READ the protected file and write
    somewhere else. The protected path is unharmed. Refusing one while allowing
    the other -- which is what checking `is there a redirection in a segment that
    names a protected path` produced -- was incoherent, and a mutation test that
    deleted the redundant check and changed NOTHING is what exposed it.

    This is not a shell parser. It cannot see through a variable, a subshell that
    rebuilds a path from pieces, or an alias, and it says so rather than implying
    a completeness it does not have.
    """
    if _background_operator(command):
        return "background verifier jobs are not permitted"

    if _names_protected(command, guarded) is None:
        return None

    for tgt in _redirect_targets(command):
        hit = _names_protected(tgt, guarded)
        if hit is not None:
            return f"redirection would write protected path {hit!r}"

    for seg in _segments(command):
        named = _names_protected(seg, guarded)
        if named is None:
            continue
        ok, why = _segment_is_read_only(seg)
        if not ok:
            return (f"segment naming protected path {named!r} is not provably "
                    f"read-only ({why})")
    return None


def shell_runner(
    workspace: Union[str, Path],
    protected_paths: Sequence[str] = (),
    *,
    repo_root: Optional[Union[str, Path]] = None,
    verifier_processes: _VerifierProcessRegistry | None = None,
) -> Callable[[str], Tuple[int, str]]:
    """Build a host-only local verifier runner, outside durable delegation.

    This is deliberately not a capability of a normal restricted worker.
    ``execute_mission`` ignores even an injected runner, because its command
    text is model-proposed.  Retaining this function supports separately
    reviewed host-owned verification and focused tests without silently making
    a generic shell available to ``reason_only`` or ``read_only_research``
    workers.

    TWO DEFECTS THIS REPAIRS, both recorded under G076:

    1. The workspace does not contain the repo, so a repo-relative verifier path
       was unresolvable. The three roots are now exported into the command's
       environment, so `$HAWKING_REPO_ROOT/tools/...` resolves deterministically
       from anywhere without the mission hardcoding an absolute path.

    2. The protected-path guard matched a SUBSTRING OF THE COMMAND TEXT, so it
       refused READS as well as WRITES. A verifier that reads a receipt is
       exactly what a receipt is for. Reads that provably cannot write are now
       allowed; everything else is still refused, and the refusal says which
       rule fired.

    The cwd is still the workspace and the repo is still not copied in, so the
    isolation S031 §21 protects is untouched -- what changed is nameability, not
    reach.
    """
    ws = Path(workspace).resolve()
    guarded = [p for p in protected_paths if str(p).strip()]
    try:
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
    except Exception:
        root = None

    def runner(command: str) -> Tuple[int, str]:
        refusal = _guard_verdict(command, guarded)
        if refusal is not None:
            return 126, f"REFUSED: {refusal}"
        env = dict(os.environ)
        if root is not None:
            env["HAWKING_REPO_ROOT"] = str(root)
        env["HAWKING_MISSION_ROOT"] = str(mission_dir(ws))
        env["HAWKING_WORKSPACE_ROOT"] = str(ws)
        popen_args: Dict[str, Any] = {
            "shell": True,
            "cwd": str(ws),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "env": env,
        }
        if os.name == "posix" and verifier_processes is None:
            # Direct (non-daemon) callers still get an isolated command group
            # for timeout cleanup. A supervised executor instead keeps its
            # verifier in its own group so daemon-loss cleanup reaches normal
            # descendants even after the shell itself has exited.
            popen_args["start_new_session"] = True
        if verifier_processes is not None:
            proc = verifier_processes.spawn(
                lambda: subprocess.Popen(command, **popen_args)
            )
        else:
            proc = subprocess.Popen(command, **popen_args)
        try:
            stdout, stderr = proc.communicate(timeout=COMMAND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if verifier_processes is not None:
                # This never returns in the daemon-owned POSIX path. Do not
                # downgrade a timed-out verifier to shell-only cleanup: its
                # descendants share the executor group and must end too.
                verifier_processes.terminate_supervised_executor_group()
            _terminate_verifier_process(proc)
            try:
                proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                # The direct runner created a private session. Its shell can
                # accept TERM while a same-group descendant ignores it, so
                # the escalation must still target the known group.
                _terminate_verifier_process(
                    proc, force=True, include_exited_leader=True
                )
                proc.communicate()
            raise
        finally:
            if verifier_processes is not None:
                verifier_processes.unregister(proc)
        return proc.returncode, (stdout or "") + (stderr or "")

    return runner


def _restricted_worker_command_runner(command: str) -> Tuple[int, str]:
    """Fail closed when a restricted worker proposes a local shell command.

    The verifier pipeline may ask the model for a discriminator, but a worker
    mode is not authority to execute that discriminator.  ``reason_only`` can
    reason from supplied evidence, and ``read_only_research`` is limited to
    Hawkingd's server-side tools; neither gets a generic local shell through
    this secondary path.  A host-owned verifier is a separate control-plane
    operation whose already-reviewed output may be supplied as evidence; it
    cannot be injected back into a durable restricted worker as a shell door.
    """

    return 126, (
        "REFUSED: restricted HAWKING workers do not execute model-proposed shell "
        "commands; use supplied evidence, Hawkingd's allowlisted research "
        "tools, or separately reviewed host evidence"
    )


def _goal_with_steers(objective: str, ws: Path) -> str:
    """Absorb pending steers into the goal handed to the pipeline."""
    queue = None
    try:
        queue = _steering_queue(ws)
        pending = queue.pending()
    except Exception:
        return objective
    if not pending:
        return objective
    lines = [f"[steer {e.kind}] {e.text}" for e in pending]
    queue.apply_pending()
    return objective + "\n\nOPERATOR STEERS (apply to remaining work):\n" + "\n".join(lines)


def _goal_with_constraints(objective: str, constraints: Sequence[str]) -> str:
    """Put the operator's constraints in front of the worker.

    `--constraint` was parsed, written into the delegation spec, and read by
    nobody: build_spec stored spec["constraints"] and execute_mission read
    objective, endpoint and protected_paths only. So every constraint an
    operator set was discarded between the command line and the model -- the
    worst shape of this bug, because the operator believes a limit is in force.
    """
    rows = [str(c).strip() for c in (constraints or ()) if str(c).strip()]
    if not rows:
        return objective
    return objective + "\n\nOPERATOR CONSTRAINTS (these bound the work):\n" + \
        "\n".join(f"- {row}" for row in rows)


def execute_mission(
    workspace: Union[str, Path],
    *,
    caller: Optional[Callable[..., Any]] = None,
    run_command: Optional[Callable[[str], Tuple[int, str]]] = None,
    verifier_processes: _VerifierProcessRegistry | None = None,
) -> Dict[str, Any]:
    """Do the delegated work and write the envelope. Holds the writer lock.

    ``run_command`` remains in the signature for compatibility with old host
    callers, but every durable worker mode is restricted and ignores it.  A
    model proposal must never become local shell/process-launch capability by
    virtue of an embedding caller supplying ``shell_runner``.  Host-owned
    verification belongs outside this worker pipeline and its reviewed result
    can be supplied as evidence on a later bounded request.
    """
    from .verifier_pipeline import run_pipeline

    ws = Path(workspace)
    spec, defect = _read_json(spec_path(ws))
    lock = MutationLock(ws)
    mission_id = (spec or {}).get("mission_id") if isinstance(spec, dict) else None
    if not lock.acquire(str(mission_id or ws.name)):
        held = lock.read() or {}
        raise DelegationBusy(
            f"{ws} is held by pid={held.get('pid')} unit={held.get('unit_id')}"
        )
    started = time.time()
    pipeline: Optional[Dict[str, Any]] = None
    blocker: Optional[str] = None
    try:
        if not isinstance(spec, dict):
            blocker = defect or "delegation spec is not an object"
        elif cancel_path(ws).is_file():
            blocker = "cancel requested before execution started"
        else:
            goal = _goal_with_constraints(
                str(spec.get("objective") or ""), spec.get("constraints") or [])
            goal = _goal_with_steers(goal, ws)
            worker_mode = _safe_worker_mode(spec.get("worker_mode"))
            caller = caller or default_caller(
                str(spec.get("endpoint") or DEFAULT_ENDPOINT),
                worker_mode=worker_mode,
            )
            # Do not let a host embedding convert the restricted model's
            # proposed verifier text into generic local execution.  The
            # compatibility parameter is intentionally ignored rather than
            # called or rejected after the model has begun work, so old call
            # sites fail closed into the ordinary evidence-only outcome.
            run_command = _restricted_worker_command_runner
            try:
                raw = run_pipeline(goal, caller, run_command)
                pipeline = {
                    "goal": raw.get("goal"),
                    "answer": raw.get("answer"),
                    "verdicts": [vars(v) for v in raw.get("verdicts") or []],
                    "obligations": [vars(o) for o in raw.get("obligations") or []],
                }
                atomic_write_json(mission_dir(ws) / "pipeline_result.json", pipeline)
            except Exception as exc:
                blocker = f"{type(exc).__name__}: {exc}"
        envelope = build_envelope(
            ws, pipeline=pipeline, blocker=blocker, started_at=started
        )
        phase = "completed" if envelope["verdict"] == "ACCEPT" else "failed"
        # state.json and the envelope must not disagree about the phase.
        state, _ = _read_json(mission_state_path(ws))
        if isinstance(state, dict):
            state["phase"] = phase
            state["last_checkpoint"] = time.time()
            atomic_write_json(mission_state_path(ws), state)
            envelope["state"] = phase
        atomic_write_json(envelope_path(ws), envelope)
        return envelope
    finally:
        lock.release(str(mission_id or ws.name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(prog: str = "hawking") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="HAWKING delegation surface"
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_run = sub.add_parser("run", help="start a delegated mission (returns at once)")
    p_run.add_argument("--goal", required=True, help="objective for the mission")
    p_run.add_argument("--verify", choices=list(VERIFY_MODES), default="standard")
    p_run.add_argument("--budget", default=None, help="JSON object, e.g. '{\"minutes\":30}'")
    p_run.add_argument("--resources", action="append", default=[])
    p_run.add_argument("--protect", action="append", default=[])
    p_run.add_argument("--constraint", action="append", default=[])
    p_run.add_argument("--expect", action="append", default=[], help="output contract path")
    p_run.add_argument(
        "--worker-mode",
        choices=list(SAFE_WORKER_MODES),
        default=DEFAULT_WORKER_MODE,
        help=("daemon-enforced authority for model calls; ordinary delegation "
              "uses reason_only"),
    )
    p_run.add_argument("--root", default=None, help="delegations root (default: cwd)")
    p_run.add_argument("--no-spawn", action="store_true", help="write the spec, run nothing")
    p_run.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="durable snapshot")
    p_status.add_argument("mission")
    p_status.add_argument("--root", default=None)
    p_status.add_argument("--json", action="store_true")

    p_steer = sub.add_parser("steer", help="queue guidance for future work")
    p_steer.add_argument("mission")
    p_steer.add_argument("text")
    p_steer.add_argument("--kind", choices=list(STEER_KINDS), default="knowledge")
    p_steer.add_argument("--root", default=None)
    p_steer.add_argument("--json", action="store_true")

    p_result = sub.add_parser("result", help="the evidence envelope")
    p_result.add_argument("mission")
    p_result.add_argument("--root", default=None)
    p_result.add_argument("--json", action="store_true")

    p_abort = sub.add_parser("abort", help="stop a delegation")
    p_abort.add_argument("mission")
    p_abort.add_argument("--reason", default="aborted by operator")
    p_abort.add_argument("--root", default=None)
    p_abort.add_argument("--json", action="store_true")
    return parser


def _render_result(env: Dict[str, Any]) -> str:
    lines = [
        f"mission {env.get('mission_id')}  state={env.get('state')}  "
        f"VERDICT={env.get('verdict')}",
    ]
    facts = env.get("verified_facts") or []
    lines.append(f"VERIFIED ({len(facts)}) — each backed by a deterministic artifact")
    for item in facts:
        art = item.get("artifact") or {}
        lines.append(f"  [VERIFIED] {item.get('claim')}")
        lines.append(f"             {art.get('kind')}: {art.get('ref')} "
                     f"(exit={art.get('exit_code')})")
    if not facts:
        lines.append("  (none)")
    hyps = env.get("hypotheses") or []
    lines.append(f"HYPOTHESIS ({len(hyps)}) — nothing here is established")
    for item in hyps:
        claim, _ = _clip(item.get("claim"), 300)
        lines.append(f"  [HYPOTHESIS] {claim}")
        if item.get("reason"):
            lines.append(f"               why not verified: {item['reason']}")
    for key in ("blocker", "remaining_uncertainty", "defects", "tests"):
        if env.get(key):
            lines.append(f"{key}: {json.dumps(env[key], default=str)[:600]}")
    if env.get("truncated"):
        lines.append("truncated: some artifact output was clipped")
    lines.append(f"next: {env.get('recommended_next_action')}")
    return "\n".join(lines)


def cli_main(argv: Sequence[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv))
    try:
        if args.verb == "run":
            budget = json.loads(args.budget) if args.budget else None
            out = run(
                args.goal,
                root=args.root,
                spawn=not args.no_spawn,
                verification=args.verify,
                budget=budget,
                allowed_resources=args.resources,
                protected_paths=args.protect,
                constraints=args.constraint,
                output_contract=args.expect,
                worker_mode=args.worker_mode,
            )
            print(json.dumps(out, indent=2) if args.json else out["mission_id"])
            return 0
        if args.verb == "status":
            out = status(args.mission, args.root)
            print(json.dumps(out, indent=2) if args.json else _render_status(out))
            return 0
        if args.verb == "steer":
            out = steer(args.mission, args.text, args.kind, args.root)
            print(json.dumps(out, indent=2) if args.json else f"steer {out['steer_id']} queued ({out['kind']})")
            return 0
        if args.verb == "result":
            out = result(args.mission, args.root)
            print(json.dumps(out, indent=2, default=str) if args.json else _render_result(out))
            return 0 if out.get("verdict") == "ACCEPT" else 2
        if args.verb == "abort":
            out = abort(args.mission, args.reason, args.root)
            print(json.dumps(out, indent=2) if args.json else
                  f"aborted {out['mission_id']} (lock_free={out['lock_free']})")
            return 0
    except DelegationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    parser.error(f"unknown verb {args.verb!r}")
    return 3


def _render_status(snap: Dict[str, Any]) -> str:
    writer = snap.get("writer") or {}
    lines = [
        f"mission {snap.get('mission_id')}  phase={snap.get('phase')}",
        f"objective: {snap.get('objective')}",
        f"writer: pid={writer.get('pid')} alive={writer.get('alive')}",
        f"units: {snap.get('units_by_status') or {}}",
        f"steers pending: {snap.get('steers_pending')}",
        f"envelope: {snap.get('envelope_present')} verdict={snap.get('verdict')}",
    ]
    if snap.get("cancel_requested"):
        lines.append(f"cancel requested: {snap.get('cancel_reason')}")
    if snap.get("defects"):
        lines.append(f"defects: {snap['defects']}")
    return "\n".join(lines)


def exec_main(argv: Sequence[str]) -> int:
    """Entry point for the spawned worker (``hawking __delegate_exec <workspace>``)."""
    if not argv:
        print("usage: hawking __delegate_exec <workspace>", file=sys.stderr)
        return 2
    verifier_processes: _VerifierProcessRegistry | None = None
    watchdog_stop: threading.Event | None = None
    supervised_executor = False
    cleanup_supervised_executor_group = False
    previous_sigterm_handler: Any | None = None
    try:
        witness = _consume_daemon_executor_capability(argv[0])
        if witness is not None:
            verifier_processes = _VerifierProcessRegistry()
            watchdog_stop = _start_supervisor_watchdog(
                witness,
                on_supervisor_lost=verifier_processes.terminate_all,
            )
            # _start_supervisor_watchdog proved this child owns a dedicated
            # executor group.  Its final cleanup happens here, before this
            # process can exit and leave a same-group stale child for a
            # parent-side reaper to identify after PID reuse is possible.
            supervised_executor = True
            # SIGTERM otherwise terminates Python before its ``finally``.
            # Install this only after the witness and dedicated-PGID proof so
            # abort/daemon-close cleanup uses the executor's own identity.
            previous_sigterm_handler = _install_supervised_executor_sigterm_cleanup()
    except DelegationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    try:
        envelope = execute_mission(argv[0], verifier_processes=verifier_processes)
        print(json.dumps({"verdict": envelope.get("verdict")}))
        sys.stdout.flush()
        return 0 if envelope.get("verdict") == "ACCEPT" else 2
    finally:
        try:
            if verifier_processes is not None:
                # Latch the boundary before best-effort leader cleanup.  A
                # verifier that has ever launched can leave an ordinary
                # same-group descendant after its shell unregisters, so an
                # exceptional cleanup path must not erase the self-owned
                # executor-group containment obligation.
                cleanup_supervised_executor_group = (
                    verifier_processes.has_launched_verifier()
                )
                # The normal restricted path never launches a model-proposed
                # verifier, but close the registry before disarming the
                # watchdog as a defense against future host-injected
                # integrations.
                verifier_processes.terminate_all()
        finally:
            try:
                if watchdog_stop is not None:
                    watchdog_stop.set()
                if supervised_executor and cleanup_supervised_executor_group:
                    # Successful POSIX cleanup kills this executor as well as
                    # every ordinary descendant in its dedicated group.
                    # Therefore it cannot return in production; its use of a
                    # self-owned group avoids a post-exit/recycled-PID signal
                    # ambiguity.
                    _terminate_owned_supervised_executor_group()
            finally:
                if previous_sigterm_handler is not None:
                    # The normal no-verifier path returns with ordinary exit
                    # semantics; restore the embedding process's handler
                    # rather than leaking daemon-specific policy into a
                    # direct unit-test caller.
                    signal.signal(signal.SIGTERM, previous_sigterm_handler)
