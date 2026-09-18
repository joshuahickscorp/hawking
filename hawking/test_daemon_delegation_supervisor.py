"""Daemon ownership tests for durable HAWKING delegation workers."""

from __future__ import annotations

import os
from pathlib import Path
import hashlib
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from hawking import delegate
from hawking.serve import DaemonDelegationSupervisor


class _Process:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False

    def poll(self):  # noqa: ANN201
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float):  # noqa: ANN201, ARG002
        return self.returncode


class _TrackedVerifier:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):  # noqa: ANN201
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = 0


class _StartResponse:
    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args):  # noqa: ANN002, ANN204
        return False

    def read(self) -> bytes:
        return b'{"pid": 4242}'


def _workspace(root: Path) -> Path:
    workspace = root / ".hawking" / "delegations" / "mission-a"
    spec = workspace / ".hawking" / "mission" / "delegation_spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}")
    return workspace


def test_daemon_supervisor_accepts_only_prepared_local_delegations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    created: list[_Process] = []
    signalled_groups: list[tuple[int, int]] = []

    spawned_env: list[dict[str, str]] = []
    spawned_options: list[dict[str, object]] = []

    def spawn(*_args, **kwargs):  # noqa: ANN001, ANN202
        spawned_env.append(kwargs["env"])
        spawned_options.append(kwargs)
        process = _Process()
        created.append(process)
        return process

    def signal_group(pid: int, signum: int) -> None:
        signalled_groups.append((pid, signum))
        created[0].returncode = 0

    monkeypatch.setattr(
        "hawking.serve.process_start_token", lambda pid: f"start-token-{pid}"
    )
    monkeypatch.setattr("hawking.serve.subprocess.Popen", spawn)
    monkeypatch.setattr("hawking.serve.os.killpg", signal_group)
    supervisor = DaemonDelegationSupervisor(tmp_path)
    started = supervisor.start(str(workspace))
    duplicate = supervisor.start(str(workspace))

    assert started == {
        "started": True,
        "pid": 4242,
        "workspace": str(workspace.resolve()),
        "owner": "hawkingd",
    }
    assert duplicate["started"] is False
    assert len(created) == 1
    assert spawned_options[0]["start_new_session"] is True
    token = spawned_env[0][delegate.DAEMON_EXEC_CAPABILITY_ENV]
    capability_path = delegate._daemon_executor_capability_path(workspace)
    payload = delegate._read_json(capability_path)[0]
    assert isinstance(payload, dict)
    assert payload["token_sha256"] == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert payload["supervisor_pid"] == os.getpid()
    assert payload["supervisor_start_token"] == f"start-token-{os.getpid()}"
    supervisor.close()
    # The witnessed executor owns same-group cleanup on SIGTERM.  Once its
    # direct leader exits, Hawkingd must not signal that potentially recycled
    # numeric PGID later.
    assert signalled_groups == [(4242, signal.SIGTERM)]
    assert created[0].returncode == 0
    assert capability_path.exists() is False


def test_daemon_close_escalates_only_a_still_live_executor_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live leader still proves its executor-only group identity."""

    workspace = _workspace(tmp_path)
    process = _Process()
    signals: list[tuple[int, int]] = []

    def signal_group(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == signal.SIGKILL:
            process.returncode = 0

    monkeypatch.setattr("hawking.serve.os.killpg", signal_group)
    supervisor = DaemonDelegationSupervisor(tmp_path)
    supervisor._children[str(workspace.resolve())] = process

    supervisor.close()

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]


def test_daemon_supervisor_fails_closed_without_a_process_witness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    supervisor = DaemonDelegationSupervisor(tmp_path)
    monkeypatch.setattr("hawking.serve.process_start_token", lambda _pid: None)
    monkeypatch.setattr(
        "hawking.serve.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("child must not start without a witness"),
    )

    with pytest.raises(RuntimeError, match="process-incarnation witness"):
        supervisor.start(str(workspace))
    assert delegate._daemon_executor_capability_path(workspace).exists() is False


def test_supervisor_witness_rejects_a_recycled_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness = delegate.DaemonSupervisorWitness(pid=4242, start_token="old-start")
    monkeypatch.setattr(delegate, "process_start_token", lambda _pid: "new-start")
    assert delegate._supervisor_witness_is_live(witness) is False


@pytest.mark.skipif(os.name != "posix", reason="Hawking native worker is POSIX/macOS")
def test_supervised_watchdog_refuses_a_shared_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness = delegate.DaemonSupervisorWitness(pid=4242, start_token="daemon-start")
    monkeypatch.setattr(delegate, "process_start_token", lambda _pid: "daemon-start")
    monkeypatch.setattr(delegate.os, "getpgrp", lambda: os.getpid() + 1)
    with pytest.raises(delegate.DelegationError, match="dedicated process group"):
        delegate._start_supervisor_watchdog(witness)


def test_daemon_supervisor_refuses_workspace_outside_delegation_root(tmp_path: Path) -> None:
    outsider = tmp_path / "outside"
    outsider.mkdir()
    supervisor = DaemonDelegationSupervisor(tmp_path)
    with pytest.raises(PermissionError):
        supervisor.start(str(outsider))


def test_live_daemon_endpoint_never_falls_back_to_detached_client_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "mission"
    delegate.run(
        "read-only review",
        workspace=workspace,
        spawn=False,
        endpoint="http://127.0.0.1:8011/v1/chat/completions",
    )

    def unavailable(*_args, **_kwargs):  # noqa: ANN001, ANN202
        raise urllib.error.URLError("old daemon image")

    monkeypatch.setattr("hawking.delegate.urllib.request.urlopen", unavailable)
    monkeypatch.setattr(
        "hawking.delegate.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("must not create an unowned worker"),
    )
    assert delegate._spawn_executor(workspace) is None


def test_no_env_default_delegation_uses_daemon_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary CLI/API default must never take the detached-worker path."""
    workspace = tmp_path / "mission"
    seen: list[tuple[str, object]] = []
    monkeypatch.delenv("HAWKING_ENDPOINT", raising=False)

    def start(request, timeout):  # noqa: ANN001, ANN202
        seen.append((request.full_url, timeout))
        return _StartResponse()

    monkeypatch.setattr("hawking.delegate.urllib.request.urlopen", start)
    monkeypatch.setattr(
        "hawking.delegate.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "default delegation must be owned by hawkingd, not a detached worker"
        ),
    )

    created = delegate.run("read-only review", workspace=workspace, spawn=True)
    spec = delegate._read_json(delegate.spec_path(workspace))[0]

    assert isinstance(spec, dict)
    assert spec["endpoint"] == delegate.DEFAULT_ENDPOINT
    assert created["writer_pid"] == 4242
    assert seen == [("http://127.0.0.1:8011/hawkingd/delegations/start", 3.0)]


def test_env_override_to_openwebui_is_queued_and_never_detaches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambient legacy :8080 endpoint cannot bypass Hawkingd ownership."""

    workspace = tmp_path / "mission"
    monkeypatch.setenv("HAWKING_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
    monkeypatch.delenv("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", raising=False)
    monkeypatch.setattr(
        "hawking.delegate.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("must not create an unmanaged worker"),
    )

    created = delegate.run("read-only review", workspace=workspace, spawn=True)
    spec = delegate._read_json(delegate.spec_path(workspace))[0]

    assert isinstance(spec, dict)
    assert spec["endpoint"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert created["writer_pid"] is None
    assert created["spawned"] is False


def test_hidden_executor_verb_refuses_direct_workspace_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    monkeypatch.delenv(delegate.DAEMON_EXEC_CAPABILITY_ENV, raising=False)
    monkeypatch.delenv("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", raising=False)
    monkeypatch.setattr(
        delegate,
        "execute_mission",
        lambda *_args, **_kwargs: pytest.fail("direct hidden verb must not execute a mission"),
    )

    assert delegate.exec_main([str(workspace)]) == 3


def test_daemon_executor_capability_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    token = delegate.issue_daemon_executor_capability(workspace)
    monkeypatch.setenv(delegate.DAEMON_EXEC_CAPABILITY_ENV, token)
    monkeypatch.delenv("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", raising=False)
    monkeypatch.setattr(
        delegate,
        "execute_mission",
        lambda *_args, **_kwargs: {"verdict": "BLOCKED"},
    )

    assert delegate.exec_main([str(workspace)]) == 2
    assert delegate._daemon_executor_capability_path(workspace).exists() is False
    assert delegate.exec_main([str(workspace)]) == 3


def test_daemon_executor_starts_and_stops_its_parent_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    witness = delegate.DaemonSupervisorWitness(pid=4242, start_token="daemon-start")
    token = delegate.issue_daemon_executor_capability(workspace, supervisor=witness)
    stop = threading.Event()
    observed: list[delegate.DaemonSupervisorWitness | None] = []
    lost_callbacks: list[object] = []
    executor_registries: list[object] = []
    monkeypatch.setenv(delegate.DAEMON_EXEC_CAPABILITY_ENV, token)

    def start_watchdog(supplied, *, on_supervisor_lost):  # noqa: ANN001, ANN202
        observed.append(supplied)
        lost_callbacks.append(on_supervisor_lost)
        return stop

    monkeypatch.setattr(
        delegate,
        "_start_supervisor_watchdog",
        start_watchdog,
    )
    monkeypatch.setattr(
        delegate,
        "execute_mission",
        lambda *_args, **kwargs: (
            executor_registries.append(kwargs.get("verifier_processes"))
            or {"verdict": "BLOCKED"}
        ),
    )
    group_cleanup: list[bool] = []
    monkeypatch.setattr(
        delegate,
        "_terminate_owned_supervised_executor_group",
        lambda: group_cleanup.append(True),
    )

    assert delegate.exec_main([str(workspace)]) == 2
    assert observed == [witness]
    assert len(lost_callbacks) == 1 and callable(lost_callbacks[0])
    assert len(executor_registries) == 1
    assert isinstance(executor_registries[0], delegate._VerifierProcessRegistry)
    assert stop.is_set() is True
    # The ordinary restricted worker never starts a host-injected verifier, so
    # it retains its ordinary verdict/exception exit semantics.
    assert group_cleanup == []


def test_daemon_executor_latches_group_cleanup_before_registry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing leader cleanup cannot bypass self-owned containment."""

    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    witness = delegate.DaemonSupervisorWitness(pid=4242, start_token="daemon-start")
    token = delegate.issue_daemon_executor_capability(workspace, supervisor=witness)
    stop = threading.Event()
    group_cleanup: list[bool] = []
    restored_handlers: list[tuple[object, object]] = []
    sentinel_handler = object()
    monkeypatch.setenv(delegate.DAEMON_EXEC_CAPABILITY_ENV, token)
    monkeypatch.setattr(
        delegate,
        "_start_supervisor_watchdog",
        lambda *_args, **_kwargs: stop,
    )
    monkeypatch.setattr(
        delegate,
        "_install_supervised_executor_sigterm_cleanup",
        lambda: sentinel_handler,
    )
    monkeypatch.setattr(
        delegate.signal,
        "signal",
        lambda signum, handler: restored_handlers.append((signum, handler)),
    )
    monkeypatch.setattr(
        delegate,
        "execute_mission",
        lambda *_args, **_kwargs: {"verdict": "BLOCKED"},
    )
    monkeypatch.setattr(
        delegate._VerifierProcessRegistry,
        "has_launched_verifier",
        lambda _self: True,
    )

    def fail_registry_cleanup(_self) -> None:  # noqa: ANN001
        raise RuntimeError("simulated verifier registry failure")

    monkeypatch.setattr(
        delegate._VerifierProcessRegistry,
        "terminate_all",
        fail_registry_cleanup,
    )
    monkeypatch.setattr(
        delegate,
        "_terminate_owned_supervised_executor_group",
        lambda: group_cleanup.append(True),
    )

    with pytest.raises(RuntimeError, match="simulated verifier registry failure"):
        delegate.exec_main([str(workspace)])
    assert stop.is_set() is True
    assert group_cleanup == [True]
    assert restored_handlers == [(signal.SIGTERM, sentinel_handler)]


def test_daemon_loss_registry_ends_a_tracked_verifier_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _TrackedVerifier()
    registry = delegate._VerifierProcessRegistry()
    assert registry.spawn(lambda: process) is process
    groups: list[tuple[int, int]] = []

    def kill_group(pid: int, signum: int) -> None:
        groups.append((pid, signum))
        process.returncode = 0

    monkeypatch.setattr(delegate.os, "killpg", kill_group)
    registry.terminate_all()

    # A supervised shell shares the executor group, so its PID must not be
    # treated as a group ID. The watchdog owns the later executor-group kill.
    assert groups == []
    assert process.terminated is True

    with pytest.raises(delegate.DelegationError, match="supervisor was lost"):
        registry.spawn(lambda: pytest.fail("closed registry must not spawn a verifier"))


@pytest.mark.skipif(os.name != "posix", reason="Hawking native worker is POSIX/macOS")
def test_supervised_verifier_timeout_targets_the_executor_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = delegate._VerifierProcessRegistry()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(delegate.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setattr(
        delegate.os,
        "killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )

    with pytest.raises(delegate.DelegationError, match="verifier timeout"):
        registry.terminate_supervised_executor_group()
    assert signals == [(os.getpid(), delegate.signal.SIGKILL)]


@pytest.mark.skipif(os.name != "posix", reason="group cleanup is POSIX-only")
def test_supervised_executor_group_cleanup_helper_kills_a_same_group_child() -> None:
    """The executor-side primitive includes an ordinary same-group child."""
    script = """
import os
import signal
import subprocess
import sys
import time
from hawking.delegate import _terminate_owned_supervised_executor_group

child = subprocess.Popen([
    sys.executable,
    '-c',
    'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)',
])
print(child.pid, flush=True)
_terminate_owned_supervised_executor_group()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    child_start: str | None = None
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        child_start = delegate.process_start_token(child_pid)
        assert process.wait(timeout=5) == -signal.SIGKILL
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("same-group child survived the executor's normal-completion cleanup")
    finally:
        # If the product code regresses, keep this negative-control fixture
        # recoverable rather than leaving a 30-second process behind.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        if child_pid is not None:
            try:
                if child_start is not None and delegate.process_start_token(child_pid) == child_start:
                    os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="group cleanup is POSIX-only")
def test_supervised_exec_main_persists_verdict_then_reaps_unregistered_verifier_child(
    tmp_path: Path,
) -> None:
    """Exercise the real executor finalizer, not only its group-kill helper.

    A host-owned verifier shell can unregister after it exits while an ordinary
    child remains in the daemon executor's group.  ``exec_main`` must write and
    flush the result first, then use its self-owned group to kill that stale
    child before a later daemon reaper would have only a PID to identify it.
    """

    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    parent_start = delegate.process_start_token(os.getpid())
    if not parent_start:
        pytest.skip("cannot establish a process-incarnation witness for this test")
    token = delegate.issue_daemon_executor_capability(
        workspace,
        supervisor=delegate.DaemonSupervisorWitness(
            pid=os.getpid(), start_token=parent_start
        ),
    )
    marker = tmp_path / "unregistered-verifier-child.json"
    script = r'''
import json
from pathlib import Path
import signal
import subprocess
import sys

from hawking import delegate

workspace = sys.argv[1]
marker = Path(sys.argv[2])

def fake_execute(_workspace, *, verifier_processes=None):
    assert verifier_processes is not None
    child = verifier_processes.spawn(lambda: subprocess.Popen([
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]))
    # Simulate a completed verifier shell whose ordinary descendant outlives
    # the shell.  Final cleanup must use the sticky launch bit, not the live
    # registry set, to reach this child.
    verifier_processes.unregister(child)
    delegate.atomic_write_json(marker, {
        "pid": child.pid,
        "start_token": delegate.process_start_token(child.pid),
    })
    delegate.atomic_write_json(
        delegate.envelope_path(_workspace),
        {"schema": "hawking.test.envelope.v1", "verdict": "ACCEPT"},
    )
    return {"verdict": "ACCEPT"}

delegate.execute_mission = fake_execute
raise SystemExit(delegate.exec_main([workspace]))
'''
    env = dict(os.environ)
    env.pop("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", None)
    env[delegate.DAEMON_EXEC_CAPABILITY_ENV] = token
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(workspace), str(marker)],
        cwd=str(Path(__file__).resolve().parents[1]),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    child_pid: int | None = None
    child_start: str | None = None
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert stderr == ""
        assert stdout == '{"verdict": "ACCEPT"}\n'
        # The executor deliberately kills itself only after stdout is flushed
        # and the durable envelope exists, so Hawkingd can retain its result
        # even though the verifier-bearing executor exits by SIGKILL.
        assert process.returncode == -signal.SIGKILL
        document = delegate._read_json(delegate.envelope_path(workspace))[0]
        assert document == {"schema": "hawking.test.envelope.v1", "verdict": "ACCEPT"}
        marker_document = delegate._read_json(marker)[0]
        assert isinstance(marker_document, dict)
        child_pid = marker_document.get("pid")
        child_start = marker_document.get("start_token")
        assert isinstance(child_pid, int) and child_pid > 0
        assert isinstance(child_start, str) and child_start

        # PID existence alone can mistake a zombie or recycled PID for the
        # child.  Compare the recorded incarnation while waiting for init to
        # reap the killed orphan.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if delegate.process_start_token(child_pid) != child_start:
                break
            time.sleep(0.02)
        else:
            pytest.fail("unregistered same-group verifier child survived exec_main cleanup")
    finally:
        # If a regression leaves either fixture alive, clean it up without
        # masking the assertion that exposed the regression.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        if child_pid is not None:
            try:
                if child_start is None or delegate.process_start_token(child_pid) == child_start:
                    os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="group cleanup is POSIX-only")
def test_abort_of_supervised_executor_reaps_a_same_group_verifier_child(
    tmp_path: Path,
) -> None:
    """The actual abort signal reaches the executor's containment hook.

    Default SIGTERM would skip ``exec_main``'s finalizer.  This exercises a
    verifier child that was started and then unregistered, so neither the
    foreground registry nor a later daemon reaper is relied on to clean it.
    """

    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    parent_start = delegate.process_start_token(os.getpid())
    if not parent_start:
        pytest.skip("cannot establish a process-incarnation witness for this test")
    token = delegate.issue_daemon_executor_capability(
        workspace,
        supervisor=delegate.DaemonSupervisorWitness(
            pid=os.getpid(), start_token=parent_start
        ),
    )
    marker = tmp_path / "abort-verifier-child.json"
    script = r'''
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from hawking import delegate
from hawking.resources import MutationLock

workspace = sys.argv[1]
marker = Path(sys.argv[2])

def fake_execute(_workspace, *, verifier_processes=None):
    assert verifier_processes is not None
    lock = MutationLock(_workspace)
    assert lock.acquire("abort-host-verifier")
    child = verifier_processes.spawn(lambda: subprocess.Popen([
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]))
    verifier_processes.unregister(child)
    delegate.atomic_write_json(marker, {
        "pid": child.pid,
        "start_token": delegate.process_start_token(child.pid),
    })
    while True:
        time.sleep(0.1)

delegate.execute_mission = fake_execute
raise SystemExit(delegate.exec_main([workspace]))
'''
    env = dict(os.environ)
    env.pop("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", None)
    env[delegate.DAEMON_EXEC_CAPABILITY_ENV] = token
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(workspace), str(marker)],
        cwd=str(Path(__file__).resolve().parents[1]),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    child_pid: int | None = None
    child_start: str | None = None
    try:
        deadline = time.monotonic() + 3.0
        while not marker.exists() and time.monotonic() < deadline:
            assert process.poll() is None, process.stderr.read() if process.stderr else ""
            time.sleep(0.02)
        assert marker.exists(), "supervised executor never reached the verifier fixture"
        marker_document = delegate._read_json(marker)[0]
        assert isinstance(marker_document, dict)
        child_pid = marker_document.get("pid")
        child_start = marker_document.get("start_token")
        assert isinstance(child_pid, int) and child_pid > 0
        assert isinstance(child_start, str) and child_start

        abort_script = r'''
import json
import sys
from hawking import delegate

print(json.dumps(delegate.abort(sys.argv[1], reason="test operator abort")))
'''
        aborted = subprocess.run(
            [sys.executable, "-c", abort_script, str(workspace)],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=8,
        )
        assert aborted.returncode == 0, aborted.stderr
        abort_result = json.loads(aborted.stdout)
        assert abort_result["signalled_pid"] == process.pid
        assert delegate.envelope_path(workspace).is_file()
        envelope = delegate._read_json(delegate.envelope_path(workspace))[0]
        assert isinstance(envelope, dict) and envelope.get("verdict") == "ABORTED"

        assert process.wait(timeout=5) == -signal.SIGKILL
        # The abort client is not the executor's parent, so it conservatively
        # may report a lock as live while the test parent still owns its zombie.
        # After reaping the executor, the normal stale-lock rule recovers it.
        assert delegate.MutationLock(workspace).try_break_stale() is True

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if delegate.process_start_token(child_pid) != child_start:
                break
            time.sleep(0.02)
        else:
            pytest.fail("same-group verifier child survived abort containment")
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            try:
                if child_start is None or delegate.process_start_token(child_pid) == child_start:
                    os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_explicit_dev_executor_still_consumes_an_issued_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev-only detached escape must not leave a stale one-use token."""

    workspace = tmp_path / "mission"
    delegate.run("read-only review", workspace=workspace, spawn=False)
    token = delegate.issue_daemon_executor_capability(workspace)
    monkeypatch.setenv(delegate.DAEMON_EXEC_CAPABILITY_ENV, token)
    monkeypatch.setenv("HAWKING_ALLOW_UNSUPERVISED_DELEGATION", "1")
    executor_registries: list[object] = []
    monkeypatch.setattr(
        delegate,
        "execute_mission",
        lambda *_args, **kwargs: (
            executor_registries.append(kwargs.get("verifier_processes"))
            or {"verdict": "BLOCKED"}
        ),
    )

    assert delegate.exec_main([str(workspace)]) == 2
    assert executor_registries == [None]
    assert delegate._daemon_executor_capability_path(workspace).exists() is False
