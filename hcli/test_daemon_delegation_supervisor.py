"""Daemon ownership tests for durable HCLI delegation workers."""

from __future__ import annotations

from pathlib import Path
import urllib.error

import pytest

from hcli import delegate
from hcli.serve import DaemonDelegationSupervisor


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


def _workspace(root: Path) -> Path:
    workspace = root / ".hcli" / "delegations" / "mission-a"
    spec = workspace / ".hcli" / "mission" / "delegation_spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}")
    return workspace


def test_daemon_supervisor_accepts_only_prepared_local_delegations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    created: list[_Process] = []

    def spawn(*_args, **_kwargs):  # noqa: ANN001, ANN202
        process = _Process()
        created.append(process)
        return process

    monkeypatch.setattr("hcli.serve.subprocess.Popen", spawn)
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
    supervisor.close()
    assert created[0].terminated is True


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

    monkeypatch.setattr("hcli.delegate.urllib.request.urlopen", unavailable)
    monkeypatch.setattr(
        "hcli.delegate.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("must not create an unowned worker"),
    )
    assert delegate._spawn_executor(workspace) is None
