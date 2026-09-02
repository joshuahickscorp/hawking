"""The long-lived daemon is named for itself, not for one model.

`ps` used to show the supervisor and worker as a bare interpreter line,
`python3 -m hcli.agentos.resident --supervise <state>`, which says nothing
about what the process is and implies a single model when a supervisor may
hold several bodies in parallel or in succession. The daemon is `hawkingd`;
`hcli` remains the client that talks to it.
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

import hcli.agentos.resident as R
from hcli.cli import DAEMON_SHIM, install_shims


def test_daemon_argv_prefers_the_named_executable(tmp_path, monkeypatch):
    fake = tmp_path / DAEMON_SHIM
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    argv = R.daemon_argv("--supervise", "/tmp/state.json")
    assert argv[0] == str(fake), f"daemon did not launch under its own name: {argv}"
    assert argv[1:] == ["--supervise", "/tmp/state.json"]


def test_daemon_argv_falls_back_when_the_shim_is_absent(tmp_path, monkeypatch):
    """A source checkout with no installed shim must still be able to start.

    A daemon that cannot launch because its own name is missing would be a poor
    trade for a nicer `ps` line.
    """
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python-with-no-sibling"))
    argv = R.daemon_argv("--worker", "/tmp/state.json")
    assert argv[:3] == [sys.executable, "-m", "hcli.hawkingd"], argv
    assert argv[3:] == ["--worker", "/tmp/state.json"]


def test_daemon_argv_finds_the_shim_beside_the_interpreter(tmp_path, monkeypatch):
    """A detached supervisor does not necessarily inherit the installing PATH."""
    binhome = tmp_path / "bin"
    binhome.mkdir()
    (binhome / "python3").write_text("", encoding="utf-8")
    sibling = binhome / DAEMON_SHIM
    sibling.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sibling.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "executable", str(binhome / "python3"))
    argv = R.daemon_argv("--supervise", "/s.json")
    assert argv[0] == str(sibling), argv


def test_an_unknown_role_is_refused():
    """argv is built from a role, so a typo must not silently launch nothing."""
    for bad in ("--supervisor", "supervise", "", "--serve"):
        try:
            R.daemon_argv(bad, "/s.json")
        except ValueError:
            continue
        raise AssertionError(f"daemon_argv accepted unknown role {bad!r}")


def test_daemon_main_routes_both_long_lived_roles():
    """The entry point must dispatch, not fall through to the client parser."""
    import inspect

    src = inspect.getsource(R.daemon_main)
    assert "--supervise" in src and "ResidentSupervisor" in src
    assert "--worker" in src and "_worker_main" in src


def test_install_writes_a_daemon_shim_that_is_not_the_client():
    with tempfile.TemporaryDirectory() as tmp:
        assert install_shims(home=tmp) == 0
        bin_dir = Path(tmp) / ".local" / "bin"
        daemon = bin_dir / DAEMON_SHIM
        client = bin_dir / "hcli"
        assert daemon.is_file(), f"{DAEMON_SHIM} shim was not installed"
        assert daemon.stat().st_mode & stat.S_IXUSR, "daemon shim is not executable"
        dtext = daemon.read_text()
        assert "-m hcli.hawkingd" in dtext, (
            f"daemon shim does not run the daemon module:\n{dtext}"
        )
        assert dtext != client.read_text(), (
            "the daemon shim is a copy of the client; it would not route "
            "--supervise/--worker"
        )
        assert ".local/share/hcli/current" in dtext, "daemon shim lost its PYTHONPATH"
