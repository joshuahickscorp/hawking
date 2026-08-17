"""Resident Genesis protocol, stopfile, and daemon fallback.

The live 15 GB body is not required. These tests drive the stub server (a
real process on a Unix socket) and a stub greedy binary so liveness, the
stopfile, and the unattended-loop fallback can fail loudly without swapping
the box.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESIDENT = REPO / "tools" / "agentos" / "genesis_resident.py"
DAEMON = REPO / "tools" / "ascent_daemon.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


resident = _load(RESIDENT, "genesis_resident")
daemon = _load(DAEMON, "ascent_daemon")


def _stub_greedy(path: Path, text: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "print('GENERATED_TEXT_VERBATIM: " + text + "')\n"
        "print('FALLBACKS: 0')\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _short_sock(name: str) -> Path:
    # macOS AF_UNIX sun_path is 104 bytes; pytest tmp_path is far longer.
    return Path(f"/tmp/gr-{os.getpid()}-{name}.sock")


def _start_stub(tmp_path: Path, reply: str = "resident-proposal") -> tuple[subprocess.Popen, Path, Path]:
    sock = _short_sock(tmp_path.name[-20:])
    try:
        sock.unlink()
    except FileNotFoundError:
        pass
    stop = tmp_path / "GENESIS_STOP"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(RESIDENT),
            "serve",
            "--stub",
            "--stub-reply",
            reply,
            "--socket",
            str(sock),
            "--stopfile",
            str(stop),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if sock.exists() and resident.body_is_up(sock):
            return proc, sock, stop
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"stub exited {proc.returncode}: {out} {err}")
        time.sleep(0.02)
    proc.kill()
    raise AssertionError("stub did not become live")


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def test_health_reports_body_resident_and_uptime(tmp_path: Path) -> None:
    proc, sock, _stop = _start_stub(tmp_path)
    try:
        info = resident.health(sock, timeout=2)
        assert info is not None
        assert info["body_resident"] is True
        assert info["load_count"] == 1
        assert info["serve_count"] == 0
        assert info["pid"] == proc.pid
        assert resident.process_alive(proc.pid)
        assert info["uptime_ns"] > 0
        assert info["load_ns"] > 0
    finally:
        _stop_proc(proc)


def test_three_proposes_one_load(tmp_path: Path) -> None:
    proc, sock, _stop = _start_stub(tmp_path, reply="serve-text")
    try:
        texts = []
        for i in range(3):
            resp = resident.propose(f"prompt-{i}", sock_path=sock, max_new_tokens=8)
            assert resp is not None
            texts.append(resp["text"])
            assert resp["load_count"] == 1
            assert resp["serve_index"] == i + 1
            assert resp["body_resident"] is True
            assert resp["pid"] == proc.pid
        assert texts == ["serve-text", "serve-text", "serve-text"]
        info = resident.health(sock, timeout=2)
        assert info is not None
        assert info["load_count"] == 1
        assert info["serve_count"] == 3
        assert resident.process_alive(proc.pid)
    finally:
        _stop_proc(proc)


def test_stale_socket_without_process_is_not_live(tmp_path: Path) -> None:
    sock = _short_sock("dead")
    # A leftover socket file is not liveness.
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock))
    listener.close()
    assert sock.exists()
    assert resident.health(sock, timeout=0.2) is None
    assert resident.body_is_up(sock) is False
    assert resident.propose("x", sock_path=sock) is None


def test_status_file_is_ignored(tmp_path: Path) -> None:
    proc, sock, _stop = _start_stub(tmp_path)
    try:
        (tmp_path / "status").write_text("dead\n")
        (tmp_path / "RUNNING").write_text("no\n")
        assert resident.body_is_up(sock) is True
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=2)
        deadline = time.monotonic() + 2
        while resident.process_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert resident.process_alive(proc.pid) is False
        assert (tmp_path / "status").read_text() == "dead\n"
        assert resident.body_is_up(sock) is False
    finally:
        _stop_proc(proc)


def test_stopfile_exits_cleanly(tmp_path: Path) -> None:
    proc, sock, stop = _start_stub(tmp_path)
    assert resident.body_is_up(sock)
    stop.write_text("")
    code = proc.wait(timeout=3)
    assert code == 0
    assert resident.body_is_up(sock) is False
    assert resident.process_alive(proc.pid) is False


def test_daemon_uses_resident_when_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc, sock, _stop = _start_stub(tmp_path, reply="from-resident")
    try:
        monkeypatch.setenv("GENESIS_RESIDENT_SOCK", str(sock))
        # Shell-out would fail: no binary. Resident must supply the text.
        monkeypatch.delenv("GENESIS_BIN", raising=False)
        got = daemon.genesis_proposes("weight_addressing 21293103 ns")
        assert got == "from-resident"
    finally:
        _stop_proc(proc)


def test_daemon_fallback_when_service_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc, sock, _stop = _start_stub(tmp_path, reply="from-resident")
    greedy = _stub_greedy(tmp_path / "greedy", "from-shell")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    monkeypatch.setenv("GENESIS_RESIDENT_SOCK", str(sock))
    monkeypatch.setenv("GENESIS_BIN", str(greedy))
    monkeypatch.setenv("GENESIS_ARTIFACT", str(artifact))
    monkeypatch.setenv("GENESIS_TOKENIZER", str(tokenizer))
    assert daemon.genesis_proposes("host.foo 1 ns") == "from-resident"
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=2)
    deadline = time.monotonic() + 2
    while resident.body_is_up(sock) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert resident.body_is_up(sock) is False
    got = daemon.genesis_proposes("host.foo 1 ns")
    assert got == "from-shell"


def test_daemon_fallback_when_service_never_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sock = tmp_path / "missing.sock"
    greedy = _stub_greedy(tmp_path / "greedy", "shell-only")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setenv("GENESIS_RESIDENT_SOCK", str(sock))
    monkeypatch.setenv("GENESIS_BIN", str(greedy))
    monkeypatch.setenv("GENESIS_ARTIFACT", str(artifact))
    monkeypatch.setenv("GENESIS_TOKENIZER", str(tmp_path / "tok.json"))
    assert daemon.genesis_proposes("metal.bar 2 ns") == "shell-only"


def test_daemon_returns_empty_when_nothing_is_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENESIS_RESIDENT_SOCK", str(tmp_path / "no.sock"))
    monkeypatch.setenv("GENESIS_BIN", str(tmp_path / "missing-bin"))
    monkeypatch.setenv("GENESIS_ARTIFACT", str(tmp_path / "missing-art"))
    t0 = time.monotonic()
    got = daemon.genesis_proposes("anything")
    elapsed = time.monotonic() - t0
    assert got == ""
    assert elapsed < 1.0, "a down service must not stall the loop"


def test_reload_bumps_generation_on_stub(tmp_path: Path) -> None:
    proc, sock, _stop = _start_stub(tmp_path)
    try:
        first = resident.health(sock, timeout=2)
        assert first is not None
        assert first["load_count"] == 1
        resp = resident.request_reload(sock, artifact=tmp_path / "next", generation=1, artifact_sha="ab")
        assert resp is not None
        assert resp.get("ok") is True
        assert resp["load_count"] == 2
        assert resp["generation"] == 1
        third = resident.propose("after-reload", sock_path=sock)
        assert third is not None
        assert third["load_count"] == 2
    finally:
        _stop_proc(proc)
