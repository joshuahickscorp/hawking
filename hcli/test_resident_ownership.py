"""Negative controls for resident adoption wiring, plus the positive case.

The decision core in tools/future/resident_adoption.py is pure. These tests
drive the caller it never had: observe a live process, prove it can generate,
adopt only when every check passes. A capability nothing calls does not exist.
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hcli.resident_ownership import (
    ADOPTED,
    REAPED_THEN_SPAWNED,
    SPAWNED,
    CompetingOwnerError,
    ResidentHandle,
    adopt_or_spawn,
)
from hcli.resources import process_start_token
import hcli.resident_ownership as RO

MODEL = "m-abc"
CONFIG = "c-def"
IDENT = {"model_hash": MODEL, "config_hash": CONFIG}


class _Live(http.server.BaseHTTPRequestHandler):
    """Answers /v1/models and actually generates one token."""

    def do_GET(self):  # noqa: N802
        body = json.dumps({"object": "list", "data": [{"id": "ok"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        body = json.dumps({"choices": [{"message": {"content": "1"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _Hang(_Live):
    """Listens. Cannot generate. The orphaned mlx_lm.server."""

    block = threading.Event()

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        self.block.wait(timeout=30)


def _serve(handler):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _sleep_body():
    return subprocess.Popen(
        ["sleep", "120"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _dead_pid():
    proc = _sleep_body()
    pid = proc.pid
    proc.kill()
    proc.wait(timeout=2)
    return pid


def _kill(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture
def world(tmp_path):
    srv, url = _serve(_Live)
    body = _sleep_body()
    spawned = []
    try:
        start = process_start_token(body.pid)
        assert start, "could not read process start identity"
        yield {
            "path": tmp_path / "own.json",
            "url": url,
            "body": body,
            "start": start,
            "spawned": spawned,
            "srv": srv,
        }
    finally:
        _kill(body)
        for proc in spawned:
            _kill(proc)
        srv.shutdown()


def _write(world, **kw):
    rec = {
        "schema": RO._core.SCHEMA,
        "pid": world["body"].pid,
        "proc_start": world["start"],
        "model_hash": MODEL,
        "config_hash": CONFIG,
        "owner_generation": 7,
        "owner_pid": _dead_pid(),
        "heartbeat_epoch": time.time() - 5.0,
        "rss_bytes": 1_000_000,
        "endpoint": world["url"],
        "ready": True,
    }
    rec.update(kw)
    world["path"].write_text(json.dumps(rec))
    return rec


def _spawn(world, model_hash=MODEL, config_hash=CONFIG):
    def spawn_fn():
        proc = _sleep_body()
        world["spawned"].append(proc)
        start = process_start_token(proc.pid) or ""
        return ResidentHandle(
            pid=proc.pid,
            proc_start=start,
            endpoint=world["url"],
            model_hash=model_hash,
            config_hash=config_hash,
            owner_generation=0,
        )

    return spawn_fn


def _must_not_spawn():
    def spawn_fn():
        raise AssertionError("valid resident must be adopted, not spawned")

    return spawn_fn


def test_valid_resident_is_adopted(world):
    """Positive control: a genuinely valid resident IS adopted."""
    _write(world)
    handle, outcome = adopt_or_spawn(IDENT, _must_not_spawn(), world["path"])
    assert outcome == ADOPTED, outcome
    assert handle.pid == world["body"].pid
    assert world["body"].poll() is None
    assert handle.owner_generation == 8
    loaded = RO._core.load(world["path"])
    assert loaded is not None
    assert loaded.owner_generation == 8


def test_stale_pid_is_reaped(world):
    _write(world, pid=_dead_pid(), proc_start="Sat Sep  6 17:00:00 2026")
    handle, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED
    assert handle.pid != world["body"].pid


def test_pid_reuse_is_reaped_and_does_not_kill_the_new_process(world):
    """Pid exists but its start time differs — a recycled pid is a different process."""
    _write(world, proc_start="Sat Sep  6 17:45:00 2026")
    handle, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED
    assert world["body"].poll() is None, "PID reuse must not kill the innocent process"
    assert handle.pid != world["body"].pid


def test_wrong_model_is_reaped(world):
    _write(world)
    ident = {"model_hash": "m-other", "config_hash": CONFIG}
    _, outcome = adopt_or_spawn(
        ident, _spawn(world, model_hash="m-other"), world["path"]
    )
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED


def test_wrong_config_is_reaped(world):
    _write(world)
    ident = {"model_hash": MODEL, "config_hash": "c-other"}
    _, outcome = adopt_or_spawn(
        ident, _spawn(world, config_hash="c-other"), world["path"]
    )
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED


def test_dead_process_endpoint_refuses_is_reaped(world):
    _write(world, endpoint="http://127.0.0.1:1/v1")
    _, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED


def test_partial_startup_listens_but_cannot_generate_is_reaped(world):
    """Endpoint listens, cannot generate. Listener checks must not adopt this."""
    hang = http.server.HTTPServer(("127.0.0.1", 0), _Hang)
    threading.Thread(target=hang.serve_forever, daemon=True).start()
    hang_url = f"http://127.0.0.1:{hang.server_address[1]}/v1"
    try:
        _write(world, endpoint=hang_url)
        _, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
        assert outcome != ADOPTED, outcome
        assert outcome == REAPED_THEN_SPAWNED
    finally:
        _Hang.block.set()
        hang.shutdown()
        _Hang.block.clear()


def test_corrupted_ownership_record_is_reaped(world):
    world["path"].write_text("{not json")
    _, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == SPAWNED
    loaded = RO._core.load(world["path"])
    assert loaded is not None


def test_truncated_record_missing_fields_is_reaped(world):
    world["path"].write_text(json.dumps({"pid": 1}))
    _, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == SPAWNED


def test_two_simultaneous_adopters_exactly_one_wins(world):
    _write(world)
    outcomes: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        try:
            _, outcome = adopt_or_spawn(IDENT, _must_not_spawn(), world["path"])
            with lock:
                outcomes.append(outcome)
        except CompetingOwnerError:
            with lock:
                outcomes.append("LOST")
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()
    assert not errors, errors
    assert outcomes.count(ADOPTED) == 1, outcomes
    assert outcomes.count("LOST") == 1, outcomes
    assert world["body"].poll() is None


def test_owner_crash_during_transfer_is_reaped(world):
    """Claim held, owner gone, record still names a different owner — do not adopt."""
    rec = _write(world)
    transferrer = _dead_pid()
    assert transferrer != rec["owner_pid"]
    world["path"].with_suffix(".claim").write_text(
        json.dumps({"owner_pid": transferrer, "generation": 8})
    )
    _, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome != ADOPTED, outcome
    assert outcome == REAPED_THEN_SPAWNED


def test_previous_owner_death_after_adopt_is_still_adoptable(world):
    """The 211.3 s motive: pool died, claim matches the record owner, body is healthy."""
    owner = _dead_pid()
    _write(world, owner_pid=owner)
    world["path"].with_suffix(".claim").write_text(
        json.dumps({"owner_pid": owner, "generation": 7})
    )
    handle, outcome = adopt_or_spawn(IDENT, _must_not_spawn(), world["path"])
    assert outcome == ADOPTED, outcome
    assert handle.pid == world["body"].pid


def test_decision_core_is_the_authority(world, monkeypatch):
    """Wiring must call decide(), not reimplement it. A capability nothing calls does not exist."""
    seen = []
    real = RO._core.decide

    def wrapped(rec, obs, now):
        seen.append(dict(obs))
        return real(rec, obs, now)

    monkeypatch.setattr(RO._core, "decide", wrapped)
    _write(world)
    _, outcome = adopt_or_spawn(IDENT, _must_not_spawn(), world["path"])
    assert outcome == ADOPTED
    assert seen, "decide() was never called"
    obs = seen[0]
    assert obs["pid_alive"] is True
    assert obs["responsive"] is True
    assert obs["model_hash"] == MODEL
    assert obs["proc_start"] == world["start"]


def test_no_record_spawns(world):
    handle, outcome = adopt_or_spawn(IDENT, _spawn(world), world["path"])
    assert outcome == SPAWNED
    assert handle.pid == world["spawned"][0].pid
    loaded = RO._core.load(world["path"])
    assert loaded is not None
    assert loaded.owner_generation == 1


def test_unix_jsonl_resident_is_adoptable(tmp_path):
    """Primary hawking-native path: generate over a Unix socket after stdin EOF."""
    sock_path = f"/tmp/g005-resident-{os.getpid()}.sock"
    stop = threading.Event()

    def server():
        import socket as _s

        srv = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv.bind(sock_path)
        srv.listen(4)
        srv.settimeout(0.3)
        try:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except _s.timeout:
                    continue
                with conn:
                    buf = b""
                    while not stop.is_set():
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            rec = json.loads(line.decode())
                            reply = json.dumps(
                                {
                                    "id": rec.get("id"),
                                    "status": "ok",
                                    "text": "1",
                                }
                            ) + "\n"
                            conn.sendall(reply.encode())
        finally:
            srv.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)

    threading.Thread(target=server, daemon=True).start()
    deadline = time.time() + 2
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.02)
    body = _sleep_body()
    try:
        start = process_start_token(body.pid)
        path = tmp_path / "own.json"
        rec = {
            "schema": RO._core.SCHEMA,
            "pid": body.pid,
            "proc_start": start,
            "model_hash": MODEL,
            "config_hash": CONFIG,
            "owner_generation": 1,
            "owner_pid": _dead_pid(),
            "heartbeat_epoch": time.time() - 5,
            "rss_bytes": 1_000_000,
            "endpoint": f"unix:{sock_path}",
            "ready": True,
        }
        path.write_text(json.dumps(rec))
        handle, outcome = adopt_or_spawn(IDENT, _must_not_spawn(), path)
        assert outcome == ADOPTED, outcome
        assert handle.pid == body.pid
    finally:
        stop.set()
        _kill(body)


def test_serving_layer_calls_adopt_or_spawn():
    """S001: adoption belongs inside RuntimePool.start, not beside it."""
    source = Path(__file__).resolve().parent.joinpath("runtime.py").read_text()
    assert "adopt_or_spawn(" in source
    start = source[source.find("def start(self)") : source.find("def _start_slot")]
    assert "adopt_or_spawn(" in start, "RuntimePool.start does not call adopt_or_spawn"
