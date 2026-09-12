from __future__ import annotations

import socket
import sqlite3

from hcli.web import enforce_gravity_only_webui, next_webui_port


def test_next_webui_port_skips_an_occupied_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    occupied = sock.getsockname()[1]
    try:
        assert next_webui_port("127.0.0.1", occupied) == occupied + 1
    finally:
        sock.close()


def test_surface_child_uses_the_launchers_package_not_the_worktree(monkeypatch,
                                                                  tmp_path):
    import hcli.web as web

    captured = {}

    class FakeProcess:
        pid = 17

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return FakeProcess()

    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(web.os, "getcwd", lambda: str(tmp_path / "older-worktree"))
    monkeypatch.setattr(
        web.shutil,
        "which",
        lambda name: str(tmp_path / "hawkingd") if name == "hawkingd" else None,
    )
    proc, _ = web.start_surface(
        "KIMI_P0_OPERATIONAL", "127.0.0.1", 8011, tmp_path / "logs", write=True
    )
    assert proc.pid == 17
    assert captured["argv"][:2] == [str(tmp_path / "hawkingd"), "serve"]
    package_root = str(web.Path(web.__file__).resolve().parent.parent)
    assert captured["env"]["PYTHONPATH"].split(web.os.pathsep)[0] == package_root
    assert captured["cwd"].endswith("older-worktree")


def test_enforce_gravity_only_webui_updates_persistent_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = sqlite3.connect(data_dir / "webui.db")
    db.execute(
        "CREATE TABLE config (key TEXT PRIMARY KEY, value JSON NOT NULL, "
        "updated_at BIGINT)"
    )
    db.execute(
        "INSERT INTO config(key, value, updated_at) VALUES (?, ?, ?)",
        ("evaluation.arena.enable", "true", 0),
    )
    db.execute(
        "INSERT INTO config(key, value, updated_at) VALUES (?, ?, ?)",
        ("evaluation.arena.models", "[]", 0),
    )
    db.execute(
        "INSERT INTO config(key, value, updated_at) VALUES (?, ?, ?)",
        ("openai.api_base_urls", '["http://127.0.0.1:8014/v1"]', 0),
    )
    db.commit()
    db.close()

    enforce_gravity_only_webui(data_dir, "http://127.0.0.1:8011/v1")

    db = sqlite3.connect(data_dir / "webui.db")
    rows = dict(db.execute(
        "SELECT key, value FROM config WHERE key IN "
        "('evaluation.arena.enable', 'evaluation.arena.models', "
        "'openai.enable', 'openai.api_base_urls', 'openai.api_keys')"
    ).fetchall())
    db.close()
    assert rows == {
        "evaluation.arena.enable": "false",
        "evaluation.arena.models": "[]",
        "openai.enable": "true",
        "openai.api_base_urls": '["http://127.0.0.1:8011/v1"]',
        "openai.api_keys": '["hawking-local"]',
    }


def test_owned_webui_manager_spawns_and_stops_a_daemon_child(monkeypatch, tmp_path):
    import hcli.web as web

    class FakeProcess:
        pid = 9001

        def __init__(self):
            self.signals = []
            self.returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, signal):
            self.signals.append(signal)
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    proc = FakeProcess()
    log = tmp_path / "webui.log"
    monkeypatch.setattr(web, "STATE_DIR", tmp_path / "web")
    monkeypatch.setattr(web, "start_webui", lambda *args, **kwargs: (proc, log))

    manager = web.OwnedWebUIManager()
    child = manager.start(8087, "http://127.0.0.1:8014/v1", 1234)
    assert child["pid"] == 9001
    assert child["port"] == 8087
    assert child["requester_pid"] == 1234
    assert manager.snapshot()[0]["state"] == "running"

    assert manager.stop(8087, requester_pid=1234) is True
    assert proc.signals
    manager.close()


def test_daemon_owned_webui_survives_launcher_exit(monkeypatch, tmp_path):
    import hcli.web as web

    class FakeProcess:
        pid = 9002

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    proc = FakeProcess()
    monkeypatch.setattr(web, "STATE_DIR", tmp_path / "web")
    monkeypatch.setattr(
        web, "start_webui", lambda *args, **kwargs: (proc, tmp_path / "webui.log")
    )

    manager = web.OwnedWebUIManager()
    child = manager.start(8088, "http://127.0.0.1:8011/v1")
    assert child["owner"] == "hawkingd"
    assert child["requester_pid"] == 0
    assert manager.snapshot()[0]["owner"] == "hawkingd"

    # A launcher PID is intentionally not required to stop a daemon-owned UI.
    assert manager.stop(8088, requester_pid=1234) is True
    manager.close()


def test_stop_uses_live_health_and_lease_when_process_title_is_unavailable(
        monkeypatch, capsys):
    import hcli.hawkingd as hawkingd
    import hcli.web as web

    killed = []
    monkeypatch.setattr(hawkingd, "daemon_lease_metadata", lambda: {
        "role": "serve", "pid": 4242, "port": 8011,
    })
    monkeypatch.setattr(web, "surface_health", lambda _host, _port: {
        "owner": {"pid": 4242}, "resident": "KIMI_P0_OPERATIONAL",
    })
    def fake_kill(pid, sig):
        killed.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError(pid)

    monkeypatch.setattr(web.os, "kill", fake_kill)
    monkeypatch.setattr(
        web.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stop fell back to argv matching despite exact ownership")),
    )

    assert web.stop_main([]) == 0
    assert killed and killed[0][0] == 4242
    assert "stopped endpoint pid 4242" in capsys.readouterr().out
