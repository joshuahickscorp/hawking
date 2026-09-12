"""One command: a browser chat connected to the Hawking resident.

Open WebUI is already installed on this machine and is a complete, maintained
chat frontend. Building a second one would be a worse copy of it, so this wires
it up instead: start `hcli serve` (the OpenAI surface over the persistent
resident), start Open WebUI pointed at that surface, and open the browser.

WHAT THIS DELIBERATELY DOES NOT DO. It does not reimplement a UI, a session
store, a model picker or an auth system -- Open WebUI has all four. It owns
exactly the three things Open WebUI cannot know: which resident to load, where
the OpenAI surface is listening, and whether that surface is actually answering.

HEALTH MEANS ANSWERING, NOT LISTENING. A prior scar in this repo: a server
answered /v1/models with HTTP 200 in 4 ms while /v1/chat/completions returned
nothing, and readiness said READY. So reuse of an already-running surface is
gated on /health reporting a resident identity, and the caller is told which
resident it reused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from .serve import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT
from .process_identity import branded_python_entrypoint

WEBUI_PORT = 8080
STATE_DIR = Path.home() / ".hcli" / "web"


class OwnedWebUIManager:
    """Keep every Open WebUI client as a child of the resident daemon.

    Open WebUI is a client surface, not a model owner. It is nevertheless a
    Python process, so letting ``hcli web`` spawn it made the process tree lie
    about Hawking ownership. The live daemon now creates, tracks, reaps, and
    stops these children. Multiple ports remain supported without creating a
    second resident or provider process.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._children: dict[int, dict[str, Any]] = {}
        self._closed = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop,
            name="hawkingd-webui-reaper",
            daemon=True,
        )
        self._reaper.start()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _reap_loop(self) -> None:
        while not self._closed.wait(1.0):
            with self._lock:
                stale = [
                    port for port, row in self._children.items()
                    if (row["proc"].poll() is not None
                        or (int(row["requester_pid"]) > 0
                            and not self._pid_alive(int(row["requester_pid"]))))
                ]
            for port in stale:
                self.stop(port)

    def start(self, port: int, base_url: str,
              requester_pid: int | None = None) -> dict[str, Any]:
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError("Open WebUI port must be between 1 and 65535")
        # A browser is a client surface, but its lifetime is not the lifetime
        # of the shell command that opened it.  Zero/None means daemon-owned:
        # the daemon must not reap the UI when `hcli web` exits.  Positive
        # requester PIDs remain accepted for tests and older callers that
        # explicitly want a short-lived lease.
        requester_pid = int(requester_pid or 0)
        with self._lock:
            current = self._children.get(port)
            if current is not None and current["proc"].poll() is None:
                if (int(current["requester_pid"]) == requester_pid
                        or int(current["requester_pid"]) == 0
                        or requester_pid == 0):
                    return self._row(current)
                raise RuntimeError(f"Open WebUI port {port} is already owned")
            if current is not None:
                self._children.pop(port, None)
            session_dir = STATE_DIR / f"webui-{port}"
            data_dir = session_dir / "data"
            proc, log = start_webui(
                base_url, port, STATE_DIR, data_dir=data_dir
            )
            row = {
                "proc": proc,
                "port": port,
            "requester_pid": requester_pid,
            "owner": "hawkingd" if requester_pid == 0 else "hcli-client",
                "data_dir": data_dir,
                "log": log,
                "base_url": base_url,
            }
            self._children[port] = row
            return self._row(row)

    @staticmethod
    def _row(row: dict[str, Any]) -> dict[str, Any]:
        proc = row["proc"]
        return {
            "pid": int(proc.pid),
            "port": int(row["port"]),
            "requester_pid": int(row["requester_pid"]),
            "owner": str(row.get("owner") or "hcli-client"),
            "data_dir": str(row["data_dir"]),
            "log": str(row["log"]),
            "endpoint": str(row["base_url"]),
            "state": "running" if proc.poll() is None else "exited",
        }

    def stop(self, port: int, *, requester_pid: int | None = None) -> bool:
        with self._lock:
            row = self._children.get(int(port))
            if row is not None and requester_pid is not None \
                    and int(row["requester_pid"]) > 0 \
                    and int(row["requester_pid"]) != int(requester_pid):
                raise PermissionError(f"Open WebUI port {port} belongs to another client")
            row = self._children.pop(int(port), None)
        if row is None:
            return False
        proc = row["proc"]
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
            except (OSError, ProcessLookupError):
                pass
        return True

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._row(row) for row in self._children.values()]

    def close(self) -> None:
        self._closed.set()
        for port in list(self._children):
            self.stop(port)


def _get_json(url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def surface_health(host: str, port: int) -> Optional[Dict[str, Any]]:
    """The resident behind an already-running surface, or None.

    Named `health` rather than `up` on purpose: a socket that accepts is not a
    resident that answers.
    """
    body = _get_json(f"http://{host}:{port}/health")
    if isinstance(body, dict) and body.get("resident"):
        return body
    return None


def wait_for_surface(host: str, port: int, *, timeout: float,
                     proc: Optional[subprocess.Popen] = None,
                     log: Optional[Path] = None) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = surface_health(host, port)
        if health:
            return health
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log and log.is_file():
                tail = log.read_text(errors="replace")[-1500:]
            raise RuntimeError(
                f"the resident surface exited with code {proc.returncode} before it "
                f"answered. Its log is {log}:\n{tail}")
        time.sleep(1.0)
    raise TimeoutError(
        f"no resident answered on http://{host}:{port}/health within {timeout:.0f}s"
        + (f"; see {log}" if log else ""))


def open_webui_command() -> Optional[str]:
    return shutil.which("open-webui") or (
        str(Path.home() / ".local/bin/open-webui")
        if (Path.home() / ".local/bin/open-webui").is_file() else None)


def start_surface(model: str, host: str, port: int, log_dir: Path,
                  write: bool = False) -> tuple:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "serve.log"
    handle = log.open("w")
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        package_root + (os.pathsep + inherited if inherited else "")
    )
    # Route the long-lived surface through the canonical named daemon owner.
    # The installed daemon shim selects the native Hawking executable and -P
    # deployment boundary. Keep a source-checkout fallback for contributors
    # who have not installed shims yet.
    daemon = shutil.which("hawkingd")
    daemon_argv = (
        [daemon, "serve"]
        if daemon
        else [sys.executable, "-P", "-m", "hcli.hawkingd", "serve"]
    )
    proc = subprocess.Popen(
        [*daemon_argv, "--model", model,
         "--host", host, "--port", str(port), *(["--write"] if write else [])],
        # The surface inherits the directory the USER ran `hcli web` in, not the
        # repo this file happens to live in -- otherwise every session would
        # claim Hawking as its context no matter where it was opened.
        cwd=os.getcwd(),
        env=env,
        stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True)
    return proc, log


def start_webui(base_url: str, port: int, log_dir: Path,
                data_dir: Optional[Path] = None) -> tuple:
    data_dir = data_dir or (log_dir / "open-webui")
    data_dir.mkdir(parents=True, exist_ok=True)
    command = open_webui_command()
    if not command:
        raise RuntimeError(
            "open-webui is not installed. Install it with:\n"
            "    uv tool install open-webui\n"
            "or pass --no-webui to run only the OpenAI surface and point your own "
            "client at it.")
    log_dir.mkdir(parents=True, exist_ok=True)
    # Each client owns its own log beside its own database. This matters now
    # that the daemon can supervise several UI ports at once.
    log = data_dir.parent / "webui.log"
    env = dict(os.environ)
    env.update({
        # Both spellings: Open WebUI renamed this to the plural form and still
        # reads the singular one.
        "OPENAI_API_BASE_URL": base_url,
        "OPENAI_API_BASE_URLS": base_url,
        "OPENAI_API_KEY": env.get("OPENAI_API_KEY") or "hawking-local",
        "OPENAI_API_KEYS": env.get("OPENAI_API_KEY") or "hawking-local",
        # Hawking gets its OWN Open WebUI data directory. Measured: WEBUI_AUTH
        # is refused when the database already has users ("You can't turn off
        # authentication because there are existing users"), and this machine
        # has an install from July -- so the shared directory left the page
        # stuck on "Signing in to Open WebUI" forever. A separate DATA_DIR is a
        # fresh installation, where auth-off is honoured, and it leaves that
        # other install and its accounts untouched.
        "DATA_DIR": str(data_dir),
        # Local single-user surface bound to loopback; a login wall on a
        # localhost chat is friction, not security.
        "WEBUI_AUTH": "False",
        "ENABLE_OLLAMA_API": "False",
        # Open WebUI injects a built-in "Arena Model" even when the upstream
        # model endpoint returns only admitted bodies.  HCLI's normal surface
        # is Gravity-only, so disable that unrelated evaluation model at the
        # process boundary as well as filtering Hawking's own catalog.
        "ENABLE_EVALUATION_ARENA_MODELS": "False",
        "EVALUATION_ARENA_MODELS": "[]",
        # Metadata generation is useful for a hosted general-purpose UI, but
        # here it sent the resident the user's entire mission three additional
        # times (title, tags, follow-ups). Those hidden requests competed with
        # the real turn for GPU time and each entered HCLI's tool loop as if it
        # were a second agent mission. Keep the ordinary Hawking client to one
        # model request per user turn; the conversation remains identifiable by
        # its first-message preview.
        "ENABLE_TITLE_GENERATION": "False",
        "ENABLE_TAGS_GENERATION": "False",
        "ENABLE_FOLLOW_UP_GENERATION": "False",
        # The client directories intentionally persist chat history across
        # daemon restarts. Open WebUI also persists application configuration
        # in that database, where an older True value otherwise outranks the
        # process environment above. HCLI owns this constrained local surface,
        # so its launch contract remains authority for model/task settings.
        "ENABLE_PERSISTENT_CONFIG": "False",
        "PORT": str(port),
    })
    handle = log.open("w")
    proc = subprocess.Popen(
        [*branded_python_entrypoint(command, "webui"),
         "serve", "--port", str(port)],
        env=env, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True)
    return proc, log


def request_daemon_webui(base_url: str, port: int,
                         requester_pid: int | None = None) -> dict[str, Any]:
    """Ask the resident daemon to create a client UI as its child."""
    root = base_url.rsplit("/v1", 1)[0].rstrip("/")
    request = urllib.request.Request(
        root + "/hawkingd/webui",
        data=json.dumps({
            "port": int(port),
            # Omit the shell PID by default.  hawkingd then owns the client
            # for its full browser lifetime, so the launcher can exit.
            **({"requester_pid": int(requester_pid)}
               if requester_pid is not None else {}),
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        body = json.loads(response.read() or b"{}")
    if not isinstance(body, dict) or not body.get("pid"):
        raise RuntimeError(f"hawkingd returned an invalid WebUI child record: {body!r}")
    return body


def stop_daemon_webui(base_url: str, port: int, requester_pid: int | None = None) -> None:
    """Ask the daemon to stop the UI client created by this HCLI session."""
    root = base_url.rsplit("/v1", 1)[0].rstrip("/")
    request = urllib.request.Request(
        root + "/hawkingd/webui/stop",
        data=json.dumps({
            "port": int(port),
            "requester_pid": int(requester_pid or os.getpid()),
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0):
            return
    except (OSError, urllib.error.HTTPError):
        # The daemon may already be shutting down. Its child cleanup is the
        # authoritative fallback, so this remains best-effort.
        return


def wait_for_http(url: str, *, timeout: float, proc: Optional[subprocess.Popen] = None,
                  log: Optional[Path] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status < 500:
                    return True
        except urllib.error.HTTPError:
            return True  # answering, even if it wants auth
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            tail = log.read_text(errors="replace")[-1500:] if log and log.is_file() else ""
            raise RuntimeError(
                f"open-webui exited with code {proc.returncode}:\n{tail}")
        time.sleep(1.0)
    return False


def next_webui_port(host: str, requested: int = WEBUI_PORT) -> int:
    """Choose a free UI port without ever creating another model surface.

    The Open WebUI process is a short-lived client surface.  The KIMI HTTP
    endpoint remains owned by ``hawkingd`` and is reused independently.  A
    second blind session therefore gets its own UI listener and data directory
    while sharing the one resident/provider process.
    """
    for port in range(int(requested), int(requested) + 64):
        # A wildcard listener (for example Open WebUI bound to ``*:8080``)
        # can still allow a narrow ``127.0.0.1`` bind on macOS when
        # SO_REUSEADDR is set.  Probe the actual connect path first so a
        # second HCLI client never mistakes an occupied UI port for a free
        # one.
        try:
            with socket.create_connection((host, port), timeout=0.1):
                continue
        except OSError:
            pass
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError(
        f"could not find a free Open WebUI port in {requested}-{requested + 63}"
    )


def enforce_gravity_only_webui(
    data_dir: Path, base_url: str, *, timeout: float = 10.0
) -> None:
    """Persist the Gravity-only model policy and live Hawking endpoint.

    Open WebUI's arena and OpenAI connection settings are persistent: process
    environment defaults do not override rows already present in ``webui.db``.
    A web-client port can outlive a daemon endpoint, so persisting only the
    arena policy can leave a healthy UI dialing a stale HCLI port.  The
    per-port database belongs to HCLI; bind it to the exact health-checked
    Hawking endpoint before handing the UI to a user.  Failure is fail-closed.
    """
    db_path = data_dir / "webui.db"
    deadline = time.time() + max(1.0, float(timeout))
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        if not db_path.is_file():
            time.sleep(0.1)
            continue
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=2.0)
            now = int(time.time())
            wanted = {
                "evaluation.arena.enable": False,
                "evaluation.arena.models": [],
                "openai.enable": True,
                "openai.api_base_urls": [base_url],
                "openai.api_keys": ["hawking-local"],
            }
            for key, value in wanted.items():
                conn.execute(
                    "INSERT INTO config(key, value, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                    "updated_at=excluded.updated_at",
                    (key, json.dumps(value), now),
                )
            conn.commit()
            rows = dict(conn.execute(
                "SELECT key, value FROM config WHERE key IN "
                "('evaluation.arena.enable', 'evaluation.arena.models', "
                "'openai.enable', 'openai.api_base_urls', 'openai.api_keys')"
            ).fetchall())
            expected = {key: json.dumps(value) for key, value in wanted.items()}
            if rows == expected:
                return
            last_error = RuntimeError(
                f"unexpected persisted Hawking WebUI configuration: {rows!r}"
            )
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            last_error = exc
        finally:
            if conn is not None:
                conn.close()
        time.sleep(0.2)
    raise RuntimeError(
        f"could not enforce Gravity-only Open WebUI configuration at {db_path}"
        + (f": {last_error}" if last_error else "")
    )


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hcli web",
        description="Open a browser chat connected to the Hawking resident.")
    # POSITIONAL, because `hcli web Qwen3-14B` is what a person types. --model
    # stays as an alias so existing scripts keep working.
    ap.add_argument("model", nargs="?", default=None,
                    help="which body to load: a name from `hcli use`, or a path")
    ap.add_argument("--model", dest="model_flag", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port for the OpenAI surface (default {DEFAULT_PORT})")
    ap.add_argument("--webui-port", type=int, default=WEBUI_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-webui", action="store_true",
                    help="start only the OpenAI surface and print its URL")
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--write", action="store_true",
                    help="grant repo-scoped write authority (see `hcli build`)")
    a = ap.parse_args(list(argv or []))
    a.model = a.model or a.model_flag

    model = a.model or DEFAULT_MODEL
    log_dir = STATE_DIR

    existing = surface_health(a.host, a.port)
    if existing is None and a.port == DEFAULT_PORT:
        # A live daemon may have been started explicitly on another port. Use
        # its advertised endpoint only after a real health check; never trust
        # stale lock-file text as proof that a surface exists.
        from .hawkingd import daemon_lease_metadata

        incumbent = daemon_lease_metadata()
        try:
            incumbent_port = int(incumbent.get("port"))
        except (TypeError, ValueError):
            incumbent_port = 0
        if incumbent.get("host") in {None, a.host} and incumbent_port:
            incumbent_health = surface_health(a.host, incumbent_port)
            if incumbent_health is not None:
                a.port = incumbent_port
                existing = incumbent_health
    base_url = f"http://{a.host}:{a.port}/v1"
    surface_proc = None
    if existing:
        print(f"reusing the resident already answering on http://{a.host}:{a.port} "
              f"({existing.get('resident')})")
        serve_log = log_dir / "serve.log"
    else:
        print(f"starting resident {Path(model).stem} ...", flush=True)
        surface_proc, serve_log = start_surface(model, a.host, a.port, log_dir,
                                                write=bool(a.write))
        try:
            existing = wait_for_surface(a.host, a.port, timeout=a.ready_timeout,
                                        proc=surface_proc, log=serve_log)
        except Exception as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2

    print(json.dumps({
        "resident": existing.get("resident"),
        "openai_base_url": base_url,
        "sampling": existing.get("sampling"),
        "serve_log": str(serve_log),
    }, indent=2))

    if a.no_webui:
        print(f"\nPoint any OpenAI client at {base_url}")
        return 0

    # Never attach a new `hcli web` invocation to another UI process.  Pick a
    # fresh listener when 8080 (or the requested starting port) is occupied;
    # the endpoint above is still the one shared hawkingd surface.
    try:
        a.webui_port = next_webui_port(a.host, a.webui_port)
    except RuntimeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    ui = f"http://{a.host}:{a.webui_port}"
    print(f"starting the web interface on {ui} ...", flush=True)
    webui_log_dir = log_dir / f"webui-{a.webui_port}"
    try:
        child = request_daemon_webui(base_url, a.webui_port)
        webui_log = Path(str(child["log"]))
        ui_data_dir = Path(str(child["data_dir"]))
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if not wait_for_http(ui, timeout=180.0, proc=None, log=webui_log):
        print(f"the web interface did not answer on {ui}; see {webui_log}",
              file=sys.stderr)
        stop_daemon_webui(base_url, a.webui_port, os.getpid())
        return 2
    try:
        enforce_gravity_only_webui(ui_data_dir, base_url)
    except RuntimeError as exc:
        stop_daemon_webui(base_url, a.webui_port, os.getpid())
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    banner = "  BUILD MODE -- repo-scoped write authority\n" if a.write else ""
    print(f"\n{banner}  {ui}\n  resident {existing.get('resident')}  via {base_url}")
    print(f"  logs: {serve_log}  {webui_log}")
    if not a.no_browser:
        webbrowser.open(ui)
    # The daemon owns both the provider and the Open WebUI child.  This
    # command is a launcher, not another resident: return after the browser
    # surface is ready so `ps` has one Hawking parent instead of one Python
    # wrapper per browser/blind session.  `hcli stop` and the daemon control
    # endpoint remain the lifecycle controls.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


def stop_main(argv: Optional[list] = None) -> int:
    """`hcli stop` -- put down what `hcli web` or `hcli serve` started.

    Scoped ON PURPOSE. It matches this session's own surface and web interface
    and nothing else: the repo has an orphan reaper whose job is unowned
    residents, and a stop command that shoots anything resembling a Hawking
    process would eventually kill somebody's overnight campaign.
    """
    import signal as _signal
    import subprocess as _subprocess

    ap = argparse.ArgumentParser(
        prog="hcli stop", description="Stop the browser chat and its endpoint.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--webui-port", type=int, default=WEBUI_PORT)
    a = ap.parse_args(list(argv or []))

    stopped = []
    # Process titles are a convenience, not ownership. A deployed interpreter
    # without the optional setproctitle package keeps ``python -m
    # hcli.hawkingd`` in argv, so pgrep-by-title missed the real daemon and
    # stopped only one WebUI child. Resolve the lease hint through live health
    # and require both independent PID observations to agree before signalling.
    from .hawkingd import daemon_lease_metadata
    incumbent = daemon_lease_metadata()
    health = surface_health(DEFAULT_HOST, a.port)
    try:
        lease_pid = int(incumbent.get("pid") or 0)
        health_pid = int(((health or {}).get("owner") or {}).get("pid") or 0)
        lease_port = int(incumbent.get("port") or 0)
    except (TypeError, ValueError):
        lease_pid = health_pid = lease_port = 0
    if (lease_pid > 1 and lease_pid == health_pid and lease_port == a.port
            and incumbent.get("role") == "serve"):
        try:
            os.kill(lease_pid, _signal.SIGTERM)
            # A returned signal is not a completed shutdown.  Starting a new
            # surface as soon as the HTTP listener disappears can overlap the
            # old provider's teardown and leave it reparented to launchd.  The
            # lease/health agreement above identifies this exact daemon, so
            # wait for that owner to leave before telling a caller it is safe
            # to start another resident.
            deadline = time.time() + 15.0
            while time.time() < deadline:
                try:
                    os.kill(lease_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                print(f"shutdown pending for endpoint pid {lease_pid}",
                      file=sys.stderr)
                return 2
            stopped.append(f"endpoint pid {lease_pid}")
        except ProcessLookupError:
            pass
    if stopped:
        for line in stopped:
            print(f"stopped {line}")
        return 0

    for label, pattern in (("endpoint", f"hawkingd serve :{a.port}"),
                           ("endpoint", f"hcli serve .*--port {a.port}"),
                           ("endpoint", "hcli serve"),
                           ("web interface", f"open-webui serve --port {a.webui_port}")):
        found = _subprocess.run(["pgrep", "-f", pattern],
                                capture_output=True, text=True)
        for pid in [p for p in found.stdout.split() if p.isdigit()]:
            if int(pid) == os.getpid():
                continue
            try:
                os.kill(int(pid), _signal.SIGTERM)
                stopped.append(f"{label} pid {pid}")
            except ProcessLookupError:
                pass
        if stopped and label == "endpoint":
            break

    if not stopped:
        print("nothing of ours was running.")
        return 0
    for line in stopped:
        print(f"stopped {line}")
    return 0
