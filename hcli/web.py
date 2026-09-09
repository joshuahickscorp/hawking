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
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from .serve import DEFAULT_HOST, DEFAULT_PORT

WEBUI_PORT = 8080
STATE_DIR = Path.home() / ".hcli" / "web"


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
    proc = subprocess.Popen(
        [sys.executable, "-m", "hcli", "serve", "--model", model,
         "--host", host, "--port", str(port), *(["--write"] if write else [])],
        # The surface inherits the directory the USER ran `hcli web` in, not the
        # repo this file happens to live in -- otherwise every session would
        # claim Hawking as its context no matter where it was opened.
        cwd=os.getcwd(),
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
    log = log_dir / "webui.log"
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
        "PORT": str(port),
    })
    handle = log.open("w")
    proc = subprocess.Popen(
        [command, "serve", "--port", str(port)],
        env=env, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True)
    return proc, log


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

    model = a.model or str(
        Path(__file__).resolve().parent / "hawking-native.sealed-3.14.json")
    base_url = f"http://{a.host}:{a.port}/v1"
    log_dir = STATE_DIR

    existing = surface_health(a.host, a.port)
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
        if surface_proc is not None:
            print("Press Ctrl-C to stop.")
            try:
                surface_proc.wait()
            except KeyboardInterrupt:
                surface_proc.send_signal(signal.SIGINT)
        return 0

    ui = f"http://{a.host}:{a.webui_port}"
    if wait_for_http(ui, timeout=2.0):
        print(f"reusing the web interface already serving on {ui}")
        webui_proc, webui_log = None, log_dir / "webui.log"
    else:
        print(f"starting the web interface on {ui} ...", flush=True)
        try:
            webui_proc, webui_log = start_webui(base_url, a.webui_port, log_dir)
        except RuntimeError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        if not wait_for_http(ui, timeout=180.0, proc=webui_proc, log=webui_log):
            print(f"the web interface did not answer on {ui}; see {webui_log}",
                  file=sys.stderr)
            return 2

    banner = "  BUILD MODE -- repo-scoped write authority\n" if a.write else ""
    print(f"\n{banner}  {ui}\n  resident {existing.get('resident')}  via {base_url}")
    print(f"  logs: {serve_log}  {webui_log}")
    if not a.no_browser:
        webbrowser.open(ui)
    print("\nPress Ctrl-C to stop what this command started.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        for proc in (webui_proc, surface_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        print("stopped")
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
