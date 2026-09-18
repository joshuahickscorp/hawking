"""The public ``h web`` cutover into the current Hawking Web surface.

This module is intentionally a launcher, not a runtime.  ``hawkingd`` remains
the sole daemon and ``WebReleaseStore`` remains the H-Web asset owner.  The
launcher only proves the current local contract, reuses a healthy daemon, and
opens the browser.  It never starts a resident model or imports the historical
HCLI launch path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .remote_cognition import openrouter_credential_available
from .serve import (
    HAWKING_AUTO_ORCHESTRATION_REVISION,
    HWEB_BUILDER_SESSION_REVISION,
    HWEB_CAPABILITY_PROJECTION_REVISION,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8014
DEFAULT_BROWSER_URL = "http://hawking.localhost:8014/"
REQUIRED_ENDPOINTS = frozenset({
    "/health", "/v1/models", "/hawking/review",
    "/hawking/web/release", "/hawking/web/permissions", "/hawking/web/session", "/hawking/web/models",
    "/hawking/web/chat", "/hawking/web/goal", "/hawking/web/events", "/hawking/web/build/session",
    "/hawking/web/attachments", "/hawking/web/artifacts",
    "/hawking/web/artifacts/config", "/hawking/web/memory/search",
    "/hawking/web/markdown",
})
REQUIRED_CLOUD_MODELS = (
    "deepseek/deepseek-v4.1-flash",
    "moonshotai/kimi-k3",
)


class WebLaunchError(RuntimeError):
    """The current Hawking Web contract is unavailable or incompatible."""


def discover_repo_root(explicit: Optional[str] = None) -> Path:
    """Resolve the current source checkout, never the historical HCLI staging tree."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    raw = os.environ.get("HAWKING_ROOT")
    if raw:
        candidates.append(Path(raw).expanduser())
    here = Path.cwd().resolve()
    candidates.extend([here, *here.parents])
    # This is the current user checkout used by the installed public shim.  It
    # is a source fallback only; HAWKING_ROOT and the current working tree win.
    candidates.append(Path.home() / "Downloads" / "hawking")
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "hawking" / "serve.py").is_file() and (root / "pyproject.toml").is_file():
            return root
    raise WebLaunchError("current Hawking source checkout was not found")


def _json_get(url: str, *, timeout: float = 3.0, headers: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise WebLaunchError(f"{url} returned HTTP {response.status}")
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise WebLaunchError(f"Hawking endpoint unavailable: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise WebLaunchError(f"{url} returned a non-object response")
    return value


def _json_post(url: str, payload: Mapping[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    """Call the local daemon without putting build authority in argv or logs."""
    encoded = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url, data=encoded, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise WebLaunchError(f"{url} returned HTTP {response.status}")
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # HTTPError is also a URLError.  Keep its bounded server explanation so
        # a workspace/contract rejection is actionable, but never reflect the
        # handoff request or an arbitrary response body into the terminal.
        message = "no Hawking error detail"
        try:
            decoded = json.loads(exc.read(4096).decode("utf-8", "replace"))
            error = decoded.get("error") if isinstance(decoded, Mapping) else None
            detail = error.get("message") if isinstance(error, Mapping) else None
            if isinstance(detail, str) and detail.strip():
                message = " ".join(detail.split())[:300]
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        raise WebLaunchError(
            f"Hawking builder session mint failed (HTTP {exc.code}): {message}"
        ) from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise WebLaunchError(
            f"Hawking builder session unavailable: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise WebLaunchError("Hawking builder session returned a non-object response")
    return value


def _endpoint(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _contract_snapshot(base: str, *, timeout: float = 3.0, profile: str = "read") -> dict[str, Any]:
    health = _json_get(_endpoint(base, "/health"), timeout=timeout)
    owner = health.get("owner") if isinstance(health.get("owner"), Mapping) else {}
    endpoints = {str(item) for item in health.get("endpoints", []) if isinstance(item, str)}
    missing = sorted(REQUIRED_ENDPOINTS - endpoints)
    if str(owner.get("daemon") or "") != "hawkingd":
        raise WebLaunchError("the endpoint is not owned by hawkingd")
    if missing:
        raise WebLaunchError("current hawkingd contract is missing required endpoints")
    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 1:
        raise WebLaunchError("hawkingd health did not identify a live owner")
    if health.get("hweb_capability_projection_revision") != HWEB_CAPABILITY_PROJECTION_REVISION:
        raise WebLaunchError(
            "running hawkingd lacks the current H-Web capability projection contract"
        )
    if health.get("hawking_auto_orchestration_revision") != HAWKING_AUTO_ORCHESTRATION_REVISION:
        raise WebLaunchError("running hawkingd lacks the current Auto orchestration contract")
    if profile == "build" and health.get("hweb_builder_session_revision") != HWEB_BUILDER_SESSION_REVISION:
        raise WebLaunchError("running hawkingd lacks the current H-Web builder-session contract")

    models = _json_get(_endpoint(base, "/v1/models"), timeout=timeout)
    catalog = models.get("hawking_catalog") if isinstance(models.get("hawking_catalog"), Mapping) else {}
    if catalog.get("search_available") is not True:
        raise WebLaunchError("Hawking model search is not available")
    release = _json_get(_endpoint(base, "/hawking/web/release"), timeout=timeout)
    current = release.get("current") if isinstance(release.get("current"), Mapping) else {}
    if not current.get("release_id") or current.get("contract") != "hawking.web.contract.v1":
        raise WebLaunchError("H-Web has no valid current release")
    # A candidate is intentionally not activated here.  Promotion belongs to
    # the already-qualified release owner, not to an ergonomic shell command.
    return {
        "health": health,
        "models": models,
        "catalog": dict(catalog),
        "release": release,
        "pid": pid,
        "current_release": str(current["release_id"]),
    }


def _cloud_roster_status(base: str, *, timeout: float = 3.0) -> tuple[int, int]:
    """Check the small current cloud pool through Hawking search, not a provider call."""
    found = 0
    for model_id in REQUIRED_CLOUD_MODELS:
        term = model_id.rsplit("/", 1)[-1].split("-", 1)[-1]
        try:
            payload = _json_get(
                _endpoint(base, "/v1/models") + "?" + urllib.parse.urlencode({"search": term}),
                timeout=timeout,
            )
        except WebLaunchError:
            continue
        ids = {
            str(row.get("id"))
            for row in payload.get("data", [])
            if isinstance(row, Mapping) and row.get("id")
        }
        if model_id in ids:
            found += 1
    return found, len(REQUIRED_CLOUD_MODELS)


def _controlled_restart(root: Path, *, host: str, port: int, timeout: float) -> dict[str, Any]:
    """Use Hawking's existing identity-checked stop path, then one daemon start."""
    from .web import stop_main

    stopped = stop_main(["--port", str(port)])
    if stopped != 0:
        raise WebLaunchError("qualified Hawking stop/recovery path refused the restart")
    log_dir = root / ".hawking" / "web-launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hawkingd-recovery.log"
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(root) + ((os.pathsep + existing_path) if existing_path else "")
    command = [
        sys.executable, "-P", "-m", "hawking.hawkingd", "serve",
        "--remote-goal-write", "--host", host, "--port", str(port),
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + max(1.0, timeout)
    while time.monotonic() < deadline:
        try:
            return _contract_snapshot(f"http://{host}:{port}", timeout=1.0)
        except WebLaunchError:
            if process.poll() is not None:
                break
            time.sleep(0.2)
    raise WebLaunchError("hawkingd did not become healthy after controlled recovery")


def launch(
    *,
    root: Optional[Path] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    browser_url: str = DEFAULT_BROWSER_URL,
    no_browser: bool = False,
    timeout: float = 8.0,
    opener: Callable[[str], bool] = webbrowser.open,
    snapshot: Callable[..., dict[str, Any]] = _contract_snapshot,
    restart: Callable[..., dict[str, Any]] = _controlled_restart,
    profile: str = "read",
) -> int:
    if profile not in {"read", "build"}:
        raise WebLaunchError("H-Web launch profile must be read or build")
    root = (root or discover_repo_root()).resolve()
    if not openrouter_credential_available():
        raise WebLaunchError("OpenRouter credential is unavailable from the canonical secure resolver")
    base = f"http://{host}:{port}"
    try:
        try:
            state = snapshot(base, timeout=timeout, profile=profile)
        except TypeError:
            # Injectable test probes from the earlier read-only launcher had a
            # two-argument signature.  Production always uses the typed probe;
            # retain that test seam without duplicating launch machinery.
            state = snapshot(base, timeout=timeout)
        reused = True
    except WebLaunchError:
        state = restart(root, host=host, port=port, timeout=max(30.0, timeout))
        # Recovery returns a health snapshot from the daemon itself.  Verify
        # the requested profile after it is healthy instead of assuming an old
        # recovery callback happened to start the right release.
        try:
            state = snapshot(base, timeout=timeout, profile=profile)
        except TypeError:
            state = snapshot(base, timeout=timeout)
        reused = False
    release = state["current_release"]
    found, total = _cloud_roster_status(base, timeout=timeout)
    launch_url = browser_url
    if profile == "read" and not no_browser:
        launch_url = _endpoint(browser_url, "/hawking/web/read")
    if profile == "build" and not no_browser:
        minted = _json_post(
            _endpoint(base, "/hawking/web/build/session"),
            {"workspace": str(root)}, timeout=timeout,
        )
        handoff = minted.get("handoff_url")
        if not isinstance(handoff, str) or not handoff.startswith("/"):
            raise WebLaunchError("hawkingd did not mint a valid builder-session handoff")
        # The one-time bearer never reaches stdout, shell history, or a
        # process argument.  It exists only in this browser navigation.
        launch_url = _endpoint(browser_url, handoff)
    if not no_browser:
        if not opener(launch_url):
            raise WebLaunchError("the current Hawking Web URL could not be opened")
    print("hawking web ready")
    print(f"  url {browser_url}")
    print(f"  hawkingd {'reused' if reused else 'recovered'} pid {state['pid']}")
    print(f"  web release {release}")
    print(f"  OpenRouter credential keychain: ready")
    print(f"  cloud roster {found}/{total} via Hawking search; Auto available")
    if profile == "build":
        print(
            "  authority build (server-owned session)"
            if not no_browser
            else "  authority build (browser handoff deferred by --no-browser)"
        )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="hawking web", description="Open the current Hawking Web surface")
    parser.add_argument("--root", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default=DEFAULT_HOST, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=argparse.SUPPRESS)
    parser.add_argument("--url", default=DEFAULT_BROWSER_URL, help=argparse.SUPPRESS)
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("read", "build"), default="read", help=argparse.SUPPRESS)
    # Module invocation (`h web` / `h build`) supplies argv through sys.argv;
    # an explicit list is used only by the CLI router and focused tests.
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return launch(
            root=discover_repo_root(args.root), host=args.host, port=args.port,
            browser_url=args.url, no_browser=args.no_browser, profile=args.profile,
        )
    except WebLaunchError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "DEFAULT_BROWSER_URL", "DEFAULT_HOST", "DEFAULT_PORT", "REQUIRED_CLOUD_MODELS",
    "WebLaunchError", "discover_repo_root", "launch", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
