"""Read-only physical connectivity census for the HCLI/AgentOS boundary."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .backends import is_remote_endpoint
from .persist import atomic_write_json


SCHEMA = "hcli.agentos.connectivity.v1"


def _tool(name: str) -> Dict[str, Any]:
    path = shutil.which(name)
    return {"status": "AVAILABLE" if path else "UNAVAILABLE", "executable": path}


def _version(name: str, *args: str) -> Dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"status": "UNAVAILABLE", "executable": None}
    try:
        proc = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=5, check=False
        )
        text = (proc.stdout or proc.stderr or "").strip().splitlines()
        return {
            "status": "AVAILABLE" if proc.returncode == 0 else "BROKEN",
            "executable": path,
            "version": text[0][:240] if text else None,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "BROKEN", "executable": path, "error": type(exc).__name__}


def _public_probe(url: str) -> Dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        url, headers={"User-Agent": "hcli-agentos-connectivity/1"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            response.read(512)
            return {
                "status": "AVAILABLE",
                "http_status": getattr(response, "status", None),
                "url": url,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    except urllib.error.HTTPError as exc:
        # Reachable-but-authenticated or rate-limited is still connectivity;
        # callers must distinguish it from an unreachable network.
        return {
            "status": "AUTH_REQUIRED" if exc.code in {401, 403} else "AVAILABLE",
            "http_status": exc.code,
            "url": url,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "UNAVAILABLE",
            "url": url,
            "error": type(exc).__name__,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _auth_surface(executable: str | tuple[str, ...], env_names: tuple[str, ...]) -> Dict[str, Any]:
    """Validate existing auth without returning command output or tokens."""
    candidates = (executable,) if isinstance(executable, str) else executable
    path = next((shutil.which(item) for item in candidates if shutil.which(item)), None)
    present = any(bool(os.environ.get(name)) for name in env_names)
    report: Dict[str, Any] = {
        "status": "AUTHENTICATED" if present else ("UNAVAILABLE" if path is None else "AUTH_REQUIRED"),
        "executable": path,
        "credential_present": present,
        "credential_value": "[REDACTED]" if present else None,
        "auth_available": present,
        "account": None,
        "source": "environment-presence-only" if present else "not-validated",
    }
    if present or path is None:
        return report

    if "gh" in candidates:
        argv = [path, "auth", "status", "--hostname", "github.com"]
    else:
        argv = [path, "auth", "whoami"]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        # Only extract a non-sensitive account-shaped token. Never include
        # stdout/stderr, which may contain a credential on a misbehaving CLI.
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        import re

        match = re.search(r"(?i)\baccount\s+([A-Za-z0-9_.-]{1,80})", combined)
        if match is None:
            match = re.search(r"(?i)\busername\s*[:=]\s*([A-Za-z0-9_.-]{1,80})", combined)
        report["account"] = match.group(1) if match else None
        report["auth_available"] = proc.returncode == 0
        report["status"] = "AUTHENTICATED" if proc.returncode == 0 else "AUTH_REQUIRED"
        report["source"] = "cli-auth-status"
    except (OSError, subprocess.SubprocessError) as exc:
        report["status"] = "BROKEN"
        report["source"] = "cli-auth-status"
        report["error"] = type(exc).__name__
    return report


def _vmcp_surface(repo: Path, vmcp_src: Path) -> Dict[str, Any]:
    """Use the discovered adapter as the VMCP probe, not a marker directory."""
    try:
        from .vmcp_adapter import inspect_vmcp

        report = inspect_vmcp(repo)
        return {
            "status": report.get("status", "PARTIAL"),
            "hcli_marker": str(repo / "hcli" / "vmcp"),
            "source": report.get("source") or str(vmcp_src),
            "package_materialized": bool((vmcp_src / "visionmcp").is_dir()),
            "confidence": report.get("confidence"),
            "unresolved": report.get("unresolved", []),
        }
    except Exception as exc:  # noqa: BLE001 - census reports broken seams
        return {
            "status": "BROKEN",
            "hcli_marker": str(repo / "hcli" / "vmcp"),
            "source": str(vmcp_src),
            "package_materialized": (vmcp_src / "visionmcp").is_dir(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_connectivity(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    network: bool = True,
) -> Dict[str, Any]:
    """Probe the named seams without downloading models or mutating services."""
    repo = Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    ws = Path(workspace or repo).expanduser().resolve()
    vmcp_src = repo / "visionmcp" / "src"
    modellake = Path("/Volumes/corpdrive/hawking-modellake")
    metal = _tool("system_profiler") if platform.system() == "Darwin" else {"status": "UNAVAILABLE", "reason": "not macOS"}
    if metal.get("status") == "AVAILABLE":
        # The executable exists; a display/Metal report is an optional
        # supporting observation and is deliberately bounded.
        try:
            proc = subprocess.run(
                [str(metal["executable"]), "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            metal["report_available"] = proc.returncode == 0 and bool(proc.stdout.strip())
            metal["status"] = "AVAILABLE" if metal["report_available"] else "PARTIAL"
        except (OSError, subprocess.SubprocessError):
            metal["report_available"] = False

    surfaces: Dict[str, Any] = {
        "filesystem": {
            "status": "AVAILABLE" if repo.is_dir() and ws.is_dir() else "BROKEN",
            "repo_root": str(repo),
            "workspace": str(ws),
            "writable_workspace": os.access(ws, os.W_OK),
        },
        "shell": _tool(os.environ.get("SHELL", "sh")),
        "git": _version("git", "--version"),
        "python": {"status": "AVAILABLE", "executable": sys.executable, "version": platform.python_version()},
        "rust": _version("rustc", "--version"),
        "cargo": _version("cargo", "--version"),
        "metal": metal,
        "vmcp": _vmcp_surface(repo, vmcp_src),
        "modellake": {
            "status": "AVAILABLE" if modellake.is_dir() else "NOT_MOUNTED",
            "root": str(modellake),
            "managed_by": "tools/odyssey/modellake.py",
        },
        "hf_cli": _auth_surface(("hf", "huggingface-cli"), ("HF_TOKEN", "HF_ACCESS_TOKEN")),
        "github_cli": _auth_surface("gh", ("GH_TOKEN", "GITHUB_TOKEN")),
    }
    if network:
        surfaces["web"] = _public_probe("https://www.rfc-editor.org/")
        surfaces["github"] = _public_probe("https://api.github.com/")
        surfaces["huggingface"] = _public_probe("https://huggingface.co/")
    else:
        for name in ("web", "github", "huggingface"):
            surfaces[name] = {"status": "NOT_PROBED", "reason": "network probe disabled"}

    available = [
        name
        for name, item in surfaces.items()
        if item.get("status") in {"AVAILABLE", "AUTHENTICATED"}
    ]
    return {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "repo_root": str(repo),
        "workspace": str(ws),
        "network_probed": bool(network),
        "surfaces": surfaces,
        "available_surfaces": available,
        "secrets_policy": "credential values are never read into the receipt; only presence is recorded",
        "download_performed": False,
        "external_write_performed": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--workspace")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = probe_connectivity(args.repo_root, workspace=args.workspace, network=not args.no_network)
    if args.emit:
        atomic_write_json(Path(args.emit).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "main", "probe_connectivity"]
