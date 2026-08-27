"""Read-only adapter for the repository's discovered VisionMCP surface.

The adapter deliberately does not guess a VMCP protocol.  It locates the
materialized package, imports only its public metadata module, and reports the
factory/profile/tool surface discovered from source.  A future live transport
can be added behind the same boundary without making HCLI's model provider
contract depend on VisionMCP internals.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


SCHEMA = "hcli.vmcp.adapter.v1"
_TOOL_RE = re.compile(r"@mcp\.tool\(\s*name\s*=\s*[\"']([^\"']+)")
VMCP_READ_ONLY_TOOLS = frozenset({
    "system.doctor",
    "project.status",
    "vision.capabilities",
    "vision.observe",
    "vision.query",
    "vision.verify",
    "vision.progress",
    "vision.list_artifacts",
    "vision.get_artifact",
})


def _candidate_source_roots(repo_root: Optional[str | os.PathLike[str]] = None) -> list[Path]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else None
    values = []
    configured = os.environ.get("HCLI_VMCP_ROOT")
    if configured:
        values.append(Path(configured).expanduser())
    if repo:
        values.extend([repo / "visionmcp" / "src", repo / "visionmcp"])
    here = Path(__file__).resolve()
    values.extend([here.parents[1] / "visionmcp" / "src", here.parents[1] / "visionmcp"])
    result: list[Path] = []
    for value in values:
        root = value.resolve(strict=False)
        package = root / "visionmcp"
        if package.is_dir() and (package / "api.py").is_file() and root not in result:
            result.append(root)
    return result


@contextmanager
def _on_import_path(root: Path) -> Iterator[None]:
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass


def _source_tools(package: Path) -> list[str]:
    names: set[str] = set()
    for path in sorted((package / "mcp").glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(_TOOL_RE.findall(text))
    return sorted(names)


def _module_surface(root: Path) -> Dict[str, Any]:
    package = root / "visionmcp"
    report: Dict[str, Any] = {
        "source_root": str(root),
        "package_root": str(package),
        "source_modules": [],
        "importable": False,
        "api_versions": None,
        "profiles": None,
        "factory": None,
        "tools_from_source": _source_tools(package),
    }
    for relative in ("api.py", "profiles.py", "mcp/factory.py", "mcp/core_server.py"):
        path = package / relative
        if path.is_file():
            report["source_modules"].append({
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
    with _on_import_path(root):
        try:
            api = importlib.import_module("visionmcp.api")
            versions = getattr(api, "public_api_versions", None)
            if callable(versions):
                report["api_versions"] = versions()
            profiles = importlib.import_module("visionmcp.profiles")
            list_profiles = getattr(profiles, "list_profiles", None)
            if callable(list_profiles):
                report["profiles"] = list_profiles(include_internal=True)
            factory = importlib.import_module("visionmcp.mcp.factory")
            report["factory"] = {
                "module": "visionmcp.mcp.factory",
                "create_server": callable(getattr(factory, "create_server", None)),
                "run_server": callable(getattr(factory, "run_server", None)),
            }
            # Construct the real core host in a disposable project root. This
            # is an in-process registration check, not a claim that a worker
            # or perception backend executed.
            from visionmcp.mcp.factory import create_server

            with tempfile.TemporaryDirectory(prefix="hcli-vmcp-inspect-") as temp:
                host = create_server(Path(temp), profile="core")
                report["live_core_surface"] = {
                    "constructed": True,
                    "tool_names": sorted(
                        str(tool.name) for tool in host._tool_manager.list_tools()
                    ),
                    "worker_started": False,
                }
            report["importable"] = True
        except Exception as exc:  # noqa: BLE001 - inspection must report blockers
            report["import_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _sha256(path: Path) -> Optional[str]:
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def inspect_vmcp(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    profile: str = "core",
) -> Dict[str, Any]:
    """Inspect the discovered VMCP/VisionMCP API without starting a server."""
    started = time.time()
    roots = _candidate_source_roots(repo_root)
    if not roots:
        return {
            "schema": SCHEMA,
            "status": "UNAVAILABLE",
            "profile_requested": profile,
            "reason": "no materialized visionmcp source root was found",
            "retrieved_at": started,
            "source": None,
            "confidence": "high",
            "unresolved": ["VMCP package source"],
        }
    report = _module_surface(roots[0])
    source_uri = f"file://{roots[0] / 'visionmcp'}"
    profiles = report.get("profiles")
    selected = None
    if isinstance(profiles, list):
        selected = next((item for item in profiles if isinstance(item, dict) and item.get("name") == profile), None)
    unresolved = []
    if not report.get("importable"):
        unresolved.append("public VisionMCP metadata import")
    if selected is None:
        unresolved.append(f"requested profile {profile!r}")
    return {
        "schema": SCHEMA,
        "status": "AVAILABLE" if report.get("importable") else "PARTIAL",
        "profile_requested": profile,
        "selected_profile": selected,
        "source": source_uri,
        "retrieved_at": started,
        "exact_uri": source_uri,
        "confidence": "high" if report.get("importable") and selected is not None else "medium",
        "unresolved": unresolved,
        "api": report,
        "inference_boundary": "source/API metadata is evidence; VMCP inference is not physical truth",
    }


def call_vmcp(
    repo_root: Optional[str | os.PathLike[str]],
    *,
    projects_root: str | os.PathLike[str],
    profile: str = "core",
    tool: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call one allowlisted real VMCP core tool in-process.

    HCLI does not expose the full VisionMCP laboratory as an untyped escape
    hatch.  The public read/evidence tools are the safe bridge; mutations and
    experimental tools remain behind VMCP's own governed interfaces.
    """
    name = str(tool or "").strip()
    if name not in VMCP_READ_ONLY_TOOLS:
        raise PermissionError(f"VMCP tool is not in the HCLI read-only allowlist: {name}")
    roots = _candidate_source_roots(repo_root)
    if not roots:
        raise FileNotFoundError("no materialized VisionMCP source root was found")
    root = roots[0]
    source_uri = f"file://{root / 'visionmcp'}"
    with _on_import_path(root):
        factory = importlib.import_module("visionmcp.mcp.factory")
        create_server = getattr(factory, "create_server", None)
        if not callable(create_server):
            raise RuntimeError("VisionMCP factory does not expose create_server")
        host = create_server(Path(projects_root).expanduser().resolve(), profile=profile)
        registered = host._tool_manager.get_tool(name)
        if registered is None:
            raise LookupError(f"VMCP profile {profile!r} does not expose {name!r}")
        value = asyncio.run(registered.run(dict(arguments or {}), convert_result=False))
    return {
        "schema": "hcli.vmcp.call.v1",
        "status": "OBSERVED",
        "profile": profile,
        "tool": name,
        "result": value,
        "evidence": {
            "source": source_uri,
            "exact_uri": source_uri,
            "retrieved_at": time.time(),
            "confidence": "high",
            "ambiguity": "tool result is evidence from the VMCP boundary; it is not model inference or physical truth",
        },
        "worker_started": False,
    }


__all__ = ["SCHEMA", "VMCP_READ_ONLY_TOOLS", "call_vmcp", "inspect_vmcp"]
