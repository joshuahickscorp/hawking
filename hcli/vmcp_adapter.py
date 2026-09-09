"""Perception surface for HCLI. SUBLATED -- no foreign package.

This module used to proxy to `visionmcp`, a separate gitignored checkout: it
put that package on sys.path, imported `visionmcp.mcp.factory`, built its MCP
host and drove one of its ~303 tools. Nine were ever allowlisted, and only
those nine were ever reachable.

Those nine are now implemented natively in `hcli.perception`, so hawking no
longer depends on a repository it does not contain. Same names, same calling
convention, same allowlist gate. Two differences, both deliberate:

  * `vision.observe` classifies by magic bytes via file_eye, needing neither
    Pillow nor OpenCV. The foreign path required both -- cv2 unconditionally,
    through a lazy proxy that a plain grep does not see.
  * `vision.verify` binds a capture to its SUBJECT PATH, not only to a digest,
    so identical bytes at a different path do not silently verify.

The public surface is unchanged for callers: VMCP_READ_ONLY_TOOLS,
inspect_vmcp, call_vmcp, and the two source-scan helpers.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

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


def _candidate_source_roots(
    repo_root: Optional[str | os.PathLike[str]] = None,
) -> list[Path]:
    """Retained for callers that still ask where a foreign source tree might be.

    After sublation there is no foreign source root to find. Returning an empty
    list is the honest answer, and it is what makes `vmcp.tools` report
    n_in_visionmcp as None rather than inventing a count.
    """
    return []


def _source_tools(package: Path) -> list[str]:
    """No foreign package to scan. Kept so the old call site does not break."""
    return []


def inspect_vmcp(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    profile: str = "core",
) -> Dict[str, Any]:
    """What this host can perceive, and by what implementation."""
    from .perception import TOOL_NAMES, list_profiles, public_api_versions

    return {
        "implementation": "hcli.perception (hawking-native)",
        "sublated": True,
        "foreign_package_required": False,
        "api_versions": public_api_versions(),
        "profiles": list_profiles(include_internal=True),
        "factory": {
            "module": "hcli.perception",
            "create_server": True,
            "run_server": False,
            "note": "in-process only; nothing here speaks stdio or JSON-RPC, "
                    "because call_vmcp never needed it to",
        },
        "tools": list(TOOL_NAMES),
        "allowlisted": sorted(VMCP_READ_ONLY_TOOLS),
        "dependencies": {"pillow": False, "opencv": False, "numpy": False},
    }


def call_vmcp(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    projects_root: str | os.PathLike[str] = ".",
    profile: str = "core",
    tool: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one allowlisted perception tool.

    Signature and return shape are UNCHANGED from the proxying version --
    callers in hcli/tool_registry.py pass repo_root positionally and tool by
    keyword, and read back schema/status/profile/tool/result/evidence. Only the
    implementation moved.

    The allowlist gate is unchanged: anything outside VMCP_READ_ONLY_TOOLS
    raises PermissionError before any work happens.
    """
    name = str(tool or "")
    if name not in VMCP_READ_ONLY_TOOLS:
        raise PermissionError(
            f"{name} is not in the read-only perception allowlist: "
            f"{sorted(VMCP_READ_ONLY_TOOLS)}")
    from .perception import create_server

    root = Path(projects_root).expanduser().resolve()
    host = create_server(root, profile=profile)
    registered = host._tool_manager.get_tool(name)
    if registered is None:
        raise LookupError(f"profile {profile!r} does not expose {name!r}")
    value = asyncio.run(registered.run(dict(arguments or {}), convert_result=False))
    return {
        "schema": "hcli.vmcp.call.v1",
        "status": "OBSERVED",
        "profile": profile,
        "tool": name,
        "result": value,
        "evidence": {
            "source": "hcli.perception",
            "exact_uri": "hcli/perception/tools.py",
            "retrieved_at": time.time(),
            "confidence": "high",
            "ambiguity": "tool result is evidence from the local perception boundary; "
                         "it is not model inference or physical truth",
        },
        "worker_started": False,
    }
