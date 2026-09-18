"""Hawking's allowlisted local perception surface.

The nine read-only perception tools are implemented in ``hawking.perception``
and are invoked in-process.  The surface deliberately has no auxiliary visual
runtime: file observation uses magic bytes, and verification binds a capture
to its declared subject path rather than only a digest.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

PERCEPTION_READ_ONLY_TOOLS = frozenset({
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


def inspect_perception(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    profile: str = "core",
) -> Dict[str, Any]:
    """What this host can perceive, and by what implementation."""
    from .perception import TOOL_NAMES, list_profiles, public_api_versions

    return {
        "status": "AVAILABLE",
        "implementation": "hawking.perception (hawking-native)",
        "implementation_state": "HAWKING_NATIVE",
        "external_runtime_required": False,
        "api_versions": public_api_versions(),
        "profiles": list_profiles(include_internal=True),
        "factory": {
            "module": "hawking.perception",
            "create_server": True,
            "run_server": False,
            "note": "in-process only; nothing here speaks stdio or JSON-RPC, "
                    "because call_perception never needed it to",
        },
        "tools": list(TOOL_NAMES),
        "allowlisted": sorted(PERCEPTION_READ_ONLY_TOOLS),
        "dependencies": {"pillow": False, "opencv": False, "numpy": False},
    }


def call_perception(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    projects_root: str | os.PathLike[str] = ".",
    profile: str = "core",
    tool: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one allowlisted perception tool.

    Callers in ``hawking.tool_registry`` pass ``repo_root`` positionally and a
    typed tool name by keyword.  Anything outside
    ``PERCEPTION_READ_ONLY_TOOLS``
    raises PermissionError before any work happens.
    """
    name = str(tool or "")
    if name not in PERCEPTION_READ_ONLY_TOOLS:
        raise PermissionError(
            f"{name} is not in the read-only perception allowlist: "
            f"{sorted(PERCEPTION_READ_ONLY_TOOLS)}")
    from .perception import create_server

    root = Path(projects_root).expanduser().resolve()
    host = create_server(root, profile=profile)
    registered = host._tool_manager.get_tool(name)
    if registered is None:
        raise LookupError(f"profile {profile!r} does not expose {name!r}")
    value = asyncio.run(registered.run(dict(arguments or {}), convert_result=False))
    return {
        "schema": "hawking.perception.call.v1",
        "status": "OBSERVED",
        "profile": profile,
        "tool": name,
        "result": value,
        "evidence": {
            "source": "hawking.perception",
            "exact_uri": "hawking/perception/tools.py",
            "retrieved_at": time.time(),
            "confidence": "high",
            "ambiguity": "tool result is evidence from the local perception boundary; "
                         "it is not model inference or physical truth",
        },
        "worker_started": False,
    }
