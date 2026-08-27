"""Operational proof for the HCLI ↔ VMCP evidence boundary."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json
from hcli.tool_registry import READ_ONLY, RESEARCH, ToolResult, default_tool_registry


SCHEMA = "hcli.agentos.vmcp_gate.v1"
_SECRET_RE = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_:-]+|sk-[A-Za-z0-9_-]+)"
)


def _row(name: str, result: ToolResult) -> Dict[str, Any]:
    value = result.value if isinstance(result.value, dict) else {}
    row: Dict[str, Any] = {
        "tool": name,
        "ok": bool(result.ok),
        "failure_class": result.failure_class,
        "error": result.error,
    }
    for key in ("status", "source", "exact_uri", "confidence", "unresolved", "sha256", "bytes", "count"):
        if key in value:
            row[key] = value[key]
    if name == "vmcp.inspect":
        api = value.get("api") if isinstance(value.get("api"), dict) else {}
        live = api.get("live_core_surface") if isinstance(api.get("live_core_surface"), dict) else {}
        row["live_core_surface"] = {
            "constructed": bool(live.get("constructed")),
            "tool_names": list(live.get("tool_names") or []),
            "worker_started": bool(live.get("worker_started")),
        }
        row["selected_profile"] = value.get("selected_profile")
    elif name == "vmcp.query":
        row["profile"] = value.get("profile")
        row["tool"] = value.get("tool")
        body = value.get("result") if isinstance(value.get("result"), dict) else {}
        row["result_schema"] = body.get("api_versions")
        row["available_count"] = len(body.get("available") or []) if isinstance(body.get("available"), list) else None
        row["blocked_count"] = len(body.get("blocked") or []) if isinstance(body.get("blocked"), list) else None
        evidence = value.get("evidence") if isinstance(value.get("evidence"), dict) else {}
        row["source"] = evidence.get("source")
        row["exact_uri"] = evidence.get("exact_uri")
        row["retrieved_at"] = evidence.get("retrieved_at")
        row["confidence"] = evidence.get("confidence")
    elif name == "web.search":
        row["provider"] = value.get("provider")
        row["source_url"] = value.get("source_url")
        row["retrieved_at"] = value.get("retrieved_at")
        row["sources"] = [
            {"title": item.get("title"), "url": item.get("url")}
            for item in (value.get("results") or [])[:8]
            if isinstance(item, dict)
        ]
    elif name == "filesystem.read":
        row["path"] = value.get("path")
        row["bytes"] = value.get("bytes")
        row["sha256"] = value.get("sha256")
    return row


def _secret_free(value: Any) -> bool:
    return _SECRET_RE.search(json.dumps(value, sort_keys=True, default=str)) is None


def run_vmcp_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    search_query: str = "VisionMCP public API evidence tools",
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """Prove one source-backed hypothesis through the callable VMCP seam."""
    root = Path(workspace or os.getcwd()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root or root).expanduser().resolve()
    registry = default_tool_registry(root, repo_root=repo, permissions={READ_ONLY, RESEARCH})
    started = time.time()
    calls: list[Dict[str, Any]] = []

    def call(name: str, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        result = registry.invoke(name, arguments)
        calls.append(_row(name, result))
        return result

    external = call("web.search", {"query": search_query, "max_results": 5, "timeout_s": timeout_s})
    inspected = call("vmcp.inspect", {"profile": "core"})
    queried = call("vmcp.query", {"profile": "core", "tool": "vision.capabilities", "arguments": {}})
    api_file = repo / "visionmcp" / "src" / "visionmcp" / "api.py"
    local = call("filesystem.read", {"path": str(api_file), "max_bytes": 64 * 1024})
    local_validated = bool(
        local
        and local.ok
        and isinstance(local.value, dict)
        and isinstance(local.value.get("sha256"), str)
        and "visionmcp.tools/v1" in str(local.value.get("content") or "")
    )
    vmcp_live = bool(
        inspected
        and inspected.ok
        and isinstance(inspected.value, dict)
        and isinstance(inspected.value.get("api"), dict)
        and inspected.value["api"].get("live_core_surface", {}).get("constructed") is True
    )
    vmcp_observed = bool(
        queried
        and queried.ok
        and isinstance(queried.value, dict)
        and queried.value.get("tool") == "vision.capabilities"
    )
    checks = {
        "external_search": bool(external and external.ok and isinstance(external.value, dict) and external.value.get("count", 0) > 0),
        "vmcp_source_rediscovered": vmcp_live,
        "vmcp_tool_called": vmcp_observed,
        "local_deterministic_validation": local_validated,
    }
    hypothesis = (
        "The discovered VMCP core profile is callable and exposes a versioned "
        "capability/evidence surface; optional backends must remain capability-gated."
    )
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "qualification": "VMCP_OPERATIONAL" if all(checks.values()) else "VMCP_NOT_OPERATIONAL",
        "started_at": started,
        "finished_at": time.time(),
        "workspace": str(root),
        "repo_root": str(repo),
        "checks": checks,
        "calls": calls,
        "hypothesis": hypothesis,
        "conclusion": "HCLI records the VMCP capability response as evidence; deterministic source hashing validates only the local API contract, not visual or model inference.",
        "inference_boundary": "VMCP observation/decompilation is evidence and never physical truth or verifier authority.",
        "next_action": "route a concrete local visual/repository target through a profile-specific VMCP tool and retain its receipt before making a capability claim",
    }
    report["credentials_secret_free"] = _secret_free(report)
    if not report["credentials_secret_free"]:
        report["status"] = "FAILED"
        report["qualification"] = "VMCP_NOT_OPERATIONAL"
        report["blocker"] = "credential-shaped data appeared in the VMCP receipt"
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HCLI_AGENTOS_VMCP_GATE.json"
    report["receipt_path"] = str(destination)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--repo-root")
    parser.add_argument("--search-query", default="VisionMCP public API evidence tools")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=12.0)
    args = parser.parse_args(argv)
    report = run_vmcp_gate(
        args.workspace,
        repo_root=args.repo_root,
        emit=args.emit,
        search_query=args.search_query,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "run_vmcp_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
