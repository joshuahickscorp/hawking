"""Operational proof for the HAWKING ↔ PERCEPTION evidence boundary."""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[1]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.verify import status_causality as sc

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hawking.persist import atomic_write_json
from hawking.tool_registry import READ_ONLY, RESEARCH, ToolResult, default_tool_registry


SCHEMA = "hawking.perception_gate.v1"
_SECRET_RE = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_:-]+|sk-[A-Za-z0-9_-]+)"
)


FIVE_RECORDED_FIELDS = sc.FIVE_RECORDED_FIELDS
records_five_fields = sc.records_five_fields


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
    if name == "perception.inspect":
        row["native_module"] = value.get("implementation")
        row["tool_names"] = list(value.get("tools") or [])
        row["external_runtime_required"] = value.get("external_runtime_required")
    elif name == "perception.query":
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
    elif name in {"fs", "fs.read", "filesystem.read"}:
        row["path"] = value.get("path")
        row["bytes"] = value.get("bytes")
        row["sha256"] = value.get("sha256")
    return row


def _secret_free(value: Any) -> bool:
    return _SECRET_RE.search(json.dumps(value, sort_keys=True, default=str)) is None



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    calls = report.get("calls") or []
    unmet = [name for name, value in checks.items() if value is not True]
    call_rows = []
    for row in calls:
        if not isinstance(row, dict):
            continue
        call_rows.append(
            {
                "tool": row.get("tool"),
                "ok": row.get("ok"),
                "failure_class": row.get("failure_class"),
                "status": row.get("status"),
            }
        )
    if not checks and not calls:
        return {
            "probe_performed": "",
            "direct_observation": "",
            "interpretation": str(report.get("status") or ""),
            "probe_kind": "",
            "claim_kind": None,
        }
    status = str(report.get("status") or "")
    return {
        "probe_performed": (
            "ToolRegistry invoke of web.search, perception.inspect(profile=core), "
            "perception.query(vision.capabilities), filesystem.read of Hawking perception tools; "
            "credential-shaped-string scan of the receipt"
        ),
        "direct_observation": (
            f"calls={call_rows}; credentials_secret_free={report.get('credentials_secret_free')!r}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; unmet={unmet!r}"
        ),
        "interpretation": (
            "Hawking perception was callable and locally hash-validated; receipt had no credential-shaped data"
            if status == "PASSED"
            else f"Hawking perception checks unmet: {unmet or ['secret-shaped data']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_perception_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return sc.stamp_gate(
        report,
        source="hawking/perception_gate.py::run_perception_gate",
        **payload,
    )

def run_perception_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    search_query: str = "Hawking native perception evidence surface",
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """Prove one source-backed hypothesis through Hawking perception."""
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
    inspected = call("perception.inspect", {"profile": "core"})
    queried = call("perception.query", {"profile": "core", "tool": "vision.capabilities", "arguments": {}})
    # The local surface must remain complete without a foreign visual runtime.
    api_file = repo / "hawking" / "perception" / "tools.py"
    # The canonical `fs` family is the model-facing owner.  Keep the
    # filesystem.read alias callable for compatibility, but production code
    # should stop paying the alias-selection tax.
    local = call("fs", {"op": "read", "path": str(api_file), "max_bytes": 64 * 1024})
    local_validated = bool(
        local
        and local.ok
        and isinstance(local.value, dict)
        and isinstance(local.value.get("sha256"), str)
        and "hawking.perception" in str(local.value.get("content") or "")
    )
    perception_live = bool(
        inspected
        and inspected.ok
        and isinstance(inspected.value, dict)
        and inspected.value.get("implementation_state") == "HAWKING_NATIVE"
        and inspected.value.get("external_runtime_required") is False
        and len(inspected.value.get("tools") or []) == 9
    )
    perception_observed = bool(
        queried
        and queried.ok
        and isinstance(queried.value, dict)
        and queried.value.get("tool") == "vision.capabilities"
    )
    checks = {
        "external_search": bool(external and external.ok and isinstance(external.value, dict) and external.value.get("count", 0) > 0),
        "perception_native_surface": perception_live,
        "perception_tool_called": perception_observed,
        "local_deterministic_validation": local_validated,
    }
    hypothesis = (
        "Hawking perception is callable and exposes a versioned "
        "capability/evidence surface; optional backends must remain capability-gated."
    )
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "qualification": "PERCEPTION_OPERATIONAL" if all(checks.values()) else "PERCEPTION_NOT_OPERATIONAL",
        "started_at": started,
        "finished_at": time.time(),
        "workspace": str(root),
        "repo_root": str(repo),
        "checks": checks,
        "calls": calls,
        "hypothesis": hypothesis,
        "conclusion": "Hawking records its perception capability response as evidence; deterministic source hashing validates only the local API contract, not visual or model inference.",
        "inference_boundary": "Perception observation/decompilation is evidence and never physical truth or verifier authority.",
        "next_action": "route a concrete local visual/repository target through a profile-specific Hawking perception tool and retain its receipt before making a capability claim",
    }
    report["credentials_secret_free"] = _secret_free(report)
    if not report["credentials_secret_free"]:
        report["status"] = "FAILED"
        report["qualification"] = "PERCEPTION_NOT_OPERATIONAL"
        report["blocker"] = "credential-shaped data appeared in the perception receipt"
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / "HAWKING_PERCEPTION_GATE.json"
    report["receipt_path"] = str(destination)
    payload = causality_payload(report)
    record_perception_causality(report, **payload)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--repo-root")
    parser.add_argument("--search-query", default="Hawking native perception evidence surface")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=12.0)
    args = parser.parse_args(argv)
    report = run_perception_gate(
        args.workspace,
        repo_root=args.repo_root,
        emit=args.emit,
        search_query=args.search_query,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "causality_payload", "record_perception_causality", "records_five_fields", "run_perception_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
