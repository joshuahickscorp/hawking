"""Operational proof for the HCLI ↔ VMCP evidence boundary."""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import sys

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


FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def _record_gate_causality(
    report: Dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change status/qualification/checks.

    An unsupplied observation is UNTESTED, never a restatement of PASSED/FAILED.
    OVERREACHING is recorded beside the verdict; it does not override it.
    """
    status_before = report.get("status")
    qual_before = report.get("qualification")
    checks_before = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    status = str(report.get("status") or "")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source=source,
    )
    for key in FIVE_RECORDED_FIELDS:
        report[key] = rec[key]
    report["causality_verdict"] = rec["verdict"]
    report["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        report["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        report["claim_kind"] = rec["claim_kind"]
    checks_after = dict(report["checks"]) if isinstance(report.get("checks"), dict) else report.get("checks")
    if (
        report.get("status") != status_before
        or report.get("qualification") != qual_before
        or checks_after != checks_before
    ):
        raise RuntimeError("status_causality.emit mutated the gate verdict")
    return rec


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
            "ToolRegistry invoke of web.search, vmcp.inspect(profile=core), "
            "vmcp.query(vision.capabilities), filesystem.read of visionmcp api.py; "
            "credential-shaped-string scan of the receipt"
        ),
        "direct_observation": (
            f"calls={call_rows}; credentials_secret_free={report.get('credentials_secret_free')!r}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; unmet={unmet!r}"
        ),
        "interpretation": (
            "VMCP core profile was callable and locally hash-validated; receipt had no credential-shaped data"
            if status == "PASSED"
            else f"VMCP evidence-boundary checks unmet: {unmet or ['secret-shaped data']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_vmcp_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/vmcp_gate.py::run_vmcp_gate",
        **payload,
    )

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
    payload = causality_payload(report)
    record_vmcp_causality(report, **payload)
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


__all__ = ["SCHEMA", "causality_payload", "record_vmcp_causality", "records_five_fields", "run_vmcp_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
