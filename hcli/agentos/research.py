"""Bounded public-research operational gate for AgentOS.

The gate exercises the same typed ToolRegistry that a mission uses.  It is
deliberately small and repeatable: public web search, an official source
fetch, GitHub repository inspection, Hugging Face revision/file access, and a
hash-checked download of one harmless metadata file.  It records provenance
and capability observations, never model content or credentials.
"""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.verify import status_causality as sc

import sys

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json
from hcli.tool_registry import (
    COSTLY,
    READ_ONLY,
    RESEARCH,
    REVERSIBLE_REPO,
    REVERSIBLE_RUNTIME,
    ToolResult,
    default_tool_registry,
)


SCHEMA = "hcli.agentos.research_gate.v1"
DEFAULT_REPO = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_SEARCH = "Cloudflare Agents SDK official documentation"
OFFICIAL_SOURCE = "https://www.rfc-editor.org/rfc/rfc9110.txt"
GITHUB_SOURCE = "https://api.github.com/repos/cloudflare/agents"
HF_FILE = "config.json"
_TOKEN_RE = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_:-]+|sk-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9-]+)"
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret|private[_-]?key|bearer|token)\s*[:=]\s*[^\s,;]+"
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


def _safe_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _credential_free(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True, default=str)
    return _TOKEN_RE.search(text) is None and _ASSIGNMENT_RE.search(text) is None


def _result_row(name: str, result: ToolResult) -> Dict[str, Any]:
    """Keep only gate evidence; omit fetched prose and request credentials."""
    document = result.to_dict()
    value = document.get("value") if isinstance(document, dict) else None
    row: Dict[str, Any] = {
        "tool": name,
        "ok": bool(document.get("ok")),
        "failure_class": document.get("failure_class"),
        "error": document.get("error"),
        "mutation": document.get("mutation"),
    }
    if not isinstance(value, dict):
        return row
    for key in (
        "provider",
        "kind",
        "count",
        "total_count",
        "bytes_read",
        "truncated",
        "status",
        "repo",
        "requested_revision",
        "resolved_revision",
        "path",
        "sha256",
        "destination",
        "bytes",
        "download_performed",
        "atomic_publish",
        "authenticated",
        "auth_available",
        "credential_values_recorded",
    ):
        if key in value:
            row[key] = _safe_json(value[key])
    if isinstance(value.get("source_url"), str):
        row["source_url"] = value["source_url"]
    elif isinstance(value.get("final_url"), str):
        row["source_url"] = value["final_url"]
    provenance = value.get("provenance")
    if isinstance(provenance, dict):
        row["retrieved_at"] = provenance.get("retrieved_at")
        row["exact_uri"] = provenance.get("source_url") or provenance.get("exact_uri")
    if "retrieved_at" in value:
        row["retrieved_at"] = value.get("retrieved_at")
    if name == "web.search" and isinstance(value.get("results"), list):
        row["sources"] = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
            }
            for item in value["results"][:10]
            if isinstance(item, dict)
        ]
    if name == "github.fetch" and isinstance(value.get("content"), str):
        try:
            payload = json.loads(value["content"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            row["repository"] = {
                key: payload.get(key)
                for key in ("full_name", "html_url", "default_branch", "description", "updated_at")
                if payload.get(key) is not None
            }
    if name == "huggingface.fetch_file" and isinstance(value.get("content"), str):
        try:
            payload = json.loads(value["content"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            row["metadata_keys"] = sorted(str(key) for key in payload)[:200]
    return row


def _run(registry: Any, name: str, arguments: Dict[str, Any]) -> tuple[ToolResult, Dict[str, Any]]:
    result = registry.invoke(name, arguments)
    return result, _result_row(name, result)



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    calls = report.get("calls") or []
    auth = report.get("auth") if isinstance(report.get("auth"), dict) else {}
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
                "download_performed": row.get("download_performed"),
                "atomic_publish": row.get("atomic_publish"),
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
            "ToolRegistry research seam: web.search, official web.fetch, github.search, "
            "github.fetch, huggingface.resolve, huggingface.fetch_file, "
            "huggingface.download of one metadata file with expected_sha256; "
            "credential-shaped-string scan of the receipt"
        ),
        "direct_observation": (
            f"calls={call_rows}; resolved_revision={report.get('resolved_revision')!r}; "
            f"auth={auth}; credentials_secret_free={report.get('credentials_secret_free')!r}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; unmet={unmet!r}"
        ),
        "interpretation": (
            "bounded public research tools returned source-backed metadata and the receipt was credential-free"
            if status == "PASSED"
            else f"research-tool checks unmet: {unmet or ['secret-shaped data']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_research_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/research.py::run_research_gate",
        **payload,
    )

def run_research_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    repo: str = DEFAULT_REPO,
    search_query: str = DEFAULT_SEARCH,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """Run and persist the minimum research-operational gate."""
    root = Path(workspace or tempfile.mkdtemp(prefix="hcli-research-gate-")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(repo_root or root).expanduser().resolve()
    started = time.time()
    permissions = {READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME, COSTLY}
    registry = default_tool_registry(root, repo_root=repository, permissions=permissions)
    calls: list[Dict[str, Any]] = []
    results: Dict[str, ToolResult] = {}

    def call(name: str, arguments: Dict[str, Any]) -> Optional[ToolResult]:
        try:
            result, row = _run(registry, name, arguments)
        except Exception as exc:  # registry normally contains failures itself
            row = {
                "tool": name,
                "ok": False,
                "failure_class": type(exc).__name__,
                "error": str(exc),
            }
            result = None
        calls.append(row)
        if result is not None:
            results[name] = result
        return result

    search = call("web.search", {"query": search_query, "max_results": 5, "timeout_s": timeout_s})
    official = call("web.fetch", {"url": OFFICIAL_SOURCE, "max_bytes": 8192, "timeout_s": timeout_s})
    github_search = call(
        "github.search",
        {"query": "org:cloudflare agents sdk", "kind": "repositories", "max_results": 5, "timeout_s": timeout_s},
    )
    github = call("github.fetch", {"url": GITHUB_SOURCE, "max_bytes": 16384, "timeout_s": timeout_s})
    resolved = call("huggingface.resolve", {"repo": repo, "revision": "main"})
    revision = None
    if resolved is not None and resolved.ok and isinstance(resolved.value, dict):
        revision = resolved.value.get("resolved_revision")
    hf_fetch = None
    hf_download = None
    download_destination = None
    with tempfile.TemporaryDirectory(prefix="hcli-hf-metadata-", dir=str(root)) as download_root:
        if revision:
            hf_fetch = call(
                "huggingface.fetch_file",
                {"repo": repo, "revision": str(revision), "path": HF_FILE, "max_bytes": 128 * 1024, "timeout_s": timeout_s},
            )
            expected = None
            if hf_fetch is not None and hf_fetch.ok and isinstance(hf_fetch.value, dict):
                expected = hf_fetch.value.get("sha256")
            download_destination = str(Path(download_root) / HF_FILE)
            hf_download = call(
                "huggingface.download",
                {
                    "repo": repo,
                    "revision": str(revision),
                    "path": HF_FILE,
                    "destination": download_destination,
                    "confirm": True,
                    "max_bytes": 128 * 1024,
                    "expected_sha256": expected,
                    "timeout_s": timeout_s,
                },
            )

    checks = {
        "web_search": bool(search and search.ok and isinstance(search.value, dict) and search.value.get("count", 0) > 0),
        "official_fetch": bool(official and official.ok),
        "github_search": bool(github_search and github_search.ok),
        "github_repository": bool(github and github.ok),
        "huggingface_resolve": bool(resolved and resolved.ok and revision),
        "huggingface_file": bool(hf_fetch and hf_fetch.ok),
        "huggingface_download": bool(
            hf_download
            and hf_download.ok
            and isinstance(hf_download.value, dict)
            and hf_download.value.get("download_performed") is True
            and hf_download.value.get("atomic_publish") is True
        ),
    }
    source_provenance = [
        {
            "source": row.get("source_url") or row.get("exact_uri"),
            "exact_uri": row.get("exact_uri") or row.get("source_url"),
            "retrieved_at": row.get("retrieved_at"),
            "tool": row.get("tool"),
            "sha256": row.get("sha256"),
            "confidence": "high" if row.get("ok") else "unavailable",
        }
        for row in calls
        if row.get("source_url") or row.get("exact_uri")
    ]
    auth = {
        "github_auth_available": next((row.get("auth_available") for row in calls if row.get("tool") == "github.search"), None),
        "github_authenticated": next((row.get("authenticated") for row in calls if row.get("tool") == "github.search"), None),
        "huggingface_authenticated": next((row.get("authenticated") for row in calls if row.get("tool") == "huggingface.resolve"), None),
        "credential_values_recorded": False,
    }
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "qualification": "RESEARCH_OPERATIONAL" if all(checks.values()) else "RESEARCH_NOT_OPERATIONAL",
        "started_at": started,
        "finished_at": time.time(),
        "workspace": str(root),
        "repo_root": str(repository),
        "checks": checks,
        "calls": calls,
        "sources": source_provenance,
        "auth": auth,
        "resolved_revision": revision,
        "download_destination": download_destination,
        "claim_boundary": "This gate proves bounded public research and metadata acquisition through typed tools; it does not authorize external writes or model-weight acquisition.",
        "next_action": "feed source-backed hypotheses to Doctor/VMCP and validate them with protected local measurements",
    }
    report["credentials_secret_free"] = _credential_free(report)
    if not report["credentials_secret_free"]:
        report["status"] = "FAILED"
        report["qualification"] = "RESEARCH_NOT_OPERATIONAL"
        report["blocker"] = "credential-shaped data appeared in the research receipt"
    receipt = root / ".hcli" / "receipts" / "research-gate.json"
    report["receipt_path"] = str(receipt)
    payload = causality_payload(report)
    record_research_causality(report, **payload)
    atomic_write_json(receipt, report)
    if emit:
        destination = Path(emit).expanduser().resolve()
        atomic_write_json(destination, report)
        report["emit_path"] = str(destination)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--repo-root")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--search-query", default=DEFAULT_SEARCH)
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=12.0)
    args = parser.parse_args(argv)
    report = run_research_gate(
        args.workspace,
        repo_root=args.repo_root,
        repo=args.repo,
        search_query=args.search_query,
        emit=args.emit,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["SCHEMA", "causality_payload", "record_research_causality", "records_five_fields", "run_research_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
