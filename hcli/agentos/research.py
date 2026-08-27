"""Bounded public-research operational gate for AgentOS.

The gate exercises the same typed ToolRegistry that a mission uses.  It is
deliberately small and repeatable: public web search, an official source
fetch, GitHub repository inspection, Hugging Face revision/file access, and a
hash-checked download of one harmless metadata file.  It records provenance
and capability observations, never model content or credentials.
"""
from __future__ import annotations

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


__all__ = ["SCHEMA", "run_research_gate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
