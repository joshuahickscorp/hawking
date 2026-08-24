"""Compile backend evidence. Context is a cache; raw traces stay on disk."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)"
)
_CMD_RE = re.compile(r"^\$\s+(.+)$", re.MULTILINE)
_BACKTICK_CMD = re.compile(r"`([^`]+)`")
_ERROR_RE = re.compile(r"(?im)^(error|failed|traceback|exception)[:\s].+$")

PROVENANCE_KEYS = frozenset(
    {"raw_report_path", "evidence_refs", "compact_path", "workspace"}
)
_TINY = 256
_SUMMARY_LIMIT = 400


def payload_dumps(compact: Dict[str, Any]) -> str:
    """Context-facing bytes: signal only, no provenance, no empty schema."""
    # Passthrough: envelope stays on the object; measure the body only.
    if compact.get("passthrough"):
        return json.dumps(
            {"final_summary": compact.get("final_summary") or ""},
            sort_keys=True,
            separators=(",", ":"),
        )
    payload = {
        k: v
        for k, v in compact.items()
        if k not in PROVENANCE_KEYS
        and k != "passthrough"
        and v not in (None, [], "", {})
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compile_backend_report(
    *,
    backend: str,
    task_id: str,
    raw_text: str = "",
    raw_report_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    extra = extra if isinstance(extra, dict) else {}
    text = _body(raw_text or "")
    summary = _summary(text)
    files = list(dict.fromkeys(_FILE_RE.findall(text)))[:24]
    commands = list(dict.fromkeys(_CMD_RE.findall(text)))[:24]
    if not commands:
        commands = [
            m.strip()
            for m in _BACKTICK_CMD.findall(text)
            if any(tok in m for tok in ("python", "pytest", "cargo", "grok", "hcli"))
        ][:12]
    found_errors = [m.group(0).strip() for m in _ERROR_RE.finditer(text)][:12]
    if errors:
        found_errors = list(errors) + found_errors
    claims = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")) and len(stripped) < 240:
            claims.append(stripped[2:].strip())
        if len(claims) >= 8:
            break
    compact = {
        "backend": backend,
        "task_id": task_id,
        "final_summary": summary,
        "claims": claims,
        "evidence_refs": [],  # do not duplicate raw_report_path
        "files_touched": files,
        "commands_executed": commands,
        "verifier_inputs": list(extra.get("verifier_inputs") or []),
        "errors": found_errors,
        "raw_report_path": raw_report_path,
        "mode": extra.get("mode"),
        "status": extra.get("status"),
        "passthrough": False,
    }
    encoded = payload_dumps(compact)
    raw_len = len(raw_text or "")
    if raw_len and (len(encoded) >= raw_len) and raw_len < _TINY:
        # Passthrough is a SIZE decision, not a licence to dump the raw trace.
        # Assigning raw_text verbatim here leaked reasoning blocks and tool
        # payloads (<think>..., {"tool": "shell", "cmd": "cat /etc/passwd"})
        # straight into the compact record for any report under _TINY bytes.
        # Keep the same line filter, just without the summary's early stop at
        # the first blank line.
        compact["final_summary"] = _summary(raw_text, limit=_SUMMARY_LIMIT, whole=True)
        compact["passthrough"] = True
    return compact


def _body(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return data["text"]
    return text


def _summary(text: str, limit: int = _SUMMARY_LIMIT, whole: bool = False) -> str:
    """Sanitized, de-duplicated prose from a backend report.

    ``whole=True`` keeps scanning past the first blank line instead of stopping
    at the leading block. The line filter is identical either way: reasoning
    blocks and tool-call payloads never survive, whichever mode is used.
    """
    if not text or not text.strip():
        return "(no report yet)"
    lines: List[str] = []
    prev = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and not whole:
                break
            continue
        lower = stripped.lower()
        if lower.startswith(("<think", "tool call", "function call")):
            continue
        if stripped.startswith("{") and "tool" in stripped[:40]:
            continue
        if stripped == prev:
            continue  # consecutive noise does not occupy the summary budget
        prev = stripped
        lines.append(stripped)
        if sum(len(item) for item in lines) >= limit:
            break
    blob = " ".join(lines).strip()
    if len(blob) > limit:
        blob = blob[: limit - 3].rstrip() + "..."
    return blob or "(empty after compaction)"
