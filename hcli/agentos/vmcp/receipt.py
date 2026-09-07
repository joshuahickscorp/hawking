"""E.4 ToolReceipt plus local-only subprocess helpers.

A truth-affecting tool with no trace is not part of the proof. Nothing in
this module talks to the network; subprocess env is stripped of proxies
and known network tools are refused before exec.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "hawking.vmcp.tool_receipt.v1"

NETWORK_BASENAMES = frozenset(
    {
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "sftp",
        "telnet",
        "nmap",
        "ftp",
        "rsync",
        "socat",
    }
)

_PROXY_KEYS = frozenset(
    {
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "ftp_proxy",
        "FTP_PROXY",
        "no_proxy",
        "NO_PROXY",
    }
)

_DANGEROUS = (
    ("rm", "-rf", "/"),
    ("rm", "-fr", "/"),
    ("rm", "-rf", "/*"),
    ("mkfs",),
    ("shutdown",),
    ("reboot",),
    ("halt",),
    ("diskutil", "erase"),
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def content_digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def argv_list(command: Any) -> list[str]:
    if command is None:
        return []
    if isinstance(command, str):
        parts = command.strip().split()
        return [p for p in parts if p]
    if isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
        return [str(x) for x in command if str(x) != ""]
    return [str(command)]


def basename_of(argv: Sequence[str]) -> str:
    if not argv:
        return ""
    return os.path.basename(str(argv[0]))


def network_tool_refused(argv: Sequence[str]) -> str | None:
    base = basename_of(argv).lower()
    if base in NETWORK_BASENAMES:
        return f"NETWORK_TOOL_REFUSED:{base}"
    return None


def dangerous_command(argv: Sequence[str]) -> str | None:
    if not argv:
        return None
    lowered = tuple(str(a).lower() for a in argv)
    for prefix in _DANGEROUS:
        if lowered[: len(prefix)] == prefix:
            return "DANGEROUS_COMMAND:" + " ".join(prefix)
    if lowered[:2] == ("kill", "-9"):
        return "DANGEROUS_COMMAND:kill -9"
    if lowered[:1] == ("dd",) and any("of=/dev/" in a for a in lowered):
        return "DANGEROUS_COMMAND:dd-of-dev"
    return None


def local_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in list(env):
        if key in _PROXY_KEYS:
            env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    env["GIT_MERGE_AUTOEDIT"] = "no"
    env["NO_NETWORK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def tool_receipt(
    *,
    tool: str,
    invocation: Sequence[str] | Mapping[str, Any],
    status: str,
    started_at: str | None = None,
    elapsed_ms: float | None = None,
    version: str | None = None,
    input_ids: Sequence[str] | None = None,
    input_hashes: Sequence[str] | None = None,
    output_ids: Sequence[str] | None = None,
    output_hashes: Sequence[str] | None = None,
    limitations: Sequence[str] | None = None,
    verifier: str | None = None,
    canary: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "tool": tool,
        "version": version,
        "invocation": list(invocation) if not isinstance(invocation, Mapping) else dict(invocation),
        "input_ids": list(input_ids or []),
        "input_hashes": list(input_hashes or []),
        "output_ids": list(output_ids or []),
        "output_hashes": list(output_hashes or []),
        "started_at": started_at or utc_now(),
        "elapsed_ms": None if elapsed_ms is None else float(elapsed_ms),
        "status": status,
        "limitations": list(limitations or []),
        "verifier": verifier,
        "canary": None if canary is None else dict(canary),
        "evidence_tier": "FUNCTIONAL_SIM",
        "gpu_authority": False,
        "network_used": False,
    }
    if extra:
        for key, value in extra.items():
            if key not in rec:
                rec[key] = value
    rec["receipt_sha256"] = content_digest({k: v for k, v in rec.items() if k != "receipt_sha256"})
    return rec
