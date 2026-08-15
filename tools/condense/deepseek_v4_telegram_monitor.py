#!/usr/bin/env python3.12
"""Detached, deduplicated Telegram watcher for DeepSeek and Proto-Frankenstein.

Reads only the local stream manifest and supervisor receipt.  Credentials stay
in the existing macOS Keychain services; no secret is accepted on the command
line, written to state, or included in errors.  A status change is delivered at
most once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity"
MANIFEST = ARTIFACT / "manifest.json"
SUPERVISOR_RECEIPT = ROOT / "receipts/deepseek_v4_resource_supervisor_launchd.json"
STATE = ROOT / "reports/condense/deepseek_v4_telegram_monitor/state.json"
FRANK_DESKTOP = Path.home() / "Desktop" / "hawking-frankenstein"
FRANK_ACTIVE = FRANK_DESKTOP / "ACTIVE_STATUS.json"
FRANK_BRIDGE_ACTIVE = FRANK_DESKTOP / "BRIDGE_ACTIVE_STATUS.json"
FRANK_BRIDGE_RUN = (
    FRANK_DESKTOP
    / "proto-frankenstein"
    / "bridge-train-real"
    / "BRIDGE_TRAIN_REAL_FIRST_RUN.json"
)
FRANK_TERMINAL = (
    FRANK_DESKTOP / "proto-frankenstein" / "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
)
TOKEN_SERVICE = "com.hawking.doctorv5.telegram.bot-token"
CHAT_SERVICE = "com.hawking.doctorv5.telegram.chat-id"
KEYCHAIN_ACCOUNT = "hawking"
MAX_JSON_BYTES = 128 * 1024 * 1024


class MonitorError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_JSON_BYTES:
        raise MonitorError("status input exceeds the monitor safety bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MonitorError("status input is not a JSON object")
    return value


def _keychain_value(service: str) -> str | None:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service, "-w"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def snapshot() -> dict[str, Any]:
    manifest = _read_json(MANIFEST)
    receipt = _read_json(SUPERVISOR_RECEIPT)
    frank_active = _read_json(FRANK_ACTIVE)
    frank_bridge_active = _read_json(FRANK_BRIDGE_ACTIVE)
    frank_bridge_run = _read_json(FRANK_BRIDGE_RUN)
    frank_terminal = _read_json(FRANK_TERMINAL)
    status = manifest.get("status") if manifest else "STREAM_NOT_SEALED"
    artifact = manifest.get("artifact") if manifest else {}
    if not isinstance(artifact, dict):
        artifact = {}
    return {
        "schema": "hawking.ascension.telegram_monitor.v2",
        "status": status,
        "manifest_present": manifest is not None,
        "manifest_created_at": manifest.get("created_at") if manifest else None,
        "source_bytes": artifact.get("source_index_total_size_bytes"),
        "chunk_count": artifact.get("content_addressed_chunk_count"),
        "supervisor_status": receipt.get("status") if receipt else None,
        "supervisor_stop_reason": receipt.get("stop_reason") if receipt else None,
        "frankenstein": {
            "operator_state": frank_active.get("state") if frank_active else "NO_STATUS",
            "bridge_guard_state": (
                frank_bridge_active.get("state") if frank_bridge_active else "NO_STATUS"
            ),
            "bridge_status": (
                frank_bridge_run.get("status") if frank_bridge_run else "NOT_RUN"
            ),
            "bridge_promotion": (
                frank_bridge_run.get("promotion_verdict")
                if frank_bridge_run
                else "NOT_RUN"
            ),
            "terminal_status": (
                frank_terminal.get("status") if frank_terminal else "NOT_SEALED"
            ),
            "terminal_endpoint": (
                frank_terminal.get("endpoint") if frank_terminal else None
            ),
        },
    }


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _message(value: dict[str, Any]) -> str:
    size = value.get("source_bytes")
    size_text = f"{int(size) / 1_000_000_000:.1f} GB" if isinstance(size, int) else "unknown size"
    frank = value.get("frankenstein") or {}
    return (
        "Hawking Ascension update\n"
        f"DeepSeek: {value['status']}\n"
        f"Artifact: {size_text}, {value.get('chunk_count', 'unknown')} chunks\n"
        f"Proto operator: {frank.get('operator_state', 'unknown')}\n"
        f"Bridge: {frank.get('bridge_status', 'unknown')} / "
        f"{frank.get('bridge_promotion', 'unknown')}\n"
        f"Proto terminal: {frank.get('terminal_status', 'NOT_SEALED')}"
    )


def _send(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id.isdigit() or len(text) > 4096:
        raise MonitorError("Telegram credentials or message shape are invalid")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode({"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "hawking-deepseek-v4-monitor/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read(2 * 1024 * 1024))
    except Exception as exc:
        raise MonitorError(f"Telegram delivery failed: {type(exc).__name__}") from None
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise MonitorError("Telegram delivery was refused")


def _write_state(value: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".state.", dir=STATE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, STATE)
    finally:
        Path(temp).unlink(missing_ok=True)


def run() -> dict[str, Any]:
    current = snapshot()
    digest = _digest(current)
    previous = _read_json(STATE) or {}
    if previous.get("digest") == digest:
        return {"ok": True, "sent": False, "reason": "unchanged", "status": current["status"]}
    token = _keychain_value(TOKEN_SERVICE)
    chat_id = _keychain_value(CHAT_SERVICE)
    if token is None or chat_id is None:
        raise MonitorError("Telegram Keychain credentials are not armed")
    _send(token, chat_id, _message(current))
    _write_state({"digest": digest, "snapshot": current})
    return {"ok": True, "sent": True, "status": current["status"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "run"))
    args = parser.parse_args(argv)
    try:
        result = snapshot() if args.command == "status" else run()
    except (MonitorError, OSError, json.JSONDecodeError) as exc:
        print(f"DEEPSEEK_V4_TELEGRAM_MONITOR_REFUSED: {exc}", file=sys.stderr)
        return 75
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
