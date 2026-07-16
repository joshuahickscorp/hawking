#!/usr/bin/env python3.12
"""Idempotent Telegram notifications for Doctor V5 model/rate rungs.

The notifier is deliberately outside the live queue.  It reads only compact
campaign/result JSON plus cheap host health probes.  It never reads model
payloads, mutates Doctor state, or treats a Telegram delivery as evidence.

Credentials are retrieved from macOS Keychain and never accepted as notifier or
launchd command arguments.  ``configure-token`` prompts without echo.  After the
user sends the new bot one Telegram message, ``discover-chat`` stores the single
private chat ID in Keychain.  No secret is written to the repository, plist,
log, or state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import re
import secrets
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "reports/condense/doctor_v5_ultra/campaign.json"
OBSERVER = ROOT / "reports/condense/doctor_v5_ultra/post_120b/observer_state.json"
RESULTS = ROOT / "reports/condense/doctor_v5_ultra/results"
DISPOSITIONS = ROOT / "reports/condense/doctor_v5_ultra/dispositions"
OUTPUT_ROOT = ROOT / "reports/condense/doctor_v5_unbound/telegram_notifier"
STATE = OUTPUT_ROOT / "state.json"
LOCK = OUTPUT_ROOT / "notifier.lock"
LOG = OUTPUT_ROOT / "notifier.log"
ERROR_LOG = OUTPUT_ROOT / "notifier.error.log"
PLIST = Path.home() / "Library/LaunchAgents/com.hawking.doctorv5.telegram.plist"
LABEL = "com.hawking.doctorv5.telegram"
TOKEN_SERVICE = "com.hawking.doctorv5.telegram.bot-token"
CHAT_SERVICE = "com.hawking.doctorv5.telegram.chat-id"
KEYCHAIN_ACCOUNT = "hawking"
STATE_SCHEMA = "hawking.doctor_v5_telegram_notifier_state.v1"
CAMPAIGN_SCHEMA = "hawking.doctor_v5_ultra_campaign.v1"
RESULT_SCHEMA = "hawking.doctor_v5_adapter_result.v1"
DISPOSITION_SCHEMA = "hawking.doctor_v5_ultra_disposition.v1"
TARGET_RATES = ("4", "3", "2", "1")
BRANCHES = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")
TERMINAL_STATUSES = frozenset({"complete", "negative", "unsupported"})
BRANCH_LABELS = {
    "codec_control": "codec",
    "doctor_static": "static",
    "doctor_conditional": "conditional",
    "doctor_full": "full",
}
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_MESSAGE_CHARS = 4000
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RESULT_KEYS = {
    "schema", "policy_version", "completed_at", "request_sha256", "adapter",
    "status", "output_artifacts", "metrics", "evidence_class",
    "quality_claims_permitted", "source_deletion_permitted", "result_sha256",
}
DISPOSITION_KEYS = {
    "schema", "version", "plan_sha256", "cell_id", "cell_identity_sha256",
    "status", "reason_code", "detail", "evidence_artifacts", "recorded_at",
    "quality_claims_permitted", "source_deletion_permitted", "disposition_sha256",
}
# Hawking's existing competitive criterion: no more than +8% PPL versus the
# exact source baseline.  This is deliberately stricter than a generic
# "usable" quantization threshold because rung notifications drive promotion.
GOOD_PPL_DELTA_MAX = 0.08
GOOD_CAPABILITY_DELTA_MIN = -0.05


class NotifierError(RuntimeError):
    """Notification configuration or a compact campaign input is invalid."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sealed(value: dict[str, Any], field: str) -> bool:
    return value.get(field) == _hash_value({k: v for k, v in value.items() if k != field})


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _workspace_file(raw: Any, *, maximum_bytes: int = MAX_JSON_BYTES) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise NotifierError("workspace evidence path is invalid")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_symlink():
        raise NotifierError(f"symlink evidence input is forbidden: {candidate}")
    try:
        path = candidate.resolve(strict=True)
        path.relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise NotifierError(f"evidence input escapes or is absent: {candidate}") from exc
    if not path.is_file() or path.stat().st_size > maximum_bytes:
        raise NotifierError(f"unsafe or oversized evidence input: {path}")
    return path


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise NotifierError(f"unsafe or oversized JSON input: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotifierError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NotifierError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _keychain_get(service: str) -> str | None:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
         "-s", service, "-w"],
        text=True, capture_output=True, check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _keychain_set(service: str, value: str) -> None:
    if not value or "\n" in value or "\r" in value:
        raise NotifierError("refusing empty or multiline Keychain value")
    result = subprocess.run(
        ["/usr/bin/security", "add-generic-password", "-U", "-a",
         KEYCHAIN_ACCOUNT, "-s", service, "-w", value],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise NotifierError("macOS Keychain rejected the credential")


def _telegram(token: str, method: str, payload: dict[str, Any] | None = None) -> Any:
    if not re.fullmatch(r"[0-9]{6,16}:[A-Za-z0-9_-]{20,}", token):
        raise NotifierError("Telegram bot token shape is invalid")
    data = urllib.parse.urlencode(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "hawking-doctor-v5-notifier/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(2 * 1024 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Do not stringify the request or URL: it contains the token.
        raise NotifierError(f"Telegram {method} request failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NotifierError(f"Telegram {method} returned invalid JSON") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        description = value.get("description") if isinstance(value, dict) else None
        raise NotifierError(f"Telegram {method} refused: {description or 'unknown error'}")
    return value.get("result")


def configure_token() -> dict[str, Any]:
    token = getpass.getpass("Paste BotFather token (input hidden): ").strip()
    if not re.fullmatch(r"[0-9]{6,16}:[A-Za-z0-9_-]{20,}", token):
        raise NotifierError("BotFather token shape is invalid")
    identity = _telegram(token, "getMe")
    _keychain_set(TOKEN_SERVICE, token)
    return {"configured": True, "bot_username": identity.get("username"),
            "token_stored_in": "macOS Keychain"}


def discover_chat() -> dict[str, Any]:
    token = _keychain_get(TOKEN_SERVICE)
    if not token:
        raise NotifierError("bot token is not configured")
    updates = _telegram(token, "getUpdates", {"timeout": 0, "allowed_updates": '["message"]'})
    chats: dict[str, dict[str, Any]] = {}
    for update in updates if isinstance(updates, list) else []:
        message = update.get("message") if isinstance(update, dict) else None
        chat = message.get("chat") if isinstance(message, dict) else None
        if isinstance(chat, dict) and chat.get("type") == "private" \
                and isinstance(chat.get("id"), int):
            chats[str(chat["id"])] = chat
    if not chats:
        raise NotifierError("no private chat found; send the bot a message, then retry")
    if len(chats) != 1:
        raise NotifierError("multiple private chats found; automatic selection is unsafe")
    chat_id, chat = next(iter(chats.items()))
    _keychain_set(CHAT_SERVICE, chat_id)
    return {"configured": True, "chat_type": "private",
            "chat_name": chat.get("first_name") or chat.get("username") or "private chat",
            "chat_id_stored_in": "macOS Keychain"}


def _state() -> dict[str, Any]:
    if not STATE.exists():
        value = {"schema": STATE_SCHEMA, "created_at": _now(), "updated_at": _now(),
                 "primed": False, "delivered": {}, "important_events": {},
                 "state_sha256": ""}
        value["state_sha256"] = _hash_value({k: v for k, v in value.items()
                                              if k != "state_sha256"})
        return value
    value = _read_json(STATE)
    if value.get("schema") != STATE_SCHEMA or not _sealed(value, "state_sha256"):
        raise NotifierError("notifier state identity is invalid")
    return value


def _save_state(value: dict[str, Any]) -> None:
    value = dict(value)
    value["updated_at"] = _now()
    value.pop("state_sha256", None)
    value["state_sha256"] = _hash_value(value)
    _atomic_json(STATE, value)


def _cell_map(campaign: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    cells = campaign.get("cells")
    if not isinstance(cells, list) or len(cells) != 320:
        raise NotifierError("campaign does not contain the exact 320 cells")
    return {
        (str(cell["model_label"]), str(cell["rate_id"]), str(cell["branch"])): cell
        for cell in cells
    }


def _validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema") != CAMPAIGN_SCHEMA \
            or not isinstance(campaign.get("version"), str) \
            or not _is_sha(campaign.get("plan_sha256")) \
            or campaign.get("source_deletion_permitted") is not False \
            or not _sealed(campaign, "campaign_sha256"):
        raise NotifierError("campaign identity or safety boundary is invalid")
    _cell_map(campaign)


def _terminal_rung_candidates(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    cells = _cell_map(campaign)
    labels = sorted({key[0] for key in cells}, key=lambda value: float(value.rstrip("BT")))
    rungs: list[dict[str, Any]] = []
    for label in labels:
        for rate in TARGET_RATES:
            rows = [cells.get((label, rate, branch)) for branch in BRANCHES]
            if all(isinstance(row, dict)
                   and row.get("status") in TERMINAL_STATUSES for row in rows):
                event_id = f"rung/{label}/{rate}bpw"
                rungs.append({"event_id": event_id, "model_label": label,
                              "rate_id": rate, "cells": rows})
    return rungs


def _validated_result(cell: dict[str, Any]) -> dict[str, Any]:
    result_paths = cell.get("result_paths")
    declared = result_paths.get("result") if isinstance(result_paths, dict) else None
    path = _workspace_file(declared)
    expected = (RESULTS / str(cell.get("cell_id")) / "result.json").resolve(strict=False)
    if path != expected:
        raise NotifierError(f"result path binding is invalid for {cell.get('cell_id')}")
    result = _read_json(path)
    metrics = result.get("metrics")
    adapter = result.get("adapter")
    campaign_cell = metrics.get("campaign_cell") if isinstance(metrics, dict) else None
    exact_binding = {
        field: cell.get(field)
        for field in ("branch", "cell_id", "cell_identity_sha256", "model_label", "rate_id")
    }
    if set(result) != RESULT_KEYS or result.get("schema") != RESULT_SCHEMA \
            or result.get("status") != "complete" \
            or result.get("result_sha256") != cell.get("result_sha256") \
            or not _is_sha(result.get("result_sha256")) \
            or not _sealed(result, "result_sha256") \
            or result.get("request_sha256") != cell.get("request_sha256") \
            or not isinstance(adapter, dict) \
            or adapter.get("adapter_id") != cell.get("adapter_id") \
            or campaign_cell != exact_binding \
            or result.get("evidence_class") != "provisional_engineering_evidence" \
            or result.get("quality_claims_permitted") is not False \
            or result.get("source_deletion_permitted") is not False:
        raise NotifierError(f"result evidence is invalid for {cell.get('cell_id')}")
    return result


def _validated_disposition(cell: dict[str, Any],
                           campaign: dict[str, Any]) -> dict[str, Any]:
    path = _workspace_file(cell.get("disposition_path"))
    expected = (DISPOSITIONS / f"{cell.get('cell_id')}.json").resolve(strict=False)
    if path != expected:
        raise NotifierError(f"disposition path binding is invalid for {cell.get('cell_id')}")
    disposition = _read_json(path)
    if set(disposition) != DISPOSITION_KEYS \
            or disposition.get("schema") != DISPOSITION_SCHEMA \
            or disposition.get("version") != campaign.get("version") \
            or disposition.get("plan_sha256") != campaign.get("plan_sha256") \
            or disposition.get("cell_id") != cell.get("cell_id") \
            or disposition.get("cell_identity_sha256") != cell.get("cell_identity_sha256") \
            or disposition.get("status") != cell.get("status") \
            or disposition.get("status") not in {"negative", "unsupported"} \
            or disposition.get("disposition_sha256") != cell.get("disposition_sha256") \
            or not _is_sha(disposition.get("disposition_sha256")) \
            or not _sealed(disposition, "disposition_sha256") \
            or not isinstance(disposition.get("reason_code"), str) \
            or not disposition["reason_code"] \
            or not isinstance(disposition.get("detail"), str) \
            or not disposition["detail"] \
            or disposition.get("quality_claims_permitted") is not False \
            or disposition.get("source_deletion_permitted") is not False:
        raise NotifierError(f"disposition evidence is invalid for {cell.get('cell_id')}")
    artifacts = disposition.get("evidence_artifacts")
    if not isinstance(artifacts, list):
        raise NotifierError(f"disposition evidence list is invalid for {cell.get('cell_id')}")
    roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) \
                or set(artifact) != {"role", "path", "sha256", "bytes"} \
                or not isinstance(artifact.get("role"), str) \
                or not artifact["role"] or artifact["role"] in roles \
                or not _is_sha(artifact.get("sha256")) \
                or isinstance(artifact.get("bytes"), bool) \
                or not isinstance(artifact.get("bytes"), int) \
                or artifact["bytes"] < 0:
            raise NotifierError(
                f"disposition evidence artifact {index} is invalid for {cell.get('cell_id')}"
            )
        roles.add(artifact["role"])
        artifact_path = _workspace_file(artifact["path"])
        digest, size = _sha_file(artifact_path)
        if digest != artifact["sha256"] or size != artifact["bytes"]:
            raise NotifierError(
                f"disposition evidence artifact {index} changed for {cell.get('cell_id')}"
            )
    return disposition


def _result_metrics(cell: dict[str, Any],
                    result: dict[str, Any] | None = None) -> dict[str, Any]:
    result = result if isinstance(result, dict) else _validated_result(cell)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    physical = metrics.get("physical_accounting") \
        if isinstance(metrics.get("physical_accounting"), dict) else {}
    quality = metrics.get("quality_observation")
    if not isinstance(quality, dict):
        outcome = metrics.get("treatment_outcome")
        quality = outcome.get("quality_observation") if isinstance(outcome, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    ppl = quality.get("ppl") if isinstance(quality.get("ppl"), dict) else {}
    capability = quality.get("capability") \
        if isinstance(quality.get("capability"), dict) else {}
    started, completed = cell.get("started_at"), cell.get("completed_at")
    wall = None
    if isinstance(started, str) and isinstance(completed, str):
        wall = (dt.datetime.fromisoformat(completed)
                - dt.datetime.fromisoformat(started)).total_seconds()
    return {
        "actual_bpw": physical.get("all_in_model_payload_bpw"),
        "target_bpw": physical.get("target_physical_bpw"),
        "physical_target_met": physical.get("target_met"),
        "ppl_delta": ppl.get("relative_delta"),
        "capability_delta": capability.get("absolute_delta"),
        "wall_seconds": wall,
        "attempts": cell.get("attempts"),
        "quality_status": quality.get("status"),
    }


def _validated_terminal_rung(rung: dict[str, Any],
                             campaign: dict[str, Any]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    dispositions: dict[str, dict[str, Any]] = {}
    for cell in rung["cells"]:
        branch = str(cell["branch"])
        status = str(cell["status"])
        row = {"cell_id": cell["cell_id"], "branch": branch, "status": status}
        if status == "complete":
            result = _validated_result(cell)
            row["result_sha256"] = result["result_sha256"]
            metrics[branch] = _result_metrics(cell, result)
        else:
            disposition = _validated_disposition(cell, campaign)
            row["disposition_sha256"] = disposition["disposition_sha256"]
            dispositions[branch] = disposition
        evidence.append(row)
    return {
        **rung,
        "metrics": metrics,
        "dispositions": dispositions,
        "evidence_root_sha256": _hash_value(evidence),
    }


def complete_rungs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evidence-validated target rungs closed by any terminal status."""
    _validate_campaign(campaign)
    return [
        _validated_terminal_rung(rung, campaign)
        for rung in _terminal_rung_candidates(campaign)
    ]


def _fmt(value: Any, *, percent: bool = False, signed: bool = False) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if percent:
        return f"{value * 100:+.1f}%"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _pareto_structure(
        eligible: list[tuple[str, dict[str, Any]]]) -> dict[str, list[str]]:
    """Separate result-efficient branches from branches dominated on this rung."""
    active: list[str] = []
    dominated: list[str] = []
    for branch, row in eligible:
        is_dominated = False
        for other_branch, other in eligible:
            if other_branch == branch:
                continue
            no_worse = (
                float(other["actual_bpw"]) <= float(row["actual_bpw"])
                and float(other["ppl_delta"]) <= float(row["ppl_delta"])
                and float(other["capability_delta"]) >= float(row["capability_delta"])
            )
            strictly_better = (
                float(other["actual_bpw"]) < float(row["actual_bpw"])
                or float(other["ppl_delta"]) < float(row["ppl_delta"])
                or float(other["capability_delta"]) > float(row["capability_delta"])
            )
            if no_worse and strictly_better:
                is_dominated = True
                break
        (dominated if is_dominated else active).append(branch)
    return {"active": active, "dominated": dominated}


def _rung_decision(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the physical-first condenser gate to measured branches only."""
    eligible: list[tuple[str, dict[str, Any]]] = []
    for branch in BRANCHES:
        row = metrics.get(branch)
        if not isinstance(row, dict):
            continue
        actual, target = row.get("actual_bpw"), row.get("target_bpw")
        ppl, capability = row.get("ppl_delta"), row.get("capability_delta")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value))
               for value in (actual, target, ppl, capability)):
            eligible.append((branch, row))
    if not eligible:
        return {"result": "UNAVAILABLE", "optimization_possible": False,
                "optimization_scope": "none", "evidence_value": "INCOMPLETE",
                "pareto_active": [], "dominated_branches": [],
                "best_branch": None, "best": None,
                "reason": "no completed branch has the required physical and quality metrics"}
    structure = _pareto_structure(eligible)
    quality_passes = [
        item for item in eligible
        if float(item[1]["ppl_delta"]) <= GOOD_PPL_DELTA_MAX
        and float(item[1]["capability_delta"]) >= GOOD_CAPABILITY_DELTA_MIN
    ]
    pool = quality_passes or eligible
    branch, best = min(pool, key=lambda item: (
        float(item[1]["actual_bpw"]), float(item[1]["ppl_delta"]),
        -float(item[1]["capability_delta"]), BRANCHES.index(item[0]),
    ))
    quality_good = (float(best["ppl_delta"]) <= GOOD_PPL_DELTA_MAX
                    and float(best["capability_delta"]) >= GOOD_CAPABILITY_DELTA_MIN)
    physical_good = float(best["actual_bpw"]) <= float(best["target_bpw"])
    good = quality_good and physical_good
    if good:
        reason = "physical and quality gates pass; the next lower rung is testable"
    elif quality_good:
        gap = float(best["actual_bpw"]) - float(best["target_bpw"])
        reason = f"quality passes; physical target misses by +{gap:.3f} bpw"
    elif physical_good:
        reason = (f"physical target passes; PPL {_fmt(best['ppl_delta'], percent=True)} "
                  f"vs +{GOOD_PPL_DELTA_MAX * 100:.1f}% limit")
    else:
        gap = float(best["actual_bpw"]) - float(best["target_bpw"])
        reason = (f"both gates miss: physical +{gap:.3f} bpw; "
                  f"PPL {_fmt(best['ppl_delta'], percent=True)} "
                  f"vs +{GOOD_PPL_DELTA_MAX * 100:.1f}% limit")
    model_headroom = quality_good or physical_good
    speed_headroom = bool(structure["dominated"])
    if model_headroom and speed_headroom:
        optimization_scope = "model + speed"
    elif model_headroom:
        optimization_scope = "model"
    elif speed_headroom:
        optimization_scope = "speed only"
    else:
        optimization_scope = "none"
    return {"result": "GOOD" if good else "BAD",
            "optimization_possible": optimization_scope != "none",
            "optimization_scope": optimization_scope,
            "evidence_value": "PROMOTABLE" if good else "USEFUL NEGATIVE",
            "pareto_active": structure["active"],
            "dominated_branches": structure["dominated"],
            "best_branch": branch, "best": best, "reason": reason}


def _health() -> dict[str, Any]:
    def command(argv: list[str]) -> str:
        result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=8)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    pressure = command(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    swap = command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap)
    swap_mb = None
    if match:
        swap_mb = float(match.group(1)) * (1024 if match.group(2) == "G" else 1)
    thermal = command(["/usr/bin/pmset", "-g", "therm"])
    stat = os.statvfs(ROOT)
    return {
        "pressure": int(pressure) if pressure.isdigit() else None,
        "swap_mb": swap_mb,
        "thermal_green": "No thermal warning level has been recorded" in thermal,
        "disk_free_gb": stat.f_bavail * stat.f_frsize / 1_000_000_000,
    }


def _progress(campaign: dict[str, Any]) -> dict[str, Any]:
    cells = campaign["cells"]
    complete = [cell for cell in cells if cell.get("status") == "complete"]
    terminal = [cell for cell in cells if cell.get("status") in TERMINAL_STATUSES]
    total_passes = sum(int(cell["exact_stored_parameter_count"]) for cell in cells)
    done_passes = sum(int(cell["exact_stored_parameter_count"]) for cell in complete)
    terminal_passes = sum(int(cell["exact_stored_parameter_count"]) for cell in terminal)
    return {"complete": len(complete), "terminal": len(terminal), "total": len(cells),
            "weighted_complete": done_passes / total_passes,
            "weighted_terminal": terminal_passes / total_passes}


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _next_rung(campaign: dict[str, Any], observer: dict[str, Any] | None
               ) -> dict[str, Any] | None:
    """Return the next active block, or the first unfinished target rung."""
    cells = _cell_map(campaign)
    rates = ((observer or {}).get("eta") or {}).get(
        "branch_rate_seconds_per_billion", {})
    fallback = ((observer or {}).get("eta") or {}).get(
        "branch_seconds_per_billion", {})
    rates = rates if isinstance(rates, dict) else {}
    fallback = fallback if isinstance(fallback, dict) else {}

    def estimate(row: dict[str, Any]) -> float | None:
        branch, rate = str(row["branch"]), str(row["rate_id"])
        per_billion = rates.get(f"{branch}@{rate}", fallback.get(branch))
        parameters = row.get("exact_stored_parameter_count")
        if not isinstance(per_billion, (int, float)) \
                or not isinstance(parameters, (int, float)):
            return None
        seconds = float(per_billion) * float(parameters) / 1_000_000_000
        if row.get("status") == "running" and isinstance(row.get("started_at"), str):
            try:
                elapsed = (dt.datetime.now(dt.timezone.utc)
                           - dt.datetime.fromisoformat(row["started_at"])).total_seconds()
                seconds = max(0.0, seconds - elapsed)
            except ValueError:
                pass
        return seconds

    running = [row for row in cells.values() if row.get("status") == "running"
               and str(row.get("rate_id")) in TARGET_RATES]
    if running:
        ranked = sorted(
            ((estimate(row), row) for row in running),
            key=lambda item: (math.inf if item[0] is None else item[0],
                              float(str(item[1]["model_label"]).rstrip("BT"))),
        )
        seconds, row = ranked[0]
        return {"model_label": row["model_label"], "rate_id": str(row["rate_id"]),
                "branch": str(row["branch"]), "scope": "block",
                "remaining_seconds": seconds}

    labels = sorted({key[0] for key in cells}, key=lambda value: float(value.rstrip("BT")))
    for label in labels:
        for rate in TARGET_RATES:
            rows = [cells.get((label, rate, branch)) for branch in BRANCHES]
            if not all(isinstance(row, dict) for row in rows):
                continue
            remaining = [
                row for row in rows if row.get("status") not in TERMINAL_STATUSES
            ]
            if not remaining:
                continue
            seconds = 0.0
            for row in remaining:
                row_seconds = estimate(row)
                if row_seconds is None:
                    seconds = math.nan
                    break
                seconds += row_seconds
            return {"model_label": label, "rate_id": rate,
                    "branch": None, "scope": "rung",
                    "remaining_seconds": seconds if math.isfinite(seconds) else None}
    return None


def _eta_block(campaign: dict[str, Any], observer: dict[str, Any] | None) -> list[str]:
    next_rung = _next_rung(campaign, observer)
    if next_rung is None:
        next_line = "Next block: none remaining"
    elif isinstance(next_rung["remaining_seconds"], (int, float)):
        branch = (f" {BRANCH_LABELS[next_rung['branch']]}"
                  if next_rung.get("branch") in BRANCH_LABELS else "")
        noun = "block" if next_rung.get("scope") == "block" else "rung"
        next_line = (f"Next {noun}: {next_rung['model_label']} @ "
                     f"{next_rung['rate_id']} bpw{branch} — "
                     f"~{_duration(next_rung['remaining_seconds'])} remaining")
    else:
        next_line = (f"Next block: {next_rung['model_label']} @ {next_rung['rate_id']} bpw"
                     " — ETA learning")

    boundary = ((observer or {}).get("eta") or {}).get("to_120b_boundary", {})
    point_at = boundary.get("point_at") if isinstance(boundary, dict) else None
    overall_line = "Overall sub-120B: ETA learning"
    if isinstance(point_at, str):
        try:
            point = dt.datetime.fromisoformat(point_at).astimezone()
            remaining = (point - dt.datetime.now(dt.timezone.utc).astimezone()).total_seconds()
            date = point.strftime("%b %d").replace(" 0", " ")
            clock = point.strftime("%I:%M %p").lstrip("0")
            zone = point.tzname() or "local"
            overall_line = (f"Overall sub-120B: {date}, {clock} {zone}"
                            f" (~{_duration(remaining)} remaining, provisional)")
        except ValueError:
            pass
    return ["ETA", next_line, overall_line]


def format_rung(rung: dict[str, Any], campaign: dict[str, Any],
                observer: dict[str, Any] | None = None) -> str:
    metrics = rung.get("metrics")
    if not isinstance(metrics, dict):
        raise NotifierError("rung evidence was not validated before formatting")
    decision = _rung_decision(metrics)
    best = decision["best"]
    statuses = {str(cell["branch"]): str(cell["status"]) for cell in rung["cells"]}
    all_complete = all(status == "complete" for status in statuses.values())
    measured = [branch for branch in BRANCHES if statuses.get(branch) == "complete"]
    negative = [branch for branch in BRANCHES if statuses.get(branch) == "negative"]
    unsupported = [branch for branch in BRANCHES if statuses.get(branch) == "unsupported"]
    result_icon = {"GOOD": "✅", "BAD": "❌"}.get(decision["result"], "⚪")
    if decision["result"] == "UNAVAILABLE":
        optimization = "UNAVAILABLE"
    elif decision["optimization_possible"]:
        optimization = f"YES — {decision['optimization_scope'].upper()} ⚡"
    else:
        optimization = "NO"
    closure = "complete" if all_complete else "closed"
    result_label = "Result" if all_complete else "Measured result"
    optimization_label = (
        "Optimization possible"
        if all_complete else "Optimization possible from measured evidence"
    )
    evidence_value = decision["evidence_value"]
    if not measured:
        evidence_value = "DISPOSITION ONLY"
    elif not all_complete:
        evidence_value += " (measured branches only)"
    lines = [
        f"🏔 Hawking: {rung['model_label']} @ {rung['rate_id']} bpw {closure}",
        "",
        f"{result_label}: {decision['result']} {result_icon}",
        f"{optimization_label}: {optimization}",
        f"Evidence: {evidence_value}",
        f"Reason: {decision['reason']}",
    ]
    if not all_complete:
        coverage = [f"{len(measured)} measured"]
        if negative:
            coverage.append(f"{len(negative)} negative disposition")
        if unsupported:
            coverage.append(f"{len(unsupported)} adaptively deferred")
        lines.append("Coverage: " + " | ".join(coverage))
        if measured:
            lines.append(
                "Measured: " + ", ".join(BRANCH_LABELS[branch] for branch in measured)
            )
        if negative:
            lines.append(
                "Negative: " + ", ".join(BRANCH_LABELS[branch] for branch in negative)
            )
        if unsupported:
            lines.append(
                "Deferred: " + ", ".join(BRANCH_LABELS[branch] for branch in unsupported)
            )
        if negative or unsupported:
            lines.append("Disposition branches carry no quality or source-deletion claim.")
    active = decision["pareto_active"]
    dominated = decision["dominated_branches"]
    if active:
        branch_scope = "branches" if all_complete else "measured branches"
        speed = (
            f"{len(active)}/{len(active) + len(dominated)} "
            f"{branch_scope} Pareto-active"
        )
        if dominated:
            labels = ", ".join(BRANCH_LABELS[branch] for branch in dominated)
            speed += f"; dominated here: {labels}"
        else:
            speed += "; no empirical pruning signal"
        lines.append(f"Speed signal: {speed}")
    if isinstance(best, dict):
        wall = best["wall_seconds"]
        hours = f"{wall / 3600:.2f}h" if isinstance(wall, (int, float)) else "n/a"
        leader = "Best" if decision["result"] == "GOOD" else "Density leader"
        lines += [
            f"{leader}: {BRANCH_LABELS[decision['best_branch']]}",
            f"Actual {_fmt(best['actual_bpw'])}/{_fmt(best['target_bpw'])} bpw | "
            f"PPL {_fmt(best['ppl_delta'], percent=True)} | "
            f"capability {_fmt(best['capability_delta'], signed=True)}",
            f"Time {hours} | attempts {best['attempts']}",
        ]
    else:
        lines.append("Best: unavailable (quality metrics missing)")
    lines += [""] + _eta_block(campaign, observer)
    progress = _progress(campaign)
    health = _health()
    lines += [
        "",
        f"Progress: {progress['complete']}/{progress['total']} measured "
        f"({progress['weighted_complete'] * 100:.2f}% weighted) | "
        f"{progress['terminal']}/{progress['total']} terminal "
        f"({progress['weighted_terminal'] * 100:.2f}% weighted)",
        f"Host: pressure {health['pressure']} | swap {_fmt(health['swap_mb'])} MB | "
        f"disk {health['disk_free_gb']:.1f} GB | "
        f"thermal {'green' if health['thermal_green'] else 'warning'}",
    ]
    lines += ["", "Provisional until the signed physical release gate."]
    return "\n".join(lines)[:MAX_MESSAGE_CHARS]


def _send(text: str) -> dict[str, Any]:
    token = _keychain_get(TOKEN_SERVICE)
    chat_id = _keychain_get(CHAT_SERVICE)
    if not token or not chat_id:
        raise NotifierError("Telegram token/chat are not fully configured")
    result = _telegram(token, "sendMessage", {
        "chat_id": chat_id, "text": text,
        "disable_web_page_preview": "true",
    })
    if not isinstance(result, dict) or not isinstance(result.get("message_id"), int):
        raise NotifierError("Telegram sendMessage response lacks a message ID")
    return {"message_id": result["message_id"], "sent_at": _now()}


def prime() -> dict[str, Any]:
    campaign = _read_json(CAMPAIGN)
    state = _state()
    existing = complete_rungs(campaign)
    for rung in existing:
        state["delivered"].setdefault(rung["event_id"], {
            "status": "primed-existing-no-message",
            "evidence_root_sha256": rung["evidence_root_sha256"],
            "recorded_at": _now(),
        })
    state["primed"] = True
    _save_state(state)
    return {"primed": True, "existing_rungs_suppressed": len(existing)}


def run_once(*, sender: Any = _send) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already-running", "sent": 0}
        campaign = _read_json(CAMPAIGN)
        _validate_campaign(campaign)
        observer = _read_json(OBSERVER) if OBSERVER.exists() else None
        state = _state()
        if state.get("primed") is not True:
            raise NotifierError("notifier must be primed before delivery")
        sent = 0
        for candidate in _terminal_rung_candidates(campaign):
            if candidate["event_id"] in state["delivered"]:
                continue
            rung = _validated_terminal_rung(candidate, campaign)
            delivery = sender(format_rung(rung, campaign, observer))
            state["delivered"][rung["event_id"]] = {
                "status": "delivered",
                "evidence_root_sha256": rung["evidence_root_sha256"],
                **delivery,
            }
            _save_state(state)
            sent += 1
        blocked = campaign.get("counts", {}).get("blocked-execution", 0)
        queue_status = campaign.get("queue_status")
        important = None
        if blocked:
            important = f"blocked-execution/{blocked}"
        elif queue_status in ("error", "failed", "stopped"):
            important = f"queue-status/{queue_status}"
        if important and important not in state["important_events"]:
            delivery = sender(
                f"⚠️ Hawking important event: {important}\n"
                f"Coverage {campaign.get('counts', {}).get('complete', 'n/a')}/320."
            )
            state["important_events"][important] = delivery
            _save_state(state)
            sent += 1
        return {"status": "ok", "sent": sent,
                "known_rungs": len(state["delivered"])}


def install_launch_agent() -> dict[str, Any]:
    if not _keychain_get(TOKEN_SERVICE) or not _keychain_get(CHAT_SERVICE):
        raise NotifierError("configure token and chat before installing launchd")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    document = {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "run"],
        "WorkingDirectory": str(ROOT),
        "StartInterval": 300,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(ERROR_LOG),
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLIST.with_name(f".{PLIST.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            plistlib.dump(document, handle, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, PLIST)
    finally:
        temporary.unlink(missing_ok=True)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["/bin/launchctl", "bootout", domain, str(PLIST)],
                   capture_output=True, check=False)
    loaded = subprocess.run(["/bin/launchctl", "bootstrap", domain, str(PLIST)],
                            text=True, capture_output=True, check=False)
    if loaded.returncode != 0:
        raise NotifierError("launchd refused the Telegram notifier service")
    return {"installed": True, "label": LABEL, "interval_seconds": 300,
            "plist": str(PLIST), "credentials": "macOS Keychain"}


def status() -> dict[str, Any]:
    state = _state()
    campaign = _read_json(CAMPAIGN)
    _validate_campaign(campaign)
    terminal_rungs = _terminal_rung_candidates(campaign)
    return {
        "token_configured": _keychain_get(TOKEN_SERVICE) is not None,
        "chat_configured": _keychain_get(CHAT_SERVICE) is not None,
        "primed": state.get("primed") is True,
        "known_rungs": len(state.get("delivered", {})),
        "currently_complete_target_rungs": sum(
            all(cell.get("status") == "complete" for cell in rung["cells"])
            for rung in terminal_rungs
        ),
        "currently_terminal_target_rungs": len(terminal_rungs),
        "launch_agent_installed": PLIST.exists(),
        "target_rates": list(TARGET_RATES),
        "state": str(STATE),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "status", "configure-token", "discover-chat", "prime", "send-test",
        "install", "run",
    ))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status": result = status()
        elif args.command == "configure-token": result = configure_token()
        elif args.command == "discover-chat": result = discover_chat()
        elif args.command == "prime": result = prime()
        elif args.command == "send-test":
            result = _send("✅ Hawking Doctor V5 Telegram notifications are connected.")
        elif args.command == "install": result = install_launch_agent()
        else: result = run_once()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (NotifierError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
