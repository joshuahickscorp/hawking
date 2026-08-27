"""Bounded live residency proof for one profile-driven native resident.

The gate measures process/lifecycle properties only.  It does not score model
quality and it does not promote generated prose into a verified fact.  A
different native provider can pass the same gate by implementing the profile's
JSONL contract.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.resident_gate.v1"
DEFAULT_COUNT = 20
DEFAULT_PROMPTS = (
    ("factual", "Answer with one short sentence: what is the capital of France?"),
    ("arithmetic", "Compute 12 * 9. Reply with the integer and nothing else."),
    ("structured_json", 'Return exactly one JSON object: {"ok":true,"value":7}'),
    ("code", "Give one Python expression that reverses the string 'abc'."),
    ("tool_planning", "Name one read-only filesystem check a verifier could perform."),
)


def _profile_path(profile: Optional[str], repo_root: Optional[Path]) -> Path:
    value = profile or os.environ.get("HCLI_HAWKING_NATIVE_CONFIG")
    if value:
        return Path(value).expanduser().resolve()
    root = repo_root or Path(__file__).resolve().parents[2]
    return (root / "hcli" / "hawking-native.sealed-3.14.json").resolve()


def _text(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(choice.get("text"), str):
            return choice["text"]
    return raw.get("text") if isinstance(raw.get("text"), str) else None


def _response_summary(raw: Any) -> Dict[str, Any]:
    hawking = raw.get("hawking") if isinstance(raw, dict) else {}
    health = hawking.get("resident_health") if isinstance(hawking, dict) else {}
    text = _text(raw)
    return {
        "id": raw.get("id") if isinstance(raw, dict) else None,
        "text_len": len(text) if isinstance(text, str) else None,
        "text_preview": repr(text)[:240] if isinstance(text, str) else None,
        "generated_tokens": hawking.get("generated_tokens") if isinstance(hawking, dict) else None,
        "prompt_tokens": hawking.get("prompt_tokens") if isinstance(hawking, dict) else None,
        "fallbacks": hawking.get("fallbacks") if isinstance(hawking, dict) else None,
        "pid": health.get("pid") if isinstance(health, dict) else None,
        "model_open_count": health.get("model_open_count") if isinstance(health, dict) else None,
        "weight_upload_count": health.get("weight_upload_count") if isinstance(health, dict) else None,
        "restart_count": health.get("restart_count") if isinstance(health, dict) else None,
        "error": raw.get("error") if isinstance(raw, dict) else None,
    }


def _write_receipt(report: Dict[str, Any], emit: Optional[str], repo_root: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RESIDENT_GATE.json"
    if not destination.is_absolute():
        destination = repo_root / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_resident_gate(
    workspace: Optional[str | os.PathLike[str]] = None,
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str] = None,
    count: int = DEFAULT_COUNT,
    timeout_s: float = 180.0,
    model_tokens: int = 32,
    emit: Optional[str] = None,
) -> Dict[str, Any]:
    """Send sequential requests through one resident and persist the proof."""
    from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector

    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    work = Path(workspace).expanduser().resolve() if workspace else repo
    work.mkdir(parents=True, exist_ok=True)
    count = max(1, min(100, int(count)))
    timeout_s = max(0.1, float(timeout_s))
    model_tokens = max(1, int(model_tokens))
    profile_path = _profile_path(profile, repo)
    config = HawkingNativeConfig.from_file(str(profile_path))
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "LIVE_RESIDENT_SEQUENTIAL_PROOF",
        "started_at": time.time(),
        "workspace": str(work),
        "profile_path": str(profile_path),
        "requested_count": count,
        "state_isolation_contract": {
            "source": "ascension_qwen38_resident.serve_request",
            "reset_before_each_request": True,
            "quality_claim": "not measured by this gate",
        },
        "requests": [],
        "errors": [],
    }
    if config.effective_mode() != "resident":
        report["status"] = "FAILED"
        report["errors"].append({"type": "ConfigurationError", "error": "resident-gate requires a resident profile mode"})
        report["finished_at"] = time.time()
        _write_receipt(report, emit, repo)
        return report

    old_tokens = os.environ.get("HCLI_MODEL_TOKENS")
    connector = HawkingNativeConnector(config)
    first_pid: Optional[int] = None
    try:
        connector.start(timeout=timeout_s)
        ready = connector.identity().get("resident_health") or {}
        report["ready"] = {
            "pid": ready.get("pid"),
            "model_open_count": ready.get("model_open_count"),
            "weight_upload_count": ready.get("weight_upload_count"),
            "resident_identity": ready.get("resident_identity"),
            "protocol": ready.get("protocol"),
        }
        os.environ["HCLI_MODEL_TOKENS"] = str(model_tokens)
        for index in range(count):
            category, base_prompt = DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)]
            prompt = base_prompt
            sentinel: Optional[str] = None
            if index == 5:
                sentinel = f"STATE_LEAK_SENTINEL_{uuid.uuid4().hex}"
                prompt = f"Ignore all prior requests. Reply exactly: ISOLATED_OK. Sentinel: {sentinel}"
                category = "state_reset_probe"
            started = time.perf_counter()
            try:
                raw = connector.complete_payload(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": model_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=timeout_s,
                )
                summary = _response_summary(raw)
                pid = summary.get("pid")
                if first_pid is None:
                    first_pid = pid
                summary.update({
                    "index": index + 1,
                    "category": category,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "nonempty": bool(summary.get("text_len")),
                })
                if sentinel is not None:
                    reply_text = _text(raw) or ""
                    summary["isolation_probe"] = {
                        "expected_marker": "ISOLATED_OK" in reply_text,
                        "sentinel_absent_from_reply": sentinel not in reply_text,
                    }
                report["requests"].append(summary)
            except Exception as exc:  # noqa: BLE001 - the gate records the boundary
                report["errors"].append({
                    "index": index + 1,
                    "category": category,
                    "type": type(exc).__name__,
                    "error": str(exc)[:1600],
                })
                break
        final_health = connector.identity().get("resident_health") or {}
        report["final_health"] = {
            "pid": final_health.get("pid"),
            "model_open_count": final_health.get("model_open_count"),
            "weight_upload_count": final_health.get("weight_upload_count"),
            "restart_count": final_health.get("restart_count"),
        }
    except Exception as exc:  # noqa: BLE001 - the gate records startup failure
        report["errors"].append({"type": type(exc).__name__, "error": str(exc)[:1600]})
    finally:
        connector.stop()
        if old_tokens is None:
            os.environ.pop("HCLI_MODEL_TOKENS", None)
        else:
            os.environ["HCLI_MODEL_TOKENS"] = old_tokens

    rows = report["requests"]
    pids = {row.get("pid") for row in rows if row.get("pid") is not None}
    report["checks"] = {
        "resident_started": bool(report.get("ready", {}).get("pid")),
        "requested_count_reached": len(rows) == count,
        "one_pid_reused": len(pids) == 1 and next(iter(pids), None) == first_pid,
        "one_model_open": report.get("ready", {}).get("model_open_count") == 1 and all(row.get("model_open_count") == 1 for row in rows),
        "one_weight_upload": report.get("ready", {}).get("weight_upload_count") == 1 and all(row.get("weight_upload_count") == 1 for row in rows),
        "all_nonempty": bool(rows) and all(row.get("nonempty") is True for row in rows),
        "zero_fallbacks": bool(rows) and all(row.get("fallbacks") == 0 for row in rows),
        "no_restart": all(row.get("restart_count") == 0 for row in rows),
        "state_reset_isolated": any(
            row.get("isolation_probe", {}).get("expected_marker") is True
            and row.get("isolation_probe", {}).get("sentinel_absent_from_reply") is True
            for row in rows
        ),
        "unique_responses": len({row.get("id") for row in rows}) == len(rows),
        "no_errors": not report["errors"],
    }
    report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    report["finished_at"] = time.time()
    _write_receipt(report, emit, repo)
    return report


__all__ = ["DEFAULT_COUNT", "DEFAULT_PROMPTS", "SCHEMA", "run_resident_gate"]
