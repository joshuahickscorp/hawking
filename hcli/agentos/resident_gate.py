"""Bounded live residency proof for one profile-driven native resident.

The gate measures process/lifecycle properties only.  It does not score model
quality and it does not promote generated prose into a verified fact.  A
different native provider can pass the same gate by implementing the profile's
JSONL contract.
"""
from __future__ import annotations

import sys
from pathlib import Path as _CausalityPath
_CAUSALITY_ROOT = _CausalityPath(__file__).resolve().parents[2]
if str(_CAUSALITY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CAUSALITY_ROOT))
from tools.future import status_causality as sc

import sys

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



def causality_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    errors = report.get("errors") or []
    rows = report.get("requests") or []
    ready = report.get("ready") if isinstance(report.get("ready"), dict) else {}
    final_health = report.get("final_health") if isinstance(report.get("final_health"), dict) else {}
    unmet = [name for name, value in checks.items() if value is not True]
    if not checks and not errors and not rows:
        return {
            "probe_performed": "",
            "direct_observation": "",
            "interpretation": str(report.get("status") or ""),
            "probe_kind": "",
            "claim_kind": None,
        }
    pids = sorted({row.get("pid") for row in rows if isinstance(row, dict) and row.get("pid") is not None})
    isolation = [
        row.get("isolation_probe")
        for row in rows
        if isinstance(row, dict) and row.get("isolation_probe")
    ]
    status = str(report.get("status") or "")
    return {
        "probe_performed": (
            "HawkingNativeConnector.start + sequential complete_payload of "
            f"requested_count={report.get('requested_count')} prompts against "
            f"profile {report.get('profile_path')}; recorded per-request pid, "
            "model_open_count, weight_upload_count, fallbacks, restart_count, "
            "response id, nonempty text, and one isolation sentinel at index 5"
        ),
        "direct_observation": (
            f"ready.pid={ready.get('pid')!r}; ready.model_open_count={ready.get('model_open_count')!r}; "
            f"ready.weight_upload_count={ready.get('weight_upload_count')!r}; "
            f"n_requests={len(rows)}; n_errors={len(errors)}; pids={pids}; "
            f"final_health={final_health}; isolation_probe={isolation}; "
            f"checks={{{', '.join(f'{k}={v!r}' for k, v in sorted(checks.items()))}}}; unmet={unmet!r}"
        ),
        "interpretation": (
            "every named residency lifecycle check in this run's measured flags was True"
            if status == "PASSED"
            else f"one or more named residency lifecycle checks did not hold: {unmet or ['no checks recorded']}"
        ),
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE if status == "PASSED" else sc.CLAIM_MEASURED_UNMET,
    }


def record_resident_causality(report: Dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    payload = kwargs or causality_payload(report)
    return _record_gate_causality(
        report,
        source="hcli/agentos/resident_gate.py::run_resident_gate",
        **payload,
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
        payload = causality_payload(report)
        record_resident_causality(report, **payload)
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
    payload = causality_payload(report)
    record_resident_causality(report, **payload)
    _write_receipt(report, emit, repo)
    return report


__all__ = ["DEFAULT_COUNT", "DEFAULT_PROMPTS", "SCHEMA", "causality_payload", "record_resident_causality", "records_five_fields", "run_resident_gate"]
