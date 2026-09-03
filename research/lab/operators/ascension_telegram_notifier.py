"""Detached, milestone-only Telegram observer for the physical Ascension campaign.

This is deliberately an *observer*, not a controller.  It reads sealed receipts
and compact mutable status documents, sends a small number of operator-facing
messages through the already configured macOS Keychain Telegram bot, and never
promotes a model or mutates a worker, packer, runtime, or tournament gate.

The state machine is fail-closed around delivery: before a Bot API call it
durably records ``SEND_STARTED``; a transport error becomes
``AMBIGUOUS_BLOCKED`` and is never retried automatically.  A successful Bot API
response is persisted as a redacted receipt containing only the HTTP status,
message id, and server date.  Tokens and chat IDs never reach JSON, logs, or
process arguments.

Receipt contracts watched by this observer (all must be hash-sealed):

* ``QWEN{30,80}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json``
* ``QWEN{30,80}_NATIVE_GENERATION_RECEIPT.json``
* ``QWEN{30,80}_CAPABILITY_EVALUATION_RECEIPT.json``
* ``QWEN{30,80}_HCLI_RECEIPT.json``
* ``QWEN{30,80}_TG10_OPERATIONAL_PASS.json``
* ``QWEN{30,80}_TG3_QUALIFICATION_RECEIPT.json``

The TG receipts must explicitly prove the complete artifact BPW, native Metal
execution, autoregressive prompt-dependent generation, HCLI, zero fallback,
and median BASE_TRUE_TPS.  Component probes, prefill rates, rooflines, router
rates, and unsealed status files are intentionally incapable of producing a
TPS or TG notification.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lab.operators import ascension_physical_gatekeeper as gatekeeper
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical"
NOTIFIER_ROOT = PHYSICAL_ROOT / "notifications/telegram"
STATE_SCHEMA = "hawking.ascension.physical_telegram_notifier_state.v1"
STATUS_SCHEMA = "hawking.ascension.physical_telegram_notifier_status.v1"
RECEIPT_SCHEMA = "hawking.ascension.physical_telegram_delivery_receipt.v1"
MAX_STATUS_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_MESSAGE_CHARS = 4096
MAX_TRACKED_EVENTS = 512
HEARTBEAT_STALE_SECONDS = 20 * 60
FRONTIER_MIN_INTERVAL_SECONDS = 30 * 60
DISK_EMERGENCY_FLOOR_BYTES = 160 * 1024**3

# This is the already-live v0/GLM production Telegram credential location.
# The notifier uses only the token and private chat credentials; the GLM HMAC
# service is currently absent on this host, so it is not silently invented.
KEYCHAIN_ACCOUNT = "hawking-glm52-gravity"
TOKEN_SERVICE = "com.hawking.glm52.gravity.telegram.bot-token"
CHAT_SERVICE = "com.hawking.glm52.gravity.telegram.private-chat-id"
TOKEN_RE = re.compile(r"[0-9]{6,16}:[A-Za-z0-9_-]{20,}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# These are the exact native-runtime receipt names and positive contract used
# by the physical tournament gatekeeper.  Do not widen this to a filename
# substring: a component result or an unfinished decoder receipt is not a
# complete native-runtime milestone.
EXACT_RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
EXACT_RUNTIME_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
EXACT_RUNTIME_SUPERSESSION_SCHEMA = (
    "hawking.ascension.physical_exact_full_token_runtime_supersession.v1"
)
EXACT_RUNTIME_REQUIRED_FACTS = (
    "native_exact_decoder",
    "full_token_execution",
    "all_layers_executed",
    "all_weight_tensors_bound",
    "tokenizer_bound",
    "prompt_template_bound",
    "model_alone",
    "no_fallback",
    "raw_bf16_teacher_not_runtime_participant",
)
MODEL_IDS: Mapping[str, str] = {
    "qwen30": "Qwen3-Coder-30B-A3B-Instruct",
    "qwen80": "Qwen3-Coder-Next-80B",
}


class TelegramNotifierError(RuntimeError):
    """The observer cannot safely inspect or deliver an event."""


@dataclass(frozen=True)
class ModelPaths:
    key: str
    prefix: str
    root: Path
    worker_status: Path
    complete_status: Path
    admission_status: Path
    admission_receipt: Path
    runtime_status: Path
    runtime_receipt: Path
    runtime_supersession: Path
    tg3_status: Path


@dataclass(frozen=True)
class NotifierPaths:
    physical_root: Path
    notifier_root: Path

    @property
    def state(self) -> Path:
        return self.notifier_root / "ASCENSION_TELEGRAM_NOTIFIER_STATE.json"

    @property
    def status(self) -> Path:
        return self.notifier_root / "ASCENSION_TELEGRAM_NOTIFIER_STATUS.json"

    @property
    def lock(self) -> Path:
        return self.notifier_root / ".ascension-telegram-notifier.lock"

    @property
    def deliveries(self) -> Path:
        return self.notifier_root / "deliveries"

    @property
    def gate_status(self) -> Path:
        return self.physical_root / "lifecycle/ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json"

    @property
    def dashboard(self) -> Path:
        return self.physical_root.parent / "lifecycle/ASCENSION_DUAL_MANAGER_PHYSICAL_STATUS.json"

    def model(self, key: str) -> ModelPaths:
        if key not in {"qwen30", "qwen80"}:
            raise TelegramNotifierError("unknown Ascension model key")
        prefix = key.upper()
        root = self.physical_root / key
        worker_name = "QWEN30_DUAL_GRAVITY_STATUS.json" if key == "qwen30" else "QWEN80_DUAL_GRAVITY_STATUS.json"
        return ModelPaths(
            key=key,
            prefix=prefix,
            root=root,
            worker_status=root / "evolution" / worker_name,
            complete_status=root / "complete-gravity" / f"{prefix}_COMPLETE_GRAVITY_STATUS.json",
            admission_status=root / "complete-gravity/complete-admission" / f"{prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_STATUS.json",
            admission_receipt=root / "complete-gravity" / f"{prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json",
            runtime_status=root / "complete-runtime" / f"{prefix}_COMPLETE_RUNTIME_STATUS.json",
            runtime_receipt=root / "complete-runtime" / f"{prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
            runtime_supersession=root
            / "complete-runtime"
            / f"{prefix}_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION.json",
            tg3_status=root / "tg3" / f"{prefix}_TG3_ASCENT_STATUS.json",
        )


DEFAULT_PATHS = NotifierPaths(PHYSICAL_ROOT, NOTIFIER_ROOT)


# Event IDs are intentionally stable and compact.  No candidate-level event
# exists in this catalog: representation changes are emitted only when a
# sealed champion changes, with a thirty-minute per-model suppression window.
EVENT_CATALOG: tuple[str, ...] = (
    "operator_path_test",
    "qwen_complete_native_runtime",
    "qwen_native_runtime_revoked",
    "qwen_first_coherent_generation",
    "qwen_capability_pass",
    "qwen_capability_fail",
    "qwen_tg10_operational_pass",
    "qwen_tg_rung",
    "qwen_tg3_qualified",
    "qwen_worker_stalled",
    "qwen_worker_recovered",
    "qwen_physical_representation_frontier_improved",
    "qwen_complete_binary_admitted",
    "global_both_tg10_operational",
    "global_both_tg3_qualified",
    "global_tournament_automatically_launched",
    "global_tournament_complete",
    "global_human_decision_required",
    "global_resource_emergency",
    "global_detached_worker_failure",
    "global_campaign_resumed",
)

QUALIFICATION_RECEIPT_CONTRACT: Mapping[str, Any] = {
    "native_runtime": {
        "filename": "QWEN{30,80}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
        "schema": EXACT_RUNTIME_SCHEMA,
        "status": EXACT_RUNTIME_STATUS,
        "supersession_filename": "QWEN{30,80}_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION.json",
        "supersession_schema": EXACT_RUNTIME_SUPERSESSION_SCHEMA,
        "supersession_rule": "a sealed revocation suppresses the old runtime and every dependent HCLI/TPS/TG/capability notification",
        "required_runtime_facts": EXACT_RUNTIME_REQUIRED_FACTS,
        "timing_scope": "complete_model_token_loop",
        "minimum_measured_token_count": 1,
    },
    "generation": "QWEN{30,80}_NATIVE_GENERATION_RECEIPT.json (sealed positive receipt)",
    "capability": "QWEN{30,80}_CAPABILITY_EVALUATION_RECEIPT.json (sealed PASS or FAIL receipt)",
    "tg10": {
        "filename": "QWEN{30,80}_TG10_OPERATIONAL_PASS.json",
        "minimum_median_base_true_tps": 100.0,
    },
    "tg3": {
        "filename": "QWEN{30,80}_TG3_QUALIFICATION_RECEIPT.json",
        "minimum_median_base_true_tps": 333.0,
    },
    "required_explicit_tg_facts": (
        "complete_bpw <= 1.5",
        "complete_native_model=true",
        "real_metal=true",
        "autoregressive_generation=true",
        "hcli_pass=true",
        "fallback_count=0",
        "median_base_true_tps",
    ),
    "explicitly_rejected_as_model_tps": (
        "component_probe_tps",
        "router_tps",
        "projection_tps",
        "prefill_tps",
        "roofline",
        "speculative_accepted_tps",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path, *, maximum_bytes: int = MAX_STATUS_BYTES) -> dict[str, Any] | None:
    try:
        status = path.stat()
        if not path.is_file() or status.st_size > maximum_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _sealed_document(path: Path) -> dict[str, Any] | None:
    document = _read_json(path)
    if document is None:
        return None
    try:
        return verify(document, label=str(path))
    except SealIntegrityError:
        return None


def _safe_relative(path: Path, physical_root: Path) -> str:
    try:
        return str(path.relative_to(physical_root.parent))
    except ValueError:
        return str(path.name)


def _status_text(document: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("status", "phase", "verdict", "outcome", "result"):
        value = document.get(key)
        if isinstance(value, str):
            parts.append(value.upper())
    return " ".join(parts)


def _is_positive_status(document: Mapping[str, Any]) -> bool:
    text = _status_text(document)
    if any(word in text for word in ("BLOCKED", "UNIMPLEMENTED", "FAIL", "REJECT", "NOT_", "WAITING")):
        return False
    return any(word in text for word in ("PASS", "EARNED", "QUALIFIED", "COMPLETE", "DELIVERED"))


def _is_negative_status(document: Mapping[str, Any]) -> bool:
    text = _status_text(document)
    return any(word in text for word in ("FAIL", "REJECT", "NEGATIVE")) and "NOT_FAILED" not in text


def _walk_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key), nested
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _field(value: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    wanted = {alias.lower() for alias in aliases}
    for key, nested in _walk_values(value):
        if key.lower() in wanted:
            return nested
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _pass_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        text = value.upper()
        return any(word in text for word in ("PASS", "EARNED", "COMPLETE", "QUALIFIED", "TRUE")) and not any(
            word in text for word in ("BLOCKED", "FAIL", "NOT_", "WAITING")
        )
    if isinstance(value, Mapping):
        return _is_positive_status(value)
    return False


def _zero_fallback(value: Mapping[str, Any]) -> bool:
    count = _field(value, ("fallback_count", "fallbacks", "fallback_count_total"))
    number = _number(count)
    if number is not None:
        return number == 0.0
    fallback = _field(value, ("fallback", "fallback_used", "used_fallback"))
    return fallback is False or (isinstance(fallback, str) and fallback.upper() in {"NONE", "ZERO", "FALSE"})


def _evidence_scalar(document: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    return _field(document, aliases)


def _tg_receipt_qualifies(document: Mapping[str, Any], *, minimum_tps: float) -> tuple[bool, dict[str, Any]]:
    """Reject anything short of a sealed, complete full-token proof.

    These aliases are deliberately narrow.  A future runtime must make each
    required fact explicit instead of relying on a prose phrase or a component
    metric that happens to contain the word "TPS".
    """

    bpw = _number(_evidence_scalar(document, ("complete_bpw", "complete_physical_bpw", "complete_artifact_bpw")))
    tps = _number(_evidence_scalar(document, ("median_base_true_tps", "base_true_tps_median")))
    native = _pass_value(_evidence_scalar(document, ("complete_native_model", "complete_native_runtime", "native_model_pass")))
    metal = _pass_value(_evidence_scalar(document, ("real_metal", "native_metal_execution", "metal_execution_pass")))
    generation = _pass_value(_evidence_scalar(document, ("autoregressive_generation", "prompt_dependent_generation", "coherent_generation_pass")))
    hcli = _pass_value(_evidence_scalar(document, ("hcli_pass", "hcli", "hcli_generation_pass")))
    fallback_zero = _zero_fallback(document)
    facts = {
        "complete_bpw": bpw,
        "median_base_true_tps": tps,
        "complete_native_model": native,
        "real_metal": metal,
        "autoregressive_generation": generation,
        "hcli_pass": hcli,
        "fallback_zero": fallback_zero,
    }
    return (
        _is_positive_status(document)
        and bpw is not None
        and bpw <= 1.5
        and tps is not None
        and tps >= minimum_tps
        and native
        and metal
        and generation
        and hcli
        and fallback_zero,
        facts,
    )


def _pid_snapshot(pid: Any) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {"pid": None, "alive": False, "observed_ppid": None}
    try:
        os.kill(pid, 0)
    except OSError:
        return {"pid": pid, "alive": False, "observed_ppid": None}
    observed_ppid: int | None = None
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        candidate = result.stdout.strip()
        if candidate.isdigit():
            observed_ppid = int(candidate)
    except (OSError, subprocess.SubprocessError):
        pass
    return {"pid": pid, "alive": True, "observed_ppid": observed_ppid}


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _keychain_value(service: str) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _credential_shape(token: str | None, chat_id: str | None) -> bool:
    return bool(token and chat_id and TOKEN_RE.fullmatch(token) and chat_id.isascii() and chat_id.isdigit() and int(chat_id) > 0)


def _remote_get_me(token: str) -> bool:
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/getMe",
        data=b"",
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "hawking-ascension-telegram/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read(MAX_RECEIPT_BYTES + 1))
            return bool(
                response.status == 200
                and isinstance(body, Mapping)
                and body.get("ok") is True
                and isinstance(body.get("result"), Mapping)
                and body["result"].get("is_bot") is True
            )
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return False


def credential_status(
    *,
    reader: Callable[[str], str | None] = _keychain_value,
    validate_remote: bool = False,
    remote_validator: Callable[[str], bool] = _remote_get_me,
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    token = reader(TOKEN_SERVICE)
    chat_id = reader(CHAT_SERVICE)
    shaped = _credential_shape(token, chat_id)
    remote_ok = remote_validator(token) if shaped and validate_remote and token is not None else None
    return (
        {
            "provider": "macOS Keychain com.hawking.glm52.gravity.telegram.*",
            "token_present": bool(token),
            "private_chat_present": bool(chat_id),
            "credential_shape_valid": shaped,
            "bot_api_validated": remote_ok,
            "secrets_persisted": False,
        },
        (token, chat_id) if shaped and token is not None and chat_id is not None else None,
    )


def _telegram_send(token: str, chat_id: str, text: str) -> dict[str, Any]:
    if not _credential_shape(token, chat_id) or not text or len(text) > MAX_MESSAGE_CHARS:
        raise TelegramNotifierError("Telegram credentials or message shape are invalid")
    payload = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "hawking-ascension-telegram/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(MAX_RECEIPT_BYTES + 1)
            if len(raw) > MAX_RECEIPT_BYTES:
                raise TelegramNotifierError("Telegram response exceeded the safe size limit")
            body = json.loads(raw)
    except TelegramNotifierError:
        raise
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        raise TelegramNotifierError("Telegram sendMessage transport failed") from None
    result = body.get("result") if isinstance(body, Mapping) else None
    message_id = result.get("message_id") if isinstance(result, Mapping) else None
    date = result.get("date") if isinstance(result, Mapping) else None
    if response.status != 200 or not isinstance(body, Mapping) or body.get("ok") is not True or not isinstance(message_id, int) or isinstance(message_id, bool):
        raise TelegramNotifierError("Telegram sendMessage did not return a validated delivery")
    return {
        "bot_api_http_status": 200,
        "message_id": message_id,
        "message_date": date if isinstance(date, int) and not isinstance(date, bool) else None,
    }


def _initial_state(now: str) -> dict[str, Any]:
    return seal(
        {
            "schema": STATE_SCHEMA,
            "created_at": now,
            "updated_at": now,
            "armed": False,
            "baseline_fingerprints": {},
            "events": {},
            "worker_health": {},
            "test_event": None,
        }
    )


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(_utc_now())
    document = _read_json(path)
    if document is None:
        raise TelegramNotifierError("notifier state is unreadable")
    try:
        checked = verify(document, label=str(path))
    except SealIntegrityError as exc:
        raise TelegramNotifierError("notifier state seal is invalid") from exc
    if checked.get("schema") != STATE_SCHEMA:
        raise TelegramNotifierError("notifier state schema is invalid")
    if not isinstance(checked.get("events"), Mapping) or not isinstance(checked.get("baseline_fingerprints"), Mapping):
        raise TelegramNotifierError("notifier state fields are invalid")
    return checked


def _save_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    base = {key: value for key, value in dict(state).items() if key != "seal_sha256"}
    base["updated_at"] = _utc_now()
    sealed = seal(base)
    _atomic_json(path, sealed)
    return sealed


def _receipt_path(paths: NotifierPaths, dedupe_key: str) -> Path:
    return paths.deliveries / f"{dedupe_key}.json"


def _render(event: Mapping[str, Any]) -> str:
    model = event.get("model")
    heading = "Hawking Ascension physical"
    if isinstance(model, str):
        heading += f" · {model.upper()}"
    text = "\n".join(
        [
            heading,
            str(event["event_id"]),
            str(event["summary"]),
            f"evidence={event['evidence_path']}",
        ]
    )
    if len(text) > MAX_MESSAGE_CHARS:
        raise TelegramNotifierError("rendered Telegram message is too long")
    return text


def _delivery_receipt(
    *, event: Mapping[str, Any], dedupe_key: str, message: str, status: str, delivery: Mapping[str, Any] | None, reason: str | None = None
) -> dict[str, Any]:
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "recorded_at": _utc_now(),
            "status": status,
            "event_id": event["event_id"],
            "model": event.get("model"),
            "dedupe_key": dedupe_key,
            "event_fingerprint": event["fingerprint"],
            "evidence_path": event["evidence_path"],
            "message_sha256": _sha256(message),
            "delivery": dict(delivery) if delivery is not None else None,
            "reason": reason,
            "secret_boundary": "no token or chat identifier is persisted",
        }
    )


def _event(
    event_id: str,
    *,
    model: str | None,
    summary: str,
    evidence_path: str,
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    if event_id not in EVENT_CATALOG:
        raise TelegramNotifierError("unknown notification event")
    payload = {"event_id": event_id, "model": model, "evidence_path": evidence_path, "facts": dict(facts)}
    fingerprint = _sha256(payload)
    return {**payload, "summary": summary, "fingerprint": fingerprint}


def _model_evidence(paths: ModelPaths) -> list[tuple[Path, dict[str, Any]]]:
    """Read only bounded runtime/evaluation directories, never candidates or weights."""

    roots = (
        paths.root / "complete-runtime",
        paths.root / "tg3",
        paths.root / "evaluation",
        paths.root / "hcli",
        paths.root / "benchmarks",
        paths.root / "agent-os",
        paths.root / "complete-gravity/complete-admission",
    )
    found: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("*.json"))[:256]:
            document = _sealed_document(candidate)
            if document is not None:
                found.append((candidate, document))
    return found


def _evidence_by_name(entries: Sequence[tuple[Path, dict[str, Any]]], *, words: Sequence[str]) -> tuple[Path, dict[str, Any]] | None:
    wanted = tuple(word.upper() for word in words)
    for path, document in entries:
        name = path.name.upper()
        if all(word in name for word in wanted):
            return path, document
    return None


def _sealed_champion(paths: ModelPaths) -> tuple[dict[str, Any] | None, Path]:
    candidate = paths.root / "evolution/CHAMPIONS.json"
    return _sealed_document(candidate), candidate


def _gatekeeper_spec(key: str) -> gatekeeper.ModelSpec:
    for spec in gatekeeper.MODEL_SPECS:
        if spec.key == key:
            return spec
    raise TelegramNotifierError(f"unknown Ascension model key for runtime authority: {key}")


def _runtime_qualification_current(
    snapshot: Mapping[str, Any], document: Mapping[str, Any] | None = None
) -> bool:
    """Require a currently eligible runtime before emitting dependent events.

    When a sealed supersession exists, a later corrected canonical runtime may
    coexist with older HCLI/TG/capability records.  Those historical records
    are not allowed to become new notifications unless they bind the current
    runtime seal.  With no supersession, this retains the existing strictly
    receipt-shaped notification behavior.
    """

    authority = snapshot.get("runtime_authority")
    if not isinstance(authority, Mapping) or authority.get("current_runtime_eligible") is not True:
        return False
    if document is None or authority.get("supersession_present") is not True:
        return True
    binding = document.get("binding") if isinstance(document.get("binding"), Mapping) else {}
    return binding.get("runtime_receipt_seal_sha256") == authority.get(
        "canonical_runtime_receipt_seal_sha256"
    )


def _model_snapshot(paths: ModelPaths, *, now: float) -> dict[str, Any]:
    worker = _read_json(paths.worker_status) or {}
    runtime = _read_json(paths.runtime_status) or {}
    runtime_receipt = _sealed_document(paths.runtime_receipt)
    runtime_authority = gatekeeper.runtime_receipt_supersession_state(
        _gatekeeper_spec(paths.key),
        runtime_path=paths.runtime_receipt,
        supersession_path=paths.runtime_supersession,
    )
    complete = _read_json(paths.complete_status) or {}
    admission = _sealed_document(paths.admission_receipt)
    tg3_status = _read_json(paths.tg3_status) or {}
    champion, champion_path = _sealed_champion(paths)
    evidence = _model_evidence(paths)
    pid_info = _pid_snapshot(worker.get("pid"))
    timestamp = _parse_timestamp(worker.get("recorded_at"))
    age = None if timestamp is None else max(0.0, now - timestamp)
    health = bool(pid_info["alive"] and age is not None and age <= HEARTBEAT_STALE_SECONDS)
    return {
        "key": paths.key,
        "prefix": paths.prefix,
        "worker": worker,
        "runtime": runtime,
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_path": paths.runtime_receipt,
        "runtime_supersession_path": paths.runtime_supersession,
        "runtime_authority": runtime_authority,
        "complete": complete,
        "admission": admission,
        "tg3_status": tg3_status,
        "champion": champion,
        "champion_path": champion_path,
        "evidence": evidence,
        "pid": pid_info,
        "status_age_seconds": age,
        "healthy": health,
    }


def _frontier_facts(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    champion = snapshot.get("champion")
    if not isinstance(champion, Mapping):
        return None
    fastest = champion.get("current_fastest_component") if isinstance(champion.get("current_fastest_component"), Mapping) else {}
    lowest = champion.get("current_lowest_bpw_component") if isinstance(champion.get("current_lowest_bpw_component"), Mapping) else {}
    if not fastest and not lowest:
        return None
    return {
        "champion_seal_sha256": champion.get("seal_sha256"),
        "fastest_component_candidate": fastest.get("candidate_id"),
        "fastest_component_matvecs_per_second": fastest.get("mps_matvecs_per_second"),
        "lowest_component_candidate": lowest.get("candidate_id"),
        "lowest_component_bpw": lowest.get("physical_bpw"),
        "claim_boundary": "component frontier only; never model TPS or qualification",
    }


def _admission_event(snapshot: Mapping[str, Any], physical_root: Path) -> dict[str, Any] | None:
    admission = snapshot.get("admission")
    status = str(admission.get("status") or "") if isinstance(admission, Mapping) else ""
    if not isinstance(admission, Mapping) or not (status.startswith("EARNED_") and "ADMITTED" in status):
        return None
    path = Path(snapshot["worker"].get("complete_pack", {}).get("status_path", "")) if isinstance(snapshot.get("worker"), Mapping) else None
    evidence = _safe_relative(path, physical_root) if path is not None and str(path) else f"physical/{snapshot['key']}/complete-gravity"
    return _event(
        "qwen_complete_binary_admitted",
        model=str(snapshot["key"]),
        summary="complete binary artifact admitted; this is not a native decoder, HCLI, TPS, or tournament qualification claim.",
        evidence_path=evidence,
        facts={"admission_seal_sha256": admission.get("seal_sha256"), "status": status},
    )


def _receipt_event(snapshot: Mapping[str, Any], physical_root: Path, *, label: str, name_words: Sequence[str], event_id: str, summary: str, require_positive: bool = True, require_negative: bool = False, require_current_runtime: bool = False) -> dict[str, Any] | None:
    match = _evidence_by_name(snapshot["evidence"], words=name_words)
    if match is None:
        return None
    path, document = match
    if require_current_runtime and not _runtime_qualification_current(snapshot, document):
        return None
    if require_positive and not _is_positive_status(document):
        return None
    if require_negative and not _is_negative_status(document):
        return None
    return _event(
        event_id,
        model=str(snapshot["key"]),
        summary=summary,
        evidence_path=_safe_relative(path, physical_root),
        facts={"receipt_seal_sha256": document.get("seal_sha256"), "status": document.get("status"), "phase": document.get("phase")},
    )


def _exact_native_runtime_event(snapshot: Mapping[str, Any], physical_root: Path) -> dict[str, Any] | None:
    """Return only a gatekeeper-shaped exact native-runtime milestone.

    This notifier deliberately does not infer readiness from arbitrary positive
    receipts in ``complete-runtime``.  The receipt must be at the canonical
    model-specific path and contain the exact positive runtime contract.  It
    remains a runtime milestone only: capability, HCLI, TPS, TG, and tournament
    events have their own independent sealed gates.
    """

    document = snapshot.get("runtime_receipt")
    path = snapshot.get("runtime_receipt_path")
    key = snapshot.get("key")
    if not isinstance(document, Mapping) or not isinstance(path, Path) or not isinstance(key, str):
        return None
    if not _runtime_qualification_current(snapshot):
        return None
    if (
        document.get("schema") != EXACT_RUNTIME_SCHEMA
        or document.get("status") != EXACT_RUNTIME_STATUS
    ):
        return None
    binding = document.get("binding")
    runtime = document.get("runtime")
    if not isinstance(binding, Mapping) or not isinstance(runtime, Mapping):
        return None
    if binding.get("model_id") != MODEL_IDS.get(key):
        return None
    if not all(
        _is_sha256(binding.get(field))
        for field in (
            "source_content_identity_sha256",
            "source_revalidation_seal_sha256",
            "complete_artifact_admission_seal_sha256",
        )
    ):
        return None
    if not all(runtime.get(field) is True for field in EXACT_RUNTIME_REQUIRED_FACTS):
        return None
    if not _is_positive_int(runtime.get("measured_token_count")):
        return None
    if runtime.get("timing_scope") != "complete_model_token_loop":
        return None
    return _event(
        "qwen_complete_native_runtime",
        model=key,
        summary="sealed exact native full-token runtime receipt observed; this is not a coherence, HCLI, TPS, TG, or tournament qualification claim.",
        evidence_path=_safe_relative(path, physical_root),
        facts={
            "receipt_seal_sha256": document.get("seal_sha256"),
            "runtime_schema": EXACT_RUNTIME_SCHEMA,
            "runtime_status": EXACT_RUNTIME_STATUS,
            "measured_token_count": runtime.get("measured_token_count"),
            "timing_scope": runtime.get("timing_scope"),
        },
    )


def _native_runtime_revocation_event(
    snapshot: Mapping[str, Any], physical_root: Path
) -> dict[str, Any] | None:
    """Expose a one-time operator correction only from the sealed contract.

    This does not infer a regression from an ordinary blocked status.  It is
    emitted solely when the v1 supersession resolver validated an explicit
    revocation and that record attests withdrawal of the native HTTP adapter.
    """

    authority = snapshot.get("runtime_authority")
    key = snapshot.get("key")
    path = snapshot.get("runtime_supersession_path")
    if not isinstance(authority, Mapping) or not isinstance(key, str) or not isinstance(path, Path):
        return None
    if (
        authority.get("state") != "CURRENT_RUNTIME_REVOKED"
        or authority.get("current_runtime_eligible") is not False
        or not _is_sha256(authority.get("supersession_seal_sha256"))
    ):
        return None
    document = _sealed_document(path)
    if not isinstance(document, Mapping):
        return None
    if (
        document.get("schema") != EXACT_RUNTIME_SUPERSESSION_SCHEMA
        or not str(document.get("status") or "").startswith("REVOKED_")
    ):
        return None
    invalidates = document.get("invalidates")
    if not isinstance(invalidates, Mapping) or invalidates.get(
        "native_http_adapter_and_transport_handoff_bound_to_runtime_sha"
    ) is not True:
        return None
    return _event(
        "qwen_native_runtime_revoked",
        model=key,
        summary="sealed native-runtime correction invalidated the prior runtime and withdrew its bound serving path; corrected requalification is in progress.",
        evidence_path=_safe_relative(path, physical_root),
        facts={
            "supersession_seal_sha256": authority.get("supersession_seal_sha256"),
            "revoked_runtime_receipt_seal_sha256": authority.get(
                "superseded_runtime_receipt_seal_sha256"
            ),
            "defective_runtime_executable_sha256": authority.get(
                "defective_runtime_executable_sha256"
            ),
            "serving_path_withdrawn": True,
        },
    )


def _tg_event(snapshot: Mapping[str, Any], physical_root: Path, *, tg: str, minimum_tps: float, event_id: str, summary: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    match = _evidence_by_name(snapshot["evidence"], words=(tg,))
    if match is None:
        return None, None
    path, document = match
    if not _runtime_qualification_current(snapshot, document):
        return None, None
    passed, facts = _tg_receipt_qualifies(document, minimum_tps=minimum_tps)
    if not passed:
        return None, {"path": path, "document": document, "facts": facts}
    return (
        _event(
            event_id,
            model=str(snapshot["key"]),
            summary=summary,
            evidence_path=_safe_relative(path, physical_root),
            facts={"receipt_seal_sha256": document.get("seal_sha256"), **facts},
        ),
        {"path": path, "document": document, "facts": facts},
    )


def _resource_state(dashboard: Mapping[str, Any]) -> dict[str, Any]:
    resources = dashboard.get("global_resources") if isinstance(dashboard.get("global_resources"), Mapping) else {}
    free_disk = _number(resources.get("free_disk_bytes"))
    reserve = _number(resources.get("minimum_reserved_free_bytes")) or float(DISK_EMERGENCY_FLOOR_BYTES)
    memory_text = str(resources.get("host", {}).get("memory_pressure", "")) if isinstance(resources.get("host"), Mapping) else ""
    swap_text = str(resources.get("host", {}).get("swap", "")) if isinstance(resources.get("host"), Mapping) else ""
    thermal_text = str(resources.get("host", {}).get("thermal", "")) if isinstance(resources.get("host"), Mapping) else ""
    memory_percent = None
    match = re.search(r"free percentage:\s*(\d+)%", memory_text, flags=re.IGNORECASE)
    if match:
        memory_percent = int(match.group(1))
    thermal_bad = "warning" in thermal_text.lower() and "no thermal warning" not in thermal_text.lower() and "no performance warning" not in thermal_text.lower()
    disk_bad = free_disk is not None and free_disk < reserve
    memory_bad = memory_percent is not None and memory_percent < 10
    swap_bad = bool(re.search(r"used\s*=\s*(?!0(?:\.0+)?[A-Za-z])[^\s,]+", swap_text, flags=re.IGNORECASE))
    return {
        "emergency": bool(disk_bad or memory_bad or swap_bad or thermal_bad),
        "free_disk_bytes": free_disk,
        "minimum_reserved_free_bytes": reserve,
        "memory_free_percent": memory_percent,
        "swap_nonzero": swap_bad,
        "thermal_warning": thermal_bad,
    }


def _tournament_events(gate: Mapping[str, Any], physical_root: Path) -> list[dict[str, Any]]:
    try:
        checked = verify(gate, label="physical tournament gate")
    except SealIntegrityError:
        return []
    path = "physical/lifecycle/ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json"
    status = str(checked.get("status") or "")
    runtime_phase = str(checked.get("runtime_phase") or "")
    execution = checked.get("tournament_execution") if isinstance(checked.get("tournament_execution"), Mapping) else {}
    facts = {"gate_seal_sha256": checked.get("seal_sha256"), "status": status, "runtime_phase": runtime_phase, "execution_status": execution.get("status")}
    events: list[dict[str, Any]] = []
    combined = f"{status} {runtime_phase} {execution.get('status', '')}".upper()
    if "MANAGER_TOURNAMENT_RUNNING" in combined or "TOURNAMENT_RUNNING" in combined:
        events.append(_event("global_tournament_automatically_launched", model=None, summary="protected gate transitioned to the running tournament state.", evidence_path=path, facts=facts))
    if "COMPLETE" in combined and "NOT_LAUNCHED" not in combined:
        events.append(_event("global_tournament_complete", model=None, summary="protected tournament reports completion; human review may be required.", evidence_path=path, facts=facts))
    if "HUMAN_DECISION_REQUIRED" in combined or "REVIEW_REQUIRED" in combined:
        events.append(_event("global_human_decision_required", model=None, summary="protected tournament gate requests a human decision.", evidence_path=path, facts=facts))
    return events


class AscensionTelegramNotifier:
    """Read-only campaign observer with a durable, idempotent delivery outbox."""

    def __init__(
        self,
        *,
        paths: NotifierPaths = DEFAULT_PATHS,
        credential_reader: Callable[[str], str | None] = _keychain_value,
        sender: Callable[[str, str, str], Mapping[str, Any]] = _telegram_send,
        remote_validator: Callable[[str], bool] = _remote_get_me,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.paths = paths
        self._credential_reader = credential_reader
        self._sender = sender
        self._remote_validator = remote_validator
        self._clock = clock

    def _locked(self) -> Any:
        self.paths.notifier_root.mkdir(parents=True, exist_ok=True)
        handle = open(self.paths.lock, "a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _status_payload(self, *, state: Mapping[str, Any], snapshot: Mapping[str, Any] | None, credential: Mapping[str, Any], last_error: str | None = None) -> dict[str, Any]:
        workers: dict[str, Any] = {}
        if isinstance(snapshot, Mapping):
            for key, value in (snapshot.get("models") or {}).items():
                if not isinstance(value, Mapping):
                    continue
                pid_snapshot = value.get("pid") if isinstance(value.get("pid"), Mapping) else {}
                runtime_authority = (
                    value.get("runtime_authority")
                    if isinstance(value.get("runtime_authority"), Mapping)
                    else {}
                )
                workers[str(key)] = {
                    "pid": pid_snapshot.get("pid"),
                    "pid_alive": pid_snapshot.get("alive", False),
                    "observed_ppid": pid_snapshot.get("observed_ppid"),
                    "status_age_seconds": value.get("status_age_seconds"),
                    "healthy": value.get("healthy"),
                    "worker_phase": (value.get("worker") or {}).get("phase") if isinstance(value.get("worker"), Mapping) else None,
                    "runtime_phase": (value.get("runtime") or {}).get("phase") if isinstance(value.get("runtime"), Mapping) else None,
                    "runtime_authority_state": runtime_authority.get("state"),
                    "runtime_authority_currently_eligible": runtime_authority.get("current_runtime_eligible"),
                    "runtime_supersession_seal_sha256": runtime_authority.get("supersession_seal_sha256"),
                }
        events = state.get("events") if isinstance(state.get("events"), Mapping) else {}
        delivered = sum(1 for record in events.values() if isinstance(record, Mapping) and record.get("lifecycle") == "DELIVERED")
        ambiguous = sum(1 for record in events.values() if isinstance(record, Mapping) and record.get("lifecycle") == "AMBIGUOUS_BLOCKED")
        body = {
            "schema": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "ARMED" if state.get("armed") else "UNARMED",
            "credential": dict(credential),
            "event_catalog": list(EVENT_CATALOG),
            "qualification_receipt_contract": QUALIFICATION_RECEIPT_CONTRACT,
            "delivery": {"delivered_event_count": delivered, "ambiguous_blocked_event_count": ambiguous, "test_event": state.get("test_event")},
            "workers": workers,
            "no_per_candidate_spam": True,
            "claim_boundary": "notifications are observational; only sealed evidence can produce completion-shaped messages",
            "last_error": last_error,
        }
        return seal(body)

    def _write_status(self, *, state: Mapping[str, Any], snapshot: Mapping[str, Any] | None, credential: Mapping[str, Any], last_error: str | None = None) -> None:
        _atomic_json(self.paths.status, self._status_payload(state=state, snapshot=snapshot, credential=credential, last_error=last_error))

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        models = {key: _model_snapshot(self.paths.model(key), now=now) for key in ("qwen30", "qwen80")}
        dashboard = _read_json(self.paths.dashboard) or {}
        gate = _read_json(self.paths.gate_status) or {}
        return {"models": models, "dashboard": dashboard, "gate": gate, "resource": _resource_state(dashboard)}

    def _candidate_events(self, snapshot: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        tg10_by_model: dict[str, Mapping[str, Any]] = {}
        tg3_by_model: dict[str, Mapping[str, Any]] = {}
        old_health = state.get("worker_health") if isinstance(state.get("worker_health"), Mapping) else {}
        for key in ("qwen30", "qwen80"):
            model = snapshot["models"][key]
            admission = _admission_event(model, self.paths.physical_root)
            if admission is not None:
                candidates.append(admission)
            runtime = _exact_native_runtime_event(model, self.paths.physical_root)
            if runtime is not None:
                candidates.append(runtime)
            runtime_revocation = _native_runtime_revocation_event(model, self.paths.physical_root)
            if runtime_revocation is not None:
                candidates.append(runtime_revocation)
            generation = _receipt_event(
                model,
                self.paths.physical_root,
                label="generation",
                name_words=(model["prefix"], "GENERATION", "RECEIPT"),
                event_id="qwen_first_coherent_generation",
                summary="sealed native prompt-dependent generation receipt observed.",
                require_current_runtime=True,
            )
            if generation is not None:
                candidates.append(generation)
            cap_pass = _receipt_event(
                model,
                self.paths.physical_root,
                label="capability pass",
                name_words=(model["prefix"], "CAPABILITY", "RECEIPT"),
                event_id="qwen_capability_pass",
                summary="sealed capability evaluation pass observed.",
                require_current_runtime=True,
            )
            if cap_pass is not None:
                candidates.append(cap_pass)
            cap_fail = _receipt_event(
                model,
                self.paths.physical_root,
                label="capability fail",
                name_words=(model["prefix"], "CAPABILITY", "RECEIPT"),
                event_id="qwen_capability_fail",
                summary="sealed capability evaluation failure observed.",
                require_positive=False,
                require_negative=True,
                require_current_runtime=True,
            )
            if cap_fail is not None:
                candidates.append(cap_fail)
            tg10, tg10_proof = _tg_event(
                model,
                self.paths.physical_root,
                tg="TG10",
                minimum_tps=100.0,
                event_id="qwen_tg10_operational_pass",
                summary="TG10 operational pass: sealed complete-token BASE_TRUE_TPS is at least 100 with fallback=0.",
            )
            if tg10 is not None:
                candidates.append(tg10)
                tg10_by_model[key] = tg10_proof or {}
            tg3, tg3_proof = _tg_event(
                model,
                self.paths.physical_root,
                tg="TG3",
                minimum_tps=333.0,
                event_id="qwen_tg3_qualified",
                summary="TG3 qualification pass: sealed complete-token BASE_TRUE_TPS is at least 333 with fallback=0.",
            )
            if tg3 is not None:
                candidates.append(tg3)
                tg3_by_model[key] = tg3_proof or {}
            # Any sealed positive non-TG3/TG10 TG receipt can describe a later rung.
            for path, document in model["evidence"]:
                name = path.name.upper()
                if "TG" not in name or "RECEIPT" not in name or "TG10" in name or "TG3" in name:
                    continue
                if not _runtime_qualification_current(model, document):
                    continue
                match = re.search(r"TG(\d+)", name)
                if match is None or not _is_positive_status(document):
                    continue
                candidates.append(
                    _event(
                        "qwen_tg_rung",
                        model=key,
                        summary=f"sealed {match.group(0)} rung receipt observed.",
                        evidence_path=_safe_relative(path, self.paths.physical_root),
                        facts={"tg_rung": match.group(0), "receipt_seal_sha256": document.get("seal_sha256")},
                    )
                )
            frontier = _frontier_facts(model)
            if frontier is not None:
                candidates.append(
                    _event(
                        "qwen_physical_representation_frontier_improved",
                        model=key,
                        summary="sealed component representation champion changed; it is not model TPS or capability evidence.",
                        evidence_path=_safe_relative(model["champion_path"], self.paths.physical_root),
                        facts=frontier,
                    )
                )
            was_healthy = old_health.get(key) if isinstance(old_health, Mapping) else None
            healthy = bool(model["healthy"])
            health_facts = {"pid": model["pid"].get("pid"), "pid_alive": model["pid"].get("alive"), "observed_ppid": model["pid"].get("observed_ppid"), "status_age_seconds": model.get("status_age_seconds")}
            if was_healthy is True and not healthy:
                candidates.append(_event("qwen_worker_stalled", model=key, summary="detached worker is stale or its declared PID is no longer alive.", evidence_path=_safe_relative(self.paths.model(key).worker_status, self.paths.physical_root), facts=health_facts))
                candidates.append(_event("global_detached_worker_failure", model=None, summary=f"{key} detached worker failure/stall observed.", evidence_path=_safe_relative(self.paths.model(key).worker_status, self.paths.physical_root), facts={"worker": key, **health_facts}))
            if was_healthy is False and healthy:
                candidates.append(_event("qwen_worker_recovered", model=key, summary="detached worker recovered with a live PID and fresh status.", evidence_path=_safe_relative(self.paths.model(key).worker_status, self.paths.physical_root), facts=health_facts))

        if len(tg10_by_model) == 2:
            candidates.append(_event("global_both_tg10_operational", model=None, summary="both managers have independently sealed TG10 operational passes.", evidence_path="physical/notifications/telegram", facts={key: value.get("document", {}).get("seal_sha256") for key, value in tg10_by_model.items()}))
        if len(tg3_by_model) == 2:
            candidates.append(_event("global_both_tg3_qualified", model=None, summary="both managers have independently sealed TG3 qualification passes.", evidence_path="physical/notifications/telegram", facts={key: value.get("document", {}).get("seal_sha256") for key, value in tg3_by_model.items()}))
        resource = snapshot["resource"]
        old_resource = state.get("resource_emergency")
        if resource["emergency"] and old_resource is not True:
            candidates.append(_event("global_resource_emergency", model=None, summary="resource emergency threshold crossed; consult the physical dashboard.", evidence_path="lifecycle/ASCENSION_DUAL_MANAGER_PHYSICAL_STATUS.json", facts=resource))
        if all(bool(snapshot["models"][key]["healthy"]) for key in ("qwen30", "qwen80")) and any(old_health.get(key) is False for key in ("qwen30", "qwen80")):
            candidates.append(_event("global_campaign_resumed", model=None, summary="both detached Qwen workers are healthy again after a recorded stall.", evidence_path="lifecycle/ASCENSION_DUAL_MANAGER_PHYSICAL_STATUS.json", facts={key: snapshot["models"][key]["pid"].get("pid") for key in ("qwen30", "qwen80")}))
        candidates.extend(_tournament_events(snapshot["gate"], self.paths.physical_root))
        return candidates

    def _record_send_started(self, state: Mapping[str, Any], *, dedupe_key: str, event: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
        updated = dict(state)
        events = dict(updated.get("events") or {})
        events[dedupe_key] = {
            "event_id": event["event_id"],
            "model": event.get("model"),
            "fingerprint": event["fingerprint"],
            "lifecycle": "SEND_STARTED",
            "receipt_path": str(receipt_path),
            "recorded_at": _utc_now(),
        }
        updated["events"] = events
        return _save_state(self.paths.state, updated)

    def _deliver(self, state: Mapping[str, Any], event: Mapping[str, Any], credentials: tuple[str, str]) -> dict[str, Any]:
        dedupe_key = _sha256({"schema": "hawking.ascension.telegram_dedupe.v1", "fingerprint": event["fingerprint"]})
        events = state.get("events") if isinstance(state.get("events"), Mapping) else {}
        old = events.get(dedupe_key)
        if isinstance(old, Mapping) and old.get("lifecycle") in {"DELIVERED", "SEND_STARTED", "AMBIGUOUS_BLOCKED"}:
            return dict(state)
        if len(events) >= MAX_TRACKED_EVENTS:
            raise TelegramNotifierError("notifier event ledger reached its safety limit")
        receipt_path = _receipt_path(self.paths, dedupe_key)
        current = self._record_send_started(state, dedupe_key=dedupe_key, event=event, receipt_path=receipt_path)
        message = _render(event)
        try:
            delivered = dict(self._sender(credentials[0], credentials[1], message))
        except Exception:
            receipt = _delivery_receipt(event=event, dedupe_key=dedupe_key, message=message, status="AMBIGUOUS_BLOCKED", delivery=None, reason="transport_or_response_validation_failed")
            _atomic_json(receipt_path, receipt)
            updated = dict(current)
            records = dict(updated["events"])
            records[dedupe_key] = {**dict(records[dedupe_key]), "lifecycle": "AMBIGUOUS_BLOCKED", "receipt_path": str(receipt_path), "recorded_at": _utc_now()}
            updated["events"] = records
            return _save_state(self.paths.state, updated)
        if delivered.get("bot_api_http_status") != 200 or not isinstance(delivered.get("message_id"), int):
            raise TelegramNotifierError("sender did not return a validated delivery receipt")
        receipt = _delivery_receipt(event=event, dedupe_key=dedupe_key, message=message, status="DELIVERED", delivery=delivered)
        _atomic_json(receipt_path, receipt)
        updated = dict(current)
        records = dict(updated["events"])
        records[dedupe_key] = {**dict(records[dedupe_key]), "lifecycle": "DELIVERED", "receipt_path": str(receipt_path), "recorded_at": _utc_now()}
        updated["events"] = records
        return _save_state(self.paths.state, updated)

    @staticmethod
    def _fingerprints(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        return {str(event["event_id"]) + ":" + str(event.get("model") or "global"): str(event["fingerprint"]) for event in events}

    def arm(self) -> dict[str, Any]:
        lock = self._locked()
        try:
            state, snapshot, credential = self._arm_locked()
            return {"armed": True, "sent": 0, "status_path": str(self.paths.status), "state_path": str(self.paths.state)}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _arm_locked(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Prime current facts without delivery; caller must hold ``self._locked``."""

        state = _load_state(self.paths.state)
        snapshot = self.snapshot()
        candidates = self._candidate_events(snapshot, state)
        updated = dict(state)
        updated["armed"] = True
        updated["baseline_fingerprints"] = self._fingerprints(candidates)
        updated["worker_health"] = {key: bool(snapshot["models"][key]["healthy"]) for key in ("qwen30", "qwen80")}
        updated["resource_emergency"] = bool(snapshot["resource"]["emergency"])
        state = _save_state(self.paths.state, updated)
        credential, _ = credential_status(
            reader=self._credential_reader,
            validate_remote=False,
            remote_validator=self._remote_validator,
        )
        self._write_status(state=state, snapshot=snapshot, credential=credential)
        return state, snapshot, credential

    def test_delivery(self) -> dict[str, Any]:
        lock = self._locked()
        try:
            state = _load_state(self.paths.state)
            if not state.get("armed"):
                # Arm silently before the single test: historical campaign facts
                # must never trigger a stale event blast.
                state, _, _ = self._arm_locked()
            prior = state.get("test_event")
            if isinstance(prior, Mapping) and prior.get("lifecycle") in {"DELIVERED", "SEND_STARTED", "AMBIGUOUS_BLOCKED"}:
                credential, _ = credential_status(
                    reader=self._credential_reader,
                    validate_remote=False,
                    remote_validator=self._remote_validator,
                )
                self._write_status(state=state, snapshot=self.snapshot(), credential=credential)
                return {"sent": False, "reason": "test_event_already_recorded", "test_event": dict(prior)}
            credential, credentials = credential_status(
                reader=self._credential_reader,
                validate_remote=True,
                remote_validator=self._remote_validator,
            )
            if credentials is None or credential.get("bot_api_validated") is not True:
                self._write_status(state=state, snapshot=self.snapshot(), credential=credential, last_error="Telegram credentials are not locally and remotely valid")
                return {"sent": False, "reason": "credentials_not_validated", "credential": credential}
            event = _event(
                "operator_path_test",
                model=None,
                summary="OPERATOR_PATH_TEST: Telegram is armed for Ascension milestones. This is a bounded delivery test, not a model, TPS, TG, or tournament claim.",
                evidence_path="physical/notifications/telegram",
                facts={"operator_path_test": True, "campaign": "ascension_physical"},
            )
            state = self._deliver(state, event, credentials)
            records = state.get("events") if isinstance(state.get("events"), Mapping) else {}
            record = next((value for value in records.values() if isinstance(value, Mapping) and value.get("fingerprint") == event["fingerprint"]), None)
            updated = dict(state)
            updated["test_event"] = dict(record) if isinstance(record, Mapping) else {"lifecycle": "UNKNOWN"}
            state = _save_state(self.paths.state, updated)
            self._write_status(state=state, snapshot=self.snapshot(), credential=credential)
            return {"sent": bool(isinstance(record, Mapping) and record.get("lifecycle") == "DELIVERED"), "test_event": dict(record) if isinstance(record, Mapping) else None}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def run_once(self) -> dict[str, Any]:
        lock = self._locked()
        try:
            state = _load_state(self.paths.state)
            snapshot = self.snapshot()
            credential, credentials = credential_status(
                reader=self._credential_reader,
                validate_remote=False,
                remote_validator=self._remote_validator,
            )
            candidates = self._candidate_events(snapshot, state)
            if not state.get("armed"):
                updated = dict(state)
                updated["armed"] = True
                updated["baseline_fingerprints"] = self._fingerprints(candidates)
                updated["worker_health"] = {key: bool(snapshot["models"][key]["healthy"]) for key in ("qwen30", "qwen80")}
                updated["resource_emergency"] = bool(snapshot["resource"]["emergency"])
                state = _save_state(self.paths.state, updated)
                self._write_status(state=state, snapshot=snapshot, credential=credential)
                return {"armed_now": True, "sent": 0, "reason": "baseline_primed"}
            baseline = state.get("baseline_fingerprints") if isinstance(state.get("baseline_fingerprints"), Mapping) else {}
            sent = 0
            for event in candidates:
                key = str(event["event_id"]) + ":" + str(event.get("model") or "global")
                if baseline.get(key) == event["fingerprint"]:
                    continue
                # Component champion updates receive both a material-change and
                # a time gate; every other milestone is deduped by receipt seal.
                if event["event_id"] == "qwen_physical_representation_frontier_improved":
                    prior = next((record for record in (state.get("events") or {}).values() if isinstance(record, Mapping) and record.get("event_id") == event["event_id"] and record.get("model") == event.get("model") and record.get("lifecycle") == "DELIVERED"), None)
                    prior_at = _parse_timestamp(prior.get("recorded_at")) if isinstance(prior, Mapping) else None
                    if prior_at is not None and self._clock() - prior_at < FRONTIER_MIN_INTERVAL_SECONDS:
                        continue
                if credentials is None:
                    continue
                existing_record = next(
                    (
                        record
                        for record in (state.get("events") or {}).values()
                        if isinstance(record, Mapping) and record.get("fingerprint") == event["fingerprint"]
                    ),
                    None,
                )
                state = self._deliver(state, event, credentials)
                delivered_record = next((record for record in (state.get("events") or {}).values() if isinstance(record, Mapping) and record.get("fingerprint") == event["fingerprint"]), None)
                if (
                    not isinstance(existing_record, Mapping)
                    and isinstance(delivered_record, Mapping)
                    and delivered_record.get("lifecycle") == "DELIVERED"
                ):
                    sent += 1
            updated = dict(state)
            updated["worker_health"] = {key: bool(snapshot["models"][key]["healthy"]) for key in ("qwen30", "qwen80")}
            updated["resource_emergency"] = bool(snapshot["resource"]["emergency"])
            state = _save_state(self.paths.state, updated)
            self._write_status(state=state, snapshot=snapshot, credential=credential)
            return {"armed_now": False, "sent": sent, "candidate_events": len(candidates)}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def watch(self, *, idle_seconds: float) -> int:
        if idle_seconds <= 0:
            raise TelegramNotifierError("idle_seconds must be positive")
        stopping = False

        def stop(_signal: int, _frame: Any) -> None:
            nonlocal stopping
            stopping = True

        old_term = signal.signal(signal.SIGTERM, stop)
        old_int = signal.signal(signal.SIGINT, stop)
        try:
            while not stopping:
                try:
                    self.run_once()
                except Exception:
                    # The durable status is best effort here; never let a
                    # notification bug alter any campaign process.
                    pass
                if not stopping:
                    time.sleep(idle_seconds)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "arm", "test", "run", "watch"))
    parser.add_argument("--physical-root", type=Path, default=PHYSICAL_ROOT)
    parser.add_argument("--notifier-root", type=Path, default=NOTIFIER_ROOT)
    parser.add_argument("--idle-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = NotifierPaths(args.physical_root.resolve(), args.notifier_root.resolve())
    notifier = AscensionTelegramNotifier(paths=paths)
    try:
        if args.command == "status":
            state = _load_state(paths.state) if paths.state.exists() else _initial_state(_utc_now())
            credential, _ = credential_status(validate_remote=False)
            snapshot = notifier.snapshot()
            notifier._write_status(state=state, snapshot=snapshot, credential=credential)
            result: Mapping[str, Any] = _read_json(paths.status) or {}
        elif args.command == "arm":
            result = notifier.arm()
        elif args.command == "test":
            result = notifier.test_delivery()
        elif args.command == "run":
            result = notifier.run_once()
        else:
            return notifier.watch(idle_seconds=args.idle_seconds)
    except (TelegramNotifierError, OSError, ValueError) as exc:
        # Deliberately do not include exception text: a platform exception may
        # echo a request URL containing a bot token.
        print("ASCENSION_TELEGRAM_NOTIFIER_REFUSED", file=sys.stderr)
        return 75
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
