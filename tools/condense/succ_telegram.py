#!/usr/bin/env python3.12
"""Successor Telegram service: first-class notification subsystem (master goal 14).

The eco_* layer had only a one-shot status formatter (eco_status). The successor
control plane needs a real notification service: a durable, deduplicated, reboot-safe
event notifier that speaks for every required successor lifecycle event (14.1), keeps
its content honest and terse (14.2), and is reliable under crashes and outages (14.3).

Design boundaries (non-interference is load-bearing):
  - OWN state store under reports/condense/event_horizon_successor/telegram/ . It never
    touches the campaign notifier's delivered-event ledger under doctor_v5_unbound/ .
  - It REUSES the campaign notifier's hardened send primitives (_send / _telegram /
    _keychain_get, and the Keychain service names) by loading that module by file path,
    exactly as eco_status._default_sender does. It never re-implements the wire format
    and never accepts a token or chat id as an argument.
  - A send is observable but NEVER evidence. A delivery failure returns a status object,
    it does not raise into the caller. The successor's state machine must be able to fire
    a notification from any transition without the notification being able to abort it.

Reliability (14.3):
  - Every notification is idempotent, keyed by a dedup event_id = hash(kind + salient
    fields). A persisted delivered-ledger plus a monotonic notification cursor survive a
    reboot, so a restarted controller replaying its transitions does not flood the chat.
  - Sends use bounded-backoff retry with an injectable sleeper (no real sleep in tests).
  - The waiting_old_release heartbeat is rate limited by a coarse time bucket, so a
    controller that waits for days emits at most one heartbeat per bucket.
  - The sender is injectable and there is a dry-run formatter (compose_event), so tests
    exercise composition, redaction, idempotency, and backoff with no network or Keychain.
  - Delivery metadata (message id, timestamp, attempts) is stored; secrets never are.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, hash_value, now_iso, atomic_write_json,
    read_json_safe, repo_root,
)

SCHEMA = "hawking.successor.telegram.v1"
MAX_MESSAGE_CHARS = 4000

# Substrings that mark a context key as secret bearing. compose_event redacts these
# before the value can reach message text, the machine summary, or the dedup key.
_SECRET_HINTS = (
    "token", "secret", "password", "passwd", "api_key", "apikey", "chat_id",
    "credential", "authorization", "cookie", "bearer",
)


class TelegramServiceError(EcoError):
    """Fail-closed error in the successor Telegram service (not a delivery failure)."""


# ── configuration ──────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Config:
    state_root: Path
    retry_max_attempts: int = 4
    retry_base_seconds: float = 1.0
    retry_cap_seconds: float = 30.0
    heartbeat_bucket_seconds: int = 3600
    max_message_chars: int = MAX_MESSAGE_CHARS

    @property
    def state_path(self) -> Path:
        return self.state_root / "telegram" / "state.json"


def successor_root() -> Path:
    """The successor-only report/state namespace. Deliberately NOT the campaign root and
    NOT the eco_* frontier_eco root, so nothing here can touch campaign-owned state."""
    return repo_root() / "reports" / "condense" / "event_horizon_successor"


def default_config(state_root: str | os.PathLike[str] | None = None) -> Config:
    sroot = Path(state_root) if state_root else successor_root()
    return Config(state_root=sroot)


# ── event catalog (14.1): kind -> icon, label, salient dedup fields, provisional footer ──
# salient = the fields that define the event's identity for deduplication. Two emits with
# the same kind and the same salient values are the same notification.
EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "controller_installed":      {"icon": "🛰", "label": "controller installed", "salient": ("generation", "controller_tree_sha256")},
    "real_test_delivered":       {"icon": "✅", "label": "real test delivered", "salient": ("generation",)},
    "watcher_started":           {"icon": "👁", "label": "watcher started", "salient": ("generation", "campaign_root")},
    "waiting_old_release_heartbeat": {"icon": "⏳", "label": "waiting for old release", "salient": ("bucket",)},
    "release_gates_changed":     {"icon": "🚦", "label": "release gates changed", "salient": ("terminal", "reporter_sealed", "quiescent", "signed")},
    "transition_armed":          {"icon": "🔧", "label": "transition armed", "salient": ("from_state", "to_state", "cause_seq"), "provisional": True},
    "transition_executed":       {"icon": "➡️", "label": "transition executed", "salient": ("from_state", "to_state", "event_seq"), "provisional": True},
    "transition_refused":        {"icon": "🛑", "label": "transition refused", "salient": ("from_state", "to_state", "reason")},
    "queue_summary":             {"icon": "📋", "label": "queue summary", "salient": ("running", "pending", "terminal", "digest_at")},
    "next_experiment":           {"icon": "🎯", "label": "next experiment", "salient": ("candidate_identity",)},
    "admission_pass":            {"icon": "🟢", "label": "resource admission pass", "salient": ("candidate_identity", "reservation_id")},
    "admission_fail":            {"icon": "🔴", "label": "resource admission fail", "salient": ("candidate_identity", "reason")},
    "experiment_started":        {"icon": "🚀", "label": "experiment started", "salient": ("experiment_id",)},
    "experiment_checkpointed":   {"icon": "💾", "label": "experiment checkpointed", "salient": ("experiment_id", "checkpoint_seq")},
    "experiment_completed":      {"icon": "🏁", "label": "experiment completed", "salient": ("experiment_id", "result_sha256")},
    "experiment_failed":         {"icon": "💥", "label": "experiment failed", "salient": ("experiment_id", "reason")},
    "experiment_retired":        {"icon": "🗄", "label": "experiment retired", "salient": ("experiment_id", "reason")},
    "event_horizon_bracket_moved": {"icon": "📉", "label": "Event Horizon bracket moved", "salient": ("model_label", "bracket", "old_bpw", "new_bpw"), "provisional": True},
    "new_extreme":               {"icon": "🥇", "label": "new EXTREME", "salient": ("model_label", "bpw", "result_sha256"), "provisional": True},
    "new_balanced":              {"icon": "🥈", "label": "new BALANCED", "salient": ("model_label", "bpw", "result_sha256"), "provisional": True},
    "new_fidelity":              {"icon": "🥉", "label": "new FIDELITY", "salient": ("model_label", "bpw", "result_sha256"), "provisional": True},
    "resource_alert":            {"icon": "⚠️", "label": "resource alert", "salient": ("kind_detail", "value", "threshold")},
    "gc_plan":                   {"icon": "🧹", "label": "GC plan", "salient": ("plan_sha256", "reclaimable_gb")},
    "gc_completed":              {"icon": "🧼", "label": "GC completed", "salient": ("plan_sha256", "reclaimed_gb")},
    "drain":                     {"icon": "🚿", "label": "drain", "salient": ("reason",)},
    "resume":                    {"icon": "🔁", "label": "resume", "salient": ("resumed_state", "event_cursor_seq")},
    "crash":                     {"icon": "🔥", "label": "crash detected", "salient": ("component", "detail")},
    "transition_72b":            {"icon": "🧗", "label": "72B transition", "salient": ("phase", "state")},
    "readiness_120b_changed":    {"icon": "📶", "label": "120B readiness changed", "salient": ("ready", "reason")},
    "frontier_row_created":      {"icon": "🌌", "label": "frontier row created", "salient": ("label", "total_b", "regime")},
    "frontier_row_state_changed": {"icon": "🌠", "label": "frontier row state changed", "salient": ("label", "old_state", "new_state")},
    "daily_digest":              {"icon": "📰", "label": "daily digest", "salient": ("digest_date",)},
    "final_seal":                {"icon": "🔒", "label": "final seal", "salient": ("generation", "seal_sha256")},
    # ── Gravity (sub-bit-first law) notifications (master goal section 18) ──
    "gravity_policy_enabled":    {"icon": "🪐", "label": "Gravity policy enabled", "salient": ("parent", "policy_version")},
    "gravity_start_rate":        {"icon": "🪂", "label": "Gravity start rate", "salient": ("parent", "stress_rate")},
    "gravity_tournament_started":{"icon": "🎲", "label": "sub-bit representation tournament", "salient": ("parent", "rate")},
    "gravity_feasibility_completed": {"icon": "🧪", "label": "Gravity feasibility completed", "salient": ("parent", "rate", "tier")},
    "gravity_diagnosis":         {"icon": "🩺", "label": "Gravity diagnosis", "salient": ("parent", "rate", "diagnosis")},
    "gravity_rescue_started":    {"icon": "🚑", "label": "Doctor rescue started", "salient": ("parent", "rate", "treatment")},
    "gravity_rescue_result":     {"icon": "💊", "label": "Doctor rescue result", "salient": ("parent", "rate", "result"), "provisional": True},
    "gravity_byte_allocation_changed": {"icon": "⚖️", "label": "byte allocation changed", "salient": ("parent", "rate", "base_bpw", "doctor_bpw")},
    "gravity_representation_changed": {"icon": "🔀", "label": "representation changed", "salient": ("parent", "rate", "family")},
    "gravity_descend":           {"icon": "🪨", "label": "descending to lower rate", "salient": ("parent", "from_rate", "to_rate")},
    "gravity_ascend":            {"icon": "🧗", "label": "ascending to higher rate", "salient": ("parent", "from_rate", "to_rate")},
    "gravity_escape_requested":  {"icon": "🛎", "label": "Escape Receipt requested", "salient": ("parent", "rate")},
    "gravity_escape_decision":   {"icon": "🎟", "label": "Escape Receipt decision", "salient": ("parent", "decision", "receipt_sha256"), "provisional": True},
    "gravity_first_pass":        {"icon": "🌅", "label": "first passing rate", "salient": ("parent", "rate"), "provisional": True},
    "gravity_lower_boundary":    {"icon": "📛", "label": "lower failing boundary", "salient": ("parent", "rate")},
    "gravity_event_horizon":     {"icon": "🕳", "label": "Event Horizon", "salient": ("parent", "rate", "whole_bpw"), "provisional": True},
    "gravity_bpw_composition":   {"icon": "🧾", "label": "physical BPW composition", "salient": ("parent", "whole_bpw", "base_bpw", "doctor_bpw")},
    "gravity_queue_eta":         {"icon": "📅", "label": "Gravity queue and ETA", "salient": ("parent", "next_probe", "eta_range")},
    "gravity_daily":             {"icon": "🗞", "label": "Gravity daily summary", "salient": ("digest_date",)},
}

_PROVISIONAL_FOOTER = "Provisional until the signed physical release gate."


def _is_secret_key(key: str) -> bool:
    low = key.lower()
    return any(hint in low for hint in _SECRET_HINTS)


def _redact(value: Any) -> Any:
    """Recursively replace secret-bearing values so they can never reach text, summary,
    or the dedup key. Applied to a copy; the caller's context is not mutated."""
    if isinstance(value, dict):
        return {k: ("[redacted]" if _is_secret_key(str(k)) else _redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def _human(field: str) -> str:
    return field.replace("_", " ")


def compose_event(event_kind: str, context: dict[str, Any] | None) -> dict[str, Any]:
    """Dry-run formatter. Returns {schema, kind, text, event_id, summary, composed_at}.

    text is terse, house style (no em or en dashes). event_id is the dedup key:
    hash of the kind plus the redacted salient fields, so it is stable across processes
    and never carries a secret. Secrets in context are redacted everywhere.
    """
    if not isinstance(event_kind, str) or not event_kind:
        raise TelegramServiceError("event_kind must be a non-empty string")
    ctx = context if isinstance(context, dict) else {}
    red = _redact(ctx)
    entry = EVENT_CATALOG.get(event_kind, {"icon": "🛰", "label": event_kind.replace("_", " "),
                                           "salient": tuple(sorted(red))})
    salient_fields = tuple(entry.get("salient", ()))
    salient = {f: red.get(f) for f in salient_fields}

    lines = [f"{entry['icon']} Hawking successor: {entry['label']}"]
    # Emit salient lines first (identity), then any other non-secret context lines.
    shown: set[str] = set()
    for f in salient_fields:
        if f in red and red[f] is not None:
            lines.append(f"{_human(f)}: {red[f]}")
            shown.add(f)
    for k in sorted(red):
        if k in shown or red[k] is None:
            continue
        lines.append(f"{_human(k)}: {red[k]}")
    if entry.get("provisional"):
        lines.append(_PROVISIONAL_FOOTER)
    text = "\n".join(str(x) for x in lines)[:MAX_MESSAGE_CHARS]

    event_id = hash_value({"kind": event_kind, "salient": salient})
    return {
        "schema": SCHEMA,
        "kind": event_kind,
        "text": text,
        "event_id": event_id,
        "summary": {"kind": event_kind, "salient": salient, "context": red},
        "composed_at": now_iso(),
    }


# ── notifier reuse (load the hardened campaign primitives by file path) ──────────────────
_NOTIFIER: Any = None


def _load_notifier() -> Any:
    global _NOTIFIER
    if _NOTIFIER is not None:
        return _NOTIFIER
    import importlib.util
    mod_path = Path(_HERE) / "doctor_v5_telegram_rung_notifier.py"
    spec = importlib.util.spec_from_file_location("doctor_v5_telegram_rung_notifier", mod_path)
    if spec is None or spec.loader is None:
        raise TelegramServiceError("cannot load the campaign notifier primitives")
    notifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notifier)
    _NOTIFIER = notifier
    return notifier


def _default_sender(text: str) -> dict[str, Any]:
    """Send via the campaign notifier's hardened, Keychain-backed sender. Raises
    NotifierError when creds are absent; emit() treats that as an observable send failure."""
    return _load_notifier()._send(text)  # type: ignore[attr-defined]


def _creds_available() -> bool:
    """True when both the bot token and chat id are present in Keychain. Never returns
    or logs the values themselves."""
    try:
        notifier = _load_notifier()
        token = notifier._keychain_get(notifier.TOKEN_SERVICE)
        chat = notifier._keychain_get(notifier.CHAT_SERVICE)
        return bool(token) and bool(chat)
    except Exception:  # noqa: BLE001 - probe must never raise into status/emit
        return False


# ── persisted, self-sealed state (survives reboot) ──────────────────────────────────────
def _empty_state() -> dict[str, Any]:
    stamp = now_iso()
    state = {
        "schema": SCHEMA,
        "created_at": stamp,
        "updated_at": stamp,
        "notification_cursor": 0,
        "delivered": {},          # event_id -> {kind, sent_at, receipt, attempts, seq}
        "heartbeat_buckets": {},  # str(bucket) -> {sent_at}
        "last_attempt": None,     # {kind, event_id, at}
        "last_result": None,      # sent | already-sent | send-failed
    }
    return seal_field(state, "state_sha256")


def _load_state(cfg: Config) -> dict[str, Any]:
    if not cfg.state_path.exists():
        return _empty_state()
    value = read_json_safe(cfg.state_path)
    if value.get("schema") != SCHEMA or not sealed(value, "state_sha256"):
        # Corrupt or tampered state: fail closed. We refuse to send rather than risk a
        # flood from a state whose delivered-ledger we cannot trust.
        raise TelegramServiceError("successor telegram state identity is invalid")
    return value


def _save_state(cfg: Config, state: dict[str, Any]) -> None:
    body = {k: v for k, v in state.items() if k != "state_sha256"}
    body["updated_at"] = now_iso()
    atomic_write_json(cfg.state_path, seal_field(body, "state_sha256"))


# ── emit (idempotent, bounded-backoff, non-raising) ─────────────────────────────────────
def emit(event_kind: str, context: dict[str, Any] | None = None, *,
         cfg: Config | None = None,
         sender: Callable[[str], dict[str, Any]] = _default_sender,
         force: bool = False,
         sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Compose and deliver one successor notification, idempotently.

    - Dedup: if this event_id was already delivered and not force, returns already-sent
      without resending (this is what makes reboot-replay safe).
    - Retry: on a raising sender, retries up to cfg.retry_max_attempts with bounded
      backoff (base * 2**n, capped), sleeping via the injected sleeper.
    - Non-raising: any failure (send exhaustion, corrupt state, compose error) returns a
      status dict. This function never raises into the caller's evidence path.
    """
    cfg = cfg or default_config()
    try:
        composed = compose_event(event_kind, context)
    except TelegramServiceError as exc:
        return {"status": "compose-failed", "kind": event_kind, "sent": 0,
                "error": str(exc)}
    event_id = composed["event_id"]

    try:
        state = _load_state(cfg)
    except TelegramServiceError as exc:
        # Fail closed: do not send against untrusted state.
        return {"status": "send-failed", "kind": event_kind, "event_id": event_id,
                "sent": 0, "error": str(exc)}

    delivered = state.setdefault("delivered", {})
    if event_id in delivered and not force:
        state["last_attempt"] = {"kind": event_kind, "event_id": event_id, "at": now_iso()}
        state["last_result"] = "already-sent"
        try:
            _save_state(cfg, state)
        except OSError:
            pass
        return {"status": "already-sent", "kind": event_kind, "event_id": event_id, "sent": 0}

    attempts = max(1, int(cfg.retry_max_attempts))
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            receipt = sender(composed["text"])
        except Exception as exc:  # noqa: BLE001 - a send failure is data, never a crash
            last_error = type(exc).__name__
            if attempt < attempts:
                delay = min(cfg.retry_base_seconds * (2 ** (attempt - 1)), cfg.retry_cap_seconds)
                try:
                    sleeper(delay)
                except Exception:  # noqa: BLE001
                    pass
            continue
        # success: record delivery metadata (never the secret), advance the cursor
        cursor = int(state.get("notification_cursor", 0)) + 1
        delivered[event_id] = {
            "kind": event_kind,
            "sent_at": now_iso(),
            "receipt": _redact(receipt) if isinstance(receipt, dict) else receipt,
            "attempts": attempt,
            "seq": cursor,
        }
        state["notification_cursor"] = cursor
        state["last_attempt"] = {"kind": event_kind, "event_id": event_id, "at": now_iso()}
        state["last_result"] = "sent"
        try:
            _save_state(cfg, state)
        except OSError as exc:
            # Delivered but could not persist the ledger: report it. A duplicate on the
            # next replay is preferable to a lost delivery record, but we surface it.
            return {"status": "sent-unpersisted", "kind": event_kind, "event_id": event_id,
                    "sent": 1, "attempts": attempt, "error": type(exc).__name__}
        return {"status": "sent", "kind": event_kind, "event_id": event_id, "sent": 1,
                "attempts": attempt, "seq": cursor, "receipt": delivered[event_id]["receipt"]}

    # exhausted all attempts: observable failure, not a crash
    state["last_attempt"] = {"kind": event_kind, "event_id": event_id, "at": now_iso()}
    state["last_result"] = "send-failed"
    try:
        _save_state(cfg, state)
    except OSError:
        pass
    return {"status": "send-failed", "kind": event_kind, "event_id": event_id, "sent": 0,
            "attempts": attempts, "error": last_error or "unknown"}


# ── heartbeat (rate limited by coarse time bucket) ──────────────────────────────────────
def heartbeat(context: dict[str, Any] | None = None, *,
              cfg: Config | None = None,
              sender: Callable[[str], dict[str, Any]] = _default_sender,
              clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Emit at most one waiting_old_release heartbeat per coarse time bucket.

    The bucket index is part of the event's salient fields, so emit() dedups naturally:
    a controller that waits for days sends one heartbeat per bucket, no more.
    """
    cfg = cfg or default_config()
    bucket = int(clock() // max(1, int(cfg.heartbeat_bucket_seconds)))
    ctx = dict(context or {})
    ctx["bucket"] = bucket
    result = emit("waiting_old_release_heartbeat", ctx, cfg=cfg, sender=sender)
    # Record the bucket for status/next_heartbeat, best effort.
    if result.get("status") == "sent":
        try:
            state = _load_state(cfg)
            state.setdefault("heartbeat_buckets", {})[str(bucket)] = {"sent_at": now_iso()}
            _save_state(cfg, state)
        except (TelegramServiceError, OSError):
            pass
    return {**result, "bucket": bucket}


# ── status (no secrets) ──────────────────────────────────────────────────────────────────
def telegram_status(cfg: Config | None = None, *,
                    creds_probe: Callable[[], bool] = _creds_available,
                    clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Report service health without ever printing a secret."""
    cfg = cfg or default_config()
    configured = bool(creds_probe())
    try:
        state = _load_state(cfg) if cfg.state_path.exists() else _empty_state()
        state_state = "ok"
    except TelegramServiceError:
        state = _empty_state()
        state_state = "corrupt-fail-closed"

    if not configured:
        service_state = "unconfigured"
    elif state.get("last_result") == "send-failed":
        service_state = "degraded"
    else:
        service_state = "configured"

    buckets = state.get("heartbeat_buckets") or {}
    next_heartbeat = None
    if buckets:
        try:
            last_bucket = max(int(b) for b in buckets)
            next_epoch = (last_bucket + 1) * max(1, int(cfg.heartbeat_bucket_seconds))
            import datetime as _dt
            next_heartbeat = _dt.datetime.fromtimestamp(
                next_epoch, _dt.timezone.utc).isoformat(timespec="seconds")
        except (ValueError, OverflowError, OSError):
            next_heartbeat = None
    else:
        next_heartbeat = "due-now"

    return {
        "schema": SCHEMA,
        "configured": configured,
        "service_state": service_state,
        "state_integrity": state_state,
        "last_attempt": state.get("last_attempt"),
        "last_result": state.get("last_result"),
        "notification_cursor": state.get("notification_cursor", 0),
        "delivered_count": len(state.get("delivered", {})),
        "cursor_path": str(cfg.state_path),
        "heartbeat_bucket_seconds": cfg.heartbeat_bucket_seconds,
        "next_heartbeat": next_heartbeat,
    }


# ── offline selftest ─────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    import tempfile
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(state_root=Path(d) / "event_horizon_successor",
                     retry_max_attempts=3, retry_base_seconds=0.0, retry_cap_seconds=0.0)

        sent_texts: list[str] = []

        def fake_sender(text: str) -> dict[str, Any]:
            sent_texts.append(text)
            return {"message_id": len(sent_texts), "sent_at": now_iso()}

        def no_sleep(_seconds: float) -> None:
            return None

        # 1. compose_event redacts secrets and never leaks them into text or event_id
        composed = compose_event("watcher_started", {
            "generation": "gen-1", "campaign_root": "/x/doctor_v5_ultra",
            "bot_token": "123456:SUPERSECRETVALUE", "chat_id": "999888",
        })
        if "SUPERSECRETVALUE" in composed["text"] or "999888" in composed["text"]:
            raise TelegramServiceError("secret leaked into composed text")
        if "SUPERSECRETVALUE" in composed["event_id"]:
            raise TelegramServiceError("secret leaked into event_id")
        if composed["summary"]["context"].get("bot_token") != "[redacted]":
            raise TelegramServiceError("secret not redacted in summary")
        results["redaction"] = True

        # 2. emit an event -> sent once
        ctx = {"experiment_id": "exp-001", "candidate_identity": "a" * 64}
        r1 = emit("experiment_started", ctx, cfg=cfg, sender=fake_sender, sleeper=no_sleep)
        if r1["status"] != "sent" or r1["sent"] != 1 or len(sent_texts) != 1:
            raise TelegramServiceError(f"first emit did not send once: {r1}")

        # 3. emit the same event again -> deduped (not resent)
        r2 = emit("experiment_started", ctx, cfg=cfg, sender=fake_sender, sleeper=no_sleep)
        if r2["status"] != "already-sent" or r2["sent"] != 0 or len(sent_texts) != 1:
            raise TelegramServiceError(f"dedup failed: {r2} texts={len(sent_texts)}")

        # 4. force -> resent
        r3 = emit("experiment_started", ctx, cfg=cfg, sender=fake_sender, sleeper=no_sleep,
                  force=True)
        if r3["status"] != "sent" or r3["sent"] != 1 or len(sent_texts) != 2:
            raise TelegramServiceError(f"force resend failed: {r3}")
        results["idempotent"] = True
        results["force_resend"] = True

        # 5. a sender that raises -> emit returns send-failed and does not crash; retries
        attempt_count = {"n": 0}

        def raising_sender(text: str) -> dict[str, Any]:
            attempt_count["n"] += 1
            raise RuntimeError("network down")

        r4 = emit("resource_alert", {"kind_detail": "swap", "value": 42.0, "threshold": 8.0},
                  cfg=cfg, sender=raising_sender, sleeper=no_sleep)
        if r4["status"] != "send-failed" or r4["sent"] != 0:
            raise TelegramServiceError(f"raising sender not handled: {r4}")
        if attempt_count["n"] != cfg.retry_max_attempts:
            raise TelegramServiceError(f"retry count wrong: {attempt_count['n']}")
        results["send_failed_non_raising"] = True
        results["bounded_retry"] = attempt_count["n"]

        # a failed event is NOT recorded as delivered, so a later success can still deliver
        r5 = emit("resource_alert", {"kind_detail": "swap", "value": 42.0, "threshold": 8.0},
                  cfg=cfg, sender=fake_sender, sleeper=no_sleep)
        if r5["status"] != "sent":
            raise TelegramServiceError(f"recovery after failure did not send: {r5}")

        # 6. heartbeat is rate limited by bucket: same bucket -> one send
        fixed_time = 1_000_000.0

        def clock_a() -> float:
            return fixed_time

        h1 = heartbeat({"reason": "campaign running"}, cfg=cfg, sender=fake_sender, clock=clock_a)
        h2 = heartbeat({"reason": "campaign running"}, cfg=cfg, sender=fake_sender, clock=clock_a)
        if h1["status"] != "sent" or h2["status"] != "already-sent":
            raise TelegramServiceError(f"heartbeat bucket dedup failed: {h1} {h2}")
        # next bucket -> sends again
        def clock_b() -> float:
            return fixed_time + cfg.heartbeat_bucket_seconds + 1

        h3 = heartbeat({"reason": "campaign running"}, cfg=cfg, sender=fake_sender, clock=clock_b)
        if h3["status"] != "sent":
            raise TelegramServiceError(f"next-bucket heartbeat did not send: {h3}")
        results["heartbeat_rate_limited"] = True

        # 7. reboot safety: a fresh state load sees the delivered ledger and dedups
        r6 = emit("experiment_started", ctx, cfg=cfg, sender=fake_sender, sleeper=no_sleep)
        if r6["status"] != "already-sent":
            raise TelegramServiceError("post-reboot dedup failed (ledger not durable)")

        # 8. corrupt state -> emit fails closed (does not send)
        raw = cfg.state_path.read_text()
        import json as _json
        doc = _json.loads(raw)
        doc["delivered"] = {}  # tamper without re-sealing
        cfg.state_path.write_text(_json.dumps(doc))
        before = len(sent_texts)
        r7 = emit("crash", {"component": "watcher", "detail": "oom"},
                  cfg=cfg, sender=fake_sender, sleeper=no_sleep)
        if r7["status"] != "send-failed" or len(sent_texts) != before:
            raise TelegramServiceError(f"corrupt state not fail-closed: {r7}")
        results["corrupt_state_fail_closed"] = True

        # rewrite valid state, then status (offline creds probe injected, no Keychain)
        atomic_write_json(cfg.state_path, _empty_state())
        st = telegram_status(cfg, creds_probe=lambda: False, clock=clock_a)
        if st["configured"] is not False or st["service_state"] != "unconfigured":
            raise TelegramServiceError(f"status must show unconfigured offline: {st}")
        if "SUPERSECRET" in _json.dumps(st):
            raise TelegramServiceError("status leaked a secret")
        results["status_no_secret"] = True

    return {"ok": True, **results}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Successor Telegram service (master goal 14).")
    ap.add_argument("--state-root", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--compose", metavar="KIND", default=None,
                    help="dry-run: compose_event for KIND (context as --ctx k=v ...)")
    ap.add_argument("--ctx", nargs="*", default=[], help="context key=value pairs for --compose")
    ap.add_argument("--emit", metavar="KIND", default=None, help="compose and send KIND")
    ap.add_argument("--go", action="store_true", help="required with --emit to actually send")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        sys.exit(0)

    cfg = default_config(args.state_root)

    if args.status:
        print(json.dumps(telegram_status(cfg), indent=2, sort_keys=True))
        sys.exit(0)

    if args.compose or args.emit:
        ctx: dict[str, Any] = {}
        for pair in args.ctx:
            if "=" in pair:
                k, v = pair.split("=", 1)
                ctx[k] = v
        kind = args.compose or args.emit
        if args.emit and args.go:
            print(json.dumps(emit(kind, ctx, cfg=cfg, force=args.force), indent=2, sort_keys=True))
        else:
            composed = compose_event(kind, ctx)
            print(composed["text"])
            print("\n--- (dry run; pass --emit KIND --go to deliver) ---")
        sys.exit(0)

    ap.print_help()
