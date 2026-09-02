"""One shared formatter for HCLI's live event stream.

TUI._on_event fused three concerns into one stateful method: text formatting,
phase/status bookkeeping, and terminal writes. That fusion is why the resident
watcher (hcli/agentos/resident.py:watch_resident) could not reuse any of it and
grew its own bare `msg`/`event` printer instead — the two surfaces render the
same mission differently by accident, not by design.

This module is the formatting half, pulled out pure: given one event dict, say
what a human should see. No self, no locks, no terminal writes, no imports
from hcli.tui (that keeps this module trivially embeddable in a read-only
viewer that must never import the TUI's turn-lifecycle/threading machinery).
Any caller — the interactive TUI or a resident watcher — supplies its own
state (current phase, open/closed turn, a screen to draw on) and calls these
three pure functions to decide what changed and what to print.

INVARIANT: model text/content must never be copied onto a phase or status
word. A phase is always one of a small closed vocabulary (see _status_word) —
free text (the model's own output, a tool's raw argument, an error message)
only ever reaches render_event's transcript lines, never event_phase's return
value. That boundary is what stops chain-of-thought from leaking onto a status
line a bystander glances at. Every phase-producing branch below runs its input
through _status_word, whose regex rejects anything containing whitespace or
longer than 64 chars and falls back to "working" — so even a hostile/garbled
`data["phase"]` cannot smuggle prose onto the phase word.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --- duplicated (not imported) from hcli.tui, kept byte-for-byte in sync ---
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"reasoning_content\s*[:=].*?(?=\n\n|$)", re.DOTALL)
_RAW_PARENT_RE = re.compile(r"RAW PARENT", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"\{\s*\"tool\"\s*:.*?\}", re.DOTALL)
_HTTP_RE = re.compile(r"HTTP/[0-9.]+\s+[0-9]{3}", re.MULTILINE)

_PASTE_ECHO_LIMIT = 600

_STATUS_ALIASES = {
    "thinking": "thinking",
    "model": "thinking",
    "model_call": "thinking",
    "working": "working",
    "evidence": "evidence",
    "gathering": "evidence",
    "evidence_gathering": "evidence",
    "compiling": "compiling",
    "validating": "validating",
    "mutating": "mutating",
    "idle": "idle",
}
_STATUS_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_TERMINAL_STATUSES = {"error", "failed", "cancelled", "unverified"}


def _sanitize_output(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _REASONING_RE.sub("", text)
    text = _RAW_PARENT_RE.sub("", text)
    text = _TOOL_JSON_RE.sub("", text)
    text = _HTTP_RE.sub("", text)
    return text.strip()


def _summarize_paste(text: str, limit: int = _PASTE_ECHO_LIMIT) -> str:
    body = text or ""
    if len(body) <= limit:
        return body
    lines = body.splitlines() or [""]
    head = " ".join(lines[0].split())
    if len(head) > 72:
        head = head[:71].rstrip() + "…"
    return f"[pasted text, {len(body):,} chars, {len(lines):,} lines] {head}"


def _status_word(raw: Any) -> str:
    """Map an event label to a short status word. Never pass through prose."""
    text = str(raw or "").replace("…", "").strip()
    key = text.lower().replace("-", "_").replace(" ", "_")
    if key in _STATUS_ALIASES:
        return _STATUS_ALIASES[key]
    if text and _STATUS_WORD_RE.fullmatch(text):
        return text
    return "working"


def _tool_name(data: Dict[str, Any]) -> str:
    return str(data.get("tool") or data.get("name") or "tool").strip() or "tool"


def _short_val(v: Any, limit: int = 24) -> str:
    s = str(v)
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def _short_args(data: Dict[str, Any], limit: int = 60) -> str:
    raw = data.get("args")
    if raw is None:
        raw = data.get("arguments")
    if raw is None:
        return ""
    if isinstance(raw, dict):
        text = ", ".join(f"{k}={_short_val(v)}" for k, v in raw.items())
    else:
        text = str(raw)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _tool_outcome(data: Dict[str, Any]) -> str:
    outcome = "ok" if bool(data.get("ok")) else "failed"
    elapsed = data.get("elapsed_s")
    extra = f"  {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
    return f"{outcome}{extra}"


def _int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _lines(raw: str) -> List[str]:
    """The one choke point every transcript line passes through, mirroring
    TUI._note: sanitize, then drop it entirely if that leaves nothing."""
    line = _sanitize_output(str(raw or ""))
    return [line] if line else []


def _safe_event(event: Any):
    if not isinstance(event, dict):
        return None, {}
    etype = event.get("type")
    data = event.get("data")
    if not isinstance(data, dict):
        data = {}
    return etype, data


def render_event(event: dict) -> List[str]:
    """Display lines for one event dict {"type": str, "data": dict}.

    Returns [] for events that produce no visible output (and for anything
    unrecognized or malformed — this never raises).
    """
    etype, data = _safe_event(event)

    if etype == "activity_completed":
        return _lines(f"✓ {_status_word(data.get('label') or 'done')}")
    if etype == "user_message":
        return _lines(f"You: {_summarize_paste(str(data.get('text', '')))}")
    if etype == "runtime_loading":
        return _lines(f"… loading resident: {str(data.get('model') or 'resident')}")
    if etype == "runtime_ready":
        admitted = data.get("admitted")
        suffix = f" ({admitted} admitted)" if admitted is not None else ""
        return _lines(f"✓ resident ready{suffix}")
    if etype == "final_response":
        text = data.get("content") or data.get("text") or data.get("message") or ""
        return _lines(_sanitize_output(str(text)))
    if etype == "error":
        return _lines(f"✗ {data.get('message') or data.get('error') or 'error'}")
    if etype == "rollback":
        return _lines(f"✗ rollback: {data.get('reason', 'rollback')}")
    if etype == "validation_failed":
        return _lines("✗ validation failed")
    if etype == "goal_completed":
        st = str(data.get("status") or "")
        if st in {"failed", "cancelled"}:
            return _lines(f"✗ goal {st}")
        return []
    if etype == "steer_queued":
        return _lines("✓ Steer queued")
    if etype == "bank_queued":
        return _lines(f"▣ Banked {data.get('id')}: {_clip(data.get('goal'), 100)}")
    if etype == "bank_started":
        return _lines(f"▶ Bank starting {data.get('id')}: {_clip(data.get('goal'), 100)}")
    if etype == "bank_finished":
        return _lines(f"✓ Bank finished {data.get('id')} status={data.get('status')}")
    if etype == "bank_dropped":
        return _lines(f"✓ Bank dropped {data.get('id')}")
    if etype == "bank_cleared":
        return _lines(f"✓ Bank cleared {data.get('removed', 0)} goal(s)")
    if etype == "warning":
        return _lines(f"! {data.get('message') or data.get('error') or 'warning'}")
    if etype == "evidence_gathering_finished":
        n = data.get("file_count")
        if n is None:
            n = len(data.get("files") or data.get("evidence_files") or [])
        return _lines(f"evidence  {_int_or(n)} files")
    if etype == "goal_compiled":
        n = data.get("workunits")
        if n is None:
            n = data.get("unit_count")
        return _lines(f"compiled  {_int_or(n)} units")
    # Tool lines shaped like Claude Code: the call, then its outcome
    # underneath. Args/outcome are short and structured (name, ok, elapsed) —
    # never the tool's raw output text — so nothing free-form lands here.
    if etype == "tool_call_started":
        return _lines(f"⏺ {_tool_name(data)}({_short_args(data)})")
    if etype in ("tool_call_finished", "tool_invoked"):
        # Pure function, no cross-event memory: unlike TUI._on_event (which
        # tracks _suppress_tool_invoked to print this outcome only once), a
        # caller that subscribes to both tool_call_finished and tool_invoked
        # must do its own dedup — this always renders the outcome line.
        # _lines()/_sanitize_output strips, which would eat the leading
        # two-space indent that makes "⎿" read as nested under the call
        # line above it — sanitize the outcome text alone, then indent.
        outcome = _sanitize_output(_tool_outcome(data))
        return [f"  ⎿ {outcome}"] if outcome else []
    # activity_started, workunit_started/completed, transcript_cleared,
    # evidence_gathering_started, model_call_started/finished, heartbeat,
    # validation_started, mutation_prepared, and any unknown type: no
    # transcript line (these only ever moved a status/phase word in the TUI).
    return []


def event_phase(event: dict) -> Optional[str]:
    """The phase word this event sets ("thinking", "validating", ...), or
    None if the event does not change phase.

    Only events that fed TUI._phase (via _set_phase or a direct assignment)
    are covered — events that instead wrote a decorated one-off string
    straight to TUI.status (runtime_loading, runtime_ready, workunit_*,
    bank_started, transcript_cleared) bypassed the phase machinery entirely
    and so report None here, matching the source they were lifted from.

    activity_completed's original guard ("only reset to idle if the current
    status is not already terminal") depends on state this pure function is
    not given; it always reports "idle" and leaves the terminal-guard to
    whatever stateful caller applies phase words to a real session.
    """
    etype, data = _safe_event(event)

    if etype == "activity_started":
        return _status_word(data.get("label") or "working")
    if etype == "activity_completed":
        return "idle"
    if etype == "evidence_gathering_started":
        return "evidence"
    if etype == "model_call_started":
        return "thinking"
    if etype == "tool_call_started":
        return _status_word(_tool_name(data))
    if etype == "validation_started":
        return "validating"
    if etype == "mutation_prepared":
        return "mutating"
    if etype in ("error", "rollback", "validation_failed"):
        return "error"
    if etype == "final_response":
        st = str(data.get("status") or "")
        return st if st in _TERMINAL_STATUSES else None
    if etype == "goal_completed":
        st = str(data.get("status") or "")
        return st if st in {"failed", "cancelled"} else None
    if etype == "heartbeat":
        # No access to "the phase before this heartbeat" here (that lived in
        # TUI._phase); fall back straight to "thinking" instead. Still runs
        # through _status_word, so a bogus/oversized data["phase"] cannot
        # leak free text onto the phase word.
        return _status_word(data.get("phase") or "thinking")
    return None


def is_terminal_event(event: dict) -> bool:
    """True for events that end a turn (goal_completed, error, rollback)."""
    etype, data = _safe_event(event)

    if etype in ("error", "rollback", "validation_failed"):
        return True
    if etype == "goal_completed":
        return str(data.get("status") or "") in {"failed", "cancelled"}
    if etype == "final_response":
        return str(data.get("status") or "") in _TERMINAL_STATUSES
    return False
