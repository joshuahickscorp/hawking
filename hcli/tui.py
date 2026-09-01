from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from typing import Any, Callable, Dict, List, Optional

from .events import Event, EventBus

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"reasoning_content\s*[:=].*?(?=\n\n|$)", re.DOTALL)
_RAW_PARENT_RE = re.compile(r"RAW PARENT", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"\{\s*\"tool\"\s*:.*?\}", re.DOTALL)
_HTTP_RE = re.compile(r"HTTP/[0-9.]+\s+[0-9]{3}", re.MULTILINE)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# The frame is measured in terminal COLUMNS, never in len(): the rows carry
# box-drawing characters, and model output carries CJK and emoji, which occupy
# two columns each. Counting code points is what made every row a different
# width and left the right border ragged.
_MIN_WIDTH = 40
_MAX_WIDTH = 100
_FALLBACK_WIDTH = 80


def sanitize_output(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _REASONING_RE.sub("", text)
    text = _RAW_PARENT_RE.sub("", text)
    text = _TOOL_JSON_RE.sub("", text)
    text = _HTTP_RE.sub("", text)
    return text.strip()


def _char_width(ch: str) -> int:
    # Combining marks, variation selectors and joiners render into the previous
    # cell; East Asian Wide/Fullwidth (and most emoji) take two.
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Columns `text` occupies once ANSI escapes are stripped."""
    # ponytail: per-codepoint, not grapheme clusters, so a ZWJ/skin-tone emoji
    # sequence over-counts. Reach for a grapheme segmenter if that ever shows.
    return sum(_char_width(ch) for ch in _ANSI_RE.sub("", text))


def _fit(text: str, width: int) -> List[str]:
    """Split one logical line into chunks of at most `width` display columns.

    Wraps on a space when there is one worth breaking at, hard-cuts otherwise,
    so no content is silently dropped and nothing exceeds the frame.
    """
    text = _ANSI_RE.sub("", text)
    rows: List[str] = []
    chunk = ""
    used = 0
    for ch in text:
        cw = _char_width(ch)
        if used + cw > width:
            cut = chunk.rfind(" ")
            if cut > width // 2:
                rows.append(chunk[:cut])
                chunk = chunk[cut + 1:]
            else:
                rows.append(chunk)
                chunk = ""
            used = display_width(chunk)
        chunk += ch
        used += cw
    rows.append(chunk)
    return rows


def frame_width() -> int:
    """Real terminal width, clamped to something readable, 80 when not a tty."""
    try:
        cols = shutil.get_terminal_size((_FALLBACK_WIDTH, 24)).columns
    except Exception:
        cols = _FALLBACK_WIDTH
    return max(_MIN_WIDTH, min(_MAX_WIDTH, cols))


def _rule(left: str, right: str, width: int, label: str = "") -> str:
    head = f"{left} {label} " if label else left
    return head + "─" * max(width - display_width(head) - 1, 0) + right


def _rows(text: str, width: int) -> List[str]:
    """Every visual line of `text`, each padded to exactly `width` columns."""
    out: List[str] = []
    for logical in text.expandtabs(4).splitlines() or [""]:
        for line in _fit(logical, width - 4):
            out.append("│ " + line + " " * (width - 4 - display_width(line)) + " │")
    return out


class TUI:
    def __init__(self, event_bus: EventBus, workspace: str, model_name: str = "local", runtime_count: int = 1):
        self.bus = event_bus
        self.workspace = workspace
        self.model_name = model_name
        self.runtime_count = runtime_count
        self.transcript: List[str] = []
        self.status: str = "idle"
        self._running = False
        self._prompt_fn: Optional[Callable[[str], str]] = None
        self._detect_prompt()

    def _detect_prompt(self):
        # prompt_toolkit on a pipe warns and emits bare carriage returns into
        # the frame, so it only gets the terminal.
        if not sys.stdin.isatty():
            self._prompt_fn = lambda msg: input(msg)
            return
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.history import InMemoryHistory
            self._prompt_fn = lambda msg: pt_prompt(msg, history=InMemoryHistory())
        except ImportError:
            self._prompt_fn = lambda msg: input(msg)

    def render_header(self) -> str:
        w = frame_width()
        info = f"{os.path.basename(self.workspace)}  {self.model_name}  {self.runtime_count} runtime(s)"
        return "\n".join([_rule("┌", "┐", w, "HCLI")] + _rows(info, w))

    def render_status(self) -> str:
        return "\n".join(_rows(self.status, frame_width()))

    def render_transcript(self) -> str:
        w = frame_width()
        if not self.transcript:
            return "\n".join(_rows("(no activity yet)", w))
        lines: List[str] = []
        for entry in self.transcript[-20:]:
            lines.extend(_rows(entry, w))
        return "\n".join(lines)

    def render(self) -> str:
        w = frame_width()
        parts = [
            self.render_header(),
            _rule("├", "┤", w),
            self.render_transcript(),
            _rule("├", "┤", w),
            self.render_status(),
            _rule("└", "┘", w),
        ]
        return "\n".join(parts)

    def _on_event(self, event: Event):
        data = event.data or {}
        if event.type == "activity_started":
            self.status = f"● {data.get('label', 'working')}"
            self.transcript.append(f"● {data.get('label', 'working')}")
        elif event.type == "activity_completed":
            self.status = "idle"
            self.transcript.append(f"✓ {data.get('label', 'done')}")
        elif event.type == "user_message":
            self.transcript.append(f"You: {data.get('text', '')}")
        elif event.type == "final_response":
            text = data.get("content") or data.get("text") or data.get("message") or ""
            self.transcript.append(sanitize_output(str(text)))
            st = str(data.get("status") or "")
            if st in {"failed", "cancelled", "error", "unverified"}:
                self.status = st
        elif event.type == "error":
            self.status = "error"
            self.transcript.append(f"✗ {data.get('message') or data.get('error') or 'error'}")
        elif event.type == "rollback":
            self.status = "error"
            self.transcript.append(f"✗ rollback: {data.get('reason', 'rollback')}")
        elif event.type == "validation_failed":
            self.status = "error"
            self.transcript.append("✗ validation failed")
        elif event.type == "goal_completed":
            st = str(data.get("status") or "")
            if st in {"failed", "cancelled"}:
                self.status = st
                self.transcript.append(f"✗ goal {st}")
        elif event.type == "steer_queued":
            self.transcript.append("✓ Steer queued")
        elif event.type == "transcript_cleared":
            # The transcript really is emptied; render_transcript() then shows
            # "(no activity yet)", so the operator still sees the clear happen.
            # The acknowledgement goes on the status line, not into the
            # transcript we were just asked to empty.
            self.transcript = []
            self.status = str(data.get("content") or "Transcript cleared")
        elif event.type == "warning":
            # Without this arm a handler that emitted only a warning renders
            # blank, which is how /help and /mission looked "broken".
            msg = data.get("message") or data.get("error") or "warning"
            self.transcript.append(f"! {msg}")

    def run(self, on_input: Callable[[str], None]) -> int:
        self.bus.subscribe(self._on_event)
        # One closed box per turn: printing a header and then unterminated
        # transcript lines is what left the frame hanging open.
        print(self.render())
        while True:
            try:
                user_input = self._prompt_fn("> ")
            except (EOFError, KeyboardInterrupt):
                print("\n[hcli] exiting")
                break
            text = user_input.strip()
            if not text:
                continue
            on_input(text)
            if text in ("/exit", "/quit"):
                break
            # The prompt leaves the cursor mid-line; start the box on its own.
            print("\n" + self.render())
        return 0
