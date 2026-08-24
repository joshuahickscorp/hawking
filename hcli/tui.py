from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

from .events import Event, EventBus

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"reasoning_content\s*[:=].*?(?=\n\n|$)", re.DOTALL)
_RAW_PARENT_RE = re.compile(r"RAW PARENT", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"\{\s*\"tool\"\s*:.*?\}", re.DOTALL)
_HTTP_RE = re.compile(r"HTTP/[0-9.]+\s+[0-9]{3}", re.MULTILINE)


def sanitize_output(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _REASONING_RE.sub("", text)
    text = _RAW_PARENT_RE.sub("", text)
    text = _TOOL_JSON_RE.sub("", text)
    text = _HTTP_RE.sub("", text)
    return text.strip()


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
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.history import InMemoryHistory
            self._prompt_fn = lambda msg: pt_prompt(msg, history=InMemoryHistory())
        except ImportError:
            self._prompt_fn = lambda msg: input(msg)

    def render_header(self) -> str:
        return f"┌ HCLI ───────────────────────────────────────────────────────────────┐\n│ {os.path.basename(self.workspace)}  {self.model_name}  {self.runtime_count} runtime(s) │"

    def render_status(self) -> str:
        return f"│ {self.status} │"

    def render_transcript(self) -> str:
        if not self.transcript:
            return "│ (no activity yet) │"
        lines = []
        for line in self.transcript[-20:]:
            lines.append(f"│ {line} │")
        return "\n".join(lines)

    def render_input(self) -> str:
        return "├────────────────────────────────────────────────────────────────────┤\n│ > "

    def render(self) -> str:
        parts = [self.render_header(), "├" + "─" * 68 + "┤", self.render_transcript(), self.render_status(), self.render_input()]
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
        print(self.render_header())
        print("├" + "─" * 68 + "┤")
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
            print(self.render_transcript())
        return 0