from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional, TextIO

from .events import Event, EventBus
from .mission import mission_state_path
from .session_ledger import SessionLedger
from .stream_render import (
    _PASTE_ECHO_LIMIT,
    _TERMINAL_STATUSES,
    event_phase,
    is_terminal_event,
    render_event,
    sanitize_output,
    status_word,
    summarize_paste,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# The frame is measured in terminal COLUMNS, never in len(): the rows carry
# box-drawing characters, and model output carries CJK and emoji, which occupy
# two columns each. Counting code points is what made every row a different
# width and left the right border ragged.
_MIN_WIDTH = 40
_MAX_WIDTH = 100
_FALLBACK_WIDTH = 80


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


# Braille dots, Claude-Code-style live spinner. Plain ASCII columns (width 1
# each under _char_width, none are East-Asian-wide), so the \r-overwrite math
# in _write_status_live needs no special case for them.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TUI:
    def __init__(
        self,
        event_bus: EventBus,
        workspace: str,
        model_name: str = "local",
        runtime_count: int = 1,
        bank_snapshot_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        stream: Optional[TextIO] = None,
        tty: Optional[bool] = None,
        live_interval: float = 1.0,
        ledger: Optional[SessionLedger] = None,
    ):
        self.bus = event_bus
        self.workspace = workspace
        self.model_name = model_name
        self.runtime_count = runtime_count
        self.bank_snapshot_fn = bank_snapshot_fn
        self.transcript: List[str] = []
        self.status: str = "idle"
        self._running = False
        self._prompt_fn: Optional[Callable[[str], str]] = None
        self._stream: TextIO = sys.stdout if stream is None else stream
        if tty is None:
            tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._tty = bool(tty)
        self._live_interval = max(0.05, float(live_interval))
        self._print_lock = threading.Lock()
        self._in_turn = False
        self._stop_live = threading.Event()
        self._live_thread: Optional[threading.Thread] = None
        self._status_open = False
        self._status_width = 0
        self._phase = "idle"
        self._phase_t0 = time.monotonic()
        self._prompt_tokens: Optional[int] = None
        self._last_status_text = ""
        self._suppress_tool_invoked = False
        self._tool_calls_this_turn = 0
        # Set right before an auto-steered plain line is dispatched, holding
        # the note to show in place of the raw "You: ..." echo; also doubles
        # as the flag that tells the final_response arm to swallow /steer's
        # own "Steer queued" confirmation, since the note already said so.
        self._auto_steer_note: Optional[str] = None
        # One ledger for the life of the session, not one per turn: it is the
        # instance itself (`_last_offered`) that remembers what was already
        # offered so an unchanged dirty tree does not nag every turn.
        #
        # Injectable, and OFF unless a caller supplies one. A ledger built here
        # from `workspace` resolves an enclosing git repository, which under a
        # unit test is whatever tree the developer happens to be sitting in: a
        # TUI test then asserted against a transcript carrying this repo's own
        # "96 file(s) changed" line. A view must not read the state of a
        # repository nobody handed it.
        self._ledger = ledger
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
        bank_line = "Bank: unavailable"
        if self.bank_snapshot_fn is not None:
            try:
                snapshot = self.bank_snapshot_fn()
            except Exception:
                snapshot = {}
            if isinstance(snapshot, dict) and snapshot.get("available"):
                queued = int(snapshot.get("queued_count") or 0)
                running = int(snapshot.get("running_count") or 0)
                next_item = snapshot.get("next")
                next_goal = ""
                if isinstance(next_item, dict):
                    next_goal = " ".join(str(next_item.get("goal") or "").split())
                if len(next_goal) > 54:
                    next_goal = next_goal[:53].rstrip() + "…"
                bank_line = f"Bank: queued={queued} running={running}"
                if next_goal:
                    bank_line += f"  next: {next_goal}"
            elif isinstance(snapshot, dict):
                bank_line = f"Bank: unavailable ({snapshot.get('reason', 'unknown')})"
        return "\n".join(
            [_rule("┌", "┐", w, "HCLI")] + _rows(info, w) + _rows(bank_line, w)
        )

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

    def _set_phase(self, word: str) -> None:
        self._phase = status_word(word)
        self._phase_t0 = time.monotonic()
        self.status = self._format_status()

    def _format_status(self) -> str:
        # Never assembled from event text/content -- only the closed phase
        # vocabulary (status_word) and small counters land here. That is
        # the whole chain-of-thought-leak boundary; see _on_event.
        word = self._phase or "idle"
        if word == "idle" or word in _TERMINAL_STATUSES:
            return word
        elapsed = max(0, int(time.monotonic() - self._phase_t0))
        spin = _SPINNER_FRAMES[int(time.monotonic() * 5) % len(_SPINNER_FRAMES)]
        extra = ""
        if word == "thinking" and self._prompt_tokens:
            extra += f"  {self._prompt_tokens} tok"
        if self._tool_calls_this_turn:
            plural = "" if self._tool_calls_this_turn == 1 else "s"
            extra += f"  {self._tool_calls_this_turn} tool{plural}"
        return f"{spin} {word}… {elapsed}s{extra}  (ctrl-c to interrupt)"

    def _close_status_locked(self) -> None:
        if self._status_open:
            try:
                self._stream.write("\n")
                self._stream.flush()
            except BrokenPipeError:
                pass
            self._status_open = False
            self._status_width = 0

    def _write_raw(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except BrokenPipeError:
            pass

    def _println(self, text: str = "") -> None:
        with self._print_lock:
            self._close_status_locked()
            if text and not text.endswith("\n"):
                text = text + "\n"
            elif not text:
                text = "\n"
            self._write_raw(text)

    def _write_transcript_line(self, line: str) -> None:
        if not self._in_turn:
            return
        # Wrap instead of letting the terminal hard-wrap raggedly: _fit is
        # the same wrapper render_transcript() uses inside the box, so a
        # line that streams in live already matches the shape it will have
        # once the turn ends and the full frame reprints it. A line that
        # fits does one pass through _fit and comes back as itself.
        chunks = _fit(line, frame_width() - 2)
        with self._print_lock:
            self._close_status_locked()
            for i, chunk in enumerate(chunks):
                text = chunk if i == 0 else "  " + chunk
                self._write_raw(text + "\n")

    def _write_status_live(self) -> None:
        text = self._format_status()
        self.status = text
        if not self._in_turn:
            return
        with self._print_lock:
            if not self._tty:
                self._write_raw(text + "\n")
                self._last_status_text = text
                return
            pad = max(0, self._status_width - display_width(text))
            self._write_raw("\r" + text + (" " * pad))
            self._status_width = display_width(text)
            self._status_open = True
            self._last_status_text = text

    def _note(self, line: str) -> None:
        line = sanitize_output(str(line or ""))
        if not line:
            return
        self.transcript.append(line)
        self._write_transcript_line(line)

    def _begin_turn(self) -> None:
        self._end_turn()
        self._in_turn = True
        self._stop_live = threading.Event()
        self._status_open = False
        self._status_width = 0
        self._prompt_tokens = None
        self._suppress_tool_invoked = False
        self._tool_calls_this_turn = 0
        self._auto_steer_note = None
        self._set_phase("working")
        self._live_thread = threading.Thread(
            target=self._live_loop,
            name="hcli-tui-live",
            daemon=True,
        )
        self._live_thread.start()

    def _end_turn(self) -> None:
        self._stop_live.set()
        thread = self._live_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._live_thread = None
        self._in_turn = False
        with self._print_lock:
            self._close_status_locked()
        if self.status not in _TERMINAL_STATUSES and self._phase not in _TERMINAL_STATUSES:
            self._phase = "idle"
            self.status = "idle"

    def _live_loop(self) -> None:
        # Paint immediately so a blocked on_input is never a silent second,
        # then tick at live_interval (1 Hz in production). Capture the Event
        # this thread was started with so a later turn cannot resurrect it.
        stop = self._stop_live
        while True:
            if self._phase not in _TERMINAL_STATUSES:
                self.status = self._format_status()
                self._write_status_live()
            if stop.wait(self._live_interval):
                break

    def _note_ledger(self, *, at_exit: bool) -> None:
        """A one-line, never-blocking offer: "you have accumulated work,
        here are the numbers you would not otherwise see, try /land."

        Printed with `_println` (not `_note`) because at exit the loop
        breaks before the box is ever redrawn again -- `_note` alone would
        leave this sitting unseen in `self.transcript`. Any failure here
        (no repo, git missing) must never interrupt the turn loop, so
        everything is best-effort.
        """
        try:
            if self._ledger is None:
                return
            prompt, reason = self._ledger.should_prompt(at_exit=at_exit)
            if not prompt:
                return
            stats = ", ".join(self._ledger.render())
        except Exception:
            return
        line = sanitize_output(f"○ uncommitted work: {stats} ({reason}) — try /land")
        if not line:
            return
        self.transcript.append(line)
        self._println(line)

    def _mission_running(self) -> bool:
        """Is the workspace's persisted mission mid-run, in this process or
        a resident's worker on the same workspace.

        Reads .hcli/mission/state.json directly -- the exact shape
        Mission.checkpoint() writes (mission.py) -- rather than guessing
        one: `phase` is the literal string "running" for exactly the
        window between Mission.run() starting and Mission._finish()
        landing a terminal phase. Cross-process safe to read (checkpoint()
        writes it via atomic_write_json); never raises -- no mission yet,
        a fresh workspace, or a read racing a write all come back False.
        """
        try:
            with open(mission_state_path(self.workspace), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return False
        return isinstance(data, dict) and data.get("phase") == "running"

    def _tool_name(self, data: Dict[str, Any]) -> str:
        return str(data.get("tool") or data.get("name") or "tool").strip() or "tool"

    def _tool_outcome_line(self, data: Dict[str, Any]) -> str:
        name = self._tool_name(data)
        ok = bool(data.get("ok"))
        mark = "✓" if ok else "✗"
        outcome = "ok" if ok else "failed"
        elapsed = data.get("elapsed_s")
        extra = ""
        if isinstance(elapsed, (int, float)):
            extra = f"  {elapsed:.1f}s"
        return f"{mark} {name}  {outcome}{extra}"

    def _on_event(self, event: Event):
        data = event.data or {}
        etype = event.type
        ev = {"type": etype, "data": data}
        # stream_render.render_event decides WHAT a line says (and enforces,
        # via its own sanitize/status-word choke points, that raw model
        # text/content never becomes a phase word); TUI decides WHERE that
        # text lands and owns everything stream_render cannot see from one
        # event alone: direct one-off status strings, elapsed-time bookkeeping,
        # per-turn counters, and the tool_call_finished/tool_invoked dedup.
        # Heartbeat / phase events never copy text/content onto the status
        # line — that is how chain-of-thought would leak.

        if etype == "user_message":
            # Auto-steer (see run()) already announced this turn's intent as
            # "✓ steering: ..."; showing the raw text again as "You: ..." and
            # then /steer's own "Steer queued" confirmation right after would
            # be the same fact printed three times.
            if self._auto_steer_note is not None:
                self._note(self._auto_steer_note)
            else:
                for line in render_event(ev):
                    self._note(line)
            return
        if etype == "final_response":
            if self._auto_steer_note is not None:
                self._auto_steer_note = None
                return
            for line in render_event(ev):
                self._note(line)
            if is_terminal_event(ev):
                phase = event_phase(ev)
                if phase:
                    self.status = self._phase = phase
            return
        if etype in ("error", "rollback", "validation_failed", "goal_completed"):
            if is_terminal_event(ev):
                phase = event_phase(ev)
                self.status = self._phase = phase
            for line in render_event(ev):
                self._note(line)
            return
        if etype == "activity_started":
            self._set_phase(event_phase(ev) or "working")
            self._write_status_live()
            return
        if etype == "activity_completed":
            if self.status not in _TERMINAL_STATUSES:
                self._set_phase("idle")
            for line in render_event(ev):
                self._note(line)
            return
        if etype == "runtime_loading":
            self.status = "● loading resident"
            for line in render_event(ev):
                self._note(line)
            return
        if etype == "runtime_ready":
            self.status = "● resident ready"
            for line in render_event(ev):
                self._note(line)
            return
        if etype == "workunit_started":
            self.status = "● generating"
            return
        if etype == "workunit_completed":
            self.status = "● finalizing"
            return
        if etype == "bank_started":
            self.status = f"● bank {data.get('id')}"
            for line in render_event(ev):
                self._note(line)
            return
        if etype == "transcript_cleared":
            # The transcript really is emptied; render_transcript() then shows
            # "(no activity yet)", so the operator still sees the clear happen.
            # The acknowledgement goes on the status line, not into the
            # transcript we were just asked to empty.
            self.transcript = []
            self.status = str(data.get("content") or "Transcript cleared")
            return
        if etype == "evidence_gathering_started":
            self._set_phase(event_phase(ev) or "evidence")
            self._write_status_live()
            return
        if etype == "model_call_started":
            tokens = data.get("prompt_tokens")
            try:
                self._prompt_tokens = int(tokens) if tokens is not None else None
            except (TypeError, ValueError):
                self._prompt_tokens = None
            self._set_phase(event_phase(ev) or "thinking")
            self._write_status_live()
            return
        if etype == "model_call_finished":
            elapsed = data.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                self._phase_t0 = time.monotonic() - float(elapsed)
            self.status = self._format_status()
            return
        if etype == "heartbeat":
            # event_phase() cannot see TUI's own running _phase, so its
            # fallback (no data["phase"]) lands flat on "thinking". TUI has
            # that state, so it keeps the richer fallback chain here rather
            # than losing it by calling through the pure function.
            phase = status_word(data.get("phase") or self._phase or "thinking")
            self._phase = phase
            elapsed = data.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                self._phase_t0 = time.monotonic() - float(elapsed)
            tokens = data.get("prompt_tokens")
            if tokens is not None:
                try:
                    self._prompt_tokens = int(tokens)
                except (TypeError, ValueError):
                    pass
            self.status = self._format_status()
            self._write_status_live()
            return
        if etype == "tool_call_started":
            for line in render_event(ev):
                self._note(line)
            self._tool_calls_this_turn += 1
            self._set_phase(event_phase(ev) or self._tool_name(data))
            self._write_status_live()
            return
        if etype == "tool_call_finished":
            for line in render_event(ev):
                self._note(line)
            self._suppress_tool_invoked = True
            return
        if etype == "tool_invoked":
            if self._suppress_tool_invoked:
                self._suppress_tool_invoked = False
            else:
                # Legacy standalone path: no tool_call_started line ran
                # ahead of this one to carry the tool's name, so (unlike
                # tool_call_finished's paired "  ⎿ outcome") this still
                # names the tool on its own line.
                self._note(self._tool_outcome_line(data))
            return
        if etype == "validation_started":
            self._set_phase(event_phase(ev) or "validating")
            self._write_status_live()
            return
        if etype == "mutation_prepared":
            self._set_phase(event_phase(ev) or "mutating")
            self._write_status_live()
            return
        # steer_queued, bank_queued/finished/dropped/cleared, warning,
        # evidence_gathering_finished, goal_compiled: text-only notes with no
        # extra TUI state to track.
        for line in render_event(ev):
            self._note(line)

    def run(self, on_input: Callable[[str], None]) -> int:
        self.bus.subscribe(self._on_event)
        # One closed box per turn: printing a header and then unterminated
        # transcript lines is what left the frame hanging open.
        self._println(self.render())
        while True:
            try:
                user_input = self._prompt_fn("> ")
            except (EOFError, KeyboardInterrupt):
                self._println("[hcli] exiting")
                break
            text = user_input.strip()
            if not text:
                continue
            # Auto-steer: a plain line while a mission is already running
            # would otherwise start a second, competing goal. Route it
            # through /steer instead -- a hidden alias, not a new command --
            # so it reaches the running mission as knowledge rather than
            # racing it. A line already spelled as a command (/steer
            # included) is untouched.
            routed_text = text
            if not text.startswith(("/", "\\")) and self._mission_running():
                routed_text = "/steer " + text
            # Grok's live bracketing, plus the fault isolation that keeps a
            # failed turn from ending the SESSION. `on_input` was unguarded, so
            # any EngineError -- a context preflight refusal, a provider fault,
            # a bad tool argument -- propagated out of run() and killed the
            # REPL: the "> " prompt never came back and there was no way to
            # type /status or /steer to find out why. Ctrl-C cancels the TURN.
            self._begin_turn()
            if routed_text != text:
                head = text if len(text) <= 60 else text[:59].rstrip() + "…"
                self._auto_steer_note = f"✓ steering: {head}"
            try:
                on_input(routed_text)
            except KeyboardInterrupt:
                self.status = "cancelled"
                self._note("✗ cancelled (Ctrl-C); session still open")
            except Exception as exc:
                self.status = "error"
                self._note(f"✗ {type(exc).__name__}: {exc}")
            finally:
                self._end_turn()
            exiting = text in ("/exit", "/quit")
            self._note_ledger(at_exit=exiting)
            if exiting:
                break
            # The prompt leaves the cursor mid-line; start the box on its own.
            self._println()
            self._println(self.render())
        return 0
