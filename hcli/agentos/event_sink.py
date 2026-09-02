"""Durable event stream the resident's worker writes and watchers tail.

Today the worker subscribes one callback to the whole EventBus and only
acts on "runtime_ready" (see resident.py:_worker_main) - every other event
(tool calls, phases, model text, notes) is emitted and dropped on the floor.
EventSink is the missing durable half: one JSON line per event, appended to
``<workspace>/.hcli/mission/events.jsonl``, so a reader in another process
(``hcli resident watch``) can tail real activity by byte offset instead of
polling heartbeats and mission.log event names.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

_EVENTS_FILENAME = "events.jsonl"


class EventSink:
    """Appends one JSON object per line; rotates to ``.1`` past ``max_bytes``.

    write() never raises - the worker's EventBus dispatches to it synchronously
    (hcli/events.py EventBus.emit has no per-subscriber error handling), so a
    raised exception here would take down a live mission. Anything swallowed
    is counted in ``.dropped`` so the failure is at least visible.
    """

    def __init__(self, workspace: Union[str, Path], *, max_bytes: int = 8 * 1024 * 1024) -> None:
        self._path = Path(workspace) / ".hcli" / "mission" / _EVENTS_FILENAME
        self._max_bytes = max_bytes
        self._seq = 0
        self.dropped = 0

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: Any) -> None:
        try:
            if isinstance(event, dict):
                etype = event.get("type")
                data = event.get("data", {})
            else:
                etype = getattr(event, "type", None)
                data = getattr(event, "data", {})
            row: Dict[str, Any] = {"t": time.time(), "type": etype, "data": data, "seq": self._seq}
            try:
                line = json.dumps(row, default=str) + "\n"
            except TypeError:
                # something in `data` refused even the str() fallback path
                # (e.g. a non-string-keyed dict) - coerce the whole blob.
                row["data"] = str(data)
                line = json.dumps(row, default=str) + "\n"

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._seq += 1
        except Exception:
            self.dropped += 1

    def _rotate_if_needed(self) -> None:
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < self._max_bytes:
            return
        rotated = self._path.parent / (self._path.name + ".1")
        try:
            os.replace(self._path, rotated)  # replaces any existing .1
        except OSError:
            pass

    def close(self) -> None:
        # No handle is held open between writes (each write is its own
        # open/append/flush/close), so there is nothing to release here.
        pass


def read_events(path: Union[str, Path], *, offset: int = 0, limit: int = 200) -> Tuple[List[Dict[str, Any]], int]:
    """Read up to ``limit`` events appended after byte ``offset``.

    Returns ``(events, new_offset)``. Never raises: a missing file returns
    ``([], 0)``; a shrunk file (rotated out from under a stale offset) is
    detected and restarts from 0; a partial trailing line (writer mid-append,
    or real corruption) is left unconsumed - ``new_offset`` stops before it
    so the next call re-reads it once it is complete.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return [], 0
    if size < offset:
        offset = 0

    events: List[Dict[str, Any]] = []
    new_offset = offset
    try:
        with open(path, "r", encoding="utf-8") as handle:
            handle.seek(offset)
            while len(events) < limit:
                line = handle.readline()
                if not line.endswith("\n"):
                    break  # EOF, or a torn trailing line - don't advance past it
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass  # corrupt but complete line - skip it, still advance past
                new_offset = handle.tell()
    except OSError:
        return [], offset
    return events, new_offset
