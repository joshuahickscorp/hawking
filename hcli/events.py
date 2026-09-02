from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


# Live-activity events. Engine and Controller emit these on EventBus;
# the TUI renders them. Payloads must never carry model text or
# chain-of-thought — elapsed time, counts, tool names, and ok/failed only.
MODEL_CALL_STARTED = "model_call_started"
MODEL_CALL_FINISHED = "model_call_finished"
EVIDENCE_GATHERING_STARTED = "evidence_gathering_started"
EVIDENCE_GATHERING_FINISHED = "evidence_gathering_finished"
TOOL_CALL_STARTED = "tool_call_started"
TOOL_CALL_FINISHED = "tool_call_finished"
HEARTBEAT = "heartbeat"

LIVE_EVENT_TYPES: Tuple[str, ...] = (
    MODEL_CALL_STARTED,
    MODEL_CALL_FINISHED,
    EVIDENCE_GATHERING_STARTED,
    EVIDENCE_GATHERING_FINISHED,
    TOOL_CALL_STARTED,
    TOOL_CALL_FINISHED,
    HEARTBEAT,
)


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subscribers: List[Callable[[Event], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Event], None]):
        with self._lock:
            self._subscribers.append(callback)

    def emit(self, event: Event):
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            cb(event)

# HCLI_EVENT_COMPAT_V1

def _hcli_event_from_parts(event_type, payload=None):
    """Construct the native event object from product-style event parts."""

    import inspect
    import time

    event_cls = Event
    data = {} if payload is None else payload

    # First prefer the natural positional shape used by most event records.
    try:
        return event_cls(event_type, data)
    except TypeError:
        pass

    signature = inspect.signature(event_cls)

    type_aliases = {
        "type",
        "event_type",
        "kind",
        "name",
        "topic",
    }

    payload_aliases = {
        "payload",
        "data",
        "detail",
        "details",
        "body",
        "meta",
        "metadata",
    }

    timestamp_aliases = {
        "timestamp",
        "time",
        "ts",
        "created_at",
    }

    text_aliases = {
        "message",
        "text",
        "content",
    }

    kwargs = {}

    for name, parameter in signature.parameters.items():
        if name in type_aliases:
            kwargs[name] = event_type
            continue

        if name in payload_aliases:
            kwargs[name] = data
            continue

        if name in timestamp_aliases:
            if parameter.default is inspect.Parameter.empty:
                kwargs[name] = time.time()
            continue

        if name in text_aliases:
            if parameter.default is inspect.Parameter.empty:
                if isinstance(data, dict):
                    kwargs[name] = str(
                        data.get("message")
                        or data.get("content")
                        or data.get("text")
                        or event_type
                    )
                else:
                    kwargs[name] = str(data)
            continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        raise TypeError(
            "Cannot map required event constructor field "
            f"{name!r} for {event_cls.__name__}"
        )

    return event_cls(**kwargs)


_HCLI_ORIGINAL_EVENTBUS_EMIT = EventBus.emit


def _hcli_compatible_eventbus_emit(self, event, payload=None):
    """Accept both native Event objects and emit(type, payload)."""

    if isinstance(event, str):
        event = _hcli_event_from_parts(
            event,
            payload,
        )

    return _HCLI_ORIGINAL_EVENTBUS_EMIT(
        self,
        event,
    )


_hcli_compatible_eventbus_emit._hcli_event_compat = True
EventBus.emit = _hcli_compatible_eventbus_emit

