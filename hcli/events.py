from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._subscribers: List[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]):
        self._subscribers.append(callback)

    def emit(self, event: Event):
        for cb in self._subscribers:
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

