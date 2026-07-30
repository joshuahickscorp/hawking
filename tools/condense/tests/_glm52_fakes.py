"""Shared GLM-52 test doubles (lane J2)."""
from __future__ import annotations
from typing import Any, Callable, Mapping

class FakeKeychain:
    def __init__(
        self,
        values: Mapping[str, str] | None = None,
        *,
        discard_writes: bool = False,
        discard: bool | None = None,
    ) -> None:
        self.values = dict(values or {})
        self.discard_writes = bool(discard if discard is not None else discard_writes)
        self.set_calls: list[tuple[str, str]] = []

    def get(self, service: str) -> str | None:
        return self.values.get(service)

    def set(self, service: str, value: str) -> None:
        self.set_calls.append((service, value))
        if not self.discard_writes:
            self.values[service] = value
