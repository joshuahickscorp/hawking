#!/usr/bin/env python3.12
"""Exclusive controller lease — one implementation for every campaign.

Preserves the crash/resume semantics historical controllers reimplemented:
non-blocking exclusive flock, sealed owner record, process-local registry so a
second controller in the same process refuses to double-hold, and clean release.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


LEASE_SCHEMA = "hawking.condense.controller_lease.v1"

_PROCESS_LEASES: set[str] = set()
_PROCESS_LEASES_LOCK = threading.Lock()


class LeaseError(RuntimeError):
    """Lease acquisition or invariant failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class LeaseOwner:
    campaign_id: str
    owner: str
    controller_epoch: str
    pid: int
    acquired_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEASE_SCHEMA,
            "campaign_id": self.campaign_id,
            "owner": self.owner,
            "controller_epoch": self.controller_epoch,
            "pid": self.pid,
            "acquired_at": self.acquired_at,
        }


class SingletonLease:
    """Exclusive non-blocking controller lease with a durable owner record."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_id: str,
        controller_epoch: str = "1",
        owner: str = "condense-engine",
    ) -> None:
        self.path = Path(path)
        self.campaign_id = campaign_id
        self.controller_epoch = controller_epoch
        self.owner = owner
        self._handle: TextIO | None = None
        self._registry_key: str | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def assert_held(self) -> None:
        if not self.held:
            raise LeaseError("controller mutation refused: singleton lease is not held")

    def acquire(self, *, blocking: bool = False) -> LeaseOwner:
        if self.held:
            raise LeaseError("lease already held by this instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve()) if self.path.exists() else str(self.path)
        # Resolve after create for stable registry key.
        handle = open(self.path, "a+", encoding="utf-8")
        try:
            key = str(self.path.resolve())
            with _PROCESS_LEASES_LOCK:
                if key in _PROCESS_LEASES:
                    raise LeaseError(
                        f"lease already held in this process: {key}"
                    )
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as exc:
                handle.close()
                raise LeaseError(
                    f"lease busy: {self.path} (another controller holds it)"
                ) from exc
            record = LeaseOwner(
                campaign_id=self.campaign_id,
                owner=self.owner,
                controller_epoch=self.controller_epoch,
                pid=os.getpid(),
                acquired_at=_utc_now(),
            )
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(record.to_dict(), indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            with _PROCESS_LEASES_LOCK:
                _PROCESS_LEASES.add(key)
            self._handle = handle
            self._registry_key = key
            return record
        except BaseException:
            try:
                handle.close()
            except OSError:
                pass
            raise

    def release(self) -> None:
        if not self.held:
            return
        assert self._handle is not None
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                self._handle.close()
            finally:
                self._handle = None
                if self._registry_key is not None:
                    with _PROCESS_LEASES_LOCK:
                        _PROCESS_LEASES.discard(self._registry_key)
                    self._registry_key = None

    def __enter__(self) -> "SingletonLease":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def read_owner(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None
