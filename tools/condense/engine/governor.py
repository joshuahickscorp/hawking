#!/usr/bin/env python3.12
"""Resource governor — one policy surface for disk, memory, and lease floors.

Historical campaigns each inlined free-disk floors, swap checks, and caffeinate
guards. This module is the single place those limits are evaluated.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ResourceLimits:
    min_free_disk_bytes: int = 0
    min_free_inodes: int = 0
    # Soft advisory only; never a capability claim.
    max_concurrent_workers: int = 1
    require_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ResourceLimits":
        raw = raw or {}
        return cls(
            min_free_disk_bytes=int(raw.get("min_free_disk_bytes") or 0),
            min_free_inodes=int(raw.get("min_free_inodes") or 0),
            max_concurrent_workers=int(raw.get("max_concurrent_workers") or 1),
            require_path=raw.get("require_path"),
        )


@dataclass(frozen=True)
class ResourceSample:
    path: str
    free_disk_bytes: int
    total_disk_bytes: int
    free_inodes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "free_disk_bytes": self.free_disk_bytes,
            "total_disk_bytes": self.total_disk_bytes,
            "free_inodes": self.free_inodes,
        }


class ResourceGovernor:
    """Fail-closed resource gate evaluated before heavy campaign steps."""

    def __init__(self, limits: ResourceLimits, *, root: Path | None = None) -> None:
        self.limits = limits
        self.root = Path(root or limits.require_path or ".")

    def sample(self, path: Path | None = None) -> ResourceSample:
        target = Path(path or self.root)
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
        free_inodes: int | None = None
        try:
            st = os.statvfs(target)
            free_inodes = int(st.f_favail)
        except (AttributeError, OSError):
            free_inodes = None
        return ResourceSample(
            path=str(target),
            free_disk_bytes=int(usage.free),
            total_disk_bytes=int(usage.total),
            free_inodes=free_inodes,
        )

    def evaluate(self, sample: ResourceSample | None = None) -> list[str]:
        """Return a list of failure reasons; empty means pass."""
        sample = sample or self.sample()
        failures: list[str] = []
        if (
            self.limits.min_free_disk_bytes > 0
            and sample.free_disk_bytes < self.limits.min_free_disk_bytes
        ):
            failures.append(
                f"free_disk_bytes {sample.free_disk_bytes} < "
                f"min {self.limits.min_free_disk_bytes}"
            )
        if (
            self.limits.min_free_inodes > 0
            and sample.free_inodes is not None
            and sample.free_inodes < self.limits.min_free_inodes
        ):
            failures.append(
                f"free_inodes {sample.free_inodes} < "
                f"min {self.limits.min_free_inodes}"
            )
        if self.limits.require_path:
            req = Path(self.limits.require_path)
            if not req.exists():
                failures.append(f"require_path missing: {req}")
        return failures

    def allow(self) -> tuple[bool, ResourceSample, list[str]]:
        sample = self.sample()
        failures = self.evaluate(sample)
        return (not failures, sample, failures)

    def require(self) -> ResourceSample:
        ok, sample, failures = self.allow()
        if not ok:
            raise RuntimeError(
                "resource governor refused: " + "; ".join(failures)
            )
        return sample
