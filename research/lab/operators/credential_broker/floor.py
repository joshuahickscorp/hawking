"""Disk-floor proofs shared by all acquisition programmes.

Proven implementations generalised here:
- DeepSeek ``assert_floor``: free - next_allocation >= 15 GiB, sampled
  before/during/after every range (``lab.operators.deepseek_v4_stream_executor``)
- Kimi ``_floor_check``: 15 GiB free-space floor before metadata admission
- GLM ``DISK_FLOOR_BYTES`` (default 75e9) + ``glm52_grounding`` operational
  reserve: free >= reserve + additional reserved
- GLM layer stream: 25 GiB free-space floor + source-only reclaim

Bible §7: disk-floor proof is mandatory before acquisition.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

FreeBytesProvider = Callable[[Path], int]

# Non-negotiable minimum used by DeepSeek header streaming and Kimi admission.
# Individual programmes may raise (never lower) this floor via preflight.
DEFAULT_PROTECTED_FLOOR_BYTES = 15 * 1024**3


class FloorViolation(RuntimeError):
    """Filesystem free space would breach the protected floor."""


@dataclass(frozen=True)
class FloorProof:
    stage: str
    observed_at: str
    free_bytes: int
    additional_bytes: int
    remaining_bytes: int
    protected_floor_bytes: int
    status: str  # PASS

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "observed_at": self.observed_at,
            "free_bytes": self.free_bytes,
            "additional_bytes": self.additional_bytes,
            "remaining_bytes": self.remaining_bytes,
            "protected_floor_bytes": self.protected_floor_bytes,
            "status": self.status,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_free(root: Path, provider: FreeBytesProvider | None) -> int:
    if provider is None:
        try:
            return int(shutil.disk_usage(root).free)
        except OSError as exc:
            raise FloorViolation(f"cannot sample filesystem free bytes: {exc}") from exc
    value = provider(root)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FloorViolation("free_bytes provider must return a non-negative integer")
    return value


def assert_disk_floor(
    workspace_root: str | Path,
    *,
    protected_floor_bytes: int = DEFAULT_PROTECTED_FLOOR_BYTES,
    additional_bytes: int = 0,
    stage: str,
    free_bytes_provider: FreeBytesProvider | None = None,
    absolute_minimum_floor_bytes: int = DEFAULT_PROTECTED_FLOOR_BYTES,
) -> FloorProof:
    """Sample and enforce ``free - additional >= protected_floor``.

    ``additional_bytes`` is the next in-flight allocation (range, shard, or
    intermediate). Programmes must not lower ``protected_floor_bytes`` below
    ``absolute_minimum_floor_bytes`` (default 15 GiB).
    """
    if not isinstance(stage, str) or not stage.strip():
        raise FloorViolation("floor stage must be a non-empty string")
    if isinstance(protected_floor_bytes, bool) or not isinstance(protected_floor_bytes, int):
        raise FloorViolation("protected_floor_bytes must be an integer")
    if protected_floor_bytes < absolute_minimum_floor_bytes:
        raise FloorViolation(
            f"protected floor cannot be below the non-negotiable "
            f"{absolute_minimum_floor_bytes}-byte minimum"
        )
    if isinstance(additional_bytes, bool) or not isinstance(additional_bytes, int) or additional_bytes < 0:
        raise FloorViolation("additional_bytes must be a non-negative integer")

    root = Path(workspace_root)
    free = _read_free(root, free_bytes_provider)
    remaining = free - additional_bytes
    if remaining < protected_floor_bytes:
        raise FloorViolation(
            f"disk floor crossed at {stage}: free={free} next_bytes={additional_bytes} "
            f"floor={protected_floor_bytes}"
        )
    return FloorProof(
        stage=stage.strip(),
        observed_at=_utc_now(),
        free_bytes=free,
        additional_bytes=additional_bytes,
        remaining_bytes=remaining,
        protected_floor_bytes=protected_floor_bytes,
        status="PASS",
    )
