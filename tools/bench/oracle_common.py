"""Shared helpers for tools/bench/oracle_*.py experiments."""
from __future__ import annotations

import os
import resource
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def rss_gb() -> float:
    """Resident set size in GiB (macOS maxrss is bytes; Linux is KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: values > 10**9 are bytes (macOS); else KiB (Linux).
    if usage > 10**9:
        return usage / (1024**3)
    return usage / (1024**2)


def check_rss(limit_gb: float = 14.0) -> None:
    if rss_gb() > limit_gb:
        raise SystemExit(f"RSS {rss_gb():.2f} GiB exceeds limit {limit_gb}")


def rel_rmse(a, b, eps: float = 1e-12) -> float:
    if np is None:
        raise RuntimeError("numpy required")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(a)) + eps
    return num / den


def repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for _ in range(10):
        if (p / "tools" / "foundry" / "lab_harness").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()
