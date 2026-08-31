"""Shared plumbing for the tools/future sidecar.

Every module in this package writes a sealed JSON receipt under receipts/future/
and never asserts a hardware number. The bench block below is the single place
that enforces the second half of that.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "receipts" / "future"

# Anything the sidecar could accidentally claim without hardware authority.
HARDWARE_FIELDS = frozenset(
    {
        "tps",
        "accepted_tps",
        "token_ns",
        "complete_token_ns",
        "gpu_ns",
        "joules_per_token",
        "bandwidth_gbps",
        "wall_ns",
        "dispatch_ns",
    }
)


def bench_block(recorded_by: str) -> dict[str, Any]:
    """The only bench state this campaign is allowed to record.

    Claude/Grok have no protected GPU lease, so every receipt produced here is
    STATIC_ONLY with state UNKNOWN. S032-style rule: a budget or a plan is not
    a physical measurement.
    """
    return {
        "state": "UNKNOWN",
        "measurement_state": "STATIC_ONLY",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_by": recorded_by,
        "machine": "Apple host CPU; receipt/header metadata only",
        "gpu_authority": False,
        "rule": "no hardware measurement claim without hardware",
    }


class HardwareClaimError(ValueError):
    """Raised when a sidecar receipt tries to assert a measured hardware value."""


def _assert_no_hardware_claims(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                raise HardwareClaimError(
                    f"{here} = {value!r}: sidecar has no GPU authority, "
                    f"hardware fields must be null/UNKNOWN"
                )
            _assert_no_hardware_claims(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_hardware_claims(value, f"{path}[{i}]")


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """Attach a content hash over everything except the hash itself."""
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    doc["seal_sha256"] = hashlib.sha256(blob).hexdigest()
    return doc


def write_receipt(name: str, doc: dict[str, Any], recorded_by: str) -> Path:
    """Validate, seal and write a sidecar receipt. Returns its path."""
    doc.setdefault("bench", bench_block(recorded_by))
    doc.setdefault("claim_boundary", "Static sidecar artifact. No hardware measurement.")
    _assert_no_hardware_claims(doc)
    seal(doc)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS / name
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# Seconds before a read-only git query is abandoned. The tree is ~43GB and dirty,
# so `git status` can run for minutes; a query with no timeout is a query that
# can hang a caller forever.
GIT_TIMEOUT_S = 120


def git(*args: str) -> str:
    """A READ-ONLY git query that cannot take, or strand, the index lock.

    Every caller in this package reads: show, ls-tree, rev-parse, status,
    worktree list. `git status` refreshes and therefore WRITES .git/index.lock
    on a tree this size, and a git killed while holding it leaves a stale lock
    that blocks every later commit in the repo -- which has happened repeatedly
    here, each time with a lock several minutes old and no process holding it.

    --no-optional-locks tells git not to take that lock for a query that does
    not need it. The timeout stops a slow query becoming a hung caller. A
    timeout returns empty, which every caller already treats as "not found".
    """
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=REPO, capture_output=True, text=True, check=False,
            timeout=GIT_TIMEOUT_S,
        ).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_mtime(root: Path, skip: tuple[str, ...] = ()) -> tuple[float, str | None]:
    """Newest mtime under root, and which file it was. (0.0, None) if empty."""
    best, who = 0.0, None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "target"}]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if any(s in p for s in skip):
                continue
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m > best:
                best, who = m, os.path.relpath(p, REPO)
    return best, who
