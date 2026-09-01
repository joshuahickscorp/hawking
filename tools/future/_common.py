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
from typing import Iterable, Sequence,  Any

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


class ReceiptPathCollision(ValueError):
    """Two producers, one path. The later writer would destroy the earlier one."""


def _refuse_foreign_overwrite(out: Path, doc: dict[str, Any], recorded_by: str) -> None:
    """A receipt path belongs to ONE producer.

    tps_budget.py and causal_budget_71.py both wrote
    RESIDENT_71TPS_CAUSAL_BUDGET.json with different schemas. The later writer
    won and silently destroyed every citation resolving against `ladder[]` and
    `measured_now` - four rows of the roof-anchor audit stopped resolving, and
    the audit honestly reported "field is not a resolvable path in this receipt"
    about a field that HAD been resolvable the day it was written. Nothing
    raised. The overwrite is a WRITE, and writes succeed.

    An overwrite by the same producer, or a schema-compatible one, is normal
    regeneration and is allowed. A DIFFERENT producer writing a DIFFERENT schema
    over an existing receipt is the collision, and it raises.
    """
    if not out.is_file():
        return
    try:
        prior = json.loads(out.read_text())
    except (ValueError, OSError):
        return  # unreadable prior is not evidence of ownership
    if not isinstance(prior, dict):
        return
    prior_by = prior.get("recorded_by") or (prior.get("bench") or {}).get("recorded_by")
    if not prior_by or prior_by == recorded_by:
        return
    prior_schema = prior.get("schema")
    new_schema = doc.get("schema")
    if prior_schema is None or new_schema is None or prior_schema == new_schema:
        return
    raise ReceiptPathCollision(
        f"{out.name} was written by {prior_by} with schema {prior_schema!r}; "
        f"{recorded_by} would overwrite it with schema {new_schema!r}. A receipt "
        "path belongs to one producer - give this one its own name rather than "
        "destroying the other's citations."
    )


def write_receipt(name: str, doc: dict[str, Any], recorded_by: str) -> Path:
    """Validate, seal and write a sidecar receipt. Returns its path."""
    doc.setdefault("bench", bench_block(recorded_by))
    doc.setdefault("claim_boundary", "Static sidecar artifact. No hardware measurement.")
    _assert_no_hardware_claims(doc)
    seal(doc)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS / name
    _refuse_foreign_overwrite(out, doc, recorded_by)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return out


class MeasurementProvenanceError(ValueError):
    """Raised when a receipt asserts a hardware number without saying when, under
    what load, and whether the GPU lane lock was held."""


def measurement_provenance(
    *,
    lock_held: bool,
    loadavg: str | None = None,
    lane: str | None = None,
    measured_at: str | None = None,
    retrofit: bool = False,
) -> dict[str, Any]:
    """Provenance every hardware number needs and none of them carried.

    write_receipt REFUSES hardware fields, so measurement receipts hand-rolled
    their own json.dumps and inherited no bench block at all. The consequence
    showed up the day /tmp/hawking-gpu-lane.lock was found wedged as a stale file:
    placing the four headline measurements against that contention window had to
    be done from GIT LANDING TIMES, because none of them recorded when it ran.
    Landing time is a proxy - a receipt can be produced long before it lands - and
    that margin happened to be hours. Contention windows are discovered after the
    fact, so the stamp has to exist before anyone knows they need it.
    """
    if loadavg is None:
        try:
            one, five, fifteen = os.getloadavg()
            loadavg = f"{{ {one:.2f} {five:.2f} {fifteen:.2f} }}"
        except (OSError, AttributeError):
            loadavg = None
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if retrofit and measured_at is None:
        # A receipt REGENERATED from a stored raw measurement must not stamp the
        # regeneration time as the measurement time. That would be a fabricated
        # provenance - precisely the thing this block exists to prevent - and it
        # is an easy mistake because the writer happens to be running now.
        return {
            "measured_at": None,
            "measured_at_source": "RETROFIT_UNKNOWN",
            "receipt_regenerated_at": now,
            "gpu_lane_lock_held": None,
            "loadavg": loadavg,
            "lane": lane,
            "absolutes_are_measured_under_load": True,
            "why": (
                "this receipt predates measurement provenance; the raw capture "
                "carries no timestamp, so the measurement time is genuinely "
                "unknown and is recorded as unknown rather than invented"
            ),
        }
    return {
        "measured_at": measured_at or now,
        "measured_at_source": "CAPTURED" if measured_at else "WRITE_TIME",
        "gpu_lane_lock_held": bool(lock_held),
        "loadavg": loadavg,
        "lane": lane,
        "absolutes_are_measured_under_load": True,
        "why": (
            "a hardware number that cannot be placed in time cannot be audited "
            "against a contention window"
        ),
    }


REQUIRED_PROVENANCE = ("measured_at", "gpu_lane_lock_held", "loadavg")


def _has_hardware_number(node: Any) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                return True
            if _has_hardware_number(value):
                return True
    elif isinstance(node, list):
        return any(_has_hardware_number(v) for v in node)
    return False


def write_measured_receipt(
    path: str | Path,
    doc: dict[str, Any],
    recorded_by: str,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a receipt that IS allowed to carry hardware numbers - and therefore
    must say when it measured them, under what load, and whether it held the lane
    lock. The sidecar writer refuses hardware; this one requires provenance for it.
    """
    doc.setdefault("recorded_by", recorded_by)
    if provenance is not None:
        doc["measurement_provenance"] = provenance
    prov = doc.get("measurement_provenance")
    if _has_hardware_number(doc):
        if not isinstance(prov, dict):
            raise MeasurementProvenanceError(
                f"{recorded_by}: receipt carries a hardware number and no "
                f"measurement_provenance block; use measurement_provenance()"
            )
        missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
        if missing:
            raise MeasurementProvenanceError(
                f"{recorded_by}: measurement_provenance is missing {missing}; "
                f"a hardware number that cannot be placed in time cannot be "
                f"audited against a contention window"
            )
        if not prov.get("measured_at") and prov.get("measured_at_source") != "RETROFIT_UNKNOWN":
            raise MeasurementProvenanceError(
                f"{recorded_by}: measured_at is empty; landing time is a proxy, "
                f"not a measurement time"
            )
    seal(doc)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
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


class UnknownFlag(SystemExit):
    """A CLI was handed a flag it does not implement."""


def require_known_flags(known: "Iterable[str]", argv: "Sequence[str] | None" = None) -> None:
    """Refuse an unrecognised flag instead of ignoring it.

    Modules that dispatch with `if "--record" in sys.argv` treat every other
    argument as absent. So `--build` - the verb most of tools/future uses -
    printed a freshly computed table, exited 0, and WROTE NOTHING. The terminal
    showed current numbers while the receipt on disk stayed stale, and that cost
    two silently-stale receipts before it was noticed (path_to_71,
    causal_budget_71).

    A tool that reports success without doing the work is the failure this
    campaign keeps finding in its own checks. Call this first in __main__.
    """
    import sys as _sys
    args = list(argv if argv is not None else _sys.argv[1:])
    ok = set(known)
    bad = [a for a in args if a.startswith("-") and a.split("=", 1)[0] not in ok]
    if bad:
        raise UnknownFlag(
            f"unknown flag(s) {bad}; known flags are {sorted(ok)}. Refusing "
            "rather than running with the argument silently ignored."
        )
