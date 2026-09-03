#!/usr/bin/env python3
"""ModelLake lifecycle state, read from disk at call time.

ONE capability with structured fields, not eight new gates. The roadmap already
tracks identity, hashing and promotion; what it could not answer was the
operational question -- how many specimens are sealed, how many are half-arrived,
how many the resident can actually see, and whether anything is still acquiring.

Everything here is counted off the filesystem when asked. The roadmap carried
"Flash is still downloading" long after it was false, and prose cannot outvote a
directory listing.

An unmounted volume reports VOLUME_ABSENT with null counts. ABSENT IS NOT EMPTY:
reporting an unmounted lake as an empty one is how a scheduler concludes there is
no science to do.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

LAKE = Path("/Volumes/corpdrive/hawking-modellake")
RECEIPT = "MODELLAKE_LIFECYCLE.json"


def _acquisition_workers() -> list[str]:
    """Processes actually acquiring right now, not a configured intent."""
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "modellake_watch"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.split(None, 1)[-1][:120] for ln in out.splitlines() if ln.strip()]


def lifecycle() -> dict[str, Any]:
    if not LAKE.is_dir():
        return {
            "state": "VOLUME_ABSENT",
            "root": str(LAKE),
            "why": "the lake volume is not mounted; absent is NOT empty",
            "sealed_specimens": None,
            "manifests": None,
            "partial_specimens": None,
            "payload_complete_unpromoted": None,
            "curriculum_eligible": None,
            "active_acquisition_workers": _acquisition_workers(),
        }

    specimens = sorted(p for p in (LAKE / "specimens").iterdir() if p.is_dir()) \
        if (LAKE / "specimens").is_dir() else []
    manifests = {p.stem for p in (LAKE / "manifests").glob("*.json")} \
        if (LAKE / "manifests").is_dir() else set()

    # A specimen with no manifest cannot be retired or reacquired: the recipe is
    # missing, so it is payload-complete but not fully promoted.
    unpromoted = [p.name for p in specimens if p.name not in manifests]
    partials = [p.name for p in specimens if (p / ".partial").exists()]

    workers = _acquisition_workers()
    return {
        "state": "READ_FROM_DISK",
        "root": str(LAKE),
        "sealed_specimens": len(specimens),
        "manifests": len(manifests),
        "partial_specimens": len(partials),
        "payload_complete_unpromoted": len(unpromoted),
        "unpromoted_examples": unpromoted[:5],
        # Sealing is what makes a specimen eligible; no manual campaign creation
        # is required merely because another model arrived.
        "curriculum_eligible": len([p for p in specimens if p.name in manifests]),
        "hcli_visible": len(specimens),
        "active_acquisition_workers": workers,
        "acquiring_now": bool(workers),
        "startup_reconciliation": (
            "counts are recomputed from the filesystem on every call; there is no "
            "cached roster to drift out of date"
        ),
        "retained_verified_bytes_per_wall_second": None,
        "why_no_rate": (
            "a throughput rate needs two timestamped observations. Reporting one "
            "from a single reading would be a fabricated measurement."
        ),
    }


def build() -> Path:
    from tools.future._common import write_receipt

    doc = lifecycle()
    doc.update({
        "schema": "hawking.future.modellake_lifecycle.v1",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": (
            "a filesystem census of the specimen lake. It proves what is on disk, "
            "not that any specimen is correct, complete or servable."
        ),
    })
    return write_receipt(RECEIPT, doc, "tools.future.modellake_lifecycle")


if __name__ == "__main__":
    print(build())
