#!/usr/bin/env python3.12
"""The second Odyssey gate: is this substrate fit to train on at all?

`ODYSSEY_LAUNCH_AUTHORIZED` answers "has a human authorized a run".  This answers a
different question -- "can the artifact generate" -- and the two are independent.  Before
this existed the fence was the only gate, so any agent that flipped it would have started
T0-T7 against a substrate that cannot complete "2 + 2 =".

Integrity is not capability.  Math-Preserve is byte-perfect at 282/282 shards and 59,585
frozen decisions, and it is still refused, because none of that was ever evidence that it
can generate.

    python3.12 tools/odyssey/substrate_capability.py            # report every substrate
    python3.12 tools/odyssey/substrate_capability.py --check    # exit 1 if the default
                                                               # substrate is refused

Silence is not a pass: an artifact with no entry is UNVERIFIED and treated as REFUSED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Runnable both as `python3 tools/odyssey/substrate_capability.py` and as
# `python3 -m tools.odyssey.substrate_capability`. A gate nobody can invoke the
# obvious way is a gate that gets skipped.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.odyssey._paths import EXPECTED_INDEX_SHA256, LAUNCH_DIR, MATH_ARTIFACT

CAPABILITY = LAUNCH_DIR / "SUBSTRATE_CAPABILITY.json"


class SubstrateRefused(RuntimeError):
    """Raised instead of returning a value, so a caller cannot ignore it by accident."""


def _load() -> dict:
    if not CAPABILITY.is_file():
        raise SubstrateRefused(
            f"{CAPABILITY} is missing. Without the capability register every substrate is "
            "UNVERIFIED, and UNVERIFIED is refused."
        )
    return json.loads(CAPABILITY.read_text())


def verdict_for(index_sha256: str) -> dict:
    """The recorded verdict for an artifact, by content hash. Unknown hash -> UNVERIFIED."""
    doc = _load()
    for s in doc["substrates"]:
        if s.get("artifact_index_sha256") == index_sha256:
            return s
    return {
        "name": f"<unknown artifact {index_sha256[:12]}>",
        "artifact_index_sha256": index_sha256,
        "capability_verdict": doc["default_for_unlisted"]["capability_verdict"],
        "capability_reason": doc["default_for_unlisted"]["why"],
    }


def measured_rate(artifact: Path) -> str | None:
    """The artifact's own complete BPW as an exact rational "num/den", or None.

    Read from the artifact, never from the register. The register records what was proven;
    this reports what is actually on disk, and the two disagreeing is the whole point of
    checking.
    """
    for name in ("PACK_RECEIPT.json", "ALLOCATION.json"):
        p = artifact / name
        if not p.is_file():
            continue
        doc = json.loads(p.read_text())
        for holder in (doc, doc.get("allocation", {})):
            if isinstance(holder, dict) and holder.get("complete_bpw_exact"):
                return str(holder["complete_bpw_exact"])
    return None


def assert_trainable(index_sha256: str, artifact: Path | None = None) -> None:
    """Raise unless this exact artifact is recorded as fit to train on.

    Called by the stage runner *in addition to* the fence check. Both gates must pass;
    neither can stand in for the other.

    Under the capability-first Gravity law a proof is bound to an artifact hash AND to the
    rate it was measured at. A proof taken at 6/5 BPW says nothing about the same pipeline
    repacked at 9/10: every rung of the ladder must be proven at that rung. When `artifact`
    is given, the recorded rate is checked against the rate actually on disk, so a repack
    that changes BPW without re-running the gates is refused rather than inheriting an
    approval it never earned.
    """
    s = verdict_for(index_sha256)
    v = s.get("capability_verdict")
    if v != "APPROVED":
        raise SubstrateRefused(
            f"substrate {s['name']} has capability_verdict={v}; refusing to train.\n"
            f"  reason: {s.get('capability_reason')}\n"
            f"  This is a SEPARATE gate from ODYSSEY_LAUNCH_AUTHORIZED. Flipping the fence\n"
            f"  does not clear it, and it is bound to the artifact's content hash."
        )
    if artifact is None:
        return
    on_disk = measured_rate(Path(artifact))
    proven = s.get("proven_at_rate")
    if on_disk is None:
        return  # nothing to compare against; the hash binding still stands
    if proven is None:
        raise SubstrateRefused(
            f"substrate {s['name']} is APPROVED but records no proven_at_rate, while the\n"
            f"  artifact on disk measures {on_disk} BPW. An approval that does not say what\n"
            f"  rate it was earned at cannot be checked, and uncheckable is refused."
        )
    if str(proven) != on_disk:
        raise SubstrateRefused(
            f"substrate {s['name']} was proven at {proven} BPW but measures {on_disk} on disk.\n"
            f"  Capability does not inherit across rates: every rung must be proven at that\n"
            f"  rung. Re-run the capability gate against this pack."
        )


def main() -> int:
    doc = _load()
    print(f"capability register: {CAPABILITY}")
    for s in doc["substrates"]:
        print(
            f"  {s['name']:34s} integrity={s.get('integrity_verdict','?'):20s} "
            f"capability={s['capability_verdict']}"
        )
        print(f"      {s.get('capability_reason','')}")
    print(f"  unlisted artifacts -> {doc['default_for_unlisted']['capability_verdict']} "
          f"(treated as {doc['default_for_unlisted']['treated_as']})")

    if "--check" in sys.argv:
        try:
            assert_trainable(EXPECTED_INDEX_SHA256)
        except SubstrateRefused as e:
            print(f"\nCHECK FAILED (this is the correct outcome today):\n{e}")
            return 1
        print(f"\nCHECK PASSED for {MATH_ARTIFACT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
