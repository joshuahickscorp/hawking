"""G030 / S012 §38: the NX promotion contract, as a checker that VERIFIES.

A contract that accepts {"standalone_execution": true} is a rubber stamp. This
campaign has already produced two receipts whose headline numbers were wrong
because a self-reported field went unchecked -- a sparse channel that cost
nothing in the accounting, and an answer recorded ok=True with no work behind it.

So every condition here demands an EVIDENCE REFERENCE, and the three that can be
checked mechanically are checked:

  * complete accounting     counted bytes must equal the artifact's real size
  * no hidden dense parent  no path may point outside the artifact
  * immutable identity      the recorded hash must match the artifact

The rest require a named receipt. A condition asserted with a bare boolean and no
receipt is REFUSED, not believed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CONDITIONS = (
    "standalone_execution",
    "complete_persistent_accounting",
    "no_hidden_dense_parent",
    "capability_qualification",
    "oiii_adversarial_qualification",
    "physical_qualification",
    "restart_recovery_qualification",
    "machine_runtime_contract",
    "independent_verification",
    "immutable_artifact_identity",
)

# Conditions this module can settle itself. The rest need a receipt.
MECHANICAL = ("complete_persistent_accounting", "no_hidden_dense_parent",
              "immutable_artifact_identity")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_accounting(manifest: dict, artifact_bytes: int) -> tuple[bool, str]:
    """Counted bytes must equal what is actually on disk."""
    counted = manifest.get("counted_bytes")
    if not isinstance(counted, int):
        return False, "counted_bytes absent or not an integer"
    if counted != artifact_bytes:
        return False, (f"accounting says {counted} but the artifact is {artifact_bytes} "
                       f"({artifact_bytes - counted:+d}) -- something is uncounted")
    return True, f"counted {counted} == artifact {artifact_bytes}"


def check_no_hidden_parent(manifest: dict) -> tuple[bool, str]:
    """No referenced path may live outside the artifact."""
    refs = manifest.get("external_references")
    if refs is None:
        return False, "external_references absent -- absence of a field is not absence of a parent"
    bad = [r for r in refs if r]
    if bad:
        return False, f"references outside the artifact: {bad[:3]}"
    return True, "no external references"


def check_identity(manifest: dict, artifact: Path) -> tuple[bool, str]:
    claimed = manifest.get("sha256")
    if not claimed:
        return False, "no sha256 recorded"
    if not artifact.exists():
        return False, f"artifact absent: {artifact}"
    actual = _sha256(artifact)
    if actual != claimed:
        return False, f"hash mismatch: recorded {claimed[:12]} actual {actual[:12]}"
    return True, f"sha256 {actual[:12]} matches"


def evaluate(manifest: dict, artifact: Path | None = None) -> dict:
    """Full contract. Returns the verdict and WHY, per condition."""
    results: dict[str, dict] = {}
    size = artifact.stat().st_size if (artifact and artifact.exists()) else -1

    for c in CONDITIONS:
        entry = manifest.get(c)
        if c in MECHANICAL:
            if c == "complete_persistent_accounting":
                ok, why = check_accounting(manifest, size)
            elif c == "no_hidden_dense_parent":
                ok, why = check_no_hidden_parent(manifest)
            else:
                ok, why = check_identity(manifest, artifact) if artifact else (False, "no artifact")
            results[c] = {"ok": ok, "why": why, "checked": "mechanically"}
            continue
        # Everything else needs a receipt. A bare boolean is not evidence.
        if isinstance(entry, dict) and entry.get("receipt"):
            results[c] = {"ok": bool(entry.get("ok")), "why": f"receipt {entry['receipt']}",
                          "checked": "by receipt"}
        else:
            results[c] = {"ok": False,
                          "why": "no receipt cited; a bare claim is not evidence",
                          "checked": "refused"}

    missing = [c for c, r in results.items() if not r["ok"]]
    return {"promotable": not missing, "conditions": results, "missing": missing,
            "verdict": ("NX" if not missing
                        else "NR-derived candidate; promotion refused")}
