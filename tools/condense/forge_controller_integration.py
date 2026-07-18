#!/usr/bin/env python3.12
"""Integrate the Gravity Forge program into the ONE merged successor controller (Section 8).

Materializes the 120B Forge source-bound program THROUGH the controller API
(succ_gravity.materialize_forge_program), proves the controller refuses to launch it (default-off,
one heavy controller), writes the authoritative FORGE_PROGRAM.json, and registers it additively into
GRAVITY_STATE.json (top-level `forge_program`, re-sealed). Idempotent: re-running with the same
program sha does not duplicate. Launch stays disabled; this never starts the heavy run.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import succ_gravity as sgv          # noqa: E402
import succ_gravity_policy as gp    # noqa: E402
from eco_common import seal_field, sealed  # noqa: E402

STATE = "reports/condense/subbit_frontier/GRAVITY_STATE.json"
MANIFEST = "reports/condense/subbit_frontier/GRAVITY_120B_PROVENANCE.json"
PROGRAM_OUT = "reports/condense/gravity_forge/FORGE_PROGRAM.json"
QUEUE = "reports/condense/event_horizon_successor/queue/queue.json"


def register_queue_row(prog: dict[str, Any], parent_label: str = "120B") -> bool:
    """Wire the Forge program into the LIVE successor queue as a candidate on the parent row so the
    one controller can queue / drain / resume it. Additive + idempotent + re-sealed; launch stays
    disabled. Returns True if the queue was changed."""
    if not os.path.exists(QUEUE):
        return False
    q = json.load(open(QUEUE))
    row = q.get("rows", {}).get(parent_label)
    if row is None:
        return False
    fam = prog["representation_family"]
    cands = row.setdefault("candidate_representation_families", [])
    ref = {"program_sha256": prog["program_sha256"], "rate": prog["rate"]["label"],
           "family": fam, "runner": prog.get("forge_runner"), "launch": "DISABLED"}
    changed = False
    if fam not in cands:
        cands.append(fam); changed = True
    if row.get("forge_program") != ref:
        row["forge_program"] = ref; changed = True
    if changed:
        q = seal_field({k: v for k, v in q.items() if k != "queue_sha256"}, "queue_sha256")
        Path(QUEUE).write_text(json.dumps(q, indent=2, sort_keys=True, default=str))
    return changed


def materialize(parent_label: str = "120B", family: str = "transform_pq") -> dict[str, Any]:
    man = json.load(open(MANIFEST)).get("manifest_sha256")
    stress = Fraction(gp.compute_stress_start(parent_label)["chosen_stress_rate"]["label"])
    return sgv.materialize_forge_program(parent_label, rate=stress, family=family,
                                         source_manifest_sha256=man)


def integrate(parent_label: str = "120B", family: str = "transform_pq") -> dict[str, Any]:
    prog = materialize(parent_label, family)
    assert sealed(prog, "program_sha256"), "forge program failed to seal"

    # the controller REFUSES to launch it (default-off, one heavy controller) - proof of governance
    launchable, reasons = sgv.program_launchable(
        prog, policy=None, heavy_lock=sgv.HeavyLock(held_by=None), admission_passed=False)

    Path(PROGRAM_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRAM_OUT).write_text(json.dumps(prog, indent=2, sort_keys=True, default=str))

    # register additively into the live gravity state (idempotent, re-sealed)
    registered = False
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        existing = st.get("forge_program")
        if not (existing and existing.get("program_sha256") == prog["program_sha256"]):
            st["forge_program"] = prog
            st = seal_field({k: v for k, v in st.items() if k != "state_doc_sha256"}, "state_doc_sha256")
            Path(STATE).write_text(json.dumps(st, indent=2, sort_keys=True, default=str))
            registered = True

    queue_changed = register_queue_row(prog, parent_label)
    queue_has_forge = False
    if os.path.exists(QUEUE):
        queue_has_forge = bool(json.load(open(QUEUE)).get("rows", {}).get(parent_label, {}).get("forge_program"))

    return {"program_sha256": prog["program_sha256"][:16],
            "representation_family": prog["representation_family"],
            "kind": prog["kind"], "rate": prog["rate"]["label"],
            "controller_refuses_launch": (not launchable), "refusal_reasons": reasons,
            "registered_in_state": registered or bool(json.load(open(STATE)).get("forge_program")),
            "registered_as_successor_row": queue_has_forge, "queue_changed": queue_changed,
            "launch": "DISABLED"}


def main(argv: list[str] | None = None) -> int:
    r = integrate()
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
