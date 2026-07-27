#!/usr/bin/env python3.12
"""Generate the data source matrix and acquisition queue FROM the offline manifest.

The campaign's own rule is not to maintain handwritten duplicate lists, so these two
artifacts are derived rather than authored.  `RAMANUJAN_OFFLINE_MANIFEST.json` is the single
source; edit that and regenerate.

    python3.12 -m ramanujan.gen_data_matrix
    python3.12 -m ramanujan.gen_data_matrix --check   # fails if the generated files drifted
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ramanujan" / "RAMANUJAN_OFFLINE_MANIFEST.json"
MATRIX = ROOT / "ramanujan" / "RAMANUJAN_DATA_SOURCE_MATRIX.json"
QUEUE = ROOT / "ramanujan" / "RAMANUJAN_ACQUISITION_QUEUE.json"

# Which of the seven required bindings each source can already answer. A source that is
# locally generated answers licence and location the moment the toolchain exists; one that
# needs an external corpus cannot answer any of them until a human decides.
REQUIRED_BINDINGS = [
    "license", "version", "hash", "split", "deduplication",
    "contamination_boundary", "evidence_status", "local_offline_location",
]


def build() -> tuple[dict, dict]:
    man = json.loads(MANIFEST.read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows, queue = [], []
    for s in man["sources"]:
        locally_generated = "locally" in (s.get("acquisition") or "").lower() or \
                            "derived" in (s.get("acquisition") or "").lower() or \
                            "generated" in (s.get("acquisition") or "").lower()
        needs_decision = bool(s.get("requires_user_decision"))
        blocked_by = s.get("blocked_by")

        bindings = {b: "PENDING" for b in REQUIRED_BINDINGS}
        if locally_generated:
            # Derived from Mathlib inherits Apache-2.0; the rest resolve at generation time.
            bindings["license"] = "INHERITED_FROM_MATHLIB_APACHE_2_0" if "Mathlib" in str(s) else "LOCALLY_GENERATED"
            bindings["contamination_boundary"] = "ENFORCED_BY_EXISTING_BARRIER"

        rows.append({
            "id": s["id"],
            "name": s["name"],
            "purpose": s["purpose"],
            "status": s["status"],
            "acquisition_mode": "LOCALLY_GENERATED" if locally_generated else "EXTERNAL",
            "requires_user_decision": needs_decision,
            "blocked_by": blocked_by,
            "bindings": bindings,
            "bindings_satisfied": sum(1 for v in bindings.values() if v != "PENDING"),
            "bindings_required": len(REQUIRED_BINDINGS),
        })

        queue.append({
            "id": s["id"],
            "name": s["name"],
            "action": (
                "await a licensing decision from the user" if needs_decision
                else f"generate locally once {blocked_by}" if blocked_by
                else "generate locally"
            ),
            "blocked_by": blocked_by or ("user licensing decision" if needs_decision else None),
            "unblocked_by_toolchain_install": bool(blocked_by and "install" not in str(blocked_by).lower()
                                                   and ("Lean" in str(blocked_by) or "Mathlib" in str(blocked_by))),
            "network_required": bool(needs_decision),
            "safe_under_light_only": not needs_decision,
        })

    local = [r for r in rows if r["acquisition_mode"] == "LOCALLY_GENERATED"]
    external = [r for r in rows if r["acquisition_mode"] == "EXTERNAL"]

    matrix = {
        "schema": "hawking.ramanujan.data_source_matrix.v1",
        "at": now,
        "generated_from": "ramanujan/RAMANUJAN_OFFLINE_MANIFEST.json",
        "do_not_hand_edit": "regenerate with python3.12 -m ramanujan.gen_data_matrix",
        "required_bindings": REQUIRED_BINDINGS,
        "sources": rows,
        "summary": {
            "total": len(rows),
            "locally_generated": len(local),
            "external_needing_a_licensing_decision": len(external),
            "present_on_disk": sum(1 for r in rows if r["status"] == "PRESENT"),
            "bindings_fully_satisfied": sum(1 for r in rows if r["bindings_satisfied"] == len(REQUIRED_BINDINGS)),
        },
        "the_load_bearing_fact": (
            f"{len(local)} of {len(rows)} sources are generated locally from a working Lean and "
            "Mathlib and need no licensing decision and no download. One bounded toolchain "
            "install therefore unblocks the majority of Ramanujan's data."
        ),
        "hard_rule": man["hard_rule_on_teacher_traces"]["rule"],
    }

    q = {
        "schema": "hawking.ramanujan.acquisition_queue.v1",
        "at": now,
        "generated_from": "ramanujan/RAMANUJAN_OFFLINE_MANIFEST.json",
        "resource_mode": "LIGHT_ONLY",
        "queue": queue,
        "runnable_under_light_only_right_now": [
            q_["id"] for q_ in queue if q_["safe_under_light_only"] and not q_["blocked_by"]
        ],
        "awaiting_user_decision": [q_["id"] for q_ in queue if q_["blocked_by"] == "user licensing decision"],
        "awaiting_toolchain": [q_["id"] for q_ in queue if q_["blocked_by"] and "install" not in str(q_["blocked_by"])
                               and q_["blocked_by"] != "user licensing decision"],
        "note": "nothing here downloads a model parent. Acquisition of external corpora is a licensing decision and is not performed by this campaign.",
    }
    return matrix, q


def main() -> int:
    matrix, q = build()
    if "--check" in sys.argv:
        for path, want in ((MATRIX, matrix), (QUEUE, q)):
            if not path.exists():
                print(f"MISSING {path}")
                return 1
            have = json.loads(path.read_text())
            have.pop("at", None)
            w = dict(want)
            w.pop("at", None)
            if have != w:
                print(f"DRIFT in {path.name}: regenerate")
                return 1
        print("generated artifacts match the manifest")
        return 0

    MATRIX.write_text(json.dumps(matrix, indent=2) + "\n")
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    print(f"{matrix['summary']['locally_generated']}/{matrix['summary']['total']} locally generated; "
          f"{len(q['awaiting_user_decision'])} await a licensing decision")
    print(f"wrote {MATRIX.name}, {QUEUE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
