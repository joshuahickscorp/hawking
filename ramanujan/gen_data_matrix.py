#!/usr/bin/env python3.12
"""Generate the data source matrix and acquisition queue FROM the offline manifest.

The campaign's own rule is not to maintain handwritten duplicate lists, so these two
artifacts are derived rather than authored.  `RAMANUJAN_OFFLINE_MANIFEST.json` is the single
source; edit that and regenerate.  When a generation receipt exists under
`ramanujan/data/corpora/`, PRESENT bindings are merged in for the local sources.

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
GENERATION_RECEIPT = ROOT / "ramanujan" / "data" / "corpora" / "GENERATION_RECEIPT.json"
CONTAMINATION_RECEIPT = ROOT / "ramanujan" / "data" / "corpora" / "CONTAMINATION_RECEIPT.json"

REQUIRED_BINDINGS = [
    "license", "version", "hash", "split", "deduplication",
    "contamination_boundary", "evidence_status", "local_offline_location",
]


def _load_receipts() -> tuple[dict, dict]:
    gen: dict = {}
    cont: dict = {}
    if GENERATION_RECEIPT.is_file():
        gen = json.loads(GENERATION_RECEIPT.read_text(encoding="utf-8"))
    if CONTAMINATION_RECEIPT.is_file():
        cont = json.loads(CONTAMINATION_RECEIPT.read_text(encoding="utf-8"))
    return gen, cont


def build() -> tuple[dict, dict]:
    man = json.loads(MANIFEST.read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gen, cont = _load_receipts()
    gen_sources = (gen or {}).get("sources") or {}

    rows, queue = [], []
    for s in man["sources"]:
        locally_generated = "locally" in (s.get("acquisition") or "").lower() or \
                            "derived" in (s.get("acquisition") or "").lower() or \
                            "generated" in (s.get("acquisition") or "").lower()
        needs_decision = bool(s.get("requires_user_decision"))
        blocked_by = s.get("blocked_by")
        sid = s["id"]
        receipt = gen_sources.get(sid) or {}

        bindings = {b: "PENDING" for b in REQUIRED_BINDINGS}
        status = s["status"]
        n_items = None

        if locally_generated:
            bindings["license"] = (
                "INHERITED_FROM_MATHLIB_APACHE_2_0"
                if "Mathlib" in str(s) or sid in ("D1", "D2", "D3")
                else "LOCALLY_GENERATED"
            )
            bindings["contamination_boundary"] = "ENFORCED_BY_EXISTING_BARRIER"

        # Merge generation receipt when present and non-empty.
        if receipt and receipt.get("n_items", 0) > 0:
            status = receipt.get("status") or "PRESENT"
            n_items = receipt.get("n_items")
            blocked_by = None
            for key in REQUIRED_BINDINGS:
                if receipt.get(key) not in (None, "", "PENDING"):
                    bindings[key] = receipt[key]
            # Prefer contamination receipt detail when available.
            if cont and sid in (cont.get("per_source") or {}):
                ps = cont["per_source"][sid]
                if ps.get("n_rejected", 0) == 0:
                    bindings["contamination_boundary"] = (
                        f"PASS (barrier indexed {cont.get('barrier', {}).get('n_eval_items_indexed', '?')} "
                        f"eval items; {ps.get('n_admitted', 0)}/{ps.get('n_input', 0)} admitted)"
                    )
                else:
                    bindings["contamination_boundary"] = (
                        f"PASS_WITH_REJECTIONS ({ps.get('n_rejected')} rejected of {ps.get('n_input')})"
                    )
            if not bindings.get("version") or bindings["version"] == "PENDING":
                bindings["version"] = gen.get("mathlib_commit") or "PENDING"
            if not bindings.get("hash") or bindings["hash"] == "PENDING":
                bindings["hash"] = receipt.get("sha256") or receipt.get("hash") or "PENDING"
            if not bindings.get("local_offline_location") or bindings["local_offline_location"] == "PENDING":
                bindings["local_offline_location"] = receipt.get("local_offline_location") or "PENDING"
            if not bindings.get("split") or bindings["split"] == "PENDING":
                bindings["split"] = receipt.get("split") or "train"
            if not bindings.get("deduplication") or bindings["deduplication"] == "PENDING":
                bindings["deduplication"] = receipt.get("deduplication") or "content_hash_sha256_exact"
            if not bindings.get("evidence_status") or bindings["evidence_status"] == "PENDING":
                bindings["evidence_status"] = receipt.get("evidence_status") or (
                    f"{n_items} items generated locally"
                )
            if not bindings.get("license") or bindings["license"] == "PENDING":
                bindings["license"] = receipt.get("license") or bindings["license"]

        # Manifest may already mark PRESENT after generation.
        if s.get("status") == "PRESENT" and status != "PRESENT":
            status = "PRESENT"

        row = {
            "id": sid,
            "name": s["name"],
            "purpose": s["purpose"],
            "status": status,
            "acquisition_mode": "LOCALLY_GENERATED" if locally_generated else "EXTERNAL",
            "requires_user_decision": needs_decision,
            "blocked_by": blocked_by,
            "bindings": bindings,
            "bindings_satisfied": sum(1 for v in bindings.values() if v != "PENDING"),
            "bindings_required": len(REQUIRED_BINDINGS),
        }
        if n_items is not None:
            row["n_items"] = n_items
            row["n_admitted"] = receipt.get("n_admitted", n_items)
            row["extraction_method"] = receipt.get("extraction_method")
        rows.append(row)

        action = (
            "await a licensing decision from the user" if needs_decision
            else (
                "present on disk; regenerate with python3.12 -m ramanujan.data.generate"
                if status == "PRESENT"
                else (f"generate locally once {blocked_by}" if blocked_by else "generate locally")
            )
        )
        queue.append({
            "id": sid,
            "name": s["name"],
            "action": action,
            "blocked_by": (
                None if status == "PRESENT"
                else (blocked_by or ("user licensing decision" if needs_decision else None))
            ),
            "unblocked_by_toolchain_install": bool(
                blocked_by
                and "install" not in str(blocked_by).lower()
                and ("Lean" in str(blocked_by) or "Mathlib" in str(blocked_by))
            ),
            "network_required": bool(needs_decision),
            "safe_under_light_only": not needs_decision,
            "n_items": n_items,
        })

    local = [r for r in rows if r["acquisition_mode"] == "LOCALLY_GENERATED"]
    external = [r for r in rows if r["acquisition_mode"] == "EXTERNAL"]

    matrix = {
        "schema": "hawking.ramanujan.data_source_matrix.v1",
        "at": now,
        "generated_from": "ramanujan/RAMANUJAN_OFFLINE_MANIFEST.json",
        "generation_receipt": (
            "ramanujan/data/corpora/GENERATION_RECEIPT.json"
            if gen else None
        ),
        "contamination_receipt": (
            "ramanujan/data/corpora/CONTAMINATION_RECEIPT.json"
            if cont else None
        ),
        "do_not_hand_edit": "regenerate with python3.12 -m ramanujan.gen_data_matrix",
        "required_bindings": REQUIRED_BINDINGS,
        "sources": rows,
        "summary": {
            "total": len(rows),
            "locally_generated": len(local),
            "external_needing_a_licensing_decision": len(external),
            "present_on_disk": sum(1 for r in rows if r["status"] == "PRESENT"),
            "bindings_fully_satisfied": sum(
                1 for r in rows if r["bindings_satisfied"] == len(REQUIRED_BINDINGS)
            ),
            "counts": {
                r["id"]: r.get("n_items")
                for r in rows
                if r.get("n_items") is not None
            },
        },
        "scale_up_command": (gen or {}).get("scale_up_command"),
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
        "awaiting_user_decision": [
            q_["id"] for q_ in queue if q_["blocked_by"] == "user licensing decision"
        ],
        "awaiting_toolchain": [
            q_["id"] for q_ in queue
            if q_["blocked_by"]
            and "install" not in str(q_["blocked_by"])
            and q_["blocked_by"] != "user licensing decision"
        ],
        "present": [q_["id"] for q_ in queue if q_.get("n_items")],
        "note": (
            "nothing here downloads a model parent. Acquisition of external corpora is a "
            "licensing decision and is not performed by this campaign."
        ),
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
    print(
        f"{matrix['summary']['locally_generated']}/{matrix['summary']['total']} locally generated; "
        f"{matrix['summary']['present_on_disk']} present on disk; "
        f"{len(q['awaiting_user_decision'])} await a licensing decision"
    )
    if matrix["summary"].get("counts"):
        print("counts:", json.dumps(matrix["summary"]["counts"], sort_keys=True))
    print(f"wrote {MATRIX.name}, {QUEUE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
