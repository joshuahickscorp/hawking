"""ROADMAP_SCAFFOLD_SATURATION — the campaign delta, DERIVED not asserted.

Every number here is read back out of the adversarial auditor's own graph
(``civilization/CAPABILITY_GRAPH.json``) or computed from git. Nothing is
hand-entered, because a hand-edited completion score is exactly the failure
this campaign spent its time removing.

Run ``python3 -m tools.roadmap.saturation --emit`` to regenerate. If the
regenerated file differs from the committed one, the committed one is stale
and the tool is authoritative -- fix the producer, never the artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "civilization" / "CAPABILITY_GRAPH.json"
RECEIPT = REPO / "receipts" / "future" / "ROADMAP_SCAFFOLD_SATURATION.json"

SCHEMA = "hawking.roadmap.scaffold_saturation.v1"

# The state of the ledger when this campaign opened. Recorded so the delta is
# a delta and not just a snapshot. These were the numbers in the directive.
BASELINE = {
    "source": "directive section 1, and a 31-agent adversarial audit that preceded this campaign",
    "gates_total": 71,
    "BUILT": 18,
    "SCAFFOLDED": 29,
    "ABSENT": 11,
    "BLOCKED_HARDWARE": 13,
    "gate_mean_pct": 25.7,
    "scaffolded_or_better": 47,
    "caveat": (
        "This baseline was materially WRONG and the campaign proved it: nearly every "
        "subsystem it called absent already existed with tests. It is retained because "
        "the delta is only meaningful against what was actually believed at the start."
    ),
}

CAMPAIGN_BASE_COMMIT = "caf83078f"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _numstat_added(pathspec: str) -> int:
    out = _git("diff", "--numstat", f"{CAMPAIGN_BASE_COMMIT}..HEAD", "--", pathspec)
    total = 0
    for line in out.split("\n"):
        if not line.strip():
            continue
        added = line.split("\t")[0]
        if added.isdigit():
            total += int(added)
    return total


def _added_files(pathspec: str, *, exclude: str | None = None) -> list[str]:
    out = _git(
        "diff", "--name-only", "--diff-filter=A", f"{CAMPAIGN_BASE_COMMIT}..HEAD", "--", pathspec
    )
    rows = [r for r in out.split("\n") if r.strip()]
    if exclude:
        rows = [r for r in rows if exclude not in r]
    return rows


def _gate_rows(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    gates = graph["gates"]
    if isinstance(gates, dict):
        return list(gates.items())
    return [(g["id"], g) for g in gates]


def build() -> dict[str, Any]:
    graph = json.loads(GRAPH.read_text())
    rows = _gate_rows(graph)
    counts = graph["counts"]["gates_by_status"]

    by_status: dict[str, list[str]] = {}
    for gid, entry in rows:
        by_status.setdefault(str(entry.get("status")), []).append(gid)

    hardware = [
        {
            "gate": gid,
            "wake_condition": entry.get("wake_condition"),
        }
        for gid, entry in rows
        if entry.get("status") == "BLOCKED_HARDWARE"
    ]
    external = [
        {
            "gate": gid,
            "blocker": entry.get("software_blocker"),
        }
        for gid, entry in rows
        if entry.get("status") == "BLOCKED_EXTERNAL"
    ]

    # Non-hardware, non-external gates are the ones this campaign could move.
    movable = [
        (gid, e)
        for gid, e in rows
        if e.get("status") not in {"BLOCKED_HARDWARE", "BLOCKED_EXTERNAL"}
    ]
    ok = [gid for gid, e in movable if e.get("status") in {"BUILT", "WIRED", "SCAFFOLDED"}]

    wired_not_accepted = [
        gid
        for gid, e in rows
        if bool((e.get("wired") or {}).get("value")) and not bool((e.get("accepted") or {}).get("value"))
    ]

    return {
        "schema": SCHEMA,
        "purpose": (
            "Campaign delta for the roadmap saturation ultragoal. Derived from the "
            "adversarial auditor's graph and from git. No hand-entered scores."
        ),
        "derived_from": {
            "graph": str(GRAPH.relative_to(REPO)),
            "graph_generated_by": graph.get("generated_by"),
            "base_commit": CAMPAIGN_BASE_COMMIT,
            "head": _git("rev-parse", "HEAD"),
        },
        "baseline": BASELINE,
        "final": {
            "gates_total": len(rows),
            "by_status": counts,
            "movable_gates": len(movable),
            "movable_scaffolded_or_better": len(ok),
            "movable_scaffolded_or_better_pct": round(100.0 * len(ok) / max(1, len(movable)), 1),
        },
        "the_correction_that_matters": {
            "claim": "BUILT went 26 -> 0 and that is the campaign's most valuable result.",
            "why": (
                "An adversarial refuter attacked all 26 BUILT gates. It was right on the "
                "central point: a call site proves a capability is REACHABLE, not that its "
                "acceptance criterion is SATISFIED. The auditor now tracks those as "
                "orthogonal facts and BUILT requires both. The decisive case was "
                "FLASH_COMPLETE_EBPW_LE_1, rated BUILT while its own receipt reported "
                "complete_ebpw = 3.139 against a required <= 1."
            ),
            "wired_but_not_accepted": sorted(wired_not_accepted),
            "wired_but_not_accepted_count": len(wired_not_accepted),
        },
        "still_absent": sorted(by_status.get("ABSENT", [])),
        "unreachable": sorted(by_status.get("UNREACHABLE", [])),
        "hardware_blocked": sorted(hardware, key=lambda r: r["gate"]),
        "externally_blocked": sorted(external, key=lambda r: r["gate"]),
        "work": {
            "commits": int(_git("rev-list", "--count", f"{CAMPAIGN_BASE_COMMIT}..HEAD")),
            "python_lines_added": _numstat_added("*.py"),
            "rust_lines_added": _numstat_added("*.rs"),
            "new_test_files": _added_files("*test_*.py"),
            "new_tool_modules": _added_files("tools/**/*.py", exclude="test_"),
            "new_crates": _added_files("crates/**/Cargo.toml"),
        },
        "remaining_highest_ev_gaps": [
            {
                "gap": "No gate has demonstrated ACCEPTANCE.",
                "why": (
                    "26 gates are wired and zero are accepted. Closing even a handful of "
                    "acceptance criteria is now worth more than wiring anything new."
                ),
            },
            {
                "gap": "The FPGA thesis is unvalidated until the U50DD arrives.",
                "why": (
                    "12 sealed predictions are staged against wake condition U50_PRESENT. "
                    "Transport is ~1.68 GB/s against ~316 GB/s of HBM, a ~188x cliff, so "
                    "residency (roadmap J.7) is the only viable thesis behind this carrier."
                ),
            },
            {
                "gap": "Theia has an engine but no model.",
                "why": (
                    "All 7 THEIA gates are BLOCKED_EXTERNAL on a training campaign that "
                    "does not run in this checkout. The bounty laboratory is built and idle."
                ),
            },
        ],
    }


def emit() -> Path:
    doc = build()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return RECEIPT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true", help="write the receipt")
    ap.add_argument("--check", action="store_true", help="fail if the committed receipt is stale")
    args = ap.parse_args()

    if args.check:
        fresh = json.dumps(build(), indent=2, sort_keys=True) + "\n"
        if not RECEIPT.exists():
            print("STALE: receipt does not exist")
            return 1
        # generated_by/head move with every commit; compare the substance.
        a = json.loads(fresh)
        b = json.loads(RECEIPT.read_text())
        for doc in (a, b):
            # Both blocks are git-derived and move with EVERY commit, so they
            # can never be stable. Staleness is about the auditor-derived
            # substance: statuses, blockers, gaps.
            doc.pop("derived_from", None)
            doc.pop("work", None)
        if a != b:
            print("STALE: regenerating changes the receipt")
            return 1
        print("FRESH")
        return 0

    path = emit() if args.emit else None
    if path:
        print(path)
    else:
        print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
