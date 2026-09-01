#!/usr/bin/env python3
"""G098: obligations carry a mission class, and Odyssey launch consults one.

S026 §44: THE LEDGER IS NOT THE MISSION. Driving the unmet count to zero stopped
being a precondition for starting Odyssey. Every unmet obligation is classified
and the launch gate reads only the LAUNCH_CRITICAL subset.

Classification is read from the ledger where an obligation declares it and
DERIVED where it does not, from what the obligation's own text says it blocks.
Nothing is classified by guessing importance: an obligation whose text gives no
signal is UNCLASSIFIED and the gate treats it as blocking, because silently
demoting something nobody understood is how a launch gate becomes decorative.

    python3 tools/future/mission_class.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/mission_class.py"
RECEIPT_NAME = "MISSION_CLASS.json"
GOAL = Path.home() / ".claude/ultragoal/hawking-global-parallel-civilization/GOAL.md"

CLASSES = ("LAUNCH_CRITICAL", "TPS_CRITICAL", "PROMOTION_CRITICAL", "DEFERRED")

# Derivation rules, applied in order, to obligations that declare no class.
# Each is a (class, why, matcher) triple over the obligation's own lowercased
# text. These encode what the obligation SAYS it blocks, not how important it
# feels.
DERIVE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("LAUNCH_CRITICAL",
     "the obligation's own text makes it a precondition for autonomous operation",
     ("autonomy trial", "torture passed", "power torture", "scheduler liveness",
      "no idle", "model-bearing")),
    ("TPS_CRITICAL",
     "the obligation targets resident token time",
     ("tps", "ms/token", "state machine, not four matrices", "decode",
      "bandwidth", "kernel")),
    ("PROMOTION_CRITICAL",
     "the obligation gates promoting an artifact rather than running the mission",
     ("promot", "qualification", "seal", "pareto")),
    ("DEFERRED",
     "housekeeping: it changes no measurement and blocks no experiment",
     ("git history", "attribution", "co-authored-by", "force-push", "rename")),
)


class ClassifyRefused(RuntimeError):
    """The ledger is missing, or a declared class is not one of the four."""


def _ledger() -> str:
    if not GOAL.is_file():
        raise ClassifyRefused(f"{GOAL} is not on disk")
    return GOAL.read_text()


def _blocks() -> list[dict[str, str]]:
    """Every UNMET obligation with its id and its own text."""
    txt = _ledger()
    lines = txt.splitlines()
    out: list[dict[str, str]] = []
    cur: dict[str, Any] | None = None
    for line in lines:
        m = re.match(r"^- \[ \] (G\d+)\b(.*)$", line)
        if m:
            if cur:
                out.append({"id": cur["id"], "text": "\n".join(cur["body"])})
            cur = {"id": m.group(1), "body": [m.group(2)]}
            continue
        if re.match(r"^- \[.\]", line) or re.match(r"^## ", line):
            if cur:
                out.append({"id": cur["id"], "text": "\n".join(cur["body"])})
                cur = None
            continue
        if cur is not None:
            cur["body"].append(line)
    if cur:
        out.append({"id": cur["id"], "text": "\n".join(cur["body"])})
    return out


def classify_one(ob: dict[str, str]) -> dict[str, Any]:
    text = ob["text"]
    declared = re.search(r"class:\s*([A-Z_]+)", text)
    if declared:
        c = declared.group(1)
        if c not in CLASSES:
            raise ClassifyRefused(
                f"{ob['id']} declares class {c!r}, which is not one of {CLASSES}"
            )
        return {"id": ob["id"], "mission_class": c, "source": "DECLARED",
                "why": "the obligation states its own class"}
    low = text.lower()
    for cls, why, needles in DERIVE:
        hit = next((n for n in needles if n in low), None)
        if hit:
            return {"id": ob["id"], "mission_class": cls, "source": "DERIVED",
                    "why": why, "matched": hit}
    return {
        "id": ob["id"], "mission_class": "UNCLASSIFIED", "source": "NONE",
        "why": (
            "the obligation's text gives no signal about what it blocks. It is "
            "NOT demoted: the gate treats UNCLASSIFIED as blocking, because "
            "silently demoting something nobody understood is how a launch gate "
            "becomes decorative."
        ),
    }


def classified() -> list[dict[str, Any]]:
    return [classify_one(b) for b in _blocks()]


def launch_gate() -> dict[str, Any]:
    """S026 §44: the gate reads LAUNCH_CRITICAL and UNCLASSIFIED, nothing else."""
    rows = classified()
    blocking = [r for r in rows
                if r["mission_class"] in ("LAUNCH_CRITICAL", "UNCLASSIFIED")]
    by_class: dict[str, list[str]] = {}
    for r in rows:
        by_class.setdefault(r["mission_class"], []).append(r["id"])
    return {
        "n_unmet_total": len(rows),
        "by_class": {k: sorted(v) for k, v in sorted(by_class.items())},
        "blocking_ids": sorted(r["id"] for r in blocking),
        "n_blocking": len(blocking),
        "green": not blocking,
        "rule": (
            "Odyssey launch consults LAUNCH_CRITICAL and UNCLASSIFIED only. "
            "TPS_CRITICAL, PROMOTION_CRITICAL and DEFERRED obligations stay open "
            "and keep being worked; they do not hold the launch."
        ),
        "what_this_changes": (
            f"{len(rows)} obligations are unmet and {len(blocking)} of them "
            "block launch. Before this classifier the answer was that all "
            f"{len(rows)} did, which is the ceremonial gate S026 §44 named."
        ),
    }


def build() -> dict[str, Any]:
    g = launch_gate()
    return {
        "obligation": "G098",
        "authority": "S026 §44",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "classes": list(CLASSES),
        "classified": classified(),
        "launch_gate": g,
        "derivation_is_from_the_text_not_from_importance": (
            "an obligation that declares a class keeps it. One that does not is "
            "matched against what its own text says it blocks - an autonomy "
            "trial, a token-time target, a promotion, or housekeeping. No rule "
            "here encodes a judgement about how important an obligation feels."
        ),
        "unclassified_blocks_rather_than_defers": (
            "the default is BLOCKING. A classifier whose unknown case defers is "
            "a classifier that empties the gate as soon as it stops "
            "understanding the ledger."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({"launch_gate": doc["launch_gate"],
                      "classified": doc["classified"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
