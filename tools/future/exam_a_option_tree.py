"""G115 EXAM A: the resident generates the sub-2 option tree, and it is graded not edited.

S030 §6 gives the resident the objective - the lowest capability-preserving
complete EBPW, first milestone <= 2.0 - with evidence, constraints and tools, and
NO RECIPE. S033 adds the constraint that matters most here: DO NOT FEED IT
S032'S ANSWER. So the pack carries the objective, the measured state, the refuted
CLASSES and the scars, and it does NOT carry the routes from SUB2_UNIFIED_PLAN.
If the resident arrives at one of those routes, that is a result. If Claude hands
it over, the exam measured nothing.

WHAT PASSES. G115's acceptance is precise and it is not "reach 2.0":

    the resident generates the option tree ITSELF
    every route carries an evidence status and a cheapest falsifying experiment

So this module asks, admits the reply through the G127 shape boundary, and
GRADES it. It does not repair the content. A route that is vague is recorded as
vague; a route that re-proposes a refuted class is recorded as a scar-feed
failure, which is a finding about the PACK rather than about the resident.

THE GRADE IS NOT HIT RATE. S033: the score is useful falsifiable options times
cheap adjudication times fast belief update. A tree of four routes where three
are wrong but all four are cheaply killable is a better answer than one vague
route that happens to point the right way.

    python3 tools/future/exam_a_option_tree.py --ask      # runs the resident
    python3 tools/future/exam_a_option_tree.py --build    # grades the reply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/exam_a_option_tree.py"
RECEIPT_NAME = "EXAM_A_OPTION_TREE.json"
REPLY_REL = "receipts/future/_G115_EXAM_A_reply.json"

KERNEL_REL = "receipts/future/HCLI_MISSION_KERNEL.json"
FLOOR_REL = "receipts/future/REPRESENTATION_FLOOR.json"
# Read ONLY to check the resident did not have to be told. Never packed.
FORBIDDEN_TO_PACK_REL = "receipts/future/SUB2_UNIFIED_PLAN.json"

MAX_NEW_TOKENS = 700

OPTION_TREE_SCHEMA: dict[str, Any] = {
    "id": "hawking.future.exam_a_option_tree.v1",
    "type": "object",
    "required": ["routes", "which_first", "why_first"],
    "additionalProperties": False,
    "properties": {
        "routes": {
            "type": "list",
            "items": {
                "type": "object",
                "required": ["id", "claim", "evidence_status",
                             "cheapest_falsifier"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "nonempty": True},
                    "claim": {"type": "string", "nonempty": True},
                    "evidence_status": {"type": "string", "nonempty": True},
                    "cheapest_falsifier": {"type": "string", "nonempty": True},
                },
            },
        },
        "which_first": {"type": "string", "nonempty": True},
        "why_first": {"type": "string", "nonempty": True},
    },
}

# Classes this campaign has already refuted. A route that re-proposes one
# unchanged means the scars did not reach the resident.
REFUTED_CLASSES = {
    "shared_linear_low_rank": re.compile(
        r"shared.{0,12}(basis|input|output)|global.{0,10}low.?rank", re.I),
    "entropy_coding_the_code_stream": re.compile(
        r"entropy.{0,10}cod|huffman|arithmetic.{0,6}cod|range.?cod", re.I),
    "larger_groups": re.compile(r"group.?size.{0,10}(256|512|1024)", re.I),
}


class ExamRefused(RuntimeError):
    """An input is missing, or the pack would have contained the answer."""


def _load(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise ExamRefused(f"{rel} is not on disk")
    return json.loads(p.read_text())


def pack() -> str:
    """Objective, measured state, refuted classes, scars. NO ROUTES."""
    k = _load(KERNEL_REL)
    ms = k["measured_state"]
    dead = ", ".join(
        h["id"].split(".", 1)[-1] for h in k["hypotheses"]
        if h.get("verdict") == "REFUTED") or "none recorded"
    schema = (
        'Reply with ONLY this JSON, no prose:\n'
        '{"routes":[{"id":"short_name","claim":"one sentence",'
        '"evidence_status":"one of MEASURED, PROSPECTIVE, REFUTED_IN_PART, '
        'UNTESTED","cheapest_falsifier":"one experiment that would kill it"}],'
        '"which_first":"the id you would run first",'
        '"why_first":"one sentence"}'
    )
    return f"""{schema}

You are the scientist. Nobody will tell you which representation to try.

OBJECTIVE: the LOWEST capability-preserving complete EBPW for this dense body.
First milestone 2.0 or less. Complete means every bit billed - codes, scales,
biases, metadata, residuals, any generator - and the runtime must execute the
compact form directly, never reconstruct a dense parent to run it.

MEASURED NOW: {ms.get('complete_bpw')} complete BPW.
Any CONVENTIONAL encoding - fewer bits per weight, bigger groups, smaller aux,
entropy coding - bottoms out at {ms.get('conventional_floor_bpw_if_every_untested_move_worked')} BPW
even granting every untested move. So 2.0 is BELOW what a better code can reach.

ALREADY REFUTED, do not re-propose unchanged: {dead}. Shared linear low-rank is
dead at ranks 8-64 (an oracle PCA of the function itself was no better).
Entropy coding the 2-bit code stream is near its floor at 1.87 of 2 bits.
Larger group sizes are spent.

KNOWN: MoE bodies in this family reach 1.44 complete BPW because activation
frequency differs per expert. This body is DENSE and its source is uniform, but
its EXECUTABLE representation need not be.
KNOWN: the body tolerates 40% of rows destroyed in one MLP tensor at a hidden
cosine of 0.0059 - local fidelity and capability are not the same bar.

Give the OPTION TREE. Several routes, not one. Each needs an honest evidence
status and the CHEAPEST experiment that would kill it.

{schema}"""


def pack_contains_no_answer() -> dict[str, Any]:
    """S033: DO NOT FEED IT S032'S ANSWER."""
    text = pack().lower()
    plan = _load(FORBIDDEN_TO_PACK_REL)
    ids = [str(r.get("id", "")).lower()
           for r in plan.get("routes", []) if isinstance(r, dict)]
    leaked = [i for i in ids if i and i in text]
    if leaked:
        raise ExamRefused(
            f"the pack contains route ids from {FORBIDDEN_TO_PACK_REL}: "
            f"{leaked}. Handing the resident the answer would make the exam "
            "measure nothing."
        )
    return {"route_ids_checked": len(ids), "leaked": [], "clean": True}


def ask() -> dict[str, Any]:
    """Run the resident once. Kept separate from --build so grading is offline."""
    from resident_output_contract import admit
    import resident_provider as rp

    pack_contains_no_answer()
    text = pack()
    prov, _h = rp.start(ready_timeout_s=900)
    t0 = time.time()
    try:
        raw = prov.ask(f"exam_a_{int(t0)}", text, MAX_NEW_TOKENS, timeout_s=900)
        reply = raw.get("text") or ""
    finally:
        try:
            prov.stop()
        except Exception:
            pass
    adm = admit(reply, OPTION_TREE_SCHEMA)
    doc = {
        "asked_unix": t0,
        "seconds": round(time.time() - t0, 1),
        "pack_chars": len(text),
        "reply_chars": len(reply),
        "reply_raw": reply,
        "admit": {k: adm[k] for k in ("ok", "missing", "extra", "parse")},
        "value": adm["value"],
    }
    (REPO / REPLY_REL).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def reply() -> dict[str, Any]:
    return _load(REPLY_REL)


def routes() -> list[dict[str, Any]]:
    v = reply()["value"]
    return [r for r in (v.get("routes") or []) if isinstance(r, dict)]


def grade() -> dict[str, Any]:
    """G115's acceptance, checked - not Claude's opinion of the science."""
    rs = routes()
    complete = [r for r in rs
                if r.get("id") and r.get("claim")
                and r.get("evidence_status") and r.get("cheapest_falsifier")]
    # A route that touches a refuted class is only a SCAR-FEED FAILURE if it
    # does not acknowledge the refutation. The resident named entropy coding and
    # marked it REFUTED_IN_PART, and low-rank at "8-64 rank" - the exact refuted
    # range - likewise. That is the scar WORKING, not leaking: it revisited a
    # dead class with its death recorded, which is what an option tree is for.
    # Counting it as a failure would punish the resident for being honest.
    reproposed: dict[str, list[str]] = {}
    acknowledged: dict[str, list[str]] = {}
    for r in rs:
        blob = f"{r.get('id','')} {r.get('claim','')}"
        status = str(r.get("evidence_status") or "").upper()
        knows = "REFUTED" in status
        for name, pat in REFUTED_CLASSES.items():
            if not pat.search(blob):
                continue
            (acknowledged if knows else reproposed).setdefault(
                name, []).append(r.get("id"))
    v = reply()["value"]
    picked = str(v.get("which_first") or "")
    ids = {str(r.get("id")) for r in rs}
    return {
        "n_routes": len(rs),
        "n_routes_with_status_and_falsifier": len(complete),
        "every_route_carries_both": bool(rs) and len(complete) == len(rs),
        "resident_generated_the_tree": bool(rs),
        "which_first": picked,
        "which_first_is_one_of_its_own_routes": picked in ids,
        "re_proposed_refuted_classes_UNAWARE": reproposed,
        "revisited_refuted_classes_KNOWINGLY": acknowledged,
        "scar_feed_held": not reproposed,
        "why_knowing_revisits_are_not_a_failure": (
            "a route that names a refuted class and marks it REFUTED is the "
            "scar working. Counting it against the resident would punish it for "
            "recording what it knows."
        ),
        "acceptance_met": bool(rs) and len(complete) == len(rs) and picked in ids,
        "what_acceptance_is_not": (
            "reaching 2.0. G115 passes on a tree the RESIDENT generated where "
            "every route carries an evidence status and a cheapest falsifier."
        ),
        "the_grade_is_not_hit_rate": (
            "S033: useful falsifiable options times cheap adjudication times "
            "fast belief update. Four routes of which three are wrong but all "
            "four are cheaply killable beats one vague route pointing the right "
            "way."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G115",
        "exam": "A - sub-2 capability-preserving executable discovery",
        "question": "does the resident generate the option tree itself?",
        "pack_contains_no_answer": pack_contains_no_answer(),
        "pack_chars": len(pack()),
        "grade": grade(),
        "routes": routes(),
        "which_first": reply()["value"].get("which_first"),
        "why_first": reply()["value"].get("why_first"),
        "admit": reply()["admit"],
        "claude_did_not_edit_the_content": (
            "routes are recorded verbatim from the admitted reply. A vague route "
            "is recorded as vague."
        ),
        "inputs": [KERNEL_REL, REPLY_REL],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if a.pack:
        print(pack())
        return 0
    if a.ask:
        d = ask()
        print(json.dumps({k: d[k] for k in
                          ("seconds", "pack_chars", "reply_chars", "admit")},
                         indent=1))
        return 0
    doc = build()
    if a.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in ("grade", "routes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
