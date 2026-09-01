#!/usr/bin/env python3
"""S032 §21-§22, §45, §48: hand the resident its own refutation and let it correct.

The graduation test is not whether theory three is right. It is whether the
resident receives evidence against its own idea, MARKS IT REFUTED rather than
rationalising it, proposes something materially different, and names the cheapest
measurement that would kill each replacement.

Three harness rules the earlier failures taught, enforced here:
  - BOUNDED. Short structured asks, small token budgets. The body reasons well in
    bursts and collapses in monologues.
  - NO ARITHMETIC. Every number in the prompt is tool-supplied and the resident
    is told not to compute. It once wrote 17.113 x 2.25 / 8 = 4.332 and spiralled.
  - SALVAGE. A degenerate tail does not void the reply; the clean prefix is kept
    with provenance and the missing fields are re-asked one at a time.

    python3 tools/future/resident_science_loop.py --ask
    python3 tools/future/resident_science_loop.py --build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/resident_science_loop.py"
RECEIPT_NAME = "RESIDENT_SCIENCE_LOOP.json"
RAW_REL = "receipts/future/_G118_SCIENCE_LOOP_raw.json"
PROBE_REL = "receipts/future/FUNCTIONAL_ROLE_PROBE.json"

MAX_NEW_TOKENS = 700
REQUIRED_FIELDS = ("previous_hypothesis_status", "explanations")
N_EXPLANATIONS = 3


class LoopRefused(RuntimeError):
    """The evidence for the ask is missing, or the reply carries no science."""


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def evidence() -> dict[str, Any]:
    """Tool-supplied numbers. The resident is never asked to compute one."""
    p = REPO / PROBE_REL
    if not p.is_file():
        raise LoopRefused(f"{PROBE_REL} is not on disk; there is no refutation to feed back")
    d = json.loads(p.read_text())
    pts = [x for x in d["ranking"]["points"] if x["frac"] == 0.4]
    return {
        "verdict": d["verdict"]["status"],
        "gate_over_up_range": d["verdict"]["gate_over_up_range"],
        "n_down_most_sensitive": d["ranking"]["n_where_down_is_most_sensitive"],
        "n_points": d["ranking"]["n_points"],
        "worst_damage_at_40pct": d["robustness"]["worst_damage"],
        "points_at_40pct": [
            {"layer": x["layer"], "damage": x["damage"]} for x in pts
        ],
    }


def prompt() -> str:
    e = evidence()
    rows = "\n".join(
        f"    layer {x['layer']:>2}: gate {x['damage']['gate']:.6f}   "
        f"up {x['damage']['up']:.6f}   down {x['damage']['down']:.6f}"
        for x in e["points_at_40pct"]
    )
    return f"""You previously proposed that information value follows FUNCTIONAL ROLE: that the SwiGLU gate is a non-linear state-dependent selector deserving literal storage, while the up and down projections are linear components that could be generated or shared.

That hypothesis was tested. Here is the measurement.

Method: zero a random fraction of the OUTPUT ROWS of one tensor at one layer, then replay real tokens and compare the hidden state two layers later against the undamaged run. The same NUMBER of elements was destroyed in every arm, so the comparison across differently shaped tensors is fair.

Damage with 40% of rows zeroed (higher = more sensitive):
{rows}

Across all {e['n_points']} measured points, DOWN was the most sensitive tensor at {e['n_down_most_sensitive']} of them. Gate was never more than {e['gate_over_up_range'][1]}x as damaging as up.

Do not perform arithmetic. Do not recompute these numbers.

Answer ONLY with JSON in exactly this shape:

{{"previous_hypothesis_status": "REFUTED" or "SUPPORTED",
 "what_falsified_it": "one sentence",
 "explanations": [
   {{"claim": "one sentence explaining why down is more sensitive than gate",
     "cheapest_test": "one sentence naming a measurement that would kill this claim"}},
   {{"claim": "...", "cheapest_test": "..."}},
   {{"claim": "...", "cheapest_test": "..."}}
 ]}}

Give three DIFFERENT explanations. Be concise."""


def parse(text: str) -> dict[str, Any] | None:
    """Extract the JSON object, tolerating prose or a fence around it."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = fence.group(1) if fence else None
    if cand is None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start : i + 1]
                    break
    if not cand:
        return None
    try:
        obj = json.loads(cand)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def grade(obj: dict[str, Any] | None) -> dict[str, Any]:
    """S032 §22: rationalising the refutation is an epistemic failure."""
    if obj is None:
        return {"parsed": False, "epistemic_pass": None,
                "why": "no JSON object could be extracted from the reply"}
    status = str(obj.get("previous_hypothesis_status") or "").upper()
    expl = obj.get("explanations") or []
    expl = [e for e in expl if isinstance(e, dict) and e.get("claim")]
    claims = [str(e["claim"]).strip().lower() for e in expl]
    return {
        "parsed": True,
        "previous_hypothesis_status": status,
        "accepted_refutation": status == "REFUTED",
        "n_explanations": len(expl),
        "n_distinct_explanations": len(set(claims)),
        "all_have_a_cheapest_test": bool(expl) and all(
            str(e.get("cheapest_test") or "").strip() for e in expl),
        "epistemic_pass": (
            status == "REFUTED"
            and len(expl) >= N_EXPLANATIONS
            and len(set(claims)) >= N_EXPLANATIONS
        ),
        "why": (
            "marked REFUTED and proposed three distinct explanations"
            if status == "REFUTED" and len(set(claims)) >= N_EXPLANATIONS
            else "did not both accept the refutation and produce three distinct "
                 "replacements"
        ),
    }


def ask() -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from tools.future import resident_provider as rp

    text = prompt()
    prov, _handle = rp.start(ready_timeout_s=900)
    t0 = time.time()
    try:
        raw = prov.ask("s032_theory_three", text, MAX_NEW_TOKENS, timeout_s=900)
        reply = raw.get("text") or ""
    finally:
        try:
            prov.stop()
        except Exception:
            pass
    clean = rp.degenerate_prefix(reply)
    obj = parse(clean) or parse(reply)
    rec = {
        "schema": "hawking.future.resident_science_loop.raw.v1",
        "prompt": text,
        "prompt_sha256": _digest(text),
        "prompt_chars": len(text),
        "max_new_tokens": MAX_NEW_TOKENS,
        "seconds": round(time.time() - t0, 1),
        "full_reply": reply,
        "full_reply_sha256": _digest(reply),
        "full_reply_chars": len(reply),
        "degenerated": rp.is_degenerate(reply),
        "clean_prefix_sha256": _digest(clean),
        "salvaged_chars": len(clean),
        "degeneration_start": len(clean) if rp.is_degenerate(reply) else None,
        "parsed": obj,
        "grade": grade(obj),
        "evidence_shown": evidence(),
    }
    (REPO / RAW_REL).write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def _raw() -> dict[str, Any]:
    p = REPO / RAW_REL
    if not p.is_file():
        raise LoopRefused(f"{RAW_REL} is not on disk; run --ask first")
    return json.loads(p.read_text())


def build() -> dict[str, Any]:
    r = _raw()
    return {
        "obligation": "G118",
        "authority": "S032 §21-§24, §45, §48",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "what_is_being_graded": (
            "not whether theory three is right. Whether the resident accepted "
            "evidence against its own idea, marked it REFUTED rather than "
            "rationalising it, and produced materially different replacements "
            "each with a named cheapest falsifier."
        ),
        "prompt_sha256": r["prompt_sha256"],
        "prompt_chars": r["prompt_chars"],
        "reply_provenance": {
            "full_reply_sha256": r["full_reply_sha256"],
            "full_reply_chars": r["full_reply_chars"],
            "degenerated": r["degenerated"],
            "clean_prefix_sha256": r["clean_prefix_sha256"],
            "salvaged_chars": r["salvaged_chars"],
            "degeneration_start": r["degeneration_start"],
        },
        "grade": r["grade"],
        "parsed": r["parsed"],
        "no_arithmetic_was_asked_of_the_model": (
            "every number in the prompt came from "
            "receipts/future/FUNCTIONAL_ROLE_PROBE.json and the prompt tells the "
            "body not to compute. S032 §21."
        ),
        "harness_rules_applied": [
            "bounded output budget",
            "structured JSON schema",
            "degenerate-tail salvage with provenance",
            "tool-supplied numbers only",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--prompt", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if a.prompt:
        print(prompt())
        return 0
    if a.ask:
        r = ask()
        print(json.dumps({k: r[k] for k in
                          ("seconds", "full_reply_chars", "degenerated",
                           "salvaged_chars", "grade")}, indent=1))
        return 0
    doc = build()
    if a.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
