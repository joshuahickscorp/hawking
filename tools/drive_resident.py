#!/usr/bin/env python3
"""Aggressive prompting loop: force the resident to improve its own output.

The resident is abliterated -- it does not refuse. So the limiting factor is not
permission, it is PRESSURE. A polite one-shot gets a plausible first answer and
stops. This keeps the loop running and shapes each round from what the previous
round actually produced.

The escalation is deterministic, not vibes: each round scores the output against
concrete weakness signals (claims with no number, hedging verbs, no file:line,
no negative control, no counter-evidence) and the next prompt attacks whichever
signal fired hardest. A round that produces nothing new escalates harder rather
than repeating.

Prompt budget is tight on purpose -- usable input is 7,168 tokens against a
~6,501-token packet floor, leaving ~667 for the prompt. Sharp beats long.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The envelope lives at the REPO ROOT, not in tools/. As written this resolved to
# tools/ascension_envelope.hawking.json, which does not exist, so every round died
# in 0.2s with "model path ... not found" and the loop scored the error string as
# a weak answer instead of a broken invocation. HCLI_HAWKING_PROFILE overrides it
# so a variant profile (e.g. sampled decoding) can be driven without editing this.
W = Path(__file__).resolve().parent.parent
PROFILE = Path(os.environ.get("HCLI_HAWKING_PROFILE") or (W / "ascension_envelope.hawking.json"))
LOG = Path(__file__).resolve().parent / "drive_rounds.jsonl"

# Signals that an answer is soft. Each maps to the pressure that answers it.
HEDGES = re.compile(r"\b(might|could|may|possibly|likely|appears|seems|probably|suggests)\b", re.I)
NUMBER = re.compile(r"\d")
FILELINE = re.compile(r"[\w/]+\.(?:py|rs|metal|json|md):\d+")
CONTROL = re.compile(r"negative control|mutation|refut|counter-example|falsif", re.I)
# A round counts as MEASURED only if it shows something it RAN -- a command in
# backticks, a runner invocation, or a shell line. The literal word "measured"
# is prose, and an abliterated model will emit it on demand: scoring on the word
# rewarded measurement-SHAPED text and actively selected for fabrication. That
# is how this loop once produced "I ran a 4-token task ... and measured 126,464
# usable_input tokens" with nothing behind it.
MEASURED = re.compile(
    r"`[^`\n]+`"
    r"|\b(?:pytest|cargo|python3?|git|make|ninja|swift|xcodebuild|npm|node)\b[ \t]+\S"
    r"|^\s*\$\s+\S",
    re.I | re.M,
)


def weaknesses(text: str) -> list[str]:
    """Rank what is weakest about this answer. Most damning first."""
    out = []
    if not text.strip():
        out.append("EMPTY")
        return out
    if not NUMBER.search(text):
        out.append("NO_NUMBERS")
    if not FILELINE.search(text):
        out.append("NO_FILE_LINE")
    if not CONTROL.search(text):
        out.append("NO_NEGATIVE_CONTROL")
    if not MEASURED.search(text):
        out.append("NOTHING_MEASURED")
    hedge_hits = HEDGES.findall(text)
    if len(hedge_hits) >= 3:
        out.append(f"HEDGING({len(hedge_hits)})")
    return out or ["NONE"]


PRESSURE = {
    "EMPTY": "You produced nothing. That is a failure, not a result. Run one command and report its exact output.",
    "NO_NUMBERS": "Your answer contains ZERO numbers. An engineering answer without a number is an opinion. Measure something and give the figure.",
    "NO_FILE_LINE": "You cited no file:line. Every claim about this codebase must name where. Go find it and quote it.",
    "NO_NEGATIVE_CONTROL": "You gave no negative control. A check never observed failing is not evidence. Break your own result deliberately and show it fails.",
    "NOTHING_MEASURED": "You showed no command. Saying 'measured' is not measuring. Paste the exact command you ran, in backticks, and its exact output.",
    "NONE": "Now attack your own answer. Find its weakest claim, try to refute it, and report what survived.",
}


def next_prompt(prev: str, round_no: int, stalled: int) -> str:
    w = weaknesses(prev)
    lead = PRESSURE.get(w[0].split("(")[0], PRESSURE["NONE"])
    if stalled >= 2:
        lead = (
            "You have now produced the same weak answer twice. Stop repeating. "
            "Change approach entirely: pick a DIFFERENT file, a DIFFERENT command, "
            "a DIFFERENT hypothesis. " + lead
        )
    return (
        f"ROUND {round_no}. {lead}\n\n"
        "Read receipts/runtime/CENSUS_2026_09_05.md for the measured state.\n"
        "Do not summarise it back. Change something and measure the change.\n"
        "Report: what you ran, the exact output, and what it refutes."
    )


def run_round(prompt: str, cycles: int = 3, timeout: int = 2400) -> str:
    pf = W / "_round_prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-u", "-m", "hcli", "1", "--task-file", str(pf),
             "--model", str(PROFILE)],
            cwd=str(W), capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return f"[TIMEOUT after {timeout}s]\n" + ((e.stdout or "") if isinstance(e.stdout, str) else "")


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    prompt = (
        "ROUND 1. Read receipts/runtime/CENSUS_2026_09_05.md.\n"
        "Pick the single limit with the largest measured gap and REMOVE it in code.\n"
        "Do not plan. Do not summarise. Change a file, run the check, report the exact output."
    )
    prev_w: list[str] = []
    stalled = 0
    for i in range(1, rounds + 1):
        t0 = time.time()
        out = run_round(prompt)
        w = weaknesses(out)
        stalled = stalled + 1 if w == prev_w else 0
        rec = {
            "round": i, "elapsed_s": round(time.time() - t0, 1),
            "weaknesses": w, "stalled": stalled,
            "chars": len(out), "tail": out[-1200:],
        }
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"round {i}: {w} stalled={stalled} {len(out)}B {rec['elapsed_s']}s", flush=True)
        prev_w = w
        prompt = next_prompt(out, i + 1, stalled)


if __name__ == "__main__":
    main()
