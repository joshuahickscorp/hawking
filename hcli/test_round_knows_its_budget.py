"""A round that does not know its horizon plans for a turn it will not get.

Rounds 22 and 24 both measured real dense anatomy on the same specimen, reported
it correctly, and ended with "Next: record ... with a receipt" -- without ever
calling odyssey.record_measurement. MEASURED stayed at 94 across both.

That reads as a model ignoring an instruction until you look at what it was
told. The tool-loop prompt says "Results come back as OBSERVATIONS and you are
asked again", and nothing anywhere states how many times. The budget is only
mentioned once it is GONE, in the final no-tools turn. A round deferring the
write to a next iteration is reasoning correctly about a horizon the harness
declined to describe.

So the observations block must carry the remaining budget on EVERY turn, not
only at exhaustion.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hcli.engine import Engine  # noqa: E402

OBS = [{"tool": "odyssey.dense_anatomy", "ok": True, "text": "k_proj 41.09%"}]


def main() -> int:
    fails = []
    block = Engine._observations_block.__get__(object.__new__(Engine))

    mid = block(OBS, used=1, budget=8)
    if "7" not in mid or "remain" not in mid.lower():
        fails.append(f"mid-loop block does not state the remaining budget: {mid[-160:]!r}")
    if "EXHAUSTED" in mid:
        fails.append("mid-loop block claims the budget is exhausted when 7 remain")

    # The last usable turn must not merely COUNT down to one -- it must say that
    # deferring loses the action. Asserting only "1" and "remain" here passed
    # while the last-chance branch was disabled, because the ordinary countdown
    # string contains both.
    last = block(OBS, used=7, budget=8)
    if "LAST CHANCE" not in last.upper():
        fails.append(f"the final usable turn does not warn that deferring loses it: {last[-200:]!r}")
    if "not happen" not in last and "lost" not in last:
        fails.append("the last-chance turn does not say what is lost by planning a next step")
    # and the ordinary turn must NOT cry wolf
    if "LAST CHANCE" in mid.upper():
        fails.append("a mid-loop turn with 7 remaining claims to be the last chance")

    # Exhaustion still reads as exhaustion.
    fin = block(OBS, final=True, used=8, budget=8)
    if "EXHAUSTED" not in fin:
        fails.append("the exhausted turn lost its TOOL BUDGET EXHAUSTED notice")

    # And an unknown budget must not invent one.
    unk = block(OBS)
    if "remain" in unk.lower():
        fails.append("a block with no budget information claims a remaining count")

    # THE CALL SITE. A block that can carry the horizon and is never given it is
    # the same defect wearing a fix. Every helper test in this campaign that
    # skipped this check passed while the feature stayed unreachable.
    src = (REPO / "hcli" / "engine.py").read_text()
    body = src.split("def _observations_block", 1)[1]
    if "used=len(observations)" not in body or "budget=self.MAX_TOOL_OBSERVATIONS" not in body:
        fails.append("no call site passes used/budget -- the horizon is computed and never sent")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — the round is told its horizon on every turn")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
