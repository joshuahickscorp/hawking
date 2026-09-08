"""One confirmatory repeat is not a runaway loop. (G033, S010 10-11)

The loop closes the moment EVERY call in a round repeats one already made:

    if all(item.get("repeat") for item in round_observations):
        return "bounded_observation_round"

Rounds 22, 24 and 25 each measured real dense anatomy, re-read the ledger to
confirm it -- a correct verification instinct -- and that confirmation ended the
loop before they could write. All three closed this way with observations still
in budget.

S010: repeat detection should be INFORMATION, not automatic silent termination,
AND runaway protection must survive. So the contract is:

    ONE all-repeat round      -> keep going, the round may be confirming
    TWO IN A ROW              -> close, this is a no-progress loop

with the observation budget still bounding everything above it.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hcli.engine import Engine  # noqa: E402

C = Engine._tool_loop_closure
OK = {"ok": True}
REP = {"ok": True, "repeat": True}


def main() -> int:
    fails = []

    # ONE confirmatory repeat round must NOT end the loop.
    if C([REP], 2, 0, 8, 1, repeat_rounds=1) is not None:
        fails.append("a single all-repeat round closed the loop -- confirmation is not a runaway")

    # TWO IN A ROW must. Runaway protection survives.
    if C([REP], 3, 0, 8, 1, repeat_rounds=2) != "bounded_observation_round":
        fails.append("two consecutive all-repeat rounds did NOT close the loop -- "
                     "no-progress cycling is unbounded")

    # A round with ANY fresh call is progress, whatever else repeated.
    if C([REP, OK], 3, 0, 8, 1, repeat_rounds=1) is not None:
        fails.append("a round containing a fresh call was treated as no progress")

    # The budget still bounds everything above the repeat rule.
    if C([OK], 8, 0, 8, 1, repeat_rounds=0) != "observation_budget_8":
        fails.append("the observation budget stopped closing the loop")

    # And a failed round still closes past tolerance -- unchanged.
    if C([{"ok": False}], 2, 3, 8, 1, repeat_rounds=0) != "failed_call":
        fails.append("failure tolerance stopped closing the loop")

    # The budget must be checked BEFORE the repeat allowance, or a repeat could
    # buy a round past the ceiling.
    if C([REP], 8, 0, 8, 1, repeat_rounds=1) != "observation_budget_8":
        fails.append("an all-repeat round at the budget ceiling was allowed to continue")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — one confirmatory repeat survives, two in a row "
          f"close, budget and failure rules unchanged")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
