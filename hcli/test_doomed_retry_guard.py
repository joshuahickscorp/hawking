"""A retry with less room than the attempt that ran out of room cannot succeed.

Round 23 measured it: the structured-output contract was granted 2474, then 990,
then 990 completion tokens against prompts of 5571, 5876 and 5959 -- all three
ending finish_reason "length". The engine re-derives the budget against the
POSTED prompt and only ever shrinks it, and every retry appends its own error
note to that prompt, so each answer to "you ran out of room" carried less room.
Two of the three attempts were spent on arithmetic that guaranteed their failure,
and the round closed reporting what reads like a model that cannot follow a
schema.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hcli.backends import (  # noqa: E402
    CompletionResult, StructuredOutputContract, StructuredOutputExhausted,
    doomed_retry,
)

SCHEMA = {"type": "object", "properties": {"kind": {"type": "string"}},
          "required": ["kind"]}


def _contract(max_attempts=3):
    return StructuredOutputContract(schema=SCHEMA, instruction="Return one JSON object.",
                                    max_attempts=max_attempts)


def _lengths(budgets):
    """complete_fn that always truncates for length, walking `budgets`.

    The budget for attempt N must already be on the payload WHEN attempt N is
    recorded, so each call sets the budget for the NEXT one. Setting it on entry
    made every attempt report its predecessor's budget and the guard fired on a
    growing sequence -- a defect in this harness, not in the guard.
    """
    calls = {"n": 0}

    def fn(payload, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        if i + 1 < len(budgets):
            payload["max_tokens"] = budgets[i + 1]
        return CompletionResult(raw={}, text='{"kind": "answ',
                                finish_reason="length",
                                completion_tokens=budgets[i] if i < len(budgets) else None)
    return fn, calls


def main() -> int:
    fails = []

    # THE DEFECT. Budget shrinks after a length truncation -> the guard must
    # stop, not spend the remaining attempts.
    c = _contract()
    fn, calls = _lengths([2474, 990, 990])
    try:
        c.enforce(fn, {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2474})
        fails.append("shrinking budget after a length truncation did not raise")
    except StructuredOutputExhausted as exc:
        if "less room than the attempt that ran out of room" not in str(exc):
            fails.append(f"raised, but not with the budget diagnosis: {str(exc)[:120]}")
        if calls["n"] > 2:
            fails.append(f"spent {calls['n']} attempts; the guard should stop at the 2nd")
    except Exception as exc:  # noqa: BLE001
        fails.append(f"unexpected {type(exc).__name__}: {exc}")

    # NEGATIVE CONTROL 1: the predicate itself, tested directly.
    # A payload-level fixture cannot express "the budget grew": the contract
    # shrinks the number twice on its own -- apply() reserves for the schema
    # instruction (990 -> 985) and the repair payload derives again (-> 945) --
    # regardless of what a caller sets. So the policy is checked where it lives.
    cases = [
        # (attempt, last_finish_reason, requested, prev, expected)
        (2, "length", 990, 2474, True),    # the round-23 shape
        (2, "length", 985, 985, True),     # flat is doomed too: no more room
        (2, "length", 4096, 990, False),   # GREW -- a legitimate retry
        (2, "stop", 990, 2474, False),     # schema violation, not length
        (1, "length", 990, 2474, False),   # first attempt has no predecessor
        (2, "length", None, 2474, False),  # unknown budget: do not guess
        (2, "length", 990, None, False),
    ]
    for attempt, fr, req, prev, want in cases:
        got = doomed_retry(attempt, fr, req, prev)
        if got != want:
            fails.append(f"doomed_retry({attempt}, {fr!r}, {req}, {prev}) = {got}, want {want}")

    # NEGATIVE CONTROL 2: a SCHEMA violation (not a length truncation) with a
    # flat budget must still retry -- a smaller budget is irrelevant there.
    calls3 = {"n": 0}

    def bad_shape(payload, timeout=None):
        calls3["n"] += 1
        return CompletionResult(raw={}, text='{"wrong": 1}', finish_reason="stop")

    c3 = _contract()
    try:
        c3.enforce(bad_shape, {"messages": [{"role": "user", "content": "x"}],
                               "max_tokens": 990})
    except StructuredOutputExhausted as exc:
        if "less room than the attempt" in str(exc):
            fails.append("a SCHEMA violation was refused by the length guard")
    except Exception:
        pass
    if calls3["n"] < 3:
        fails.append(f"a schema violation stopped after {calls3['n']} attempts; all 3 are owed")

    for f in fails:
        print("FAIL:", f)
    print(f"{'FAILED' if fails else 'PASS'} — doomed retry refused, growing budget and "
          f"schema violations still retried")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
