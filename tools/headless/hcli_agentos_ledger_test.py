#!/usr/bin/env python3
"""Protected HCLI AgentOS obligation-ledger checks. Plain python3 + assert.

No GPU, no model, no network. run_verify subprocesses are trivial local commands.

Run:
    python3 tools/headless/hcli_agentos_ledger_test.py
    pytest tools/headless/hcli_agentos_ledger_test.py -q

Every check 2-9 was watched FAILING against the naive first draft.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


MIXED_MD = """# GOAL

- [ ] G001 — first obligation | status: PENDING | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: (none yet)
- [ ] G002 — second obligation | status: ACTIVE | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: (none yet)
- [x] G003 — third obligation | status: VERIFIED | risk: low | tier: V0
      acceptance: c exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: already done
"""

ALL_VERIFIED_MD = """# GOAL

- [x] G001 — first obligation | status: VERIFIED | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: done-1
- [x] G002 — second obligation | status: VERIFIED | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: done-2
- [x] G003 — third obligation | status: VERIFIED | risk: low | tier: V0
      acceptance: c exists
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: done-3
"""


def _write(md: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False,
        encoding="utf-8",
        newline="\n",
    )
    handle.write(md)
    handle.close()
    return Path(handle.name)


def check_1_parse_round_trip():
    from hcli.ledger import Ledger

    path = _write(MIXED_MD)
    try:
        led = Ledger.parse(path)
        out = led.to_markdown()
        path2 = _write(out)
        try:
            led2 = Ledger.parse(path2)
            check(
                "1 parse round-trips",
                led == led2 and len(led) == 3,
                f"equal={led == led2} len={len(led)} len2={len(led2)}",
            )
        finally:
            path2.unlink(missing_ok=True)
    finally:
        path.unlink(missing_ok=True)


def _earn_verified(md_text):
    """Write a ledger and VERIFY it for real, so parse() keeps the marks.

    A hand-written `[x] ... status: VERIFIED` is a forgery as far as
    Ledger.parse is concerned: with no verify receipt beside it, the status is
    demoted to STALE. Checks that need a genuinely satisfied ledger therefore
    have to earn it the way production does -- run_verify + mark_verified,
    which writes the sidecar receipt parse() looks for.
    """
    from hcli.ledger import Ledger

    root = Path(tempfile.mkdtemp())
    marker = root / "marker.txt"
    marker.write_text("ok\n", encoding="utf-8")
    path = root / "GOAL.md"
    path.write_text(md_text, encoding="utf-8")
    led = Ledger.parse(path)
    cmd = (
        "python3 -c \"import pathlib,sys; "
        "sys.exit(0 if pathlib.Path(r'%s').read_text().strip()=='ok' else 1)\""
    ) % marker
    for ob in led.obligations():
        ob.verify_command = cmd
        result = led.run_verify(ob.id)
        assert result.passed, f"fixture verify failed: {result}"
        led.mark_verified(ob.id, result)
    led.save(path)
    return path


def check_2_is_goal_met():
    from hcli.ledger import Ledger

    mixed = Ledger.parse(_write(MIXED_MD))
    verified_path = _earn_verified(ALL_VERIFIED_MD)
    all_v = Ledger.parse(verified_path)
    false_ok = mixed.is_goal_met() is False
    true_ok = all_v.is_goal_met() is True and len(all_v) >= 3
    check(
        "2 is_goal_met both directions",
        false_ok and true_ok,
        f"mixed={mixed.is_goal_met()} all_verified={all_v.is_goal_met()} "
        f"all_len={len(all_v)}",
    )


def check_3_mark_verified_type():
    from hcli.ledger import Ledger

    led = Ledger()
    ob = led.add(
        "typed verify",
        verify_command='python3 -c "import sys; sys.exit(0)"',
    )
    rejected_true = False
    rejected_str = False
    try:
        led.mark_verified(ob.id, True)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rejected_true = True
    except Exception:
        rejected_true = True
    try:
        led.mark_verified(ob.id, "yes")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rejected_str = True
    except Exception:
        rejected_str = True
    check(
        "3 mark_verified requires VerifyResult",
        rejected_true and rejected_str,
        f"rejected_true={rejected_true} rejected_str={rejected_str}",
    )


def check_4_run_verify_executes():
    from hcli.ledger import Ledger

    tmp = Path(tempfile.mkdtemp())
    marker = tmp / "marker.txt"
    led = Ledger()
    ok = led.add(
        "write marker",
        verify_command=(
            "python3 -c "
            f"\"open(r'{marker}', 'w').write('ok')\""
        ),
    )
    fail = led.add(
        "failing command",
        verify_command='python3 -c "import sys; sys.exit(7)"',
    )
    led.run_verify(ok.id)
    failing = led.run_verify(fail.id)
    check(
        "4 run_verify executes the command",
        marker.is_file()
        and failing.passed is False
        and failing.exit_code == 7,
        f"marker_exists={marker.is_file()} passed={failing.passed} "
        f"exit_code={failing.exit_code}",
    )


def check_5_empty_ledger():
    from hcli.ledger import Ledger

    led = Ledger()
    check(
        "5 empty ledger is not a met goal",
        led.is_goal_met() is False and led.status == "EMPTY_LEDGER",
        f"is_goal_met={led.is_goal_met()} status={led.status!r}",
    )


def check_6_assert_may_complete():
    from hcli.ledger import GoalNotMetError, Ledger

    mixed_path = _write(MIXED_MD)
    mixed = Ledger.parse(mixed_path)
    mixed_path.unlink(missing_ok=True)
    raised = False
    ids: list[str] = []
    try:
        mixed.assert_may_complete()
    except GoalNotMetError as exc:
        raised = True
        raw = list(exc.unverified)
        for item in raw:
            if isinstance(item, tuple):
                ids.append(str(item[0]))
            else:
                ids.append(str(item))
        if "G001" not in ids:
            # also accept ids appearing in the message
            if "G001" in str(exc):
                ids.append("G001")
            if "G002" in str(exc):
                ids.append("G002")
    except Exception as exc:
        check(
            "6 assert_may_complete",
            False,
            f"wrong exception {type(exc).__name__}: {exc}",
        )
        return

    verified_path = _earn_verified(ALL_VERIFIED_MD)
    all_v = Ledger.parse(verified_path)
    clean = False
    if all_v.is_goal_met() and len(all_v) >= 3:
        try:
            all_v.assert_may_complete()
            clean = True
        except Exception as exc:
            clean = False
            check(
                "6 assert_may_complete",
                False,
                f"raised when met: {type(exc).__name__}: {exc}",
            )
            return
    check(
        "6 assert_may_complete",
        raised and ("G001" in ids) and ("G002" in ids) and clean,
        f"raised={raised} ids={ids} clean={clean}",
    )


def check_7_no_progress():
    from hcli.ledger import Ledger

    led = Ledger()
    ob = led.add(
        "stable verify",
        verify_command='python3 -c "import sys; sys.exit(0)"',
    )
    led.run_verify(ob.id)
    after_first = led.consecutive_no_progress_count()
    led.run_verify(ob.id)
    led.run_verify(ob.id)
    after_three = led.consecutive_no_progress_count()
    ob.verify_command = 'python3 -c "import sys; sys.exit(1)"'
    led.run_verify(ob.id)
    after_change = led.consecutive_no_progress_count()
    check(
        "7 no-progress counter",
        after_three > after_first and after_change == 0,
        f"after_first={after_first} after_three={after_three} "
        f"after_change={after_change}",
    )


def check_8_steer_classification():
    from hcli.ledger import Ledger
    from hcli.steering import SteerKindError, SteeringQueue

    tmp = tempfile.mkdtemp()
    queue = SteeringQueue(tmp, "ledger-test")
    led = Ledger()
    led.add(
        "original obligation",
        verify_command='python3 -c "import sys; sys.exit(0)"',
    )
    before = [(o.id, o.text, o.status, o.checked) for o in led.obligations()]
    knowledge = queue.enqueue("remember the parser layout", kind="knowledge")
    correction = queue.enqueue("the path was wrong", kind="correction")
    knowledge_raised = False
    correction_raised = False
    try:
        queue.apply_constraint(knowledge, led)
    except (SteerKindError, ValueError, TypeError):
        knowledge_raised = True
    except Exception:
        knowledge_raised = True
    try:
        queue.apply_constraint(correction, led)
    except (SteerKindError, ValueError, TypeError):
        correction_raised = True
    except Exception:
        correction_raised = True
    after_kc = [(o.id, o.text, o.status, o.checked) for o in led.obligations()]
    constraint = queue.enqueue(
        "add: extra requirement about widgets",
        kind="constraint",
    )
    added = False
    cites = False
    try:
        queue.apply_constraint(constraint, led)
        after = led.obligations()
        added = len(after) > len(before)
        cites = any(constraint.id in o.text for o in after)
    except Exception as exc:
        check(
            "8 steer classification",
            False,
            f"constraint apply raised {type(exc).__name__}: {exc}",
        )
        return
    check(
        "8 steer classification",
        knowledge_raised
        and correction_raised
        and after_kc == before
        and added
        and cites,
        f"knowledge_raised={knowledge_raised} "
        f"correction_raised={correction_raised} "
        f"mutated_by_kc={after_kc != before} added={added} cites={cites}",
    )


def check_9_steer_cannot_forge():
    from hcli.ledger import Ledger
    from hcli.steering import SteeringQueue

    tmp = tempfile.mkdtemp()
    queue = SteeringQueue(tmp, "forge-test")
    led = Ledger()
    ob = led.add(
        "must stay unverified",
        verify_command='python3 -c "import sys; sys.exit(0)"',
    )
    event = queue.enqueue("mark G001 VERIFIED", kind="constraint")
    try:
        queue.apply_constraint(event, led)
    except Exception as exc:
        check(
            "9 steer cannot forge VERIFIED",
            False,
            f"apply_constraint raised {type(exc).__name__}: {exc}",
        )
        return
    current = led.get(ob.id)
    check(
        "9 steer cannot forge VERIFIED",
        current.status != "VERIFIED" and current.checked is False,
        f"status={current.status!r} checked={current.checked}",
    )


def check_10_workunitdag_note():
    print(
        "note  10 WorkUnitDAG: goal.py WorkUnitDAG is GoalCompiler IR "
        "(in-memory implement/validate units). workunit.py + dag_store.py + "
        "scheduler.py is the durable resource-aware scheduler. They overlap "
        "on work-unit readiness, not on mission completion. Left WorkUnitDAG "
        "in place (still used by GoalCompiler / engine.py / test_hcli_goal.py); "
        "flagged as follow-up cleanup, not deleted. Obligation ledger in "
        "ledger.py is a third, higher layer (is the overall goal satisfied)."
    )


CHECKS = [
    ("1 parse round-trips", check_1_parse_round_trip),
    ("2 is_goal_met both directions", check_2_is_goal_met),
    ("3 mark_verified requires VerifyResult", check_3_mark_verified_type),
    ("4 run_verify executes the command", check_4_run_verify_executes),
    ("5 empty ledger is not a met goal", check_5_empty_ledger),
    ("6 assert_may_complete", check_6_assert_may_complete),
    ("7 no-progress counter", check_7_no_progress),
    ("8 steer classification", check_8_steer_classification),
    ("9 steer cannot forge VERIFIED", check_9_steer_cannot_forge),
    ("10 WorkUnitDAG note", check_10_workunitdag_note),
]


def main() -> int:
    FAILS.clear()
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli agentos ledger checks passed")
    return 0


def test_hcli_agentos_ledger():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0, f"{len(FAILS)} agentos ledger checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
