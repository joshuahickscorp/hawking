from __future__ import annotations

import sys
import pathlib
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.ledger import (
    DEFAULT_VERIFY_TIMEOUT,
    ENFORCEMENT_FAILURE,
    GOAL_MET,
    GOAL_NOT_MET,
    NO_PROGRESS_THRESHOLD,
    SAFETY_DISARM,
    TERMINAL_BLOCKER,
    WATCHDOG_L4,
    WATCHDOG_MESSAGES,
    EnforcementFailureError,
    GoalNotMetError,
    Ledger,
    Obligation,
    SafetyDisarmError,
    TerminalBlockerError,
    VerifyResult,
)


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

DISAGREE_CHECKBOX_MD = """- [x] G001 — checkbox says done | status: PENDING | risk: high | tier: V2
      acceptance: none
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: (none yet)
"""

DISAGREE_STATUS_MD = """- [ ] G001 — status says done | status: VERIFIED | risk: high | tier: V2
      acceptance: none
      verify: python3 -c "import sys; sys.exit(0)"
      evidence: (none yet)
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


class TestLedgerParse(unittest.TestCase):
    def test_parse_round_trip_objects_equal(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        out = led.to_markdown()
        path2 = _write(out)
        self.addCleanup(lambda: path2.unlink(missing_ok=True))
        led2 = Ledger.parse(path2)
        self.assertEqual(led, led2)
        self.assertEqual(len(led), 3)

    def test_byte_for_byte_when_unchanged(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.to_markdown(), MIXED_MD)

    def test_checkbox_status_disagreement_is_not_verified(self):
        p1 = _write(DISAGREE_CHECKBOX_MD)
        p2 = _write(DISAGREE_STATUS_MD)
        self.addCleanup(lambda: p1.unlink(missing_ok=True))
        self.addCleanup(lambda: p2.unlink(missing_ok=True))
        a = Ledger.parse(p1)
        b = Ledger.parse(p2)
        self.assertFalse(a.is_goal_met())
        self.assertFalse(b.is_goal_met())
        self.assertEqual(a.status, "GOAL_NOT_MET")
        self.assertEqual(b.status, "GOAL_NOT_MET")
        self.assertEqual(a.get("G001").status, "PENDING")
        self.assertTrue(a.get("G001").checked)
        # status: VERIFIED on disk with no run_verify receipt is demoted to
        # STALE -- unsatisfied, but distinguishable from never-attempted.
        self.assertEqual(b.get("G001").status, "STALE")
        self.assertFalse(b.get("G001").checked)


class TestLedgerGovernor(unittest.TestCase):
    def test_empty_ledger_is_missing_not_met(self):
        led = Ledger()
        self.assertFalse(led.is_goal_met())
        self.assertEqual(led.status, "EMPTY_LEDGER")
        self.assertEqual(led.outcome(), ENFORCEMENT_FAILURE)
        with self.assertRaises(EnforcementFailureError) as ctx:
            led.assert_may_complete()
        self.assertEqual(ctx.exception.ledger_status, ENFORCEMENT_FAILURE)

    def test_is_goal_met_false_until_all_verified(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertFalse(led.is_goal_met())
        self.assertEqual(led.status, "GOAL_NOT_MET")

    def test_hand_marked_verified_is_not_goal_met(self):
        """Correction: on-disk VERIFIED with no run_verify receipt is not evidence.

        The previous assertion treated a hand-edited GOAL.md as a met goal.
        That was exploit I. Hand-marked VERIFIED now loads as PENDING.
        """
        md = MIXED_MD.replace("[ ] G001", "[x] G001").replace(
            "status: PENDING", "status: VERIFIED", 1
        ).replace("[ ] G002", "[x] G002").replace(
            "status: ACTIVE", "status: VERIFIED", 1
        )
        path = _write(md)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertFalse(led.is_goal_met())
        self.assertEqual(led.status, "GOAL_NOT_MET")
        for ob in led.obligations():
            self.assertNotEqual(ob.status, "VERIFIED")
            self.assertFalse(ob.checked)

    def test_assert_may_complete_lists_unverified_ids(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        with self.assertRaises(GoalNotMetError) as ctx:
            led.assert_may_complete()
        ids = [item[0] for item in ctx.exception.unverified]
        self.assertIn("G001", ids)
        self.assertIn("G002", ids)
        # G003 is hand-marked VERIFIED in MIXED_MD with no receipt.
        # That is not evidence; including it here is a correction of the
        # previous "already done" exemption, not a weakening.
        self.assertIn("G003", ids)

    def test_mark_verified_rejects_bare_true_and_string(self):
        led = Ledger()
        ob = led.add("x", verify_command='python3 -c "import sys; sys.exit(0)"')
        with self.assertRaises((TypeError, ValueError)):
            led.mark_verified(ob.id, True)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            led.mark_verified(ob.id, "yes")  # type: ignore[arg-type]
        self.assertNotEqual(ob.status, "VERIFIED")

    def test_mark_verified_rejects_fabricated_verify_result(self):
        led = Ledger()
        ob = led.add("x", verify_command='python3 -c "import sys; sys.exit(0)"')
        fake = VerifyResult(True, "nope", 0, obligation_id=ob.id)
        with self.assertRaises((TypeError, ValueError)):
            led.mark_verified(ob.id, fake)
        self.assertNotEqual(ob.status, "VERIFIED")

    def test_mark_status_cannot_set_verified(self):
        led = Ledger()
        ob = led.add("x", verify_command='python3 -c "import sys; sys.exit(0)"')
        with self.assertRaises(ValueError):
            led.mark_status(ob.id, "VERIFIED")
        self.assertEqual(ob.status, "PENDING")


class TestLedgerRunVerify(unittest.TestCase):
    def test_run_verify_writes_marker_and_records_failure(self):
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
        ok_result = led.run_verify(ok.id)
        fail_result = led.run_verify(fail.id)
        self.assertTrue(marker.is_file())
        self.assertTrue(ok_result.passed)
        self.assertEqual(ok_result.exit_code, 0)
        self.assertFalse(fail_result.passed)
        self.assertEqual(fail_result.exit_code, 7)
        self.assertTrue(fail.evidence)

    def test_empty_verify_command_does_not_pass(self):
        led = Ledger()
        ob = led.add("no command", verify_command="")
        result = led.run_verify(ob.id)
        self.assertFalse(result.passed)
        self.assertTrue(result.output)

    def test_timeout_is_not_passed(self):
        led = Ledger()
        ob = led.add(
            "sleep",
            verify_command='python3 -c "import time; time.sleep(10)"',
        )
        result = led.run_verify(ob.id, timeout=0.2)
        self.assertFalse(result.passed)
        self.assertIn("TIMEOUT", result.output.upper())

    def test_crash_is_not_passed(self):
        led = Ledger()
        ob = led.add(
            "abort",
            verify_command='python3 -c "import os; os.abort()"',
        )
        result = led.run_verify(ob.id)
        self.assertFalse(result.passed)

    def test_mark_verified_requires_passing_fresh_result(self):
        tmp = Path(tempfile.mkdtemp())
        marker = tmp / "ok.txt"
        led = Ledger()
        ob = led.add(
            "ok",
            verify_command=(
                "python3 -c "
                f"\"open(r'{marker}', 'w').write('ok')\""
            ),
        )
        result = led.run_verify(ob.id)
        self.assertTrue(result.passed)
        led.mark_verified(ob.id, result)
        self.assertEqual(ob.status, "VERIFIED")
        self.assertTrue(ob.checked)
        self.assertTrue(led.is_goal_met())
        self.assertEqual(led.status, "GOAL_MET")

    def test_mark_verified_rejects_failing_result(self):
        led = Ledger()
        ob = led.add(
            "fail",
            verify_command='python3 -c "import sys; sys.exit(1)"',
        )
        result = led.run_verify(ob.id)
        self.assertFalse(result.passed)
        with self.assertRaises(ValueError):
            led.mark_verified(ob.id, result)
        self.assertNotEqual(ob.status, "VERIFIED")

    def test_no_progress_increments_then_resets(self):
        self.assertGreaterEqual(NO_PROGRESS_THRESHOLD, 1)
        self.assertGreater(DEFAULT_VERIFY_TIMEOUT, 0)
        tmp = Path(tempfile.mkdtemp())
        marker = tmp / "stable.txt"
        led = Ledger()
        ob = led.add(
            "stable",
            verify_command=(
                "python3 -c "
                f"\"open(r'{marker}', 'w').write('ok')\""
            ),
        )
        led.run_verify(ob.id)
        after_first = led.consecutive_no_progress_count()
        led.run_verify(ob.id)
        led.run_verify(ob.id)
        after_three = led.consecutive_no_progress_count()
        self.assertGreater(after_three, after_first)
        ob.verify_command = 'python3 -c "import sys; sys.exit(1)"'
        led.run_verify(ob.id)
        self.assertEqual(led.consecutive_no_progress_count(), 0)

    def test_vacuous_verify_command_does_not_pass(self):
        led = Ledger()
        ob = led.add(
            "tautology",
            verify_command="python3 -c 'raise SystemExit(0)'",
        )
        result = led.run_verify(ob.id)
        self.assertFalse(result.passed)
        self.assertIn("VACUOUS", result.output)

    def test_forged_verified_without_receipt_is_not_satisfied(self):
        md = """- [x] G001 — forged | status: VERIFIED | risk: high | tier: V2
      acceptance: none
      verify: python3 -c "print(1)"
      evidence: hand-marked
"""
        path = _write(md)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        # STALE, not PENDING: a forged VERIFIED is unsatisfied either way, but
        # STALE keeps "was marked verified, evidence not available here"
        # distinguishable from "never attempted".
        self.assertEqual(led.get("G001").status, "STALE")
        self.assertFalse(led.get("G001").checked)
        self.assertFalse(led.is_goal_met())

    def test_verified_survives_reparse_only_with_receipt(self):
        tmp = Path(tempfile.mkdtemp())
        marker = tmp / "ok.txt"
        goal = tmp / "GOAL.md"
        led = Ledger()
        led._path = goal
        ob = led.add(
            "real",
            verify_command=(
                "python3 -c "
                f"\"open(r'{marker}', 'w').write('ok')\""
            ),
        )
        result = led.run_verify(ob.id)
        self.assertTrue(result.passed)
        led.mark_verified(ob.id, result)
        goal.write_text(led.to_markdown(), encoding="utf-8")
        again = Ledger.parse(goal)
        self.assertEqual(again.get(ob.id).status, "VERIFIED")
        self.assertTrue(again.is_goal_met())

        other = Path(tempfile.mkdtemp())
        orphan = other / "GOAL.md"
        orphan.write_text(led.to_markdown(), encoding="utf-8")
        forged = Ledger.parse(orphan)
        self.assertNotEqual(forged.get(ob.id).status, "VERIFIED")
        self.assertFalse(forged.is_goal_met())


class TestLedgerAdd(unittest.TestCase):
    def test_add_generates_monotonic_ids(self):
        led = Ledger()
        a = led.add("one")
        b = led.add("two")
        self.assertEqual(a.id, "G001")
        self.assertEqual(b.id, "G002")
        self.assertEqual(a.status, "PENDING")
        self.assertFalse(a.checked)

    def test_obligation_dataclass_fields(self):
        ob = Obligation(id="G001", text="t")
        self.assertEqual(ob.status, "PENDING")
        self.assertEqual(ob.evidence, "(none yet)")


ALL_BLOCKED_MD = """# GOAL
terminal_blocker: missing GPU firmware

- [ ] G001 — first obligation | status: BLOCKED | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: stuck
- [ ] G002 — second obligation | status: BLOCKED | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: stuck
"""

PARTIAL_BLOCKED_MD = """# GOAL
terminal_blocker: missing GPU firmware

- [ ] G001 — first obligation | status: BLOCKED | risk: high | tier: V2
      acceptance: a exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: stuck
- [ ] G002 — second obligation | status: PENDING | risk: medium | tier: V1
      acceptance: b exists
      verify: python3 -c "import sys; sys.exit(1)"
      evidence: (none yet)
"""


def _verified_ledger_with_receipts(tmpdir):
    """A ledger whose VERIFIED marks are backed by real verify receipts.

    Hand-marking `[x]` / `status: VERIFIED` in the markdown is exactly the
    forgery Ledger.parse now refuses, so a test that wants GOAL_MET has to earn
    it the way production does: run_verify + mark_verified, which writes the
    sidecar receipt that parse checks.
    """
    root = pathlib.Path(tmpdir)
    marker = root / "marker.txt"
    marker.write_text("ok\n", encoding="utf-8")
    path = root / "GOAL.md"
    path.write_text(MIXED_MD, encoding="utf-8")
    led = Ledger.parse(path)
    # A real command that reads real state and can genuinely fail. An
    # unfailable one (`sys.exit(0)`) is rejected as VACUOUS_COMMAND, which is
    # the whole point of the verifier.
    cmd = (
        "python3 -c \"import pathlib,sys; "
        "sys.exit(0 if pathlib.Path(r'%s').read_text().strip()=='ok' else 1)\""
    ) % marker
    for ob in led.obligations():
        ob.verify_command = cmd
        result = led.run_verify(ob.id)
        assert result.passed, f"helper verify did not pass: {result}"
        # run_verify records evidence; mark_verified is what promotes the
        # status and writes the sidecar receipt that parse() later checks.
        led.mark_verified(ob.id, result)
    led.save(path)
    return path


class TestLedgerOutcomes(unittest.TestCase):
    def test_outcome_goal_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _verified_ledger_with_receipts(tmp)
            led = Ledger.parse(path)
            self.assertEqual(led.outcome(), GOAL_MET)
            led.assert_may_complete()

    def test_hand_marked_verified_is_stale_not_met(self):
        """The forgery this replaced: checkboxes are not evidence.

        Before the freshness rule, editing `[ ]` to `[x]` and PENDING to
        VERIFIED in the markdown was enough to make the ledger report GOAL_MET.
        """
        md = MIXED_MD.replace("[ ] G001", "[x] G001").replace(
            "status: PENDING", "status: VERIFIED", 1
        ).replace("[ ] G002", "[x] G002").replace(
            "status: ACTIVE", "status: VERIFIED", 1
        )
        path = _write(md)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.outcome(), GOAL_NOT_MET)
        self.assertTrue(
            all(ob.status == "STALE" for ob in led.obligations()),
            [ob.status for ob in led.obligations()],
        )
        with self.assertRaises(Exception):
            led.assert_may_complete()

    def test_outcome_goal_not_met(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.outcome(), GOAL_NOT_MET)
        with self.assertRaises(GoalNotMetError):
            led.assert_may_complete()

    def test_outcome_terminal_blocker_is_not_goal_met(self):
        path = _write(ALL_BLOCKED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.terminal_blocker, "missing GPU firmware")
        self.assertEqual(led.outcome(), TERMINAL_BLOCKER)
        self.assertNotEqual(led.outcome(), GOAL_MET)
        self.assertFalse(led.is_goal_met())
        with self.assertRaises(TerminalBlockerError) as ctx:
            led.assert_may_complete()
        self.assertIn("NOT GOAL_MET", str(ctx.exception))
        self.assertEqual(ctx.exception.ledger_status, TERMINAL_BLOCKER)
        self.assertEqual(ctx.exception.blocker, "missing GPU firmware")

    def test_terminal_blocker_requires_every_unresolved_blocked(self):
        path = _write(PARTIAL_BLOCKED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.terminal_blocker, "missing GPU firmware")
        self.assertEqual(led.outcome(), GOAL_NOT_MET)
        with self.assertRaises(GoalNotMetError):
            led.assert_may_complete()

    def test_safety_disarm_is_caller_requested(self):
        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        self.assertEqual(led.outcome(), GOAL_NOT_MET)
        self.assertEqual(led.outcome(budget_exceeded=True), SAFETY_DISARM)
        with self.assertRaises(SafetyDisarmError) as ctx:
            led.assert_may_complete(budget_exceeded=True)
        self.assertEqual(ctx.exception.ledger_status, SAFETY_DISARM)
        self.assertIn("NOT completion", str(ctx.exception))

    def test_safety_disarm_does_not_override_goal_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _verified_ledger_with_receipts(tmp)
            led = Ledger.parse(path)
            self.assertEqual(led.outcome(budget_exceeded=True), GOAL_MET)


class TestLedgerWatchdog(unittest.TestCase):
    def test_dual_hash_ladder_and_reset(self):
        led = Ledger()
        ob = led.add(
            "stable",
            verify_command='python3 -c "import sys; sys.exit(0)"',
        )
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 0)
        self.assertEqual(led.watchdog_message(), "")
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 1)
        self.assertEqual(led.watchdog_message(), WATCHDOG_MESSAGES[1])
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 2)
        self.assertIn("MANDATORY REPLAN", led.watchdog_message())
        self.assertEqual(led.watchdog_message(), WATCHDOG_MESSAGES[2])
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 3)
        self.assertIn("CHANGE STRATEGY OR ESCALATE", led.watchdog_message())
        self.assertEqual(led.watchdog_message(), WATCHDOG_MESSAGES[3])
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 4)
        self.assertEqual(led.watchdog_message(), WATCHDOG_L4)
        self.assertIn("ROOT CAUSE OR BLOCKER PROOF", led.watchdog_message())
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 4)
        ob.verify_command = 'python3 -c "import sys; sys.exit(1)"'
        led.run_verify(ob.id)
        self.assertEqual(led.watchdog_tier(), 0)
        self.assertEqual(led.consecutive_no_progress_count(), 0)

    def test_stall_requires_both_hashes(self):
        led = Ledger()
        ob = led.add(
            "stable",
            verify_command='python3 -c "import sys; sys.exit(0)"',
        )
        led.observe_progress()
        led.observe_progress()
        self.assertEqual(led.watchdog_tier(), 1)
        led.replace_text(ob.id, "rewritten without touching evidence")
        led.observe_progress()
        self.assertEqual(led.watchdog_tier(), 0)
        led.observe_progress()
        self.assertEqual(led.watchdog_tier(), 1)
        ob.evidence = "fresh evidence line"
        led._dirty = True
        led.observe_progress()
        self.assertEqual(led.watchdog_tier(), 0)


class TestLedgerApplyConstraint(unittest.TestCase):
    def test_constraint_persists_to_same_path(self):
        from hcli.steering import SteeringQueue

        path = _write(MIXED_MD)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        led = Ledger.parse(path)
        tmp = path.parent
        queue = SteeringQueue(str(tmp), "ledger-persist")
        event = queue.enqueue(
            "add: extra requirement about widgets",
            kind="constraint",
        )
        led.apply_constraint(event, queue)
        led2 = Ledger.parse(path)
        self.assertGreater(len(led2), 3)
        self.assertTrue(any("widgets" in ob.text for ob in led2.obligations()))
        self.assertTrue(any(event.id in ob.text for ob in led2.obligations()))
        self.assertEqual(led2.get("G001").text, led.get("G001").text)
        self.assertEqual(led2.get("G001").status, "PENDING")


if __name__ == "__main__":
    unittest.main()
