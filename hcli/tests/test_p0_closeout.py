"""P0 closeout gates: unfailable verifiers, Grok terminal state, receipt evidence, DAG.

Each test in this module was first run against the unmodified tree. Failures
from that run are recorded in the closeout report; a ModuleNotFoundError is
called out as missing-module, not as behavioural evidence.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]

from hcli.controller import Controller
from hcli.engine import Engine
from hcli.events import EventBus
from hcli.goal import GoalCompiler
from hcli.ledger import Ledger
from hcli.workspace import Workspace
from hcli.workunit import WorkUnit

UNFAILABLE = (
    re.compile(r"SystemExit\(\s*0\s*\)"),
    re.compile(r"^\s*true\s*$"),
    re.compile(r"^\s*exit\s+0\s*$"),
    re.compile(r"^\s*:\s*$"),
)

_DIRECTIVE = Path.home() / ".claude/ultragoal/hawking-headless-v3/ultragoal-directive.md"

_SYNTHETIC_LARGE = """# Large structured goal

## Phase A — Context authority
Must unify context budget in context_budget.py. Tests must fail when the
authority is split.

## Phase B — Command plane
Must wire /ultragoal through Controller.start_ultragoal. A missing method
is a failed check, not a pass.

## Phase C — Report compaction
Validate compact reports in report_compiler.py. Empty evidence is a failure.

## Phase D — Grok backend
Grok WorkUnits whose terminal state is failed must never be accepted.

## Phase E — Verifier authority
Never emit python3 -c 'raise SystemExit(0)' as a verify command.

## Final acceptance
cargo test and the named pytest files must collect at least one case.
"""


def _directive_text() -> str:
    if _DIRECTIVE.is_file():
        return _DIRECTIVE.read_text(encoding="utf-8")
    return _SYNTHETIC_LARGE


def _controller(workspace: str) -> Controller:
    return Controller(
        workspace=Workspace(workspace),
        runtime_count=1,
        model=None,
        bus=EventBus(),
    )


def _verify_lines(goal_md: str) -> list:
    return [
        line.strip()[len("verify:") :].strip()
        for line in goal_md.splitlines()
        if line.strip().startswith("verify:")
    ]


def _wu(uid: str, **kwargs) -> WorkUnit:
    wu = WorkUnit(id=uid, role="work", description=uid)
    for key, value in kwargs.items():
        setattr(wu, key, value)
    return wu


class TestP0_1UnfailableVerifiers(unittest.TestCase):
    def test_compiled_obligations_have_no_unfailable_verify_commands(self):
        text = _directive_text()
        with tempfile.TemporaryDirectory(prefix="p0-1-") as tmp:
            ctrl = _controller(tmp)
            try:
                result = ctrl.start_ultragoal(text)
                goal_md = (Path(tmp) / ".hcli" / "GOAL.md").read_text(
                    encoding="utf-8"
                )
            finally:
                ctrl.shutdown()

        verifies = _verify_lines(goal_md)
        bad = [v for v in verifies if any(rx.search(v) for rx in UNFAILABLE)]
        self.assertTrue(verifies, "compiled GOAL.md must carry verify: fields")
        self.assertEqual(bad, [], f"unfailable verify commands: {bad}")
        self.assertGreater(len(result.get("obligation_ids") or []), 0)

        ledger = Ledger.from_markdown(goal_md)
        for ob in ledger.obligations():
            self.assertNotEqual(
                (ob.acceptance or "").strip(),
                (ob.text or "").strip(),
                f"{ob.id} acceptance restates the obligation",
            )
            self.assertNotEqual(
                (ob.acceptance or "").strip(),
                "Requested behavior is implemented and validated.",
            )
            cmd = (ob.verify_command or "").strip()
            if cmd:
                self.assertFalse(
                    any(rx.search(cmd) for rx in UNFAILABLE),
                    f"{ob.id} verify is unfailable: {cmd!r}",
                )


class TestP0_3GrokTerminalState(unittest.TestCase):
    def test_failed_grok_state_not_accepted_despite_passing_verifier(self):
        from hcli.executors import WorkUnitExecutor

        with tempfile.TemporaryDirectory(prefix="p0-3-fail-") as tmp:
            ran = Path(tmp) / "verifier-ran.txt"
            wu = _wu(
                "g-fail",
                preferred_backend="grok",
                verifier=f"python3 -c \"open(r'{ran}', 'w').write('ran'); raise SystemExit(0)\"",
            )

            class Fake:
                def consult(self, prompt, **kwargs):
                    return SimpleNamespace(task_id="g-fail-task")

                def wait(self, task_id, timeout=3600.0, poll_interval=0.5):
                    return {"state": "failed", "exit_code": 1, "task_id": task_id}

            raw = WorkUnitExecutor(tmp, grok_bridge=Fake()).execute(wu, {})
            validation = raw.get("validation") or {}
            accepted = bool(raw.get("accepted") or validation.get("ok"))
            self.assertFalse(accepted, raw)
            self.assertFalse(validation.get("ok"), raw)
            self.assertFalse(ran.is_file(), "verifier must not run after failed state")

    def test_done_grok_state_accepted_with_passing_verifier(self):
        from hcli.executors import WorkUnitExecutor

        with tempfile.TemporaryDirectory(prefix="p0-3-done-") as tmp:
            ran = Path(tmp) / "verifier-ran.txt"
            wu = _wu(
                "g-done",
                preferred_backend="grok",
                verifier=f"python3 -c \"open(r'{ran}', 'w').write('ran'); raise SystemExit(0)\"",
            )

            class Fake:
                def consult(self, prompt, **kwargs):
                    return SimpleNamespace(task_id="g-done-task")

                def wait(self, task_id, timeout=3600.0, poll_interval=0.5):
                    return {"state": "done", "exit_code": 0, "task_id": task_id}

            raw = WorkUnitExecutor(tmp, grok_bridge=Fake()).execute(wu, {})
            validation = raw.get("validation") or {}
            self.assertTrue(validation.get("ok"), raw)
            self.assertTrue(ran.is_file(), "verifier must run after done state")
            self.assertNotEqual(raw.get("accepted"), False)


class TestP0_11ErrorPathEvidence(unittest.TestCase):
    def test_engine_source_has_no_empty_evidence_literal(self):
        import hcli.engine as engine_mod

        src = Path(engine_mod.__file__).read_text(encoding="utf-8")
        hits = [
            i
            for i, line in enumerate(src.splitlines(), 1)
            if re.search(r"^\s*evidence\s*=\s*\[\s*\]\s*,?\s*$", line)
        ]
        self.assertEqual(hits, [], f"empty evidence literals at {hits}")

    def test_forced_failure_receipt_records_gathered_evidence(self):
        with tempfile.TemporaryDirectory(prefix="p0-11-") as tmp:
            root = Path(tmp)
            (root / "alpha.py").write_text("a = 1\n", encoding="utf-8")
            (root / "beta.py").write_text("b = 2\n", encoding="utf-8")
            engine = Engine(
                workspace=Workspace(str(root)),
                event_bus=EventBus(),
                runtime_count=1,
                model_name="/m.gguf",
            )

            def boom(prompt, evidence, compiled):
                raise RuntimeError("forced failure after gather")

            engine._call_model = boom
            result = engine.execute("read alpha.py and beta.py")
            self.assertEqual(result["status"], "failed")
            receipt_path = result.get("receipt")
            self.assertTrue(receipt_path)
            receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(receipt.get("evidence_files") or []),
                ["alpha.py", "beta.py"],
            )


class TestP0_12WorkUnitDagFromObligations(unittest.TestCase):
    def test_many_obligations_produce_many_traceable_workunits(self):
        text = _directive_text()
        with tempfile.TemporaryDirectory(prefix="p0-12-") as tmp:
            ctrl = _controller(tmp)
            try:
                result = ctrl.start_ultragoal(text)
                dag = json.loads(
                    (Path(tmp) / ".hcli" / "dag.json").read_text(encoding="utf-8")
                )
                goal_md = (Path(tmp) / ".hcli" / "GOAL.md").read_text(
                    encoding="utf-8"
                )
            finally:
                ctrl.shutdown()

        obligation_ids = list(result.get("obligation_ids") or [])
        units = dag.get("units") or {}
        self.assertGreater(len(obligation_ids), 2, obligation_ids)
        self.assertGreater(len(units), 2, list(units))
        self.assertGreater(len(result.get("workunit_ids") or []), 2)

        named = 0
        for uid, payload in units.items():
            blob = uid + " " + json.dumps(payload, default=str)
            covers = [oid for oid in obligation_ids if oid in blob]
            self.assertTrue(
                covers,
                f"workunit {uid} does not name any obligation id from {obligation_ids[:8]}",
            )
            named += 1
        self.assertEqual(named, len(units))

        compiled = GoalCompiler().compile(text)
        dag_ir = compiled["workunits"]
        self.assertGreater(len(dag_ir.units), 2)


if __name__ == "__main__":
    unittest.main()
