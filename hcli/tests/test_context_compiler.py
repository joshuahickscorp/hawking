"""Context compiler pressure tests.

These were first driven against the unmodified compiler. Failures from that
run (see the module docstring of each class) are the evidence the assertions
are real, not re-derived from reading source.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.context_budget import (
    PacketBudgetError,
    fit_or_refuse,
    resolve,
)
from hcli.engine import EngineError
from hcli.executors import consult_worker, execute_workunit
from hcli.goal import (
    EvidenceIdentity,
    GoalCompiler,
    StaleEvidenceError,
    WorkerPacket,
    assert_packet_evidence_fresh,
    compile_worker_context,
    contains_goal_dump,
    identity_for_path,
    refuse_goal_dump,
)
from hcli.workunit import WorkUnit, emit_repair


ROOT = "THIS IS THE GOVERNING ULTRAGOAL. " + ("keep-out-" * 30)
assert len(ROOT) >= 80


def _wu(uid="u1", **kwargs):
    return WorkUnit(
        id=uid,
        role=kwargs.pop("role", "work"),
        description=kwargs.pop("description", "do the thing"),
        **kwargs,
    )


def _compiled(goal=ROOT, **extra):
    payload = {
        "goal": goal,
        "invariants": ["must not leak the parent"],
        "acceptance_criteria": ["tests pass"],
        "referenced_files": extra.pop("referenced_files", []),
    }
    payload.update(extra)
    return payload


def _packet(wu, compiled=None, **kwargs):
    compiled = compiled if compiled is not None else _compiled()
    units = kwargs.pop("units", {wu.id: wu})
    return compile_worker_context(
        wu,
        compiled,
        phase=kwargs.pop("phase", "running"),
        units=units,
        steering=kwargs.pop("steering", []),
        **kwargs,
    )


def _rewrite(path: Path, text: str) -> None:
    old_mtime = path.stat().st_mtime_ns if path.exists() else None
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    if old_mtime is not None and st.st_mtime_ns == old_mtime:
        os.utime(path, ns=(st.st_atime_ns, old_mtime + 1))


class TestGoalDumpRefusalFires(unittest.TestCase):
    """Watched against the pre-fix compiler:

    - WorkerPacket(prompt='GOAL: dumped') raised ValueError (already).
    - execute_workunit(..., {'prompt': 'GOAL: dumped'}) raised EngineError
      (already).
    - WorkerPacket(prompt='GOAL : '+root) was ACCEPTED — the substring
      check looked for 'GOAL:' and missed the spaced header.
    - wrapping a real compiled packet with 'GOAL:\\n' refused (already).
    """

    def test_workerpacket_refuses_goal_header(self):
        with self.assertRaises(ValueError) as ctx:
            WorkerPacket(
                unit_id="x",
                phase="running",
                workunit="x implementation do it",
                invariants=(),
                acceptance=(),
                neighborhood=(),
                evidence_paths=(),
                steering=(),
                prompt="GOAL: dumped",
            )
        self.assertIn("GOAL:", str(ctx.exception))

    def test_workerpacket_refuses_spaced_header_that_used_to_slip(self):
        # Constructed slip-through: "GOAL :" is not the substring "GOAL:".
        spaced = "GOAL : " + ROOT
        self.assertNotIn("GOAL:", spaced)
        self.assertTrue(contains_goal_dump(spaced))
        with self.assertRaises(ValueError) as ctx:
            WorkerPacket(
                unit_id="x",
                phase="running",
                workunit="x",
                invariants=(),
                acceptance=(),
                neighborhood=(),
                evidence_paths=(),
                steering=(),
                prompt=spaced,
            )
        self.assertIn("GOAL:", str(ctx.exception))

    def test_engine_wrap_of_compiled_packet_fires_refusal(self):
        wu = _wu(description="edit foo.py")
        packet = _packet(wu, _compiled(referenced_files=["foo.py"]))
        self.assertNotIn("GOAL:", packet.prompt)
        wrapped = f"GOAL:\n{packet.prompt}"
        with self.assertRaises(ValueError) as ctx:
            WorkerPacket(
                unit_id=packet.unit_id,
                phase=packet.phase,
                workunit=packet.workunit,
                invariants=packet.invariants,
                acceptance=packet.acceptance,
                neighborhood=packet.neighborhood,
                evidence_paths=packet.evidence_paths,
                steering=packet.steering,
                prompt=wrapped,
            )
        self.assertIn("GOAL:", str(ctx.exception))

    def test_execute_workunit_refusal_fires_on_dumped_prompt(self):
        class Boom:
            def __init__(self):
                self._gather_evidence = lambda _prompt: []
                self.goal_compiler = type("C", (), {})()
                self.goal_compiler.compile = lambda _text: {}

            def execute(self, prompt):
                raise AssertionError("must not execute: " + prompt[:80])

        with self.assertRaises(EngineError) as ctx:
            execute_workunit(
                Boom(),
                _wu(),
                {"prompt": "GOAL: dumped"},
            )
        self.assertIn("GOAL:", str(ctx.exception))


class TestRootGoalCannotReachWorker(unittest.TestCase):
    """Watched against the pre-fix compiler:

    - _sanitize_goal_header('GOAL: '+root) rewrote the header to GOAL_ and
      LEFT THE BODY. compile_worker_context put that body on OBJECTIVE.
    - steering=[root] put the full parent into the packet.
    - A repair unit's failure_context JSON carried GOAL_\\n + a prefix of
      the parent; the 400-char silent slice is why the FULL root was
      absent, not because the dump was refused.
    """

    def test_rewrite_only_sanitizer_would_have_leaked_the_body(self):
        leaked = ("GOAL: " + ROOT).replace("GOAL:", "GOAL_")
        self.assertNotIn("GOAL:", leaked)
        self.assertIn(ROOT, leaked)
        wu = _wu(description="GOAL: " + ROOT)
        packet = _packet(wu)
        self.assertNotIn("GOAL:", packet.prompt)
        self.assertFalse(contains_goal_dump(packet.prompt))
        self.assertNotIn(ROOT, packet.prompt)
        self.assertNotIn("THIS IS THE GOVERNING ULTRAGOAL.", packet.prompt)

    def test_steering_append_of_root_is_excised(self):
        wu = _wu(description="edit foo.py")
        packet = _packet(wu, steering=["[constraint] " + ROOT])
        self.assertNotIn(ROOT, packet.prompt)
        self.assertNotIn("GOAL:", packet.prompt)
        self.assertIn("[ROOT_GOAL_OMITTED]", packet.prompt)

    def test_repair_unit_failure_context_dump_is_redacted(self):
        orig = _wu(uid="orig", description="do thing")
        orig.status = "failed"
        orig.attempts = 1
        units = {"orig": orig}
        repair = emit_repair(
            units,
            orig,
            context={
                "prompt": "GOAL:\n" + ROOT + "\n\nDETERMINISTIC EVIDENCE:\n(none)"
            },
        )
        self.assertIsNotNone(repair)
        self.assertIn("GOAL:", str(repair.failure_context))
        self.assertIn(ROOT, str(repair.failure_context))
        packet = compile_worker_context(
            repair,
            _compiled(),
            phase="running",
            units=units,
            steering=[],
            failure_context=repair.failure_context,
        )
        self.assertNotIn("GOAL:", packet.prompt)
        self.assertFalse(contains_goal_dump(packet.prompt))
        self.assertNotIn(ROOT, packet.prompt)
        self.assertIn("FAILURE_CONTEXT", packet.prompt)

    def test_long_goal_forcing_compaction_still_excludes_root(self):
        long_goal = ROOT + (" more obligation text." * 400)
        compiled = GoalCompiler().compile(long_goal)
        compiled["invariants"] = [
            "must keep this invariant " + ("I" * 80) for _ in range(40)
        ]
        compiled["acceptance_criteria"] = [
            "tests pass " + ("A" * 40) for _ in range(20)
        ]
        child = WorkUnit(
            id="child",
            role="work",
            description="touch target.py",
            dependencies=["dep0", "dep1", "dep2"],
        )
        units = {
            "child": child,
            "dep0": WorkUnit(
                id="dep0", role="work", description="d0", status="completed"
            ),
            "dep1": WorkUnit(
                id="dep1", role="work", description="d1", status="failed"
            ),
            "dep2": WorkUnit(
                id="dep2", role="work", description="d2", status="pending"
            ),
        }
        packet = compile_worker_context(
            child,
            compiled,
            phase="running",
            units=units,
            steering=["[knowledge] " + long_goal],
            char_cap=400,
        )
        self.assertNotIn(long_goal, packet.prompt)
        self.assertNotIn("GOAL:", packet.prompt)
        # The unique body of the parent (not the short first sentence the
        # compiler may keep as the unit objective) must not appear.
        self.assertNotIn("keep-out-" * 8, packet.prompt)
        self.assertTrue(packet.truncated)
        self.assertIn("TRUNCATION:", packet.prompt)
        self.assertTrue(packet.omitted)


class TestVisibleTruncationOrRefuse(unittest.TestCase):
    """Watched against the pre-fix compiler:

    - char_cap=80 produced a 79-char prompt ending in '...'.
    - CONSTRAINT_MUST_SURVIVE was DROPPED.
    - No TRUNCATION: section, no truncated attribute.
    """

    def test_over_cap_compaction_is_visible_and_keeps_constraints(self):
        wu = WorkUnit(
            id="child",
            role="work",
            description="touch target.py",
            dependencies=["dep0", "dep1", "dep2"],
        )
        units = {
            "child": wu,
            "dep0": WorkUnit(
                id="dep0", role="work", description="d0", status="completed"
            ),
            "dep1": WorkUnit(
                id="dep1", role="work", description="d1", status="failed"
            ),
            "dep2": WorkUnit(
                id="dep2", role="work", description="d2", status="pending"
            ),
        }
        constraint = (
            "CONSTRAINT_MUST_SURVIVE: never drop this constraint " + ("X" * 200)
        )
        packet = compile_worker_context(
            wu,
            {
                "invariants": [
                    "must keep this invariant " + ("I" * 80) for _ in range(40)
                ],
                "acceptance_criteria": [
                    "tests pass " + ("A" * 40) for _ in range(20)
                ],
                "referenced_files": ["target.py"],
            },
            phase="running",
            units=units,
            steering=[constraint],
            char_cap=800,
        )
        self.assertLessEqual(len(packet.prompt), 800)
        self.assertIn("CONSTRAINT_MUST_SURVIVE", packet.prompt)
        self.assertNotIn("status=failed", packet.prompt)
        self.assertTrue(packet.truncated)
        self.assertIn("TRUNCATION:", packet.prompt)
        self.assertTrue(packet.omitted)
        self.assertIn("TRUNCATION:", packet.budget_signal)

    def test_last_resort_slice_is_refused_not_silent(self):
        wu = _wu(description="touch t.py")
        huge = "CONSTRAINT_MUST_SURVIVE " + ("X" * 500)
        with self.assertRaises(PacketBudgetError) as ctx:
            compile_worker_context(
                wu,
                _compiled(goal="short task must pass"),
                phase="running",
                units={wu.id: wu},
                steering=[huge],
                char_cap=80,
            )
        exc = ctx.exception
        self.assertIn("refused", str(exc).lower())
        self.assertIn("silent truncation is not allowed", str(exc).lower())
        self.assertGreater(exc.shortfall, 0)
        self.assertGreater(exc.prompt_len, exc.cap)

    def test_packet_plus_evidence_over_budget_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp) / "blob.txt"
            blob.write_text("E" * 120000, encoding="utf-8")
            wu = _wu(description="read blob.txt")
            compiled = _compiled(
                goal="short task about blob.txt must pass",
                referenced_files=["blob.txt"],
            )
            budget = resolve(
                ctx_size=32768, n_parallel=1, generation_reserve=4096
            )
            with self.assertRaises(PacketBudgetError) as ctx:
                compile_worker_context(
                    wu,
                    compiled,
                    phase="running",
                    units={wu.id: wu},
                    steering=[],
                    workspace=tmp,
                    budget=budget,
                )
            exc = ctx.exception
            self.assertIsNotNone(exc.result)
            self.assertFalse(exc.result.ok)
            self.assertEqual(exc.result.kind, "worker")
            self.assertGreater(exc.shortfall, 0)
            self.assertIn("refused", str(exc).lower())
            # Same numbers, caller-visible without going through compile.
            packet = compile_worker_context(
                wu,
                compiled,
                phase="running",
                units={wu.id: wu},
                steering=[],
                workspace=tmp,
            )
            evidence_tokens = packet.evidence[0].size // 3
            with self.assertRaises(PacketBudgetError) as ctx2:
                fit_or_refuse(
                    budget,
                    packet.prompt,
                    evidence_tokens=evidence_tokens,
                    kind="worker",
                )
            self.assertFalse(ctx2.exception.result.ok)


class TestEvidenceIdentityAcrossPacketBoundary(unittest.TestCase):
    """Watched against the pre-fix compiler:

    - WorkerPacket had evidence_paths (names only) and no evidence
      identity. Engine._assert_evidence_fresh silently re-read a mutated
      file into the payload.
    """

    def test_gather_stamps_path_size_mtime_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.txt"
            body = "snapshot-AAAA"
            notes.write_text(body, encoding="utf-8")
            wu = _wu(description="read notes.txt")
            packet = compile_worker_context(
                wu,
                _compiled(
                    goal="short task about notes.txt must pass",
                    referenced_files=["notes.txt"],
                ),
                phase="running",
                units={wu.id: wu},
                steering=[],
                workspace=tmp,
            )
            self.assertEqual(packet.evidence_paths, ("notes.txt",))
            self.assertEqual(len(packet.evidence), 1)
            ident = packet.evidence[0]
            self.assertIsInstance(ident, EvidenceIdentity)
            st = notes.stat()
            self.assertEqual(ident.path, "notes.txt")
            self.assertEqual(ident.size, st.st_size)
            self.assertEqual(ident.mtime_ns, st.st_mtime_ns)
            self.assertEqual(
                ident.sha256, hashlib.sha256(body.encode("utf-8")).hexdigest()
            )

    def test_stale_file_mutated_between_gather_and_use_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.txt"
            notes.write_text("snapshot-AAAA", encoding="utf-8")
            wu = _wu(description="read notes.txt")
            gathered = identity_for_path(notes, root=Path(tmp))
            packet = compile_worker_context(
                wu,
                _compiled(
                    goal="short task about notes.txt must pass",
                    referenced_files=["notes.txt"],
                ),
                phase="running",
                units={wu.id: wu},
                steering=[],
                workspace=tmp,
                evidence=[gathered],
            )
            self.assertEqual(packet.evidence[0].sha256, gathered.sha256)
            _rewrite(notes, "snapshot-BBBB")
            with self.assertRaises(StaleEvidenceError) as ctx:
                assert_packet_evidence_fresh(packet, tmp)
            message = str(ctx.exception)
            self.assertIn("stale evidence refused", message)
            self.assertIn("notes.txt", message)
            self.assertIn(gathered.sha256, message)
            now = identity_for_path(notes, root=Path(tmp))
            self.assertIn(now.sha256, message)
            self.assertNotEqual(gathered.sha256, now.sha256)

    def test_unchanged_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.txt"
            notes.write_text("snapshot-AAAA", encoding="utf-8")
            wu = _wu(description="read notes.txt")
            packet = compile_worker_context(
                wu,
                _compiled(
                    goal="short task about notes.txt must pass",
                    referenced_files=["notes.txt"],
                ),
                phase="running",
                units={wu.id: wu},
                steering=[],
                workspace=tmp,
            )
            verified = assert_packet_evidence_fresh(packet, tmp)
            self.assertEqual(verified[0].sha256, packet.evidence[0].sha256)


class TestPacketDeterminism(unittest.TestCase):
    """Same unit + same evidence must compile to the same packet.

    Watched: failure_context key insertion order changed the FAILURE_CONTEXT
    JSON (json.dumps without sort_keys), which would break a packet cache.
    """

    def test_repeated_compile_is_byte_identical(self):
        wu = _wu(
            description="edit foo.py",
            failure_context={"z": 1, "a": 2},
        )
        compiled = _compiled(
            goal="short task about foo.py must pass",
            referenced_files=["foo.py"],
        )
        kwargs = dict(
            phase="running",
            units={wu.id: wu},
            steering=["[constraint] stay in foo.py"],
        )
        first = compile_worker_context(wu, compiled, **kwargs)
        second = compile_worker_context(wu, compiled, **kwargs)
        self.assertEqual(first.prompt, second.prompt)
        self.assertEqual(first, second)

    def test_failure_context_key_order_does_not_break_caching(self):
        compiled = _compiled(goal="short task must pass")
        wu_a = _wu(description="x", failure_context={"z": 1, "a": 2})
        wu_b = _wu(description="x", failure_context={"a": 2, "z": 1})
        a = compile_worker_context(
            wu_a, compiled, phase="running", units={wu_a.id: wu_a}, steering=[]
        )
        b = compile_worker_context(
            wu_b, compiled, phase="running", units={wu_b.id: wu_b}, steering=[]
        )
        self.assertEqual(a.prompt, b.prompt)

    def test_consult_uses_compiled_packet_not_root(self):
        wu = _wu(description="edit foo.py")
        packet = _packet(
            wu, _compiled(referenced_files=["foo.py"])
        )
        seen = {}

        class Bridge:
            def consult(self, prompt, **kwargs):
                seen["prompt"] = prompt
                return "handle"

        result = consult_worker(Bridge(), packet)
        self.assertEqual(result, "handle")
        self.assertEqual(seen["prompt"], packet.prompt)
        self.assertNotIn(ROOT, seen["prompt"])
        refuse_goal_dump(seen["prompt"], ROOT)

    def test_what_varies_named(self):
        """Fields that vary, and whether they break packet caching.

        - evidence.mtime_ns / sha256 / size: identity of the file at gather.
          A rewrite between compiles produces a different packet.evidence
          tuple; that MUST break caching, otherwise a stale snapshot would
          cache-hit. Tested in TestEvidenceIdentityAcrossPacketBoundary.
        - steering list order: preserved as the caller supplied it. Two
          compiles with the same events in the same order match; reordering
          constraints is a different packet (operator intent).
        - SteerEvent.id / timestamp: not copied into the packet (only
          kind+text), so they do not break caching.
        - WorkUnit.ready_at / running_at / finished_at: not in the packet.
        - ROOT_REF path: supplied by the caller; stable for a workspace.
        """
        wu = _wu(description="edit foo.py")
        compiled = _compiled(
            goal="short task about foo.py must pass",
            referenced_files=["foo.py"],
        )
        p1 = compile_worker_context(
            wu,
            compiled,
            phase="running",
            units={wu.id: wu},
            steering=["[constraint] a", "[knowledge] b"],
        )
        p2 = compile_worker_context(
            wu,
            compiled,
            phase="running",
            units={wu.id: wu},
            steering=["[knowledge] b", "[constraint] a"],
        )
        self.assertNotEqual(p1.prompt, p2.prompt)
        self.assertNotEqual(p1.steering, p2.steering)


if __name__ == "__main__":
    unittest.main()
