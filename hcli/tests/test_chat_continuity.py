"""Crossing a context window and crossing a process boundary.

These are S035 s25's adversarial tests, made runnable:

  TEST A  a fact given early survives compaction and is retrievable
  TEST D  compact between RED and the mutation; the exact next action resumes
  TEST E  kill the process after a checkpoint; the objective comes back

Plus the property that makes compaction affordable at all: the invariant
leading region is untouched, so the resident's prefix cache still hits.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_continuity import (
    checkpoint,
    compact,
    estimate_tokens,
    resume,
    resume_block,
)
from hcli.chat_state import ChatSession


def _conversation(turns=40, size=400):
    out = [{"role": "system", "content": "TOOLS AND REPO IDENTITY -- invariant"}]
    out.append({"role": "user", "content":
                "DECISIVE: we rejected chunk 16 because dispatches tripled"})
    for i in range(turns):
        out.append({"role": "assistant", "content": f"turn {i} " + "x" * size})
        out.append({"role": "user", "content": f"next {i} " + "y" * size})
    return out


class TestCompaction(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.session = ChatSession.load(self._tmp.name, "s1")

    def test_a_small_conversation_is_left_alone(self):
        messages = _conversation(turns=1)
        kept, report = compact(messages, self.session, window=8192)
        self.assertFalse(report.compacted)
        self.assertEqual(kept, messages)

    def test_a_long_conversation_is_compacted(self):
        messages = _conversation()
        kept, report = compact(messages, self.session, window=4096)
        self.assertTrue(report.compacted)
        self.assertLess(report.after_tokens, report.before_tokens)
        self.assertGreater(report.evicted_turns, 0)

    def test_the_invariant_prefix_is_untouched(self):
        # This is what keeps the 30x native prefix reuse. Rewriting the front
        # of the prompt would trade a physical win for a semantic tidy-up.
        messages = _conversation()
        kept, _ = compact(messages, self.session, window=4096)
        self.assertEqual(kept[0], messages[0])

    def test_the_recent_thread_is_kept(self):
        messages = _conversation()
        kept, _ = compact(messages, self.session, window=4096)
        self.assertEqual(kept[-1], messages[-1])
        self.assertEqual(kept[-2], messages[-2])

    def test_TEST_A_the_evicted_fact_is_archived_not_destroyed(self):
        messages = _conversation()
        kept, report = compact(messages, self.session, window=4096)
        flattened = " ".join(str(m.get("content")) for m in kept)
        self.assertNotIn("DECISIVE", flattened, "fixture wrong: it was not evicted")
        archive = Path(report.archive)
        self.assertTrue(archive.is_file())
        body = archive.read_text(encoding="utf-8")
        self.assertIn("DECISIVE", body,
                      "an evicted turn was destroyed, not compacted")
        self.assertIn("chunk 16", body)

    def test_the_marker_tells_the_model_where_the_rest_went(self):
        messages = _conversation()
        kept, report = compact(messages, self.session, window=4096)
        marker = " ".join(str(m.get("content")) for m in kept)
        self.assertIn("archived at", marker)
        self.assertIn(report.archive, marker)


class TestCheckpointAndResume(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.session = ChatSession.load(self._tmp.name, "s1")
        self.session.objective = "make long runs survive"
        self.session.add_plan("Context gravity", "prose",
                              steps=["evidence store", "compaction", "resume"])

    def test_TEST_E_the_objective_survives_a_new_process(self):
        plan = self.session.plan()
        plan.current_step = 1
        self.session.update_plan(plan)
        checkpoint(self.session, resident="sealed-3.14")
        self.session.save()

        # a new process: nothing in memory, only the workspace
        revived = ChatSession.load(self._tmp.name, "s1")
        record = resume(revived)
        self.assertIsNotNone(record)
        self.assertEqual(record["objective"], "make long runs survive")
        self.assertEqual(record["next_action"], "compaction")
        self.assertIn("P1 step 2", record["active_workunit"])

    def test_TEST_D_the_exact_next_action_is_what_resumes(self):
        plan = self.session.plan()
        plan.current_step = 2
        self.session.update_plan(plan)
        checkpoint(self.session, next_action="write the RED continuation test")
        record = resume(ChatSession.load(self._tmp.name, "s1"))
        self.assertEqual(record["next_action"], "write the RED continuation test")
        self.assertIn("write the RED", resume_block(record))

    def test_no_checkpoint_resumes_to_nothing(self):
        self.assertIsNone(resume(ChatSession.load(self._tmp.name, "fresh")))
        self.assertEqual(resume_block(None), "")

    def test_a_moved_repository_is_reported_not_replayed(self):
        import subprocess
        root = Path(self._tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "one"],
                       cwd=root, check=True)
        session = ChatSession.load(str(root), "s2")
        session.objective = "obj"
        from hcli.chat_state import _head_commit
        session.add_plan("P", "b", steps=["a"], repo_commit=_head_commit(str(root)))
        checkpoint(session)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "two"],
                       cwd=root, check=True)
        record = resume(ChatSession.load(str(root), "s2"))
        self.assertTrue(record["diverged"],
                        "the repository moved and resume did not notice")
        self.assertIn("re-inspect", record["divergence"])
        self.assertIn("DIVERGED", resume_block(record))

    def test_the_checkpoint_uses_the_field_names_the_resident_already_reads(self):
        # One reader must serve both surfaces, so a durable objective does not
        # depend on which one wrote it.
        record = checkpoint(self.session)
        for field in ("objective", "active_workunit", "hypothesis",
                      "next_action", "evidence_refs"):
            self.assertIn(field, record)


if __name__ == "__main__":
    unittest.main()
