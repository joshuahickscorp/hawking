"""The folder you opened HCLI in, and what it is allowed to do to an answer.

Two failures are guarded here, and they pull in opposite directions:

  * a session opened in a repo that knows nothing about it (what `hcli web` did
    before this: asked what files it could see, it explained the `ls` command)
  * a session opened in a repo that turns EVERY question into a question about
    that repo, which the campaign's own rule forbids -- cwd is context, not
    identity.

The third is a precision failure that produced confident nonsense: four files
named `catalog.py` were all injected and the model described a blend of them.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from hcli.repo_context import RepoContext, inject


def _repo(tmp: Path, git: bool = True) -> Path:
    (tmp / "hcli").mkdir(parents=True)
    (tmp / "tools").mkdir(parents=True)
    (tmp / "README.md").write_text("# widget\n\nA thing that widgets.\n")
    (tmp / "hcli" / "catalog.py").write_text("CATALOG_MARKER = 'the real catalog'\n")
    (tmp / "tools" / "catalog.py").write_text("OTHER_MARKER = 'a different catalog'\n")
    (tmp / "hcli" / "serve.py").write_text("SERVE_MARKER = 'serving'\n")
    if git:
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "x"], cwd=tmp, check=True)
    return tmp


class TestDetection(unittest.TestCase):
    def test_a_git_repo_is_detected_from_a_subdirectory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _repo(Path(raw).resolve())
            found = RepoContext.detect(str(root / "hcli"))
            self.assertIsNotNone(found)
            self.assertEqual(found.root, root)
            self.assertTrue(found.is_git)

    def test_an_empty_directory_is_not_context(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertIsNone(RepoContext.detect(raw))


class TestRetrievalPrecision(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _repo(Path(self._tmp.name).resolve())
        self.ctx = RepoContext.detect(str(self.root))
        self.addCleanup(self._tmp.cleanup)

    def test_a_named_path_returns_that_file_alone(self):
        # THE DEFECT: asking about hcli/catalog.py returned it AND every other
        # catalog.py, and the answer was a fluent blend of files that describes
        # none of them.
        picked = [p for p, _ in self.ctx.files_for("what is hcli/catalog.py for?")]
        self.assertEqual(picked, ["hcli/catalog.py"], picked)

    def test_a_named_path_beats_a_stem_match_elsewhere(self):
        picked = [p for p, _ in self.ctx.files_for("how does hcli/serve.py work")]
        self.assertEqual(picked[0], "hcli/serve.py")

    def test_a_vague_question_may_return_several(self):
        picked = [p for p, _ in self.ctx.files_for("what does the catalog do")]
        self.assertGreaterEqual(len(picked), 2, picked)

    def test_a_question_matching_nothing_returns_nothing(self):
        self.assertEqual(self.ctx.files_for("what is the capital of France"), [])


class TestInjection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _repo(Path(self._tmp.name).resolve())
        self.ctx = RepoContext.detect(str(self.root))
        self.addCleanup(self._tmp.cleanup)

    def test_no_context_changes_nothing(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(inject(msgs, None), msgs)

    def test_the_repo_is_named_and_the_file_is_included(self):
        out = inject([{"role": "user", "content": "what is hcli/catalog.py for?"}],
                     self.ctx)
        self.assertEqual(out[0]["role"], "system")
        system = out[0]["content"]
        self.assertIn(self.root.name, system)
        self.assertIn("CATALOG_MARKER", system)
        self.assertNotIn("OTHER_MARKER", system, "an unrelated file was injected")
        self.assertEqual(out[-1]["role"], "user")

    def test_the_block_licenses_ignoring_the_repo(self):
        # cwd is CONTEXT, not IDENTITY: a general question asked inside a repo
        # must still get a general answer, so the prompt has to say so.
        system = inject([{"role": "user", "content": "hello"}],
                        self.ctx)[0]["content"]
        self.assertIn("general", system.lower())

    def test_the_stable_block_is_byte_identical_across_questions(self):
        # This is what makes prefix reuse work. If the invariant block moved or
        # changed between turns, every follow-up would re-prefill the whole
        # context: measured 0.5s versus 20s on the native resident.
        first = inject([{"role": "user", "content": "what is hcli/serve.py"}],
                       self.ctx)[0]["content"]
        second = inject([{"role": "user", "content": "what is hcli/catalog.py"}],
                        self.ctx)[0]["content"]
        stable = self.ctx.stable_block()
        self.assertTrue(first.startswith(stable))
        self.assertTrue(second.startswith(stable),
                        "the invariant prefix changed between turns")

    def test_the_conversation_is_preserved_in_order(self):
        msgs = [{"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"}]
        out = inject(msgs, self.ctx)
        self.assertEqual([m["content"] for m in out[1:]], ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()
