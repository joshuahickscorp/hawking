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
from unittest.mock import patch

from hcli.repo_context import RepoContext, inject, prewarm_native_gravity_index


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

    def test_prewarm_starts_native_child_without_a_query(self):
        root = Path("/tmp/hawking-native-prewarm-fixture")
        binary = Path("/tmp/hawking-gravityd-fixture")
        with patch.dict("os.environ", {"HCLI_NATIVE_GRAVITY": "1"}, clear=False), \
             patch("hcli.repo_context._native_gravity_binary", return_value=binary), \
             patch("hcli.repo_context._NativeGravityIndex") as spawned, \
             patch("hcli.repo_context._NATIVE_GRAVITY_INDEXES", {}):
            self.assertTrue(prewarm_native_gravity_index(root))
            spawned.assert_called_once_with(root, binary)

    def test_native_binary_cache_drops_stale_paths_and_sees_later_install(self):
        import hcli.repo_context as repo_context

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            binary = root / "hawking-gravityd"
            binary.write_bytes(b"placeholder")
            binary.chmod(0o755)
            with patch.dict("os.environ", {"HCLI_GRAVITYD_BIN": str(binary)}, clear=False):
                repo_context._NATIVE_GRAVITY_BINARY_CACHE.pop((str(root), str(binary)), None)
                self.assertEqual(repo_context._native_gravity_binary(root), binary)
                binary.unlink()
                self.assertNotEqual(repo_context._native_gravity_binary(root), binary)
                binary.write_bytes(b"reinstalled")
                binary.chmod(0o755)
                self.assertEqual(repo_context._native_gravity_binary(root), binary)


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

    def test_goal_only_question_uses_distinctive_content_for_orientation(self):
        target = self.root / "hcli" / "session_transport.py"
        target.write_text(
            "OpenAI-compatible streaming browser chat persists session state "
            "across refresh and restart.\n"
        )
        with patch.object(
            self.ctx,
            "_candidate_files",
            side_effect=AssertionError("content orientation must not walk the tree"),
        ):
            picked = self.ctx.paths_for(
                "Determine whether OpenAI-compatible streaming browser chat "
                "preserves durable session state across refresh or restart."
            )
        self.assertTrue(picked)
        self.assertEqual(picked[0], "hcli/session_transport.py")


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
