"""Deterministic tests for HAIDER scoped-edit validation.

Run:
    python3 tools/haider/test_haider_edit.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_tool_bridge as p0
import haider


class ScopedEditValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # Create a simple file to edit.
        self.file_path = os.path.join(self.root, "hello.py")
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("# hello\nprint('world')\n")
        self.guard = p0.RepositoryGuard(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_scoped_edit(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "hello.py",
            "# hello",
            "# hello world",
        )
        self.assertTrue(ok, f"expected ok, got err={err}")
        self.assertEqual(full_path, os.path.realpath(self.file_path))
        self.assertIn("# hello", original)

    def test_nonexistent_path_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "does_not_exist.py",
            "foo",
            "bar",
        )
        self.assertFalse(ok)
        self.assertIn("not a regular file", err)

    def test_missing_old_text_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "hello.py",
            "THIS TEXT IS NOT IN THE FILE",
            "replacement",
        )
        self.assertFalse(ok)
        self.assertIn("old_text not found", err)

    def test_noop_edit_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "hello.py",
            "# hello",
            "# hello",
        )
        self.assertFalse(ok)
        self.assertIn("no-op", err)

    def test_out_of_root_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "../outside.py",
            "foo",
            "bar",
        )
        self.assertFalse(ok)
        self.assertIn("path rejected", err)

    def test_empty_path_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "",
            "foo",
            "bar",
        )
        self.assertFalse(ok)
        self.assertIn("path is empty", err)

    def test_empty_old_text_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "hello.py",
            "",
            "bar",
        )
        self.assertFalse(ok)
        self.assertIn("old_text is empty", err)

    def test_empty_new_text_rejection(self):
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "hello.py",
            "# hello",
            "",
        )
        self.assertFalse(ok)
        self.assertIn("new_text is empty", err)

    def test_multiple_match_rejection(self):
        # Create a file with repeated text.
        multi_path = os.path.join(self.root, "multi.py")
        with open(multi_path, "w", encoding="utf-8") as f:
            f.write("x = 1\nx = 1\nx = 1\n")
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "multi.py",
            "x = 1",
            "x = 2",
        )
        self.assertFalse(ok)
        self.assertIn("matches 3 locations", err)

    def test_single_match_ok(self):
        # Create a file with unique text.
        single_path = os.path.join(self.root, "single.py")
        with open(single_path, "w", encoding="utf-8") as f:
            f.write("alpha\nbeta\ngamma\n")
        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            "single.py",
            "beta",
            "BETA",
        )
        self.assertTrue(ok, f"expected ok, got err={err}")



    def test_dot_git_config_rejected(self):
        git_dir = os.path.join(self.root, ".git")
        os.makedirs(git_dir, exist_ok=True)

        path = os.path.join(git_dir, "config")

        with open(path, "w", encoding="utf-8") as f:
            f.write("[core]\n")

        ok, err, full_path, original = haider.validate_scoped_edit(
            self.guard,
            ".git/config",
            "[core]",
            "[core]\nfoo = bar",
        )

        self.assertFalse(ok)
        self.assertIn(".git", err)
        self.assertIsNone(full_path)

    def test_nested_dot_git_rejected(self):
        ok, err, _, _ = haider.validate_scoped_edit(
            self.guard,
            "foo/.git/config",
            "x",
            "y",
        )

        self.assertFalse(ok)
        self.assertIn(".git", err)


class ExtractEditJsonTests(unittest.TestCase):
    def test_direct_json(self):
        result = haider._extract_edit_json('{"path":"a.py","old_text":"x","new_text":"y"}')
        self.assertIsNotNone(result)
        self.assertEqual(result["path"], "a.py")

    def test_json_in_prose(self):
        result = haider._extract_edit_json('Here is the edit: {"path":"a.py","old_text":"x","new_text":"y"} done')
        self.assertIsNotNone(result)
        self.assertEqual(result["path"], "a.py")

    def test_json_in_code_fence(self):
        result = haider._extract_edit_json('```json\n{"path":"a.py","old_text":"x","new_text":"y"}\n```')
        self.assertIsNotNone(result)
        self.assertEqual(result["path"], "a.py")

    def test_no_json(self):
        result = haider._extract_edit_json("I will make an edit to the file.")
        self.assertIsNone(result)

    def test_non_dict_json(self):
        result = haider._extract_edit_json("[1, 2, 3]")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
