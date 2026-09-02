"""No-op mutations and content hashes. Rollback of a created file."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from hcli.mutation import (
    MutationError,
    apply_mutation_operations,
    content_fingerprint,
    mutation_content_hash,
    rollback_mutation,
)


class _Guard:
    def resolve(self, path: str) -> str:
        return path


class TestMutationNoOpAndHash(unittest.TestCase):
    def test_empty_operations_are_noop(self):
        with self.assertRaises(MutationError) as ctx:
            apply_mutation_operations(_Guard(), [])
        self.assertIn("NO_OP_MUTATION", str(ctx.exception))

    def test_real_replace_records_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "a.py")
            Path(path).write_text("x = 1\n", encoding="utf-8")
            result = apply_mutation_operations(
                _Guard(),
                [
                    {
                        "op": "replace",
                        "path": path,
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    }
                ],
            )
            self.assertTrue(result["content_hash"])
            self.assertEqual(result["content_hash"], mutation_content_hash(result))
            files = result["files"]
            self.assertEqual(len(files), 1)
            self.assertNotEqual(files[0]["sha256_before"], files[0]["sha256_after"])
            same = content_fingerprint([path])
            Path(path).write_text("x = 2\n", encoding="utf-8")
            self.assertEqual(content_fingerprint([path]), same)
            Path(path).write_text("x = 3\n", encoding="utf-8")
            self.assertNotEqual(content_fingerprint([path]), same)

    def test_identical_replace_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "a.py")
            Path(path).write_text("x = 1\n", encoding="utf-8")
            with self.assertRaises(MutationError) as ctx:
                apply_mutation_operations(
                    _Guard(),
                    [
                        {
                            "op": "replace",
                            "path": path,
                            "old_text": "x = 1",
                            "new_text": "x = 1",
                        }
                    ],
                )
            self.assertIn("NO_OP_MUTATION", str(ctx.exception))

    def test_rewrites_tests_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_a.py")
            Path(path).write_text("x = 1\n", encoding="utf-8")
            result = apply_mutation_operations(
                _Guard(),
                [
                    {
                        "op": "replace",
                        "path": path,
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    }
                ],
            )
            self.assertTrue(result["rewrites_tests"])
            self.assertEqual(result["test_paths"], [path])

    def test_rollback_removes_created_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "new.py")
            result = apply_mutation_operations(
                _Guard(),
                [{"op": "create", "path": path, "content": "x = 1\n"}],
            )
            self.assertTrue(Path(path).is_file())
            rollback_mutation(result)
            self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
