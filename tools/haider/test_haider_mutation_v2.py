import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__)),
)

import haider
import p0_tool_bridge as p0


class MutationV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.guard = p0.RepositoryGuard(self.root)

        Path(self.root, "a.py").write_text(
            "HEADER\nVALUE = 1\nFOOTER\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_replace(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ],
        )

        self.assertIsNotNone(result)
        self.assertIn(
            "VALUE = 2",
            Path(self.root, "a.py").read_text(),
        )

    def test_insert_after(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "insert_after",
                    "path": "a.py",
                    "anchor": "HEADER\n",
                    "text": "INSERTED\n",
                }
            ],
        )

        self.assertIsNotNone(result)

        text = Path(self.root, "a.py").read_text()

        self.assertIn(
            "HEADER\nINSERTED\n",
            text,
        )

    def test_insert_before(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "insert_before",
                    "path": "a.py",
                    "anchor": "FOOTER",
                    "text": "BEFORE\n",
                }
            ],
        )

        self.assertIsNotNone(result)

        self.assertIn(
            "BEFORE\nFOOTER",
            Path(self.root, "a.py").read_text(),
        )

    def test_create(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "create",
                    "path": "test_new.py",
                    "content": "VALUE = 42\n",
                }
            ],
        )

        self.assertIsNotNone(result)

        self.assertEqual(
            Path(
                self.root,
                "test_new.py",
            ).read_text(),
            "VALUE = 42\n",
        )

    def test_create_existing_rejected(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "create",
                    "path": "a.py",
                    "content": "oops\n",
                }
            ],
        )

        self.assertIsNone(result)

    def test_git_rejected(self):
        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "create",
                    "path": ".git/config",
                    "content": "bad\n",
                }
            ],
        )

        self.assertIsNone(result)

    def test_all_validated_before_write(self):
        original = Path(
            self.root,
            "a.py",
        ).read_text()

        result = haider.apply_mutation_operations(
            self.guard,
            [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                },
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "DOES NOT EXIST",
                    "new_text": "x",
                },
            ],
        )

        self.assertIsNone(result)

        self.assertEqual(
            Path(
                self.root,
                "a.py",
            ).read_text(),
            original,
        )

    def test_max_operations(self):
        allowed = {
            "operations": [
                {
                    "op": "create",
                    "path": f"x{i}.txt",
                    "content": "x",
                }
                for i in range(20)
            ]
        }

        too_many = {
            "operations": [
                {
                    "op": "create",
                    "path": f"x{i}.txt",
                    "content": "x",
                }
                for i in range(21)
            ]
        }

        self.assertIsNotNone(
            haider._candidate_operations_valid(
                allowed,
                self.guard,
            )
        )

        self.assertFalse(
            haider._candidate_operations_valid(
                too_many,
                self.guard,
            )
        )

if __name__ == "__main__":
    unittest.main()
