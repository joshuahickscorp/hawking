"""FORENSICS preserves the failure context before cleanup; REVERSE IDs a file.

S036 s48: preserve evidence BEFORE cleanup -- a daemon that destroys failure
evidence cannot improve. S036 s46: reverse engineering is normal engineering;
the basic rung is metadata + magic-byte classification.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.forensics_capability import capture, identify


class TestCapture(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "parser.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-qm", "one"], cwd=self.root, check=True)

    def test_it_captures_the_dirty_files_with_hashes(self):
        (self.root / "parser.py").write_text("x = 2\n", encoding="utf-8")
        snap = capture(str(self.root), note="parser crashed")
        self.assertEqual(snap.dirty, ["parser.py"],
                         "the porcelain path was mangled")
        self.assertIn("parser.py", snap.changed_hashes,
                      "the dirty file was not hashed")

    def test_it_persists_before_returning(self):
        snap = capture(str(self.root), reason="crash")
        self.assertTrue(Path(snap.path).is_file(),
                        "the snapshot was not written to disk")

    def test_it_records_head_and_recent_commits(self):
        snap = capture(str(self.root))
        self.assertTrue(snap.head)
        self.assertTrue(any("one" in c for c in snap.recent_commits))

    def test_capture_never_raises_even_in_a_non_repo(self):
        with TemporaryDirectory() as plain:
            snap = capture(plain)  # not a git repo -- must not raise
            self.assertEqual(snap.dirty, [])


class TestIdentify(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_python_file_is_utf8_text(self):
        (self.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual(identify(str(self.root / "a.py"))["kind"], "utf-8 text")

    def test_an_elf_blob_is_recognised(self):
        (self.root / "blob").write_bytes(b"\x7fELF\x02\x01\x01\x00")
        self.assertEqual(identify(str(self.root / "blob"))["kind"], "ELF executable")

    def test_a_macho_blob_is_recognised(self):
        (self.root / "m").write_bytes(b"\xcf\xfa\xed\xfe" + b"\x00" * 8)
        self.assertEqual(identify(str(self.root / "m"))["kind"], "Mach-O 64-bit")

    def test_binary_noise_is_flagged_not_guessed(self):
        (self.root / "n").write_bytes(b"\x80\x81\x82\x83")
        self.assertEqual(identify(str(self.root / "n"))["kind"],
                         "binary (unrecognised)")

    def test_a_missing_file_is_an_error_not_a_crash(self):
        self.assertIn("error", identify(str(self.root / "nope")))

    def test_it_carries_a_hash(self):
        (self.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        self.assertTrue(identify(str(self.root / "a.py"))["sha256_16"])


if __name__ == "__main__":
    unittest.main()
