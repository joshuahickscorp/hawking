from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.cli import install_shims


class TestInstallShims(unittest.TestCase):
    def test_install_shims_writes_identical_hcli_and_jhcli(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = install_shims(home=tmp)
            self.assertEqual(rc, 0)
            bin_dir = Path(tmp) / ".local" / "bin"
            hcli = bin_dir / "hcli"
            jhcli = bin_dir / "jhcli"
            self.assertTrue(hcli.is_file())
            self.assertTrue(jhcli.is_file())
            self.assertEqual(hcli.read_text(), jhcli.read_text())
            for path in (hcli, jhcli):
                mode = path.stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR)
            text = hcli.read_text()
            self.assertIn("-m hcli", text)
            self.assertIn(".local/share/hcli/current", text)
            current = Path(tmp) / ".local" / "share" / "hcli" / "current"
            self.assertTrue(current.is_symlink())
            pkg = current / "hcli" / "cli.py"
            self.assertTrue(pkg.is_file())
            self.assertIn("install-shims", pkg.read_text())


if __name__ == "__main__":
    unittest.main()
