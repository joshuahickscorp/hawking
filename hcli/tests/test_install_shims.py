from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.cli import INSTALL_STAMP, install_shims


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
            self.assertIn("-P -m hcli", text)
            self.assertIn(".local/share/hcli/current", text)
            current = Path(tmp) / ".local" / "share" / "hcli" / "current"
            self.assertTrue(current.is_symlink())
            pkg = current / "hcli" / "cli.py"
            self.assertTrue(pkg.is_file())
            self.assertIn("install-shims", pkg.read_text())

    def test_install_bundles_a_built_native_gravity_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            package = source_root / "hcli"
            package.mkdir(parents=True)
            (package / "cli.py").write_text("# fixture\n", encoding="utf-8")
            native = source_root / "workspace/ops/build/rust/release/hawking-gravityd"
            native.parent.mkdir(parents=True)
            native.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            native.chmod(0o755)
            native_hcli = source_root / "workspace/ops/build/rust/release/hcli"
            native_hcli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            native_hcli.chmod(0o755)
            from unittest.mock import patch
            with patch("hcli.cli.__file__", str(package / "cli.py")):
                self.assertEqual(install_shims(home=str(root / "home")), 0)
            installed = (root / "home/.local/share/hcli/current").resolve()
            bundled = installed / "hawking-gravityd"
            self.assertTrue(bundled.is_file())
            self.assertTrue(bundled.stat().st_mode & stat.S_IXUSR)
            stamp = __import__("json").loads((installed / INSTALL_STAMP).read_text())
            self.assertEqual(stamp["native_gravityd"]["status"], "bundled")
            self.assertEqual(stamp["native_hcli"]["status"], "bundled")
            self.assertTrue((installed / "hcli-rust").is_file())
            self.assertTrue((installed / "hcli-rust").stat().st_mode & stat.S_IXUSR)

    def test_installed_snapshot_cannot_be_shadowed_by_the_launch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(install_shims(home=tmp), 0)
            poison = Path(tmp) / "poison"
            (poison / "hcli").mkdir(parents=True)
            (poison / "hcli" / "__init__.py").write_text(
                'raise RuntimeError("cwd package shadowed deployed HCLI")\n',
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["HOME"] = tmp
            done = subprocess.run(
                [str(Path(tmp) / ".local/bin/hcli"), "--help"],
                cwd=poison,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("HCLI — autonomous local model engineering", done.stdout)
            self.assertNotIn("cwd package shadowed", done.stderr)


if __name__ == "__main__":
    unittest.main()
