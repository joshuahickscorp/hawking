from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hawking.cli import INSTALL_STAMP, install_shims


class TestInstallShims(unittest.TestCase):
    def test_install_shims_writes_identical_hawking_and_short_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = install_shims(home=tmp)
            self.assertEqual(rc, 0)
            bin_dir = Path(tmp) / ".local" / "bin"
            hawking = bin_dir / "hawking"
            short = bin_dir / "h"
            self.assertTrue(hawking.is_file())
            self.assertTrue(short.is_file())
            self.assertEqual(hawking.read_text(), short.read_text())
            for path in (hawking, short):
                mode = path.stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR)
            text = hawking.read_text()
            self.assertIn("-m hawking", text)
            self.assertIn("-P -m hawking", text)
            self.assertIn(".local/share/hawking/current", text)
            current = Path(tmp) / ".local" / "share" / "hawking" / "current"
            self.assertTrue(current.is_symlink())
            pkg = current / "hawking" / "cli.py"
            self.assertTrue(pkg.is_file())
            self.assertIn("install-shims", pkg.read_text())

    def test_install_bundles_a_built_native_gravity_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            package = source_root / "hawking"
            package.mkdir(parents=True)
            (package / "cli.py").write_text("# fixture\n", encoding="utf-8")
            native = source_root / "workspace/ops/build/rust/release/hawking-gravityd"
            native.parent.mkdir(parents=True)
            native.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            native.chmod(0o755)
            native_hawking = source_root / "workspace/ops/build/rust/release/hawking"
            native_hawking.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            native_hawking.chmod(0o755)
            native_process_authority = source_root / "workspace/ops/build/rust/release/hawking-process-authority"
            native_process_authority.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            native_process_authority.chmod(0o755)
            from unittest.mock import patch
            with patch("hawking.cli.__file__", str(package / "cli.py")):
                self.assertEqual(install_shims(home=str(root / "home")), 0)
            installed = (root / "home/.local/share/hawking/current").resolve()
            bundled = installed / "hawking-gravityd"
            self.assertTrue(bundled.is_file())
            self.assertTrue(bundled.stat().st_mode & stat.S_IXUSR)
            stamp = __import__("json").loads((installed / INSTALL_STAMP).read_text())
            self.assertEqual(stamp["native_gravityd"]["status"], "bundled")
            self.assertEqual(stamp["native_hawking"]["status"], "bundled")
            self.assertEqual(stamp["native_process_authority"]["status"], "bundled")
            self.assertTrue((installed / "hawking-rust").is_file())
            self.assertTrue((installed / "hawking-rust").stat().st_mode & stat.S_IXUSR)
            self.assertTrue((installed / "hawking-process-authority").is_file())

    def test_installed_snapshot_cannot_be_shadowed_by_the_launch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(install_shims(home=tmp), 0)
            poison = Path(tmp) / "poison"
            (poison / "hawking").mkdir(parents=True)
            (poison / "hawking" / "__init__.py").write_text(
                'raise RuntimeError("cwd package shadowed deployed HAWKING")\n',
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["HOME"] = tmp
            done = subprocess.run(
                [str(Path(tmp) / ".local/bin/hawking"), "--help"],
                cwd=poison,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("HAWKING — autonomous local model engineering", done.stdout)
            self.assertNotIn("cwd package shadowed", done.stderr)


if __name__ == "__main__":
    unittest.main()
