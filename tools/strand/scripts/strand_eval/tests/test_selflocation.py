# Self-location PROOF tests (audit 3.3): the CLI works when invoked from /,
# via a symlink, and from a copied package path (with STRAND_ROOT). The exact
# REPO_ROOT incident class (script resolves //Cargo.toml) must die loudly.
# Torch-free: uses the `where` and `ledger` subcommands only.
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_TESTS = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.dirname(_TESTS)                       # tools/strand_eval
REPO = os.path.dirname(os.path.dirname(PKG_DIR))        # the repo root
CLI_SH = os.path.join(REPO, "scripts", "strand-eval")
PY = sys.executable


def run(cmd, cwd=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)


class TestSelfLocation(unittest.TestCase):
    def test_invoked_from_root_cwd(self):
        r = run([CLI_SH, "where"], cwd="/")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"repo_root  : {os.path.realpath(REPO)}", r.stdout)

    def test_module_cli_from_root_cwd(self):
        r = run([PY, os.path.join(PKG_DIR, "cli.py"), "where"], cwd="/")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("repo_root", r.stdout)

    def test_via_symlink(self):
        with tempfile.TemporaryDirectory(prefix="se_sym_") as td:
            link = os.path.join(td, "strand-eval-link")
            os.symlink(CLI_SH, link)
            r = run([link, "where"], cwd="/")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(os.path.realpath(REPO), r.stdout)

    def test_copied_package_with_strand_root(self):
        # the package copied OUT of the repo: self-locates via STRAND_ROOT
        with tempfile.TemporaryDirectory(prefix="se_copy_") as td:
            dst = os.path.join(td, "strand_eval")
            shutil.copytree(PKG_DIR, dst, ignore=shutil.ignore_patterns("tests", "__pycache__"))
            r = run([PY, os.path.join(dst, "cli.py"), "where"], cwd="/",
                    env={"STRAND_ROOT": REPO})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(os.path.realpath(REPO), r.stdout)

    def test_copied_package_without_root_dies_loudly(self):
        with tempfile.TemporaryDirectory(prefix="se_lost_") as td:
            dst = os.path.join(td, "strand_eval")
            shutil.copytree(PKG_DIR, dst, ignore=shutil.ignore_patterns("tests", "__pycache__"))
            env = {"STRAND_ROOT": ""}
            r = run([PY, os.path.join(dst, "cli.py"), "where"], cwd="/", env=env)
            self.assertNotEqual(r.returncode, 0,
                                "a lost copy must REFUSE to guess, not invent a root")
            self.assertIn("STRAND_ROOT", r.stdout + r.stderr)

    def test_shell_wrapper_copied_offrepo_dies_loudly(self):
        # the REPO_ROOT incident shape: script cp'd to /workspace-like dir
        with tempfile.TemporaryDirectory(prefix="se_ship_") as td:
            dst = os.path.join(td, "strand-eval")
            shutil.copy2(CLI_SH, dst)
            r = run([dst, "where"], cwd="/", env={"STRAND_ROOT": ""})
            self.assertEqual(r.returncode, 3, f"expected loud death, got: {r.stdout}{r.stderr}")
            self.assertIn("not the strand repo root", r.stderr)

    def test_ledger_check_cwd_independent(self):
        r = run([CLI_SH, "ledger", "check"], cwd="/")
        # empty/clean ledger -> rc 0; the point is it found the ledger from /
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("record(s)", r.stdout)


if __name__ == "__main__":
    unittest.main()
