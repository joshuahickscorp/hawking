"""Control-plane performance: in-process syntax check, pytest stays isolated.

py_compile is the stdlib parser, not an isolation boundary. Contained pytest
of model-written tests is; this file locks that distinction so a later 20%
LOC cut cannot inline untrusted tests into the Engine.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hcli.engine import Engine
from hcli.events import EventBus
from hcli.mutation import compile_python_file, validate_python_syntax
from hcli.workspace import Workspace

REPO = Path(__file__).resolve().parents[4]

VALID = "def add(a, b):\n    return a + b\n"
INVALID = "def add(a, b)\n    return a + b\n"
TEST_ADD = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


class TestCompilePythonFile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_valid_source_ok(self):
        path = self.root / "ok.py"
        path.write_text(VALID, encoding="utf-8")
        got = compile_python_file(path)
        self.assertTrue(got["ok"])
        self.assertEqual(got["exit_code"], 0)
        self.assertEqual(got["stderr"], "")

    def test_syntax_error_not_ok(self):
        path = self.root / "bad.py"
        path.write_text(INVALID, encoding="utf-8")
        got = compile_python_file(path)
        self.assertFalse(got["ok"])
        self.assertEqual(got["exit_code"], 1)
        self.assertIn("SyntaxError", got["stderr"])

    def test_missing_file_not_ok(self):
        got = compile_python_file(self.root / "absent.py")
        self.assertFalse(got["ok"])
        self.assertEqual(got["exit_code"], 1)

    def test_validate_python_syntax_bool(self):
        good = self.root / "good.py"
        bad = self.root / "bad.py"
        good.write_text(VALID, encoding="utf-8")
        bad.write_text(INVALID, encoding="utf-8")
        self.assertTrue(validate_python_syntax(str(good)))
        self.assertFalse(validate_python_syntax(str(bad)))

    def test_does_not_spawn_py_compile(self):
        path = self.root / "ok.py"
        path.write_text(VALID, encoding="utf-8")
        src = inspect.getsource(compile_python_file)
        self.assertIn("compile(", src)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("Popen", src)
        with patch("subprocess.run") as mock_run:
            self.assertTrue(validate_python_syntax(str(path)))
            mock_run.assert_not_called()


class TestEngineValidateCeremony(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.engine = Engine(
            workspace=Workspace(str(self.root)),
            event_bus=EventBus(),
            runtime_count=1,
            model_name="/missing.gguf",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_syntax_error_record_keeps_py_compile_kind(self):
        bad = self.engine.root / "bad.py"
        bad.write_text(INVALID, encoding="utf-8")
        validation = self.engine._validate([bad], ["bad.py"])
        self.assertFalse(validation["ok"])
        syntax = [c for c in validation["checks"] if c.get("kind") == "py_compile"]
        self.assertEqual(len(syntax), 1)
        self.assertEqual(syntax[0]["exit_code"], 1)
        self.assertIn("SyntaxError", syntax[0]["stderr"])

    def test_syntax_error_does_not_spawn_interpreter(self):
        bad = self.engine.root / "bad.py"
        bad.write_text(INVALID, encoding="utf-8")
        with patch("hcli.engine.subprocess.run") as mock_run, patch(
            "hcli.engine.subprocess.Popen"
        ) as mock_popen:
            validation = self.engine._validate([bad], ["test_missing.py"])
        self.assertFalse(validation["ok"])
        mock_run.assert_not_called()
        mock_popen.assert_not_called()

    def test_pytest_process_boundary_stays(self):
        (self.engine.root / "calc.py").write_text(VALID, encoding="utf-8")
        (self.engine.root / "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        recorded: list = []
        real_popen = subprocess.Popen

        def wrapped(argv, *args, **kwargs):
            recorded.append(list(argv))
            return real_popen(argv, *args, **kwargs)

        with patch("hcli.engine.subprocess.Popen", wrapped):
            validation = self.engine._validate(
                [self.engine.root / "calc.py", self.engine.root / "test_calc.py"],
                ["test_calc.py"],
            )
        self.assertTrue(validation.get("ok"), validation)
        pytest_argv = [a for a in recorded if "-m" in a and "pytest" in a]
        self.assertEqual(len(pytest_argv), 1, recorded)
        self.assertTrue(
            all("py_compile" not in a for a in recorded),
            recorded,
        )
        self.assertTrue(
            all("-c" not in a or a[a.index("-c") + 1] != "import pytest" for a in recorded),
            recorded,
        )

    def test_pytest_importable_uses_find_spec(self):
        with patch("importlib.util.find_spec", return_value=None) as mock_spec, patch(
            "hcli.engine.subprocess.run"
        ) as mock_run:
            self.engine._pytest_importable_cached = None
            self.assertFalse(self.engine._pytest_importable())
            mock_spec.assert_called_with("pytest")
            mock_run.assert_not_called()
        self.engine._pytest_importable_cached = None
        self.assertTrue(self.engine._pytest_importable())


class TestPackageInitLazy(unittest.TestCase):
    def test_from_hcli_import_controller_still_works(self):
        from hcli import Controller
        from hcli.controller import Controller as Direct

        self.assertIs(Controller, Direct)

    def test_parse_haider_args_is_eager(self):
        import hcli

        self.assertIn("parse_haider_args", hcli.__dict__)
        self.assertTrue(callable(hcli.parse_haider_args))


class TestHelpPath(unittest.TestCase):
    def test_help_does_not_load_runtime_or_heavy_stacks(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTHONPROFILEIMPORTTIME", None)
        script = r"""
import runpy, sys
sys.argv = ["hcli", "--help"]
try:
    runpy.run_module("hcli", run_name="__main__")
except SystemExit as exc:
    code = 0 if exc.code in (0, None) else int(exc.code or 1)
else:
    code = 0
mods = set(sys.modules)
hcli = sorted(m for m in mods if m == "hcli" or m.startswith("hcli."))
heavy = [
    n
    for n in (
        "mlx",
        "mlx_lm",
        "torch",
        "numpy",
        "scipy",
        "cv2",
        "open3d",
        "visionmcp",
        "PIL",
        "prompt_toolkit",
    )
    if n in mods
]
forbidden = [m for m in hcli if m not in ("hcli", "hcli.cli")]
print("CODE", code)
print("HCLI", ",".join(hcli))
print("HEAVY", ",".join(heavy))
print("FORBIDDEN", ",".join(forbidden))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = {
            ln.split(" ", 1)[0]: (ln.split(" ", 1)[1] if " " in ln else "")
            for ln in proc.stdout.splitlines()
            if ln.startswith(("CODE ", "HCLI ", "HEAVY ", "FORBIDDEN "))
        }
        self.assertEqual(lines.get("CODE"), "0", proc.stdout)
        self.assertEqual(lines.get("HEAVY"), "", proc.stdout)
        self.assertEqual(lines.get("FORBIDDEN"), "", proc.stdout)
        hcli_mods = [m for m in lines.get("HCLI", "").split(",") if m]
        self.assertEqual(hcli_mods, ["hcli", "hcli.cli"], proc.stdout)


if __name__ == "__main__":
    unittest.main()
