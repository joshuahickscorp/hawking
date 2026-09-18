"""Hawking perception organs import without an auxiliary runtime."""
from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PURE = {
    "hawking.perception.receipts": "hawking/perception/receipts.py",
    "hawking.perception.terminal": "hawking/perception/terminal.py",
    "hawking.perception.doctor": "hawking/perception/doctor.py",
    "hawking.perception.file_eye": "hawking/perception/file_eye.py",
}


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_pure_organs_import_only_hawking_or_stdlib_modules():
    allowed = set(sys.stdlib_module_names) | {"__future__", "hawking"}
    for module, relative_path in PURE.items():
        unexpected = _import_roots(REPO / relative_path) - allowed
        assert not unexpected, f"{module} imports non-local runtime(s): {sorted(unexpected)}"


def test_pure_organs_import_in_a_cold_interpreter():
    code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"for module in {tuple(PURE)!r}:\n"
        "    importlib.import_module(module)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-1500:]


def test_perception_organs_are_importable_in_this_process():
    for module in PURE:
        assert importlib.import_module(module)
