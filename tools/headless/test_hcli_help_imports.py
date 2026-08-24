"""`python3 -m hcli --help` must not import the runtime graph or heavy stacks."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEAVY = (
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
ALLOWED_HCLI = {"hcli", "hcli.cli"}


def _help_modules():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPROFILEIMPORTTIME", None)
    script = r"""
import runpy, sys
heavy_names = list(sys.argv[1:])
sys.argv = ["hcli", "--help"]
try:
    runpy.run_module("hcli", run_name="__main__")
except SystemExit as exc:
    code = 0 if exc.code in (0, None) else int(exc.code or 1)
else:
    code = 0
mods = set(sys.modules)
hcli = sorted(m for m in mods if m == "hcli" or m.startswith("hcli."))
print("CODE", code)
print("HCLI", ",".join(hcli))
print("HEAVY", ",".join(n for n in heavy_names if n in mods))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, *HEAVY],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = {}
    for line in proc.stdout.splitlines():
        if line.startswith(("CODE ", "HCLI ", "HEAVY ")):
            key, _, rest = line.partition(" ")
            parsed[key] = rest
    return parsed, proc.stdout


def test_help_exits_zero_without_controller():
    parsed, raw = _help_modules()
    assert parsed.get("CODE") == "0", raw
    hcli = [m for m in parsed.get("HCLI", "").split(",") if m]
    assert hcli == ["hcli", "hcli.cli"], raw
    extra = [m for m in hcli if m not in ALLOWED_HCLI]
    assert extra == [], raw


def test_help_does_not_import_mlx_torch_or_vmcp():
    parsed, raw = _help_modules()
    assert parsed.get("HEAVY") == "", raw
