from __future__ import annotations

import os
from pathlib import Path
import plistlib
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/condense"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools import kimi_k26_campaign as campaign  # noqa: E402
import kimi_k26_doctor_prime as doctor  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402


FLOOR = 5 * 1024**3


def test_all_python_enforcement_agrees_on_exact_floor() -> None:
    assert campaign.MIN_RESERVE == FLOOR
    assert campaign.TARGET_RESERVE == FLOOR
    assert doctor.disk_floor_bytes() == FLOOR
    assert manager.MIN_FREE == FLOOR


def test_conflicting_environment_is_rejected() -> None:
    environment = {**os.environ, "KIMI_K26_DISK_FLOOR_BYTES": str(FLOOR + 1)}
    result = subprocess.run(
        [sys.executable, "-c", "from tools import kimi_k26_campaign"],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "must equal exactly 5368709120" in result.stderr


def test_doctor_rejects_conflicting_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KIMI_K26_DISK_FLOOR_BYTES", str(FLOOR - 1))
    with pytest.raises(RuntimeError, match="must equal exactly"):
        doctor.disk_floor_bytes()


def test_floor_boundary_and_atomic_write_boundary() -> None:
    assert campaign.disk_floor_green(FLOOR - 1) is False
    assert campaign.disk_floor_green(FLOOR) is False
    assert campaign.disk_floor_green(FLOOR + 1) is True
    assert campaign.can_start_atomic_write(FLOOR + 1024, 1024) is False
    assert campaign.can_start_atomic_write(FLOOR + 1025, 1024) is True


def test_source_gate_does_not_reintroduce_two_shard_floor() -> None:
    source = (ROOT / "tools/kimi_k26_campaign.py").read_text(encoding="utf-8")
    assert "max(MIN_RESERVE, 2 * manifest" not in source


def test_static_launchd_declares_exact_floor() -> None:
    path = ROOT / "deploy/launchd/com.hawking.kimi-k26-doctor-prime.plist"
    with path.open("rb") as handle:
        value = plistlib.load(handle)
    assert int(value["EnvironmentVariables"]["KIMI_K26_DISK_FLOOR_BYTES"]) == FLOOR
