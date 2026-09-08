"""A function named metal_status allocated on the CPU and returned "mlx cpu usable".

    def metal_status() -> tuple[bool, str]:
        mx.set_default_device(mx.cpu)
        x = mx.zeros((2, 2))
        mx.eval(x)
        return True, "mlx cpu usable"

It would return True on a machine with no GPU at all, because it never touches one.

A Grok lane found this the expensive way. It saw `mx.default_device()` report
`Device(gpu, 0)`, which names a default and does not load a device, and then died on
`mx.eval(mx.ones((8,8)))` with "[metal::load_device] No Metal device available. This
typically occurs in headless, sandboxed, or virtualized macOS sessions." Its own scar
line names this function: both checks "would have licensed a fake 'Metal is up' and
then died inside mlx_lm".

The G004 capability arm is gated on exactly this answer, so a check that cannot say no
is worse than no check: it converts a resource refusal into a mid-run crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "future"))

# The interpreter that has pytest here (3.14) has no mlx; the one with mlx
# (/usr/local/bin/python3.12) has no pytest. So the import is per-test: the SOURCE test
# below runs EVERYWHERE and is the one that catches the defect, while the two runtime
# tests skip rather than pretend. A module-level importorskip took the source test with
# them and left this file reporting "1 skipped" -- a check that skips on the machine it
# was written for is not a check.


def test_it_evaluates_on_the_GPU_DEVICE_not_the_cpu(monkeypatch):
    real = pytest.importorskip("mlx.core", reason="mlx lives in /usr/local/bin/python3.12")
    """The invariant: if the GPU cannot evaluate, the answer is False."""
    import lowrank_nr

    def boom(*a, **k):
        raise RuntimeError("[metal::load_device] No Metal device available")

    monkeypatch.setattr(real, "eval", boom)
    ok, why = lowrank_nr.metal_status()
    assert ok is False, f"metal_status said {ok!r} while every eval raised: {why}"
    assert "metal" in why.lower(), why


def test_the_reason_is_not_a_cpu_claim():
    """"mlx cpu usable" is a true sentence about the wrong device."""
    pytest.importorskip("mlx.core", reason="mlx lives in /usr/local/bin/python3.12")
    import lowrank_nr
    ok, why = lowrank_nr.metal_status()
    assert "cpu usable" not in why.lower(), (
        f"metal_status still reports about the CPU: {why!r}")


def test_the_source_no_longer_pins_the_default_device_to_cpu():
    """Setting the default to cpu inside a Metal check is the defect itself."""
    src = (Path(__file__).resolve().parents[2] / "tools" / "future" / "lowrank_nr.py").read_text()
    body = src.split("def metal_status")[1].split("\ndef ")[0]
    assert "mx.set_default_device(mx.cpu)" not in body, body
    assert "mx.gpu" in body, f"the check never names the GPU device: {body}"
