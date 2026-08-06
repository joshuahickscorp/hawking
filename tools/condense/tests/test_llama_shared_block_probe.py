from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "llama_shared_block_probe.py"
SPEC = importlib.util.spec_from_file_location("llama_shared_block_probe", MODULE_PATH)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_shared_block_preserves_shape_and_charges_unique_bytes_once() -> None:
    model = probe.SharedResidual(hidden=6, width=3, steps=4)
    assert model(torch.zeros(2, 6)).shape == (2, 6)
    bill = probe.physical(hidden=4096, width=128, steps=4)
    assert bill["shared_recurrence_steps"] == 4
    assert bill["executed_macs_per_token"] == 4 * 2 * 4096 * 128
    assert bill["sequential_matvecs_per_token"] == 8
    assert bill["prospective_unique_active_fp16_bytes_per_token"] < 3_000_000
