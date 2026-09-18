"""Focused regression for the P14_PCIE_FPGA qualification-support gate.

Exercises the executable behavior added to
``tools/accelerator/accelerator_runner.py``: the qualification-support
fragment must embed the evaluated contract-phase gate and evidence fragments,
and the gate must be open only while every claim flag is False and the
boundary string is pinned.

Boundary: this is a no-release/no-hardware-qualification check. It asserts
nothing about phase completion, release, or hardware qualification.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.accelerator.accelerator_runner import (
    P14_PCIE_CONTRACT_PHASE_DESCRIPTOR,
    p14_pcie_contract_phase_gate_fragment,
    p14_pcie_contract_phase_qualification_support,
)


def test_qualification_support_embeds_open_gate_and_evidence() -> None:
    support = p14_pcie_contract_phase_qualification_support()
    assert support["gate"]["gate_open"] is True
    assert support["gate"]["gate_reason"] == "all_claims_false_boundary_pinned"
    assert support["evidence"]["boundary"] == "no-release/no-hardware-qualification"
    assert support["evidence"]["claims_release"] is False
    assert support["evidence"]["claims_hardware_qualification"] is False


def test_gate_closes_when_a_claim_flag_is_not_false() -> None:
    original = P14_PCIE_CONTRACT_PHASE_DESCRIPTOR["claims_release"]
    try:
        P14_PCIE_CONTRACT_PHASE_DESCRIPTOR["claims_release"] = True
        gate = p14_pcie_contract_phase_gate_fragment()
        assert gate["gate_open"] is False
        assert gate["gate_reason"] == "failed:claims_release"
    finally:
        P14_PCIE_CONTRACT_PHASE_DESCRIPTOR["claims_release"] = original
    assert p14_pcie_contract_phase_gate_fragment()["gate_open"] is True