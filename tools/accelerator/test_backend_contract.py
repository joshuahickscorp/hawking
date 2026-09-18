"""Backend contract pins.

Required coverage:
  - enumerate backends through the registry; >=4; each answers capability and cost
  - FPGA cannot emit HARDWARE_MEASURED by any code path
  - repatriation: a real receipt becomes a neutral law a second backend consumes
  - CUDA is named, not instantiated
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_cuda_runtime_all_exports_are_unique():
    from tools.accelerator import cuda_runtime

    names = cuda_runtime.__all__
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert duplicates == [], f"duplicate __all__ exports: {duplicates}"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend_contract as bc  # noqa: E402
import repatriation_audit as audit  # noqa: E402
from semantic_transport import COST_MODEL, HARDWARE_MEASURED  # noqa: E402

# P14_PCIE_IMPLEMENTATION_CONTRACT_V2: bounded qualification-support tranche.
# This test file is pinned to the contract phase; no release or hardware
# qualification is claimed by this annotation.
P14_PCIE_IMPLEMENTATION_CONTRACT_V2 = "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"


def test_p14_pcie_implementation_contract_v2_constant_is_pinned():
    """Bounded qualification-support tranche: the contract phase constant is
    pinned in both the runner and this test module. No release, phase, or
    hardware qualification is claimed by this test."""
    import accelerator_runner as runner
    assert runner.P14_PCIE_IMPLEMENTATION_CONTRACT_V2 == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"
    assert P14_PCIE_IMPLEMENTATION_CONTRACT_V2 == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"


def test_p14_pcie_contract_phase_receipt_fragment_boundary():
    """Bounded qualification-support tranche: the receipt fragment names the
    contract phase and its explicit boundary. No release, phase completion,
    or hardware qualification is claimed by this test."""
    import accelerator_runner as runner
    fragment = runner.p14_pcie_contract_phase_receipt_fragment()
    assert fragment["schema"] == "hawking.accelerator.p14_pcie_contract_phase.v1"
    assert fragment["contract_phase"] == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"
    assert fragment["boundary"] == "no-release/no-hardware-qualification"
    assert fragment["claims_phase_completion"] is False
    assert fragment["claims_release"] is False
    assert fragment["claims_hardware_qualification"] is False


def test_p14_pcie_contract_phase_evidence_fragment_is_bounded():
    """Bounded qualification-support tranche: the evidence fragment names the
    contract phase and its explicit boundary. No release, phase completion,
    or hardware qualification is claimed by this test."""
    import accelerator_runner as runner
    fragment = runner.p14_pcie_contract_phase_evidence_fragment()
    assert fragment["schema"] == "hawking.accelerator.p14_pcie_contract_evidence.v1"
    assert fragment["contract_phase"] == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"
    assert fragment["boundary"] == "no-release/no-hardware-qualification"
    assert fragment["claims_phase_completion"] is False
    assert fragment["claims_release"] is False
    assert fragment["claims_hardware_qualification"] is False


def test_p14_pcie_contract_phase_qualification_support_is_bounded():
    """Bounded qualification-support tranche: the qualification-support
    descriptor names the contract phase and its explicit boundary. No release,
    phase completion, or hardware qualification is claimed by this test."""
    import accelerator_runner as runner
    fragment = runner.p14_pcie_contract_phase_qualification_support()
    assert fragment["schema"] == "hawking.accelerator.p14_pcie_contract_qualification_support.v1"
    assert fragment["contract_phase"] == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"
    assert fragment["boundary"] == "no-release/no-hardware-qualification"
    assert fragment["claims_phase_completion"] is False
    assert fragment["claims_release"] is False
    assert fragment["claims_hardware_qualification"] is False


def test_p14_pcie_contract_phase_boundary_is_bounded():
    from tools.accelerator.accelerator_runner import (
        P14_PCIE_IMPLEMENTATION_CONTRACT_V2,
        p14_pcie_contract_phase_boundary,
    )

    fragment = p14_pcie_contract_phase_boundary()
    assert fragment["contract_phase"] == P14_PCIE_IMPLEMENTATION_CONTRACT_V2
    assert fragment["boundary"] == "no-release/no-hardware-qualification"
    assert fragment["claims_phase_completion"] is False
    assert fragment["claims_release"] is False
    assert fragment["claims_hardware_qualification"] is False


RECEIPT = "receipts/headless/ACCELERATOR_MACHINE_GENOME.json"


def _program():
    return bc.sample_elementwise_program()


def test_registry_enumerates_at_least_four_backends_that_answer_capability_and_cost():
    backends = bc.enumerate_backends()
    assert len(backends) >= 4
    ids = [b.backend_id for b in backends]
    assert ids == ["CPU", "METAL", "ANE", "FPGA-HWIR"]
    program = _program()
    for backend in backends:
        cap = backend.capabilities(program)
        cst = backend.cost(program)
        assert cap.backend_id == backend.backend_id
        assert cap.domain_kind is backend.domain_kind
        assert cap.evidence_tier in bc.EVIDENCE_TIERS
        assert cst.backend_id == backend.backend_id
        assert cst.cost.label == COST_MODEL
        assert cst.evidence_tier in bc.EVIDENCE_TIERS
        report = backend.validate_program(program)
        assert report.ok, report.errors
        lowered = backend.lower(program)
        assert lowered.backend_id == backend.backend_id
        executed = backend.execute(program)
        assert executed.backend_id == backend.backend_id
        assert executed.ok is True


def test_capability_and_cost_snapshot_calls_every_backend():
    snap = bc.capability_and_cost_snapshot(_program())
    assert snap["count"] >= 4
    assert snap["ids"] == ["CPU", "METAL", "ANE", "FPGA-HWIR"]
    for row in snap["backends"]:
        assert row["capability"]["backend_id"] == row["backend_id"]
        assert row["cost"]["cost"]["label"] == COST_MODEL


def test_fpga_cannot_emit_hardware_measured_by_any_code_path():
    fpga = bc.get_backend("FPGA-HWIR")
    program = _program()
    cap = fpga.capabilities(program)
    cst = fpga.cost(program)
    lowered = fpga.lower(program)
    executed = fpga.execute(program)
    law = bc.law_from_machine_genome()
    consumed = fpga.consume_law(law, program)

    payloads = [
        cap.to_dict(),
        cst.to_dict(),
        lowered.to_dict(),
        executed.to_dict(),
        consumed.to_dict(),
        fpga.execution_domain().to_dict(),
    ]
    tiers = set()
    for payload in payloads:
        tiers |= bc.collect_evidence_tiers(payload)
    assert HARDWARE_MEASURED not in tiers, tiers
    assert cap.physical is False
    assert cap.present is False
    assert executed.simulated is True
    assert cst.cost.label == COST_MODEL
    assert consumed.evidence_tier != HARDWARE_MEASURED
    assert consumed.cost.label == COST_MODEL
    with pytest.raises(bc.FpgaHardwareClaimError, match="HARDWARE_MEASURED"):
        bc.fpga_evidence_tier(HARDWARE_MEASURED)
    with pytest.raises(bc.FpgaHardwareClaimError):
        bc.BackendCost(
            backend_id="FPGA-HWIR",
            cost=cst.cost,
            evidence_tier=HARDWARE_MEASURED,
        )


def test_repatriation_metal_machine_genome_consumed_by_fpga():
    """Named receipt: receipts/headless/ACCELERATOR_MACHINE_GENOME.json

    Metal measured 589.73 GB/s (HARDWARE_MEASURED, INSTANCE). FPGA-HWIR
    consumes that number as uma_dram_bandwidth_gb_s and still reports
    COST_MODEL -- the consumer does not inherit the source's tier.
    """
    trace = bc.repatriate_machine_bandwidth(consumer_id="FPGA-HWIR")
    assert trace.source_receipt == RECEIPT
    assert trace.source_backend == "METAL"
    assert trace.consumer_backend == "FPGA-HWIR"
    assert trace.law.law_id == "AKB-MACHINE-BANDWIDTH"
    assert trace.law.evidence_tier == HARDWARE_MEASURED
    assert trace.law.hawking_primitive == "MemoryTierIdentity"
    bw = trace.consumer_cost.features["uma_dram_bandwidth_gb_s"]
    assert bw == pytest.approx(589.73)
    assert trace.consumer_cost.evidence_tier == COST_MODEL
    assert trace.consumer_cost.cost.label == COST_MODEL
    assert trace.consumer_cost.source_law_id == "AKB-MACHINE-BANDWIDTH"
    assert trace.binding["promotes_physical_law"] is False
    assert trace.binding["genericity"] == "CANDIDATE_UNVERIFIED"
    cpu = bc.get_backend("CPU")
    cpu_cost = cpu.consume_law(trace.law, _program())
    assert cpu_cost.features["uma_dram_bandwidth_gb_s"] == pytest.approx(589.73)
    assert cpu_cost.source_law_id == "AKB-MACHINE-BANDWIDTH"


def test_ane_reports_neural_engine_present_from_mlcomputeplan_profile():
    ane = bc.get_backend("ANE")
    cap = ane.capabilities(_program())
    assert cap.present is True
    assert cap.physical is True
    assert "NEURAL_ENGINE" in cap.devices
    lowered = ane.lower(_program())
    assert lowered.target == "mlcomputeplan"
    assert lowered.artifact["public_api_only"] is True
    assert "NEURAL_ENGINE" in lowered.artifact["supported"]
    executed = ane.execute(_program())
    assert executed.evidence_tier == bc.FUNCTIONAL_SIM
    assert executed.simulated is True


def test_cpu_execute_is_a_real_host_run():
    cpu = bc.get_backend("CPU")
    result = cpu.execute(_program())
    assert result.ok is True
    assert result.simulated is False
    assert result.evidence_tier == HARDWARE_MEASURED
    assert result.outputs["n"] == 4096


def test_cuda_is_named_not_registered():
    with pytest.raises(bc.BackendNotRegistered, match="CUDA"):
        bc.get_backend("CUDA")
    assert "CUDA" not in bc.list_backend_ids()


def test_unknown_primitive_is_refused_at_program_construction():
    with pytest.raises(bc.BackendContractError, match="atlas primitive"):
        bc.ProgramOp(
            op_id="bad",
            primitive="DenseMatmul",
            node_kind=bc.sample_elementwise_program().ops[0].node_kind,
        )


def test_metal_lower_calls_air_msl():
    metal = bc.get_backend("METAL")
    lowered = metal.lower(_program())
    assert lowered.target == "air-msl"
    assert "thread_position_in_grid" in lowered.artifact["msl"]
    cap = metal.capabilities(_program())
    assert cap.present is True
    assert cap.physical is True


def test_fpga_lower_is_hwir_static_only():
    fpga = bc.get_backend("FPGA-HWIR")
    lowered = fpga.lower(_program())
    assert lowered.target == "hwir"
    assert lowered.artifact["qualification"] == "STATIC_ONLY"
    assert lowered.artifact["device_budget"]["declared_not_measured"] is True
    assert HARDWARE_MEASURED not in bc.collect_evidence_tiers(lowered.to_dict())


def test_audit_backend_contract_check_calls_the_registry():
    row = audit.backend_contract_checks()
    assert row["check_id"] == "backend-contract"
    assert row["passed"] is True, row["observed"]
    observed = row["observed"]
    assert observed["count"] >= 4
    assert set(observed["backend_ids"]) >= {"CPU", "METAL", "ANE", "FPGA-HWIR"}
    assert observed["fpga_hardware_measured_refused"] is True
    assert observed["repatriation"]["source_receipt"] == RECEIPT
    assert observed["repatriation"]["law_id"] == "AKB-MACHINE-BANDWIDTH"
    assert observed["repatriation"]["feature_gb_s"] == pytest.approx(589.73)
    assert "HARDWARE_MEASURED" not in observed["fpga_evidence_tiers"]
def test_backend_contract_qualification_support_marker():
    """Bounded qualification-support check for the backend contract module.

    Asserts the marker surface is importable and stable. Does not assert
    hardware qualification, release readiness, or phase completion.
    """
    from tools.accelerator import backend_contract

    assert backend_contract.qualification_support_marker() == (
        "P13_NVIDIA_IMPLEMENTATION_CONTRACT_V2"
    )# P14_PCIE_IMPLEMENTATION_CONTRACT_V2 focused qualification-support test:
# asserts the runner exposes the bounded contract descriptor and that it
# does not claim hardware qualification or release completion.
def test_backend_contract_descriptor_boundary():
    from tools.accelerator.accelerator_runner import backend_contract_descriptor
    desc = backend_contract_descriptor()
    assert desc["contract"] == "P14_PCIE_IMPLEMENTATION_CONTRACT_V2"
    assert desc["hardware_qualified"] is False
    assert desc["release_complete"] is False


def test_validate_backend_contract_accepts_complete_contract():
    from tools.accelerator.backend_contract import validate_backend_contract

    assert validate_backend_contract(
        {"name": "cuda", "version": 2, "capabilities": ["matmul"]}
    )


def test_validate_backend_contract_rejects_missing_field():
    from tools.accelerator.backend_contract import validate_backend_contract

    assert not validate_backend_contract({"name": "cuda", "version": 2})


def test_validate_backend_contract_rejects_non_mapping():
    from tools.accelerator.backend_contract import validate_backend_contract

    assert not validate_backend_contract(None)


def test_backend_contract_does_not_claim_hardware_qualification():
    from tools.accelerator import backend_contract

    assert backend_contract.HARDWARE_QUALIFICATION_CLAIMED is False


def test_p13_contract_v2_qualification_support_marker():
    """Bounded P13 qualification-support check: contract module exposes a stable
    version marker so downstream NVIDIA implementation work can assert the
    accepted contract revision without re-deriving source mapping."""
    from tools.accelerator import backend_contract

    marker = getattr(backend_contract, "CONTRACT_REVISION", None)
    assert marker is not None, "backend_contract must expose CONTRACT_REVISION"
    assert isinstance(marker, str) and marker, "CONTRACT_REVISION must be a non-empty str"


def test_p13_nvidia_contract_v2_qualification_support_marker():
    """Bounded qualification-support marker for P13_NVIDIA_IMPLEMENTATION_CONTRACT_V2.

    Asserts the backend contract module exposes the P13 NVIDIA contract marker so the
    implementation contract can be qualified without hardware. This is a
    qualification-support check only; it makes no release or hardware claim.
    """
    import tools.accelerator.backend_contract as backend_contract

    assert backend_contract is not None
    assert backend_contract.p13_nvidia_contract_v2_marker() == "P13_NVIDIA_IMPLEMENTATION_CONTRACT_V2"


def test_describe_backend_capability_is_offline_and_deterministic():
    from tools.accelerator.backend_contract import describe_backend_capability

    first = describe_backend_capability("cuda", available=False, reason="no_device")
    second = describe_backend_capability("cuda", available=False, reason="no_device")

    assert first == second
    assert first["backend"] == "cuda"
    assert first["available"] is False
    assert first["reason"] == "no_device"
    assert first["probed"] is False


def test_describe_backend_capability_defaults_and_validation():
    from tools.accelerator.backend_contract import describe_backend_capability

    ok = describe_backend_capability("cpu", available=True)
    assert ok["available"] is True
    assert ok["reason"] == "available"
    assert ok["probed"] is False

    try:
        describe_backend_capability("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty backend_name")


def test_nvidia_backend_capability_flags_exposed():
    from tools.accelerator import backend_contract

    flags = backend_contract.NVIDIA_BACKEND_CAPABILITY_FLAGS
    assert isinstance(flags, (tuple, frozenset, set))
    assert "cuda" in flags


def test_no_hardware_qualification_boundary():
    from tools.accelerator import backend_contract
    assert getattr(backend_contract, "NO_HARDWARE_QUALIFICATION_BOUNDARY", False) is True


def test_nvidia_route_declaration_is_hardware_free():
    """Qualification-support: the declared NVIDIA route must be pure data and
    must not require hardware."""
    from tools.accelerator import backend_contract

    decl = backend_contract.nvidia_route_declaration()
    assert isinstance(decl, dict)
    for field in ("backend", "provider", "route", "requires_hardware"):
        assert field in decl, f"missing route field: {field}"
    assert decl["backend"] == "nvidia"
    assert decl["requires_hardware"] is False


def test_p13_no_hardware_qualification_boundary():
    import backend_contract

    assert backend_contract.NO_HARDWARE_QUALIFICATION is True
    assert backend_contract.NO_RELEASE is True# P13_NVIDIA_IMPLEMENTATION_CONTRACT_V2 qualification-support assertions.
# Bounded contract-surface check only: no hardware execution, no runtime
# behavior change, no release or hardware-qualification claim.
def test_nvidia_backend_identifier_is_stable():
    from tools.accelerator import backend_contract

    assert backend_contract.NVIDIA_BACKEND_ID == "nvidia"
    assert backend_contract.nvidia_backend_identifier() == "nvidia"


def test_nvidia_backend_capabilities_declared():
    from tools.accelerator import backend_contract

    caps = backend_contract.nvidia_backend_capabilities()
    assert "cuda_runtime" in caps
    assert "backend_contract_v2" in caps