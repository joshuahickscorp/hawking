from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_postrun.py"
SPEC = importlib.util.spec_from_file_location("appendix_postrun", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_postrun_plan_is_deterministic_and_dependency_closed() -> None:
    plan = MODULE.build_plan()
    assert plan == MODULE.build_plan()
    ids = {stage["id"] for stage in plan["stages"]}
    assert all(set(stage["depends_on"]) <= ids for stage in plan["stages"])
    assert plan["counts"]["tq_device_cells"]["deferred"] == 496
    assert plan["counts"]["mapped_existing_gates"] == 9
    assert plan["counts"]["missing_artifact_adapters"] == 0
    assert plan["execution_supported"] is False


def test_vendor_gates_cannot_promote_a_runtime_default() -> None:
    plan = MODULE.build_plan()
    vendor = [gate for gate in plan["gates"] if gate["tier"] == "vendor_microbench"]
    assert vendor
    assert all(not gate["strict_device_receipt_emitted"] for gate in vendor)
    assert all(gate["source_surface"] for gate in vendor)
    final = next(gate for gate in plan["gates"] if gate["id"] == "hawking_core_runtime_matrix")
    assert final["command"] is not None
    assert "--cell-id" in final["command"]
    assert "--residual-artifact" in final["command"]
    assert "--residual-tensor" in final["command"]
    assert final["finalize_command"] is not None
    assert final["strict_device_receipt_emitted"] is True
    spec = next(gate for gate in plan["gates"] if gate["id"] == "tq_native_batched_verifier")
    assert spec["command"] is not None
    assert spec["finalize_command"] is not None
    assert spec["strict_device_receipt_emitted"] is True


def test_release_probe_build_is_explicit_and_still_non_executing() -> None:
    plan = MODULE.build_plan()
    gate = next(
        gate for gate in plan["gates"]
        if gate["id"] == "build_hawking_tq_release_probes"
    )
    assert gate["tier"] == "static_compile"
    assert gate["command"] == [
        "cargo", "build", "--release", "-p", "hawking", "--features", "tq",
        "--bin", "hawking-tq-device-probe", "--bin", "hawking-tq-spec-probe",
    ]
    assert gate["requires_exclusive_heavy_lease"] is False
    assert gate["strict_device_receipt_emitted"] is False
    stage = next(stage for stage in plan["stages"] if stage["id"] == "A1_compile_surfaces")
    assert gate["id"] in stage["gates"]


def test_status_is_fail_closed_on_owner_or_missing_compiler() -> None:
    plan = MODULE.build_plan()
    assert not MODULE.status(
        plan, active_owners=[{"pid": 7}], metal_compiler=None,
        probe_available=True, spec_probe_available=True, platform_name="Darwin"
    )["device_environment_ready"]
    assert not MODULE.status(
        plan, active_owners=[], metal_compiler=None,
        probe_available=False, spec_probe_available=True, platform_name="Darwin"
    )["device_environment_ready"]
    assert MODULE.status(
        plan, active_owners=[], metal_compiler=None,
        probe_available=True, spec_probe_available=True, platform_name="Darwin"
    )["device_environment_ready"]
    assert not MODULE.status(
        plan, active_owners=[], metal_compiler=None,
        probe_available=True, spec_probe_available=True, platform_name="Darwin"
    )["execution_ready"]
