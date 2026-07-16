from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "doctor_v5_physical_ab_controller.py"
SPEC = importlib.util.spec_from_file_location("doctor_v5_physical_ab_controller", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(label: str) -> dict:
    return {
        "path": f"/tmp/doctor-v5-physical/{label}",
        "sha256": MODULE.canonical_sha256(label),
        "bytes": 1,
    }


def _counter(facet: str, nonce: str) -> dict:
    return MODULE._stamp({
        "schema": MODULE.COUNTER_SCHEMA,
        "facet": facet,
        "run_nonce": nonce,
        "energy_j": 1.0,
        "cpu_time_ns": 100,
        "read_bytes": 10,
        "write_bytes": 10,
        "peak_rss_bytes": 1000,
        "sample_count": 5,
        "directly_measured": True,
        "estimated": False,
    }, "counter_payload_sha256")


def _run(facet: str, role: str, repeat: int, boundary_sha: str) -> dict:
    nonce = MODULE.canonical_sha256(f"{facet}:{role}:{repeat}:nonce")
    shared_input = _artifact(f"{facet}:input:{repeat}")
    shared_science = _artifact(f"{facet}:science:{repeat}")
    output = _artifact(f"{facet}:exact-output:{repeat}")
    invocation = _artifact(f"{facet}:{role}:invocation:{repeat}")
    environment = _artifact(f"{facet}:{role}:environment:{repeat}")
    counter = _counter(facet, nonce)
    attestation = _artifact(f"{facet}:{role}:attestation:{repeat}")
    program = _artifact(f"{facet}:{role}:program")
    value = {
        "role": role,
        "repeat": repeat,
        "run_nonce": nonce,
        "boundary_attestation_sha256": boundary_sha,
        "program": program,
        "benchmark_runner": _artifact(f"{facet}:benchmark-runner"),
        "invocation_manifest": invocation,
        "environment_manifest": environment,
        "input_manifest": shared_input,
        "output_manifest": output,
        "scientific_receipt": shared_science,
        "counter_attestation": attestation,
        "owner_inventory_before": _artifact(f"{facet}:{role}:owners-before:{repeat}"),
        "owner_inventory_after": _artifact(f"{facet}:{role}:owners-after:{repeat}"),
        "invocation_sha256": invocation["sha256"],
        "environment_sha256": environment["sha256"],
        "started_at_unix_ns": 1000 + repeat * 10 + (2 if role == "candidate" else 0),
        "ended_at_unix_ns": 1005 + repeat * 10 + (2 if role == "candidate" else 0),
        "exit_code": 0,
        "skipped": False,
        "owner_count_before": 0,
        "owner_count_after": 0,
        "thermal_before": "nominal",
        "thermal_after": "nominal",
        "memory_pressure_before": "normal",
        "memory_pressure_after": "normal",
        "swap_before_mb": 100.0,
        "swap_after_mb": 100.0,
        "disk_free_before_bytes": MODULE.MIN_TOTAL_DISK_ADMISSION_BYTES,
        "disk_free_after_bytes": MODULE.MIN_TOTAL_DISK_ADMISSION_BYTES,
        "counter_payload": counter,
        "counter_attestation_binding_sha256": MODULE.canonical_sha256({
            "counter_payload_sha256": counter["counter_payload_sha256"],
            "counter_attestation_sha256": attestation["sha256"],
            "run_nonce": nonce,
            "program_sha256": program["sha256"],
            "input_manifest_sha256": shared_input["sha256"],
            "output_manifest_sha256": output["sha256"],
        }),
        "exact_output_sha256": output["sha256"],
    }
    return MODULE._stamp(value, "run_sha256")


def _disk_receipt(plan: dict, boundary_sha: str) -> dict:
    order = []
    for repeat in range(MODULE.MIN_REPEATS):
        order.extend(
            [f"candidate:{repeat}", f"baseline:{repeat}"]
            if repeat % 2 else [f"baseline:{repeat}", f"candidate:{repeat}"]
        )
    runs = [
        _run("disk_lifecycle", role, repeat, boundary_sha)
        for repeat in range(MODULE.MIN_REPEATS)
        for role in ("baseline", "candidate")
    ]
    by_label = {f"{row['role']}:{row['repeat']}": row for row in runs}
    for rank, label in enumerate(order):
        row = by_label[label]
        row["started_at_unix_ns"] = 1000 + rank * 10
        row["ended_at_unix_ns"] = row["started_at_unix_ns"] + 5
        replacement = MODULE._stamp(row, "run_sha256")
        runs[runs.index(row)] = replacement
    return MODULE._stamp({
        "schema": MODULE.FACET_SCHEMA,
        "facet": "disk_lifecycle",
        "status": "pass",
        "scope": "physical-owner-free-real-artifact",
        "structural_only": False,
        "plan_sha256": plan["plan_sha256"],
        "source_manifest_sha256": plan["source_manifest"]["manifest_sha256"],
        "boundary_attestation_sha256": boundary_sha,
        "paired_protocol": {
            "warmups_per_arm": 1,
            "repeats_per_arm": MODULE.MIN_REPEATS,
            "randomized_interleaved": True,
            "order": order,
            "order_sha256": MODULE.canonical_sha256(order),
        },
        "runs": runs,
        "payload": {
            "minimum_disk_free_bytes": MODULE.MIN_TOTAL_DISK_ADMISSION_BYTES,
            "exclusive_partial_creation": True,
            "atomic_finalization": True,
            "source_identity_before_after_exact": True,
            "parent_source_deleted": False,
            "gc_only_ephemeral_refcount_zero": True,
            "crash_resume_and_rollback_exact": True,
        },
        "runtime_defaults_changed": False,
        "source_files_deleted": False,
        "completed_evidence_mutated": False,
        "component_speedups_multiplied": False,
    }, "receipt_sha256")


def _observer(ready: bool) -> dict:
    return MODULE._stamp({
        "schema": "hawking.doctor_v5_post_120b_observer_state.v1",
        "final_interpretation_ready": ready,
    }, "state_sha256")


def test_plan_is_exact_and_controller_has_no_execution_surface() -> None:
    plan = MODULE.build_plan()
    assert plan == MODULE.build_plan()
    assert plan["counts"] == {
        "thread_profile_cells": 800,
        "thread_profile_selections": 200,
        "block_parallel_cells": 200,
        "facets": 10,
    }
    assert plan["thread_candidates"] == [8, 12, 16, 20]
    assert plan["execution_capability"] is False
    assert plan["claim_contract"]["component_speedups_multiplied"] is False
    executor = plan["executor_manifest"]
    assert executor["trusted_executor_available"] is True
    assert executor["runner_source"]["path"] == "tools/condense/doctor_v5_physical_ab_executor.py"
    assert executor["commands_executable"] is False
    assert executor["commands_executable_after_all_admission_gates"] is True
    assert len(executor["commands"]) == 10
    assert all(row["implemented"] is True for row in executor["commands"])
    assert all(row["currently_admitted"] is False for row in executor["commands"])
    assert MODULE.MIN_DISK_RESERVE_BYTES == 150_000_000_000
    assert MODULE.MIN_SCRATCH_RESERVE_BYTES == 64_000_000_000
    assert MODULE.MIN_TOTAL_DISK_ADMISSION_BYTES == 214_000_000_000
    dry = MODULE.build_dry_run()
    assert dry["would_execute"] is False and dry["commands"] == []
    assert dry["heavy_lease_acquired"] is False
    assert len(dry["future_command_contracts"]) == 10


def test_structural_scaffolding_scores_exactly_zero_physical() -> None:
    card = MODULE.build_scorecard(None, verify_files=False)
    assert card["physical_rating"] == "0/10"
    assert card["physical_score"] == 0
    assert card["structural_scaffolding_physical_score"] == 0
    assert all(not row["green"] and row["physical_points"] == 0 for row in card["facets"])
    assert all(row["structural_points"] == 0 for row in card["facets"])
    assert card["eta_promotion_authorized"] is False


def test_closed_release_boundary_does_not_evaluate_caller_packet() -> None:
    status = MODULE.build_status(
        observer=_observer(False), active_owners=[{"pid": 7}],
        packet={"forged": True}, verify_files=False,
    )
    assert status["release_boundary_open"] is False
    assert status["physical_packet_evaluated"] is False
    assert status["scorecard"]["physical_rating"] == "0/10"
    assert any("not loaded" in row for row in status["blockers"])
    assert any("program adapter is absent" in row for row in status["blockers"])


def test_opaque_physical_facet_cannot_pass_without_executor_evidence_and_adapter() -> None:
    plan = MODULE.build_plan()
    boundary_sha = "a" * 64
    receipt = _disk_receipt(plan, boundary_sha)
    initial = MODULE._facet_errors(
        receipt, facet="disk_lifecycle", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("executor arm evidence is absent" in row for row in initial)

    forged = copy.deepcopy(receipt)
    forged["runs"][0]["executor_arm_evidence"] = {
        "schema": "hawking.doctor_v5_physical_ab_arm_sidecars.v1",
        "synthetic": False,
        "direct_counter_validated": True,
        "sidecars_sha256": "a" * 64,
    }
    forged["runs"][0]["executor_arm_evidence_artifact"] = _artifact("forged-sidecars")
    forged["runs"][0] = MODULE._stamp(forged["runs"][0], "run_sha256")
    forged = MODULE._stamp(forged, "receipt_sha256")
    forged_errors = MODULE._facet_errors(
        forged, facet="disk_lifecycle", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("arm evidence schema is invalid" in row for row in forged_errors)
    assert any("source-reviewed concrete" in row for row in forged_errors)

    damaged = copy.deepcopy(receipt)
    damaged["runs"][0]["thermal_after"] = "fair"
    damaged["runs"][0] = MODULE._stamp(damaged["runs"][0], "run_sha256")
    damaged = MODULE._stamp(damaged, "receipt_sha256")
    errors = MODULE._facet_errors(
        damaged, facet="disk_lifecycle", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("thermal/pressure" in row for row in errors)


def test_counter_estimates_and_pair_output_mismatch_fail_closed() -> None:
    plan = MODULE.build_plan()
    boundary_sha = "b" * 64
    receipt = _disk_receipt(plan, boundary_sha)
    counter = receipt["runs"][0]["counter_payload"]
    counter["directly_measured"] = False
    counter["estimated"] = True
    receipt["runs"][0]["counter_payload"] = MODULE._stamp(counter, "counter_payload_sha256")
    receipt["runs"][0] = MODULE._stamp(receipt["runs"][0], "run_sha256")
    receipt["runs"][1]["exact_output_sha256"] = "f" * 64
    receipt["runs"][1] = MODULE._stamp(receipt["runs"][1], "run_sha256")
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    errors = MODULE._facet_errors(
        receipt, facet="disk_lifecycle", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("direct, not estimated" in row for row in errors)
    assert any("input/output/scientific parity differs" in row for row in errors)


def test_thread_profile_payload_requires_every_exact_candidate_without_fallback() -> None:
    plan = MODULE.build_plan()
    measurements = [
        {
            "cell_id": row["id"], "exact_output": True,
            "wall_seconds": 1.0, "receipt_sha256": MODULE.canonical_sha256(row["id"]),
        }
        for row in plan["thread_profile_cells"]
    ]
    selections = [
        {
            "tier": tier, "rate": rate, "branch": branch,
            "threads": 8, "nearest_fallback": False,
        }
        for tier in MODULE.TIERS for rate in MODULE.RATES for branch in MODULE.BRANCHES
    ]
    payload = {"measurements": measurements, "selections": selections}
    assert MODULE._domain_payload_errors("thread_profiles", payload, plan=plan) == []
    payload["measurements"].pop()
    payload["selections"][0]["nearest_fallback"] = True
    errors = MODULE._domain_payload_errors("thread_profiles", payload, plan=plan)
    assert any("exact 8/12/16/20" in row for row in errors)
    assert any("fallback" in row for row in errors)


def test_full_stack_rejects_multiplied_component_speedups() -> None:
    plan = MODULE.build_plan()
    payload = {
        "paired_speedups": [2.0] * MODULE.MIN_REPEATS,
        "conservative_speedup": 2.0,
        "all_required_components_exercised": True,
        "all_outputs_exact": True,
        "all_scientific_receipts_exact": True,
        "component_speedups_multiplied": True,
        "eta_segment": "sub-120b-doctor",
    }
    errors = MODULE._domain_payload_errors("full_stack_parity_ab", payload, plan=plan)
    assert any("full-stack paired" in row for row in errors)


def test_full_stack_speedups_must_derive_from_bound_run_intervals() -> None:
    plan = MODULE.build_plan()
    boundary_sha = "d" * 64
    receipt = _disk_receipt(plan, boundary_sha)
    receipt["facet"] = "full_stack_parity_ab"
    for index, run in enumerate(receipt["runs"]):
        counter = run["counter_payload"]
        counter["facet"] = "full_stack_parity_ab"
        run["counter_payload"] = MODULE._stamp(counter, "counter_payload_sha256")
        run["counter_attestation_binding_sha256"] = MODULE.canonical_sha256({
            "counter_payload_sha256": run["counter_payload"]["counter_payload_sha256"],
            "counter_attestation_sha256": run["counter_attestation"]["sha256"],
            "run_nonce": run["run_nonce"],
            "program_sha256": run["program"]["sha256"],
            "input_manifest_sha256": run["input_manifest"]["sha256"],
            "output_manifest_sha256": run["output_manifest"]["sha256"],
        })
        receipt["runs"][index] = MODULE._stamp(run, "run_sha256")
    receipt["payload"] = {
        "paired_speedups": [1.0] * MODULE.MIN_REPEATS,
        "conservative_speedup": 1.0,
        "all_required_components_exercised": True,
        "all_outputs_exact": True,
        "all_scientific_receipts_exact": True,
        "component_speedups_multiplied": False,
        "eta_segment": "sub-120b-doctor",
    }
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    initial = MODULE._facet_errors(
        receipt, facet="full_stack_parity_ab", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("executor arm evidence is absent" in row for row in initial)
    forged = copy.deepcopy(receipt)
    forged["payload"]["paired_speedups"] = [2.0] * MODULE.MIN_REPEATS
    forged["payload"]["conservative_speedup"] = 2.0
    forged = MODULE._stamp(forged, "receipt_sha256")
    errors = MODULE._facet_errors(
        forged, facet="full_stack_parity_ab", plan=plan,
        boundary_sha=boundary_sha, verify_files=False,
    )
    assert any("not derived from bound run intervals" in row for row in errors)


def test_post120_qualification_rejects_sub120_receipt_reuse() -> None:
    value = MODULE._stamp({
        "schema": MODULE.POST120_SCHEMA,
        "post120_handoff_sha256": "a" * 64,
        "status": "physical-exact-qualified",
        "gptoss_coverage": {
            "source_units": 615, "rates": 10, "branches": 4,
            "cells": 40, "jobs": 24600, "skips": 0, "exact": True,
        },
        "segment_facet_receipts": {
            facet: _artifact(f"post120:{facet}") for facet in MODULE.FACETS[:-1]
        },
        "higher_tier_receipts": {
            name: _artifact(f"higher:{name}")
            for name in ("DeepSeek-V4-Flash", "Kimi-K2.6", "DeepSeek-V4-Pro")
        },
        "sub120_receipts_reused": True,
        "runtime_defaults_changed": False,
    }, "qualification_sha256")
    errors = MODULE._post120_errors(
        value, handoff={"handoff_sha256": "a" * 64}, verify_files=False,
    )
    assert any("reused sub120" in row for row in errors)
    assert any("exact segment scope was not file-verified" in row for row in errors)


def test_owner_and_resource_receipts_reject_opaque_or_weakened_samples() -> None:
    plan = MODULE.build_plan()
    owner = MODULE._stamp({
        "schema": "hawking.doctor_v5_physical_ab_owner_snapshot.v1",
        "plan_sha256": plan["plan_sha256"], "contract_sha256": "c" * 64,
        "facet": "disk_lifecycle", "phase": "measured", "role": "baseline",
        "repeat": 0, "run_nonce": "d" * 64, "position": "before",
        "observed_at_unix_ns": 1, "ps_program": _artifact("ps"),
        "shared_heavy_lease": {
            "path": str(MODULE.ROOT / "reports" / "cron" / "studio_heavy.lock"),
            "st_dev": 1, "st_ino": 2, "inherited_descriptor": True, "held": True,
        },
        "owners": [{"pid": 9}], "owner_count": 1,
        "probe_ok": True, "synthetic": False,
    }, "snapshot_sha256")
    errors = MODULE._owner_snapshot_errors(
        owner, plan=plan, contract_sha256="c" * 64, facet="disk_lifecycle",
        role="baseline", repeat=0, nonce="d" * 64, position="before",
        verify_files=False,
    )
    assert any("opaque/synthetic/nonzero" in row for row in errors)

    limits = {
        "minimum_disk_free_bytes": MODULE.MIN_DISK_RESERVE_BYTES,
        "maximum_swap_used_bytes": 10,
    }
    resource = MODULE._stamp({
        "schema": "hawking.doctor_v5_physical_ab_resource_guard.v1",
        "plan_sha256": plan["plan_sha256"], "contract_sha256": "c" * 64,
        "facet": "disk_lifecycle", "phase": "measured", "role": "baseline",
        "repeat": 0, "run_nonce": "d" * 64, "position": "before",
        "observed_at_unix_ns": 1, "limits": limits,
        "snapshot": {
            "probe_ok": True, "pressure_level": 1, "thermal_state": 0,
            "power_source": "AC Power", "swap_used_bytes": 11,
            "disk_free_bytes": MODULE.MIN_TOTAL_DISK_ADMISSION_BYTES - 1,
        },
        "health_errors": [], "healthy": True, "synthetic": False,
    }, "receipt_sha256")
    errors = MODULE._resource_guard_errors(
        resource, plan=plan, contract_sha256="c" * 64, facet="disk_lifecycle",
        role="baseline", repeat=0, nonce="d" * 64, position="before",
    )
    assert any("disk+scratch" in row for row in errors)
    assert any("swap is invalid" in row for row in errors)
