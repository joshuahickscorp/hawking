from __future__ import annotations

import copy
import fcntl
import os
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import doctor_v5_physical_ab_controller as controller  # noqa: E402
import doctor_v5_physical_ab_executor as executor  # noqa: E402
import doctor_v5_physical_counter_barrier as barrier  # noqa: E402


def _write(path: pathlib.Path, raw: bytes, *, executable: bool = False) -> pathlib.Path:
    path.write_bytes(raw)
    if executable:
        path.chmod(0o755)
    return path


def _artifact(path: pathlib.Path, *, executable: bool = False) -> dict:
    return executor._artifact_identity(path, executable=executable, allow_empty=True)


def _valid_seed(repeats: int = 5) -> str:
    for index in range(10_000):
        seed = executor.canonical_sha256(f"seed:{index}")
        try:
            executor.build_randomized_order(seed, repeats)
        except executor.ContractError:
            continue
        return seed
    raise AssertionError("could not find counterbalanced seed")


def _argv(role: str, program_sha256: str) -> dict:
    return executor._stamp({
        "schema": executor.ARGV_SCHEMA,
        "role": role,
        "program_sha256": program_sha256,
        "program_abi": "hawking.doctor_v5_physical_ab_program.v1",
        "abi_reviewed": True,
        "direct_exec": True,
        "writes_confined_to_dynamic_paths": True,
        "argv": [
            "--inputs", "{INPUT_MANIFEST_PATH}",
            "--output", "{OUTPUT_PATH}",
            "--science", "{SCIENTIFIC_RECEIPT_PATH}",
            "--payload", "{FACET_PAYLOAD_PATH}",
            "--nonce", "{RUN_NONCE}",
        ],
        "placeholders": sorted(executor.ARM_PLACEHOLDERS),
        "environment": {"LANG": "C"},
        "cwd": str(ROOT),
        "stdin": "devnull",
        "shell": False,
        "mutates_live_doctor": False,
        "mutates_runtime_defaults": False,
        "deletes_sources": False,
    }, "manifest_sha256")


def _launch_contract(plan: dict, tmp_path: pathlib.Path) -> dict:
    baseline = _write(tmp_path / "baseline", b"#!/bin/false\n# baseline\n", executable=True)
    candidate = _write(tmp_path / "candidate", b"#!/bin/false\n# candidate\n", executable=True)
    baseline_argv = tmp_path / "baseline_argv.json"
    candidate_argv = tmp_path / "candidate_argv.json"
    baseline_argv.write_text(__import__("json").dumps(
        _argv("baseline", _artifact(baseline, executable=True)["sha256"]), sort_keys=True,
    ))
    candidate_argv.write_text(__import__("json").dumps(
        _argv("candidate", _artifact(candidate, executable=True)["sha256"]), sort_keys=True,
    ))
    input_manifest = _write(tmp_path / "inputs.json", b"{}")
    collector_authority = _write(tmp_path / "collector_authority.json", b"{}")
    seed = _valid_seed()
    order = executor.build_randomized_order(seed, 5)
    return executor._stamp({
        "schema": executor.LAUNCH_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "source_manifest_sha256": plan["source_manifest"]["manifest_sha256"],
        "executor_source_sha256": plan["executor_manifest"]["runner_source"]["sha256"],
        "facet": "disk_lifecycle",
        "baseline_program": _artifact(baseline, executable=True),
        "baseline_argv_manifest": _artifact(baseline_argv),
        "candidate_program": _artifact(candidate, executable=True),
        "candidate_argv_manifest": _artifact(candidate_argv),
        "input_manifest": _artifact(input_manifest),
        "collector_authority": _artifact(collector_authority),
        "pairing": {
            "warmups_per_arm": 1,
            "repeats_per_arm": 5,
            "random_seed_sha256": seed,
            "randomized_interleaved": True,
            "order": order,
            "order_sha256": executor.canonical_sha256(order),
        },
        "run_limits": {
            "timeout_seconds": 60,
            "collector_ready_timeout_seconds": 10,
            "max_stdout_bytes": 1024,
            "max_stderr_bytes": 1024,
            "max_output_bytes": 1024,
            "max_scientific_receipt_bytes": 1024,
            "max_facet_payload_bytes": 1024,
            "maximum_swap_used_bytes": 8_000_000_000,
            "maximum_swap_growth_bytes": 1_000_000_000,
            "minimum_disk_free_bytes": controller.MIN_TOTAL_DISK_ADMISSION_BYTES,
        },
        "output_policy": {
            "root": str(executor.EXECUTION_ROOT),
            "exclusive_create": True,
            "immutable_sidecars": True,
            "atomic_final_receipt": True,
            "source_deletion_permitted": False,
        },
        "mutation_policy": {
            "live_doctor_mutation": False,
            "completed_evidence_mutation": False,
            "runtime_default_mutation": False,
            "result_overwrite": False,
            "source_deletion": False,
        },
    }, "contract_sha256")


def test_status_and_dry_run_are_nonexecuting_and_report_real_gaps() -> None:
    observer = controller._stamp({
        "schema": "hawking.doctor_v5_post_120b_observer_state.v1",
        "final_interpretation_ready": True,
    }, "state_sha256")
    status = executor.build_status(
        observer=observer, owners=[],
        collector_status={
            "execution_ready": False,
            "blockers": ["direct process-joule backend absent"],
        },
        env={},
    )
    assert status["execution_ready"] is False
    assert status["shared_heavy_lease_opened"] is False
    assert status["input_artifacts_hashed"] is False
    assert status["benchmark_spawned"] is False
    assert set(status["missing_program_adapter_facets"]) == set(controller.FACETS)
    assert any("process-joule" in row for row in status["blockers"])
    assert any("program adapter is absent" in row for row in status["blockers"])
    dry = executor.build_dry_run(facet="disk_lifecycle")
    assert dry["would_execute"] is False and dry["commands"] == []
    assert dry["would_hash_inputs"] is False and dry["would_create_output"] is False


def test_execute_exits_75_before_loading_or_creating_when_admission_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(executor, "build_status", lambda: {
        "blockers": ["collector authority is closed"]
    })
    monkeypatch.setattr(
        executor, "_load_bundle_from_args",
        lambda _args: (_ for _ in ()).throw(AssertionError("bundle was loaded")),
    )
    output = tmp_path / "must-not-exist.json"
    rc = executor.main([
        "execute", "--facet", "disk_lifecycle", "--plan", "/missing/plan",
        "--launch-contract", "/missing/launch", "--release-authority", "/missing/release",
        "--baseline-program", "/missing/base", "--baseline-argv-manifest", "/missing/base-argv",
        "--candidate-program", "/missing/candidate", "--candidate-argv-manifest", "/missing/candidate-argv",
        "--input-manifest", "/missing/inputs", "--execution-scope", "/missing/scope",
        "--collector-authority", "/missing/collector",
        "--output", str(output),
    ])
    assert rc == 75
    assert not output.exists()


def test_execute_blockers_are_facet_specific_but_global_gates_remain() -> None:
    status = {
        "blockers": [
            "a concrete hash-bound launch bundle is required",
            "thread_profiles: reviewed concrete baseline/candidate program adapter is absent",
            "disk_lifecycle: reviewed concrete baseline/candidate program adapter is absent",
            "direct process-joule backend absent",
        ]
    }
    blockers = executor._hard_blockers_for_facet(status, "disk_lifecycle")
    assert "thread_profiles: reviewed concrete baseline/candidate program adapter is absent" \
        not in blockers
    assert "disk_lifecycle: reviewed concrete baseline/candidate program adapter is absent" \
        in blockers
    assert "direct process-joule backend absent" in blockers


def test_randomized_pairs_are_exact_deterministic_and_counterbalanced() -> None:
    seed = _valid_seed(7)
    order = executor.build_randomized_order(seed, 7)
    assert order == executor.build_randomized_order(seed, 7)
    assert len(order) == 14 and len(set(order)) == 14
    for index in range(0, len(order), 2):
        left_role, left_repeat = order[index].split(":")
        right_role, right_repeat = order[index + 1].split(":")
        assert left_repeat == right_repeat
        assert {left_role, right_role} == {"baseline", "candidate"}
    assert {order[index].split(":")[0] for index in range(0, len(order), 2)} \
        == {"baseline", "candidate"}
    with pytest.raises(executor.ContractError):
        executor.build_randomized_order(seed, 4)


def test_arm_argv_rejects_shell_loader_injection_and_placeholder_tampering() -> None:
    value = _argv("baseline", "a" * 64)
    assert executor.validate_argv_manifest(
        value, role="baseline", program_sha256="a" * 64,
    ) == []
    shell = copy.deepcopy(value)
    shell["shell"] = True
    shell = executor._stamp(shell, "manifest_sha256")
    assert any("no shell" in row for row in executor.validate_argv_manifest(
        shell, role="baseline", program_sha256="a" * 64,
    ))
    loader = copy.deepcopy(value)
    loader["environment"]["DYLD_INSERT_LIBRARIES"] = "/tmp/inject.dylib"
    loader = executor._stamp(loader, "manifest_sha256")
    assert any("protected environment" in row for row in executor.validate_argv_manifest(
        loader, role="baseline", program_sha256="a" * 64,
    ))
    missing = copy.deepcopy(value)
    missing["argv"].remove("{RUN_NONCE}")
    missing = executor._stamp(missing, "manifest_sha256")
    assert any("placeholder coverage" in row for row in executor.validate_argv_manifest(
        missing, role="baseline", program_sha256="a" * 64,
    ))


def test_launch_contract_is_blocked_without_source_reviewed_facet_adapter(
    tmp_path: pathlib.Path,
) -> None:
    plan = controller.build_plan()
    contract = _launch_contract(plan, tmp_path)
    errors = executor.validate_launch_contract(
        contract, plan=plan, facet="disk_lifecycle", verify_files=True,
    )
    assert any("no reviewed concrete physical program adapter" in row for row in errors)
    escaped = copy.deepcopy(contract)
    escaped["output_policy"]["root"] = str(ROOT / "reports" / "condense")
    escaped = executor._stamp(escaped, "contract_sha256")
    errors = executor.validate_launch_contract(
        escaped, plan=plan, facet="disk_lifecycle", verify_files=False,
    )
    assert any("confine" in row for row in errors)


def test_collector_argv_is_exact_generic_barrier_and_no_shell() -> None:
    program = executor._artifact_identity(
        pathlib.Path(barrier.__file__), executable=True,
    )
    value = executor._stamp({
        "schema": executor.COLLECTOR_ARGV_SCHEMA,
        "program_sha256": program["sha256"],
        "collector_abi": "hawking.doctor_v5_physical_counter_barrier.v1",
        "abi_reviewed": True,
        "argv": executor.EXACT_COUNTER_BARRIER_ARGV,
        "placeholders": sorted(executor.COLLECTOR_PLACEHOLDERS),
        "environment": {"LANG": "C"},
        "cwd": str(ROOT), "stdin": "devnull", "shell": False,
        "inherits_shared_heavy_lease": True,
        "opens_heavy_lease": False,
        "launches_benchmark_with_shell": False,
    }, "manifest_sha256")
    assert executor.validate_collector_argv(value, program_sha256=program["sha256"]) == []
    reordered = copy.deepcopy(value)
    reordered["argv"] = list(reversed(reordered["argv"]))
    reordered = executor._stamp(reordered, "manifest_sha256")
    assert any("exact reviewed" in row for row in executor.validate_collector_argv(
        reordered, program_sha256=program["sha256"],
    ))


def test_inherited_lease_is_proved_but_never_acquired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    lock_path = _write(tmp_path / "heavy.lock", b"")
    monkeypatch.setattr(executor, "HEAVY_LOCK", lock_path)
    handle = lock_path.open("r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(handle.fileno())
        authority = {
            "shared_heavy_lease": {
                "lock_file": _artifact(lock_path),
                "st_dev": int(info.st_dev), "st_ino": int(info.st_ino),
            }
        }
        fd, errors = executor._validate_inherited_lease(
            authority, env={executor.HEAVY_LEASE_FD_ENV: str(handle.fileno())},
        )
        assert fd == handle.fileno() and errors == []
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        fd, errors = executor._validate_inherited_lease(
            authority, env={executor.HEAVY_LEASE_FD_ENV: str(handle.fileno())},
        )
        assert fd is None
        assert any("not already held" in row for row in errors)
    finally:
        handle.close()


def test_resource_admission_combines_disk_and_scratch_and_bounds_swap_growth() -> None:
    limits = {
        "maximum_swap_used_bytes": 8_000_000_000,
        "minimum_disk_free_bytes": controller.MIN_TOTAL_DISK_ADMISSION_BYTES,
    }
    green = {
        "probe_ok": True, "errors": [], "pressure_level": 1,
        "thermal_state": 0, "power_source": "AC Power",
        "swap_used_bytes": 2_000_000_000,
        "disk_free_bytes": controller.MIN_TOTAL_DISK_ADMISSION_BYTES,
    }
    assert executor._resource_errors(green, limits=limits) == []
    red_disk = dict(green)
    red_disk["disk_free_bytes"] = controller.MIN_TOTAL_DISK_ADMISSION_BYTES - 1
    assert any("reserve" in row for row in executor._resource_errors(red_disk, limits=limits))
    after = dict(green)
    after["swap_used_bytes"] += 1_000_000_001
    assert any("growth" in row for row in executor._swap_growth_errors(
        green, after, maximum_growth_bytes=1_000_000_000,
    ))


def test_barrier_collect_exits_75_before_touching_paths(tmp_path: pathlib.Path) -> None:
    paths = [tmp_path / name for name in (
        "request", "ready", "started", "stop", "counter", "attestation",
    )]
    rc = barrier.main([
        "collect", "--request", str(paths[0]), "--ready", str(paths[1]),
        "--arm-started", str(paths[2]), "--stop", str(paths[3]),
        "--counter-output", str(paths[4]), "--counter-attestation", str(paths[5]),
    ])
    assert rc == 75
    assert not any(path.exists() for path in paths)
    status = barrier.build_status(
        env={}, system="Darwin", euid=0, powermetrics_path="/usr/bin/powermetrics",
    )
    assert status["execution_ready"] is False
    assert status["direct_process_energy_joules_available"] is False
    assert any("energy-impact" in row for row in status["blockers"])


def test_counter_validator_rejects_estimates_and_unbound_raw_capture(tmp_path: pathlib.Path) -> None:
    nonce = "1" * 64
    raw = _write(tmp_path / "capture.raw", b"direct samples")
    counter = barrier._stamp({
        "schema": barrier.COUNTER_SCHEMA, "facet": "disk_lifecycle",
        "run_nonce": nonce, "energy_j": 1.0, "cpu_time_ns": 100,
        "read_bytes": 10, "write_bytes": 10, "peak_rss_bytes": 1000,
        "sample_count": 5, "directly_measured": True, "estimated": False,
    }, "counter_payload_sha256")
    interval = {
        "started_at_unix_ns": 100, "ended_at_unix_ns": 200,
        "started_at_continuous_ns": 300, "ended_at_continuous_ns": 400,
    }
    attestation = barrier._stamp({
        "schema": barrier.ATTESTATION_SCHEMA, "plan_sha256": "2" * 64,
        "contract_sha256": "3" * 64, "facet": "disk_lifecycle",
        "phase": "measured", "role": "baseline", "repeat": 0,
        "run_nonce": nonce, "collector_authority_sha256": "4" * 64,
        "collector_program_sha256": "5" * 64, "benchmark_program_sha256": "6" * 64,
        "invocation_sha256": "7" * 64, "execution_interval": interval,
        "capture_interval": {
            "started_at_unix_ns": 90, "ended_at_unix_ns": 210,
            "started_at_continuous_ns": 290, "ended_at_continuous_ns": 410,
        },
        "directly_measured": True, "estimated": False,
        "domains": list(barrier.COUNTER_FIELDS),
        "counter_payload_sha256": counter["counter_payload_sha256"],
        "output_sha256": "8" * 64, "scientific_sha256": "9" * 64,
        "stdout_sha256": "a" * 64, "stderr_sha256": "b" * 64,
        "raw_captures": [_artifact(raw)],
    }, "attestation_sha256")
    assert barrier.validate_backend_result(
        counter, attestation, facet="disk_lifecycle", role="baseline", repeat=0,
        run_nonce=nonce, program_sha256="6" * 64,
        started_at_unix_ns=100, ended_at_unix_ns=200,
        started_at_continuous_ns=300, ended_at_continuous_ns=400,
    ) == []
    estimated = copy.deepcopy(attestation)
    estimated["estimated"] = True
    estimated = barrier._stamp(estimated, "attestation_sha256")
    assert any("direct domains" in row for row in barrier.validate_backend_result(
        counter, estimated, facet="disk_lifecycle", role="baseline", repeat=0,
        run_nonce=nonce, program_sha256="6" * 64,
        started_at_unix_ns=100, ended_at_unix_ns=200,
        started_at_continuous_ns=300, ended_at_continuous_ns=400,
    ))
    raw.write_bytes(b"changed")
    assert any("raw capture" in row for row in barrier.validate_backend_result(
        counter, attestation, facet="disk_lifecycle", role="baseline", repeat=0,
        run_nonce=nonce, program_sha256="6" * 64,
        started_at_unix_ns=100, ended_at_unix_ns=200,
        started_at_continuous_ns=300, ended_at_continuous_ns=400,
    ))


def test_scientific_receipt_is_structured_bound_and_preserves_negative_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = {
        "adapter_id": "disk-lifecycle-v1",
        "baseline_program_sha256": "1" * 64,
        "baseline_argv_manifest_sha256": "2" * 64,
        "candidate_program_sha256": "3" * 64,
        "candidate_argv_manifest_sha256": "4" * 64,
        "scientific_receipt_schema": executor.SCIENTIFIC_RECEIPT_SCHEMA,
        "scientific_validator": (
            "doctor_v5_physical_ab_executor.validate_scientific_receipt.v1"
        ),
    }
    monkeypatch.setitem(executor.PROGRAM_ADAPTER_REGISTRY, "disk_lifecycle", adapter)
    bundle = {
        "plan": {"plan_sha256": "a" * 64},
        "contract": {"facet": "disk_lifecycle"},
        "program_adapter": adapter,
    }
    inputs = {"sha256": "b" * 64}
    output = {"sha256": "c" * 64}
    payload = {"sha256": "d" * 64}
    value = executor._stamp({
        "schema": executor.SCIENTIFIC_RECEIPT_SCHEMA,
        "plan_sha256": "a" * 64, "facet": "disk_lifecycle",
        "adapter_id": "disk-lifecycle-v1",
        "input_manifest_sha256": "b" * 64,
        "output_sha256": "c" * 64,
        "facet_payload_sha256": "d" * 64,
        "exact_output": True, "skipped": False,
        "negative_evidence_preserved": True, "synthetic": False,
    }, "receipt_sha256")
    assert executor.validate_scientific_receipt(
        value, bundle=bundle, input_manifest=inputs, output=output,
        facet_payload=payload,
    ) == []
    damaged = copy.deepcopy(value)
    damaged["negative_evidence_preserved"] = False
    damaged = executor._stamp(damaged, "receipt_sha256")
    assert any("negative evidence" in row for row in executor.validate_scientific_receipt(
        damaged, bundle=bundle, input_manifest=inputs, output=output,
        facet_payload=payload,
    ))
