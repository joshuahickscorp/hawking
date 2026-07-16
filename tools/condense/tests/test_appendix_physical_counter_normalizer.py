from __future__ import annotations

import copy
import fcntl
import json
import os
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
TESTS = CONDENSE / "tests"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(TESTS))

import appendix_physical_counter_collector as collector  # noqa: E402
import appendix_physical_counter_normalizer as normalizer  # noqa: E402
import appendix_process_joule_collector as process_joule  # noqa: E402
import physical_counter_attestation  # noqa: E402
from test_appendix_physical_counter_collector import _bundle  # noqa: E402


def _capture(bundle: dict, kind: str, source: str, *, pid: int = 1234) -> dict:
    targets, errors = collector._phase_targets(bundle, kind)
    assert errors == []
    common = {
        "probe_pid": pid,
        "run_nonce": bundle["execution_authority"]["run_nonce"],
        "probe_argv_sha256": bundle["execution_authority"]["argv_sha256"],
        "capture_started_at_unix_ns": 9_000,
        "capture_ended_at_unix_ns": 21_000,
        "capture_started_at_continuous_ns": 29_000,
        "capture_ended_at_continuous_ns": 41_000,
    }
    records = []
    for ordinal, target in enumerate(targets):
        interval = target["interval"]
        if source == "energy":
            counters_before = {field: 0 for field in process_joule.RUSAGE_V6_U64_FIELDS}
            counters_before["ri_proc_start_abstime"] = 777
            counters_after = dict(counters_before)
            counters_after.update({
                "ri_user_time": 10 + ordinal,
                "ri_system_time": 5 + ordinal,
                "ri_instructions": 100 + ordinal,
                "ri_cycles": 200 + ordinal,
                "ri_pinstructions": 90 + ordinal,
                "ri_pcycles": 180 + ordinal,
                "ri_energy_nj": 250_000_000 + ordinal,
                "ri_penergy_nj": 200_000_000 + ordinal,
            })
            def snapshot(counters: dict, start_u: int, end_u: int, start_c: int, end_c: int) -> dict:
                return process_joule._stamp({
                    "schema": process_joule.SNAPSHOT_SCHEMA,
                    "backend_id": process_joule.BACKEND_ID,
                    "pid": pid,
                    "ri_uuid": "ab" * 16,
                    "read_started_at_unix_ns": start_u,
                    "read_ended_at_unix_ns": end_u,
                    "read_started_at_continuous_ns": start_c,
                    "read_ended_at_continuous_ns": end_c,
                    "counters": counters,
                }, "snapshot_sha256")
            before = snapshot(
                counters_before,
                interval["wall_started_unix_ns"] - 2, interval["wall_started_unix_ns"] - 1,
                interval["continuous_started_ns"] - 2,
                interval["continuous_started_ns"] - 1,
            )
            after = snapshot(
                counters_after,
                interval["wall_ended_unix_ns"] + 1, interval["wall_ended_unix_ns"] + 2,
                interval["continuous_ended_ns"] + 1,
                interval["continuous_ended_ns"] + 2,
            )
            records.append(process_joule.phase_record(
                before=before, after=after,
                phase_marker_sha256=target["marker"], interval_sha256=target["interval_sha256"],
                interval_started_at_unix_ns=interval["wall_started_unix_ns"],
                interval_ended_at_unix_ns=interval["wall_ended_unix_ns"],
                interval_started_at_continuous_ns=interval["continuous_started_ns"],
                interval_ended_at_continuous_ns=interval["continuous_ended_ns"],
            ))
            continue
        record = {
            "source_sample_id": f"{source}:{ordinal}",
            "phase_marker_sha256": target["marker"],
            "interval_sha256": target["interval_sha256"],
            "process_id": pid,
            "run_nonce": common["run_nonce"],
            "interval_started_at_unix_ns": interval["wall_started_unix_ns"],
            "interval_ended_at_unix_ns": interval["wall_ended_unix_ns"],
            "interval_started_at_continuous_ns": interval["continuous_started_ns"],
            "interval_ended_at_continuous_ns": interval["continuous_ended_ns"],
            "measurement_scope": (
                "exact-probe-process" if source == "energy"
                else "exact-probe-process+exact-metal-registry-id"
            ),
            "attribution": "direct-counter",
            "estimated": False,
            "apportioned": False,
        }
        record.update({
            "gpu_time_ns": 100 + ordinal,
            "physical_bytes": 1_000 + ordinal,
            "occupancy_percent": 50.0,
            "bandwidth_bytes_per_second": 2_000.0 + ordinal,
        })
        records.append(record)
    if source == "energy":
        return process_joule.build_probe_counter_block(
            records=records, library=process_joule.library_provenance(), probe_pid=pid,
        )
    value = {
        "schema": (
            normalizer.DIRECT_JOULE_CAPTURE_SCHEMA
            if source == "energy" else normalizer.METAL_CAPTURE_SCHEMA
        ),
        "backend_id": (
            normalizer.DIRECT_JOULE_BACKEND
            if source == "energy" else normalizer.METAL_BACKEND
        ),
        "metal_registry_id": None if source == "energy" else "metal-test-1",
        **common,
        "records": records,
    }
    return normalizer._stamp(value, "capture_sha256")


def _write(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_contract_pins_schema_and_rejects_powermetrics_energy_impact() -> None:
    value = normalizer.contract()
    assert value["contract_sha256"] == normalizer.CONTRACT_SHA256
    assert value["output_schema"] == collector.ATTRIBUTED_SCHEMA
    assert normalizer.UNSUPPORTED_POWERMETRICS_BACKEND in value["rejected_energy_sources"]
    assert value["energy_semantics"] == {
        "quantity": "energy", "unit": "joule", "scope": "exact-probe-process",
        "attribution": "direct-counter", "estimated": False, "apportioned": False,
    }
    assert normalizer.status()["production_process_joule_backend_admitted"] is False


def test_normalize_binds_pid_phase_captures_device_and_lease(tmp_path: pathlib.Path) -> None:
    bundle = _bundle("device")
    energy = _capture(bundle, "device", "energy")
    bundle["raw_probe"]["process_energy_counters"] = copy.deepcopy(energy)
    bundle["raw_bundle_sha256"] = collector.canonical_sha256({
        key: value for key, value in bundle.items() if key != "raw_bundle_sha256"
    })
    metal = _capture(bundle, "device", "metal")
    bundle_path, energy_path, metal_path = (
        tmp_path / "bundle.json", tmp_path / "energy.json", tmp_path / "metal.json"
    )
    _write(bundle_path, bundle)
    _write(energy_path, energy)
    _write(metal_path, metal)
    lease = (tmp_path / "lease").open("a+b")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = normalizer.normalize(
            kind="device", bundle_path=bundle_path, energy_path=energy_path,
            metal_path=metal_path, probe_pid=1234,
            run_nonce=bundle["execution_authority"]["run_nonce"],
            metal_registry_id="metal-test-1", lease_fd=lease.fileno(),
        )
        lease_stat = os.fstat(lease.fileno())
        errors = collector.validate_attributed_samples(
            bundle, manifest, kind="device", expected_probe_pid=1234,
            expected_capture_sha256s={
                "process_joule": physical_counter_attestation.file_identity(energy_path)["sha256"],
                "xctrace": physical_counter_attestation.file_identity(metal_path)["sha256"],
            },
            expected_metal_registry_id="metal-test-1",
            expected_lease={"device": lease_stat.st_dev, "inode": lease_stat.st_ino},
        )
    finally:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
    assert errors == []
    assert manifest["normalizer"]["contract_sha256"] == normalizer.CONTRACT_SHA256
    assert all(row["process_id"] == 1234 for row in manifest["samples"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(backend_id=normalizer.UNSUPPORTED_POWERMETRICS_BACKEND), "schema/backend"),
        (lambda value: value["records"][0].update(estimated=True), "record"),
        (lambda value: value["records"][0].update(apportioned=True), "record"),
        (lambda value: value["records"][0].update(process_id=9999), "record"),
        (lambda value: value["records"][0].update(unit="energy-impact"), "record"),
    ],
)
def test_energy_capture_adversarial_mutations_fail_closed(
    tmp_path: pathlib.Path, mutation, message: str,  # type: ignore[no-untyped-def]
) -> None:
    bundle = _bundle("device")
    energy = _capture(bundle, "device", "energy")
    mutation(energy)
    energy = process_joule._stamp(energy, "counters_sha256")
    bundle["raw_probe"]["process_energy_counters"] = copy.deepcopy(energy)
    bundle["raw_bundle_sha256"] = collector.canonical_sha256({
        key: value for key, value in bundle.items() if key != "raw_bundle_sha256"
    })
    metal = _capture(bundle, "device", "metal")
    bundle_path, energy_path, metal_path = (
        tmp_path / "bundle.json", tmp_path / "energy.json", tmp_path / "metal.json"
    )
    _write(bundle_path, bundle)
    _write(energy_path, energy)
    _write(metal_path, metal)
    lease = (tmp_path / "lease").open("a+b")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(normalizer.NormalizerError, match=message):
            normalizer.normalize(
                kind="device", bundle_path=bundle_path, energy_path=energy_path,
                metal_path=metal_path, probe_pid=1234,
                run_nonce=bundle["execution_authority"]["run_nonce"],
                metal_registry_id="metal-test-1", lease_fd=lease.fileno(),
            )
    finally:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()


def test_unlocked_or_missing_inherited_lease_fails_closed(tmp_path: pathlib.Path) -> None:
    bundle = _bundle("device")
    energy, metal = _capture(bundle, "device", "energy"), _capture(bundle, "device", "metal")
    bundle["raw_probe"]["process_energy_counters"] = copy.deepcopy(energy)
    bundle["raw_bundle_sha256"] = collector.canonical_sha256({
        key: value for key, value in bundle.items() if key != "raw_bundle_sha256"
    })
    paths = [tmp_path / name for name in ("bundle.json", "energy.json", "metal.json")]
    for path, value in zip(paths, (bundle, energy, metal)):
        _write(path, value)
    with pytest.raises(normalizer.NormalizerError, match="lease"):
        normalizer.normalize(
            kind="device", bundle_path=paths[0], energy_path=paths[1], metal_path=paths[2],
            probe_pid=1234, run_nonce=bundle["execution_authority"]["run_nonce"],
            metal_registry_id="metal-test-1", lease_fd=-1,
        )
