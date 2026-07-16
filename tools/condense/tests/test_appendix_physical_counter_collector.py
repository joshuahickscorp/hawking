from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import appendix_physical_counter_collector as collector  # noqa: E402
import appendix_physical_counter_normalizer as trusted_normalizer  # noqa: E402
import appendix_physical_evidence_gate as evidence_gate  # noqa: E402
import physical_counter_attestation  # noqa: E402


def _bundle(kind: str) -> dict:
    nonce = "1" * 64
    intervals = []
    pairs = []
    shape = [(None, index) for index in range(2)] if kind == "device" else [
        (batch, 0) for batch in range(1, 9)
    ]
    for sequence, (batch, iteration) in enumerate(shape):
        role = "candidate" if kind == "device" else "verifier"
        baseline = {
            "schema": "hawking.physical_phase_interval.v1",
            "run_nonce": nonce,
            "sequence": sequence * 2,
            "phase": "trial",
            "role": "baseline",
            "batch": batch,
            "iteration": iteration,
            "wall_started_unix_ns": 11_000 + sequence * 20,
            "wall_ended_unix_ns": 11_005 + sequence * 20,
            "continuous_started_ns": 31_000 + sequence * 20,
            "continuous_ended_ns": 31_005 + sequence * 20,
            "elapsed_ns": 5,
        }
        baseline["interval_sha256"] = collector.canonical_sha256(baseline)
        intervals.append(baseline)
        interval = {
            "schema": "hawking.physical_phase_interval.v1",
            "run_nonce": nonce,
            "sequence": sequence * 2 + 1,
            "phase": "trial",
            "role": role,
            "batch": batch,
            "iteration": iteration,
            "wall_started_unix_ns": 11_010 + sequence * 20,
            "wall_ended_unix_ns": 11_015 + sequence * 20,
            "continuous_started_ns": 31_010 + sequence * 20,
            "continuous_ended_ns": 31_015 + sequence * 20,
            "elapsed_ns": 5,
        }
        interval["interval_sha256"] = collector.canonical_sha256(interval)
        intervals.append(interval)
        pair = {
            "schema": "hawking.physical_phase_pair.v1",
            "run_nonce": nonce,
            "phase": "trial",
            "batch": batch,
            "iteration": iteration,
            "first_role": "baseline",
            "baseline_interval_sha256": baseline["interval_sha256"],
            ("candidate_interval_sha256" if kind == "device" else "verifier_interval_sha256"): interval["interval_sha256"],
        }
        pair["phase_marker_sha256"] = collector.canonical_sha256(pair)
        pairs.append(pair)
    phase = {
        "schema": collector.PHASE_SCHEMA,
        "run_nonce": nonce,
        "clock_source": "mach_absolute_time_plus_system_time_unix_epoch",
        "probe_started_wall_unix_ns": 10_500,
        "probe_ended_wall_unix_ns": 12_000,
        "probe_started_continuous_ns": 30_500,
        "probe_ended_continuous_ns": 32_000,
        "intervals": intervals,
        "pairs": pairs,
    }
    phase["phase_markers_sha256"] = collector.canonical_sha256(phase)
    raw = {
        "artifact": {"sha256": "a" * 64},
        "runtime_path": "stored",
        "phase_markers": phase,
    }
    if kind == "device":
        raw.update({
            "tensor": {"name": "ffn_down"},
            "benchmark": {
                "trials": 2,
                "trial_phase_marker_sha256": [
                    pair["phase_marker_sha256"] for pair in pairs
                ],
            },
        })
    else:
        raw["measurement_protocol"] = {
            "phase_markers_sha256": phase["phase_markers_sha256"],
            "batches": [
                {
                    "b": batch,
                    "repeats": [
                        {"phase_marker_sha256": pair["phase_marker_sha256"]}
                        for pair in pairs if pair["batch"] == batch
                    ],
                }
                for batch in range(1, 9)
            ],
        }
    bundle = {
        "raw_probe": raw,
        "execution_authority": {
            "run_nonce": nonce,
            "argv_sha256": "9" * 64,
            "started_at_unix_ns": 10_000,
            "ended_at_unix_ns": 20_000,
            "started_at_continuous_ns": 30_000,
            "ended_at_continuous_ns": 40_000,
        },
    }
    bundle["raw_bundle_sha256"] = collector.canonical_sha256(bundle)
    return bundle


def _manifest(bundle: dict, kind: str) -> dict:
    required = collector.DEVICE_DOMAINS if kind == "device" else collector.SPEC_DOMAINS
    collectors = []
    for collector_id in ("process_joule", "xctrace"):
        collectors.append({
            "id": collector_id,
            "backend_id": (
                trusted_normalizer.DIRECT_JOULE_BACKEND
                if collector_id == "process_joule" else trusted_normalizer.METAL_BACKEND
            ),
            "raw_capture_sha256": ("b" if collector_id == "process_joule" else "c") * 64,
            "available": True,
            "privilege_verified": True,
            "process_attributed": True,
            "phase_attributed": True,
            "directly_measured": True,
            "estimated": False,
            "apportioned": False,
            "domains": [d for d in required if collector.DOMAIN_COLLECTOR[d] == collector_id],
            "capture_started_at_unix_ns": 9_000,
            "capture_ended_at_unix_ns": 21_000,
            "capture_started_at_continuous_ns": 29_000,
            "capture_ended_at_continuous_ns": 41_000,
        })
    targets, errors = collector._phase_targets(bundle, kind)
    assert errors == []
    samples = []
    for ordinal, target in enumerate(targets):
        samples.append({
            "ordinal": ordinal,
            "batch": target["batch"],
            "repeat": target["iteration"] if kind == "spec" else None,
            "phase_marker_sha256": target["marker"],
            "interval_sha256": target["interval_sha256"],
            "interval_started_at_unix_ns": target["interval"]["wall_started_unix_ns"],
            "interval_ended_at_unix_ns": target["interval"]["wall_ended_unix_ns"],
            "interval_started_at_continuous_ns": target["interval"]["continuous_started_ns"],
            "interval_ended_at_continuous_ns": target["interval"]["continuous_ended_ns"],
            "run_nonce": bundle["execution_authority"]["run_nonce"],
            "process_id": 1234,
            "energy_j": 0.1 + ordinal / 1000,
            "gpu_time_ns": 10 + ordinal,
            "physical_bytes": 100 + ordinal,
            "occupancy_percent": 50.0 if kind == "device" else None,
            "bandwidth_bytes_per_second": 1000.0 if kind == "device" else None,
            "source_sample_ids": {domain: [f"{domain}:{ordinal}"] for domain in required},
            "energy_provenance": {
                "backend_id": trusted_normalizer.DIRECT_JOULE_BACKEND,
                "quantity": "energy",
                "unit": "joule",
                "scope": "exact-probe-process",
                "attribution": "direct-counter",
                "estimated": False,
                "apportioned": False,
                "source_process_id": 1234,
            },
        })
    manifest = {
        "schema": collector.ATTRIBUTED_SCHEMA,
        "kind": kind,
        "normalizer": {
            "schema": trusted_normalizer.SCHEMA,
            "contract_sha256": trusted_normalizer.CONTRACT_SHA256,
            "binary": physical_counter_attestation.file_identity(
                pathlib.Path(trusted_normalizer.__file__),
            ),
        },
        "lease": {"inherited": True, "device": 1, "inode": 2},
        "probe_pid": 1234,
        "probe_argv_sha256": bundle["execution_authority"]["argv_sha256"],
        "metal_registry_id": "metal-test-1",
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": bundle["raw_probe"]["artifact"]["sha256"],
        "runtime_path": bundle["raw_probe"]["runtime_path"],
        "run_nonce": bundle["execution_authority"]["run_nonce"],
        "phase_markers_sha256": bundle["raw_probe"]["phase_markers"]["phase_markers_sha256"],
        "collectors": collectors,
        "samples": samples,
    }
    return collector._stamp(manifest, "manifest_sha256")


def test_config_and_cli_surface_are_default_off() -> None:
    config = collector.build_config()
    assert config["default_off"] is True
    assert config["collection_cli_exposed"] is False
    assert config["authoritative_outputs"]["legacy_v1_projection_emitted"] is False
    trace = next(row for row in config["collectors"] if row["id"] == "xctrace")
    assert trace["required_template"] == "Metal System Trace"
    assert trace["command-line-tools-stub_is_capable"] is False
    assert collector.dry_run("device")["would_execute"] is False


def test_status_refuses_missing_privilege_receipts_and_never_opens_lease() -> None:
    value = collector.status(
        euid=501,
        binary_paths={
            "process_joule": str(pathlib.Path(trusted_normalizer.process_joule.__file__)),
            "xctrace": "/usr/bin/xctrace",
        },
        final_ready=True,
        active_heavy_owner_count=0,
    )
    assert value["execution_ready"] is False
    assert value["shared_heavy_lease_opened"] is False
    assert any("powermetrics energy-impact proxy" in blocker for blocker in value["blockers"])
    assert any("attribution receipt" in blocker for blocker in value["blockers"])


def test_status_does_not_confuse_xctrace_stub_presence_with_capability() -> None:
    value = collector.status(
        euid=0,
        binary_paths={
            "process_joule": str(pathlib.Path(trusted_normalizer.process_joule.__file__)),
            "xctrace": "/usr/bin/xctrace",
        },
        capability_checks={
            "process_joule": (True, "libproc present"),
            "xctrace": (False, "full Xcode is not selected"),
        },
        capability_receipts={"process_joule": True, "xctrace": True},
        final_ready=True,
        active_heavy_owner_count=0,
    )
    trace = next(row for row in value["collectors"] if row["id"] == "xctrace")
    assert trace["available"] is True
    assert trace["runtime_capable"] is False
    assert value["execution_ready"] is False
    assert any("full Xcode is not selected" in blocker for blocker in value["blockers"])


def test_status_defaults_to_live_observer_and_owner_inventory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = tmp_path / "observer.json"
    observer.write_text(json.dumps({"final_interpretation_ready": True}), encoding="utf-8")
    monkeypatch.setattr(collector, "OBSERVER_PATH", observer)
    monkeypatch.setattr(
        collector.spec_reentry_scaffold,
        "active_heavy_owners",
        lambda: [{"pid": 42, "command": "fixture-owner"}],
    )
    value = collector.status(
        euid=0,
        binary_paths={
            "process_joule": str(pathlib.Path(trusted_normalizer.process_joule.__file__)),
            "xctrace": "/Applications/Xcode.app/xctrace",
        },
        capability_checks={"process_joule": True, "xctrace": True},
        capability_receipts={"process_joule": True, "xctrace": True},
    )
    assert value["final_interpretation_ready"] is True
    assert value["active_heavy_owner_count"] == 1
    assert "heavy owners remain" in value["blockers"]


def test_device_normalizer_emits_authoritative_gate_compatible_v2() -> None:
    bundle = _bundle("device")
    payload = collector.normalize_v2(bundle, _manifest(bundle, "device"), kind="device")
    assert payload["schema"] == evidence_gate.DEVICE_COUNTER_SCHEMA
    assert evidence_gate._device_counter_errors(payload, raw=bundle["raw_probe"]) == []


def test_spec_normalizer_emits_b1_b8_gate_compatible_v2() -> None:
    bundle = _bundle("spec")
    payload = collector.normalize_v2(bundle, _manifest(bundle, "spec"), kind="spec")
    assert payload["schema"] == evidence_gate.SPEC_COUNTER_SCHEMA
    assert evidence_gate._spec_counter_errors(payload, raw=bundle["raw_probe"], repeats=1) == []


def test_normalizer_rejects_unavailable_unprivileged_unattributable_source() -> None:
    bundle = _bundle("device")
    manifest = _manifest(bundle, "device")
    manifest["collectors"][0]["available"] = False
    manifest["collectors"][0]["privilege_verified"] = False
    manifest["collectors"][0]["phase_attributed"] = False
    manifest = collector._stamp(manifest, "manifest_sha256")
    errors = collector.validate_attributed_samples(bundle, manifest, kind="device")
    assert any("unavailable" in error for error in errors)
    assert any("unprivileged" in error for error in errors)
    assert any("unattributable" in error for error in errors)
    with pytest.raises(ValueError):
        collector.normalize_v2(bundle, manifest, kind="device")


def test_normalizer_rejects_incomplete_clock_window_and_wrong_phase_id() -> None:
    bundle = _bundle("device")
    manifest = _manifest(bundle, "device")
    manifest["collectors"][1]["capture_started_at_continuous_ns"] = 30_001
    manifest["samples"][0]["phase_marker_sha256"] = "f" * 64
    manifest = collector._stamp(manifest, "manifest_sha256")
    errors = collector.validate_attributed_samples(bundle, manifest, kind="device")
    assert any("wall+continuous" in error for error in errors)
    assert any("exact phase marker" in error for error in errors)


def test_normalizer_rejects_proxy_energy_pid_spoof_and_source_reuse() -> None:
    bundle = _bundle("device")
    manifest = _manifest(bundle, "device")
    manifest["samples"][0]["energy_provenance"]["attribution"] = "time-apportioned"
    manifest["samples"][0]["energy_provenance"]["estimated"] = True
    manifest["samples"][0]["process_id"] = 9999
    manifest["samples"][1]["source_sample_ids"]["energy"] = list(
        manifest["samples"][0]["source_sample_ids"]["energy"]
    )
    manifest = collector._stamp(manifest, "manifest_sha256")
    errors = collector.validate_attributed_samples(
        bundle, manifest, kind="device", expected_probe_pid=1234,
    )
    assert any("not a direct, non-apportioned process-joule" in error for error in errors)
    assert any("lacks process attribution" in error for error in errors)
    assert any("reuses energy source record IDs" in error for error in errors)


def test_normalizer_binds_sealed_capture_device_and_inherited_lease() -> None:
    bundle = _bundle("device")
    manifest = _manifest(bundle, "device")
    errors = collector.validate_attributed_samples(
        bundle, manifest, kind="device", expected_probe_pid=1234,
        expected_capture_sha256s={"process_joule": "d" * 64, "xctrace": "c" * 64},
        expected_metal_registry_id="metal-other",
        expected_lease={"device": 1, "inode": 3},
    )
    assert any("raw capture differs" in error for error in errors)
    assert any("registry ID differs" in error for error in errors)
    assert any("lease differs" in error for error in errors)
