from __future__ import annotations

import copy
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import appendix_contract  # noqa: E402
import physical_counter_attestation as contract  # noqa: E402
from physical_counter_fixtures import (  # noqa: E402
    attestation,
    execution_authority,
    phase_markers,
)


def _bundle(tmp_path: pathlib.Path) -> dict:
    raw = {"artifact": {"sha256": "d" * 64}}
    authority = execution_authority(tmp_path, raw)
    base = {
        "raw_probe": raw,
        "raw_probe_sha256": appendix_contract.canonical_sha256(raw),
        "execution_authority": authority,
    }
    base["raw_bundle_sha256"] = appendix_contract.canonical_sha256(base)
    return base


def _validate(tmp_path: pathlib.Path) -> tuple[dict, dict, dict, list[str]]:
    bundle = _bundle(tmp_path)
    payload = {"energy_j_total": 1.25, "gpu_time_ns": 4000, "physical_bytes": 8192}
    evidence = attestation(
        tmp_path, bundle=bundle, counter_payload=payload,
        domains=("energy", "gpu_time", "physical_bytes"), sample_count=20,
    )
    errors = contract.validate(
        evidence,
        raw_bundle_sha256=bundle["raw_bundle_sha256"],
        artifact_sha256=bundle["raw_probe"]["artifact"]["sha256"],
        execution_authority=bundle["execution_authority"],
        counter_payload=payload,
        required_domains=("energy", "gpu_time", "physical_bytes"),
        minimum_samples=20,
    )
    return bundle, payload, evidence, errors


def test_file_bound_counter_attestation_passes(tmp_path: pathlib.Path) -> None:
    _, _, _, errors = _validate(tmp_path)
    assert errors == []


def test_fabricated_values_and_source_labels_cannot_retain_attestation(
    tmp_path: pathlib.Path,
) -> None:
    bundle, payload, evidence, _ = _validate(tmp_path)
    forged = copy.deepcopy(payload)
    forged["energy_j_total"] = 1
    errors = contract.validate(
        evidence,
        raw_bundle_sha256=bundle["raw_bundle_sha256"],
        artifact_sha256=bundle["raw_probe"]["artifact"]["sha256"],
        execution_authority=bundle["execution_authority"],
        counter_payload=forged,
        required_domains=("energy", "gpu_time", "physical_bytes"),
        minimum_samples=20,
    )
    assert "counter attestation is not bound to normalized counter values" in errors


def test_capture_and_probe_binary_tampering_are_detected(tmp_path: pathlib.Path) -> None:
    bundle, payload, evidence, _ = _validate(tmp_path)
    pathlib.Path(evidence["domains"][0]["raw_capture"]["path"]).write_bytes(b"tampered")
    errors = contract.validate(
        evidence,
        raw_bundle_sha256=bundle["raw_bundle_sha256"],
        artifact_sha256=bundle["raw_probe"]["artifact"]["sha256"],
        execution_authority=bundle["execution_authority"],
        counter_payload=payload,
        required_domains=("energy", "gpu_time", "physical_bytes"),
        minimum_samples=20,
    )
    assert any("raw_capture differs" in error for error in errors)

    probe = pathlib.Path(bundle["execution_authority"]["probe_binary"]["path"])
    probe.write_bytes(b"different probe")
    authority_errors = contract.validate_execution_authority(
        bundle["execution_authority"], raw_probe_sha256=bundle["raw_probe_sha256"],
    )
    assert any("probe_binary differs" in error for error in authority_errors)


def test_incomplete_interval_and_domain_coverage_fail(tmp_path: pathlib.Path) -> None:
    bundle, payload, evidence, _ = _validate(tmp_path)
    evidence["domains"][0]["capture_started_at_unix_ns"] = 11_000_000_000
    evidence = contract.stamp(evidence)
    errors = contract.validate(
        evidence,
        raw_bundle_sha256=bundle["raw_bundle_sha256"],
        artifact_sha256=bundle["raw_probe"]["artifact"]["sha256"],
        execution_authority=bundle["execution_authority"],
        counter_payload=payload,
        required_domains=("energy", "gpu_time", "physical_bytes"),
        minimum_samples=20,
    )
    assert "counter domain energy does not cover the workload interval" in errors


def _markers() -> dict:
    pairs = [
        {
            "phase": "warmup", "batch": None, "iteration": 0,
            "first_role": "baseline", "comparison_role": "candidate",
        },
        {
            "phase": "trial", "batch": None, "iteration": 0,
            "first_role": "candidate", "comparison_role": "candidate",
        },
    ]
    return phase_markers(
        pairs=pairs,
        singles=[{
            "phase": "parity", "role": "candidate_q12",
            "batch": None, "iteration": 0,
        }],
    )


def test_phase_markers_bind_nonce_dual_clocks_and_pair_intervals() -> None:
    markers = _markers()
    assert contract.validate_phase_markers(
        markers,
        run_nonce="2" * 64,
        workload_started_at_unix_ns=10_000_000_000,
        workload_ended_at_unix_ns=20_000_000_000,
        workload_elapsed_continuous_ns=10_000_000_000,
    ) == []


def test_phase_marker_time_hash_nonce_and_pair_tampering_fail_closed() -> None:
    markers = _markers()
    markers["run_nonce"] = "3" * 64
    markers["intervals"][0]["continuous_ended_ns"] = (
        markers["intervals"][0]["continuous_started_ns"] - 1
    )
    markers["pairs"][1]["candidate_interval_sha256"] = markers["pairs"][0][
        "candidate_interval_sha256"
    ]
    errors = contract.validate_phase_markers(markers, run_nonce="2" * 64)
    assert any("execution authority run_nonce" in error for error in errors)
    assert any("timing is invalid" in error for error in errors)
    assert any("interval_sha256 mismatch" in error for error in errors)
    assert any("reuses an interval" in error or "identities" in error for error in errors)
    assert "phase_markers.phase_markers_sha256 mismatch" in errors
