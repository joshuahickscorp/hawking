"""Fail-closed tests for binding lineage to an observed resident process."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lab.lineage.identity import GENESIS_MODEL
from lab.qwen38_protected_run_verifier import VERIFICATION_SCHEMA
from tools import genesis_seat


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Any]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    physical_bpw = 4.252735126866492
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
                "complete_physical_bpw": physical_bpw,
            },
            sort_keys=True,
        )
    )
    kernel = tmp_path / "qwen_uniform_q4.metal"
    kernel.write_text("kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128() {}\n")
    resident_executable = tmp_path / "genesis-resident"
    resident_executable.write_bytes(b"resident executable fixture\n")
    resident_executable.chmod(0o755)
    measurement_runtime = tmp_path / "ascension_qwen38_hybrid_greedy"
    measurement_runtime.write_bytes(b"measurement runtime fixture")
    measurement_runtime.chmod(0o755)

    wall_ns = 37_100_000
    tps = 1_000_000_000.0 / wall_ns
    capture = {
        "schema": "hawking.ascent.qwen38_complete_token_wall.v1",
        "timing_label": "DIRTY_ENGINEERING",
        "authority": {
            "headline_complete_wall_ns_per_token": wall_ns,
            "headline_complete_tps": tps,
        },
        "identity": {
            "model": GENESIS_MODEL,
            "fallbacks": 0,
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        },
        "vehicle": {
            "artifact_root": str(artifact),
            "complete_physical_bpw": physical_bpw,
        },
    }
    capture_path = tmp_path / "QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json"
    capture_path.write_text(json.dumps(capture, sort_keys=True))
    measurement_runtime_sha = _sha(measurement_runtime)
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS",
        "capture_sha256": _sha(capture_path),
        "protected_binding": {
            "artifact_root": str(artifact),
            "artifact_manifest_path": str(manifest),
            "artifact_manifest_sha256": _sha(manifest),
            "runtime_executable_path": str(measurement_runtime),
            "runtime_executable_sha256": measurement_runtime_sha,
            "kernel_source_path": str(kernel),
            "kernel_source_sha256": _sha(kernel),
            "candidate_hashes_were_not_authority": True,
        },
        "measurement": {
            "model": GENESIS_MODEL,
            "fallbacks": 0,
            "derived_headline_complete_wall_ns_per_token": wall_ns,
            "derived_complete_tps": tps,
        },
        "claim_boundary": {
            "wall_rederived_from_raw_capture": True,
            "capture_origin_attested": False,
        },
    }
    verification_path = tmp_path / "QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.verify.json"
    verification_path.write_text(json.dumps(verification, sort_keys=True))

    lineage, _genesis = genesis_seat.build_seated_lineage(artifact_manifest=manifest)
    assert lineage.current is not None
    lineage.current.identity["artifact"] = str(artifact)
    state_path = tmp_path / "GENESIS_LINEAGE_CURRENT.json"
    state_path.write_text(json.dumps(lineage.to_dict(), indent=2))

    roles = list(genesis_seat.EXPECTED_SESSION_ROLES)
    health = {
        "ok": True,
        "protocol": genesis_seat.RESIDENT_PROTOCOL,
        "pid": 4242,
        "body_resident": True,
        "load_count": 1,
        "resident_weight_bytes": 14_297_675_776,
        "session_count": 4,
        "session_roles": roles,
        "session_workspace_bytes": {role: 173_703_168 for role in roles},
        "session_semantics": {
            role: {"kind": kind}
            for role, kind in genesis_seat.EXPECTED_SESSION_KINDS.items()
        },
        "lineage_children": 0,
        "artifact": str(artifact),
        "artifact_sha": _sha(manifest),
        "generation": 0,
        "reload_error": None,
    }
    return {
        "artifact": artifact,
        "manifest": manifest,
        "kernel": kernel,
        "resident_executable": resident_executable,
        "measurement_runtime": measurement_runtime,
        "capture_path": capture_path,
        "capture": capture,
        "verification_path": verification_path,
        "verification": verification,
        "state_path": state_path,
        "health": health,
        "wall_ns": wall_ns,
        "tps": tps,
        "physical_bpw": physical_bpw,
        "measurement_runtime_sha": measurement_runtime_sha,
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _bind(
    fixture: dict[str, Any],
    *,
    health: dict[str, Any] | None = None,
    alive: bool = True,
    executable: Path | None = None,
) -> dict[str, Any]:
    observed = fixture["health"] if health is None else health
    return genesis_seat.bind_live_state(
        verification_path=fixture["verification_path"],
        state_file=fixture["state_path"],
        capture_path=fixture["capture_path"],
        artifact_root=fixture["artifact"],
        kernel_source=fixture["kernel"],
        sock_path=Path("/tmp/test-genesis-resident.sock"),
        health_query=lambda _path: observed,
        alive_query=lambda _pid: alive,
        executable_query=lambda _pid: executable or fixture["resident_executable"],
    )


def test_bind_live_records_exact_observation_and_keeps_lkg_not_live(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    updated = _bind(fixture)
    persisted = json.loads(fixture["state_path"].read_text())
    assert persisted == updated
    current = persisted["slots"]["CURRENT"]
    lkg = persisted["slots"]["LAST_KNOWN_GOOD"]
    assert current["live"] is True
    assert current["launched"] is True
    assert current["artifact_sha"] == _sha(fixture["manifest"])
    assert current["runtime_sha"] == _sha(fixture["resident_executable"])
    assert current["kernel_genome_sha"] == _sha(fixture["kernel"])
    assert current["complete_token_ns"] == fixture["wall_ns"]
    assert current["tps"] == pytest.approx(fixture["tps"])
    assert current["physical_bpw"] == fixture["physical_bpw"]
    assert current["representation_bpw"] == 4.2527
    assert current["identity"]["measurement_runtime_sha256"] == fixture[
        "measurement_runtime_sha"
    ]
    assert current["identity"]["protected_verification_sha256"] == _sha(
        fixture["verification_path"]
    )
    assert lkg["live"] is False
    assert lkg["launched"] is False
    event = persisted["events"][-1]
    assert event["kind"] == "bind_live_observed"
    assert event["payload"]["schema"] == genesis_seat.LIVE_BINDING_SCHEMA
    assert event["payload"]["session_roles"] == list(genesis_seat.EXPECTED_SESSION_ROLES)
    assert event["payload"]["capture_origin_attested"] is False
    assert event["payload"]["verification_sha256"] == _sha(
        fixture["verification_path"]
    )
    assert len(event["payload"]["observation_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("body_resident", False, "body_resident"),
        ("load_count", 2, "exactly one body load"),
        ("session_roles", ["parent", "child_a", "child_b"], "roles must be exactly"),
        ("session_count", 3, "session_count"),
        ("artifact_sha", "0" * 64, "artifact SHA"),
        ("stub", True, "stub resident"),
        ("reload_error", "failed reload", "reload error"),
        ("lineage_children", 1, "zero lineage children"),
    ],
)
def test_resident_health_refusals_leave_state_unchanged(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    fixture = _fixture(tmp_path)
    health = copy.deepcopy(fixture["health"])
    health[field] = bad_value
    before = fixture["state_path"].read_bytes()

    with pytest.raises(genesis_seat.BindLiveError, match=message):
        _bind(fixture, health=health)

    assert fixture["state_path"].read_bytes() == before


def test_worker_session_semantics_are_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    health = copy.deepcopy(fixture["health"])
    health["session_semantics"]["child_a"]["kind"] = "lineage_child"
    before = fixture["state_path"].read_bytes()

    with pytest.raises(genesis_seat.BindLiveError, match="worker_session"):
        _bind(fixture, health=health)

    assert fixture["state_path"].read_bytes() == before


def test_dead_pid_or_wrong_process_executable_refuses_without_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = fixture["state_path"].read_bytes()
    with pytest.raises(genesis_seat.BindLiveError, match="pid is not live"):
        _bind(fixture, alive=False)
    assert fixture["state_path"].read_bytes() == before

    wrong = tmp_path / "not-genesis"
    wrong.write_bytes(b"wrong executable\n")
    wrong.chmod(0o755)
    with pytest.raises(genesis_seat.BindLiveError, match="does not own"):
        _bind(fixture, executable=wrong)
    assert fixture["state_path"].read_bytes() == before


def test_pid_exit_during_binding_refuses_without_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = fixture["state_path"].read_bytes()
    liveness = iter((True, False))

    with pytest.raises(genesis_seat.BindLiveError, match="exited during"):
        genesis_seat.bind_live_state(
            verification_path=fixture["verification_path"],
            state_file=fixture["state_path"],
            capture_path=fixture["capture_path"],
            artifact_root=fixture["artifact"],
            kernel_source=fixture["kernel"],
            sock_path=Path("/tmp/test-genesis-resident.sock"),
            health_query=lambda _path: fixture["health"],
            alive_query=lambda _pid: next(liveness),
            executable_query=lambda _pid: fixture["resident_executable"],
        )

    assert fixture["state_path"].read_bytes() == before


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("capture", "immutable timing capture"),
        ("kernel", "kernel_source_sha256"),
        ("tps", "TPS is inconsistent"),
        ("physical_bpw", "physical BPW differs"),
        ("runtime", "runtime SHA"),
        ("origin_attested", "origin-unattested"),
        ("timing_label", "DIRTY_ENGINEERING"),
    ],
)
def test_measurement_binding_refusals_are_atomic(
    tmp_path: Path, target: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    if target == "capture":
        fixture["capture"]["timing_label"] = "tampered"
        _write(fixture["capture_path"], fixture["capture"])
    elif target == "kernel":
        fixture["kernel"].write_text("different kernel source\n")
    elif target == "tps":
        fixture["verification"]["measurement"]["derived_complete_tps"] = 999.0
        _write(fixture["verification_path"], fixture["verification"])
    elif target == "physical_bpw":
        fixture["capture"]["vehicle"]["complete_physical_bpw"] = 3.0
        _write(fixture["capture_path"], fixture["capture"])
        fixture["verification"]["capture_sha256"] = _sha(fixture["capture_path"])
        _write(fixture["verification_path"], fixture["verification"])
    elif target == "runtime":
        fixture["measurement_runtime"].write_bytes(b"changed measurement runtime\n")
        fixture["measurement_runtime"].chmod(0o755)
    elif target == "origin_attested":
        fixture["verification"]["claim_boundary"]["capture_origin_attested"] = True
        _write(fixture["verification_path"], fixture["verification"])
    else:
        fixture["capture"]["timing_label"] = "OFFICIAL"
        _write(fixture["capture_path"], fixture["capture"])
        fixture["verification"]["capture_sha256"] = _sha(fixture["capture_path"])
        _write(fixture["verification_path"], fixture["verification"])
    before = fixture["state_path"].read_bytes()

    with pytest.raises(genesis_seat.BindLiveError, match=message):
        _bind(fixture)

    assert fixture["state_path"].read_bytes() == before


def test_atomic_replace_failure_preserves_previous_state(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    before = fixture["state_path"].read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(genesis_seat.os, "replace", fail_replace)
    with pytest.raises(genesis_seat.BindLiveError, match="atomic lineage replace failed"):
        _bind(fixture)

    assert fixture["state_path"].read_bytes() == before
    assert not list(tmp_path.glob(".GENESIS_LINEAGE_CURRENT.json.bind-live-*"))
