from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import appendix_physical_counter_executor as executor  # noqa: E402
import appendix_physical_counter_request as request_builder  # noqa: E402
import appendix_physical_release_packet as release_packet  # noqa: E402


def _json(path: pathlib.Path, value: dict) -> pathlib.Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_builder_contract_is_default_off_and_hash_bound() -> None:
    config = request_builder.build_config()
    assert config["default_off"] is True
    assert config["execution_capability"] is False
    assert config["opens_heavy_lease"] is False
    assert config["starts_collector_or_probe"] is False
    assert config["authority_registry_sha256"]
    assert request_builder.dry_run("device")["would_write_request"] is False


def test_typed_workload_builders_measure_file_identities(tmp_path: pathlib.Path) -> None:
    probe = tmp_path / "probe"
    probe.write_bytes(b"probe")
    artifact = tmp_path / "artifact.tq"
    artifact.write_bytes(b"artifact")
    release = {
        "probes": {
            "device": {"path": str(probe), "sha256": "a" * 64, "size_bytes": 5},
            "spec": {"path": str(probe), "sha256": "a" * 64, "size_bytes": 5},
        },
    }
    value = request_builder.device_workload(
        release_build=release, artifact=artifact, runtime_path="stored",
        cell_id="cell", tensor="ffn_down",
    )
    assert value["artifact"]["sha256"] == hashlib.sha256(b"artifact").hexdigest()
    assert value["probe"] == release["probes"]["device"]
    assert value["residual_artifact"] is None


def test_key_path_parser_rejects_missing_duplicate_and_extra_keys(tmp_path: pathlib.Path) -> None:
    with pytest.raises(request_builder.RequestBuildError, match="differs"):
        request_builder._parse_key_paths(
            [f"a={tmp_path / 'a'}"], expected={"a", "b"}, label="parent",
        )
    with pytest.raises(request_builder.RequestBuildError, match="duplicate"):
        request_builder._parse_key_paths(
            [f"a={tmp_path / 'a'}", f"a={tmp_path / 'b'}"],
            expected={"a"}, label="authority",
        )


def test_builder_constructs_complete_stamped_request_without_handwritten_json(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = {
        key: _json(tmp_path / f"{key}.json", {})
        for key in request_builder.PARENT_KEYS
    }
    authority_paths = {
        key: _json(tmp_path / f"{key}.json", {})
        for key in executor.AUTHORITY_SPECS
    }
    monkeypatch.setattr(executor, "_release_errors", lambda *_a, **_k: [])
    monkeypatch.setattr(executor, "_workload_errors", lambda *_a, **_k: [])
    monkeypatch.setattr(
        executor, "validate_authorities",
        lambda *_a, **_k: ([], {}),
    )
    output_directory = executor.REPORT_ROOT / (
        "pytest-request-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    request = request_builder.build_request(
        kind="device", parents=parents, authority_paths=authority_paths,
        workload={}, output_directory=output_directory, scratch_reserve_gb=5,
        observer={"final_interpretation_ready": True}, active_owners=[],
        live_host_sha256="a" * 64, verify_files=False,
        signature_verifier=lambda _e, _p: (True, ""),
    )
    assert request["schema"] == executor.REQUEST_SCHEMA
    assert request["release"]["authority_registry"]["registry_sha256"]
    assert set(request["authorities"]) == set(executor.AUTHORITY_SPECS)
    assert request["runtime_default_mutation_requested"] is False
    unstamped = dict(request)
    claimed = unstamped.pop("request_sha256")
    assert claimed == executor.canonical_sha256(unstamped)


def test_release_parent_set_accepts_exact_prepare_release_pre_post_chain(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "result.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8",
    )
    index = release_packet.appendix_corpus.build_index(
        corpus_root, active_owners=[], source_base_commit="a" * 40,
    )
    observed = 123
    owner_snapshot = {
        "schema": "hawking.appendix_owner_snapshot.v1",
        "exclusive_shared_heavy_lease_held": True,
        "lock_path": "/tmp/studio_heavy.lock", "lock_device": 1,
        "lock_inode": 2, "owners": [], "observed_at_unix_ns": observed,
    }
    owner_sha = release_packet.canonical_sha256(owner_snapshot)
    observation = release_packet._stamp({
        "schema": release_packet.BOUNDARY_OBSERVATION_SCHEMA,
        "observer_state_sha256": "a" * 64,
        "final_packet_file_sha256": "b" * 64,
        "final_packet_canonical_sha256": "c" * 64,
        "all_recorded_hashes_verified": True,
        "verified_reference_count": 9,
        "verified_references_sha256": "d" * 64,
        "owner_snapshot": owner_snapshot,
        "owner_snapshot_sha256": owner_sha,
        "resource_snapshot": {"ok": True, "pressure_level": 1, "swap_used_mb": 0.0},
        "ram_swap_guard_healthy": True,
        "observed_at_unix_ns": observed,
        "default_mutation_requested": False,
    }, "observation_sha256")
    boundary = release_packet._stamp({
        "schema": release_packet.evidence_gate.RELEASE_BOUNDARY_SCHEMA,
        "final_interpretation_ready": True,
        "final_packet_sha256": "b" * 64,
        "observer_state_sha256": "a" * 64,
        "all_recorded_hashes_verified": True,
        "active_heavy_owner_count": 0,
        "owner_snapshot_sha256": owner_sha,
        "ram_swap_guard_healthy": True,
        "observed_at_unix_ns": observed,
    }, "attestation_sha256")
    monkeypatch.setattr(
        release_packet.spec_reentry_scaffold, "active_heavy_owners", lambda: [],
    )
    pre, _ = release_packet.build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=124, verification_phase="pre_release_build",
    )
    post, attestation = release_packet.build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=125, verification_phase="post_release_build",
        parent_verification_receipt=pre,
    )
    release = {
        "boundary_attestation": boundary,
        "boundary_observation": observation,
        "corpus_index": index,
        "corpus_verification": attestation,
        "corpus_prebuild_verification_receipt": pre,
        "corpus_verification_receipt": post,
        "source_manifest": {}, "release_build": {}, "authority_registry": {},
    }
    monkeypatch.setattr(
        executor.release_packet, "validate_release_boundary_attestation",
        lambda *_a, **_k: [],
    )
    seen_parent: list[dict] = []

    def validate_corpus_chain(*_args, **kwargs):
        seen_parent.append(kwargs["parent_verification_receipt"])
        return []

    monkeypatch.setattr(
        executor.release_packet, "validate_corpus_verification", validate_corpus_chain,
    )
    monkeypatch.setattr(
        executor.release_packet, "validate_clean_source_manifest", lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        executor.release_packet, "validate_release_build_receipt", lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        executor.authority_root, "validate_registry", lambda *_a, **_k: [],
    )
    assert executor._release_errors(
        release, observer={}, verify_files=False,
    ) == []
    assert seen_parent == [pre]
    without_pre = dict(release)
    without_pre.pop("corpus_prebuild_verification_receipt")
    assert executor._release_errors(
        without_pre, observer={}, verify_files=False,
    ) == ["release parent set is incomplete or unexpected"]
