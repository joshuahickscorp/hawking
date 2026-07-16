from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import doctor_v5_physical_ab_controller as controller  # noqa: E402
import doctor_v5_physical_ab_executor as executor  # noqa: E402
import doctor_v5_physical_adapter_registry as registry  # noqa: E402


PLAN_SHA256 = "a" * 64
SOURCE_SHA256 = "b" * 64


def _write_json(path: pathlib.Path, value: object) -> pathlib.Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _artifact(path: pathlib.Path, *, executable: bool = False) -> dict:
    return registry.file_identity(path, executable=executable)


def _argv(role: str, program_sha256: str, facet: str) -> dict:
    return registry._stamp({
        "schema": registry.ARGV_MANIFEST_SCHEMA,
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
        "environment": {"LANG": "C", "HAWKING_ADAPTER_FACET": facet},
        "cwd": str(ROOT),
        "stdin": "devnull",
        "shell": False,
        "mutates_live_doctor": False,
        "mutates_runtime_defaults": False,
        "deletes_sources": False,
    }, "manifest_sha256")


def _install_ephemeral_trust(
    monkeypatch: pytest.MonkeyPatch, root: pathlib.Path, name: str = "release",
) -> pathlib.Path:
    key = root / name
    subprocess.run(
        [
            str(registry.SSH_KEYGEN), "-q", "-t", "ed25519", "-N", "",
            "-f", str(key),
        ],
        cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True, env=registry.SSH_ENV,
    )
    key.chmod(0o600)
    fields = key.with_suffix(".pub").read_text(encoding="ascii").split()
    allowed = root / f"{name}.allowed_signers"
    allowed.write_text(
        f"{registry.SIGNER_IDENTITY} {fields[0]} {fields[1]}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(registry, "DEFAULT_ALLOWED_SIGNERS", allowed)
    monkeypatch.setattr(
        registry, "PINNED_ALLOWED_SIGNERS_SHA256",
        hashlib.sha256(allowed.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        registry, "PINNED_PUBLIC_KEY_BLOB_SHA256",
        hashlib.sha256(f"{fields[0]} {fields[1]}".encode("ascii")).hexdigest(),
    )
    return key


def _ten_requests(root: pathlib.Path) -> list[dict]:
    baseline = root / "baseline-program"
    candidate = root / "candidate-program"
    baseline.write_bytes(b"#!/bin/sh\nexit 0\n# reviewed baseline fixture\n")
    candidate.write_bytes(b"#!/bin/sh\nexit 0\n# reviewed candidate fixture\n")
    baseline.chmod(0o755)
    candidate.chmod(0o755)
    baseline_identity = _artifact(baseline, executable=True)
    candidate_identity = _artifact(candidate, executable=True)
    requests: list[dict] = []
    for facet in registry.FACETS:
        facet_root = root / facet
        facet_root.mkdir()
        baseline_argv_path = _write_json(
            facet_root / "baseline.argv.json",
            _argv("baseline", baseline_identity["sha256"], facet),
        )
        candidate_argv_path = _write_json(
            facet_root / "candidate.argv.json",
            _argv("candidate", candidate_identity["sha256"], facet),
        )
        source_manifest = registry._stamp({
            "schema": controller.SOURCE_UNIT_MANIFEST_SCHEMA,
            "segment": "sub-120b-doctor",
            "model": "Doctor-V5",
            "tier": "3B-through-72B",
            "units": [{
                "source_unit_id": f"fixture:{facet}",
                "source_sha256": registry.canonical_sha256(f"source:{facet}"),
            }],
        }, "manifest_sha256")
        source_manifest_path = _write_json(
            facet_root / "source-units.json", source_manifest,
        )
        scope = registry._stamp({
            "schema": registry.EXECUTION_SCOPE_SCHEMA,
            "segment": "sub-120b-doctor",
            "model": "Doctor-V5",
            "tier": "3B-through-72B",
            "parameter_scope": "3B-through-72B",
            "facet": facet,
            "source_units": 1,
            "source_unit_manifest": _artifact(source_manifest_path),
            "rates": ["0.1"],
            "branches": ["codec_control"],
            "cells": 1,
            "jobs": 1,
            "skips": 0,
        }, "scope_sha256")
        scope_path = _write_json(facet_root / "scope.json", scope)
        input_path = _write_json(facet_root / "inputs.json", {"facet": facet})
        collector_path = _write_json(
            facet_root / "collector.json", {"facet": facet, "default_off": True},
        )
        seed = registry.canonical_sha256(f"pairing:{facet}")
        launch = registry._stamp({
            "schema": registry.LAUNCH_CONTRACT_SCHEMA,
            "plan_sha256": PLAN_SHA256,
            "source_manifest_sha256": SOURCE_SHA256,
            "executor_source_sha256": registry.canonical_sha256("executor-fixture"),
            "facet": facet,
            "baseline_program": baseline_identity,
            "baseline_argv_manifest": _artifact(baseline_argv_path),
            "candidate_program": candidate_identity,
            "candidate_argv_manifest": _artifact(candidate_argv_path),
            "input_manifest": _artifact(input_path),
            "execution_scope": _artifact(scope_path),
            "collector_authority": _artifact(collector_path),
            "pairing": {
                "warmups_per_arm": 1,
                "repeats_per_arm": 5,
                "random_seed_sha256": seed,
                "randomized_interleaved": True,
                "order": [],
                "order_sha256": registry.canonical_sha256([]),
            },
            "run_limits": {"fixture_only": True},
            "output_policy": {
                "root": str(facet_root / "output"),
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
        launch_path = _write_json(facet_root / "launch.json", launch)
        requests.append({
            "schema": registry.ENTRY_REQUEST_SCHEMA,
            "adapter_id": f"sub120:{facet}:fixture-v1",
            "segment": "sub-120b-doctor",
            "model": "Doctor-V5",
            "tier": "3B-through-72B",
            "parameter_scope": "3B-through-72B",
            "facet": facet,
            "baseline_program_path": str(baseline),
            "baseline_argv_manifest_path": str(baseline_argv_path),
            "candidate_program_path": str(candidate),
            "candidate_argv_manifest_path": str(candidate_argv_path),
            "execution_scope_path": str(scope_path),
            "launch_contract_path": str(launch_path),
        })
    return requests


@pytest.fixture
def signed_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> dict:
    key = _install_ephemeral_trust(monkeypatch, tmp_path)
    monkeypatch.setattr(
        registry, "_current_controller_bindings",
        lambda: (PLAN_SHA256, SOURCE_SHA256),
    )
    requests = _ten_requests(tmp_path)
    issued = time.time_ns() - 1_000_000_000
    document = registry.build_registry(
        requests, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256,
        issued_at_unix_ns=issued, valid_seconds=300,
    )
    return {"key": key, "requests": requests, "registry": document}


def test_status_distinguishes_ready_issuer_from_absent_production_descriptors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    _install_ephemeral_trust(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "DEFAULT_ENVELOPE", tmp_path / "absent.json")
    value = registry.status()
    assert value["issuer_ready_for_concrete_inputs"] is True
    assert value["independent_verifier_available"] is True
    assert value["production_descriptor_state"] == "absent"
    assert value["production_descriptors_verified"] == 0
    assert value["exact_ten_production_descriptors_verified"] is False
    assert value["execution_granted"] is False
    assert value["private_key_read"] is False


def test_exact_ten_descriptor_draft_is_semantically_bound_and_default_off(
    signed_fixture: dict,
) -> None:
    document = signed_fixture["registry"]
    assert registry.validate_registry(
        document, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256, verify_files=True,
    ) == []
    assert [row["facet"] for row in document["entries"]] == list(registry.FACETS)
    assert len({row["adapter_id"] for row in document["entries"]}) == 10
    assert document["registry_grants_execution"] is False
    assert document["activation_requested"] is False
    assert document["runtime_defaults_changed"] is False
    envelope_stub = {"registry": document}
    sub120, all_entries = registry.normalized_registries(envelope_stub)
    assert set(sub120) == set(registry.FACETS)
    assert len(all_entries) == 10


def test_real_sshsig_round_trip_and_independent_verification(
    signed_fixture: dict, tmp_path: pathlib.Path,
) -> None:
    signature = tmp_path / "registry.sig"
    envelope_path = tmp_path / "registry.envelope.json"
    envelope = registry.sign_registry(
        signed_fixture["registry"], private_key=signed_fixture["key"],
        signature_output=signature, envelope_output=envelope_path,
    )
    assert registry.validate_envelope(
        envelope, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256,
    ) == []
    verification = registry.verify_envelope_path(
        envelope_path, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256,
    )
    assert verification["signature_verified"] is True
    assert verification["exact_ten_facet_sub120_verified"] is True
    assert verification["verified_adapter_count"] == 10
    assert verification["execution_granted"] is False
    assert verification["physical_execution_claimed"] is False
    sub120, post120, errors = registry.load_registries(
        plan_sha256=PLAN_SHA256, source_manifest_sha256=SOURCE_SHA256,
        envelope_path=envelope_path,
    )
    assert errors == [] and len(sub120) == 10 and len(post120) == 10


def test_draft_sign_verify_cli_is_complete_and_still_default_off(
    signed_fixture: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    requests_path = _write_json(tmp_path / "requests.json", signed_fixture["requests"])
    draft_path = tmp_path / "draft.json"
    assert registry.main([
        "draft", "--entry-requests", str(requests_path),
        "--plan-sha256", PLAN_SHA256,
        "--source-manifest-sha256", SOURCE_SHA256,
        "--valid-seconds", "300", "--output", str(draft_path),
    ]) == 0
    monkeypatch.setattr(
        registry, "_current_controller_bindings",
        lambda: (PLAN_SHA256, SOURCE_SHA256),
    )
    signature_path = tmp_path / "cli.sig"
    envelope_path = tmp_path / "cli.envelope.json"
    assert registry.main([
        "sign", "--registry", str(draft_path),
        "--private-key", str(signed_fixture["key"]),
        "--signature-output", str(signature_path),
        "--envelope-output", str(envelope_path),
    ]) == 0
    assert registry.main(["verify", "--envelope", str(envelope_path)]) == 0
    verification = registry.verify_envelope_path(
        envelope_path, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256,
    )
    assert verification["signature_verified"] is True
    assert verification["execution_granted"] is False
    assert verification["physical_execution_claimed"] is False


def test_forgery_expiry_wrong_signer_and_source_drift_fail_closed(
    signed_fixture: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path,
) -> None:
    signature = tmp_path / "registry.sig"
    envelope_path = tmp_path / "registry.envelope.json"
    envelope = registry.sign_registry(
        signed_fixture["registry"], private_key=signed_fixture["key"],
        signature_output=signature, envelope_output=envelope_path,
    )

    forged = copy.deepcopy(envelope)
    entry = forged["registry"]["entries"][0]
    entry["adapter_id"] += "-forged"
    forged["registry"]["entries"][0] = registry._stamp(entry, "entry_sha256")
    forged["registry"] = registry._stamp(forged["registry"], "registry_sha256")
    forged = registry._stamp(forged, "envelope_sha256")
    errors = registry.validate_envelope(
        forged, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256,
    )
    assert any("SSHSIG verification failed" in row for row in errors)

    expires = envelope["registry"]["expires_at_unix_ns"]
    assert any("not current" in row for row in registry.validate_envelope(
        envelope, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256, now_unix_ns=expires,
        verify_signature=False,
    ))
    assert any("plan/source" in row for row in registry.validate_envelope(
        envelope, plan_sha256="c" * 64,
        source_manifest_sha256=SOURCE_SHA256, verify_signature=False,
    ))

    excessive = copy.deepcopy(envelope)
    excessive["registry"]["expires_at_unix_ns"] = (
        excessive["registry"]["issued_at_unix_ns"]
        + (registry.MAX_VALID_SECONDS + 1) * 1_000_000_000
    )
    excessive["registry"] = registry._stamp(
        excessive["registry"], "registry_sha256",
    )
    excessive = registry._stamp(excessive, "envelope_sha256")
    assert any("duration" in row for row in registry.validate_envelope(
        excessive, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256, verify_signature=False,
    ))

    wrong_key = _install_ephemeral_trust(monkeypatch, tmp_path, "wrong")
    # Restore the trusted public-key pins while retaining the untrusted key.
    trusted_fields = signed_fixture["key"].with_suffix(".pub").read_text(
        encoding="ascii",
    ).split()
    trusted_allowed = tmp_path / "trusted-again.allowed_signers"
    trusted_allowed.write_text(
        f"{registry.SIGNER_IDENTITY} {trusted_fields[0]} {trusted_fields[1]}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(registry, "DEFAULT_ALLOWED_SIGNERS", trusted_allowed)
    monkeypatch.setattr(
        registry, "PINNED_ALLOWED_SIGNERS_SHA256",
        hashlib.sha256(trusted_allowed.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        registry, "PINNED_PUBLIC_KEY_BLOB_SHA256",
        hashlib.sha256(
            f"{trusted_fields[0]} {trusted_fields[1]}".encode("ascii")
        ).hexdigest(),
    )
    with pytest.raises(registry.RegistryError, match="does not match"):
        registry.sign_registry(
            signed_fixture["registry"], private_key=wrong_key,
            signature_output=tmp_path / "wrong.sig",
            envelope_output=tmp_path / "wrong.envelope.json",
        )
    assert not (tmp_path / "wrong.sig").exists()

    pathlib.Path(
        signed_fixture["registry"]["entries"][0]["baseline_program"]["path"]
    ).write_bytes(b"changed after signing")
    assert any("signed identity" in row for row in registry.validate_envelope(
        envelope, plan_sha256=PLAN_SHA256,
        source_manifest_sha256=SOURCE_SHA256, verify_signature=False,
    ))


def test_cross_scope_duplicate_id_and_unvalidated_signing_are_rejected(
    signed_fixture: dict, tmp_path: pathlib.Path,
) -> None:
    duplicate = copy.deepcopy(signed_fixture["requests"])
    duplicate[1]["adapter_id"] = duplicate[0]["adapter_id"]
    with pytest.raises(registry.RegistryError, match="reuses an adapter id"):
        registry.build_registry(
            duplicate, plan_sha256=PLAN_SHA256,
            source_manifest_sha256=SOURCE_SHA256,
        )

    first_argv = pathlib.Path(
        signed_fixture["requests"][0]["baseline_argv_manifest_path"]
    )
    second_argv = pathlib.Path(
        signed_fixture["requests"][1]["baseline_argv_manifest_path"]
    )
    second_argv.write_bytes(first_argv.read_bytes())
    second_launch_path = pathlib.Path(
        signed_fixture["requests"][1]["launch_contract_path"]
    )
    second_launch = json.loads(second_launch_path.read_text(encoding="utf-8"))
    second_launch["baseline_argv_manifest"] = _artifact(second_argv)
    _write_json(
        second_launch_path, registry._stamp(second_launch, "contract_sha256"),
    )
    with pytest.raises(registry.RegistryError, match="reuses baseline_argv_manifest"):
        registry.build_registry(
            signed_fixture["requests"], plan_sha256=PLAN_SHA256,
            source_manifest_sha256=SOURCE_SHA256,
        )

    first = signed_fixture["requests"][0]
    scope_path = pathlib.Path(first["execution_scope_path"])
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["facet"] = "thread_profiles"
    _write_json(scope_path, registry._stamp(scope, "scope_sha256"))
    with pytest.raises(registry.RegistryError, match="exact adapter"):
        registry.build_registry(
            signed_fixture["requests"], plan_sha256=PLAN_SHA256,
            source_manifest_sha256=SOURCE_SHA256,
        )

    malformed = copy.deepcopy(signed_fixture["registry"])
    malformed["activation_requested"] = True
    malformed = registry._stamp(malformed, "registry_sha256")
    with pytest.raises(registry.RegistryError, match="refusing to sign invalid"):
        registry.sign_registry(
            malformed, private_key=signed_fixture["key"],
            signature_output=tmp_path / "must-not-exist.sig",
            envelope_output=tmp_path / "must-not-exist.envelope.json",
        )
    assert not (tmp_path / "must-not-exist.sig").exists()


def test_adapter_selection_never_falls_back_across_segment_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = {"adapter_id": "sub"}
    post = {"adapter_id": "post"}
    monkeypatch.setitem(executor.PROGRAM_ADAPTER_REGISTRY, "disk_lifecycle", sub)
    monkeypatch.setitem(
        controller.POST120_PROGRAM_ADAPTER_REGISTRY,
        ("gpt-oss-120b", "GPT-OSS", "disk_lifecycle"), post,
    )
    assert executor._adapter_for_scope(
        facet="disk_lifecycle", segment="sub-120b-doctor", model="Doctor-V5",
    ) is sub
    assert executor._adapter_for_scope(
        facet="disk_lifecycle", segment="gpt-oss-120b", model="GPT-OSS",
    ) is post
    assert executor._adapter_for_scope(
        facet="disk_lifecycle", segment="post-120b-higher-tier", model="Unknown",
    ) is None
    assert executor._adapter_for_scope(
        facet="disk_lifecycle", segment=None, model=None,
    ) is None


def test_adapter_immutable_output_rejects_writable_short_and_swapped_parent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable = tmp_path / "registry.json"
    writable.write_bytes(b"same")
    writable.chmod(0o666)
    with pytest.raises(registry.RegistryError, match="single-link mode"):
        registry._atomic_bytes(writable, b"same")

    short = tmp_path / "short.json"
    monkeypatch.setattr(registry.os, "write", lambda *_args, **_kwargs: 0)
    with pytest.raises(OSError, match="short write"):
        registry._atomic_bytes(short, b"registry")
    assert not short.exists()
    assert list(tmp_path.glob(".short.json.*.tmp")) == []

    monkeypatch.undo()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_open = registry._open_dir_nofollow

    def swapped(path: pathlib.Path, *, create: bool) -> int:
        if not create:
            return os.open(
                replacement,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        return original_open(path, create=create)

    monkeypatch.setattr(registry, "_open_dir_nofollow", swapped)
    swapped_output = tmp_path / "swapped.json"
    with pytest.raises(registry.RegistryError, match="parent was replaced"):
        registry._atomic_bytes(swapped_output, b"registry")
    assert not swapped_output.exists()
