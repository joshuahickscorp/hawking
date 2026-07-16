from __future__ import annotations

import copy
import fcntl
import json
import os
import pathlib
import stat
import sys
from dataclasses import dataclass

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))

import appendix_contract  # noqa: E402
import appendix_physical_counter_authority as authority_root  # noqa: E402
import appendix_physical_counter_executor as executor  # noqa: E402
import physical_counter_attestation  # noqa: E402
import ram_scheduler  # noqa: E402


def _file(path: pathlib.Path, content: bytes = b"evidence") -> dict:
    path.write_bytes(content)
    return physical_counter_attestation.file_identity(path)


def _authority(
    tmp_path: pathlib.Path, *, subject: str = "normalizer", kind: str = "capability",
    claims: list[str] | None = None, abi: str | None = None,
    release_build_sha256: str = "b" * 64, issued: int = 10, expires: int = 100,
) -> dict:
    binary = _file(tmp_path / f"{subject}-{kind}.bin")
    allowed = physical_counter_attestation.file_identity(authority_root.DEFAULT_ALLOWED_SIGNERS)
    signature = _file(tmp_path / f"{subject}-{kind}.sig")
    receipt = executor._stamp({
        "schema": executor.AUTHORITY_SCHEMA,
        "receipt_kind": kind,
        "subject": subject,
        "host_hardware_uuid_sha256": "a" * 64,
        "binary": binary,
        "command_abi_sha256": abi or executor.ABI_HASHES.get(subject, "c" * 64),
        "claims": claims or sorted(executor.AUTHORITY_SPECS["normalizer_capability"][2]),
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": expires,
        "release_build_sha256": release_build_sha256,
    }, "receipt_sha256")
    return executor._stamp({
        "schema": executor.ENVELOPE_SCHEMA,
        "receipt": receipt,
        "signer_identity": authority_root.SIGNER_IDENTITY,
        "signature_namespace": executor.SSHSIG_NAMESPACE,
        "allowed_signers": allowed,
        "detached_signature": signature,
    }, "envelope_sha256")


def _minimal_request(tmp_path: pathlib.Path) -> dict:
    unique_output = executor.REPORT_ROOT / (
        "pytest-no-start-" + __import__("hashlib").sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    request = {
        "schema": executor.REQUEST_SCHEMA,
        "kind": "device",
        "release": {},
        "workload": {},
        "authorities": {},
        "output_directory": str(unique_output),
        "scratch_reserve_gb": 2,
        "additional_parent_receipt_sha256": [],
        "execution_requested": True,
        "runtime_default_mutation_requested": False,
    }
    return executor._stamp(request, "request_sha256")


def _write_request(path: pathlib.Path, request: dict) -> None:
    path.write_text(__import__("json").dumps(request), encoding="utf-8")


def _xctrace_evidence_fixture(
    request: dict, *, raw_bundle: dict, execution_authority: dict,
    probe_pid: int = 123,
) -> dict:
    output = pathlib.Path(request["output_directory"])
    identity = lambda name, digest: {
        "path": str(output / name), "sha256": digest, "size_bytes": 1,
    }
    exports = {
        table: identity(f"xctrace.export.{table}.xml", f"{20 + index:064x}")
        for index, table in enumerate(executor.xctrace_adapter.REQUIRED_TABLES)
    }
    authorities = {
        key: request["authorities"][key] for key in executor.XCTRACE_AUTHORITY_KEYS
    }
    capture_sha = "a" * 64
    receipt_sha = "b" * 64
    return executor._stamp({
        "schema": executor.XCTRACE_EXPORT_EVIDENCE_SCHEMA,
        "adapter_schema": executor.xctrace_adapter.SCHEMA,
        "adapter_contract_sha256": executor.xctrace_adapter.CONTRACT_SHA256,
        "kind": request["kind"], "probe_pid": probe_pid,
        "run_nonce": execution_authority["run_nonce"],
        "probe_argv_sha256": execution_authority["argv_sha256"],
        "metal_registry_id": "metal-1",
        "xctrace_binary": request["authorities"]["xctrace_capability"][
            "receipt"
        ]["binary"],
        "xctrace_authority_chain_sha256": executor.canonical_sha256(authorities),
        "profile_identity": {
            "path": str(executor.xctrace_adapter.DEFAULT_PROFILE.absolute()),
            "sha256": "3" * 64, "size_bytes": 1,
        },
        "profile_sha256": "4" * 64,
        "raw_bundle_identity": identity("raw_bundle.json", "5" * 64),
        "raw_bundle_sha256": raw_bundle["raw_bundle_sha256"],
        "trace_identity": {
            "schema": "hawking.xctrace_trace_tree_identity.v1",
            "path": str(output / "metal.trace"), "total_size_bytes": 1,
            "files": [{
                "relative_path": "trace.data", "sha256": "6" * 64,
                "size_bytes": 1,
            }],
            "tree_sha256": "7" * 64,
        },
        "toc_identity": identity("xctrace.toc.export", "8" * 64),
        "export_identities": exports,
        "canonical_capture_identity": identity("xctrace.canonical.json", "9" * 64),
        "capture_sha256": capture_sha,
        "adapter_receipt_identity": identity("xctrace.adapter-receipt.json", "c" * 64),
        "adapter_receipt_sha256": receipt_sha,
        "lease": {"inherited": True, "device": 1, "inode": 2},
        "output_directory": {
            "held": True, "path": str(output), "device": 10, "inode": 11,
        },
        "file_backed_validation": {
            "receipt_sha256": receipt_sha, "capture_sha256": capture_sha,
            "all_provenance_files_reopened": True,
            "physical_evidence_eligible": True,
        },
        "physical_evidence_eligible": True,
    }, "xctrace_export_evidence_sha256")


def _result_chain(tmp_path: pathlib.Path) -> tuple[dict, dict, dict, dict]:
    authority_dir = tmp_path / "authorities"
    authority_dir.mkdir()
    probe = _file(tmp_path / "release-probe", b"release-probe")
    authorities = _full_authorities(authority_dir, "a" * 64, probe_identity=probe)
    request = executor._stamp({
        "schema": executor.REQUEST_SCHEMA,
        "kind": "device",
        "release": {
            "boundary_attestation": {"attestation_sha256": "7" * 64},
            "boundary_observation": {"observation_sha256": "8" * 64},
            "corpus_index": {"index_sha256": "9" * 64},
            "corpus_verification": {},
            "corpus_prebuild_verification_receipt": {},
            "corpus_verification_receipt": {},
            "source_manifest": {"manifest_sha256": "c" * 64},
            "release_build": {"receipt_sha256": "b" * 64},
            "authority_registry": authority_root.load_default_registry(),
        },
        "workload": {"probe": probe},
        "authorities": authorities,
        "output_directory": str(executor.REPORT_ROOT / ("pytest-result-" + tmp_path.name)),
        "scratch_reserve_gb": 2,
        "additional_parent_receipt_sha256": [],
        "execution_requested": True,
        "runtime_default_mutation_requested": False,
    }, "request_sha256")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    raw_bundle = {"raw_bundle_sha256": "1" * 64}
    execution_authority = {
        "run_nonce": "d" * 64, "argv_sha256": "e" * 64,
    }
    evidence = {
        "cell_id": "device-cell", "raw_bundle": raw_bundle, "receipt": {},
        "execution_authority": execution_authority, "counter_payload": {},
        "counter_attestation": {},
    }
    evidence["xctrace_export_evidence"] = _xctrace_evidence_fixture(
        request, raw_bundle=raw_bundle, execution_authority=execution_authority,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    provenance_path = tmp_path / "process_joule_provenance.json"
    provenance_path.write_text(
        json.dumps(executor.process_joule.library_provenance()), encoding="utf-8",
    )
    execution = executor._stamp({
        "schema": executor.EXECUTION_SCHEMA,
        "request_sha256": request["request_sha256"],
        "request_file": physical_counter_attestation.file_identity(request_path),
        "authority_chain_sha256": executor.canonical_sha256(authorities),
        "kind": "device", "run_nonce": "d" * 64,
        "probe_pid": 123, "probe_argv_sha256": "e" * 64,
        "external_trace_started_before_barrier": True,
        "capture_readiness": {},
        "execution_started_at_unix_ns": 20,
        "execution_ended_at_unix_ns": 30,
        "raw_bundle_sha256": "1" * 64,
        "counter_payload_sha256": "2" * 64,
        "counter_attestation_sha256": "3" * 64,
        "xctrace_export_evidence_sha256": evidence["xctrace_export_evidence"][
            "xctrace_export_evidence_sha256"
        ],
        "core_evidence_sha256": executor.canonical_sha256(evidence),
        "evidence_file": physical_counter_attestation.file_identity(evidence_path),
        "process_joule_provenance_file": physical_counter_attestation.file_identity(
            provenance_path,
        ),
        "prelaunch_release_cas": executor._stamp({"phase": "pre"}, "cas_sha256"),
        "final_release_cas": executor._stamp({"phase": "final"}, "cas_sha256"),
        "runtime_defaults_changed": False,
    }, "execution_receipt_sha256")
    draft = executor.build_result_attestation(
        request=request, execution_receipt=execution, evidence=evidence,
        attested_at_unix_ns=40,
    )
    return request, execution, evidence, draft


def _signed_result(tmp_path: pathlib.Path, draft: dict) -> dict:
    signature = _file(tmp_path / "result.sig", b"test-result-signature")
    registry = authority_root.load_default_registry()
    return executor._stamp({
        "schema": executor.RESULT_ENVELOPE_SCHEMA,
        "attestation": draft,
        "signer_identity": registry["signer_identity"],
        "signature_namespace": executor.RESULT_SSHSIG_NAMESPACE,
        "allowed_signers": authority_root.allowed_signers_identity(registry),
        "detached_signature": signature,
    }, "envelope_sha256")


def test_capability_contract_is_separate_and_preserves_collector_default_off() -> None:
    value = executor.execution_capability_contract()
    assert value["execute_surface_exposed"] is True
    assert value["present_execution_admission"] is False
    assert value["collector_normalizer_collection_cli_exposed"] is False
    assert value["collector_config_sha256"]
    authority = executor.authority_requirements()
    assert value["authority_requirements_sha256"] == authority["requirements_sha256"]
    assert authority["grants_execution_by_itself"] is False
    assert executor._hash_errors(value, "capability_sha256", label="capability") == []


def test_status_and_dry_run_never_claim_present_execution() -> None:
    value = executor.status(
        euid=501, final_ready=False, heavy_owner_count=4,
        inherited_lease_fds=(), process_joule_present=True, full_xcode_present=False,
    )
    assert value["execution_ready"] is False
    assert value["collection_started"] is False
    assert value["execute_surface_exposed"] is True
    assert value["collector_normalizer_default_off_preserved"] is True
    assert value["powermetrics_energy_impact_eligible"] is False
    assert any("full-Xcode" in blocker for blocker in value["blockers"])
    dry = executor.dry_run("spec")
    assert dry["would_execute"] is False
    assert dry["would_start_collectors"] is False
    assert dry["would_open_heavy_lease"] is False


def test_signed_authority_requires_current_hash_bound_sshsig(tmp_path: pathlib.Path) -> None:
    envelope = _authority(tmp_path)
    seen: list[bytes] = []

    def verifier(_envelope: dict, payload: bytes) -> tuple[bool, str]:
        seen.append(payload)
        return True, "verified"

    errors = executor.validate_signed_authority(
        envelope, expected_subject="normalizer", expected_kind="capability",
        required_claims=executor.AUTHORITY_SPECS["normalizer_capability"][2],
        expected_release_build_sha256="b" * 64, now_unix_ns=50,
        registry=authority_root.load_default_registry(),
        expected_abi_sha256=executor.ABI_HASHES["normalizer"],
        signature_verifier=verifier,
    )
    assert errors == []
    assert seen == [appendix_contract.canonical_bytes(envelope["receipt"])]

    forged = copy.deepcopy(envelope)
    forged["receipt"]["claims"].append("invented-claim")
    errors = executor.validate_signed_authority(
        forged, expected_subject="normalizer", expected_kind="capability",
        required_claims=executor.AUTHORITY_SPECS["normalizer_capability"][2],
        expected_release_build_sha256="b" * 64, now_unix_ns=50,
        registry=authority_root.load_default_registry(),
        expected_abi_sha256=executor.ABI_HASHES["normalizer"],
        signature_verifier=verifier,
    )
    assert any("receipt_sha256 mismatch" in error for error in errors)
    assert any("envelope_sha256 mismatch" in error for error in errors)


def test_authority_rejects_expiry_wrong_abi_and_failed_signature(tmp_path: pathlib.Path) -> None:
    expired = _authority(tmp_path, expires=40)
    errors = executor.validate_signed_authority(
        expired, expected_subject="normalizer", expected_kind="capability",
        required_claims=executor.AUTHORITY_SPECS["normalizer_capability"][2],
        expected_release_build_sha256="b" * 64, now_unix_ns=50,
        registry=authority_root.load_default_registry(),
        expected_abi_sha256="d" * 64,
        signature_verifier=lambda _e, _p: (False, "bad signature"),
    )
    assert any("not currently valid" in error for error in errors)
    assert any("different argv contract" in error for error in errors)
    # Structural failures short-circuit signature execution.  With a fresh,
    # otherwise-valid receipt the detached signature itself is authoritative.
    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    fresh = _authority(fresh_root)
    errors = executor.validate_signed_authority(
        fresh, expected_subject="normalizer", expected_kind="capability",
        required_claims=executor.AUTHORITY_SPECS["normalizer_capability"][2],
        expected_release_build_sha256="b" * 64, now_unix_ns=50,
        registry=authority_root.load_default_registry(),
        expected_abi_sha256=executor.ABI_HASHES["normalizer"],
        signature_verifier=lambda _e, _p: (False, "operator signature mismatch"),
    )
    assert any("SSHSIG verification failed" in error for error in errors)


def test_authority_set_cannot_omit_privilege_or_attribution() -> None:
    errors, receipts = executor.validate_authorities(
        {}, release_build_sha256="b" * 64, now_unix_ns=1,
        registry=authority_root.load_default_registry(),
        expected_host_hardware_uuid_sha256="a" * 64,
        verify_files=False, signature_verifier=lambda _e, _p: (True, ""),
    )
    assert receipts == {}
    assert errors == ["authority set is incomplete or unexpected"]


def test_envelope_cannot_supply_its_own_allowed_signers(tmp_path: pathlib.Path) -> None:
    envelope = _authority(tmp_path)
    forged_allowed = _file(tmp_path / "attacker.allowed", b"attacker ssh-ed25519 AAAAattack\n")
    envelope["allowed_signers"] = forged_allowed
    envelope = executor._stamp(envelope, "envelope_sha256")
    verifier_called = False

    def verifier(_envelope: dict, _payload: bytes) -> tuple[bool, str]:
        nonlocal verifier_called
        verifier_called = True
        return True, "self-authorized"

    errors = executor.validate_signed_authority(
        envelope, expected_subject="normalizer", expected_kind="capability",
        required_claims=executor.AUTHORITY_SPECS["normalizer_capability"][2],
        expected_release_build_sha256="b" * 64, now_unix_ns=50,
        registry=authority_root.load_default_registry(),
        expected_abi_sha256=executor.ABI_HASHES["normalizer"],
        signature_verifier=verifier,
    )
    assert any("non-pinned trust root" in error for error in errors)
    assert verifier_called is False


def _full_authorities(
    tmp_path: pathlib.Path, host_sha256: str,
    probe_identity: dict | None = None,
) -> dict:
    registry = authority_root.load_default_registry()
    allowed = authority_root.allowed_signers_identity(registry)
    probe_identity = probe_identity or {
        "path": "/absolute/release-probe", "sha256": "4" * 64, "size_bytes": 1,
    }
    binary_by_subject = {
        "xctrace": {"path": str(executor.FULL_XCODE_XCTRACE), "sha256": "2" * 64, "size_bytes": 1},
        "normalizer": authority_root.trusted_normalizer_identity()["file"],
        "process-joule": probe_identity,
        "release-probe": probe_identity,
        "release-orchestrator": {"path": "/absolute/release-orchestrator", "sha256": "5" * 64, "size_bytes": 1},
    }
    output = {}
    for key, (subject, kind, required) in executor.AUTHORITY_SPECS.items():
        claims = set(required)
        if key == "xctrace_capability":
            claims.add("graceful_exit_code=0")
        if key == "process_joule_capability":
            provenance = executor.process_joule.library_provenance()
            claims.update({
                f"dyld_shared_cache_uuid={provenance['dyld_shared_cache_uuid']}",
                f"os_build={provenance['os_build']}", f"machine={provenance['machine']}",
                f"proc_libversion_major={provenance['proc_libversion_major']}",
                f"proc_libversion_minor={provenance['proc_libversion_minor']}",
                f"resource_header_sha256={provenance['resource_header']['sha256']}",
                f"libproc_header_sha256={provenance['libproc_header']['sha256']}",
                f"struct_layout_sha256={provenance['struct_layout_sha256']}",
                f"library_provenance_sha256={provenance['provenance_sha256']}",
            })
        if key == "device_identity":
            claims.update({
                "registry_id=metal-1", "name=M3 Ultra", "architecture=arm64",
                "os_build=24A", "driver_build=metal-1",
            })
        if key == "inherited_lease":
            claims.update({
                "lock_device=1", "lock_inode=2", f"parent_pid={os.getppid()}",
                "observer_state_sha256=" + "6" * 64,
                "release_boundary_attestation_sha256=" + "7" * 64,
            })
        abi = executor.ABI_HASHES.get(subject, executor.ABI_HASHES["device-probe"])
        receipt = executor._stamp({
            "schema": executor.AUTHORITY_SCHEMA, "receipt_kind": kind,
            "subject": subject, "host_hardware_uuid_sha256": host_sha256,
            "binary": binary_by_subject[subject], "command_abi_sha256": abi,
            "claims": sorted(claims), "issued_at_unix_ns": 10,
            "expires_at_unix_ns": 100, "release_build_sha256": "b" * 64,
        }, "receipt_sha256")
        signature = _file(tmp_path / f"{key}.sig")
        output[key] = executor._stamp({
            "schema": executor.ENVELOPE_SCHEMA, "receipt": receipt,
            "signer_identity": registry["signer_identity"],
            "signature_namespace": registry["sshsig_namespace"],
            "allowed_signers": allowed, "detached_signature": signature,
        }, "envelope_sha256")
    return output


def test_authority_host_identity_must_match_live_measurement(tmp_path: pathlib.Path) -> None:
    errors, _receipts = executor.validate_authorities(
        _full_authorities(tmp_path, "a" * 64), release_build_sha256="b" * 64,
        now_unix_ns=50, registry=authority_root.load_default_registry(),
        expected_host_hardware_uuid_sha256="f" * 64, verify_files=False,
        signature_verifier=lambda _e, _p: (True, "verified"),
    )
    assert "authority receipts do not match the live measured host UUID" in errors


def test_probe_argv_requires_exact_process_joule_provenance_for_both_kinds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tmp_path / "process_joule_provenance.json"
    raw = tmp_path / "raw.json"
    cell = {
        "id": "cell", "model": "model", "tensor_family": "tensor",
    }
    monkeypatch.setattr(
        executor.tq_runtime_matrix, "build_matrix", lambda: {"cells": [cell]},
    )
    device = {
        "kind": "device", "release": {"release_build": {"source_base_commit": "a" * 40}},
        "workload": {
            "probe": {"path": "/probe"}, "cell_id": "cell",
            "artifact": {"path": "/artifact"}, "runtime_path": "stored",
            "warmups": 1, "trials": 2, "residual_artifact": None,
        },
    }
    device_argv = executor._probe_argv(device, raw, provenance)
    assert device_argv.count("--process-joule-provenance") == 1
    assert device_argv[device_argv.index("--process-joule-provenance") + 1] == str(provenance)

    monkeypatch.setattr(
        executor.spec_tq_runner, "_cells",
        lambda _runtime, _label: ({"id": "parity"}, {"id": "curve"}, set()),
    )
    spec = {
        "kind": "spec", "release": {"release_build": {"source_base_commit": "a" * 40}},
        "workload": {
            "probe": {"path": "/probe"}, "weights": {"path": "/weights"},
            "artifact": {"path": "/artifact"}, "prompts": {"path": "/prompts"},
            "runtime_path": "stored", "generated_tokens": 1,
            "warmups_per_batch": 1, "repeats_per_batch": 2, "label": "TEST",
        },
    }
    spec_argv = executor._probe_argv(spec, raw, provenance)
    assert spec_argv.count("--process-joule-provenance") == 1
    assert spec_argv[spec_argv.index("--process-joule-provenance") + 1] == str(provenance)


def test_stale_process_joule_provenance_fails_authority_binding(
    tmp_path: pathlib.Path,
) -> None:
    authorities = _full_authorities(tmp_path, "a" * 64)
    receipts = {key: envelope["receipt"] for key, envelope in authorities.items()}
    provenance = executor.process_joule.library_provenance()
    assert executor._process_provenance_claim_errors(provenance, receipts) == []
    stale = copy.deepcopy(provenance)
    stale["provenance_sha256"] = "f" * 64
    errors = executor._process_provenance_claim_errors(stale, receipts)
    assert any("library_provenance_sha256" in error for error in errors)


def test_operator_sealed_result_validates_and_core_tamper_fails_closed(
    tmp_path: pathlib.Path,
) -> None:
    request, execution, evidence, draft = _result_chain(tmp_path)
    signed = _signed_result(tmp_path, draft)
    verified = lambda _envelope, _payload: (True, "verified")
    sealed = executor.build_sealed_evidence(
        request=request, execution_receipt=execution, evidence=evidence,
        signed_result_attestation=signed, sealed_at_unix_ns=50,
        verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    )
    assert executor.validate_sealed_evidence(
        sealed, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    ) == []

    failed_result_signature = executor.validate_sealed_evidence(
        sealed, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=lambda _envelope, _payload: (False, "forged"),
    )
    assert any("result SSHSIG verification failed" in error for error in failed_result_signature)

    forged = copy.deepcopy(sealed)
    forged["core_evidence"]["cell_id"] = "forged-cell"
    forged = executor._stamp(forged, "sealed_evidence_sha256")
    errors = executor.validate_sealed_evidence(
        forged, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    )
    assert any("exact core evidence" in error or "dynamic execution bindings" in error for error in errors)

    attacker_allowed_path = tmp_path / "attacker.allowed"
    attacker_allowed_path.write_bytes(b"attacker ssh-ed25519 AAAAattack\n")
    substituted = copy.deepcopy(sealed)
    substituted["signed_result_attestation"]["allowed_signers"] = (
        physical_counter_attestation.file_identity(attacker_allowed_path)
    )
    substituted["signed_result_attestation"] = executor._stamp(
        substituted["signed_result_attestation"], "envelope_sha256",
    )
    substituted = executor._stamp(substituted, "sealed_evidence_sha256")
    result_verifier_called = False

    def should_not_verify(_envelope, _payload):  # type: ignore[no-untyped-def]
        nonlocal result_verifier_called
        result_verifier_called = True
        return True, "attacker"

    errors = executor.validate_sealed_evidence(
        substituted, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=should_not_verify,
    )
    assert any("non-pinned trust root" in error for error in errors)
    assert result_verifier_called is False


def test_xctrace_export_evidence_exact_bindings_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    request, execution, evidence, _draft = _result_chain(tmp_path)
    value = evidence["xctrace_export_evidence"]
    assert executor._xctrace_export_evidence_errors(
        value, request=request, execution_receipt=execution,
        core_evidence=evidence, verify_files=False,
    ) == []

    mutations = (
        lambda row: row["profile_identity"].update({"path": "/tmp/attacker-profile"}),
        lambda row: row.update({"xctrace_authority_chain_sha256": "f" * 64}),
        lambda row: row["lease"].update({"inode": 999}),
        lambda row: row.update({"probe_argv_sha256": "f" * 64}),
        lambda row: row["export_identities"].pop("counters"),
        lambda row: row.update({"adapter_receipt_sha256": "f" * 64}),
        lambda row: row.update({"physical_evidence_eligible": False}),
    )
    for mutate in mutations:
        forged = copy.deepcopy(value)
        mutate(forged)
        forged = executor._stamp(forged, "xctrace_export_evidence_sha256")
        assert executor._xctrace_export_evidence_errors(
            forged, request=request, execution_receipt=execution,
            core_evidence=evidence, verify_files=False,
        )

    assert executor._xctrace_export_evidence_errors(
        {"schema": []}, request=request, execution_receipt=execution,
        core_evidence=evidence, verify_files=False,
    )


def test_result_signature_file_tamper_short_circuits_sshsig(
    tmp_path: pathlib.Path,
) -> None:
    request, execution, evidence, draft = _result_chain(tmp_path)
    signed = _signed_result(tmp_path, draft)
    pathlib.Path(signed["detached_signature"]["path"]).write_bytes(b"tampered-signature")
    verifier_called = False

    def verifier(_envelope, _payload):  # type: ignore[no-untyped-def]
        nonlocal verifier_called
        verifier_called = True
        return True, "should not run"

    errors = executor.validate_result_envelope(
        signed, request=request, execution_receipt=execution, evidence=evidence,
        verify_files=True, signature_verifier=verifier,
    )
    assert any("differs from the immutable file" in error for error in errors)
    assert verifier_called is False


def test_sealed_validator_malformed_nested_release_never_raises(
    tmp_path: pathlib.Path,
) -> None:
    request, execution, evidence, draft = _result_chain(tmp_path)
    signed = _signed_result(tmp_path, draft)
    verified = lambda _envelope, _payload: (True, "verified")
    sealed = executor.build_sealed_evidence(
        request=request, execution_receipt=execution, evidence=evidence,
        signed_result_attestation=signed, sealed_at_unix_ns=50,
        verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    )
    malformed = copy.deepcopy(sealed)
    malformed["request"]["release"] = {}
    malformed["request"] = executor._stamp(malformed["request"], "request_sha256")
    malformed = executor._stamp(malformed, "sealed_evidence_sha256")
    errors = executor.validate_sealed_evidence(
        malformed, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    )
    assert errors
    assert any("release build" in error or "inputs are invalid" in error for error in errors)


def test_result_draft_sign_seal_cli_and_dynamic_forgery_rejection(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, execution, evidence, _draft = _result_chain(tmp_path)
    request_path = tmp_path / "request.json"
    evidence_path = tmp_path / "evidence.json"
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    draft_path = tmp_path / "draft.json"
    assert executor.main([
        "--draft-result", str(request_path),
        "--execution-receipt", str(execution_path),
        "--evidence", str(evidence_path), "--output", str(draft_path),
    ]) == 0
    drafted = json.loads(draft_path.read_text(encoding="utf-8"))
    assert executor.validate_result_attestation(
        drafted, request=request, execution_receipt=execution, evidence=evidence,
    ) == []

    forged = copy.deepcopy(drafted)
    forged["core_evidence_sha256"] = "f" * 64
    forged = executor._stamp(forged, "result_attestation_sha256")
    forged_path = tmp_path / "forged-draft.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    signer_called = False

    def fake_sign(attestation, *, private_key, detached_signature_output,
                  envelope_output, registry=None):  # type: ignore[no-untyped-def]
        nonlocal signer_called
        signer_called = True
        detached_signature_output.write_bytes(b"test-result-signature")
        active_registry = authority_root.load_default_registry()
        signed = executor._stamp({
            "schema": executor.RESULT_ENVELOPE_SCHEMA,
            "attestation": attestation,
            "signer_identity": active_registry["signer_identity"],
            "signature_namespace": executor.RESULT_SSHSIG_NAMESPACE,
            "allowed_signers": authority_root.allowed_signers_identity(active_registry),
            "detached_signature": physical_counter_attestation.file_identity(
                detached_signature_output,
            ),
        }, "envelope_sha256")
        envelope_output.write_text(json.dumps(signed), encoding="utf-8")
        return signed

    monkeypatch.setattr(authority_root, "sign_result_attestation", fake_sign)
    with pytest.raises(executor.EvidenceError, match="refused invalid result draft"):
        executor.sign_result_files(
            request_path=request_path, execution_receipt_path=execution_path,
            evidence_path=evidence_path, draft_path=forged_path,
            private_key=tmp_path / "operator-key",
            detached_signature_output=tmp_path / "forged.sig",
            envelope_output=tmp_path / "forged.json",
        )
    assert signer_called is False

    signed_path = tmp_path / "signed.json"
    signature_path = tmp_path / "signed.sig"
    assert executor.main([
        "--sign-result", str(draft_path), "--request", str(request_path),
        "--execution-receipt", str(execution_path), "--evidence", str(evidence_path),
        "--private-key", str(tmp_path / "operator-key"),
        "--signature-output", str(signature_path),
        "--envelope-output", str(signed_path),
    ]) == 0
    assert signer_called is True
    assert signature_path.is_file()

    original_seal = executor.seal_result_files
    verified = lambda _envelope, _payload: (True, "verified")

    def fake_seal(**kwargs):  # type: ignore[no-untyped-def]
        return original_seal(
            **kwargs, verify_files=False,
            authority_signature_verifier=verified,
            result_signature_verifier=verified,
        )

    monkeypatch.setattr(executor, "seal_result_files", fake_seal)
    sealed_path = tmp_path / "sealed.json"
    assert executor.main([
        "--seal-result", str(signed_path), "--request", str(request_path),
        "--execution-receipt", str(execution_path), "--evidence", str(evidence_path),
        "--output", str(sealed_path),
    ]) == 0
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert sealed["core_evidence"] == evidence
    assert executor.validate_sealed_evidence(
        sealed, verify_files=False, authority_signature_verifier=verified,
        result_signature_verifier=verified,
    ) == []


def test_inherited_fd_without_exclusive_flock_is_rejected(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "heavy.lock"
    held = lock.open("a+")
    observer_sha = "6" * 64
    boundary_sha = "7" * 64
    claims = [
        f"lock_device={os.fstat(held.fileno()).st_dev}",
        f"lock_inode={os.fstat(held.fileno()).st_ino}",
        f"parent_pid={os.getppid()}", f"observer_state_sha256={observer_sha}",
        f"release_boundary_attestation_sha256={boundary_sha}",
    ]
    receipt = {"claims": claims}
    env = {ram_scheduler.HEAVY_LEASE_FD_ENV: str(held.fileno())}
    try:
        _fd, errors = executor._validate_inherited_lease(
            receipt, env=env, expected_observer_sha256=observer_sha,
            expected_boundary_sha256=boundary_sha, lock_path=lock,
        )
        assert any("does not own an exclusive flock" in error for error in errors)
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _fd, errors = executor._validate_inherited_lease(
            receipt, env=env, expected_observer_sha256=observer_sha,
            expected_boundary_sha256=boundary_sha, lock_path=lock,
        )
        assert errors == []
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


class NeverStartBackend(executor.ProcessBackend):
    def __init__(self) -> None:
        self.starts = 0

    def spawn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.starts += 1
        raise AssertionError("no process may start")


def test_missing_xcode_exits_before_any_process_even_when_nonroot(tmp_path: pathlib.Path) -> None:
    request = _minimal_request(tmp_path)
    path = tmp_path / "request.json"
    _write_request(path, request)
    backend = NeverStartBackend()
    with pytest.raises(executor.AdmissionBlocked, match="full-Xcode"):
        executor.execute(
            path, acknowledgement=request["request_sha256"], backend=backend, euid=501,
        )
    assert backend.starts == 0


def test_cli_maps_admission_block_to_exit_75(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _minimal_request(tmp_path)
    path = tmp_path / "request.json"
    _write_request(path, request)

    def blocked(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise executor.AdmissionBlocked("synthetic admission")

    monkeypatch.setattr(executor, "execute", blocked)
    assert executor.main([
        "--execute", str(path),
        "--acknowledge-request-sha256", request["request_sha256"],
    ]) == 75


def test_missing_full_xcode_exits_before_any_process(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _minimal_request(tmp_path)
    path = tmp_path / "request.json"
    _write_request(path, request)
    monkeypatch.setattr(executor, "FULL_XCODE_XCTRACE", tmp_path / "missing-xctrace")
    backend = NeverStartBackend()
    with pytest.raises(executor.AdmissionBlocked, match="full-Xcode"):
        executor.execute(
            path, acknowledgement=request["request_sha256"], backend=backend, euid=0,
        )
    assert backend.starts == 0


@dataclass
class FakeProcess:
    name: str
    pid: int
    alive: bool = True
    barrier_reader: int | None = None


class OrderedBackend(executor.ProcessBackend):
    def __init__(self, raw_by_name: dict[str, pathlib.Path]) -> None:
        self.events: list[str] = []
        self.raw_by_name = raw_by_name
        self.next_pid = 4100

    def spawn(self, name, argv, *, env, pass_fds, stdout_path, stderr_path):  # type: ignore[no-untyped-def]
        self.events.append(f"spawn:{name}")
        self.next_pid += 1
        reader = os.dup(pass_fds[-1]) if name == "probe" else None
        if name in self.raw_by_name:
            self.raw_by_name[name].write_bytes(b"capture-started")
        return FakeProcess(name=name, pid=self.next_pid, barrier_reader=reader)

    def alive(self, process):  # type: ignore[no-untyped-def]
        return process.alive

    def wait(self, process, timeout):  # type: ignore[no-untyped-def]
        self.events.append(f"wait:{process.name}")
        if process.barrier_reader is not None:
            token = os.read(process.barrier_reader, 1)
            assert token == (b"G" if process.alive else b"")
            os.close(process.barrier_reader)
            process.barrier_reader = None
        process.alive = False
        return 0

    def interrupt(self, process):  # type: ignore[no-untyped-def]
        self.events.append(f"interrupt:{process.name}")

    def terminate(self, process):  # type: ignore[no-untyped-def]
        self.events.append(f"terminate:{process.name}")
        process.alive = False

    def kill(self, process):  # type: ignore[no-untyped-def]
        self.events.append(f"kill:{process.name}")
        process.alive = False


def _orchestration_paths(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "probe_stdout": tmp_path / "probe.out",
        "probe_stderr": tmp_path / "probe.err",
        "xctrace_raw": tmp_path / "xc.raw",
        "xctrace_stdout": tmp_path / "xc.out",
        "xctrace_stderr": tmp_path / "xc.err",
    }


def test_orchestration_starts_xctrace_before_readiness_and_probe(
    tmp_path: pathlib.Path,
) -> None:
    paths = _orchestration_paths(tmp_path)
    backend = OrderedBackend({"xctrace": paths["xctrace_raw"]})

    def readiness(_backend, process, path):  # type: ignore[no-untyped-def]
        backend.events.append(f"ready:{process.name}")
        assert path.read_bytes() == b"capture-started"
        return {"ready_at_unix_ns": 1, "ready_at_continuous_ns": 2}

    lease = (tmp_path / "lease").open("wb")
    try:
        result = executor.orchestrate_capture(
            backend=backend, probe_argv=["/absolute/probe", "--output", "/tmp/raw"],
            probe_env={}, lease_fd=lease.fileno(), output_paths=paths,
            probe_timeout_seconds=1, readiness_wait=readiness,
        )
    finally:
        lease.close()
    assert backend.events[:3] == ["spawn:probe", "spawn:xctrace", "ready:xctrace"]
    assert backend.events.count("spawn:probe") == 1
    assert backend.events.index("ready:xctrace") < backend.events.index("wait:probe")
    assert result.probe_returncode == 0
    assert set(result.collector_argv) == {"xctrace"}


def test_readiness_failure_never_releases_probe_barrier(tmp_path: pathlib.Path) -> None:
    paths = _orchestration_paths(tmp_path)
    backend = OrderedBackend({"xctrace": paths["xctrace_raw"]})

    def fail_readiness(_backend, process, _path):  # type: ignore[no-untyped-def]
        backend.events.append(f"ready-fail:{process.name}")
        raise executor.AdmissionBlocked("synthetic readiness failure")

    lease = (tmp_path / "lease").open("wb")
    try:
        with pytest.raises(executor.AdmissionBlocked, match="synthetic readiness"):
            executor.orchestrate_capture(
                backend=backend, probe_argv=["/absolute/probe"], probe_env={},
                lease_fd=lease.fileno(), output_paths=paths,
                probe_timeout_seconds=1, readiness_wait=fail_readiness,
            )
    finally:
        lease.close()
    # The only probe wait follows an explicit terminate in finally.  The fake
    # reader observes EOF, never the release token, and therefore raises if the
    # implementation accidentally claims the barrier was released.
    assert "terminate:probe" in backend.events
    assert backend.events.index("terminate:probe") < backend.events.index("wait:probe")


def test_barrier_child_rejects_wrong_token_without_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"X")
    os.close(write_fd)
    called = False

    def forbidden(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("execve must not be called")

    monkeypatch.setattr(executor.os, "execve", forbidden)
    assert executor._barrier_child(str(read_fd), ["/absolute/probe"]) == executor.EXIT_BLOCKED
    assert called is False


def test_capture_sealing_and_trace_tree_freeze_are_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "capture"
    source.write_bytes(b"raw-counter-bytes")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(executor.EvidenceError, match="not a regular file"):
        executor.seal_capture(link, tmp_path / "bad.sealed")
    identity = executor.seal_capture(source, tmp_path / "good.sealed")
    assert identity["sha256"] == __import__("hashlib").sha256(b"raw-counter-bytes").hexdigest()
    assert pathlib.Path(identity["path"]).stat().st_mode & 0o222 == 0

    trace = tmp_path / "metal.trace"
    trace.mkdir()
    nested = trace / "nested"
    nested.mkdir()
    (trace / "b").write_bytes(b"B")
    (nested / "a").write_bytes(b"A")
    frozen = executor.freeze_trace_tree(trace)
    assert executor.xctrace_adapter.trace_tree_identity(
        trace, require_immutable=True,
    ) == frozen
    assert stat.S_IMODE(trace.stat().st_mode) == 0o555
    assert stat.S_IMODE(nested.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in (
        trace / "b", nested / "a",
    ))
    with pytest.raises(executor.EvidenceError, match="sealable capture"):
        executor.seal_capture(trace, tmp_path / "forbidden.tar")

    unsafe = tmp_path / "unsafe.trace"
    unsafe.mkdir()
    (unsafe / "link").symlink_to(source)
    with pytest.raises(executor.EvidenceError, match="symlink"):
        executor.freeze_trace_tree(unsafe)

    hardlinked = tmp_path / "hardlinked.trace"
    hardlinked.mkdir()
    (hardlinked / "one").write_bytes(b"same-inode")
    os.link(hardlinked / "one", hardlinked / "two")
    with pytest.raises(executor.EvidenceError, match="hard-linked"):
        executor.freeze_trace_tree(hardlinked)


def test_request_rejects_output_escape_and_default_mutation(tmp_path: pathlib.Path) -> None:
    request = _minimal_request(tmp_path)
    request["output_directory"] = str(tmp_path / "escape")
    request["runtime_default_mutation_requested"] = True
    request = executor._stamp(request, "request_sha256")
    errors = executor._validate_request_structure(request)
    assert any("escapes" in error for error in errors)
    assert any("runtime default" in error for error in errors)


def test_release_parent_binding_is_hash_chained_and_deduplicated() -> None:
    receipt = appendix_contract.stamp_receipt({
        "bindings": {"parent_receipt_sha256": []},
        "status": "synthetic",
    })
    release = {
        "corpus_index": {"index_sha256": "a" * 64},
        "release_build": {"receipt_sha256": "b" * 64},
        "source_manifest": {"manifest_sha256": "c" * 64},
    }
    bound = executor._bind_release_parents(
        receipt, release=release, extra=["d" * 64, "a" * 64],
        parity_parent="e" * 64,
    )
    bindings = bound["bindings"]
    assert bindings["parent_receipt_sha256"] == ["a" * 64, "b" * 64, "d" * 64, "e" * 64]
    assert bindings["source_manifest_sha256"] == "c" * 64
    unstamped = copy.deepcopy(bound)
    claimed = unstamped.pop("receipt_sha256")
    assert claimed == appendix_contract.canonical_sha256(unstamped)
