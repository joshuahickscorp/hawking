from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import sys
import time

import pytest


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import doctor_v5_higher_tier_authority as authority  # noqa: E402
import doctor_v5_higher_tier_scaffold as higher  # noqa: E402
import doctor_v5_post120_acceleration_scaffold as acceleration  # noqa: E402
from higher_tier_authority_fixtures import (  # noqa: E402
    VERIFIED, ephemeral_trust, fake_operator_seal, make_core_manifest,
)


def _admission(manifest: dict) -> dict:
    return higher.build_admission_plan(
        manifest, total_memory_bytes=100_000_000_000,
        process_budget_bytes=80_000_000_000,
        control_resident_bytes=5_000_000_000,
        safety_margin_bytes=10_000_000_000,
        logical_cpu_count=20, maximum_lanes=8,
    )


def test_ephemeral_operator_crypto_roundtrip_and_exact_10x4_integration(
    tmp_path: Path,
) -> None:
    manifest = make_core_manifest(tmp_path)
    issued = time.time_ns()
    with ephemeral_trust(tmp_path) as key:
        draft = authority.build_manifest_attestation(
            manifest, issued_at_unix_ns=issued, valid_seconds=60,
        )
        signature = tmp_path / "manifest.sshsig"
        envelope_path = tmp_path / "manifest-envelope.json"
        envelope = authority.sign_manifest_attestation(
            draft, manifest=manifest, private_key=key,
            detached_signature_output=signature, envelope_output=envelope_path,
            now_unix_ns=issued + 1,
        )
        sealed = authority.build_sealed_manifest(
            manifest=manifest, signed_attestation=envelope,
            sealed_at_unix_ns=issued + 2,
        )
        assert authority.validate_sealed_manifest(
            sealed, now_unix_ns=issued + 3,
        ) == []
        core, errors = higher.unwrap_exact_source_manifest(sealed)
        assert errors == [] and core == manifest
        admission = _admission(manifest)
        plan = acceleration.build_higher_tier_acceleration_plan(
            sealed, admission, created_at="2026-07-15T00:00:00+00:00",
        )
        assert acceleration.validate_higher_tier_acceleration_plan(
            plan, sealed, admission,
        ) == []
        assert len(plan["cells"]) == 40
        assert plan["parents"]["sealed_manifest_sha256"] \
            == sealed["sealed_manifest_sha256"]
        assert all(
            row["manifest_attestation_sha256"]
            == draft["attestation_sha256"] for row in plan["cells"]
        )
        assert stat.S_IMODE(signature.stat().st_mode) == 0o444
        assert stat.S_IMODE(envelope_path.stat().st_mode) == 0o444


def test_raw_reviewed_self_hashed_manifest_never_builds_exact_matrix(
    tmp_path: Path,
) -> None:
    manifest = make_core_manifest(tmp_path)
    errors = higher.validate_source_manifest(
        manifest, require_exact_wiring=True, verify_files=True,
    )
    assert any("operator-signed sealed" in error for error in errors)
    with pytest.raises(acceleration.AccelerationScaffoldError, match="operator-signed"):
        acceleration.build_higher_tier_acceleration_plan(
            manifest, _admission(manifest),
        )


def test_false_121b_in_4k_and_unknown_codec_are_rejected() -> None:
    logical = 121_000_000_001
    tensor = {
        "tensor_key": "false.weight", "role": "model_parameter",
        "logical_dtype": "BF16", "storage_encoding": "invented-121b-in-4k",
        "shape": [logical], "logical_parameters": logical,
        "stored_parameters": logical, "packing_overhead_bytes": 0,
        "stored_bytes": 4096, "source_id": "s",
        "absolute_byte_range": [0, 4096], "range_sha256": "a" * 64,
    }
    document = authority._stamp({
        "schema": authority.PARAMETER_AUTHORITY_SCHEMA,
        "model": {
            "label": "False-121B", "hf_id_or_source_id": "false/121b",
            "family": "false", "architecture_kind": "dense",
        },
        "tensors": [tensor], "logical_parameters": logical,
        "stored_parameters": logical, "tensor_count": 1,
        "tensor_layout_sha256": authority.canonical_sha256([tensor]),
        "parameter_ranges_sha256": authority.canonical_sha256([{
            "tensor_key": "false.weight", "source_id": "s",
            "absolute_byte_range": [0, 4096], "stored_bytes": 4096,
            "range_sha256": "a" * 64,
        }]),
    }, "authority_sha256")
    summary, _tensors, errors = authority.validate_parameter_authority(
        document,
        model={
            "label": "False-121B", "hf_id_or_source_id": "false/121b",
            "family": "false", "architecture_kind": "dense",
        },
        sources={"s": {"bytes": 4096}},
    )
    assert summary is None
    assert any("storage encoding" in error for error in errors)


def test_tensor_ranges_must_be_nonoverlapping_and_exactly_cover_sources() -> None:
    count = 80
    base = {
        "role": "model_parameter", "logical_dtype": "BF16",
        "storage_encoding": "tq-0.1bpw", "shape": [count],
        "logical_parameters": count, "stored_parameters": count,
        "packing_overhead_bytes": 0, "stored_bytes": 1,
        "source_id": "s", "absolute_byte_range": [0, 1],
        "range_sha256": "a" * 64,
    }
    tensors = [dict(base, tensor_key="a"), dict(base, tensor_key="b")]
    range_rows = [{
        "tensor_key": row["tensor_key"], "source_id": "s",
        "absolute_byte_range": [0, 1], "stored_bytes": 1,
        "range_sha256": "a" * 64,
    } for row in tensors]
    document = authority._stamp({
        "schema": authority.PARAMETER_AUTHORITY_SCHEMA,
        "model": {
            "label": "M", "hf_id_or_source_id": "m", "family": "f",
            "architecture_kind": "dense",
        },
        "tensors": tensors, "logical_parameters": 160,
        "stored_parameters": 160, "tensor_count": 2,
        "tensor_layout_sha256": authority.canonical_sha256(tensors),
        "parameter_ranges_sha256": authority.canonical_sha256(range_rows),
    }, "authority_sha256")
    summary, _rows, errors = authority.validate_parameter_authority(
        document,
        model={
            "label": "M", "hf_id_or_source_id": "m", "family": "f",
            "architecture_kind": "dense",
        },
        sources={"s": {"bytes": 1}},
    )
    assert summary is None
    assert any("overlap" in error for error in errors)


def test_malformed_nan_and_non_object_rows_never_raise(tmp_path: Path) -> None:
    manifest = make_core_manifest(tmp_path)
    malformed = copy.deepcopy(manifest)
    malformed["sources"].append("not-an-object")
    malformed["work_units"][0]["source_ranges"].append(float("nan"))
    bindings, errors = authority.inspect_core_manifest(malformed)
    assert bindings is None and errors

    parameter_path = Path(manifest["model"]["parameter_authority"]["artifact"]["path"])
    parameter = json.loads(parameter_path.read_text(encoding="utf-8"))
    parameter["tensors"][0]["shape"] = [float("nan")]
    summary, rows, errors = authority.validate_parameter_authority(
        parameter, model=manifest["model"],
        sources={row["source_id"]: row for row in manifest["sources"]},
    )
    assert summary is None and rows == {} and errors


@pytest.mark.parametrize(
    "mutator",
    [
        lambda core: core["architecture_adapter"].__setitem__("adapter_id", "substituted"),
        lambda core: core["tokenizer_binding"].__setitem__("reviewed", False),
        lambda core: core["lifecycle_manifest"].__setitem__(
            "source_deletion_permitted", True
        ),
        lambda core: core["transport_manifest"].__setitem__("immutable", False),
        lambda core: core["sources"][0].__setitem__("immutable_version", "substituted"),
        lambda core: core["work_units"][0]["source_ranges"][0].__setitem__(
            "range_sha256", "f" * 64
        ),
    ],
)
def test_core_authority_substitutions_break_operator_attestation(
    tmp_path: Path, mutator,
) -> None:  # type: ignore[no-untyped-def]
    manifest = make_core_manifest(tmp_path)
    sealed = fake_operator_seal(tmp_path, manifest)
    attacked = copy.deepcopy(sealed)
    mutator(attacked["core_manifest"])
    attacked["core_manifest"] = authority._stamp(
        attacked["core_manifest"], "manifest_sha256",
    )
    attacked = authority._stamp(attacked, "sealed_manifest_sha256")
    errors = authority.validate_sealed_manifest(
        attacked, verify_files=True, signature_verifier=VERIFIED,
    )
    assert errors


def test_expiry_future_max_validity_signer_namespace_trust_and_signature_fail(
    tmp_path: Path,
) -> None:
    manifest = make_core_manifest(tmp_path)
    sealed = fake_operator_seal(tmp_path, manifest)
    attestation = sealed["signed_manifest_attestation"]["attestation"]
    assert any("expired" in error for error in authority.validate_sealed_manifest(
        sealed, now_unix_ns=attestation["expires_at_unix_ns"] + 1,
        signature_verifier=VERIFIED,
    ))
    assert any("not yet valid" in error for error in authority.validate_sealed_manifest(
        sealed, now_unix_ns=attestation["issued_at_unix_ns"] - 1,
        signature_verifier=VERIFIED,
    ))
    with pytest.raises(authority.HigherTierAuthorityError, match="validity"):
        authority.build_manifest_attestation(
            manifest, valid_seconds=authority.MAX_VALIDITY_SECONDS + 1,
        )
    attacks = []
    for field, value in (
        ("signer_identity", "attacker"),
        ("signature_namespace", "wrong-namespace"),
    ):
        attacked = copy.deepcopy(sealed)
        attacked["signed_manifest_attestation"][field] = value
        attacked["signed_manifest_attestation"] = authority._stamp(
            attacked["signed_manifest_attestation"], "envelope_sha256",
        )
        attacks.append(attacked)
    trust = copy.deepcopy(sealed)
    trust["signed_manifest_attestation"]["allowed_signers"]["sha256"] = "f" * 64
    trust["signed_manifest_attestation"] = authority._stamp(
        trust["signed_manifest_attestation"], "envelope_sha256",
    )
    attacks.append(trust)
    for attacked in attacks:
        attacked = authority._stamp(attacked, "sealed_manifest_sha256")
        assert authority.validate_sealed_manifest(
            attacked, signature_verifier=VERIFIED,
        )
    forged = authority.validate_sealed_manifest(
        sealed, signature_verifier=lambda *_args: (False, "forged"),
    )
    assert any("SSHSIG verification failed" in error for error in forged)


def test_remote_source_and_range_authority_hashes_are_explicitly_signed(
    tmp_path: Path,
) -> None:
    manifest = make_core_manifest(tmp_path)
    draft = authority.build_manifest_attestation(manifest, valid_seconds=60)
    assert set(draft["remote_source_authority_sha256"]) == {"shard-0"}
    assert len(draft["remote_range_receipt_sha256"]) == 1


def test_immutable_outputs_reject_writable_same_short_write_and_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(b"same")
    writable.chmod(0o600)
    with pytest.raises(authority.HigherTierAuthorityError, match="single-link mode"):
        authority._atomic_bytes(writable, b"same")

    original_write = authority.operator_root.os.write
    monkeypatch.setattr(authority.operator_root.os, "write", lambda *_args: 0)
    short = tmp_path / "short.json"
    with pytest.raises(OSError, match="short write"):
        authority._atomic_bytes(short, b"sealed")
    assert not short.exists()
    monkeypatch.setattr(authority.operator_root.os, "write", original_write)

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_open = authority.operator_root._open_dir_nofollow

    def swapped(path: Path, *, create: bool) -> int:
        if not create:
            return os.open(
                replacement, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        return original_open(path, create=create)

    monkeypatch.setattr(authority.operator_root, "_open_dir_nofollow", swapped)
    output = tmp_path / "swapped.json"
    with pytest.raises(authority.HigherTierAuthorityError, match="parent was replaced"):
        authority._atomic_bytes(output, b"sealed")
    assert not output.exists()
