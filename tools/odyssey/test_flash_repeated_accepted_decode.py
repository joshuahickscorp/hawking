from __future__ import annotations

import importlib.util
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SOURCE = Path(__file__).with_name("flash_repeated_accepted_decode.py")
SPEC = importlib.util.spec_from_file_location("flash_repeated_accepted_decode", SOURCE)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

_TEST_OWNER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
_TEST_OWNER_PUBLIC_KEY = _TEST_OWNER_PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
)


@pytest.fixture(autouse=True)
def _synthetic_canonical_model_lake(monkeypatch, tmp_path: Path) -> None:
    """Synthetic files/key are admitted only by explicit in-process test overrides."""
    monkeypatch.setattr(runner, "MODEL_LAKE_ROOT", tmp_path / "lake")
    monkeypatch.setattr(runner, "_load_machine_admin_owner_public_key", lambda: _TEST_OWNER_PUBLIC_KEY)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(document: dict) -> dict:
    body = {key: value for key, value in document.items() if key != "seal_sha256"}
    body["seal_sha256"] = runner._compact_utf8_sorted_seal(body)
    return body


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _test_source_input_identity(model_root: Path) -> dict:
    return {
        "schema": runner.SOURCE_INPUT_IDENTITY_SCHEMA,
        "model": runner.REPO_ID,
        "pinned_revision": runner.PINNED_REVISION,
        "model_root": str(model_root.resolve()),
        "model_lake_manifest": {
            "path": "/fixture/canonical-flash-manifest.json",
            "sha256": "a" * 64,
            "repo": runner.REPO_ID,
            "revision": runner.PINNED_REVISION,
        },
        "config": {"file": "config.json", "sha256": "b" * 64, "bytes": 1},
        "safetensors_index": {
            "file": "model.safetensors.index.json",
            "sha256": "c" * 64,
            "bytes": 1,
        },
        "tokenizer": {"file": "tokenizer.json", "sha256": "d" * 64, "bytes": 1},
    }


def _native_session(
    references: list[int] | None = None,
    *,
    model_root: Path = runner.DEFAULT_MODEL_ROOT,
    source_input_identity: dict | None = None,
) -> dict:
    references = [6, 7] if references is None else references
    token_ids = [*runner.PROMPT_IDS, *references]
    source_input_identity = (
        _test_source_input_identity(model_root)
        if source_input_identity is None
        else source_input_identity
    )
    document = {
        "schema": runner.NATIVE_SESSION_SCHEMA,
        "status": runner.NATIVE_SESSION_STATUS,
        "model": runner.REPO_ID,
        "pinned_revision": runner.PINNED_REVISION,
        "root": str(model_root.resolve()),
        "source_input_identity": source_input_identity,
        "token_ids": token_ids,
        "prompt_token_ids": list(runner.PROMPT_IDS),
        "reference_generated_token_ids": references,
        "candidate_token_id": references[0],
        "accepted_generation_tokens": len(references),
        "terminal": {
            "candidate_accepted": True,
            "reference_checks": [
                {
                    "generation_index": generation_index,
                    "input_state_index": len(runner.PROMPT_IDS) - 1 + generation_index,
                    "expected_token_id": token_id,
                    "predicted_token_id": token_id,
                    "accepted": True,
                }
                for generation_index, token_id in enumerate(references)
            ],
        },
        "execution": {
            "process_boundary": "one native process",
            "source_reset_or_reprefill": False,
            "source_payload_bytes_read": 8192,
            "state_memory": {
                "total_persistent_bytes": 4096,
                "growth_bytes_per_additional_token": 512,
            },
        },
    }
    return _reseal(document)


def _write_owner_authorization(authorization_path: Path, receipt: Path, document: dict) -> None:
    provider = document["provider"]
    payload = {
        "authorization_scope": runner.OWNER_AUTHORIZATION_SCOPE,
        "reference_contract_sha256": _sha256(receipt),
        "reference_contract_seal_sha256": document["seal_sha256"],
        "provider_implementation_sha256": provider["implementation"]["sha256"],
        "provider_runtime_lock_sha256": provider["runtime_lock"]["sha256"],
        "provider_producer_sha256": provider["producer"]["sha256"],
        "logits_trace_sha256": document["logits_trace"]["sha256"],
        "logits_trace_seal_sha256": document["logits_trace"]["seal_sha256"],
        "shard_ledger_sha256": document["source"]["shard_ledger"]["sha256"],
        "shard_ledger_seal_sha256": document["source"]["shard_ledger"]["seal_sha256"],
        "model_lake_manifest_sha256": document["source"]["model_lake_manifest"]["sha256"],
        "issued_unix_ns": 0,
        "expires_unix_ns": 2**63 - 1,
        "nonce": "test-owner-authorization-nonce-0001",
    }
    signature = _TEST_OWNER_PRIVATE_KEY.sign(runner._canonical_compact_utf8_sorted(payload))
    _write_json(authorization_path, {
        "schema": runner.OWNER_AUTHORIZATION_SCHEMA,
        "owner_public_key_sha256": hashlib.sha256(_TEST_OWNER_PUBLIC_KEY).hexdigest(),
        "payload": payload,
        "signature_ed25519_hex": signature.hex(),
    })


def test_native_source_closure_covers_ple_control_semantics() -> None:
    paths = set(runner._native_source_closure_paths())
    assert SOURCE.resolve() in {path.resolve() for path in paths}
    assert runner.ROOT / "crates/hawking-core/src/flash_ple.rs" in paths
    assert runner.ROOT / "crates/hawking-core/src/model/source_safetensors.rs" in paths
    assert runner.ROOT / "crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs" in paths


def test_repeated_session_requires_sealed_native_schema_and_exact_boundaries():
    references = [6, 7]
    good = _native_session(references)
    source_input_identity = good["source_input_identity"]
    runner._verify_repeated(
        good,
        references,
        model_root=runner.DEFAULT_MODEL_ROOT,
        source_input_identity=source_input_identity,
    )

    bad_seal = json.loads(json.dumps(good))
    bad_seal["execution"]["source_reset_or_reprefill"] = True
    with pytest.raises(ValueError, match="seal"):
        runner._verify_repeated(
            bad_seal,
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    reset = _reseal(bad_seal)
    with pytest.raises(ValueError, match="reset/re-prefill"):
        runner._verify_repeated(
            reset,
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    wrong_schema = json.loads(json.dumps(good))
    wrong_schema["schema"] = "hawking.flash.stateful_complete_token_session.v0"
    with pytest.raises(ValueError, match="schema"):
        runner._verify_repeated(
            _reseal(wrong_schema),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    shifted_input = json.loads(json.dumps(good))
    shifted_input["terminal"]["reference_checks"][1]["input_state_index"] = 0
    with pytest.raises(ValueError, match="token/input boundary"):
        runner._verify_repeated(
            _reseal(shifted_input),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    shifted_tokens = json.loads(json.dumps(good))
    shifted_tokens["token_ids"][-1] = 8
    with pytest.raises(ValueError, match="token IDs or prompt/reference boundary"):
        runner._verify_repeated(
            _reseal(shifted_tokens),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    wrong_source = json.loads(json.dumps(good))
    wrong_source["pinned_revision"] = "unsealed"
    with pytest.raises(ValueError, match="revision"):
        runner._verify_repeated(
            _reseal(wrong_source),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    wrong_root = json.loads(json.dumps(good))
    wrong_root["root"] = "/tmp/not-the-canonical-flash-source"
    with pytest.raises(ValueError, match="canonical ModelLake source"):
        runner._verify_repeated(
            _reseal(wrong_root),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )

    wrong_inputs = json.loads(json.dumps(good))
    wrong_inputs["source_input_identity"]["config"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source input identity"):
        runner._verify_repeated(
            _reseal(wrong_inputs),
            references,
            model_root=runner.DEFAULT_MODEL_ROOT,
            source_input_identity=source_input_identity,
        )


def _write_external_reference_contract(tmp_path: Path) -> tuple[Path, Path, dict, dict[str, Path]]:
    slug = runner.MODEL_LAKE_SLUG
    lake = runner.MODEL_LAKE_ROOT
    source = lake / "specimens" / slug
    manifests = lake / "manifests"
    source.mkdir(parents=True)
    manifests.mkdir()
    vocab_size = 6_000
    config = source / "config.json"
    index = source / "model.safetensors.index.json"
    tokenizer = source / "tokenizer.json"
    shard = source / "model-00001-of-00001.safetensors"
    _write_json(config, {"vocab_size": None, "text_config": {"vocab_size": vocab_size}})
    _write_json(index, {"weight_map": {"model.layers.0.weight": shard.name}})
    _write_json(tokenizer, {"version": "test"})
    shard.write_bytes(b"verified source shard bytes")

    manifest = manifests / f"{slug}.json"
    _write_json(manifest, {
        "repo": runner.REPO_ID,
        "revision": runner.PINNED_REVISION,
        "resolved_sha": runner.PINNED_REVISION,
        "path": str(source.resolve()),
        "bytes": shard.stat().st_size,
        "n_files": 4,
    })
    manifest_sha256 = _sha256(manifest)

    implementation = tmp_path / "reference-provider.py"
    runtime_lock = tmp_path / "reference-runtime.lock"
    producer = tmp_path / "reference-producer.py"
    implementation.write_text("print('independent provider')\n", encoding="utf-8")
    runtime_lock.write_text("reference-runtime==test\n", encoding="utf-8")
    producer.write_text("print('producer')\n", encoding="utf-8")

    ledger = _reseal({
        "schema": runner.SOURCE_SHARD_LEDGER_SCHEMA,
        "status": runner.SOURCE_SHARD_LEDGER_STATUS,
        "source": {
            "model": runner.REPO_ID,
            "pinned_revision": runner.PINNED_REVISION,
            "model_root": str(source.resolve()),
            "model_lake_manifest_sha256": manifest_sha256,
            "safetensors_index_sha256": _sha256(index),
        },
        "verification": {
            "method": "sha256_exact_file_bytes",
            "all_indexed_shards_verified": True,
        },
        "shards": [{
            "path": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": _sha256(shard),
            "verification": "SHA256_EXACT_FILE_BYTES",
        }],
    })
    ledger_path = tmp_path / "verified-source-shards.json"
    _write_json(ledger_path, ledger)

    references = [6, 7]
    trace_steps = []
    payloads: list[Path] = []
    for generation_index, token_id in enumerate(references):
        logits = [-20.0] * vocab_size
        logits[token_id] = 3.0
        payload = tmp_path / f"logits-{generation_index:06d}.f32"
        payload.write_bytes(struct.pack(f"<{vocab_size}f", *logits))
        payloads.append(payload)
        trace_steps.append({
            "generation_index": generation_index,
            "input_token_ids": [*runner.PROMPT_IDS, *references[:generation_index]],
            "expected_token_id": token_id,
            "argmax_token_id": token_id,
            "logits_f32": {
                "path": str(payload),
                "sha256": _sha256(payload),
                "dtype": "F32_LE",
                "elements": vocab_size,
                "bytes": payload.stat().st_size,
            },
        })
    trace = _reseal({
        "schema": runner.LOGITS_TRACE_SCHEMA,
        "status": runner.LOGITS_TRACE_STATUS,
        "source": {
            "model": runner.REPO_ID,
            "pinned_revision": runner.PINNED_REVISION,
            "config_sha256": _sha256(config),
            "safetensors_index_sha256": _sha256(index),
            "tokenizer_sha256": _sha256(tokenizer),
            "model_lake_manifest_sha256": manifest_sha256,
            "shard_ledger_sha256": _sha256(ledger_path),
        },
        "provider_artifacts": {
            "implementation_sha256": _sha256(implementation),
            "runtime_lock_sha256": _sha256(runtime_lock),
            "producer_sha256": _sha256(producer),
        },
        "prompt_token_ids": list(runner.PROMPT_IDS),
        "generated_token_ids": references,
        "sampling": {"method": "greedy_argmax", "do_sample": False, "temperature": 0},
        "dtype": "F32_LE",
        "argmax_tie_break": "lowest_token_id",
        "vocab_size": vocab_size,
        "steps": trace_steps,
    })
    trace_path = tmp_path / "external-logits-trace.json"
    _write_json(trace_path, trace)

    document = _reseal({
        "schema": runner.EXTERNAL_REFERENCE_SCHEMA,
        "status": runner.EXTERNAL_REFERENCE_STATUS,
        "source": {
            "model": runner.REPO_ID,
            "pinned_revision": runner.PINNED_REVISION,
            "config_sha256": _sha256(config),
            "safetensors_index_sha256": _sha256(index),
            "ple_inclusive": True,
            "model_lake_manifest": {
                "path": str(manifest),
                "sha256": manifest_sha256,
                "repo": runner.REPO_ID,
                "revision": runner.PINNED_REVISION,
            },
            "shard_ledger": {
                "path": str(ledger_path),
                "sha256": _sha256(ledger_path),
                "schema": runner.SOURCE_SHARD_LEDGER_SCHEMA,
                "status": runner.SOURCE_SHARD_LEDGER_STATUS,
                "seal_sha256": ledger["seal_sha256"],
            },
        },
        "tokenizer": {"file": tokenizer.name, "sha256": _sha256(tokenizer)},
        "prompt_token_ids": list(runner.PROMPT_IDS),
        "generated_token_ids": references,
        "sampling": {"method": "greedy_argmax", "do_sample": False, "temperature": 0},
        "provider": {
            "independent_from_hawking_native_executor": True,
            "implementation_version": "test-reference-2",
            "implementation": {"path": str(implementation), "sha256": _sha256(implementation)},
            "runtime_lock": {"path": str(runtime_lock), "sha256": _sha256(runtime_lock)},
            "producer": {"path": str(producer), "sha256": _sha256(producer)},
        },
        "logits_trace": {
            "path": str(trace_path),
            "sha256": _sha256(trace_path),
            "schema": runner.LOGITS_TRACE_SCHEMA,
            "status": runner.LOGITS_TRACE_STATUS,
            "seal_sha256": trace["seal_sha256"],
        },
    })
    receipt = tmp_path / "external-reference.json"
    _write_json(receipt, document)
    authorization = tmp_path / "external-reference.owner-authorization.json"
    _write_owner_authorization(authorization, receipt, document)
    return source, receipt, document, {
        "manifest": manifest,
        "ledger": ledger_path,
        "trace": trace_path,
        "implementation": implementation,
        "runtime_lock": runtime_lock,
        "producer": producer,
        "payload0": payloads[0],
        "authorization": authorization,
    }


def _write_trace_and_refresh_contract(receipt: Path, document: dict, trace_path: Path, trace: dict) -> dict:
    trace = _reseal(trace)
    _write_json(trace_path, trace)
    document["logits_trace"]["sha256"] = _sha256(trace_path)
    document["logits_trace"]["seal_sha256"] = trace["seal_sha256"]
    document = _reseal(document)
    _write_json(receipt, document)
    return document


def test_admitted_v2_reference_requires_all_bound_local_artifacts(tmp_path: Path) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)

    authority = runner.load_external_reference_contract(
        receipt,
        source,
        owner_authorization_path=artifacts["authorization"],
    )
    assert authority["mode"] == "admitted_external_complete_source_greedy_reference_v2"
    assert authority["generated_token_ids"] == [6, 7]
    assert authority["seal_sha256"] == document["seal_sha256"]
    assert authority["logits_trace"]["computed_argmax_token_ids"] == [6, 7]
    assert authority["source"]["shard_ledger"]["indexed_shard_count"] == 1
    assert authority["source_input_identity"] == authority["source"]["source_input_identity"]
    assert authority["source_input_identity"]["model_root"] == str(source)


def test_external_reference_refuses_a_root_outside_the_configured_canonical_model_lake(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    monkeypatch.setattr(runner, "MODEL_LAKE_ROOT", tmp_path / "different-lake")
    with pytest.raises(ValueError, match="configured canonical ModelLake Flash specimen"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_v1_reference_contracts_are_explicitly_withheld(tmp_path: Path) -> None:
    source, receipt, document, _ = _write_external_reference_contract(tmp_path)
    document["schema"] = "hawking.flash.external_greedy_reference.v1"
    _write_json(receipt, _reseal(document))
    with pytest.raises(ValueError, match="V1 contracts are withheld"):
        runner.load_external_reference_contract(receipt, source)


def test_v2_reference_rejects_malformed_duplicate_json_keys(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    receipt.write_text(
        '{"schema":"hawking.flash.external_greedy_reference.v2",'
        '"schema":"hawking.flash.external_greedy_reference.v2"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_v2_noncanonical_capture_still_requires_owner_authorization(tmp_path: Path, monkeypatch) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    with pytest.raises(ValueError, match="noncanonical or foreign capture producer requires --owner-authorization"):
        runner.load_external_reference_contract(receipt, source)

    monkeypatch.setattr(runner, "_load_machine_admin_owner_public_key", lambda: None)
    with pytest.raises(ValueError, match="machine-admin owner trust anchor is unavailable"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def _promote_fixture_to_local_canonical(
    tmp_path: Path,
    receipt: Path,
    document: dict,
    artifacts: dict,
    monkeypatch,
) -> dict:
    monkeypatch.setattr(runner, "_canonical_capture_repo_roots", lambda: (tmp_path.resolve(),))
    producer_dir = tmp_path / "tools" / "odyssey"
    producer_dir.mkdir(parents=True, exist_ok=True)
    producer = producer_dir / runner.CANONICAL_CAPTURE_PRODUCER_BASENAME
    producer.write_text(artifacts["producer"].read_text(encoding="utf-8"), encoding="utf-8")
    artifacts["producer"] = producer
    document["provider"]["producer"] = {"path": str(producer), "sha256": _sha256(producer)}
    document["provider"]["execution_schedule"] = runner.CANONICAL_CAPTURE_EXECUTION_SCHEDULE
    trace = json.loads(artifacts["trace"].read_text(encoding="utf-8"))
    trace["provider_artifacts"]["producer_sha256"] = _sha256(producer)
    document = _write_trace_and_refresh_contract(receipt, document, artifacts["trace"], trace)
    packager = producer_dir / runner.CANONICAL_CAPTURE_PACKAGER_BASENAME
    packager.write_text("print('canonical packager')\n", encoding="utf-8")
    artifacts["packager"] = packager
    _write_json(receipt.parent / "capture.json", {
        "status": "CAPTURED_EXTERNAL_COMPLETE_SOURCE_GREEDY_REFERENCE__OWNER_AUTHORIZATION_REQUIRED",
        "owner_authorization_present": False,
        "provider": {
            "hawking_packager": {"path": str(packager), "sha256": _sha256(packager)},
        },
    })
    return document


def test_historical_capture_blob_fallback_is_exactly_receipt_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "external-reference.v2.json"
    receipt.write_text("sealed historical reference\n", encoding="utf-8")
    current = tmp_path / "tools" / "odyssey" / runner.CANONICAL_CAPTURE_PRODUCER_BASENAME
    current.parent.mkdir(parents=True)
    current.write_text("current migrated source\n", encoding="utf-8")
    historical = b"sealed historical source\n"
    historical_sha = hashlib.sha256(historical).hexdigest()
    monkeypatch.setattr(runner, "_HISTORICAL_LOCAL_CAPTURE_RECEIPT", receipt)
    monkeypatch.setattr(
        runner,
        "_HISTORICAL_LOCAL_CAPTURE_RECEIPT_SHA256",
        _sha256(receipt),
    )
    monkeypatch.setattr(runner, "_HISTORICAL_LOCAL_CAPTURE_ARTIFACTS", {
        "producer": {
            "path": current,
            "sha256": historical_sha,
            "git_snapshot": "fixture-snapshot",
            "git_path": "tools/odyssey/historical-worker.py",
        },
    })

    def fake_git_show(command, **_kwargs):
        assert command == [
            "git", "show", "fixture-snapshot:tools/odyssey/historical-worker.py",
        ]
        return SimpleNamespace(returncode=0, stdout=historical, stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", fake_git_show)
    binding = {"path": str(current.resolve()), "sha256": historical_sha}
    verified = runner._verify_historical_local_capture_artifact(
        binding,
        receipt=receipt,
        field="producer",
    )
    assert verified is not None
    assert verified["sha256"] == historical_sha
    assert verified["historical_git_snapshot"] == "fixture-snapshot"

    assert runner._verify_historical_local_capture_artifact(
        {**binding, "path": str(tmp_path / "other.py")},
        receipt=receipt,
        field="producer",
    ) is None
    receipt.write_text("tampered\n", encoding="utf-8")
    assert runner._verify_historical_local_capture_artifact(
        binding,
        receipt=receipt,
        field="producer",
    ) is None


def test_v2_local_canonical_capture_admits_without_owner_signature(tmp_path: Path, monkeypatch) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)
    _promote_fixture_to_local_canonical(tmp_path, receipt, document, artifacts, monkeypatch)
    authority = runner.load_external_reference_contract(receipt, source)
    assert authority["admission_class"] == runner.LOCAL_CANONICAL_ADMISSION_STATUS
    assert authority["science_status"] == "UNEARNED"
    assert authority["owner_authorization"]["admission_class"] == runner.LOCAL_CANONICAL_ADMISSION_STATUS
    assert authority["owner_authorization"]["owner_signature_present"] is False
    assert authority["generated_token_ids"] == [6, 7]


def test_v2_local_canonical_rejects_missing_sibling_capture(tmp_path: Path, monkeypatch) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)
    _promote_fixture_to_local_canonical(tmp_path, receipt, document, artifacts, monkeypatch)
    (receipt.parent / "capture.json").unlink()
    with pytest.raises(ValueError, match="requires sibling capture.json"):
        runner.load_external_reference_contract(receipt, source)


def test_v2_local_canonical_rejects_tampered_packager_bytes(tmp_path: Path, monkeypatch) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)
    _promote_fixture_to_local_canonical(tmp_path, receipt, document, artifacts, monkeypatch)
    artifacts["packager"].write_text("tampered packager\n", encoding="utf-8")
    with pytest.raises(ValueError, match="packager"):
        runner.load_external_reference_contract(receipt, source)


def test_v2_owner_authorization_must_bind_the_exact_contract_inputs(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    authorization = json.loads(artifacts["authorization"].read_text(encoding="utf-8"))
    authorization["payload"]["logits_trace_sha256"] = "0" * 64
    authorization["signature_ed25519_hex"] = _TEST_OWNER_PRIVATE_KEY.sign(
        runner._canonical_compact_utf8_sorted(authorization["payload"])
    ).hex()
    _write_json(artifacts["authorization"], authorization)
    with pytest.raises(ValueError, match="does not bind the exact V2 contract inputs"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_v2_owner_authorization_rejects_a_tampered_signature(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    authorization = json.loads(artifacts["authorization"].read_text(encoding="utf-8"))
    authorization["signature_ed25519_hex"] = "0" * 128
    _write_json(artifacts["authorization"], authorization)
    with pytest.raises(ValueError, match="signature verification failed"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_external_reference_rejects_a_missing_provider_runtime_lock(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    artifacts["runtime_lock"].unlink()
    with pytest.raises(ValueError, match="runtime_lock"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_external_reference_requires_the_native_tokenizer_input(tmp_path: Path) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)
    document["tokenizer"]["file"] = "alternate-tokenizer.json"
    _write_json(receipt, _reseal(document))
    with pytest.raises(ValueError, match="native source tokenizer.json"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_external_reference_rejects_a_tampered_provider_implementation(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    artifacts["implementation"].write_text("tampered provider\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provider implementation SHA-256"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_external_reference_rejects_a_tampered_shard_ledger_artifact(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    artifacts["ledger"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source shard ledger SHA-256"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_external_reference_rejects_a_tampered_logits_payload(tmp_path: Path) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    artifacts["payload0"].write_bytes(b"x" * artifacts["payload0"].stat().st_size)
    with pytest.raises(ValueError, match="F32 logits payload.*SHA-256"):
        runner.load_external_reference_contract(receipt, source)


def test_external_reference_recomputes_actual_f32_argmax_after_all_seals_are_refreshed(tmp_path: Path) -> None:
    source, receipt, document, artifacts = _write_external_reference_contract(tmp_path)
    trace = json.loads(artifacts["trace"].read_text(encoding="utf-8"))
    vocab_size = trace["vocab_size"]
    payload = Path(trace["steps"][0]["logits_f32"]["path"])
    logits = [-20.0] * vocab_size
    logits[0] = 5.0
    payload.write_bytes(struct.pack(f"<{vocab_size}f", *logits))
    trace["steps"][0]["logits_f32"]["sha256"] = _sha256(payload)
    document = _write_trace_and_refresh_contract(receipt, document, artifacts["trace"], trace)
    _write_owner_authorization(artifacts["authorization"], receipt, document)

    with pytest.raises(ValueError, match="actual F32 logits argmax"):
        runner.load_external_reference_contract(
            receipt,
            source,
            owner_authorization_path=artifacts["authorization"],
        )


def test_dry_run_is_flash_specific_and_never_starts_gpu_work(capsys):
    assert runner.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DRY_RUN"
    assert payload["reference_contract"]["required"] is True
    assert "no Hawking-native reference discovery" in payload["phase_1"]
    assert payload["phase_2_ple_pre_layer_state_bank"].endswith(".ple_pre_layer_state_bank")
    assert payload["claim_boundary"].startswith("no GPU work")


def test_non_dry_run_reserves_every_native_output_before_contract_or_gpu_work(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_run",
        lambda command: (_ for _ in ()).throw(AssertionError("native runner must not start")),
    )

    existing_out = tmp_path / "existing-session.json"
    existing_out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        runner.main(["--out", str(existing_out)])

    sidecar_out = tmp_path / "sidecar-session.json"
    sidecar = sidecar_out.with_name(f"{sidecar_out.stem}.runner.json")
    sidecar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        runner.main(["--out", str(sidecar_out)])

    bank_out = tmp_path / "bank-session.json"
    existing_bank = tmp_path / "existing-bank"
    existing_bank.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        runner.main([
            "--out", str(bank_out),
            "--ple-pre-layer-state-bank-out", str(existing_bank),
        ])

    artifact_out = tmp_path / "artifact-session.json"
    runner._native_session_artifact_dir(artifact_out).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        runner.main(["--out", str(artifact_out)])

    nested_out = tmp_path / "nested-session.json"
    with pytest.raises(ValueError, match="collide or nest"):
        runner.main([
            "--out", str(nested_out),
            "--ple-pre-layer-state-bank-out", str(nested_out / "bank"),
        ])


def test_runner_sidecar_create_is_exclusive_and_never_replaces_a_racing_path(tmp_path: Path) -> None:
    sidecar = tmp_path / "session.runner.json"
    sidecar.write_text("already-owned\n", encoding="utf-8")
    with pytest.raises(ValueError, match="became occupied.*refusing to overwrite"):
        runner._write_new_runner_sidecar(sidecar, {"schema": "test"})
    assert sidecar.read_text(encoding="utf-8") == "already-owned\n"

    fresh = tmp_path / "fresh.runner.json"
    runner._write_new_runner_sidecar(fresh, {"schema": "test"})
    assert json.loads(fresh.read_text(encoding="utf-8")) == {"schema": "test"}


def test_repeated_decode_command_uses_an_explicit_prebuilt_binary_not_cargo_run(tmp_path: Path) -> None:
    binary = tmp_path / "flash_stateful_complete_token_session"
    command = runner._flash_binary_command(
        binary,
        tmp_path / "source",
        [*runner.PROMPT_IDS, 6, 7],
        len(runner.PROMPT_IDS),
        tmp_path / "out.json",
    )
    assert command[0] == str(binary)
    assert "cargo" not in command
    assert "run" not in command
    assert "--wrapper-admitted-source-control" in command
    assert command[command.index("--supervising-launcher-pid") + 1] == str(runner.os.getpid())
    assert "--wrapper-admitted-source-control" in command


def test_runner_sidecar_is_sealed_and_binds_the_exact_native_session(tmp_path: Path) -> None:
    references = [6, 7]
    session = tmp_path / "native-session.json"
    model_root = tmp_path / "source-root"
    session_doc = _native_session(references, model_root=model_root)
    _write_json(session, session_doc)
    source_input_identity = session_doc["source_input_identity"]
    sidecar = _reseal({
        "schema": runner.RUNNER_SIDECAR_SCHEMA,
        "status": runner.RUNNER_SIDECAR_STATUS,
        "model_root": str(model_root),
        "source_input_identity": source_input_identity,
        "prompt_token_ids": list(runner.PROMPT_IDS),
        "reference_generated_token_ids": references,
        "verification_receipt": str(session),
        "native_session": {
            "path": str(session),
            "sha256": _sha256(session),
            "seal_sha256": session_doc["seal_sha256"],
            "schema": runner.NATIVE_SESSION_SCHEMA,
            "status": runner.NATIVE_SESSION_STATUS,
            "root": str(model_root),
            "source_input_identity": source_input_identity,
        },
        "reference_authority": {
            "mode": "admitted_external_complete_source_greedy_reference_v2",
            "source_input_identity": source_input_identity,
            "owner_authorization": {"owner_public_key_sha256": "e" * 64},
        },
        "claim_boundary": "test only",
        "promotion_allowed": False,
    })
    runner._verify_runner_sidecar(
        sidecar,
        session_path=session,
        session_doc=session_doc,
        model_root=model_root,
        references=references,
        source_input_identity=source_input_identity,
    )

    tampered = json.loads(json.dumps(sidecar))
    tampered["native_session"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact native session path, bytes, seal, schema, and status"):
        runner._verify_runner_sidecar(
            _reseal(tampered),
            session_path=session,
            session_doc=session_doc,
            model_root=model_root,
            references=references,
            source_input_identity=source_input_identity,
        )

    tampered_seal = json.loads(json.dumps(sidecar))
    tampered_seal["claim_boundary"] = "changed without a seal"
    with pytest.raises(ValueError, match="runner sidecar.*seal"):
        runner._verify_runner_sidecar(
            tampered_seal,
            session_path=session,
            session_doc=session_doc,
            model_root=model_root,
            references=references,
            source_input_identity=source_input_identity,
        )


def test_pre_ple_state_bank_requires_every_payload_and_session_binding(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session_doc = {
        "schema": "hawking.flash.stateful_complete_token_session.v1",
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "seal_sha256": "a" * 64,
    }
    session.write_text(json.dumps(session_doc), encoding="utf-8")
    bank = tmp_path / "bank"
    bank.mkdir()
    raw = (b"\0\0\0\0") * runner.PLE_PRE_LAYER_STATE_WIDTH
    state = bank / "state-step-000000.f32"
    state.write_bytes(raw)
    body = {
        "schema": runner.PLE_PRE_LAYER_STATE_BANK_SCHEMA,
        "status": runner.PLE_PRE_LAYER_STATE_BANK_STATUS,
        "model": runner.REPO_ID,
        "pinned_revision": runner.PINNED_REVISION,
        "session_receipt": {
            "path": str(session),
            "sha256": hashlib.sha256(session.read_bytes()).hexdigest(),
            "seal_sha256": session_doc["seal_sha256"],
            "schema": session_doc["schema"],
            "status": session_doc["status"],
        },
        "token_ids": [7],
        "prompt_length": 1,
        "capture": {
            "layer": 0,
            "state_width": runner.PLE_PRE_LAYER_STATE_WIDTH,
            "upstream_PLE_inclusive_source_trajectory": False,
        },
        "states": [{
            "step": 0,
            "token_id": 7,
            "layer": 0,
            "path": str(state),
            "dtype": "F32_LE",
            "elements": runner.PLE_PRE_LAYER_STATE_WIDTH,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "finite": True,
        }],
    }
    body["seal_sha256"] = runner._compact_utf8_sorted_seal(body)
    manifest = bank / "manifest.json"
    manifest.write_text(json.dumps(body, indent=2), encoding="utf-8")

    verified = runner._verify_ple_pre_layer_state_bank(
        manifest,
        session_path=session,
        session_doc=session_doc,
        token_ids=[7],
        prompt_len=1,
    )
    assert verified["state_count"] == 1
    assert verified["payload_bytes"] == len(raw)

    state.write_bytes(raw[:-4])
    try:
        runner._verify_ple_pre_layer_state_bank(
            manifest,
            session_path=session,
            session_doc=session_doc,
            token_ids=[7],
            prompt_len=1,
        )
    except ValueError as exc:
        assert "does not bind its F32 payload" in str(exc)
    else:
        raise AssertionError("truncated state payload must fail closed")


def test_native_lane_preflight_ignores_only_zombies_and_sees_every_protected_owner(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "101 Z flash_stateful_complete_token_session --root fixture\n"
                "102 Z hawkingd --child fixture\n"
            )
        ),
    )
    assert runner._hawking_gpu_is_busy() is False

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "201 S flash_stateful_complete_token_session --root fixture\n"
                "202 R python tools/odyssey/flash_repeated_accepted_decode.py\n"
                "203 S python tools/odyssey/flash_route_union_control.py\n"
                "204 S flash_noetic_complete_layer0\n"
                "205 S mlx_vlm.server --model kimi\n"
                "206 S hawkingd --serve\n"
            )
        ),
    )
    lane = runner._protected_native_lane()
    assert lane["clean"] is False
    assert {match["marker"] for match in lane["matches"]} == {
        "flash_stateful_complete_token_session",
        "flash_repeated_accepted_decode.py",
        "flash_route_union_control.py",
        "flash_noetic_complete_layer0",
        "mlx_vlm.server",
        "hawkingd",
    }
    assert runner._hawking_gpu_is_busy() is True


def test_native_lane_preflight_ignores_its_direct_launcher(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 900)
    monkeypatch.setattr(runner.os, "getppid", lambda: 899)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "899 S /bin/zsh -c python3 tools/odyssey/flash_source_boundary_capture_mlx.py\n"
                "901 S unrelated-process\n"
            )
        ),
    )
    assert runner._protected_native_lane()["clean"] is True


def test_native_lane_preflight_ignores_only_its_ancestor_transaction_family(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 400)
    monkeypatch.setattr(runner.os, "getppid", lambda: 300)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "100 1 Ss /bin/zsh -c task launcher\n"
                "200 100 S python3 tools/odyssey/flash_source_boundary_transaction.py "
                "--native-binary target/release/examples/flash_stateful_complete_token_session\n"
                "300 200 S bash tools/gpu_lane_lock.sh flash-source-boundary-token0\n"
                "400 300 S python3 tools/odyssey/flash_source_boundary_transaction.py --leased-execute\n"
                "500 1 S flash_stateful_complete_token_session --root competing-fixture\n"
            )
        ),
    )
    lane = runner._protected_native_lane()
    assert lane["clean"] is False
    assert lane["matches"] == [{
        "pid": "500",
        "state": "S",
        "command": "flash_stateful_complete_token_session --root competing-fixture",
        "marker": "flash_stateful_complete_token_session",
    }]


def test_non_dry_run_refuses_a_visible_protected_lane_even_with_allow_busy_gpu(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source, receipt, _, artifacts = _write_external_reference_contract(tmp_path)
    monkeypatch.setattr(
        runner,
        "_protected_native_lane",
        lambda: {
            "clean": False,
            "matches": [{"pid": "9", "state": "S", "marker": "hawkingd", "command": "hawkingd"}],
            "observation": "test snapshot",
        },
    )
    monkeypatch.setattr(
        runner,
        "_run",
        lambda command: (_ for _ in ()).throw(AssertionError("native runner must not start")),
    )
    out = tmp_path / "blocked-session.json"
    assert runner.main([
        "--root", str(source),
        "--out", str(out),
        "--reference-contract", str(receipt),
        "--owner-authorization", str(artifacts["authorization"]),
        "--allow-busy-gpu",
    ]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED_CLEAN_GPU_LANE"
    assert payload["allow_busy_gpu_ignored"] is True
    assert payload["observed_native_lane_preflight"]["matches"][0]["marker"] == "hawkingd"
