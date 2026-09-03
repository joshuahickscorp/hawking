"""Focused regression coverage for the durable dual Gravity worker codecs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

import lab.operators.ascension_dual_gravity_worker as dual_gravity
from lab.operators.ascension_dual_gravity_worker import (
    DualGravityWorker,
    EXPANDED_REPRESENTATION_SCHEDULE_VERSION,
    IDENTITY_SCHEMA,
    LEGACY_REPRESENTATION_SCHEDULE_VERSION,
    MAGIC_ACTIVATION,
    MAGIC_ADDITIVE,
    MAGIC_HADAMARD,
    Proposal,
    SPECS,
    SOURCE_REVALIDATION_SCHEMA,
    Target,
    _activation_corrected_codec,
    _additive_residual_codec,
    _binary_codec,
    _decode_activation_corrected_codec,
    _decode_additive_residual_codec,
    _decode_hadamard_lattice_codec,
    _encode,
    _hadamard_lattice_codec,
    _parse_container,
    _representation_config,
    _residual_codec,
    _regular_file_identity,
    _sha256,
    _ternary_codec,
    _uniform_codec,
)
from lab.receipts import seal


def test_diverse_physical_codecs_keep_the_exact_source_region_shape() -> None:
    values = np.linspace(-1.0, 1.0, 513, dtype=np.float32).reshape(27, 19)

    codecs = (
        _binary_codec(values),
        _uniform_codec(values, bits=2),
        _uniform_codec(values, bits=3),
        _ternary_codec(values, threshold_multiplier=0.8),
        _residual_codec(values, outlier_ratio=0.01),
    )

    for codec in codecs:
        assert codec.reconstruction.shape == values.shape
        assert np.isfinite(codec.reconstruction).all()
        assert codec.payload
        assert codec.metadata["elements"] == values.size


def test_teacher_low_rank_genome_is_a_real_component_training_candidate() -> None:
    values = np.arange(256, dtype=np.float32).reshape(16, 16) / 100.0
    proposal = Proposal(
        sequence=5,
        generation=0,
        target=Target("unit.weight", "routed_expert_gate", True, "unit"),
        representation="teacher_low_rank_q3",
        config={"rank": 4, "bits": 3, "train_steps": 0},
        candidate_id="unit-low-rank",
    )

    codec, training = _encode(values, proposal)

    assert codec.reconstruction.shape == values.shape
    assert codec.metadata["rank"] == 4
    assert training["status"] == "NOT_REQUESTED"


def test_deterministic_proposal_rotates_representation_and_target() -> None:
    worker = DualGravityWorker(SPECS["qwen30"])
    identity = {"content_identity_sha256": "a" * 64}
    targets = (
        Target("a.weight", "moe_router", True, "a"),
        Target("b.weight", "routed_expert_gate", True, "b"),
    )
    first = worker._proposal({"next_proposal_index": 0}, targets, identity)
    seventh = worker._proposal({"next_proposal_index": 7}, targets, identity)

    assert first.target.name == "a.weight"
    assert first.representation == "binary_sign_scale128"
    assert seventh.target.name == "b.weight"
    assert first.candidate_id != seventh.candidate_id


def test_new_representation_families_decode_their_exact_physical_bytes() -> None:
    values = np.random.default_rng(19).standard_normal((17, 31), dtype=np.float32)
    candidates = (
        (
            _hadamard_lattice_codec(values, bits=3, group_size=64),
            _decode_hadamard_lattice_codec,
            MAGIC_HADAMARD,
            lambda meta: meta["scale_bytes"] + meta["code_bytes"],
        ),
        (
            _additive_residual_codec(values, group_size=64),
            _decode_additive_residual_codec,
            MAGIC_ADDITIVE,
            lambda meta: meta["base_scale_bytes"]
            + meta["residual_scale_bytes"]
            + meta["base_code_bytes"]
            + meta["residual_code_bytes"],
        ),
        (
            _activation_corrected_codec(values, bits=3, group_size=64, calibration_seed=101),
            _decode_activation_corrected_codec,
            MAGIC_ACTIVATION,
            lambda meta: meta["base_uniform_header"]["scale_bytes"]
            + meta["base_uniform_header"]["code_bytes"]
            + meta["activation_correction"]["correction_bytes"],
        ),
    )

    for codec, decoder, magic, body_bytes in candidates:
        header, body = _parse_container(codec.payload, expected_magic=magic)
        decoded = decoder(codec.payload)

        assert header == codec.metadata
        assert len(body) == body_bytes(codec.metadata)
        assert codec.reconstruction.shape == values.shape
        assert np.isfinite(codec.reconstruction).all()
        np.testing.assert_array_equal(decoded, codec.reconstruction)

    correction = candidates[-1][0].metadata["activation_correction"]
    assert correction["corrected_direction_output"]["relative_l2"] <= correction["baseline_direction_output"]["relative_l2"] + 1e-6


def test_new_representation_genomes_dispatch_through_the_worker_encoder() -> None:
    values = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) / 127.0
    target = Target("unit.weight", "moe_router", True, "unit")
    expected_schemas = {
        "hadamard_lattice_q3_group128": "hawking.gravity.hadamard_lattice_group.v1",
        "additive_residual_codebook_q2x2": "hawking.gravity.additive_residual_codebook.v1",
        "activation_corrected_rowwise_q3": "hawking.gravity.activation_corrected_rowwise.v1",
    }

    for sequence, (representation, expected_schema) in enumerate(expected_schemas.items()):
        proposal = Proposal(
            sequence=sequence,
            generation=1,
            target=target,
            representation=representation,
            config=_representation_config(representation, generation=1),
            candidate_id=f"new-codec-{sequence}",
            schedule_version=EXPANDED_REPRESENTATION_SCHEDULE_VERSION,
            schedule_phase="unit-test",
            schedule_boundary_sequence=0,
            schedule_start_generation=1,
        )
        codec, training = _encode(values, proposal)

        assert codec.metadata["schema"] == expected_schema
        assert codec.reconstruction.shape == values.shape
        assert training["status"] == "NOT_APPLICABLE"


def test_schedule_migration_keeps_v1_positions_and_adds_families_only_after_boundary() -> None:
    worker = DualGravityWorker(SPECS["qwen30"])
    identity = {"content_identity_sha256": "b" * 64}
    targets = (
        Target("a.weight", "moe_router", True, "a"),
        Target("b.weight", "routed_expert_gate", True, "b"),
    )
    state: dict[str, object] = {"next_proposal_index": 0}

    first = worker._proposal(state, targets, identity)
    boundary = state["representation_schedule"]["v2_start_sequence"]
    state["next_proposal_index"] = 7
    seventh = worker._proposal(state, targets, identity)
    state["next_proposal_index"] = boundary - 1
    final_legacy = worker._proposal(state, targets, identity)
    state["next_proposal_index"] = boundary
    first_v2 = worker._proposal(state, targets, identity)
    state["next_proposal_index"] = boundary + 7
    first_hadamard = worker._proposal(state, targets, identity)
    state["next_proposal_index"] = boundary + 8
    first_additive = worker._proposal(state, targets, identity)
    state["next_proposal_index"] = boundary + 9
    first_activation_corrected = worker._proposal(state, targets, identity)

    assert boundary == len(targets) * 7
    assert state["representation_schedule"]["status"] == "PINNED_FUTURE_EXPANSION_BOUNDARY"
    assert first.schedule_version == LEGACY_REPRESENTATION_SCHEDULE_VERSION
    assert (first.target.name, first.representation) == ("a.weight", "binary_sign_scale128")
    assert (seventh.target.name, seventh.representation) == ("b.weight", "binary_sign_scale128")
    assert (final_legacy.target.name, final_legacy.representation) == ("b.weight", "uniform_q4_group64")
    assert first_v2.schedule_version == EXPANDED_REPRESENTATION_SCHEDULE_VERSION
    assert first_v2.generation == 1
    assert first_v2.representation == "binary_sign_scale128"
    assert first_hadamard.representation == "hadamard_lattice_q3_group128"
    assert first_additive.representation == "additive_residual_codebook_q2x2"
    assert first_activation_corrected.representation == "activation_corrected_rowwise_q3"
    assert first_activation_corrected.schedule_boundary_sequence == boundary


def test_schedule_upgrade_pins_the_next_complete_legacy_generation_for_live_progress() -> None:
    worker = DualGravityWorker(SPECS["qwen30"])
    identity = {"content_identity_sha256": "c" * 64}
    targets = (
        Target("a.weight", "moe_router", True, "a"),
        Target("b.weight", "routed_expert_gate", True, "b"),
    )
    # A prior v1 process has already crossed the first 14-position generation.
    state: dict[str, object] = {"next_proposal_index": 16}
    current = worker._proposal(state, targets, identity)
    plan = state["representation_schedule"]

    assert current.schedule_version == LEGACY_REPRESENTATION_SCHEDULE_VERSION
    assert plan["v2_start_generation"] == 2
    assert plan["v2_start_sequence"] == 28
    state["next_proposal_index"] = 28
    first_v2 = worker._proposal(state, targets, identity)
    assert first_v2.schedule_version == EXPANDED_REPRESENTATION_SCHEDULE_VERSION
    assert first_v2.generation == 2


def test_state_kv_snapshot_uses_verified_nested_outcome(tmp_path) -> None:
    worker = DualGravityWorker(replace(SPECS["qwen30"], root=tmp_path))
    lane = tmp_path / "state-kv"
    receipt_path = lane / "QWEN30_ATTENTION_KV_STATE_CODEC_RECEIPT.json"
    receipt = seal({"schema": "test.state-kv.receipt.v1", "component": "attention_kv"})
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    status = seal(
        {
            "schema": "test.state-kv.status.v1",
            "phase": "COMPLETE_SOURCE_BOUND_COMPONENT_STATE_CODEC_RESEARCH",
            "heartbeat": 3,
            "outcome": {
                "artifact_count": 4,
                "geometry": {"ledger": {"layer_count": 48}},
                "receipt_path": str(receipt_path),
                "receipt_seal_sha256": receipt["seal_sha256"],
            },
        }
    )
    (lane / "QWEN30_STATE_KV_STATUS.json").write_text(json.dumps(status), encoding="utf-8")

    snapshot = worker._state_kv_snapshot()

    assert snapshot["status"] == "COMPLETE_SOURCE_BOUND_COMPONENT_STATE_CODEC_RESEARCH"
    assert snapshot["measured_source_bound_component_lane"] is True
    assert snapshot["geometry"]["ledger"]["layer_count"] == 48
    assert snapshot["source_component_receipts"][0]["status"] == "VERIFIED"


def test_runtime_snapshot_never_infers_tps_from_missing_runtime_receipts(tmp_path) -> None:
    worker = DualGravityWorker(replace(SPECS["qwen80"], root=tmp_path))

    snapshot = worker._runtime_snapshot()

    assert snapshot["current_base_true_tps"] is None
    assert snapshot["hcli_status"] == "BLOCKED_NO_COMPLETE_NATIVE_RUNTIME"
    assert snapshot["tg_rung"] is None
    assert snapshot["current_kernel_bottleneck"] == "EXACT_COMPLETE_NATIVE_DECODER_AND_FULL_TOKEN_GRAPH_NOT_YET_IMPLEMENTED"


def _source_revalidation_fixture(tmp_path):
    """Build a tiny sealed source body and compiler receipt without real weights."""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weight_map = {
        "model.layers.0.weight": "model-00001.safetensors",
        "model.layers.1.weight": "model-00002.safetensors",
    }
    shard_bytes = {
        "model-00001.safetensors": b"first tiny source shard\x00\x01",
        "model-00002.safetensors": b"second tiny source shard\x02\x03",
    }
    for name, payload in shard_bytes.items():
        (source_dir / name).write_bytes(payload)
    index_path = source_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}, sort_keys=True), encoding="utf-8")

    spec = replace(SPECS["qwen30"], source_dir=source_dir, root=tmp_path / "physical")
    worker = DualGravityWorker(spec)
    verified_shards = [
        {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(shard_bytes.items())
    ]
    source_content = {
        "repository": spec.repository,
        "revision": spec.revision,
        "architecture": spec.architecture,
        "verified_weight_shards": verified_shards,
        "control_files": [
            {
                "path": "model.safetensors.index.json",
                "bytes": index_path.stat().st_size,
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            }
        ],
    }
    identity = seal(
        {
            "schema": IDENTITY_SCHEMA,
            "status": "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND",
            "source_content": source_content,
            "content_identity_sha256": _sha256(source_content),
        }
    )
    expected_hashes = {row["path"]: row["sha256"] for row in verified_shards}
    receipt = seal(
        {
            "schema": SOURCE_REVALIDATION_SCHEMA,
            "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
            "source_repository": spec.repository,
            "source_revision": spec.revision,
            "source_model_dir": str(source_dir.resolve()),
            "index_sha256": source_content["control_files"][0]["sha256"],
            "weight_map_sha256": _sha256(dict(sorted(weight_map.items()))),
            "sealed_shard_hashes_sha256": _sha256(expected_hashes),
            "sealed_shard_count": len(verified_shards),
            "shards": {
                row["path"]: {
                    "expected_sha256": row["sha256"],
                    "observed_sha256": row["sha256"],
                    "expected_bytes": row["bytes"],
                    "file_identity": _regular_file_identity(source_dir / row["path"], label=row["path"]),
                }
                for row in verified_shards
            },
            "observed_total_bytes": sum(row["bytes"] for row in verified_shards),
        }
    )
    receipt_path = spec.root / "complete-gravity" / "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return worker, identity, weight_map, receipt, receipt_path, source_dir


def test_current_source_revalidation_reuses_sealed_full_shard_proof_without_rehashing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, identity, weight_map, receipt, _, _ = _source_revalidation_fixture(tmp_path)

    def reject_unexpected_full_hash(*_args, **_kwargs):
        raise AssertionError("per-candidate source revalidation must not full-hash source shards")

    monkeypatch.setattr(dual_gravity, "_sha256_file", reject_unexpected_full_hash)
    proof = worker._current_source_revalidation(
        identity,
        weight_map,
        target_shard="model-00001.safetensors",
    )

    assert proof["status"] == "VERIFIED_SEALED_CURRENT_SOURCE_REVALIDATION_BEFORE_TENSOR_READ"
    assert proof["receipt_seal_sha256"] == receipt["seal_sha256"]
    assert proof["indexed_shard_count"] == 2
    assert proof["target_shard"] == "model-00001.safetensors"
    assert proof["target_shard_sha256"] == receipt["shards"]["model-00001.safetensors"]["expected_sha256"]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("resealed_hash_mismatch", "source revalidation receipt hash"),
        ("invalid_seal", "source revalidation receipt is not trustworthy"),
        ("symlink_substitution", "not a symlink"),
    ),
)
def test_current_source_revalidation_fails_closed_for_bad_or_substituted_evidence(
    tmp_path, case: str, message: str
) -> None:
    worker, identity, weight_map, receipt, receipt_path, source_dir = _source_revalidation_fixture(tmp_path)
    if case == "resealed_hash_mismatch":
        replacement = json.loads(json.dumps(receipt))
        replacement["shards"]["model-00001.safetensors"]["observed_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(seal(replacement), sort_keys=True), encoding="utf-8")
    elif case == "invalid_seal":
        replacement = dict(receipt)
        replacement["status"] = "MUTATED_AFTER_SEAL"
        receipt_path.write_text(json.dumps(replacement, sort_keys=True), encoding="utf-8")
    else:
        replacement_path = tmp_path / "replacement.safetensors"
        replacement_path.write_bytes((source_dir / "model-00001.safetensors").read_bytes())
        source_shard = source_dir / "model-00001.safetensors"
        source_shard.unlink()
        source_shard.symlink_to(replacement_path)

    with pytest.raises(dual_gravity.DualGravityError, match=message):
        worker._current_source_revalidation(
            identity,
            weight_map,
            target_shard="model-00001.safetensors",
        )
