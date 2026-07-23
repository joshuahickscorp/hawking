#!/usr/bin/env python3.12
"""Offline tests for the strict GLM-5.2 adapter and deterministic twin."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
CONDENSE = HERE.parent
REPO_ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_adapter as A  # noqa: E402
import glm52_synthetic as S  # noqa: E402


def _official_config() -> dict:
    config = json.loads(json.dumps(dict(A.OFFICIAL_CONFIG_CONTRACT)))
    config["indexer_types"] = list(A.OFFICIAL_INDEXER_TYPES)
    config["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 75
    return config


@pytest.fixture()
def fixture(tmp_path: Path) -> S.SyntheticFixture:
    return S.build_synthetic_fixture(tmp_path / "glm52-fixture")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _mutate_header(path: Path, tensor_name: str, mutation) -> None:
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    mutation(header[tensor_name])
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > header_len:
        raise AssertionError("test mutation unexpectedly grew the safetensors header")
    encoded += b" " * (header_len - len(encoded))
    path.write_bytes(raw[:8] + encoded + raw[8 + header_len:])


def test_official_schema_census_and_exact_shapes_without_source_download():
    geometry = A.validate_config(_official_config(), profile=A.PROFILE_OFFICIAL)
    specs = A.expected_tensor_specs(geometry)
    summary = A.schema_summary(geometry)
    assert len(specs) == 59_585
    assert summary["core_tensor_count"] == 58_794
    assert summary["mtp_tensor_count"] == 791
    assert summary["core_logical_elements"] == 743_377_019_904
    assert summary["mtp_logical_elements"] == 9_952_920_576
    assert summary["index_topk"] == 2_048
    assert summary["routed_expert_count"] == 256
    assert summary["shared_expert_count"] == 1
    assert summary["experts_per_token"] == 8
    assert summary["router_selection"] == "top-8-of-256"
    assert summary["router_weight_sum_after_scaling"] == 2.5
    assert summary["router_scoring_function"] == "sigmoid"
    assert summary["router_topk_method"] == "noaux_tc"
    assert summary["router_normalizes_topk_probability"] is True
    assert sum(spec.element_count for spec in specs.values()) == 753_329_940_480
    assert sum(spec.byte_count for spec in specs.values()) == 1_506_659_919_872

    assert specs["model.layers.0.self_attn.q_b_proj.weight"].shape == (16_384, 2_048)
    assert specs["model.layers.0.self_attn.kv_b_proj.weight"].shape == (28_672, 512)
    assert specs["model.layers.0.self_attn.indexer.wq_b.weight"].shape == (4_096, 2_048)
    assert specs["model.layers.3.mlp.experts.255.gate_proj.weight"].shape == (2_048, 6_144)
    correction = specs["model.layers.77.mlp.gate.e_score_correction_bias"]
    assert correction.dtype == "F32" and correction.shape == (256,)
    assert specs["model.layers.78.eh_proj.weight"].shape == (6_144, 12_288)


def test_official_and_synthetic_config_contracts_fail_closed():
    official = _official_config()
    official["num_experts_per_tok"] = 7
    with pytest.raises(A.Glm52AdapterError, match="config contract mismatch"):
        A.validate_config(official, profile=A.PROFILE_OFFICIAL)

    synthetic = S.synthetic_config()
    synthetic["indexer_types"][4] = "full"
    with pytest.raises(A.Glm52AdapterError, match="IndexShare pattern mismatch"):
        A.validate_config(synthetic, profile=A.PROFILE_SYNTHETIC)


def test_synthetic_fixture_has_exact_architecture_and_hf_views(fixture: S.SyntheticFixture):
    full = fixture.full_inventory
    main = fixture.main_only_inventory
    assert full.geometry.hidden_size == 32
    assert full.geometry.num_hidden_layers == 7
    assert full.geometry.physical_layer_count == 8
    assert full.geometry.mtp_layer == 7
    assert full.geometry.n_routed_experts == 256
    assert full.geometry.num_experts_per_tok == 8
    assert full.geometry.n_shared_experts == 1
    assert full.geometry.indexer_types == A.SYNTHETIC_INDEXER_TYPES
    assert fixture.config["max_position_embeddings"] == 1_048_576
    assert full.tensor_count == 3_978 and len(full.shards) == 3
    assert main.tensor_count == 3_187 and len(main.shards) == 2
    assert len(full.mtp_names) == 791 and not main.mtp_names
    assert full.core_names == main.core_names
    assert len(fixture.index["weight_map"]) == 3_978
    assert len(fixture.main_only_index["weight_map"]) == 3_187
    assert (fixture.full_dir / "model.safetensors.index.json").is_file()
    assert (fixture.main_only_dir / "model.safetensors.index.json").is_file()
    assert fixture.metadata["views"]["main_only"]["is_exact_core_filter"] is True


def test_fixture_is_byte_deterministic(tmp_path: Path):
    first = S.build_synthetic_fixture(tmp_path / "first")
    second = S.build_synthetic_fixture(tmp_path / "second")
    assert first.metadata["shards"] == second.metadata["shards"]
    assert first.metadata["views"]["full"]["index_sha256"] == second.metadata["views"]["full"]["index_sha256"]
    assert first.index == second.index
    assert first.config == second.config


def test_indexshare_omissions_and_sources_are_exact(fixture: S.SyntheticFixture):
    names = set(fixture.full_inventory.tensors)
    for layer in (0, 1, 2, 6, 7):
        assert f"model.layers.{layer}.self_attn.indexer.wk.weight" in names
    for layer in (3, 4, 5):
        assert not any(name.startswith(f"model.layers.{layer}.self_attn.indexer.") for name in names)
        spec = fixture.full_inventory.tensors[
            f"model.layers.{layer}.self_attn.q_a_proj.weight"
        ].spec
        assert spec.indexer_source_layer == 2
    assert fixture.full_inventory.tensors[
        "model.layers.6.self_attn.indexer.wk.weight"
    ].spec.indexer_source_layer == 6
    assert fixture.full_inventory.tensors[
        "model.layers.7.self_attn.indexer.wk.weight"
    ].spec.indexer_source_layer == 7
    with pytest.raises(A.Glm52AdapterError, match="must omit"):
        A.tensor_spec(
            fixture.full_inventory.geometry,
            "model.layers.4.self_attn.indexer.wk.weight",
        )


def test_core_mtp_boundary_and_mtp_extras(fixture: S.SyntheticFixture):
    full = fixture.full_inventory
    assert all(row.spec.section == A.MTP for name, row in full.tensors.items() if name.startswith("model.layers.7."))
    assert all(not name.startswith("model.layers.7.") for name in full.core_names)
    for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
        name = f"model.layers.7.{suffix}"
        assert full.tensors[name].spec.section == A.MTP
    with pytest.raises(A.Glm52AdapterError, match="outside"):
        A.tensor_spec(full.geometry, "model.layers.8.input_layernorm.weight")
    with pytest.raises(A.Glm52AdapterError, match="unknown or misplaced"):
        A.tensor_spec(full.geometry, "model.layers.6.eh_proj.weight")


def test_bounded_reader_decodes_bf16_and_f32_and_counts_exact_bytes(fixture: S.SyntheticFixture):
    reader = fixture.full_reader(max_tensor_bytes=8_192)
    bf16_name = "model.layers.3.mlp.experts.9.gate_proj.weight"
    f32_name = "model.layers.3.mlp.gate.e_score_correction_bias"
    bf16 = reader.tensor(bf16_name)
    f32 = reader.tensor(f32_name)
    assert bf16.dtype == np.float32 and bf16.shape == (16, 32)
    assert f32.dtype == np.float32 and f32.shape == (256,)
    assert np.array_equal(f32, np.arange(256, dtype=np.float32) / np.float32(65_536.0))
    expected_bytes = fixture.full_inventory.tensors[bf16_name].byte_count + fixture.full_inventory.tensors[f32_name].byte_count
    assert reader.read_calls == 2
    assert reader.payload_bytes_read == expected_bytes
    assert reader.peak_payload_request_bytes == max(
        fixture.full_inventory.tensors[bf16_name].byte_count,
        fixture.full_inventory.tensors[f32_name].byte_count,
    )


def test_bounded_reader_refuses_over_limit_and_unknown(fixture: S.SyntheticFixture):
    reader = fixture.full_reader(max_tensor_bytes=128)
    with pytest.raises(A.Glm52AdapterError, match="bounded read refused"):
        reader.tensor("model.layers.3.mlp.experts.0.gate_proj.weight")
    with pytest.raises(A.Glm52AdapterError, match="absent"):
        reader.tensor("model.layers.3.mlp.experts.256.gate_proj.weight")
    assert reader.read_calls == 0 and reader.payload_bytes_read == 0


def test_router_fixture_is_tie_free(fixture: S.SyntheticFixture):
    reader = fixture.full_reader(max_tensor_bytes=32_768)
    router = reader.tensor("model.layers.3.mlp.gate.weight")
    scores = router @ np.ones((32,), dtype=np.float32)
    assert len(np.unique(scores)) == 256
    assert np.all(np.diff(scores) > 0)
    correction = reader.tensor("model.layers.3.mlp.gate.e_score_correction_bias")
    assert np.all(np.diff(correction) > 0)


def test_gate_up_pack_is_gate_then_up_without_interleaving(fixture: S.SyntheticFixture):
    reader = fixture.full_reader(max_tensor_bytes=8_192)
    gate_name, up_name = A.expert_gate_up_names(3, 17)
    gate = reader.tensor(gate_name)
    up = reader.tensor(up_name)
    assert not np.array_equal(gate, up)
    packed = A.pack_expert_gate_up(reader, 3, 17)
    assert packed.shape == (32, 32)
    assert np.array_equal(packed[:16], gate)
    assert np.array_equal(packed[16:], up)
    split_gate, split_up = A.split_expert_gate_up(packed, fixture.full_inventory.geometry)
    assert np.array_equal(split_gate, gate) and np.array_equal(split_up, up)
    assert A.GATE_UP_ORDER == ("gate_proj", "up_proj")


def test_missing_and_unknown_index_names_are_rejected(fixture: S.SyntheticFixture):
    index_path = fixture.full_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    missing = "model.layers.3.mlp.experts.0.gate_proj.weight"
    shard = index["weight_map"].pop(missing)
    _write_json(index_path, index)
    with pytest.raises(A.Glm52AdapterError, match="checkpoint tensor census mismatch"):
        A.verify_checkpoint(fixture.full_dir, profile=A.PROFILE_SYNTHETIC)

    index["weight_map"][missing] = shard
    index["weight_map"]["model.layers.3.mlp.experts.256.gate_proj.weight"] = shard
    _write_json(index_path, index)
    with pytest.raises(A.Glm52AdapterError, match="checkpoint tensor census mismatch"):
        A.verify_checkpoint(fixture.full_dir, profile=A.PROFILE_SYNTHETIC)


def test_duplicate_index_key_is_rejected_before_header_access(fixture: S.SyntheticFixture):
    index_path = fixture.full_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = "model.layers.3.mlp.experts.0.gate_proj.weight"
    shard = index["weight_map"].pop(name)
    rest = json.dumps(index["weight_map"], sort_keys=True, separators=(",", ":"))[1:-1]
    duplicate_map = json.dumps(name) + ":" + json.dumps(shard) + "," + json.dumps(name) + ":" + json.dumps(shard)
    if rest:
        duplicate_map += "," + rest
    raw = (
        "{\"metadata\":"
        + json.dumps(index["metadata"], separators=(",", ":"))
        + ",\"weight_map\":{"
        + duplicate_map
        + "}}"
    )
    index_path.write_text(raw)
    with pytest.raises(A.Glm52AdapterError, match="duplicate JSON key"):
        A.verify_checkpoint(fixture.full_dir, profile=A.PROFILE_SYNTHETIC)


def test_wrong_header_shape_and_dtype_are_rejected(fixture: S.SyntheticFixture):
    name = "model.layers.3.mlp.experts.0.gate_proj.weight"
    shard_name = fixture.index["weight_map"][name]
    shard_path = fixture.full_dir / shard_name
    _mutate_header(shard_path, name, lambda row: row.__setitem__("shape", [32, 16]))
    with pytest.raises(A.Glm52AdapterError, match="schema mismatch"):
        A.verify_checkpoint(fixture.full_dir, profile=A.PROFILE_SYNTHETIC)

    second = S.build_synthetic_fixture(fixture.root.parent / "dtype-tamper")
    bf16_name = "model.layers.3.mlp.experts.1.gate_proj.weight"
    bf16_shard = second.full_dir / second.index["weight_map"][bf16_name]
    _mutate_header(bf16_shard, bf16_name, lambda row: row.__setitem__("dtype", "F32"))
    with pytest.raises(A.Glm52AdapterError, match="shape/dtype extent mismatch"):
        A.verify_checkpoint(second.full_dir, profile=A.PROFILE_SYNTHETIC)


def test_unindexed_shard_and_changed_shard_identity_are_rejected(fixture: S.SyntheticFixture):
    extra = fixture.full_dir / "unindexed.safetensors"
    extra.write_bytes(b"not a shard")
    with pytest.raises(A.Glm52AdapterError, match="checkpoint shard set mismatch"):
        A.verify_checkpoint(fixture.full_dir, profile=A.PROFILE_SYNTHETIC)
    extra.unlink()

    reader = fixture.full_reader(max_tensor_bytes=8_192)
    name = "model.layers.3.mlp.experts.0.gate_proj.weight"
    record = fixture.full_inventory.tensors[name]
    shard = fixture.full_inventory.shards[record.shard]
    with shard.path.open("r+b") as handle:
        handle.seek(record.absolute_start)
        first = handle.read(1)
        handle.seek(record.absolute_start)
        handle.write(bytes([first[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(A.Glm52AdapterError, match="identity changed"):
        reader.tensor(name)


def test_main_only_view_cannot_read_mtp(fixture: S.SyntheticFixture):
    reader = fixture.main_only_reader(max_tensor_bytes=32_768)
    assert reader.tensor("model.layers.6.mlp.gate.weight").shape == (256, 32)
    with pytest.raises(A.Glm52AdapterError, match="absent"):
        reader.tensor("model.layers.7.mlp.gate.weight")


def test_streaming_window_retains_complete_index_but_reads_only_resident_shards(
    fixture: S.SyntheticFixture,
    tmp_path: Path,
):
    resident = sorted(set(fixture.index["weight_map"].values()))[:2]
    hydrated = tmp_path / "hydrated-window"
    hydrated.mkdir()
    for shard in resident:
        os.link(fixture.full_dir / shard, hydrated / shard)
    window = {
        "window_id": "W001",
        "source_shards": resident,
        "carry_in_shards": [resident[0]],
        "new_fetch_shards": [resident[1]],
        "refetch_shards": [],
        "carry_out_shards": [resident[0]],
        "evict_after_seal_shards": [resident[1]],
    }
    inventory = A.verify_streaming_window(
        fixture.full_dir,
        hydrated,
        window,
        profile=A.PROFILE_SYNTHETIC,
    )
    expected_names = {
        name
        for name, shard in fixture.index["weight_map"].items()
        if shard in set(resident)
    }
    assert set(inventory.shards) == set(resident)
    assert set(inventory.tensors) == expected_names
    assert len(inventory.index["weight_map"]) == fixture.full_inventory.tensor_count
    smallest = min(inventory.tensors, key=lambda name: inventory.tensors[name].byte_count)
    assert A.BoundedSafetensorsReader(inventory).tensor(smallest).size > 0

    (hydrated / resident[1]).unlink()
    with pytest.raises(A.Glm52AdapterError, match="hydrated streaming shard set mismatch"):
        A.verify_streaming_window(
            fixture.full_dir,
            hydrated,
            window,
            profile=A.PROFILE_SYNTHETIC,
        )


def test_streaming_window_rejects_partial_global_index_and_bad_carry_algebra(
    fixture: S.SyntheticFixture,
    tmp_path: Path,
):
    shard = sorted(set(fixture.index["weight_map"].values()))[0]
    hydrated = tmp_path / "hydrated-one"
    hydrated.mkdir()
    os.link(fixture.full_dir / shard, hydrated / shard)
    window = {
        "window_id": "W000",
        "source_shards": [shard],
        "carry_in_shards": [],
        "new_fetch_shards": [shard],
        "refetch_shards": [],
        "carry_out_shards": [],
        "evict_after_seal_shards": [shard],
    }

    malformed = dict(window)
    malformed["evict_after_seal_shards"] = []
    with pytest.raises(A.Glm52AdapterError, match="source minus carry-out"):
        A.verify_streaming_window(
            fixture.full_dir,
            hydrated,
            malformed,
            profile=A.PROFILE_SYNTHETIC,
        )

    control = tmp_path / "partial-control"
    control.mkdir()
    os.link(fixture.full_dir / "config.json", control / "config.json")
    partial_index = json.loads(json.dumps(fixture.index))
    partial_index["weight_map"].pop(next(iter(partial_index["weight_map"])))
    _write_json(control / "model.safetensors.index.json", partial_index)
    with pytest.raises(A.Glm52AdapterError, match="complete source index tensor census"):
        A.verify_streaming_window(
            control,
            hydrated,
            window,
            profile=A.PROFILE_SYNTHETIC,
        )


def test_official_tokenizer_and_chat_template_assemble_offline() -> None:
    manifest = json.loads(
        (REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json").read_text(encoding="utf-8")
    )
    root = Path(manifest["one_copy"]["snapshot_view"])
    assembly = A.load_official_tokenizer_assembly(root)
    assert assembly.vocabulary_size == 154_856
    assert assembly.padded_model_vocabulary_size == 154_880
    assert assembly.model_max_length == 1_048_576
    assert assembly.eos_token_id == assembly.pad_token_id == 154_820
    assert assembly.generation_eos_token_ids == (154_820, 154_827, 154_829)
    assert assembly.generation_pad_token_id == 154_820
    assert assembly.asset_sha256 == {
        name: digest for name, (_size, digest) in A.OFFICIAL_TOKENIZER_ASSETS.items()
    }
    receipt = assembly.assemble_chat(
        (
            {"role": "system", "content": "You are exact."},
            {"role": "user", "content": "Return 2+2."},
        )
    )
    assert receipt["token_count"] == 23
    assert receipt["begins_with_glm_mask_sop"] is True
    assert receipt["ends_with_generation_think"] is True
    assert receipt["generation_eos_token_ids"] == [154_820, 154_827, 154_829]

    tool_receipt = assembly.assemble_chat(
        (
            {"role": "user", "content": "Call square."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "A tool is appropriate.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "square", "arguments": {"x": 4}},
                    }
                ],
            },
            {"role": "tool", "content": "16"},
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "square",
                    "description": "Square an integer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            },
        ),
        enable_thinking=False,
    )
    assert tool_receipt["tool_count"] == 1
    assert tool_receipt["contains_tool_catalog"] is True
    assert tool_receipt["contains_tool_call"] is True
    assert tool_receipt["contains_tool_response"] is True
    assert tool_receipt["ends_with_generation_think"] is True

    with pytest.raises(A.Glm52AdapterError, match="unsupported fields"):
        assembly.assemble_chat(({"role": "user", "content": "x", "extra": 1},))
    with pytest.raises(A.Glm52AdapterError, match="arguments must be an object"):
        assembly.assemble_chat((
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}],
            },
        ))
