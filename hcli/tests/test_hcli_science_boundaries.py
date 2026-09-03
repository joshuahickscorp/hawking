from __future__ import annotations

import json
import hashlib
import struct

from hcli.agentos.benchmark_boundary import (
    DIAGNOSTIC_CONTAMINATED,
    QUALIFIED_PROTECTED,
    classify_window,
)
from hcli.agentos.flash_executable import run_flash_executable_scaffold
from hcli.agentos import flash_executable
from hcli.agentos import flash_tensor_probe
from hcli.agentos import flash_representation_experiment
from hcli.agentos import flash_transform_parity
from hcli.agentos import flash_loader_roundtrip
from hcli.agentos import flash_graph_component
from hcli.agentos import flash_component_campaign
from hcli.agentos import flash_router_graph
from hcli.agentos import flash_router_selection
from hcli.flash_next import PINNED_REVISION
from hcli.agentos.fpga_preboard import simulate_partition
from hcli.agentos.protected_benchmark_watcher import _classify_blockers
from hcli.agentos.protected_accelerator_benchmark import _aggregate, _request_record
from hcli.agentos_cli import build_parser
from hcli.nomenclature import (
    CANONICAL_PIPELINE,
    COMPATIBILITY_ALIASES,
    NOMENCLATURE_VERSION,
)
from hcli.physical_graph import compile_physical_graph


def _quiet_sample() -> dict:
    return {"quiet": True, "contenders": [], "method": "test"}


def test_benchmark_boundary_cannot_be_overridden_by_caller_qualification():
    result = classify_window(
        {"quiet": False, "contenders": [{"comm": "WindowServer"}]},
        _quiet_sample(),
        {"state": "QUIESCED"},
        qualification=True,
    )
    assert result["benchmark_class"] == DIAGNOSTIC_CONTAMINATED
    assert result["qualification"] is False
    assert result["NOT_FOR_PROMOTION"] is True


def test_quiet_boundary_is_explicitly_protected_but_still_not_promotion_by_default():
    result = classify_window(_quiet_sample(), _quiet_sample(), {"state": "QUIESCED"})
    assert result["benchmark_class"] == QUALIFIED_PROTECTED
    assert result["qualification"] is True
    assert result["NOT_FOR_PROMOTION"] is True


def test_flash_executable_scaffold_writes_honest_budgets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["status"] == "SCAFFOLD_ONLY"
    assert manifest["promotion_allowed"] is False
    assert manifest["native_loader"]["status"] == "NOT_IMPLEMENTED"
    assert manifest["complete_token_timing"]["accepted_tps"] is None
    assert result["ebpw_budget"]["measured"]["complete_system_ebpw"] is None
    assert result["token_ns_budget"]["system_ledger"]["complete_generation_wall_ns"] is None


def test_flash_executable_ingests_exact_layer0_moe_receipt_without_promoting(tmp_path):
    receipt = tmp_path / "FLASH_NOETIC_EXACT_HYPERCONNECTION_NATIVE.json"
    receipt.write_text(json.dumps({
        "status": "PASSED",
        "schema": "hawking.flash_noetic_exact_hyperconnection_native.v1",
        "qualification": "EXACT_LAYER0_ROUTED_SHARED_MOE_CANDIDATE",
        "layer": 0,
        "execution": {
            "complete_layer0_moe_candidate": True,
            "complete_moe_combine": True,
            "device_intermediate_no_host_roundtrip": True,
        },
        "physical_graph": {"fingerprint": "f" * 64},
        "noetic_ir": {"source_independent": True},
        "source_selection_parity": {"status": "MISMATCH", "top_k_overlap_count": 8},
        "promotion_allowed": False,
    }), encoding="utf-8")

    summary = flash_executable._native_exact_hyperconnection_summary(tmp_path, receipt)

    assert summary["status"] == "PASSED"
    assert summary["complete_layer0_moe_candidate"] is True
    assert summary["complete_moe_combine"] is True
    assert summary["device_intermediate_no_host_roundtrip"] is True
    assert summary["source_selection_parity"]["status"] == "MISMATCH"
    assert summary["promotion_allowed"] is False


def test_flash_executable_prefers_current_modellake_receipt_names(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    lake = tmp_path / "lake"
    final = lake / "specimens" / flash_executable.LAKE_SLUG
    final.mkdir(parents=True)
    (lake / "manifests").mkdir(parents=True)
    (lake / "manifests" / f"{flash_executable.LAKE_SLUG}.json").write_text(
        json.dumps({"resolved_sha": flash_executable.PINNED_REVISION}), encoding="utf-8"
    )
    # Both names are intentionally present: the old HCLI names remain sealed
    # compatibility receipts, while the descriptive names are the current
    # source-specimen observations.
    (receipts / "HCLI_MODELLAKE_FLASH_CENSUS.json").write_text(json.dumps({
        "status": "PASSED",
        "flash_target_manifest": {"final_present": False},
    }), encoding="utf-8")
    (receipts / "MODELLAKE_FLASH_NEXT_CENSUS.json").write_text(json.dumps({
        "status": "PASSED",
        "flash_target_manifest": {"final_present": True},
    }), encoding="utf-8")
    (receipts / "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json").write_text(json.dumps({
        "status": "LEGACY",
    }), encoding="utf-8")
    (receipts / "MODELLAKE_FLASH_NEXT_SUPERVISION.json").write_text(json.dumps({
        "status": "CURRENT",
    }), encoding="utf-8")
    monkeypatch.setattr(flash_executable, "LAKE_ROOT", lake)

    result = flash_executable._modellake_identity(repo)

    assert result["census_receipt"]["path"].endswith("MODELLAKE_FLASH_NEXT_CENSUS.json")
    assert result["supervision_receipt"]["path"].endswith("MODELLAKE_FLASH_NEXT_SUPERVISION.json")
    assert result["census_final_present"] is True
    assert result["observed_job_status"] == "CURRENT"
    assert result["status"] == "VERIFIED_FINAL_IDENTITY"


def test_flash_tensor_probe_reads_bounded_slice_and_keeps_claim_boundary(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_tensor_probe.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    shard_name = "model-00002-of-00131.safetensors"
    tensor_name = flash_tensor_probe.DEFAULT_TENSOR
    values = [0x3F80, 0x4000] * 32
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_tensor_probe.LAKE_SLUG}.json").write_text(
        json.dumps({"repo": "Qwen/Qwen3.8-Flash-Next", "revision": PINNED_REVISION, "resolved_sha": PINNED_REVISION, "n_files": 2}), encoding="utf-8"
    )
    monkeypatch.setattr(flash_tensor_probe, "LAKE_ROOT", lake)
    receipt = tmp_path / "probe.json"

    result = flash_tensor_probe.run_flash_tensor_probe(
        root=specimen,
        sample_bytes=len(payload),
        emit=receipt,
    )

    assert result["status"] == "PASSED"
    assert result["source_label"] == "[V]"
    assert result["candidate_label"] == "[D]"
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False
    assert result["source_tensor"]["slice_bytes"] == len(payload)
    assert result["dense_vs_packed_low_bit"]["candidate"]["effective_bits_per_value"] == 4.25
    assert result["dense_vs_packed_low_bit"]["comparison"]["capability_parity"] == "NOT_TESTED"


def test_flash_executable_scaffold_ingests_probe_as_bounded_evidence(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_FIRST_TENSOR_PROBE.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "candidate_label": "[D]",
        "tensor_name": "probe.tensor",
        "organ": {"id": "routed_experts", "label": "[D]"},
        "dense_vs_packed_low_bit": {
            "candidate": {"scheme": "test", "effective_bits_per_value": 4.25},
            "comparison": {"candidate_is_smaller": True},
        },
        "body_mutated": False,
        "model_loaded": False,
    }), encoding="utf-8")
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    assert result["manifest"]["source_tensor_probe"]["status"] == "PASSED"
    assert result["manifest"]["chosen_representation"]["status"] == "BOUNDED_SLICE_OBSERVED_NOT_WHOLE_MODEL"
    assert result["manifest"]["complete_token_timing"]["accepted_tps"] is None


def test_flash_executable_scaffold_ingests_full_transform_as_tensor_ebpw_only(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "candidate_label": "[D]",
        "tensor_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "source_tensor": {"shape": [2, 4, 64], "payload_bytes": 1024, "layout": "row-major"},
        "candidates": {
            "independent_q4_g64": {
                "candidate_bytes": 300,
                "effective_bits_per_value": 4.6875,
                "weight_reconstruction": {"cosine": 0.99},
                "reference_vector": {"cosine": 0.98},
            },
            "shared_bf16_basis_nf4_residual": {
                "candidate_bytes": 320,
                "effective_bits_per_value": 5.0,
                "weight_reconstruction": {"cosine": 0.995},
                "reference_vector": {"cosine": 0.99},
            },
        },
        "comparison": {
            "dense_bytes": 1024,
            "full_payload_read": True,
            "capability_parity": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "transform_parity": {"status": "PASSED", "pack_unpack_parity": True},
        "next_experiment": {"id": "flash-routed-expert-bounded-loader-roundtrip"},
        "body_mutated": False,
        "model_loaded": False,
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["source_transform_parity"]["status"] == "PASSED"
    assert manifest["chosen_representation"]["status"] == "FULL_TENSOR_TRANSFORM_OBSERVED_NOT_WHOLE_MODEL"
    assert result["ebpw_budget"]["bounded_tensor_observation"]["is_complete_system"] is False
    assert result["ebpw_budget"]["bounded_tensor_observation"]["candidates"]["independent_q4_g64"]["effective_bits_per_value"] == 4.6875
    assert manifest["promotion_allowed"] is False


def test_flash_representation_experiment_uses_source_layout_and_direct_candidate_dots(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_representation_experiment.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_representation_experiment.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    # Two experts x two complete rows x 64 columns: enough to exercise the
    # source [expert, row, column] layout and one full G64 group per row.
    values = []
    for expert in range(2):
        for row in range(2):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 2, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_representation_experiment.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_representation_experiment, "LAKE_ROOT", lake)

    result = flash_representation_experiment.run_flash_representation_experiment(
        root=specimen,
        expert_indices=[0, 1],
        row_count=2,
        emit=tmp_path / "representation.json",
    )

    assert result["status"] == "PASSED"
    assert result["source_tensor"]["layout"].startswith("row-major [expert, row, column]")
    assert result["source_tensor"]["values_read"] == 256
    assert result["candidates"]["independent_q4_g64"]["direct_representation_dot"] is True
    assert result["candidates"]["shared_bf16_basis_nf4_residual"]["direct_representation_dot"] is True
    assert result["comparison"]["same_source_rows"] is True
    assert result["comparison"]["capability_parity"] == "NOT_TESTED"
    assert result["body_mutated"] is False


def test_flash_transform_parity_streams_complete_tensor_without_runtime_claim(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_transform_parity.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_transform_parity.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    values = []
    for expert in range(2):
        for row in range(4):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 4, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_transform_parity.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_transform_parity, "LAKE_ROOT", lake)

    result = flash_transform_parity.run_flash_transform_parity(
        root=specimen,
        chunk_rows=2,
        emit=tmp_path / "transform.json",
    )

    assert result["status"] == "PASSED"
    assert result["source_tensor"]["bytes_read_pass_one"] == len(payload)
    assert result["source_tensor"]["bytes_read_pass_two"] == len(payload)
    assert result["transform_parity"]["complete_source_payload_read"] is True
    assert result["transform_parity"]["pack_unpack_parity"] is True
    assert result["comparison"]["full_payload_read"] is True
    assert result["whole_model_capability"] == "NOT_TESTED"
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False


def test_flash_loader_roundtrip_serializes_descriptor_without_model_load(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_loader_roundtrip.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_loader_roundtrip.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    values = []
    for expert in range(2):
        for row in range(4):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 4, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_loader_roundtrip.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_transform_parity, "LAKE_ROOT", lake)
    monkeypatch.setattr(flash_loader_roundtrip, "LAKE_ROOT", lake)
    transform_receipt = tmp_path / "transform.json"
    transform_result = flash_transform_parity.run_flash_transform_parity(
        root=specimen,
        chunk_rows=2,
        emit=transform_receipt,
    )
    assert transform_result["status"] == "PASSED"

    result = flash_loader_roundtrip.run_flash_loader_roundtrip(
        root=specimen,
        transform_receipt=transform_receipt,
        candidate_id="independent_q4_g64",
        row_count=2,
        emit=tmp_path / "loader.json",
    )

    assert result["status"] == "PASSED"
    assert result["native_loader"] == "BOUNDED_DESCRIPTOR_ROUNDTRIP_ONLY"
    assert result["loader_roundtrip"]["descriptor_json_roundtrip"] is True
    assert result["loader_roundtrip"]["code_pack_unpack_parity"] is True
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False


def test_flash_graph_component_compiles_validated_noetic_receipts_without_promotion(tmp_path):
    root = tmp_path / "repo"
    receipts = root / "receipts" / "headless"
    receipts.mkdir(parents=True)
    specimen = str(tmp_path / "specimen")
    revision = PINNED_REVISION
    manifest = {
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "revision": revision,
        "resolved_sha": revision,
        "path": specimen,
    }
    tensor = {
        "tensor_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "dtype": "BF16",
        "shape": [512, 1280, 2560],
        "layout": "row-major [expert, row, column]",
        "group_size": 64,
    }
    candidate = {
        "candidate_bytes": 891289600,
        "candidate_sha256": "c" * 64,
        "effective_bits_per_value": 4.25,
        "status": "FULL_TENSOR_TRANSFORM_ONLY",
    }
    transform = {
        "schema": "hcli.agentos.flash_transform_parity.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "model_lake_manifest": manifest,
        "candidates": {"independent_q4_g64": candidate},
    }
    descriptor = {
        "schema": "hcli.noetic.representation_descriptor.v1",
        "candidate_id": "independent_q4_g64",
        "source_tensor": tensor,
        "full_transform_reference": candidate,
    }
    loader = {
        "schema": "hcli.agentos.flash_loader_roundtrip.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "model_lake_manifest": manifest,
        "candidate_id": "independent_q4_g64",
        "representation_descriptor": descriptor,
        "body_mutated": False,
        "model_loaded": False,
    }
    kernel = {
        "schema": "hawking.flash_noetic_q4_kernel_parity.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "model_lake_manifest": manifest,
        "source_tensor": {**tensor, "selected_block_bytes": 655360},
        "noetic_descriptor": {"schema": "hcli.noetic.representation_descriptor.v1"},
        "noetic_representation": {"candidate_id": "independent_q4_g64"},
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_LOAD",
            "candidate_id": "independent_q4_g64",
            "descriptor_sha256": "d" * 64,
        },
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "kernel_registered": True,
            "dispatches_per_sample": 1,
        },
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
    }
    transform_path = receipts / "transform.json"
    loader_path = receipts / "loader.json"
    kernel_path = receipts / "kernel.json"
    transform_path.write_text(json.dumps(transform), encoding="utf-8")
    loader_path.write_text(json.dumps(loader), encoding="utf-8")
    kernel_path.write_text(json.dumps(kernel), encoding="utf-8")

    result = flash_graph_component.run_flash_graph_component(
        repo_root=root,
        transform_receipt=transform_path,
        loader_receipt=loader_path,
        kernel_receipt=kernel_path,
        emit=receipts / "graph.json",
    )

    assert result["status"] == "PASSED"
    assert result["component_status"] == "BOUNDED_COMPONENT_COMPILED"
    assert result["physical_graph"]["compiler_stage"] == "PhysicalGraphCompiler"
    assert result["noetic_ir"]["source_independent"] is False
    assert result["candidate_body_persisted"] is False
    assert result["promotion_allowed"] is False

    body_kernel = json.loads(json.dumps(kernel))
    body_kernel["native_loader"].update({
        "status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
        "source_independent_execution": True,
        "candidate_body_persisted": True,
    })
    body_kernel["candidate_body"] = {
        "path": str(tmp_path / "component.bin"),
        "bytes": 174080,
        "source_independent": True,
    }
    body_kernel_path = receipts / "body-kernel.json"
    body_kernel_path.write_text(json.dumps(body_kernel), encoding="utf-8")
    body_result = flash_graph_component.run_flash_graph_component(
        repo_root=root,
        transform_receipt=transform_path,
        loader_receipt=loader_path,
        kernel_receipt=body_kernel_path,
        emit=receipts / "body-graph.json",
    )

    assert body_result["status"] == "PASSED"
    assert body_result["source_backed"] is False
    assert body_result["source_independent_execution"] is True
    assert body_result["candidate_body_persisted"] is True
    assert "noetic_component_body_load" in {
        node["id"] for node in body_result["physical_graph"]["computation"]
    }


def test_flash_component_campaign_composes_two_source_independent_blocks(tmp_path):
    root = tmp_path / "repo"
    receipts = root / "receipts" / "headless"
    receipts.mkdir(parents=True)
    specimen = str(tmp_path / "specimen")
    revision = PINNED_REVISION
    manifest = {
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "revision": revision,
        "resolved_sha": revision,
        "path": specimen,
    }
    tensor = {
        "tensor_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "dtype": "BF16",
        "shape": [512, 1280, 2560],
        "layout": "row-major [expert, row, column]",
        "group_size": 64,
    }
    candidate = {
        "candidate_bytes": 891289600,
        "candidate_sha256": "c" * 64,
        "effective_bits_per_value": 4.25,
        "status": "FULL_TENSOR_TRANSFORM_ONLY",
    }
    transform = {
        "schema": "hcli.agentos.flash_transform_parity.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "model_lake_manifest": manifest,
        "candidates": {"independent_q4_g64": candidate},
    }
    descriptor = {
        "schema": "hcli.noetic.representation_descriptor.v1",
        "candidate_id": "independent_q4_g64",
        "source_tensor": tensor,
        "full_transform_reference": candidate,
    }
    loader = {
        "schema": "hcli.agentos.flash_loader_roundtrip.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "model_lake_manifest": manifest,
        "candidate_id": "independent_q4_g64",
        "representation_descriptor": descriptor,
        "body_mutated": False,
        "model_loaded": False,
    }
    transform_path = receipts / "transform.json"
    loader_path = receipts / "loader.json"
    transform_path.write_text(json.dumps(transform), encoding="utf-8")
    loader_path.write_text(json.dumps(loader), encoding="utf-8")
    specs = []
    for index, row_start in enumerate((0, 128)):
        body_bytes = bytes([index + 1]) * 8
        body_path = tmp_path / f"body-{index}.bin"
        body_path.write_bytes(body_bytes)
        body_receipt_path = receipts / f"body-{index}.json"
        body = {
            "schema": "hcli.agentos.flash_noetic_component_body.v1",
            "status": "PASSED",
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": revision,
            "root": specimen,
            "model_lake_manifest": manifest,
            "candidate_id": "independent_q4_g64",
            "source_independent": True,
            "candidate_body_persisted": True,
            "body_mutated": False,
            "model_loaded": False,
            "source_block": {**tensor, "expert_index": 0, "row_start": row_start, "row_count": 128, "bytes": 655360, "payload_sha256": "a" * 64},
            "body": {
                "path": str(body_path),
                "sha256": hashlib.sha256(body_bytes).hexdigest(),
                "bytes": len(body_bytes),
            },
        }
        body_receipt_path.write_text(json.dumps(body), encoding="utf-8")
        kernel_receipt_path = receipts / f"kernel-{index}.json"
        kernel = {
            "schema": "hawking.flash_noetic_q4_kernel_parity.v1",
            "status": "PASSED",
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": revision,
            "root": specimen,
            "model_lake_manifest": manifest,
            "source_tensor": {**tensor, "selected_expert": 0, "selected_row_start": row_start, "selected_row_count": 128, "selected_block_bytes": 655360, "selected_block_sha256": "a" * 64},
            "noetic_descriptor": {"schema": "hcli.noetic.representation_descriptor.v1"},
            "noetic_representation": {"candidate_id": "independent_q4_g64"},
            "native_loader": {"status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD", "candidate_id": "independent_q4_g64", "descriptor_sha256": "d" * 64, "source_independent_execution": True, "candidate_body_persisted": True},
            "native_kernel": {"kernel": "qwen_uniform_q4_group64_matvec", "kernel_registered": True, "dispatches_per_sample": 1},
            "parity": {"within_tolerance": True},
            "candidate_body": {"path": str(body_path), "sha256": body["body"]["sha256"], "bytes": len(body_bytes), "receipt_path": str(body_receipt_path), "source_independent": True},
            "body_mutated": False,
            "model_loaded": False,
        }
        kernel_receipt_path.write_text(json.dumps(kernel), encoding="utf-8")
        specs.append({"id": f"e0_r{row_start}_128", "body_receipt": str(body_receipt_path), "kernel_receipt": str(kernel_receipt_path)})

    result = flash_component_campaign.run_flash_component_campaign(
        repo_root=root,
        loader_receipt=loader_path,
        transform_receipt=transform_path,
        component_specs=specs,
        emit=receipts / "campaign.json",
    )

    assert result["status"] == "PASSED"
    assert result["component_status"] == "BOUNDED_MULTI_COMPONENT_COMPILED"
    assert result["component_count"] == 2
    assert result["candidate_body_persisted"] is True
    assert result["physical_graph"]["qualification"] == "BOUNDED_MULTI_COMPONENT_ONLY"
    assert any(node["id"].startswith("e0_r128_128:") for node in result["physical_graph"]["computation"])
    assert result["noetic_ir"]["complete_model"] is False


def test_flash_router_graph_compiles_rank_two_source_independent_body(tmp_path):
    root = tmp_path / "repo"
    receipts = root / "receipts" / "headless"
    receipts.mkdir(parents=True)
    specimen = str(tmp_path / "specimen")
    revision = PINNED_REVISION
    body_bytes = bytes(range(8))
    body_path = tmp_path / "router.bin"
    body_path.write_bytes(body_bytes)
    body_receipt_path = receipts / "router-body.json"
    tensor = {
        "tensor_name": "model.language_model.layers.0.mlp.gate.weight",
        "dtype": "BF16",
        "shape": [512, 2560],
    }
    descriptor = {
        "schema": "hcli.noetic.representation_descriptor.v1",
        "candidate_id": "independent_q4_g64",
        "source_tensor": {**tensor, "layout": "row-major [row, column]", "group_size": 64},
        "storage": {
            "code_dtype": "uint4_packed",
            "code_offset": 8,
            "nibble_order": "low_nibble_then_high_nibble_row_major",
            "scale_dtype": "little_endian_float16",
        },
        "transform_reference": {"status": "BOUNDED_TENSOR_TRANSFORM_ONLY"},
        "loader_policy": {"source_mutation": False, "model_load": False, "dense_rematerialization": "forbidden"},
    }
    body = {
        "schema": "hcli.agentos.flash_noetic_component_body.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "component_kind": "router",
        "candidate_id": "independent_q4_g64",
        "source_independent": True,
        "candidate_body_persisted": True,
        "body_mutated": False,
        "model_loaded": False,
        "source_block": {**tensor, "row_start": 0, "row_count": 128, "bytes": 655360, "payload_sha256": "b" * 64},
        "representation_descriptor": descriptor,
        "body": {"path": str(body_path), "sha256": hashlib.sha256(body_bytes).hexdigest(), "bytes": len(body_bytes)},
    }
    body_receipt_path.write_text(json.dumps(body), encoding="utf-8")
    kernel_receipt_path = receipts / "router-kernel.json"
    kernel = {
        "schema": "hawking.flash_noetic_q4_kernel_parity.v1",
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": revision,
        "root": specimen,
        "source_tensor": {**tensor, "selected_row_start": 0, "selected_row_count": 128, "selected_block_bytes": 655360, "selected_block_sha256": "b" * 64},
        "native_loader": {"status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD", "source_independent_execution": True, "candidate_body_persisted": True},
        "native_kernel": {"kernel": "qwen_uniform_q4_group64_matvec", "kernel_registered": True, "dispatches_per_sample": 1},
        "parity": {"within_tolerance": True},
        "candidate_body": {"path": str(body_path), "sha256": body["body"]["sha256"], "bytes": len(body_bytes)},
        "body_mutated": False,
        "model_loaded": False,
    }
    kernel_receipt_path.write_text(json.dumps(kernel), encoding="utf-8")
    result = flash_router_graph.run_flash_router_graph(
        repo_root=root,
        body_receipt=body_receipt_path,
        kernel_receipt=kernel_receipt_path,
        emit=receipts / "router-graph.json",
    )
    assert result["status"] == "PASSED"
    assert result["component_status"] == "BOUNDED_ROUTER_MATRIX_COMPILED"
    assert result["physical_graph"]["qualification"] == "BOUNDED_ROUTER_MATRIX_ONLY"
    assert result["noetic_ir"]["complete_model"] is False
    assert result["promotion_allowed"] is False


def test_flash_router_selection_applies_stable_fp32_softmax_top_k_and_normalization():
    import numpy as np

    result = flash_router_selection.select_router(
        np,
        np.asarray([0.0, 2.0, 1.0, 2.0, -1.0], dtype=np.float32),
        top_k=3,
        norm_topk_prob=True,
    )
    assert result["expert_ids"] == [1, 3, 2]
    assert result["probabilities_finite"] is True
    assert abs(result["selected_weight_sum"] - 1.0) < 1e-6
    assert result["selected_weights"][0] == result["selected_weights"][1]


def test_canonical_nomenclature_is_versioned_without_renaming_legacy_terms():
    assert NOMENCLATURE_VERSION == "HAWKING_NOMENCLATURE_V1"
    assert CANONICAL_PIPELINE[0] == "SourceSpecimen"
    assert CANONICAL_PIPELINE[-1] == "ResidentInstance"
    assert COMPATIBILITY_ALIASES["quantization"] == "GravityOperator"
    assert COMPATIBILITY_ALIASES["artifact"] == "SemanticInspectionRequired"
    graph = compile_physical_graph({"model_id": "flash-next", "organs": []})
    assert graph["nomenclature_version"] == NOMENCLATURE_VERSION
    assert graph["semantic_type"] == "PhysicalGraphPlan"
    assert graph["compiler_stage"] == "PhysicalGraphCompiler"


def test_flash_executable_ingests_bounded_kernel_evidence_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_NOETIC_Q4_KERNEL_PARITY.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "derived_label": "[D]",
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "source_tensor": {"selected_block_bytes": 128},
        "noetic_representation": {"candidate_id": "independent_q4_g64"},
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_LOAD",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "gpu_timing": {"gpu_ns_median": 123},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
    }), encoding="utf-8")
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    assert result["manifest"]["source_kernel_parity"]["status"] == "PASSED"
    assert result["manifest"]["native_loader"]["bounded_native_descriptor_load_status"] == "BOUNDED_NOETIC_DESCRIPTOR_LOAD"
    assert result["manifest"]["native_kernels"]["status"] == "PLAN_ONLY"
    assert result["manifest"]["complete_token_timing"]["accepted_tps"] is None
    assert result["manifest"]["promotion_allowed"] is False


def test_flash_executable_records_shared_expert_kernel_lane_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    shared = receipts / "shared-kernel.json"
    shared.write_text(json.dumps({
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": PINNED_REVISION,
        "source_tensor": {
            "tensor_name": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "shape": [640, 2560],
            "selected_row_start": 0,
            "selected_row_count": 128,
        },
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
            "source_independent_execution": True,
            "candidate_body_persisted": True,
        },
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "gpu_timing": {"gpu_ns_median": 456},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "promotion_allowed": False,
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(
        repo_root=repo,
        science_receipt=source,
        shared_expert_kernel_parity_receipt=shared,
    )

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["status"] == "SCAFFOLD_ONLY"
    assert manifest["source_shared_expert_kernel_parity"]["status"] == "PASSED"
    assert manifest["native_kernels"]["bounded_shared_expert_matrix_evidence"]["status"] == "PASSED"
    assert manifest["native_kernels"]["bounded_shared_expert_matrix_evidence"]["source_independent_execution"] is True
    assert manifest["promotion_allowed"] is False


def test_flash_executable_records_deltanet_kernel_lane_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    deltanet = receipts / "deltanet-kernel.json"
    deltanet.write_text(json.dumps({
        "status": "PASSED",
        "repo": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": PINNED_REVISION,
        "source_tensor": {
            "tensor_name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "shape": [10240, 2560],
            "selected_row_start": 0,
            "selected_row_count": 128,
        },
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
            "source_independent_execution": True,
            "candidate_body_persisted": True,
        },
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "gpu_timing": {"gpu_ns_median": 789},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "promotion_allowed": False,
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(
        repo_root=repo,
        science_receipt=source,
        deltanet_kernel_parity_receipt=deltanet,
    )

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["source_deltanet_kernel_parity"]["status"] == "PASSED"
    evidence = manifest["native_kernels"]["bounded_deltanet_matrix_evidence"]
    assert evidence["status"] == "PASSED"
    assert evidence["source_independent_execution"] is True
    assert manifest["runtime_genome"]["deltanet_kernel_parity_receipt"] == str(deltanet.resolve())
    assert manifest["promotion_allowed"] is False


def test_flash_executable_records_sparse_attention_kernel_lane_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    sparse = receipts / "sparse-kernel.json"
    sparse.write_text(json.dumps({
        "status": "PASSED",
        "source_tensor": {
            "tensor_name": "model.language_model.layers.11.self_attn.indexer.index_qk_proj.weight",
            "shape": [640, 2560],
            "selected_row_start": 0,
            "selected_row_count": 128,
        },
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
            "source_independent_execution": True,
            "candidate_body_persisted": True,
        },
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "gpu_timing": {"gpu_ns_median": 321},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "promotion_allowed": False,
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(
        repo_root=repo,
        science_receipt=source,
        sparse_attention_kernel_parity_receipt=sparse,
    )

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["source_sparse_attention_kernel_parity"]["status"] == "PASSED"
    evidence = manifest["native_kernels"]["bounded_sparse_attention_matrix_evidence"]
    assert evidence["status"] == "PASSED"
    assert evidence["source_independent_execution"] is True
    assert manifest["runtime_genome"]["sparse_attention_kernel_parity_receipt"] == str(sparse.resolve())
    assert manifest["promotion_allowed"] is False


def test_flash_executable_records_mtp_gate_kernel_lane_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    mtp = receipts / "mtp-kernel.json"
    mtp.write_text(json.dumps({
        "status": "PASSED",
        "source_tensor": {
            "tensor_name": "mtp.layers.0.mlp.gate.weight",
            "shape": [512, 2560],
            "selected_row_start": 0,
            "selected_row_count": 128,
        },
        "native_loader": {
            "status": "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD",
            "source_independent_execution": True,
            "candidate_body_persisted": True,
        },
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "gpu_timing": {"gpu_ns_median": 321},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "promotion_allowed": False,
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(
        repo_root=repo,
        science_receipt=source,
        mtp_gate_kernel_parity_receipt=mtp,
    )

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["source_mtp_gate_kernel_parity"]["status"] == "PASSED"
    evidence = manifest["native_kernels"]["bounded_mtp_gate_matrix_evidence"]
    assert evidence["status"] == "PASSED"
    assert evidence["source_independent_execution"] is True
    assert manifest["runtime_genome"]["mtp_gate_kernel_parity_receipt"] == str(mtp.resolve())
    assert manifest["promotion_allowed"] is False


def test_fpga_partition_simulation_is_model_specific_and_never_hardware():
    qwen = simulate_partition("qwen27")
    flash = simulate_partition("flash-next")
    assert len(qwen["scenarios"]) == 3
    assert len(flash["scenarios"]) >= 6
    assert all(row["label"] == "[S]" for row in qwen["scenarios"] + flash["scenarios"])
    assert all(row["physical_execution"] is False for row in qwen["scenarios"] + flash["scenarios"])
    assert {row["decision"] for row in flash["scenarios"]} <= {"MIXED_CANDIDATE", "REJECT_MIXED_IF_NOT_BEAT"}


def test_watcher_never_classifies_modellake_as_pausable():
    classes = _classify_blockers([
        {"job_id": "lake", "label": "modellake-flash-next-acquire", "argv": ["python3", "tools/odyssey/modellake.py", "acquire"], "pid": 7},
        {"job_id": "bench", "label": "hcli-diagnostic", "argv": ["python3", "-m", "hcli", "agentos", "accelerator-regression"], "pid": 8},
    ])
    assert classes["blockers"][0]["kind"] == "MODELLAKE_UNTOUCHABLE"
    assert classes["pausable_hcli_jobs"][0]["job_id"] == "bench"


def test_protected_accelerator_benchmark_normalizes_provider_metrics_without_model_assumptions():
    raw = {
        "hawking": {
            "generated_tokens": 4,
            "new_token_ids": [10, 11, 12, 13],
            "fallbacks": 0,
            "prompt_tokens": 7,
            "resident_health": {"pid": 42, "model_open_count": 1, "weight_upload_count": 1},
            "native_metrics": {
                "gpu_ns": 80,
                "gpu_ns_per_generated_token": 20,
                "wall_minus_gpu_ns": 8,
                "dispatches": 40,
                "dispatches_per_generated_token": 10,
                "prefill": {"steps": 7, "wall_ns": 70},
                "decode": {"steps": 4, "wall_ns": 100},
                "kernel_genome": {"histogram": [["test_kernel", 4]]},
                "capability": {"complete_token_accounting": True},
            },
        }
    }
    row = _request_record(raw, elapsed_ns=100, index=1, phase="measure")
    summary = _aggregate([row, {**row, "index": 2}])
    assert summary["kernel_genome_exact_and_stable"] is True
    assert summary["generated_token_ids_exact_and_stable"] is True
    assert row["complete_wall_ns_per_token"] == 25
    assert row["gpu_ns_per_token"] == 20
    assert row["wall_minus_gpu_ns_per_token"] == 5
    assert row["wall_minus_gpu_metric_source"] == "derived_complete_wall_minus_gpu"
    assert row["native_wall_minus_gpu_ns"] == 8
    assert row["native_wall_minus_gpu_ns_per_token"] == 2
    assert row["native_wall_minus_gpu_metric_source"] == "derived_from_native_wall_minus_gpu_ns"
    assert summary["native_wall_minus_gpu_ns"]["median"] == 8
    assert summary["native_wall_minus_gpu_ns_per_token"]["median"] == 2
    assert row["dispatches_per_token"] == 10
    assert row["capability_sanity"]["status"] == "PASS"


def test_cli_exposes_general_science_surfaces():
    parser = build_parser()
    assert parser.parse_args(["qwen27-runtime-archaeology"]).command == "qwen27-runtime-archaeology"
    assert parser.parse_args(["qwen27-mlp-ab"]).command == "qwen27-mlp-ab"
    assert parser.parse_args(["flash-executable"]).command == "flash-executable"
    assert parser.parse_args(["flash-tensor-probe"]).command == "flash-tensor-probe"
    assert parser.parse_args(["flash-representation-experiment"]).command == "flash-representation-experiment"
    assert parser.parse_args(["flash-transform-parity"]).command == "flash-transform-parity"
    assert parser.parse_args(["flash-loader-roundtrip"]).command == "flash-loader-roundtrip"
    assert parser.parse_args(["flash-component-body"]).command == "flash-component-body"
    assert parser.parse_args(["flash-matrix-body"]).command == "flash-matrix-body"
    assert parser.parse_args(["flash-vector-body"]).command == "flash-vector-body"
    assert parser.parse_args(["flash-router-graph"]).command == "flash-router-graph"
    assert parser.parse_args(["flash-router-selection"]).command == "flash-router-selection"
    assert parser.parse_args(["flash-router-representation-ab"]).command == "flash-router-representation-ab"
    assert parser.parse_args(["flash-component-campaign"]).command == "flash-component-campaign"
    assert parser.parse_args(["flash-graph-component"]).command == "flash-graph-component"
    assert parser.parse_args(["flash-executable", "--kernel-parity-receipt", "kernel.json"]).kernel_parity_receipt == "kernel.json"
    assert parser.parse_args(["flash-executable", "--deltanet-kernel-parity-receipt", "deltanet.json"]).deltanet_kernel_parity_receipt == "deltanet.json"
    assert parser.parse_args(["flash-executable", "--sparse-attention-kernel-parity-receipt", "sparse.json"]).sparse_attention_kernel_parity_receipt == "sparse.json"
    assert parser.parse_args(["flash-executable", "--mtp-gate-kernel-parity-receipt", "mtp.json"]).mtp_gate_kernel_parity_receipt == "mtp.json"
    assert parser.parse_args(["flash-executable", "--graph-component-receipt", "graph.json"]).graph_component_receipt == "graph.json"
    assert parser.parse_args(["flash-executable", "--router-selection-receipt", "selection.json"]).router_selection_receipt == "selection.json"
    assert parser.parse_args(["flash-executable", "--native-router-selection-receipt", "native-selection.json"]).native_router_selection_receipt == "native-selection.json"
    assert parser.parse_args(["flash-executable", "--native-routed-expert-dispatch-receipt", "native-dispatch.json"]).native_routed_expert_dispatch_receipt == "native-dispatch.json"
    assert parser.parse_args(["flash-executable", "--native-gate-up-swiglu-receipt", "native-gate-up.json"]).native_gate_up_swiglu_receipt == "native-gate-up.json"
    assert parser.parse_args(["flash-executable", "--native-expert-composition-receipt", "native-composition.json"]).native_expert_composition_receipt == "native-composition.json"
    assert parser.parse_args(["flash-executable", "--native-shared-expert-composition-receipt", "native-shared-composition.json"]).native_shared_expert_composition_receipt == "native-shared-composition.json"
    assert parser.parse_args(["flash-executable", "--native-shared-residual-hyperconnection-receipt", "native-shared-residual.json"]).native_shared_residual_hyperconnection_receipt == "native-shared-residual.json"
    assert parser.parse_args(["flash-executable", "--native-exact-hyperconnection-receipt", "native-exact.json"]).native_exact_hyperconnection_receipt == "native-exact.json"
    assert parser.parse_args(["flash-executable", "--router-representation-ab-receipt", "router-ab.json"]).router_representation_ab_receipt == "router-ab.json"
    assert parser.parse_args(["protected-bench-watch"]).command == "protected-bench-watch"
    assert parser.parse_args(["protected-accelerator-bench"]).command == "protected-accelerator-bench"
    parsed = parser.parse_args([
        "protected-accelerator-bench",
        "--fusion-env",
        "HAWKING_QWEN38_FUSE_MLP=pair",
        "--fusion-env",
        "HAWKING_QWEN38_FUSE_GQA_QKV=1",
    ])
    assert parsed.fusion_env == [
        ("HAWKING_QWEN38_FUSE_MLP", "pair"),
        ("HAWKING_QWEN38_FUSE_GQA_QKV", "1"),
    ]
