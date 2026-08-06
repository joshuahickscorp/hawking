"""Focused contracts for manifest-only DSV4F expert-residency analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity
from lab.receipts import seal, verify


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _descriptor(name: str, *, shape: list[int] | None = None, bytes_: int = 8, dtype: str = "BF16") -> dict:
    return {"name": name, "bytes": bytes_, "dtype": dtype, "shape": shape or [1]}


def _config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "hidden_size": 16,
        "num_hidden_layers": 43,
        "n_routed_experts": 256,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "head_dim": 8,
        "sliding_window": 4,
        "index_head_dim": 4,
        "index_topk": 3,
        "hc_mult": 4,
        "vocab_size": 64,
        "num_hash_layers": 3,
        "max_position_embeddings": 256,
        "compress_ratios": [0, 0, 4, 128] * 10 + [0, 0, 4],
    }


def _inference() -> dict:
    return {
        "n_layers": 43,
        "n_routed_experts": 256,
        "n_activated_experts": 2,
        "head_dim": 8,
        "window_size": 4,
        "hc_mult": 4,
    }


def _add_layer(tensors: dict[str, dict], config: dict, layer: int) -> None:
    prefix = f"layers.{layer}."
    for suffix in (
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "attn.wq_a.weight",
        "attn.wq_a.scale",
        "attn.wq_b.weight",
        "attn.wq_b.scale",
        "attn.wkv.weight",
        "attn.wkv.scale",
        "attn.wo_a.weight",
        "attn.wo_a.scale",
        "attn.wo_b.weight",
        "attn.wo_b.scale",
        "attn_norm.weight",
        "ffn_norm.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ):
        tensors[prefix + suffix] = _descriptor(prefix + suffix)
    ratio = config["compress_ratios"][layer]
    if ratio:
        tensors[prefix + "attn.compressor.wkv.weight"] = _descriptor(
            prefix + "attn.compressor.wkv.weight"
        )
    if ratio == 4:
        tensors[prefix + "attn.indexer.wq_b.weight"] = _descriptor(
            prefix + "attn.indexer.wq_b.weight"
        )
    if layer < config["num_hash_layers"]:
        table = prefix + "ffn.gate.tid2eid"
        tensors[table] = _descriptor(table, shape=[config["vocab_size"], 2], bytes_=config["vocab_size"] * 2 * 8, dtype="I64")
    else:
        tensors[prefix + "ffn.gate.bias"] = _descriptor(prefix + "ffn.gate.bias")
    tensors[prefix + "ffn.gate.weight"] = _descriptor(prefix + "ffn.gate.weight", bytes_=100)
    for expert in range(config["n_routed_experts"]):
        for suffix in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"):
            name = prefix + f"ffn.experts.{expert}.{suffix}"
            tensors[name] = _descriptor(name, bytes_=10)
    for suffix in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"):
        name = prefix + f"ffn.shared_experts.{suffix}"
        tensors[name] = _descriptor(name, bytes_=20)


def _full_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    artifact = tmp_path / "full.gravity"
    config = _config()
    inference = _inference()
    config_path = artifact / "metadata" / "config.json"
    inference_path = artifact / "metadata" / "inference" / "config.json"
    model_path = artifact / "metadata" / "inference" / "model.py"
    _write_json(config_path, config)
    _write_json(inference_path, inference)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("# source-bound test model\n", encoding="utf-8")
    tensors: dict[str, dict] = {}
    for layer in range(43):
        _add_layer(tensors, config, layer)
    tensors["embed.weight"] = _descriptor("embed.weight", shape=[64, 16], bytes_=64 * 16 * 2)
    tensors["norm.weight"] = _descriptor("norm.weight")
    tensors["hc_head_base"] = _descriptor("hc_head_base")
    tensors["hc_head_fn"] = _descriptor("hc_head_fn")
    tensors["hc_head_scale"] = _descriptor("hc_head_scale")
    tensors["head.weight"] = _descriptor("head.weight", shape=[64, 16], bytes_=64 * 16 * 2)
    tensors["mtp.0.stub"] = _descriptor("mtp.0.stub")
    total = sum(item["bytes"] for item in tensors.values())
    manifest = seal(
        {
            "schema": gravity.FULL_ARTIFACT_SCHEMA,
            "status": "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
            "source": {
                "repository": gravity.REPOSITORY,
                "revision": gravity.REVISION,
                "metadata_assets": {
                    "config.json": {"sha256": gravity._sha256(config_path.read_bytes())},
                    "inference/config.json": {"sha256": gravity._sha256(inference_path.read_bytes())},
                    "inference/model.py": {"sha256": gravity._sha256(model_path.read_bytes())},
                },
            },
            "artifact": {"total_tensor_bytes": total},
            "storage": {"source_parent_retained": False},
            "runtime_adapter": {"id": None, "registration": None, "metal_dispatches": 0},
            "tensors": tensors,
        }
    )
    _write_json(artifact / "manifest.json", manifest)
    monkeypatch.setattr(gravity, "FULL_EXPECTED_TENSOR_COUNT", len(tensors))
    return artifact


def _latent_receipt(path: Path, artifact: Path) -> Path:
    manifest = json.loads((artifact / "manifest.json").read_text())
    assets = manifest["source"]["metadata_assets"]
    source_hashes = {name: value["sha256"] for name, value in assets.items()}
    route_set = "a" * 64
    other_route_set = "b" * 64
    value = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_latent_route_receipt.v1",
            "status": "SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_LATENT_ROUTE_CAPTURE",
            "artifact": {
                "seal_sha256": "c" * 64,
                "diagnostic_scope": {"selected_layer": 4, "not_full_model": True},
            },
            "source_hash_binding": {
                "repository": gravity.REPOSITORY,
                "revision": gravity.REVISION,
                "source_parent_retained": False,
                "metadata_asset_sha256": source_hashes,
            },
            "capture": {
                "collection_limits": {
                    "raw_prompts_retained": False,
                    "raw_completions_retained": False,
                    "raw_hidden_states_retained": False,
                },
                "route_aggregate": {
                    "total_source_forwards": 2,
                    "expert_frequency": {"2": 3, "9": 1},
                    "route_set_frequency": {route_set: 1, other_route_set: 1},
                    "route_set_transition_frequency": {f"{route_set}:{other_route_set}": 1},
                    "raw_route_sequence_retained": False,
                },
                "membership_partition": {
                    "excluded_from": ["fit", "calibration", "public_test", "hidden_test"]
                },
            },
        }
    )
    _write_json(path, value)
    return path


def test_static_expert_residency_is_manifest_only_and_keeps_hash_router_lookup_row_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _full_artifact(tmp_path, monkeypatch)
    latent = _latent_receipt(tmp_path / "bounded-latent.json", artifact)

    report = gravity.static_expert_residency_report(
        artifact, out=tmp_path / "static-residency.json", layer4_latent_route_receipt=latent
    )

    verify(report, label="static expert residency report")
    assert report["analysis_mode"] == {
        "manifest_and_metadata_only": True,
        "tensor_payload_bytes_read": 0,
        "runtime_forward_executed": False,
        "gpu_dispatches": 0,
        "command_buffers": 0,
        "base_true_tps": False,
        "training_or_distillation": False,
    }
    assert len(report["per_layer"]) == 43
    layer0 = report["per_layer"][0]
    layer3 = report["per_layer"][3]
    assert layer0["routed_expert_activation_contract"]["per_expert_bundle_logical_bytes_when_uniform"] == 60
    assert layer0["routed_expert_activation_contract"]["top_k_routed_expert_logical_bytes"] == 120
    assert layer0["router_state_contract"]["hash_route_table_selected_row_logical_source_bytes"] == 16
    assert layer0["attention_and_control_logical_bytes"]["router_selected_logical_bytes_per_decode_token"] == 116
    assert layer3["router_state_contract"]["hash_route_table_selected_row_logical_source_bytes"] is None
    assert report["bounded_layer4_route_observations"]["frequency_ranked_experts"][0] == {
        "expert_id": 2,
        "selection_count": 3,
    }
    assert "trace_shards" not in json.dumps(report, sort_keys=True)
    assert report["unavailable_until_native_runtime"]["hot_expert_cache_hit_rate"] == "NOT_MEASURED"


def test_static_expert_residency_parser_has_no_runtime_or_tps_switches() -> None:
    args = gravity._parser().parse_args(
        [
            "static-expert-residency",
            "--full-artifact-dir",
            "/tmp/full.gravity",
            "--layer4-latent-route-receipt",
            "/tmp/routes.json",
            "--out",
            "/tmp/static.json",
        ]
    )
    assert args.command == "static-expert-residency"
    assert args.full_artifact_dir == "/tmp/full.gravity"
    assert args.layer4_latent_route_receipt == "/tmp/routes.json"
