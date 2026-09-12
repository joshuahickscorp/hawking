import json
from pathlib import Path

from tools.accelerator.architecture_atlas import build_atlas

from hcli.architecture import ArchitectureRecognizer
from hcli.tool_registry import default_tool_registry


def test_metadata_recognizer_can_project_the_atlas_without_claiming_execution(tmp_path):
    model_dir = tmp_path / "qwen27"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "Qwen3.8-27B",
                "model_type": "qwen3_next",
                "architectures": ["Qwen3NextForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 48,
                "num_experts": 160,
                "num_experts_per_tok": 8,
            }
        )
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "shard-1.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "shard-1.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "shard-1.safetensors",
                    "model.layers.0.mlp.experts.0.down_proj.weight": "shard-1.safetensors",
                }
            }
        )
    )

    report = ArchitectureRecognizer().inspect(
        model_dir,
        architecture_atlas=build_atlas(),
        backend="metal",
    )

    graph = report["physical_graph"]
    projection = graph["architecture_repatriation"]
    assert projection["selected_behavior_ids"]
    assert projection["atlas_fingerprint"] == build_atlas()["fingerprint"]
    assert report["qualification"]["native_execution_verified"] is False
    assert report["qualification"]["promotion_allowed"] is False
    assert graph["execution_policy"]["architecture_repatriation"]["measurement_authority"].startswith("protected")
    assert "ane" in graph["device_placement"]["candidates"]
    assert graph["provider_context"]["kind"] == "ANEProvider"
    ane = report["ane_model_characterization"]
    assert ane["model_specific"] is True
    assert ane["source_seal"].startswith("metadata:")
    assert ane["source_seal_scope"].endswith("not_weight_identity")
    assert ane["promotion"]["status"] == "WITHHELD"
    assert {row["organ_id"] for row in ane["organs"]} >= {"attention", "moe_experts"}
    assert all(row["measurement_state"] == "UNMEASURED" for row in ane["organs"])
    assert all(
        set(row["backend_matrix"]) == {"cpu", "gpu", "ane"}
        for row in ane["organs"]
    )
    assert all(
        row["backend_matrix"]["gpu"]["measurement_state"] == "UNMEASURED"
        for row in ane["organs"]
    )

    registry = default_tool_registry(
        tmp_path,
        repo_root=Path(__file__).resolve().parents[1],
    )
    inspected = registry.invoke(
        "architecture.inspect",
        {"path": str(model_dir), "backend": "metal"},
    )
    assert inspected.ok is True
    assert inspected.value["physical_graph"]["architecture_repatriation"]["atlas_fingerprint"] == build_atlas()["fingerprint"]
    assert inspected.value["ane_model_characterization"]["model_specific"] is True


def test_architecture_tool_attaches_rust_megakernel_plan_when_native_owner_is_enabled(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 64}))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "shard.safetensors"}})
    )
    monkeypatch.setenv("HCLI_NATIVE_GRAVITY", "1")
    registry = default_tool_registry(tmp_path, repo_root=Path(__file__).resolve().parents[1])
    inspected = registry.invoke("architecture.inspect", {"path": str(model_dir)})
    assert inspected.ok is True
    native = inspected.value["physical_graph"]["native_mega_kernel"]
    assert native["schema"] == "hcli.physical_graph.mega_kernel.v1"
    assert native["registered_native_lowerings"][0]["promotion_eligible"] is False
