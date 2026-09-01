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
