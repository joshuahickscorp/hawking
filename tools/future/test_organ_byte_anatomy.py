from __future__ import annotations

from tools.future.organ_byte_anatomy import classify


def test_qwen4_exp_stacked_experts_have_distinct_routed_owner() -> None:
    assert classify("model.language_model.layers.1.mlp.experts.gate_up_proj") == (
        "routed_experts", "language"
    )
    assert classify("model.language_model.layers.1.mlp.experts.down_proj") == (
        "routed_experts", "language"
    )
    assert classify("mtp.layers.0.mlp.experts.down_proj") == (
        "mtp_routed_experts", "language"
    )
    assert classify("mtp.layers.0.mlp.shared_expert_gate.weight") == (
        "mtp_shared_expert_router", "language"
    )


def test_qwen4_exp_state_and_vision_organs_are_not_silently_unknown() -> None:
    assert classify("model.language_model.layers.0.linear_attn.in_proj_qkv.weight") == (
        "recurrent_state", "language"
    )
    assert classify("model.language_model.layers.0.mlp.shared_expert_gate.weight") == (
        "shared_expert_router", "language"
    )
    assert classify("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight") == (
        "hyper_connection", "language"
    )
    assert classify("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight") == (
        "positional_local_embedding", "language"
    )
    assert classify("model.visual.blocks.0.attn.qkv.weight") == ("vision", "vision")


def test_qwen4_rules_do_not_become_a_catch_all() -> None:
    assert classify("model.language_model.layers.1.mlp.experts.quantum_proj") == (
        "UNCLASSIFIED", "unknown"
    )
    assert classify("model.language_model.layers.1.alien_state.weight") == (
        "UNCLASSIFIED", "unknown"
    )
