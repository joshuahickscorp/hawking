from __future__ import annotations

import json

from tools.odyssey.flash_route_union_census import EXPERTS, LAYERS, TOP_K, derive


def test_dense_teacher_route_union_requires_complete_distinct_coverage(tmp_path):
    tokens = [11, 12]
    rows = []
    for layer in range(LAYERS):
        for step in range(len(tokens)):
            routes = [(layer + step + offset) % EXPERTS for offset in range(TOP_K)]
            rows.append({"layer": layer, "step": step, "route_ids": routes})
    session = {
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "token_ids": tokens,
        "accepted_generation_tokens": 2,
        "execution": {"source_reset_or_reprefill": False, "process_boundary": "one native process"},
        "segments": [{"steps": rows}],
    }
    result = derive(session, tmp_path, expert_family_bytes=LAYERS * EXPERTS * 2)
    assert result["coverage"]["complete"] is True
    assert result["expert_union"]["selected_rows"] == LAYERS * (TOP_K + 1)
    assert result["expert_bank_byte_control"]["teacher_union_bf16_bytes"] == (
        LAYERS * (TOP_K + 1) * 2
    )


def test_attention_rows_inherit_their_layer_from_the_session_segment(tmp_path):
    tokens = [11, 12]
    rows = [
        {"step": step, "route_ids": [(3 + step + offset) % EXPERTS for offset in range(TOP_K)]}
        for step in range(len(tokens))
    ]
    nested = tmp_path / "layer-3-attention.json"
    nested.write_text(json.dumps({"steps": rows}))
    session = {
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "token_ids": tokens,
        "accepted_generation_tokens": 1,
        "execution": {"source_reset_or_reprefill": False, "process_boundary": "one native process"},
        "segments": [
            {"steps": [
                {"layer": layer, "step": step, "route_ids": [(layer + step + offset) % EXPERTS for offset in range(TOP_K)]}
                for layer in list(range(3)) + list(range(4, LAYERS))
                for step in range(len(tokens))
            ]},
            {"layer": 3, "receipt": str(nested)},
        ],
    }
    result = derive(session, tmp_path, expert_family_bytes=LAYERS * EXPERTS * 2)
    assert result["coverage"]["complete"] is True
