"""Unit tests for the scalar-only independent Llama checkpoint bisection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import llama_checkpoint_bisect as bisect  # noqa: E402


def test_oracle_parser_keeps_first_batched_eval_and_rejects_reshape_duplicates() -> None:
    lines = [
        "common_debug_cb_eval: embd = (f32) GET_ROWS(token_embd.weight{4, 8}, inp_tokens{2, 1}) = {4, 2, 1, 1}\n",
        " sum = 1.0\n",
        "common_debug_cb_eval: Qcur-0 = (f32) MUL_MAT(blk.0.attn_q.weight{4, 4}, attn_norm-0{4, 2}) = {4, 2, 1, 1}\n",
        " sum = 2.0\n",
        "common_debug_cb_eval: Qcur-0 = (f32) RESHAPE(Qcur-0{4, 2}, }) = {2, 2, 2, 1}\n",
        " sum = 99.0\n",
        "common_debug_cb_eval: result_output = (f32) MUL_MAT(output.weight{4, 8}, result_norm{4, 2}) = {8, 2, 1, 1}\n",
        " sum = 3.0\n",
        "common_debug_cb_eval: l_out-0 = (f32) ADD(ffn_out-0{4, 2}, ffn_inp-0{4, 2}) = {4, 2, 1, 1}\n",
        " sum = 6.0\n",
        "common_debug_cb_eval: embd = (f32) GET_ROWS(token_embd.weight{4, 8}, inp_tokens{1, 1}) = {4, 1, 1, 1}\n",
        " sum = 4.0\n",
        "common_debug_cb_eval: Qcur-0 = (f32) MUL_MAT(blk.0.attn_q.weight{4, 4}, attn_norm-0{4, 1}) = {4, 1, 1, 1}\n",
        " sum = 5.0\n",
    ]
    sums, prompt_len = bisect.parse_oracle_lines(lines)
    assert prompt_len == 2
    assert sums == {
        "embedding": 1.0,
        "layer.0.q_raw": 2.0,
        "logits": 3.0,
        "layer.0.layer_out": 6.0,
    }


def test_hawking_summary_aggregates_only_prompt_positions(tmp_path: Path) -> None:
    layer = {
        "layer": 0,
        "attn_norm_sum": 1.0,
        "q_raw_sum": 2.0,
        "k_raw_sum": 3.0,
        "v_raw_sum": 4.0,
        "q_rope_sum": 5.0,
        "k_rope_sum": 6.0,
        "attn_out_sum": 7.0,
        "ffn_input_sum": 8.0,
        "ffn_norm_sum": 9.0,
        "ffn_gate_sum": 10.0,
        "ffn_up_sum": 11.0,
        "ffn_swiglu_sum": 12.0,
        "ffn_out_sum": 13.0,
        "layer_out_sum": 14.0,
    }
    record = {
        "position": 0,
        "token_id": 1,
        "embedding_sum": 1.0,
        "layers": [layer],
        "final_norm_sum": 2.0,
        "logits_sum": 3.0,
        "greedy_token_id": 4,
    }
    second = json.loads(json.dumps(record))
    second.update({"position": 1, "token_id": 2})
    generated = json.loads(json.dumps(record))
    generated.update({"position": 2, "token_id": 4, "embedding_sum": 100.0})
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "schema": bisect.SUMMARY_SCHEMA,
        "prompt_token_ids": [1, 2],
        "records": [record, second, generated],
    }))
    _, sums = bisect.load_hawking_summary(path)
    assert sums["embedding"] == 2.0
    assert sums["layer.0.ffn_out"] == 26.0
    assert sums["logits"] == 6.0


def test_compare_returns_the_first_surface_in_execution_order() -> None:
    oracle = {"embedding": 1.0, "layer.0.q_raw": 10.0, "layer.0.k_raw": 20.0}
    hawking = {"embedding": 1.0, "layer.0.q_raw": 10.5, "layer.0.k_raw": 999.0}
    _, first, missing = bisect.compare(oracle, hawking, abs_tolerance=0.001, rel_tolerance=0.0)
    assert not missing
    assert first is not None
    assert first["surface"] == "layer.0.q_raw"


def test_sequential_logits_find_the_first_prompt_position_and_greedy_mismatch() -> None:
    hawking = {
        "prompt_token_ids": [1, 2],
        "records": [
            {"position": 0, "token_id": 1, "logits_sum": 1.0, "greedy_token_id": 3},
            {"position": 1, "token_id": 2, "logits_sum": 2.0, "greedy_token_id": 4},
        ],
    }
    oracle = {
        "schema": bisect.SEQUENTIAL_ORACLE_SCHEMA,
        "prompt_token_ids": [1, 2],
        "records": [
            {"position": 0, "token_id": 1, "logits_sum": 1.0, "greedy_token_id": 3},
            {"position": 1, "token_id": 2, "logits_sum": 2.0, "greedy_token_id": 5},
        ],
    }
    comparisons, first, error = bisect.compare_sequential_logits(hawking, oracle, 0.001, 0.0)
    assert error is None
    assert len(comparisons) == 2
    assert first is not None
    assert first["position"] == 1
    assert not first["greedy_pass"]


def test_sequential_logits_rejects_different_tokenization() -> None:
    hawking = {"prompt_token_ids": [1], "records": []}
    oracle = {"schema": bisect.SEQUENTIAL_ORACLE_SCHEMA, "prompt_token_ids": [2], "records": []}
    _, _, error = bisect.compare_sequential_logits(hawking, oracle, 0.001, 0.0)
    assert error == "sequential oracle prompt token ids do not match Hawking"


def test_vector_checkpoint_reports_the_worst_element_without_retaining_vectors() -> None:
    hawking = {
        "records": [{
            "debug_vector": {"surface": "layer.17.v_raw", "values": [1.0, -2.0, 3.0]},
        }],
    }
    oracle = {
        "checkpoint": {
            "name": "Vcur-17",
            "captured": True,
            "f32": True,
            "values": [1.0, -1.5, 2.0],
        },
    }
    comparison, error = bisect.compare_checkpoint_vector(
        hawking, oracle, "layer.17.v_raw", "Vcur-17"
    )
    assert error is None
    assert comparison is not None
    assert comparison["value_count"] == 3
    assert comparison["max_abs_error_index"] == 2
    assert comparison["max_abs_error"] == 1.0
