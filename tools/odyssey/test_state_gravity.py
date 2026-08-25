"""G037 pins."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/STATE_GRAVITY.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G037 receipt not built")


def rec():
    return json.load(open(R))


def test_the_hybrid_state_split_is_censused_from_config_not_assumed():
    c = rec()["state_census"]
    assert c["full_attention_layers"] + c["linear_attention_layers"] == c["layers"]
    assert c["full_attention_layers"] < c["layers"] / 2


def test_kv_grows_with_context_and_recurrent_state_does_not():
    c = rec()["state_census"]
    assert c["kv"]["grows_with_context"] is True
    assert c["recurrent"]["grows_with_context"] is False
    assert c["crossover_tokens"] > 0


def test_the_kv_precision_ladder_is_labelled_bytes_only():
    d = rec()
    assert len(d["kv_precision_ladder"]) >= 3
    assert "bytes only" in d["kv_ladder_caveat"]
    assert "no KV quantization path" in d["kv_ladder_caveat"]


def test_the_kv_ladder_is_arithmetically_consistent():
    for s in rec()["kv_precision_ladder"]:
        assert s["reduction_x"] == pytest.approx(32.0 / (s["k_bits"] + s["v_bits"]), rel=1e-3)


def test_prefill_was_measured_at_several_lengths():
    p = rec()["prefill"]
    assert len(p["measured_points"]) >= 4
    assert len(p["marginal_cost"]) >= 3
    ns = [x["prompt_tokens"] for x in p["measured_points"]]
    assert ns == sorted(ns) and ns[-1] / ns[0] > 20


def test_prompt_tokens_are_not_cheaper_than_decode_tokens():
    """The whole finding. If prefill were batched this ratio would be far below 1."""
    p = rec()["prefill"]
    assert p["prompt_token_vs_decode_token_ratio"] > 1.0


def test_marginal_prefill_cost_rises_with_length():
    m = [x["marginal_ms_per_prompt_token"] for x in rec()["prefill"]["marginal_cost"]]
    assert m[-1] > m[0]


def test_measurement_and_inference_are_kept_apart():
    """'No batched prefill' is an inference from a timing signature, not a kernel trace."""
    v = rec()["prefill"]["measured_vs_inferred"]
    assert "MEASURED" in v and "INFERRED" in v
    assert "it is an inference" in v["INFERRED"]
    assert v["how_to_settle_it"]


def test_prefix_sharing_uses_a_measured_cost_and_a_measured_prefix():
    p = rec()["prefix_sharing"]
    assert p["shared_prefix"]["shared_tokens"] > 0
    assert p["measured_marginal_ms_per_token"] > 0
    assert p["ttft_saving_ms_if_prefix_reused"] > 0


def test_prefix_sharing_is_declared_a_projection_not_a_delivered_win():
    p = rec()["prefix_sharing"]
    assert "IS_A_PROJECTION" in p
    assert "no prefix cache exists" in p["IS_A_PROJECTION"]


def test_kv_precision_is_ranked_last_and_the_reason_is_structural():
    r = rec()["ranking"]
    assert "batched_prefill" in list(r)[0]
    assert "16 of 64" in r["3_kv_precision"]
