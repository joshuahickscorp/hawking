from __future__ import annotations

from tools.odyssey.ple_access_trace import (
    _shift_right_ignore_eos,
    build_contract,
    trace_token_segments,
)


def _small_contract():
    return build_contract(
        vocab_size=1000,
        eos_token_id=999,
        ngram_size=3,
        heads_per_ngram=1,
        ngram_vocab_size_base=31,
        make_vocab_divisible_by=8,
        ple_embed_dim=16,
        ple_layer_index=0,
        source_layer_index=1,
        seed=1234,
    )


def _rows(trace):
    return [
        [int(access["combined_row"]) for access in token["accesses"]]
        for token in trace
    ]


def test_scalar_contract_matches_small_reference_implementation_probe() -> None:
    contract = _small_contract()
    trace = trace_token_segments(contract, [[3, 4, 5, 6]])
    # Golden values were captured from Qwen4ExpTextNGramEmbedding.forward in
    # transformers 5.17.0 with its embedding replaced by a recording module.
    assert _rows(trace) == [[13, 64], [4, 65], [27, 61], [23, 34]]


def test_stateful_chunking_matches_one_prefill_for_a_continuous_sequence() -> None:
    contract = _small_contract()
    one_shot = trace_token_segments(contract, [[3, 4, 5, 6]])
    chunked = trace_token_segments(contract, [[3, 4], [5], [6]])
    assert _rows(chunked) == _rows(one_shot)


def test_eos_breaks_ngram_history() -> None:
    contract = _small_contract()
    after_eos = trace_token_segments(contract, [[3, 4, 999, 6]])[-1:]
    fresh = trace_token_segments(contract, [[6]])
    assert _rows(after_eos) == _rows(fresh)
    assert _shift_right_ignore_eos([3, 999, 6], 1, 999) == [999, 3, 999]
