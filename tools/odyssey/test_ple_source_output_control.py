from __future__ import annotations

import numpy as np

from tools.odyssey.ple_access_trace import build_contract, trace_token_segments
from tools.odyssey.ple_source_output_control import (
    LTR_F32_REDUCTION_POLICY,
    PLESupport,
    VECTOR_REDUCTION_POLICY,
    _ltr_f32_sum,
    bf16_bytes_to_f32,
    evaluate_ple_sequence,
    evaluate_ple_zero_state_context_batch,
)


def _small_contract():
    return build_contract(
        vocab_size=29,
        eos_token_id=28,
        ngram_size=3,
        heads_per_ngram=1,
        ngram_vocab_size_base=31,
        make_vocab_divisible_by=8,
        ple_embed_dim=4,
        ple_layer_index=0,
        source_layer_index=0,
        seed=1234,
    )


def _synthetic_reference_fixture():
    """A tiny F32 fixture captured from transformers 5.17.0 PLE forward.

    The parameter construction deliberately mirrors the reference probe's
    ``named_parameters`` sequence.  Its expected output below was generated
    by ``Qwen4ExpTextPLELayer.forward`` using the installed reference source,
    not by this evaluator.
    """
    contract = _small_contract()
    assert contract.padded_vocab_size == 72
    cursor = 0

    def values(shape):
        nonlocal cursor
        count = int(np.prod(shape))
        result = (
            np.arange(cursor, cursor + count, dtype=np.float32) / np.float32(97.0)
            - np.float32(0.4)
        ).reshape(shape)
        cursor += count
        return result

    table = values((contract.padded_vocab_size, contract.head_dim))
    support = PLESupport(
        key_proj=values((8, 4)),
        value_proj=values((4, 4)),
        norm_key=values((8,)),
        norm_query=values((8,)),
        norm_conv=values((8,)),
        conv=values((8, 4)),
        bindings=[],
    )
    trace = trace_token_segments(contract, [[3, 4]])
    embeddings = np.stack(
        [
            np.concatenate([table[access["combined_row"]] for access in token["accesses"]])
            for token in trace
        ]
    )
    hidden = np.array(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            [0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            [
                5.019091606140137,
                5.30023193359375,
                5.58831787109375,
                5.883541584014893,
                5.630568027496338,
                5.930059909820557,
                6.236894130706787,
                6.551270961761475,
            ],
            [
                7.05069637298584,
                7.39200496673584,
                7.740334987640381,
                8.095877647399902,
                7.945893287658691,
                8.313895225524902,
                8.689323425292969,
                9.072372436523438,
            ],
        ],
        dtype=np.float32,
    )
    return support, hidden, embeddings, expected


def test_bf16_payload_conversion_preserves_ieee_sign_and_exponent() -> None:
    raw = bytes.fromhex("803f80bf0000807f")
    actual = bf16_bytes_to_f32(raw)
    assert actual.dtype == np.dtype("<f4")
    assert actual[0] == np.float32(1.0)
    assert actual[1] == np.float32(-1.0)
    assert actual[2] == np.float32(0.0)
    assert np.isinf(actual[3]) and actual[3] > 0


def test_ple_formula_matches_reference_forward_golden() -> None:
    support, hidden, embeddings, expected = _synthetic_reference_fixture()
    injection, _, _ = evaluate_ple_sequence(
        hidden,
        embeddings,
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
    )
    np.testing.assert_allclose(injection, expected, rtol=2e-6, atol=2e-6)


def test_ple_convolution_state_survives_a_token_boundary() -> None:
    support, hidden, embeddings, _ = _synthetic_reference_fixture()
    full, full_state, _ = evaluate_ple_sequence(
        hidden,
        embeddings,
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
    )
    first, state_after_first, _ = evaluate_ple_sequence(
        hidden[:1],
        embeddings[:1],
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
    )
    second, state_after_second, _ = evaluate_ple_sequence(
        hidden[1:],
        embeddings[1:],
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
        initial_conv_state=state_after_first,
    )
    np.testing.assert_allclose(np.concatenate([first, second]), full, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_after_second, full_state, rtol=1e-6, atol=1e-6)


def test_zero_state_context_batch_matches_independent_sequence_evaluations() -> None:
    """The fast local-context primitive must preserve the source formula."""
    support, hidden, embeddings, _ = _synthetic_reference_fixture()
    batched, _ = evaluate_ple_zero_state_context_batch(
        hidden,
        embeddings[0],
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
    )
    expected = np.concatenate(
        [
            evaluate_ple_sequence(
                hidden[index : index + 1],
                embeddings[0:1],
                support,
                hidden_size=4,
                hc_count=2,
                ngram_size=3,
                eps=1e-6,
                initial_conv_state=None,
            )[0]
            for index in range(hidden.shape[0])
        ],
        axis=0,
    )
    np.testing.assert_allclose(batched, expected, rtol=1e-6, atol=1e-6)


def test_ltr_f32_sum_matches_an_explicit_scalar_f32_accumulator() -> None:
    values = np.array(
        [[1.0e20, 1.0, -1.0e20, 3.0], [0.125, -0.5, 0.25, 0.75]], dtype=np.float32
    )
    expected = []
    for row in values:
        total = np.float32(0.0)
        for value in row:
            total = np.float32(total + value)
        expected.append(total)
    np.testing.assert_array_equal(_ltr_f32_sum(values, axis=-1), np.array(expected, dtype=np.float32))


def test_ltr_f32_formula_preserves_split_sequence_state() -> None:
    support, hidden, embeddings, _ = _synthetic_reference_fixture()
    full, full_state, _ = evaluate_ple_sequence(
        hidden,
        embeddings,
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
        reduction_policy=LTR_F32_REDUCTION_POLICY,
    )
    first, first_state, _ = evaluate_ple_sequence(
        hidden[:1],
        embeddings[:1],
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
        reduction_policy=LTR_F32_REDUCTION_POLICY,
    )
    second, second_state, _ = evaluate_ple_sequence(
        hidden[1:],
        embeddings[1:],
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
        initial_conv_state=first_state,
        reduction_policy=LTR_F32_REDUCTION_POLICY,
    )
    np.testing.assert_array_equal(np.concatenate([first, second]), full)
    np.testing.assert_array_equal(second_state, full_state)

    vectorized, _, _ = evaluate_ple_sequence(
        hidden,
        embeddings,
        support,
        hidden_size=4,
        hc_count=2,
        ngram_size=3,
        eps=1e-6,
        reduction_policy=VECTOR_REDUCTION_POLICY,
    )
    assert np.all(np.isfinite(vectorized))
