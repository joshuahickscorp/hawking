"""Mechanism identity: paraphrases collide, distinct actions do not."""

from __future__ import annotations

from ascent.mechanism_identity import (
    fingerprint,
    is_bottleneck_name,
    same_mechanism,
    tokenize,
)


def test_fuse_paraphrases_share_canonical_id() -> None:
    a = "fusing tiny kernels into the following GEMV"
    b = "fuse small Metal kernels into the next GEMV"
    c = "merge microkernels into the trailing GEMV"
    fa, fb, fc = fingerprint(a), fingerprint(b), fingerprint(c)
    assert fa.canonical_id == fb.canonical_id == fc.canonical_id == "fuse_tiny_kernels_into_gemv"
    assert same_mechanism(a, b).same
    assert same_mechanism(a, c).same


def test_session_share_paraphrase_matches() -> None:
    a = "N sessions sharing one resident weight body to amortize DRAM"
    b = "concurrent sessions share weights to cut DRAM"
    assert same_mechanism(a, b).same
    assert fingerprint(a).canonical_id == "shared_resident_weight_amortize"


def test_cross_token_cache_paraphrase_matches() -> None:
    a = "cross-token cache reuse"
    b = "retain weights between tokens"
    assert same_mechanism(a, b).same


def test_distinct_mechanisms_do_not_match() -> None:
    fuse = "fusing tiny kernels into the following GEMV"
    assign = "assign codecs per layer and per head on the coherent vehicle"
    skip = "do not read every weight every token"
    assert not same_mechanism(fuse, assign).same
    assert not same_mechanism(assign, skip).same
    assert not same_mechanism(fuse, skip).same


def test_bottleneck_name_is_not_a_mechanism() -> None:
    assert is_bottleneck_name("weight_addressing")
    assert is_bottleneck_name("weight_addressing 21293103 ns")
    assert is_bottleneck_name("deltanet")
    assert not is_bottleneck_name("fusing tiny kernels into the following GEMV")
    assert not is_bottleneck_name("assign codecs per layer and per head")


def test_empty_texts_are_not_the_same_mechanism() -> None:
    match = same_mechanism("", "")
    assert not match.same
    assert match.reason == "empty_mechanism"
    assert tokenize("") == []


def test_gaussian_alias() -> None:
    a = "evaluate or fit compression on gaussian / synthetic proxy activations"
    b = "gaussian activations"
    assert same_mechanism(a, b).same
    assert fingerprint(a).canonical_id == "gaussian_synthetic_activations"
