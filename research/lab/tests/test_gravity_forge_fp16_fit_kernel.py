"""The fp16 fit kernel must be faster WITHOUT costing reconstruction error.

Odyssey has a whole ModelLake still to fit, so a second removed from the inner
k-means loop is multiplied by every future specimen. v3 halves the width of the
[step, k] score block, which is where the cost was measured to be: on Kimi's
expert bank on MPS, 11.5M subvectors against k=1024, fp32 1.33s and fp16 0.98s.

v3 is lossy, and its safety turns out to depend on the DATA rather than on the
kernel. Scores scale with |v||c|, so a block holding a 1e-5 row beside a 0.9 row
cannot represent both in 11 bits of mantissa. Measured, varying only the row-norm
dynamic range:

    max/min      4.9   agreement 99.84%   SSE +0.00007%
    max/min     5453   agreement 99.90%   SSE +0.00002%
    max/min   425549   agreement 99.89%   SSE +0.09538%
    max/min    2.6e9   agreement 91.73%   SSE +38.3%

Two things are pinned from that. First, v3 is free in the homogeneous regime.
Second -- and this is the load-bearing one -- AGREEMENT DOES NOT BOUND ERROR:
in the third row agreement went UP while SSE got 1400x worse, because the rows it
lost were the high-norm ones that dominate the sum. Gating a lossy assign on an
agreement rate is the same magnitude-blindness that let a 0.01x body score
1.000000 on a cosine gate. The gate here is SSE; the guard is the dynamic range
that predicts it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import lab.operators.gravity_forge as forge


SSE_CEILING = 1e-4          # 0.01%; measured 0.0004% on Kimi, 0.00007% synthetic


def _specimen(spread=0.0, n=60_000, d=16, k=256, seed=0):
    """``spread`` is the log-normal sigma of the row norms. spread=0 is the
    homogeneous regime v3 is licensed for; large spread is the cliff."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g)
    if spread:
        v = v * torch.exp(torch.randn(n, 1, generator=g) * spread)
    cb = v[torch.randperm(n, generator=g)[:k]].clone()
    return v, cb


def _assign_with(kernel, v, cb):
    prev = forge.FIT_KERNEL
    forge.FIT_KERNEL = kernel
    try:
        v2 = (v * v).sum(1, keepdim=True)
        return forge._argmin_chunked(v, v2, cb, forge._chunk_rows(v.shape[0], cb.shape[0]))
    finally:
        forge.FIT_KERNEL = prev


def _sse(v, cb, idx):
    return float(((v - cb[idx]) ** 2).sum())


def _rel_sse(v, cb, a_ref, a_test):
    ref = _sse(v, cb, a_ref)
    return (_sse(v, cb, a_test) - ref) / ref


def _norm_range(v):
    rn = v.norm(dim=1)
    return float(rn.max() / rn.min())


class TestTheKernelIsRegisteredAndOptIn:
    def test_it_is_a_declared_kernel(self):
        assert "v3_lean_argmin_fp16" in forge.FIT_KERNELS

    def test_it_is_not_the_default(self):
        # A lossy kernel that turns itself on is a silent codec change.
        assert forge.FIT_KERNEL != "v3_lean_argmin_fp16"


class TestFp16IsFreeInTheRegimeItIsLicensedFor:
    def test_homogeneous_rows_cost_no_meaningful_error(self):
        v, cb = _specimen(spread=0.0)
        assert _norm_range(v) < forge.FP16_ROW_NORM_RANGE_LIMIT
        rel = _rel_sse(v, cb, _assign_with("v2_lean_argmin", v, cb),
                       _assign_with("v3_lean_argmin_fp16", v, cb))
        assert abs(rel) <= SSE_CEILING, f"fp16 moved SSE by {rel*100:.5f}%"

    def test_moderate_spread_is_still_free(self):
        v, cb = _specimen(spread=1.0)
        assert _norm_range(v) > 1e3, "this case is meant to be non-trivial"
        rel = _rel_sse(v, cb, _assign_with("v2_lean_argmin", v, cb),
                       _assign_with("v3_lean_argmin_fp16", v, cb))
        assert abs(rel) <= SSE_CEILING, f"fp16 moved SSE by {rel*100:.5f}%"

    def test_it_assigns_every_row_in_range(self):
        v, cb = _specimen()
        a3 = _assign_with("v3_lean_argmin_fp16", v, cb)
        assert a3.shape[0] == v.shape[0]
        assert int(a3.min()) >= 0 and int(a3.max()) < cb.shape[0]


class TestTheGuardCatchesTheCliff:
    def test_a_wide_norm_range_falls_back_and_is_therefore_exact(self):
        # spread=2.5 measured +38% SSE UNGUARDED. Guarded it must be exact,
        # because the fallback IS v2.
        v, cb = _specimen(spread=2.5)
        # Literal, not forge.FP16_ROW_NORM_RANGE_LIMIT: reading the constant here
        # would make this precondition fail whenever the limit is mutated, and the
        # behavioural assert below would never be reached. A mutation check must
        # trip the behaviour, not the setup.
        assert _norm_range(v) > 1e4, "this specimen must sit past the shipped limit"
        a2 = _assign_with("v2_lean_argmin", v, cb)
        a3 = _assign_with("v3_lean_argmin_fp16", v, cb)
        assert torch.equal(a2, a3), (
            "the guard did not fire on a specimen whose norm range exceeds the limit")

    def test_the_unguarded_kernel_really_would_have_been_wrong(self):
        # Negative control for the guard. If fp16 were harmless here, the guard
        # would be ceremony and this file would be proving nothing.
        v, cb = _specimen(spread=2.5)
        a2 = _assign_with("v2_lean_argmin", v, cb)
        cb_h = cb.to(torch.float16)
        half = ((cb.float() * cb.float()).sum(1) * 0.5).to(torch.float16)
        a_raw = (half - v.to(torch.float16) @ cb_h.t()).argmin(1)
        rel = _rel_sse(v, cb, a2, a_raw)
        assert rel > SSE_CEILING, (
            f"unguarded fp16 cost only {rel*100:.5f}% on the cliff specimen -- "
            "the guard is protecting against nothing")

    def test_agreement_would_not_have_caught_it(self):
        # The reason the guard reads norms and the gate reads SSE. This is the
        # finding, so it is pinned rather than left in a comment.
        v, cb = _specimen(spread=1.5)
        a2 = _assign_with("v2_lean_argmin", v, cb)
        cb_h = cb.to(torch.float16)
        half = ((cb.float() * cb.float()).sum(1) * 0.5).to(torch.float16)
        a_raw = (half - v.to(torch.float16) @ cb_h.t()).argmin(1)
        agree = float((a2 == a_raw).float().mean())
        rel = _rel_sse(v, cb, a2, a_raw)
        assert agree > 0.99 and rel > SSE_CEILING, (
            f"agreement {agree*100:.3f}% / SSE {rel*100:.5f}% -- the case where a high "
            "agreement rate hides a real error regression no longer reproduces, so the "
            "justification for guarding on norms instead of agreement is stale")


class TestAGuttedKernelIsRejected:
    def test_four_bit_scores_destroy_the_minimiser(self):
        v, cb = _specimen()
        a2 = _assign_with("v2_lean_argmin", v, cb)
        half = (cb * cb).sum(1) * 0.5
        scores = half - v @ cb.t()
        lo, hi = scores.min(), scores.max()
        a_bad = ((scores - lo) / (hi - lo) * 15).round().argmin(1)
        assert _rel_sse(v, cb, a2, a_bad) > SSE_CEILING


class TestChunkRowsAccountsForScoreWidth:
    def test_a_half_width_score_doubles_the_rows_per_block(self):
        n, k = 10_000_000, 1024
        assert forge._chunk_rows(n, k, 2) == 2 * forge._chunk_rows(n, k, 4)

    def test_the_default_is_unchanged_at_four_bytes(self):
        # The signature grew a parameter; every existing caller must be unaffected.
        n, k = 10_000_000, 1024
        assert forge._chunk_rows(n, k) == forge._chunk_rows(n, k, 4)

    def test_it_never_returns_a_zero_block(self):
        assert forge._chunk_rows(1, 1 << 30, 4) >= 1


class TestTheExistingKernelsAreUntouched:
    def test_v1_and_v2_still_agree_on_a_clean_specimen(self):
        v, cb = _specimen()
        a1 = _assign_with("v1_full_distance", v, cb)
        a2 = _assign_with("v2_lean_argmin", v, cb)
        assert _rel_sse(v, cb, a1, a2) <= SSE_CEILING

    def test_v2_refactored_into_a_helper_returns_what_it_did(self):
        v, cb = _specimen()
        step = forge._chunk_rows(v.shape[0], cb.shape[0])
        assert torch.equal(_assign_with("v2_lean_argmin", v, cb),
                           forge._argmin_chunked_v2(v, cb, step))

    def test_the_chunked_path_matches_the_single_block_path(self):
        # The refactor moved a loop. A step that forces chunking must agree with
        # one that does not.
        v, cb = _specimen(n=20_000)
        assert torch.equal(forge._argmin_chunked_v2(v, cb, 1 << 30),
                           forge._argmin_chunked_v2(v, cb, 4096))
