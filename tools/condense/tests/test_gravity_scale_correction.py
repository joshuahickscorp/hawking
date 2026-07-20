#!/usr/bin/env python3.12
"""Synthetic, fast. No model source touched."""
from __future__ import annotations

import numpy as np
import pytest

import gravity_scale_correction as SC


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_recovers_known_scalar_gain(rng):
    w = rng.standard_normal((48, 64)).astype(np.float32)
    packed = (w / 13.4).astype(np.float32)
    assert SC.fit_scale(w, packed) == pytest.approx(13.4, rel=1e-4)
    assert SC.rel_error(w, SC.apply_scale(packed, 13.4)) < 1e-5


def test_scalar_gain_beats_uncorrected_under_noise(rng):
    w = rng.standard_normal((48, 64)).astype(np.float32)
    packed = (w / 9.0 + 0.02 * rng.standard_normal(w.shape)).astype(np.float32)
    g = SC.fit_scale(w, packed)
    assert SC.rel_error(w, SC.apply_scale(packed, g)) < 0.5 * SC.rel_error(w, packed)


def test_cosine_is_gain_invariant(rng):
    w = rng.standard_normal((16, 32)).astype(np.float32)
    packed = (w * 0.031 + 0.01 * rng.standard_normal(w.shape)).astype(np.float32)
    before = SC.cosine(w, packed)
    after = SC.cosine(w, SC.apply_scale(packed, SC.fit_scale(w, packed)))
    assert after == pytest.approx(before, abs=1e-5)


def test_rowwise_recovers_per_row_gains(rng):
    w = rng.standard_normal((24, 40)).astype(np.float32)
    true_g = rng.uniform(2.0, 20.0, size=24).astype(np.float32)
    packed = (w / true_g[:, None]).astype(np.float32)
    g = SC.fit_scale_rowwise(w, packed)
    assert g.shape == (24,)
    np.testing.assert_allclose(g, true_g, rtol=1e-3)
    assert SC.rel_error(w, SC.apply_scale(packed, g)) < 1e-4


def test_rowwise_never_worse_than_scalar(rng):
    w = rng.standard_normal((24, 40)).astype(np.float32)
    packed = (w / rng.uniform(2.0, 20.0, size=24)[:, None]
              + 0.02 * rng.standard_normal((24, 40))).astype(np.float32)
    s = SC.rel_error(w, SC.apply_scale(packed, SC.fit_scale(w, packed)))
    r = SC.rel_error(w, SC.apply_scale(packed, SC.fit_scale_rowwise(w, packed)))
    assert r <= s + 1e-9


def test_activation_weighted_fit_differs_and_wins_on_its_own_measure(rng):
    w = rng.standard_normal((12, 8)).astype(np.float32)
    packed = (w * rng.uniform(0.05, 0.2, size=8)[None, :]).astype(np.float32)  # anisotropic damage
    acts = (rng.standard_normal((256, 8)) * np.array([9, 1, 1, 1, 1, 1, 1, 1])).astype(np.float32)
    g_w = SC.fit_scale(w, packed)
    g_a = SC.fit_scale(w, packed, acts)
    assert abs(g_a - g_w) > 1e-3
    assert (SC.rel_error(w, SC.apply_scale(packed, g_a), acts)
            <= SC.rel_error(w, SC.apply_scale(packed, g_w), acts) + 1e-9)


def test_degenerate_inputs_are_identity_not_nan():
    w = np.ones((4, 4), dtype=np.float32)
    assert SC.fit_scale(w, np.zeros((4, 4), np.float32)) == 1.0
    np.testing.assert_array_equal(SC.fit_scale_rowwise(w, np.zeros((4, 4), np.float32)),
                                  np.ones(4, np.float32))


def test_bit_costs_are_exact():
    assert SC.scale_bits(12032) == 12032 * 3 * 16
    assert SC.scale_bits(12032, n_organs=1) == 12032 * 16
    assert SC.scale_bits_rowwise(12032, (1536, 1536, 4096)) == 12032 * 7168 * 16


def test_shape_guards():
    w = np.ones((4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        SC.fit_scale(w, np.ones((4, 4), np.float32))
    with pytest.raises(ValueError):
        SC.fit_scale(w, w, np.ones((3, 5), np.float32))
