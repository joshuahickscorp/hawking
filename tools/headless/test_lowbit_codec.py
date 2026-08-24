"""The low-bit arms of the composition boundary hunt must actually code bits.

The first hunt reported `requantize-q1 grouped-64` FAILS with cos/gain/sa/rel_l2
identical to `zeroed down_proj` to four decimals. It matched the zero control
because it WAS the zero control: bound = 2^(bits-1)-1 is 0 at bits=1, so every
weight clipped to 0. A negative that reproduces the deletion control exactly is
a broken codec, not a refuted idea.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noetic_composition import requantize_absmax


def test_binary_is_not_deletion():
    w = np.random.RandomState(0).randn(4096).astype(np.float32)
    out = requantize_absmax(w, 1)
    assert np.count_nonzero(out) == out.size, "1-bit must keep every weight, not zero them"
    assert np.unique(np.abs(out)).size <= 4096 // 64, "1-bit must be one magnitude per group"


def test_binary_hits_the_sign_code_optimum():
    # For unit Gaussian, the best possible sign code alpha*sign(w) leaves
    # rel_l2 = sqrt(1 - 2/pi) ~= 0.6028. Per-group alpha may beat it slightly.
    w = np.random.RandomState(0).randn(1 << 16).astype(np.float32)
    rel = np.linalg.norm(requantize_absmax(w, 1) - w) / np.linalg.norm(w)
    assert 0.55 <= rel <= 0.62, f"1-bit rel_l2 {rel:.4f} is off the sign-code optimum 0.6028"


def test_error_falls_as_bits_rise():
    w = np.random.RandomState(1).randn(1 << 14).astype(np.float32)
    rel = [np.linalg.norm(requantize_absmax(w, b) - w) / np.linalg.norm(w) for b in (1, 2, 3, 4)]
    assert rel == sorted(rel, reverse=True), f"error must fall monotonically with bits: {rel}"


def test_zero_bits_is_still_deletion():
    w = np.random.RandomState(2).randn(256).astype(np.float32)
    assert not np.any(requantize_absmax(w, 0)), "bits=0 is the deletion control and must stay so"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("4/4 passed")
