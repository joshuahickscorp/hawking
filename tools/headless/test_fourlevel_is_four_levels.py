"""A codec billed at 2 bits per weight must emit exactly four codes.

This is a correction to my own work. `_fourlevel_fitted` used
`rint(G/delta * 2.0) / 2.0` under a comment claiming "half-integers at +/-0.5,
+/-1.5". rint() also returns whole integers, so it emitted SEVEN levels
{-1.5,-1,-0.5,0,0.5,1,1.5}, which is log2(7) = 2.807 bits, not 2, with 49.7% of
units landing off the 4-level grid.

Everything billed from it at "2.25 bpw" was really about 3.06 bpw -- including the
whole-model composition arm this campaign recorded as SURVIVING at 2.25 and then
reasoned from. The bit budget was the claim; the codec never met it.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "headless"))
from fractional_bit_canon import _fourlevel_fitted

LEGAL = {-1.5, -0.5, 0.5, 1.5}


def _units(out, g=64):
    O = out.reshape(-1, g)
    amax = np.abs(O).max(-1, keepdims=True)
    scale = np.where(amax > 0, amax / 1.5, 1.0)
    return np.round(O / scale, 4)


def test_exactly_four_levels():
    W = np.random.RandomState(0).randn(4096, 128).astype(np.float32)
    u = np.unique(_units(_fourlevel_fitted(W, 64)))
    assert len(u) == 4, f"a 2-bit code has 4 levels, got {len(u)}: {u}"
    assert set(np.round(u, 4).tolist()) <= LEGAL, u


def test_no_unit_lands_off_the_grid():
    W = np.random.RandomState(1).randn(2048, 256).astype(np.float32)
    units = _units(_fourlevel_fitted(W, 64))
    off = np.isin(units, list(LEGAL), invert=True).mean()
    assert off == 0.0, f"{off:.1%} of units are off the 4-level grid"


def test_holds_for_several_group_sizes():
    W = np.random.RandomState(2).randn(1024, 512).astype(np.float32)
    for g in (32, 64, 128):
        u = np.unique(_units(_fourlevel_fitted(W, g), g))
        assert len(u) <= 4, f"group {g} produced {len(u)} levels"


def test_it_still_reconstructs_something_useful():
    """Snapping to four codes must not turn the codec into the deletion control."""
    W = np.random.RandomState(3).randn(4096, 128).astype(np.float32)
    out = _fourlevel_fitted(W, 64)
    assert np.count_nonzero(out) == out.size, "a 4-level grid has no zero code"
    rel = np.linalg.norm(out - W) / np.linalg.norm(W)
    assert 0.05 < rel < 0.6, f"rel_l2 {rel:.4f} is not a plausible 2-bit reconstruction"


if __name__ == "__main__":
    n = 0
    for _name, f in sorted(globals().items()):
        if _name.startswith("test_"):
            f(); n += 1; print(f"ok  {_name}")
    print(f"{n} passed")
