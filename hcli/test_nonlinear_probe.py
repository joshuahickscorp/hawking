"""G003: a 4th-moment probe must be able to see what a spectrum cannot, and
must not report sharing on a matched null.

The linear participation ratio is a second-order statistic. Every measured
expert stack sits within 0.007% of a norm-matched Gaussian under it. This
module's producer asks whether the energy profiles (elementwise squares)
still share a low-dimensional description. A probe without the four controls
below is not a discriminator -- it is a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
import nonlinear_probe as nl  # noqa: E402
from dense_anatomy import _write_safetensors  # noqa: E402
from nonlinear_probe import (  # noqa: E402
    CONTROL_D,
    CONTROL_N,
    NonlinearProbeUnavailable,
    is_live,
    load_expert_stack,
    make_gaussian,
    make_linear,
    make_sign_magnitude,
    make_sparse_mixture,
    make_tanh_shared,
    probe,
    verdict,
)


def test_gaussian_is_dead_on_both_statistics():
    """Negative control. A false LIVE is the 0.95-at-n=20 defect, at 4th order."""
    rng = np.random.default_rng(0)
    r = probe(make_gaussian(CONTROL_N, CONTROL_D, rng))
    assert r["flag"] == "DEAD", r
    assert not r["linear"]["live"] and not r["energy"]["live"], r
    assert abs(r["energy"]["deficit_pct"]) < 2.0, r["energy"]
    assert abs(r["linear"]["deficit_pct"]) < 2.0, r["linear"]
    assert "non-linear sharing is DEAD" in r["verdict"], r["verdict"]


def test_tanh_shared_map_is_detected():
    """Positive: rows = tanh(z @ W) for shared W. Must be DETECTED."""
    rng = np.random.default_rng(1)
    r = probe(make_tanh_shared(CONTROL_N, CONTROL_D, 3, rng))
    assert r["energy"]["live"], r["energy"]
    assert r["energy"]["deficit_pct"] > 20.0, r["energy"]


def test_linear_rank3_is_detected_by_spectrum_AND_energy():
    """Linear structure the spectrum sees, this probe must not miss.

    A miss means the probe is weaker than the spectrum, not complementary.
    """
    rng = np.random.default_rng(2)
    r = probe(make_linear(CONTROL_N, CONTROL_D, 3, rng))
    assert r["linear"]["live"], r["linear"]
    assert r["energy"]["live"], r["energy"]
    assert r["linear"]["deficit_pct"] > 50.0, r["linear"]
    assert r["energy"]["deficit_pct"] > 20.0, r["energy"]
    assert r["flag"] == "BOTH_LIVE", r


def test_sign_magnitude_is_missed_by_spectrum_and_caught_by_energy():
    """THE control: shared |m|, independent signs.

    Linear Gram is a random walk of size ||m||^2/sqrt(d) -- spectrally flat.
    Energy Gram is rank-1. If energy does not go LIVE here, the probe adds
    nothing over the spectrum, which is itself the answer.
    """
    rng = np.random.default_rng(3)
    r = probe(make_sign_magnitude(CONTROL_N, CONTROL_D, rng, shared=True))
    assert not r["linear"]["live"], ("spectrum saw sign-magnitude", r["linear"])
    assert r["linear"]["deficit_pct"] < 2.0, r["linear"]
    assert r["energy"]["live"], r["energy"]
    assert r["energy"]["deficit_pct"] > 50.0, r["energy"]
    assert r["flag"] == "NONLINEAR_LIVE", r
    assert "non-linear sharing is LIVE" in r["verdict"], r["verdict"]


def test_independent_magnitudes_are_not_sharing():
    """Each row has its own |m|. Non-Gaussian marginals are not structure."""
    rng = np.random.default_rng(4)
    r = probe(make_sign_magnitude(CONTROL_N, CONTROL_D, rng, shared=False))
    assert not r["energy"]["live"], r["energy"]
    assert abs(r["energy"]["deficit_pct"]) < 2.0, r["energy"]


def test_heterogeneous_row_norms_are_not_energy_sharing():
    """Permutation null must not misread scale heterogeneity as alignment."""
    rng = np.random.default_rng(5)
    scales = np.exp(rng.standard_normal((CONTROL_N, 1), dtype=np.float32) * 2.0)
    r = probe(make_gaussian(CONTROL_N, CONTROL_D, rng, scales=scales))
    assert not r["energy"]["live"], r["energy"]
    assert abs(r["energy"]["deficit_pct"]) < 2.0, r["energy"]
    assert not r["linear"]["live"], r["linear"]


def test_sparse_mixture_is_not_energy_sharing():
    """Half-sparse / half-dense independent rows are not aligned energy.

    A Gaussian energy-null of matching energy-row-norms scores ~+6.8% LIVE
    on this construction at n=20, d=2048 -- it does not preserve the sparse
    magnitude histogram. The permutation null sits near zero. Replacing
    `_permutation_null` with that Gaussian is the mutation this test exists
    to catch; a source-string check is not a substitute.
    """
    rng = np.random.default_rng(9)
    r = probe(make_sparse_mixture(CONTROL_N, CONTROL_D, rng))
    assert not r["energy"]["live"], r["energy"]
    assert abs(r["energy"]["deficit_pct"]) < 2.0, r["energy"]
    assert not r["linear"]["live"], r["linear"]


def test_null_reports_its_own_sampling_error():
    rng = np.random.default_rng(6)
    r = probe(make_gaussian(CONTROL_N, CONTROL_D, rng))
    for side in ("linear", "energy"):
        c = r[side]
        assert c["null_seeds"] >= 2, c
        assert c["deficit_stderr_pct"] == c["deficit_stderr_pct"], f"{side} stderr NaN"
        assert c["deficit_stderr_pct"] > 0.0, (side, c)


def test_live_requires_deficit_above_twice_stderr():
    """A 0.04 pp gap the null cannot resolve is not LIVE, even if > 0."""
    dead = {"deficit_pct": 1.9, "deficit_stderr_pct": 0.01, "n": 20}
    assert not is_live(dead)
    unresolved = {"deficit_pct": 2.1, "deficit_stderr_pct": 1.5, "n": 20}
    assert not is_live(unresolved), "2.1% with se=1.5 pp is not resolved"
    live = {"deficit_pct": 8.0, "deficit_stderr_pct": 0.1, "n": 20}
    assert is_live(live)


def test_verdict_names_the_complementary_case():
    energy = {"deficit_pct": 80.0, "deficit_stderr_pct": 0.05, "n": 20}
    linear = {"deficit_pct": 0.2, "deficit_stderr_pct": 0.05, "n": 20}
    v = verdict(energy, linear)
    assert "non-linear sharing is LIVE" in v
    assert "spectrum is DEAD" in v


def test_verdict_names_a_dead_stack():
    energy = {"deficit_pct": 0.01, "deficit_stderr_pct": 0.02, "n": 128}
    linear = {"deficit_pct": 0.002, "deficit_stderr_pct": 0.01, "n": 128}
    v = verdict(energy, linear)
    assert "non-linear sharing is DEAD" in v
    assert "LIVE" not in v


def test_consume_path_matches_copy_path():
    """The lake path squares in place. A memory saving that changes the
    number is not a saving.
    """
    rng = np.random.default_rng(12)
    X = make_gaussian(CONTROL_N, CONTROL_D, rng)
    a = probe(X, consume=False)
    b = probe(X.copy(), consume=True)
    assert a["energy"]["deficit_pct"] == b["energy"]["deficit_pct"], (a, b)
    assert a["linear"]["deficit_pct"] == b["linear"]["deficit_pct"], (a, b)
    assert a["flag"] == b["flag"]


def test_probe_refuses_fewer_than_three_rows():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="at least 3 rows"):
        probe(rng.standard_normal((2, 64), dtype=np.float32))


def test_loader_reads_one_projection_one_tensor_at_a_time(tmp_path):
    rng = np.random.default_rng(8)
    tensors = {
        f"model.layers.0.mlp.experts.{i}.down_proj.weight":
            rng.standard_normal((16, 8), dtype=np.float32)
        for i in range(8)
    }
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = rng.standard_normal(
        (16, 8), dtype=np.float32
    )
    _write_safetensors(str(tmp_path / "m.safetensors"), tensors)
    X, meta = load_expert_stack(str(tmp_path), layer=0, projection="down_proj")
    assert X.shape == (8, 128), (X.shape, meta)
    assert meta["projection"] == "down_proj"
    assert meta["storage"] == "per-expert"
    r = probe(X)
    assert r["n"] == 8 and r["d"] == 128


def test_loader_refuses_prequantized_experts(tmp_path):
    tensors = {
        f"model.layers.0.mlp.experts.{i}.down_proj.weight": np.zeros((8, 8), dtype=np.int8)
        for i in range(4)
    }
    _write_safetensors(str(tmp_path / "m.safetensors"), tensors)
    with pytest.raises(NonlinearProbeUnavailable, match="PRE-QUANTIZED"):
        load_expert_stack(str(tmp_path), layer=0)


def test_loader_uses_census_read_header():
    """Wiring is existence: the lake path must call census.read_header, not
    merely import it. A private header reader would silently skip the module
    the contract named.
    """
    src = Path(nl.__file__).read_text()
    body = src[src.index("def load_expert_stack"):src.index("def probe_snapshot")]
    assert "read_header(" in body, "load_expert_stack never calls read_header"
    assert "from lake_scheme_census import" in src


def test_energy_statistic_is_the_square_not_the_raw_stack():
    """A mutation that skips the square makes energy == linear on sign-magnitude
    and the 4th control collapses. Anchor the producer, not a comment.
    """
    src = Path(nl.__file__).read_text()
    body = src[src.index("def _energy_stack"):src.index("def _permutation_null")]
    assert "np.square" in body, "energy stack is not the elementwise square"


def test_energy_null_is_a_permutation_not_a_gaussian():
    src = Path(nl.__file__).read_text()
    body = src[src.index("def _permutation_null"):src.index("def _gaussian_null_from_norms")]
    assert "permuted" in body, "energy null is not a per-row permutation"
    # the Gaussian null is for the LINEAR half; energy must not call it
    probe_body = src[src.index("def probe("):src.index("def make_gaussian")]
    # energy draws go through _permutation_null; linear through _gaussian_null
    assert "_permutation_null(Y" in probe_body
    assert "_gaussian_null_from_norms(lin_norms" in probe_body
