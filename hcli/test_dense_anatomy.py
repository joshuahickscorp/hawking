"""Dense-body Odyssey-I anatomy: refusals are named, structure is a deficit, factoring is arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
import dense_anatomy as da  # noqa: E402
from dense_anatomy import (  # noqa: E402
    DenseAnatomyUnavailable,
    anatomy_from_safetensors,
    calibrated_stack,
    effective_rank_calibrated,
    factoring_report,
    hypotheses,
    _write_safetensors,
)

LAKE_QWEN = Path("/Volumes/corpdrive/hawking-modellake/specimens/"
                 "Qwen--Qwen3-0.6B@c1899de289a0")


def _dense_weights(n_layers: int, shape=(32, 24), kind="full", seed=0, organ="q_proj"):
    rng = np.random.default_rng(seed)
    tensors = {}
    if kind == "low":
        B = rng.standard_normal((3, shape[1]), dtype=np.float32)
        for i in range(n_layers):
            C = rng.standard_normal((shape[0], 3), dtype=np.float32)
            tensors[f"model.layers.{i}.self_attn.{organ}.weight"] = C @ B
    elif kind == "i8":
        for i in range(n_layers):
            tensors[f"model.layers.{i}.self_attn.{organ}.weight"] = np.zeros(shape, dtype=np.int8)
    elif kind == "expert":
        for i in range(n_layers):
            tensors[f"model.layers.{i}.mlp.experts.0.down_proj.weight"] = (
                rng.standard_normal(shape, dtype=np.float32)
            )
    else:
        for i in range(n_layers):
            tensors[f"model.layers.{i}.self_attn.{organ}.weight"] = (
                rng.standard_normal(shape, dtype=np.float32)
            )
    return tensors


def test_no_safetensors_named_refusal(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"\0")
    with pytest.raises(DenseAnatomyUnavailable, match=r"no \.safetensors"):
        anatomy_from_safetensors(str(tmp_path))


def test_expert_tensors_refuse_pointing_at_the_other_module(tmp_path):
    _write_safetensors(str(tmp_path / "m.safetensors"), _dense_weights(4, kind="expert"))
    with pytest.raises(DenseAnatomyUnavailable, match="representational_anatomy.py"):
        anatomy_from_safetensors(str(tmp_path))


def test_i8_payload_refused_as_prequantized(tmp_path):
    _write_safetensors(str(tmp_path / "m.safetensors"), _dense_weights(4, kind="i8"))
    with pytest.raises(DenseAnatomyUnavailable, match="pre-quantized"):
        anatomy_from_safetensors(str(tmp_path))


def test_nested_safetensors_are_found(tmp_path):
    nest = tmp_path / "low_noise_model"
    nest.mkdir()
    _write_safetensors(str(nest / "m.safetensors"), _dense_weights(4, kind="low", shape=(32, 24)))
    a = anatomy_from_safetensors(str(tmp_path))
    assert a["n_shards"] == 1
    assert a["hypotheses"], a


def test_synthetic_low_rank_organ_is_detected(tmp_path):
    _write_safetensors(str(tmp_path / "m.safetensors"),
                       _dense_weights(4, kind="low", shape=(80, 40), seed=1))
    a = anatomy_from_safetensors(str(tmp_path))
    mid = next(r for r in a["within_tensor"] if r["depth"] == "middle")
    assert mid["deficit_pct"] > 50.0, mid
    assert any("ALIVE" in h for h in a["hypotheses"]), a["hypotheses"]


def test_synthetic_full_rank_organ_deficit_near_zero(tmp_path):
    _write_safetensors(str(tmp_path / "m.safetensors"),
                       _dense_weights(4, kind="full", shape=(128, 64), seed=2))
    a = anatomy_from_safetensors(str(tmp_path))
    mid = next(r for r in a["within_tensor"] if r["depth"] == "middle")
    assert abs(mid["deficit_pct"]) < 5.0, mid


def test_break_even_arithmetic_says_factoring_saves():
    fac = factoring_report(2048, 768, 495)
    assert abs(fac["byte_ratio"] - 0.886) < 1e-3, fac
    assert fac["pays"] is True, fac
    assert fac["factored_elems"] == 495 * (2048 + 768)
    assert fac["dense_elems"] == 2048 * 768
    h = hypotheses({
        "within_tensor": [{
            "organ": "down_proj", "depth": "middle", "layer": 0,
            "shape": [2048, 768], "deficit_pct": 13.5, "ratio": 0.7171,
            "null_ratio": 0.8286, **fac,
        }],
        "cross_layer": [],
    })
    text = " ".join(h)
    assert "PAYS" in text, h
    assert "costs more than dense" not in text


def test_fewer_than_three_layers_refused(tmp_path):
    _write_safetensors(str(tmp_path / "m.safetensors"), _dense_weights(2, kind="full"))
    with pytest.raises(DenseAnatomyUnavailable, match="fewer than 3 layers"):
        anatomy_from_safetensors(str(tmp_path))


@pytest.mark.slow
def test_qwen3_06b_organ_ordering_reproduces():
    if not LAKE_QWEN.is_dir():
        pytest.skip(
            "Qwen3-0.6B lake specimen absent under "
            "/Volumes/corpdrive/hawking-modellake/specimens/; "
            "organ-ordering assertion DID NOT RUN"
        )
    a = anatomy_from_safetensors(str(LAKE_QWEN))
    mids = {r["organ"]: r["deficit_pct"]
            for r in a["within_tensor"] if r["depth"] == "middle"}
    order = ["k_proj", "q_proj", "o_proj", "gate_proj", "up_proj"]
    missing = [o for o in order if o not in mids]
    assert not missing, (missing, sorted(mids))
    vals = [mids[o] for o in order]
    assert vals == sorted(vals, reverse=True), mids
    published = {"k_proj": 41.37, "q_proj": 38.84, "o_proj": 23.62,
                 "gate_proj": 20.17, "up_proj": 11.60}
    for organ, expected in published.items():
        assert abs(mids[organ] - expected) < 2.0, (organ, mids[organ], expected)
    cl = {r["organ"]: r["deficit_pct"] for r in a["cross_layer"]}
    assert cl["gate_proj"] < 1.0, cl
    assert cl["q_proj"] < 1.0, cl
    text = " ".join(a["hypotheses"])
    assert "ALIVE" in text and "organ-dependent" in text, a["hypotheses"]
    assert "cross-layer sharing is DEAD" in text, a["hypotheses"]


def test_calibrated_stack_matches_spectral_null_invariants():
    rng = np.random.default_rng(7)
    R = rng.standard_normal((12, 1 << 14), dtype=np.float32)
    c = calibrated_stack(R)
    assert abs(c["ratio_n"] - 11 / 12) < 2e-2, c
    assert abs(c["deficit_pct"]) < 2.0, c
    B = rng.standard_normal((3, 1 << 14), dtype=np.float32)
    C = rng.standard_normal((16, 3), dtype=np.float32)
    live = calibrated_stack(C @ B)
    assert live["deficit_pct"] > 50.0, live


def test_effective_rank_null_is_aspect_ratio_dependent():
    rng = np.random.default_rng(3)
    square = effective_rank_calibrated(rng.standard_normal((96, 96), dtype=np.float32))
    wide = effective_rank_calibrated(rng.standard_normal((288, 96), dtype=np.float32))
    assert wide["null_ratio"] > square["null_ratio"] + 0.05, (square, wide)


# --- the null carries its own noise -----------------------------------------

def test_the_null_reports_its_own_sampling_error():
    """A single null draw is a random variable, not a constant.

    Forty seeds on a 1024x1024 organ give mean 41.297 with stdev 0.045 pp. Two
    adjacent pairs in the first Wan2.2-T2V-A14B ordering were 0.04 and 0.05 pp
    apart and were reported as ranked.
    """
    rng = np.random.default_rng(11)
    W = rng.standard_normal((512, 512), dtype=np.float32)
    r = da.effective_rank_calibrated(W)
    assert r["null_seeds"] >= 2, r
    assert r["deficit_stderr_pct"] == r["deficit_stderr_pct"], "stderr is NaN"
    assert r["deficit_stderr_pct"] > 0.0, "a multi-draw null cannot have zero spread"
    assert abs(r["deficit_pct"]) < 1.0, ("a random matrix must sit at its own null", r)


def test_organs_closer_than_the_null_noise_are_not_ordered():
    o = [("a", 25.31, 0.045), ("b", 25.27, 0.045),
         ("c", 20.14, 0.045), ("d", 9.70, 0.045), ("e", 9.65, 0.045)]
    ranks = {x["organ"]: x["rank"] for x in da.resolved_ordering(o)}
    assert ranks["a"] == ranks["b"], "0.04 pp apart was reported as ordered"
    assert ranks["d"] == ranks["e"], "0.05 pp apart was reported as ordered"
    assert ranks["c"] not in (ranks["a"], ranks["d"]), "5 pp apart must separate"


def test_a_genuinely_separated_ordering_is_still_ordered():
    """The tie rule must not flatten everything -- that would be the opposite bug."""
    o = [("k", 41.24, 0.04), ("q", 38.81, 0.04), ("o", 23.68, 0.04), ("up", 11.59, 0.04)]
    ranks = [x["rank"] for x in da.resolved_ordering(o)]
    assert ranks == [0, 1, 2, 3], ranks


def test_expert_as_a_substring_is_not_an_expert_organ(tmp_path):
    """pi0_base fell through BOTH anatomy modules and was measured by neither.

    Its keys read paligemma_with_expert.gemma_expert..., a SUBMODULE NAME, and
    767 of them matched a substring test. dense_anatomy refused it as "has
    experts, the other module owns it" while representational_anatomy refused it
    as "not a per-expert layout". Same shape as counting `hawkingd` because it
    contains `awk`.
    """
    rng = np.random.default_rng(3)
    tensors = {
        f"paligemma_with_expert.paligemma.model.layers.{i}.self_attn.q_proj.weight":
        rng.standard_normal((8, 8), dtype=np.float32) for i in range(3)
    }
    _write_safetensors(str(tmp_path / "m.safetensors"), tensors)
    try:
        da.anatomy_from_safetensors(str(tmp_path))
    except da.DenseAnatomyUnavailable as exc:
        assert "HAS expert tensors" not in str(exc), (
            "refused on the substring again: " + str(exc))


def test_a_real_indexed_expert_set_is_still_refused(tmp_path):
    """The fix must not let a genuine MoE body into the dense path."""
    rng = np.random.default_rng(4)
    tensors = {f"model.layers.0.mlp.experts.{i}.down_proj.weight":
               rng.standard_normal((8, 8), dtype=np.float32) for i in range(12)}
    _write_safetensors(str(tmp_path / "m.safetensors"), tensors)
    with pytest.raises(da.DenseAnatomyUnavailable, match="HAS expert tensors"):
        da.anatomy_from_safetensors(str(tmp_path))
