"""G025: representation budget is allocated BY FUNCTION, from measured sensitivity.

The two mixed-geometry arms are a controlled experiment: same class, same
calibration, one projection group changed. That is what licenses per-organ
allocation instead of uniform treatment.
"""
import json
from pathlib import Path

T = json.loads((Path(__file__).resolve().parent.parent
                / "receipts" / "future" / "O003_ALLOCATION_TABLE.json").read_text())
ORGANS = {o["organ"]: o for o in T["ORGANS"]}


def test_every_organ_carries_share_and_execution_frequency():
    for o in T["ORGANS"]:
        assert isinstance(o["weight_share"], float), o["organ"]
        assert o.get("execution_frequency"), o["organ"]
        assert o.get("allocation_reading"), o["organ"]


def test_down_proj_is_the_more_likelihood_sensitive_organ_per_weight():
    d = ORGANS["down_proj"]["likelihood_density"]
    g = ORGANS["gate_proj + up_proj"]["likelihood_density"]
    assert d > g, f"down {d} not above gate/up {g}"
    assert d / g > 2.0, f"ratio {d/g:.2f} -- the allocation claim needs a real gap"


def test_the_two_organs_move_diversity_in_opposite_directions():
    d = ORGANS["down_proj"]["diversity_density"]
    g = ORGANS["gate_proj + up_proj"]["diversity_density"]
    assert d < 0 < g, f"expected opposite signs, got down {d} and gate/up {g}"


def test_the_router_is_an_exact_island_and_costs_almost_nothing():
    r = ORGANS["router (mlp.gate)"]
    assert r["weight_share"] < 0.001, r["weight_share"]
    assert "bf16" in r["allocation_reading"]


def test_reconstruction_error_is_demoted_and_distinct_ratio_cannot_replace_the_gate():
    d = T["DEMOTIONS_ENFORCED"]
    assert "may not rank final candidates" in d["reconstruction_error"]
    assert "may not replace the conjunction gate" in d["distinct_ratio"]


def test_the_method_is_controlled():
    m = T["method"]
    assert "ONE projection group" in m and "holds the other" in m, m
    assert T["baseline"]["spec"], "no baseline arm recorded"
