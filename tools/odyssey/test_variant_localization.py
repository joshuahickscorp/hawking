"""G034 pins. The obligation is written on a premise the evidence refutes, and the
attribution it would otherwise inherit is confounded."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/VARIANT_LOCALIZATION.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G034 receipt not built")


def rec():
    return json.load(open(R))


def test_the_obligations_premise_is_recorded_as_refuted_not_quietly_worked_around():
    d = rec()["premise_check"]
    assert d["premise_holds"] is False
    assert d["measured_variantA"] == d["measured_clean"] == 0
    assert "REFUTED" in d["verdict"]


def test_ladder_comparability_was_established_before_any_attribution():
    """The clean body has scored 3, 14 and 0 on the same suite. Without a shared harness
    generation every attribution built on the ladder is worthless."""
    d = rec()["harness_comparability"]
    assert d["ladder_is_comparable"] is True
    assert len(set(d["max_completion_tokens_per_body"].values())) == 1


def test_the_non_comparable_run_is_excluded_and_says_why():
    d = rec()["harness_comparability"]["excluded"]
    assert d["passed"] == 14
    assert d["max_completion_tokens"] == 512
    assert "<think>" in d["why"]


def test_degradation_is_monotone_in_empty_replies():
    """Scores that track a physical signal, not scorer noise."""
    d = rec()["harness_comparability"]["degradation_is_monotone"]
    assert d["sealed-3.14"] < d["variantA-2.98"] < d["clean-2.60"]


def test_variant_b_separates_the_two_confounded_coordinates():
    d = rec()
    assert d["confound"]["coordinates_moved"] == 2
    assert d["variant_b"]["capability"]["passed"] == 24
    assert d["localization"]["verdict"] == "MLP_PER_GROUP_BIAS_IS_LOAD_BEARING"


def test_the_container_is_exonerated_by_evidence_not_by_assumption():
    """variantB runs the NEW container on every non-MLP organ and works."""
    r = rec()["localization"]["reading"]
    assert "container" in r and "innocent" in r
    assert "HGRAVU01" in r


def test_the_loss_is_localized_to_named_axes_not_to_an_organ_verdict():
    """S011 §2 forbids concluding 'attention must stay q4'."""
    d = rec()["axis_localization"]
    assert set(d["axes_lost"]) == {"coding", "self_correction"}
    assert len(d["axes_identical_to_sealed"]) == 5
    for k in d["axes_identical_to_sealed"]:
        assert abs(d["sealed_per_axis"][k] - d["variant_b_per_axis"][k]) < 1e-9


def test_the_previously_unpriced_non_mlp_change_now_has_a_price():
    """The floor effect made this UNPRICED; variantB prices it against a living body."""
    ps = rec()["marginal_information_allocator"]["purchases"]
    nm = next(p for p in ps if "non-MLP" in p["buy"])
    assert nm["capability_points_bought"] == 6
    assert nm["added_bpw"] > 0
    assert set(nm["buys_only"]) == {"coding", "self_correction"}


def test_allocator_ranks_purchases_by_points_per_bit():
    ps = rec()["marginal_information_allocator"]["purchases"]
    mlp = next(p for p in ps if "MLP per-group bias" in p["buy"])
    nm = next(p for p in ps if "non-MLP" in p["buy"])
    assert mlp["points_per_bpw"] > nm["points_per_bpw"]
    assert mlp["points_per_bpw"] == pytest.approx(96.0)


def test_pareto_candidate_beats_sealed_on_density_and_is_the_only_sub_3_body_that_works():
    d = rec()["pareto_candidate"]
    assert d["complete_ebpw_physical"] < 3.0
    assert int(d["capability"].split("/")[0]) > 0
    assert Path(d["artifact_root"]).is_dir()


# --- the correction G039 forced -------------------------------------------------

def test_the_axis_localization_claim_is_marked_refuted():
    """variantB passed both code WorkUnits in the HCLI bench, which is impossible if
    'q3 non-MLP cannot emit a python code block'."""
    c = rec().get("CORRECTION_axis_localization_refuted")
    assert c, "the refuted claim is not recorded"
    assert c["status"].startswith("REFUTED")


def test_the_refutation_carries_its_own_probe_data():
    c = rec()["CORRECTION_axis_localization_refuted"]["probe"]
    assert len(c["results"]) == 2
    for body, cells in c["results"].items():
        assert len(cells) >= 4, f"{body} needs several prompt cells to show sensitivity"


def test_every_failure_is_the_same_budget_runaway():
    """If failures had varied causes the bimodal claim would be wrong."""
    cells = rec()["CORRECTION_axis_localization_refuted"]["probe"]["results"]
    for body, cs in cells.items():
        for name, c in cs.items():
            if not c["verified"]:
                assert c["unterminated_think"] is True, (body, name)
                assert c["tokens"] == 1536, (body, name)
            else:
                assert c["tokens"] < 600, (body, name)


def test_the_inversion_is_recorded_not_smoothed_over():
    """variantB is more prompt-robust than sealed on this item, the opposite of the
    axis table."""
    c = rec()["CORRECTION_axis_localization_refuted"]["probe"]
    assert c["variantB_cells_passed"] > c["sealed_cells_passed"]


def test_what_survives_and_what_does_not_are_both_enumerated():
    c = rec()["CORRECTION_axis_localization_refuted"]
    assert len(c["what_survives"]) >= 2
    assert len(c["what_does_not_survive"]) >= 2
    assert any("bias" in s for s in c["what_survives"])


def test_the_allocator_carries_a_regime_caveat():
    """Its points/bpw come from the no-system-prompt regime and cannot be quoted as an
    unconditional law."""
    a = rec()["marginal_information_allocator"]
    assert "REGIME_CAVEAT" in a
    assert "system prompt" in a["REGIME_CAVEAT"].lower()
