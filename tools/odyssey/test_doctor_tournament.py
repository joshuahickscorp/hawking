"""G035 pins."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/DOCTOR_TOURNAMENT.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G035 receipt not built")


def rec():
    return json.load(open(R))


def test_the_canonical_s011_8_order_is_used_in_full():
    d = rec()
    assert d["canonical_order"] == [
        "ELIMINATE", "COORDINATES", "SHARE", "FACTOR", "GENERATE", "CODEBOOK", "ROUTE",
        "SENSITIVITY", "HEAL", "QUANTIZE", "NATIVE", "STATE", "CONDITIONAL", "DECODING"]
    assert set(d["stages"]) == set(d["canonical_order"])


def test_every_technique_in_the_library_is_routed():
    d = rec()
    assert d["techniques_routed_to_a_stage"] == d["library_size"] == 39
    assert d["unrouted"] == []


def test_all_three_organs_were_probed():
    """Probing 3/31/59 for every organ sampled only full-attention layers and dropped
    deltanet silently."""
    d = rec()["diagnosis"]
    assert {"mlp", "attention_gqa", "deltanet"} <= set(d)
    for o, v in d.items():
        assert v["measurements"], o


def test_no_probe_tensor_was_silently_missing():
    assert rec()["missing_probe_tensors"] == []


def test_preconditions_are_measured_from_real_weights():
    for o, v in rec()["diagnosis"].items():
        for k in ("mean_excess_kurtosis", "mean_r90_over_full_rank",
                  "max_cross_layer_cosine"):
            assert v[k] is not None, (o, k)


def test_a_stage_with_no_techniques_is_not_called_refuted():
    d = rec()["stages"]
    for st, v in d.items():
        if v["n_techniques"] == 0:
            assert v["verdict"] == "NO_TECHNIQUE_IN_LIBRARY", st


def test_share_is_closed_by_a_measured_near_orthogonality():
    d = rec()
    assert d["stages"]["SHARE"]["verdict"] == "PROBE_REFUTES_CLOSE_CHEAPLY"
    for o, v in d["diagnosis"].items():
        assert abs(v["max_cross_layer_cosine"]) < 0.05, o


def test_coordinates_is_resolved_per_organ_not_by_a_single_max():
    """A max() over organs hides which organ carries the signal."""
    c = rec()["coordinates_per_organ"]
    assert {"mlp", "attention_gqa", "deltanet"} <= set(c)
    assert c["mlp"]["holds"] is False
    assert c["deltanet"]["holds"] is True


def test_coordinates_respects_the_s011_16_prior_negative():
    """Rotation already failed whole-model; only a changed-condition probe is allowed."""
    r = rec()["coordinates_s011_16_routing"]
    assert "NEGATIVE" in r["prior"]
    assert "DELTANET ALONE" in r["the_only_sanctioned_probe"]
    assert "not that redistributing it will pay" in r["not_a_prediction"]


def test_thresholds_are_published_with_their_sensitivity():
    d = rec()
    assert d["thresholds"]
    assert "judgment calls" in d["threshold_honesty"]
    assert "would flip" in d["threshold_honesty"]


def test_four_stages_are_adjudicated_by_campaign_receipts_that_exist():
    d = rec()["stages"]
    adj = {s: v for s, v in d.items()
           if v["verdict"] == "ADJUDICATED_BY_CAMPAIGN_MEASUREMENT"}
    assert len(adj) >= 4
    root = Path(__file__).resolve().parents[2]
    for s, v in adj.items():
        assert (root / v["campaign_receipt"]).is_file(), (s, v["campaign_receipt"])


def test_the_remaining_frontier_is_named_and_not_endorsed():
    d = rec()
    assert d["remaining_frontier"]
    assert d["techniques_on_the_frontier"] > 0
    assert "untested, not endorsed" in d["frontier_note"]


def test_nothing_was_ranked_by_bit_width():
    assert "no stage is ranked by bit width" in rec()["no_blind_sweep"]


def test_the_probe_states_its_own_limit():
    m = rec()["probe_method"]
    assert "not a proof" in m["limit"]
