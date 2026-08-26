"""Doctor must rank, must skip with a reason, and must never prune."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import doctor_transfer as dt

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/DOCTOR_TRANSFER.json"
LIB = REPO / "receipts/headless/DOCTOR_TECHNIQUE_LIBRARY.json"


def test_all_39_techniques_survive():
    """A Qwen failure must never globally prune a technique (directive §81)."""
    lib = json.load(open(LIB))
    assert len(lib["techniques"]) == 39
    assert all(t["decision"] == "KEEP" for t in lib["techniques"])
    d = json.load(open(R))
    assert d["all_techniques_still_KEEP"] is True
    assert d["n_techniques_in_library"] == 39


def test_every_organ_gets_a_ranked_prescription():
    d = json.load(open(R))
    assert d["n_organs"] >= 5
    for o in d["per_organ"]:
        assert o["prescription"], o["organ"]
        scores = [p["rank_score"] for p in o["prescription"]]
        assert scores == sorted(scores, reverse=True), o["organ"]


def test_every_skip_carries_a_reason():
    d = json.load(open(R))
    for o in d["per_organ"]:
        for s in o["skipped"]:
            assert s["skip_reason"]
            assert s["grade"] == "UNLIKELY"


def test_skipped_techniques_are_not_deleted():
    """UNLIKELY is a ranking, not a deletion."""
    d = json.load(open(R))
    lib_ids = {t["id"] for t in json.load(open(LIB))["techniques"]}
    for o in d["per_organ"]:
        for s in o["skipped"]:
            assert s["technique"] in lib_ids


def test_prior_failures_actually_attach():
    """Matching negatives to techniques by id substring attached nothing; it is by family."""
    d = json.load(open(R))
    n = sum(len(p["prior_failures_elsewhere"])
            for o in d["per_organ"] for p in o["prescription"])
    assert n > 0, "no prescription carries a prior failure -- the matcher is broken again"
    for o in d["per_organ"]:
        for p in o["prescription"]:
            for f in p["prior_failures_elsewhere"]:
                assert f["reopen_condition"], "a warning with no reopening condition is a dead end"


def test_three_zero_search_is_explicit():
    d = json.load(open(R))
    assert set(d["three_kinds_of_zero"]) == {
        "ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION", "ZERO_EXECUTION"}
    for o in d["per_organ"]:
        covered = set(o["three_zeros_covered"])
        missing = set(o["three_zeros_missing"])
        assert covered | missing == set(d["three_kinds_of_zero"]), o["organ"]
        assert not (covered & missing)


def test_canonical_order_is_encoded_not_narrated():
    d = json.load(open(R))
    assert len(d["canonical_doctor_order"]) == 9
    assert d["canonical_doctor_order"][0].startswith("SHOULD THIS STRUCTURE EXIST")
    for o in d["per_organ"]:
        assert o["canonical_order_asked"] == d["canonical_doctor_order"]


def test_search_space_reduction_is_computed_from_real_counts():
    d = json.load(open(R))
    q = d["prescription_quality"]
    n_exp = q["experiments_to_run"]["value"]
    assert n_exp == len(d["distinct_experiments_prescribed"])
    assert 0 < q["search_space_reduction"]["value"] < 1
