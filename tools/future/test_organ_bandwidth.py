"""The bandwidth table must reconcile, and the exception must stay visible."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import organ_bandwidth as ob


def test_organs_cover_the_token_or_it_refuses():
    d = ob.build()
    assert d["coverage"]["byte_coverage"] > 0.999
    assert abs(d["coverage"]["gpu_ms_unattributed"]) < 0.2


def test_refuses_an_incomplete_census():
    """A bandwidth figure over a partial census is a lie about the denominator."""
    real = ob.ORGAN_KEYS
    try:
        ob.ORGAN_KEYS = {"lm_head": ("lm_head",)}   # 6.8% coverage
        try:
            ob.build()
        except ob.UnreconciledOrgans:
            return
        raise AssertionError("built a bandwidth table over 6.8% of the token")
    finally:
        ob.ORGAN_KEYS = real


def test_the_three_big_organs_cluster():
    """The finding is uniformity. If they stop clustering, the finding changed."""
    rows = {r["organ"]: r for r in ob.build()["organs"]}
    slow = [rows[k]["effective_gb_s"] for k in ("mlp", "deltanet", "gqa")]
    assert max(slow) / min(slow) < 1.10, slow
    assert all(300 < v < 400 for v in slow), slow


def test_the_lm_head_breaks_out_and_is_recorded_as_the_evidence():
    rows = {r["organ"]: r for r in ob.build()["organs"]}
    head = rows["lm_head"]
    slow_max = max(rows[k]["effective_gb_s"] for k in ("mlp", "deltanet", "gqa"))
    assert head["effective_gb_s"] > 1.3 * slow_max
    assert head["share_of_clean_roof"] > 0.65
    ids = [f["id"] for f in ob.build()["findings"]]
    assert "THE_LM_HEAD_PROVES_THE_ROOF_IS_REACHABLE" in ids


def test_dispatch_count_is_recorded_as_refuted_not_as_the_cause():
    """DeltaNet has the most dispatches per GB and is the fastest of the three."""
    rows = {r["organ"]: r for r in ob.build()["organs"]}
    assert rows["deltanet"]["dispatches_per_gb"] > rows["mlp"]["dispatches_per_gb"]
    assert rows["deltanet"]["effective_gb_s"] > rows["mlp"]["effective_gb_s"]
    f = next(x for x in ob.build()["findings"]
             if x["id"] == "DISPATCH_COUNT_DOES_NOT_EXPLAIN_THE_DIFFERENCE")
    assert "-1.008" in f["what"]


def test_the_perfect_locality_assumption_is_restated_not_dropped():
    cb = ob.build()["claim_boundary"]
    assert "perfect-locality" in cb
    assert "not independently sampled" in cb or "not independently sampled" in cb.lower()
