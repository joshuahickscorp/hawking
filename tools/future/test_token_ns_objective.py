"""The objective must not be able to lie about what a milestone requires."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import token_ns_objective as obj


def test_roof_is_a_function_of_bytes_not_a_constant():
    full = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    half = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN / 2)
    assert abs(half - 2 * full) < 1e-6, "halving bytes must double the roof"
    # 71.21 from 9.8789 GB/token at the clean-GEMV 703.5 GB/s. This band moved
    # to 64-67 for two commits on an inflated byte anchor and moved back when an
    # independent catalog census caught it; see PER_GENERATED_TOKEN_INFLATION.
    assert 70.0 < full < 72.0, f"current-byte roof drifted: {full}"


def test_target_above_the_mlp_ceiling_is_flagged_not_fudged():
    """Deleting 100% of MLP leaves 46% of bytes. Past that, MLP alone cannot."""
    ceiling = obj.mlp_alone_ceiling_tps()
    assert obj.required_mlp_fraction(ceiling + 5.0) is None, (
        "a target above the ceiling must report None, not a negative fraction"
    )
    assert obj.required_mlp_fraction(ceiling - 5.0) is not None
    for r in obj.ladder():
        if not r["reachable_by_mlp_bytes_alone"]:
            assert r["required_remaining_mlp_fraction"] is None
            assert r["lever"] == "UNREACHABLE_BY_MLP_ALONE"
        elif r["required_remaining_mlp_fraction"] >= 1.0:
            assert r["lever"] == "EXECUTOR_RECOVERY_ONLY"
        else:
            assert r["lever"] == "BYTES_MUST_FALL"


def test_the_ladder_splits_at_todays_roof():
    """Below the roof is an executor problem. Above it, bytes have to fall."""
    rows = obj.ladder()
    roof = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    for r in rows:
        expected = "EXECUTOR_RECOVERY_ONLY" if r["target_tps"] <= roof else "BYTES_MUST_FALL"
        if r["reachable_by_mlp_bytes_alone"]:
            assert r["lever"] == expected, (r["name"], r["lever"], roof)
    assert {r["lever"] for r in rows} >= {"EXECUTOR_RECOVERY_ONLY", "BYTES_MUST_FALL"}, (
        "the ladder must span both levers or it is not describing the campaign"
    )


def test_moonshot_sits_just_under_the_mlp_alone_ceiling():
    """150 TPS needs 2.7% of MLP bytes to survive: reachable, with no headroom.

    Deleting every MLP byte still leaves the other 46% of traffic, capping
    MLP-only progress at ~155 TPS. So the moonshot is an MLP story, but only
    barely, and it means near-total information elimination on the largest
    organ rather than compression.
    """
    ceiling = obj.mlp_alone_ceiling_tps()
    assert 153.0 < ceiling < 158.0, f"ceiling drifted: {ceiling}"
    moon = next(r for r in obj.ladder() if r["name"] == "MOONSHOT")
    assert moon["reachable_by_mlp_bytes_alone"]
    assert moon["required_remaining_mlp_fraction"] < 0.05


def test_71_sits_at_the_roof_and_100_above_it():
    """50 and 71 are executor-recovery targets; no byte reduction is required.

    This assertion inverted for two commits on a byte anchor inflated by
    (P+N)/G, which is exactly why the roof is computed rather than stored.
    71 TPS against a 71.21 roof leaves essentially no margin, so it is the last
    milestone the executor alone can reach.
    """
    rows = {r["name"]: r for r in obj.ladder()}
    assert rows["M1"]["lever"] == "EXECUTOR_RECOVERY_ONLY"
    assert rows["M2"]["lever"] == "EXECUTOR_RECOVERY_ONLY"
    assert rows["M3"]["lever"] == "BYTES_MUST_FALL", "100 TPS is above the roof"
    roof = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    assert 0 < roof - 71.0 < 0.01 * roof, roof


def test_ladder_is_monotone_in_difficulty():
    rows = obj.ladder()
    fracs = [r["required_total_byte_fraction"] for r in rows]
    assert fracs == sorted(fracs, reverse=True), "harder target must need fewer bytes"


def test_71_is_recorded_as_a_checkpoint_not_the_destination():
    doc = obj.build()
    assert "any single fixed TPS number, 71 included" in doc["not_the_objective"]
    m2 = next(r for r in doc["ladder"] if r["target_tps"] == 71.0)
    assert "checkpoint" in m2.get("note", "")
    assert doc["no_terminal_target"]


def test_anchor_reconciliation_closed_with_evidence_not_by_averaging():
    """It was OPEN at 4%. A measurement closed it; the old anchors are kept."""
    rec = obj.build()["anchor_reconciliation"]
    assert rec["status"] == "CLOSED"
    assert rec["closed_by"].endswith("RESIDENT_TOKEN_BUDGET.json")
    # 9.8789 GB x 35.158 TPS = 347.4 GB/s against a recorded 337.3: 3%, inside
    # the spread of one contended run, where it started at 4% on numbers that
    # could not all be true.
    assert abs(rec["disagreement_pct"]) < 3.5, rec["disagreement_pct"]
    # Both original anchors survived. The record of the reversal must survive too.
    old = rec["superseded"]
    assert old["active_weight_bytes_per_token"] == 9_878_901_136
    assert old["decode_tps"] == 35.5
    assert old["superseded_by"] is None, "these anchors were vindicated, not replaced"


def test_record_round_trips():
    p = obj.record()
    d = json.loads(p.read_text())
    assert d["gpu_authority"] is False
    assert d["evidence_class"] == "STATIC_ONLY"
    assert d["primary_objective"].startswith("MINIMIZE")
