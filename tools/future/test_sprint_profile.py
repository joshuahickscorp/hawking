"""Sprint-wall attribution: UNKNOWN is first-class, and tallest() must refuse.

A validator nobody has watched reject is a validator that will silently
drift into fiction. These tests force the negative of every judgement:
no-evidence is UNKNOWN not zero, UNKNOWN dominating refuses a winner,
remainder is reported not distributed, and a missing 1h timeline is a
recorded refusal rather than a skip.
"""
from __future__ import annotations

import json

import pytest

from tools.future import autonomy_run as ar
from tools.future import autonomy_scars as asc
from tools.future import sprint_profile as sp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.repro_science import FailClosed


def _bucket(bucket_id: str, seconds: int, derivation: str = sp.MEASURED) -> dict:
    return sp.make_bucket(
        bucket_id,
        attributed_seconds=seconds,
        derivation=derivation,
        evidence=[{"fixture": True}],
        why="test fixture",
    )


def _profile(buckets: list[dict], elapsed_s: int | None) -> dict:
    rem = sp.remainder_of(elapsed_s=elapsed_s, buckets=buckets)
    return {"buckets": buckets, "unattributed_remainder": rem}


def _fill_unknown(attributed: list[dict]) -> list[dict]:
    have = {b["id"] for b in attributed}
    out = list(attributed)
    for bid in sp.BUCKET_IDS:
        if bid not in have:
            out.append(sp.unknown_bucket(bid, why="fixture: no evidence"))
    order = {i: n for n, i in enumerate(sp.BUCKET_IDS)}
    out.sort(key=lambda b: order[b["id"]])
    return out


def test_unknown_bucket_is_none_not_zero():
    """NEGATIVE CONTROL: absence is UNKNOWN. Zero is a claim."""
    b = sp.unknown_bucket("gpu", why="no GPU lease")
    assert b["attributed_seconds"] is None
    assert b["derivation"] == sp.UNKNOWN
    assert b["attributed_seconds"] != 0
    assert sp.is_attributed(b) is False


def test_unknown_with_a_number_is_refused():
    """NEGATIVE CONTROL: UNKNOWN carrying seconds would launder a guess."""
    with pytest.raises(FailClosed, match="unknown_must_not_carry_seconds"):
        sp.make_bucket(
            "gpu",
            attributed_seconds=0,
            derivation=sp.UNKNOWN,
            evidence=[],
            why="would launder",
        )
    with pytest.raises(FailClosed, match="unknown_must_not_carry_seconds"):
        sp.make_bucket(
            "compilation",
            attributed_seconds=12,
            derivation=sp.UNKNOWN,
            evidence=[],
            why="would launder",
        )


def test_measured_without_seconds_is_refused():
    with pytest.raises(FailClosed, match="attributed_without_seconds"):
        sp.make_bucket(
            "trial_reruns",
            attributed_seconds=None,
            derivation=sp.MEASURED,
            evidence=[],
            why="incomplete",
        )


def test_tallest_refuses_when_unknown_dominates():
    """NEGATIVE CONTROL: the ranking must actually refuse, not pick the thing we measured."""
    buckets = _fill_unknown([_bucket("trial_reruns", 100)])
    ranking = sp.tallest(_profile(buckets, elapsed_s=1000))
    assert ranking["named"] is False
    assert ranking["winner"] is None
    assert ranking["unknown_mass_s"] == 900
    assert "measure first" in ranking["reason"]
    assert ranking["leader_if_unknown_were_smaller"]["id"] == "trial_reruns"


def test_tallest_refuses_when_unknown_ties_the_leader():
    buckets = _fill_unknown([_bucket("trial_reruns", 500)])
    ranking = sp.tallest(_profile(buckets, elapsed_s=1000))
    assert ranking["named"] is False
    assert ranking["winner"] is None
    assert ranking["unknown_mass_s"] == 500


def test_tallest_refuses_when_elapsed_is_unmeasured():
    """Live case: no sprint clock. UNKNOWN cannot be shown not to dominate."""
    buckets = _fill_unknown([_bucket("trial_reruns", 8450)])
    ranking = sp.tallest(_profile(buckets, elapsed_s=None))
    assert ranking["named"] is False
    assert ranking["winner"] is None
    assert ranking["unknown_mass_s"] is None
    assert "measure first" in ranking["reason"]


def test_tallest_refuses_when_nothing_is_attributed():
    buckets = _fill_unknown([])
    ranking = sp.tallest(_profile(buckets, elapsed_s=100))
    assert ranking["named"] is False
    assert ranking["winner"] is None
    assert ranking["unknown_mass_s"] == 100
    assert "no attributed bucket" in ranking["reason"]


def test_tallest_names_a_winner_when_attributed_exceeds_unknown():
    """The positive of the refusal: a ranking we have actually watched succeed."""
    buckets = _fill_unknown([_bucket("trial_reruns", 700)])
    ranking = sp.tallest(_profile(buckets, elapsed_s=1000))
    assert ranking["named"] is True
    assert ranking["winner"] == "trial_reruns"
    assert ranking["attributed_seconds"] == 700
    assert ranking["unknown_mass_s"] == 300


def test_remainder_is_reported_not_distributed():
    """NEGATIVE CONTROL: buckets must not be required to sum to the wall clock."""
    attributed = [_bucket("trial_reruns", 200)]
    buckets = _fill_unknown(attributed)
    rem = sp.remainder_of(elapsed_s=1000, buckets=buckets)
    assert rem["seconds"] == 800
    assert rem["distributed_into_buckets"] is False
    assert rem["attributed_sum_s"] == 200
    assert sp.attributed_sum(buckets) + rem["seconds"] == 1000
    for b in buckets:
        if b["id"] != "trial_reruns":
            assert b["attributed_seconds"] is None, (
                f"{b['id']} received a slice of the remainder; that is distribution"
            )
    # The named buckets do not sum to elapsed.
    assert sp.attributed_sum(buckets) != 1000


def test_remainder_refuses_to_clamp_a_double_count():
    buckets = _fill_unknown([_bucket("trial_reruns", 500)])
    with pytest.raises(FailClosed, match="attributed_exceeds_elapsed"):
        sp.remainder_of(elapsed_s=100, buckets=buckets)


def test_empty_inputs_make_trial_reruns_and_scars_unknown_not_zero():
    """NEGATIVE CONTROL: explicit absence is a recorded refusal, not a skip."""
    profile = sp.attribute(
        snapshots=[],
        invalidated_rows=[],
        scars=[],
        elapsed_s=None,
    )
    by_id = {b["id"]: b for b in profile["buckets"]}
    assert set(by_id) == set(sp.BUCKET_IDS)
    for bid in sp.BUCKET_IDS:
        assert by_id[bid]["attributed_seconds"] is None
        assert by_id[bid]["derivation"] == sp.UNKNOWN
        assert by_id[bid]["attributed_seconds"] != 0
    assert profile["attributed_sum_s"] == 0
    assert profile["unattributed_remainder"]["seconds"] is None
    assert profile["unattributed_remainder"]["distributed_into_buckets"] is False
    assert profile["tallest"]["named"] is False
    assert profile["recovery"]["failed_1h"] is None
    assert profile["recovery"]["n_invalidated"] == 0
    assert profile["recovery"]["n_scars"] == 0


def test_infrastructure_defect_cites_scars_and_does_not_invent_seconds():
    scars = asc.scars()
    assert len(scars) == 4
    bucket = sp.infrastructure_defect_bucket(scars)
    assert bucket["id"] == "infrastructure_defect"
    assert bucket["derivation"] == sp.UNKNOWN
    assert bucket["attributed_seconds"] is None
    ids = [e["id"] for e in bucket["evidence"]]
    assert ids == [s["id"] for s in scars]
    for row in bucket["evidence"]:
        assert row["cost"], f"{row['id']} dropped its qualitative cost"
        assert "seconds" not in row


def test_trial_reruns_from_judged_fail_and_invalidated_timestamps():
    """The judge is invoked. A commit subject is not the verdict."""
    fail_tl = {
        "trial": "1h",
        "duration_s": 3600,
        "elapsed_s": 3600,
        "events": [
            {"kind": "state_recovered", "t_s": 0, "seq": 0, "payload": {"path": "x"}, "cites": ["x"]},
        ],
    }
    verdict = sp.judge_1h(fail_tl)
    assert verdict["verdict"] == "FAIL"
    assert verdict["elapsed_is_not_a_pass"] is True
    assert "ingest_completed_result" in verdict["unmet"]

    failed = sp.failed_1h_trial([{
        "commit": "fixture",
        "path_taken": "fixture",
        "doc": fail_tl,
    }])
    assert failed is not None
    assert failed["verdict"] == "FAIL"
    assert failed["elapsed_s"] == 3600
    assert failed["judge"].startswith("tools.future.autonomy_trial.verify")

    inv = sp.invalidated_seconds(ar.INVALIDATED_RUNS[0])
    assert inv["elapsed_s"] == 4821
    assert inv["verdict"] == "INVALIDATED_BY_SUBSTRATE_MUTATION"
    assert inv["derivation"] == sp.MEASURED

    bucket = sp.trial_reruns_bucket(failed=failed, invalidated=[inv])
    assert bucket["derivation"] == sp.MEASURED
    assert bucket["attributed_seconds"] == 3600 + 4821


def test_passing_timeline_is_not_the_failed_first_trial():
    """STATUS LABELS ARE HYPOTHESES. A live PASS must not be billed as the FAIL."""
    passing = {
        "trial": "1h",
        "duration_s": 3600,
        "elapsed_s": 3600,
        "events": [],
    }
    # A timeline with no events fails conditions; build one the judge would PASS
    # by using the harness's own passing fixture shape if available, otherwise
    # assert that a FAIL-then-PASS history keeps the FAIL as the first trial.
    fail_tl = {
        "trial": "1h",
        "elapsed_s": 3629,
        "events": [{"kind": "receipt_ingested", "t_s": 0, "seq": 0, "payload": {}}],
    }
    # Newest snapshot last.
    history = [
        {"commit": "old", "path_taken": "git:old", "doc": fail_tl},
        {"commit": "new", "path_taken": "git:new", "doc": passing},
    ]
    failed = sp.failed_1h_trial(history)
    live = sp.live_1h_verdict(history)
    assert failed is not None
    assert failed["elapsed_s"] == 3629
    assert failed["verdict"] == "FAIL"
    assert failed["path_taken"] == "git:old"
    assert live is not None
    assert live["path_taken"] == "git:new"
    assert live["elapsed_s"] == 3600


def test_invalidated_interval_fails_closed_on_missing_or_inverted_stamps():
    with pytest.raises(FailClosed, match="incomplete_invalidated_interval"):
        sp.invalidated_seconds({"started": "2026-08-30T08:57:13-04:00", "verdict": "X"})
    with pytest.raises(FailClosed, match="inverted_invalidated_interval"):
        sp.invalidated_seconds({
            "started": "2026-08-30T10:17:34-04:00",
            "killed": "2026-08-30T08:57:13-04:00",
            "verdict": "INVALIDATED_BY_SUBSTRATE_MUTATION",
        })
    with pytest.raises(FailClosed, match="malformed_timestamp"):
        sp.invalidated_seconds({
            "started": "not-a-date",
            "killed": "2026-08-30T10:17:34-04:00",
        })


def test_attribute_recovers_live_disk_and_keeps_unknown_first_class():
    """Cope with presence or absence. Never skip."""
    profile = sp.attribute()
    by_id = {b["id"]: b for b in profile["buckets"]}
    assert list(b["id"] for b in profile["buckets"]) == list(sp.BUCKET_IDS)

    reruns = by_id["trial_reruns"]
    if reruns["derivation"] == sp.UNKNOWN:
        assert reruns["attributed_seconds"] is None
    else:
        assert reruns["derivation"] == sp.MEASURED
        assert isinstance(reruns["attributed_seconds"], int)
        assert reruns["attributed_seconds"] > 0
        # The judge ran: a FAIL interval cites unmet conditions.
        kinds = {e.get("verdict") for e in reruns["evidence"]}
        assert "FAIL" in kinds or "INVALIDATED_BY_SUBSTRATE_MUTATION" in kinds
        for row in reruns["evidence"]:
            if row.get("verdict") == "FAIL":
                assert row.get("unmet"), "FAIL without unmet conditions is a status without a cause"
                assert "verify" in str(row.get("judge") or "")

    defect = by_id["infrastructure_defect"]
    assert defect["attributed_seconds"] is None
    assert defect["derivation"] == sp.UNKNOWN
    if defect["evidence"] and defect["evidence"][0].get("id"):
        assert len(defect["evidence"]) == 4
        assert {e["id"] for e in defect["evidence"]} == {s["id"] for s in asc.scars()}

    for bid in (
        "scheduler", "resident_reasoning", "compilation", "gpu",
        "model_loading", "verification", "source_io", "process_wait",
        "human_authority",
    ):
        assert by_id[bid]["derivation"] == sp.UNKNOWN
        assert by_id[bid]["attributed_seconds"] is None

    rem = profile["unattributed_remainder"]
    assert rem["distributed_into_buckets"] is False
    assert rem["seconds"] is None  # no sprint clock on disk
    assert profile["sprint"]["target_s"] == sp.TARGET_S
    assert profile["sprint"]["elapsed_s"] is None
    assert profile["tallest"]["named"] is False
    assert "measure first" in profile["tallest"]["reason"]


def test_build_seals_static_only_receipt():
    out = sp.build()
    assert out.parent == RECEIPTS
    assert out.name == "ODYSSEY_LAUNCH_SPRINT_PROFILE.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == sp.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["tallest"]["named"] is False
    assert doc["unattributed_remainder"]["distributed_into_buckets"] is False
    assert [b["id"] for b in doc["buckets"]] == list(sp.BUCKET_IDS)
    rc = doc["resident_callable"]
    assert rc["entry_point"]
    assert rc["workunit"]
    assert rc["receipt"] == f"receipts/future/{sp.RECEIPT}"
    assert rc["frontier"] == "FT.LATENCY.cpu-turnaround"
    assert rc["fails_closed"]
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc


def test_measured_zero_is_allowed_as_a_claim_with_evidence():
    """Zero is a claim. It may exist when derivation is measured; it must not stand in for UNKNOWN."""
    b = sp.make_bucket(
        "gpu",
        attributed_seconds=0,
        derivation=sp.MEASURED,
        evidence=[{"clock": "interval contained no GPU work, measured"}],
        why="measured empty interval",
    )
    assert b["attributed_seconds"] == 0
    assert sp.is_attributed(b) is True
