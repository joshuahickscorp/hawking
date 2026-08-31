"""Campaign scars are guards, not stories. The index must refuse at least one."""
from __future__ import annotations

import json
import subprocess
import sys

from tools.future import campaign_scars as cs
from tools.future import negative_index as ni
from tools.future import status_causality as sc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, REPO, _assert_no_hardware_claims

SEVEN = (
    "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR",
    "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED",
    "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
    "PRIORITY_ZERO_FALSY_OR_DEFAULT",
    "EVENT_TIMESTAMP_UNIT_MISMATCH",
    "ADJACENCY_IS_NOT_OVERLAP",
    "SHARED_INDEX_BARE_COMMIT_SWEEPS_FOREIGN_STAGE",
)


def test_seven_distinct_scars_record_the_contract_fields():
    ids = cs.scar_ids()
    assert ids == list(SEVEN), ids
    assert len(set(ids)) == 7
    assert cs.missing_fields() == []
    for scar in cs.scars():
        for field in cs.REQUIRED_FIELDS:
            assert str(scar[field]).strip(), f"{scar['id']} missing {field}"
        assert scar["verdict"] == "FALSIFIED"
        assert scar["hypothesis_family"] == scar["id"]
        assert "964" in scar["observed"] or scar["id"] != "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED"
        assert "2.34" in scar["observed"] or scar["id"] != "EVENT_TIMESTAMP_UNIT_MISMATCH"


def test_build_emits_sealed_receipt_with_seven_scars():
    out = cs.build()
    assert out.parent == RECEIPTS
    assert out.name == "CAMPAIGN_SCARS.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == cs.SCHEMA
    assert doc["n_scars"] == 7
    assert doc["scar_ids"] == list(SEVEN)
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert len(doc["scars"]) == 7
    assert len(doc["entries"]) == 7
    assert doc["scars_missing_required_fields"] == []
    assert doc["claim_checker"]["n_author_overreaches"] >= 3
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")


def test_negative_index_ingests_the_seven_and_marks_them_refuse_eligible():
    cs.build()
    scars = ni.ingest(force=True)
    by_id = {}
    for s in scars:
        if s.source_path.endswith("CAMPAIGN_SCARS.json") and s.parse_status == ni.PARSED:
            by_id[s.original_id] = s
    missing = [i for i in SEVEN if i not in by_id]
    assert not missing, f"index missed campaign scars: {missing}"
    for sid, scar in by_id.items():
        assert scar.refuse_eligible, f"{sid} not refuse_eligible verdict={scar.verdict!r}"
        assert scar.hypothesis_family
        assert scar.reopen_condition and scar.reopen_condition != ni.UNRECORDED
        assert scar.claim_refuted and scar.claim_refuted != ni.UNRECORDED


def test_refuse_if_dead_fires_on_at_least_one_of_the_seven():
    """Acceptance: negative_index --refuse returns refused for a campaign scar."""
    cs.build()
    pool = ni.ingest(force=True)
    hits = []
    for sid in SEVEN:
        refusal = ni.refuse_if_dead({"hypothesis_family": sid}, scars=pool)
        if refusal is not None:
            hits.append(refusal)
            assert refusal["refused"] is True
            assert refusal["source_path"].endswith("CAMPAIGN_SCARS.json")
            assert sid in refusal["scar_id"] or sid == refusal.get("original_id")
    assert hits, "refuse_if_dead did not fire on any of the seven campaign scars"


def test_cli_refuse_returns_refused_for_a_campaign_scar():
    cs.build()
    family = SEVEN[0]
    proc = subprocess.run(
        [
            sys.executable,
            "tools/future/negative_index.py",
            "--refuse",
            json.dumps({"hypothesis_family": family}),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["refused"] is True
    assert family.lower() in json.dumps(doc).lower() or family in str(doc.get("original_id"))
    assert "CAMPAIGN_SCARS.json" in doc["source_path"]


def test_prefill_over_generated_token_is_refuse_eligible():
    cs.build()
    refusal = ni.refuse_if_dead(
        {"hypothesis_family": "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR"},
        scars=ni.ingest(force=True),
    )
    assert refusal is not None
    assert refusal["refused"] is True
    blob = json.dumps(refusal).lower()
    assert "prefill_over_generated_token_denominator" in blob


def test_shared_index_scar_refuses_bare_commit_hypothesis():
    cs.build()
    refusal = ni.refuse_if_dead(
        {"hypothesis_family": "SHARED_INDEX_BARE_COMMIT_SWEEPS_FOREIGN_STAGE"},
        scars=ni.ingest(force=True),
    )
    assert refusal is not None
    assert refusal["refused"] is True


def test_claim_checker_is_not_a_lookup_table():
    """Flags classify. A novel structured case must fire without being seeded."""
    row = cs.check_claim(
        observation="Path.exists('/stale/path') is False",
        conclusion="the model is missing",
        entails=False,
        consistent_with_negation=True,
        contradicts=False,
    )
    assert row["verdict"] == cs.OVERREACHING
    supported = cs.check_claim(
        observation="n_files=144 verified=144 mismatched=0",
        conclusion="published digests match the hashes recomputed here",
        entails=True,
        consistent_with_negation=False,
        contradicts=False,
    )
    assert supported["verdict"] == cs.SUPPORTED
    unknown = cs.check_claim("some novel observation", "some novel conclusion")
    assert unknown["verdict"] == cs.UNDERDETERMINED


def test_seeded_campaign_claims_cover_today_including_author_overreach():
    rows = cs.check_campaign_claims()
    by_id = {r["id"]: r for r in rows}
    assert by_id["CC.PREFILL_AS_PER_TOKEN"]["verdict"] == cs.OVERREACHING
    assert by_id["CC.964_AS_PRODUCTION"]["verdict"] == cs.OVERREACHING
    assert by_id["CC.964_AS_PRODUCTION"]["author_was_the_one_who_concluded"] is True
    assert by_id["CC.SOURCE_FIELDS_AS_RUNNING"]["verdict"] == cs.OVERREACHING
    assert by_id["CC.PRIORITY_ZERO_VIA_OR"]["verdict"] == cs.CONTRADICTED
    assert by_id["CC.MIXED_UNITS_NO_OVERLAP"]["verdict"] == cs.CONTRADICTED
    assert by_id["CC.ADJACENCY_AS_OVERLAP"]["verdict"] == cs.OVERREACHING
    assert by_id["CC.ADJACENCY_AS_OVERLAP"]["author_was_the_one_who_concluded"] is True
    assert by_id["CC.BARE_COMMIT_INTENDED_PATHS"]["verdict"] == cs.OVERREACHING
    assert by_id["CC.BARE_COMMIT_INTENDED_PATHS"]["author_was_the_one_who_concluded"] is True
    assert by_id["CC.CATALOG_CENSUS_INFLATION"]["verdict"] == cs.SUPPORTED
    assert by_id["CC.STATIC_AND_LIVE_DISPATCH"]["verdict"] == cs.SUPPORTED
    assert by_id["CC.MISSING_ENV_WHICH_GRAPH"]["verdict"] == cs.UNDERDETERMINED
    assert by_id["CC.STRINGS_ABSENT_MAY_BE_STRIPPED"]["verdict"] == cs.UNDERDETERMINED
    assert by_id["CC.UNFUSED_INHERITS_INCUMBENT"]["verdict"] == cs.OVERREACHING
    assert {r["verdict"] for r in rows} >= {
        cs.SUPPORTED, cs.OVERREACHING, cs.UNDERDETERMINED, cs.CONTRADICTED,
    }


def test_mixed_timestamp_units_are_contradicted_as_no_overlap():
    row = cs.check_claim(case_id="CC.MIXED_UNITS_NO_OVERLAP")
    assert row["verdict"] == cs.CONTRADICTED
    assert "2.34" in str(row["observation"])
    assert row["contradicts"] is True


def test_964_as_production_is_the_authors_overreach():
    row = cs.check_claim(case_id="CC.964_AS_PRODUCTION")
    assert row["verdict"] == cs.OVERREACHING
    assert row["author_was_the_one_who_concluded"] is True
    assert row["scar_id"] == "ENVIRONMENT_MISMATCH_UNFUSED_VS_SEALED"


def test_check_claim_lives_in_status_causality_not_a_fork():
    assert cs.check_claim is sc.check_claim
    assert cs.UNDERDETERMINED == sc.UNDERDETERMINED
    assert cs.CONTRADICTED == sc.CONTRADICTED
    # challenge() keeps its original three verdicts.
    assert sc.UNTESTED in sc.VERDICTS
    assert sc.UNDERDETERMINED not in sc.VERDICTS
    assert sc.UNDERDETERMINED in sc.CLAIM_CHECK_VERDICTS


def test_challenge_still_classifies_historical_overreach():
    """Extending the module must not break the status-challenge path."""
    row = sc.challenge("BLOCKED_NO_METAL_GPU")
    assert row["verdict"] == sc.OVERREACHING
