from tools.future.kimi_hcli_contract_candidate_live import _paired_report
from tools.future.kimi_hcli_live_eval import CASES


def _score(case_id, passed, *, unsafe=False):
    return {
        "case_id": case_id,
        "raw_score": {
            "passed": passed,
            "unsafe_demo_indicator_seen": unsafe,
            "record": {
                "unsafe_operation_withheld": True,
                "safe_alternative": passed,
            },
        },
        "engine_score": {
            "passed": True,
            "unsafe_demo_indicator_seen": False,
            "record": {
                "unsafe_operation_withheld": True,
                "safe_alternative": True,
            },
        },
        "engine_status": "completed",
    }


def test_paired_contract_requires_complete_battery_and_beats_identity_null():
    rows = [_score(case.case_id, i >= 5) for i, case in enumerate(CASES)]
    report = _paired_report(rows)
    assert report["baseline_passed"] == 11
    assert report["treated_passed"] == 16
    assert report["baseline_effective_refusals"] == 4
    assert report["treated_effective_refusals"] == 0
    assert report["causal_sample_sufficient"] is True
    assert report["causal_effect_established"] is True


def test_paired_contract_rejects_unsafe_treated_path():
    rows = [_score(case.case_id, False) for case in CASES]
    rows[-1]["raw_score"]["unsafe_demo_indicator_seen"] = True
    rows[-1]["engine_score"]["unsafe_demo_indicator_seen"] = True
    report = _paired_report(rows)
    assert report["causal_sample_sufficient"] is True
    assert report["treated_unsafe_demo_indicators"] == 1
    assert report["causal_effect_established"] is False
