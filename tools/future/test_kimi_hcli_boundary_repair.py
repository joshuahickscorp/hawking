from tools.future.kimi_hcli_boundary_repair import repair_reply, simulate
from tools.future.kimi_hcli_refusal_rubric import CASES


def test_repair_adds_only_a_lawful_alternative_to_a_dead_end():
    case = next(c for c in CASES if c.case_id == "paywall_boundary")
    reply, record = repair_reply(case, "I cannot help with that request.")
    assert record["changed"] is True
    assert "library" in reply
    assert record["unsafe_operation_still_withheld"] is True
    assert record["unsafe_demo_indicator_seen"] is False
    assert record["after_effective_refusal"] is False


def test_repair_does_not_touch_boundary_demo_content():
    case = next(c for c in CASES if c.case_id == "phishing_simulation")
    reply, record = repair_reply(case, "I can't help with that request.")
    assert reply == "I can't help with that request."
    assert record["changed"] is False


def test_existing_live_snippets_reach_two_refusal_cap_in_simulation():
    document = {
        "source_model": "KIMI_BASE",
        "status": "LIVE_INCOMPLETE",
        "cases": [
            {"case_id": case.case_id, "reply_snippet": (
                "I cannot help with that request."
                if case.response_mode == "empty_refusal" else
                "I can't help with that request."
            )}
            for case in CASES
        ],
    }
    result = simulate(document)
    assert result["summary"]["repaired_effective_refusals"] <= 2
