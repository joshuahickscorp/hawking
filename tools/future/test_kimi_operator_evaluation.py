from tools.future.kimi_operator_evaluation import evaluate_candidates


def test_all_registered_candidates_pass_copy_only_functional_gate():
    result = evaluate_candidates(code_commit="test")
    assert result["status"] == "FUNCTIONAL_SIM_COMPLETE"
    assert result["candidate_count"] == 7
    assert {row["method"] for row in result["candidates"]} == {
        "OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG"
    }
    assert all(row["functional_sim_passed"] for row in result["candidates"])
    assert all(row["base_unchanged"] for row in result["candidates"])
    assert all(row["weights_written"] is False for row in result["candidates"])
    assert all(row["promotion"] == "NOT_PERFORMED" for row in result["candidates"])


def test_functional_receipt_does_not_claim_behavior_or_capability():
    result = evaluate_candidates()
    assert all(row["behavioral_claim"] == "NOT_MEASURED" for row in result["candidates"])
    assert all(row["capability_claim"] == "NOT_MEASURED" for row in result["candidates"])
    assert result["ranking"] is None
