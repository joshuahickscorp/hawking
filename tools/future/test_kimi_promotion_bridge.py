from tools.future.kimi_promotion_bridge import FAMILY_ID, plan


def test_bridge_refuses_before_nx_when_any_gate_is_missing():
    result = plan(
        {
            "status": "PROMOTION_REFUSED",
            "missing_gates": ["causal_behavior", "hcli"],
            "candidates": [{
                "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA",
                "missing_gates": ["causal_behavior", "hcli"],
            }],
        },
        "KIMI_OPERATOR_CANDIDATE_OPA",
    )
    assert result["status"] == "PROMOTION_REFUSED"
    assert result["noetic_family"] == FAMILY_ID
    assert result["artifact_created"] is False
    assert result["weights_written"] is False


def test_bridge_refuses_allowed_gate_without_artifact_and_nova_lineage():
    result = plan(
        {
            "status": "PROMOTION_ALLOWED",
            "evidence": {"evaluation_receipt": "receipt.json"},
            "candidates": [{
                "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA",
                "missing_gates": [],
                "artifact_hash": None,
            }],
        },
        "KIMI_OPERATOR_CANDIDATE_OPA",
    )
    assert result["status"] == "PROMOTION_REFUSED"
    assert "candidate_artifact_hash" in result["missing_provenance"]
    assert "nova_lineage_receipt" in result["missing_provenance"]
