from tools.future.kimi_odyssey_bridge import plan


def test_kimi_worker_is_withheld_while_promotion_gate_is_refused():
    result = plan(
        {
            "status": "PROMOTION_REFUSED",
            "missing_gates": ["capability", "physical"],
            "candidates": [{
                "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA",
                "missing_gates": ["capability", "physical"],
            }],
        },
        "KIMI_OPERATOR_CANDIDATE_OPA",
    )
    assert result["status"] == "KIMI_WORKER_WITHHELD"
    assert result["scope"] == []


def test_ready_worker_scope_is_bounded_and_rejects_authority_expansion():
    gate = {
        "status": "PROMOTION_ALLOWED",
        "candidates": [{
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA",
            "missing_gates": [],
        }],
    }
    result = plan(gate, "KIMI_OPERATOR_CANDIDATE_OPA")
    assert result["status"] == "READY_FOR_BOUNDED_ODYSSEY_WORKUNITS"
    assert "artifact.promote" not in result["scope"]
    bad = plan(
        gate,
        "KIMI_OPERATOR_CANDIDATE_OPA",
        requested_scope=["model.inspect", "external.write"],
    )
    assert bad["status"] == "KIMI_WORKER_WITHHELD"
