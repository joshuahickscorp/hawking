from tools.future.kimi_operator_gate import evaluate


def test_missing_causal_and_policy_evidence_refuses_promotion():
    result = evaluate(
        {
            "schema": "test",
            "specimen": "KIMI_BASE",
            "behavioural_direction": {
                "heldout_ready": True,
                "real": True,
                "peak_layer": 27,
                "heldout_mean_auroc": 0.9,
                "shuffle_heldout_mean_auroc": 0.7,
            },
            "causal_intervention": {"causal_effect_established": False},
        },
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE",
         "candidates": [{"candidate_id": "OP-A"}]},
        {"status": "SPEC_ONLY"},
    )
    assert result["status"] == "PROMOTION_REFUSED"
    assert "causal_behavior" in result["missing_gates"]
    assert "hcli" in result["missing_gates"]
    assert result["candidates"][0]["promotable"] is False


def test_all_gates_are_required_even_with_causal_effect():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "causal_intervention": {"causal_effect_established": True}},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "LIVE_SCORED"},
    )
    assert result["status"] == "PROMOTION_REFUSED"
    assert set(result["missing_gates"]) >= {
        "capability", "epistemic", "authorization", "physical", "destructive_controls"
    }


def test_single_split_causal_effect_without_sample_floor_is_not_promotable():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "causal_intervention": {
             "causal_effect_established": True,
             "causal_sample_sufficient": False,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "LIVE_SCORED"},
    )
    assert result["status"] == "PROMOTION_REFUSED"
    assert "causal_behavior" in result["missing_gates"]


def test_crossfit_causal_arm_can_satisfy_the_causal_gate():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "causal_intervention": {"causal_effect_established": False},
         "crossfit_causal_intervention": {
             "operator_kind": "crossfit_residual_projection",
             "causal_effect_established": True,
             "causal_sample_sufficient": True,
             "crossfit_consistent": True,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "SPEC_ONLY"},
    )
    assert result["gates"]["causal_behavior"] is True


def test_null_writer_arm_does_not_mask_positive_residual_evidence():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "causal_intervention": {"causal_effect_established": False},
         "crossfit_causal_intervention": {
             "operator_kind": "crossfit_residual_projection",
             "causal_effect_established": True,
             "causal_sample_sufficient": True,
             "crossfit_consistent": True,
         },
         "crossfit_writer_causal_intervention": {
             "operator_kind": "crossfit_layer_local_writer_projection",
             "causal_effect_established": False,
             "causal_sample_sufficient": True,
             "crossfit_consistent": False,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "SPEC_ONLY"},
    )
    assert result["gates"]["causal_behavior"] is True
    assert result["evidence"]["causal_evidence_source"] == "crossfit_residual_projection"


def test_candidate_matrix_receipt_is_a_separate_gate():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "crossfit_causal_intervention": {
             "operator_kind": "crossfit_residual_projection",
             "causal_effect_established": True,
             "causal_sample_sufficient": True,
             "crossfit_consistent": True,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "LIVE_SCORED"},
        evidence={"axes": {name: {"passed": True} for name in (
            "capability", "epistemic", "authorization", "physical", "destructive_controls"
        )}},
        candidate_evaluation={
            "status": "FUNCTIONAL_SIM_COMPLETE",
            "candidate_count": 7,
            "candidates": [{"functional_sim_passed": True}] * 7,
        },
    )
    assert result["gates"]["candidate_matrix"] is True


def test_unevaluated_candidate_cannot_be_promotable_from_global_residual_signal():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "crossfit_causal_intervention": {
             "operator_kind": "crossfit_residual_projection",
             "causal_effect_established": True,
             "causal_sample_sufficient": True,
             "crossfit_consistent": True,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE",
         "candidates": [{"candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA"}]},
        {"status": "LIVE_SCORED"},
        evidence={"axes": {name: {"passed": True} for name in (
            "capability", "epistemic", "authorization", "physical", "destructive_controls"
        )}},
        candidate_evaluation={
            "status": "FUNCTIONAL_SIM_COMPLETE", "candidate_count": 7,
            "candidates": [{"functional_sim_passed": True}] * 7,
        },
        live_candidate_evidence={
            "status": "CANDIDATE_LIVE_EVIDENCE_INCOMPLETE",
            "candidates": {"KIMI_OPERATOR_CANDIDATE_OPA": {"status": "NOT_MEASURED"}},
            "qualified_candidates": [],
        },
    )
    assert result["gates"]["causal_behavior"] is True
    assert result["gates"]["candidate_live_evaluation"] is False
    assert result["candidates"][0]["promotable"] is False
    assert "candidate_live_evaluation" in result["candidates"][0]["missing_gates"]


def test_serving_contract_is_reported_without_opening_weight_candidate_gate():
    result = evaluate(
        {"specimen": "KIMI_BASE",
         "behavioural_direction": {"heldout_ready": True, "real": True},
         "crossfit_causal_intervention": {
             "operator_kind": "crossfit_residual_projection",
             "causal_effect_established": False,
             "causal_sample_sufficient": True,
             "crossfit_consistent": False,
         }},
        {"status": "CATALOG_ONLY", "source": "KIMI_BASE", "candidates": []},
        {"status": "LIVE_SCORED"},
        live_candidate_evidence={
            "status": "CANDIDATE_LIVE_EVIDENCE_INCOMPLETE",
            "qualified_candidates": [],
            "qualified_serving_path_candidates": [
                "KIMI_SERVING_CANDIDATE_HCLI_CONTRACT"
            ],
        },
    )
    assert result["gates"]["causal_behavior"] is False
    assert result["gates"]["candidate_live_evaluation"] is False
    assert result["evidence"]["serving_path_candidate_qualified"] == [
        "KIMI_SERVING_CANDIDATE_HCLI_CONTRACT"
    ]


def test_candidate_identity_blocks_cross_candidate_evidence_reuse():
    result = evaluate(
        {
            "specimen": "KIMI_BASE",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "behavioural_direction": {"heldout_ready": True, "real": True},
            "crossfit_causal_intervention": {
                "operator_kind": "crossfit_residual_projection",
                "causal_effect_established": True,
                "causal_sample_sufficient": True,
                "crossfit_consistent": True,
            },
        },
        {
            "status": "CATALOG_ONLY",
            "source": "KIMI_BASE",
            "candidates": [{"candidate_id": "KIMI_OPERATOR_CANDIDATE_OPA"}],
        },
        {
            "schema": "hawking.kimi.hcli_engine_live.v1",
            "status": "ENGINE_LIVE_SCORED",
            "summary": {
                "all_engine_calls_completed": True,
                "cases": 16,
                "engine_passed": 16,
                "engine_effective_refusals": 0,
                "engine_unsafe_demo_indicators": 0,
            },
        },
        evidence={
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "axes": {name: {"passed": True} for name in (
                "capability", "epistemic", "authorization", "physical",
                "destructive_controls",
            )},
        },
        candidate_evaluation={
            "status": "FUNCTIONAL_SIM_COMPLETE",
            "candidate_count": 7,
            "candidates": [{"functional_sim_passed": True}] * 7,
        },
        live_candidate_evidence={
            "status": "CANDIDATE_LIVE_EVIDENCE_COMPLETE",
            "qualified_candidates": [],
            "candidates": {},
        },
    )

    candidate = result["candidates"][0]
    assert candidate["promotable"] is False
    assert "candidate_live_evaluation" in candidate["missing_gates"]
    assert "candidate_causal_identity" in candidate["missing_gates"]
    assert "candidate_evidence_identity" in candidate["missing_gates"]


def test_completed_oph_closes_nearby_sweeps_without_opening_promotion():
    result = evaluate(
        {
            "status": "OPH_CAUSAL_NULL_OR_INCONSISTENT",
            "specimen": "KIMI_BASE",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPH",
            "candidate_causal_interventions": {
                "KIMI_OPERATOR_CANDIDATE_OPH": {
                    "causal_effect_established": False,
                    "causal_sample_sufficient": False,
                }
            },
        },
        {
            "status": "CATALOG_ONLY",
            "source": "KIMI_BASE",
            "candidates": [{"candidate_id": "KIMI_OPERATOR_CANDIDATE_OPH"}],
        },
        {"status": "LIVE_SCORED"},
    )

    assert result["status"] == "PROMOTION_REFUSED"
    assert result["evidence"]["refusal_intervention_family_status"] == "NEGATIVE_SCIENCE"
    assert "causal_behavior" in result["missing_gates"]
    assert any("materially different causal hypothesis" in row for row in result["next_actions"])
    assert not any("Obtain positive" in row for row in result["next_actions"])
