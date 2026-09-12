from tools.future.kimi_operator_live_evidence import build


def test_unmeasured_candidates_are_not_qualified():
    result = build({})
    assert result["status"] == "CANDIDATE_LIVE_EVIDENCE_INCOMPLETE"
    assert result["qualified_candidates"] == []
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPA"]["status"] == "NOT_MEASURED"


def test_oph_device_busy_is_deferred_not_scientific_evidence():
    result = build(
        {
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPH",
            "candidate_method": "OPH",
            "status": "OPH_DEVICE_BUSY",
            "device_preflight": {"owner": "hawkingd", "owner_pid": 21611},
            "candidate_causal_interventions": {
                "KIMI_OPERATOR_CANDIDATE_OPH": {
                    "status": "OPH_DEVICE_BUSY",
                    "causal_effect_established": False,
                    "causal_sample_sufficient": False,
                }
            },
            "claim_boundary": "No model was loaded.",
        },
        source_path="oph-device-busy.json",
    )

    row = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPH"]
    assert row["status"] == "DEFERRED_DEVICE_BUSY"
    assert row["causal_effect_established"] is False
    assert row["source_receipt"] == "oph-device-busy.json"


def test_completed_oph_null_supersedes_prior_deferred_receipts():
    result = build(
        {
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPH",
            "candidate_method": "OPH",
            "status": "OPH_DEVICE_BUSY",
            "claim_boundary": "No model was loaded.",
        },
        source_path="oph-device-busy.json",
        supplementary_sources={
            "oph-router-live.json": {
                "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPH",
                "candidate_method": "OPH",
                "status": "OPH_CAUSAL_NULL_OR_INCONSISTENT",
                "candidate_causal_interventions": {
                    "KIMI_OPERATOR_CANDIDATE_OPH": {
                        "status": "INSUFFICIENT_BEHAVIOR_ROWS",
                        "operator_kind": "crossfit_router_logit_projection",
                        "causal_effect_established": False,
                        "causal_sample_sufficient": False,
                    }
                },
            }
        },
    )

    row = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPH"]
    assert row["status"] == "REJECTED_CAUSAL_NULL"
    assert row["measurement_status"] == "INSUFFICIENT_BEHAVIOR_ROWS"
    assert row["source_receipt"] == "oph-router-live.json"
    assert result["qualified_candidates"] == []


def test_positive_serving_contract_is_separate_from_weight_candidates():
    result = build(
        {},
        source_path="locality.json",
        serving_hcli_candidate={
            "candidate_id": "KIMI_SERVING_CANDIDATE_HCLI_CONTRACT",
            "status": "SERVING_CANDIDATE_HCLI_EFFECT_ESTABLISHED",
            "paired_report": {
                "causal_effect_established": True,
                "causal_sample_sufficient": True,
            },
        },
    )
    assert result["qualified_candidates"] == []
    assert result["qualified_serving_path_candidates"] == [
        "KIMI_SERVING_CANDIDATE_HCLI_CONTRACT"
    ]


def test_selected_expert_null_is_bound_to_opf_only():
    result = build({
        "moe_causal_intervention": {
            "causal_effect_established": False,
            "causal_sample_sufficient": False,
            "causal_sample_counts": {"refused": 2, "complied": 3},
            "refusal_rate_delta": 0.0,
            "layer": 26,
            "expert_ids": [23],
            "include_shared": False,
        }
    })
    opf = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPF"]
    assert opf["status"] == "REJECTED_CAUSAL_NULL"
    assert opf["causal_effect_established"] is False
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPA"]["status"] == "NOT_MEASURED"


def test_shared_selected_expert_evidence_binds_to_opg():
    result = build({
        "moe_causal_intervention": {
            "include_shared": True,
            "causal_effect_established": False,
            "causal_sample_sufficient": True,
            "layer": 26,
            "expert_ids": [23],
        }
    }, source_path="shared-receipt.json")
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPF"]["status"] == "NOT_MEASURED"
    row = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPG"]
    assert row["status"] == "REJECTED_CAUSAL_NULL"
    assert row["source_receipt"] == "shared-receipt.json"


def test_candidate_label_wins_over_generic_moe_shorthand_in_same_receipt():
    result = build({
        "candidate_causal_interventions": {
            "KIMI_OPERATOR_CANDIDATE_OPG": {
                "candidate_method": "OPG",
                "candidate_scope": "peak_layer_shared_plus_selected_experts",
                "operator_kind": "shared_plus_selected_expert_weight_projection",
                "causal_effect_established": False,
                "causal_sample_sufficient": True,
            }
        },
        "moe_causal_intervention": {
            "include_shared": False,
            "causal_effect_established": False,
            "causal_sample_sufficient": True,
        },
    }, source_path="opg-receipt.json")
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPG"]["method"] == "OPG"
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPG"]["source_receipt"] == "opg-receipt.json"
    assert result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPF"]["status"] == "NOT_MEASURED"


def test_candidate_labelled_crossfit_evidence_binds_to_ope():
    result = build({
        "candidate_causal_interventions": {
            "KIMI_OPERATOR_CANDIDATE_OPE": {
                "candidate_method": "OPE",
                "candidate_scope": "single_selected_layer_shared_writer",
                "operator_kind": "crossfit_layer_local_writer_projection",
                "causal_effect_established": True,
                "causal_sample_sufficient": True,
                "crossfit_consistent": False,
                "positive_fold_count": 1,
                "layer": 26,
            }
        }
    }, source_path="ope-receipt.json")
    ope = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPE"]
    assert ope["status"] == "REJECTED_CROSSFIT_INCONSISTENT"
    assert ope["source_receipt"] == "ope-receipt.json"
    assert result["qualified_candidates"] == []


def test_candidate_hcli_regression_is_retained_as_negative_evidence():
    result = build(
        {
            "candidate_causal_interventions": {
                "KIMI_OPERATOR_CANDIDATE_OPE": {
                    "candidate_method": "OPE",
                    "operator_kind": "crossfit_layer_local_writer_projection",
                    "causal_effect_established": True,
                    "causal_sample_sufficient": True,
                    "crossfit_consistent": True,
                }
            }
        },
        source_path="ope-receipt.json",
        candidate_hcli={
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "status": "CANDIDATE_HCLI_NULL_OR_INCONSISTENT",
            "hcli_contract_intervention": {
                "causal_effect_established": False,
                "causal_sample_sufficient": True,
                "baseline_passed": 5,
                "treated_passed": 4,
                "null_passed": 5,
                "pass_rate_delta": -0.0625,
                "null_pass_rate_delta": 0.0,
            },
        },
    )
    ope = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPE"]
    assert ope["hcli_contract"]["treated_passed"] == 4
    assert result["qualified_candidates"] == []


def test_nested_translation_negative_experiment_is_bound_when_supplied():
    result = build(
        {},
        experimental_hcli_translation={
            "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_TRANSLATION",
            "status": "EXPERIMENTAL_HCLI_FIT_TRANSLATION_NEGATIVE",
            "report": {
                "selection_scope": "nested_outer_train_only",
                "causal_effect_established": False,
            },
        },
    )
    bound = result["experimental_hcli_evaluations"]["residual_translation_nested_selection"]
    assert bound["status"] == "EXPERIMENTAL_HCLI_FIT_TRANSLATION_NEGATIVE"
    assert bound["report"]["selection_scope"] == "nested_outer_train_only"


def test_nested_writer_negative_experiment_is_bound_when_supplied():
    result = build(
        {},
        experimental_hcli_nested={
            "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER",
            "status": "EXPERIMENTAL_HCLI_FIT_WRITER_NEGATIVE",
            "report": {
                "selection_scope": "nested_outer_train_only",
                "causal_effect_established": False,
            },
        },
    )
    bound = result["experimental_hcli_evaluations"]["writer_projection_nested_selection"]
    assert bound["status"] == "EXPERIMENTAL_HCLI_FIT_WRITER_NEGATIVE"
    assert bound["report"]["selection_scope"] == "nested_outer_train_only"


def test_prefill_translation_negative_experiment_is_bound_when_supplied():
    result = build(
        {},
        experimental_hcli_prefill={
            "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_PREFILL_TRANSLATION",
            "status": "EXPERIMENTAL_HCLI_FIT_TRANSLATION_NEGATIVE",
            "prefill_only": True,
            "report": {
                "causal_effect_established": False,
                "prefill_only": True,
            },
        },
    )
    bound = result["experimental_hcli_evaluations"]["prefill_translation"]
    assert bound["prefill_only"] is True
    assert bound["report"]["causal_effect_established"] is False


def test_writer_output_translation_negative_experiment_is_bound_when_supplied():
    result = build(
        {},
        experimental_writer_translation_prefill={
            "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_WRITER_TRANSLATION",
            "status": "EXPERIMENTAL_HCLI_FIT_WRITER_NEGATIVE",
            "prefill_only": True,
            "report": {
                "operator_kind": "crossfit_hcli_contract_writer_output_translation_prefill",
                "causal_effect_established": False,
            },
        },
    )
    bound = result["experimental_hcli_evaluations"]["writer_output_translation_prefill"]
    assert bound["prefill_only"] is True
    assert bound["report"]["operator_kind"].endswith("_prefill")


def test_conditional_translation_negative_experiment_is_bound_when_supplied():
    result = build(
        {},
        experimental_conditional_translation={
            "candidate_id": "KIMI_EXPERIMENTAL_HCLI_FIT_CONDITIONAL_TRANSLATION",
            "status": "EXPERIMENTAL_HCLI_FIT_TRANSLATION_NEGATIVE",
            "conditional_pass": True,
            "report": {
                "causal_effect_established": False,
                "operator_kind": "crossfit_hcli_contract_conditional_residual_translation_prefill",
            },
        },
    )
    bound = result["experimental_hcli_evaluations"]["conditional_translation_prefill"]
    assert bound["conditional_pass"] is True
    assert bound["report"]["causal_effect_established"] is False


def test_candidate_epistemic_parity_is_bound_to_candidate_without_qualifying_it_alone():
    result = build(
        {
            "candidate_causal_interventions": {
                "KIMI_OPERATOR_CANDIDATE_OPE": {
                    "candidate_method": "OPE",
                    "operator_kind": "crossfit_layer_local_writer_projection",
                    "causal_effect_established": False,
                    "causal_sample_sufficient": True,
                    "crossfit_consistent": False,
                }
            }
        },
        source_path="ope-receipt.json",
        candidate_epistemic={
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "status": "CANDIDATE_EPISTEMIC_PARITY_PASSED",
            "parity": {
                "no_regression": True,
                "baseline_passed": 7,
                "treated_passed": 7,
                "null_passed": 6,
                "brier_delta": 0.0,
                "ece_delta": 0.0,
            },
        },
    )
    ope = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPE"]
    assert ope["epistemic_parity"]["no_regression"] is True
    assert result["qualified_candidates"] == []


def test_candidate_physical_parity_is_bound_to_candidate_without_qualifying_it_alone():
    result = build(
        {
            "candidate_causal_interventions": {
                "KIMI_OPERATOR_CANDIDATE_OPE": {
                    "candidate_method": "OPE",
                    "operator_kind": "crossfit_layer_local_writer_projection",
                    "causal_effect_established": False,
                    "causal_sample_sufficient": True,
                    "crossfit_consistent": False,
                }
            }
        },
        source_path="ope-receipt.json",
        candidate_physical={
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "status": "CANDIDATE_PHYSICAL_PARITY_PASSED",
            "parity": {
                "no_physical_regression": True,
                "treated_to_baseline_ratio": 1.0003,
                "null_to_baseline_ratio": 1.0009,
                "same_task_count": True,
            },
        },
    )
    ope = result["candidates"]["KIMI_OPERATOR_CANDIDATE_OPE"]
    assert ope["physical_parity"]["no_physical_regression"] is True
    assert result["qualified_candidates"] == []
