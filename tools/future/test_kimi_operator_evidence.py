from tools.future.kimi_operator_evidence import build


def test_evidence_envelope_keeps_unqualified_axes_explicit():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "FAST AND NOT YET QUALIFIED"}},
        hcli={"status": "LIVE_INCOMPLETE", "summary": {"all_passed": False}},
        locality={"schema": "test"},
    )
    assert result["status"] == "EVIDENCE_INCOMPLETE"
    assert result["summary"]["passed"] == 0
    assert set(result["summary"]["missing"]) == {
        "capability", "epistemic", "authorization", "physical", "destructive_controls"
    }


def test_only_explicit_qualified_receipts_close_axes():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "QUALIFIED"}},
        hcli={"status": "LIVE_SCORED", "summary": {"all_passed": True}},
        locality={"schema": "test"},
    )
    assert result["axes"]["capability"]["passed"] is True
    assert result["axes"]["authorization"]["passed"] is True
    assert result["axes"]["epistemic"]["passed"] is False
    assert result["axes"]["physical"]["passed"] is False
    assert result["axes"]["destructive_controls"]["passed"] is False


def test_independent_hcli_controls_can_close_only_destructive_axis():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "FAST AND NOT YET QUALIFIED"}},
        hcli={"status": "LIVE_INCOMPLETE", "summary": {"all_passed": False}},
        locality={"schema": "test"},
        controls={
            "schema": "hawking.kimi.hcli.boundary_controls.v1",
            "status": "CONTROLS_PASSED",
            "summary": {"all_passed": True},
        },
    )
    assert result["axes"]["destructive_controls"]["passed"] is True
    assert result["axes"]["authorization"]["passed"] is False


def test_base_epistemic_baseline_does_not_close_candidate_axis():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "FAST AND NOT YET QUALIFIED"}},
        hcli={"status": "LIVE_INCOMPLETE", "summary": {"all_passed": False}},
        locality={"schema": "test"},
        epistemic={
            "schema": "hawking.kimi.epistemic_live.v1",
            "status": "EPISTEMIC_LIVE_SCORED",
            "summary": {"qualified": True},
        },
    )
    assert result["axes"]["epistemic"]["baseline_passed"] is True
    assert result["axes"]["epistemic"]["passed"] is False


def test_resident_engine_contract_closes_authorization_without_candidate_parity():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "FAST AND NOT YET QUALIFIED"}},
        hcli={"status": "LIVE_INCOMPLETE", "summary": {"all_passed": False}},
        locality={"schema": "test"},
        engine_hcli={
            "schema": "hawking.kimi.hcli_engine_live.v1",
            "status": "ENGINE_LIVE_SCORED",
            "summary": {
                "all_engine_calls_completed": True,
                "cases": 16,
                "engine_passed": 16,
                "engine_effective_refusals": 0,
            },
        },
    )
    assert result["axes"]["authorization"]["passed"] is True
    assert result["axes"]["epistemic"]["passed"] is False
    assert result["summary"]["passed"] == 1


def test_ope_candidate_parity_closes_only_epistemic_and_physical_axes():
    result = build(
        g019={"K2_CAPABILITY_FIRST_RESULT": {"verdict": "FAST AND NOT YET QUALIFIED"}},
        hcli={"status": "LIVE_INCOMPLETE", "summary": {"all_passed": False}},
        locality={"schema": "test"},
        candidate_epistemic={
            "schema": "hawking.kimi.candidate_epistemic_live.v1",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "status": "CANDIDATE_EPISTEMIC_PARITY_PASSED",
            "parity": {"no_regression": True},
        },
        candidate_physical={
            "schema": "hawking.kimi.candidate_physical_live.v1",
            "candidate_id": "KIMI_OPERATOR_CANDIDATE_OPE",
            "status": "CANDIDATE_PHYSICAL_PARITY_PASSED",
            "parity": {"no_physical_regression": True, "same_task_count": True},
        },
    )
    assert result["axes"]["epistemic"]["passed"] is True
    assert result["axes"]["physical"]["passed"] is True
    assert result["candidate_id"] == "KIMI_OPERATOR_CANDIDATE_OPE"
    assert result["candidate_evidence_scope"] == ["epistemic", "physical"]
    assert result["summary"]["passed"] == 2
