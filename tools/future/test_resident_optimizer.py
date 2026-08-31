"""Tests for the bounded resident optimizer. Promotion is impossible here."""
from __future__ import annotations

import json

import pytest

from hcli.workunit import WorkUnit
from tools.future import resident_optimizer as ro
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_and_selftest_emit_sealed_receipt():
    out = ro.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "RESIDENT_OPTIMIZER.json"
    assert doc["schema"] == "hawking.future.resident_optimizer.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["built"] is True
    _assert_no_hardware_claims(doc)


def test_generate_emits_bounded_hypotheses_with_evidence_parents():
    opt = ro.ResidentOptimizer()
    hyps = opt.generate(seed=0)
    assert hyps
    assert len(hyps) <= opt.bound().max_hypotheses
    spent = sum(int(h["cost_units"]) for h in hyps)
    assert spent <= opt.bound().max_total_cost_units
    kinds = {h["kind"] for h in hyps}
    assert kinds <= set(ro.KINDS)
    for hyp in hyps:
        assert hyp["evidence_parents"]
        assert hyp["status"] == "PROPOSED"
        assert hyp["verified"] is False
        assert hyp["evidence_class"] == "STATIC_ONLY"
        assert hyp["bench_state"] == "UNKNOWN"
        assert hyp["era"] in ro.ERA_NUMERALS
        assert hyp["era"] != "VI"
        assert hyp["may_promote"] is False
        assert hyp["delegation"]["reimplemented"] is False
        assert hyp["verifier"] not in ro.WEAK_VERIFIERS
        ro.validate_delegated_body(hyp["kind"], hyp["delegated_body"])


def test_ranking_is_expected_information_per_unit_cost():
    rows = [
        {
            "id": "cheap-high",
            "kind": "compiler-pass",
            "expected_information": 3,
            "cost_units": 1,
            "statement": "a",
            "evidence_parents": ["x"],
            "delegated_body": {},
        },
        {
            "id": "expensive-high",
            "kind": "compiler-pass",
            "expected_information": 3,
            "cost_units": 6,
            "statement": "b",
            "evidence_parents": ["x"],
            "delegated_body": {},
        },
        {
            "id": "cheap-low",
            "kind": "compiler-pass",
            "expected_information": 1,
            "cost_units": 1,
            "statement": "c",
            "evidence_parents": ["x"],
            "delegated_body": {},
        },
    ]
    ranked = ro.rank_hypotheses(rows)
    assert [r["id"] for r in ranked] == ["cheap-high", "cheap-low", "expensive-high"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["information_per_cost"]["rule"].startswith("rank by expected_information")


def test_bound_clips_after_rank():
    tight = ro.ResidentOptimizer(ro.OptimizerBound(max_hypotheses=2, max_total_cost_units=24))
    hyps = tight.generate(seed=0)
    assert len(hyps) == 2
    full = ro.rank_hypotheses(ro.candidate_catalog(tight.bound()))
    assert [h["id"] for h in hyps] == [h["id"] for h in full[:2]]


def test_kinds_are_delegated_not_reimplemented():
    opt = ro.ResidentOptimizer()
    hyps = opt.generate(seed=0)
    by_kind = {k: [h for h in hyps if h["kind"] == k] for k in ro.KINDS}
    # Default bound of 9 admits the whole 3+3+3 catalog.
    assert by_kind["compiler-pass"]
    assert by_kind["transfer"]
    assert by_kind["hardware-profile"]
    for kind, rows in by_kind.items():
        owner = ro.KIND_OWNERS[kind]
        for hyp in rows:
            assert hyp["delegation"]["owner_module"] == owner["owner_module"]
            assert hyp["delegation"]["owner_schema"] == owner["owner_schema"]
            assert hyp["workunit_species"] == owner["workunit_species"]
            assert hyp["verifier"] == owner["verifier"]
            for field in owner["required_fields"]:
                assert field in hyp["delegated_body"]
    lpc = by_kind["compiler-pass"][0]["delegated_body"]
    assert lpc["contamination_class"] == "STATIC_ONLY"
    assert lpc["latency"] is None
    assert lpc["resident_bytes"] is None
    transfer = by_kind["transfer"][0]["delegated_body"]
    assert transfer["time_to_first_useful_executable_ns"] is None
    assert transfer["scope"] != "GENERIC_VERIFIED"
    for cand in transfer["transfer_candidates"]:
        assert cand.get("promotion_requested") is False
    hw = by_kind["hardware-profile"][0]["delegated_body"]
    assert hw["predicted_effect"]["magnitude_class"] in {
        "UNKNOWN",
        "SUB_PERCENT",
        "SINGLE_DIGIT_FRACTION",
        "FACTOR",
    }
    assert not isinstance(hw["predicted_effect"].get("magnitude"), (int, float))


def test_empty_evidence_parents_are_refused():
    with pytest.raises(ro.BoundViolation, match="evidence parents"):
        ro.make_hypothesis(
            id="RO-BAD",
            kind="compiler-pass",
            statement="x",
            evidence_parents=(),
            expected_information=2,
            cost_units=1,
            delegated_body=ro._lpc_body(ro.COMPILER_PASS_SEEDS[0]),
        )


def test_workunit_economy_uses_hcli_constructor_and_budgets():
    opt = ro.ResidentOptimizer()
    hyps = opt.generate(seed=0)
    eco = opt.economy(hyps)
    assert eco["count"] == len(hyps)
    assert eco["budget"]["attempts"] == 3
    assert eco["budget"]["max_repair_depth"] == 3
    assert eco["budget"]["max_repairs_per_root"] == 6
    assert eco["budget"]["gpu_windows_held"] == 0
    for stop in eco["stop_conditions"]:
        assert not ro._stop_grants_authority(stop)
    for row in eco["work_units"]:
        roundtrip = WorkUnit.from_dict(row)
        assert roundtrip.id == row["id"]
        assert roundtrip.verifier
        assert row["may_promote"] is False
        assert row["may_modify_verifier"] is False
        assert row["classification"] == "STATIC_ONLY"
        assert row["effect_class"] == "READ_ONLY"


def test_recovery_contract_points_at_live_gates():
    contract = ro.recovery_contract()
    assert contract["runs_recovery"] is False
    assert contract["resident_py"] is None
    owners = contract["owners"]
    assert owners["recovery_gate"] == "hcli/agentos/recovery.py"
    assert owners["resident_gate"] == "hcli/agentos/resident_gate.py"
    assert "run_resident_kill" in owners["autonomy_a3_resident_kill"]
    assert "run_process_kill" in owners["autonomy_a4_process_kill"]
    assert "run_idempotency_crash" in owners["autonomy_a5_idempotency"]
    assert "the proposer is never the admitter" in contract["self_evolution"]["recovered_principle"]


def test_proposer_and_verifier_are_distinct_without_shared_mutable_state():
    opt = ro.ResidentOptimizer()
    assert opt.proposer is not opt.verifier
    assert type(opt.proposer) is not type(opt.verifier)
    assert opt.shared_mutable_state() is False
    assert opt.proposer.store_id() != opt.verifier.store_id()
    hyps = opt.generate(seed=0)
    before_emitted = [h["id"] for h in opt.proposer.emitted()]
    verdict = opt.verifier.inspect(hyps[0])
    assert verdict["verdict"] in {"STRUCTURALLY_SOUND", "STRUCTURALLY_UNSOUND"}
    assert verdict["promotion"] == "IMPOSSIBLE_FROM_THIS_LANE"
    assert verdict["settles_physical_claim"] is False
    assert [h["id"] for h in opt.proposer.emitted()] == before_emitted
    assert hyps[0]["verified"] is False
    # Writing a verdict must not flip the proposer's copy.
    opt.verifier.record_verdict(hyps[0]["id"], "UNSETTLED")
    assert all(h["verified"] is False for h in opt.proposer.emitted())


def test_negative_control_proposer_cannot_mark_own_proposal_verified():
    opt = ro.ResidentOptimizer()
    hyps = opt.generate(seed=0)
    with pytest.raises(ro.VerifierSeparationError, match="cannot mark its own proposal verified"):
        opt.proposer.mark_verified(hyps[0]["id"])
    with pytest.raises(ro.VerifierSeparationError, match="cannot mark its own proposal verified"):
        opt.mark_verified(hyps[0]["id"])


def test_negative_control_proposer_cannot_weaken_a_verifier():
    opt = ro.ResidentOptimizer()
    with pytest.raises(ro.VerifierSeparationError, match="cannot weaken a verifier"):
        opt.proposer.weaken_verifier("self")
    with pytest.raises(ro.VerifierSeparationError, match="cannot weaken a verifier"):
        opt.weaken_verifier("none")
    with pytest.raises(ro.VerifierSeparationError, match="cannot weaken itself"):
        opt.verifier.weaken()


def test_negative_control_proposer_cannot_widen_own_authority():
    opt = ro.ResidentOptimizer()
    with pytest.raises(ro.VerifierSeparationError, match="cannot widen its own authority"):
        opt.proposer.widen_authority("self_promotion")
    with pytest.raises(ro.VerifierSeparationError, match="cannot widen its own authority"):
        opt.widen_authority("promote_self")
    with pytest.raises(ro.VerifierSeparationError, match="cannot assign '_authority'"):
        opt.proposer._authority = frozenset({"self_promotion"})


def test_separation_guard_is_watched_failing():
    results = ro._prove_verifier_separation()
    assert len(results) == 3
    assert all(row["refused"] is True for row in results)
    trials = {row["trial"] for row in results}
    assert trials == {"mark_verified", "weaken_verifier", "widen_authority"}


def test_promote_does_not_exist():
    opt = ro.ResidentOptimizer()
    assert not hasattr(ro.ResidentOptimizer, "promote")
    assert not hasattr(ro.Proposer, "promote")
    assert "promote" not in vars(ro.ResidentOptimizer)
    with pytest.raises(AttributeError):
        opt.promote()  # type: ignore[attr-defined]


def test_bound_refuses_self_promotion_flags():
    with pytest.raises(ro.BoundViolation, match="cannot grant promotion"):
        ro.OptimizerBound(may_promote=True)
    with pytest.raises(ro.BoundViolation, match="cannot grant promotion"):
        ro.OptimizerBound(may_modify_verifier=True)
    with pytest.raises(ro.BoundViolation, match="cannot grant promotion"):
        ro.OptimizerBound(may_widen_authority=True)
    with pytest.raises(ro.BoundViolation, match="forbidden authority"):
        ro.OptimizerBound(allowed_authority=frozenset({"read_receipts", "self_promotion"}))
    with pytest.raises(ro.BoundViolation, match="Era VI"):
        ro.OptimizerBound(era="VI")
    with pytest.raises(ro.BoundViolation, match="Odyssey IV"):
        ro.OptimizerBound(odyssey="IV")
    with pytest.raises(ro.BoundViolation, match="GPU window"):
        ro.OptimizerBound(gpu_windows_held=1)


def test_sidecar_verifier_cannot_record_a_promotion_class():
    v = ro.IsolatedVerifier()
    with pytest.raises(ro.VerifierSeparationError, match="cannot record"):
        v.record_verdict("RO-CP-001", "VERIFIED")
    with pytest.raises(ro.VerifierSeparationError, match="cannot record"):
        v.record_verdict("RO-CP-001", "PROTECTED_ABSOLUTE")
    with pytest.raises(ro.VerifierSeparationError, match="cannot record"):
        v.record_verdict("RO-CP-001", "PROMOTED")


def test_compiler_pass_cannot_claim_protected_lpc_row():
    body = ro._lpc_body(ro.COMPILER_PASS_SEEDS[0])
    body["contamination_class"] = "PROTECTED_ABSOLUTE"
    with pytest.raises(ro.DelegationError, match="PROTECTED_ABSOLUTE"):
        ro.validate_delegated_body("compiler-pass", body)
    body["contamination_class"] = "DIAGNOSTIC_RELATIVE"
    with pytest.raises(ro.DelegationError, match="DIAGNOSTIC_RELATIVE"):
        ro.validate_delegated_body("compiler-pass", body)


def test_vocabulary_has_five_eras_and_three_odysseys():
    assert len(ro.ERAS) == 5
    assert len(ro.ODYSSEYS) == 3
    assert "VI" not in "".join(ro.ERAS)
    assert "IV" not in "".join(ro.ODYSSEYS)
    opt = ro.ResidentOptimizer()
    blob = json.dumps(list(opt.generate(seed=0)))
    assert "Era VI" not in blob
    assert "Odyssey IV" not in blob


def test_receipt_records_built_not_promoted_and_recovered_paths():
    doc = json.loads(ro.build().read_text())
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promotion"]["possible_from_this_lane"] is False
    assert doc["promotion"]["emits"] == "PROPOSAL"
    assert doc["verifier_separation"]["promote_exists_on_optimizer"] is False
    assert doc["verifier_separation"]["shared_mutable_state"] is False
    recovered = {item["path"]: item for item in doc["recovered_implementation"]}
    assert recovered["hcli/agentos/resident.py"]["present"] is False
    assert recovered["hcli/agentos/resident_gate.py"]["present"] is True
    assert recovered["hcli/verifier_pipeline.py"]["present"] is True
    assert recovered["hcli/agentos/recovery.py"]["present"] is True
    assert "FPGA" in doc["vocabulary"]["fpga_is"]
    assert "not a civilization" in doc["vocabulary"]["fpga_is"]
