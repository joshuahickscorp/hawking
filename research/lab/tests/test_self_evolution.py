"""Tests for HCLI self-evolution admission pipeline (Bible §23)."""
from __future__ import annotations

import pytest

from lab.hcli.self_evolution import (
    MIN_CLASS_SAMPLES_FOR_BIAS,
    MIN_REPLAY_TRACES,
    AdmissionPipeline,
    AdmissionStage,
    EvolutionRefusal,
    GraveyardRefused,
    ProposalKind,
    SelfEvolutionEngine,
)


def _propose(pipe: AdmissionPipeline, **kw):
    defaults = dict(
        kind=ProposalKind.BEHAVIORAL_RULE,
        body={"rule": "prefer sealed receipts over narrative claims"},
        proposer="executor_30b",
        source_trajectory_id="traj-verified-001",
    )
    defaults.update(kw)
    return pipe.propose(**defaults)


def _pass_through_to_compat(pipe: AdmissionPipeline, pid: str) -> None:
    traces = [f"trace-{i}" for i in range(MIN_REPLAY_TRACES)]
    pipe.run_replay(
        pid,
        historical_trace_ids=traces,
        outcomes={t: "PASS" for t in traces},
    )
    pipe.run_hidden_validation(pid, hidden_bundle={"pass": True, "train_eval_disjoint": True})
    pipe.run_compat_test(pid, compat_report={"pass": True, "detail": "ok"})


def test_proposal_kinds_match_bible():
    expected = {
        "memory_update",
        "behavioral_rule",
        "skill_update",
        "tool_wrapper",
        "tool_retirement",
        "search_index_update",
        "new_benchmark",
        "new_anti_pattern",
    }
    assert {k.value for k in ProposalKind} == expected


def test_full_admission_happy_path_sealed():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    _pass_through_to_compat(pipe, p.proposal_id)
    assert p.stage is AdmissionStage.COMPAT_TEST
    doc = pipe.protected_admit(p.proposal_id, admitter="protected_controller")
    assert doc["verdict"] == "ACCEPT"
    assert "seal_sha256" in doc
    assert p.stage is AdmissionStage.ADMITTED
    assert doc["fabricated_accept"] is False


def test_proposer_cannot_admit_own_proposal():
    pipe = AdmissionPipeline()
    p = _propose(pipe, proposer="executor_30b")
    _pass_through_to_compat(pipe, p.proposal_id)
    with pytest.raises(EvolutionRefusal, match="tribunal separation"):
        pipe.protected_admit(p.proposal_id, admitter="executor_30b")


def test_single_trace_replay_is_pending_never_admit():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    result = pipe.run_replay(
        p.proposal_id,
        historical_trace_ids=["only-one"],
        outcomes={"only-one": "PASS"},
    )
    assert result.status == "PENDING"
    assert p.stage is AdmissionStage.PENDING
    # Cannot sneak through to admission.
    doc = pipe.protected_admit(p.proposal_id, admitter="controller")
    assert doc["verdict"] == "PENDING"
    assert doc["fabricated_accept"] is False


def test_missing_hidden_validation_is_honest_pending():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    traces = ["a", "b"]
    pipe.run_replay(p.proposal_id, historical_trace_ids=traces, outcomes={"a": "PASS", "b": "PASS"})
    result = pipe.run_hidden_validation(p.proposal_id, hidden_bundle=None)
    assert result.status == "PENDING"
    doc = pipe.protected_admit(p.proposal_id, admitter="controller")
    assert doc["verdict"] == "PENDING"


def test_replay_failure_buries_and_learns():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    pipe.run_replay(
        p.proposal_id,
        historical_trace_ids=["t0", "t1"],
        outcomes={"t0": "PASS", "t1": "FAIL"},
    )
    assert p.stage is AdmissionStage.BURIED
    assert p.verdict == "REJECT"
    buried = list(pipe.graveyard())
    assert len(buried) == 1
    assert buried[0].proposal_id == p.proposal_id
    # Ledger learned the rejection.
    outcomes = pipe.ledger.outcomes_for_kind(ProposalKind.BEHAVIORAL_RULE)
    assert any(r.kind == "buried" for r in outcomes)


def test_free_resurrection_refused():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    pipe.bury(p.proposal_id, reason="counterexample", actor="reviewer")
    with pytest.raises(GraveyardRefused, match="free resurrection|premise-changing"):
        pipe.revive(p.proposal_id, because="I changed my mind", actor="controller")
    assert p.stage is AdmissionStage.BURIED


def test_revive_requires_post_burial_premise_change():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    pipe.bury(p.proposal_id, reason="failed hidden eval", actor="controller")
    # Premise change *before* burial does not count — append after burial.
    row = pipe.ledger.append(
        "new_historical_trace_corpus",
        {"corpus_id": "hist-v2", "reason": "expanded traces"},
        actor="controller",
    )
    revived = pipe.revive(
        p.proposal_id,
        because="new corpus changes premises",
        actor="controller",
        premise_change_seq=row.seq,
    )
    assert revived.stage is AdmissionStage.PROPOSED
    assert revived.verdict is None
    assert revived.proposal_id not in {x.proposal_id for x in pipe.graveyard()}


def test_revive_refuses_pre_burial_premise_seq():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    # A ledger row that exists before burial cannot un-bury.
    early = pipe.ledger.append(
        "new_historical_trace_corpus",
        {"corpus_id": "too-early"},
        actor="controller",
    )
    pipe.bury(p.proposal_id, reason="fail", actor="controller")
    with pytest.raises(GraveyardRefused, match="strictly after burial"):
        pipe.revive(
            p.proposal_id,
            because="old evidence",
            actor="controller",
            premise_change_seq=early.seq,
        )


def test_never_bias_from_single_class_sample():
    engine = SelfEvolutionEngine()
    p = engine.propose_from_trajectory(
        kind=ProposalKind.TOOL_WRAPPER,
        body={"tool": "rg", "wrapper": "bounded"},
        proposer="executor",
        trajectory_id="t1",
    )
    _pass_through_to_compat(engine.pipeline, p.proposal_id)
    engine.pipeline.protected_admit(p.proposal_id, admitter="controller")
    summary = engine.learn_summary()
    assert summary["by_kind"]["tool_wrapper"]["total"] == 1
    assert summary["by_kind"]["tool_wrapper"]["may_bias_routing"] is False
    assert summary["min_class_samples_for_bias"] == MIN_CLASS_SAMPLES_FOR_BIAS


def test_bias_unlocked_after_enough_class_samples():
    engine = SelfEvolutionEngine()
    for i in range(MIN_CLASS_SAMPLES_FOR_BIAS):
        p = engine.propose_from_trajectory(
            kind=ProposalKind.NEW_ANTI_PATTERN,
            body={"pattern": f"ap-{i}"},
            proposer=f"exec-{i}",
            trajectory_id=f"traj-{i}",
        )
        if i % 2 == 0:
            _pass_through_to_compat(engine.pipeline, p.proposal_id)
            engine.pipeline.protected_admit(p.proposal_id, admitter="controller")
        else:
            engine.pipeline.bury(p.proposal_id, reason="not useful", actor="controller")
    summary = engine.learn_summary()
    row = summary["by_kind"]["new_anti_pattern"]
    assert row["total"] >= MIN_CLASS_SAMPLES_FOR_BIAS
    assert row["admitted"] + row["buried"] == row["total"]
    assert row["may_bias_routing"] is True


def test_proposals_never_overwritten():
    pipe = AdmissionPipeline()
    _propose(pipe, proposal_id="fixed-id")
    with pytest.raises(EvolutionRefusal, match="already exists"):
        _propose(pipe, proposal_id="fixed-id")


def test_evaluate_status_ready_for_admission():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    _pass_through_to_compat(pipe, p.proposal_id)
    status = pipe.evaluate_status(p.proposal_id)
    assert status["overall"] == "READY_FOR_PROTECTED_ADMISSION"
    assert status["fabricated_accept"] is False


def test_compat_failure_buries():
    pipe = AdmissionPipeline()
    p = _propose(pipe)
    traces = ["a", "b"]
    pipe.run_replay(p.proposal_id, historical_trace_ids=traces, outcomes={"a": "PASS", "b": "PASS"})
    pipe.run_hidden_validation(p.proposal_id, hidden_bundle={"pass": True})
    pipe.run_compat_test(p.proposal_id, compat_report={"pass": False, "detail": "breaks API"})
    assert p.stage is AdmissionStage.BURIED
