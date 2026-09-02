"""Tests for the generic capability-evaluation interface.

Two genuinely different evaluators must register and run through one
registry. A live-resident source must be refused. An import is not a call
site — evaluate() is.
"""
from __future__ import annotations

import json

import pytest

from tools.future import capability_eval as ce
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_interface_admits_required_domains_and_subject_kinds():
    reg = ce.default_registry()
    for domain in (
        "reasoning",
        "coding",
        "tool_use",
        "mission_autonomy",
        "domain_capability",
    ):
        assert reg.admits_domain(domain)
    for kind in ("resident", "nx", "representation", "heterogeneous"):
        assert reg.admits_subject_kind(kind)
    with pytest.raises(ce.EvalRefused):
        ce.EvalSubject(kind="mmlu-only", identity="x")


def test_two_genuinely_different_evaluators_register():
    reg = ce.default_registry()
    ids = set(reg.list_evaluators())
    assert "reasoning.predicate" in ids
    assert "coding.execute" in ids
    assert len(ids) >= 2
    domains = {reg._evaluators[i].domain for i in ids}
    assert "reasoning" in domains
    assert "coding" in domains
    assert len(domains) >= 2


def test_no_single_hardcoded_benchmark():
    assert ce.default_registry().list_evaluators() != ("mmlu",)
    assert getattr(ce, "DEFAULT_BENCHMARK", None) is None
    src = open(ce.__file__, encoding="utf-8").read()
    assert "MMLU" not in src
    assert "GSM8K" not in src
    assert "HumanEval" not in src


def test_run_calls_evaluate_not_just_import(monkeypatch):
    reg = ce.default_registry()
    called: list[str] = []
    real_reason = reg._evaluators["reasoning.predicate"].evaluate
    real_code = reg._evaluators["coding.execute"].evaluate

    def wrap_reason(subject, source):
        called.append("reasoning.predicate")
        return real_reason(subject, source)

    def wrap_code(subject, source):
        called.append("coding.execute")
        return real_code(subject, source)

    monkeypatch.setattr(reg._evaluators["reasoning.predicate"], "evaluate", wrap_reason)
    monkeypatch.setattr(reg._evaluators["coding.execute"], "evaluate", wrap_code)
    subject = ce.EvalSubject(kind="representation", identity="fixture")
    results = reg.run_all(subject, ce.passing_scripted_source())
    assert called == ["reasoning.predicate", "coding.execute"]
    assert results["reasoning.predicate"]["passed"] == results["reasoning.predicate"]["total"]
    assert results["coding.execute"]["passed"] == results["coding.execute"]["total"]


def test_reasoning_and_coding_both_run_through_one_interface():
    reg = ce.default_registry()
    subject = ce.EvalSubject(kind="nx", identity="fixture-nx")
    results = reg.run_all(subject, ce.passing_scripted_source())
    r = results["reasoning.predicate"]
    c = results["coding.execute"]
    assert r["evaluator_id"] == "reasoning.predicate"
    assert c["evaluator_id"] == "coding.execute"
    assert r["domain"] == "reasoning"
    assert c["domain"] == "coding"
    assert r["scoring"] != c["scoring"]
    assert r["hardcoded_benchmark"] is None
    assert c["hardcoded_benchmark"] is None
    assert r["passed"] == 2
    assert c["passed"] == 2
    assert r["subject_kind"] == "nx"
    assert c["subject_kind"] == "nx"


def test_failing_source_is_rejected_by_both_evaluators():
    reg = ce.default_registry()
    subject = ce.EvalSubject(kind="heterogeneous", identity="fixture-fail")
    results = reg.run_all(subject, ce.failing_scripted_source())
    assert results["reasoning.predicate"]["passed"] == 0
    assert results["coding.execute"]["passed"] == 0


def test_coding_execute_actually_calls_the_function():
    """Mutation-shaped: a function that ignores inputs must fail the cases."""
    ev = ce.CodingExecuteEvaluator()
    item = ev.items()[0]
    passed, detail = ev._score(item, "```python\ndef dedupe(xs):\n    return xs\n```")
    assert passed is False
    assert "expect" in detail or "case" in detail


def test_live_resident_source_is_refused():
    reg = ce.default_registry()
    with pytest.raises(ce.LiveResidentRefused):
        reg.run(
            "reasoning.predicate",
            ce.EvalSubject(kind="resident", identity="live"),
            ce.LiveResidentSource(),
        )


def test_build_runs_both_evaluators_and_seals_receipt():
    out = ce.build()
    assert out.parent == RECEIPTS
    assert out.name == "CAPABILITY_EVAL.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.capability_eval.v1"
    assert doc["hardcoded_benchmark"] is None
    assert doc["live_resident_refusal_fired"] is True
    ids = {e["id"] for e in doc["registered_evaluators"]}
    assert "reasoning.predicate" in ids
    assert "coding.execute" in ids
    assert len(doc["runs"]) >= 2
    kinds = {r["subject_kind"] for r in doc["runs"]}
    assert "representation" in kinds
    assert "nx" in kinds
    for run in doc["runs"]:
        results = run["results"]
        assert results["reasoning.predicate"]["passed"] == 2
        assert results["coding.execute"]["passed"] == 2
    fail = doc["negative_control_failing_source"]
    assert fail["reasoning.predicate"]["passed"] == 0
    assert fail["coding.execute"]["passed"] == 0
    _assert_no_hardware_claims(doc)


def test_representation_family_hook_scores_the_same_axes():
    """CALL SITE: score_representation_family. An import is not a score."""
    axes = {k: 1 if k not in {"execute_match", "chain_complete", "is_sub2_executable",
                               "reconstructs_dense_parent", "consumes_representation_directly"}
            else True
            for k in ce.REPRESENTATION_AXES}
    axes["complete_ebpw"] = 4.0
    axes["stored_bytes"] = 100
    axes["stored_bpw"] = 4.0
    axes["billed_ms"] = 0.1
    axes["executable_bytes"] = 100
    axes["is_sub2_executable"] = False
    axes["reconstructs_dense_parent"] = False
    axes["consumes_representation_directly"] = True
    axes["execute_match"] = True
    axes["chain_complete"] = True
    incumbent = dict(axes)
    candidate = dict(axes)
    candidate["complete_ebpw"] = 3.0
    candidate["stored_bytes"] = 80
    candidate["stored_bpw"] = 3.2
    candidate["billed_ms"] = 0.05
    candidate["executable_bytes"] = 80
    # CALL SITE of the hook, not an import.
    score = ce.score_representation_family(
        candidate_id="cand",
        candidate=candidate,
        incumbent_id="inc",
        incumbent=incumbent,
    )
    assert score["same_axes_as_incumbent"] is True
    assert score["subject_kind"] == "representation"
    assert score["axes"] == list(ce.REPRESENTATION_AXES)
    assert score["per_axis"]["complete_ebpw"]["candidate_smaller"] is True
    assert score["hardcoded_benchmark"] is None


def test_representation_family_hook_refuses_a_partial_axis_set():
    cand = {k: 0 for k in ce.REPRESENTATION_AXES}
    inc = {k: 0 for k in ce.REPRESENTATION_AXES}
    del cand["billed_ms"]
    with pytest.raises(ce.EvalRefused, match="missing"):
        ce.score_representation_family(
            candidate_id="cand",
            candidate=cand,
            incumbent_id="inc",
            incumbent=inc,
        )


def test_registry_refuses_unknown_evaluator():
    reg = ce.default_registry()
    with pytest.raises(ce.EvalRefused, match="no evaluator"):
        reg.run(
            "mmlu",
            ce.EvalSubject(kind="resident", identity="x"),
            ce.passing_scripted_source(),
        )
