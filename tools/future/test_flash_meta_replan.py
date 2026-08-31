"""Tests for the Flash meta re-plan.

Negative controls (must actually fire):
  * a family whose mechanism is not the failing codec is UNTOUCHED
  * extrapolation refuses a rank when the series cannot support one
  * the coherence contract is read from the screen and is never assigned here
  * a passing screen does not falsify the codec
  * n_fit < rank is THIN under NS-014
A skipped test is a P0. Absent receipts must be a recorded refusal, not an omitted case.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools.future import flash_meta_replan as fmr
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _contract(**overrides):
    base = {
        "min_heldout_cosine": 999 / 1000,
        "max_heldout_relative_fro_error": 1 / 20,
        "must_beat_per_expert_q4": True,
        "fit_holdout_required": True,
    }
    base.update(overrides)
    return base


def _rank(
    rank: int,
    err: float,
    *,
    cos: float = 0.9,
    q4: float = 0.1,
    bpw: float | None = None,
    passed: bool = False,
    fit_rows: int = 204,
    heldout_rows: int = 52,
    first: str = "held-out function error",
):
    if bpw is None:
        bpw = rank * 0.006356837606837607
    return {
        "rank": rank,
        "fit_rows": fit_rows,
        "heldout_rows": heldout_rows,
        "fit_relative_fro_error": err * 0.9,
        "heldout_relative_fro_error": err,
        "heldout_cosine": cos,
        "per_expert_q4_heldout_relative_fro_error": q4,
        "beats_per_expert_q4_on_heldout": err < q4,
        "diagnostic_factor_equivalent_bpw": bpw,
        "surface_failure_gates": [] if passed else [first, "held-out function cosine"],
        "first_surface_failure": None if passed else first,
        "surface_gate_pass": passed,
    }


def _screen(
    *,
    rows: list[dict] | None = None,
    contract: dict | None = None,
    status: str = "OFFLINE_META_SURFACE_GATE_FAILED",
    kind: str = "shared_input_latent_plus_expert_local_output_readout",
    organ: str = "layer_4.routed_experts.gate_up_proj",
    next_gate: str | None = (
        "collect broader teacher traces, distill router/hidden/routed-output/"
        "terminal-logit surfaces, then build a serializer"
    ),
    fit_rows: int = 204,
    heldout_rows: int = 52,
):
    if rows is None:
        rows = [
            _rank(4, 0.5284),
            _rank(8, 0.4718),
            _rank(16, 0.4369),
            _rank(32, 0.4121),
            _rank(64, 0.3906),
        ]
    return {
        "schema": "hawking.flash.meta_coherence_screen.v1",
        "status": status,
        "representation": {"kind": kind, "organ": organ, "physical_ebpw": None},
        "coherence_contract": contract if contract is not None else _contract(),
        "teacher_trace": {"rows": 256, "width": 2560, "unique_row_hashes": 256},
        "surface": {"organ": "gate_up_proj", "rows": rows, "fit_rows": fit_rows, "heldout_rows": heldout_rows},
        "next_gate": next_gate,
        "measurement_state": {"promotion_allowed": False, "physical_ebpw": "NULL_BY_RULE"},
    }


def _sub1(families: list[dict] | None = None, q4_bpw: float = 4.25):
    if families is None:
        families = [
            {
                "family": "routed_experts",
                "program": "expert-local latent code + shared tile decoder + route-margin repair",
                "source_fraction": 0.685,
                "meta_bpw_target": 0.88,
            },
            {
                "family": "ngram_embedding",
                "program": "frequency-tiered n-gram generator + hot exact islands + residual symbols",
                "source_fraction": 0.284,
                "meta_bpw_target": 0.7,
            },
            {
                "family": "norm",
                "program": "exact norm island",
                "source_fraction": 1e-6,
                "meta_bpw_target": 16.0,
            },
        ]
    return {
        "schema": "hawking.flash.meta_representation.v1",
        "family_budget": families,
        "current_evidence": {
            "bounded_routed_q4_component_bpw": q4_bpw,
            "bounded_routed_q4_status": "PASSED",
        },
        "metric": {"prospective_target": 0.8871807728336929, "physical_ebpw": None},
    }


def _index(*original_ids: str):
    scars = []
    for oid in original_ids or ("NS-014",):
        scars.append(
            {
                "scar_id": f"receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json#{oid}",
                "original_id": oid,
                "source_path": "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
                "hypothesis_family": (
                    "fit_a_rank_r_or_full_dim_codec_on_fewer_captured_rows_"
                    "than_the_fitted_dimension_then_trust_the_score"
                ),
                "failure_mechanism": (
                    "Fit a rank-r or full-dim codec on fewer captured rows than "
                    "the fitted dimension, then trust the score"
                ),
                "reopen_condition": (
                    "never trust a score from an underdetermined fit. Re-score only "
                    "when n_fit >= the claimed rank (and, for a full-dim claim, "
                    "n_fit >= dim), with rank not clamped."
                ),
                "refuse_eligible": True,
            }
        )
    return {"schema": "hawking.future.negative_index.v1", "scars": scars}


def test_build_emits_sealed_receipt():
    out = fmr.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_META_REPLAN.json"
    assert doc["schema"] == fmr.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert len(doc["seal_sha256"]) == 64
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert doc["gate_stands"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_REPRESENTATION.meta-gates-3-9"
    # Copes either way: present screen yields a scoped judgement; absent yields REFUSED.
    assert doc["falsification"]["verdict"] in {
        "FALSIFIED_ON_STATED_SCOPE",
        "NOT_FALSIFIED_BY_THIS_RECEIPT",
        "REFUSED",
    }


def test_selftest_aliases_build():
    assert fmr.selftest is fmr.build or callable(fmr.selftest)
    assert fmr.selftest().name == "FLASH_META_REPLAN.json"


def test_absent_screen_is_refusal_not_pass():
    """NEGATIVE CONTROL: missing measurement is not a kill and not a pass."""
    result = fmr.falsification(None, None)
    assert result["verdict"] == "REFUSED"
    assert result["verdict"] != "FALSIFIED_ON_STATED_SCOPE"
    assert result["dead"] == []
    assert any("sub-1" in x for x in result["not_dead"])
    extra = fmr.rank_extrapolation(None, None)
    assert extra["verdict"] == "REFUSED"
    assert extra["rank_required"] is None
    under = fmr.underdetermination_check(n_fit=None, ranks=None, input_width=None)
    assert under["verdict"] == "REFUSED"
    plan = fmr.replan(screen=None, teacher=None, sub1=None, index_doc=None)
    assert plan["gate_stands"] is True
    assert plan["falsification"]["verdict"] == "REFUSED"
    assert plan["families"]["ok"] is False
    assert plan["next_capture"]["ok"] is False


def test_passing_screen_does_not_falsify():
    """NEGATIVE CONTROL: a rank that meets the contract is not a family kill."""
    rows = [
        _rank(4, 0.4, passed=False),
        _rank(8, 0.2, passed=False),
        _rank(16, 0.04, cos=0.9995, passed=True, first=""),
    ]
    result = fmr.falsification(_screen(rows=rows))
    assert result["verdict"] == "NOT_FALSIFIED_BY_THIS_RECEIPT"
    assert result["passing_ranks"] == [16]
    assert result["failed_ranks"] == [4, 8]
    assert result["dead"] == []


def test_missing_gate_flag_is_refusal_not_a_silent_fail():
    """NEGATIVE CONTROL: absent surface_gate_pass is not rounded into a fail."""
    rows = [_rank(4, 0.5), _rank(8, 0.4)]
    del rows[1]["surface_gate_pass"]
    result = fmr.falsification(_screen(rows=rows))
    assert result["verdict"] == "REFUSED"
    assert "surface_gate_pass" in result["reason"]
    assert result["dead"] == []


def test_falsification_scope_is_one_organ_one_codec():
    result = fmr.falsification(_screen())
    assert result["verdict"] == "FALSIFIED_ON_STATED_SCOPE"
    assert result["scope"]["kind"] == "shared_input_latent_plus_expert_local_output_readout"
    assert result["scope"]["organ"] == "layer_4.routed_experts.gate_up_proj"
    assert result["scope"]["fit_rows"] == 204
    assert result["scope"]["heldout_rows"] == 52
    assert result["scope"]["ranks"] == [4, 8, 16, 32, 64]
    assert len(result["dead"]) == 1
    assert result["dead"][0]["what"] == result["scope"]["kind"]
    blob = " ".join(result["not_dead"]).lower()
    assert "sub-1" in blob
    assert "other eight" in blob
    assert "down_proj" in blob


def test_untouched_family_is_not_marked_down():
    """NEGATIVE CONTROL: n-gram does not inherit the latent+readout failure."""
    plan = fmr.replan(
        screen=_screen(),
        teacher=None,
        sub1=_sub1(),
        index_doc=_index(),
    )
    by_name = {r["family"]: r for r in plan["families"]["families"]}
    assert by_name["ngram_embedding"]["evidence_effect"] == "UNTOUCHED"
    assert by_name["ngram_embedding"]["shares_failing_mechanism"] is False
    assert by_name["norm"]["evidence_effect"] == "UNTOUCHED"
    assert "ngram_embedding" in plan["families"]["untouched"]
    assert "norm" in plan["families"]["untouched"]
    assert plan["families"]["untouched"]


def test_failing_family_inherits_against():
    plan = fmr.replan(
        screen=_screen(),
        teacher=None,
        sub1=_sub1(),
        index_doc=_index(),
    )
    by_name = {r["family"]: r for r in plan["families"]["families"]}
    assert by_name["routed_experts"]["evidence_effect"] == "INHERITS_AGAINST"
    assert by_name["routed_experts"]["shares_failing_mechanism"] is True
    assert "routed_experts" in plan["families"]["inherits_against"]
    # Untouched sorts before inherited-against.
    names = [r["family"] for r in plan["families"]["families"]]
    assert names.index("ngram_embedding") < names.index("routed_experts")


def test_routed_experts_without_program_is_unresolved_not_a_markdown():
    """NEGATIVE CONTROL: name match alone is not mechanism match."""
    sub1 = _sub1(
        [
            {"family": "routed_experts", "program": "", "source_fraction": 0.6},
            {"family": "ngram_embedding", "program": "n-gram generator", "source_fraction": 0.3},
        ]
    )
    plan = fmr.replan(screen=_screen(), teacher=None, sub1=sub1, index_doc=_index())
    by_name = {r["family"]: r for r in plan["families"]["families"]}
    assert by_name["routed_experts"]["evidence_effect"] == "UNRESOLVED"
    assert by_name["routed_experts"]["shares_failing_mechanism"] is None
    assert by_name["ngram_embedding"]["evidence_effect"] == "UNTOUCHED"


def test_extrapolation_refuses_non_monotone_series():
    """NEGATIVE CONTROL: a curve nobody could trust reports no rank."""
    points = [
        _rank(4, 0.50),
        _rank(8, 0.20),
        _rank(16, 0.60),
        _rank(32, 0.30),
        _rank(64, 0.40),
    ]
    extra = fmr.rank_extrapolation(points, _contract())
    assert extra["verdict"] == "REFUSED"
    assert extra["rank_required"] is None
    assert extra["dominated_by_construction"] is None
    assert "monotone" in extra["reason"]


def test_extrapolation_refuses_when_floor_is_above_the_contract():
    """Five diminishing-return points with an obvious floor must not invent a rank."""
    # err ~ 0.35 + 0.4/sqrt(rank)  — floor 0.35 > contract 0.05
    points = [
        _rank(4, 0.35 + 0.4 / (4 ** 0.5)),
        _rank(8, 0.35 + 0.4 / (8 ** 0.5)),
        _rank(16, 0.35 + 0.4 / (16 ** 0.5)),
        _rank(32, 0.35 + 0.4 / (32 ** 0.5)),
        _rank(64, 0.35 + 0.4 / (64 ** 0.5)),
    ]
    extra = fmr.rank_extrapolation(points, _contract(), q4_bpw=4.25)
    assert extra["verdict"] == "REFUSED"
    assert extra["rank_required"] is None
    assert extra["implied_floor"] >= extra["target_heldout_error"]
    assert extra["dominated_by_construction"] is None


def test_extrapolation_reports_a_rank_when_the_curve_supports_one():
    """The refuse path has a positive complement: a clean 1/rank curve must report."""
    points = [
        _rank(4, 0.4 / 4, bpw=0.1 * 4),
        _rank(8, 0.4 / 8, bpw=0.1 * 8),
        _rank(16, 0.4 / 16, bpw=0.1 * 16),
        _rank(32, 0.4 / 32, bpw=0.1 * 32),
        _rank(64, 0.4 / 64, bpw=0.1 * 64),
    ]
    extra = fmr.rank_extrapolation(points, _contract(), q4_bpw=4.25)
    assert extra["verdict"] in {"EXTRAPOLATED", "OBSERVED_NOT_EXTRAPOLATED"}
    assert extra["rank_required"] is not None
    # 0.4 / r = 0.05  => r = 8
    assert abs(extra["rank_required"] - 8.0) < 0.2
    assert extra["diagnostic_bpw_at_rank"] is not None
    assert extra["dominated_by_construction"] is False


def test_extrapolation_flags_domination_when_bpw_exceeds_q4():
    points = [
        _rank(4, 0.8 / (4 ** 0.25), bpw=2.0 * 4),
        _rank(8, 0.8 / (8 ** 0.25), bpw=2.0 * 8),
        _rank(16, 0.8 / (16 ** 0.25), bpw=2.0 * 16),
        _rank(32, 0.8 / (32 ** 0.25), bpw=2.0 * 32),
        _rank(64, 0.8 / (64 ** 0.25), bpw=2.0 * 64),
    ]
    extra = fmr.rank_extrapolation(points, _contract(), q4_bpw=4.25)
    if extra["verdict"] == "EXTRAPOLATED":
        assert extra["diagnostic_bpw_at_rank"] is not None
        if extra["diagnostic_bpw_at_rank"] > 4.25:
            assert extra["dominated_by_construction"] is True
        else:
            assert extra["dominated_by_construction"] is False
    else:
        # A floor above the contract is also an honest refusal, not a guessed rank.
        assert extra["verdict"] == "REFUSED"
        assert extra["rank_required"] is None


def test_extrapolation_uses_the_screen_contract_not_a_local_gate():
    """NEGATIVE CONTROL: a looser contract must be the one that is tested."""
    points = [
        _rank(4, 0.40),
        _rank(8, 0.20),
        _rank(16, 0.10),
        _rank(32, 0.05),
        _rank(64, 0.025),
    ]
    loose = _contract(max_heldout_relative_fro_error=0.2)
    extra = fmr.rank_extrapolation(points, loose)
    assert extra["target_heldout_error"] == 0.2
    assert extra["rank_required"] is not None
    # Rank 8 already has err 0.20; that is observed, not a 0.05-gate result.
    assert extra["rank_required"] <= 8.0 + 1e-9


def test_module_source_does_not_redefine_the_coherence_contract():
    """NEGATIVE CONTROL: the gate is not silently lowered in this file."""
    src = Path(fmr.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = (999 / 1000, 1 / 20)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if any(abs(node.value - b) < 1e-12 for b in banned):
                hits.append((node.lineno, node.value))
    assert hits == [], f"coherence-contract literals assigned in module: {hits}"
    check = fmr._module_does_not_redefine_contract()
    assert check["ok"] is True
    # A fake screen with a different gate must be the gate that is used.
    fake = _screen(contract=_contract(min_heldout_cosine=0.7, max_heldout_relative_fro_error=0.2))
    lifted = fmr.contract_from_screen(fake)
    assert lifted["ok"] is True
    assert lifted["min_heldout_cosine"] == 0.7
    assert lifted["max_heldout_relative_fro_error"] == 0.2
    extra = fmr.rank_extrapolation(fake["surface"]["rows"], lifted)
    assert extra["target_heldout_error"] == 0.2


def test_replan_does_not_lower_the_contract():
    screen = _screen()
    plan = fmr.replan(screen=screen, teacher=None, sub1=_sub1(), index_doc=_index())
    assert plan["gate_stands"] is True
    assert plan["contract"]["ok"] is True
    assert plan["contract"]["min_heldout_cosine"] == screen["coherence_contract"]["min_heldout_cosine"]
    assert (
        plan["contract"]["max_heldout_relative_fro_error"]
        == screen["coherence_contract"]["max_heldout_relative_fro_error"]
    )
    assert plan["contract"]["must_beat_per_expert_q4"] is True
    assert plan["contract"]["min_heldout_cosine"] == 999 / 1000
    assert plan["contract"]["max_heldout_relative_fro_error"] == 1 / 20


def test_underdetermination_negative_thin_rank():
    """NEGATIVE CONTROL: NS-014 must be observed rejecting a thin rank."""
    result = fmr.underdetermination_check(
        n_fit=10,
        ranks=[4, 8, 16, 32],
        input_width=2560,
        index_doc=_index("NS-014"),
        claimed_full_dim=False,
    )
    assert result["verdict"] in {"THIN", "MIXED"}
    by_rank = {r["rank"]: r for r in result["ranks"]}
    assert by_rank[4]["trustworthy"] is True
    assert by_rank[8]["trustworthy"] is True
    assert by_rank[16]["thin"] is True
    assert by_rank[32]["thin"] is True
    assert 16 in result["thin_ranks"]
    assert 4 in result["trustworthy_ranks"]
    # Mixed must not be rounded into all-trustworthy.
    assert result["verdict"] != "TRUSTWORTHY"


def test_underdetermination_trustworthy_when_n_fit_ge_rank():
    result = fmr.underdetermination_check(
        n_fit=204,
        ranks=[4, 8, 16, 32, 64],
        input_width=2560,
        index_doc=_index("NS-014", "NNS-007"),
        claimed_full_dim=False,
    )
    assert result["verdict"] == "TRUSTWORTHY"
    assert result["thin_ranks"] == []
    assert result["trustworthy_ranks"] == [4, 8, 16, 32, 64]
    assert result["index_consulted"] is True
    assert any(s.get("original_id") == "NS-014" for s in result["scars_cited"])


def test_underdetermination_full_dim_uses_width_not_rank():
    """204 rows vs width 2560 is thin for a full-dim claim, even if rank-r would pass."""
    result = fmr.underdetermination_check(
        n_fit=204,
        ranks=[64],
        input_width=2560,
        index_doc=_index("NS-014"),
        claimed_full_dim=True,
    )
    assert result["verdict"] == "THIN"
    assert result["ranks"][0]["thin"] is True
    assert "input_width" in result["ranks"][0]["criterion"]


def test_underdetermination_clamped_rank_is_thin():
    result = fmr.underdetermination_check(
        n_fit=204,
        ranks=[64],
        input_width=2560,
        index_doc=_index("NS-014"),
        rank_clamped_to_n_fit=True,
    )
    assert result["verdict"] == "THIN"
    assert result["ranks"][0]["trustworthy"] is False


def test_named_next_surfaces_extracted_or_refused():
    ok = fmr.named_next_surfaces(_screen())
    assert ok["ok"] is True
    assert ok["surfaces"] == ["router", "hidden", "routed-output", "terminal-logit"]
    missing = fmr.named_next_surfaces(_screen(next_gate="fit more ranks of gate_up_proj"))
    assert missing["ok"] is False
    assert missing["surfaces"] == []
    assert fmr.named_next_surfaces(None)["ok"] is False


def test_contract_from_screen_refuses_partial_contract():
    screen = _screen()
    del screen["coherence_contract"]["min_heldout_cosine"]
    lifted = fmr.contract_from_screen(screen)
    assert lifted["ok"] is False
    assert lifted["min_heldout_cosine"] is None


def test_real_receipt_copes_and_keeps_the_gate_when_screen_is_present():
    """If the real-256 screen is loadable, the receipt must keep its contract."""
    doc = json.loads(fmr.build().read_text())
    screen, rel = fmr.load_named(fmr.SCREEN_REL)
    if screen is None:
        assert doc["falsification"]["verdict"] == "REFUSED"
        assert doc["contract"]["ok"] is False
        return
    assert rel == fmr.SCREEN_REL
    cc = screen["coherence_contract"]
    assert doc["contract"]["ok"] is True
    assert doc["contract"]["min_heldout_cosine"] == cc["min_heldout_cosine"]
    assert doc["contract"]["max_heldout_relative_fro_error"] == cc["max_heldout_relative_fro_error"]
    assert doc["contract"]["must_beat_per_expert_q4"] is True
    assert doc["gate_stands"] is True
    assert doc["falsification"]["verdict"] == "FALSIFIED_ON_STATED_SCOPE"
    assert doc["falsification"]["scope"]["kind"] == screen["representation"]["kind"]
    assert doc["falsification"]["scope"]["organ"] == screen["representation"]["organ"]
    assert doc["at_least_one_family_untouched"] is True
    assert "ngram_embedding" in doc["untouched_families"]
    extra = doc["rank_extrapolation"]
    # Five diminishing-return points: a supported floor sits above the gate, so
    # a rank is refused rather than invented. A reported rank is only legal if
    # the curve actually reached the contract.
    if extra["verdict"] == "REFUSED":
        assert extra["rank_required"] is None
    else:
        assert extra["rank_required"] is not None
    under = doc["underdetermination"]
    assert under["verdict"] in {"TRUSTWORTHY", "THIN", "MIXED", "REFUSED"}
    if under["verdict"] == "TRUSTWORTHY":
        assert under["thin_ranks"] == []
    capture = doc["next_capture"]
    if capture["ok"]:
        assert "router" in capture["surfaces"]
        assert "terminal-logit" in capture["surfaces"]
        spend = (capture.get("spend") or "").lower()
        assert "never been measured" in spend
        assert "not to another rank sweep" in spend
        assert any("lower" in x for x in capture.get("do_not") or [])
    else:
        assert capture["surfaces"] == []


def test_corpus_comparison_needs_two_corpora_to_call_a_trend(monkeypatch):
    """One screen is not a direction of travel.

    The 256-row failure had an obvious objection -- a 204/52 split is thin. The
    answer was a 4x corpus, and the comparison IS the finding. A module that
    reports a single screen as though it showed a trend would have thrown that
    away.
    """
    out = fmr.corpus_comparison()
    assert out["verdict"] in {"OVERFIT_ON_THE_SMALLER_CORPUS", "STABLE_ACROSS_CORPORA",
                              "INCOMPARABLE"}
    if out["verdict"] == "INCOMPARABLE":
        assert "not a trend" in out["why"]
        return
    assert set(out["corpora"]) == {"256", "1024"}
    assert out["still_bounded"], "the comparison must state what it does not cover"


def test_more_data_made_this_family_worse_and_left_the_baseline_alone():
    """The signature of overfitting, measured rather than argued."""
    out = fmr.corpus_comparison()
    if out["verdict"] == "INCOMPARABLE":
        return
    deltas = out["heldout_error_delta_by_rank"]
    assert deltas, "no per-rank comparison was produced"
    assert all(v > 0 for v in deltas.values()), (
        "every rank must degrade for the overfit verdict to stand"
    )
    # The comparator is what makes the degradation meaningful: if Q4 had moved
    # too, a harder held-out set would explain both.
    assert abs(out["q4_baseline_shift"]) < 0.01, (
        "the Q4 baseline moved materially; the degradation is not attributable "
        "to the codec alone"
    )
    assert out["every_rank_degraded_with_more_data"] is True
    assert out["verdict"] == "OVERFIT_ON_THE_SMALLER_CORPUS"


def test_neither_corpus_lets_this_family_reach_the_gate():
    out = fmr.corpus_comparison()
    if out["verdict"] == "INCOMPARABLE":
        return
    for name, row in out["corpora"].items():
        if not row.get("present"):
            continue
        assert row["any_rank_passed"] is False, f"corpus {name} reported a passing rank"
        ext = row["extrapolation"]
        if ext.get("verdict") == "EXTRAPOLATED":
            # A rank the curve technically reaches is only meaningful if it is
            # affordable. 5.9e18 is a refutation written as a number.
            assert ext.get("dominated_by_construction") is True, (
                "an affordable rank was claimed to reach the gate"
            )
