#!/usr/bin/env python3.12
"""Adversarial tests for the Hawking Gravity law (master goal section 19).

These prove that Gravity is ENFORCED by code, not documentation: Hawking starts below
one BPW when Gravity applies, cannot finalize above one BPW without a sealed Escape
Receipt, and rejects every named non-justification. The search-policy tests replay the
synthetic outcomes whose expected Event Horizon is 0.50 (not 1.00), and the 11 invariants
proven by the sealed diagnostic sim are folded in.
"""
import json
import pathlib
import sys
from fractions import Fraction

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_gravity as g          # noqa: E402
import succ_gravity_policy as gp  # noqa: E402
import succ_gravity_receipts as gr  # noqa: E402
import succ_queue as q            # noqa: E402
import succ_telegram as tg        # noqa: E402
from eco_common import sealed, atomic_write_json, read_json_safe  # noqa: E402


# -- module selftests --------------------------------------------------------------------
def test_policy_selftest():
    assert gp.selftest()["ok"] is True


def test_receipts_selftest():
    assert gr.selftest()["ok"] is True


def test_engine_selftest():
    assert g.selftest()["ok"] is True


# -- 1. Hawking starts below one BPW when Gravity applies --------------------------------
def test_starts_below_one_bpw():
    for label in gp.PARENT_PRIORS:
        ss = gp.compute_stress_start(label)
        rate = gp.rate_from_identity(ss["chosen_stress_rate"])
        assert gp.is_subbit(rate), (label, rate)
        state = g.new_parent_state(label)
        assert gp.is_subbit(gp.rate_from_identity(state["initial_stress_rate"])), label


# -- 2. Cannot finalize above one BPW without an Escape Receipt --------------------------
def test_cannot_finalize_above_one_bpw_without_receipt():
    ok, why = gp.can_finalize_extreme(whole_bpw=1.5, coverage={"satisfied": True},
                                      escape_receipt=None, escape_receipt_valid=False)
    assert not ok and any("Escape Receipt" in r for r in why)
    good = gr._fully_justified_example()
    ok2, _ = gp.can_finalize_extreme(whole_bpw=1.5, coverage={"satisfied": True},
                                     escape_receipt=good, escape_receipt_valid=True)
    assert ok2


# -- 3. A missing result cannot authorize escape ----------------------------------------
def test_missing_experiment_not_justification():
    base = gr._fully_justified_example()
    thin = {k: v for k, v in base.items() if k != "receipt_sha256"}
    thin["attempted_subbit_rates"] = []
    from eco_common import seal_field
    thin = seal_field(thin, "receipt_sha256")
    ok, why = gr.escape_receipt_valid(thin)
    assert not ok and any("missing experiment" in r for r in why)


# -- 4. A scheduling deferral cannot authorize escape -----------------------------------
def test_scheduler_deferral_not_justification():
    base = gr._fully_justified_example()
    from eco_common import seal_field
    d = {k: v for k, v in base.items() if k != "receipt_sha256"}
    d["diagnosis_by_rate"] = {"1/2": "deferred", "1/3": "deferred", "1/4": "deferred"}
    d = seal_field(d, "receipt_sha256")
    ok, why = gr.escape_receipt_valid(d)
    assert not ok and any("deferral" in r for r in why)


# -- 5. One failed scalar codec cannot authorize escape ---------------------------------
def test_failed_scalar_codec_not_justification():
    base = gr._fully_justified_example()
    from eco_common import seal_field
    s = {k: v for k, v in base.items() if k != "receipt_sha256"}
    s["attempted_representation_families"] = ["scalar_trellis_tqv2"]
    s["evidence_level_by_family"] = {"scalar_trellis_tqv2": "F2"}
    s = seal_field(s, "receipt_sha256")
    ok, why = gr.escape_receipt_valid(s)
    assert not ok and any("scalar codec" in r for r in why)


# -- 6. Unsupported Doctor treatments cannot be selected --------------------------------
def test_unsupported_treatment_never_selected():
    assert gp.select_treatment(["lora_kd", "blockwise_qat", "strand_hessian"], degraded=True) is None
    assert gp.treatment_reachable("lora_kd") is False
    assert gp.treatment_reachable("gptoss_codec_control") is False
    # gpt-oss family has no executable Doctor module yet
    assert gp.treatment_reachable("doctor_static", architecture_family="gpt-oss-moe") is False
    assert gp.select_treatment(["doctor_full", "doctor_static"], degraded=True) == "doctor_full"


# -- 7. Doctor bytes count toward total BPW ---------------------------------------------
def test_doctor_bytes_counted_in_whole():
    p = g._sim_giant_moe()
    treated = p.whole_bpw(Fraction(1, 3), treated=True)
    untreated = p.whole_bpw(Fraction(1, 3), treated=False)
    assert treated >= untreated + p.doctor_reserve
    # and the conservation-law sum counts doctor explicitly
    whole = gp.whole_artifact_bpw({"base": 0.5, "doctor": 0.4, "packaging": 0.15})
    assert abs(whole - 1.05) < 1e-9


# -- 8. Exact rational rates preserve identity ------------------------------------------
def test_exact_rational_identity():
    assert Fraction(1, 10) + Fraction(1, 5) == Fraction(3, 10)
    assert 0.1 + 0.2 != 0.3                      # the float trap exists...
    for r in gp.RATE_LADDER:                     # ...the ladder round-trips exactly
        assert gp.rate_from_identity(gp.rate_identity(r)) == r
    assert gp.parse_rate("11/20") == Fraction(11, 20)
    with pytest.raises(gp.GravityError):
        gp.parse_rate("0.33")                    # a rounded decimal is not an identity


# -- 9. The scheduler tries another representation before ascending when required --------
def test_tries_other_representation_before_ascending():
    p = g._sim_synthetic()   # two material families; 0.33 mixed-fails on both
    s = g.InvertedSearch(g.HeavyLock(None))
    s.evaluate_rate(p, Fraction(1, 3), Fraction(1))
    families_at_third = {r.family for r in s.log if r.rate == Fraction(1, 3)}
    assert len(families_at_third) >= 2, families_at_third   # tried a 2nd representation
    assert g.gp.count_material_families(families_at_third) >= 2


# -- 10. Computation collapse can trigger ascent ----------------------------------------
def test_collapse_triggers_ascent():
    p = g._sim_synthetic()
    s = g.InvertedSearch(g.HeavyLock(None))
    out = s.search(p, start=Fraction(1, 4), contract_max_whole=Fraction(1))
    kinds = [t["outcome"] for t in out["trajectory"]]
    assert "fail_collapse" in kinds
    assert out["passing_rate"] is not None
    assert gp.parse_rate(out["passing_rate"]) > Fraction(1, 4)   # climbed above the collapse


# -- 11. A pass triggers descent --------------------------------------------------------
def test_pass_triggers_descent():
    p = g._sim_giant_moe()
    s = g.InvertedSearch(g.HeavyLock(None))
    out = s.search(p, start=Fraction(11, 20), contract_max_whole=Fraction(1))
    rates = [gp.parse_rate(t["rate"]) for t in out["trajectory"]]
    # after the first pass the search moves to a strictly lower rate at some point
    assert any(rates[i + 1] < rates[i] for i in range(len(rates) - 1))
    assert out["evidenced_floor"]


# -- 12. A physical-impossibility receipt can authorize ascent --------------------------
def test_physical_impossibility_authorizes_ascent():
    pr = gr.physical_impossibility_receipt(
        parent_identity={"label": "72B"}, exact_revision="pinned",
        target_region="< 0.20 bpw whole", min_whole_bpw=0.24, envelope_bpw=0.20,
        byte_budget={c: 0.02 for c in gr._PHYS_BYTE_CATEGORIES})
    state = {"receipts": {"physical_impossibility": pr}, "representation_families_tested": []}
    cov = gp.subbit_coverage_status(state)
    assert cov["satisfied"] and cov["route"] == "C"
    ok, _ = gp.can_finalize_extreme(whole_bpw=1.25, coverage=cov,
                                    escape_receipt=gr._fully_justified_example(),
                                    escape_receipt_valid=True)
    assert ok


# -- 13. Gravity state survives crash and resume ----------------------------------------
def test_state_survives_crash_resume(tmp_path):
    state = g.new_parent_state("72B")
    state = g.advance_state(state, "GRAVITY_DIAGNOSTIC")
    state = g.advance_state(state, "GRAVITY_F0", experiment_id="exp-a")
    path = tmp_path / "gravity_state.json"
    atomic_write_json(path, state)
    reloaded = read_json_safe(path)
    ok, why = g.verify_state_chain(reloaded)
    assert ok, why
    assert reloaded["fsm_state"] == "GRAVITY_F0"
    # resume continues the chain from the reloaded state
    resumed = g.advance_state(reloaded, "GRAVITY_F1")
    ok2, _ = g.verify_state_chain(resumed)
    assert ok2 and resumed["fsm_state"] == "GRAVITY_F1"


# -- 14. Duplicate heavy launch is refused ----------------------------------------------
def test_duplicate_heavy_launch_refused():
    state = g.new_parent_state("72B")
    state = g.advance_state(state, "GRAVITY_DIAGNOSTIC")
    state = g.advance_state(state, "GRAVITY_F0", experiment_id="exp-dup")
    state = g.advance_state(state, "GRAVITY_F1")
    with pytest.raises(g.GravityEngineError):
        g.advance_state(state, "GRAVITY_F0", experiment_id="exp-dup")


# -- 15. Gravity cannot create a second controller --------------------------------------
def test_cannot_become_second_heavy_controller():
    lock = g.HeavyLock(held_by="doctor_v5_disk25_successor")
    s = g.InvertedSearch(lock, lane_id="hawking-gravity")
    s.probe(g._sim_giant_moe(), Fraction(1, 2), "scalar_trellis_tqv2", Fraction(1))
    assert lock.held_by == "doctor_v5_disk25_successor"   # lane never seized the lock
    progs = g.materialize_live_parent_programs("72B", source_manifest_sha256="a" * 64)
    ok, why = g.program_launchable(progs["subbit_stress"], policy=gp.default_policy(),
                                   heavy_lock=lock, env={"HAWKING_GRAVITY_ENABLED": "1"},
                                   admission_passed=True)
    assert not ok and any("second heavy controller" in r for r in why)


# -- 16. Queue state and Telegram survive restart ---------------------------------------
def test_queue_and_telegram_survive_restart(tmp_path):
    state = g.new_parent_state("72B")
    row = q.make_row(parent_label="72B", current_status="planned", architecture_family="qwen2.5-dense")
    aug = g.augment_row(row, state)
    path = tmp_path / "queue.json"
    from eco_common import seal_field
    atomic_write_json(path, seal_field({"schema": q.QUEUE_SCHEMA, "rows": {"72B": aug}}, "queue_sha256"))
    doc = read_json_safe(path)
    assert sealed(doc["rows"]["72B"], "row_sha256")
    assert doc["rows"]["72B"]["gravity"]["gravity_state"] == "GRAVITY_UNINITIALIZED"
    # telegram dedup key is process-stable, so a restart replay does not double-send
    a = tg.compose_event("gravity_policy_enabled", {"parent": "72B", "policy_version": "2026-07-17.1"})
    b = tg.compose_event("gravity_policy_enabled", {"parent": "72B", "policy_version": "2026-07-17.1"})
    assert a["event_id"] == b["event_id"]


# -- 17. The same parent cannot receive conflicting Gravity identities -------------------
def test_no_conflicting_parent_identities():
    a = g.new_parent_state("72B")
    b = g.new_parent_state("72B", initial_stress_rate=Fraction(1, 2))   # different stress rate
    with pytest.raises(g.GravityEngineError):
        g.assert_no_identity_conflict([a, b])
    # identical identities do not conflict
    assert g.assert_no_identity_conflict([g.new_parent_state("72B"), g.new_parent_state("72B")])


# -- 18. Escape Receipts are tamper-detectable ------------------------------------------
def test_escape_receipt_tamper_detected():
    good = gr._fully_justified_example()
    assert gr.escape_receipt_valid(good)[0]
    tampered = dict(good)
    tampered["parameter_count"] = 1
    assert not gr.escape_receipt_valid(tampered)[0]
    # every receipt type is tamper-detectable
    sr = gr.structural_incompatibility_receipt(
        parent_identity={"label": "1T"}, exact_revision="x", component="vision.attn",
        attempted_representation="binary_pattern_codebooks", incompatibility="no factorization",
        evidence_level="F2", alternatives_considered=["additive_codebooks"], reopening_condition="x")
    sr_bad = dict(sr); sr_bad["component"] = "y"
    assert gr.verify_receipt(sr)[0] and not gr.verify_receipt(sr_bad)[0]


# -- 19. F0-F2 evidence cannot masquerade as F4 -----------------------------------------
def test_f0_f2_cannot_masquerade_as_f4():
    state = g.new_parent_state("72B")
    state = {**{k: v for k, v in state.items() if k != "state_sha256"},
             "current_evidence_level": "F2"}
    from eco_common import seal_field
    state = seal_field(state, "state_sha256")
    ok, why = g.can_finalize_event_horizon(state)
    assert not ok and any("F4" in r for r in why)
    # only replicated F4 finalizes
    state_f4 = {**{k: v for k, v in state.items() if k != "state_sha256"},
                "current_evidence_level": "F4"}
    state_f4 = seal_field(state_f4, "state_sha256")
    assert g.can_finalize_event_horizon(state_f4)[0]


# -- 20. Whole-artifact BPW is authoritative --------------------------------------------
def test_whole_artifact_bpw_authoritative():
    # 0.50 base is sub-bit, but 0.50 + 0.40 doctor + 0.15 overhead = 1.05 is NOT
    whole = gp.whole_artifact_bpw({"base": 0.50, "doctor": 0.40, "packaging": 0.15})
    assert not gp.is_subbit_artifact(whole)
    # a lower BODY rate with heavy healing does not beat a smaller complete artifact
    A = g.ExperimentResult("P", Fraction(1, 10), "scalar_trellis_tqv2", "doctor_full",
                           g.Signal.DEGRADED, g.Outcome.PASS, Fraction(1, 10), Fraction(9, 10))
    B = g.ExperimentResult("P", Fraction(4, 5), "scalar_trellis_tqv2", None,
                           g.Signal.OK, g.Outcome.PASS, Fraction(4, 5), Fraction(17, 20))
    assert g.better_artifact(A, B) is B          # 0.85 whole < 0.90 whole
    assert A.body_bpw < B.body_bpw               # ...even though A's body is far lower


# -- synthetic search: expected Event Horizon is 0.50, not 1.00 -------------------------
def test_synthetic_event_horizon_is_half():
    p = g._sim_synthetic()
    s = g.InvertedSearch(g.HeavyLock("doctor_v5_disk25_successor"))
    out = s.search(p, start=Fraction(1, 4), contract_max_whole=Fraction(1))
    assert out["passing_rate"] == "1/2"                  # Event Horizon = 0.50
    assert out["lower_boundary"] == "1/3"                # evidenced fail directly below
    assert out["evidenced_floor"]
    assert gp.parse_rate(out["passing_rate"]) != Fraction(1, 1)   # NOT 1.00


# -- gravitational acquisition changes priority only, never quality ---------------------
def test_gravity_bonus_pulls_toward_subbit():
    st = g.new_parent_state("72B")
    subbit = {"model_label": "72B", "rate": "1/2", "family": "binary_latent_factors",
              "can_change_extreme": True}
    above = {"model_label": "72B", "rate": "3/2", "family": "scalar_trellis_tqv2"}
    assert g.gravity_bonus(subbit, st) > g.gravity_bonus(above, st)
    ranked = g.rank_candidates([above, subbit], {"72B": st})
    assert ranked[0]["rate"] == "1/2"     # sub-bit candidate prioritized


def test_physically_impossible_candidate_penalized():
    st = g.new_parent_state("685B")
    ok_probe = {"model_label": "685B", "rate": "11/20", "family": "scalar_trellis_tqv2"}
    dead = {"model_label": "685B", "rate": "11/20", "family": "scalar_trellis_tqv2",
            "physically_impossible": True}
    assert g.gravity_bonus(dead, st) < g.gravity_bonus(ok_probe, st)


# -- real parents defer sub-bit today (no packer below 1.34), never a false collapse -----
def test_real_parent_defers_subbit_no_false_floor():
    p = g.parent_from_prior("72B")
    s = g.InvertedSearch(g.HeavyLock(None))
    r = s.probe(p, Fraction(1, 2), "scalar_trellis_tqv2", Fraction(8))
    assert r.outcome is g.Outcome.DEFERRED         # no wired sub-bit packer -> deferral
    assert r.outcome is not g.Outcome.FAIL_COLLAPSE
    out = s.search(p, start=Fraction(1, 2), contract_max_whole=Fraction(8))
    assert out["passing_rate"] is None             # honest: no false floor claimed
    assert not out["evidenced_floor"]


# -- policy manifest, state doc, validation doc all seal --------------------------------
def test_deliverables_seal():
    assert sealed(gp.build_policy_manifest(), "policy_sha256")
    doc = g.build_state_doc(list(gp.PARENT_PRIORS), live_parent="72B", source_manifest_sha256="a" * 64)
    assert sealed(doc, "state_doc_sha256")
    assert doc["gravity_enabled"] is False
    assert not doc["source_bound_programs"]["subbit_stress"]["is_subbit"] is None
    val = g.build_validation_doc()
    assert sealed(val, "validation_sha256") and val["all_selftests_ok"]


# -- integration: selector binding applies G(x) as priority-only ------------------------
def test_selector_binding_applies_gravity_priority():
    import succ_engine as eng
    states = {"72B": g.new_parent_state("72B")}
    plan = {"parents": [{"binding": {"model_label": "72B"}, "params_b": 72.7,
                         "event_horizon_bracket": {"lowest_pass_bpw": None},
                         "boundary_probes_needed": [
                             {"rate_bpw": 0.8, "current_verdict": "INCONCLUSIVE",
                              "next_feasibility_tier": "F1", "doctor_program": {"promote": []}},
                             {"rate_bpw": 4.0, "current_verdict": "INCONCLUSIVE",
                              "next_feasibility_tier": "F3_full_model_quality", "doctor_program": {"promote": []}},
                         ]}]}
    plain = eng.next_experiment(plan)
    bonus = g.gravity_bonus_binding(states, gp.build_policy_manifest())
    governed = eng.next_experiment(plan, gravity_bonus=bonus)
    # Gravity pulls the sub-bit (0.8) probe to the top; the plain selector need not
    assert governed["selected"]["rate_bpw"] == 0.8
    assert "gravity_bonus" in governed["selected"]
    # and it never touches the quality verdict
    assert governed["selected"]["verdict"] == "INCONCLUSIVE"


# -- integration: EXTREME finalization gate refuses unjustified >1 BPW EXTREME -----------
def test_finalize_extreme_gate_refuses_unjustified():
    rows = [
        {"model_label": "72B", "tier": "EXTREME", "whole_bpw": 1.5},   # above 1, no receipt
        {"model_label": "685B", "tier": "EXTREME", "whole_bpw": 0.72},  # sub-bit, allowed
    ]
    states = {"72B": g.new_parent_state("72B"), "685B": g.new_parent_state("685B")}
    out = g.finalize_extreme_gate(rows, states, receipts={})
    refused_labels = {r["label"] for r in out["refused"]}
    allowed_labels = {r.get("model_label") for r in out["allowed"]}
    assert "72B" in refused_labels          # >1 BPW EXTREME without coverage+receipt: refused
    assert "685B" in allowed_labels         # sub-bit EXTREME: allowed
    # with coverage + a valid receipt, the >1 BPW row is allowed
    states["72B"]["representation_families_tested"] = [
        {"family": "binary_latent_factors", "evidence_level": "F2"},
        {"family": "additive_codebooks", "evidence_level": "F2"}]
    receipts = {"72B": {"escape_receipt": gr._fully_justified_example()}}
    out2 = g.finalize_extreme_gate(rows, states, receipts=receipts)
    assert "72B" not in {r["label"] for r in out2["refused"]}


# -- integration: Gravity notifications compose in the approved terse style --------------
def test_gravity_notifications_compose():
    composed = g.notify_gravity("gravity_event_horizon",
                                {"parent": "72B", "rate": "4/5", "whole_bpw": 0.83}, dry_run=True)
    assert composed["text"].startswith("🕳 Hawking successor: Event Horizon")
    assert "Provisional until the signed physical release gate." in composed["text"]
    # unknown kind fails closed
    with pytest.raises(g.GravityEngineError):
        g.notify_gravity("not_a_kind", {}, dry_run=True)
    # injectable emit is used for real sends
    seen = {}
    def fake_emit(kind, ctx):
        seen["kind"] = kind
        return {"status": "sent"}
    r = g.notify_gravity("gravity_ascend", {"parent": "72B", "from_rate": "1/2", "to_rate": "4/5"},
                         emit=fake_emit, dry_run=False)
    assert seen["kind"] == "gravity_ascend" and r["status"] == "sent"


# -- integration: Gravity daily summary carries the required fields ----------------------
def test_gravity_daily_summary_fields():
    states = {label: g.new_parent_state(label) for label in gp.PARENT_PRIORS}
    summ = g.gravity_daily_summary(states, digest_date="2026-07-17")
    assert summ["digest_date"] == "2026-07-17"
    p72 = next(p for p in summ["parents"] if p["parent"] == "72B")
    for field in ("gravity_enabled", "stress_start_rate", "current_rate", "current_representation",
                  "escape_receipt_state", "next_experiment", "eta_range"):
        assert field in p72
    assert p72["stress_start_rate"] == "4/5"


# -- integration: arm_frontier arms but never launches ----------------------------------
def test_arm_frontier_is_launch_gated():
    doc = g.arm_frontier(live_parent="72B", source_manifest_sha256="a" * 64)
    assert sealed(doc, "frontier_armed_sha256")
    assert doc["operational_status"] == "integrated_armed_launch_gated"
    assert doc["launch_gate"]["launchable_now"] is False
    assert doc["gravity_enabled"] is False        # default-off; heavy launch deferred
    assert set(doc["giant_frontier_rows_augmented"]) == {
        "deepseek-v3.2-685b", "kimi-k2.6-1t", "deepseek-v4-pro-1.6t"}


# -- default-off: nothing activates until every gate holds ------------------------------
def test_default_off_until_all_gates():
    pol = gp.default_policy()
    assert gp.gravity_enabled(policy=pol, env={"HAWKING_GRAVITY_ENABLED": "1"}) is False
    from eco_common import seal_field
    enabled = {k: v for k, v in pol.items() if k != "policy_sha256"}
    enabled["enabled"] = True
    enabled["activation_gates"] = {k: True for k in enabled["activation_gates"]}
    enabled = seal_field(enabled, "policy_sha256")
    assert gp.gravity_enabled(policy=enabled, env={"HAWKING_GRAVITY_ENABLED": "1"}) is True
    # ...but the env flag alone is not enough
    assert gp.gravity_enabled(policy=enabled, env={}) is False
