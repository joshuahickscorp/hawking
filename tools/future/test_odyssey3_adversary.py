"""Odyssey III adversary: generator, ranking, closed loop, watched refusals."""
from __future__ import annotations

import json

import pytest

from tools.future import odyssey3_adversary as o3
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def _cosine_law() -> dict:
    return next(l for l in o3.fixture_laws() if l["law_id"] == "LAW-COSINE-ADEQUACY-GATE")


def _law_by_id(law_id: str) -> dict:
    return next(l for l in o3.fixture_laws() if l["law_id"] == law_id)


def test_build_emits_sealed_receipt():
    out = o3.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ODYSSEY3_ADVERSARY.json"
    assert doc["schema"] == "hawking.future.odyssey3_adversary.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["n_laws"] >= 1
    assert doc["n_attacks"] == doc["n_laws"] * len(o3.ATTACK_FAMILIES)
    assert doc["closed_loop"]["moved_down"] is True
    assert doc["closed_loop"]["scope_before"] != doc["closed_loop"]["scope_after"]
    assert o3.is_downgrade(
        doc["closed_loop"]["scope_before"], doc["closed_loop"]["scope_after"]
    )
    _assert_no_hardware_claims(doc)


def test_selftest_downgrades_cosine_law():
    loop = o3.selftest()
    assert loop["law_id"] == "LAW-COSINE-ADEQUACY-GATE"
    assert loop["scope_before"] == "GENERIC_VERIFIED"
    assert loop["scope_after"] != "GENERIC_VERIFIED"
    assert o3.scope_rank(loop["scope_after"]) > o3.scope_rank(loop["scope_before"])
    assert loop["scope_update"]["direction"] == "DOWN"
    assert loop["scope_update"]["moved"] is True
    assert loop["hold_negative_control"]["moved"] is False
    assert loop["hold_negative_control"]["scope_after"] == "GENERIC_VERIFIED"


def test_law_schema_exact_fields():
    for law in o3.fixture_laws():
        assert tuple(k for k in o3.LAW_FIELDS if k in law) == o3.LAW_FIELDS
        o3.validate_law(law)


def test_missing_law_field_is_refused():
    law = dict(_cosine_law())
    law.pop("counterexample_requirement")
    with pytest.raises(o3.LawSchemaError) as ei:
        o3.validate_law(law)
    assert "counterexample_requirement" in str(ei.value)


def test_already_refuted_law_is_refused():
    law = dict(_cosine_law())
    law["scope"] = "REFUTED"
    with pytest.raises(o3.LawSchemaError) as ei:
        o3.validate_law(law)
    assert "already REFUTED" in str(ei.value)


def test_unknown_scope_is_refused():
    law = dict(_cosine_law())
    law["scope"] = "ERA_VI"  # there is no Era VI; there is no such scope
    with pytest.raises(o3.LawSchemaError):
        o3.validate_law(law)


def test_nine_families_each_emit_executable_spec():
    law = _cosine_law()
    attacks = {a["family"]: a for a in o3.generate_attacks(law)}
    assert set(attacks) == set(o3.ATTACK_FAMILIES)
    assert len(attacks) == 9
    for family, spec in attacks.items():
        for field in o3.ATTACK_SPEC_FIELDS:
            assert field in spec, f"{family} missing {field}"
        assert spec["command"][0] == "python3"
        assert "--replay-attack" in spec["command"]
        assert spec["law_id"] == law["law_id"]
        assert spec["evidence_class"] == "STATIC_ONLY"
        assert spec["bench_state"] == "UNKNOWN"
        assert spec["cost_units"] >= 1
        assert 0.0 < spec["p_refutation"] <= 1.0
        assert spec["expected_if_law_holds"]
        assert spec["expected_if_law_false"]
        assert spec["falsifier"]
        assert o3.is_downgrade(law["scope"], spec["target_scope_if_refuted"])


def test_every_fixture_law_receives_at_least_one_attack():
    for law in o3.fixture_laws():
        plan = o3.emit_for_law(law)
        assert plan["n_attacks"] >= 1
        assert plan["selected_attack_id"]
        assert plan["ranked_attacks"][0]["attack_id"] == plan["selected_attack_id"]


def test_selection_emits_cheapest_capable_first():
    law = _cosine_law()
    ranked = o3.rank_attacks(o3.generate_attacks(law))
    scores = [a["selection_score"] for a in ranked]
    assert scores == sorted(scores)
    # Cosine-as-gate: measurement_trap is both cheapest and highest prior.
    assert ranked[0]["family"] == "measurement_trap"
    assert ranked[0]["cost_units"] == o3.FAMILY_COST["measurement_trap"]
    # ranking is deterministic
    ranked2 = o3.rank_attacks(o3.generate_attacks(law))
    assert [a["attack_id"] for a in ranked] == [a["attack_id"] for a in ranked2]


def test_selection_prefers_causal_over_weak_measurement_on_non_metric_law():
    law = _law_by_id("LAW-PACKED-GEMV-DIRECT-TRANSFER")
    ranked = o3.rank_attacks(o3.generate_attacks(law))
    # Packed GEMV is layout/compiler shaped, not a cosine certificate.
    assert ranked[0]["family"] != "goodhart"
    assert ranked[0]["family"] != "measurement_trap"
    assert ranked[0]["selection_score"] <= min(a["selection_score"] for a in ranked)


def test_affine_seeded_law_leads_with_holdout_not_cosine_trap():
    law = _law_by_id("LAW-AFFINE-SEEDED-MATCHED-BITS")
    ranked = o3.rank_attacks(o3.generate_attacks(law))
    assert ranked[0]["family"] == "blind_holdout"


def test_apply_result_refutation_downgrades_scope():
    """NEGATIVE CONTROL: a refuting result actually moves scope DOWN."""
    law = _cosine_law()
    before = law["scope"]
    attack = o3.rank_attacks(o3.generate_attacks(law))[0]
    update = o3.apply_result(
        law,
        attack,
        {
            "verdict": "REFUTED",
            "synthetic": True,
            "reason": "synthetic refutation",
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
        },
    )
    assert update["scope_before"] == before
    assert update["scope_after"] != before
    assert o3.is_downgrade(before, update["scope_after"])
    assert update["direction"] == "DOWN"
    assert update["moved"] is True
    assert update["law_after"]["scope"] == update["scope_after"]
    # original law object is not mutated
    assert law["scope"] == before


def test_apply_result_hold_does_not_move_scope():
    law = _cosine_law()
    attack = o3.generate_attacks(law)[0]
    update = o3.apply_result(
        law,
        attack,
        {"verdict": "HOLDS", "synthetic": True, "evidence_class": "STATIC_ONLY"},
    )
    assert update["moved"] is False
    assert update["direction"] == "NONE"
    assert update["scope_after"] == update["scope_before"] == law["scope"]


def test_refutation_that_would_not_move_scope_is_refused():
    """NEGATIVE CONTROL: the loop bug is watched failing, not assumed absent."""
    law = _cosine_law()
    attack = dict(o3.generate_attacks(law)[0])
    attack["target_scope_if_refuted"] = law["scope"]  # same scope: illegal
    with pytest.raises(o3.ScopeUnmovedError) as ei:
        o3.apply_result(
            law,
            attack,
            {"verdict": "REFUTED", "synthetic": True, "evidence_class": "STATIC_ONLY"},
        )
    assert law["law_id"] in str(ei.value)
    assert "does not change scope" in str(ei.value)


def test_emitter_rejects_law_with_no_attack(monkeypatch):
    """NEGATIVE CONTROL: a law with no generated attack is refused, not published.

    The refusal must actually fire. Stubbing generate_attacks is the only way
    to reach the empty-plan path, because a valid law always yields nine
    families — that is the point of the guard.
    """
    law = _cosine_law()
    monkeypatch.setattr(o3, "generate_attacks", lambda _law: [])
    with pytest.raises(o3.NoAttackError) as ei:
        o3.emit_for_law(law)
    msg = str(ei.value)
    assert law["law_id"] in msg
    assert "refused" in msg


def test_build_refuses_empty_law_list(monkeypatch):
    monkeypatch.setattr(o3, "load_laws", lambda: ([], {"source": "empty"}))
    with pytest.raises(o3.NoAttackError) as ei:
        o3.build()
    assert "no laws" in str(ei.value)


def test_load_laws_falls_back_to_fixture_when_store_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(o3, "LAW_STORE", tmp_path / "ODYSSEY2_LAW_STORE.json")
    laws, meta = o3.load_laws()
    assert meta["store_present"] is False
    assert meta["source"] == "inline_fixture"
    assert {l["law_id"] for l in laws} == {l["law_id"] for l in o3.fixture_laws()}


def test_store_present_is_preferred(tmp_path, monkeypatch):
    store_law = dict(_cosine_law())
    store_law["law_id"] = "LAW-FROM-STORE"
    store_law["statement"] = "store-authored law used to prove the loader prefers disk"
    payload = {
        "schema": "hawking.future.odyssey2_law_store.v1",
        "laws": [store_law],
    }
    path = tmp_path / "ODYSSEY2_LAW_STORE.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(o3, "LAW_STORE", path)
    laws, meta = o3.load_laws()
    assert meta["store_present"] is True
    assert meta["source"] == "receipts/future/ODYSSEY2_LAW_STORE.json"
    assert [l["law_id"] for l in laws] == ["LAW-FROM-STORE"]


def test_invalid_store_records_are_skipped_not_fatal(tmp_path, monkeypatch):
    bad = dict(_cosine_law())
    bad.pop("scope")
    good = dict(_cosine_law())
    good["law_id"] = "LAW-VALID-IN-STORE"
    path = tmp_path / "ODYSSEY2_LAW_STORE.json"
    path.write_text(json.dumps({"entries": [bad, good]}))
    monkeypatch.setattr(o3, "LAW_STORE", path)
    laws, meta = o3.load_laws()
    assert meta["n_store_valid"] == 1
    assert meta["n_store_rejected"] == 1
    assert laws[0]["law_id"] == "LAW-VALID-IN-STORE"


def test_scale_invariance_trap_catches_cosine_blindness():
    row = o3.run_scale_invariance_trap(dim=256, seed=0, scale=0.01)
    assert row["trap_fired"] is True
    assert row["cosine"] == pytest.approx(1.0, abs=1e-9)
    assert row["rel_fro"] == pytest.approx(0.99, abs=1e-6)


def test_skip_as_pass_trap_fires_on_all_skipped_suite():
    row = o3.run_skip_as_pass_trap()
    assert row["trap_fired"] is True
    assert row["naive_pass"] is True
    assert row["honest_pass"] is False


def test_skip_as_pass_trap_silent_when_all_passed():
    row = o3.run_skip_as_pass_trap(
        [{"name": "a", "status": "PASSED"}, {"name": "b", "status": "PASSED"}]
    )
    assert row["trap_fired"] is False
    assert row["honest_pass"] is True


def test_receipt_reread_trap_fires_only_when_receipt_replaces_a_run():
    fired = o3.run_receipt_reread_trap(ran_generator=False, read_checked_in_receipt=True)
    clean = o3.run_receipt_reread_trap(ran_generator=True, read_checked_in_receipt=False)
    assert fired["trap_fired"] is True
    assert clean["trap_fired"] is False


def test_measurement_trap_execute_is_cpu_and_fires():
    law = _cosine_law()
    spec = next(a for a in o3.generate_attacks(law) if a["family"] == "measurement_trap")
    result = o3.execute_attack(spec, law)
    assert result["evidence_class"] == "STATIC_ONLY"
    assert result["bench_state"] == "UNKNOWN"
    assert result["physical_arm"] == "not_run"
    assert result["verdict"] == "TRAP_TRIGGERED"
    assert "scale_invariance" in result["traps_fired"]
    assert "skip_counted_as_pass" in result["traps_fired"]


def test_physical_families_stay_inconclusive_without_gpu():
    law = _cosine_law()
    spec = next(a for a in o3.generate_attacks(law) if a["family"] == "negative_transfer")
    result = o3.execute_attack(spec, law)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["physical_arm"] == "not_run"
    assert result["bench_state"] == "UNKNOWN"


def test_contamination_trap_targets_machine_local_or_lower():
    law = _cosine_law()
    spec = next(a for a in o3.generate_attacks(law) if a["family"] == "contamination_trap")
    assert spec["target_scope_if_refuted"] == "MACHINE_LOCAL"
    assert o3.is_downgrade(law["scope"], "MACHINE_LOCAL")


def test_law_scope_attack_is_exactly_one_step():
    for law in o3.fixture_laws():
        spec = next(a for a in o3.generate_attacks(law) if a["family"] == "law_scope")
        assert spec["target_scope_if_refuted"] == o3.one_step_down(law["scope"])


def test_apply_result_trap_triggered_is_a_refutation():
    law = _cosine_law()
    spec = next(a for a in o3.generate_attacks(law) if a["family"] == "measurement_trap")
    update = o3.apply_result(
        law, spec, {"verdict": "TRAP_TRIGGERED", "synthetic": True}
    )
    assert update["moved"] is True
    assert update["direction"] == "DOWN"
    assert update["scope_after"] == "REFUTED"


def test_receipt_does_not_claim_hardware_numbers():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 12.0})
    plan = o3.emit_for_law(_cosine_law())
    _assert_no_hardware_claims(plan)
    _assert_no_hardware_claims(o3.selftest())


def test_no_odyssey_iv_and_no_era_vi_in_receipt():
    doc = json.loads(o3.build().read_text())
    blob = json.dumps(doc)
    assert "Odyssey IV" not in blob
    assert "Era VI" not in blob
    assert doc["odysseys"][-1].startswith("III")
    assert doc["eras"][-1].startswith("V")
