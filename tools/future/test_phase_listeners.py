"""Phase II / III listeners: vacuity, underspecification, identity, empty store."""
from __future__ import annotations

import json

import pytest

from tools.future import odyssey2_law_store as ols
from tools.future import phase_listeners as pl
from tools.future import workgraph as wg
from tools.future import workunit_species as wus
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _local(**over) -> dict:
    law = pl._fixture_local()
    law.update(over)
    return law


def _generic(**over) -> dict:
    law = pl._fixture_generic()
    law.update(over)
    return law


def test_build_emits_sealed_receipt():
    out = pl.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "PHASE_LISTENERS.json"
    assert doc["schema"] == pl.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["barrier"] is None
    assert doc["phase_ii_depends_on_phase_iii"] is False
    assert doc["phase_iii_depends_on_phase_ii"] is False
    assert doc["listen"]["concludes_law"] is False
    assert doc["listen"]["spawns_work"] is True
    assert doc["listen"]["performs_science"] is False
    assert doc["law_schema"] == list(ols.LAW_FIELDS)
    controls = doc["negative_controls"]
    assert controls["vacuous_attack_rejected"] is True
    assert controls["underspecified_refused"] is True
    assert controls["identity_transfer_rejected"] is True
    assert controls["empty_store_zero_units"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"].startswith("tools.future.phase_listeners")
    assert "FT.ODYSSEY_TRANSFER" in doc["resident_callable"]["frontier"]
    assert "FT.ODYSSEY_ADVERSARY" in doc["resident_callable"]["frontier"]
    _assert_no_hardware_claims(doc)


def test_selftest_fires_all_four_negative_controls():
    loop = pl.selftest()
    assert loop["vacuous_attack_rejected"] is True
    assert loop["underspecified_refused"] is True
    assert loop["identity_transfer_rejected"] is True
    assert loop["empty_store_zero_units"] is True
    assert loop["concludes_law"] is False
    assert loop["barrier"] is None
    assert loop["in_domain_family"] != "negative_transfer"


def test_vacuous_attack_is_rejected_as_vacuous():
    """NEGATIVE CONTROL: out-of-domain input is vacuous, not a useful miss."""
    law = _local()
    attack = pl.make_vacuous_attack(law)
    verdict = pl.classify_attack(law, attack)
    assert verdict["reason_code"] == pl.VACUOUS
    assert verdict["vacuous"] is True
    assert verdict["in_domain"] is False
    assert verdict["violations"]
    assert "outside" in verdict["reason"]
    with pytest.raises(pl.VacuousAttackError) as ei:
        pl.spawn_attack_unit(law, attack)
    assert law["law_id"] in str(ei.value)
    assert "vacuous" in str(ei.value)


def test_vacuous_attack_different_organ_is_rejected():
    law = _local()
    attack = {
        "family": "blind_holdout",
        "inputs": {**pl.origin_inputs(law), "organ_class": "lm_head"},
    }
    verdict = pl.classify_attack(law, attack)
    assert verdict["reason_code"] == pl.VACUOUS
    assert any(v["axis"] == "organ_class" for v in verdict["violations"])


def test_in_domain_attack_is_accepted():
    law = _local()
    attack = {
        "family": "blind_holdout",
        "inputs": pl.origin_inputs(law),
        "expected_information_gain": pl.INFO_HIGH,
        "would_prove": "holdout fails",
        "would_not_prove": "not a transfer miss",
    }
    verdict = pl.classify_attack(law, attack)
    assert verdict["reason_code"] == pl.USEFUL
    assert verdict["in_domain"] is True
    unit = pl.spawn_attack_unit(law, attack)
    wus.validate_emitted_unit(unit)
    assert unit["concludes_law"] is False
    assert unit["listener"] == pl.PHASE_III
    assert unit["species"] == "odyssey_iii_adversarial_experiment"


def test_underspecified_law_cannot_be_attacked():
    """NEGATIVE CONTROL: no recorded preconditions → refuse, do not attack anyway."""
    law = pl._fixture_underspecified()
    assert law["preconditions"] == []
    pre = pl.recorded_preconditions(law)
    assert pre["underspecified"] is True
    plan = pl.attacks(law)
    assert plan["decision"] == pl.REFUSED
    assert plan["reason_code"] == pl.UNDERSPECIFIED
    assert plan["ranked"] == []
    assert plan["selected"] is None
    assert "underspecified" in plan["reason"]
    with pytest.raises(pl.UnderspecifiedLawError) as ei:
        pl.spawn_attack_unit(law, {"family": "blind_holdout", "inputs": {}})
    assert "underspecified" in str(ei.value)


def test_absent_preconditions_without_explicit_key_are_underspecified():
    law = _local(
        source_model="UNKNOWN",
        architecture_family="UNKNOWN",
        organ_class="",
        backend="UNKNOWN",
        source_device="UNKNOWN",
    )
    # no preconditions key: derived from unnamed axes → absent
    assert "preconditions" not in law
    pre = pl.recorded_preconditions(law)
    assert pre["underspecified"] is True
    assert pre["source"] == "absent"
    plan = pl.attacks(law)
    assert plan["reason_code"] == pl.UNDERSPECIFIED


def test_transfer_target_identical_to_origin_is_rejected():
    """NEGATIVE CONTROL: origin is not a transfer."""
    law = _local()
    origin = {
        "target_school": "Qwen27",
        "target_model": "Qwen3.8-27B",
        "target_architecture_family": "dense_hybrid_transformer",
    }
    verdict = pl.classify_transfer(law, origin)
    assert verdict["reason_code"] == pl.NOT_A_TRANSFER
    assert "not a transfer" in verdict["reason"]
    with pytest.raises(pl.NotATransferError) as ei:
        pl.spawn_transfer_unit(law, origin)
    assert "not a transfer" in str(ei.value)
    plan = pl.transfer_targets(law)
    assert all(
        not pl._same_model(t.get("target_model"), law["source_model"])
        for t in plan["ranked"]
    )
    assert all(t.get("target_school") != "Qwen27" for t in plan["ranked"])
    assert any(r["reason_code"] == pl.NOT_A_TRANSFER for r in plan["rejected"])


def test_empty_store_emits_zero_units_and_says_why():
    """NEGATIVE CONTROL: empty store is a recorded zero, not an error."""
    result = pl.listen(laws=[])
    assert result["n_units"] == 0
    assert result["n_phase_ii"] == 0
    assert result["n_phase_iii"] == 0
    assert result["units"] == []
    assert result["reason_code"] == pl.EMPTY_STORE
    assert result["reason"]
    assert "zero" in result["reason"]
    assert result["concludes_law"] is False
    q = pl.qualifying_laws(laws=[])
    assert q["n"] == 0
    assert q["laws"] == []


def test_qualifying_laws_use_odyssey_ii_field_set():
    q = pl.qualifying_laws(laws=[_local(), _generic()])
    assert q["n"] == 2
    for row in q["laws"]:
        law = row["law"]
        for field in ols.LAW_FIELDS:
            assert field in law
        assert row["qualifies_for_attack"] is True
        assert row["preconditions"]
    ids = [r["law_id"] for r in q["laws"]]
    assert ids == sorted(ids)


def test_qualifying_laws_copes_with_real_or_absent_store():
    """Store may be git-only in this checkout. Absence is empty, never a skip."""
    q = pl.qualifying_laws()
    assert q["n"] >= 0
    assert q["store"]["source"] in {pl.LAW_STORE_REL, "absent"}
    for row in q["laws"]:
        assert row["law_id"]
        for field in ("scope", "source_model", "organ_class"):
            assert field in row["law"]
        assert row["law"]["scope"] in ols.SCOPES


def test_attacks_rank_strongest_useful_not_cheapest_out_of_domain():
    law = _local()
    plan = pl.attacks(law)
    assert plan["decision"] == pl.SPAWN
    assert plan["selected"] is not None
    assert plan["selected"]["family"] != "negative_transfer"
    assert plan["selected"]["expected_information_gain"] >= plan["ranked"][-1]["expected_information_gain"]
    gains = [a["expected_information_gain"] for a in plan["ranked"]]
    assert gains == sorted(gains, reverse=True)
    for spec in plan["ranked"]:
        assert spec["would_prove"]
        assert spec["would_not_prove"]
        assert spec["concludes_law"] is False
        assert pl.classify_attack(law, spec)["in_domain"] is True
    assert any(r.get("reason_code") == pl.VACUOUS and r.get("constructed") for r in plan["rejected"])


def test_generic_candidate_second_family_is_in_domain():
    law = _generic()
    plan = pl.attacks(law)
    nt = next(a for a in plan["ranked"] if a["family"] == "negative_transfer")
    verdict = pl.classify_attack(law, nt)
    assert verdict["in_domain"] is True
    assert verdict["reason_code"] == pl.USEFUL


def test_generic_candidate_different_organ_is_vacuous():
    law = _generic()
    attack = {
        "family": "representation_overfit",
        "inputs": {**pl.origin_inputs(law), "organ_class": "lm_head"},
    }
    verdict = pl.classify_attack(law, attack)
    assert verdict["reason_code"] == pl.VACUOUS


def test_listen_spawns_both_listeners_without_concluding():
    law = _local()
    generic = _generic()
    result = pl.listen(laws=[law, generic])
    assert result["n_phase_iii"] >= 1
    assert result["barrier"] is None
    assert result["phase_ii_depends_on_phase_iii"] is False
    assert result["phase_iii_depends_on_phase_ii"] is False
    assert result["concludes_law"] is False
    assert result["performs_science"] is False
    assert result["spawns_work"] is True
    blob = json.dumps(result["phase_iii"])
    assert "HOLDS" not in blob
    assert "REFUTED" not in blob
    for unit in result["units"]:
        wus.validate_emitted_unit(unit)
        assert unit["provider"] == "future.phase_listeners"
        assert unit["concludes_law"] is False
        assert unit["gpu_authority"] is False
        assert unit["classification"] == "STATIC_ONLY"
        wg.make_unit(
            id=unit["id"],
            role=unit["role"],
            description=unit["description"],
            dependencies=unit.get("dependencies") or [],
            resource_lane=unit["resource_lane"],
            mutation_scope=unit["mutation_scope"],
            verifier=unit["verifier"],
            expected_information_gain=int(unit["expected_information_gain"]),
            cost_units=int(unit["cost_units"]),
            species=unit.get("species"),
            effect_class=unit.get("effect_class") or "READ_ONLY",
        )


def test_listen_spawns_phase_ii_when_transfer_engine_proposes(monkeypatch):
    law = _local()

    def _propose(_law, target):
        if target in {"Qwen27", "Qwen3.8-27B"}:
            return []
        return [
            {
                "target_school": "Flash",
                "target_model": "Qwen/Qwen3.8-Flash-Next",
                "target_architecture_family": "qwen4_exp",
            }
        ]

    monkeypatch.setattr(pl.ols, "transfer_candidates", _propose)
    result = pl.listen(laws=[law])
    assert result["n_phase_ii"] >= 1
    schools = {row["target_school"] for row in result["phase_ii"]}
    assert "Flash" in schools
    assert "Qwen27" not in schools
    assert result["n_phase_iii"] == 1


def test_attacks_do_not_call_apply_result_or_promote(monkeypatch):
    """Listeners spawn work. They must not close the science loop themselves."""
    called = {"promote": 0, "apply": 0}

    def _boom_promote(*_a, **_k):
        called["promote"] += 1
        raise AssertionError("listeners must not promote")

    def _boom_apply(*_a, **_k):
        called["apply"] += 1
        raise AssertionError("listeners must not apply_result")

    monkeypatch.setattr(ols, "promote", _boom_promote)
    import tools.future.odyssey3_adversary as o3

    monkeypatch.setattr(o3, "apply_result", _boom_apply)
    pl.listen(laws=[_local()])
    pl.attacks(_local())
    pl.transfer_targets(_local())
    assert called["promote"] == 0
    assert called["apply"] == 0


def test_receipt_does_not_claim_hardware_numbers():
    with pytest.raises(Exception):
        _assert_no_hardware_claims({"tps": 12.0})
    plan = pl.attacks(_local())
    _assert_no_hardware_claims(plan)
    _assert_no_hardware_claims(pl.selftest())
    for key in HARDWARE_FIELDS:
        assert plan.get(key) in (None, "UNKNOWN") or key not in plan


def test_explicit_preconditions_override_derivation():
    law = _local(
        preconditions={"source_model": "Qwen3.8-27B", "organ_class": "mlp", "backend": "Metal"}
    )
    pre = pl.recorded_preconditions(law)
    assert pre["source"] == "explicit"
    assert {c["axis"] for c in pre["constraints"]} == {"source_model", "organ_class", "backend"}
    ok = pl.classify_attack(law, {"inputs": {**pl.origin_inputs(law), "backend": "Metal"}})
    assert ok["in_domain"] is True
    bad = pl.classify_attack(law, {"inputs": {**pl.origin_inputs(law), "backend": "CUDA"}})
    assert bad["reason_code"] == pl.VACUOUS


def test_generic_wildcard_organ_is_underspecified_for_attack():
    """GENERIC_CANDIDATE + organ_class=cross_model records no restricting axis."""
    law = _generic(organ_class="cross_model")
    pre = pl.recorded_preconditions(law)
    assert pre["underspecified"] is True
    plan = pl.attacks(law)
    assert plan["reason_code"] == pl.UNDERSPECIFIED
    assert plan["ranked"] == []
    q = pl.qualifying_laws(laws=[law])
    assert q["n"] == 1
    assert q["laws"][0]["qualifies_for_transfer"] is True
    assert q["laws"][0]["qualifies_for_attack"] is False


def test_wildcard_organ_does_not_vacuously_constrain():
    law = _local(organ_class="cross_model")
    pre = pl.recorded_preconditions(law)
    assert all(c["axis"] != "organ_class" for c in pre["constraints"])
    assert any(c["axis"] == "source_model" for c in pre["constraints"])
    # A different organ on the same model is still in the claimed model-local domain.
    verdict = pl.classify_attack(
        law, {"inputs": {"source_model": "Qwen3.8-27B", "organ_class": "lm_head"}}
    )
    assert verdict["in_domain"] is True


def test_no_odyssey_iv_and_no_era_vi():
    doc = json.loads(pl.build().read_text())
    blob = json.dumps(doc)
    assert "Odyssey IV" not in blob
    assert "Era VI" not in blob
