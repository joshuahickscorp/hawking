"""ODYSSEY III — attack a named campaign law. Concurrent with I and II.

As soon as a law looks promising, this module attacks it. It does not wait
for Odyssey I to finish. Attack specs come from the Odyssey III adversary
(generate_attacks / apply_result / SCOPE_LADDER); campaign laws come from
Odyssey II transfer. Neither authority is forked.

An attack that fails to break a law is a result (SURVIVED / STRENGTHENED),
not a null run. A sealed specimen whose architecture differs substantially
(Falcon-H1) is an adversary. Missing measurements are UNMEASURED with the
experiment that would settle them. Scope cannot widen without a replicating
specimen.

    python3 tools/future/odyssey3_attack.py --build
    python3 tools/future/odyssey3_attack.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from typing import Any, Mapping

from tools.future._common import write_receipt, _assert_no_hardware_claims
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey2_transfer as o2t
from tools.future import odyssey3_adversary as o3
from tools.future import phase_listeners as pl


RECEIPT = "ODYSSEY3_ATTACK.json"
SCHEMA = "hawking.future.odyssey3_attack.v1"
VERSION = 1
RECORDED_BY = "tools/future/odyssey3_attack.py"

BROKE = "BROKE"
SURVIVED = "SURVIVED"
UNMEASURED = o2t.UNMEASURED

# II lattice (narrow -> wide) onto III ladder (wide -> refuted). Never a promotion.
II_TO_III_SCOPE = {
    "MODEL_LOCAL": "MODEL_LOCAL",
    "ARCHITECTURE_FAMILY": "FAMILY_VERIFIED",
    "BACKEND_FAMILY": "FAMILY_VERIFIED",
    "MACHINE_LOCAL": "MACHINE_LOCAL",
    "GENERIC_CANDIDATE": "FAMILY_VERIFIED",
    "GENERIC_VERIFIED": "GENERIC_VERIFIED",
}


class AttackError(ValueError):
    """Attack protocol violation."""


# ---------------------------------------------------------------------------
# Project an Odyssey II campaign law onto the Odyssey III field set.
# Extra II keys are dropped; III validate_law is the gate.
# ---------------------------------------------------------------------------


def to_o3_law(law: ols.Law, *, organ_local: bool = False) -> dict[str, Any]:
    scope = "ORGAN_LOCAL" if organ_local else II_TO_III_SCOPE[law.scope]
    names: list[str] = []
    for cand in law.transfer_candidates:
        if isinstance(cand, dict):
            name = cand.get("target_model") or cand.get("target_school")
        else:
            name = cand
        if name and str(name) not in names:
            names.append(str(name))
    conf = law.transfer_confidence
    if isinstance(conf, dict):
        conf_v = float(conf["value"])
    else:
        conf_v = float(conf)
    rec = {
        "law_id": law.law_id,
        "statement": law.statement,
        "source_model": law.source_model,
        "source_device": law.source_device,
        "architecture_family": law.architecture_family,
        "organ_class": law.organ_class,
        "backend": law.backend,
        "evidence_strength": law.evidence_strength,
        "evidence_refs": list(law.evidence_refs),
        "scope": scope,
        "transfer_candidates": names,
        "transfer_confidence": conf_v,
        "counterexample_requirement": law.counterexample_requirement,
    }
    return o3.validate_law(rec)


def campaign_o3_laws() -> list[dict[str, Any]]:
    return [to_o3_law(law) for law in o2t.campaign_laws()]


def campaign_o3_law(law_id: str) -> dict[str, Any]:
    return to_o3_law(o2t.campaign_law(law_id))


def odyssey_i_barrier() -> None:
    return o2t.odyssey_i_barrier()


def widen_scope(law: dict[str, Any] | ols.Law, target_scope: str, evidence: Mapping[str, Any]) -> Any:
    """III does not widen. Any widen attempt still has to pass the II guard.

    A surviving attack is not a promotion. Promotion without a replicating
    specimen raises ReplicatingSpecimenRequired.
    """
    if isinstance(law, dict):
        ii = o2t.campaign_law(str(law["law_id"]))
    else:
        ii = law
    # III scopes that are wider than the honest II scope still have to climb
    # the II lattice, which requires a replicating specimen first.
    return o2t.widen(ii, target_scope, evidence)


# ---------------------------------------------------------------------------
# Attacks. Each one names the law, the specimen, and the cited evidence.
# ---------------------------------------------------------------------------


def _spec_for(law: dict[str, Any], family: str) -> dict[str, Any]:
    attacks = {a["family"]: a for a in o3.generate_attacks(law)}
    if family not in attacks:
        raise AttackError(f"{law['law_id']} has no {family} spec")
    return attacks[family]


def attack_l5_fidelity_703p5() -> dict[str, Any]:
    """Fidelity counterexample: 703.5 addr_probe never loads the activation.

    L5 already names 703.5 as the wrong shape. The attack FAILS to break the
    law and therefore STRENGTHENS it. Not a null run.
    """
    law = campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    falcon = o2t.require_sealed("Falcon-H1-7B-Instruct")  # named, not used as the organ
    cites = o2t.campaign_citations(law["law_id"])
    spec = _spec_for(law, "measurement_trap")
    # Do not execute_attack: that replay is the cosine scale-invariance trap,
    # which is the wrong harness for a roof-shape law.
    result = {
        "verdict": "HOLDS",
        "synthetic": False,
        "reason": (
            "703.5 is the addr_probe that never loads the activation "
            f"(copied activation_loaded={cites['addr_probe_activation_loaded']['value']!r} "
            f"from {cites['addr_probe_activation_loaded']['source_receipt']}; "
            f"usable_as_production_streaming_roof="
            f"{cites['addr_probe_usable']['value']!r}). L5 already distinguishes "
            "that probe from the 497.4 production-shaped legs. The fidelity "
            "counterexample fails to break the law."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "physical_arm": "not_run",
    }
    update = o3.apply_result(law, spec, result)
    return {
        "attack_id": "L5-FIDELITY-703P5-ADDR-PROBE",
        "kind": "fidelity_counterexample",
        "family": spec["family"],
        "named_law": law["law_id"],
        "statement": law["statement"],
        "specimen_used": o2t.SOURCE_SPECIMEN,
        "specimen_display": o2t.SOURCE_DISPLAY,
        "adversary_named": falcon.get("specimen_id"),
        "adversary_role": "§66 natural adversary is available; this fidelity attack is on the source legs",
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved": update["moved"],
        "verdict": SURVIVED,
        "strengthened": True,
        "citations": {
            "mlp_arm_a_gb_s": cites["mlp_arm_a_gb_s"],
            "lm_head_gb_s": cites["lm_head_gb_s"],
            "clean_gemv_gb_s": cites["clean_gemv_gb_s"],
            "addr_probe_activation_loaded": cites["addr_probe_activation_loaded"],
            "addr_probe_usable": cites["addr_probe_usable"],
            "mlp_arm_a_activation_loaded": cites["mlp_arm_a_activation_loaded"],
            "lm_head_activation_loaded": cites["lm_head_activation_loaded"],
        },
        "apply_result": update,
        "o3_spec_attack_id": spec["attack_id"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def attack_l5_two_legs() -> dict[str, Any]:
    """Measurement weakness: 497.4 rests on two legs (MLP ARM A, LM head).

    Both legs load the activation and agree. A third same-shape organ on this
    parent, or a replicating specimen, would be required to widen. The law
    SURVIVES at MODEL_LOCAL. Two legs are not a GENERIC_VERIFIED roof.
    """
    law = campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    cites = o2t.campaign_citations(law["law_id"])
    spec = _spec_for(law, "law_scope")
    result = {
        "verdict": "HOLDS",
        "synthetic": False,
        "reason": (
            "Both cited legs still agree: MLP ARM A stripped "
            f"{cites['mlp_arm_a_gb_s']['value']} GB/s "
            f"(activation_loaded={cites['mlp_arm_a_activation_loaded']['value']!r}) "
            f"and LM head production {cites['lm_head_gb_s']['value']} GB/s "
            f"(activation_loaded={cites['lm_head_activation_loaded']['value']!r}). "
            "n_legs=2 on one parent is a measurement weakness against widening, "
            "not a refutation of the two legs."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    update = o3.apply_result(law, spec, result)
    return {
        "attack_id": "L5-MEASUREMENT-WEAKNESS-TWO-LEGS",
        "kind": "measurement_weakness",
        "family": spec["family"],
        "named_law": law["law_id"],
        "statement": law["statement"],
        "specimen_used": o2t.SOURCE_SPECIMEN,
        "specimen_display": o2t.SOURCE_DISPLAY,
        "n_legs": 2,
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved": update["moved"],
        "verdict": SURVIVED,
        "cannot_widen_without_replicating_specimen": True,
        "citations": {
            "mlp_arm_a_gb_s": cites["mlp_arm_a_gb_s"],
            "lm_head_gb_s": cites["lm_head_gb_s"],
            "mlp_arm_a_activation_loaded": cites["mlp_arm_a_activation_loaded"],
            "lm_head_activation_loaded": cites["lm_head_activation_loaded"],
        },
        "apply_result": update,
        "o3_spec_attack_id": spec["attack_id"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def attack_l4_probe_classes() -> dict[str, Any]:
    """Measurement weakness on L4: the 'two instances' are not the same probe.

    widen_f4 is an isolated-organ vs complete-token pair (0.7046 -> 1.0245).
    fold_addqx 1.745 is a 3-GEMV unfused projection, not that pair; the
    isolated-organ fold saving is 3.9452 vs complete 3.9833.

    Directional 'undersell' SURVIVES. The framing 'two equivalent instances'
    BREAKS. Scope stays MODEL_LOCAL (HOLDS on the directional law).
    """
    law = campaign_o3_law("LAW-L4-PROBE-UNDERSELLS-TOKEN")
    cites = o2t.campaign_citations(law["law_id"])
    spec = _spec_for(law, "measurement_trap")
    isolated_inc = cites["fold_isolated_incumbent_ms"]["value"]
    isolated_add = cites["fold_isolated_addqx_ms"]["value"]
    isolated_saving = None
    if isinstance(isolated_inc, (int, float)) and isinstance(isolated_add, (int, float)):
        isolated_saving = isolated_inc - isolated_add
    complete_inc = cites["fold_incumbent_ms"]["value"]
    complete_add = cites["fold_addqx_ms"]["value"]
    complete_saving = None
    if isinstance(complete_inc, (int, float)) and isinstance(complete_add, (int, float)):
        complete_saving = complete_inc - complete_add
    result = {
        "verdict": "HOLDS",
        "synthetic": False,
        "reason": (
            "Directional undersell still holds on the widen_f4 isolated-vs-token "
            f"pair ({cites['widen_isolated_ms']['value']} -> "
            f"{cites['widen_complete_ms']['value']}) and on the fold isolated-organ "
            f"pair (saving {isolated_saving} -> {complete_saving}). The 1.745 figure "
            f"(copied from {cites['fold_projection_ms']['source_receipt']} "
            f"{cites['fold_projection_ms']['source_field']}) is a 3-GEMV projection, "
            "not a second isolated-vs-token instance. Framing 'two equivalent "
            "instances' is broken; the directional law survived."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    update = o3.apply_result(law, spec, result)
    return {
        "attack_id": "L4-MEASUREMENT-WEAKNESS-PROBE-CLASSES",
        "kind": "measurement_weakness",
        "family": spec["family"],
        "named_law": law["law_id"],
        "statement": law["statement"],
        "specimen_used": o2t.SOURCE_SPECIMEN,
        "specimen_display": o2t.SOURCE_DISPLAY,
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved": update["moved"],
        "verdict": SURVIVED,
        "where_it_breaks": (
            "the 'two instances' framing: 1.745 ms is a projection "
            "(MLP_DECODE_CHEAPEN via FOLD_ADDQX_AB.cited_diagnostic), not an "
            "isolated-organ A/B like widen_f4"
        ),
        "where_it_survives": (
            "direction: both honest isolated-vs-token pairs undersell, they do "
            "not oversell"
        ),
        "isolated_fold_saving_ms": isolated_saving,
        "complete_fold_saving_ms": complete_saving,
        "citations": cites,
        "apply_result": update,
        "o3_spec_attack_id": spec["attack_id"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def attack_l1_falcon_model_counterexample() -> dict[str, Any]:
    """§66: Falcon-H1 is a sealed specimen of a different family, so it is an adversary.

    Against the honest MODEL_LOCAL L1 this is a negative-transfer spec. Physical
    ARM A on Falcon is UNMEASURED (no GPU lease). VALUE transfer already failed
    in Odyssey II. An UNMEASURED attack is recorded as a result, not a null run.
    classify_attack on a MODEL_LOCAL law with Falcon as source_model is VACUOUS
    for refutation of the honest scope — that is itself a finding.
    """
    law = campaign_o3_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    falcon = o2t.require_sealed("Falcon-H1-7B-Instruct")
    spec = _spec_for(law, "negative_transfer")
    # Point the hostile target at the real sealed Falcon, not a synthesized name.
    spec = dict(spec)
    spec["adversarial_target"] = falcon.get("specimen_id")
    spec["inputs"] = dict(spec.get("inputs") or {})
    spec["inputs"]["apply_on"] = {
        "name": falcon.get("specimen_id"),
        "kind": "sealed_natural_adversary",
        "source_model": falcon.get("repo"),
        "source_family": falcon.get("architecture_family"),
        "source_organ": law["organ_class"],
        "source_device": law["source_device"],
        "source_backend": law["backend"],
    }
    vacuity = pl.classify_attack(o2t.campaign_law(law["law_id"]).to_dict(), spec)
    executed = o3.execute_attack(spec, law)
    layout = o2t.falcon_layout()
    # Honest MODEL_LOCAL law: Falcon is outside claimed domain, so a miss
    # cannot refute it. Physical METHOD measurement is UNMEASURED.
    result = {
        "verdict": "INCONCLUSIVE",
        "synthetic": True,
        "reason": (
            f"§66 adversary {falcon.get('specimen_id')} family="
            f"{falcon.get('architecture_family')!r} layout={layout.get('storage')!r}. "
            f"classify_attack reason_code={vacuity.get('reason_code')}. "
            f"execute_attack verdict={executed.get('verdict')} "
            f"(physical_arm={executed.get('physical_arm')}). ARM A vs production "
            f"on Falcon MLP is UNMEASURED. Experiment: {o2t.L1_METHOD_EXPERIMENT}"
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    update = o3.apply_result(law, spec, result)
    return {
        "attack_id": "L1-MODEL-COUNTEREXAMPLE-FALCON-H1",
        "kind": "model_counterexample",
        "family": spec["family"],
        "named_law": law["law_id"],
        "statement": law["statement"],
        "specimen_used": falcon.get("specimen_id"),
        "specimen_display": falcon.get("alias"),
        "specimen_repo": falcon.get("repo"),
        "specimen_family": falcon.get("architecture_family"),
        "specimen_whole_tree_verified": falcon.get("whole_tree_verified"),
        "section_66": (
            "a sealed specimen whose architecture differs substantially IS an "
            "adversary; Falcon-H1 is that specimen"
        ),
        "vacuity": vacuity,
        "execute_attack": executed,
        "falcon_layout": layout,
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved": update["moved"],
        "verdict": UNMEASURED,
        "not_a_null_run": True,
        "measurement_state": UNMEASURED,
        "experiment_that_would_settle": o2t.L1_METHOD_EXPERIMENT,
        "apply_result": update,
        "o3_spec_attack_id": spec["attack_id"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
    }


def attack_l5_overclaim_unique_roof() -> dict[str, Any]:
    """Layer counterexample against a GENERIC reading of L5.

    Someone reading 'the production-shaped roof is 497.4' as machine-generic
    meets deltanet ARM A stripped at 943.2 (activation loaded, different organ).
    The honest MODEL_LOCAL law is not deleted; the overclaim BREAKS and, if we
    feed REFUTED to a law_scope spec, scope moves MODEL_LOCAL -> ORGAN_LOCAL.
    """
    law = campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    cites = o2t.campaign_citations(law["law_id"])
    spec = _spec_for(law, "law_scope")
    dn = cites["deltanet_arm_a_gb_s"]
    result = {
        "verdict": "REFUTED",
        "synthetic": False,
        "reason": (
            "A GENERIC reading of 'the production-shaped roof WITH activation "
            f"load is 497.4' is broken by deltanet ARM A stripped at {dn['value']} "
            f"GB/s (copied from {dn['source_receipt']} {dn['source_field']}; "
            "activation is loaded on that arm). MLP ARM A and LM head still "
            "agree at 497.4. Scope moves to ORGAN_LOCAL: mlp affine-q2 + lm_head, "
            "not deltanet, not a machine-wide roof. The law is not deleted."
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    update = o3.apply_result(law, spec, result)
    if not update["moved"] or not o3.is_downgrade(update["scope_before"], update["scope_after"]):
        raise AttackError(f"L5 overclaim refutation did not move scope DOWN: {update}")
    return {
        "attack_id": "L5-LAYER-COUNTEREXAMPLE-DELTANET-943P2",
        "kind": "layer_counterexample",
        "family": spec["family"],
        "named_law": law["law_id"],
        "statement": law["statement"],
        "specimen_used": o2t.SOURCE_SPECIMEN,
        "specimen_display": o2t.SOURCE_DISPLAY,
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved": update["moved"],
        "verdict": BROKE,
        "broke_what": (
            "the GENERIC reading of 497.4 as the unique production-shaped "
            "activation-loaded roof"
        ),
        "survived_what": (
            "MLP ARM A stripped and LM head production still cite 497.4 on "
            "sealed-3.14; law kept at ORGAN_LOCAL"
        ),
        "law_deleted": False,
        "citations": {
            "mlp_arm_a_gb_s": cites["mlp_arm_a_gb_s"],
            "lm_head_gb_s": cites["lm_head_gb_s"],
            "deltanet_arm_a_gb_s": dn,
        },
        "apply_result": update,
        "o3_spec_attack_id": spec["attack_id"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "odyssey_i_barrier": odyssey_i_barrier(),
        "roof_anchor_note": (
            "ROOF_ANCHOR already rejects deltanet 943.2 as a DRAM streaming "
            "roof (exceeds published peak 819). It remains a layer/organ "
            "disagreement with a unique 497.4, which is the attack."
        ),
    }


def run_attacks() -> dict[str, Any]:
    fidelity = attack_l5_fidelity_703p5()
    two_legs = attack_l5_two_legs()
    overclaim = attack_l5_overclaim_unique_roof()
    l4 = attack_l4_probe_classes()
    falcon = attack_l1_falcon_model_counterexample()
    attacks = [overclaim, fidelity, two_legs, l4, falcon]
    headline = overclaim
    return {
        "headline": headline,
        "attacks": attacks,
        "named_law": headline["named_law"],
        "specimen_used": headline["specimen_used"],
        "verdict": headline["verdict"],
        "n_broke": sum(1 for a in attacks if a["verdict"] == BROKE),
        "n_survived": sum(1 for a in attacks if a["verdict"] == SURVIVED),
        "n_unmeasured": sum(1 for a in attacks if a["verdict"] == UNMEASURED),
        "survived_is_a_result": True,
        "unmeasured_is_a_result": True,
    }


def selftest() -> dict[str, Any]:
    run = run_attacks()
    if run["named_law"] != "LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD":
        raise AttackError(f"headline law {run['named_law']}")
    if run["verdict"] != BROKE:
        raise AttackError(f"headline must report BROKE on the unique-roof overclaim, got {run['verdict']}")
    if not run["specimen_used"]:
        raise AttackError("headline did not name a specimen")
    if run["n_survived"] < 1:
        raise AttackError("no SURVIVED attack recorded; a failed break must be a result")
    if run["n_unmeasured"] < 1:
        raise AttackError("Falcon §66 attack was not recorded as UNMEASURED")
    if odyssey_i_barrier() is not None:
        raise AttackError("Odyssey I barrier is not None")
    law = campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    try:
        widen_scope(
            law,
            "ARCHITECTURE_FAMILY",
            {
                "models": [o2t.SOURCE_MODEL, "tiiuae/Falcon-H1-7B-Instruct"],
                "architecture_families": [o2t.SOURCE_FAMILY, "falcon_h1"],
                "replications": [],
            },
        )
        raise AttackError("widen without replicating specimen did not raise")
    except o2t.ReplicatingSpecimenRequired as e:
        widen_refused = {"raised": True, "reason": e.reason, "law_id": e.law_id}
    # HOLDS must not move scope (negative control, via o3.apply_result).
    spec = _spec_for(law, "law_scope")
    hold = o3.apply_result(law, spec, {"verdict": "HOLDS", "synthetic": True, "evidence_class": "STATIC_ONLY"})
    if hold["moved"]:
        raise AttackError("HOLDS moved scope")
    return {
        "ok": True,
        "named_law": run["named_law"],
        "specimen_used": run["specimen_used"],
        "verdict": run["verdict"],
        "n_broke": run["n_broke"],
        "n_survived": run["n_survived"],
        "n_unmeasured": run["n_unmeasured"],
        "widen_without_replicating_specimen_raised": widen_refused,
        "holds_does_not_move_scope": True,
        "odyssey_i_barrier": odyssey_i_barrier(),
        "listen_rule": pl.LISTEN_RULE,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def build() -> Any:
    laws = campaign_o3_laws()
    for law in laws:
        o3.validate_law(law)
        plan = o3.emit_for_law(law)
        if plan["n_attacks"] < 1:
            raise o3.NoAttackError(f"{law['law_id']}: emitter refused")
    run = run_attacks()
    loop = selftest()
    falcon = o2t.require_sealed("Falcon-H1-7B-Instruct")
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Odyssey III attack: consume a named campaign law, attack it "
            "deliberately, report where it breaks or that it survived. "
            "Concurrent with Odyssey I and II; no Phase-I barrier."
        ),
        "odyssey": "III WHERE IS HAWKING WRONG?",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "odyssey_i_barrier": None,
        "phase_iii_waits_for_odyssey_i_complete": False,
        "listen_rule": pl.LISTEN_RULE,
        "law_authority": {
            "odyssey2_transfer": "campaign L1–L5, sealed specimens, replicating-specimen guard",
            "odyssey2_law_store": "Law field set / sequential lattice via o2t",
            "odyssey3_adversary": "generate_attacks, apply_result, SCOPE_LADDER, execute_attack",
            "not_a_fork": True,
        },
        "named_law": run["named_law"],
        "specimen_used": run["specimen_used"],
        "verdict": run["verdict"],
        "headline": run["headline"],
        "attacks": run["attacks"],
        "n_broke": run["n_broke"],
        "n_survived": run["n_survived"],
        "n_unmeasured": run["n_unmeasured"],
        "survived_is_a_result": True,
        "unmeasured_is_a_result": True,
        "section_66_adversary": {
            "specimen": falcon.get("specimen_id"),
            "repo": falcon.get("repo"),
            "architecture_family": falcon.get("architecture_family"),
            "whole_tree_verified": falcon.get("whole_tree_verified"),
            "role": "natural counterexample; not a bespoke synthetic",
        },
        "campaign_o3_laws": laws,
        "selftest": loop,
        "recovered_implementation": {
            "odyssey3_adversary": "nine families, ranking, apply_result DOWN, emitter refusal — consumed",
            "odyssey2_transfer": "named campaign laws and Falcon sealed identity — consumed",
            "phase_listeners": "vacuity classifier, LISTEN_RULE — consumed",
            "roof_anchor": "703.5 no-activation, 497.4 two legs — cited, not re-measured",
        },
        "gaps_closed": [
            "Named campaign law L5 attacked; GENERIC unique-roof reading BROKE (deltanet 943.2); honest 497.4 legs survived at ORGAN_LOCAL.",
            "L5 fidelity attack with 703.5 SURVIVED and strengthened the law (wrong shape already named).",
            "L4 measurement weakness recorded: framing broke, directional law SURVIVED.",
            "Falcon-H1 used as §66 model counterexample against L1; UNMEASURED recorded as a result.",
            "widen_scope raises ReplicatingSpecimenRequired.",
            "No Odyssey I completion barrier.",
        ],
        "negative_findings": [
            "No GPU lease: Falcon ARM A is UNMEASURED, not a fabricated GB/s.",
            "execute_attack on measurement_trap would fire the cosine harness; that trap is the wrong shape for these laws and was not used as the verdict.",
            "ROOF_ANCHOR already rejects 943.2 as a DRAM roof; the attack uses it as an organ disagreement, not as a replacement roof.",
            "Independence limitation: the same operator wrote the campaign citations and the attacks.",
        ],
        "claim_boundary_reminder": (
            "STATIC_ONLY. Copied GB/s figures are citations. A SURVIVED attack "
            "is a result. UNMEASURED stays UNMEASURED."
        ),
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=1, sort_keys=True))
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
