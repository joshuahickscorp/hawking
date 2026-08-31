"""Tests for the Flash organ pivot.

Negative controls that must actually fire:
  * a restatement of a killed family is REFUSED, naming the family and the receipt
  * an organ with no census bytes is UNRANKABLE, not guessed
  * the scar records organ, surface, split and refuses to generalise
  * ranking does not claim EBPW or capability
A skipped test is a P0. Absent inputs are recorded refusals, not omitted cases.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import flash_organ_pivot as fop
from tools.future import flash_schools as fs
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _killed(*names: str) -> list[dict]:
    ids = names or fop.BUILTIN_KILLED_FAMILY_IDS
    return [
        {
            "family": name,
            "any_pass": False,
            "n_beats_q4": 0,
            "n_passed_contract": 0,
            "best_heldout_relative_fro_error": 0.5,
            "killed_by": fop.KILLED_BY,
            "authority": "test",
            "is_comparator": name == "per_expert_q4_control",
        }
        for name in ids
    ]


def _scar(**overrides) -> dict:
    base = {
        "status": "SCOPED_SCAR",
        "organ": fop.EXHAUSTED_ORGAN,
        "tensor": fop.EXHAUSTED_TENSOR,
        "surface": fop.EXHAUSTED_SURFACE,
        "split": {"fit_rows": 819, "heldout_rows": 205, "teacher_rows": 1024},
        "refuses_to_generalise": True,
        "generalises_to_other_expert_tensors": False,
        "generalises_to_other_layers": False,
        "generalises_to_other_flash_organs": False,
        "generalises_to_other_mechanisms": False,
        "any_family_passed_contract": False,
        "any_family_beats_q4": False,
        "killed_by": fop.KILLED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    base.update(overrides)
    return base


def _inv(families: list[dict] | None = None, *, reachable: bool = True, router: int | None = 2_621_440) -> dict:
    rows = families if families is not None else [
        {"family": "routed_experts", "bytes": 246_625_075_200, "tensor_count": 98, "fraction": 0.685},
        {"family": "ngram_embedding", "bytes": 102_400_491_520, "tensor_count": 128, "fraction": 0.284},
        {"family": "linear_attention_hyperconnection", "bytes": 4_820_291_328, "tensor_count": 520, "fraction": 0.013},
        {"family": "embedding_lm_head", "bytes": 2_564_759_064, "tensor_count": 10, "fraction": 0.007},
        {"family": "full_attention", "bytes": 1_337_609_728, "tensor_count": 117, "fraction": 0.0037},
        {"family": "mlp_hyperconnection", "bytes": 1_311_502_048, "tensor_count": 353, "fraction": 0.0036},
        {"family": "shared_expert", "bytes": 481_940_480, "tensor_count": 196, "fraction": 0.0013},
        {"family": "other", "bytes": 457_917_440, "tensor_count": 120, "fraction": 0.0013},
        {"family": "norm", "bytes": 376_320, "tensor_count": 116, "fraction": 1e-6},
    ]
    by_family = {r["family"]: r for r in rows}
    return {
        "census_source": "test",
        "reachable": reachable,
        "families": rows,
        "by_family": by_family,
        "router_tensor_bytes": router,
        "router_source": "test_overlay" if router else "unavailable",
        "specimen_bytes": 359_999_963_128,
        "n_census_families": len(rows),
        "largest_tensors": [],
    }


def _screen() -> dict:
    return {
        "any_family_passed_contract": False,
        "promotion_allowed": False,
        "n_fit": 819,
        "n_heldout": 205,
        "q4_error": 0.1014202494182455,
        "harness": {"ok": True},
        "specimen": {
            "tensor": fop.EXHAUSTED_TENSOR,
            "n_experts": 314,
            "weight_shape": [314, 1280, 2560],
        },
        "state": {"rows": 1024},
        "families": [
            {
                "family": name,
                "any_pass": False,
                "n_beats_q4": 0,
                "n_passed_contract": 0,
                "wins_the_screen": False,
                "rows": [
                    {
                        "scored": True,
                        "heldout_relative_fro_error": (
                            0.10142 if name == "per_expert_q4_control"
                            else 1.18497 if name == "sparse_residual_on_cheap_backbone"
                            else 0.56
                        ),
                        "algebra": (
                            "y_e = W_shared @ x + U_e @ (V_e @ x)"
                            if name == "sparse_residual_on_cheap_backbone"
                            else None
                        ),
                    }
                ],
            }
            for name in fop.BUILTIN_KILLED_FAMILY_IDS
        ],
    }


def test_build_emits_sealed_receipt():
    out = fop.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "FLASH_ORGAN_PIVOT.json"
    assert doc["schema"] == fop.SCHEMA
    assert doc["version"] == 1
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["status"] == "STATIC_PIVOT_RANKING"
    assert doc["status_is_not_a_causal_claim"] is True
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["resident_callable"]["frontier"] == "FT.MODEL_REPRESENTATION.meta-gates-3-9"
    assert doc["resident_callable"]["receipt"] == "receipts/future/FLASH_ORGAN_PIVOT.json"


def test_module_parses():
    src = Path(fop.__file__).read_text()
    ast.parse(src)
    assert "raise NotImplementedError" not in src
    assert "\npass\n" not in src
    assert "TODO" not in src
    assert "skip(" not in src


def test_restatement_of_killed_family_is_refused_by_name():
    """NEGATIVE CONTROL: a restatement must refuse, naming the family and the receipt."""
    scar = _scar()
    killed = _killed()
    probe = {
        "id": "PROBE-SHARED-LATENT",
        "school": "ROUTED_EXPERTS",
        "family": "shared_input_latent_plus_expert_local_output_readout",
        "mechanism": "more ranks of shared input latent plus expert local output readout",
        "organ": fop.EXHAUSTED_ORGAN,
        "tensor": fop.EXHAUSTED_TENSOR,
    }
    with pytest.raises(fop.RestatementRefused) as ei:
        fop.refuse_if_restatement(probe, scar, killed)
    err = ei.value
    assert "REFUSED" in str(err)
    assert err.killed_family == "shared_input_latent_plus_expert_local_output_readout"
    assert err.killed_by == fop.KILLED_BY
    assert "RIVAL_CODEC_SCREEN.json" in err.killed_by

    verdict = fop.restatement_verdict(probe, scar, killed)
    assert verdict is not None
    assert verdict["status"] == "REFUSED_RESTATEMENT"
    assert verdict["killed_family"] == "shared_input_latent_plus_expert_local_output_readout"
    assert verdict["killed_by"] == fop.KILLED_BY


def test_unlabelled_low_rank_sharing_on_gate_up_is_refused():
    """NEGATIVE CONTROL: a renamed low-rank share on the exhausted surface is still refused."""
    scar = _scar()
    killed = _killed()
    probe = {
        "id": "PROBE-NEW-LABEL",
        "school": "ROUTED_EXPERTS",
        "family": "clever_new_codec",
        "mechanism": "low-rank sharing under a new label",
        "organ": fop.EXHAUSTED_ORGAN,
    }
    with pytest.raises(fop.RestatementRefused) as ei:
        fop.refuse_if_restatement(probe, scar, killed)
    assert ei.value.killed_family == "shared_input_latent_plus_expert_local_output_readout"


@pytest.mark.parametrize(
    "family",
    [
        "common_left_subspace_plus_expert_local_core",
        "common_right_subspace_plus_expert_local_core",
        "clustered_subspaces_route_conditioned",
        "dictionary_plus_per_expert_sparse_residual",
        "expert_local_small_core_plus_shared_decoder",
        "sparse_residual_on_cheap_backbone",
        "per_expert_q3_control",
        "per_expert_q2_control",
    ],
)
def test_each_killed_family_is_refused_on_gate_up(family):
    scar = _scar()
    killed = _killed()
    probe = {
        "id": f"PROBE-{family}",
        "school": "ROUTED_EXPERTS",
        "family": family,
        "mechanism": family.replace("_", " "),
        "organ": fop.EXHAUSTED_ORGAN,
    }
    with pytest.raises(fop.RestatementRefused) as ei:
        fop.refuse_if_restatement(probe, scar, killed)
    assert ei.value.killed_family == family
    assert "RIVAL_CODEC_SCREEN.json" in ei.value.killed_by


def test_same_family_on_down_proj_is_not_a_restatement():
    """NEGATIVE CONTROL of the refusal: the scar is scoped; down_proj is a different tensor."""
    scar = _scar()
    killed = _killed()
    probe = {
        "id": "LIVE-DOWN-LEFT",
        "school": "ROUTED_EXPERTS",
        "family": "common_left_subspace_plus_expert_local_core",
        "mechanism": "common left subspace on down_proj",
        "organ": "layer_4.routed_experts.down_proj",
        "surface": "down_proj",
    }
    fop.refuse_if_restatement(probe, scar, killed)  # must not raise
    assert fop.restatement_verdict(probe, scar, killed) is None
    row = fop.classify_candidate(probe, scar=scar, killed=killed, source_bytes=100)
    assert row["status"] == "RANKED"


def test_ngram_product_codebooks_are_not_a_gate_up_restatement():
    scar = _scar()
    killed = _killed()
    probe = {
        "id": "LIVE-NGRAM-PQ",
        "school": "NGRAM",
        "family": "product_codebooks",
        "mechanism": "product codebooks of the n-gram table",
        "organ": "ngram_embedding",
        "surface": "n-gram table lookup",
    }
    fop.refuse_if_restatement(probe, scar, killed)
    row = fop.classify_candidate(probe, scar=scar, killed=killed, source_bytes=1024)
    assert row["status"] == "RANKED"
    assert row["status"] != "REFUSED_RESTATEMENT"


def test_organ_with_no_census_bytes_is_unrankable():
    """NEGATIVE CONTROL: no census bytes → UNRANKABLE, not a guessed rank."""
    empty = _inv(families=[], reachable=True, router=None)
    row = fop.census_bytes_for("NGRAM", empty)
    assert row["rankable"] is False
    assert row["bytes"] is None
    with pytest.raises(fop.UnrankableOrgan) as ei:
        fop.require_census_bytes(row)
    assert "UNRANKABLE" in str(ei.value)
    assert ei.value.school == "NGRAM"

    ranked = fop.rank_school(
        "NGRAM",
        inventory=empty,
        scar=_scar(),
        killed=_killed(),
    )
    assert ranked["status"] == "UNRANKABLE"
    assert ranked["rank"] is None
    assert ranked["expected_ig_per_cost_milli"] is None
    assert "guess" in (ranked.get("unrankable_reason") or "").lower() or "no census" in (ranked.get("unrankable_reason") or "").lower()


def test_function_organs_without_bytes_are_unrankable_not_zeroed():
    inv = _inv()
    for school in ("KV_STATE", "DECODING", "MTP_SPECULATION"):
        row = fop.rank_school(school, inventory=inv, scar=_scar(), killed=_killed())
        assert row["status"] == "UNRANKABLE", school
        assert row["expected_ig_per_cost_milli"] is None
        assert row["function_organ"] is True


def test_override_zero_bytes_is_unrankable_even_when_census_exists():
    inv = _inv()
    row = fop.rank_school(
        "NGRAM",
        inventory=inv,
        scar=_scar(),
        killed=_killed(),
        source_bytes_override=0,
    )
    assert row["status"] == "UNRANKABLE"
    assert row["rank"] is None


def test_scar_is_scoped_and_refuses_to_generalise():
    scar = fop.scoped_scar(_screen(), screen_rel=fop.RIVAL_REL)
    assert scar["status"] == "SCOPED_SCAR"
    assert scar["organ"] == fop.EXHAUSTED_ORGAN
    assert scar["surface"] == fop.EXHAUSTED_SURFACE
    assert scar["split"]["fit_rows"] == 819
    assert scar["split"]["heldout_rows"] == 205
    assert scar["split"]["teacher_rows"] == 1024
    assert scar["refuses_to_generalise"] is True
    assert scar["generalises_to_other_expert_tensors"] is False
    assert scar["generalises_to_other_layers"] is False
    assert scar["generalises_to_other_flash_organs"] is False
    assert scar["generalises_to_other_mechanisms"] is False
    assert scar["any_family_passed_contract"] is False
    assert scar["any_family_beats_q4"] is False
    assert scar["not_a_capability_result"] is True
    assert scar["not_physical_ebpw"] is True
    assert scar["residual_on_shared_expert_worse_than_predicting_zero"] is True


def test_absent_screen_does_not_invent_a_pass():
    """NEGATIVE CONTROL: missing screen is REFUSED_UNAVAILABLE, not a generalised scar."""
    scar = fop.scoped_scar(None, screen_rel=None)
    assert scar["status"] == "REFUSED_UNAVAILABLE"
    assert scar["refuses_to_generalise"] is True
    assert scar["any_family_passed_contract"] is None
    assert scar["split"] is None


def test_ranking_does_not_claim_ebpw_or_capability():
    ranking = fop.rank_all(inventory=_inv(), screen=_screen(), screen_rel=fop.RIVAL_REL)
    for row in ranking["schools"]:
        assert row.get("not_ebpw") is True
        assert row.get("not_capability") is True
        assert "ebpw" not in (row.get("status") or "").lower()
        assert row.get("evidence_class") == "STATIC_ONLY"
        assert row.get("gpu_authority") is False
    for row in ranking["ranked"]:
        assert row["status"] == "RANKED"
        assert isinstance(row["expected_ig_per_cost_milli"], int)
        assert "capability_result" not in row
        assert row.get("not_a_measurement") is True


def test_ngram_outranks_tiny_protected_islands_when_census_present():
    ranking = fop.rank_all(inventory=_inv(), screen=_screen(), screen_rel=fop.RIVAL_REL)
    by = {r["school"]: r for r in ranking["ranked"]}
    assert "NGRAM" in by
    assert "NORMALIZATION" in by
    assert by["NGRAM"]["rank"] < by["NORMALIZATION"]["rank"]
    assert by["NGRAM"]["expected_ig_per_cost_milli"] > by["NORMALIZATION"]["expected_ig_per_cost_milli"]
    assert by["NGRAM"]["rank"] < by["FULL_ATTENTION"]["rank"]
    # Density of gate_up is not a license to rank more ranks of killed families.
    refused_ids = {p["id"] for p in ranking["restatement_probes"] if p["status"] == "REFUSED_RESTATEMENT"}
    assert "PROBE-GATEUP-SHARED-LATENT-RANK128" in refused_ids
    live_ids = {p["id"] for p in ranking["restatement_probes"] if p["status"] != "REFUSED_RESTATEMENT"}
    assert "PROBE-NGRAM-PRODUCT-CODEBOOKS" in live_ids
    assert "PROBE-DOWN-COMMON-LEFT" in live_ids


def test_ranking_can_return_the_negative():
    """A validator nobody has watched reject is not a validator.

    Two negatives: restatement refuses, and a live organ can rank below another.
    """
    ranking = fop.rank_all(inventory=_inv(), screen=_screen(), screen_rel=fop.RIVAL_REL)
    assert ranking["n_restatement_probes_refused"] >= 8
    ranks = [r["rank"] for r in ranking["ranked"]]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)
    # ROUTER overlay is tiny; NGRAM must beat it when both rank.
    by = {r["school"]: r for r in ranking["ranked"]}
    if "ROUTER" in by and "NGRAM" in by:
        assert by["NGRAM"]["expected_ig_per_cost_milli"] > by["ROUTER"]["expected_ig_per_cost_milli"]


def test_prove_negative_controls_actually_fire():
    inv = _inv()
    scar = _scar()
    killed = _killed()
    proof = fop.prove_negative_controls(scar, killed, inv)
    assert proof["restatement_refused"] is True
    assert proof["restatement_killed_family"] == "shared_input_latent_plus_expert_local_output_readout"
    assert "RIVAL_CODEC_SCREEN.json" in (proof["restatement_killed_by"] or "")
    assert proof["unrankable_without_bytes"] is True
    assert proof["down_proj_common_left_not_refused"] is True
    assert proof["ngram_product_codebooks_not_refused"] is True
    assert proof["watched_fail"] is True


def test_receipt_carries_scoped_scar_and_watched_fails():
    doc = json.loads(fop.build().read_text())
    scar = doc["scoped_scar"]
    assert scar["organ"] == fop.EXHAUSTED_ORGAN
    assert scar["surface"] == fop.EXHAUSTED_SURFACE
    assert scar["refuses_to_generalise"] is True
    if scar["status"] == "SCOPED_SCAR":
        assert scar["split"]["fit_rows"] == 819
        assert scar["split"]["heldout_rows"] == 205
        assert scar["any_family_passed_contract"] is False
        assert scar["any_family_beats_q4"] is False
    proof = doc["negative_control"]
    assert proof["restatement_refused"] is True
    assert proof["unrankable_without_bytes"] is True
    claims = doc["claims"]
    assert claims["ranking_is_ebpw"] is False
    assert claims["ranking_is_capability"] is False
    assert claims["ranking_is_hardware"] is False
    assert claims["ranking_is_expected_information_gain_per_cost"] is True
    assert claims["scar_is_scoped"] is True
    assert claims["scar_organ"] == fop.EXHAUSTED_ORGAN
    for row in doc["ranked_schools"]:
        assert row["not_ebpw"] is True
        assert row["not_capability"] is True
    # Unspecified killed-family probes on gate_up must show up refused.
    refused = [p for p in doc["restatement_probes"] if p["status"] == "REFUSED_RESTATEMENT"]
    assert refused
    assert all("killed_family" in p for p in refused)
    assert all("RIVAL_CODEC_SCREEN.json" in p["killed_by"] for p in refused)


def test_census_unreachable_does_not_guess_zero_for_weight_organs():
    """Unreachability is not a zero. Weight organs become UNRANKABLE, not rank-0."""
    inv = _inv(families=[], reachable=False, router=None)
    row = fop.census_bytes_for("NGRAM", inv)
    assert row["rankable"] is False
    assert row["provenance"] == "census_unreachable"
    ranked = fop.rank_school("NGRAM", inventory=inv, scar=_scar(), killed=_killed())
    assert ranked["status"] == "UNRANKABLE"


def test_router_overlay_is_cited_not_guessed():
    inv = _inv(families=[], reachable=True, router=2_621_440)
    row = fop.census_bytes_for("ROUTER", inv)
    assert row["rankable"] is True
    assert row["bytes"] == 2_621_440
    assert "overlay" in row["provenance"] or "overlay" in row["reason"]


def test_ig_score_refuses_to_look_like_ebpw():
    ngram = 102_400_491_520
    ig = fop.expected_ig_per_cost_milli(ngram, 8, 1)
    assert isinstance(ig, int)
    # MiB * weight; not bits-per-weight.
    assert ig == (ngram // (1024 * 1024)) * 8
    tiny = fop.expected_ig_per_cost_milli(376_320, 1, 2)
    assert tiny == 0  # 376 kB < 1 MiB
    with pytest.raises(ValueError):
        fop.expected_ig_per_cost_milli(100, 1, 0)


def test_on_exhausted_surface_does_not_eat_other_layers_or_down():
    scar = _scar()
    assert fop.on_exhausted_surface({"organ": fop.EXHAUSTED_ORGAN}, scar) is True
    assert fop.on_exhausted_surface({"tensor": fop.EXHAUSTED_TENSOR}, scar) is True
    assert fop.on_exhausted_surface({"organ": "layer_4.routed_experts.down_proj"}, scar) is False
    assert fop.on_exhausted_surface({"organ": "layer_12.routed_experts.gate_up_proj"}, scar) is False
    assert fop.on_exhausted_surface({"organ": "ngram_embedding"}, scar) is False


def test_unknown_school_fails_closed():
    with pytest.raises(fs.UnknownSchoolError):
        fop.census_bytes_for("NOT_A_SCHOOL", _inv())


def test_hardware_fields_absent_from_receipt():
    doc = json.loads(fop.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                assert k not in HARDWARE_FIELDS or not isinstance(v, (int, float)), here
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    _assert_no_hardware_claims(doc)


def test_next_workunits_refuse_gate_up_resweep():
    ranking = fop.rank_all(inventory=_inv(), screen=_screen(), screen_rel=fop.RIVAL_REL)
    units = fop.next_workunits(ranking)
    assert units
    refuse = next(u for u in units if u["id"].endswith("do_not_resweep_gate_up"))
    assert refuse["action"] == "REFUSE"
    assert "gate_up" in refuse["reason"]
    for u in units:
        assert u.get("gpu_authority") is False


def test_q4_is_comparator_not_a_contract_pass():
    scar = fop.scoped_scar(_screen(), screen_rel=fop.RIVAL_REL)
    assert scar["q4_is_best_tested_local_comparator"] is True
    assert scar["any_family_passed_contract"] is False
    killed = fop.killed_families_from_screen(_screen())
    q4 = next(k for k in killed if k["family"] == "per_expert_q4_control")
    assert q4["is_comparator"] is True
    assert q4["any_pass"] is False
