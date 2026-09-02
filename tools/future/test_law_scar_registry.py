"""Scoped law/scar registry: one local failure must not become a global ban.

Named real receipts (not fixtures):

* Scar: receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#NNS-004
  (shared_basis_across_experts; measured on qwen3-80b / gpt-oss-120b:F0 /
  qwen3-235b-a22b:F1; NOT measured on dsv4f)
* Law: receipts/future/ODYSSEY2_LAW_STORE.json#LAW-COLD-CONTROL-BEAT-TRANSFER-SEED
  (MODEL_LOCAL on Qwen/Qwen3-30B-A3B moe_expert)
* Law machine binding: receipts/future/CLAIM_SCOPE.json#LAW-MLP-ARITHMETIC-SENSITIVITY

The load-bearing behaviour is scope_covers / scar_blocks_candidate /
retrieve_law. scar_reevaluator.consult_candidate is the HCLI-facing call
site: it invokes autonomy_scars.consult.
"""
from __future__ import annotations

import json

from tools.future import autonomy_scars as asc
from tools.future import scar_reevaluator as sr


NOETIC_REL = "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json"
O2_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"
CLAIM_REL = "receipts/future/CLAIM_SCOPE.json"


def _noetic_nns004() -> dict:
    doc = asc._load_json(NOETIC_REL)
    hits = [e for e in doc["entries"] if e.get("id") == "NNS-004"]
    assert hits, "NNS-004 missing from named receipt"
    return hits[0]


def _o2_law(law_id: str) -> dict:
    doc = asc._load_json(O2_REL)
    hits = [row for row in doc["laws"] if row.get("law_id") == law_id]
    assert hits, f"{law_id} missing from named receipt"
    return hits[0]


def _claim_law(law_id: str) -> dict:
    doc = asc._load_json(CLAIM_REL)
    hits = [row for row in doc["laws"] if row.get("law_id") == law_id]
    assert hits, f"{law_id} missing from named receipt"
    return hits[0]


def test_out_of_scope_retry_is_not_blocked_by_scoped_scar():
    """Over-generalization guard: NNS-004 does not ban a DSV4F retry.

    STATIC. The receipt itself says NOT measured on dsv4f.
    """
    scar = asc.scar_from_noetic_entry(_noetic_nns004(), NOETIC_REL)
    in_scope = {
        "model": "qwen3-80b",
        "organ": "gate",
        "hypothesis_family": "cross_expert_structure",
    }
    out_of_scope = {
        "model": "deepseek-v4-flash",
        "organ": "gate",
        "hypothesis_family": "cross_expert_structure",
    }
    blocked = asc.scar_blocks_candidate(scar, in_scope)
    assert blocked is not None, "in-scope Q80 cross-expert retry must be blocked"
    assert blocked["scar_id"] == "NNS-004"
    assert blocked["refused"] is True

    allowed = asc.scar_blocks_candidate(scar, out_of_scope)
    assert allowed is None, (
        f"DSV4F retry was banned by a scar that was not measured on dsv4f: {allowed}"
    )

    verdict = asc.consult(out_of_scope, scars=[scar], laws=[])
    assert verdict["blocked"] is False
    assert verdict["entry_point"] == "tools.future.autonomy_scars.consult"
    assert any(row["scar_id"] == "NNS-004" for row in verdict["out_of_scope_not_blocked"])


def test_law_retrieved_outside_measured_scope_is_refused_or_flagged():
    """A MODEL_LOCAL law is not silently generalized to another parent."""
    law = asc.law_from_odyssey2_record(_o2_law("LAW-COLD-CONTROL-BEAT-TRANSFER-SEED"), O2_REL)
    in_scope = {
        "model": "Qwen/Qwen3-30B-A3B",
        "organ": "moe_expert",
        "lattice": "MODEL_LOCAL",
    }
    other_parent = {
        "model": "glm-5.2",
        "organ": "moe_expert",
        "lattice": "MODEL_LOCAL",
    }
    widened = {
        "model": "Qwen/Qwen3-30B-A3B",
        "organ": "moe_expert",
        "lattice": "GENERIC_VERIFIED",
    }
    hit = asc.retrieve_law(law.identity, in_scope, laws=[law])
    assert hit["found"] is True
    assert hit["usable"] is True
    assert hit["refused"] is False
    assert hit["law"]["identity"] == "LAW-COLD-CONTROL-BEAT-TRANSFER-SEED"

    flagged = asc.retrieve_law(law.identity, other_parent, laws=[law])
    assert flagged["found"] is True
    assert flagged["usable"] is False
    assert flagged["refused"] is True
    assert flagged["flag"] == "OUT_OF_SCOPE"

    widened_hit = asc.retrieve_law(law.identity, widened, laws=[law])
    assert widened_hit["usable"] is False
    assert widened_hit["flag"] == "OUT_OF_SCOPE"

    try:
        asc.retrieve_law(law.identity, other_parent, laws=[law], require_in_scope=True)
        raise AssertionError("out-of-scope retrieve_law must raise when required")
    except asc.OutOfScopeError as err:
        assert err.identity == law.identity
        assert err.flag == "OUT_OF_SCOPE"


def test_real_round_trip_from_named_receipts():
    """Round-trip a real scar and a real law. Named receipts, not fixtures."""
    raw_scar = _noetic_nns004()
    scar = asc.scar_from_noetic_entry(raw_scar, NOETIC_REL)
    again = asc.round_trip_scar(scar)
    assert again == scar
    assert again.identity == "NNS-004"
    assert again.source_path == NOETIC_REL
    assert again.reopen_if == raw_scar["reopen_condition"]
    assert again.reason == raw_scar["claim_refuted"]
    assert again.failed_mechanism == "shared_basis_across_experts"
    assert "qwen3-80b" in again.scope.models
    assert "qwen3-235b-a22b" in again.scope.models
    assert "gpt-oss-120b" in again.scope.models
    assert "deepseek-v4-flash" in again.scope.excluded_models
    assert "deepseek-v4-flash" not in again.scope.models
    assert again.evidence
    assert any("QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json" in ev for ev in again.evidence)
    blob = json.dumps(again.to_dict(), sort_keys=True)
    assert "NNS-004" in blob
    assert "reopen_if" in blob

    raw_law = _o2_law("LAW-COLD-CONTROL-BEAT-TRANSFER-SEED")
    law = asc.law_from_odyssey2_record(raw_law, O2_REL)
    law_again = asc.round_trip_law(law)
    assert law_again == law
    assert law_again.identity == "LAW-COLD-CONTROL-BEAT-TRANSFER-SEED"
    assert law_again.claim == raw_law["statement"]
    assert "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json" in law_again.evidence
    assert law_again.scope.lattice == "MODEL_LOCAL"
    assert "qwen3-30b-a3b" in law_again.scope.models
    assert law_again.falsifier == raw_law["counterexample_requirement"]
    assert law_again.confidence["value"] == raw_law["transfer_confidence"]["value"]
    assert law_again.machine_binding == raw_law["source_device"]
    assert "identity" in law_again.to_dict()
    assert "scar_ids" in law_again.to_dict()
    assert "transfer_tests" in law_again.to_dict()

    raw_claim = _claim_law("LAW-MLP-ARITHMETIC-SENSITIVITY")
    claim_law = asc.law_from_claim_scope_record(raw_claim, CLAIM_REL)
    claim_again = asc.round_trip_law(claim_law)
    assert claim_again == claim_law
    assert "M3" in claim_again.machine_binding or "m3" in claim_again.machine_binding.lower()
    assert claim_again.experiment_identity
    assert claim_again.scope.machine
    other_machine = asc.retrieve_law(
        claim_law.identity,
        {
            "model": "qwen3.8-27b",
            "organ": "mlp",
            "machine": "apple_host_cpu",
            "lattice": "MODEL_LOCAL",
        },
        laws=[claim_law],
    )
    assert other_machine["flag"] == "OUT_OF_SCOPE"
    assert other_machine["usable"] is False


def test_consult_candidate_is_the_hcli_call_site_and_respects_scope():
    """scar_reevaluator.consult_candidate must invoke autonomy_scars.consult."""
    out = sr.consult_candidate(
        {
            "model": "deepseek-v4-flash",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert out["consults"] == "tools.future.autonomy_scars.consult"
    assert out["entry_point"] == "tools.future.scar_reevaluator.consult_candidate"
    assert out["blocked"] is False
    assert any(row.get("scar_id") == "NNS-004" for row in out["out_of_scope_not_blocked"])

    banned = sr.consult_candidate(
        {
            "model": "qwen3-80b",
            "organ": "gate",
            "hypothesis_family": "cross_expert_structure",
        }
    )
    assert banned["blocked"] is True
    assert any(row.get("scar_id") == "NNS-004" for row in banned["blocked_by"])


def test_general_physical_campaign_scar_still_blocks_any_parent():
    """Unbound process scars remain global; the guard is not 'never block'."""
    registry = asc.load_registry()
    hits = [s for s in registry.scars if s.identity == "PREFILL_OVER_GENERATED_TOKEN_DENOMINATOR"]
    assert hits, "campaign scar missing from registry"
    scar = hits[0]
    for model in ("qwen3.8-27b", "glm-5.2", "deepseek-v4-flash"):
        hit = asc.scar_blocks_candidate(
            scar,
            {"model": model, "hypothesis_family": "prefill_over_generated_token_denominator"},
        )
        assert hit is not None, f"GENERAL_PHYSICAL scar failed to block {model}"
