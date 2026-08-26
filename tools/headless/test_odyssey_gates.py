"""Structural guards on the gates that had no test of their own.

Each of these asserts the property the gate exists to have, not that a file is present.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
sys.path.insert(0, str(RH.parent))


def load(name):
    p = RH / f"{name}.json"
    assert p.exists(), f"missing receipt {name}"
    return json.load(open(p))


# ------------------------------------------------------------ capability contract

def test_threshold_comes_from_the_incumbent_not_the_candidate():
    d = load("QWEN_CAPABILITY_QUALIFICATION")
    assert d["threshold_rule"]["basis"].startswith("the artifact production runs today")
    inc = next(r for r in d["results"] if r["role"] == "incumbent (production)")
    for ax, cell in inc["per_axis"].items():
        assert cell["incumbent_rate"] == inc["per_axis"][ax]["rate"]


def test_at_least_one_body_fails_the_contract():
    """A contract nothing can fail is not a contract."""
    d = load("QWEN_CAPABILITY_QUALIFICATION")
    assert d["verdict"]["fails_contract"], "no candidate fails: the threshold is vacuous"


def test_the_dead_body_is_recorded_as_dead():
    d = load("QWEN_CAPABILITY_QUALIFICATION")
    dead = next(r for r in d["results"] if "2.5970" in r["density"])
    assert dead["overall"]["rate"] < 0.15
    assert len(dead["axes_below_threshold"]) >= 5
    assert "hygiene" in d["vacuous_if_empty"]


# ------------------------------------------------------------ cross-model laws

def test_architecture_general_requires_two_distinct_families():
    """Promotion is earned by measurement on a second family, never waived."""
    d = load("CROSS_MODEL_LAWS")
    fams = set(d["architecture_families_measured"])
    ag = [l for l in d["laws"] if l["level"] == "ARCHITECTURE_GENERAL"]
    if ag:
        assert len(fams) >= 2, f"ARCHITECTURE_GENERAL claimed on families {fams}"
        for l in ag:
            models = {m for m in l["measured_on_models"]}
            assert len(models) >= 2, l["id"]
    assert d["promotion_rule"]["min_architecture_families"]["ARCHITECTURE_GENERAL"] == 2


def test_the_promoted_law_cites_the_second_family():
    d = load("CROSS_MODEL_LAWS")
    ag = [l for l in d["laws"] if l["level"] == "ARCHITECTURE_GENERAL"]
    for l in ag:
        assert any("FALCON" in c.upper() or "falcon" in c for c in l["evidence"]), l["id"]
        fam = json.load(open(RH / "MATCHED_BITS_FALCON_H1.json"))
        assert fam["law_holds_here"] is True
        assert fam["n_tiers_seeded_wins"] == fam["n_tiers"]


def test_the_2_25_floor_is_qwen_specific_and_says_where_it_broke():
    d = load("CROSS_MODEL_LAWS")
    law = next(l for l in d["laws"] if l["id"] == "LAW-MLP-FLOOR-2.25")
    assert law["level"] == "QWEN_SPECIFIC"
    assert "Qwen/Qwen3-30B-A3B" in law.get("refuted_on_models", [])


def test_every_law_says_why_it_is_not_higher():
    d = load("CROSS_MODEL_LAWS")
    for l in d["laws"]:
        assert l["why_not_higher"].strip(), l["id"]
        assert l["evidence"], l["id"]


# ------------------------------------------------------------ compounding

def test_all_five_demonstrations_and_no_adversary_won():
    d = load("MODEL_2_COMPOUNDING")
    assert d["n_demonstrations"] == 5
    assert d["n_happened"] == 5
    assert d["n_adversaries_won"] == 0
    for dem in d["demonstrations"]:
        assert dem["adversary"]["check"], dem["demonstration"]


def test_the_kernel_reuse_claim_states_its_limitation():
    """Kernels were SELECTED, not executed on specimen #2. That must be in the receipt."""
    d = load("MODEL_2_COMPOUNDING")
    k = next(x for x in d["demonstrations"] if "kernels reused" in x["demonstration"])
    assert "honest_limitation" in k["adversary"]
    assert "not kernel execution" in k["adversary"]["honest_limitation"]


def test_stop_rule_is_encoded():
    d = load("MODEL_2_COMPOUNDING")
    assert "stop scaling Odyssey" in d["directive_stop_rule"]
    assert d["stop_rule_triggered"] is False


# ------------------------------------------------------------ transfer proof

def test_the_unfavourable_result_is_reported():
    d = load("ODYSSEY_TRANSFER_PROVEN")
    assert d["honest_note"], "the losing metric must be in the receipt"
    assert d["cold"]["evaluations_run"] < d["transfer"]["evaluations_run"], \
        "under the loose target cold wins; if that flips, the note is stale"


def test_matched_bits_is_actually_matched():
    d = load("ODYSSEY_TRANSFER_PROVEN")
    for t in d["matched_bits_comparison"]["tiers"]:
        assert t["seeded_is_better"], t["bpw"]
        assert t["error_ratio_generic_over_seeded"] > 1.0


def test_activations_are_real_and_the_expert_mapping_was_verified():
    d = load("ODYSSEY_TRANSFER_PROVEN")["activations"]
    assert d["real_not_synthetic"] is True
    assert d["expert_mapping_verified"] is True
    assert d["routed"] is True


# ------------------------------------------------------------ physical compiler

def test_every_collapse_is_numerically_checked():
    d = load("PHYSICAL_GRAPH_COMPILER")
    cols = d["physical_operator_graph"]["collapses"]
    assert cols
    for c in cols:
        assert c["semantic_justification"].strip()
        assert c.get("numerically_equivalent") or c.get("selection_identical"), c["collapse"]
        assert c["n_physical_nodes"] < c["n_source_nodes"]


def test_interactions_are_measured_not_asserted():
    d = load("PHYSICAL_GRAPH_COMPILER")
    assert len(d["interactions"]) >= 3
    for i in d["interactions"]:
        assert i["measured"], i["relation"]
        assert i["evidence"]


# ------------------------------------------------------------ pipeline + gate

def test_blocked_stages_are_blocked_not_skipped():
    d = load("NOETIC_COMPILER_PIPELINE")
    blocked = [s for s in d["stages"] if s["status"] == "BLOCKED"]
    assert blocked
    for b in blocked:
        assert b["missing_capability"] and b["what_would_unblock"]
    assert d["n_manual_interventions"] == 0


def test_no_device_profile_is_claimed_qualified_without_a_measurement():
    d = load("NOETIC_COMPILER_PIPELINE")
    for p in d["device_profiles"]:
        if p["status"] != "QUALIFIED":
            assert p["reason"].strip()


def test_the_retirement_gate_refuses_and_deleted_nothing():
    d = load("QWEN_RETIREMENT_GATE")
    assert d["RETIREMENT_READY"] is False
    assert d["refusals"]
    assert d["nothing_was_deleted"] is True
    assert d["preserve_set_complete"] is True
    for item in d["preserve_set"]:
        assert item["present"], item["what"]
    for r in d["relegatable"]:
        if r["present"]:
            assert r["reversible"], r["what"]


# ------------------------------------------------------------ adversarial sweep

def test_the_sweep_can_catch_a_weak_gate():
    d = load("ODYSSEY_ADVERSARIAL_SWEEP")
    sv = d["self_validation"]
    assert sv["weak_gate_caught"] is True
    assert sv["strong_gate_survives"] is True
    assert sv["valid"] is True


def test_every_gate_took_all_ten_attacks():
    d = load("ODYSSEY_ADVERSARIAL_SWEEP")
    assert len(d["attacks"]) == 10
    for r in d["results"]:
        assert r["n_attacks"] == 10, r["id"]


def test_the_independence_limitation_is_stated():
    d = load("ODYSSEY_ADVERSARIAL_SWEEP")
    assert "not the gate's implementer" in d["independence_limitation"]
    assert "402" in d["independence_limitation"]


# ------------------------------------------------------------ learning curve

def test_learning_curve_marks_unmeasured_rather_than_guessing():
    d = load("ODYSSEY_LEARNING_CURVE")
    assert d["n_specimens"] >= 2
    for row in d["rows"]:
        for k, v in row.items():
            if isinstance(v, dict) and v.get("status") == "UNMEASURED":
                assert v["reason"].strip()
                assert v["value"] is None


def test_learning_curve_states_what_it_does_not_yet_show():
    """Three rows now, including a second architecture family. The scope note must still
    name what is missing rather than implying the curve is complete."""
    d = load("ODYSSEY_LEARNING_CURVE")
    assert d["n_specimens"] >= 3
    assert "do NOT yet show" in d["honest_scope"] or "do not establish" in d["honest_scope"]
    assert d["marginal_transfer_specimen_2_to_3"]["architecture_family"].endswith(
        "(first specimen outside the family)")


# ------------------------------------------------------------ frozen density field

def test_reported_density_tracks_the_bytes_actually_written():
    """A density that cannot move is not a measurement.

    complete_ebpw was built from hardcoded per-organ rates, so repacking the same model
    with attention at q4 instead of q3 added 1,288,519,664 bytes and the reported figure
    did not change at all. It agreed with the physical figure to seven decimals for the
    body the constants were written for, which is why it survived.
    """
    import re
    src = (REPO / "tools/headless/whole_model_native.py").read_text()
    assert "complete_ebpw = physical_ebpw" in src, \
        "complete_ebpw must follow the payload bytes, not the design constants"
    assert "design_ebpw_from_hardcoded_rates" in src
    assert "agree_within_1e_3" in src

    # the two artifacts on disk are the regression case: same design constants, different bytes
    import json as _json
    a = _json.load(open("/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors/MIX_REPORT.json"))
    b = _json.load(open("/Users/scammermike/noetic/VARIANT_A_MLP_ONLY/MIX_REPORT.json"))
    assert b["payload_bytes"] > a["payload_bytes"]
    phys_a = a["complete_ebpw_physical_codes_plus_scales"]
    phys_b = b["complete_ebpw_physical_codes_plus_scales"]
    assert phys_b > phys_a, "the physical figure must move when the bytes move"
    # and the old frozen field is identical across both, which is the bug being pinned
    assert abs(a["complete_ebpw"] - b["complete_ebpw"]) < 1e-9, \
        "these two receipts predate the fix; they pin the frozen-field regression"


# ------------------------------------------------------------ canonical authority

def test_canonical_libraries_are_not_silently_downgraded():
    """Two producers write these paths; whichever ran last wins.

    tools/headless/genome_libraries.py emits the v1 form of REPRESENTATION_LIBRARY.json
    and KERNEL_LIBRARY.json to the same canonical paths this campaign's v2 producers own,
    and test_genome_libraries.py runs it. A full `pytest tools/` therefore reverts them,
    which silently drops 13 representation families and every kernel completeness field.
    """
    for name, want in (("REPRESENTATION_LIBRARY", ".representation_library.v2"),
                       ("KERNEL_LIBRARY", ".kernel_library.v2")):
        d = load(name)
        assert str(d["schema"]).endswith(want), (
            f"{name} is at {d['schema']}, not {want}. Something rewrote the canonical "
            f"path -- rerun tools/headless/{name.lower()}.py before trusting any gate "
            f"that cites it.")


def test_the_two_v2_libraries_carry_what_v1_does_not():
    r = load("REPRESENTATION_LIBRARY")
    assert r["n_families"] >= 17 and r["v1_preservation"]["lost"] == []
    k = load("KERNEL_LIBRARY")
    assert k["n_complete"] == k["n_kernels"] and k["n_rejected"] == 0


# ------------------------------------------------------------ campaign receipt integrity

CAMPAIGN_RECEIPTS = {
    "ARCHITECTURE_RECOGNIZER": "architecture_recognizer.v1",
    "ORGAN_FRONTIER_MATRIX": "organ_frontier_matrix.v1",
    "REPRESENTATION_LIBRARY": "representation_library.v2",
    "KERNEL_LIBRARY": "kernel_library.v2",
    "SUPEROPERATOR_LIBRARY": "superoperator_library.v1",
    "NOETIC_NEGATIVE_SCIENCE": "negative_science.v2",
    "QWEN_TRANSFER_REPORT": "qwen_transfer_report.v1",
    "QWEN_TRANSFER_REHEARSAL": "transfer_rehearsal.v1",
    "ODYSSEY_TRANSFER_PROVEN": "cold_vs_transfer.v1",
    "MODEL_2_SELECTION": "model2_selection.v1",
    "MODEL_2_COMPOUNDING": "model2_compounding.v1",
    "CROSS_MODEL_LAWS": "cross_model_laws.v1",
    "ODYSSEY_LEARNING_CURVE": "odyssey_learning_curve.v1",
    "EXPERT_FAMILY_GENOME": "expert_family_genome.v1",
    "DOCTOR_TRANSFER": "doctor_transfer.v1",
    "MAXX_RESOURCE_PIPELINE": "maxx_resource_pipeline.v1",
    "LANE_HEARTBEATS": "lane_heartbeats.v1",
    "QWEN_CLEAN_REBUILD": "qwen_clean_rebuild.v1",
    "QWEN_TEXTBOOK_V1": "textbook_trace.v1",
    "MODEL_LAKE_ROLLING_PIPELINE": "model_lake.v1",
    "PHYSICAL_GRAPH_COMPILER": "physical_graph_compiler.v1",
    "NOETIC_COMPILER_PIPELINE": "noetic_compiler_pipeline.v1",
    "QWEN_RETIREMENT_GATE": "qwen_retirement_gate.v1",
    "ODYSSEY_ADVERSARIAL_SWEEP": "adversarial_sweep.v1",
    "QWEN_CAPABILITY_QUALIFICATION": "capability_contract.v1",
    "RECONSTRUCTION_ISOLATION": "reconstruction_isolation.v1",
    "COMPOSITION_ATTRIBUTION": "composition_attribution.v1",
}


def test_every_campaign_receipt_is_present_and_at_its_own_schema():
    """One receipt being rewritten by a foreign producer is not a hypothetical.

    A full `pytest tools/` reverted two of these to a v1 form written by a different tool.
    Nothing failed loudly; the gates citing them simply started pointing at different
    contents. This catches that on the next run instead of by luck.
    """
    wrong = []
    for name, want in CAMPAIGN_RECEIPTS.items():
        p = RH / f"{name}.json"
        if not p.exists():
            wrong.append((name, "MISSING"))
            continue
        d = json.load(open(p))
        if not str(d.get("schema", "")).endswith(want):
            wrong.append((name, f"schema {d.get('schema')} != ...{want}"))
        elif d.get("pass") is False:
            wrong.append((name, "pass=false"))
    assert not wrong, wrong


def test_no_campaign_receipt_claims_to_be_hand_authored():
    for name in CAMPAIGN_RECEIPTS:
        p = RH / f"{name}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        if "hand_authored" in d:
            assert d["hand_authored"] is False, name
