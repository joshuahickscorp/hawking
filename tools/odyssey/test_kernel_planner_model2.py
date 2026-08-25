"""G023 KernelPlanner pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "KERNEL_PLANNER_MODEL2.json"
KL = RH / "KERNEL_LIBRARY.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="planner not run")


def rec():
    return json.load(open(R))


def test_the_planner_names_model_2_as_its_specimen():
    """The stage it replaces never mentioned model #2 at all."""
    assert rec()["specimen"]["repo"] == "Qwen/Qwen3-30B-A3B"


def test_the_organ_graph_comes_from_the_recognizer_not_a_hand_list():
    d = rec()
    assert "ARCHITECTURE_RECOGNIZER" in d["specimen"]["organ_graph_source"]
    census = json.load(open(RH / "ARCHITECTURE_RECOGNIZER.json"))
    organs = set()
    for k in ("specimens", "heldout_specimens"):
        for s in census.get(k, []) or []:
            if (s.get("result") or {}).get("repo") == d["specimen"]["repo"]:
                organs = {o["organ"] for o in s["result"]["organs"]}
    assert {r["organ"] for r in d["organ_plan"]} <= organs


def test_moe_kernels_are_registered_and_pass_the_library_validator():
    lib = json.load(open(KL))
    reg = lib["moe_registration"]
    assert reg["n_added"] == 18
    assert reg["declared_in_shader"] == reg["referenced_by_runtime"] == reg["in_both"]
    assert lib["n_rejected"] == 0
    assert {"moe_expert", "moe_expert_gate_up"} <= set(reg["organs_now_covered"])


def test_registered_kernels_carry_no_invented_measurements():
    """No parity or throughput number exists for these; claiming one would be forged."""
    lib = json.load(open(KL))
    added = set(lib["moe_registration"]["added"])
    for k in lib["kernels"]:
        if k["kernel_identity"] in added:
            for f in ("measurements", "supported_capability_regime", "parity"):
                assert k[f]["kind"] == "ABSENT", (k["kernel_identity"], f)
                assert k[f]["absent_reason"], (k["kernel_identity"], f)


def test_registered_kernels_point_at_a_shader_that_declares_them():
    root = Path(__file__).resolve().parents[2]
    lib = json.load(open(KL))
    added = [k for k in lib["kernels"]
             if k["kernel_identity"] in set(lib["moe_registration"]["added"])]
    assert added
    src = (root / added[0]["shader"]).read_text()
    for k in added:
        assert f"kernel void {k['kernel_identity']}" in src, k["kernel_identity"]


def test_section_71_is_checked_per_organ_AND_representation():
    """Matching on organ name alone reported moe_expert COVERED by 18 kernels while the
    planner had seeded q2_affine, which none of them executes."""
    d = rec()
    for r in d["organ_plan"]:
        assert r["may_be_evaluated"] is (r["selected_representation"] is not None)
        for fam in r["seeded_families_in_score_order"]:
            assert "family" in fam and "n_competent_kernels" in fam
    assert d["section_71_rule"]["rule"].startswith("representations are never evaluated")


def test_the_planner_considers_every_seeded_family_not_only_the_top_one():
    for r in rec()["organ_plan"]:
        assert len(r["seeded_families_in_score_order"]) >= 2, r["organ"]


def test_a_top_seed_without_a_competent_kernel_is_downgraded_not_accepted():
    d = rec()
    moe = next(r for r in d["organ_plan"] if r["organ"] == "moe_expert")
    assert moe["seeded_families_in_score_order"][0]["family"] == "q2_affine"
    assert moe["seeded_families_in_score_order"][0]["n_competent_kernels"] == 0
    assert moe["downgraded_from_top_seed"] is True
    assert moe["selected_representation"] != "q2_affine"
    assert moe["n_competent_kernels"] > 0


# superseded by test_moe_expert_still_cannot_take_f32_passthrough: the refusal is no
# longer keyed on organ type but on measured mass, so the old field no longer exists.


def test_a_norm_organ_may_select_f32_passthrough():
    n = next(r for r in rec()["organ_plan"] if r["organ"] == "rmsnorm")
    assert n["selected_representation"] == "leftover_f32"
    assert n["may_be_evaluated"] is True


def test_covered_organs_with_kernels_are_unqualified_until_parity_runs():
    """Competence is not qualification."""
    for r in rec()["organ_plan"]:
        if r["status"] == "COVERED" and r["n_competent_kernels"]:
            assert "UNQUALIFIED" in r["qualification"]


def test_the_stage_is_complete_and_says_what_that_does_not_mean():
    """Complete planning is not a compilable model: two stages remain blocked and every
    competent kernel still has parity ABSENT."""
    d = rec()
    assert d["n_gaps"] == 0 and d["stage_status"] == "RAN_COMPLETE"
    assert "does NOT mean model #2 can be compiled" in d["honest_summary"]
    assert "parity ABSENT" in d["honest_summary"]


def test_the_pipeline_receipt_reflects_the_new_stage_status():
    p = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    kp = next(s for s in p["stages"] if s["stage"] == "KernelPlanner")
    assert kp["status"] == "AUTOMATIC"
    assert kp["output"]["n_gaps"] == 0
    assert p["n_stages_automatic_on_model_2"] == 6
    blocked = [s["stage"] for s in p["stages"] if s["status"] == "BLOCKED"]
    assert blocked == ["DeviceCompiler", "NoeticExecutable"]


def test_a_selector_alone_cannot_cover_a_gemv_organ():
    """A top-k select touches no weights, so no quantized family may be selected for the
    router on the strength of one. It resolves as a passthrough instead."""
    r = next(x for x in rec()["organ_plan"] if x["organ"] == "moe_router")
    for f in r["seeded_families_in_score_order"]:
        if f["family"] != "leftover_f32":
            assert f["n_weight_bearing"] == 0
            assert f["n_competent_kernels"] == 0
    assert r["selected_representation"] == "leftover_f32"


def test_the_router_resolved_by_fixing_the_upstream_stage_not_this_one():
    """The seed came from the RepresentationPlanner becoming mass-aware, not from this
    stage inventing one."""
    import json as _j
    reh = _j.load(open(RH / "QWEN_TRANSFER_REHEARSAL.json"))
    row = next(r for r in reh["plan"]["organ_plan"] if r["organ"] == "moe_router")
    assert row["mass_aware_seed_added"] is True
    assert "leftover_f32" in [s["family"] for s in row["seeded_representations"]]
    assert row["organ_mass_share"] <= 0.005


def test_only_organs_under_the_ceiling_got_a_mass_aware_seed():
    import json as _j
    reh = _j.load(open(RH / "QWEN_TRANSFER_REHEARSAL.json"))
    for r in reh["plan"]["organ_plan"]:
        if r.get("mass_aware_seed_added"):
            assert r["organ_mass_share"] <= 0.005, r["organ"]


def test_binary_group_is_not_treated_as_binary_sparse_residual():
    """A plain binary kernel has no residual path."""
    import importlib.util, sys
    mp = Path(__file__).resolve().parent / "kernel_planner_model2.py"
    spec = importlib.util.spec_from_file_location("_kp", mp)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_kp"] = m
    spec.loader.exec_module(m)
    # check the VALUE SET, not a text span: binary_group legitimately appears as its own
    # key between binary_sparse_residual and ternary
    assert m.REPRESENTATION_EXECUTES["binary_sparse_residual"] == {"binary_sparse_residual"}
    assert "binary_group" in m.REPRESENTATION_EXECUTES["binary_group"]


def test_the_second_registration_round_verified_both_declaration_and_reference():
    lib = json.load(open(KL))
    r = lib["organ_registration_round2"]
    assert r["n_added"] == 6
    assert "referenced from src/" in r["each_verified"]
    root = Path(__file__).resolve().parents[2]
    added = set(r["added"])
    for k in lib["kernels"]:
        if k["kernel_identity"] in added:
            assert f"kernel void {k['kernel_identity']}" in (root / k["shader"]).read_text()
            assert k["parity"]["kind"] == "ABSENT" and k["parity"]["absent_reason"]


def test_passthrough_eligibility_is_decided_by_measured_mass_not_organ_type():
    """Refusing f32 for every GEMV organ was too crude: moe_router and moe_expert are
    both GEMV organs and only one of them should qualify."""
    d = rec()
    s = d["organ_mass_shares"]
    ceil = d["passthrough_mass_ceiling"]
    assert s["moe_expert"] > ceil
    assert s["moe_router"] <= ceil
    assert "Organ type is not the test" in d["passthrough_rule"]


def test_organ_mass_shares_are_computed_from_the_specimen_config():
    d = rec()
    assert abs(sum(d["organ_mass_shares"].values()) - 1.0) < 1e-5
    assert d["organ_param_counts"]["moe_expert"] > d["organ_param_counts"]["moe_router"]


def test_moe_expert_still_cannot_take_f32_passthrough():
    """The size rule must not reopen the case it was built to close."""
    moe = next(r for r in rec()["organ_plan"] if r["organ"] == "moe_expert")
    f32 = [f for f in moe["seeded_families_in_score_order"]
           if f["family"] == "leftover_f32"]
    assert f32 and f32[0]["passthrough_rejected_as_too_large"] is True
    assert moe["selected_representation"] != "leftover_f32"


def test_no_upstream_defect_remains_now_that_the_planner_is_mass_aware():
    assert rec()["upstream_defects_found"] == []
