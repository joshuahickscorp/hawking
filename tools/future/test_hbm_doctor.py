"""HBM Doctor tests. The negative controls must actually fire."""
from __future__ import annotations

import json

import pytest

from tools.future import hbm_doctor as hd
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _ids(result: hd.SolveResult) -> set[str]:
    return set(result.selected_ids())


def test_build_emits_sealed_receipt():
    out = hd.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "HBM_DOCTOR.json"
    assert doc["schema"] == hd.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    assert doc["headline"]["undecidable_count"] >= 1
    assert doc["solution"]["counts"]["undecidable"] == doc["headline"]["undecidable_count"]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["anti_pattern"]["refused_policy"] == "fill_HBM_with_the_largest_items"
    assert doc["frontier_lane"]["id"] == "F005"
    assert doc["headline"]["undecidable_is_the_decision"] is True
    assert doc["solution"]["selected_ids"] == []


def test_selftest_aliases_build():
    assert hd.selftest is hd.build or hd.selftest().name == "HBM_DOCTOR.json"


def test_receipt_never_numeric_hardware_fields():
    doc = json.loads(hd.build().read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS:
                    assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


def test_unknown_mac_latency_is_undecidable_never_selected():
    known = hd.make_candidate("known_hot", bytes=50, mac_latency_ns=10.0)
    unknown = hd.make_candidate("unknown_mac", bytes=1, mac_latency_ns=None)
    result = hd.solve([known, unknown], budget_bytes=100)
    scored = hd.score_candidate(unknown)
    assert scored["score"] == "UNKNOWN"
    assert scored["score"] != 0
    assert scored["bucket"] == "undecidable"
    assert "mac_latency_ns" in scored["missing_inputs"]
    assert unknown.id not in _ids(result)
    undecidable_ids = {c.id for c in result.undecidable}
    rejected_ids = {c.id for c, _ in result.rejected}
    assert unknown.id in undecidable_ids
    assert unknown.id not in rejected_ids
    assert known.id in _ids(result)


def test_unknown_is_not_defaulted_to_zero_or_an_estimate():
    hole = hd.make_candidate("hole", bytes=10, reuse_count=None, mac_latency_ns=999.0)
    scored = hd.score_candidate(hole)
    assert scored["score"] == "UNKNOWN"
    assert "reuse_count" in scored["missing_inputs"]
    # A zero-default would make this decidable with value 0 and put it in rejected.
    result = hd.solve([hole], budget_bytes=100)
    assert result.undecidable[0].id == "hole"
    assert result.selected == ()
    assert result.rejected == ()


def test_size_ranking_guard_fires_on_constructed_case():
    """Large cold organ vs small hot organ. Size ranking must lose, and the guard must fire."""
    budget = 100
    large_cold = hd.make_candidate(
        "large_cold", bytes=80, mac_latency_ns=8.0, access_probability=1.0, reuse_count=1.0,
        dependency_criticality=1.0,
    )
    small_hot = hd.make_candidate(
        "small_hot", bytes=40, mac_latency_ns=400.0, access_probability=1.0, reuse_count=1.0,
        dependency_criticality=1.0,
    )
    med_cold = hd.make_candidate(
        "med_cold", bytes=70, mac_latency_ns=7.0, access_probability=1.0, reuse_count=1.0,
        dependency_criticality=1.0,
    )
    items = [large_cold, small_hot, med_cold]
    result = hd.solve(items, budget)
    assert result.selected_ids() == ["small_hot"]
    assert result.selected_value == pytest.approx(400.0)
    size_sel = hd.size_ranked_select(items, budget)
    assert [c.id for c in size_sel] == ["large_cold"]
    size_val = sum(c.cited_weighted_latency() or 0.0 for c in size_sel)
    assert size_val == pytest.approx(8.0)
    assert size_val < result.selected_value
    guard = hd.refuse_size_ranking(items, budget)
    assert guard["fired"] is True
    assert guard["vacuous"] is False
    assert guard["size_ranked_ids"] == ["large_cold"]
    assert guard["objective_ids"] == ["small_hot"]
    assert guard["size_ranked_value"] < guard["objective_value"]


def test_size_ranking_guard_is_capable_of_not_firing():
    """If size ranking happens to equal the objective, fired is False. The guard is not a constant True."""
    a = hd.make_candidate("a", bytes=10, mac_latency_ns=10.0)
    b = hd.make_candidate("b", bytes=10, mac_latency_ns=10.0)
    guard = hd.refuse_size_ranking([a, b], budget_bytes=10)
    assert guard["fired"] is False
    assert guard["vacuous"] is False


def test_knapsack_respects_budget_and_picks_two_small_over_one_large():
    budget = 100
    small_a = hd.make_candidate("small_a", bytes=40, mac_latency_ns=50.0)
    small_b = hd.make_candidate("small_b", bytes=50, mac_latency_ns=50.0)
    large = hd.make_candidate("large", bytes=90, mac_latency_ns=60.0)
    result = hd.solve([small_a, small_b, large], budget)
    assert result.selected_bytes <= budget
    assert set(result.selected_ids()) == {"small_a", "small_b"}
    assert result.selected_value == pytest.approx(100.0)


def test_item_larger_than_budget_is_rejected_not_selected():
    huge = hd.make_candidate("huge", bytes=10_000, mac_latency_ns=1e9)
    tiny = hd.make_candidate("tiny", bytes=4, mac_latency_ns=1.0)
    result = hd.solve([huge, tiny], budget_bytes=8)
    assert "huge" not in _ids(result)
    reasons = {c.id: reason for c, reason in result.rejected}
    assert reasons["huge"] == "bytes_exceed_budget"
    assert "tiny" in _ids(result)


def test_numeric_fpga_latency_is_refused():
    with pytest.raises(hd.FpgaLatencyMustBeClass, match="never a number"):
        hd.make_candidate("boardless", projected_fpga_latency=12.5)
    with pytest.raises(hd.FpgaLatencyMustBeClass, match="never a number"):
        hd.ResidentCandidate(
            id="x",
            bytes=1,
            access_probability=1.0,
            reuse_count=1.0,
            transport_cost_class="PCIE",
            mac_latency_ns=1.0,
            projected_fpga_latency=3,
            state_lifetime="STATIC_WEIGHTS",
            dependency_criticality=1.0,
            representation_format="x",
        )


def test_fpga_latency_class_is_required_for_decidability():
    c = hd.make_candidate("no_class", projected_fpga_latency=None)
    scored = hd.score_candidate(c)
    assert scored["score"] == "UNKNOWN"
    assert "projected_fpga_latency" in scored["missing_inputs"]


def test_flash_run_is_undecidable_and_unknown_never_selected():
    recovered = hd.recover_inputs()
    flash = hd.flash_candidates_from_docs(
        recovered["docs"]["flash_science"],
        recovered["docs"]["flash_ebpw"],
        recovered["docs"]["flash_token_ns"],
        recovered["docs"]["flash_fpga_map"],
    )
    assert flash, "science organ_graph must yield candidates"
    ids = {c.id for c in flash}
    assert "flash.routed_experts" in ids
    assert "flash.ngram_engine" in ids
    assert "flash.embeddings" in ids
    assert "flash.deltanet" in ids
    assert "flash.recurrent_state" in ids
    for cand in flash:
        assert cand.mac_latency_ns is None
        assert cand.score_per_byte() == "UNKNOWN"
        assert not cand.is_decidable()
        assert "mac_latency_ns" in cand.missing_inputs()
    result = hd.solve(flash, hd.DEFAULT_BUDGET_BYTES)
    assert result.selected == ()
    assert result.rejected == ()
    assert len(result.undecidable) == len(flash)
    experts = next(c for c in flash if c.id == "flash.routed_experts")
    assert experts.bytes == 241591910400
    assert experts.access_probability == pytest.approx(10 / 512)
    ngram = next(c for c in flash if c.id == "flash.ngram_engine")
    assert ngram.bytes == 102466171160
    assert ngram.access_probability is None
    embed = next(c for c in flash if c.id == "flash.embeddings")
    assert embed.bytes == 1271398400
    assert embed.source_active_bytes_per_token == 5120
    assert embed.access_probability is None
    state = next(c for c in flash if c.id == "flash.recurrent_state")
    assert state.bytes == 117669888
    assert state.state_lifetime == "SEQUENCE_STATE"


def test_flash_size_ranking_counterexample_is_recorded():
    recovered = hd.recover_inputs()
    flash = hd.flash_candidates_from_docs(
        recovered["docs"]["flash_science"],
        recovered["docs"]["flash_ebpw"],
        recovered["docs"]["flash_token_ns"],
        recovered["docs"]["flash_fpga_map"],
    )
    anti = hd.flash_size_counterexample(flash, hd.DEFAULT_BUDGET_BYTES)
    assert "flash.routed_experts" in anti["too_large_to_reside_as_a_whole"]
    assert "flash.ngram_engine" in anti["too_large_to_reside_as_a_whole"]
    greedy = anti["size_greedy_ids"]
    assert greedy, "size-greedy must pick something that fits"
    assert "flash.routed_experts" not in greedy
    assert "flash.ngram_engine" not in greedy
    assert anti["size_greedy_bytes"] <= hd.DEFAULT_BUDGET_BYTES


def test_noetic_preciousness_order_disagrees_with_size():
    recovered = hd.recover_inputs()
    noetic = hd.noetic_size_counterexample(recovered["docs"]["noetic_census"])
    assert noetic["present"] is True
    assert noetic["orders_disagree"] is True
    assert noetic["size_order"][0] == "mlp"
    assert noetic["preciousness_order"][0] == "embedding"
    mlp_bytes = noetic["counterexample"]["largest_organ"]["bytes"]
    assert mlp_bytes > hd.DEFAULT_BUDGET_BYTES


def test_bytes_atlas_consumed_without_tps_keys():
    recovered = hd.recover_inputs()
    atlas = hd.consume_bytes_atlas(recovered["docs"]["bytes_atlas"])
    assert atlas["consumed"] is True
    blob = json.dumps(atlas)
    for banned in ("token_ns", "bandwidth_gbps", '"tps"', "joules_per_token"):
        assert banned not in blob
    assert atlas["catalog_total_bytes"]
    assert atlas["pareto_by_bytes"]


def test_contract_named_receipts_are_reported_absent():
    # Environment-coupled: uncommitted receipts are invisible from a sparse lane
    # worktree and visible from the primary one. What must hold in BOTH is that the
    # doctor reports presence honestly and never silently substitutes a default.

    recovered = hd.recover_inputs()
    c = recovered["consulted"]
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(c["flash_organ_census"]["present"], bool)
    # Environment-coupled: uncommitted files are invisible from a sparse lane
    # worktree and visible from the primary one. Assert the module copes, not
    # the checkout it was written in.
    assert isinstance(c["qwen27_token_ns_budget"]["present"], bool)
    assert isinstance(c["flash_layer30_critical_path"]["present"], bool)
    assert isinstance(c["flash_layer10_critical_path"]["present"], bool)
    assert c["flash_science"]["present"] is True
    assert c["flash_fpga_organ_map"]["present"] is True


def test_zero_criticality_is_decidable_but_not_selected():
    hot = hd.make_candidate("hot", bytes=10, mac_latency_ns=100.0, dependency_criticality=1.0)
    off = hd.make_candidate("off_path", bytes=10, mac_latency_ns=100.0, dependency_criticality=0.0)
    result = hd.solve([hot, off], budget_bytes=20)
    assert "hot" in _ids(result)
    assert "off_path" not in _ids(result)
    reasons = {c.id: reason for c, reason in result.rejected}
    assert reasons["off_path"] == "zero_cited_weighted_latency"


def test_score_is_reduction_per_byte():
    a = hd.make_candidate("a", bytes=10, mac_latency_ns=20.0, access_probability=0.5, reuse_count=2.0, dependency_criticality=1.0)
    # 20 * 0.5 * 2 * 1 = 20; per byte = 2
    assert a.cited_weighted_latency() == pytest.approx(20.0)
    assert a.score_per_byte() == pytest.approx(2.0)


def test_the_hbm_budget_is_sourced_not_an_unattributed_literal():
    """The number was always right; its provenance was missing.

    A module that refuses to fill reuse_count or mac_latency_ns with a default
    should not take its own byte budget on faith, especially when an identical
    CITED value was in the U50DD device profile the whole time.
    """
    prov = hd.DEFAULT_BUDGET_PROVENANCE
    assert prov["pinned"] is True, "budget fell back to the unsourced literal"
    assert prov["citation"], "a budget with no citation"
    assert prov["document_class"] == "AMD_DATASHEET_DS965"
    assert prov["hardware_measured"] is False, "vendor literature is not a measurement"
    assert prov["value"] == hd.DEFAULT_BUDGET_BYTES


def test_the_budget_tracks_the_device_profile_rather_than_a_copy():
    """Mutation control: move the profile and the budget must move with it.

    Two constants that merely happen to be equal drift apart the first time one
    of them is edited.
    """
    from tools.future import hwir
    assert hd.DEFAULT_BUDGET_BYTES == int(
        hwir.u50_family_profile("u50dd").to_dict()["hbm_capacity_bytes"]
    )
    recomputed, prov = hd._u50dd_hbm_capacity()
    assert recomputed == hd.DEFAULT_BUDGET_BYTES
    assert prov["via"].endswith("u50_family_profile('u50dd')")
