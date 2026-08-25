"""N039 Qwen completion receipt: generated from receipts, citations resolve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from qwen_completion_receipt import (  # noqa: E402
    ABSENT,
    GENERATOR,
    RECEIPT,
    REQUIRED_INPUTS,
    SCHEMA,
    build,
    citation_exists,
    numeric,
    unresolved_citations,
    write,
)

DOCS = None


def docs() -> dict:
    global DOCS
    if DOCS is None:
        built = build()
        write(built)
        DOCS = built
    return DOCS


def _disk() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/qwen_completion_receipt.py"
    )
    return json.loads(RECEIPT.read_text())


def test_generator_writes_schema_and_cpu_discipline():
    d = docs()
    on_disk = _disk()
    assert on_disk["schema"] == SCHEMA
    assert d["schema"] == SCHEMA
    assert on_disk["generated_by"] == GENERATOR
    assert on_disk["hand_authored"] is False
    assert on_disk["did_not_touch_gpu"] is True
    assert on_disk["did_not_run_cargo_or_metal_benchmarks"] is True
    assert on_disk["did_not_load_a_model"] is True
    assert on_disk["did_not_mutate_parent"] is True
    assert on_disk["did_not_rederive_measured_numbers"] is True
    assert on_disk["unmeasured_is_absent"] is True
    assert on_disk["odyssey_textbook"] == 1
    assert on_disk["specimen_retired_by_this_lane"] is False


def test_required_inputs_are_present():
    d = docs()
    paths = [r["path"] for r in d["required_inputs"]]
    assert paths == list(REQUIRED_INPUTS)
    for row in d["required_inputs"]:
        assert row["present"] is True, row
        assert row["required"] is True
        assert citation_exists(row["path"]), row["path"]


def test_every_citation_resolves_on_disk_or_in_git():
    missing = unresolved_citations(docs())
    assert missing == [], f"completion-receipt citations that do not exist: {missing}"
    missing_disk = unresolved_citations(_disk())
    assert missing_disk == [], missing_disk


def test_citation_walker_fails_a_bogus_receipt():
    fake = {
        "citations": ["receipts/headless/DOES_NOT_EXIST_N039.json"],
        "source": "receipts/headless/ALSO_NOT_A_RECEIPT.json",
    }
    bad = unresolved_citations(fake)
    assert "receipts/headless/DOES_NOT_EXIST_N039.json" in bad
    assert "receipts/headless/ALSO_NOT_A_RECEIPT.json" in bad
    assert citation_exists("receipts/headless/BYTES_FRONTIER.json") is True
    assert citation_exists("receipts/headless/HYBRID_OPERATOR.json") is True
    assert citation_exists("receipts/headless/BINARY_HEALING.json") is True


def _vector() -> dict:
    v = docs()["completion_vector"]
    assert v, "completion vector missing"
    return v


def test_completion_vector_has_every_section_84_field():
    v = _vector()
    required = (
        "strongest_coherent_complete_ebpw",
        "fastest_coherent_token_ns",
        "lowest_organ_ebpw_vs_lowest_coherent",
        "binary_coherence_tax",
        "shared_basis_verdict",
        "hybrid_verdict",
        "mlp_density_floor",
        "three_roofs_and_production",
        "residual_bottleneck",
        "concurrency_equilibrium",
        "dispatch_frontier",
        "reusable_kernels_and_organs",
        "negative_science_index",
        "RETIREMENT_READY",
    )
    for key in required:
        assert key in v, key
        assert v[key], key


def test_strongest_coherent_is_q2f_2_25_with_token_ns():
    v = _vector()["strongest_coherent_complete_ebpw"]
    assert v["name"] == "q2f_g64"
    assert v["coherent"] is True
    bpw = numeric(v["complete_ebpw"])
    assert bpw == 2.25
    ns = numeric(v["COMPLETE_TOKEN_NS"])
    assert ns == 27_547_874
    assert v["complete_ebpw"]["source"]
    assert citation_exists(v["complete_ebpw"]["source"])
    assert citation_exists(v["source"])


def test_fastest_coherent_is_q2f_27_55ms_no_coherent_faster():
    v = _vector()["fastest_coherent_token_ns"]
    assert v["name"] == "q2f_g64"
    assert v["coherent"] is True
    assert v["no_coherent_candidate_is_faster"] is True
    ns = numeric(v["COMPLETE_TOKEN_NS"])
    assert ns == 27_547_874
    assert abs(v["ms"] - 27.55) < 0.01
    faster = v["faster_incoherent_bodies"]
    ids = {r["id"] for r in faster}
    assert "binary_g64" in ids
    assert "shared_basis_k2" in ids
    for r in faster:
        assert r["COMPLETE_TOKEN_NS"] < ns


def test_mlp_density_floor_is_2_25_measured_four_independent_ways():
    floor = _vector()["mlp_density_floor"]
    assert "2.25" in floor["headline"]
    assert floor["n_independent_ways"] == 4
    assert numeric(floor["value"]) == 2.25
    ways = floor["ways"]
    assert len(ways) == 4
    families = [w["family"] for w in ways]
    assert families == [
        "q2f_composition",
        "binary_healing",
        "shared_basis",
        "hybrid_operator",
    ]
    for w in ways:
        assert w["independent"] is True
        assert w["source"]
        assert citation_exists(w["source"]), w["source"]
        assert numeric(w["floor_bpw"]) == 2.25
        for c in w.get("citations") or []:
            assert citation_exists(c), c
    assert ways[1]["uniformly_injured"] is True
    assert ways[1]["n_heals_coherent"] == 0
    assert ways[2]["kernel_competent"] is True
    assert ways[2]["coherent_shared_basis_beats_q2f"] is False
    assert ways[3]["coherent_hybrid_beats_q2f"] is False
    assert floor["do_not_transfer_to_other_organs"] is True
    assert docs()["headline"] == floor["headline"]
    assert docs()["finding"]["mlp_density_floor_bpw"] == 2.25
    assert docs()["finding"]["n_independent_ways"] == 4


def test_binary_coherence_tax_is_uniform_and_full_q2f_body():
    tax = _vector()["binary_coherence_tax"]
    assert tax["injury_uniform"] is True
    assert tax["tax_is_full_q2f_mlp_body"] is True
    assert tax["n_that_reached_coherent_generation"] == 0
    assert tax["coherent_healed_body_still_faster_than_q2f"] is False
    only = tax["only_coherent_reference"]
    assert only["id"] == "q2f"
    assert numeric(only["mlp_body_bpw"]) == 2.25
    assert numeric(only["mlp_tax_ebpw"]) == 1.0
    assert only["coherent"] is True
    assert citation_exists(tax["source"])
    injured = tax["injured_body"]
    assert injured["bpw"] == 1.25
    assert injured["COMPLETE_TOKEN_NS"] == 23_431_791


def test_shared_basis_kernel_competent_density_dead_below_2_25():
    v = _vector()["shared_basis_verdict"]
    assert v["kernel"] == "competent"
    assert v["kernel_competent"] is True
    assert v["representation_below_2_25"] == "dead"
    assert v["coherent_shared_basis_beats_q2f"] is False
    ns = numeric(v["k2_complete_token_ns"])
    assert ns == 24_554_625
    assert ns < 27_547_874
    assert "competent" in v["verdict"] and "dead below 2.25" in v["verdict"]
    for c in v["citations"]:
        assert citation_exists(c), c


def test_hybrid_confirms_floor_as_fourth_way():
    v = _vector()["hybrid_verdict"]
    assert v["coherent_hybrid_beats_q2f"] is False
    assert v["confirms_2_25_floor_as_fourth_way"] is True
    assert v["n_hybrid_fused_operators"] >= 2
    assert (v.get("q2f_baseline") or {}).get("bpw") == 2.25
    assert citation_exists(v["source"])


def test_lowest_organ_ebpw_vs_lowest_coherent():
    v = _vector()["lowest_organ_ebpw_vs_lowest_coherent"]
    organ = v["lowest_organ_complete_ebpw"]
    assert organ["organ"] in {"mlp_gate_up", "mlp_down"}
    assert numeric(organ["complete_ebpw"]) == 2.25
    assert numeric(v["lowest_coherent_complete_ebpw"]) == 2.25
    local = v["lowest_local_probe_ebpw_reached"]
    assert abs(numeric(local) - 0.53125) < 1e-9
    assert local["coherent"] is False
    floors = v["other_organ_local_floors"]
    assert numeric(floors["deltanet"]) == 4.125
    assert numeric(floors["gqa"]) == 4.25
    assert numeric(floors["embedding_output"]) == 4.125
    assert floors["do_not_transfer_mlp"] is True
    assert numeric(floors["mlp_survive_bpw"]) == 2.25
    assert numeric(floors["mlp_fail_bpw"]) == 1.85


def test_three_roofs_and_production_356_7():
    v = _vector()["three_roofs_and_production"]
    assert v["never_collapsed"] is True
    assert numeric(v["DEVICE_THEORETICAL"]) == 819.0
    assert numeric(v["DEVICE_MEASURED_SUSTAINED"]) == 778.8
    reach = numeric(v["MODEL_REACHABLE"])
    assert reach is not None and abs(reach - 729.7) < 0.01
    assert numeric(v["MODEL_REACHABLE_tok_s"]) == 729.7
    prod = numeric(v["production_decode_gb_s"])
    assert prod == 356.7
    owner = numeric(v["production_decode_gb_s"]["owner_measurement"])
    assert owner is not None and round(owner, 1) == 356.7
    assert "BANDWIDTH_ROOF.json" in v["DEVICE_THEORETICAL"]["source"]
    assert "ORGAN_BANDWIDTH.json" in v["production_decode_gb_s"]["source"]
    for c in v["citations"]:
        assert citation_exists(c), c


def test_residual_bottleneck_is_organ_bound_not_dispatch_bound():
    v = _vector()["residual_bottleneck"]
    assert v["bound_class"] == "bandwidth-bound"
    assert v["organ_bound_not_dispatch_bound"] is True
    assert v["mlp_tile_is_not_the_wall"] is True
    assert v["n025"]["largest_share"] == "mlp_gate_up"
    assert numeric(v["n025"]["dispatch_628_to_580"]["baseline"]) == 628
    assert numeric(v["n025"]["dispatch_628_to_580"]["candidate"]) == 580
    q4 = numeric(v["q4_incumbent_achieved_gb_s"])
    assert q4 is not None and abs(q4 - 468.9) < 0.1


def test_concurrency_equilibrium_about_1_32x_and_q4_wins_verified_wu():
    v = _vector()["concurrency_equilibrium"]
    c2 = numeric(v["concurrent_independent_vs_c1"]["c2"])
    c4 = numeric(v["concurrent_independent_vs_c1"]["c4"])
    assert c2 is not None and abs(c2 - 1.32) < 0.01
    assert c4 is not None and abs(c4 - 1.32) < 0.01
    assert v["verified_wu_hour_ranks_q4"] is True
    winner = v["production_bench_winner"]
    assert winner["artifact"] == "q4_incumbent"
    q4_wu = numeric(winner["verified_wu_per_hour"])
    parent_wu = numeric(v["highest_aggregate_tok_s_is_not_the_winner"]["verified_wu_per_hour"])
    assert q4_wu is not None and abs(q4_wu - 669.2) < 0.1
    assert parent_wu is not None and abs(parent_wu - 491.5) < 0.1
    assert q4_wu > parent_wu
    assert v["highest_aggregate_tok_s_is_not_the_winner"]["artifact"] == "parent_a"


def test_dispatch_frontier_964_756_628_580():
    v = _vector()["dispatch_frontier"]
    assert v["sequence"] == [964, 756, 628, 580]
    vals = [numeric(s["dispatches"]) for s in v["steps"]]
    assert vals == [964, 756, 628, 580]
    for s in v["steps"]:
        assert citation_exists(s["source"]), s["source"]


def test_reusable_kernels_and_organs_counts():
    v = _vector()["reusable_kernels_and_organs"]
    assert numeric(v["n_kernels"]) == 17
    assert numeric(v["n_organs"]) == 7
    assert "mlp_gate_up" in v["organ_ids"]
    assert "mlp_down" in v["organ_ids"]
    assert "deltanet" in v["organ_ids"]
    assert numeric(v["n_representation_families"]) == 7


def test_negative_science_index_has_every_required_measured_negative():
    v = _vector()["negative_science_index"]
    needles = v["required_needles"]
    assert needles == [
        "fewer_bits_is_not_fewer_ns",
        "uniform_binary_injury",
        "shared_basis_dies_below_2_25",
        "hybrid_floor",
        "mlp_tile_is_not_the_wall",
        "gate_up_down_host_not_separated",
    ]
    by_id = {e["id"]: e for e in v["campaign_measured_negatives"]}
    for nid in needles:
        assert nid in by_id, nid
        row = by_id[nid]
        assert row["measured_negative"] is True
        assert citation_exists(row["source"]), row["source"]
    assert by_id["uniform_binary_injury"]["evidence"]["uniformly_injured"] is True
    assert by_id["hybrid_floor"]["evidence"]["coherent_hybrid_beats_q2f"] is False
    assert by_id["shared_basis_dies_below_2_25"]["evidence"]["kernel_competent"] is True
    assert (
        by_id["gate_up_down_host_not_separated"]["evidence"]["mlp_gate_up_production_separated"]
        is False
    )
    catalog_n = numeric(v["catalog"]["n_entries"])
    assert catalog_n == 31


def test_retirement_ready_is_false_and_marks_the_gap():
    d = docs()
    assert d["RETIREMENT_READY"] is False
    gate = _vector()["RETIREMENT_READY"]
    assert gate["value"] is False
    assert gate["RETIREMENT_READY"] is False
    assert gate["specimen_retired_by_this_lane"] is False
    assert "§73" in gate["s024_gate"]
    assert "§74" in gate["s024_gate"]
    cl = gate["checklist"]
    assert cl["organ_library_sealed"] is True
    assert cl["kernel_library_sealed"] is True
    assert cl["representation_library_sealed"] is True
    assert cl["transfer_receipt_sealed"] is False
    assert cl["clean_rerun_this_lane"] is False
    gap_ids = {g["id"] for g in gate["gap"]}
    assert "TRANSFER_RECEIPT" in gap_ids
    assert "CLEAN_RERUN" in gap_ids
    recipe = gate["parent_compile_recipe"]
    assert recipe["id"] == "mix_all_mlp_affine_g64_ls"
    assert citation_exists(recipe["source"])


def test_remaining_lists_open_items_with_reasons():
    left = docs()["REMAINING"]
    ids = {r["id"] for r in left}
    assert "other_organ_density_floors_not_descended" in ids
    assert "full_c1_c8_sweep" in ids
    assert "capability_rung_untested" in ids
    assert "transfer_receipt" in ids
    for row in left:
        assert row.get("why"), row["id"]
        for c in row.get("citations") or []:
            assert citation_exists(c), c


def test_every_numeric_completion_field_cites_a_real_receipt():
    """Fails if a cited receipt is missing — the N039 acceptance walker."""

    def walk(obj, path=""):
        if isinstance(obj, dict):
            if obj.get("kind") in {"CITED", "MEASURED", "DERIVED"} and obj.get("value") is not None:
                src = obj.get("source") or obj.get("source_receipt")
                assert src, f"{path} has a number with no source"
                assert citation_exists(src), f"{path} cites missing {src}"
            for k, val in obj.items():
                walk(val, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                walk(val, f"{path}[{i}]")

    walk(_vector())
    walk(docs()["specimen"])
