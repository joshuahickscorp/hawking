"""N046 LITERATURE_FRONTIER: generated survey, citations resolve, additions vs N043."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from literature_frontier import (  # noqa: E402
    COST_AXES,
    GENERATOR,
    N043_SEED,
    ORGAN_LIBRARY_IDS,
    RECEIPT,
    REQUIRED_INPUTS,
    SCHEMA,
    SURVEY_FAMILIES,
    build,
    citation_exists,
    unresolved_citations,
    write,
)

DOCS = None
ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

# Organ ids the survey is allowed to target (library + aliases used in receipts).
ALLOWED_ORGANS = set(ORGAN_LIBRARY_IDS) | {
    "mlp",
    "embedding_output",
    "session_state",
    "decode_loop",
    "whole_model",
}


def docs() -> dict:
    global DOCS
    if DOCS is None:
        built = build()
        write(built)
        DOCS = built
    return DOCS


def _disk() -> dict:
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
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
    assert on_disk["literature_is_hypothesis_not_authority"] is True
    assert on_disk["web_access"] is True
    assert on_disk["s027"] is True
    assert "§86" in on_disk["s026"]
    assert "§87" in on_disk["s026"]
    assert "§89" in on_disk["s026"]


def test_required_inputs_present():
    d = docs()
    paths = [r["path"] for r in d["required_inputs"]]
    assert paths == list(REQUIRED_INPUTS)
    for row in d["required_inputs"]:
        assert row["present"] is True, row
        assert citation_exists(row["path"]), row["path"]


def test_every_citation_resolves_on_disk_or_in_git():
    missing = unresolved_citations(docs())
    assert missing == [], f"literature citations that do not exist: {missing}"
    missing_disk = unresolved_citations(_disk())
    assert missing_disk == [], missing_disk


def test_citation_walker_fails_a_bogus_receipt():
    fake = {
        "citations": ["receipts/headless/DOES_NOT_EXIST_N046.json"],
        "source": "receipts/headless/ALSO_NOT_A_RECEIPT.json",
    }
    bad = unresolved_citations(fake)
    assert "receipts/headless/DOES_NOT_EXIST_N046.json" in bad
    assert "receipts/headless/ALSO_NOT_A_RECEIPT.json" in bad
    assert citation_exists("receipts/headless/ORGAN_LIBRARY.json") is True


def test_survey_covers_required_families():
    d = docs()
    covered = set(d["survey_families_covered"])
    required = set(SURVEY_FAMILIES)
    assert required <= covered, f"missing families: {required - covered}"
    assert set(d["survey_families_required"]) == required


def test_each_technique_is_an_actionable_hypothesis():
    d = docs()
    techs = d["techniques"]
    assert len(techs) >= 24, len(techs)
    names = [t["name"] for t in techs]
    assert len(names) == len(set(names)), "duplicate technique names"
    required_fields = (
        "name",
        "arxiv_id",
        "arxiv_date",
        "authors",
        "mechanism",
        "hawking_organ",
        "hawking_gap",
        "codebase_citations",
        "metal_feasibility",
        "metal_note",
        "expected_physical_win_axis",
        "expected_physical_win",
        "cheapest_falsifying_experiment",
        "ADD_TO_REGISTRY",
        "add_to_registry_reason",
        "rank_score",
        "literature_status",
    )
    for t in techs:
        for f in required_fields:
            assert t.get(f) not in (None, "", []), f"{t['name']} missing {f}"
        assert ARXIV_RE.match(t["arxiv_id"]), t["arxiv_id"]
        assert re.match(r"^\d{4}-\d{2}(-\d{2})?$", t["arxiv_date"]), t["arxiv_date"]
        assert t["hawking_organ"] in ALLOWED_ORGANS, t["hawking_organ"]
        assert t["expected_physical_win_axis"] in COST_AXES, t
        assert t["ADD_TO_REGISTRY"] in ("yes", "no")
        assert t["literature_status"] == "HYPOTHESIS"
        assert t["not_a_verdict"] is True
        assert t["cuda_result_is_not_metal_result"] is True
        metal_blob = (t["metal_feasibility"] + " " + t["metal_note"]).lower()
        assert any(
            tok in metal_blob
            for tok in ("metal", "cuda", "apple", "kernel", "architecture", "npu", "uma")
        ), t["name"]
        assert t["codebase_citations"], t["name"]
        for c in t["codebase_citations"]:
            assert citation_exists(c), f"{t['name']} cites missing {c}"
        assert t["rank_score"] == (t["info_gain"] * t["physical_upside"]) / t[
            "experiment_cost"
        ]


def test_techniques_ranked_by_expected_value():
    scores = [t["rank_score"] for t in docs()["techniques"]]
    assert scores == sorted(scores, reverse=True)


def test_n043_seed_is_not_re_recommended():
    d = docs()
    seed = {n.lower() for n in N043_SEED}
    for t in d["techniques"]:
        if t["n043_seed"]:
            assert t["name"] in N043_SEED or t["name"].lower() in seed
            assert t["ADD_TO_REGISTRY"] == "no", t["name"]
    rec_names = {r["name"].lower() for r in d["RECOMMENDED_ADDITIONS"]}
    overlap = rec_names & seed
    assert not overlap, f"N043 seed leaked into RECOMMENDED_ADDITIONS: {overlap}"


def test_recommended_additions_are_yes_and_sorted():
    d = docs()
    rec = d["RECOMMENDED_ADDITIONS"]
    assert rec, "empty RECOMMENDED_ADDITIONS"
    assert d["n_recommended_additions"] == len(rec)
    yes = [t for t in d["techniques"] if t["ADD_TO_REGISTRY"] == "yes" and not t["n043_seed"]]
    assert [r["name"] for r in rec] == [t["name"] for t in yes]
    scores = [r["rank_score"] for r in rec]
    assert scores == sorted(scores, reverse=True)
    # Cheap high-EV rows the survey is built around.
    rec_names = {r["name"] for r in rec}
    for must in ("HIGGS", "QTIP", "TEAL", "LayerSkip", "Quamba2"):
        assert must in rec_names, must


def test_connected_floors_match_receipts():
    d = docs()
    floors = d["connected_floors"]
    recompose = json.loads(
        (REPO / "receipts/headless/WHOLE_MODEL_RECOMPOSE.json").read_text()
    )
    assert floors["whole_model_complete_ebpw"] == recompose["current_qwen_complete_ebpw"]
    assert floors["below_3_0"] is True
    assert floors["mlp_confirmed_ebpw"] == 2.25
    assert recompose["current_qwen_complete_ebpw"] < 3.0
    roof = json.loads((REPO / "receipts/headless/ORGAN_ROOF_LEDGER.json").read_text())
    assert d["machine"]["device_measured_sustained_gb_s"] == roof["three_roofs"][
        "DEVICE_MEASURED_SUSTAINED"
    ]["value"]
    assert abs(d["machine"]["device_measured_sustained_gb_s"] - 778.8) < 0.05
    assert abs(d["machine"]["model_reachable_gb_s"] - 729.7) < 0.1


def test_negative_science_is_cited_not_restated_as_authority():
    d = docs()
    ns = d["negative_science_this_survey_respects"]
    assert ns["binary_uniformly_injured"]["source"].endswith("BINARY_HEALING.json")
    assert ns["shared_basis_dead_below_2_25"]["coherent_shared_basis_beats_q2f"] is False
    assert ns["low_rank_residual_never_heals"]["coherent_hybrid_beats_q2f"] is False
    assert ns["deltanet_in_proj_irreducible"]["capacity_ratio_state_over_qkv"] == 0.015
    closed = {row["idea"] for row in d["do_not_reopen_on_this_parent"]}
    assert any("shared" in i for i in closed)
    assert any("low-rank" in i for i in closed)
    # CompactifAI must not be a recommended addition.
    rec_names = {r["name"] for r in d["RECOMMENDED_ADDITIONS"]}
    assert "CompactifAI" not in rec_names
    compact = next(t for t in d["techniques"] if t["name"] == "CompactifAI")
    assert compact["ADD_TO_REGISTRY"] == "no"


def test_metal_feasibility_does_not_treat_cuda_as_metal():
    d = docs()
    for t in d["techniques"]:
        assert t["cuda_result_is_not_metal_result"] is True, t["name"]
        note = t["metal_note"].lower()
        if "cuda" in note:
            # CUDA mentioned => Metal must be named so the S026 §89 flag is local.
            assert "metal" in note, t["name"]


def test_cost_vector_is_s026_section_3():
    d = docs()
    assert d["cost_vector_axes_s026_3"] == list(COST_AXES)
    axes_used = {t["expected_physical_win_axis"] for t in d["techniques"]}
    assert "complete_ebpw" in axes_used
    assert "accepted_tokens_per_forward" in axes_used
    assert "kv_bytes" in axes_used
    assert "token_ns" in axes_used
