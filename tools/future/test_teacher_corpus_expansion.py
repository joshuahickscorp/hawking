"""Teacher-corpus expansion tests.

Load-bearing negatives: validate() must fail duplicate token ids and a
degenerate route union; plan() must refuse a surface the real corpus already
covers; every wall-time figure must say ESTIMATE. A validator nobody has
watched reject is a validator that will drift into fiction.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.future import teacher_corpus as tc
from tools.future import teacher_corpus_expansion as tce
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
    load_json,
)


def _walk_estimate_blocks(obj, found: list):
    if isinstance(obj, dict):
        if "wall_time_estimate" in obj or "expected_wall_time_estimate" in obj:
            found.append(obj.get("wall_time_estimate") or obj.get("expected_wall_time_estimate"))
        if obj.get("kind") == "ESTIMATE" or obj.get("label") == "ESTIMATE":
            found.append(obj)
        for v in obj.values():
            _walk_estimate_blocks(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_estimate_blocks(v, found)


def _assert_every_wall_time_is_estimate(obj) -> None:
    found: list = []
    _walk_estimate_blocks(obj, found)
    assert found, "expected at least one ESTIMATE-labelled wall-time block"
    for block in found:
        blob = json.dumps(block)
        assert "ESTIMATE" in blob
        if isinstance(block, dict):
            assert block.get("kind") == "ESTIMATE" or block.get("label") == "ESTIMATE"
            for key in HARDWARE_FIELDS:
                val = block.get(key)
                assert not isinstance(val, (int, float)), (key, val)


def test_build_emits_sealed_receipt():
    out = tce.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "TEACHER_CORPUS_EXPANSION.json"
    assert doc["schema"] == "hawking.future.teacher_corpus_expansion.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    assert load_json(out)["schema"] == doc["schema"]
    assert doc["selftest"]["real_corpus_accepted"] is True
    assert doc["selftest"]["duplicate_token_ids_refused"] is True
    assert doc["selftest"]["degenerate_route_union_refused"] is True
    assert doc["selftest"]["hidden_plan_redundant"] is True
    assert doc["real_corpus"]["validate_accepted"] is True
    assert doc["real_corpus"]["route_union_size"] == 117
    assert doc["real_corpus"]["n_unique_row_hashes"] == 256
    assert doc["dedupe_recapture_is_waste"]["verdict"] == "WASTE"
    assert "tools/future/teacher_corpus.py" in " ".join(doc["recovered_implementation"])
    _assert_every_wall_time_is_estimate(doc)


def test_validate_real_corpus_passes():
    corpus = tce.load_real_corpus()
    result = tce.validate(corpus)
    assert result["accepted"] is True
    assert result["refusals"] == []
    assert result["n_rows"] == 256
    assert result["n_unique_token_ids"] == 256
    assert result["n_unique_row_hashes"] == 256
    assert result["route_union_size"] == 117
    assert result["unique_ordered_topk_sets"] == 256
    assert result["canonical_surface"] == "hidden"
    assert result["tensor"] == tce.EXISTING_TENSOR
    assert result["pinned_revision"] == tc.FLASH_SPECIMEN["pinned_revision"]
    axes = {a["axis"]: a for a in result["axes"]}
    assert axes["unique_token_ids"]["pass"] is True
    assert axes["route_union_nondegenerate"]["pass"] is True
    assert axes["capability_domain"]["pass"] is False
    assert axes["capability_domain"]["value"] == "ABSENT"
    assert axes["capability_domain"]["required"] is False
    _assert_no_hardware_claims(result)


def test_validate_duplicate_token_ids_fails():
    fixture = tce.make_flash_corpus(duplicate_token_ids=True)
    result = tce.validate(fixture, raise_on_refuse=False)
    assert result["accepted"] is False
    assert "DUPLICATE_TOKEN_IDS" in result["refusals"]
    axes = {a["axis"]: a for a in result["axes"]}
    assert axes["unique_token_ids"]["pass"] is False
    with pytest.raises(tce.ExpansionRefused) as ei:
        tce.validate(fixture, raise_on_refuse=True)
    assert "DUPLICATE_TOKEN_IDS" in ei.value.codes


def test_validate_degenerate_route_union_fails():
    fixture = tce.make_flash_corpus(degenerate_routes=True)
    result = tce.validate(fixture, raise_on_refuse=False)
    assert result["accepted"] is False
    assert "DEGENERATE_ROUTE_UNION" in result["refusals"]
    axes = {a["axis"]: a for a in result["axes"]}
    assert axes["route_union_nondegenerate"]["pass"] is False
    assert result["route_union_size"] < tce.FLASH_MIN_ROUTE_UNION
    with pytest.raises(tce.ExpansionRefused) as ei:
        tce.validate(fixture, raise_on_refuse=True)
    assert "DEGENERATE_ROUTE_UNION" in ei.value.codes


def test_validate_absent_corpus_refuses_not_skips():
    result = tce.validate(None, raise_on_refuse=False)
    assert result["accepted"] is False
    assert "ABSENT_CORPUS" in result["refusals"]
    empty = tce.validate({}, raise_on_refuse=False)
    assert empty["accepted"] is False
    assert "ABSENT_CORPUS" in empty["refusals"]


def test_validate_thin_corpus_does_not_design_around_binary_min():
    thin = tce.make_flash_corpus(n=16)
    result = tce.validate(thin, raise_on_refuse=False)
    assert result["accepted"] is False
    assert "BINARY_MIN_ROWS_REFUSAL" in result["refusals"]
    assert result["n_unique_row_hashes"] == 16
    assert result["n_unique_row_hashes"] < tce.FLASH_MIN_ROWS
    assert tce.FLASH_MIN_ROWS == tc.BOUNDED_TARGET_ROWS == 256


def test_plan_covered_surface_refused_as_redundant():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    with pytest.raises(tce.RedundantCapture) as ei:
        tce.plan("hidden", prior=real, screen=screen, raise_on_refuse=True)
    assert "REDUNDANT_SURFACE" in ei.value.codes
    quiet = tce.plan("hidden", prior=real, screen=screen, raise_on_refuse=False)
    assert quiet["refused"] is True
    assert quiet["reason"] == "REDUNDANT_SURFACE"
    _assert_every_wall_time_is_estimate(quiet)


def test_plan_unknown_surface_refused():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    with pytest.raises(tce.UnknownSurface) as ei:
        tce.plan("combine", prior=real, screen=screen, raise_on_refuse=True)
    assert "UNKNOWN_SURFACE" in ei.value.codes
    # teacher_corpus.SURFACES names that the screen did not write are not planned.
    for name in tc.SURFACES:
        if name not in tce.declared_surfaces(screen):
            out = tce.plan(name, prior=real, screen=screen, raise_on_refuse=False)
            assert out["refused"] is True
            assert out["reason"] == "UNKNOWN_SURFACE"


def test_surfaces_needed_from_screen_not_invented():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    needed = tce.surfaces_needed(real, screen)
    names = needed["declared_from_screen"]
    assert names == ["router", "hidden", "routed-output", "terminal-logit"]
    assert set(needed["needed_names"]) <= set(names)
    assert "hidden" not in needed["needed_names"]
    assert "router" in needed["needed_names"]
    assert "routed-output" in needed["needed_names"]
    assert "terminal-logit" in needed["needed_names"]
    assert "combine" not in needed["needed_names"]
    assert "shared_expert" not in needed["needed_names"]
    assert "attention" not in needed["needed_names"]
    covered = {c["surface"] for c in needed["covered"]}
    assert covered == {"hidden"}
    for row in needed["covered"]:
        assert row["route_union_size"] == 117
        assert row["whole_model_coverage"] is False


def test_plan_zero_coverage_surfaces_join_existing_tokens():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    prior_tokens = list(tce.coverage_of(real)["token_ids"])
    for surface in ("router", "routed-output", "terminal-logit"):
        p = tce.plan(surface, prior=real, screen=screen, raise_on_refuse=True)
        assert p["accepted"] is True
        assert p["refused"] is False
        assert p["n_tokens"] == 256
        assert p["n_tokens"] >= tce.FLASH_MIN_ROWS
        assert p["n_tokens"] <= tce.FLASH_MAX_ROWS
        assert p["token_ids"] == prior_tokens
        assert p["executed"] is False
        assert p["gpu_authority"] is False
        assert p["current_binary_emits_this_surface"] is False
        assert p["pinned_revision"] == tc.FLASH_SPECIMEN["pinned_revision"]
        assert p["diversity_contract"]["min_unique_rows"] == 256
        _assert_every_wall_time_is_estimate(p)
        assert "ESTIMATE" in p["wall_time_estimate"]["text"]
        assert p["wall_time_estimate"]["kind"] == "ESTIMATE"
        assert p["wall_time_estimate"]["label"] == "ESTIMATE"


def test_plan_does_not_offer_wall_time_below_binary_min():
    est = tce.wall_time_estimate(n_tokens=16, layer=4, surface="router")
    assert est["refused"] is True
    assert "ESTIMATE" in est["text"]
    assert "256" in est["text"]


def test_dedupe_against_reports_waste_and_joinable_overlap():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    waste = tce.dedupe_against(real)
    assert waste["verdict"] == "WASTE"
    assert waste["token_overlap_fraction_of_candidate"] == 1.0
    assert waste["same_surface"] is True
    router = tce.plan("router", prior=real, screen=screen)
    join = tce.dedupe_against(real, router)
    assert join["verdict"] == "SAME_TOKENS_NEW_SURFACE"
    assert join["same_surface"] is False
    assert join["token_overlap_n"] == 256
    broader = tce.plan_broader_traces(prior=real)
    disjoint = tce.dedupe_against(real, broader)
    assert disjoint["verdict"] == "NO_OVERLAP"
    assert disjoint["token_overlap_n"] == 0
    assert broader["n_tokens"] == 256
    assert set(broader["token_ids"]).isdisjoint(set(real["token_ids"]))
    _assert_every_wall_time_is_estimate(broader)


def test_rank_by_diversity_not_row_count():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    ranked = tce.rank_plans(prior=real, screen=screen)
    order = ranked["needed_in_diversity_order"]
    assert order == ["router", "routed-output", "terminal-logit"]
    assert "hidden" not in order
    for row in ranked["plans"]:
        assert row["n_tokens"] == 256
        assert row["n_tokens"] != 4096
        _assert_every_wall_time_is_estimate(row)
    lesson = ranked["worked_example_real_corpus"]
    assert lesson["n_rows"] == 256
    assert lesson["route_union"] == 117
    assert "117" in lesson["lesson"]
    assert "4096" in ranked["rank_rule"]
    assert ranked["broader_traces"]["n_tokens"] == 256
    _assert_every_wall_time_is_estimate(ranked)


def test_funnel_bindings_do_not_mint_surfaces():
    real = tce.load_real_corpus()
    screen = tce.load_screen()
    needed = tce.surfaces_needed(real, screen)
    declared = set(needed["declared_from_screen"])
    for row in needed["funnel_bindings"]:
        if row["bound"]:
            assert row["surface"] in declared
        else:
            assert row["surface"] is None
    # teacher_corpus is a funnel input, not a surface name.
    inputs = {r["required_input"] for r in needed["funnel_bindings"]}
    assert "teacher_corpus" in inputs
    assert "teacher_corpus" not in declared
    assert "teacher_corpus" not in needed["needed_names"]


def test_parse_slash_surfaces_recovers_screen_list_only():
    text = (
        "collect broader teacher traces, distill "
        "router/hidden/routed-output/terminal-logit surfaces, then build a serializer"
    )
    assert tce.parse_slash_surfaces(text) == (
        "router",
        "hidden",
        "routed-output",
        "terminal-logit",
    )
    assert tce.parse_slash_surfaces("no surfaces named here") == ()
    add = "then add router/hidden/routed-output/terminal-logit surfaces before any meta runtime"
    assert tce.parse_slash_surfaces(add) == (
        "router",
        "hidden",
        "routed-output",
        "terminal-logit",
    )


def test_surfaces_needed_copes_when_screen_names_are_unparseable():
    fake_screen = {"next_gate": "do something else entirely"}
    fake_corpus = {"schema": "nope", "next_gate": ""}
    result = tce.surfaces_needed(fake_corpus, fake_screen, raise_on_refuse=False)
    assert result["accepted"] is False
    assert "SURFACE_NAMES_UNPARSEABLE" in result["refusals"]
    assert result["needed_names"] == []


def test_dedupe_against_absent_prior_is_refusal():
    result = tce.dedupe_against(None)
    assert result["accepted"] is False
    assert "ABSENT_PRIOR" in result["refusals"]
    assert result["verdict"] == "REFUSED"
