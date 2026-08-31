import json

import pytest

from tools.future import ngram_school as ns
from tools.future._common import RECEIPTS, HARDWARE_FIELDS


REQUIRED_FAMILIES = (
    "packed_q4_control",
    "packed_q3_control",
    "product_codebooks",
    "residual_product_quantization",
    "hierarchical_codebooks",
    "clustered_dictionaries",
    "factorized_lookup",
    "context_conditioned_lookup",
    "generated_lookup",
    "semantic_hashing",
    "literal_exception_islands",
)


def _school():
    return ns.candidates()


def _by_id(cands, ident):
    return next(c for c in cands if c["id"] == ident)


def test_build_and_selftest_emit_sealed_receipt():
    out = ns.selftest()
    assert out.parent == RECEIPTS
    assert out.name == "NGRAM_SCHOOL.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.ngram_school.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert len(doc["seal_sha256"]) == 64
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "claim_boundary" in doc
    out2 = ns.build()
    doc2 = json.loads(out2.read_text())
    assert doc2["schema"] == doc["schema"]
    assert doc2["negative_control"]["storage_only_by_keyword_raises"] is True
    assert doc2["negative_control"]["trap_dominates_q4"] is False


def test_all_required_families_present_controls_first_class():
    cands = _school()
    ids = [c["id"] for c in cands]
    assert ids[:2] == ["packed_q4_control", "packed_q3_control"]
    assert set(ids) == set(REQUIRED_FAMILIES)
    assert len(ids) == len(REQUIRED_FAMILIES)
    q4 = _by_id(cands, "packed_q4_control")
    q3 = _by_id(cands, "packed_q3_control")
    assert q4["is_control"] is True
    assert q3["is_control"] is True
    for c in cands:
        if c["id"] not in ns.CONTROL_IDS:
            assert c["is_control"] is False


def test_every_candidate_has_the_five_axis_vector():
    for c in _school():
        ns._require_axes(c)
        t = ns.axis_tuple(c)
        assert len(t) == 5
        assert t[0] == c["executable_bytes"]
        assert t[1] == c["active_lookup_bytes_per_token"]
        assert t[2] == c["lookup_operations_per_token"]


def test_organ_shape_matches_sealed_ngram_engine():
    shape = ns.organ_shape()
    assert shape["source_payload_bytes"] == 102_466_171_160
    assert shape["table_bytes"] == 102_400_491_520
    assert shape["aux_bytes"] == 65_679_640
    assert shape["source_active_bytes_per_token"] == 65_680_600
    assert shape["n_rows"] == 320_001_536
    assert shape["shard_shape"] == [2_500_012, 160]
    assert shape["shard_count"] == 128
    assert shape["lookup_rows_nominal"] == 3
    assert shape["lookup_row_bytes"] == 320
    assert shape["active_is_ple_aux_dominated"] is True
    assert shape["table_bytes"] + shape["aux_bytes"] == shape["source_payload_bytes"]


def test_q4_packing_is_4_25_bpw_group64():
    assert ns.q4_bytes(64) == 34
    assert ns.q3_bytes(64) == 26
    q4 = ns.packed_q4_control()
    q3 = ns.packed_q3_control()
    assert q4["executable_bytes"] < ns.SOURCE_BYTES
    assert q3["executable_bytes"] < q4["executable_bytes"]
    assert q4["lookup_operations_per_token"] == 3
    assert q3["lookup_operations_per_token"] == 3
    assert q4["decode_cost_class"] == "GATHER_DEQUANT_Q4"
    assert q3["decode_cost_class"] == "GATHER_DEQUANT_Q3"


def test_q3_does_not_dominate_q4_despite_fewer_bytes():
    q4 = ns.packed_q4_control()
    q3 = ns.packed_q3_control()
    assert q3["executable_bytes"] < q4["executable_bytes"]
    assert q3["active_lookup_bytes_per_token"] < q4["active_lookup_bytes_per_token"]
    assert not ns.dominates(q3, q4)
    assert not ns.dominates(q4, q3)


def test_literal_islands_are_incomparable_to_q3():
    # Islands spend bytes and a branchy decode to bandage Q3 capability risk.
    # If sensitivity were ranked worse than CONTROL_Q3, Q3 would dominate
    # islands on every axis and the family would be a dead letter.
    q3 = ns.packed_q3_control()
    islands = ns.literal_exception_islands()
    assert islands["executable_bytes"] > q3["executable_bytes"]
    assert not ns.dominates(q3, islands)
    assert not ns.dominates(islands, q3)


def test_rank_refuses_storage_only_by_keyword():
    cands = _school()
    with pytest.raises(ns.StorageOnlyRankingError, match="refuses scalar"):
        ns.rank(cands, by="executable_bytes")


def test_rank_refuses_executable_bytes_axes_only():
    cands = _school()
    with pytest.raises(ns.StorageOnlyRankingError, match="incomplete-axis"):
        ns.rank(cands, axes=("executable_bytes",))


def test_rank_refuses_any_incomplete_axis_set():
    cands = _school()
    with pytest.raises(ns.StorageOnlyRankingError):
        ns.rank(
            cands,
            axes=("executable_bytes", "active_lookup_bytes_per_token"),
        )


def test_rank_requires_controls():
    clever = [c for c in _school() if not c["is_control"]]
    with pytest.raises(ns.ControlsMissingError, match="packed Q4/Q3"):
        ns.rank(clever)


def test_rank_requires_full_vector():
    q4 = ns.packed_q4_control()
    q3 = ns.packed_q3_control()
    broken = dict(q4)
    broken["id"] = "broken"
    del broken["lookup_operations_per_token"]
    with pytest.raises(ns.IncompleteVectorError):
        ns.rank([q4, q3, broken])


def test_halved_bytes_tripled_lookups_does_not_dominate_q4():
    """Negative control: the refusal must fire, and the trap must not win.

    A storage-only sort *would* put the trap ahead of packed Q4. rank()
    must reject that sort, and Pareto must not declare the trap a winner.
    """
    q4 = ns.packed_q4_control()
    q3 = ns.packed_q3_control()
    trap = ns.storage_trap(q4)
    assert trap["executable_bytes"] == q4["executable_bytes"] // 2
    assert trap["lookup_operations_per_token"] == q4["lookup_operations_per_token"] * 3
    assert trap["executable_bytes"] < q4["executable_bytes"]
    assert trap["lookup_operations_per_token"] > q4["lookup_operations_per_token"]

    naive = ns.naive_storage_order([trap, q4])
    assert naive[0] == trap["id"], "the naive ranking this guard exists to refuse"

    assert not ns.dominates(trap, q4)
    assert not ns.dominates(q4, trap)

    with pytest.raises(ns.StorageOnlyRankingError):
        ns.rank([q4, q3, trap], by="executable_bytes")

    result = ns.rank([q4, q3, trap])
    assert result["scalar_winner"] is None
    assert result["storage_only"] == "REFUSED"
    assert q4["id"] in result["pareto_front_ids"]
    assert trap["id"] in result["pareto_front_ids"]
    assert result["pareto_front_ids"][0] != trap["id"] or len(result["pareto_front_ids"]) > 1


def test_prove_storage_only_refusal_actually_fires():
    proof = ns.prove_storage_only_refusal()
    assert proof["storage_only_by_keyword_raises"] is True
    assert proof["storage_only_axes_raises"] is True
    assert proof["trap_dominates_q4"] is False
    assert proof["q4_dominates_trap"] is False
    assert proof["naive_storage_winner"] == "storage_trap_half_bytes_triple_lookups"
    assert proof["scalar_winner"] is None
    assert "packed_q4_control" in proof["pareto_ids_with_trap"]


def test_school_rank_has_no_scalar_winner_and_keeps_controls():
    result = ns.rank(_school())
    assert result["scalar_winner"] is None
    assert result["controls_present"] == list(ns.CONTROL_IDS)
    assert "packed_q4_control" in result["pareto_front_ids"]
    assert "packed_q3_control" in result["pareto_front_ids"]
    q4 = ns.packed_q4_control()
    for c in _school():
        if c["is_control"]:
            continue
        assert not ns.dominates(c, q4), f"{c['id']} must not dominate packed Q4"


def test_product_codebooks_cut_bytes_raise_lookups():
    q4 = ns.packed_q4_control()
    pq = ns.product_codebooks()
    assert pq["executable_bytes"] < q4["executable_bytes"]
    assert pq["lookup_operations_per_token"] > q4["lookup_operations_per_token"]
    assert pq["lookup_operations_per_token"] == 3 * 16
    assert not ns.dominates(pq, q4)


def test_clever_families_do_not_beat_q4_on_storage_alone():
    q4 = ns.packed_q4_control()
    for c in _school():
        if c["is_control"]:
            continue
        if c["executable_bytes"] < q4["executable_bytes"]:
            assert c["lookup_operations_per_token"] != q4["lookup_operations_per_token"] or (
                ns.DECODE_COST_ORDER[c["decode_cost_class"]]
                > ns.DECODE_COST_ORDER[q4["decode_cost_class"]]
                or ns.SENSITIVITY_ORDER[c["capability_sensitivity_class"]]
                > ns.SENSITIVITY_ORDER[q4["capability_sensitivity_class"]]
            )


def test_receipt_has_no_numeric_hardware_fields():
    out = ns.build()
    doc = json.loads(out.read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    assert doc["analytical_only"] is True
    assert doc["no_specimen_fit"] is True
    assert doc["no_timing"] is True
    assert doc["evidence_class"] == "STATIC_ONLY"
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(doc["family_budget"]["named_receipt_present"], bool)
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc


def test_family_budget_is_proxy_not_invention():
    budget = ns.family_budget()
    assert budget["named_receipt"].endswith("FLASH_META_REPRESENTATION_SUB1.json")
    assert budget["ngram_fair_share_bytes"] == ns.SOURCE_BYTES // 2
    assert budget["ngram_representation_actual_bytes"] is None
