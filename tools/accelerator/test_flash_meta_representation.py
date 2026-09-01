from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from flash_meta_representation import (  # noqa: E402
    DEFAULT_BUDGET,
    DEFAULT_BYTES,
    DEFAULT_CENSUS,
    META_BPW_SEARCH_LADDER,
    SCHEMA,
    build_receipt,
)


def test_meta_budget_closes_below_one_without_creating_physical_ebpw_claim():
    receipt = build_receipt(
        census_path=DEFAULT_CENSUS,
        budget_path=DEFAULT_BUDGET,
        bytes_path=DEFAULT_BYTES,
    )
    metric = receipt["metric"]
    state = receipt["measurement_state"]
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "PROSPECTIVE_META_ONLY"
    assert metric["name"] == "meta_bpw"
    assert metric["prospective_target"] < 1.0
    assert metric["below_one_target"] is True
    assert metric["physical_ebpw"] is None
    search = receipt["budget_search"]
    assert search["physical_ebpw"] is None
    assert search["lowest_useful_target"] is None
    assert [row["meta_bpw_target"] for row in search["targets"]] == list(
        META_BPW_SEARCH_LADDER
    )
    assert all(
        row["status"] == "NOT_MEASURED" and row["physical_ebpw"] is None
        for row in search["targets"]
    )
    assert state["serialized_artifact"] == "NOT_BUILT"
    assert state["native_kernel"] == "NOT_BUILT"
    assert state["complete_token"] == "NOT_MEASURED"
    assert state["promotion_allowed"] is False


def test_meta_budget_accounts_for_every_census_family_and_keeps_hard_gates():
    receipt = build_receipt(
        census_path=DEFAULT_CENSUS,
        budget_path=DEFAULT_BUDGET,
        bytes_path=DEFAULT_BYTES,
    )
    families = receipt["family_budget"]
    assert families
    assert {row["family"] for row in families} == {
        "routed_experts",
        "ngram_embedding",
        "linear_attention_hyperconnection",
        "embedding_lm_head",
        "full_attention",
        "mlp_hyperconnection",
        "shared_expert",
        "other",
        "norm",
    }
    assert sum(row["source_fraction"] for row in families) == pytest.approx(1.0)
    for row in families:
        assert sum(row["ledger"].values()) == pytest.approx(row["meta_bpw_target"])
    contract = receipt["coherence_contract"]
    assert contract["router"]["topk_membership_match"] == 1.0
    assert contract["router"]["topk_order_match"] == 1.0
    assert contract["state"]["fallback_count"] == 0
