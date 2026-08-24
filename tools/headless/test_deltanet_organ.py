"""N026 DeltaNet organ: two kernel changes, rec-state parity, 25.9% share."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from deltanet_organ import (  # noqa: E402
    CHANGES,
    DELTANET_WEIGHT_READ,
    GAP_GB_S,
    KERNEL_BAD,
    KERNEL_BASELINE,
    KERNEL_F4,
    KERNEL_TG32,
    N018_PRODUCTION_GB_S,
    N025_SHARE,
    PARENT_ROOT,
    RECEIPT,
    ROOF_GB_S,
    SCHEMA,
    kernel_autopsy,
    separated,
    shader_evidence,
    structure_accounting,
)

RECEIPT_DOC = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        reuse = os.environ.get("NOETIC_DELTANET_REUSE", "1") != "0"
        if reuse and RECEIPT.is_file():
            RECEIPT_DOC = json.loads(RECEIPT.read_text())
            if RECEIPT_DOC.get("schema") == SCHEMA:
                return RECEIPT_DOC
        from deltanet_organ import build, write_receipt  # noqa: WPS433

        RECEIPT_DOC = build(live=True)
        write_receipt(RECEIPT_DOC)
    return RECEIPT_DOC


def test_separation_helper_refuses_overlap():
    assert separated([1.0, 2.0], [3.0, 4.0]) is True
    assert separated([1.0, 3.0], [2.0, 4.0]) is False
    assert separated([], [1.0]) is False


def test_fused_kernels_declared_and_default_off():
    ev = shader_evidence()
    assert ev["shader_present"]
    assert ev["all_kernels_declared"], ev
    assert ev["wired"]
    assert ev["widen_f4_present"]
    assert ev["tg32_present"]
    assert ev["coalesced_load"]
    assert ev["float4_load"]
    assert ev["does_not_write_dense_w"] is True


def test_kernel_autopsy_does_not_flag_new_kernels_defective():
    autopsy = kernel_autopsy()
    assert autopsy["any_new_kernel_defective"] is False, autopsy
    assert autopsy["missing"] == []
    names = {w["kernel"] for w in autopsy["new_kernels"]}
    assert KERNEL_F4 in names
    assert KERNEL_TG32 in names


def test_structure_names_in_proj_qkvz_as_largest_dn_consumer():
    s = structure_accounting()
    assert s["largest_deltanet_specific_information_consumer"] == "linear_attn.in_proj_qkvz"
    assert s["state_cannot_replace_in_proj_wholesale"] is True
    assert s["separated_from_gqa_and_mlp"] is True
    assert s["prior_capacity_ratio_0p015"]["matches_prior"] is True
    assert s["dense_w_materialized"] == 0


def test_receipt_schema_roof_and_no_second_27b():
    doc = receipt()
    assert RECEIPT.is_file()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["roof_gb_s"] == ROOF_GB_S
    assert doc["n018_production_gb_s"] == N018_PRODUCTION_GB_S
    assert abs(doc["gap_gb_s"] - GAP_GB_S) < 0.2
    assert abs(doc["deltanet_25p9_share"] - N025_SHARE) < 1e-9
    assert doc["dense_w_materialized"] == 0
    assert doc["prior_not_rederived"]["did_not_retry_mlp_tile"] is True
    loc = doc["parent_immutable"]
    assert Path(loc["path"]).resolve() == PARENT_ROOT.resolve() or str(PARENT_ROOT) in loc["path"]
    assert loc["outside_worktree"] is True
    assert "10+ GiB" in doc["occupancy"]["note"] or "27B" in doc["occupancy"]["note"]
    law = doc["causal_benchmark_law"]
    assert law["kernel_identity"] == KERNEL_BASELINE
    assert KERNEL_F4 in law["changes"] and KERNEL_TG32 in law["changes"]
    assert KERNEL_BAD == law["bad_control"]


def test_receipt_reports_before_and_after_two_changes():
    doc = receipt()
    m = doc["measurement"]
    assert m["kind"] in ("MEASURED", "ABSENT")
    if m["kind"] != "MEASURED":
        assert m.get("absent_reason")
        return
    organ = m["organ"]
    assert organ["before"]["status"] == "MEASURED"
    assert organ["before"]["n_reps"] >= 7
    assert organ["before"]["weight_read_bytes"] == DELTANET_WEIGHT_READ
    assert organ["before"]["gpu_ns_min"] <= organ["before"]["gpu_ns_median"] <= organ["before"]["gpu_ns_max"]
    for ch in CHANGES:
        row = organ[f"after_{ch}"]
        assert row["status"] == "MEASURED", ch
        assert row["n_reps"] >= 7, ch
        assert row["gpu_ns_min"] <= row["gpu_ns_median"] <= row["gpu_ns_max"]
        assert row["achieved_gb_s"] is not None
        assert row["dense_w_materialized"] == 0
    assert len(m["changes"]) >= 2
    names = [c["change"] for c in m["changes"]]
    assert "widen_f4" in names and "coalesce_tg32" in names
    assert "25.9" in doc["one_line"] or "25.9" in (doc.get("reading") or "")
    assert "778.8" in doc["one_line"] or "356.7" in doc["one_line"]


def test_receipt_token_ids_recurrent_parity_noop_bad():
    doc = receipt()
    m = doc["measurement"]
    if m["kind"] != "MEASURED":
        assert m.get("absent_reason")
        return
    assert m["noop_empty"]["did_not_score"] is True
    assert m["noop_empty"]["must_not_score_as_an_organ"] is True
    bad = m["bad_control"]
    assert bad["kernel"] == KERNEL_BAD
    assert bad["rejected"] is True
    for ch in m["changes"]:
        assert ch.get("token_ids_unchanged") is True, ch
        assert ch.get("token_ids_before") == ch.get("token_ids_after")
        assert ch.get("token_ids_after")
        assert len(ch["token_ids_after"]) == 16
        assert ch.get("parity_rec_state") is not None
        assert ch["parity_rec_state"] < 1e-4, ch
        assert ch["parity_rec_out"] < 1e-4, ch
        if ch.get("gpu_ns_separated") is False:
            assert "NOT SEPARATED" in (ch.get("note") or "")
    assert m["bad_control"]["token_ids"] != m["changes"][0]["token_ids_after"]


def test_receipt_states_share_recovered_and_residual():
    doc = receipt()
    m = doc["measurement"]
    if m["kind"] != "MEASURED":
        assert m.get("absent_reason")
        return
    for ch in m["changes"]:
        assert "recovered_fraction_of_deltanet_25p9_share" in ch
        assert "recovered_fraction_of_roof_gap" in ch
    assert doc.get("residual_blocker")
    assert "in_proj" in doc["residual_blocker"] or "in_proj" in (doc.get("reading") or "")
    st = doc["structure"]
    assert st["largest_deltanet_specific_information_consumer"] == "linear_attn.in_proj_qkvz"
    assert doc["kernel_autopsy"]["any_new_kernel_defective"] is False
    gated = m["gated_delta"]
    assert gated["before"]["n_reps"] >= 7
    assert gated["after_widen_f4"]["n_reps"] >= 7
    assert gated["after_coalesce_tg32"]["n_reps"] >= 7
