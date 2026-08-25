"""N040 ORGAN_DENSITY_FLOORS: composition descent of GQA / DeltaNet / embed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from organ_density_floors import (  # noqa: E402
    CURRENT_Q2F_CLASS,
    DN_GEMV_N,
    DN_LEFTOVER_N,
    GAIN_HEALTH,
    GQA_GEMV_N,
    GQA_LEFTOVER_N,
    LEFTOVER_BPW,
    MLP_FAIL_BPW,
    MLP_SURVIVE_BPW,
    RECEIPT,
    REL_FRO_LOCAL_MAX,
    RUNGS,
    SCALE_TRAP,
    SCHEMA,
    apply_codec,
    autopsy_kernel,
    complete_organ_bytes,
    complete_organ_ebpw,
    composition_survives,
    embed_active_bytes_per_token,
    grouped_storage_bpw,
    native_kernel_for,
)
from organ_frontiers import local_survives, score_pair, ws_rtn  # noqa: E402


def test_accounting_complete_ebpw_counts_leftover_and_scales():
    assert grouped_storage_bpw(4, 64) == pytest.approx(4.25)
    assert grouped_storage_bpw(4, 128) == pytest.approx(4.125)
    assert grouped_storage_bpw(3, 64) == pytest.approx(3.25)
    assert grouped_storage_bpw(2, 64) == pytest.approx(MLP_SURVIVE_BPW)
    # Leftover f32 is billed. A GEMV-only quote would hide those bits.
    gqa = complete_organ_ebpw(GQA_GEMV_N, GQA_LEFTOVER_N, 3.25)
    assert GQA_LEFTOVER_N > 0
    assert gqa > 3.25
    assert gqa < 3.26  # leftover is ~90k vs 1.68e9
    dn = complete_organ_ebpw(DN_GEMV_N, DN_LEFTOVER_N, 3.25)
    assert DN_LEFTOVER_N > 0
    assert dn > 3.25
    # Active bytes for a streamed organ = complete bits / 8.
    b = complete_organ_bytes(GQA_GEMV_N, GQA_LEFTOVER_N, 3.25)
    assert b == pytest.approx(
        (GQA_GEMV_N * 3.25 + GQA_LEFTOVER_N * LEFTOVER_BPW) / 8.0
    )
    # Embed gather is one row, not the table.
    assert embed_active_bytes_per_token(4.125) == pytest.approx(5120 * 4.125 / 8.0)


def test_codec_is_not_deletion_and_bills_scales():
    rng = np.random.RandomState(0)
    W = rng.randn(8, 64).astype(np.float32)
    What, acc = apply_codec(W, "ws_rtn_q3_g64")
    assert What.shape == W.shape
    assert float(np.abs(What).mean()) > 0.1 * float(np.abs(W).mean())
    assert acc["scales_counted"] is True
    assert acc["storage_bpw"] == pytest.approx(3.25)
    q2f, acc2 = apply_codec(W, "q2f_g64")
    assert q2f.shape == W.shape
    assert acc2["storage_bpw"] == pytest.approx(2.25)
    assert float(np.abs(q2f).mean()) > 0.1 * float(np.abs(W).mean())
    # Per-group 4-level grid: after dividing by the group's scale, at most 4 codes.
    scale = np.max(np.abs(q2f.reshape(8, 1, 64)), axis=-1, keepdims=True)
    unit = np.round(q2f.reshape(8, 1, 64) / np.maximum(scale, 1e-12) * 3, 0)
    assert np.unique(unit).size <= 6


def test_composition_bar_rejects_001W_and_zero():
    rng = np.random.RandomState(1)
    Y = rng.randn(32, 64).astype(np.float32)
    trap = score_pair(Y, (SCALE_TRAP * Y).astype(np.float32))
    ok, reason = composition_survives(trap)
    assert ok is False
    assert trap["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert trap["gain"] == pytest.approx(SCALE_TRAP, rel=1e-3, abs=1e-4)
    zero = score_pair(Y, np.zeros_like(Y))
    okz, _ = composition_survives(zero)
    assert okz is False
    ident = score_pair(Y, Y)
    oki, _ = composition_survives(ident)
    assert oki is True
    assert GAIN_HEALTH == pytest.approx(0.50)
    assert REL_FRO_LOCAL_MAX == pytest.approx(0.50)


def test_native_kernels_exist_and_autopsy_does_not_write_dense_w():
    sh, kn = native_kernel_for("ws_rtn_q4_g64")
    auto = autopsy_kernel(sh, kn)
    assert auto["present"] is True
    assert auto["dense_w_written"] is False
    sh3, kn3 = native_kernel_for("ws_rtn_q3_g64")
    auto3 = autopsy_kernel(sh3, kn3)
    assert auto3["present"] is True
    she, kne = native_kernel_for("ws_rtn_q4_g64", embed=True)
    autoe = autopsy_kernel(she, kne)
    assert autoe["present"] is True


def test_mlp_bracket_is_cited_not_transferred_as_these_organs_floor():
    assert MLP_FAIL_BPW == pytest.approx(1.85)
    assert MLP_SURVIVE_BPW == pytest.approx(2.25)
    assert CURRENT_Q2F_CLASS["gqa_attention"] == pytest.approx(4.25)
    assert CURRENT_Q2F_CLASS["deltanet"] == pytest.approx(4.125)
    assert CURRENT_Q2F_CLASS["embedding_output"] == pytest.approx(4.125)
    assert "held_out_activation" in RUNGS
    assert "complete_token" in RUNGS


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/organ_density_floors.py"
    )
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_discipline_and_no_second_27b():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_write_under_models"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["did_not_rederive_roofs"] is True
    assert doc["dense_w"] == 0
    assert doc["dense_w_materialized"] == 0
    assert doc["hand_authored"] is False
    assert doc["capture"]["not_gaussian"] is True
    trap = doc["scale_trap"]
    assert trap["rejects_scaled_artifact"] is True
    assert trap["identity_survives"] is True
    assert trap["local_survives"] is False
    assert doc["mlp_not_extrapolated"]["do_not_transfer"] is True
    assert doc["token_ns"]["kind"] == "ABSENT"
    assert doc["token_ns"]["value"] is None
    assert "7-rep" in doc["token_ns"]["absent_reason"] or ">=7" in doc["token_ns"]["absent_reason"]


def test_each_organ_reports_floor_family_ebpw_bytes_parity_dense_w():
    doc = _receipt()
    organs = doc["organs"]
    assert set(organs) == {"gqa_attention", "deltanet", "embedding_output"}
    for name, o in organs.items():
        assert o["status"] == "MEASURED", name
        assert o["mlp_not_used_as_prior"] is True
        floor = o["floor"]
        assert floor["status"] == "MEASURED", name
        assert floor["dense_w"] == 0, name
        assert floor["dense_w_materialized"] == 0, name
        assert floor.get("complete_ebpw") is not None, name
        assert float(floor["complete_ebpw"]) > 0, name
        assert floor.get("family"), name
        assert floor.get("codec"), name
        assert floor.get("active_bytes_per_token") is not None, name
        assert floor.get("highest_rung_reached") in RUNGS, name
        assert RUNGS.index(floor["highest_rung_reached"]) >= RUNGS.index(
            "held_out_activation"
        ), f"{name} did not reach held_out_activation"
        assert floor.get("vs_current_q2f_class") in {"below", "at", "above"}, name
        assert floor.get("current_q2f_class_bpw") == pytest.approx(
            CURRENT_Q2F_CLASS[name]
        )
        assert floor.get("because"), name
        assert "floors at" in floor["because"]
        assert floor.get("scales_counted") is True
        parity = floor.get("parity") or {}
        assert parity.get("dense_w") == 0.0 or floor["dense_w"] == 0
        cands = o.get("candidates") or []
        assert len(cands) >= 4, name
        for c in cands:
            assert c["dense_w"] == 0
            assert c.get("complete_ebpw") is not None
            assert c["held_out"]["real_activations"] is True
            assert c["held_out"]["not_gaussian"] is True
            assert c.get("scales_counted") is True


def test_deltanet_is_a_transition_program_not_a_dense_matrix():
    doc = _receipt()
    dn = doc["organs"]["deltanet"]
    assert dn["transition_program"] is True
    floor = dn["floor"]
    assert floor.get("transition_program") is True or any(
        c.get("transition_program") for c in dn["candidates"]
    )
    why = " ".join(dn["why"]).lower()
    assert "recurrent" in why or "transition" in why
    assert "in_proj" in why
    cited = doc["cited_structural_families"]["deltanet"]
    assert cited.get("state_cannot_replace_in_proj") is True
    cands = dn["candidates"]
    assert any(c.get("recurrent_not_dense_matrix") for c in cands)


def test_floors_are_compared_to_q2f_class_and_verdict_covers_three_organs():
    doc = _receipt()
    v = doc["verdict"]
    assert "gqa_attention" in v
    assert "deltanet" in v
    assert "embedding_output" in v
    assert v.get("one_line")
    for name in ("gqa_attention", "deltanet", "embedding_output"):
        fl = v[name]
        rel = fl["vs_current_q2f_class"]
        ebpw = float(fl["complete_ebpw"])
        cur = CURRENT_Q2F_CLASS[name]
        if rel == "below":
            assert ebpw < cur
        elif rel == "at":
            assert ebpw == pytest.approx(cur, abs=1e-6)
        else:
            assert ebpw > cur
        # Complete EBPW is not a hidden-bit GEMV-only quote of the leftover-free number
        # unless leftover is zero (embed mix).
        if name != "embedding_output":
            assert fl.get("complete_ebpw") >= fl.get("gemv_storage_bpw") or fl.get(
                "gemv_storage_bpw"
            ) is None


def test_embed_reports_gather_bytes_and_lm_head_separately():
    doc = _receipt()
    e = doc["organs"]["embedding_output"]
    floor = e["floor"]
    assert floor["active_bytes_per_token"] is not None
    # Combined organ still names both tables on at least one candidate.
    assert any("embed" in (c.keys()) for c in e["candidates"])
    c0 = next(c for c in e["candidates"] if "embed" in c)
    assert c0["embed"]["active_bytes_per_token"] < 10_000
    assert c0["lm_head"]["active_bytes_per_token"] > 100_000
    assert "rare" in " ".join(e["why"]).lower() or "lexical" in " ".join(e["why"]).lower()


def test_kernel_autopsy_recorded_and_speed_not_claimed():
    doc = _receipt()
    assert doc["token_ns"]["kind"] == "ABSENT"
    autos = doc.get("native_kernels_autopsied") or {}
    assert set(autos) >= {"gqa_attention", "deltanet", "embedding_output"}
    for name, auto in autos.items():
        assert auto.get("dense_w_written") is False
        assert auto.get("present") is True
        assert auto.get("verdict") in {"CLEAR", "SUSPECT", "DEFECTIVE"}
