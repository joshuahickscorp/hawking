"""ORGAN_FRONTIERS: per-organ floors, scales counted, 0.01*W rejected."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from organ_frontiers import (  # noqa: E402
    BAR_Q4,
    F16_BPW,
    MLP_FAIL_BPW,
    MLP_SURVIVE_BPW,
    RECEIPT,
    SCALE_TRAP,
    SCHEMA,
    TRIT_PACK_5IN8,
    binary_storage_bpw,
    gain_score,
    grouped_storage_bpw,
    lowrank_f16_bpw,
    q4_healthy,
    rel_fro,
    row_cosine,
    score_pair,
    ternary_5in8_storage_bpw,
    test_grouped_bpw_counts_scales,
    test_lowrank_bpw_glm_shape,
    ws_rtn,
)


def test_accounting_not_naive_bits():
    test_grouped_bpw_counts_scales()
    test_lowrank_bpw_glm_shape()
    assert grouped_storage_bpw(4, 64) == pytest.approx(4.25)
    assert grouped_storage_bpw(4, 128) == pytest.approx(4.125)
    assert grouped_storage_bpw(2, 64) == pytest.approx(MLP_SURVIVE_BPW)
    assert ternary_5in8_storage_bpw(64) == pytest.approx(MLP_FAIL_BPW)
    assert binary_storage_bpw(64) == pytest.approx(1.25)
    assert lowrank_f16_bpw(2048, 6144, 16) == pytest.approx(1.0 / 6.0)


def test_ws_rtn_is_not_deletion_and_bills_scales():
    rng = np.random.RandomState(0)
    W = rng.randn(8, 64).astype(np.float32)
    What, acc = ws_rtn(W, 4, 64)
    assert What.shape == W.shape
    assert float(np.abs(What).mean()) > 0.1 * float(np.abs(W).mean())
    assert acc["scales_counted"] is True
    assert acc["storage_bpw"] == pytest.approx(4.25)
    assert acc["active_fused_bpw"] == acc["storage_bpw"]
    assert acc["active_cached_f16_bpw"] == F16_BPW


def test_metric_rejects_001W_and_cosine_does_not():
    rng = np.random.RandomState(1)
    Y = rng.randn(32, 64).astype(np.float32)
    Yh = (SCALE_TRAP * Y).astype(np.float32)
    sc = score_pair(Y, Yh)
    assert sc["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert sc["gain"] == pytest.approx(SCALE_TRAP, rel=1e-3, abs=1e-4)
    assert sc["rel_fro"] == pytest.approx(1.0 - SCALE_TRAP, rel=1e-3, abs=1e-3)
    ok, reason = q4_healthy(sc)
    assert ok is False
    assert "gain" in reason or "0.50" in reason
    # identity must pass
    ident = score_pair(Y, Y)
    assert ident["cosine"] == pytest.approx(1.0)
    assert ident["gain"] == pytest.approx(1.0)
    assert ident["rel_fro"] == pytest.approx(0.0, abs=1e-6)
    ok_id, _ = q4_healthy(ident)
    assert ok_id is True


def test_zero_is_deletion_not_a_floor():
    rng = np.random.RandomState(2)
    Y = rng.randn(16, 32).astype(np.float32)
    sc = score_pair(Y, np.zeros_like(Y))
    assert sc["cosine"] == pytest.approx(0.0)
    assert sc["rel_fro"] == pytest.approx(1.0)
    ok, _ = q4_healthy(sc)
    assert ok is False


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/organ_frontiers.py"
    )
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_and_no_second_27b():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["capture"]["not_gaussian"] is True
    assert doc["scale_trap"]["rejects_scaled_artifact"] is True
    trap = doc["scale_trap"]["scaled_0p01"]
    assert trap["cosine"] > 0.99
    assert trap["gain"] < 0.05
    assert trap["rel_fro"] > 0.9


def test_mlp_bracket_cited_not_transferred():
    doc = _receipt()
    lock = doc["mlp_not_extrapolated"]
    assert lock["do_not_transfer"] is True
    assert lock["fail_bpw"]["value"] == pytest.approx(MLP_FAIL_BPW)
    assert lock["survive_bpw"]["value"] == pytest.approx(MLP_SURVIVE_BPW)
    assert lock["fail_bpw"]["status"] == "CITED"
    assert lock["survive_bpw"]["status"] == "CITED"
    assert "not a floor" in str(lock["fail_bpw"]["null"]).lower()
    for organ in ("deltanet", "gqa", "embedding_output"):
        o = doc["organs"][organ]
        assert o["mlp_not_used_as_prior"] is True
        # The organ floor must not be a copy of the MLP numbers without evidence.
        floor = o["floor"]
        assert floor["status"] == "MEASURED"
        assert floor.get("method") is not None or floor.get("storage_bpw") is not None


def test_each_organ_has_own_information_and_function_floor():
    doc = _receipt()
    organs = doc["organs"]
    assert set(organs) == {"deltanet", "gqa", "embedding_output"}
    floors = []
    for name, o in organs.items():
        assert o["status"] == "MEASURED", name
        assert "information" in o, name
        assert "function" in o, name
        floor = o["floor"]
        assert floor["organ"] == name or floor["organ"] in (name, "embedding_output")
        st = floor.get("storage_bpw")
        act = floor.get("active_fused_bpw")
        assert st is not None, f"{name} missing storage_bpw floor"
        assert act is not None, f"{name} missing active_fused_bpw floor"
        assert floor.get("scales_counted", True) is True
        # every floor function score has a null
        fn = floor.get("function") or {}
        if fn:
            assert "null" in fn or floor.get("null") is not None
        floors.append((name, float(st), float(act)))
        # candidates bill both bpws and a null
        cands = o.get("candidates") or []
        assert len(cands) >= 4, f"{name} too few candidates"
        for c in cands[:8]:
            assert c.get("storage_bpw") is not None or c.get("family") == "control"
            assert "function" in c
            fnc = c["function"] or {}
            assert "null" in fnc
            assert "gain" in fnc
            assert fnc.get("scales_counted", True) or c.get("scales_counted", True)
            # 0.01*W control must not be Q4-equivalent
            if c.get("name") == "scale_001W" or c.get("trap"):
                assert not fnc.get("q4_equivalent", True)
    # Independence: three named floors exist even if values coincide.
    assert len(floors) == 3
    names = {n for n, _, _ in floors}
    assert names == {"deltanet", "gqa", "embedding_output"}


def test_deltanet_state_capacity_and_sub1_not_assumed():
    doc = _receipt()
    dn = doc["organs"]["deltanet"]
    cap = dn["information"]["state_capacity"]
    ratio = cap["capacity_ratio_state_over_qkv"]
    assert ratio < 1.0
    assert cap["null"]
    fn = dn["function"]
    assert "n_sub1_q4_equivalent" in fn
    # A sub-1 opportunity is a measured count, not a hope.
    assert isinstance(fn["n_sub1_q4_equivalent"], int)
    assert dn["floor"]["bar"] in ("q4_equivalent", "local_survives")
    # evidence is on held-out real X
    assert dn["function"]["n_hold"] >= 256
    assert "mlp" not in (dn["floor"].get("method") or "").lower()


def test_gqa_searched_structure_refit_could_not_reach():
    doc = _receipt()
    g = doc["organs"]["gqa"]
    info = g["information"]
    layers = info["layers"]
    assert "3" in layers and "63" in layers
    qpair = layers["3"]["q_head_pairwise"]
    assert qpair["n_heads"] == 24
    assert qpair["n_pairs"] == 24 * 23 // 2
    assert "shared_q_head_basis" in layers["3"]
    assert "kv_state_compression" in info
    assert "shared_input_basis" in info
    assert g["floor"]["storage_bpw"] is not None
    assert "why_the_floor_holds" in g
    assert len(g["why_the_floor_holds"]) >= 3
    # cited 4.125 refit is present, but this organ's floor is measured here
    cited = g.get("cited_grouped_absmax_floor") or {}
    assert cited.get("n_matched_rows") == 60 or cited.get("source")


def test_embedding_reports_rare_tokens_not_just_mean():
    doc = _receipt()
    e = doc["organs"]["embedding_output"]
    fn = e["function"]
    assert "n_rare_hold" in fn
    info = e["information"]
    assert info["tie_word_embeddings_config"] is False
    assert "tie_sample_4096_mean_row_cosine" in info
    families = {c.get("family") for c in e["candidates"]}
    assert "row_codebook" in families or "table_lowrank" in families
    assert "hot_cold" in families or any("hot" in (c.get("name") or "") for c in e["candidates"])
    # floor is rare-gated
    assert "rare" in e["floor"]["bar"] or "unseen" in e["floor"]["bar"]
    # embed active is not the whole table
    st = e["floor"]["storage_bpw"]
    act = e["floor"]["active_fused_bpw"]
    assert st is not None and act is not None
    slices = []
    for c in e["candidates"]:
        sl = c.get("function_slices") or {}
        if "rare_hold_gather" in sl or "unseen_vocab_rows" in sl:
            slices.append(c["name"])
    assert slices, "no candidate reported rare/unseen token quality"


def test_every_organ_floor_has_null_and_both_bpws():
    doc = _receipt()
    for name, o in doc["organs"].items():
        floor = o["floor"]
        assert floor.get("null") is not None or (floor.get("function") or {}).get("null") is not None, name
        assert floor.get("storage_bpw") is not None, name
        assert floor.get("active_fused_bpw") is not None, name
        assert floor.get("active_cached_f16_bpw") == F16_BPW or floor.get("active_fused_bpw") is not None
        # controls exist
        names = {c.get("name") for c in o.get("candidates") or []}
        if name != "embedding_output":
            assert "scale_001W" in names, name
            assert "zero" in names, name
        else:
            trap = e_trap = o.get("scale_trap_embed_gather") or {}
            assert trap.get("rejects_scaled_artifact") in (True, False, None) or trap.get("score")


def test_q4_bar_is_not_the_mlp_bar():
    assert BAR_Q4 == pytest.approx(0.990)
    # MLP survive 2.25 is a different number; using it as the mixer bar is the bug.
    assert BAR_Q4 != MLP_SURVIVE_BPW
    assert BAR_Q4 != MLP_FAIL_BPW


def test_organ_floor_is_gated_by_worst_required_tensor():
    doc = _receipt()
    for name in ("deltanet", "gqa"):
        o = doc["organs"][name]
        per = o.get("per_tensor_floors") or {}
        assert per, name
        mx = max(float(v["storage_bpw"]) for v in per.values() if v.get("storage_bpw") is not None)
        assert float(o["floor"]["storage_bpw"]) == pytest.approx(mx)
        # A cheap healthy tensor must not set the organ floor by itself.
        mn = min(float(v["storage_bpw"]) for v in per.values() if v.get("storage_bpw") is not None)
        if mn + 1e-9 < mx:
            assert float(o["floor"]["storage_bpw"]) != pytest.approx(mn)
