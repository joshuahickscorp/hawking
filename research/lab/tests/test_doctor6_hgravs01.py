"""HGRAVS01 activation-weighted low-rank family on the doctor6 ballot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lab.operators.ascension_dual_gravity_worker import MAGIC_ACT_SVD, _parse_container
from lab.operators.doctor6.prescribe import LE_PER_BAND, prescribe
from lab.operators.doctor6.rungs import (
    RUNG_ORDER,
    apply_rung,
    list_rung_status,
    quant_binary,
    score_vs_incumbent,
)
from lab.operators.hgravs01_adapter import (
    HGRAVS01_RUNGS,
    HGRAVS01_SCHEMA,
    apply_hgravs01_to_compose_result,
    clamp_rank,
    encode_hgravs01,
    evaluate_hgravs01_candidates,
    hgravs01_budgets,
    parse_hgravs01_rung,
)
from lab.tests.test_doctor6_parallel import (
    _build_capture,
    _build_model,
    _scientific,
)

# Baseline sample size must stay 16 LE/band after this lane.
assert LE_PER_BAND == 16


def _well_conditioned_organ(*, n_fit: int = 220, hidden: int = 192, out: int = 192):
    """Matrix dims and fit rows both large enough that r192 is well-posed."""
    rng = np.random.default_rng(21)
    rank_plant = 8
    left = rng.standard_normal((out, rank_plant), dtype=np.float32) * 0.2
    right = rng.standard_normal((rank_plant, hidden), dtype=np.float32) * 0.2
    W = left @ right
    X_fit = rng.standard_normal((n_fit, hidden), dtype=np.float32)
    X_hold = rng.standard_normal((80, hidden), dtype=np.float32)
    W_inc, _ = quant_binary(W)
    return W, X_fit, X_hold, W_inc


def test_hgravs01_r128_and_r192_are_searchable_rungs() -> None:
    labels = {str(b["label"]) for b in hgravs01_budgets()}
    assert "r128_b3" in labels
    assert "r192_b4" in labels
    assert "hgravs01_r128_b3" in HGRAVS01_RUNGS
    assert "hgravs01_r192_b4" in HGRAVS01_RUNGS
    assert "hgravs01_r128_b3" in RUNG_ORDER
    assert "hgravs01_r192_b4" in RUNG_ORDER
    assert "incumbent_binary" in RUNG_ORDER
    statuses = {row["entry"]: row for row in list_rung_status()}
    assert statuses["activation_weighted"]["status"] == "live"
    assert statuses["hgravs01"]["rank_searchable"] is True
    assert parse_hgravs01_rung("hgravs01_r192_b4") == (192, 4)


def test_hgravs01_appears_as_candidate_with_rank_recorded() -> None:
    W, X_fit, X_hold, W_inc = _well_conditioned_organ(n_fit=220)
    ev = evaluate_hgravs01_candidates(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        W_incumbent=W_inc,
        target_cos=0.9857,
        max_legal_bpw=1.5,
        best_cos=-1.0,
        already_clears_target=False,
    )
    rungs = [m["rung"] for m in ev["measurements"]]
    assert "hgravs01_r128_b3" in rungs
    assert "hgravs01_r192_b4" in rungs
    by_rung = {m["rung"]: m for m in ev["measurements"]}
    r192 = by_rung["hgravs01_r192_b4"]
    assert r192["requested_rank"] == 192
    assert r192["rank"] == 192
    assert r192["rank_clamped_to_n_fit"] is False
    assert r192["n_fit_rows"] == 220
    assert r192["family"] == "hgravs01"
    assert r192["activation_weighted"] is True
    assert r192["low_rank"] is True
    assert r192["hgravs"] is True
    assert r192["schema"] == HGRAVS01_SCHEMA

    applied = apply_rung(
        "hgravs01_r192_b4",
        W,
        X_fit,
        organ_key="synth.gate_proj.weight",
        sensitivity=0.5,
        seed=0,
    )
    assert applied.meta["rank"] == 192
    assert applied.meta["requested_rank"] == 192
    assert applied.meta["family"] == "hgravs01"


def test_hgravs01_honest_billing_includes_factors_and_scales() -> None:
    W, X_fit, _, _ = _well_conditioned_organ(n_fit=220)
    encoded = encode_hgravs01(W, X_fit, rank=128, bits=3)
    header, body = _parse_container(encoded["payload"], expected_magic=MAGIC_ACT_SVD)

    ledger = encoded["ledger"]
    assert encoded["payload"][:8] == MAGIC_ACT_SVD
    assert encoded["payload_bytes"] == len(encoded["payload"])
    assert encoded["payload_bytes"] == ledger["total_bytes"]
    assert ledger["total_bytes"] == (
        ledger["magic_bytes"]
        + ledger["header_len_field_bytes"]
        + ledger["header_bytes"]
        + ledger["left_body_bytes"]
        + ledger["right_body_bytes"]
    )
    assert ledger["scale_bytes"] > 0
    assert ledger["code_bytes"] > 0
    assert ledger["factor_body_bytes"] == ledger["scale_bytes"] + ledger["code_bytes"]
    assert ledger["body_matches_ledger"] is True
    assert header["schema"] == HGRAVS01_SCHEMA
    assert header["rank"] == 128
    assert len(body) == header["left_body_bytes"] + header["right_body_bytes"]
    # BPW is the physical container, not factor codes alone.
    codes_only_bpw = 8.0 * ledger["code_bytes"] / W.size
    assert encoded["component_bpw"] > codes_only_bpw
    assert encoded["component_bpw"] == 8.0 * encoded["payload_bytes"] / W.size


def test_hgravs01_over_ceiling_is_refused() -> None:
    W, X_fit, X_hold, W_inc = _well_conditioned_organ(n_fit=220)
    billed = [
        encode_hgravs01(W, X_fit, rank=int(b["rank"]), bits=int(b["bits"]))
        for b in hgravs01_budgets()
    ]
    cheapest = min(float(e["component_bpw"]) for e in billed)
    # Ceiling below every honestly billed point — none may be admitted.
    ceiling = cheapest * 0.5
    assert ceiling < cheapest
    ev = evaluate_hgravs01_candidates(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        W_incumbent=W_inc,
        target_cos=0.0,
        max_legal_bpw=ceiling,
        best_cos=-1.0,
        already_clears_target=False,
    )
    assert ev["winner"] is None
    assert ev["measurements"]
    assert ev["dropped"]
    for row in ev["measurements"]:
        assert row["kept"] is False
        assert row["legal_under_budget"] is False
        assert row["component_bpw"] > ceiling
    assert all("over_budget" in d["reason"] for d in ev["dropped"])

    compose_like = {
        "measurements": [],
        "dropped_rungs": [],
        "chain": ["incumbent_binary"],
        "W_hat": W_inc,
        "W_incumbent": W_inc,
        "payload_bytes": 8,
        "component_bpw": 1.1,
        "prescribed_cosine": 0.5,
        "incumbent_cosine": 0.5,
        "clears_target": False,
        "fallback_to_incumbent": True,
        "meta": {"codec": "binary_g128"},
    }
    out = apply_hgravs01_to_compose_result(
        compose_like,
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        target_cos=0.9857,
        max_legal_bpw=ceiling,
    )
    assert out["chain"] == ["incumbent_binary"]
    assert out["fallback_to_incumbent"] is True
    assert any(m["rung"].startswith("hgravs01_") for m in out["measurements"])

    # Mid-band ceiling: cheaper points may stay legal; r192 must not sneak in
    # if its honest BPW is above the ceiling.
    r192_bpw = next(
        float(e["component_bpw"]) for e in billed if e["requested_rank"] == 192
    )
    mid = (cheapest + r192_bpw) / 2.0
    ev_mid = evaluate_hgravs01_candidates(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        W_incumbent=W_inc,
        target_cos=0.0,
        max_legal_bpw=mid,
        best_cos=-1.0,
        already_clears_target=False,
    )
    r192_row = next(m for m in ev_mid["measurements"] if m["rung"] == "hgravs01_r192_b4")
    if r192_bpw > mid:
        assert r192_row["kept"] is False
        assert r192_row["legal_under_budget"] is False


def test_hgravs01_row_starved_reports_reduced_rank() -> None:
    W, _, X_hold, W_inc = _well_conditioned_organ(n_fit=220)
    X_fit = np.random.default_rng(4).standard_normal((7, W.shape[1])).astype(np.float32)
    assert clamp_rank(192, 7) == 7
    encoded = encode_hgravs01(W, X_fit, rank=192, bits=4)
    assert encoded["requested_rank"] == 192
    assert encoded["achieved_rank"] <= 7
    assert encoded["rank_clamped_to_n_fit"] is True
    assert encoded["n_fit_rows"] == 7
    # Must not pretend the requested rank was achieved.
    assert encoded["achieved_rank"] != 192

    ev = evaluate_hgravs01_candidates(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        W_incumbent=W_inc,
        target_cos=0.9857,
        max_legal_bpw=8.0,
        best_cos=-1.0,
        already_clears_target=False,
    )
    r192 = next(m for m in ev["measurements"] if m["rung"] == "hgravs01_r192_b4")
    assert r192["requested_rank"] == 192
    assert r192["rank"] <= 7
    assert r192["rank_clamped_to_n_fit"] is True
    assert r192["meta"]["rank"] <= 7


def test_hgravs01_workers_1_and_8_deterministic(tmp_path: Path) -> None:
    capture = _build_capture(tmp_path, hidden=8)
    model = _build_model(tmp_path, hidden=8)
    common = dict(
        model_id="toy",
        model_dir=model,
        capture=capture,
        target_bpw=1.5,
        le_per_band=1,
        min_rows=4,
        device="cpu",
        qat_steps=2,
        qat_lr=1e-3,
        memory_bounded=True,
        max_rows_per_expert=8,
    )
    rx1 = prescribe(**common, workers=1, out_path=tmp_path / "rx1.json")
    rx8 = prescribe(**common, workers=8, out_path=tmp_path / "rx8.json")
    assert rx1.get("organs"), rx1
    hgravs_rungs = set()
    for organ in rx1["organs"]:
        names = [m.get("rung") for m in organ.get("measurements") or []]
        assert any(str(n).startswith("hgravs01_") for n in names), organ["tensor_name"]
        hgravs_rungs.update(n for n in names if str(n).startswith("hgravs01_"))
        for m in organ.get("measurements") or []:
            if str(m.get("rung", "")).startswith("hgravs01_"):
                assert "rank" in m
                assert m.get("family") == "hgravs01"
    assert "hgravs01_r128_b3" in hgravs_rungs
    assert "hgravs01_r192_b4" in hgravs_rungs
    assert [r["tensor_name"] for r in rx1["organs"]] == [
        r["tensor_name"] for r in rx8["organs"]
    ]
    assert [r["chain"] for r in rx1["organs"]] == [r["chain"] for r in rx8["organs"]]
    assert _scientific(rx1) == _scientific(rx8)


def test_hgravs01_payload_is_physical_hgravs01_container() -> None:
    W, X_fit, _, _ = _well_conditioned_organ(n_fit=64)
    encoded = encode_hgravs01(W, X_fit, rank=8, bits=3)
    assert encoded["payload"][:8] == MAGIC_ACT_SVD
    header, body = _parse_container(encoded["payload"], expected_magic=MAGIC_ACT_SVD)
    assert header["schema"] == HGRAVS01_SCHEMA
    left = header["left"]
    right = header["right"]
    assert int(left["scale_bytes"]) > 0
    assert int(right["scale_bytes"]) > 0
    assert encoded["ledger"]["scale_bytes"] == int(left["scale_bytes"]) + int(
        right["scale_bytes"]
    )
    assert len(body) == int(header["left_body_bytes"]) + int(header["right_body_bytes"])
    W_inc, _ = quant_binary(W)
    X_hold = X_fit[:16]
    score = score_vs_incumbent(
        W=W, X_hold=X_hold, W_hat=encoded["W_hat"], W_incumbent=W_inc
    )
    assert -1.0 <= score["output_cosine"] <= 1.0
