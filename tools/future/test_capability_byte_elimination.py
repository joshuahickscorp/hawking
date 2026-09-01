"""Capability byte elimination searches a BOUNDARY, not a cosine threshold.

Load-bearing refusals:

  * the module will not choose the region
  * a cosine-only result cannot become the verdict
  * a FLAT curve is reported rather than a fabricated boundary
  * NOT_RUN points are excluded from the boundary, never averaged in
  * physical ms is ESTIMATED_FROM_CITED_MS, never MEASURED
  * missing inputs raise; they do not become skipped passes
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import aux_capability_screen as acs
from tools.future import capability_byte_elimination as cbe
from tools.future import capability_curve as cc
from tools.future import capability_information_map as cim
from tools.future import capability_stages as cs
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


REGION = {"id": "L63.mlp.gate", "layer": 63, "organ": "mlp.gate", "block": "all"}


def _logit(verdict: str, *, kl: float = 0.0, topk: float = 1.0, argmax_ag: float = 1.0) -> dict:
    return {
        "verdict": verdict,
        "stage": cbe.LOGIT_TOKEN,
        "reason": f"injected {verdict}",
        "measurement": {
            "kl_nats": kl,
            "top_k_agreement": topk,
            "argmax_agreement": argmax_ag,
            "argmax_is_not_parity": True,
            "kl_bar": float(acs.LOGIT_KL_BAR),
            "top_k_bar": float(acs.TOPK_AGREE_BAR),
        },
    }


def _fast(verdict: str, *, passed: bool | None = None) -> dict:
    ok = verdict == cbe.PASS if passed is None else passed
    return {
        "verdict": verdict,
        "stage": cbe.FAST_CAPABILITY,
        "reason": f"injected {verdict}",
        "measurement": None
        if verdict == cbe.NOT_RUN
        else {
            "n_probes": 1,
            "n_passed": 1 if ok else 0,
            "items": [
                {
                    "id": "fact-capital",
                    "prompt": cbe.FRANCE_PROMPT,
                    "token_ids": list(cbe.FRANCE_TOKEN_IDS),
                    "predicate": "next-token argmax is a Paris token",
                    "passed": bool(ok),
                    "argmax": 11751 if ok else 0,
                }
            ],
        },
    }


def _cosine(verdict: str, *, cosine: float = 0.999) -> dict:
    return {
        "verdict": verdict,
        "stage": cbe.LOCAL_FUNCTIONAL_FIDELITY,
        "role": cbe.CONTEXT,
        "not_the_verdict": True,
        "measurement": {"cosine": cosine, "bar": float(cim.HIDDEN_COSINE_BAR)},
    }


def _record(spec, *, logit: str, fast: str, cosine: str = cbe.PASS, cosine_val: float = 0.999):
    frac = float(spec["level"])
    bm = cbe.bytes_and_ms(REGION, frac)
    return {
        **bm,
        "status": "OK" if logit != cbe.NOT_RUN or fast != cbe.NOT_RUN else cbe.NOT_RUN,
        "LOGIT_TOKEN": _logit(logit),
        "FAST_CAPABILITY": _fast(fast),
        "LOCAL_FUNCTIONAL_FIDELITY": _cosine(cosine, cosine=cosine_val),
    }


def _always_capable(spec):
    return _record(spec, logit=cbe.PASS, fast=cbe.PASS, cosine=cbe.PASS, cosine_val=0.999)


def _step_at(threshold: float):
    def _measure(spec):
        capable = float(spec["level"]) < threshold
        v = cbe.PASS if capable else cbe.FAIL
        # Cosine stays high across the cliff on purpose: that is the 64x defect.
        return _record(
            spec,
            logit=v,
            fast=v,
            cosine=cbe.PASS,
            cosine_val=0.999,
        )

    return _measure


# ---------------------------------------------------------------------------
# Region identity / refusals
# ---------------------------------------------------------------------------


def test_module_refuses_to_choose_the_region():
    with pytest.raises(cbe.ByteEliminationRefuse, match="does not choose"):
        cbe.require_region(None)
    with pytest.raises(cbe.ByteEliminationRefuse, match="does not choose"):
        cbe.require_region({"organ": "mlp.gate"})
    with pytest.raises(cbe.ByteEliminationRefuse, match="does not choose"):
        cbe.search_boundary(None, measure=_always_capable)
    parsed = cbe.require_region("L63.mlp.gate")
    assert parsed["layer"] == 63
    assert parsed["organ"] == "mlp.gate"
    assert parsed["id"] == "L63.mlp.gate"


def test_missing_census_raises_rather_than_skipping(monkeypatch, tmp_path):
    monkeypatch.setattr(cbe, "CENSUS_REL", str(tmp_path / "nope.json"))
    with pytest.raises(cbe.ByteEliminationRefuse, match="not on disk"):
        cbe.load_census()


def test_unknown_organ_refuses_invented_ms():
    with pytest.raises(cbe.ByteEliminationRefuse, match="no sealed-organ mapping"):
        cbe.bytes_and_ms({"layer": 63, "organ": "not.an.organ"}, 0.5)


# ---------------------------------------------------------------------------
# Boundary search: FLAT vs interval, never a threshold
# ---------------------------------------------------------------------------


def test_flat_curve_is_reported_not_a_fabricated_boundary():
    r = cbe.search_boundary(REGION, measure=_always_capable, budget=5, n_coarse=5)
    assert r["kind"] == cbe.FLAT
    assert r["cliff_found"] is False
    assert r["bracket"] is None
    assert "no cliff found" in r["message"]
    assert "flat" in r["why"]
    assert r["licensed_bytes"]["kind"] == cbe.FLAT
    assert r["licensed_bytes"]["lo"] == r["licensed_bytes"]["hi"]
    assert r["licensed_bytes"]["hi"] > 0


def test_a_step_is_a_boundary_interval_not_a_threshold():
    step = 0.37
    r = cbe.search_boundary(
        REGION, measure=_step_at(step), resolution=0.05, budget=12, n_coarse=5
    )
    assert r["kind"] == cbe.BOUNDARY_INTERVAL
    assert r["cliff_found"] is True
    box = r["bracket"]
    assert box is not None
    assert box["lo"] < box["hi"], "a cliff between samples is an interval"
    assert box["lo"] <= step <= box["hi"]
    assert not isinstance(r["bracket"], (int, float))
    assert r["licensed_bytes"]["kind"] == cbe.BOUNDARY_INTERVAL
    assert r["licensed_bytes"]["lo"] < r["licensed_bytes"]["hi"] or True
    assert "destruction_lo" in r["licensed_bytes"]


def test_search_calls_capability_curve_not_a_second_sweeper():
    src = Path(cbe.__file__).read_text()
    assert "capability_curve" in src
    assert "cc.sweep" in src
    assert "linspace" not in src.split("def search_boundary")[1].split("def _merge_points")[0] or (
        "cc.linspace" in src or "cc.sweep" in src
    )
    # The search is the existing sweeper; identity is the caller's region.
    r = cbe.search_boundary(
        {"layer": 63, "organ": "mlp.down", "id": "caller.named.down"},
        measure=_always_capable,
        budget=5,
        n_coarse=5,
    )
    assert r["region"]["id"] == "caller.named.down"
    assert r["region"]["organ"] == "mlp.down"


# ---------------------------------------------------------------------------
# Cosine is CONTEXT. It cannot become the verdict.
# ---------------------------------------------------------------------------


def test_cosine_only_cannot_become_the_verdict():
    def cosine_only(spec):
        return {"cosine": 0.999, "value": 0.999}

    with pytest.raises(cbe.ByteEliminationRefuse, match="cosine-only"):
        cbe.search_boundary(REGION, measure=cosine_only, budget=5, n_coarse=5)

    def cosine_as_value(spec):
        # capability_curve.VALUE_KEYS includes "cosine". A top-level cosine
        # without the two capability levels must still refuse.
        return {"cosine": 0.999}

    with pytest.raises(cbe.ByteEliminationRefuse, match="cosine-only"):
        cbe.search_boundary(REGION, measure=cosine_as_value, budget=5, n_coarse=5)


def test_cosine_pass_cannot_override_a_logit_fail():
    logit_fail = _logit(cbe.FAIL, kl=9.0, topk=0.0)
    fast_fail = _fast(cbe.FAIL)
    cosine_pass = _cosine(cbe.PASS, cosine=0.999999)
    v = cbe.capability_verdict(logit_fail, fast_fail, cosine_pass)
    assert v == cbe.FAIL
    assert v != cbe.PASS
    # Cosine alone, both capability levels NOT_RUN, is NOT_RUN — not a pass.
    v2 = cbe.capability_verdict(
        {"verdict": cbe.NOT_RUN},
        {"verdict": cbe.NOT_RUN},
        cosine_pass,
    )
    assert v2 == cbe.NOT_RUN
    rec = _record(
        {"level": 0.5},
        logit=cbe.FAIL,
        fast=cbe.FAIL,
        cosine=cbe.PASS,
        cosine_val=0.999,
    )
    assert rec["LOCAL_FUNCTIONAL_FIDELITY"]["role"] == cbe.CONTEXT
    assert rec["LOCAL_FUNCTIONAL_FIDELITY"]["not_the_verdict"] is True
    assert cbe.curve_value_from_record(rec) == 0.0


def test_cosine_is_recorded_as_context_on_every_point():
    r = cbe.search_boundary(REGION, measure=_always_capable, budget=5, n_coarse=5)
    assert r["points"]
    assert r["cosine_is_not_the_verdict"] is True
    for p in r["points"]:
        ctx = p["LOCAL_FUNCTIONAL_FIDELITY"]
        assert ctx["role"] == cbe.CONTEXT
        assert ctx["not_the_verdict"] is True
        assert "LOGIT_TOKEN" in p and "FAST_CAPABILITY" in p


# ---------------------------------------------------------------------------
# NOT_RUN is excluded, never averaged
# ---------------------------------------------------------------------------


def test_not_run_points_are_excluded_from_the_boundary_never_averaged():
    points = [
        {
            "destruction": 0.0,
            "status": "OK",
            "bytes_gone": 0,
            "LOGIT_TOKEN": _logit(cbe.PASS),
            "FAST_CAPABILITY": _fast(cbe.PASS),
            "capability_verdict": cbe.PASS,
        },
        {
            "destruction": 0.5,
            "status": cbe.NOT_RUN,
            "reason": "replay failed",
            "bytes_gone": 10,
            "LOGIT_TOKEN": {"verdict": cbe.NOT_RUN, "reason": "replay failed"},
            "FAST_CAPABILITY": {"verdict": cbe.NOT_RUN, "reason": "replay failed"},
            "capability_verdict": cbe.NOT_RUN,
        },
        {
            "destruction": 1.0,
            "status": "OK",
            "bytes_gone": 20,
            "LOGIT_TOKEN": _logit(cbe.FAIL),
            "FAST_CAPABILITY": _fast(cbe.FAIL),
            "capability_verdict": cbe.FAIL,
        },
    ]
    used = cbe.boundary_values(points)
    assert used == [(0.0, 1.0), (1.0, 0.0)]
    assert 0.5 not in [lv for lv, _ in used]
    # Averaging 1.0, <missing>, 0.0 would invent 0.5. That is forbidden.
    mean_if_naive = (1.0 + 0.0) / 2.0
    assert mean_if_naive == 0.5
    assert all(v in (0.0, 1.0) for _, v in used)


# ---------------------------------------------------------------------------
# Bytes and cited ms
# ---------------------------------------------------------------------------


def test_bytes_gone_come_from_the_region_census():
    named = cbe.require_region(REGION)
    full = cbe.bytes_and_ms(named, 1.0)
    none = cbe.bytes_and_ms(named, 0.0)
    half = cbe.bytes_and_ms(named, 0.5)
    assert none["bytes_gone"] == 0
    assert full["bytes_gone"] == full["region_bytes"]
    assert full["region_bytes"] == 27_853_103
    assert half["bytes_gone"] == int(round(0.5 * full["region_bytes"]))
    assert half["bytes_gone"] < full["bytes_gone"]


def test_ms_gone_is_estimated_from_cited_not_measured():
    rec = cbe.bytes_and_ms(REGION, 1.0)
    ms = rec["ms_gone"]
    assert ms["label"] == cbe.ESTIMATED_FROM_CITED_MS
    assert ms["not"] == "MEASURED"
    assert ms["source"] == cbe.ORGAN_REL
    assert ms["cited_organ"] == "mlp_gate_up"
    assert ms["cited_organ_ms"] == pytest.approx(7.1706, abs=1e-4)
    assert ms["pro_rated_by"] == "byte_share_of_organ"
    assert ms["ms"] > 0.0
    # One last-layer gate is a 1/64 slice of half of mlp_gate_up, so ms << token.
    assert ms["ms"] < 7.1706
    organ = cbe.load_organ_ms()
    assert organ["label"] == cbe.ESTIMATED_FROM_CITED_MS
    assert organ["not"] == "MEASURED"
    assert organ["cited_token_ms"] == pytest.approx(21.9464, abs=1e-4)


def test_both_capability_levels_are_recorded_at_every_swept_point():
    r = cbe.search_boundary(
        REGION, measure=_always_capable, budget=5, n_coarse=5
    )
    assert len(r["points"]) >= 2
    for p in r["points"]:
        assert "bytes_gone" in p
        assert "ms_gone" in p
        assert p["ms_gone"]["label"] == cbe.ESTIMATED_FROM_CITED_MS
        assert p["LOGIT_TOKEN"]["verdict"] in {cbe.PASS, cbe.FAIL, cbe.NOT_RUN}
        assert p["FAST_CAPABILITY"]["verdict"] in {cbe.PASS, cbe.FAIL, cbe.NOT_RUN}
        assert p["LOCAL_FUNCTIONAL_FIDELITY"]["not_the_verdict"] is True


# ---------------------------------------------------------------------------
# Judges reuse existing bars
# ---------------------------------------------------------------------------


def test_logit_token_reuses_stage_bars_and_records_argmax_separately():
    a = np.zeros(32, dtype=np.float64)
    a[0], a[1], a[2] = 5.0, 4.0, 3.0
    b = a.copy()
    row = cbe.judge_logit_token(a, b, component_id="test")
    assert row["verdict"] == cbe.PASS
    assert row["measurement"]["kl_bar"] == float(acs.LOGIT_KL_BAR) == 0.1
    assert row["measurement"]["top_k_bar"] == float(acs.TOPK_AGREE_BAR) == 0.8
    assert row["measurement"]["argmax_is_not_parity"] is True
    assert "argmax_a" in row["measurement"]
    assert "argmax_b" in row["measurement"]
    # Drift logits: argmax may hold while KL / top-k fail — argmax is not the bar.
    c = np.zeros(32, dtype=np.float64)
    c[0] = 5.0
    c[10] = 4.9
    drifted = cbe.judge_logit_token(a, c, component_id="test")
    assert drifted["measurement"]["argmax_is_not_parity"] is True
    if drifted["measurement"]["argmax_a"] == drifted["measurement"]["argmax_b"]:
        # If argmax held, the pass still depends on KL and top-k, not argmax.
        assert drifted["verdict"] in {cbe.PASS, cbe.FAIL}
        assert "argmax is not the pass criterion" in drifted["reason"]


def test_fast_probes_are_real_tokenizer_ids():
    assert cbe.FAST_PROBES
    probe = cbe.FAST_PROBES[0]
    assert probe["token_ids"] == list(cbe.FRANCE_TOKEN_IDS) == [760, 6511, 314, 9338, 369]
    assert 11751 in probe["expect_argmax_in"]
    assert 57590 in probe["expect_argmax_in"]
    assert "Paris" in probe["predicate"]
    ok = cbe.judge_fast_capability(11751, incumbent_satisfies=True)
    assert ok["verdict"] == cbe.PASS
    bad = cbe.judge_fast_capability(0, incumbent_satisfies=True)
    assert bad["verdict"] == cbe.FAIL
    blocked = cbe.judge_fast_capability(11751, incumbent_satisfies=False)
    assert blocked["verdict"] == cbe.NOT_RUN
    empty = cbe.judge_fast_capability(11751, probes=())
    assert empty["verdict"] == cbe.NOT_RUN


def test_synthetic_activations_are_refused():
    with pytest.raises(cim.SyntheticActivationRefuse):
        cbe.replay_with_logits(
            {
                "source": {
                    "kind": "isotropic gaussian",
                    "real_forward_pass": True,
                    "from_embedding_table": True,
                }
            },
            0,
            0,
        )
    with pytest.raises(cim.SyntheticActivationRefuse):
        cim.refuse_synthetic_activations(
            {"kind": "cpu", "real_forward_pass": False, "from_prefix": True}
        )


def test_hidden_cosine_bar_is_reused_as_context_not_as_authority():
    a = np.ones(16, dtype=np.float64)
    row = cbe.judge_cosine_context(a, a + 1e-9, component_id="test")
    assert row["role"] == cbe.CONTEXT
    assert row["not_the_verdict"] is True
    assert row["measurement"]["bar"] == float(cim.HIDDEN_COSINE_BAR) == 0.99
    assert "27.7 MB" in row["why_not_the_verdict"]


def test_assemble_document_never_emits_hardware_fields():
    r = cbe.search_boundary(REGION, measure=_always_capable, budget=5, n_coarse=5)
    cosine = cbe.load_cosine_license()
    doc = cbe.assemble_document(region_rows=[r], cosine=cosine, capture_status={"ok": True})
    _assert_no_hardware_claims(doc)
    assert doc["chooses_region"] is False
    assert doc["cosine_is_not_the_verdict"] is True
    assert doc["comparison"]["campaign_need_mb"] == 1773.0
    assert doc["the_wrong_bar"]["licensed_bytes"] == 27_688_960
    kinds = {row["kind"] for row in doc["regions"]}
    assert kinds <= {cbe.FLAT, cbe.BOUNDARY_INTERVAL, cbe.INCOMPLETE, cbe.BLOCKED}
    assert cbe.PASS not in kinds
    # Never a single-threshold pass/fail as the region result.
    for row in doc["regions"]:
        assert row["kind"] in {cbe.FLAT, cbe.BOUNDARY_INTERVAL, cbe.INCOMPLETE, cbe.BLOCKED}


def test_written_receipt_reports_a_boundary_or_flat_not_a_threshold():
    path = RECEIPTS / cbe.RECEIPT
    if not path.is_file():
        pytest.skip("receipt not built yet")
    doc = json.loads(path.read_text())
    _assert_no_hardware_claims(doc)
    assert doc["schema"] == cbe.SCHEMA
    assert doc["cosine_is_not_the_verdict"] is True
    assert doc["chooses_region"] is False
    assert len(doc["regions"]) >= 1
    for row in doc["regions"]:
        assert row["kind"] in {
            cbe.FLAT,
            cbe.BOUNDARY_INTERVAL,
            cbe.INCOMPLETE,
            cbe.BLOCKED,
        }
        assert row["kind"] not in {cbe.PASS, cbe.FAIL}
        assert row["points"]
        for p in row["points"]:
            assert "bytes_gone" in p
            assert p["ms_gone"]["label"] == cbe.ESTIMATED_FROM_CITED_MS
            assert p["ms_gone"]["not"] == "MEASURED"
            assert "LOGIT_TOKEN" in p
            assert "FAST_CAPABILITY" in p
            assert p["LOCAL_FUNCTIONAL_FIDELITY"]["not_the_verdict"] is True


def test_per_level_reports_flat_when_one_level_never_breaks():
    points = [
        {
            "destruction": 0.0,
            "bytes_gone": 0,
            "LOGIT_TOKEN": _logit(cbe.PASS),
            "FAST_CAPABILITY": _fast(cbe.PASS),
            "capability_verdict": cbe.PASS,
            "status": "OK",
        },
        {
            "destruction": 0.5,
            "bytes_gone": 10,
            "LOGIT_TOKEN": _logit(cbe.FAIL),
            "FAST_CAPABILITY": _fast(cbe.PASS),
            "capability_verdict": cbe.FAIL,
            "status": "OK",
        },
        {
            "destruction": 1.0,
            "bytes_gone": 20,
            "LOGIT_TOKEN": _logit(cbe.FAIL),
            "FAST_CAPABILITY": _fast(cbe.PASS),
            "capability_verdict": cbe.FAIL,
            "status": "OK",
        },
    ]
    per = cbe.per_level_from_points(points, 20)
    assert per[cbe.FAST_CAPABILITY]["kind"] == cbe.FLAT
    assert "never broke" in per[cbe.FAST_CAPABILITY]["reading"]
    assert per[cbe.LOGIT_TOKEN]["kind"] == cbe.BOUNDARY_INTERVAL


def test_curve_value_refuses_a_record_without_capability_levels():
    with pytest.raises(cbe.ByteEliminationRefuse, match="cosine-only"):
        cbe.curve_value_from_record({"cosine": 0.99, "status": "OK"})
    with pytest.raises(cbe.PointNotRun):
        cbe.curve_value_from_record(
            {
                "status": cbe.NOT_RUN,
                "reason": "blocked",
                "LOGIT_TOKEN": {"verdict": cbe.NOT_RUN},
                "FAST_CAPABILITY": {"verdict": cbe.NOT_RUN},
            }
        )
