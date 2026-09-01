"""Matched aux-merge A/B: refuse a rate on unmatched bytes or a bimodal arm."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_aux_merge_ab as ab


WEIGHT = 83_558_400


def _arm(
    ident: str,
    gb_s: float,
    *,
    streams: list[int] | None = None,
    weight_bytes: int = WEIGHT,
    spread: float = 1.02,
    n_reps: int = 11,
) -> dict:
    streams = list(streams if streams is not None else (
        ab.INCUMBENT_STREAMS if ident == "incumbent" else ab.MERGED_STREAMS
    ))
    gpu_ns = int(round(weight_bytes / gb_s)) if gb_s > 0 else 0
    lo = int(gpu_ns / (spread ** 0.5)) if spread > 0 else gpu_ns
    hi = int(gpu_ns * (spread ** 0.5)) if spread > 0 else gpu_ns
    reps = [gpu_ns] * n_reps
    if n_reps >= 2:
        reps[0] = lo
        reps[-1] = hi
    return {
        "id": ident,
        "kernel": ab.INCUMBENT_KERNEL if ident == "incumbent" else ab.MERGED_KERNEL,
        "stream_count": len(streams),
        "bytes_per_stream": streams,
        "bytes_per_thread_iteration": sum(streams),
        "unique_payload_bytes": weight_bytes,
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_min": min(reps) if reps else 0,
        "gpu_ns_max": max(reps) if reps else 0,
        "gpu_ns_reps": reps,
        "gpu_us_median": gpu_ns / 1e3,
        "rep_spread": spread,
        "steady_state": spread <= ab.STEADY_MAX_SPREAD,
        "effective_gb_s": gb_s,
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "threads_per_threadgroup": 128,
    }


def _compare(*, n: int = 39936, exact: int | None = None, max_abs: float = 0.0, rel: float = 0.0) -> dict:
    if exact is None:
        exact = n
    return {
        "n_compared": n,
        "n_bit_exact": exact,
        "max_abs_err": max_abs,
        "rel_fro": rel,
        "bit_identical": exact == n,
        "per_projection": [],
    }


def _raw(
    *,
    inc: float = 330.0,
    mer: float = 564.0,
    ab_inc: float | None = None,
    ab_mer: float | None = None,
    ba_inc: float | None = None,
    ba_mer: float | None = None,
    spread: float = 1.02,
    warmup: int = 60,
    reps: int = 11,
    inc_streams: list[int] | None = None,
    mer_streams: list[int] | None = None,
    mer_weight: int | None = None,
    compare: dict | None = None,
) -> dict:
    ab_inc = inc if ab_inc is None else ab_inc
    ab_mer = mer if ab_mer is None else ab_mer
    ba_inc = inc if ba_inc is None else ba_inc
    ba_mer = mer if ba_mer is None else ba_mer
    inc_s = inc_streams or list(ab.INCUMBENT_STREAMS)
    mer_s = mer_streams or list(ab.MERGED_STREAMS)
    mw = WEIGHT if mer_weight is None else mer_weight
    def order(a: float, b: float, seq: list[str]) -> dict:
        ia = _arm("incumbent", a, streams=inc_s, spread=spread)
        mb = _arm("merged", b, streams=mer_s, weight_bytes=mw, spread=spread)
        ratio = (b / a) if a else None
        return {
            "sequence": seq,
            "incumbent": ia,
            "merged": mb,
            "ratio_merged_over_incumbent": ratio,
        }
    return {
        "schema": "hawking.future.mlp_aux_merge_ab.raw.v1",
        "layer": 0,
        "warmup": warmup,
        "reps": reps,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "bytes_per_thread_iteration_held": 38,
        "aux_repack_on_disk": False,
        "does_not_promote": True,
        "production_shader_untouched": True,
        "concurrent_load": {"loadavg": "{ 4.10 4.20 4.30 }"},
        "concurrent_load_end": {"loadavg": "{ 4.40 4.25 4.31 }"},
        "weight_bytes": WEIGHT,
        "unique_payload_bytes": WEIGHT,
        "merged_aux_bytes": 16_711_680,
        "scale_bias_bytes": 16_711_680,
        "dispatches": 3,
        "output_compare": compare or _compare(),
        "order_ab": order(ab_inc, ab_mer, ["incumbent", "merged"]),
        "order_ba": order(ba_inc, ba_mer, ["merged", "incumbent"]),
        "pooled": order(inc, mer, ["pooled"]),
    }


def test_refuses_an_arm_whose_rep_spread_exceeds_1_10():
    raw = _raw(spread=1.20)
    with pytest.raises(ab.SpreadRefused, match="rep_spread 1.2000"):
        ab.measurement_from_raw(raw)


def test_refuses_a_rate_claim_when_byte_counts_differ():
    raw = _raw(mer_weight=WEIGHT * 2)
    with pytest.raises(ab.ByteMismatch, match="unique payload bytes differ"):
        ab.measurement_from_raw(raw)

    raw = _raw(mer_streams=[4, 4, 32])
    with pytest.raises(ab.ByteMismatch):
        ab.measurement_from_raw(raw)


def test_refuses_bytes_per_iteration_not_38():
    raw = _raw()
    raw["pooled"]["incumbent"]["bytes_per_thread_iteration"] = 36
    raw["pooled"]["incumbent"]["bytes_per_stream"] = [2, 2, 0, 32]
    with pytest.raises(ab.ByteMismatch, match="38"):
        ab.measurement_from_raw(raw)


def test_refuses_warmup_below_60():
    raw = _raw(warmup=5)
    with pytest.raises(ab.AbRefused, match="coin flip between two modes"):
        ab.measurement_from_raw(raw)


def test_refuses_empty_gpu_ns():
    raw = _raw()
    raw["pooled"]["incumbent"]["gpu_ns_median"] = 0
    with pytest.raises(ab.EmptyGpuSample):
        ab.measurement_from_raw(raw)


def test_gb_s_is_bytes_over_gpu_ns():
    assert ab.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(ab.EmptyGpuSample):
        ab.effective_gb_s(100, 0)


def test_probe_survives_when_ratio_tracks_1_708():
    # 330 * 1.708 ≈ 563.6
    doc = ab.build(ab.measurement_from_raw(_raw(inc=330.0, mer=564.0)))
    assert doc["verdict"] == ab.VERDICT_SURVIVES
    assert doc["does_not_promote"] is True
    assert abs(doc["judgement"]["ratio_merged_over_incumbent"] - ab.PROBE_RATIO) < 0.02


def test_no_lift_when_arms_measure_the_same():
    doc = ab.build(ab.measurement_from_raw(_raw(inc=330.0, mer=332.0)))
    assert doc["verdict"] == ab.VERDICT_NONE
    assert "does not survive" in doc["judgement"]["why"]


def test_hurts_when_merged_is_slower():
    doc = ab.build(ab.measurement_from_raw(_raw(inc=330.0, mer=250.0)))
    assert doc["verdict"] == ab.VERDICT_HURTS


def test_smaller_lift_is_named_as_the_probe_measuring_a_sink():
    doc = ab.build(ab.measurement_from_raw(_raw(inc=330.0, mer=400.0)))
    assert doc["verdict"] == ab.VERDICT_SMALLER
    assert "XOR/add sink" in doc["judgement"]["why"]


def test_orderings_that_disagree_are_not_a_rate_claim():
    raw = _raw(ab_inc=330.0, ab_mer=564.0, ba_inc=330.0, ba_mer=335.0, inc=400.0, mer=450.0)
    doc = ab.build(ab.measurement_from_raw(raw))
    assert doc["judgement"]["orderings_agree"] is False
    assert doc["verdict"] == ab.VERDICT_NONE
    assert "disagree" in doc["judgement"]["why"]


def test_output_compare_is_required_and_carried():
    raw = _raw()
    del raw["output_compare"]
    with pytest.raises(ab.AbRefused, match="output_compare"):
        ab.measurement_from_raw(raw)
    raw = _raw(compare=_compare(n=39936, exact=39900, max_abs=1e-6, rel=1e-7))
    measured = ab.measurement_from_raw(raw)
    oc = measured["output_compare"]
    assert oc["n_compared"] == 39936
    assert oc["n_bit_exact"] == 39900
    assert oc["bit_identical"] is False
    assert oc["max_abs_err"] == 1e-6
    assert oc["rel_fro"] == 1e-7


def test_build_writes_receipt_that_parses(tmp_path: Path):
    dest = tmp_path / "MLP_AUX_MERGE_AB.json"
    measured = ab.measurement_from_raw(_raw())
    path = ab.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == ab.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in ab.VERDICTS
    assert doc["does_not_promote"] is True
    assert doc["production_shader_untouched"] is True
    assert doc["aux_repack_on_disk"] is False
    for arm_name in ("incumbent", "merged"):
        arm = doc[arm_name]
        assert arm["effective_gb_s"] > 0
        assert arm["gpu_us_median"] > 0
        assert arm["bytes_per_thread_iteration"] == 38
        assert "rep_spread" in arm
    oc = doc["output_compare"]
    assert {"n_compared", "n_bit_exact", "max_abs_err", "rel_fro"} <= set(oc)
    assert doc["loadavg_open"]
    assert doc["loadavg_close"]
    assert "measurement_provenance" in doc
    with pytest.raises(ab.MissingArm):
        ab.record(None)


def test_cli_build_writes_the_canonical_receipt_from_raw(tmp_path: Path, monkeypatch):
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "MLP_AUX_MERGE_AB.json"
    raw_path.write_text(json.dumps(_raw()))
    rc = ab.main(["--from", str(raw_path), "--build", "--out", str(out_path)])
    assert rc == 0
    doc = json.loads(out_path.read_text())
    assert doc["schema"] == ab.SCHEMA
    assert doc["verdict"] in ab.VERDICTS


def test_committed_receipt_if_present_carries_the_required_fields():
    if not ab.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(ab.RECEIPT.read_text())
    assert doc["schema"] == ab.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in ab.VERDICTS
    assert doc["does_not_promote"] is True
    if doc["verdict"] == ab.VERDICT_BLOCKED:
        assert doc.get("blocked_error"), "BLOCKED must carry the exact error"
        blob = json.dumps(doc)
        assert '"gpu_ns"' not in blob, "a BLOCKED receipt must not invent gpu_ns"
        assert "gpu_ns_median" not in doc
        return
    for arm_name in ("incumbent", "merged"):
        arm = doc[arm_name]
        assert arm["bytes_per_thread_iteration"] == 38
        assert arm["effective_gb_s"] > 0
        assert arm["gpu_us_median"] > 0
        assert "rep_spread" in arm
        assert arm["rep_spread"] <= ab.STEADY_MAX_SPREAD
    assert doc["incumbent"]["unique_payload_bytes"] == doc["merged"]["unique_payload_bytes"]
    oc = doc["output_compare"]
    assert oc["n_compared"] > 0
    assert "n_bit_exact" in oc and "max_abs_err" in oc and "rel_fro" in oc
    assert doc.get("loadavg_open") or (doc.get("concurrent_load") or {}).get("loadavg")
    assert doc.get("loadavg_close") or (doc.get("concurrent_load_end") or {}).get("loadavg")


def test_blocked_receipt_does_not_invent_gpu_ns(tmp_path: Path):
    dest = tmp_path / "MLP_AUX_MERGE_AB.json"
    path = ab.record_blocked(
        "mlp_aux_merge_ab: no Metal-capable GPU",
        {
            "warmup": 60,
            "reps": 11,
            "concurrent_load": {"loadavg": "{ 10.95 10.32 10.17 }"},
            "concurrent_load_end": {"loadavg": "{ 10.95 10.32 10.17 }"},
        },
        path=dest,
    )
    doc = json.loads(path.read_text())
    assert doc["verdict"] == ab.VERDICT_BLOCKED
    assert "no Metal-capable GPU" in doc["blocked_error"]
    assert "gpu_ns_median" not in doc
    assert doc["does_not_promote"] is True
    assert doc["loadavg_open"]


def test_cli_build_from_blocked_raw_does_not_invent_numbers(tmp_path: Path):
    raw_path = tmp_path / "raw.json"
    out_path = tmp_path / "MLP_AUX_MERGE_AB.json"
    raw_path.write_text(json.dumps({
        "schema": "hawking.future.mlp_aux_merge_ab.raw.v1",
        "blocked": True,
        "blocked_error": "mlp_aux_merge_ab: no Metal-capable GPU",
        "warmup": 60,
        "reps": 11,
        "concurrent_load": {"loadavg": "{ 10.95 10.32 10.17 }"},
        "concurrent_load_end": {"loadavg": "{ 10.95 10.32 10.17 }"},
    }))
    rc = ab.main(["--from", str(raw_path), "--build", "--out", str(out_path)])
    assert rc == 0
    doc = json.loads(out_path.read_text())
    assert doc["verdict"] == ab.VERDICT_BLOCKED
    assert "gpu_ns_median" not in doc


def test_record_refuses_none():
    with pytest.raises(ab.MissingArm):
        ab.record(None)
