"""Issue-rate ladder: no verdict on a non-monotone curve; no bit-identity without a byte comparison."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_issue_rate_ladder as mil


WEIGHT_BYTES = 83_558_400


def _compare(
    *,
    n_bytes: int = 160_000,
    mismatch_bytes: int = 0,
    mismatch_floats: int = 0,
    cosine: float = 1.0,
    rel_fro: float = 0.0,
    max_abs: float = 0.0,
) -> dict:
    return {
        "n_bytes_compared": n_bytes,
        "n_floats_compared": n_bytes // 4,
        "n_mismatch_bytes": mismatch_bytes,
        "n_float_mismatch": mismatch_floats,
        "max_abs_err": max_abs,
        "cosine": cosine,
        "rel_fro": rel_fro,
        "compared_against": "production kernel output buffers after the timed command buffer",
    }


def _occ(*, tg: int = 128, max_threads: int = 1024) -> dict:
    return {
        "threads_per_threadgroup": tg,
        "max_total_threads_per_threadgroup": max_threads,
        "thread_execution_width": 32,
        "occupancy_of_max_threads": tg / max_threads,
        "threadgroups": 8704,
        "gpu_cores": 60,
        "threadgroups_per_core": 145.06666666666666,
        "registers_per_thread": None,
    }


def _arm(
    ident: str,
    *,
    gb_s: float,
    family: str = "ladder",
    n_accumulators: int = 1,
    n_live_floats: int = 1,
    tg: int = 128,
    compare: dict | None = None,
    bit_identical: bool | None = None,
    kernel: str | None = None,
) -> dict:
    gpu_ns = int(round(WEIGHT_BYTES / gb_s))
    out = {
        "id": ident,
        "kernel": kernel or f"issue_ladder_{ident}",
        "family": family,
        "note": "fixture",
        "n_accumulators": n_accumulators,
        "n_live_floats": n_live_floats,
        "weight_bytes": WEIGHT_BYTES,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "gpu_us_median": gpu_ns / 1e3,
        "effective_gb_s": gb_s,
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "threads_per_threadgroup": tg,
        "occupancy": _occ(tg=tg),
        "byte_compare": compare if compare is not None else _compare(),
    }
    if bit_identical is not None:
        out["bit_identical"] = bit_identical
    return out


def _linear_gb(ident: str) -> float:
    """Constant issue-rate GB/s = 3292 / total_ops_per_byte (tax table)."""
    ops = mil.INNER_LOOP_TAX[ident]["total_ops_per_byte"]
    return 3292.0 / ops


def _raw(
    *,
    ladder_gb: dict[str, float] | None = None,
    ilp_gb: dict[str, float] | None = None,
    ws_gb: dict[str, float] | None = None,
    tg_gb: dict[int, float] | None = None,
    ladder_compare: dict[str, dict] | None = None,
) -> dict:
    if ladder_gb is None:
        ladder_gb = {i: _linear_gb(i) for i in mil.LADDER_IDS}
    if ilp_gb is None:
        ilp_gb = {i: ladder_gb["production"] for i in mil.ILP_IDS}
    if ws_gb is None:
        ws_gb = {i: ladder_gb["production"] for i in mil.WS_IDS}
    if tg_gb is None:
        tg_gb = {tg: ladder_gb["production"] for tg in (64, 128, 256, 512)}
    ladder = []
    for ident in mil.LADDER_IDS:
        cmp_ = (ladder_compare or {}).get(ident)
        ladder.append(
            _arm(
                ident,
                gb_s=ladder_gb[ident],
                family="ladder",
                kernel=(
                    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"
                    if ident == "production"
                    else f"issue_ladder_{ident}"
                ),
                compare=cmp_ if cmp_ is not None else _compare(),
            )
        )
    ilp = [
        _arm(
            ident,
            gb_s=ilp_gb[ident],
            family="ilp",
            n_accumulators={"ilp2": 2, "ilp4": 4, "ilp8": 8}[ident],
            n_live_floats={"ilp2": 2, "ilp4": 4, "ilp8": 8}[ident],
        )
        for ident in mil.ILP_IDS
    ]
    ws = [
        _arm(
            ident,
            gb_s=ws_gb[ident],
            family="register_pressure",
            n_live_floats={"ws8": 8, "ws16": 16, "ws32": 32}[ident],
        )
        for ident in mil.WS_IDS
    ]
    tg = [
        _arm(
            f"tg{n}",
            gb_s=tg_gb[n],
            family="threadgroup",
            tg=n,
            kernel="issue_ladder_tg",
        )
        for n in sorted(tg_gb)
    ]
    return {
        "schema": "hawking.future.mlp_issue_rate_ladder.raw.v1",
        "layer": 0,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 1.0 1.0 1.0 }"},
        "concurrent_load_end": {"loadavg": "{ 1.1 1.0 1.0 }"},
        "organ": "mlp",
        "codec": "HGRAVF01 affine2 q2 group64",
        "geometry": "geo_tpr64_tg128",
        "production_kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "weight_bytes": WEIGHT_BYTES,
        "ladder": ladder,
        "ilp": ilp,
        "register_pressure": ws,
        "threadgroup": tg,
    }


def test_tax_table_is_monotone_and_has_four_plus_rungs():
    rungs = [mil.INNER_LOOP_TAX[i] for i in mil.LADDER_IDS]
    assert len(rungs) >= 4
    ops = [r["total_ops_per_byte"] for r in rungs]
    assert ops == sorted(ops, reverse=True)
    assert len(set(ops)) == len(ops)
    for r in rungs:
        by = r["ops_per_byte_by_class"]
        assert set(by) == {"fma", "integer", "conversion", "memory", "control"}
        assert abs(sum(by.values()) - r["total_ops_per_byte"]) < 5e-4


def test_refuses_bit_identical_without_byte_comparison():
    v = _arm("k6", gb_s=360.0)
    del v["byte_compare"]
    v["bit_identical"] = True
    with pytest.raises((mil.NoByteComparison, mil.BitIdentityClaimWithoutCompare)) as caught:
        mil.bit_identical_from_compare(v)
    msg = str(caught.value).lower()
    assert "bit-identical" in msg
    assert "byte" in msg


def test_refuses_bit_identical_when_n_bytes_compared_is_zero():
    v = _arm("k6", gb_s=360.0, compare=_compare(n_bytes=0))
    v["bit_identical"] = True
    with pytest.raises(mil.NoByteComparison) as caught:
        mil.bit_identical_from_compare(v)
    assert "n_bytes_compared" in str(caught.value)


def test_refuses_to_build_a_claimed_identical_mismatch():
    raw = _raw()
    for v in raw["ladder"]:
        if v["id"] == "k6":
            v["bit_identical"] = True
            v["byte_compare"] = _compare(mismatch_bytes=12, mismatch_floats=3, cosine=0.9)
    with pytest.raises(mil.BitIdentityClaimWithoutCompare):
        mil.measurement_from_raw(raw)


def test_derive_identical_only_from_zero_mismatch():
    ok = _arm("production", gb_s=329.2, compare=_compare())
    assert mil.bit_identical_from_compare(ok) is True
    bad = _arm(
        "k6",
        gb_s=360.0,
        compare=_compare(mismatch_bytes=4, mismatch_floats=1, cosine=0.999),
    )
    assert mil.bit_identical_from_compare(bad) is False


def test_raises_when_ladder_is_not_monotone():
    measured = mil.measurement_from_raw(_raw())
    # Swap ops so k4 is not below k6.
    measured["ladder"][2]["total_ops_per_byte"] = measured["ladder"][1]["total_ops_per_byte"]
    with pytest.raises(mil.LadderNotMonotone) as caught:
        mil.judge(measured)
    assert "not monotone" in str(caught.value)
    with pytest.raises(mil.LadderNotMonotone):
        mil.build(measured)
    # Direct helper.
    fake = [
        {"id": "a", "total_ops_per_byte": 1.0},
        {"id": "b", "total_ops_per_byte": 2.0},
        {"id": "c", "total_ops_per_byte": 0.5},
        {"id": "d", "total_ops_per_byte": 0.1},
    ]
    with pytest.raises(mil.LadderNotMonotone):
        mil.require_monotone(fake)


def test_raises_when_a_rung_is_missing():
    raw = _raw()
    raw["ladder"] = [v for v in raw["ladder"] if v["id"] != "k4"]
    with pytest.raises(mil.MissingRung) as caught:
        mil.measurement_from_raw(raw)
    assert "k4" in str(caught.value)

    raw = _raw()
    raw["ilp"] = [v for v in raw["ilp"] if v["id"] != "ilp8"]
    with pytest.raises(mil.MissingRung) as caught:
        mil.measurement_from_raw(raw)
    assert "ilp8" in str(caught.value)

    raw = _raw()
    raw["register_pressure"] = [v for v in raw["register_pressure"] if v["id"] != "ws32"]
    with pytest.raises(mil.MissingRung):
        mil.measurement_from_raw(raw)


def test_zero_gpu_ns_refuses():
    raw = _raw()
    raw["ladder"][0]["gpu_ns_median"] = 0
    with pytest.raises(mil.EmptyGpuSample):
        mil.measurement_from_raw(raw)


def test_gb_s_is_bytes_over_gpu_ns():
    assert mil.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(mil.EmptyGpuSample):
        mil.effective_gb_s(100, 0)


def test_issue_rate_bound_when_time_tracks_ops():
    doc = mil.build(mil.measurement_from_raw(_raw()))
    assert doc["verdict"] == mil.VERDICT_ISSUE
    assert doc["judgement"]["shape"]["shape"] == mil.SHAPE_LINEAR
    assert doc["judgement"]["shape"]["linear"] is True
    assert "cheapest remaining" in doc["finding"].lower() or "Cheapest remaining" in doc["finding"]
    cheap = doc["judgement"]["cheapest_remaining"]
    assert cheap["rung"] == "arm_a"
    assert cheap["dominant_class"] in ("fma", "integer", "conversion", "memory", "control")


def test_dependency_bound_when_plateau_and_ilp_jumps():
    prod = 370.0
    ladder_gb = {i: prod for i in mil.LADDER_IDS}
    ilp_gb = {"ilp2": 390.0, "ilp4": 420.0, "ilp8": 444.0}  # 444/370 = 1.20 >= 1.12
    ws_gb = {i: prod for i in mil.WS_IDS}
    doc = mil.build(
        mil.measurement_from_raw(_raw(ladder_gb=ladder_gb, ilp_gb=ilp_gb, ws_gb=ws_gb))
    )
    assert doc["judgement"]["shape"]["plateau"] is True
    assert doc["judgement"]["ilp"]["jumped"] is True
    assert doc["judgement"]["register_pressure"]["pressed"] is False
    assert doc["verdict"] == mil.VERDICT_DEP


def test_register_pressure_bound_when_plateau_and_ws_drops_at_constant_occupancy():
    prod = 370.0
    ladder_gb = {i: prod for i in mil.LADDER_IDS}
    ilp_gb = {i: prod for i in mil.ILP_IDS}
    ws_gb = {"ws8": 360.0, "ws16": 340.0, "ws32": 296.0}  # 296/370 = 0.80 <= 0.88
    doc = mil.build(
        mil.measurement_from_raw(_raw(ladder_gb=ladder_gb, ilp_gb=ilp_gb, ws_gb=ws_gb))
    )
    assert doc["judgement"]["shape"]["plateau"] is True
    assert doc["judgement"]["ilp"]["jumped"] is False
    assert doc["judgement"]["register_pressure"]["pressed"] is True
    assert doc["judgement"]["register_pressure"]["constant_occupancy"] is True
    assert doc["verdict"] == mil.VERDICT_REG


def test_mixed_when_neither_shape():
    # GB/s moves, but not with ops: production slow, middle rungs fast, ARM A back down.
    ladder_gb = {
        "production": 329.2,
        "k6": 480.0,
        "k4": 490.0,
        "k2": 400.0,
        "arm_a": 350.0,
    }
    doc = mil.build(mil.measurement_from_raw(_raw(ladder_gb=ladder_gb)))
    assert doc["judgement"]["shape"]["shape"] == mil.SHAPE_NEITHER
    assert doc["verdict"] == mil.VERDICT_MIXED
    assert "neither" in doc["judgement"]["why"].lower() or "MIXED" in doc["finding"]


def test_mixed_when_plateau_and_both_discriminators_fire():
    prod = 370.0
    ladder_gb = {i: prod for i in mil.LADDER_IDS}
    ilp_gb = {"ilp2": 400.0, "ilp4": 430.0, "ilp8": 460.0}
    ws_gb = {"ws8": 350.0, "ws16": 330.0, "ws32": 296.0}
    doc = mil.build(
        mil.measurement_from_raw(_raw(ladder_gb=ladder_gb, ilp_gb=ilp_gb, ws_gb=ws_gb))
    )
    assert doc["judgement"]["shape"]["plateau"] is True
    assert doc["judgement"]["ilp"]["jumped"] is True
    assert doc["judgement"]["register_pressure"]["pressed"] is True
    assert doc["verdict"] == mil.VERDICT_MIXED


def test_ws_drop_without_constant_occupancy_is_not_c2():
    prod = 370.0
    ladder_gb = {i: prod for i in mil.LADDER_IDS}
    raw = _raw(
        ladder_gb=ladder_gb,
        ilp_gb={i: prod for i in mil.ILP_IDS},
        ws_gb={"ws8": 360.0, "ws16": 340.0, "ws32": 296.0},
    )
    for v in raw["register_pressure"]:
        if v["id"] == "ws32":
            v["occupancy"] = _occ(tg=128, max_threads=256)  # occupancy 0.5 vs 0.125
    doc = mil.build(mil.measurement_from_raw(raw))
    assert doc["judgement"]["register_pressure"]["dropped"] is True
    assert doc["judgement"]["register_pressure"]["constant_occupancy"] is False
    assert doc["judgement"]["register_pressure"]["pressed"] is False
    assert doc["verdict"] == mil.VERDICT_MIXED


def test_record_writes_classes_and_refuses_none(tmp_path: Path):
    dest = tmp_path / "MLP_ISSUE_RATE_LADDER.json"
    measured = mil.measurement_from_raw(_raw())
    path = mil.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == mil.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert doc["took_gpu_lease"] is True
    assert "loadavg" in doc["concurrent_load"]
    assert doc["verdict"] in (
        mil.VERDICT_ISSUE,
        mil.VERDICT_DEP,
        mil.VERDICT_REG,
        mil.VERDICT_MIXED,
    )
    ids = [v["id"] for v in doc["ladder"]]
    assert ids == list(mil.LADDER_IDS)
    ops = [v["total_ops_per_byte"] for v in doc["ladder"]]
    assert ops == sorted(ops, reverse=True)
    for v in doc["ladder"]:
        assert v["effective_gb_s"] > 0
        assert set(v["ops_per_byte_by_class"]) == {
            "fma",
            "integer",
            "conversion",
            "memory",
            "control",
        }
        assert v["bit_identical"] == mil.bit_identical_from_compare(v)
        assert int(v["byte_compare"]["n_bytes_compared"]) > 0
    assert len(doc["ilp"]) == 3
    assert len(doc["register_pressure"]) == 3
    assert doc["pre_registered"]["locked_before_measurement"] is True
    with pytest.raises(mil.MissingRung):
        mil.record(None)


def test_committed_receipt_if_present_is_a_real_verdict():
    if not mil.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(mil.RECEIPT.read_text())
    assert doc["schema"] == mil.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert "loadavg" in (doc.get("concurrent_load") or {})
    assert doc["verdict"] in (
        mil.VERDICT_ISSUE,
        mil.VERDICT_DEP,
        mil.VERDICT_REG,
        mil.VERDICT_MIXED,
    )
    ids = [v["id"] for v in doc["ladder"]]
    assert ids == list(mil.LADDER_IDS)
    ops = [v["total_ops_per_byte"] for v in doc["ladder"]]
    assert len(ops) >= 4
    assert ops == sorted(ops, reverse=True)
    assert len(set(ops)) == len(ops)
    for v in doc["ladder"]:
        assert v["effective_gb_s"] > 0
        assert set(v["ops_per_byte_by_class"]) == {
            "fma",
            "integer",
            "conversion",
            "memory",
            "control",
        }
        assert int(v["byte_compare"]["n_bytes_compared"]) > 0
        assert v["bit_identical"] == mil.bit_identical_from_compare(v)
    assert {v["id"] for v in doc["ilp"]} >= set(mil.ILP_IDS)
    assert {v["id"] for v in doc["register_pressure"]} >= set(mil.WS_IDS)
    rebuilt = mil.build(
        mil.measurement_from_raw(
            {
                "layer": doc["layer"],
                "warmup": doc["warmup"],
                "reps": doc["reps"],
                "git_head": doc["git_head"],
                "artifact_root": doc["artifact_root"],
                "timing": doc["timing"],
                "concurrent_load": doc["concurrent_load"],
                "concurrent_load_end": doc.get("concurrent_load_end") or {},
                "organ": doc["organ"],
                "codec": doc["codec"],
                "geometry": doc["geometry"],
                "production_kernel": doc["production_kernel"],
                "projections": doc["projections"],
                "weight_bytes": doc["weight_bytes"],
                "ladder": doc["ladder"],
                "ilp": doc["ilp"],
                "register_pressure": doc["register_pressure"],
                "threadgroup": doc.get("threadgroup") or [],
            }
        )
    )
    assert rebuilt["verdict"] == doc["verdict"]
    assert rebuilt["production_gb_s"] == doc["production_gb_s"]
    for a, b in zip(rebuilt["ladder"], doc["ladder"]):
        assert a["id"] == b["id"]
        assert a["bit_identical"] == b["bit_identical"]
        assert a["effective_gb_s"] == b["effective_gb_s"]
        assert a["total_ops_per_byte"] == b["total_ops_per_byte"]
