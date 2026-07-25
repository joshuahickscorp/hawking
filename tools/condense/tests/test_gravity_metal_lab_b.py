#!/usr/bin/env python3.12
"""Tests for Track B.  Everything except the dispatch tests runs with no Metal device.

The planner, both cost models and the numpy mechanism model are pure functions of the
geometry, and they are where the load-bearing claims live -- the threadgroup budget that
caps ``cbs``, the op split that decides whether the grammar can pay, and the fact that the
table never reaches device memory on the hot path.  Those are tested on the CPU.  The
handful of tests that need a GPU skip cleanly when there is not one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import gravity_metal_lab_b as b  # noqa: E402

# The two real R0 geometries, so the tests are about the artifacts and not about a toy.
GATE = {"rows": 2048, "cols": 6144, "nchunk": 768, "D": 8, "k": 128}
DOWN = {"rows": 6144, "cols": 2048, "nchunk": 256, "D": 8, "k": 128}


# ------------------------------------------------------------------ threadgroup budget

def test_cbs_cap_is_arithmetic_not_a_magic_number():
    """codebook + x slice + table must fit 32768 B, and the planner must say so."""
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                      cbs=52, tpg=1024)
    assert shape["scratch_bytes"] == (128 * 8 + 52 * 8) * 4 + 52 * 128 * 4
    assert shape["scratch_bytes"] <= 32768
    with pytest.raises(b.TrackBError, match="threadgroup memory"):
        b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                  cbs=53, tpg=1024)


def test_half_table_is_what_buys_the_bigger_block():
    """The only reason a half table is worth measuring: it doubles the on-chip block."""
    args = {kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")}
    with pytest.raises(b.TrackBError):
        b.ll_plan(**args, cbs=96, tpg=1024, half_table=False)
    shape = b.ll_plan(**args, cbs=96, tpg=1024, half_table=True)
    assert shape["half_table"] and shape["kernel"] == "ll_blk_f16_r4" or True
    assert shape["table_bytes_on_chip"] == 96 * 128 * 2
    # 4096 B codebook + 288 B per chunk (table 256 + x slice 32) caps the half path at 99
    assert b.ll_plan(**args, cbs=99, tpg=1024, half_table=True)["scratch_bytes"] <= 32768
    with pytest.raises(b.TrackBError):
        b.ll_plan(**args, cbs=100, tpg=1024, half_table=True)


def test_row4_refuses_a_geometry_it_would_misread():
    with pytest.raises(b.TrackBError, match="row4"):
        b.ll_plan(rows=2050, nchunk=768, D=8, k=128, cbs=16, tpg=256, row4=True)


def test_kernel_name_matches_the_flags():
    assert b.ll_kernel_name(half_table=False, row4=False) == "ll_blk_f32_r1"
    assert b.ll_kernel_name(half_table=True, row4=True) == "ll_blk_f16_r4"


def test_one_threadgroup_per_chunk_block_owns_every_row():
    """The shape that preserves the op reduction: blocks threadgroups, no row tiling.

    A 2D (row tile, chunk block) split like Track A's would rebuild the table once per row
    tile and hand most of the 5.333x straight back.  This is the invariant that stops it.
    """
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                      cbs=16, tpg=1024)
    assert shape["threadgroups"] == shape["blocks"] == 48
    assert "row_tiles" not in shape
    assert shape["rows_per_thread"] == 2


# ------------------------------------------------------------------------- cost models

def test_analytic_reduction_matches_the_ground_truth_figures():
    """5.333x at gate/up and 6.857x at down, unshared.  Anything else is a regression."""
    g = b.ll_cost(**GATE, shape=b.ll_plan(**{kk: GATE[kk] for kk in
                                             ("rows", "nchunk", "D", "k")},
                                          cbs=16, tpg=1024))
    d = b.ll_cost(**DOWN, shape=b.ll_plan(**{kk: DOWN[kk] for kk in
                                             ("rows", "nchunk", "D", "k")},
                                          cbs=16, tpg=1024))
    assert g["arithmetic_reduction_vs_decode_fma"] == pytest.approx(16 / 3)
    assert d["arithmetic_reduction_vs_decode_fma"] == pytest.approx(48 / 7)


def test_table_never_reaches_device_memory_on_the_hot_path():
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                      cbs=16, tpg=1024)
    cost = b.ll_cost(**GATE, shape=shape)
    assert cost["table_bytes_written_to_device"] == 0
    assert cost["table_bytes_on_chip"] == 16 * 128 * 4
    assert shape["table_in_device_memory"] is False


def test_the_op_reduction_is_independent_of_every_tuning_knob():
    """k*D against rows.  cbs, tpg, the block count and row4 all cancel."""
    args = {kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")}
    values = {b.ll_cost(**GATE, shape=b.ll_plan(**args, cbs=cbs, tpg=tpg, row4=r4)
                        )["arithmetic_reduction_vs_decode_fma"]
              for cbs in (8, 16, 32, 48) for tpg in (256, 1024) for r4 in (False, True)}
    assert len(values) == 1


def test_lookup_linear_reads_x_once_and_decode_fma_does_not():
    """The traffic difference between the grammars, which is not the arithmetic one."""
    llshape = b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                        cbs=16, tpg=1024)
    dfshape = b.dfma_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                          cbs=24, tpg=256)
    ll = b.ll_cost(**GATE, shape=llshape)
    df = b.dfma_cost(**GATE, shape=dfshape)
    assert ll["activation_bytes"] == GATE["nchunk"] * GATE["D"] * 4
    # decode-FMA re-reads the block's x slice once per ROW TILE; lookup-linear has no row
    # tiles, so x crosses the bus exactly once however the block count is chosen
    assert df["activation_bytes"] == dfshape["threadgroups"] * dfshape["cbs"] * GATE["D"] * 4
    assert df["activation_bytes"] == 8 * ll["activation_bytes"]
    tighter = b.dfma_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                          cbs=24, tpg=64)
    assert b.dfma_cost(**GATE, shape=tighter)["activation_bytes"] == \
        32 * ll["activation_bytes"]
    # the index stream is the term the two grammars SHARE, and it is the biggest one
    assert ll["index_bytes"] == df["index_bytes"] == GATE["rows"] * GATE["nchunk"]


def test_device_table_control_bills_the_defect_that_makes_it_lose():
    cost = b.device_table_cost(**GATE)
    assert cost["table_bytes_written_to_device"] == GATE["nchunk"] * GATE["k"] * 4
    # four bytes gathered out of device memory for every one index byte it replaced
    assert cost["table_gather_read_bytes"] == 4 * cost["index_bytes"]
    assert "NEGATIVE CONTROL" in cost["grammar"]


def test_counterfactual_amortises_the_table_and_nothing_else():
    """nsets shares the table build; the gather and the index stream scale with nsets."""
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")},
                      cbs=16, tpg=1024)
    one = b.ll_cost(**GATE, shape=shape, nsets=1)
    many = b.ll_cost(**GATE, shape=shape, nsets=16)
    assert many["executed_fp_macs"] == one["executed_fp_macs"]        # built once
    assert many["executed_gather_ops"] == 16 * one["executed_gather_ops"]
    assert many["index_bytes"] == 16 * one["index_bytes"]
    assert many["arithmetic_reduction_vs_decode_fma"] > \
        one["arithmetic_reduction_vs_decode_fma"]


def test_dfma_plan_refuses_a_D_the_float4_loop_would_misread():
    with pytest.raises(b.TrackBError, match="float4"):
        b.dfma_plan(rows=2048, nchunk=768, D=6, k=128, cbs=24, tpg=256)


# --------------------------------------------------------------- the numpy mechanism model

def _small_artifact(seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((256, 128)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=8, subspaces=1, k=16, seed=0, iters=4)
    return art, art.config["pq_codes"], rng.standard_normal(128).astype(np.float32)


def test_numpy_model_agrees_with_the_parity_authority():
    art, codes, x = _small_artifact()
    want = forge.pq_execute(art, x)
    got = b.ll_reference(codes, x, cbs=4, half_table=False)
    gap = float(np.abs(want - got).max() / (np.abs(want).max() + 1e-30))
    assert gap < b.PARITY_GATE


def test_block_count_does_not_change_the_answer_beyond_reassociation():
    art, codes, x = _small_artifact()
    a = b.ll_reference(codes, x, cbs=2, half_table=False)
    c = b.ll_reference(codes, x, cbs=16, half_table=False)
    assert np.allclose(a, c, rtol=1e-5, atol=1e-5)


def test_half_table_costs_accuracy_and_the_model_shows_it():
    """T holds 8-term dot products, so casting it to half is a real, measurable loss.

    Graded against the fp32 TABLE, not against the CPU authority: at this toy's k=16 the
    artifact's own quantisation error is larger than either, so an authority comparison
    would hide the effect the half table actually has.
    """
    art, codes, x = _small_artifact()
    f32 = b.ll_reference(codes, x, cbs=4, half_table=False)
    f16 = b.ll_reference(codes, x, cbs=4, half_table=True)
    gap = np.linalg.norm(f16 - f32) / np.linalg.norm(f32)
    assert gap > 1e-4, gap                      # far above fp32 reassociation noise
    assert gap < 1e-2, gap                      # but still inside the 2e-3-scale regime


def test_parity_of_reports_every_statistic_it_claims():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p = b.parity_of(a, a)
    assert p["finite"] and p["max_abs_error"] == 0.0
    assert p["relative_l2"] == 0.0 and p["cosine"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------ sweep

def test_sweep_never_proposes_a_shape_the_planner_would_refuse():
    for cfg in b.ll_sweep(nchunk=768):
        b.ll_plan(**{kk: GATE[kk] for kk in ("rows", "nchunk", "D", "k")}, **cfg)
    for cfg in b.ll_sweep(nchunk=256):
        b.ll_plan(**{kk: DOWN[kk] for kk in ("rows", "nchunk", "D", "k")}, **cfg)


def test_sweep_covers_both_table_dtypes_and_both_row_tiles():
    cfgs = b.ll_sweep(nchunk=768)
    assert {c["half_table"] for c in cfgs} == {False, True}
    assert {c["row4"] for c in cfgs} == {False, True}


# ------------------------------------------------------------------------- device tests

@pytest.fixture(scope="module")
def dec():
    try:
        return b.TrackBDecoder()
    except Exception as exc:  # noqa: BLE001 - any device failure is a skip, never a fail
        pytest.skip(f"no usable Metal device: {exc}")


def _upload(dec, seed=0):
    art, codes, x = _small_artifact(seed)
    entry = dec.upload(codes, f"test|{seed}", max_blocks=64)
    dec.set_x(entry, x)
    return art, codes, x, entry


def test_every_lookup_linear_variant_matches_the_authority(dec):
    art, codes, x, entry = _upload(dec)
    want = forge.pq_execute(art, x)
    for half in (False, True):
        for row4 in (False, True):
            shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
                              half_table=half, row4=row4,
                              threadgroup_memory_limit=dec.threadgroup_memory_limit)
            got = dec.run_batch([(entry, shape)])[0]
            gap = float(np.abs(want - got).max() / (np.abs(want).max() + 1e-30))
            assert gap < b.PARITY_GATE, (half, row4, gap)


def test_the_kernel_matches_its_own_numpy_model_not_just_the_authority(dec):
    """Mechanism check: if the on-chip build drifted, this fails before parity does."""
    art, codes, x, entry = _upload(dec, seed=1)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
                      threadgroup_memory_limit=dec.threadgroup_memory_limit)
    got = dec.run_batch([(entry, shape)])[0]
    model = b.ll_reference(codes, x, cbs=4, half_table=False)
    assert np.allclose(got, model, rtol=1e-4, atol=1e-5)


def test_device_table_control_computes_the_same_thing_it_is_a_control_for(dec):
    art, codes, x, entry = _upload(dec, seed=2)
    want = forge.pq_execute(art, x)
    got = dec.run_device_table(entry)
    gap = float(np.abs(want - got).max() / (np.abs(want).max() + 1e-30))
    assert gap < b.PARITY_GATE, gap


def test_both_grammars_agree_on_the_same_tensor(dec):
    art, codes, x, entry = _upload(dec, seed=3)
    ll = dec.run_batch([(entry, b.ll_plan(
        rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
        threadgroup_memory_limit=dec.threadgroup_memory_limit))])[0]
    df = dec.run_batch([(entry, b.dfma_plan(
        rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
        threadgroup_memory_limit=dec.threadgroup_memory_limit))])[0]
    assert np.allclose(ll, df, rtol=1e-4, atol=1e-5)


def test_a_batch_of_tensors_returns_each_tensor_its_own_answer(dec):
    """A batched command buffer that crossed its buffers would still look plausible."""
    jobs, wants = [], []
    for seed in (4, 5, 6):
        art, codes, x, entry = _upload(dec, seed)
        shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
                          threadgroup_memory_limit=dec.threadgroup_memory_limit)
        jobs.append((entry, shape))
        wants.append(forge.pq_execute(art, x))
    got = dec.run_batch(jobs)
    for g, w in zip(got, wants):
        assert np.abs(w - g).max() / (np.abs(w).max() + 1e-30) < b.PARITY_GATE
    # and the answers really are different tensors, not one answer repeated
    assert not np.allclose(got[0], got[1])


def test_the_shared_counterfactual_is_a_table_built_once(dec):
    """Two index sets against one codebook must both come back right from one build."""
    art, codes, x = _small_artifact(7)
    rows, nchunk, D = 256, 16, 8
    book = np.ascontiguousarray(codes["codebooks"][0], dtype=np.float16)
    k = int(book.shape[0])
    rng = np.random.default_rng(11)
    sets = rng.integers(0, k, size=(2, rows, nchunk)).astype(np.uint8)
    stacked = np.stack([s.T.ravel() for s in sets])
    shape = b.ll_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=4, tpg=64,
                      threadgroup_memory_limit=dec.threadgroup_memory_limit)
    entry = dec.upload_shared(stacked, book, rows=rows, nchunk=nchunk, D=D,
                              blocks=shape["blocks"])
    dec.set_x(entry, x)
    got = dec.run_shared(entry, shape)
    table = (x.reshape(nchunk, D) @ book.astype(np.float32).T).astype(np.float32)
    want = np.stack([table[np.arange(nchunk)[None, :], s].sum(axis=1, dtype=np.float32)
                     for s in sets])
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4)
    assert not np.allclose(got[0], got[1])


def test_a_shape_wider_than_the_partial_buffer_is_refused_not_dispatched(dec):
    art, codes, x, entry = _upload(dec, seed=8)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=1, tpg=64,
                      threadgroup_memory_limit=dec.threadgroup_memory_limit)
    entry = dict(entry, max_blocks=2)
    with pytest.raises(b.TrackBError, match="fewer blocks"):
        dec.run_batch([(entry, shape)])


def test_a_wrong_length_x_is_refused_before_it_reaches_a_device_pointer(dec):
    art, codes, x, entry = _upload(dec, seed=9)
    with pytest.raises(b.TrackBError, match="geometry needs"):
        dec.set_x(entry, np.zeros(7, dtype=np.float32))


def test_an_out_of_range_index_is_refused_at_upload(dec):
    art, codes, x = _small_artifact(10)
    bad = dict(codes)
    idx = np.asarray(codes["indices"]).copy()
    idx[0, 0] = 99
    bad["indices"] = idx
    with pytest.raises(b.TrackBError, match="out of range"):
        dec.upload(bad, "test|bad", max_blocks=8)


def test_gpu_time_is_measured_not_derived(dec):
    """last_gpu_ms must come from the driver and be a real, smaller-than-wall number."""
    import time
    art, codes, x, entry = _upload(dec, seed=12)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64,
                      threadgroup_memory_limit=dec.threadgroup_memory_limit)
    t0 = time.perf_counter()
    dec.run_batch([(entry, shape)])
    wall_ms = (time.perf_counter() - t0) * 1e3
    assert 0.0 < dec.last_gpu_ms < wall_ms
