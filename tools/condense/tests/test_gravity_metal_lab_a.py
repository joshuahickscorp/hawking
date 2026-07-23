"""Guards for the Track A 2D split-chunk kernel.

Everything that decides whether a variant is legal -- the grid plan, the stage_x policy,
the byte accounting, the 7-bit unpack arithmetic and the reassociated reduce -- is a pure
function of the geometry, so it is tested here with no GPU in sight.  The two assertions
that genuinely need a dispatch are skipped when Metal or the device is missing, because a
test process is not guaranteed to have either.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_pack  # noqa: E402
import gravity_forge as forge  # noqa: E402
import gravity_metal_lab_a as A  # noqa: E402

R0 = dict(D=8, k=128)
GATE = dict(rows=2048, nchunk=768, **R0)      # [2048, 6144]
DOWN = dict(rows=6144, nchunk=256, **R0)      # [6144, 2048]
ATTN = dict(rows=6144, nchunk=2048, **R0)     # [6144, 16384], the 64 KB-x geometry


def _codes(rows: int, cols: int, *, D: int = 8, k: int = 128, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=D, subspaces=1, k=k, seed=seed, iters=2)
    return art.config["pq_codes"]


# ------------------------------------------------------------------ the grid, which is the point

def test_split_widens_the_grid_instead_of_repackaging_it():
    """The production defect: threads == rows regardless of THREADS.  blocks multiplies it."""
    one = A.plan(tpg=256, blocks=1, **GATE)
    many = A.plan(tpg=256, blocks=32, **GATE)
    assert one["threads_in_flight"] == GATE["rows"]
    assert many["threads_in_flight"] == GATE["rows"] * 32 == 65536
    # repackaging the same threads differently must NOT change the thread count
    assert A.plan(tpg=64, blocks=1, **GATE)["threads_in_flight"] == one["threads_in_flight"]


def test_blocks_one_skips_the_reduce_dispatch():
    assert A.plan(tpg=256, blocks=1, **GATE)["dispatches"] == 1
    assert A.plan(tpg=256, blocks=8, **GATE)["dispatches"] == 2
    assert A.plan(tpg=256, blocks=8, **GATE)["command_buffers"] == 1


def test_chunk_blocks_cover_every_chunk():
    for geom in (GATE, DOWN, ATTN):
        for blocks in (1, 3, 8, 32, 64, 97):
            shape = A.plan(tpg=256, blocks=blocks, **geom)
            assert shape["chunks_per_block"] * blocks >= geom["nchunk"]


# ------------------------------------------------------------------ stage_x is a policy

def test_attention_geometry_is_unstageable_whole_but_stageable_per_block():
    """This is the 33.3 GB/token defect and its fix, stated as an assertion."""
    whole = A.plan(tpg=256, blocks=1, stage_x_policy=True, **ATTN)
    assert not whole["stage_x"]
    assert "exceeds" in whole["stage_x_refused_reason"]
    blocked = A.plan(tpg=256, blocks=8, stage_x_policy=True, **ATTN)
    assert blocked["stage_x"]
    assert blocked["scratch_bytes"] <= A.DEFAULT_THREADGROUP_MEMORY


def test_stage_x_off_is_recorded_as_requested_off_not_as_a_refusal():
    shape = A.plan(tpg=256, blocks=8, stage_x_policy=False, **GATE)
    assert shape["stage_x"] is False
    assert shape["stage_x_requested"] is False
    assert shape["stage_x_refused_reason"] is None


def test_scratch_never_exceeds_the_threadgroup_limit():
    for geom in (GATE, DOWN, ATTN):
        for blocks in (1, 2, 8, 32, 64):
            for want in (False, True):
                shape = A.plan(tpg=256, blocks=blocks, stage_x_policy=want, **geom)
                assert shape["scratch_bytes"] <= A.DEFAULT_THREADGROUP_MEMORY


def test_kernel_name_matches_the_realised_staging_not_the_requested_one():
    shape = A.plan(tpg=256, blocks=1, stage_x_policy=True, **ATTN)
    assert shape["kernel"].split("_")[2] == "nox"


def test_vec4_is_refused_when_D_is_not_a_multiple_of_four():
    shape = A.plan(rows=64, nchunk=8, D=6, k=16, tpg=32, blocks=1, vec4=True)
    assert shape["vec4"] is False and shape["kernel"].endswith("scalar")


def test_plan_rejects_nonsense():
    with pytest.raises(A.TrackAError):
        A.plan(rows=0, nchunk=8, D=8, k=128, tpg=32, blocks=1)
    with pytest.raises(A.TrackAError):
        A.plan(tpg=0, blocks=1, **GATE)
    with pytest.raises(A.TrackAError):
        # a codebook that cannot fit threadgroup memory at all
        A.plan(rows=64, nchunk=8, D=8, k=128, tpg=32, blocks=1, threadgroup_memory_limit=1024)


# ------------------------------------------------------------------ byte accounting

def test_seven_bit_stream_is_one_eighth_smaller_than_the_uint8_one():
    kw = dict(rows=2048, cols=6144, nchunk=768, D=8, k=128)
    u8 = A.executed_bytes(shape=A.plan(tpg=256, blocks=32, bits7=False, **GATE), **kw)
    b7 = A.executed_bytes(shape=A.plan(tpg=256, blocks=32, bits7=True, **GATE), **kw)
    assert u8["index_bits_executed"] == 8 and b7["index_bits_executed"] == 7
    assert b7["index_bytes"] == (2048 * 768 * 7 + 7) // 8
    assert abs(1 - b7["index_bytes"] / u8["index_bytes"] - 0.125) < 1e-3


def test_unique_bytes_never_exceed_the_re_read_upper_bound():
    for geom, cols in ((GATE, 6144), (DOWN, 2048), (ATTN, 16384)):
        for blocks in (1, 8, 32, 64):
            for want in (False, True):
                shape = A.plan(tpg=256, blocks=blocks, stage_x_policy=want, **geom)
                t = A.executed_bytes(cols=cols, rows=geom["rows"], nchunk=geom["nchunk"],
                                     D=8, k=128, shape=shape)
                assert t["unique_total_bytes"] <= t["executed_total_bytes"]
                assert t["unique_read_bytes"] <= t["executed_read_bytes"]


def test_not_staging_x_at_the_attention_geometry_bills_the_whole_activation_per_thread():
    """The billed difference is the reason the policy exists."""
    kw = dict(rows=6144, cols=16384, nchunk=2048, D=8, k=128)
    off = A.executed_bytes(shape=A.plan(tpg=256, blocks=8, stage_x_policy=False, **ATTN), **kw)
    on = A.executed_bytes(shape=A.plan(tpg=256, blocks=8, stage_x_policy=True, **ATTN), **kw)
    assert off["activation_bytes"] == 6144 * 2048 * 8 * 4
    assert on["activation_bytes"] < off["activation_bytes"] / 100


def test_blocks_one_has_no_partial_traffic():
    t = A.executed_bytes(rows=2048, cols=6144, nchunk=768, D=8, k=128,
                         shape=A.plan(tpg=256, blocks=1, **GATE))
    assert t["partial_write_bytes"] == 0 and t["partial_read_bytes"] == 0


# ------------------------------------------------------------------ the 7-bit unpack arithmetic

def test_in_kernel_unpack_matches_the_packer_at_every_bit_phase():
    """The MSL does a 16-bit big-endian window, a shift and a mask; this is that, in numpy."""
    rng = np.random.default_rng(7)
    for count in (1, 7, 8, 9, 63, 64, 65, 4096, 12289):
        values = rng.integers(0, 128, size=count, dtype=np.uint64)
        raw = glm52_pack.pack_indices(values, 7) + b"\x00\x00"
        assert np.array_equal(A.unpack7_reference(raw, count), values.astype(np.uint8))
        # and against the packer's own inverse, so neither side is graded by itself
        assert np.array_equal(A.unpack7_reference(raw, count),
                              glm52_pack.unpack_indices(raw, count, 7)[:count])


def test_unpack_window_never_reads_past_the_padded_allocation():
    count = 1000
    raw = glm52_pack.pack_indices(np.full(count, 127, dtype=np.uint64), 7)
    last_bit = (count - 1) * 7
    assert (last_bit >> 3) + 1 < len(raw) + 2, "the +2 pad bytes are what make this safe"


# ------------------------------------------------------------------ the reassociated reduce

def test_split_reduce_matches_the_cpu_authority_within_the_fp16_codebook_floor():
    codes = _codes(256, 512)
    art = forge.PackedArtifact("product_quant", np.empty((0,), dtype=np.float32),
                               codes["rows"] * codes["cols"], forge.ByteLedger(), 0, 0,
                               {"pq_codes": codes})
    x = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    reference = forge.pq_execute(art, x)
    for blocks in (1, 4, 8, 64):
        got = A.split_reference(codes, x, blocks)
        gap = np.abs(reference - got).max() / (np.abs(reference).max() + 1e-30)
        assert gap < A.PARITY_GATE, (blocks, gap)


def test_reassociation_alone_does_not_move_the_answer_past_the_gate():
    codes = _codes(256, 512)
    x = np.random.default_rng(2).standard_normal(512).astype(np.float32)
    one = A.split_reference(codes, x, 1)
    for blocks in (4, 32):
        split = A.split_reference(codes, x, blocks)
        assert np.abs(one - split).max() / (np.abs(one).max() + 1e-30) < A.PARITY_GATE


# ------------------------------------------------------------------ verdict wording

def test_b7_verdict_has_a_noise_band_so_a_wobble_is_not_a_win():
    assert A._b7_verdict([]) == "UNMEASURED"
    assert A._b7_verdict([0.999, 1.001, 1.0]).startswith("NATIVE_7BIT_NEUTRAL")
    assert A._b7_verdict([0.5, 0.5, 0.5]).startswith("NATIVE_7BIT_WINS")
    assert A._b7_verdict([1.5, 1.5, 1.5]).startswith("NATIVE_7BIT_LOSES")


def test_every_kernel_the_planner_can_name_exists_in_the_metal_source():
    for bits7 in (False, True):
        for stage_x in (False, True):
            for vec4 in (False, True):
                name = A.kernel_name(bits7=bits7, stage_x=stage_x, vec4=vec4)
                assert f"GRAVITY_PARTIAL({name}," in A.METAL_SOURCE
    assert "kernel void gravity_pq_reduce" in A.METAL_SOURCE


def test_module_never_writes_to_the_live_artifact_directory():
    """The shards are a running campaign's output.  This file must stay read-only on them."""
    source = Path(A.__file__).read_text()
    for forbidden in ("shutil", ".unlink(", ".rename(", "os.remove", "open("):
        assert forbidden not in source, forbidden
    # the only write in the module is the report, and it goes to reports/, not to the shards
    assert source.count(".write_text(") == 1 and "args.out.write_text" in source


# ------------------------------------------------------------------ dispatch (needs a GPU)

def _decoder():
    try:
        return A.TrackADecoder()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no usable Metal device: {exc}")


def test_dispatch_matches_the_cpu_authority_for_every_compiled_variant():
    dec = _decoder()
    codes = _codes(512, 256)
    art = forge.PackedArtifact("product_quant", np.empty((0,), dtype=np.float32),
                               codes["rows"] * codes["cols"], forge.ByteLedger(), 0, 0,
                               {"pq_codes": codes})
    x = np.random.default_rng(3).standard_normal(256).astype(np.float32)
    reference = forge.pq_execute(art, x)
    entry = dec.upload(codes, "test", max_blocks=8)
    for blocks in (1, 4, 8):
        for vec4 in (False, True):
            for bits7 in (False, True):
                for want in (False, True):
                    shape = A.plan(rows=512, nchunk=32, D=8, k=128, tpg=64, blocks=blocks,
                                   vec4=vec4, bits7=bits7, stage_x_policy=want,
                                   threadgroup_memory_limit=dec.threadgroup_memory_limit)
                    got = dec.matvec(entry, x, shape)
                    parity = A.parity_of(got, reference)
                    assert parity["finite"]
                    assert parity["relative_max_gap"] < A.PARITY_GATE, (shape["kernel"], blocks)


def test_dispatch_refuses_an_x_of_the_wrong_length():
    dec = _decoder()
    codes = _codes(512, 256)
    entry = dec.upload(codes, "test_bad_x", max_blocks=4)
    shape = A.plan(rows=512, nchunk=32, D=8, k=128, tpg=64, blocks=4,
                   threadgroup_memory_limit=dec.threadgroup_memory_limit)
    with pytest.raises(A.TrackAError):
        dec.matvec(entry, np.zeros(999, dtype=np.float32), shape)
