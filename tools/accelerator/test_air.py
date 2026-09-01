"""FRONT A pins. Execution tests need MLX; schema tests do not."""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/accelerator"))
import air, receipt  # noqa: E402

np = pytest.importorskip("numpy")


def test_dtype_and_shape_are_validated():
    with pytest.raises(ValueError):
        air.AirTensor("x", (4,), dtype="int4")
    with pytest.raises(ValueError):
        air.AirTensor("x", (0,))


def test_air_is_ssa():
    p = air.AirProgram("dup", [air.AirTensor("x", (4,)), air.AirTensor("y", (4,))],
                       [air.AirOp("add", ("x", "y"), "z"),
                        air.AirOp("mul", ("x", "y"), "z")], "z")
    with pytest.raises(ValueError, match="SSA"):
        p.validate()


def test_reading_an_undefined_operand_is_refused():
    p = air.AirProgram("bad", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("nope",), "z")], "z")
    with pytest.raises(ValueError, match="undefined"):
        p.validate()


def test_unproduced_output_is_refused():
    p = air.AirProgram("bad", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("x",), "t")], "z")
    with pytest.raises(ValueError, match="never produced"):
        p.validate()


def test_intermediates_lower_as_scalars_not_buffers():
    """The bug that actually happened: an SSA intermediate was declared a scalar and
    then indexed `t[i]`, which does not compile."""
    p = air.AirProgram("f", [air.AirTensor("x", (8,)), air.AirTensor("y", (8,))],
                       [air.AirOp("saxpy", ("x", "y"), "t"),
                        air.AirOp("relu", ("t",), "z")], "z",
                       specialization={"ALPHA": 2.0})
    msl = air.lower_to_msl(p)
    # `"t[i]" not in msl` is the wrong check: it matches inside `out[i]`.
    assert not re.search(r"\bt\[i\]", msl)
    assert "float t = " in msl
    assert "out[i] = " in msl
    assert "const float ALPHA" in msl


def test_only_the_final_op_writes_the_output_buffer():
    p = air.AirProgram("f", [air.AirTensor("x", (8,)), air.AirTensor("y", (8,))],
                       [air.AirOp("mul", ("x", "y"), "a"),
                        air.AirOp("relu", ("a",), "z")], "z")
    assert air.lower_to_msl(p).count("out[i] = ") == 1


def test_scale_uses_specialization_literal_without_fabricating_scalar_buffer():
    p = air.AirProgram(
        "negate", [air.AirTensor("x", (8,))],
        [air.AirOp("scale", ("x",), "z")], "z",
        specialization={"ALPHA": -1.0},
    )
    msl = air.lower_to_msl(p)
    assert "const float ALPHA = -1.0f" in msl
    assert "ALPHA * x[i]" in msl
    assert len(p.inputs) == 1


def test_placement_is_a_field_so_egb_cannot_force_a_redesign():
    p = air.AirProgram("f", [air.AirTensor("x", (4,))],
                       [air.AirOp("relu", ("x",), "z")], "z", device="NVIDIA_GPU_0")
    assert p.device == "NVIDIA_GPU_0"


def test_receipt_refuses_a_missing_identity():
    with pytest.raises(ValueError, match="missing identities"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                      identities={"machine": {}}, result={}, claim_boundary="x",
                      passed=True)


def test_receipt_refuses_absent_without_a_reason():
    ids = {k: {"status": "ABSENT", "reason": "n/a"} for k in receipt.IDENTITIES}
    ids["transport"] = {"status": "ABSENT"}
    with pytest.raises(ValueError, match="without a reason"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="INSTANCE",
                      identities=ids, result={}, claim_boundary="x", passed=True)


def test_receipt_refuses_an_invented_experiment_class():
    ids = {k: receipt.absent("n/a") for k in receipt.IDENTITIES}
    with pytest.raises(ValueError, match="not a canonical class"):
        receipt.build(experiment_class="ACCEL-VIBES", knowledge_level="INSTANCE",
                      identities=ids, result={}, claim_boundary="x", passed=True)


def test_receipt_refuses_promotion_to_an_unknown_level():
    ids = {k: receipt.absent("n/a") for k in receipt.IDENTITIES}
    with pytest.raises(ValueError, match="not a knowledge level"):
        receipt.build(experiment_class="ACCEL-KERNEL", knowledge_level="UNIVERSAL",
                      identities=ids, result={}, claim_boundary="x", passed=True)




def test_metal_execution_matches_numpy():
    pytest.importorskip("mlx.core")
    n = 1024
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    p = air.AirProgram("t", [air.AirTensor("x", (n,)), air.AirTensor("y", (n,))],
                       [air.AirOp("saxpy", ("x", "y"), "t"),
                        air.AirOp("relu", ("t",), "z")], "z",
                       specialization={"ALPHA": 2.5})
    got = np.array(air.execute(p, {"x": x, "y": y}))
    assert np.abs(got - np.maximum(2.5 * x + y, 0)).max() < 1e-5


# --- synchronization, memory domains, side effects (the gap G043 named) ---

def test_air_represents_a_barrier_and_refuses_to_lower_it():
    """A silently dropped barrier is a race. Refusing beats emitting."""
    p = air.AirProgram("b", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z")], "z",
                       barriers=[air.AirBarrier("THREADGROUP")])
    ok, why = p.executable_on_metal_backend()
    assert not ok and "barrier" in why
    with pytest.raises(NotImplementedError, match="barrier"):
        air.lower_to_msl(p)


def test_an_unknown_sync_scope_is_refused():
    with pytest.raises(ValueError, match="unknown sync scope"):
        air.AirBarrier("GLOBAL_ISH")


def test_side_effects_are_declared_and_block_lowering():
    p = air.AirProgram("s", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z", writes_external=True)], "z")
    assert p.has_side_effects()
    with pytest.raises(NotImplementedError, match="side effects"):
        air.lower_to_msl(p)


def test_a_writable_operand_counts_as_a_side_effect():
    p = air.AirProgram("s", [air.AirTensor("x", (8,), read_only=False)],
                       [air.AirOp("relu", ("x",), "z")], "z")
    assert p.has_side_effects()


def test_an_unknown_memory_domain_is_refused():
    with pytest.raises(ValueError, match="unknown memory domain"):
        air.AirTensor("x", (8,), memory_domain="SOMEWHERE")


def test_a_foreign_domain_blocks_lowering_rather_than_being_ignored():
    p = air.AirProgram("v", [air.AirTensor("x", (8,), memory_domain="NVIDIA_VRAM_DIRECT")],
                       [air.AirOp("relu", ("x",), "z")], "z")
    with pytest.raises(NotImplementedError, match="HUMF"):
        air.lower_to_msl(p)


def test_apple_um_is_not_treated_as_foreign():
    p = air.AirProgram("v", [air.AirTensor("x", (8,), memory_domain="APPLE_UM")],
                       [air.AirOp("relu", ("x",), "z")], "z")
    assert p.executable_on_metal_backend()[0]


def test_air_domain_vocabulary_hands_to_humf_without_translation():
    """The point of sharing the vocabulary: an AIR requirement is directly a HUMF
    domain name, so no mapping layer can drift between them."""
    import humf
    p = air.AirProgram("v", [air.AirTensor("w", (8,), memory_domain="MOCK_EXTERNAL_VRAM")],
                       [air.AirOp("relu", ("w",), "z")], "z")
    req = p.required_domains()
    assert req == {"w": "MOCK_EXTERNAL_VRAM"}
    mp = humf.MockExternalMemoryProvider(capacity_bytes=1 << 20, bandwidth_gb_s=5.0)
    # the name AIR emitted is the name HUMF uses, with no translation in between
    assert req["w"] == mp.domain.name
    fabric = humf.Humf({"APPLE_UM": humf.Domain("APPLE_UM", 1 << 30, 589.73, physical=True),
                        mp.domain.name: mp.domain})
    assert req["w"] in fabric.domains


# --- AIR produces the tiled matmul rather than sitting beside a hand-written one ---

def test_air_matmul_lowering_emits_tiles_and_barriers():
    mm = air.AirMatmul("m", 64, 64, 64, tile=16)
    src = air.lower_matmul_to_msl(mm)
    assert "threadgroup float As" in src and "threadgroup float Bs" in src
    assert src.count("threadgroup_barrier") == 2
    assert mm.barrier_scopes_emitted() == ["THREADGROUP", "THREADGROUP"]


def test_threadgroup_barriers_are_executable_because_air_emits_them():
    """Barrier REPRESENTATION is refused for elementwise programs, but a matmul
    LOWERS its own barriers, so the scope is executable there. The distinction is
    the point: refusal tracks what the backend can emit, not what AIR can name."""
    mm = air.AirMatmul("m", 64, 64, 64)
    ok, _ = mm.executable_on_metal_backend()
    assert ok
    p = air.AirProgram("b", [air.AirTensor("x", (8,))],
                       [air.AirOp("relu", ("x",), "z")], "z",
                       barriers=[air.AirBarrier("THREADGROUP")])
    assert not p.executable_on_metal_backend()[0]


def test_a_matmul_operand_in_a_foreign_domain_is_refused():
    with pytest.raises(NotImplementedError, match="HUMF"):
        air.lower_matmul_to_msl(
            air.AirMatmul("v", 64, 64, 64, a_domain="NVIDIA_VRAM_DIRECT"))


@pytest.mark.parametrize("tile,msg", [(64, "exceeds 32"), (3, "power of two"),
                                      (0, "power of two")])
def test_impossible_tiles_are_refused(tile, msg):
    with pytest.raises(ValueError, match=msg):
        air.AirMatmul("t", 64, 64, 64, tile=tile).validate()


def test_tile_is_a_specialization_the_forge_can_tune():
    """Different tiles produce different kernels, which is what makes tuning possible
    at the AIR level rather than by editing MSL."""
    srcs = {t: air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, tile=t))
            for t in (8, 16, 32)}
    assert len(set(srcs.values())) == 3
    assert "As[8][8]" in srcs[8] and "As[32][32]" in srcs[32]


# --- simdgroup matmul strategy ---

def test_strategies_emit_different_kernels_and_launches():
    t = air.AirMatmul("m", 64, 64, 64, tile=16, strategy="tiled")
    s = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup")
    assert air.lower_matmul_to_msl(t) != air.lower_matmul_to_msl(s)
    assert "simdgroup_multiply_accumulate" in air.lower_matmul_to_msl(s)
    assert t.launch() != s.launch()


def test_simdgroup_emits_no_barrier_because_a_simd_runs_in_lockstep():
    assert air.AirMatmul("m", 64, 64, 64, strategy="simdgroup").barrier_scopes_emitted() == []
    assert air.AirMatmul("m", 64, 64, 64, strategy="tiled").barrier_scopes_emitted() == [
        "THREADGROUP", "THREADGROUP"]


@pytest.mark.parametrize("dim", [("m", 60), ("k", 60), ("n", 60)])
def test_simdgroup_refuses_shapes_that_are_not_multiples_of_eight(dim):
    kw = {"m": 64, "k": 64, "n": 64}
    kw[dim[0]] = dim[1]
    with pytest.raises(ValueError, match="multiple of 8"):
        air.AirMatmul("m", strategy="simdgroup", **kw).validate()


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown matmul strategy"):
        air.AirMatmul("m", 64, 64, 64, strategy="wishful").validate()


def test_tiled_has_no_shape_constraint_so_it_is_not_deleted():
    """simdgroup wins on speed but cannot take arbitrary shapes, which is why both
    strategies are kept rather than one replacing the other."""
    air.AirMatmul("m", 60, 60, 60, strategy="tiled").validate()


# --- register blocking ---

def test_block_2_emits_four_accumulators_and_four_macs():
    s = air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=2))
    assert s.count("make_filled_simdgroup_matrix") == 4
    assert s.count("simdgroup_multiply_accumulate") == 4
    assert s.count("simdgroup_load") == 4      # 4 loads feeding 4 MACs, not 2 feeding 1
    assert s.count("simdgroup_store") == 4


def test_block_1_is_the_unblocked_kernel():
    s = air.lower_matmul_to_msl(air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=1))
    assert s.count("simdgroup_multiply_accumulate") == 1
    assert s.count("simdgroup_load") == 2


def test_blocking_changes_the_launch_geometry():
    a = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=1)
    b = air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=2)
    assert a.launch() != b.launch()


def test_block_2_refuses_shapes_it_cannot_cover():
    with pytest.raises(ValueError, match="multiple of 16"):
        air.AirMatmul("m", 24, 64, 64, strategy="simdgroup", block=2).validate()


def test_unimplemented_block_is_refused():
    with pytest.raises(ValueError, match="only 1 and 2"):
        air.AirMatmul("m", 64, 64, 64, strategy="simdgroup", block=3).validate()


# --- reduction ---

def test_reduce_emits_a_barrier_and_a_simd_reduction():
    s = air.lower_reduce_to_msl(air.AirReduce("r", 1024, "sum"))
    assert "simd_sum" in s and s.count("threadgroup_barrier") == 1
    assert "simd_max" in air.lower_reduce_to_msl(air.AirReduce("r", 1024, "max"))


def test_reduce_refuses_an_unknown_op_and_a_bad_threadgroup():
    with pytest.raises(ValueError, match="unknown reduce op"):
        air.AirReduce("r", 64, "median").validate()
    with pytest.raises(ValueError, match="multiple of the"):
        air.AirReduce("r", 64, "sum", threadgroup=100).validate()


def test_reduce_partial_count_covers_every_element():
    for n in (1, 255, 256, 257, 1 << 20):
        rd = air.AirReduce("r", n, "sum", threadgroup=256)
        assert rd.partials() * 256 >= n


# --- fused softmax ---

def test_softmax_emits_two_barriers_and_both_simd_reductions():
    s = air.lower_softmax_to_msl(air.AirSoftmax("s", 8, 64))
    assert s.count("threadgroup_barrier") == 2
    assert "simd_max" in s and "simd_sum" in s
    assert air.AirSoftmax("s", 8, 64).barrier_scopes_emitted() == ["THREADGROUP"] * 2


def test_softmax_subtracts_the_row_max_for_stability():
    """Without it, exp of a large logit overflows to inf and the row becomes nan."""
    s = air.lower_softmax_to_msl(air.AirSoftmax("s", 8, 64))
    assert "rowmax" in s and "exp(x[base + c] - rowmax)" in s


def test_softmax_refuses_a_bad_threadgroup_and_empty_shapes():
    with pytest.raises(ValueError, match="multiple of 32"):
        air.AirSoftmax("s", 8, 64, threadgroup=48).validate()
    with pytest.raises(ValueError, match="must be positive"):
        air.AirSoftmax("s", 0, 64).validate()


def test_softmax_launches_one_threadgroup_per_row():
    sm = air.AirSoftmax("s", 37, 64, threadgroup=128)
    grid, tg = sm.launch()
    assert grid == (37 * 128, 1, 1) and tg == (128, 1, 1)


# --- fused attention ---

def test_attention_emits_four_barriers_and_holds_scores_in_threadgroup_memory():
    s = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32))
    assert s.count("threadgroup_barrier") == 4
    assert "threadgroup float scores[64]" in s
    assert air.AirAttention("a", 64, 64, 32).barrier_scopes_emitted() == ["THREADGROUP"] * 4


def test_attention_refuses_a_sequence_that_will_not_fit_in_threadgroup_memory():
    """This kernel is not flash-attention: it holds the whole score row, so a long
    sequence must be refused rather than silently truncated."""
    with pytest.raises(ValueError, match="online softmax"):
        air.AirAttention("a", 10, 100_000, 64).validate()


def test_causal_flag_changes_the_emitted_kernel():
    plain = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32, causal=False))
    causal = air.lower_attention_to_msl(air.AirAttention("a", 64, 64, 32, causal=True))
    assert "-INFINITY;" in causal and plain != causal


def test_attention_reports_the_materialisation_it_avoids():
    at = air.AirAttention("a", 1024, 1024, 64)
    assert at.materialised_bytes_avoided() == 2 * 1024 * 1024 * 4


# --- atomic reduction strategy ---

def test_atomic_strategy_emits_an_atomic_add_and_one_barrier():
    s = air.lower_reduce_to_msl(air.AirReduce("r", 1024, "sum", strategy="atomic"))
    assert "atomic_fetch_add_explicit" in s
    assert s.count("threadgroup_barrier") == 1


def test_atomic_strategy_refuses_max():
    """Metal has no float atomic max here, so max must stay two-stage rather than
    silently producing a wrong answer."""
    with pytest.raises(ValueError, match="sum only"):
        air.AirReduce("r", 64, "max", strategy="atomic").validate()


def test_unknown_reduce_strategy_is_refused():
    with pytest.raises(ValueError, match="unknown reduce strategy"):
        air.AirReduce("r", 64, "sum", strategy="wishful").validate()


def test_two_stage_remains_available_for_max():
    air.AirReduce("r", 64, "max", strategy="two_stage").validate()


# ---------------------------------------------------------------- scan (G046)

def test_scan_refuses_max_on_the_simd_prefix_strategy():
    """Metal has simd_prefix_inclusive_sum and _product but no prefix max. A strategy
    that quietly became a different strategy would make its own benchmark a lie."""
    with pytest.raises(ValueError, match="prefix max"):
        air.AirScan("s", 1024, "max", True, 256, "simd_prefix").validate()
    air.AirScan("s", 1024, "max", True, 256, "hillis_steele").validate()   # legal


def test_scan_refuses_exclusive_max_and_says_it_is_unbuilt_not_impossible():
    with pytest.raises(ValueError, match="NOT IMPOSSIBLE"):
        air.AirScan("s", 1024, "max", False, 256, "hillis_steele").validate()


def test_scan_strategies_emit_different_barrier_counts():
    """simd_prefix does its 32-wide scan inside the SIMD unit and needs 2 barriers;
    hillis_steele holds the block in threadgroup memory and pays 2 per doubling step.
    Conflating them would hide what the vendor primitive actually buys."""
    s = air.AirScan("s", 4096, "sum", True, 256, "simd_prefix")
    h = air.AirScan("s", 4096, "sum", True, 256, "hillis_steele")
    assert len(s.barrier_scopes_emitted()) == 2
    assert len(h.barrier_scopes_emitted()) == 1 + 2 * 8      # log2(256) steps
    assert "simd_prefix_inclusive_sum" in air.lower_scan_block_to_msl(s)
    assert "simd_prefix_inclusive_sum" not in air.lower_scan_block_to_msl(h)


def test_scan_blocks_covers_a_non_multiple():
    assert air.AirScan("s", 4099, "sum", True, 256).blocks() == 17


def test_scan_executes_and_matches_an_f64_oracle():
    pytest.importorskip("mlx.core")
    import numpy as np
    rng = np.random.default_rng(5)
    for n in (7, 1000, 4099, 1 << 16):
        x = rng.standard_normal(n).astype(np.float32)
        ref = np.cumsum(x.astype(np.float64))
        scale = float(np.max(np.abs(ref)))
        for strat in ("simd_prefix", "hillis_steele"):
            g = np.array(air.execute_scan(air.AirScan("t", n, "sum", True, 256, strat), x))
            assert np.max(np.abs(g - ref)) / scale < 1e-5, (n, strat)
        e = np.array(air.execute_scan(air.AirScan("t", n, "sum", False, 256), x))
        assert np.max(np.abs(e - (ref - x))) / scale < 1e-5
        m = np.array(air.execute_scan(air.AirScan("t", n, "max", True, 256, "hillis_steele"), x))
        assert np.array_equal(m, np.maximum.accumulate(x))


def test_a_block_local_scan_alone_is_wrong():
    """The structural fact that makes scan harder than reduction: a block cannot
    finish without every earlier block's total. Phase one ALONE must be wrong, or the
    three-phase shape is unmotivated and the correctness test proves nothing."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    n = 1 << 16
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n).astype(np.float32)
    ref = np.cumsum(x.astype(np.float64))
    sc = air.AirScan("c", n, "sum", True, 256, "simd_prefix")
    k = mx.fast.metal_kernel(name="t_blockonly", input_names=["x"],
                             output_names=["out", "sums"],
                             source=air.lower_scan_block_to_msl(sc),
                             ensure_row_contiguous=True)
    y, _ = k(inputs=[mx.array(x)], grid=(sc.blocks() * 256, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(n,), (sc.blocks(),)],
             output_dtypes=[mx.float32, mx.float32])
    mx.eval(y)
    err = np.max(np.abs(np.array(y) - ref)) / float(np.max(np.abs(ref)))
    assert err > 1e-3, f"block-local scan should be badly wrong, got {err}"


def test_kernel_cache_returns_the_same_object():
    """Building the wrapper per call put MLX's kernel lookup inside timing loops and
    showed up as 20%+ IQR. Pinned because it is a measurement correctness property,
    not a micro-optimization."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    src = air.lower_scan_offset_to_msl(air.AirScan("k", 1024, "sum", True, 256))
    a = air.metal_kernel(mx, name="t_cache", input_names=["y", "off"],
                         output_names=["out"], source=src, ensure_row_contiguous=True)
    b = air.metal_kernel(mx, name="t_cache", input_names=["y", "off"],
                         output_names=["out"], source=src, ensure_row_contiguous=True)
    assert a is b


# ------------------------------------------------------------ graphs (G046/G045)

def _body(n, src):
    return f"uint i=thread_position_in_grid.x; if(i<{n}u) out[i]={src}[i]*1.5f+0.25f;"


def _chain(n, k):
    g = air.AirGraph(f"c{k}", externals=["x"]); prev = "x"
    for j in range(k):
        g.nodes.append(air.AirGraphNode(f"n{j}", _body(n, prev), [prev], n)); prev = f"n{j}"
    return g


def _fan(n, k):
    g = air.AirGraph(f"f{k}", externals=["x"])
    for j in range(k):
        g.nodes.append(air.AirGraphNode(f"n{j}", _body(n, "x"), ["x"], n))
    return g


def test_graph_depth_and_width_separate_ordering_from_independence():
    """The two halves of what CUDA graphs and streams offer. A chain can only be
    batched; a fan is the only shape where concurrency could possibly help."""
    c, f = _chain(4096, 8), _fan(4096, 8)
    assert (c.serial_depth(), c.width()) == (8, 1)
    assert (f.serial_depth(), f.width()) == (1, 8)
    assert c.submissions() == f.submissions() == 1


def test_graph_refuses_a_forward_reference():
    g = air.AirGraph("bad", externals=["x"])
    g.nodes.append(air.AirGraphNode("a", _body(64, "later"), ["later"], 64))
    with pytest.raises(ValueError, match="neither an external nor a node"):
        g.validate()


def test_graph_refuses_duplicate_node_names():
    g = air.AirGraph("dup", externals=["x"])
    g.nodes += [air.AirGraphNode("a", _body(64, "x"), ["x"], 64),
                air.AirGraphNode("a", _body(64, "x"), ["x"], 64)]
    with pytest.raises(ValueError, match="unique"):
        g.validate()


def test_graph_executes_a_chain_in_order():
    """A chain of 1.5x+0.25 is order-sensitive: a dropped or reordered node changes
    the answer, so matching numpy proves the dependency edges were honoured."""
    pytest.importorskip("mlx.core")
    import numpy as np
    x = np.random.default_rng(1).standard_normal(4096).astype(np.float32)
    env = air.execute_graph(_chain(4096, 4), {"x": x})
    ref = x
    for _ in range(4):
        ref = ref * 1.5 + 0.25
    assert np.max(np.abs(np.array(env["n3"]) - ref)) < 1e-5


def test_eager_mode_is_a_control_not_a_feature():
    """eager=True exists only so the batched number has a baseline. It must produce
    the SAME answer -- if it did not, the comparison would be between two different
    computations rather than two submission strategies."""
    pytest.importorskip("mlx.core")
    import numpy as np
    x = np.random.default_rng(4).standard_normal(4096).astype(np.float32)
    a = air.execute_graph(_chain(4096, 4), {"x": x})["n3"]
    b = air.execute_graph(_chain(4096, 4), {"x": x}, eager=True)["n3"]
    assert np.array_equal(np.array(a), np.array(b))


# ------------------------------------------------ storage layout and reuse (G048)

def test_kernel_identity_is_the_msl_not_a_heuristic():
    import gravity_native as gnat
    assert gnat.kernel_identity(768, 2048) == gnat.kernel_identity(768, 2048)
    assert gnat.kernel_identity(768, 2048) != gnat.kernel_identity(2048, 768)
    assert gnat.kernel_identity(768, 2048) != gnat.kernel_identity(1408, 2048)


def test_fused_expert_storage_needs_preparation_and_2d_does_not():
    """The two conventions actually on disk. A fused [experts, in, 2*out] tensor
    stores its gate half TRANSPOSED, so the shape a config implies and the shape the
    bytes are in are different -- which is what makes reuse a layout claim."""
    import gravity_native as gnat
    shape, prep = gnat.stored_gate_shape([128, 2048, 1536])     # Qwen3-VL
    assert (shape, prep) == ((2048, 768), True)
    shape, prep = gnat.stored_gate_shape([768, 2048])            # Qwen3-30B-A3B
    assert (shape, prep) == ((768, 2048), False)
    shape, prep = gnat.stored_gate_shape([1408, 2048])           # Kimi-VL
    assert (shape, prep) == ((1408, 2048), False)


def test_the_fused_layout_emits_a_different_kernel_before_preparation():
    """The sharp version of the correction: the VL variant's tensor AS STORED does
    NOT share model #2's kernel, even though its GEMV shape does after preparation."""
    import gravity_native as gnat
    m2 = gnat.kernel_identity(768, 2048)
    stored, prep = gnat.stored_gate_shape([128, 2048, 1536])
    assert prep is True
    assert gnat.kernel_identity(*stored) != m2                  # as it sits on disk
    assert gnat.kernel_identity(stored[1], stored[0]) == m2      # after transposing


# ------------------------------------------- acceptance predicate (G046 / G048)

def _pack_fixture():
    import numpy as np
    import gravity_native as gnat
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 512)).astype(np.float32)
    packed, scale = gnat.pack_q4_g64(w)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    return gnat, w, packed, scale, x


def test_the_old_self_referential_predicate_accepts_an_ALL_ZEROS_pack():
    """The defect this exists to prevent, kept as an executable demonstration.

    Comparing the GPU kernel against a numpy decode of THE SAME packed bytes tests
    only that the kernel implements the representation. The original tensor never
    enters it, so a pack of pure zeros passes."""
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]

    def old_predicate(p, s):
        ref = gnat.dequantize(p, s, cols) @ x
        got = ref.copy()                      # a CORRECT kernel returns exactly this
        return float(np.max(np.abs(got - ref))) <= float(np.max(np.abs(ref))) * 1e-3

    assert old_predicate(np.zeros_like(packed), scale) is True
    assert old_predicate(packed, scale * 1000.0) is True


def test_accept_pack_rejects_every_broken_pack_the_old_one_accepted():
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]

    def verdict(p, s):
        ref = gnat.dequantize(p, s, cols) @ x
        return gnat.accept_pack(w, p, s, cols, ref, x)

    assert verdict(packed, scale)["accepted"] is True
    for label, p, s in [
        ("all zeros", np.zeros_like(packed), scale),
        ("scales x1000", packed, scale * 1000.0),
        ("scales x0.5", packed, scale * 0.5),
        ("nibbles swapped", ((packed >> 4) | (packed << 4)).astype(np.uint8), scale),
    ]:
        assert verdict(p, s)["accepted"] is False, label


def test_cosine_alone_cannot_tell_the_scaled_packs_apart():
    """The 2026-08-17 law, demonstrated rather than cited: cosine is SCALE-INVARIANT,
    so an honest pack, one scaled 1000x and one scaled 0.5x have the SAME cosine.
    The magnitude band is the entire difference between catching that and not."""
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    cos = {}
    for label, s in [("honest", scale), ("x1000", scale * 1000.0), ("x0.5", scale * 0.5)]:
        ref = gnat.dequantize(packed, s, cols) @ x
        r = gnat.accept_pack(w, packed, s, cols, ref, x)
        cos[label] = round(r["representation_fidelity"]["cosine"], 6)
    assert cos["honest"] == cos["x1000"] == cos["x0.5"], cos
    # and yet only one is accepted
    ref = gnat.dequantize(packed, scale, cols) @ x
    assert gnat.accept_pack(w, packed, scale, cols, ref, x)["accepted"] is True


def test_a_broken_KERNEL_is_caught_even_when_the_pack_is_honest():
    """The two gates are independent and both must be able to fail alone."""
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    ref = gnat.dequantize(packed, scale, cols) @ x
    bad_kernel = ref * 1.5
    r = gnat.accept_pack(w, packed, scale, cols, bad_kernel, x)
    assert r["accepted"] is False
    assert r["kernel_fidelity"]["ok"] is False
    assert r["representation_fidelity"]["ok"] is True     # the pack was fine


# ------------------------------------------------- GPU packer (G046, Front D)

@pytest.mark.parametrize("shape", [(768, 2048), (256, 512), (64, 64)])
def test_gpu_pack_is_BIT_EXACT_against_the_cpu_packer(shape):
    """Bit-exactness is achievable here and therefore REQUIRED. The quantiser is
    integer rounding of an f32 quotient and Metal's rint() is round-half-to-even
    exactly as numpy's is, so the correctness question is BINARY -- the bytes match
    or the port is wrong. A tolerance would have hidden a rounding-mode difference
    that changes one weight in a million."""
    pytest.importorskip("mlx.core")
    import numpy as np
    import gravity_native as gnat
    w = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    pc, sc = gnat.pack_q4_g64(w)
    pg, sg = gnat.pack_q4_g64_gpu(w)
    assert np.array_equal(pc, pg)
    assert np.array_equal(sc, sg)


def test_gpu_pack_bit_exact_on_the_paths_that_would_diverge():
    """The cases where a naive port silently differs: the amax==0 branch, an exact
    .5 quotient where the rounding mode decides, and a single huge outlier."""
    pytest.importorskip("mlx.core")
    import numpy as np
    import gravity_native as gnat
    rng = np.random.default_rng(1)
    cases = {}
    cases["all_zero"] = np.zeros((64, 128), np.float32)
    mixed = rng.standard_normal((64, 128)).astype(np.float32); mixed[0, :64] = 0.0
    cases["one_zero_group"] = mixed
    half = np.zeros((1, 64), np.float32)
    half[0, :] = (np.arange(64) - 32) * 0.5
    half[0, 0] = 7.0
    cases["exact_half_quotients"] = half
    cases["negatives_only"] = -np.abs(rng.standard_normal((32, 64))).astype(np.float32)
    for name, w in cases.items():
        pc, sc = gnat.pack_q4_g64(w)
        pg, sg = gnat.pack_q4_g64_gpu(w)
        assert np.array_equal(pc, pg), name
        assert np.array_equal(sc, sg), name


def test_the_f16_scale_overflows_above_a_known_absmax_and_the_gate_catches_it():
    """A pre-existing limit of the representation, not of the GPU port -- BOTH
    packers do it identically. Recorded because it is silent: the stored f16 scale
    becomes inf and the dequantized tensor becomes nan."""
    pytest.importorskip("mlx.core")
    import numpy as np
    import gravity_native as gnat
    rng = np.random.default_rng(2)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    w[3, 10] = 6e5                                   # absmax/7 exceeds f16's 65504
    pc, sc = gnat.pack_q4_g64(w)
    pg, sg = gnat.pack_q4_g64_gpu(w)
    assert np.array_equal(pc, pg)                    # the port is still faithful
    assert np.isinf(np.asarray(sc, dtype=np.float32)).any()
    x = rng.standard_normal(64).astype(np.float32)
    ref = gnat.dequantize(pc, sc, 64) @ x
    v = gnat.accept_pack(w, pc, sc, 64, ref, x)
    assert v["accepted"] is False                    # the gate refuses it


# --------------------------------------- the honest gate, made cheap (G046/G049)

def test_ref_matvec_agrees_with_the_dense_path_to_machine_epsilon():
    """The cheap reference must be the SAME reference, not a looser one.

    Verification became 61% of WorkUnit wall once the packer moved to the GPU, and
    the tempting fix is to check less. This checks exactly as much: the group-wise
    contraction and the dense rematerialization it replaced agree to float64 epsilon,
    so no threshold anywhere had to move."""
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    dense = gnat.dequantize(packed, scale, cols).astype(np.float64) @ x.astype(np.float64)
    cheap = gnat.ref_matvec(packed, scale, cols, x)
    rel = float(np.max(np.abs(cheap - dense)) / np.max(np.abs(dense)))
    assert rel < 1e-14, rel


def test_ref_matvec_never_materialises_a_dense_tensor():
    """S015 §19 in the verifier itself: the peak allocation must stay far below the
    dense tensor the old path built. Measured by tracemalloc, not by reading code."""
    import tracemalloc
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    dense_bytes = w.size * 4
    tracemalloc.start()
    gnat.ref_matvec(packed, scale, cols, x)
    _, peak_cheap = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    gnat.dequantize(packed, scale, cols).astype(np.float64) @ x.astype(np.float64)
    _, peak_dense = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_dense > dense_bytes, peak_dense       # the old path really did build it
    assert peak_cheap < peak_dense


def test_the_cheap_gate_returns_the_same_verdict_on_every_negative_control():
    """A cheaper gate is only honest if it rejects everything the expensive one did.

    Each control is run through accept_pack twice -- once with the reference computed
    the old dense way, once the new way -- and both the verdict and the numbers must
    match. A gate that got faster by getting more permissive fails here."""
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]

    def dense_ref(p, s):
        return gnat.dequantize(p, s, cols).astype(np.float64) @ x.astype(np.float64)

    controls = [
        ("honest", packed, scale, True),
        ("all zeros", np.zeros_like(packed), scale, False),
        ("scales x1000", packed, scale * 1000.0, False),
        ("scales x0.5", packed, scale * 0.5, False),
        ("nibbles swapped", ((packed >> 4) | (packed << 4)).astype(np.uint8), scale, False),
    ]
    for label, p, s, expected in controls:
        ref = dense_ref(p, s)
        cheap = gnat.accept_pack(w, p, s, cols, ref, x)
        assert cheap["accepted"] is expected, label
        assert abs(cheap["representation_fidelity"]["cosine"] -
                   float(np.dot((w.astype(np.float64) @ x.astype(np.float64)), ref) /
                         (np.linalg.norm(w.astype(np.float64) @ x.astype(np.float64))
                          * np.linalg.norm(ref)))) < 1e-12, label

    # and a broken kernel still fails gate 1 with an honest pack
    bad = gnat.accept_pack(w, packed, scale, cols, dense_ref(packed, scale) * 1.5, x)
    assert bad["accepted"] is False and bad["kernel_fidelity"]["ok"] is False


# ------------------------------------- the gate on the device (G046/G049, Front D)

def test_the_device_gate_returns_the_same_verdicts_as_the_float64_host_gate():
    """Moving the reference onto the GPU is only allowed if the DECISIONS survive.

    The numbers cannot survive -- the device accumulates in float32 -- so what is
    pinned is the verdict on the honest pack and on every negative control, which is
    the property the gate exists for."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    for label, p, s, expected in [
        ("honest", packed, scale, True),
        ("all zeros", np.zeros_like(packed), scale, False),
        ("scales x1000", packed, scale * 1000.0, False),
        ("scales x0.5", packed, scale * 0.5, False),
        ("nibbles swapped", ((packed >> 4) | (packed << 4)).astype(np.uint8), scale, False),
    ]:
        ref = gnat.ref_matvec(p, s, cols, x)
        host = gnat.accept_pack(w, p, s, cols, ref, x)
        dev = gnat.accept_pack_gpu(mx.array(w), mx.array(p),
                                   mx.array(s.astype(np.float32)), cols,
                                   mx.array(ref.astype(np.float32)), mx.array(x), mx=mx)
        assert host["accepted"] is expected, label
        assert dev["accepted"] is expected, label


def test_the_device_gate_still_catches_a_broken_kernel():
    """Gate 1 must be able to fail on its own, or moving it bought a vacuous check."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    gnat, w, packed, scale, x = _pack_fixture()
    cols = w.shape[1]
    ref = gnat.ref_matvec(packed, scale, cols, x).astype(np.float32)
    r = gnat.accept_pack_gpu(mx.array(w), mx.array(packed),
                             mx.array(scale.astype(np.float32)), cols,
                             mx.array(ref * 1.5), mx.array(x), mx=mx)
    assert r["accepted"] is False
    assert r["kernel_fidelity"]["ok"] is False
    assert r["representation_fidelity"]["ok"] is True


def test_headroom_names_the_regime_where_float32_must_not_decide():
    """A float32 gate is safe exactly when the verdict is far from its threshold.
    That is a property of the pack being judged, not of the gate, so it is reported
    with every verdict -- and it must say NO when the margin is thin."""
    import gravity_native as gnat
    thin = gnat.near_threshold_headroom(0.9900001, 1.0, 1e-9, 1e-3, 0.99, (0.9, 1.1))
    assert thin["safe_at_float32"] is False
    wide = gnat.near_threshold_headroom(0.994, 1.004, 1e-9, 1e-3, 0.99, (0.9, 1.1))
    assert wide["safe_at_float32"] is True
    assert wide["min_distance_to_a_threshold"] > thin["min_distance_to_a_threshold"]


def test_the_resident_path_packs_the_same_bytes_as_the_host_path():
    """Arm C decodes bf16 on the device and never brings the pack home. If that
    changed one byte it would be measuring a different computation."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    import gravity_native as gnat
    rng = np.random.default_rng(3)
    f32 = (rng.standard_normal((64, 256)) * 0.02).astype(np.float32)
    bits = (f32.view(np.uint32) >> 16).astype(np.uint16)          # truncate to bf16
    host = (bits.astype(np.uint32) << 16).view(np.float32).reshape(64, 256)
    dev = mx.array(bits.reshape(-1)).view(mx.bfloat16).reshape(64, 256)
    assert np.array_equal(np.array(dev.astype(mx.float32)), host)
    ph, sh = gnat.pack_q4_g64(host)
    pd, sd = gnat.pack_q4_g64_device(dev, mx=mx)
    assert np.array_equal(np.array(pd), ph)
    assert np.array_equal(np.array(sd), sh)


# ------------------------------- the representation study anchors on the real packer

def test_quantize_grouped_reproduces_the_shipped_representation_exactly():
    """The sweep's anchor point must BE ws_rtn_q4_g64, not a lookalike. If bits=4,
    group=64 diverged from the real packer by even a rounding, every other point on
    the curve would be measuring a different representation."""
    import numpy as np
    import gravity_native as gnat
    rng = np.random.default_rng(5)
    for shape in ((256, 512), (64, 64), (768, 2048)):
        w = (rng.standard_normal(shape) * 0.02).astype(np.float32)
        packed, scale = gnat.pack_q4_g64(w)
        shipped = gnat.dequantize(packed, scale, shape[1])
        study = gnat.quantize_grouped(w, bits=gnat.BITS, group=gnat.GROUP)
        assert np.array_equal(shipped, study), shape


def test_one_bit_is_the_deletion_control_not_a_cheap_option():
    """bound = 2^(bits-1)-1 is ZERO at bits=1, so absmax 1-bit deletes the tensor.
    Sealed by this campaign on 2026-08-17; pinned here so the sweep's floor cannot be
    misread as a working configuration."""
    import numpy as np
    import gravity_native as gnat
    w = np.random.default_rng(6).standard_normal((64, 128)).astype(np.float32)
    out = gnat.quantize_grouped(w, bits=1, group=64)
    assert np.count_nonzero(out) == 0
    assert gnat.fidelity(w, out)["cosine"] == 0.0


def test_bpw_counts_the_scale():
    import gravity_native as gnat
    assert gnat.bpw_grouped(4, 64) == 4.25
    assert gnat.bpw_grouped(4, 32) == 4.5        # smaller groups are NOT free
    assert gnat.bpw_grouped(2, 128) == 2.125


# ----------------------------------- barrier scopes: what each one can and cannot do

def test_barrier_msl_supplies_two_scopes_and_refuses_the_third_by_name():
    """SIMDGROUP was recorded as UNLOWERABLE for many blocks. It has an instruction and
    now emits one. DEVICE does NOT, and the refusal names the construct that does carry
    device-wide ordering rather than merely saying no."""
    import air
    assert "threadgroup_barrier" in air.barrier_msl("THREADGROUP")
    assert "simdgroup_barrier" in air.barrier_msl("SIMDGROUP")
    with pytest.raises(NotImplementedError, match="AirGraph"):
        air.barrier_msl("DEVICE")


def test_an_elementwise_program_refuses_a_barrier_for_the_RIGHT_reason():
    """The refusal used to blame the backend. That was wrong -- AIR's matmul emits
    threadgroup barriers. The real constraint is that an elementwise program has no
    shared state for a barrier to order, and the message has to say so or it teaches
    the wrong lesson to whoever reads it."""
    import air
    a = air.AirTensor("a", (16,), "f32")
    prog = air.AirProgram(name="bar_probe", inputs=[a],
                          ops=[air.AirOp("relu", ["a"], "y")], output="y",
                          barriers=[air.AirBarrier("SIMDGROUP")])
    ok, why = prog.executable_on_metal_backend()
    assert ok is False
    assert "nothing for a barrier to order" in why
    assert "PROGRAM SHAPE, not the backend" in why


def test_a_matmul_still_accepts_because_its_lowering_emits_its_own_barriers():
    """The distinction that must not rot: refusal tracks what the backend can EMIT for
    THIS program, not what AIR can name. Sat beside the elementwise refusal on purpose."""
    import air
    mm = air.AirMatmul(name="mm_probe", m=32, k=32, n=32, dtype="f32")
    ok, why = mm.executable_on_metal_backend()
    assert ok is True, why


def test_topk_sample_refuses_what_it_cannot_do():
    mk = lambda **kw: air.AirTopKSample(**{"name": "t", "rows": 2, "cols": 64,
                                           "k": 4, **kw})
    with pytest.raises(ValueError, match="exceeds 64"):
        mk(k=65, cols=1024).validate()
    with pytest.raises(ValueError, match="must be in 1"):
        mk(k=65, cols=32).validate()
    with pytest.raises(ValueError, match="temperature"):
        mk(temperature=0.0).validate()
    with pytest.raises(ValueError, match="multiple of 32"):
        mk(threadgroup=100).validate()
    # a name with a dot becomes an invalid Metal SYMBOL and fails deep inside the MLX
    # header with no mention of the name; found by naming a variant after its temperature
    with pytest.raises(ValueError, match="valid identifier"):
        air.AirTopKSample("l2_0.5", rows=2, cols=64, k=4).validate()


def test_topk_is_exact_against_a_stable_argsort():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(11)
    for rows, cols, k in [(4, 64, 5), (2, 1031, 8), (3, 128, 1)]:
        x = (rng.standard_normal((rows, cols)) * 3).astype(np.float32)
        u = rng.random(rows).astype(np.float32)
        ts = air.AirTopKSample(f"tk_{rows}_{cols}_{k}", rows=rows, cols=cols, k=k)
        _, tv, ti = air.execute_topk_sample(ts, x, u)
        _, ovals, oidx = air.topk_sample_oracle(x, u, k)
        assert (np.array(ti) == oidx).all()
        assert float(np.max(np.abs(np.array(tv) - ovals))) == 0.0


def test_ties_break_to_the_lower_index_where_ONE_thread_sees_them_all():
    """The regression a mutation of mine failed to catch. The tie-break lives in TWO
    places -- the per-thread scan and the tree reduce -- and at cols <= threadgroup
    each thread sees at most one element, so the per-thread half is DEAD CODE and a
    test at that shape proves nothing about it. The ties here sit a full threadgroup
    stride apart so one thread must break them itself."""
    pytest.importorskip("mlx.core")
    tg, cols = 256, 1024
    x = np.zeros((2, cols), np.float32)
    for c in (5, 5 + tg, 5 + 2 * tg):
        x[:, c] = 1.0
    u = np.array([0.1, 0.9], np.float32)
    ts = air.AirTopKSample("tiestride", rows=2, cols=cols, k=3, threadgroup=tg)
    _, _, ti = air.execute_topk_sample(ts, x, u)
    assert np.array(ti)[0].tolist() == [5, 5 + tg, 5 + 2 * tg]


def test_the_sample_is_a_pure_function_of_the_logits_and_u():
    """Randomness as an INPUT is what makes exact grading possible at all: same u
    gives the same answer, a different u actually moves it, and the answer matches an
    independent numpy CDF walk."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(5)
    rows, cols, k = 256, 512, 8
    x = (rng.standard_normal((rows, cols)) * 2.5).astype(np.float32)
    u = rng.random(rows).astype(np.float32)
    ts = air.AirTopKSample("pure", rows=rows, cols=cols, k=k)
    a, _, _ = air.execute_topk_sample(ts, x, u)
    b, _, _ = air.execute_topk_sample(ts, x, u)
    c, _, _ = air.execute_topk_sample(ts, x, rng.random(rows).astype(np.float32))
    och, _, _ = air.topk_sample_oracle(x, u, k)
    assert (np.array(a) == np.array(b)).all()          # reproducible
    assert (np.array(a) != np.array(c)).sum() > rows // 4   # and not trivially constant
    assert (np.array(a) == och).all()                  # matches the independent oracle


def test_the_distribution_is_checked_and_the_check_can_fail():
    """A sampler cannot be graded by equality alone: `the index is one of the top k`
    PASSES FOR A SAMPLER THAT ALWAYS RETURNS THE ARGMAX. So the frequencies are tested
    -- and the negative control is what makes that test evidence rather than ritual."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(99)
    n, cols, k = 4096, 256, 8
    logits = (rng.standard_normal(cols) * 2.0).astype(np.float32)
    X = np.repeat(logits[None, :], n, axis=0)
    u = rng.random(n).astype(np.float32)
    ts = air.AirTopKSample("dist", rows=n, cols=cols, k=k)
    ch, _, ti = air.execute_topk_sample(ts, X, u)
    order = np.array(ti)[0]
    vals = logits[order]
    p = np.exp(vals - vals.max()); p = p / p.sum()
    exp_counts = p * n
    chi2 = lambda idx: float(((np.array([(idx == t).sum() for t in order]) - exp_counts)
                              ** 2 / exp_counts).sum())
    critical = 24.322                       # chi-square upper 0.1% point, df = k-1 = 7
    assert chi2(np.array(ch)) < critical
    assert chi2(np.full(n, order[0])) > critical        # argmax-always must FAIL
    assert chi2(order[np.minimum((u * k).astype(int), k - 1)]) > critical  # uniform too


def test_norm_refuses_what_it_cannot_do():
    mk = lambda **kw: air.AirNorm(**{"name": "n", "rows": 2, "cols": 64, **kw})
    with pytest.raises(ValueError, match="must be 'rms' or 'layer'"):
        mk(mode="batch").validate()
    with pytest.raises(ValueError, match="two_pass"):
        mk(mode="layer", variance="welford").validate()
    # RMSNorm subtracts no mean, so a one-pass variance is not a choice that exists
    with pytest.raises(ValueError, match="no mean"):
        mk(mode="rms", variance="one_pass").validate()
    with pytest.raises(ValueError, match="multiple of 32"):
        mk(threadgroup=100).validate()
    with pytest.raises(ValueError, match="valid identifier"):
        air.AirNorm("n.1", rows=2, cols=64).validate()


def test_norm_matches_a_float64_oracle():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(21)
    for rows, cols, mode in [(4, 128, "rms"), (3, 4099, "layer"), (2, 37, "layer")]:
        x = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
        w = (rng.standard_normal(cols) * 0.5 + 1).astype(np.float32)
        b = (rng.standard_normal(cols) * 0.1).astype(np.float32)
        nm = air.AirNorm(f"n_{mode}_{cols}", rows=rows, cols=cols, mode=mode)
        got = np.array(air.execute_norm(nm, x, w, b))
        assert float(np.max(np.abs(got - air.norm_oracle(x, w, b, mode)))) < 1e-5


def test_the_write_after_read_regression_at_threadgroup_1024():
    """THE BUG THIS PINS WAS REAL AND MINE: phase 2 reused phase 1's scratch with
    nothing ordering its WRITE after every thread's READ of the first result. Exact at
    threadgroup 64 and 256, WRONG BY 0.061 at 1024, because 32 lanes make the read loop
    long enough for a fast thread to overtake a slow one. Found only by running the
    barrier control ACROSS WIDTHS, which is the discipline the top-k receipt argued for
    one block earlier."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(1234)
    rows, cols = 32, 4096
    x = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
    w = np.ones(cols, np.float32); b = np.zeros(cols, np.float32)
    ref = air.norm_oracle(x, w, b, "layer")
    for tgw in (256, 1024):
        nm = air.AirNorm(f"war{tgw}", rows=rows, cols=cols, mode="layer", threadgroup=tgw)
        got = np.array(air.execute_norm(nm, x, w, b))
        assert float(np.max(np.abs(got - ref))) < 1e-5, tgw


def test_the_two_reductions_do_not_share_scratch_anywhere():
    """Structural pin so the hazard cannot come back by someone tidying the slots.
    Softmax and attention carry the SAME shape and never fired -- and a race that does
    not fire is not a passing test, so they are fixed on structure, not on evidence."""
    sm = air.lower_softmax_to_msl(air.AirSoftmax("s", rows=4, cols=128))
    at = air.lower_attention_to_msl(air.AirAttention("a", seq_q=64, seq_k=64, head_dim=32))
    nm = air.lower_norm_to_msl(air.AirNorm("n", rows=4, cols=128, mode="layer"))
    for src in (sm, at, nm):
        assert "u + warp] =" in src        # the second reduction writes its own slots


def test_the_one_pass_variance_really_does_die():
    """Both strategies are kept, and the cheap one is kept as a DEMONSTRATION. A
    strategy nobody has watched fail reads as one that merely lost a style argument."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(7)
    rows, cols = 2, 4096
    w = np.ones(cols, np.float32); b = np.zeros(cols, np.float32)
    for ratio, must_be_close in ((1, True), (4096, False)):
        x = ((rng.standard_normal((rows, cols)) + ratio)).astype(np.float32)
        ref = air.norm_oracle(x, w, b, "layer")
        one = np.array(air.execute_norm(
            air.AirNorm(f"op{ratio}", rows=rows, cols=cols, mode="layer",
                        variance="one_pass"), x, w, b))
        two = np.array(air.execute_norm(
            air.AirNorm(f"tp{ratio}", rows=rows, cols=cols, mode="layer"), x, w, b))
        assert float(np.max(np.abs(two - ref))) < 1e-2      # two-pass survives both
        ok = bool(np.isfinite(one).all()) and float(np.max(np.abs(one - ref))) < 1e-3
        assert ok == must_be_close, (ratio, ok)


def test_stripping_the_first_norm_barrier_breaks_it():
    """The barrier that publishes the row mean is read by EVERY thread, so this control
    is LOUD where the top-k tree-reduce control was nearly silent -- same class of
    control, different evidential value, and the difference is whether the dependency
    is total or incidental."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    rng = np.random.default_rng(3)
    rows, cols = 32, 4096
    x = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
    w = np.ones(cols, np.float32); b = np.zeros(cols, np.float32)
    nm = air.AirNorm("bar", rows=rows, cols=cols, mode="layer")
    bad = air.lower_norm_to_msl(nm).replace(
        "threadgroup_barrier(mem_flags::mem_threadgroup);", "", 1)
    kern = mx.fast.metal_kernel(name="nobar_norm", input_names=nm.input_names(),
                                output_names=["out"], source=bad, ensure_row_contiguous=True)
    g, tg = nm.launch()
    (o,) = kern(inputs=[mx.array(x), mx.array(w), mx.array(b)], grid=g, threadgroup=tg,
                output_shapes=[(rows, cols)], output_dtypes=[mx.float32])
    mx.eval(o)
    ref = air.norm_oracle(x, w, b, "layer")
    assert float(np.max(np.abs(np.array(o) - ref))) > 1e-2


def test_batched_matvec_matches_a_float64_oracle():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(77)
    for B, R, C in [(4, 16, 64), (3, 7, 129), (17, 5, 32)]:
        w = (rng.standard_normal((B, R, C)) * 2).astype(np.float32)
        x = (rng.standard_normal((B, C)) * 2).astype(np.float32)
        bm = air.AirBatchedMatvec(f"bmv{B}_{R}_{C}", batch=B, rows=R, cols=C)
        got = np.array(air.execute_batched_matvec(bm, w, x))
        ref = np.einsum("brc,bc->br", w.astype(np.float64), x.astype(np.float64))
        assert float(np.max(np.abs(got - ref))) < 1e-4, (B, R, C)


def test_a_shared_activation_batch_collapses_to_one_taller_matvec():
    """The distinction the batched kernel exists to keep: a DECODE-time expert batch
    shares its activation, so it is arithmetically a taller matvec and needs no batched
    kernel. Only a batch with a DIFFERENT vector per element needs one."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(5)
    B, R, C = 6, 8, 64
    w = (rng.standard_normal((B, R, C)) * 2).astype(np.float32)
    xs = (rng.standard_normal(C) * 2).astype(np.float32)
    shared = np.repeat(xs[None, :], B, axis=0)
    batched = np.array(air.execute_batched_matvec(
        air.AirBatchedMatvec("shared", batch=B, rows=R, cols=C), w, shared))
    stacked = np.array(air.execute_batched_matvec(
        air.AirBatchedMatvec("stacked", batch=1, rows=B * R, cols=C),
        w.reshape(1, B * R, C), xs[None, :]))
    assert (batched.reshape(-1) == stacked.reshape(-1)).all()   # EXACT, not close
    # and a different vector per element does NOT collapse, or the distinction is empty
    diff = (rng.standard_normal((B, C)) * 2).astype(np.float32)
    other = np.array(air.execute_batched_matvec(
        air.AirBatchedMatvec("differ", batch=B, rows=R, cols=C), w, diff))
    assert not np.allclose(other, batched)


def test_batched_matvec_refuses_what_it_cannot_do():
    mk = lambda **kw: air.AirBatchedMatvec(**{"name": "b", "batch": 2, "rows": 4,
                                              "cols": 8, **kw})
    with pytest.raises(ValueError, match="multiple of 32"):
        mk(threadgroup=48).validate()
    with pytest.raises(ValueError, match="must be positive"):
        mk(batch=0).validate()
    with pytest.raises(ValueError, match="valid identifier"):
        air.AirBatchedMatvec("b.1", batch=2, rows=4, cols=8).validate()
    # one thread owns one output row, so this lowering emits NO barrier -- pinned so it
    # cannot be conflated with the reductions that do
    assert mk().barrier_scopes_emitted() == []


def test_sparse_matvec_matches_a_dense_oracle():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(101)
    for rows, cols, dens in [(64, 128, 0.05), (17, 513, 0.2), (5, 32, 1.0)]:
        a = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
        a[rng.random((rows, cols)) > dens] = 0.0
        x = (rng.standard_normal(cols) * 2).astype(np.float32)
        rp, ci, v = air.to_csr(a)
        sp = air.AirSparseMatvec(f"sp{rows}_{cols}", rows=rows, cols=cols, nnz=len(v))
        got = np.array(air.execute_sparse_matvec(sp, rp, ci, v, x))
        ref = a.astype(np.float64) @ x.astype(np.float64)
        assert float(np.max(np.abs(got - ref))) < 1e-4, (rows, cols, dens)


def test_a_structurally_empty_row_returns_zero_not_garbage():
    """A fully pruned output row is a real case. A kernel that walks a stale range
    returns plausible nonsense for exactly the rows nobody looks at."""
    pytest.importorskip("mlx.core")
    a = np.zeros((4, 16), np.float32)
    a[1, 3] = 2.0; a[3, 15] = -1.5          # rows 0 and 2 are entirely empty
    x = np.arange(16, dtype=np.float32)
    rp, ci, v = air.to_csr(a)
    sp = air.AirSparseMatvec("empty", rows=4, cols=16, nnz=len(v))
    got = np.array(air.execute_sparse_matvec(sp, rp, ci, v, x))
    assert got.tolist() == [0.0, 6.0, 0.0, -22.5]


def test_sparse_matvec_refuses_what_it_cannot_do():
    mk = lambda **kw: air.AirSparseMatvec(**{"name": "s", "rows": 4, "cols": 8,
                                             "nnz": 3, **kw})
    with pytest.raises(ValueError, match="exceeds the"):
        mk(nnz=99).validate()
    with pytest.raises(ValueError, match="must be positive"):
        mk(rows=0).validate()
    with pytest.raises(ValueError, match="valid identifier"):
        air.AirSparseMatvec("s.1", rows=4, cols=8, nnz=3).validate()
    assert mk().barrier_scopes_emitted() == []      # one thread per row, nothing to order
    assert abs(mk().density() - 3 / 32) < 1e-9


def test_causal_conv1d_matches_a_float64_oracle():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(31)
    for C, L, W in [(8, 32, 4), (5, 1, 4), (3, 2, 4), (17, 129, 3), (4, 16, 1)]:
        x = (rng.standard_normal((C, L)) * 2).astype(np.float32)
        w = (rng.standard_normal((C, W)) * 2).astype(np.float32)
        b = (rng.standard_normal(C) * 0.5).astype(np.float32)
        cv = air.AirCausalConv1d(f"cv{C}_{L}_{W}", channels=C, length=L, width=W)
        got = np.array(air.execute_causal_conv1d(cv, x, w, b))
        ref = air.causal_conv1d_oracle(x, w, b, W)
        assert float(np.max(np.abs(got - ref))) < 1e-4, (C, L, W)


def test_a_sequence_shorter_than_the_kernel_is_all_padding():
    """L=1 with W=4 means three of four taps read the zero pad. It is the DECODE shape
    for a Mamba mixer, so getting it wrong breaks generation and nothing else."""
    pytest.importorskip("mlx.core")
    w = np.array([[1.0, 10.0, 100.0, 1000.0]], np.float32)   # one channel, distinct taps
    x = np.array([[7.0]], np.float32)
    b = np.array([0.5], np.float32)
    cv = air.AirCausalConv1d("short", channels=1, length=1, width=4)
    got = np.array(air.execute_causal_conv1d(cv, x, w, b))
    # only the LAST tap sees the single element; the first three see the pad
    assert abs(float(got[0, 0]) - (1000.0 * 7.0 + 0.5)) < 1e-4


def test_peeking_at_the_future_is_caught():
    """Causality is the correctness. In an autoregressive model a conv that reads t+1
    changes no norm -- it just makes teacher forcing mysteriously easy -- so the control
    has to be explicit."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    rng = np.random.default_rng(9)
    C, L, W = 16, 32, 4
    x = (rng.standard_normal((C, L)) * 2).astype(np.float32)
    w = (rng.standard_normal((C, W)) * 2).astype(np.float32)
    b = np.zeros(C, np.float32)
    cv = air.AirCausalConv1d("peek", channels=C, length=L, width=W)
    good = np.array(air.execute_causal_conv1d(cv, x, w, b))
    src = air.lower_causal_conv1d_to_msl(cv)
    bad = src.replace(f"int src = (int)t - {W - 1} + (int)k;", "int src = (int)t + (int)k;")
    assert bad != src
    kern = mx.fast.metal_kernel(name="conv_peek_t", input_names=["x", "w", "bias"],
                                output_names=["out"], source=bad, ensure_row_contiguous=True)
    g, tg = cv.launch()
    (o,) = kern(inputs=[mx.array(x), mx.array(w), mx.array(b)], grid=g, threadgroup=tg,
                output_shapes=[(C, L)], output_dtypes=[mx.float32])
    mx.eval(o)
    assert float(np.max(np.abs(np.array(o) - good))) > 1e-2


def test_causal_conv1d_refuses_what_it_cannot_do():
    mk = lambda **kw: air.AirCausalConv1d(**{"name": "c", "channels": 4, "length": 8,
                                             "width": 4, **kw})
    with pytest.raises(ValueError, match="must be positive"):
        mk(channels=0).validate()
    with pytest.raises(ValueError, match="multiple of 32"):
        mk(threadgroup=48).validate()
    with pytest.raises(ValueError, match="valid identifier"):
        air.AirCausalConv1d("c.1", channels=4, length=8, width=4).validate()
    assert mk().barrier_scopes_emitted() == []      # depthwise, one thread per output


def test_the_tiled_matmul_handles_a_shape_that_is_not_a_multiple_of_the_tile():
    """The simdgroup receipt justified keeping the tiled strategy by saying it 'has NO
    shape constraint -- deleting tiled would have removed the only strategy that handles
    a 60x60 matmul'. That 60x60x60 was WRONG BY 3.84 until the launch rounded up to
    whole threadgroups. The kernel body was always guarded; the LAUNCH under-dispatched."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(5)
    for m, k, n in [(60, 60, 60), (8, 64, 32), (17, 64, 32)]:
        a = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
        b = (rng.standard_normal((k, n)) * 0.5).astype(np.float32)
        mm = air.AirMatmul(f"odd{m}_{k}_{n}", m=m, k=k, n=n, tile=16)
        got = np.array(air.execute_matmul(mm, a, b))
        ref = a.astype(np.float64) @ b.astype(np.float64)
        assert float(np.max(np.abs(got - ref))) < 1e-3, (m, k, n)
    # and the launch really does cover every row now
    mm = air.AirMatmul("cover", m=17, k=64, n=32, tile=16)
    grid, tg = mm.launch()
    assert grid[0] % tg[0] == 0 and grid[1] % tg[1] == 0


def test_a_graph_node_takes_its_geometry_from_the_op():
    """AirGraphNode derived a 1-D grid from an element count for many blocks, which is
    right for elementwise and wrong for everything else -- and wrong in the worst way,
    LOOKING CORRECT at 8x1024 (24 spare threadgroups running out of bounds) and wrong by
    2.62 at 8x64 (2 threadgroups for 8 rows). node_for asks the op instead."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(3)
    for rows, cols in ((8, 64), (8, 1024), (64, 128)):
        x = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
        w = np.ones(cols, np.float32)
        nm = air.AirNorm(f"gn{rows}_{cols}", rows=rows, cols=cols, mode="rms")
        node = air.node_for("normed", nm, ["x", "w"], source=air.lower_norm_to_msl(nm),
                            param_names=["x", "w"])
        assert node.launch() == nm.launch() and node.shape() == (rows, cols)
        g = air.AirGraph(f"gg{rows}_{cols}", nodes=[node], externals=["x", "w"])
        got = np.array(air.execute_graph(g, {"x": x, "w": w})["normed"])
        ref = air.norm_oracle(x, w, np.zeros(cols, np.float32), "rms")
        assert float(np.max(np.abs(got - ref))) < 1e-4, (rows, cols)


def test_a_whole_gated_mlp_block_composes_and_matches_an_f64_oracle():
    """Every AIR receipt grades ONE primitive. This campaign's own law is that LOCAL
    ADEQUACY DOES NOT COMPOSE -- stated about representations, and equally open about
    kernels. Five nodes, one submission, checked end to end."""
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(17)
    T, C, H = 8, 256, 128
    x = (rng.standard_normal((T, C)) * 0.5).astype(np.float32)
    wn = (rng.standard_normal(C) * 0.1 + 1).astype(np.float32)
    wg = (rng.standard_normal((C, H)) * 0.05).astype(np.float32)
    wd = (rng.standard_normal((H, C)) * 0.05).astype(np.float32)
    nm = air.AirNorm("bn", rows=T, cols=C, mode="rms")
    g1 = air.AirMatmul("bg", m=T, k=C, n=H, tile=16)
    g2 = air.AirMatmul("bd", m=T, k=H, n=C, tile=16)
    el = lambda n, body: f"uint i = thread_position_in_grid.x;\nif (i >= {n}u) return;\n{body}"
    nodes = [
        air.node_for("normed", nm, ["x", "wn"], source=air.lower_norm_to_msl(nm),
                     param_names=["x", "w"]),
        air.node_for("gated", g1, ["normed", "wg"], source=air.lower_matmul_to_msl(g1),
                     param_names=["A", "B"], out_name="C"),
        air.AirGraphNode("act", el(T * H, "float v = a[i]; out[i] = v / (1.0f + exp(-v));"),
                         ["gated"], n=T * H, param_names=["a"], out_shape=(T, H)),
        air.node_for("down", g2, ["act", "wd"], source=air.lower_matmul_to_msl(g2),
                     param_names=["A", "B"], out_name="C"),
        air.AirGraphNode("y", el(T * C, "out[i] = a[i] + b[i];"), ["x", "down"],
                         n=T * C, param_names=["a", "b"], out_shape=(T, C)),
    ]
    g = air.AirGraph("mlp", nodes=nodes, externals=["x", "wn", "wg", "wd"])
    got = np.array(air.execute_graph(g, {"x": x, "wn": wn, "wg": wg, "wd": wd})["y"])
    xd = x.astype(np.float64)
    r = 1.0 / np.sqrt((xd * xd).mean(axis=1, keepdims=True) + 1e-5)
    h = (xd * r * wn.astype(np.float64)) @ wg.astype(np.float64)
    ref = xd + (h / (1.0 + np.exp(-h))) @ wd.astype(np.float64)
    assert float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))) < 1e-4
    assert g.serial_depth() == 5 and g.submissions() == 1


def test_the_weight_space_gate_cannot_tell_organs_apart_and_output_space_can():
    """A per-tensor weight gate weights every input direction equally, which is what a
    Gaussian x does. Pinned on constructed tensors that reproduce the MiniLM finding:
    real activations separate two organs that weight-space cosine calls identical."""
    import gravity_native as gn
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((64, 128)) * 0.05).astype(np.float32)
    Wq = gn.quantize_grouped(W, 3, 64)
    # activations with a strong preferred direction, as real ones have
    X = (rng.standard_normal((256, 128)) * 0.1).astype(np.float64)
    X[:, :8] += 4.0
    f = gn.output_space_fidelity(W, Wq, X)
    # THE CONTROL MUST MOVE THE ANSWER, or it is a formality. Directional inputs and
    # directionless ones disagree by far more than float noise on the same pack.
    assert abs(f["cosine_real_inputs"] - f["cosine_gaussian_inputs"]) > 5e-3
    # and where the separation comes from the per-feature marginals rather than from
    # cross-feature correlation, shuffling the columns changes nothing
    assert abs(f["cosine_real_inputs"] - f["cosine_shuffled_inputs"]) < 1e-3


def test_accept_pack_reports_the_output_space_reading_without_letting_it_decide():
    """accept_pack's representation gate is already output-space, but on ONE x. When
    real activations are supplied the wider reading is REPORTED beside the verdict and
    never replaces it -- a caller with no activations must not silently get a weaker
    answer than one who has them."""
    import gravity_native as gn
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((64, 128)) * 0.05).astype(np.float32)
    packed, scale = gn.pack_q4_g64(W)
    x = rng.standard_normal(128).astype(np.float32)
    out = gn.ref_matvec(packed, scale, 128, x)
    X = (rng.standard_normal((32, 128)) * 0.1); X[:, :8] += 4.0
    with_act = gn.accept_pack(W, packed, scale, 128, out, x, activations=X)
    without = gn.accept_pack(W, packed, scale, 128, out, x)
    assert without["output_space"] is None
    assert with_act["accepted"] == without["accepted"]      # it does not decide
    assert set(with_act["output_space"]) >= {
        "cosine_real_inputs", "cosine_gaussian_inputs", "ok",
        "agrees_with_single_x_gate"}


def test_one_input_vector_is_a_noisier_gate_than_many():
    """A one-row gate flips its verdict on 12.6% of real packs. Pinned structurally:
    the spread of single-row cosines straddles the aggregate, so the verdict a
    single-x gate returns depends on which row it happened to get."""
    import gravity_native as gn
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((32, 64)) * 0.05).astype(np.float32)
    Wq = gn.quantize_grouped(W, 3, 64)
    X = np.abs(rng.standard_normal((64, 64))) * rng.choice([0.1, 3.0], (64, 1))
    singles = [gn.output_space_fidelity(W, Wq, X[i:i + 1])["cosine_real_inputs"]
               for i in range(X.shape[0])]
    full = gn.output_space_fidelity(W, Wq, X)["cosine_real_inputs"]
    assert min(singles) < full < max(singles)
    assert max(singles) - min(singles) > 10 * abs(full - float(np.mean(singles)))


def test_a_verdict_can_be_precision_safe_and_still_undecided():
    """safe_at_float32 answers a question about ARITHMETIC. When the gate's inputs are
    a SAMPLE of activations the wider source of disagreement is WHICH ROWS -- measured
    at 28.6x the float32 bar, with three of six unstable packs passing safe_at_float32
    and one of them flipping on 8.3% of 176-row resamples."""
    import gravity_native as gn
    near = gn.near_threshold_headroom(cos=0.99 + 0.0008, ratio=1.0, kernel_err=0.0,
                                      kernel_tol=1.0, min_cosine=0.99,
                                      magnitude_band=(0.9, 1.1))
    assert near["safe_at_float32"] is True              # 0.0008 clears 100 x 2.85e-06
    assert near["decided_under_resampling"] is False    # and is inside the 0.002 band
    far = gn.near_threshold_headroom(cos=0.9999, ratio=1.0, kernel_err=0.0,
                                     kernel_tol=1.0, min_cosine=0.99,
                                     magnitude_band=(0.9, 1.1))
    assert far["safe_at_float32"] and far["decided_under_resampling"]
    # AND THE BAR CANNOT BE SET AT THE WIDEST UNSTABLE PACK: an accepted pack's
    # cosine headroom is capped at 1 - min_cosine, so a 0.01 bar would call every
    # acceptance undecided. Pinned so nobody "tightens" it into vacuity.
    assert gn.near_threshold_headroom(cos=1.0, ratio=1.0, kernel_err=0.0,
                                      kernel_tol=1.0, min_cosine=0.99,
                                      magnitude_band=(0.9, 1.1)
                                      )["min_distance_to_a_threshold"] <= 0.01 + 1e-12


# --------------------------------------------------------------------------
# ACCELERATOR_BARRIER_SCOPES.json named this against itself: "NO AIR PROGRAM
# CURRENTLY DECLARES A SIMDGROUP BARRIER AND EXECUTES -- what closed is the
# INSTRUCTION and the REFUSAL'S HONESTY, not a new executable program shape."
# --------------------------------------------------------------------------

def test_a_program_DECLARES_a_simdgroup_barrier_and_the_msl_carries_it():
    rr = air.AirSimdRowReduce("t", rows=4, cols=100)
    assert rr.barrier_scopes_emitted() == ["SIMDGROUP"]
    src = air.lower_simd_row_reduce_to_msl(rr)
    assert air.barrier_msl("SIMDGROUP") in src, \
        "the declared scope must reach the source through barrier_msl, the ONE place a " \
        "lowering asks for a barrier, or the rule and the instruction can drift apart"


def test_a_WIDER_threadgroup_declaring_SIMDGROUP_scope_is_REFUSED():
    """A simdgroup barrier orders 32 lanes and nothing wider. The barrier-scopes
    receipt measured a cross-simdgroup exchange WRONG BY 9.854 with simdgroup_barrier,
    the same error as with no fence -- so emitting it over a wider group would be an
    instruction that cannot order what the program needs ordered."""
    for tg in (64, 256, 1024):
        with pytest.raises(ValueError) as e:
            air.AirSimdRowReduce("bad", rows=2, cols=64, threadgroup=tg).validate()
        assert "9.854" in str(e.value), "the refusal must cite the measurement"


def test_the_refusal_is_about_SCOPE_and_not_about_wide_threadgroups_in_general():
    """AirSoftmax runs happily at 256 with THREADGROUP scope. If this ever fails the
    rule above has been over-generalised into 'wide threadgroups are bad'."""
    sm = air.AirSoftmax("wide", rows=2, cols=64, threadgroup=256)
    sm.validate()
    assert sm.barrier_scopes_emitted() == ["THREADGROUP", "THREADGROUP"]


def test_the_simdgroup_program_EXECUTES_and_is_correct_off_a_multiple_of_32():
    pytest.importorskip("mlx.core")
    import numpy as np
    rr = air.AirSimdRowReduce("exec", rows=3, cols=33)
    x = np.random.RandomState(1).randn(3, 33).astype(np.float32)
    got = np.array(air.execute_simd_row_reduce(rr, x), dtype=np.float64)
    want = x.astype(np.float64).sum(axis=1)
    rel = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-30))
    assert rel < 1e-5, rel


def test_the_correctness_check_CAN_FAIL():
    """Without this the test above is a check that cannot fail, which this program has
    sealed five times. A wrong stride must be caught by the same oracle."""
    pytest.importorskip("mlx.core")
    import numpy as np, mlx.core as mx
    rr = air.AirSimdRowReduce("ctl", rows=4, cols=777)
    src = air.lower_simd_row_reduce_to_msl(rr).replace("c += 32u", "c += 31u")
    x = np.random.RandomState(2).randn(4, 777).astype(np.float32)
    k = mx.fast.metal_kernel(name="air_simd_ctl", input_names=["x"],
                             output_names=["out"], source=src, ensure_row_contiguous=True)
    (o,) = k(inputs=[mx.array(x)], grid=(4 * 32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(4,)], output_dtypes=[mx.float32])
    mx.eval(o)
    want = x.astype(np.float64).sum(axis=1)
    rel = np.max(np.abs(np.array(o, dtype=np.float64) - want) / np.maximum(np.abs(want), 1e-30))
    assert rel > 1e-4, "the broken stride was not caught, so the oracle proves nothing"
