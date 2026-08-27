"""THE THREADGROUP WIDTH AGAINST THE BLOCK ROTATION.

ACCELERATOR_THE_PENALTY_IS_A_STEP named this axis against itself -- "the step
sits between ONE threadgroup request and TWO, so the axis that would price it is
the THREADGROUP WIDTH itself: at tg=64 a threadgroup IS one simdgroup and the
k=32 arm would have nothing to split, while at tg=256 there are four."

BOTH HALVES OF THAT SENTENCE ARE WRONG AND THE ARITHMETIC SAYS SO BEFORE ANY ARM
RUNS, which is why these tests exist:

  1. A simdgroup on this chip is 32 lanes, measured six times in this program.
     tg=64 is TWO simdgroups and tg=256 is EIGHT, not one and four.

  2. The rotation offset is lane/k where lane = gid %% TPR, so the block
     structure is over THE 64 LANES OF ONE ROW and is INDEPENDENT OF TG.
     Raising tg packs more ROWS into a threadgroup; it gives k=32 nothing more
     or less to split. Running the sweep as named and reporting "the ratio is
     flat" would be a check whose answer was fixed before the run.

What tg DOES vary is the THREADGROUP's request count, because rows sit at
unrelated base addresses and never merge: a threadgroup holding R rows issues at
least R disjoint runs AT EVERY k, the control included. That is the well-posed
form of the question and it is what ACCELERATOR_THE_WIDTH_IS_THE_ROW measured.
"""
import pytest

import gravity_native as G

ROWS, COLS, TPR = 17408, 16384, 64
GROUPS = COLS // G.GROUP
SIMD = 32  # the hardware width, not a tuning choice


def cnt(lane):
    return (GROUPS - lane + TPR - 1) // TPR


def blk(lane, k):
    """The group sequence lane `lane` walks under a rotation by blocks of k."""
    n = cnt(lane)
    return [lane + ((i + lane // k) % n) * TPR for i in range(n)]


def strided(lane):
    return list(range(lane, GROUPS, TPR))


def runs_over(lanes, k):
    """Contiguous runs in what `lanes` request at ITERATION 0 -- the instant."""
    g = [blk(l, k)[0] for l in lanes]
    return 1 + sum(1 for i in range(1, len(g)) if g[i] != g[i - 1] + 1)


def rows_per_threadgroup(tg):
    return tg // TPR


def threadgroup_runs(tg, k):
    return rows_per_threadgroup(tg) * runs_over(range(TPR), k)


# ---------------------------------------------------------------- the arithmetic


def test_tg64_IS_TWO_SIMDGROUPS_NOT_ONE():
    """The sentence that named this axis said a tg=64 threadgroup IS one
    simdgroup. It is two, and tg=256 is eight rather than four. Pinned because
    the whole 'the k=32 arm would have nothing to split' argument rested on it."""
    assert 64 // SIMD == 2
    assert 256 // SIMD == 8
    assert TPR // SIMD == 2, "a row at tpr=64 spans TWO simdgroups"


@pytest.mark.parametrize("tg", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64])
def test_the_PER_ROW_run_count_DOES_NOT_MOVE_WITH_TG(tg, k):
    """64/k at every threadgroup width. This is why the sweep as named could not
    vary what it wanted to vary: the rotation is over lanes, and a row always
    has TPR of them however many rows share a threadgroup."""
    assert runs_over(range(TPR), k) == TPR // k
    assert runs_over(range(SIMD), k) == max(1, SIMD // k)


def test_TG_VARIES_THE_THREADGROUP_REQUEST_AND_THAT_IS_THE_WELL_POSED_AXIS():
    """The control alone sweeps one threadgroup request to eight. Rows sit at
    unrelated base addresses so they never merge into one run, which is what
    makes the control -- an arm whose per-row request is a single run at every
    tg -- the instrument for the threadgroup hypothesis."""
    assert [threadgroup_runs(tg, 64) for tg in (64, 128, 256, 512)] == [1, 2, 4, 8]
    assert [threadgroup_runs(tg, 32) for tg in (64, 128, 256, 512)] == [2, 4, 8, 16]


def test_THE_TG_AXIS_IS_NOT_DEGENERATE():
    """Anti-vacuity for the sweep itself. If every tg produced the same
    threadgroup request count the comparison would be four copies of one arm and
    a flat result would mean nothing -- the 0-of-0-reads-like-0-of-many shape
    this program has sealed repeatedly."""
    counts = {threadgroup_runs(tg, 64) for tg in (64, 128, 256, 512)}
    assert len(counts) == 4, f"the tg axis varied nothing: {counts}"


# ------------------------------------------------------------------- the arms


@pytest.mark.parametrize("k", [32, 64])
def test_the_per_lane_SET_is_identical_LANE_BY_LANE(k):
    """Every arm is a REORDERING, so the wrong-by-construction anti-vacuity
    control every earlier probe in this family used is unavailable. Its
    replacement is this: the same GROUPS, not merely the same count."""
    for lane in range(TPR):
        assert sorted(blk(lane, k)) == strided(lane), f"lane {lane} set moved at k={k}"


@pytest.mark.parametrize("k,expected", [(32, True), (64, False)])
def test_the_ORDER_differs_at_k32_and_NOT_at_the_control(k, expected):
    """Asserted in BOTH directions. An arm that silently reduced to the shipped
    order would TIE, and the tie would read as a finding; a control that did not
    reduce to it would not be a control."""
    differs = any(blk(lane, k) != strided(lane) for lane in range(TPR))
    assert differs is expected


@pytest.mark.parametrize("tg", [64, 128, 256, 512])
def test_the_TG_ARMS_DIFFER_ONLY_IN_THE_THREADGROUP_CONSTANT(tg):
    """A tg-to-tg comparison is only a comparison of launch and addresses if the
    kernel body is otherwise identical. The reduction tail sizes its scratch by
    TG, so that one number is allowed to move and nothing else is."""
    a = G.source_operand_probe(ROWS, COLS, "rotblk64", TPR, 128)
    b = G.source_operand_probe(ROWS, COLS, "rotblk64", TPR, tg)
    assert a.replace("part[128u]", "part[Nu]") == b.replace(f"part[{tg}u]", "part[Nu]")


def test_the_CONTROL_PAYS_THE_SAME_ARITHMETIC_AS_THE_ARM():
    """rotblk64 is lane/64 = 0 for every lane, so it computes the SHIPPED ORDER
    while still paying the division and the modulo. Timing the arm against the
    bare shipped kernel would confound the order with two instructions."""
    ctl = G.source_operand_probe(ROWS, COLS, "rotblk64", TPR, 128)
    arm = G.source_operand_probe(ROWS, COLS, "rotblk32", TPR, 128)
    assert "/ 64u" in ctl and "/ 32u" in arm
    assert ctl.count("%") == arm.count("%"), "the arms differ in more than the divisor"


# -------------------------------------------------------------- the geometry gate


def test_a_THREADGROUP_THAT_IS_NOT_A_WHOLE_NUMBER_OF_ROWS_IS_REFUSED():
    """REACHABLE, unlike the row-stride alignment guard an earlier block had to
    name as unreachable: tg=32 is a legal Metal threadgroup and a legal divisor
    of this grid. It is refused because the serial tail has lane 0 read TPR
    slots from its own threadgroup -- at tg=32 half the row's partials live in
    ANOTHER threadgroup and the read runs off a 32-entry array, which is silent
    garbage rather than an error."""
    with pytest.raises(ValueError, match="whole number of rows"):
        G.source_operand_probe(ROWS, COLS, "rotblk32", TPR, 32)


def test_a_GRID_THAT_DOES_NOT_DIVIDE_IS_REFUSED_FOR_THE_BARRIER_REASON():
    with pytest.raises(ValueError, match="skip the barrier"):
        G.source_operand_probe(17407, COLS, "rotblk32", TPR, 128)


def test_DRIFT_the_probe_raises_if_the_loop_it_rewrites_is_gone(monkeypatch):
    """source_tpr builds from the PRE-CONCATENATED NATIVE_MATVEC_TPR, which is
    the constant an earlier drift test of mine aimed at the wrong half of."""
    monkeypatch.setattr(G, "NATIVE_MATVEC_TPR",
                        G.NATIVE_MATVEC_TPR.replace("for (uint g = lane;", "for (uint g = 0u;"))
    with pytest.raises(AssertionError):
        G.source_operand_probe(ROWS, COLS, "rotblk32", TPR, 128)


# ---------------------------------------------------------------- and it EXECUTES


def test_THE_ARMS_AGREE_WITH_A_FLOAT64_ORACLE_AT_TWO_WIDTHS():
    """The suite that pinned only its own Python arithmetic let a mutation
    survive two blocks ago. This one runs the generated Metal: a reordering must
    agree with the oracle at EVERY tg, and it must agree with itself across tg
    since nothing about the computation depends on how rows are packed."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    rows, cols, tpr = 256, 4096, 64
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    x = rng.standard_normal(cols).astype(np.float32)
    packed, scale = G.pack_q4_g64(w)
    oracle = G.dequantize(packed, scale, cols).astype(np.float64) @ x.astype(np.float64)
    n = np.linalg.norm(oracle)
    dp, ds, dx = mx.array(packed), mx.array(scale), mx.array(x)
    got = {}
    for tg in (64, 256):
        for probe in ("rotblk64", "rotblk32"):
            src = G.source_operand_probe(rows, cols, probe, tpr, tg)
            k = mx.fast.metal_kernel(name=f"twt_{probe}_{tg}",
                                     input_names=["packed", "scales", "x"],
                                     output_names=["out"], source=src,
                                     ensure_row_contiguous=True)
            (o,) = k(inputs=[dp, ds, dx], grid=(rows * tpr, 1, 1),
                     threadgroup=(tg, 1, 1), output_shapes=[(rows,)],
                     output_dtypes=[mx.float32])
            v = np.array(o, dtype=np.float64)
            rel = float(np.linalg.norm(v - oracle) / n)
            assert rel < 1e-5, (probe, tg, rel)
            assert len(np.unique(v)) > rows // 2, "degenerate output"
            got[(probe, tg)] = v
    for probe in ("rotblk64", "rotblk32"):
        assert np.array_equal(got[(probe, 64)], got[(probe, 256)]), (
            f"{probe} changed its ANSWER with the threadgroup width; packing rows "
            "into a threadgroup must not touch the arithmetic")


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64])
def test_THE_EMITTED_DIVISOR_IS_THE_REQUESTED_ONE(k):
    """The ONLY thing that can pin the block width. Every rotation is a
    BIJECTION, so a kernel emitting the wrong divisor still computes the right
    answer and the executing test above cannot see it -- watched directly: a
    mutation replacing lane/k with lane survived every set, order and oracle
    check in this file and was caught here alone."""
    src = G.source_operand_probe(ROWS, COLS, f"rotblk{k}", TPR, 128)
    assert f"(lane / {k}u)" in src, f"rotblk{k} did not emit its own divisor"
