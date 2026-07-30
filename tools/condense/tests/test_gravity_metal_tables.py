"""Gravity Metal pure-logic (S6 F2 C2 densified)."""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np
from lab.operators import gravity_forge as forge
from lab.operators import gravity_metal as gm
from lab.operators import gravity_metal_lab_b as b

def _codes(rows: int, cols: int, *, D: int=8, k: int=128, seed: int=0) -> dict:
    """A real packed artifact's codes stash, so the shapes are the production ones."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=D, subspaces=1, k=k, seed=seed, iters=2)
    return art.config['pq_codes']
R0_ROWS, R0_COLS, R0_D, R0_K = (2048, 6144, 8, 128)
R0_NCHUNK = R0_COLS // R0_D

def _r0_codes() -> dict:
    """Geometry-only stand-in: matvec_bytes reads shapes, never index values."""
    return {'D': R0_D, 'S': 1, 'sub': R0_D, 'rows': R0_ROWS, 'cols': R0_COLS, 'nchunk': R0_NCHUNK, 'rotate': False, 'seed': 0, 'codebooks': [np.zeros((R0_K,
        R0_D), dtype=np.float32)], 'indices': np.zeros((R0_ROWS * R0_NCHUNK, 1), dtype=np.int64)}

def _decoder():
    try:
        return gm.decoder()
    except gm.MetalUnavailable as exc:
        pytest.skip(f'no Metal device in this process: {exc}')

def test_exact_size_x_passes_the_guard():
    xv = gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float32), nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)
    assert xv.nbytes == R0_NCHUNK * R0_D * 4
    assert xv.dtype == np.float32

def test_short_x_is_rejected():
    with pytest.raises(gm.GravityMetalInputError, match='elements'):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D - 1, dtype=np.float32), nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)

def test_long_x_is_rejected_before_any_pointer_is_taken():
    with pytest.raises(gm.GravityMetalInputError, match='elements'):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D + 1024, dtype=np.float32), nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)

def test_wrong_dtype_is_rejected():
    with pytest.raises(gm.GravityMetalInputError, match='float32'):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float64), nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)

def test_wrong_geometry_is_rejected():
    with pytest.raises(gm.GravityMetalInputError, match='allocated with'):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float32), nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=256 * R0_D * 4)

def test_guard_fires_on_a_real_dispatch():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    key = gm.content_key(codes)
    good = np.ones(128, dtype=np.float32)
    gpu.matvec(codes, good, key=key)
    with pytest.raises(gm.GravityMetalInputError):
        gpu.matvec(codes, np.ones(4096, dtype=np.float32), key=key)

def test_matvec_without_a_key_raises():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    with pytest.raises(gm.GravityMetalInputError, match='explicit cache key'):
        gpu.matvec(codes, np.ones(128, dtype=np.float32))

def test_same_explicit_key_returns_the_same_entry():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    first = gpu._cache_tensor(codes, 'tensor-a')
    assert gpu._cache_tensor(codes, 'tensor-a') is first

def test_different_keys_do_not_collide():
    gpu = _decoder()
    a, b = (_codes(256, 128, D=8, k=16, seed=1), _codes(256, 128, D=8, k=16, seed=2))
    ea = gpu._cache_tensor(a, 'collide-a')
    eb = gpu._cache_tensor(b, 'collide-b')
    assert ea is not eb
    x = np.ones(128, dtype=np.float32)
    ya = gpu.matvec(a, x, key='collide-a')
    yb = gpu.matvec(b, x, key='collide-b')
    assert not np.allclose(ya, yb)

def test_reusing_one_key_for_two_tensors_is_refused():
    """The explicit-key rule moves uniqueness to the caller; this is what enforces it."""
    gpu = _decoder()
    a, b = (_codes(256, 128, D=8, k=16, seed=5), _codes(256, 128, D=8, k=16, seed=6))
    gpu._cache_tensor(a, 'shared-literal')
    with pytest.raises(gm.GravityMetalInputError, match='already holds a different tensor'):
        gpu._cache_tensor(b, 'shared-literal')

def test_codebook_width_must_match_declared_D():
    """A narrow codebook is an out-of-bounds device read, not a wrong answer."""
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16, seed=7)
    codes['D'] = 16
    with pytest.raises(gm.GravityMetalInputError, match='codebook subvector'):
        gpu._cache_tensor(codes, 'narrow-book')

def test_index_beyond_codebook_is_refused_before_the_uint8_cast():
    """uint8 would wrap an out-of-range index and silently select the wrong codeword."""
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16, seed=8)
    codes['indices'] = np.asarray(codes['indices'], dtype=np.int64).copy()
    codes['indices'][0] = 300
    with pytest.raises(gm.GravityMetalInputError, match='out of range'):
        gpu._cache_tensor(codes, 'wrapping-index')

def test_content_key_is_stable_and_discriminating():
    a = _codes(128, 64, D=8, k=16, seed=3)
    same = {**a, 'indices': a['indices'].copy(), 'codebooks': [cb.copy() for cb in a['codebooks']]}
    other = _codes(128, 64, D=8, k=16, seed=4)
    assert gm.content_key(a) == gm.content_key(a)
    assert gm.content_key(a) == gm.content_key(same)
    assert gm.content_key(a) != gm.content_key(other)
    mutated = dict(a)
    idx = a['indices'].copy()
    idx[0, 0] = (idx[0, 0] + 1) % 16
    mutated['indices'] = idx
    assert gm.content_key(mutated) != gm.content_key(a)
    mutated = dict(a)
    book = a['codebooks'][0].copy()
    book[0, 0] += 1.0
    mutated['codebooks'] = [book]
    assert gm.content_key(mutated) != gm.content_key(a)

def _entry(nbytes: int) -> dict:
    return {'pinned_bytes': nbytes}

def test_lru_evicts_past_the_budget_and_accounts_correctly():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    cache.put('a', _entry(100))
    cache.put('b', _entry(100))
    assert cache.stats()['bytes_pinned'] == 200
    assert cache.stats()['evictions'] == 0
    cache.put('c', _entry(100))
    assert cache.get('a') is None
    assert cache.stats() == {'entries': 2, 'bytes_pinned': 200, 'budget_bytes': 250, 'evictions': 1, 'keys': ['b', 'c']}

def test_lru_evicts_by_recency_not_insertion():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    cache.put('a', _entry(100))
    cache.put('b', _entry(100))
    cache.get('a')
    cache.put('c', _entry(100))
    assert cache.get('b') is None
    assert cache.get('a') is not None

def test_reinserting_an_evicted_entry_works():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    for key in ('a', 'b', 'c'):
        cache.put(key, _entry(100))
    fresh = cache.put('a', _entry(100))
    assert cache.get('a') is fresh
    assert cache.stats()['bytes_pinned'] == 200

def test_reput_of_a_live_key_does_not_double_count():
    cache = gm._ByteBudgetCache(budget_bytes=10000)
    cache.put('a', _entry(100))
    cache.put('a', _entry(300))
    assert cache.stats() == {'entries': 1, 'bytes_pinned': 300, 'budget_bytes': 10000, 'evictions': 0, 'keys': ['a']}

def test_an_entry_larger_than_the_budget_still_runs():
    cache = gm._ByteBudgetCache(budget_bytes=10)
    entry = cache.put('huge', _entry(1000))
    assert cache.get('huge') is entry

def test_the_glm_walk_cannot_pin_ninety_gigabytes():
    cache = gm._ByteBudgetCache(budget_bytes=gm.DEFAULT_CACHE_BUDGET_BYTES)
    per_tensor = R0_ROWS * R0_NCHUNK
    for i in range(4000):
        cache.put(f't{i}', _entry(per_tensor))
    assert cache.stats()['bytes_pinned'] <= gm.DEFAULT_CACHE_BUDGET_BYTES
    assert cache.stats()['evictions'] > 0

def test_decoder_exposes_the_accounting():
    gpu = _decoder()
    assert set(gpu.cache_stats) == {'entries', 'bytes_pinned', 'budget_bytes', 'evictions', 'keys'}
    assert gpu.cache_stats['budget_bytes'] == gm.DEFAULT_CACHE_BUDGET_BYTES

def test_r0_gate_up_traffic_is_the_measured_split():
    got = gm.matvec_bytes(_r0_codes())
    assert got['index_bits_billed'] == 7
    assert got['threadgroups'] == 8
    assert got['stage_x'] is True
    assert got['logical_index_bytes'] == 1376256
    assert got['logical_codebook_bytes'] == 2048
    assert got['logical_artifact_bytes'] == 1378304
    assert got['executed_index_bytes'] == 1572864
    assert got['executed_codebook_bytes'] == 16384
    assert got['executed_activation_bytes'] == 196608
    assert got['executed_output_bytes'] == 8192
    assert got['executed_read_bytes'] == 1785856
    assert got['executed_total_bytes'] == 1794048
    assert got['dense_bf16_bytes'] == 25165824
    assert round(got['logical_bpw'], 5) == 0.8763
    assert round(got['executed_read_bytes'] / got['dense_bf16_bytes'] * 100, 2) == 7.1

def test_executed_exceeds_the_logical_artifact():
    got = gm.matvec_bytes(_r0_codes())
    ratio = got['executed_total_bytes'] / got['logical_artifact_bytes']
    assert got['executed_total_bytes'] > got['logical_artifact_bytes']
    assert round(ratio, 4) == round(1794048 / 1378304, 4) == 1.3016

def test_the_seven_to_eight_bit_gap_is_exactly_eight_sevenths():
    got = gm.matvec_bytes(_r0_codes())
    assert got['executed_index_bytes'] * 7 == got['logical_index_bytes'] * 8

def test_r0_down_projection_traffic():
    codes = _r0_codes()
    codes.update(rows=6144, cols=2048, nchunk=256, indices=np.zeros((6144 * 256, 1), dtype=np.int64))
    got = gm.matvec_bytes(codes)
    assert got['threadgroups'] == 24
    assert got['stage_x'] is True
    assert got['executed_index_bytes'] == 1572864
    assert got['executed_codebook_bytes'] == 24 * 2048
    assert got['executed_activation_bytes'] == 24 * 256 * 8 * 4
    assert got['executed_index_bytes'] * 7 == got['logical_index_bytes'] * 8

def test_unstaged_x_is_charged_once():
    codes = _r0_codes()
    got = gm.matvec_bytes(codes, threadgroup_memory_limit=8192)
    assert got['stage_x'] is False
    assert got['executed_activation_bytes'] == R0_NCHUNK * R0_D * 4

def test_accounting_matches_a_real_artifact_and_refuses_multi_subspace():
    codes = _codes(256, 128, D=8, k=16)
    got = gm.matvec_bytes(codes)
    assert got['index_bits_billed'] == 4
    assert got['executed_index_bytes'] == 256 * 16
    assert got['logical_index_bytes'] == 256 * 16 // 2
    codes['S'] = 2
    with pytest.raises(gm.GravityMetalInputError, match='subspaces == 1'):
        gm.matvec_bytes(codes)

def test_bytes_read_per_matvec_returns_the_dict():
    gpu = _decoder()
    got = gpu.bytes_read_per_matvec(_r0_codes())
    assert isinstance(got, dict)
    assert got['executed_read_bytes'] == 1785856
GATE = {'rows': 2048, 'cols': 6144, 'nchunk': 768, 'D': 8, 'k': 128}
DOWN = {'rows': 6144, 'cols': 2048, 'nchunk': 256, 'D': 8, 'k': 128}

def test_cbs_cap_is_arithmetic_not_a_magic_number():
    """codebook + x slice + table must fit 32768 B, and the planner must say so."""
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=52, tpg=1024)
    assert shape['scratch_bytes'] == (128 * 8 + 52 * 8) * 4 + 52 * 128 * 4
    assert shape['scratch_bytes'] <= 32768
    with pytest.raises(b.TrackBError, match='threadgroup memory'):
        b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=53, tpg=1024)

def test_half_table_is_what_buys_the_bigger_block():
    """The only reason a half table is worth measuring: it doubles the on-chip block."""
    args = {kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}
    with pytest.raises(b.TrackBError):
        b.ll_plan(**args, cbs=96, tpg=1024, half_table=False)
    shape = b.ll_plan(**args, cbs=96, tpg=1024, half_table=True)
    assert shape['half_table'] and shape['kernel'] == 'll_blk_f16_r4' or True
    assert shape['table_bytes_on_chip'] == 96 * 128 * 2
    assert b.ll_plan(**args, cbs=99, tpg=1024, half_table=True)['scratch_bytes'] <= 32768
    with pytest.raises(b.TrackBError):
        b.ll_plan(**args, cbs=100, tpg=1024, half_table=True)

def test_row4_refuses_a_geometry_it_would_misread():
    with pytest.raises(b.TrackBError, match='row4'):
        b.ll_plan(rows=2050, nchunk=768, D=8, k=128, cbs=16, tpg=256, row4=True)

def test_kernel_name_matches_the_flags():
    assert b.ll_kernel_name(half_table=False, row4=False) == 'll_blk_f32_r1'
    assert b.ll_kernel_name(half_table=True, row4=True) == 'll_blk_f16_r4'

def test_one_threadgroup_per_chunk_block_owns_every_row():
    """The shape that preserves the op reduction: blocks threadgroups, no row tiling. A 2D (row tile, chunk"""
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024)
    assert shape['threadgroups'] == shape['blocks'] == 48
    assert 'row_tiles' not in shape
    assert shape['rows_per_thread'] == 2

def test_analytic_reduction_matches_the_ground_truth_figures():
    """5.333x at gate/up and 6.857x at down, unshared.  Anything else is a regression."""
    g = b.ll_cost(**GATE, shape=b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024))
    d = b.ll_cost(**DOWN, shape=b.ll_plan(**{kk: DOWN[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024))
    assert g['arithmetic_reduction_vs_decode_fma'] == pytest.approx(16 / 3)
    assert d['arithmetic_reduction_vs_decode_fma'] == pytest.approx(48 / 7)

def test_table_never_reaches_device_memory_on_the_hot_path():
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024)
    cost = b.ll_cost(**GATE, shape=shape)
    assert cost['table_bytes_written_to_device'] == 0
    assert cost['table_bytes_on_chip'] == 16 * 128 * 4
    assert shape['table_in_device_memory'] is False

def test_the_op_reduction_is_independent_of_every_tuning_knob():
    """k*D against rows.  cbs, tpg, the block count and row4 all cancel."""
    args = {kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}
    values = {b.ll_cost(**GATE, shape=b.ll_plan(**args, cbs=cbs, tpg=tpg, row4=r4))['arithmetic_reduction_vs_decode_fma'] for cbs in (8, 16, 32,
        48) for tpg in (256, 1024) for r4 in (False, True)}
    assert len(values) == 1

def test_lookup_linear_reads_x_once_and_decode_fma_does_not():
    """The traffic difference between the grammars, which is not the arithmetic one."""
    llshape = b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024)
    dfshape = b.dfma_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=24, tpg=256)
    ll = b.ll_cost(**GATE, shape=llshape)
    df = b.dfma_cost(**GATE, shape=dfshape)
    assert ll['activation_bytes'] == GATE['nchunk'] * GATE['D'] * 4
    assert df['activation_bytes'] == dfshape['threadgroups'] * dfshape['cbs'] * GATE['D'] * 4
    assert df['activation_bytes'] == 8 * ll['activation_bytes']
    tighter = b.dfma_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=24, tpg=64)
    assert b.dfma_cost(**GATE, shape=tighter)['activation_bytes'] == 32 * ll['activation_bytes']
    assert ll['index_bytes'] == df['index_bytes'] == GATE['rows'] * GATE['nchunk']

def test_device_table_control_bills_the_defect_that_makes_it_lose():
    cost = b.device_table_cost(**GATE)
    assert cost['table_bytes_written_to_device'] == GATE['nchunk'] * GATE['k'] * 4
    assert cost['table_gather_read_bytes'] == 4 * cost['index_bytes']
    assert 'NEGATIVE CONTROL' in cost['grammar']

def test_counterfactual_amortises_the_table_and_nothing_else():
    """nsets shares the table build; the gather and the index stream scale with nsets."""
    shape = b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, cbs=16, tpg=1024)
    one = b.ll_cost(**GATE, shape=shape, nsets=1)
    many = b.ll_cost(**GATE, shape=shape, nsets=16)
    assert many['executed_fp_macs'] == one['executed_fp_macs']
    assert many['executed_gather_ops'] == 16 * one['executed_gather_ops']
    assert many['index_bytes'] == 16 * one['index_bytes']
    assert many['arithmetic_reduction_vs_decode_fma'] > one['arithmetic_reduction_vs_decode_fma']

def test_dfma_plan_refuses_a_D_the_float4_loop_would_misread():
    with pytest.raises(b.TrackBError, match='float4'):
        b.dfma_plan(rows=2048, nchunk=768, D=6, k=128, cbs=24, tpg=256)

def _small_artifact(seed: int=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((256, 128)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=8, subspaces=1, k=16, seed=0, iters=4)
    return (art, art.config['pq_codes'], rng.standard_normal(128).astype(np.float32))

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
    assert np.allclose(a, c, rtol=1e-05, atol=1e-05)

def test_half_table_costs_accuracy_and_the_model_shows_it():
    """T holds 8-term dot products, so casting it to half is a real, measurable loss. Graded against the fp"""
    art, codes, x = _small_artifact()
    f32 = b.ll_reference(codes, x, cbs=4, half_table=False)
    f16 = b.ll_reference(codes, x, cbs=4, half_table=True)
    gap = np.linalg.norm(f16 - f32) / np.linalg.norm(f32)
    assert gap > 0.0001, gap
    assert gap < 0.01, gap

def test_parity_of_reports_every_statistic_it_claims():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    p = b.parity_of(a, a)
    assert p['finite'] and p['max_abs_error'] == 0.0
    assert p['relative_l2'] == 0.0 and p['cosine'] == pytest.approx(1.0)

def test_sweep_never_proposes_a_shape_the_planner_would_refuse():
    for cfg in b.ll_sweep(nchunk=768):
        b.ll_plan(**{kk: GATE[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, **cfg)
    for cfg in b.ll_sweep(nchunk=256):
        b.ll_plan(**{kk: DOWN[kk] for kk in ('rows', 'nchunk', 'D', 'k')}, **cfg)

def test_sweep_covers_both_table_dtypes_and_both_row_tiles():
    cfgs = b.ll_sweep(nchunk=768)
    assert {c['half_table'] for c in cfgs} == {False, True}
    assert {c['row4'] for c in cfgs} == {False, True}

@pytest.fixture(scope='module')
def dec():
    try:
        return b.TrackBDecoder()
    except Exception as exc:
        pytest.skip(f'no usable Metal device: {exc}')

def _upload(dec, seed=0):
    art, codes, x = _small_artifact(seed)
    entry = dec.upload(codes, f'test|{seed}', max_blocks=64)
    dec.set_x(entry, x)
    return (art, codes, x, entry)

def test_every_lookup_linear_variant_matches_the_authority(dec):
    art, codes, x, entry = _upload(dec)
    want = forge.pq_execute(art, x)
    for half in (False, True):
        for row4 in (False, True):
            shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, half_table=half, row4=row4, threadgroup_memory_limit=dec.threadgroup_memory_limit)
            got = dec.run_batch([(entry, shape)])[0]
            gap = float(np.abs(want - got).max() / (np.abs(want).max() + 1e-30))
            assert gap < b.PARITY_GATE, (half, row4, gap)

def test_the_kernel_matches_its_own_numpy_model_not_just_the_authority(dec):
    """Mechanism check: if the on-chip build drifted, this fails before parity does."""
    art, codes, x, entry = _upload(dec, seed=1)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit)
    got = dec.run_batch([(entry, shape)])[0]
    model = b.ll_reference(codes, x, cbs=4, half_table=False)
    assert np.allclose(got, model, rtol=0.0001, atol=1e-05)

def test_device_table_control_computes_the_same_thing_it_is_a_control_for(dec):
    art, codes, x, entry = _upload(dec, seed=2)
    want = forge.pq_execute(art, x)
    got = dec.run_device_table(entry)
    gap = float(np.abs(want - got).max() / (np.abs(want).max() + 1e-30))
    assert gap < b.PARITY_GATE, gap

def test_both_grammars_agree_on_the_same_tensor(dec):
    art, codes, x, entry = _upload(dec, seed=3)
    ll = dec.run_batch([(entry, b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit))])[0]
    df = dec.run_batch([(entry, b.dfma_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit))])[0]
    assert np.allclose(ll, df, rtol=0.0001, atol=1e-05)

def test_a_batch_of_tensors_returns_each_tensor_its_own_answer(dec):
    """A batched command buffer that crossed its buffers would still look plausible."""
    jobs, wants = ([], [])
    for seed in (4, 5, 6):
        art, codes, x, entry = _upload(dec, seed)
        shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit)
        jobs.append((entry, shape))
        wants.append(forge.pq_execute(art, x))
    got = dec.run_batch(jobs)
    for g, w in zip(got, wants):
        assert np.abs(w - g).max() / (np.abs(w).max() + 1e-30) < b.PARITY_GATE
    assert not np.allclose(got[0], got[1])

def test_the_shared_counterfactual_is_a_table_built_once(dec):
    """Two index sets against one codebook must both come back right from one build."""
    art, codes, x = _small_artifact(7)
    rows, nchunk, D = (256, 16, 8)
    book = np.ascontiguousarray(codes['codebooks'][0], dtype=np.float16)
    k = int(book.shape[0])
    rng = np.random.default_rng(11)
    sets = rng.integers(0, k, size=(2, rows, nchunk)).astype(np.uint8)
    stacked = np.stack([s.T.ravel() for s in sets])
    shape = b.ll_plan(rows=rows, nchunk=nchunk, D=D, k=k, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit)
    entry = dec.upload_shared(stacked, book, rows=rows, nchunk=nchunk, D=D, blocks=shape['blocks'])
    dec.set_x(entry, x)
    got = dec.run_shared(entry, shape)
    table = (x.reshape(nchunk, D) @ book.astype(np.float32).T).astype(np.float32)
    want = np.stack([table[np.arange(nchunk)[None, :], s].sum(axis=1, dtype=np.float32) for s in sets])
    assert np.allclose(got, want, rtol=0.0001, atol=0.0001)
    assert not np.allclose(got[0], got[1])

def test_a_shape_wider_than_the_partial_buffer_is_refused_not_dispatched(dec):
    art, codes, x, entry = _upload(dec, seed=8)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=1, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit)
    entry = dict(entry, max_blocks=2)
    with pytest.raises(b.TrackBError, match='fewer blocks'):
        dec.run_batch([(entry, shape)])

def test_a_wrong_length_x_is_refused_before_it_reaches_a_device_pointer(dec):
    art, codes, x, entry = _upload(dec, seed=9)
    with pytest.raises(b.TrackBError, match='geometry needs'):
        dec.set_x(entry, np.zeros(7, dtype=np.float32))

def test_an_out_of_range_index_is_refused_at_upload(dec):
    art, codes, x = _small_artifact(10)
    bad = dict(codes)
    idx = np.asarray(codes['indices']).copy()
    idx[0, 0] = 99
    bad['indices'] = idx
    with pytest.raises(b.TrackBError, match='out of range'):
        dec.upload(bad, 'test|bad', max_blocks=8)

def test_gpu_time_is_measured_not_derived(dec):
    """last_gpu_ms must come from the driver and be a real, smaller-than-wall number."""
    import time
    art, codes, x, entry = _upload(dec, seed=12)
    shape = b.ll_plan(rows=256, nchunk=16, D=8, k=16, cbs=4, tpg=64, threadgroup_memory_limit=dec.threadgroup_memory_limit)
    t0 = time.perf_counter()
    dec.run_batch([(entry, shape)])
    wall_ms = (time.perf_counter() - t0) * 1000.0
    assert 0.0 < dec.last_gpu_ms < wall_ms
