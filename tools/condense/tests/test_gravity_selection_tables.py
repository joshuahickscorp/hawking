"""Gravity kernel selection + MoE pure-logic (S6 F2 C2 densified)."""
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
from lab.operators import gravity_kernel_select as ks
import numpy as np
from lab.operators import gravity_forge as forge
from lab.operators import gravity_metal_lab_b as labb
from lab.operators import gravity_moe_layer as gml
from lab.operators import gravity_real_fixtures as grf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
R0 = {'S': 1, 'rotate': False, 'k': 128, 'D': 8, 'rows': 2048, 'cols': 6144, 'nchunk': 768, 'index_bits_on_disk': 7, 'rung': 'R0'}
KERNELS = sorted(ks.KERNELS)

def row(kernel, **kw):
    base = {'kernel': kernel, 'artifact_compatible': True, 'parity_relative_l2': 1e-07, 'incompatibility_reasons': []}
    base.update(kw)
    return base

def _refuses(kernel, facts, needle):
    ok, reasons = ks.KERNELS[kernel].accepts(facts)
    assert not ok
    assert any((needle in r for r in reasons))

def _select(*rows, selected=None, decided_by=None, **trail_checks):
    decision = ks.select_kernel(list(rows))
    if selected is not None:
        assert decision['selected'] == selected
    if decided_by is not None:
        assert decision['decided_by'] == decided_by
    for key, pred in trail_checks.items():
        step = next((t for t in decision['decision_trail'] if t['criterion'] == key))
        pred(step)
    return decision

def test_every_kernel_accepts_the_real_R0_geometry():
    for name, kernel in ks.KERNELS.items():
        ok, reasons = kernel.accepts(R0)
        assert ok, (name, reasons)

@pytest.mark.parametrize('kernel', KERNELS)
def test_a_rotated_artifact_is_refused_by_every_kernel(kernel):
    _refuses(kernel, {**R0, 'rotate': True}, 'rotat')

@pytest.mark.parametrize('kernel', KERNELS)
def test_a_larger_codebook_than_the_index_width_is_refused(kernel):
    _refuses(kernel, {**R0, 'k': 512}, 'k=512')

@pytest.mark.parametrize('kernel', KERNELS)
def test_multiple_subspaces_are_refused(kernel):
    _refuses(kernel, {**R0, 'S': 2}, 'subspaces')

def test_lookup_linear_alone_needs_rows_divisible_by_four():
    odd = {**R0, 'rows': 2050}
    assert not ks.KERNELS['lookup_linear'].accepts(odd)[0]
    assert ks.KERNELS['decode_fma_2d'].accepts(odd)[0]
    assert ks.KERNELS['production_v2'].accepts(odd)[0]

def test_decode_fma_alone_needs_D_divisible_by_four():
    odd = {**R0, 'D': 6}
    assert not ks.KERNELS['decode_fma_2d'].accepts(odd)[0]
    assert ks.KERNELS['production_v2'].accepts(odd)[0]

def test_an_artifact_at_an_unknown_rung_is_refused():
    _refuses('production_v2', {**R0, 'rung': 'R2'}, 'R2')

def test_a_codebook_that_cannot_be_staged_is_refused():
    _refuses('production_v2', {**R0, 'k': 256, 'D': 64}, 'threadgroup memory')

def test_every_refusal_names_a_source_line():
    for kernel in ks.KERNELS.values():
        assert kernel.refusal_sites
        for condition, source in kernel.refusal_sites:
            assert condition and source
            assert source.split('.')[0] in ('gravity_metal', 'gravity_metal_lab_b')

def test_incompatibility_beats_speed():
    d = _select(row('fast_but_incompatible', artifact_compatible=False, incompatibility_reasons=['artifact is rotated'], latency_wall_median_ms=0.01),
        row('slow_but_compatible',
         latency_wall_median_ms=1.0), selected='slow_but_compatible')
    assert d['decision_trail'][0]['criterion'] == 'artifact_compatible'

def test_a_parity_failure_is_not_a_candidate():
    _select(row('wrong', parity_relative_l2=0.01, latency_wall_median_ms=0.01), row('right', latency_wall_median_ms=1.0), selected='right')

def test_everything_rejected_raises_rather_than_picking_something():
    with pytest.raises(ks.KernelSelectError):
        ks.select_kernel([row('a', artifact_compatible=False, incompatibility_reasons=['nope'])])

def test_wall_eliminates_a_grossly_slower_candidate():
    d = _select(row('a', latency_wall_median_ms=1.0), row('b', latency_wall_median_ms=0.5), selected='b')
    wall = next((t for t in d['decision_trail'] if t['criterion'] == 'latency_wall_median_ms'))
    assert wall['inside_band'] == ['b']

def test_wall_may_eliminate_but_never_decide():
    _select(
        row(
            'fast_wall_slow_gpu',
            latency_wall_median_ms=0.2779,
            latency_gpu_median_ms=0.0587,
            latency_gpu_min_ms=0.057,
        ),
        row(
            'slow_wall_fast_gpu',
            latency_wall_median_ms=0.2985,
            latency_gpu_median_ms=0.0413,
            latency_gpu_min_ms=0.04,
        ),
        selected='slow_wall_fast_gpu',
        decided_by='latency_gpu_median_ms',
        latency_wall_median_ms=lambda wall: wall['decided'] is False,
    )

def test_a_candidate_with_no_gpu_clock_is_dropped_with_the_reason_recorded():
    _select(row('production_v2', latency_wall_median_ms=0.3), row('decode_fma_2d', latency_wall_median_ms=0.29, latency_gpu_median_ms=0.03,
        latency_gpu_min_ms=0.028), selected='decode_fma_2d', latency_gpu_median_ms=lambda gpu: gpu['dropped_for_no_measurement'] == ['production_v2'])

def test_median_and_min_must_agree_before_the_gpu_clock_decides():
    _select(row('a', latency_wall_median_ms=1.0, latency_gpu_median_ms=0.1, latency_gpu_min_ms=0.09, executed_total_bytes=5, executed_fp_ops=1,
        scratch_bytes=1), row('b', latency_wall_median_ms=1.0, latency_gpu_median_ms=0.102, latency_gpu_min_ms=0.089, executed_total_bytes=4, executed_fp_ops=1,
         scratch_bytes=1), decided_by='executed_total_bytes', selected='b')

def test_inside_the_band_the_gpu_clock_breaks_the_tie():
    _select(row('a', latency_wall_median_ms=1.0, latency_gpu_median_ms=0.8), row('b', latency_wall_median_ms=1.02, latency_gpu_median_ms=0.2), selected='b',
        decided_by='latency_gpu_median_ms')

def test_a_full_tie_falls_through_to_the_resource_criteria():
    _select(
        row(
            'a',
            latency_wall_median_ms=1.0,
            latency_gpu_median_ms=0.5,
            executed_total_bytes=2000000,
            executed_fp_ops=10,
            scratch_bytes=8192,
        ),
        row(
            'b',
            latency_wall_median_ms=1.0,
            latency_gpu_median_ms=0.5,
            executed_total_bytes=1000000,
            executed_fp_ops=10,
            scratch_bytes=8192,
        ),
        selected='b',
        decided_by='executed_total_bytes',
    )

def test_an_unmeasured_criterion_cannot_decide():
    d = _select(row('a', latency_wall_median_ms=1.0), row('b', latency_wall_median_ms=1.01))
    trail = {t['criterion']: t for t in d['decision_trail']}
    assert trail['latency_gpu_median_ms']['outcome'] == 'NOT_MEASURED_ON_ANY_SURVIVOR'

def test_the_decision_trail_cites_a_source_for_every_criterion():
    for step in ks.select_kernel([row('a', latency_wall_median_ms=1.0)])['decision_trail']:
        assert step['source'], step

def test_selection_carries_no_geometry_knowledge():
    source = Path(ks.__file__).read_text().split('def select_kernel')[1].split('\ndef ')[0]
    for forbidden in ('2048', '6144', '16384', 'gate', 'down', 'attention'):
        assert forbidden not in source, forbidden

def test_every_shape_prior_names_a_measurement():
    for prior in ks.SHAPE_PRIORS:
        assert prior['measurement'].endswith('.') and '.json' in prior['measurement']
        assert prior['values']

def test_every_criterion_names_a_source():
    for criterion in ks.CRITERIA:
        assert len(criterion.source) > 40
        assert criterion.role in ('hard_filter', 'eliminate_only', 'primary', 'tiebreak')

def test_every_closed_lever_names_the_measurement_that_closed_it():
    for lever in ks.CLOSED_LEVERS:
        assert '.json' in lever['measurement']
        assert lever['verdict'].isupper()

def test_the_within_tensor_hybrid_is_declared_unimplemented_not_invented():
    assert ks.HYBRID_WITHIN_TENSOR['status'] == 'UNIMPLEMENTED_NOT_MEASURED'
    assert 'run_batch' in ks.HYBRID_WITHIN_TENSOR['what_does_exist']

def test_the_parity_gate_is_the_modules_own_tolerance_not_a_tighter_one():
    assert ks.PARITY_GATE == 0.002

def test_candidate_configs_come_only_from_the_shape_prior_values():
    allowed = {(r['grammar'], r['field']): set(r['values']) for r in ks.SHAPE_PRIORS}
    for entry in ks.candidate_configs(dict(R0)):
        if entry['kernel'] == 'production_v2':
            continue
        for field, values in ((f, v) for (g, f), v in allowed.items() if g == entry['kernel']):
            if field in entry['config']:
                assert entry['config'][field] in values, entry

def test_a_config_the_device_would_refuse_never_becomes_a_candidate():
    facts = {**R0, 'k': 256}
    kinds = {e['kernel'] for e in ks.candidate_configs(facts)}
    assert 'lookup_linear' in kinds
    for entry in ks.candidate_configs(facts):
        if entry['shape'] is not None:
            assert entry['shape']['scratch_bytes'] <= ks.THREADGROUP_MEMORY_LIMIT

def test_blocks_are_clipped_to_nchunk_so_a_tiny_geometry_still_gets_candidates():
    facts = {**R0, 'cols': 512, 'nchunk': 64, 'rows': 28672}
    entries = ks.candidate_configs(facts)
    assert {'production_v2', 'decode_fma_2d', 'lookup_linear'} == {e['kernel'] for e in entries}
    for entry in entries:
        if entry['shape'] is not None:
            assert entry['shape']['blocks'] <= 64

def test_the_census_reproduces_both_published_token_totals():
    census = ks.geometry_census()
    assert census['production_token_bytes'] == ks.PRODUCTION_TOKEN_BYTES
    assert census['artifact_floor_bytes'] == ks.ARTIFACT_FLOOR_BYTES

def test_the_census_carries_every_geometry_class_the_shards_hold():
    classes = ks.geometry_census()['classes']
    for expected in ('routed_expert::2048x6144', 'routed_expert::6144x2048', 'shared_expert::2048x6144', 'shared_expert::6144x2048', 'attention::6144x16384',
        'attention::28672x512',
         'attention::576x6144', 'attention::16384x2048', 'attention::2048x6144', 'dense_mlp::6144x12288'):
        assert expected in classes, expected

def test_the_unique_lower_bound_never_exceeds_the_executed_upper_bound():
    for blocks in (1, 8, 64):
        cbs = (768 + blocks - 1) // blocks
        shape = ks.lb.dfma_plan(rows=2048, nchunk=768, D=8, k=128, cbs=cbs, tpg=256)
        cost = ks.lb.dfma_cost(rows=2048, cols=6144, nchunk=768, D=8, k=128, shape=shape)
        lower = ks.unique_read_bytes(rows=2048, nchunk=768, D=8, k=128, blocks=blocks)
        assert lower <= cost['executed_total_bytes']

def test_the_token_ledger_beats_the_production_total_when_the_kernel_moves_fewer_bytes():
    census = ks.geometry_census()
    selections = {key: {'kernel': 'decode_fma_2d', 'config': {}, 'executed_total_bytes': entry['kernel_device_bytes'] // entry['per_token_tensors'] // 4,
        'unique_read_bytes': entry['kernel_device_bytes'] // entry['per_token_tensors'] // 8} for key,
        entry in census['classes'].items() if entry['kernel_selectable']}
    ledger = ks.token_ledger(census, selections)
    assert ledger['selected_executed_bytes_per_token'] < ks.PRODUCTION_TOKEN_BYTES
    assert ledger['reduction_vs_production_executed'] > 1.0
    assert ledger['token_ceiling_tok_s']['selected_executed_upper_bound'] > ledger['token_ceiling_tok_s']['production_kernel']

def test_an_unselected_class_carries_the_production_bytes_rather_than_vanishing():
    census = ks.geometry_census()
    ledger = ks.token_ledger(census, {})
    assert ledger['selected_executed_bytes_per_token'] == ks.PRODUCTION_TOKEN_BYTES
    bases = {r['selection_basis'] for r in ledger['rows'] if r['kernel'].startswith('UNSELECTED')}
    assert bases == {'PRODUCTION_KERNEL_BYTES_CARRIED_UNCHANGED'}

def test_all_same_index_really_is_one_codeword():
    import numpy as np
    codes = {'rows': 8, 'cols': 32, 'nchunk': 4, 'D': 8, 'S': 1, 'sub': 8, 'rotate': False, 'indices': np.zeros((32, 1), dtype=np.uint8),
        'codebooks': [np.zeros((128, 8), dtype=np.float16)]}
    out = ks.adversarial_indices(codes, 'all_same')
    assert set(np.unique(out['indices'])) == {0}
    assert out['codebooks'] is codes['codebooks']

def test_spread_index_touches_every_codeword():
    import numpy as np
    codes = {'rows': 128, 'cols': 32, 'nchunk': 4, 'D': 8, 'S': 1, 'sub': 8, 'rotate': False, 'indices': np.zeros((512, 1), dtype=np.uint8),
        'codebooks': [np.zeros((128, 8), dtype=np.float16)]}
    out = ks.adversarial_indices(codes, 'spread')
    assert len(np.unique(out['indices'])) == 128

def test_an_unknown_adversarial_mode_is_refused():
    with pytest.raises(ks.KernelSelectError):
        ks.adversarial_indices({'rows': 1, 'nchunk': 1, 'indices': None, 'codebooks': [None]}, 'whatever')

def test_the_selected_kernel_decodes_a_real_tensor_within_the_parity_gate():
    grf = pytest.importorskip('gravity_real_fixtures')
    try:
        fixtures = ks.collect_fixtures(None)
        bench = ks.Bench(dec=ks.lb.TrackBDecoder(), prod=ks.gravity_metal.decoder(), reps=2, warmup=1)
    except Exception as exc:
        pytest.skip(f'device or fixtures unavailable: {exc}')
    name, fixture = fixtures[0]
    result = ks.run_geometry(name, fixture, bench, with_dense=False)
    assert result['selected']['parity']['relative_l2'] < ks.PARITY_GATE
    assert result['activation_source'] == grf.SYNTHETIC
    assert result['expert_selection'].startswith('FIXED_LIST')
GEOM = gml.LayerGeometry(rows_a=2048, cols_a=6144, rows_b=6144, cols_b=2048, D=8, k=128)
SMALL = gml.LayerGeometry(rows_a=64, cols_a=128, rows_b=128, cols_b=64, D=8, k=128)

def test_router_status_says_absent_everywhere():
    assert gml.ROUTER_STATUS == 'ROUTER_ABSENT_FIXED_EXPERT_LIST'
    plan = gml.layer_plan(GEOM)
    assert plan['router_status'] == gml.ROUTER_STATUS
    assert 'FIXED LIST' in plan['router_note']

def test_routing_weights_are_a_fixed_normalised_vector_plus_a_shared_one():
    w = gml.routing_weights(8)
    assert len(w) == 9
    assert abs(float(w[:8].sum()) - 1.0) < 1e-06
    assert w[8] == gml.SHARED_EXPERT_WEIGHT == 1.0
    assert np.array_equal(w, gml.routing_weights(8))
    assert len(set(w[:8].tolist())) == 8

def test_routing_weights_are_not_all_ones():
    """All-ones would hide a per-expert offset bug in the combine kernel."""
    w = gml.routing_weights(8)
    assert not np.allclose(w[:8], 1.0)

def test_one_command_buffer_and_the_dispatch_count_is_the_graph():
    plan = gml.layer_plan(GEOM)
    assert plan['command_buffers_per_layer'] == 1
    assert plan['encoders_per_layer'] == 4
    assert plan['dispatches_by_stage'] == {'wave_a': 18, 'swiglu': 9, 'wave_b': 9, 'combine': 1}
    assert plan['dispatches_per_layer'] == 37

def test_command_buffer_count_meets_the_moonshot_target():
    plan = gml.layer_plan(GEOM)
    targets = dict(gml.COMMAND_BUFFER_TARGETS)
    assert plan['command_buffers_per_layer'] <= targets['moonshot']

def test_shared_expert_is_a_slot_not_a_second_graph():
    with_shared = gml.layer_plan(GEOM)
    without = gml.layer_plan(GEOM, shared=False)
    assert with_shared['tensors_executed'] - without['tensors_executed'] == 3
    assert with_shared['command_buffers_per_layer'] == without['command_buffers_per_layer']
    assert with_shared['dispatches_per_layer'] - without['dispatches_per_layer'] == 4
    assert 'routing weight 1.0' in with_shared['shared_expert_integration']

def test_scratch_fits_the_measured_threadgroup_limit():
    plan = gml.layer_plan(GEOM)
    for value in plan['scratch_bytes'].values():
        assert value <= gml.gravity_metal.DEFAULT_THREADGROUP_MEMORY

def test_plan_refuses_a_config_that_overflows_threadgroup_memory():
    with pytest.raises(labb.TrackBError):
        gml.layer_plan(GEOM, wave_a_cfg={'grammar': 'lookup-linear', 'cbs': 512, 'tpg': 1024, 'half_table': False, 'row4': True})

def test_graph_ledger_does_not_bill_the_two_stores_the_graph_never_makes():
    """Wave A and Wave B never write a y: reduce_swiglu and moe_combine consume partials. A sum of 27 per-t"""
    plan = gml.layer_plan(GEOM)
    led = plan['byte_ledger']
    ca = plan['wave_a']['per_tensor_cost']
    naive = 18 * ca['executed_total_bytes']
    assert led['wave_a_read_bytes'] + led['wave_a_write_bytes'] < naive

def test_ledger_totals_are_self_consistent():
    led = gml.layer_plan(GEOM)['byte_ledger']
    reads = led['wave_a_read_bytes'] + led['reduce_swiglu_read_bytes'] + led['wave_b_read_bytes'] + led['combine_read_bytes']
    writes = led['wave_a_write_bytes'] + led['reduce_swiglu_write_bytes'] + led['wave_b_write_bytes'] + led['combine_write_bytes']
    assert reads == led['executed_read_bytes']
    assert writes == led['executed_write_bytes']
    assert reads + writes == led['executed_total_bytes']

def test_artifact_bytes_use_the_seven_bit_packed_width():
    """The shards store 7-bit indices; the kernel uploads 8.  Both are billed, separately."""
    led = gml.layer_plan(GEOM)['byte_ledger']
    per_tensor = (2048 * 768 * 7 + 7) // 8 + 128 * 8 * 2
    assert led['layer_artifact_bytes'] == 9 * (2 * per_tensor + ((6144 * 256 * 7 + 7) // 8 + 128 * 8 * 2))
    assert led['executed_over_artifact'] > 1.0

def test_layer_moves_far_fewer_bytes_than_dense_bf16():
    led = gml.layer_plan(GEOM)['byte_ledger']
    assert led['executed_over_dense_bf16'] < 0.2

def test_pressure_verdict_picks_the_tightest_target_and_does_not_round():
    assert gml.pressure_verdict(0.5)['reached'] == 'moonshot'
    assert gml.pressure_verdict(0.76)['reached'] == 'dominance'
    assert gml.pressure_verdict(1.51)['reached'] == 'ship'
    assert gml.pressure_verdict(3.01)['reached'] == 'viable'
    assert gml.pressure_verdict(6.01)['reached'] == 'NONE'

def test_pressure_verdict_is_exclusive_at_the_boundary_upward_only():
    """0.7501 ms is not a moonshot.  Nothing rounds toward a target."""
    assert gml.pressure_verdict(0.7501)['reached'] == 'dominance'
    assert gml.pressure_verdict(0.75)['reached'] == 'moonshot'

def test_per_token_labels_every_term_and_never_zeroes_the_unmeasured():
    proj = gml.per_token_projection(1.0, source='test')
    assert proj['moe_layer_ms']['status'] == 'MEASURED'
    assert proj['attention_per_layer_ms']['status'] == 'DERIVED'
    assert proj['dense_mlp_layer_ms']['status'] == 'DERIVED'
    assert len(proj['unmeasured_terms']) >= 4
    assert any(('lm_head' in t for t in proj['unmeasured_terms']))
    assert proj['layer_counts']['sparse_moe'] + proj['layer_counts']['dense_mlp'] == 78

def test_per_token_total_is_the_sum_of_its_labelled_terms():
    proj = gml.per_token_projection(1.25, source='test')
    total = 75 * 1.25 + 3 * proj['dense_mlp_layer_ms']['value'] + 78 * proj['attention_per_layer_ms']['value']
    assert abs(proj['implied_token_ms'] - total) < 1e-09
    assert abs(proj['implied_tok_s'] - 1000.0 / total) < 1e-09

def test_per_token_is_an_upper_bound_on_tok_s():
    assert 'UPPER bound on tok/s' in gml.per_token_projection(1.0, source='t')['note']

def _fake_expert(rng, rows_a, cols_a, rows_b, cols_b, D=8, k=128):

    def codes(rows, cols):
        nchunk = cols // D
        return {'rows': rows, 'cols': cols, 'nchunk': nchunk, 'D': D, 'S': 1, 'sub': D, 'rotate': False, 'seed': 0, 'codebooks': [rng.standard_normal((k,
            D)).astype(np.float32)], 'indices': rng.integers(0, k, size=(rows * nchunk, 1)).astype(np.uint8)}
    return {'gate': codes(rows_a, cols_a), 'up': codes(rows_a, cols_a), 'down': codes(rows_b, cols_b)}

def test_cpu_authority_composes_the_graph_not_a_sum_of_projections():
    rng = np.random.default_rng(7)
    experts = [_fake_expert(rng, 16, 32, 32, 16) for _ in range(2)]
    x = rng.standard_normal(32).astype(np.float32)
    w = np.array([0.75, 0.25], dtype=np.float32)
    out = gml.cpu_layer(experts, x, w)
    want = x.astype(np.float32).copy()
    for exp, weight in zip(experts, w):
        g = forge.pq_execute(gml.artifact_of(exp['gate']), x)
        u = forge.pq_execute(gml.artifact_of(exp['up']), x)
        want = want + weight * forge.pq_execute(gml.artifact_of(exp['down']), (gml.silu(g) * u).astype(np.float32))
    assert np.allclose(out['y'], want, atol=0, rtol=0)

def test_cpu_authority_includes_the_residual():
    rng = np.random.default_rng(11)
    experts = [_fake_expert(rng, 16, 32, 32, 16)]
    x = rng.standard_normal(32).astype(np.float32)
    zero = gml.cpu_layer(experts, x, np.array([0.0], dtype=np.float32))
    assert np.allclose(zero['y'], x)

def test_cpu_authority_refuses_a_weight_count_mismatch():
    rng = np.random.default_rng(3)
    with pytest.raises(gml.MoeLayerError):
        gml.cpu_layer([_fake_expert(rng, 16, 32, 32, 16)], np.zeros(32, np.float32), np.ones(2, np.float32))

def test_silu_matches_the_kernel_expression():
    v = np.array([-4.0, -0.5, 0.0, 0.5, 4.0], dtype=np.float32)
    assert np.allclose(gml.silu(v), v / (1.0 + np.exp(-v)), atol=1e-07)

def test_swiglu_error_composes_nonlinearly_so_whole_layer_parity_is_the_number():
    """The reason the gate is on the layer output and not on a projection. A relative perturbation of the g"""
    rng = np.random.default_rng(5)
    g = rng.standard_normal(4096).astype(np.float32) * 3.0
    u = rng.standard_normal(4096).astype(np.float32)
    eps = 0.0001
    clean = gml.silu(g) * u
    perturbed = gml.silu(g * (1 + eps)) * u
    rel_in = eps
    rel_out = float(np.linalg.norm(perturbed - clean) / np.linalg.norm(clean))
    assert rel_out > rel_in

def test_parity_of_reports_the_gate_it_was_judged_against():
    p = gml.parity_of(np.ones(4, np.float32), np.ones(4, np.float32))
    assert p['gate'] == gml.PARITY_GATE == 0.002
    assert p['relative_l2'] == 0.0
    assert p['finite']

def test_parity_gate_is_the_modules_own_tolerance_not_one_over_a_million():
    """1e-6 would fail for the right reason and be misread: the codebook is cast to fp16."""
    assert gml.PARITY_GATE == 0.002

def test_parity_detects_a_wrong_answer():
    ref = np.arange(8, dtype=np.float32) + 1
    assert gml.parity_of(ref * 1.5, ref)['relative_l2'] > gml.PARITY_GATE

def test_the_refuted_numbers_cannot_be_republished():
    for name in ('dense_fp16_9.012ms', 'speedup_35.9x', 'metal_parity_1.4e-6'):
        with pytest.raises(Exception):
            gml.lab.assert_not_refuted(name=name)

def test_module_does_not_quote_the_refuted_dense_baseline():
    text = Path(gml.__file__).read_text()
    assert '9.012' not in text
    assert '35.9x' not in text

def test_module_never_opens_the_artifact_directory_for_writing():
    text = Path(gml.__file__).read_text()
    for forbidden in ('shutil', '.unlink(', '.rename(', '.replace(', 'open(', "'w'"):
        assert forbidden not in text or forbidden == 'open('
    assert 'write_text' in text
    assert str(grf.ARTIFACT_DIR) not in text

def test_reports_are_written_only_under_the_breakthrough_report_dir():
    assert gml.REPORT_DIR.name == 'breakthrough'
    text = Path(gml.__file__).read_text()
    assert text.count('write_text') == text.count('REPORT_DIR /')

def test_command_queue_depth_and_autorelease_pool_are_inherited_not_reinvented():
    """The 64-in-flight deadlock is handled by lab_b's decoder, which this module holds."""
    labb_text = Path(labb.__file__).read_text()
    assert 'newCommandQueueWithMaxCommandBufferCount_(1024)' in labb_text
    assert 'objc.autorelease_pool()' in Path(gml.__file__).read_text()

def test_the_extra_kernels_are_only_the_two_the_graph_needs():
    assert gml.EXTRA_METAL_SOURCE.count('kernel void ') == 3
    for name in ('reduce_swiglu', 'reduce_swiglu4', 'moe_combine'):
        assert f'kernel void {name}(' in gml.EXTRA_METAL_SOURCE

def test_combine_kernel_starts_from_the_residual():
    """The residual is the combine accumulator's initialiser, which is why it has no cost."""
    assert 'float acc = residual[gid];' in gml.EXTRA_METAL_SOURCE

def test_band_verdict_needs_median_and_min_to_agree():
    assert gml._band_verdict(1.2, 1.2) == 'VEC4_WINS'
    assert gml._band_verdict(1.2, 1.0) == 'NEUTRAL_WITHIN_NOISE'
    assert gml._band_verdict(0.8, 0.8) == 'SCALAR_WINS'
    assert gml._band_verdict(1.0, 1.0) == 'NEUTRAL_WITHIN_NOISE'

def test_selftest_runs_without_a_device():
    assert gml.selftest() == 0

def _loaded():
    try:
        return gml.load_layer(3)
    except Exception as exc:
        pytest.skip(f'real artifacts unavailable: {exc}')

def test_real_layer_geometry_and_activation_label():
    loaded = _loaded()
    geom = loaded['geometry']
    assert (geom.rows_a, geom.cols_a) == (2048, 6144)
    assert (geom.rows_b, geom.cols_b) == (6144, 2048)
    assert geom.D == 8 and geom.k == 128
    assert len(loaded['experts']) == grf.EXPERTS_PER_TOKEN + 1
    assert loaded['fixture_set']['router_present'] is False
    for exp in loaded['experts']:
        for proj in ('gate', 'up', 'down'):
            assert exp[proj]['fixture'].activation_source == grf.SYNTHETIC

def test_every_cache_key_is_a_distinct_content_address():
    """Never id(), never one literal reused -- gravity_metal now refuses both."""
    loaded = _loaded()
    keys = [e[p]['key'] for e in loaded['experts'] for p in ('gate', 'up', 'down')]
    assert len(keys) == len(set(keys)) == 27
    for key, exp in zip(keys, (e[p] for e in loaded['experts'] for p in ('gate', 'up', 'down'))):
        shard, name, digest = key.split('::')
        assert shard.endswith('.gravity') and name == exp['name'] and (len(digest) == 64)

def test_prepare_refuses_a_cache_key_claimed_by_two_tensors():
    loaded = _loaded()
    try:
        ex = gml.MoeLayerExecutor()
    except Exception as exc:
        pytest.skip(f'no Metal device: {exc}')
    experts = [{p: dict(e[p]) for p in ('gate', 'up', 'down')} for e in loaded['experts']]
    experts[1]['gate']['key'] = experts[0]['gate']['key']
    with pytest.raises(gml.MoeLayerError, match='claimed by both'):
        ex.prepare(experts, gml.routing_weights(8), plan=gml.layer_plan(loaded['geometry']))

def test_whole_layer_parity_against_the_cpu_authority():
    loaded = _loaded()
    try:
        ex = gml.MoeLayerExecutor()
    except Exception as exc:
        pytest.skip(f'no Metal device: {exc}')
    geom = loaded['geometry']
    weights = gml.routing_weights(8)
    x = grf.synthetic_activation(geom.cols_a, seed=gml.SEED)
    ex.prepare(loaded['experts'], weights, plan=gml.layer_plan(geom))
    ex.set_input(x)
    got = ex.run_layer()
    authority = gml.cpu_layer([{p: e[p]['codes'] for p in ('gate', 'up', 'down')} for e in loaded['experts']], x, weights)
    parity = gml.parity_of(got, authority['y'])
    assert parity['finite']
    assert parity['relative_l2'] < gml.PARITY_GATE
