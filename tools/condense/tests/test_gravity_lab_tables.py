"""Gravity lab/forge/PQ pure-logic (S6 F2 C2 densified)."""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import pathlib
import sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import json
import math
import os
from dataclasses import replace
from lab.operators import gravity_bench_lab as bl
import numpy as np
from lab.operators import gravity_forge as gf
from lab.operators import gravity_forge as forge
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _spec(**over):
    base = dict(rows=64, cols=128, batch=1, input_seed=7, input_dtype='float32', output_dtype='float32', warmup=1, reps=5, sync_boundary='none_cpu_wall_clock',
        dependency_shape='independent_calls', pack_in_timed_region=False, unpack_in_timed_region=True)
    base.update(over)
    return bl.BenchSpec(**base)

def _result(baseline, samples, spec=None, **over):
    return bl.BenchResult(baseline=baseline, spec=spec or _spec(), timings=bl.ComponentTimings(end_to_end=bl.TimingStats(tuple(samples))), **over)

def test_matched_specs_are_field_identical():
    assert bl.matched(_spec(), _spec())
    assert bl.mismatched_fields(_spec(), _spec()) == ()
    bl.require_matched(_spec(), _spec())

@pytest.mark.parametrize('field,value', [('rows', 65), ('cols', 129), ('batch', 2), ('input_seed', 8), ('input_dtype', 'float16'), ('output_dtype', 'float16'),
    ('warmup', 2), ('reps', 6), ('sync_boundary', 'per_call_host_sync'), ('dependency_shape', 'serial_dependent_chain'), ('pack_in_timed_region',
        True), ('unpack_in_timed_region',
     False)])
def test_any_differing_field_breaks_the_match(field, value):
    """Every field participates; none of them is a 'detail' that may drift."""
    other = _spec(**{field: value})
    assert not bl.matched(_spec(), other)
    assert bl.mismatched_fields(_spec(), other) == (field,)
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.require_matched(_spec(), other)
    assert _spec().fingerprint != other.fingerprint

def test_speedup_refuses_unmatched_specs():
    base = _result('cpu_authority', [1.0, 1.0, 1.0, 1.0, 1.0])
    cand = _result('custom_v2', [0.5, 0.5, 0.5, 0.5, 0.5], spec=_spec(batch=2))
    with pytest.raises(bl.MatchedBenchmarkError, match='unmatched BenchSpecs'):
        bl.speedup(base, cand)

def test_speedup_refuses_unreproduced_baseline():
    base = _result('dense_fp16_mps', [1.0, 1.0, 1.0, 1.0, 1.0], reproduced=False)
    cand = _result('custom_v2', [0.5, 0.5, 0.5, 0.5, 0.5])
    with pytest.raises(bl.MatchedBenchmarkError, match='unreproduced'):
        bl.speedup(base, cand)

def test_speedup_refuses_mismatched_timed_region():
    """GPU-only time on one side and wall clock on the other is not a matched comparison."""
    base = bl.BenchResult('cpu_authority', _spec(), bl.ComponentTimings(end_to_end=bl.TimingStats((1.0, 1.0, 1.1))))
    cand = bl.BenchResult('custom_v2', _spec(), bl.ComponentTimings(gpu_execution=bl.TimingStats((0.5, 0.5, 0.5))))
    with pytest.raises(bl.MatchedBenchmarkError, match='component mismatch'):
        bl.speedup(base, cand)

def test_speedup_on_matched_specs_reports_the_ratio_and_direction():
    base = _result('dense_fp16_mps', [0.3674] * 5)
    cand = _result('custom_v2', [0.5057] * 5)
    out = bl.speedup(base, cand)
    assert out['specs_matched'] and out['baseline'] == 'dense_fp16_mps'
    assert out['candidate'] == 'custom_v2'
    assert math.isclose(out['speedup'], 0.3674 / 0.5057, rel_tol=1e-09)
    assert out['slower_than_baseline']

def test_unknown_baseline_name_is_rejected():
    with pytest.raises(bl.MatchedBenchmarkError, match='unknown baseline'):
        _result('my_fast_kernel', [1.0, 1.0])

def test_raw_sample_statistics_are_correct_and_no_mean_is_exposed():
    samples = (1.0, 2.0, 3.0, 4.0, 100.0)
    st = bl.TimingStats(samples)
    assert st.count == 5
    assert st.min_ms == 1.0 and st.max_ms == 100.0
    assert st.median_ms == 3.0
    assert st.p95_ms == 100.0
    assert math.isclose(st.stddev_ms, math.sqrt(1902.5), rel_tol=1e-12)
    assert math.isclose(st.coefficient_of_variation, math.sqrt(1902.5) / 3.0, rel_tol=1e-12)
    assert list(st.to_json()['raw_samples_ms']) == list(samples)
    assert not any(('mean' in k for k in st.to_json()))
    assert not hasattr(st, 'mean_ms')

def test_p95_nearest_rank_on_twenty_samples():
    st = bl.TimingStats(tuple((float(i) for i in range(1, 21))))
    assert st.median_ms == 10.5
    assert st.p95_ms == 19.0

def test_timing_stats_reject_degenerate_input():
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0,))
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0, float('nan')))
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0, -1.0))

def test_contention_flag_tracks_the_documented_threshold():
    quiet = bl.TimingStats((1.0, 1.01, 0.99, 1.0, 1.02))
    assert quiet.coefficient_of_variation < bl.CONTENTION_CV_THRESHOLD
    assert not quiet.is_contended
    noisy = bl.TimingStats((1.0, 1.0, 1.0, 1.0, 5.0))
    assert noisy.coefficient_of_variation > bl.CONTENTION_CV_THRESHOLD
    assert noisy.is_contended
    assert noisy.median_ms == 1.0 and noisy.max_ms == 5.0
    assert noisy.to_json()['contention_cv_threshold'] == 0.15

def test_unmeasured_components_serialise_as_unmeasured_not_zero():
    res = _result('cpu_authority', [1.0, 1.0, 1.0])
    timings = res.to_json()['timings']
    assert set(timings) == set(bl.COMPONENTS)
    assert timings['end_to_end']['median_ms'] == 1.0
    for name in ('gpu_execution', 'host_encode', 'command_buffer', 'cold_start', 'warm_steady_state'):
        assert timings[name] == 'UNMEASURED'
        assert timings[name] != 0

def test_at_least_one_component_must_be_measured():
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.ComponentTimings()

def test_roofline_bills_against_the_measured_roofs():
    res = _result('cpu_authority', [1.0, 1.0, 1.0])
    res = replace(res, bytes_moved=736000000, flops=17703000000)
    roof = res.roofline()
    assert roof['timing_source'] == 'end_to_end'
    assert math.isclose(roof['fraction_of_bandwidth_roof'], 1.0, rel_tol=1e-09)
    assert math.isclose(roof['fraction_of_compute_roof'], 1.0, rel_tol=1e-09)
    assert bl.BANDWIDTH_ROOF_GB_S == 736.0 and bl.COMPUTE_ROOF_GFLOP_S == 17703.0

def test_roofline_without_counters_is_unmeasured():
    roof = _result('cpu_authority', [1.0, 1.0, 1.0]).roofline()
    assert roof['achieved_gb_s'] == 'UNMEASURED'
    assert roof['fraction_of_compute_roof'] == 'UNMEASURED'

@pytest.mark.parametrize('name', [c.name for c in bl.REFUTED_CLAIMS])
def test_refuted_claims_are_rejected_by_name(name):
    with pytest.raises(bl.MatchedBenchmarkError, match='REFUTED'):
        bl.assert_not_refuted(name=name)

@pytest.mark.parametrize('kind,value', [('milliseconds', 9.012), ('ratio', 35.9), ('parity', 1.4e-06)])
def test_refuted_claims_are_rejected_by_value(kind, value):
    with pytest.raises(bl.MatchedBenchmarkError, match='REFUTED'):
        bl.assert_not_refuted(kind=kind, value=value)

def test_the_retracted_headline_cannot_be_rebuilt():
    """9.012 ms / 0.2511 ms = 35.9x. Both the baseline and the ratio are rejected."""
    base = _result('dense_fp16_mps', [9.012] * 5)
    cand = _result('custom_v2', [9.012 / 35.9] * 5)
    with pytest.raises(bl.MatchedBenchmarkError, match='dense_fp16_9.012ms'):
        bl.speedup(base, cand)
    base2 = _result('dense_fp16_mps', [3.59] * 5)
    cand2 = _result('custom_v2', [0.1] * 5)
    with pytest.raises(bl.MatchedBenchmarkError, match='speedup_35.9x'):
        bl.speedup(base2, cand2)

def test_honest_numbers_pass_the_refuted_guard():
    bl.assert_not_refuted(kind='milliseconds', value=0.3674)
    bl.assert_not_refuted(kind='ratio', value=0.727)
    bl.assert_not_refuted(kind='parity', value=0.00021)

def test_spec_and_result_json_round_trip():
    spec = _spec()
    assert bl.BenchSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec
    res = replace(_result('custom_v2', [1.0, 2.0, 3.0]), bytes_moved=1024, flops=2048, notes='n')
    back = bl.BenchResult.from_json(json.loads(json.dumps(res.to_json())))
    assert back == res
    assert back.to_json() == res.to_json()

def test_report_schema_round_trip():
    base = _result('cpu_authority', [1.0, 1.0, 1.0])
    cand = _result('custom_v2', [0.5, 0.5, 0.5])
    report = bl.build_report([base, cand], [bl.speedup(base, cand)], label='t')
    loaded = json.loads(json.dumps(report))
    assert loaded['schema'] == 'hawking.glm52.matched_benchmark.v1'
    assert loaded['baselines'] == list(bl.BASELINES)
    assert loaded['machine']['bandwidth_roof_gb_s'] == 736.0
    assert loaded['machine']['gpu_cores'] == 60
    assert [c['name'] for c in loaded['refuted_claims']] == [c.name for c in bl.REFUTED_CLAIMS]
    assert loaded['matched'][0]['baseline'] == 'cpu_authority'
    assert loaded['matched'][0]['specs_matched'] is True
    assert [bl.BenchResult.from_json(r).to_json() for r in loaded['results']] == loaded['results']

def test_report_writer_refuses_a_foreign_schema(tmp_path):
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.write_report(tmp_path / 'x.json', {'schema': 'something.else.v1'})

def test_build_report_refuses_an_unasserted_speedup():
    with pytest.raises(bl.MatchedBenchmarkError, match='specs_matched'):
        bl.build_report([], [{'baseline': 'cpu_authority', 'candidate': 'custom_v2'}], label='t')

def test_measure_keeps_every_sample():
    spec = _spec(warmup=2, reps=7)
    calls = []
    stats = bl.measure(lambda: calls.append(1), spec)
    assert len(calls) == 9
    assert stats.count == 7 == len(stats.raw_samples_ms)

def test_selftest_runs_cpu_only_and_is_internally_consistent():
    report = bl.selftest(rows=64, cols=128)
    assert report['schema'] == 'hawking.glm52.matched_benchmark.v1'
    assert {r['baseline'] for r in report['results']} == {'cpu_authority', 'custom_v2'}
    assert report['selftest']['json_round_trip_stable']
    assert report['selftest']['unmatched_comparison_refused']
    assert report['selftest']['refuted_guard_live']
    for res in report['results']:
        assert res['timings']['gpu_execution'] == 'UNMEASURED'
        assert res['timings']['end_to_end']['count'] == 15

def _forge_lowrank(m=128, n=128, r=8, seed=0):
    """Cluster-local helper from test_gravity_forge (distinct from second-light PQ)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, r)).astype(np.float32) @ rng.standard_normal((r, n)).astype(np.float32) * 0.1

def test_selftest_green():
    out = gf.selftest()
    assert out['ok'] and out['accounting_invariant_holds']
    assert out['compressible_beats_random']
    assert out['deterministic_bytes']

def test_byte_ledger_counts_everything():
    """Whole-artifact bytes must include indices, codebooks, transform seed, and metadata -
    nothing is silently free (guards against hidden byte accounting)."""
    w = _forge_lowrank()
    art = gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)
    assert 'indices' in art.ledger.items and 'fp16_params' in art.ledger.items
    assert 'transform_seed' in art.ledger.items
    assert art.ledger.total_bits() > sum(art.ledger.items.values())
    assert art.physical_bytes > 0

def test_accounting_invariant_whole_ge_base_plus_doctor():
    w = _forge_lowrank()
    for art in (gf.pack_naive_rvq(w, dim=32, k=64, stages=2), gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=4, sparse_rows=4),
        gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)):
        assert art.whole_artifact_bpw >= art.base_bpw + art.doctor_bpw - 1e-06
        assert art.overhead_bpw >= -1e-06

def test_weight_error_is_not_capability_claim():
    """A family can have low weight error yet the artifact must never be flagged capability-parity;
    output_divergence must always label itself proxy_output / capability_parity False."""
    w = _forge_lowrank(r=4)
    art = gf.pack_low_rank(w, rank=4)
    assert gf._rel_error(w, art.recon) < 0.01
    assert not hasattr(art, 'capability_parity')

def test_deterministic_same_seed_same_bytes():
    w = _forge_lowrank()
    a = gf.pack_naive_rvq(w, dim=32, k=64, stages=2, seed=3)
    b = gf.pack_naive_rvq(w, dim=32, k=64, stages=2, seed=3)
    assert a.physical_bytes == b.physical_bytes

def test_shared_grammar_amortizes_codebook():
    """A larger expert cluster must lower whole-artifact BPW (shared codebook amortized), proving
    the MoE sharing lever is real and correctly accounted."""
    experts = [_forge_lowrank(seed=i) for i in range(8)]
    small = gf.pack_shared_grammar(experts[:2], dim=32, k=64, stages=2)
    big = gf.pack_shared_grammar(experts, dim=32, k=64, stages=2)
    assert big.whole_artifact_bpw < small.whole_artifact_bpw

def test_repairability_splits_base_and_doctor():
    w = _forge_lowrank()
    art = gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=4, sparse_rows=4)
    assert art.base_bpw > 0 and art.doctor_bpw > 0
    assert 'doctor_lowrank' in art.ledger.items and 'doctor_sparse_rows' in art.ledger.items

def test_no_dense_shadow_model():
    """The reconstruction returned is a bounded per-tile object the size of the tile, not a hidden
    expansion. recon.size must equal the represented weight count, never more."""
    w = _forge_lowrank(m=256, n=128, r=8)
    art = gf.pack_transform_pq(w, dim=32, subspaces=2, k=64)
    assert art.recon.size == w.size == art.n_weights

def test_four_materially_distinct_families_available():
    """Section 5 requires >=4 materially distinct families (beyond the RVQ/low-rank controls)."""
    out = gf.selftest()
    assert out['families_available'] >= 4
    w = _forge_lowrank()
    families = {'transform_pq': gf.pack_transform_pq(w, dim=32, subspaces=2, k=64).family, 'shared_grammar': gf.pack_shared_grammar([w, w * 1.01], dim=32, k=64,
         stages=2).family, 'repairability': gf.pack_repairability_shaped(w, base_dim=32, base_k=32, corr_rank=2, sparse_rows=2).family,
        'ternary': gf.pack_ternary_factor(w, rank=4).family}
    assert len(set(families.values())) == 4

def test_ternary_factor_is_ternary_and_billed_conservatively():
    """Ternary factors must be {-1,0,+1} and billed at 2 bits each (no base-3 packing claimed)."""
    w = _forge_lowrank()
    art = gf.pack_ternary_factor(w, rank=6)
    assert art.config['ternary_bits'] == 2
    assert 'ternary_factors' in art.ledger.items
    assert art.recon.size == w.size and np.isfinite(art.recon).all()

def test_activation_aware_reduces_output_error_when_activations_concentrate():
    """Section 6.2: with activations concentrated on a few input channels, activation-aware packing
    must reduce OUTPUT error ||X W^T - X What^T|| vs weight-aware at ~matched bytes, and bill the scale."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64)).astype(np.float32)
    X = rng.standard_normal((128, 64)).astype(np.float32) * 0.05
    X[:, :4] *= 40.0
    wa = gf.pack_transform_pq(W, dim=16, subspaces=2, k=16)
    aa = gf.pack_transform_pq_actaware(W, X, dim=16, subspaces=2, k=16)
    err_wa = float(np.linalg.norm(X @ W.T - X @ wa.recon.T))
    err_aa = float(np.linalg.norm(X @ W.T - X @ aa.recon.T))
    assert err_aa < err_wa
    assert aa.config.get('act_scaled') is True
    assert aa.physical_bytes >= wa.physical_bytes
CONDENSE = pathlib.Path(__file__).resolve().parents[1]
GLM52_EMBED_WEIGHTS = 951582720
GLM52_EMBED_SUBVECTORS_AT_DIM8 = GLM52_EMBED_WEIGHTS // 8

def _tensor(rows: int, dim: int, device):
    import torch
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.standard_normal((rows, dim)).astype(np.float32)).to(device)

@pytest.mark.parametrize('rows,dim,k', [(4096, 8, 64), (10000, 8, 128), (3331, 16, 32)])
def test_chunked_argmin_is_exactly_unchunked(monkeypatch, rows, dim, k):
    """Ragged final chunks included: 3331 rows never divides evenly."""
    import torch
    for device in ('cpu', forge._device()):
        values = _tensor(rows, dim, device)
        v2 = (values * values).sum(1, keepdim=True)
        cb = values[:k].clone()
        whole = forge._argmin_chunked(values, v2, cb, rows)
        chunked = forge._argmin_chunked(values, v2, cb, max(1, rows // 7))
        assert torch.equal(whole, chunked), f'chunking changed the argmin on {device}'

def test_distance_block_is_bounded_by_the_codebook_not_the_tensor():
    budget = forge._DISTANCE_BUDGET_BYTES
    huge = forge._chunk_rows(GLM52_EMBED_SUBVECTORS_AT_DIM8, 128)
    assert huge * 128 * 4 <= budget, 'the distance block must fit the budget'
    assert huge < GLM52_EMBED_SUBVECTORS_AT_DIM8, 'the embedding table must actually chunk'
    assert forge._chunk_rows(10000000, 128) == forge._chunk_rows(100000000, 128)
    assert forge._chunk_rows(10000000, 256) <= forge._chunk_rows(10000000, 128)

def test_accumulation_is_not_chunked(monkeypatch):
    """The reverted attempt chunked index_add_ and raced its own count vector."""
    import torch
    values = _tensor(5000, 8, forge._device())
    calls = {'n': 0}
    real = torch.Tensor.index_add_

    def counting(self, dim, index, source, **kwargs):
        calls['n'] += 1
        return real(self, dim, index, source, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'index_add_', counting)
    monkeypatch.setattr(forge, '_DISTANCE_BUDGET_BYTES', 64 * 4 * 101)
    iters = 3
    forge._kmeans(values, 64, iters=iters, seed=0)
    assert calls['n'] == 2 * iters, f'expected {2 * iters} accumulations, saw {calls['n']}: the accumulation is being chunked again'

def test_kmeans_matches_across_block_sizes(monkeypatch):
    """Same centroids whether one block or many, on CPU where MPS noise cannot mask it."""
    import torch
    values = _tensor(8000, 8, 'cpu')
    monkeypatch.setattr(forge, '_DISTANCE_BUDGET_BYTES', 1 << 30)
    whole = forge._kmeans(values, 64, iters=5, seed=0)
    monkeypatch.setattr(forge, '_DISTANCE_BUDGET_BYTES', 64 * 4 * 257)
    chunked = forge._kmeans(values, 64, iters=5, seed=0)
    assert torch.equal(whole, chunked), 'chunking changed the CPU k-means result'

def _second_light_lowrank(m=256, n=256, r=24, noise=0.01, seed=0):
    """Cluster-local helper from test_second_light_pq (distinct from gravity_forge).
    A genuinely low-rank matrix (rank r of min(m,n)) plus a little full-rank noise, so PQ leaves a
    meaningful residual for islands and Doctor to work on. Not a 0.5B-style toy."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((m, r)).astype(np.float32) @ rng.standard_normal((r, n)).astype(np.float32)
    return (base * 0.1 + rng.standard_normal((m, n)).astype(np.float32) * noise).astype(np.float32)

def test_pq_lifecycle_roundtrip():
    w = _second_light_lowrank()
    fam = gf.PQFamily(dim=32, subspaces=4, k=32, seed=0)
    assert fam.VERBS == ('inspect', 'fit', 'pack', 'measure', 'execute', 'validate', 'repairability')
    ins = fam.inspect(w)
    assert ins['rows'] == 256 and ins['cols'] == 256
    assert ins['valid_subvector_dims'] == [4, 8, 16]
    assert 1 <= ins['effective_rank_90'] <= 256
    assert 'residual_kurtosis' in ins
    fit = fam.fit(w)
    assert len(fit['codebooks']) == fit['S']
    assert fit['indices'].shape[1] == fit['S']
    art = fam.pack(w)
    assert art.family == 'product_quant' and art.recon.size == w.size == art.n_weights
    meas = fam.measure(w, art)
    assert meas['whole_artifact_bpw'] < 8.0 and meas['rel_error'] > 0.0
    assert meas['verdict'] in ('survives', 'degraded', 'collapse')
    rep = fam.repairability(w, art)
    assert 0.0 <= rep['rank4_capture'] <= 1.0 and 0.0 <= rep['sparse_row_capture'] <= 1.0
    assert rep['residual_rel_energy'] > 0.0

def test_pq_plain_and_rotated_both_available():
    """PQ must exist as its own geometry (plain, no Hadamard) AND as the rotated transform_pq variant."""
    w = _second_light_lowrank()
    plain = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=False)
    rotated = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=True)
    assert plain.family == 'product_quant' and plain.config['rotate'] is False
    assert rotated.family == 'transform_pq' and rotated.config['rotate'] is True
    assert 'transform_seed' not in plain.ledger.items
    assert 'transform_seed' in rotated.ledger.items

@pytest.mark.parametrize('rotate', [False, True])
def test_pq_direct_execute_matches_dense(rotate):
    w = _second_light_lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32, rotate=rotate)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(w.shape[1]).astype(np.float32)
    y = gf.pq_execute(art, x)
    y_dense = art.recon @ x
    rel = float(np.linalg.norm(y - y_dense) / (np.linalg.norm(y_dense) + 1e-12))
    assert rel < 0.0001
    val = gf.pq_validate(w, art, x)
    assert val['within_tol'] and val['no_dense_reconstruction'] is True

def test_pq_execute_batched():
    w = _second_light_lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((w.shape[1], 6)).astype(np.float32)
    Y = gf.pq_execute(art, X)
    assert Y.shape == (w.shape[0], 6)
    rel = float(np.linalg.norm(Y - art.recon @ X) / (np.linalg.norm(art.recon @ X) + 1e-12))
    assert rel < 0.0001

def test_pq_deterministic_bytes_and_recon():
    w = _second_light_lowrank()
    a = gf.pq_pack(w, dim=32, subspaces=4, k=32, seed=11)
    b = gf.pq_pack(w, dim=32, subspaces=4, k=32, seed=11)
    assert a.physical_bytes == b.physical_bytes
    assert np.array_equal(a.config['pq_codes']['indices'], b.config['pq_codes']['indices'])
    assert np.allclose(a.recon, b.recon, atol=1e-05)

def test_pq_byte_ledger_counts_indices_and_codebooks():
    w = _second_light_lowrank()
    art = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    assert 'indices' in art.ledger.items and 'fp16_params' in art.ledger.items
    assert art.ledger.total_bits() > sum(art.ledger.items.values())
    assert art.whole_artifact_bpw >= art.base_bpw + art.doctor_bpw - 1e-06

@pytest.mark.parametrize('strategy', list(gf._ISLAND_STRATEGIES))
def test_protected_island_selection_is_deterministic(strategy):
    w = _second_light_lowrank()
    resid = w - gf.pq_pack(w, dim=32, subspaces=4, k=32).recon
    act = np.abs(np.random.default_rng(5).standard_normal((10, w.shape[1]))).astype(np.float32)
    sens = np.abs(np.random.default_rng(6).standard_normal(w.shape)).astype(np.float32)
    a = gf.select_protected_islands(w, resid, strategy=strategy, budget_frac=0.05, activation=act, sensitivity=sens)
    b = gf.select_protected_islands(w, resid, strategy=strategy, budget_frac=0.05, activation=act, sensitivity=sens)
    assert np.array_equal(a['row_indices'], b['row_indices'])
    assert a['n_islands'] == int(np.ceil(0.05 * w.shape[0]))
    assert np.all(np.diff(a['row_indices']) > 0)

def test_protected_islands_are_billed_and_increase_bpw():
    w = _second_light_lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=32)
    isl = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.05, strategy='residual_energy')
    assert 'protected_islands' in isl.ledger.items
    assert isl.whole_artifact_bpw > base.whole_artifact_bpw
    assert isl.whole_artifact_bpw >= isl.base_bpw + isl.doctor_bpw - 1e-06
    assert isl.overhead_bpw >= -1e-06
    rows = isl.config['pq_codes']['island_rows']
    assert np.allclose(isl.recon[rows], w[rows], atol=1e-06)

def test_more_islands_cost_more_bytes():
    w = _second_light_lowrank()
    small = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.02)
    big = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.1)
    assert big.whole_artifact_bpw > small.whole_artifact_bpw

def test_protected_island_artifact_executes_with_exact_rows():
    w = _second_light_lowrank()
    isl = gf.pack_pq_protected_islands(w, dim=32, subspaces=4, k=32, budget_frac=0.05)
    x = np.random.default_rng(7).standard_normal(w.shape[1]).astype(np.float32)
    y = gf.pq_execute(isl, x)
    rows = isl.config['pq_codes']['island_rows']
    assert np.allclose(y[rows], w[rows] @ x, atol=0.0001)
    assert np.linalg.norm(y - isl.recon @ x) / (np.linalg.norm(isl.recon @ x) + 1e-12) < 0.0001

@pytest.mark.parametrize('treatment', list(gf._DOCTOR_TREATMENTS))
def test_doctor_stays_within_budget(treatment):
    w = _second_light_lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    budget = 8000
    out = gf.doctor_pq(w, base, byte_budget=budget, strategy=treatment)
    assert out['treatment'] == treatment
    assert out['added_bytes'] <= budget and out['within_budget']
    assert out['quality_delta'] >= -1e-06

def test_doctor_reduces_error_for_repairing_treatments():
    """The three treatments that add explicit residual capacity must strictly reduce weight error."""
    w = _second_light_lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    for treatment in ('residual_codebook', 'sparse_residual', 'protected_island_expansion'):
        out = gf.doctor_pq(w, base, byte_budget=8000, strategy=treatment)
        assert out['quality_delta'] > 0.0, treatment
        assert out['err_after'] < out['err_before'], treatment

def test_doctor_respects_tiny_budget():
    """With a tiny budget the Doctor must still not exceed it (bounded, honest)."""
    w = _second_light_lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    out = gf.doctor_pq(w, base, byte_budget=200, strategy='sparse_residual')
    assert out['added_bytes'] <= 200

def test_doctor_rejects_unknown_treatment():
    w = _second_light_lowrank()
    base = gf.pq_pack(w, dim=32, subspaces=4, k=16)
    with pytest.raises(ValueError):
        gf.doctor_pq(w, base, byte_budget=8000, strategy='not_a_treatment')

def test_cpu_metal_parity_holds():
    w = _second_light_lowrank()
    par = gf.pq_cpu_metal_parity(w, dim=32, k=32, subspaces=4)
    assert par['authoritative'] == 'cpu'
    assert par['within_tol'], par
    assert par['ranking_match']
    assert par['pass_match']
    assert 0.0 <= par['assignment_agreement'] <= 1.0
    assert par['relerr_delta'] <= par['tol']

def test_selftest_reports_second_light_signals():
    out = gf.selftest()
    assert out['pq_family_verbs'] == 7
    assert out['pq_execute_within_tol']
    assert out['pq_deterministic_bytes']
    assert out['pq_islands_increase_bpw']
    assert out['pq_doctor_reduces_error']
    assert out['pq_cpu_metal_within_tol']
