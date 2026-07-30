"""Foundry lifecycle/harness/subbit tables (S6 F2 C3 densified)."""
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
FOUNDRY = Path(__file__).resolve().parents[1]
from lab.operators import quality_contract as qc
from lab.operators import storage_modes as sm
import json
from lab.bench_harness import HARNESS_VERSION, MeasurementRecorder, ReceiptWriter, ReportRenderer, load_spec, validate_spec
from lab.bench_harness.runner import Runner
from lab.bench_harness.spec import SPEC_SCHEMA, ExperimentSpec
from fractions import Fraction
from lab.operators import subbit_closure as sc
GB = 10 ** 9

def _capability_evidence(**over):
    ev = {'real_parent_forward': True, 'real_packed_forward': True, 'split': 'holdout', 'n_tokens': 4096, 'domains': list(qc.PROTECTED_QUALITY_DOMAINS),
        'metrics': {'mean_symmetric_kl': 0.04, 'argmax_agreement': 0.97}}
    ev.update(over)
    return ev

def test_mode_signature_has_no_parameter_count():
    import inspect
    assert 'param' not in ''.join(inspect.signature(sm.choose_mode).parameters)

def test_397b_bf16_is_bigger_on_disk_than_1t_int4():
    """Precision drives disk size. Params would rank these backwards."""
    bf16_397b = 397 * 10 ** 9 * 2
    int4_1t = 1000 * 10 ** 9 * 60 // 100
    assert bf16_397b > int4_1t
    free, reserve = (700 * GB, 0)
    small_params = sm.choose_mode(bf16_397b, free, reserve, '397B-bf16')
    big_params = sm.choose_mode(int4_1t, free, reserve, '1T-int4')
    assert small_params['mode'] == sm.VULTURE_SHARD_SERIAL
    assert big_params['mode'] == sm.FULL_DISK_RESIDENT
    assert 'parameter_count' in big_params['never_decided_from']

def test_mode_ladder_by_free_space():
    manifest = 65 * GB
    assert sm.choose_mode(manifest, 200 * GB, 50 * GB, 'F0')['mode'] == sm.FULL_DISK_RESIDENT
    assert sm.choose_mode(manifest, 90 * GB, 50 * GB, 'F0')['mode'] == sm.VULTURE_SHARD_SERIAL
    assert sm.choose_mode(manifest, 60 * GB, 50 * GB, 'F0')['mode'] == sm.BOUNDED_REMOTE_RANGE

def test_reserve_is_honoured_not_borrowed():
    d = sm.choose_mode(100 * GB, 120 * GB, 30 * GB, 'F0')
    assert d['usable_bytes'] == 90 * GB
    assert d['mode'] != sm.FULL_DISK_RESIDENT

def test_expert_cache_cap_stays_at_measured_20gb():
    """64 GB gave 0 evictions and drove swap to 906 MB free on a lockstep pass."""
    assert sm.EXPERT_CACHE_CAP_BYTES == 20 * GB

@pytest.mark.parametrize('args', [(0, GB, 0, 'F0'), (-1, GB, 0, 'F0'), (GB, -1, 0, 'F0'), (GB, GB, 0, '')])
def test_bad_inputs_rejected(args):
    with pytest.raises((ValueError, TypeError)):
        sm.choose_mode(*args)

def test_param_count_passed_as_bytes_is_a_type_error():
    with pytest.raises(TypeError):
        sm.choose_mode(116000000000.0, 200 * GB, 0, 'F0')

def test_vulture_serial_pass_achieves_full_coverage():
    shards = {'s1': ['a.w', 'a.b'], 's2': ['b.w'], 's3': ['c.w', 'c.b']}
    manifest = [t for ts in shards.values() for t in ts]
    packed = []
    for tensors in shards.values():
        packed.extend(tensors)
        sm.assert_shard_release_ordered(tensors, packed)
    assert sm.assert_full_coverage(manifest, packed)['complete']

def test_skipped_tensor_is_a_coverage_error():
    with pytest.raises(sm.CoverageError):
        sm.assert_full_coverage(['a', 'b', 'c'], ['a', 'b'])
    assert sm.vulture_coverage(['a', 'b', 'c'], ['a', 'b'])['missing'] == ['c']

def test_tensor_outside_the_manifest_is_a_coverage_error():
    with pytest.raises(sm.CoverageError):
        sm.assert_full_coverage(['a'], ['a', 'ghost'])

def test_shard_released_before_its_tensors_are_packed():
    with pytest.raises(sm.CoverageError):
        sm.assert_shard_release_ordered(['a', 'b'], ['a'])
PINNED = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'

def test_release_ok_with_pinned_revision_and_retained_metadata():
    r = sm.vulture_release_ok(PINNED, {'config', 'index', 'tokenizer'})
    assert r['ok'] and r['independent_of_capability_gate'] is True

def test_release_blocked_without_pinned_immutable_revision():
    for rev in ('main', 'v1.0', '', None, 'b5c939de'):
        assert not sm.vulture_release_ok(rev, {'config', 'index', 'tokenizer'})['ok']

def test_release_blocked_without_retained_rehydration_metadata():
    r = sm.vulture_release_ok(PINNED, {'config', 'index'})
    assert not r['ok'] and 'tokenizer' in r['reasons'][0]

def test_release_does_not_require_a_passing_artifact():
    """F0 was NEGATIVE at sub-bit and still released. Release is not the gate."""
    assert sm.vulture_release_ok(PINNED, {'config', 'index', 'tokenizer'})['ok']

def test_gate_needs_both_conditions():
    ev = _capability_evidence()
    assert qc.evaluate({'mean_symmetric_kl': 0.1, 'argmax_agreement': 0.95}, ev)['passed']
    assert not qc.evaluate({'mean_symmetric_kl': 0.11, 'argmax_agreement': 0.99}, ev)['passed']
    assert not qc.evaluate({'mean_symmetric_kl': 0.01, 'argmax_agreement': 0.94}, ev)['passed']

def test_partial_metrics_are_not_a_gate():
    with pytest.raises(qc.ContractViolation):
        qc.evaluate({'mean_symmetric_kl': 0.01})

def test_extras_are_recorded_not_gating():
    r = qc.evaluate({'mean_symmetric_kl': 0.5, 'argmax_agreement': 0.4, 'logit_cosine': 0.999, 'top5_overlap': 0.8, 'candidate_perplexity': 12.3},
        _capability_evidence())
    assert r['passed'] is False and r['recorded']['logit_cosine'] == 0.999

def test_classify_each_rung():
    assert qc.classify({'metrics': {'bpw': 0.77}}) == 'PHYSICAL'
    assert qc.classify({'metrics': {'weight_recon_error': 0.66}}) == 'FUNCTIONAL_PROXY'
    assert qc.classify({'per_layer_divergence': [0.1, 0.2]}) == 'LAYER'
    assert qc.classify(_capability_evidence(split='validation')) == 'SHORT_END_TO_END'
    assert qc.classify(_capability_evidence(n_tokens=88)) == 'SHORT_END_TO_END'
    assert qc.classify(_capability_evidence()) == 'CAPABILITY'

def test_overclaim_is_rejected():
    proxy = {'metrics': {'weight_recon_error': 0.668}}
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed('CAPABILITY', proxy)
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed('LAYER', proxy)
    assert qc.assert_not_overclaimed('FUNCTIONAL_PROXY', proxy)['ok']
    assert qc.assert_not_overclaimed('PHYSICAL', proxy)['ok']

def test_unknown_class_rejected():
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_overclaimed('VIBES', _capability_evidence())

def test_only_capability_selects_a_frontier():
    assert qc.may_select_frontier(_capability_evidence())
    assert not qc.may_select_frontier(_capability_evidence(n_tokens=88))
    assert not qc.may_select_frontier({'metrics': {'bpw': 0.3}})
    assert not qc.evaluate({'mean_symmetric_kl': 0.01, 'argmax_agreement': 0.99}, _capability_evidence(split='calibration'))['may_select_frontier']

def test_protected_domains_required_for_capability():
    partial = _capability_evidence(domains=['code'])
    assert qc.classify(partial) == 'SHORT_END_TO_END'

def test_splits_must_be_disjoint_and_present():
    ok = {'calibration': ['a'], 'validation': ['b'], 'holdout': ['c']}
    assert qc.assert_splits_disjoint(ok)['sizes']['holdout'] == 1
    with pytest.raises(qc.ContractViolation):
        qc.assert_splits_disjoint({'calibration': ['a'], 'validation': ['a'], 'holdout': ['c']})
    with pytest.raises(qc.ContractViolation):
        qc.assert_splits_disjoint({'calibration': ['a'], 'validation': ['b'], 'holdout': []})

def test_unchanged_contract_matches_the_seal():
    assert qc.assert_not_weakened(dict(qc.CONTRACT))['change'] == 'unchanged'
    assert qc.contract_hash() == qc.SEALED_CONTRACT_SHA256

def test_weakening_after_a_failure_is_rejected():
    failed = qc.evaluate({'mean_symmetric_kl': 0.42, 'argmax_agreement': 0.61})
    assert failed['passed'] is False
    for weaker in ({**qc.CONTRACT, 'max_mean_symmetric_kl': 0.5}, {**qc.CONTRACT, 'min_argmax_agreement': 0.6}, {**qc.CONTRACT, 'min_capability_tokens': 88}):
        with pytest.raises(qc.ContractViolation):
            qc.assert_not_weakened(weaker)

def test_removing_a_threshold_is_weakening():
    stripped = {k: v for k, v in qc.CONTRACT.items() if k != 'min_argmax_agreement'}
    with pytest.raises(qc.ContractViolation):
        qc.assert_not_weakened(stripped)

def test_tightening_is_allowed_and_reseals():
    tighter = {**qc.CONTRACT, 'max_mean_symmetric_kl': 0.05}
    r = qc.assert_not_weakened(tighter)
    assert r['change'] == 'tightened' and r['sealed_sha256'] != qc.SEALED_CONTRACT_SHA256
sys.path.insert(0, str(FOUNDRY))

def _spec_dict(**over):
    base = {'schema': SPEC_SCHEMA, 'id': 'unit_probe', 'title': 'unit probe', 'artifact_dir': 'artifacts/runs/lab_harness_unit', 'stages': [{'id': 'echo',
        'argv': ['python3.12', '-c', "print('ok')"], 'skip_if_done': False}]}
    base.update(over)
    return base

@pytest.mark.parametrize('raw,ok', [(_spec_dict(), True), ({'schema': SPEC_SCHEMA, 'id': 'x', 'stages': [{'id': 'a', 'shell': 'true'}]}, True),
    ({'schema': 'wrong',
     'id': 'x', 'stages': [{'id': 'a', 'shell': 'true'}]}, False), ({'schema': SPEC_SCHEMA, 'id': 'x', 'stages': []}, False), ({'schema': SPEC_SCHEMA,
    'stages': [{'id': 'a', 'shell': 'true'}]}, False), ({'schema': SPEC_SCHEMA, 'id': 'x', 'stages': [{'id': 'a'}]}, False), ({'schema': SPEC_SCHEMA, 'id': 'x',
     'stages': [{'id': 'a', 'argv': ['true']}]}, True)])
def test_spec_validation_matrix(raw, ok):
    if ok:
        validate_spec(raw)
        ExperimentSpec.from_dict(raw)
    else:
        with pytest.raises(ValueError):
            validate_spec(raw)

def test_load_spec_roundtrip(tmp_path: Path):
    p = tmp_path / 's.json'
    p.write_text(json.dumps(_spec_dict()), encoding='utf-8')
    spec = load_spec(p)
    assert spec.id == 'unit_probe'
    assert len(spec.stages) == 1

def test_measurement_recorder(tmp_path: Path):
    path = tmp_path / 'm.jsonl'
    with MeasurementRecorder(path) as rec:
        rec.stage_start('s1')
        rec.metric('tps_proxy', 0, note='fence')
        rec.stage_end('s1', rc=0, seconds=0.1, state='done')
    rows = MeasurementRecorder(path).read_all()
    kinds = [r['kind'] for r in rows]
    assert kinds == ['stage_start', 'metric', 'stage_end']

def test_receipt_writer_stable_hash(tmp_path: Path):
    w = ReceiptWriter('hawking.lab.receipt.v1')
    r1 = w.build(experiment_id='e', stages=[{'id': 'a', 'state': 'done'}], status='complete')
    r2 = w.build(experiment_id='e', stages=[{'id': 'a', 'state': 'done'}], status='complete')
    assert 'content_sha256' in r1 and len(r1['content_sha256']) == 64
    out = tmp_path / 'r.json'
    w.write(out, r1)
    assert json.loads(out.read_text())['experiment_id'] == 'e'

def test_report_renderer_contains_stages():
    md = ReportRenderer().render_md(title='T', experiment_id='e', status={'state': 'complete', 'current_stage': 'a', 'uptime_seconds': 1}, stages=[{'id': 'a',
        'state': 'done', 'rc': 0, 'seconds': 0.5, 'note': ''}], measures=[{'kind': 'metric', 'name': 'x', 'value': 1}])
    assert 'complete' in md and '| a |' in md and ('x' in md)

def test_runner_dry_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'tools' / 'foundry' / 'lab_harness').mkdir(parents=True)
    spec = ExperimentSpec.from_dict(_spec_dict(artifact_dir='art', stages=[{'id': 'echo', 'shell': 'echo hi', 'skip_if_done': False}]))
    rc = Runner(spec, root=tmp_path, dry_run=True).run()
    assert rc == 0
    assert (tmp_path / 'art' / 'receipt.json').is_file()
    assert (tmp_path / 'art' / 'report.md').is_file()
    assert (tmp_path / 'art' / 'status.json').is_file()

def test_runner_real_echo(tmp_path: Path):
    spec = ExperimentSpec.from_dict(_spec_dict(artifact_dir='art', stages=[{'id': 'echo', 'argv': ['python3.12', '-c', 'print(1)'], 'skip_if_done': False}]))
    rc = Runner(spec, root=tmp_path, dry_run=False).run()
    assert rc == 0
    receipt = json.loads((tmp_path / 'art' / 'receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['stages'][0]['state'] == 'done'

def test_matrix_expands_slots(tmp_path: Path):
    spec = ExperimentSpec.from_dict(_spec_dict(artifact_dir='art', matrix=[{'n': '1'}, {'n': '2'}], stages=[{'id': 'echo', 'shell': 'echo {n}',
        'skip_if_done': False}]))
    rc = Runner(spec, root=tmp_path, dry_run=True).run()
    assert rc == 0
    receipt = json.loads((tmp_path / 'art' / 'receipt.json').read_text())
    ids = [s['id'] for s in receipt['stages']]
    assert any(('n=1' in i for i in ids)) and any(('n=2' in i for i in ids))

def test_harness_version_nonzero():
    assert HARNESS_VERSION and SPEC_SCHEMA.startswith('hawking.lab')
INV = sc.build_inventory()

def test_inventory_matches_the_real_parent_byte_for_byte():
    assert INV.params == sc.SEALED_PARAMS
    assert INV.params * 2 == sc.SEALED_TOTAL_SIZE_BYTES
    assert INV.tensors == sc.SEALED_TENSOR_COUNT
    assert sum((o.params for o in INV.organs.values())) == INV.params

def test_inventory_cross_check_rejects_a_wrong_denominator():
    bad = {'metadata': {'total_size': sc.SEALED_TOTAL_SIZE_BYTES + 2}, 'weight_map': {}}
    with pytest.raises(sc.ClosureError, match='cross-check failed'):
        sc.build_inventory(None, bad)

def test_every_tensor_must_be_allocated_exactly_once():
    half = sc.Variant(name='half_allocated', pressure_taken_from='n/a', rationale='n/a', bands=(sc.Band('norms', 10, sc.Native()),))
    with pytest.raises(sc.ClosureError, match='exactly once'):
        sc.bill(half, INV)

@pytest.mark.parametrize('variant', sc.VARIANTS, ids=lambda v: v.name)
def test_variant_is_legal_under_the_ceiling(variant):
    receipt = sc.check_ceiling(variant, INV)
    bpw = Fraction(receipt['complete_bits'], INV.params)
    assert bpw <= sc.CEILING
    assert receipt['legal'] is True
    assert receipt['headroom_bits'] >= 0

def test_there_are_at_least_three_legal_variants_taking_pressure_in_different_places():
    places = {v.pressure_taken_from for v in sc.VARIANTS}
    assert len(sc.VARIANTS) >= 3
    assert len(places) == len(sc.VARIANTS)

def test_nothing_is_excluded_as_overhead():
    """Every named slot is present and the total is their exact sum plus the reserve."""
    receipt = sc.bill(sc.V1, INV)
    slots = receipt['components_bits']
    assert set(slots) == {'indices', 'codebooks', 'scales', 'metadata', 'alignment', 'protected_islands', 'doctor', 'pass_through_tensors', 'packaging',
        'runtime_tables'}
    assert receipt['complete_bits'] == sum(slots.values()) + receipt['reserve_bits']
    for slot in ('codebooks', 'scales', 'metadata', 'alignment', 'packaging', 'pass_through_tensors'):
        assert slots[slot] > 0, slot

def test_expert_only_bpw_is_never_the_model_rate():
    receipt = sc.bill(sc.V1, INV)
    expert_bits = sum((receipt['per_organ'][o]['bits'] for o in ('expert_gate', 'expert_up', 'expert_down')))
    expert_params = sum((INV.organ(o).params for o in ('expert_gate', 'expert_up', 'expert_down')))
    expert_only = Fraction(expert_bits, expert_params)
    whole = Fraction(receipt['complete_bits'], INV.params)
    assert expert_only < whole
    assert receipt['expert_only_bpw_is_not_the_model_rate'] is True

def test_a1_1p0_replay_is_rejected():
    receipt = sc.bill(sc.A1_REPLAY, INV)
    assert receipt['legal'] is False
    assert Fraction(receipt['complete_bits'], INV.params) > sc.CEILING
    with pytest.raises(Exception) as exc:
        sc.check_ceiling(sc.A1_REPLAY, INV)
    assert 'ceiling' in str(exc.value).lower()

def test_this_ledger_is_stricter_than_the_sealed_campaign_ledger():
    """A rebudget that is only legal under a slacker ruler is not a rebudget."""
    replay = Fraction(sc.bill(sc.A1_REPLAY, INV)['complete_bits'], INV.params)
    assert replay > sc.A1_SEALED_COMPLETE_BPW

def test_raising_any_expert_rate_one_notch_breaks_the_tightest_variant():
    """The budget is genuinely binding, not padded."""
    over = sc.Variant(name='C2_plus_one_notch', pressure_taken_from='test', rationale='test', bands=tuple(sc._experts(sc.VQ(16, 4, 64), sc.VQ(16, 4, 64),
        sc.VQ(32, 7, 4)) + sc._nonexpert(sc.VQ(8, 1, 256), sc.VQ(8, 2, 256), sc.VQ(16, 3, 256))))
    with pytest.raises(Exception, match='(?i)ceiling'):
        sc.check_ceiling(over, INV)

def test_native_everything_is_wildly_illegal():
    native = sc.Variant(name='all_native', pressure_taken_from='nowhere', rationale='the parent itself', bands=tuple((sc.Band(n, o.tensors, sc.Native()) for n,
        o in INV.organs.items())))
    receipt = sc.bill(native, INV)
    assert receipt['complete_bpw_float'] > 16.0

@pytest.mark.parametrize('rate', sc.SUBBIT_PROBE_RATES, ids=lambda q: f'{q.numerator}/{q.denominator}')
def test_subbit_probes_run_at_cheap_tiers_and_are_refused_at_expensive_ones(rate):
    for tier in sc.CHEAP_TIERS:
        assert sc.may_schedule(rate, tier)['allowed'] is True
    for tier in sc.EXPENSIVE_TIERS:
        verdict = sc.may_schedule(rate, tier)
        assert verdict['allowed'] is False
        assert 'until a serious one-bit method is selected' in verdict['reason']

def test_selecting_a_one_bit_method_unlocks_subbit_capability_compute():
    before = sc.may_schedule('1/2', 'capability', one_bit_method_selected=False)
    after = sc.may_schedule('1/2', 'capability', one_bit_method_selected=True)
    assert before['allowed'] is False
    assert after['allowed'] is True

def test_the_ceiling_rate_itself_may_spend_capability_compute():
    assert sc.may_schedule('1/1', 'capability')['allowed'] is True

@pytest.mark.parametrize('rate', ['6/5', '5/4', '3/2', '2/1', '3/1'])
def test_no_tier_may_schedule_above_the_ceiling(rate):
    for tier in sc.FIDELITY_TIERS:
        verdict = sc.may_schedule(rate, tier, one_bit_method_selected=True)
        assert verdict['allowed'] is False
        assert 'upward bracketing is REJECTED' in verdict['reason']

def test_unknown_tier_is_refused_not_treated_as_cheap():
    assert sc.may_schedule('1/2', 'vibes')['allowed'] is False

def test_rates_are_exact_rationals_and_decimals_are_refused():
    with pytest.raises(sc.ClosureError, match='exact rational'):
        sc._parse_rate('0.85')
    assert sc._parse_rate('17/20') == Fraction(17, 20)

def test_vq_rate_is_exact_and_geometry_derived():
    assert sc.VQ(16, 4, 32).rate == Fraction(5, 4)
    assert sc.VQ(32, 1, 256).rate == Fraction(1, 4)
    assert sc.VQ(16, 2, 16).rate == Fraction(1, 2)
    with pytest.raises(sc.ClosureError):
        sc.VQ(16, 4, 30)
    with pytest.raises(sc.ClosureError):
        sc.VQ(16, 17, 16)

def test_complete_bpw_is_reported_as_an_exact_fraction():
    for variant in sc.VARIANTS:
        receipt = sc.bill(variant, INV)
        num, den = receipt['complete_bpw_exact'].split('/')
        exact = Fraction(int(num), int(den))
        assert exact == Fraction(receipt['complete_bits'], INV.params)
        assert exact.denominator > 1 or receipt['complete_bits'] % INV.params == 0
        assert abs(float(exact) - receipt['complete_bpw_float']) < 1e-12

def test_bit_counts_are_integers_never_floats():
    receipt = sc.bill(sc.V5, INV)
    assert all((isinstance(v, int) for v in receipt['components_bits'].values()))
    assert isinstance(receipt['complete_bits'], int)

def test_omission_does_not_shrink_the_denominator():
    receipt = sc.bill(sc.V4, INV)
    assert receipt['original_weight_count'] == INV.params
    gate = receipt['per_organ']['expert_gate']
    assert gate['params'] == INV.organ('expert_gate').params
    assert gate['realized_organ_bpw_float'] < 1.5
    assert receipt['complete_bpw_float'] <= 1.0

def test_program_is_well_formed_and_orders_methods_decisive_first():
    prog = sc.build_program(INV)
    assert len(prog['legal_variants']) == len(sc.VARIANTS)
    assert all((r['legal'] for r in prog['legal_variants']))
    assert any((not c['legal'] for c in prog['rejected_candidates']))
    ranks = [m['rank'] for m in prog['closure_methods']]
    assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))
    assert prog['decisive_first_order'][0] == 'M01_row_norm_stratified_codebooks'

def test_every_method_declares_the_five_required_fields():
    for m in sc.CLOSURE_METHODS:
        for key in ('changes', 'why_not_falsified', 'byte_cost', 'first_tier', 'falsification', 'changes_source'):
            assert m.get(key) not in (None, ''), (m['id'], key)
        assert m['first_tier'] in sc.FIDELITY_TIERS

def test_source_changing_methods_are_flagged():
    flagged = set(sc.source_changing_methods())
    assert {'M12_compressibility_training', 'M13_quantization_aware_training', 'M14_distillation_into_the_one_bit_student',
        'M15_learned_sharing_generated_weights',
         'M05_structured_expert_omission'} <= flagged
    assert 'M01_row_norm_stratified_codebooks' not in flagged

def test_program_schedules_no_escape_hatch():
    prog = sc.build_program(INV)
    blob = str(prog).lower()
    for banned in ('escape receipt', 'safety anchor', 'quality anchor', 'upward bracket'):
        assert banned not in str(prog['legal_variants']).lower()
        assert banned not in str(prog['closure_methods']).lower()
    assert '1.2' not in str([r['complete_bpw_float'] for r in prog['legal_variants']])
    assert blob
if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
