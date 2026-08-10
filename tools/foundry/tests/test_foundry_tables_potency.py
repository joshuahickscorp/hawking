"""Foundry potency/post-parent tables (S6 F2 C3 densified)."""
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
from lab.operators import gravity_potency as gp
import json
import os

# Live post-parent review (restored). Gates must execute, not skip by construction.
import importlib.util as _ilu
_ppr_path = FOUNDRY / "post_parent_review.py"
_spec = _ilu.spec_from_file_location("hawking_post_parent_review", _ppr_path)
ppr = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ppr)

@pytest.fixture()
def foundry(tmp_path, monkeypatch):
    monkeypatch.setenv('HAWKING_FOUNDRY_DIR', str(tmp_path))
    gp.seal_v1()
    gp.seal_atlas()
    return tmp_path

def _review(**over):
    body = {'schema': gp.SCHEMA_REVIEW, 'parent_id': 'qwen3-235b:F1', 'reviewer': 'frontier', 'verdict': 'accept', 'capability_receipt_sha256': 'a' * 64,
        'mean_symmetric_kl': 0.04, 'argmax_agreement': 0.97}
    body.update(over)
    return gp.seal_field(body, 'sha256')

def test_selftest_green():
    assert gp.selftest()['ok'] is True

def test_v1_is_sealed_and_immutable(foundry):
    gen = gp.latest_generation()
    assert gen['method_version'] == 'GRAVITY_METHOD_V1'
    assert gen['candidate_priors']['organ_inversion']['action'].startswith('allocate bits to gate/up')
    assert gen['storage_policy']['expert_cache_cap_bytes'] == 20 * 1024 ** 3
    doc = gp.read_json_safe(gp.registry_path())
    doc['generations']['GRAVITY_METHOD_V1']['storage_policy']['expert_cache_cap_bytes'] = 64 * 1024 ** 3
    gp.atomic_write_json(gp.registry_path(), doc)
    with pytest.raises(gp.PotencyError, match='mutated or unsealed'):
        gp.load_registry()

def test_promotion_refused_without_review(foundry):
    with pytest.raises(gp.PotencyError, match='evidence review is missing'):
        gp.promote({'generation': {'kernel_set': {}}})

def test_promotion_refused_with_unsealed_review(foundry):
    bad = dict(_review())
    bad['reviewer'] = 'tampered-after-seal'
    with pytest.raises(gp.PotencyError, match='not sealed'):
        gp.promote({'review': bad, 'generation': {'kernel_set': {}}})

def test_promotion_refused_when_contract_weakened(foundry):
    with pytest.raises(gp.PotencyError, match='weakened'):
        gp.promote({'review': _review(), 'generation': {'quality_contract': {'mean_symmetric_kl_max': '1/4'}}})
    with pytest.raises(gp.PotencyError, match='weakened'):
        gp.promote({'review': _review(), 'generation': {'quality_contract': {'next_token_argmax_agreement_min': '9/10'}}})
    with pytest.raises(gp.PotencyError, match='weakened'):
        gp.promote({'review': _review(), 'generation': {'quality_contract': {'min_capability_tokens': 88}}})

def test_promotion_with_sealed_review_seals_v2(foundry):
    gen = gp.promote({'review': _review(), 'generation': {'source_revision': {'parent': 'Qwen/Qwen3-235B-A22B'}}})
    assert gen['method_version'] == 'GRAVITY_METHOD_V2'
    assert gen['promoted_by_review_sha256'] == _review()['sha256']
    assert 'qwen3-235b:F1' in gen['parents_completed']
    assert gp.sealed(gen, 'sha256')
    assert gp.load_registry()['generations']['GRAVITY_METHOD_V1']['generation'] == 1

def test_ledger_append_and_read(foundry):
    gp.append_potency(gp.potency_row('gpt-oss-120b:F0', lowest_physical_bpw='3/10', lowest_capability_passing_bpw=None, source_bytes=65 * 10 ** 9))
    gp.append_potency(gp.potency_row('qwen3-235b:F1', lowest_physical_bpw='2/5'))
    assert len(gp.read_potency()) == 2
    row = gp.read_potency('gpt-oss-120b:F0')[0]
    assert row['lowest_physical_bpw'] == '3/10'
    assert row['lowest_capability_passing_bpw'] is None
    assert row['energy_joules'] is None
    assert gp.sealed(row, 'sha256')

def test_unknown_axis_refused(foundry):
    with pytest.raises(gp.PotencyError, match='unknown potency axes'):
        gp.potency_row('x', vanity_score=9000)

def test_report_prints_vector_and_refuses_one_number(foundry):
    gp.append_potency(gp.potency_row('gpt-oss-120b:F0', lowest_physical_bpw='3/10'))
    text = gp.report_potency()
    for axis in gp.POTENCY_AXES:
        assert axis in text
    assert 'REFUSED' in text
    with pytest.raises(gp.PotencyError, match='potency is a vector'):
        gp.collapse_to_score(gp.read_potency())
PREV = {'parent_id': 'gpt-oss-120b:F0', 'lowest_credible_bpw': '2/5', 'start_rate': '3/5'}
CEILING_METHODS = ['quantization_aware_training', 'distillation']

def test_no_senility_passes_an_aggressive_program(foundry):
    out = gp.check_no_senility({'parent_id': 'qwen3-235b:F1', 'start_rate': '3/5', 'ceiling_methods': CEILING_METHODS, 'rates': ['1/1', '3/5', '2/5', '1/3',
        '1/4']}, PREV)
    assert out['ok'] is True, out['failures']

def test_no_senility_fails_a_timid_program(foundry):
    out = gp.check_no_senility({'parent_id': 'qwen3-235b:F1', 'start_rate': '7/10', 'ceiling_methods': ['quantization_aware_training'], 'rates': ['1/1',
        '7/10']}, PREV)
    assert out['ok'] is False
    joined = ' '.join(out['failures'])
    assert 'materially distinct method family at the ceiling' in joined
    assert 'lowest credible region' in joined
    assert 'lower-rate stress point' in joined

def test_no_senility_fails_size_justified_rate_raise(foundry):
    out = gp.check_no_senility({'parent_id': 'deepseek-685b:F2', 'start_rate': '7/10',
        'start_rate_reason': '685B is much larger than the 120B parent, start safer',
         'ceiling_methods': CEILING_METHODS, 'rates': ['1/1', '7/10', '2/5', '1/3', '1/4']}, PREV)
    assert out['ok'] is False
    assert any(('size argument' in f for f in out['failures']))

def test_no_senility_fails_any_upward_start_rate_move(foundry):
    out = gp.check_no_senility({'parent_id': 'deepseek-685b:F2', 'start_rate': '7/10', 'ceiling_methods': CEILING_METHODS, 'rates': ['1/1', '7/10', '2/5',
        '1/3',
         '1/4']}, PREV)
    assert any(('upward rate change is never legal' in f for f in out['failures']))

def test_no_senility_rejects_off_ladder_and_rounded_rates(foundry):
    out = gp.check_no_senility({'rates': ['1/1', '9/17', '1/4']}, None)
    assert any(('not on the exact rate ladder' in f for f in out['failures']))
    out = gp.check_no_senility({'rates': ['0.85']}, None)
    assert any(('exact rational' in f for f in out['failures']))

def _full_sweep(rate):
    return [{'rate': rate, 'method_family': f, 'exhausted': True} for f in gp.METHOD_FAMILY_ORDER]

def test_rate_discipline_allows_lowering_after_full_sweep(foundry):
    out = gp.check_rate_discipline(_full_sweep('1/1') + _full_sweep('1/2'))
    assert out['ok'] is True, out['failures']
    assert out['may_lower_rate'] is True
    assert out['may_raise_rate'] is False

def test_rate_discipline_fails_premature_lowering(foundry):
    history = _full_sweep('1/1')[:3] + [{'rate': '1/2', 'method_family': 'quantization_aware_training'}]
    out = gp.check_rate_discipline(history)
    assert out['ok'] is False
    assert 'rate lowered 1/1 -> 1/2' in out['failures'][0]
    assert 'allocation' in out['failures'][0]

def test_rate_discipline_fails_out_of_order_family(foundry):
    out = gp.check_rate_discipline([{'rate': '1/4', 'method_family': 'allocation'}])
    assert out['ok'] is False
    assert 'before' in out['failures'][0]

def test_rate_discipline_names_the_next_family(foundry):
    out = gp.check_rate_discipline(_full_sweep('1/1')[:2])
    assert out['next_method_family'] == 'distillation'
    assert out['may_lower_rate'] is False

def test_rate_discipline_still_reads_legacy_lever_histories(foundry):
    legacy = [{'rate': '1/4', 'lever': lever, 'exhausted': True} for lever in gp.LEVER_ORDER]
    out = gp.check_rate_discipline(legacy)
    assert out['ok'] is True, out['failures']
    assert out['ordering'] == 'lever'

def test_atlas_blocks_every_dead_lever(foundry):
    for lever in gp.load_atlas()['entries']:
        assert gp.atlas_check(lever)['blocked'] is True

def test_atlas_block_carries_the_killing_measurement(foundry):
    out = gp.atlas_check('inter_expert_redundancy')
    assert out['blocked'] is True
    assert '1e-4' in out['killed_by']
    assert 'cosine' in out['reopen_condition']

def test_atlas_reopens_only_on_a_new_parent_diagnosis(foundry):
    same_parent = {'parent_id': 'gpt-oss-120b:F0', 'reopens': ['large_expert_cache'], 'measurement': 'cache hit rate 0.31'}
    assert gp.atlas_check('large_expert_cache', same_parent)['blocked'] is True
    no_measurement = {'parent_id': 'deepseek-685b:F2', 'reopens': ['large_expert_cache']}
    assert gp.atlas_check('large_expert_cache', no_measurement)['blocked'] is True
    good = {'parent_id': 'deepseek-685b:F2', 'reopens': ['large_expert_cache'], 'measurement': 'measured cross-layer expert cache hit rate 0.31'}
    out = gp.atlas_check('large_expert_cache', good)
    assert out['blocked'] is False
    assert out['reopened_by'] == 'deepseek-685b:F2'

def test_atlas_does_not_block_an_alive_lever(foundry):
    assert gp.atlas_check('row_norm_stratification')['blocked'] is False
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(_HERE))

def synthetic_evidence(run_status='honest_boundary_sealed'):
    return {'schema': 'hawking.foundry.parent_evidence.v1', 'parent': {'id': 'synth-9b', 'label': '9B', 'generation': 'FT'}, 'run_status': run_status,
        'representation': {'winners': [{'name': 'vq_d32_k65536', 'rate_bpw': 0.5, 'rel_error': 0.668}], 'rate_response': [{'rate_bpw': 0.4, 'rel_error': 0.71},
        {'rate_bpw': 0.5, 'rel_error': 0.668}]}, 'organ_sensitivity': {'organs': {'gate': {'sensitivity': 'HIGH'}, 'up': {'sensitivity': 'HIGH'},
        'down': {'sensitivity': 'LOWER'}}, 'dominant_failure_organ': 'gate', 'inversion_confirmed': True}, 'doctor': {'successes': [{'target': 'down'}],
        'failures': [{'target': 'gate'}]}, 'routing': {'calibration_tokens': 1000, 'required_calibration_tokens': 1000},
        'activation': {'inter_expert_mean_pairwise_cosine': 0.0001}, 'quality': {'capability_gate': {'mean_symmetric_kl_max': 0.1,
        'next_token_argmax_agreement_min': 0.95}, 'result': {'selected_frontier': None}, 'failures': [{'probe': 'real_forward', 'domain': 'logits'}],
        'probes': ['real_forward'], 'domains_collapsing_first': ['logits']}, 'runtime': {'timings': [{'stage': 'pack', 'seconds': 10}],
        'dominant_bottleneck': 'expert streaming'}, 'resources': {'memory_floor_gib': 20}, 'source_format': {'lessons': [{'id': 'packed_blocks'}],
        'decoder_requirements': ['packed block decode']}, 'storage': {'lessons': [{'id': 'release_after_harvest'}]}, 'assumptions': [{'id': 'organ_inversion',
        'statement': 'mlp1 sensitive', 'verdict': 'CONFIRMED', 'evidence': 'x'}, {'id': 'cache_64', 'statement': 'big cache helps', 'verdict': 'FALSIFIED',
        'evidence': 'y'},
        {'id': 'row_norm', 'statement': 'stratification helps', 'verdict': 'OPEN'},
    ],
        'methods': [
            {'id': 'm1', 'name': 'row-norm stratification', 'status': 'UNTESTED', 'transfer_breadth': 2.0},
        ],
        'next_parent': {
            'storage': {'free_gib': 500, 'required_gib': 120, 'headroom_gib': 50},
        },
    }

def adapters(acks=None):
    return [{'id': 'qwen3_moe', 'consumes_parent_lessons': ['synth-9b'], 'unverified_assumptions': ['row_norm'], 'falsification_plan': (
        're-measure gate/up rel'
        '_err at matched rate; if down beats gate the inversion prior is dead'),
         'subbit_closure_plan': {'rates': ['1/1', '1/2'], 'method_families': ['quantization_aware_training', 'distillation']}, 'rebase_acks': acks or {}}]

def test_generate_writes_every_artifact(tmp_path):
    out = str(tmp_path / 'review')
    written = ppr.generate(synthetic_evidence(), out, adapters())
    for name in ppr.REQUIRED_PER_PARENT:
        assert os.path.exists(os.path.join(out, 'SYNTH_9B_' + name)), name
    for name in ppr.REQUIRED_GLOBAL:
        assert os.path.exists(os.path.join(out, name)), name
    assert len(written) == len(ppr.REQUIRED_PER_PARENT) + len(ppr.REQUIRED_GLOBAL)
    harvest = json.load(open(written['VULTURE_HARVEST.json']))
    assert harvest['organ_sensitivity']['dominant_failure_organ'] == 'gate'
    assert harvest['provisional'] is False
    ids = {n['id'] for n in harvest['negative_transfer_constraints']}
    assert {'inter_expert_redundancy_zero', 'entropy_coding_pq_indices', 'aggressive_expert_cache'} <= ids
    review = json.load(open(written['GLOBAL_METHODOLOGY_REVIEW.json']))
    assert [a['id'] for a in review['assumptions_falsified']] == ['cache_64']
    assert set(review['questions']) == {k for k, _ in ppr._REVIEW_QUESTIONS}
    md = open(written['GLOBAL_METHODOLOGY_REVIEW.md']).read()
    assert 'dominant failure organ: gate' in md
    for ch in ('—', '–', '·'):
        assert ch not in md
    matrix = json.load(open(written['ADAPTER_REBASE_MATRIX.json']))
    entry = matrix['adapters']['qwen3_moe']
    assert entry['sensitive_organ_prior'] == 'gate'
    assert entry['cache_policy']['expert_cache_cap_gib'] == 20
    assert entry['routing_calibration_plan']['min_calibration_tokens'] == 1000
    assert any((c.startswith('DO_NOT_ATTEMPT:') for c in entry['candidate_ordering']))
    assert entry['rate_priors']['not_a_selection'] is True
    resource = json.load(open(written['RESOURCE_REBASE.json']))
    assert resource['cache_policy']['expert_cache_cap_gib'] == 20
    promo = json.load(open(written['GRAVITY_METHOD_PROMOTION.json']))
    assert all((m['potency'] <= 0 for m in promo['retired']))
    assert 'gravity_potency' in promo['potency_backend']
    ledger = [json.loads(x) for x in open(written['PROVIDER_ADAPTER_LESSON_LEDGER.jsonl']) if x.strip()]
    assert ledger and all((r['parent_id'] == 'synth-9b' for r in ledger))
    ppr.generate(synthetic_evidence(), out, adapters())
    again = [json.loads(x) for x in open(written['PROVIDER_ADAPTER_LESSON_LEDGER.jsonl']) if x.strip()]
    assert len(again) == len(ledger)

def test_refuses_weakened_capability_gate():
    ev = synthetic_evidence()
    ev['quality']['capability_gate']['mean_symmetric_kl_max'] = 0.25
    with pytest.raises(ValueError, match='weakened capability gate'):
        ppr.validate_evidence(ev)

def test_find_stale_adapters_flags_unrebased(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    stale = ppr.find_stale_adapters(ev['parent'], adapters(), matrix)
    assert stale[0]['adapter_id'] == 'qwen3_moe'
    assert 'no rebase_ack' in stale[0]['reasons'][0]
    acked = adapters({'synth-9b': {'prescription_digest': d}})
    assert ppr.find_stale_adapters(ev['parent'], acked, matrix) == []
    drifted = ppr.find_stale_adapters(ev['parent'], adapters({'synth-9b': {'prescription_digest': 'deadbeef'}}), matrix)
    assert 'digest drift' in drifted[0]['reasons'][0]

def test_stale_when_adapter_does_not_declare_falsification(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    a = adapters({'synth-9b': {'prescription_digest': d}})
    a[0]['falsification_plan'] = None
    a[0]['unverified_assumptions'] = None
    a[0]['consumes_parent_lessons'] = []
    reasons = ppr.find_stale_adapters(ev['parent'], a, matrix)[0]['reasons']
    assert len(reasons) == 3

def test_shipped_registry_starts_stale_against_f0():
    reg = json.load(open(os.path.join(_HERE, 'adapters', 'tier_a_registry.json')))
    ev = ppr.load_evidence(os.path.join(_HERE, 'evidence', 'f0_gpt_oss_120b.json'))
    stale = ppr.find_stale_adapters(ev['parent'], reg['adapters'])
    assert len(stale) == len(reg['adapters'])

def test_gate_blocks_next_parent_before_review(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / 'review')
    state = ppr.build_gate_state(out, ev, adapters())
    ok, reasons = ppr.can_launch_next_parent(state)
    assert ok is False
    assert any(('missing review artifacts' in r for r in reasons))
    assert any(('stale adapters' in r for r in reasons))

def test_gate_allows_next_parent_after_review_and_acks(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / 'review')
    written = ppr.generate(ev, out, adapters())
    matrix = json.load(open(written['ADAPTER_REBASE_MATRIX.json']))
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    state = ppr.build_gate_state(out, ev, adapters({'synth-9b': {'prescription_digest': d}}), matrix)
    ok, reasons = ppr.can_launch_next_parent(state)
    assert (ok, reasons) == (True, [])
    held = ppr.build_gate_state(out, ev, adapters({'synth-9b': {'prescription_digest': d}}), matrix, heavy_lease_held=True)
    ok2, reasons2 = ppr.can_launch_next_parent(held)
    assert ok2 is False and any(('heavy lease' in r for r in reasons2))

def test_gate_blocks_on_provisional_review(tmp_path):
    ev = synthetic_evidence(run_status='in_flight')
    out = str(tmp_path / 'review')
    written = ppr.generate(ev, out, adapters())
    assert json.load(open(written['VULTURE_HARVEST.json']))['provisional'] is True
    matrix = json.load(open(written['ADAPTER_REBASE_MATRIX.json']))
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    state = ppr.build_gate_state(out, ev, adapters({'synth-9b': {'prescription_digest': d}}), matrix)
    ok, reasons = ppr.can_launch_next_parent(state)
    assert ok is False
    assert any(('provisional' in r for r in reasons))
    assert any(('not complete or honest_boundary_sealed' in r for r in reasons))

def test_download_may_run_concurrently_while_heavy_is_blocked(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / 'review')
    state = ppr.build_gate_state(out, ev, adapters(), storage=ev['next_parent']['storage'])
    assert ppr.can_launch_next_parent(state)[0] is False
    assert ppr.can_start_next_download(state) == (True, [])

def test_download_blocked_only_by_storage(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / 'review')
    ppr.generate(ev, out, adapters())
    tight = {'free_gib': 100, 'required_gib': 120, 'headroom_gib': 50}
    ok, reasons = ppr.can_start_next_download({'storage': tight})
    assert ok is False and 'insufficient storage' in reasons[0]
    ok2, reasons2 = ppr.can_start_next_download({'storage': {'free_gib': 500, 'required_gib': 1, 'download_in_flight': True}})
    assert ok2 is False and 'already in flight' in reasons2[0]

def test_f1_qwen_bundle_is_sealed_and_blocks_on_stale_adapters(tmp_path):
    """F1 sealed at honest_boundary_sealed. The review no longer blocks; adapters do."""
    ev = ppr.load_evidence(os.path.join(_HERE, 'evidence', 'f1_qwen3_235b.json'))
    assert ev['run_status'] == 'honest_boundary_sealed'
    out = str(tmp_path / 'review')
    written = ppr.generate(ev, out, adapters())
    harvest = json.load(open(written['VULTURE_HARVEST.json']))
    assert harvest['provisional'] is False
    assert harvest['organ_sensitivity']['dominant_failure_organ'] == 'gate'
    cross = json.load(open(written['CROSS_PARENT_TRANSFER_MATRIX.json']))
    assert cross['parents']['qwen3-235b-a22b-instruct-2507']['provisional'] is False
    assert harvest['capability_gate_result']['selected_frontier'] is None
    matrix = json.load(open(written['ADAPTER_REBASE_MATRIX.json']))
    priors = next(iter(matrix['adapters'].values()))['rate_priors']
    assert priors['search_start_bpw'] <= 1.0
    assert [r['rate_bpw'] for r in priors['above_ceiling_excluded']] == [1.007471652]
    ok, _ = ppr.can_launch_next_parent(ppr.build_gate_state(out, ev, []))
    assert ok is True
    with open(os.path.join(_HERE, 'adapters', 'tier_a_registry.json')) as fh:
        registry = json.load(fh)['adapters']
    ok, reasons = ppr.can_launch_next_parent(ppr.build_gate_state(out, ev, registry, matrix))
    assert ok is False and 'stale adapters' in reasons[0]

def test_f0_then_f1_accumulate_in_cross_parent_matrix(tmp_path):
    out = str(tmp_path / 'review')
    f0 = ppr.load_evidence(os.path.join(_HERE, 'evidence', 'f0_gpt_oss_120b.json'))
    f1 = ppr.load_evidence(os.path.join(_HERE, 'evidence', 'f1_qwen3_235b.json'))
    ppr.generate(f0, out, adapters())
    written = ppr.generate(f1, out, adapters())
    cross = json.load(open(written['CROSS_PARENT_TRANSFER_MATRIX.json']))
    assert set(cross['parents']) == {'gpt-oss-120b', 'qwen3-235b-a22b-instruct-2507'}
    assert all((p['dominant_failure_organ'] == 'gate' for p in cross['parents'].values()))
    assert cross['parents']['gpt-oss-120b']['provisional'] is False
