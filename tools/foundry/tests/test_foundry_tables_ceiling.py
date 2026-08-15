"""Foundry one-bit ceiling tables (S6 F2 C3 densified)."""
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
import json
import os
from lab.operators import acquisition as acq
from lab.operators import gravity_potency as gp

# Live post-parent review (restored). Gates must execute, not skip by construction.
import importlib.util as _ilu
_ppr_path = FOUNDRY / "post_parent_review.py"
_spec = _ilu.spec_from_file_location("hawking_post_parent_review", _ppr_path)
ppr = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ppr)
from test_foundry_tables_potency import adapters, synthetic_evidence
from fractions import Fraction
from lab.operators import one_bit_ceiling as obc
ANCHOR_PROGRAM = {'parent_id': 'next:F2', 'rates': ['1.2', '1/1', '1/2'], 'ceiling_methods': ['quantization_aware_training', 'distillation'],
    'candidates': [{'id': 'A0_1p2', 'complete_bpw': '1.2', 'role': 'quality anchor'}]}

@pytest.fixture()
def foundry(tmp_path, monkeypatch):
    monkeypatch.setenv('HAWKING_FOUNDRY_DIR', str(tmp_path))
    gp.seal_v1()
    gp.seal_atlas()
    return tmp_path

def _review(**over):
    body = {'schema': gp.SCHEMA_REVIEW, 'parent_id': 'qwen3-235b:F1', 'reviewer': 'frontier', 'verdict': 'accept', 'capability_receipt_sha256': 'a' * 64}
    body.update(over)
    return gp.seal_field(body, 'sha256')

def test_ceiling_gate_names_the_illegal_rate(foundry):
    fails = gp.ceiling_failures(ANCHOR_PROGRAM)
    assert fails
    assert all(('6/5' in f for f in fails))

def test_no_senility_rejects_the_12_anchor(foundry):
    out = gp.check_no_senility(ANCHOR_PROGRAM, None)
    assert out['ok'] is False
    assert any(('6/5' in f and 'above the one-bit ceiling' in f for f in out['failures']))

def test_rate_discipline_rejects_the_12_anchor(foundry):
    out = gp.check_rate_discipline([{'rate': '1.2', 'method_family': 'quantization_aware_training'}])
    assert out['ok'] is False
    assert any(('above the one-bit ceiling' in f for f in out['failures']))

def test_next_parent_launch_gate_rejects_the_12_anchor(tmp_path):
    ev = synthetic_evidence()
    out = str(tmp_path / 'review')
    written = ppr.generate(ev, out, adapters())
    matrix = json.load(open(written['ADAPTER_REBASE_MATRIX.json']))
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    acked = adapters({'synth-9b': {'prescription_digest': d}})
    clean = ppr.build_gate_state(out, ev, acked, matrix)
    assert ppr.can_launch_next_parent(clean) == (True, [])
    dirty = ppr.build_gate_state(out, ev, acked, matrix, next_parent_program=ANCHOR_PROGRAM)
    ok, reasons = ppr.can_launch_next_parent(dirty)
    assert ok is False
    assert any(('one-bit ceiling' in r and '6/5' in r for r in reasons))

def test_acquisition_rejects_the_12_anchor(foundry):
    with pytest.raises(acq.CeilingViolation, match='6/5'):
        acq.propose({'rate': '1.2'})

def test_escape_receipt_and_safety_anchor_are_refused_everywhere(foundry):
    for key in ('escape_receipt', 'safety_anchor', 'quality_anchor'):
        program = {'rates': ['1/1', '1/2'], 'ceiling_methods': ['quantization_aware_training', 'distillation'], key: {'rate': '3/2'}}
        assert any((key in f for f in gp.ceiling_failures(program)))
        assert gp.check_no_senility(program, None)['ok'] is False
        assert any((key in r for r in ppr.ceiling_violations(program)))

def _sweep(rate, families=None):
    return [{'rate': rate, 'method_family': f, 'exhausted': True} for f in families or gp.METHOD_FAMILY_ORDER]

def test_upward_rate_change_is_never_legal(foundry):
    out = gp.check_rate_discipline(_sweep('1/2') + [{'rate': '1/1', 'method_family': 'quantization_aware_training'}])
    assert out['ok'] is False
    assert any(('upward rate change is never legal' in f for f in out['failures']))
    assert out['may_raise_rate'] is False

def test_downward_change_before_method_exhaustion_is_rejected(foundry):
    partial = _sweep('1/1', gp.METHOD_FAMILY_ORDER[:2])
    out = gp.check_rate_discipline(partial + [{'rate': '1/2', 'method_family': 'quantization_aware_training'}])
    assert out['ok'] is False
    assert any(('rate lowered 1/1 -> 1/2' in f and 'unexhausted' in f for f in out['failures']))

def test_downward_change_after_method_exhaustion_is_legal(foundry):
    out = gp.check_rate_discipline(_sweep('1/1') + [{'rate': '1/2', 'method_family': 'quantization_aware_training'}])
    assert out['ok'] is True, out['failures']

def test_ceiling_program_needs_distinct_methods_not_a_higher_rate(foundry):
    one_method = dict(ANCHOR_PROGRAM, rates=['1/1', '1/2'], candidates=[], ceiling_methods=['quantization_aware_training'])
    assert any(('materially distinct method family' in f for f in gp.check_no_senility(one_method, None)['failures']))
    dead = dict(one_method, ceiling_methods=['raw_weight_pq_vq', 'allocation'])
    assert any(('falsified raw-weight PQ/VQ family' in f for f in gp.check_no_senility(dead, None)['failures']))

def test_program_must_work_at_the_ceiling_and_stress_below(foundry):
    no_ceiling = {'rates': ['1/2', '1/4'], 'ceiling_methods': ['quantization_aware_training', 'distillation']}
    assert any(('does not work AT the ceiling' in f for f in gp.check_no_senility(no_ceiling, None)['failures']))
    no_stress = {'rates': ['1/1'], 'ceiling_methods': ['quantization_aware_training', 'distillation']}
    assert any(('no lower-rate stress point' in f for f in gp.check_no_senility(no_stress, None)['failures']))

def test_v2_promotion_still_requires_a_sealed_review(foundry):
    with pytest.raises(gp.PotencyError, match='evidence review is missing'):
        gp.promote_v2(None)
    tampered = dict(_review())
    tampered['verdict'] = 'accept-but-edited-after-seal'
    with pytest.raises(gp.PotencyError, match='not sealed'):
        gp.promote_v2(tampered)
    with pytest.raises(gp.PotencyError, match='verdict'):
        gp.promote_v2(_review(verdict='reject'))

def test_v2_binds_the_ceiling_objective_f1_and_the_atlas(foundry):
    gen = gp.promote_v2(_review())
    assert gen['method_version'] == 'GRAVITY_METHOD_V2'
    assert gp.sealed(gen, 'sha256')
    ceiling = gen['one_bit_ceiling']
    assert ceiling['identity']['label'] == '1/1'
    assert ceiling['nothing_excluded_as_overhead'] is True
    assert 'doctor_bytes' in ceiling['complete_bits_include']
    assert '1.2 safety anchor' in ceiling['forbidden_permanently']
    assert gen['objective']['subject_to'] == 'complete BPW <= 1/1'
    assert gen['objective']['rate_change_law'].startswith('downward only')
    f1 = gen['f1_negative_result']
    assert f1['family'] == 'raw_weight_pq_vq'
    assert f1['candidates']['A1_1p0']['complete_bpw'] == '1.0075'
    assert 'NOT evidence' in f1['scope']
    assert 'quantization_aware_training' in f1['not_bound_by_it']
    atlas = gen['negative_transfer_atlas']
    assert atlas['preserved'] is True
    assert 'raw_weight_pq_vq_at_one_bit' in atlas['entries']
    assert any(('1e-4' in p for p in atlas['pinned_findings']))
    assert all((gp.parse_bpw(r['label']) <= gp.ONE_BIT_CEILING for r in gen['rate_ladder']))

def test_atlas_blocks_raw_weight_pq_vq_at_one_bit(foundry):
    out = gp.atlas_check('raw_weight_pq_vq_at_one_bit')
    assert out['blocked'] is True
    assert '1.0075' in out['killed_by']
    assert 'never on raw weights' in out['reopen_condition']

def test_acquisition_cannot_construct_a_candidate_above_the_ceiling(foundry):
    with pytest.raises(acq.CeilingViolation, match='above the one-bit ceiling'):
        acq.Candidate('quantization_aware_training', '1.2', 0.9, 'asked nicely')
    with pytest.raises(acq.CeilingViolation, match='above the one-bit ceiling'):
        acq.Candidate('distillation', {'complete_bits': 12, 'original_weight_count': 10}, 0.9, 'raw bits')

def test_acquisition_refuses_a_dead_or_unknown_family(foundry):
    with pytest.raises(acq.CeilingViolation, match='falsified'):
        acq.Candidate('raw_weight_pq_vq', '1/1', 0.9, 'one more time')
    assert acq.propose({'families': ['raw_weight_pq_vq', 'vibes']}) == []

def test_acquisition_orders_by_expected_gain_at_a_fixed_ceiling(foundry):
    out = acq.propose({}, limit=7)
    assert [c.method_family for c in out] == list(gp.METHOD_FAMILY_ORDER)
    assert all((c.rate == gp.ONE_BIT_CEILING for c in out))
    gains = [c.expected_capability_gain for c in out]
    assert gains == sorted(gains, reverse=True)

def test_acquisition_only_lowers_the_rate_and_only_when_exhausted(foundry):
    at_ceiling = acq.propose({'rate': '1/1', 'exhausted': list(gp.METHOD_FAMILY_ORDER)[:2]})
    assert all((c.rate == gp.ONE_BIT_CEILING for c in at_ceiling))
    assert at_ceiling[0].method_family == 'distillation'
    lowered = acq.propose({'rate': '1/1', 'exhausted': list(gp.METHOD_FAMILY_ORDER)})
    assert lowered and all((c.rate < gp.ONE_BIT_CEILING for c in lowered))
    assert acq.next_rate_below(gp.Fraction(1, 4)) is None
    assert acq.propose({'rate': '1/4', 'exhausted': list(gp.METHOD_FAMILY_ORDER)}) == []

def test_acquisition_emit_assert_is_the_last_line(foundry):
    bad = object.__new__(acq.Candidate)
    object.__setattr__(bad, 'rate', gp.Fraction(6, 5))
    with pytest.raises(AssertionError, match='above the one-bit ceiling'):
        acq._emit([bad])

def test_proposal_document_declares_the_ceiling(foundry):
    doc = acq.proposal({}, limit=2)
    assert doc['ceiling'] == '1/1'
    assert doc['objective'] == 'maximize capability subject to complete BPW <= 1/1'
    assert all((c['complete_bpw'] == '1/1' for c in doc['candidates']))

def test_adapter_rebase_prescribes_the_ceiling_and_a_closure_plan(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    entry = matrix['adapters']['qwen3_moe']
    assert entry['one_bit_ceiling']['rule'] == 'complete_artifact_bits / original_weight_count <= 1/1'
    assert entry['subbit_closure_requirement']['objective'].startswith('maximize capability')
    assert entry['subbit_closure_plan']['method_families']
    assert entry['stopping_rules']['never'].startswith('raise the rate')

def test_adapter_without_a_closure_plan_is_stale(tmp_path):
    ev = synthetic_evidence()
    harvest = ppr.build_vulture_harvest(ev)
    matrix = ppr.build_adapter_rebase_matrix(ev, harvest, adapters())
    d = matrix['adapters']['qwen3_moe']['prescription_digest']
    a = adapters({'synth-9b': {'prescription_digest': d}})
    a[0]['subbit_closure_plan'] = None
    assert any(('subbit_closure_plan' in r for r in ppr.find_stale_adapters(ev['parent'], a, matrix)[0]['reasons']))
    b = adapters({'synth-9b': {'prescription_digest': d}})
    b[0]['subbit_closure_plan'] = {'rates': ['1.2']}
    assert any(('breaches the one-bit ceiling' in r for r in ppr.find_stale_adapters(ev['parent'], b, matrix)[0]['reasons']))

def test_a_method_carrying_an_above_ceiling_rate_is_never_promoted(tmp_path):
    ev = synthetic_evidence()
    ev['methods'].append({'id': 'anchor_1p2', 'status': 'CONFIRMED_MEASURED', 'transfer_breadth': 3.0, 'complete_bpw': '1.2'})
    promo = ppr.build_gravity_method_promotion(ev, ppr.build_vulture_harvest(ev))
    assert 'anchor_1p2' not in [m['id'] for m in promo['promoted']]
    retired = [m for m in promo['retired'] if m['id'] == 'anchor_1p2']
    assert retired and retired[0]['ceiling_violation']

def test_rate_priors_never_seed_a_search_above_the_ceiling(tmp_path):
    ev = synthetic_evidence()
    ev['representation']['rate_response'].append({'rate_bpw': 1.0075, 'rel_error': 0.5})
    priors = ppr._rate_priors(ev)
    assert priors['search_start_bpw'] <= 1.0
    assert priors['above_ceiling_excluded'] == [{'rate_bpw': 1.0075, 'rel_error': 0.5}]

def test_review_artifacts_carry_the_ceiling_law(tmp_path):
    ev = synthetic_evidence()
    ev['next_parent']['rates'] = ['1.2']
    out = str(tmp_path / 'review')
    written = ppr.generate(ev, out, adapters())
    review = json.load(open(written['GLOBAL_METHODOLOGY_REVIEW.json']))
    assert review['ceiling_law']['ceiling'] == '1/1'
    assert review['next_parent_ceiling_violations']
    md = open(written['GLOBAL_METHODOLOGY_REVIEW.md']).read()
    assert 'One-bit ceiling' in md
    for ch in ('—', '–', '·'):
        assert ch not in md
    assert os.path.exists(written['CROSS_PARENT_TRANSFER_MATRIX.json'])
    assert json.load(open(written['CROSS_PARENT_TRANSFER_MATRIX.json']))['ceiling_law']['ceiling'] == '1/1'
WEIGHTS = 1000000

def ledger(**over):
    """A complete ledger: every named slot declared, reserve declared, total 0 by default."""
    body = {c: 0 for c in obc.COMPONENTS}
    body[obc.RESERVE] = 0
    body.update(over)
    return obc.CompleteByteLedger(**body)

def test_a1_1p0_is_rejected_with_its_overage():
    with pytest.raises(obc.CeilingViolation) as exc:
        obc.assert_complete_bpw_le_one(obc.a1_1p0_ledger(), obc.A1_1P0_WEIGHTS)
    msg = str(exc.value)
    assert '1.007471652' in msg
    assert 'overage 0.007471652 BPW = 1756537856 bits' in msg
    assert 'do not raise the ceiling' in msg
    verdict = obc.a1_1p0_verdict()
    assert verdict['legal'] is False
    assert any(('one-bit ceiling violated' in r for r in verdict['reasons']))

def test_ledger_omitting_codebooks_is_incomplete_not_zero():
    fields = {c: 0 for c in obc.COMPONENTS if c != 'codebooks'}
    fields[obc.RESERVE] = 0
    with pytest.raises(obc.IncompleteLedger, match='codebooks'):
        obc.CompleteByteLedger(**fields)

def test_missing_reserve_is_incomplete():
    with pytest.raises(obc.IncompleteLedger, match=obc.RESERVE):
        obc.CompleteByteLedger(**{c: 0 for c in obc.COMPONENTS})

def test_declared_but_unset_component_is_incomplete():
    with pytest.raises(obc.IncompleteLedger, match='undeclared is not zero'):
        ledger(doctor=None)

def test_unknown_component_is_refused():
    with pytest.raises(obc.IncompleteLedger, match='unknown ledger components'):
        ledger(misc_overhead=10)

def test_negative_component_cannot_pay_for_another():
    with pytest.raises(obc.IncompleteLedger, match='negative bits'):
        ledger(packaging=-8)

def test_incomplete_ledger_dict_fails_candidate_validation():
    fields = {c: 0 for c in obc.COMPONENTS if c != 'codebooks'}
    fields[obc.RESERVE] = 0
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': fields})
    assert legal is False
    assert 'codebooks' in reasons[0]

def test_candidate_without_a_ledger_is_not_a_candidate():
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'reported_bpw': '1/2'})
    assert legal is False
    assert 'itemized ledger' in reasons[0]

@pytest.mark.parametrize('component', obc.COMPONENTS)
def test_each_named_component_counts_against_the_ceiling(component):
    assert obc.assert_complete_bpw_le_one(ledger(**{component: WEIGHTS}), WEIGHTS)['legal'] is True
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(ledger(**{component: WEIGHTS + 1}), WEIGHTS)

def test_complete_bits_sums_all_eleven_slots():
    full = ledger(**{c: 3 for c in obc.COMPONENTS}, **{obc.RESERVE: 5})
    assert full.itemized_bits() == 3 * len(obc.COMPONENTS)
    assert full.complete_bits() == 3 * len(obc.COMPONENTS) + 5

def test_one_plus_one_billionth_fails():
    w = 1000000000
    over = ledger(indices=w + 1)
    assert over.complete_bpw(w) == Fraction(w + 1, w) == Fraction(1) + Fraction(1, 10 ** 9)
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(over, w)
    tiny = 10 ** 17
    assert float(Fraction(tiny + 1, tiny)) == 1.0
    with pytest.raises(obc.CeilingViolation):
        obc.assert_complete_bpw_le_one(ledger(indices=tiny + 1), tiny)

def test_float_overage_is_not_rounded_away():
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': ledger(indices=WEIGHTS), 'target_bpw': 1.0000001})
    assert legal is False
    assert any(('upward bracketing is REJECTED' in r for r in reasons))

def test_exactly_one_with_a_declared_reserve_passes():
    lg = ledger(indices=WEIGHTS - 4096, codebooks=3000, scales=1000, metadata=40, packaging=40, runtime_tables=16, **{obc.RESERVE: 0})
    assert lg.complete_bits() == WEIGHTS
    receipt = obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    assert receipt['legal'] is True
    assert receipt['complete_bpw_exact'] == '1/1'
    legal, reasons = obc.is_legal_candidate({'candidate_id': 'legal_at_one', 'original_weight_count': WEIGHTS, 'ledger': lg, 'reported_bpw': '1/1',
        'target_bpw': '1/1'})
    assert (legal, reasons) == (True, [])

def test_reserve_is_charged_not_free():
    lg = ledger(indices=WEIGHTS, **{obc.RESERVE: 1})
    with pytest.raises(obc.CeilingViolation, match='overage'):
        obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    legal, _ = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': ledger(indices=WEIGHTS - 1, **{obc.RESERVE: 1})})
    assert legal is True

def test_sub_bit_candidate_is_legal():
    lg = ledger(indices=WEIGHTS // 2, codebooks=1000, **{obc.RESERVE: 500})
    receipt = obc.assert_complete_bpw_le_one(lg, WEIGHTS)
    assert receipt['legal'] is True
    assert float(receipt['complete_bpw_float']) < 0.51

def test_expert_only_bpw_cannot_substitute_for_whole_model():
    lg = ledger(indices=WEIGHTS // 2, pass_through_tensors=WEIGHTS, **{obc.RESERVE: 0})
    legal, reasons = obc.is_legal_candidate({'candidate_id': 'expert_only_claim', 'original_weight_count': WEIGHTS, 'ledger': lg, 'expert_only_bpw': '1/2',
        'reported_bpw': '1/2'})
    assert legal is False
    assert any(('understates the whole-model' in r for r in reasons))
    assert any(('reported_bpw' in r and '!=' in r for r in reasons))
    assert any(('one-bit ceiling violated' in r for r in reasons))

def test_partial_rate_without_a_whole_model_rate_is_refused():
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': ledger(indices=WEIGHTS // 4, **{obc.RESERVE: 0}),
        'payload_only_bpw': '1/4'})
    assert legal is False
    assert any(('may never stand in for the whole model' in r for r in reasons))

def test_non_whole_model_scope_is_refused():
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': ledger(**{obc.RESERVE: 0}), 'scope': 'experts_only'})
    assert legal is False
    assert any(('whole-model only' in r for r in reasons))

@pytest.mark.parametrize('anchor', ['6/5', '3/2', '2/1', '3/1'])
def test_upward_anchors_are_rejected(anchor):
    legal, reasons = obc.is_legal_candidate({'original_weight_count': WEIGHTS, 'ledger': ledger(indices=WEIGHTS // 2, **{obc.RESERVE: 0}),
        'target_bpw': anchor})
    assert legal is False
    assert any(('upward bracketing is REJECTED' in r for r in reasons))

def test_zero_or_negative_weight_count_is_refused():
    with pytest.raises(obc.IncompleteLedger, match='must be positive'):
        ledger(**{obc.RESERVE: 0}).complete_bpw(0)
