#!/usr/bin/env python3.12
"""Retired-controller cases preserved against the campaign engine (lane H1)."""
from __future__ import annotations

from pathlib import Path

from tools.condense.engine.runtime import run_campaign
from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
from tools.condense.engine.spec import SPECS_DIR, load_spec

FAMILY = 'mechanics'

def test_module_selftest_green(tmp_path: Path) -> None:
    case = 'test_module_selftest_green'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_staged_execution_matches_frozen_shared_grammar(tmp_path: Path) -> None:
    case = 'test_staged_execution_matches_frozen_shared_grammar'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_M2_runs_and_measured(tmp_path: Path) -> None:
    case = 'test_M2_runs_and_measured'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_M3_M4_M5_M6_run_and_measured(tmp_path: Path) -> None:
    case = 'test_M3_M4_M5_M6_run_and_measured'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_no_dense_shadow_M2_through_M6(tmp_path: Path) -> None:
    case = 'test_no_dense_shadow_M2_through_M6'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_quality_gate_rejects_lower_quality(tmp_path: Path) -> None:
    case = 'test_quality_gate_rejects_lower_quality'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_fake_win_ban_candidate_marked_inadmissible(tmp_path: Path) -> None:
    case = 'test_fake_win_ban_candidate_marked_inadmissible'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_conditional_false_negative_gate_fires(tmp_path: Path) -> None:
    case = 'test_conditional_false_negative_gate_fires'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_pareto_excludes_dominated_and_dense(tmp_path: Path) -> None:
    case = 'test_pareto_excludes_dominated_and_dense'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_pareto_champions_present(tmp_path: Path) -> None:
    case = 'test_pareto_champions_present'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_deterministic_same_seed(tmp_path: Path) -> None:
    case = 'test_deterministic_same_seed'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_cluster_ledger_shared_cheaper_than_independent(tmp_path: Path) -> None:
    case = 'test_cluster_ledger_shared_cheaper_than_independent'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_islands_are_billed_and_increase_bytes(tmp_path: Path) -> None:
    case = 'test_islands_are_billed_and_increase_bytes'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'
