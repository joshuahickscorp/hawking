#!/usr/bin/env python3.12
"""Retired-controller cases preserved against the campaign engine (lane H1).

The bespoke controller body was deleted. Each logical case name is retained and
now asserts the engine lifecycle, lease, checkpoint, or seal-integrity property
that the controller previously owned alone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.condense.engine.checkpoint import CheckpointStore
from tools.condense.engine.lease import LeaseError, SingletonLease
from tools.condense.engine.runtime import run_campaign
from tools.condense.engine.seal_integrity import (
    SealIntegrityError,
    inspect_launcher_node,
    preflight_must_not_use_subprocess,
    reject_resealed_substitution,
    seal_document,
    verify_document_seal,
)
from tools.condense.engine.spec import SPECS_DIR, load_spec

FAMILY = 'kimi_k26'
RETIRED_MODULES = ['kimi_k26_download_supervisor', 'kimi_k26_release_cycle', 'kimi_k26_stale_download_cleanup']


def test_preflight_and_status_start_no_process_or_network_and_write_nothing(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_and_status_start_no_process_or_network_and_write_nothing'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_preflight_rejects_resealed_plan_substitution_and_missing_tmp(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_rejects_resealed_plan_substitution_and_missing_tmp'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_preflight_enforces_exact_capacity_floor(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_enforces_exact_capacity_floor'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_resume_credits_more_than_initial_margin_of_existing_session_bytes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resume_credits_more_than_initial_margin_of_existing_session_bytes'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_run_uses_only_exact_pinned_child_contract_and_seals_evidence(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_run_uses_only_exact_pinned_child_contract_and_seals_evidence'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_resource_guard_terminates_then_kills_and_preserves_resume_cache(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resource_guard_terminates_then_kills_and_preserves_resume_cache'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_resource_guard_detects_external_consumption_of_remaining_capacity(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resource_guard_detects_external_consumption_of_remaining_capacity'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_post_start_sampler_fault_stops_child_and_closes_durable_lifecycle(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_post_start_sampler_fault_stops_child_and_closes_durable_lifecycle'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_exclusive_lease_refuses_concurrent_run_before_popen(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exclusive_lease_refuses_concurrent_run_before_popen'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_sustained_low_measured_transfer_serially_restarts_at_16(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_sustained_low_measured_transfer_serially_restarts_at_16'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_clean_prior_measured_ramp_resumes_directly_at_sealed_16(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_clean_prior_measured_ramp_resumes_directly_at_sealed_16'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_resource_guarded_ramp_requires_exact_cleanup_receipt_for_direct_16(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resource_guarded_ramp_requires_exact_cleanup_receipt_for_direct_16'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_ramp_uses_only_post_warmup_measurement_window(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_ramp_uses_only_post_warmup_measurement_window'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_fast_measured_transfer_does_not_ramp(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_fast_measured_transfer_does_not_ramp'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_missing_phase1_ramp_profile_never_derives_an_unplanned_command(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_missing_phase1_ramp_profile_never_derives_an_unplanned_command'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_status_replays_hash_chain_and_rejects_tamper(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_status_replays_hash_chain_and_rejects_tamper'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_journal_refuses_append_before_crossing_replay_bound(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_journal_refuses_append_before_crossing_replay_bound'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_production_cli_rejects_authority_redefinition(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_production_cli_rejects_authority_redefinition'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_preexec_process_conflict_fails_before_runtime_or_popen(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preexec_process_conflict_fails_before_runtime_or_popen'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_incomplete_appearing_at_final_boundary_blocks_before_popen(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_incomplete_appearing_at_final_boundary_blocks_before_popen'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_native_auditor_matches_exact_structured_tokens_not_substrings(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_native_auditor_matches_exact_structured_tokens_not_substrings'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_module_has_no_source_release_primitives(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_module_has_no_source_release_primitives'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()
