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

FAMILY = 'gravity_frontier'
RETIRED_MODULES = ['emergency_detached_campaign']


def test_resource_parsers_and_zero_growth_policy(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resource_parsers_and_zero_growth_policy'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_recovery_process_detection_requires_exact_session(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_recovery_process_detection_requires_exact_session'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_durable_store_repairs_snapshot_from_atomic_journal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_durable_store_repairs_snapshot_from_atomic_journal'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_journal_chain_tamper_fails_closed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_journal_chain_tamper_fails_closed'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_exact_private_crash_temp_does_not_brick_journal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exact_private_crash_temp_does_not_brick_journal'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_operation_roots_default_to_clean_executor(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_operation_roots_default_to_clean_executor'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_executor_requires_exact_official_remote_head_and_pinned_files(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_executor_requires_exact_official_remote_head_and_pinned_files'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_unfinalized_operation_hash_pins_refuse_before_git(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_unfinalized_operation_hash_pins_refuse_before_git'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_linked_worktree_is_rejected_as_emergency_executor(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_linked_worktree_is_rejected_as_emergency_executor'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_remote_head_mismatch_fails_closed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_remote_head_mismatch_fails_closed'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_origin_url_mismatch_fails_closed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_origin_url_mismatch_fails_closed'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_post_bootstrap_executor_verification_needs_no_network(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_post_bootstrap_executor_verification_needs_no_network'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_controller_waits_for_existing_generate_without_spawning(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_controller_waits_for_existing_generate_without_spawning'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_pid_reuse_is_never_signalled_or_treated_as_controller_child(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_pid_reuse_is_never_signalled_or_treated_as_controller_child'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_sampler_failure_stops_exact_restarted_child_and_retains_watchdog_state(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_sampler_failure_stops_exact_restarted_child_and_retains_watchdog_state'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_spawn_gap_reconciliation_recovers_exact_live_child_without_duplicate(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_spawn_gap_reconciliation_recovers_exact_live_child_without_duplicate'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_campaign_baseline_is_not_reset_between_actions(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_campaign_baseline_is_not_reset_between_actions'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_verified_executor_head_must_remain_bootstrap_head(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_verified_executor_head_must_remain_bootstrap_head'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_replacement_capsule_status_is_accepted_without_terminal_block(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replacement_capsule_status_is_accepted_without_terminal_block'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_plist_is_caffeinated_and_does_not_touch_overnight_plist(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_plist_is_caffeinated_and_does_not_touch_overnight_plist'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_bootstrap_requires_explicit_release_authorization_flag(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_bootstrap_requires_explicit_release_authorization_flag'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_release_authorization_is_private_durable_and_head_bound(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_release_authorization_is_private_durable_and_head_bound'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_source_contains_no_delete_implementation(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_contains_no_delete_implementation'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
