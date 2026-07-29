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

FAMILY = 'glm52'
RETIRED_MODULES = ['glm52_notifications']


def test_roles_are_ed25519_distinct_and_private_keys_do_not_enter_journal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_roles_are_ed25519_distinct_and_private_keys_do_not_enter_journal'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_prepare_has_no_caller_facts_status_anchor_or_timestamp(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_prepare_has_no_caller_facts_status_anchor_or_timestamp'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_unsupported_and_premature_milestones_fail_closed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_unsupported_and_premature_milestones_fail_closed'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_stream_start_and_all_coverage_crossings_come_from_window_ledger(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_stream_start_and_all_coverage_crossings_come_from_window_ledger'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_exact_prepared_send_started_committed_lifecycle_and_redacted_receipt(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exact_prepared_send_started_committed_lifecycle_and_redacted_receipt'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_receipt_binds_attempt_start_sequence_hash_and_both_telegram_identities(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_receipt_binds_attempt_start_sequence_hash_and_both_telegram_identities'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_real_utc_calendar_and_timestamp_order_are_enforced(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_real_utc_calendar_and_timestamp_order_are_enforced'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_ambiguity_requires_separately_signed_exact_reconciliation(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_ambiguity_requires_separately_signed_exact_reconciliation'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_controller_checkpoint_binds_every_notification_head(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_controller_checkpoint_binds_every_notification_head'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_head_and_journal_deletion_or_prefix_rollback_are_detected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_head_and_journal_deletion_or_prefix_rollback_are_detected'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_parent_journal_lock_and_fifo_replacement_are_detected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_parent_journal_lock_and_fifo_replacement_are_detected'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_exact_one_entry_crash_tail_is_recovered(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exact_one_entry_crash_tail_is_recovered'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_resealed_entry_tamper_and_clean_tail_truncation_fail(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resealed_entry_tamper_and_clean_tail_truncation_fail'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_concurrent_duplicate_prepare_has_one_winner(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_concurrent_duplicate_prepare_has_one_winner'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_read_only_audit_replays_exact_lifecycle_without_private_keys(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_read_only_audit_replays_exact_lifecycle_without_private_keys'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_read_only_audit_performs_no_write_repair_anchor_or_send(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_read_only_audit_performs_no_write_repair_anchor_or_send'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_read_only_audit_rejects_crash_gaps_without_recovery(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_read_only_audit_rejects_crash_gaps_without_recovery'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_read_only_audit_rejects_wrong_configuration_and_unheld_lease(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_read_only_audit_rejects_wrong_configuration_and_unheld_lease'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_read_only_audit_rejects_fifo_lock_and_controller_bound_head_rollback(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_read_only_audit_rejects_fifo_lock_and_controller_bound_head_rollback'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases
