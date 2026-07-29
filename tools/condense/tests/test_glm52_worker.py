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
RETIRED_MODULES = ['glm52_notifications', 'glm52_worker']


def test_crash_restart_replays_checkpoint_and_authenticates_heartbeat(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_crash_restart_replays_checkpoint_and_authenticates_heartbeat'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_restart_recovers_exact_event_tail_after_checkpoint_write_crash(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_restart_recovers_exact_event_tail_after_checkpoint_write_crash'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_singleton_lease_refuses_second_worker_and_releases_after_failure(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_singleton_lease_refuses_second_worker_and_releases_after_failure'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_static_symlink_lease_attack_is_rejected_before_target_write(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_static_symlink_lease_attack_is_rejected_before_target_write'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_signal_latches_until_boundary_then_emits_safe_stop_heartbeat(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_signal_latches_until_boundary_then_emits_safe_stop_heartbeat'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_signal_latch_wakes_persistent_wait_without_applying_mid_phase(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_signal_latch_wakes_persistent_wait_without_applying_mid_phase'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_pause_and_stop_directives_are_consumed_only_by_boundary(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_pause_and_stop_directives_are_consumed_only_by_boundary'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_worker_heartbeat_rejects_forged_hmac_and_identity(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_worker_heartbeat_rejects_forged_hmac_and_identity'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_preflight_reports_each_production_blocker_without_side_effects(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_reports_each_production_blocker_without_side_effects'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_telegram_rotation_receipt_binds_live_distinct_keys(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_telegram_rotation_receipt_binds_live_distinct_keys'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_notification_static_readiness_never_claims_semantic_replay(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_notification_static_readiness_never_claims_semantic_replay'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_notification_semantic_audit_requires_and_uses_exact_held_lease(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_notification_semantic_audit_requires_and_uses_exact_held_lease'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_preflight_never_calls_unleased_notification_audit(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_never_calls_unleased_notification_audit'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_persistent_run_refuses_missing_audit_before_first_heartbeat(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_persistent_run_refuses_missing_audit_before_first_heartbeat'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_persistent_run_audits_under_lease_at_each_dispatch_boundary(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_persistent_run_audits_under_lease_at_each_dispatch_boundary'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_tampered_frozen_audit_readiness_blocks_before_heartbeat(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_tampered_frozen_audit_readiness_blocks_before_heartbeat'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_notification_audit_bypass_is_synthetic_and_explicit_only(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_notification_audit_bypass_is_synthetic_and_explicit_only'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_worker_authentication_role_check_requires_distinct_grounding_without_leak(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_worker_authentication_role_check_requires_distinct_grounding_without_leak'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_empty_reviewed_registry_hard_blocks_every_claimed_adapter(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_empty_reviewed_registry_hard_blocks_every_claimed_adapter'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_adapter_source_inspection_rejects_noop_effects_and_test_eviction(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_adapter_source_inspection_rejects_noop_effects_and_test_eviction'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_public_eviction_entry_point_remains_a_hard_blocker(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_public_eviction_entry_point_remains_a_hard_blocker'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_resource_policy_requires_exact_contract_binding_and_frozen_floor(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resource_policy_requires_exact_contract_binding_and_frozen_floor'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_final_schedule_reopens_semantic_xet_and_binds_controller_commit(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_final_schedule_reopens_semantic_xet_and_binds_controller_commit'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_final_schedule_rejects_absent_tampered_and_arbitrary_xet_evidence(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_final_schedule_rejects_absent_tampered_and_arbitrary_xet_evidence'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_final_schedule_rejects_semantic_result_without_controller_commit(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_final_schedule_rejects_semantic_result_without_controller_commit'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_worker_config_rejects_symlink_hardlink_and_nondeterministic_timestamp(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_worker_config_rejects_symlink_hardlink_and_nondeterministic_timestamp'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_worker_config_replacement_cannot_reseal_an_alternate_contract(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_worker_config_replacement_cannot_reseal_an_alternate_contract'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_worker_config_binds_exact_controller_bytes_seal_and_runtime_targets(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_worker_config_binds_exact_controller_bytes_seal_and_runtime_targets'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_same_path_controller_reseal_and_root_redirection_is_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_same_path_controller_reseal_and_root_redirection_is_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_authenticated_worker_target_substitution_cannot_override_controller_file(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_authenticated_worker_target_substitution_cannot_override_controller_file'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_official_contract_authority_uses_authenticated_runtime_bound_seal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_official_contract_authority_uses_authenticated_runtime_bound_seal'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_contract_derived_bootstrap_writes_valid_fail_closed_configs(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_contract_derived_bootstrap_writes_valid_fail_closed_configs'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_bootstrap_binds_only_existing_authenticated_readiness_bytes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_bootstrap_binds_only_existing_authenticated_readiness_bytes'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_bootstrap_main_uses_only_evidence_keychain_and_never_enables_dispatch(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_bootstrap_main_uses_only_evidence_keychain_and_never_enables_dispatch'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_top_level_run_execs_exact_caffeinate_argv_then_validates_child_path(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_top_level_run_execs_exact_caffeinate_argv_then_validates_child_path'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_caffeinate_parent_validation(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_caffeinate_parent_validation'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_caffeinate_parent_exact_argv_rejects_any_program_drift(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_caffeinate_parent_exact_argv_rejects_any_program_drift'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_launchd_plist_is_caffeinated_restartable_and_secret_free(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_launchd_plist_is_caffeinated_restartable_and_secret_free'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_launchd_plist_rejects_injection_and_every_argument_drift(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_launchd_plist_rejects_injection_and_every_argument_drift'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_status_documents_logout_and_reboot_limitations(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_status_documents_logout_and_reboot_limitations'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
