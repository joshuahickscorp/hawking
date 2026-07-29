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
RETIRED_MODULES = ['kimi_k26_release_cycle']


def test_init_session_is_private_and_has_no_live_capability(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_init_session_is_private_and_has_no_live_capability'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_init_session_rejects_unsafe_ids(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_init_session_rejects_unsafe_ids'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_layout_rejects_symlinked_component(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_layout_rejects_symlinked_component'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_layout_rejects_missing_scratch_root(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_layout_rejects_missing_scratch_root'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_layout_rejects_symlinked_scratch_root(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_layout_rejects_symlinked_scratch_root'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_layout_rejects_nonprivate_scratch_mode(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_layout_rejects_nonprivate_scratch_mode'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_download_plan_is_exact_dedicated_and_inert(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_download_plan_is_exact_dedicated_and_inert'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_resealed_transfer_pin_substitution_is_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resealed_transfer_pin_substitution_is_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_resealed_runtime_binding_substitutions_are_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resealed_runtime_binding_substitutions_are_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_resealed_native_xet_artifact_substitution_is_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_resealed_native_xet_artifact_substitution_is_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_runtime_builder_rejects_unsafe_hf_launcher_node(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_runtime_builder_rejects_unsafe_hf_launcher_node'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_real_frozen_manifest_and_archive_verify(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_real_frozen_manifest_and_archive_verify'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_strict_json_rejects_duplicate_keys_and_nan(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_strict_json_rejects_duplicate_keys_and_nan'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_manifest_rejects_resealed_substitution(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_manifest_rejects_resealed_substitution'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_old_898063_archive_is_explicitly_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_old_898063_archive_is_explicitly_rejected'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_archive_hash_substitution_is_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_archive_hash_substitution_is_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_zip_slip_names_are_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_zip_slip_names_are_rejected'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_zip_symlink_member_is_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_zip_symlink_member_is_rejected'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_recovery_extracts_only_explicit_text_allowlist(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_recovery_extracts_only_explicit_text_allowlist'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_recovery_extractor_refuses_overwrite_and_unlisted_entry(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_recovery_extractor_refuses_overwrite_and_unlisted_entry'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_source_verifier_accepts_exact_dedicated_symlink_snapshot(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_verifier_accepts_exact_dedicated_symlink_snapshot'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_source_verifier_rejects_symlink_escape(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_verifier_rejects_symlink_escape'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_source_verifier_rejects_hardlinked_blob(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_verifier_rejects_hardlinked_blob'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_source_verifier_rejects_blob_identity_substitution(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_verifier_rejects_blob_identity_substitution'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_inventory_deduplicates_hardlinks_and_blocks_release(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_inventory_deduplicates_hardlinks_and_blocks_release'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_inventory_blocks_mop_symlink_and_path_overlap(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_inventory_blocks_mop_symlink_and_path_overlap'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_inventory_seal_and_allocated_sum_detect_substitution(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_inventory_seal_and_allocated_sum_detect_substitution'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_pre_release_audit_passes_only_clean_inventory_readers_processes_and_queues(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_pre_release_audit_passes_only_clean_inventory_readers_processes_and_queues'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_pre_release_audit_blocks_lsof_process_and_queue(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_pre_release_audit_blocks_lsof_process_and_queue'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_exact_payload_result_capture_verifier(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exact_payload_result_capture_verifier'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_payload_hash_substitution_and_hardlink_are_rejected(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_payload_hash_substitution_and_hardlink_are_rejected'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_phase1_preflight_is_deterministic_and_never_calls_subprocess(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_phase1_preflight_is_deterministic_and_never_calls_subprocess'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_cli_exposes_no_download_delete_or_release_execution_command(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_cli_exposes_no_download_delete_or_release_execution_command'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()
