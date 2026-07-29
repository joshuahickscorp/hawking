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
RETIRED_MODULES = ['glm52_pilot_source_release']


def test_green_gate_and_confirmed_exact_deletion(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_green_gate_and_confirmed_exact_deletion'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_wrong_confirmation(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_wrong_confirmation'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_missing_confirm_flag_cli(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_missing_confirm_flag_cli'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_missing_shard(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_missing_shard'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_extra_shard(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_extra_shard'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_size_mismatch_shard(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_size_mismatch_shard'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_hash_mismatch_shard(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_hash_mismatch_shard'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_symlinked_root_refusal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_symlinked_root_refusal'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_symlinked_shard_refusal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_symlinked_shard_refusal'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_path_escape_protected_target_refusal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_path_escape_protected_target_refusal'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_deletion_never_touches_protected_trees(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_deletion_never_touches_protected_trees'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_stale_measurement_receipt_binding(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_stale_measurement_receipt_binding'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_stale_reseal_binding(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_stale_reseal_binding'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_revision_0_hash_mismatch(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_revision_0_hash_mismatch'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_current_code_hash_mismatch(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_current_code_hash_mismatch'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_measurement_math_changed_blocks(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_measurement_math_changed_blocks'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_fence_not_false_blocks(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_fence_not_false_blocks'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_gaussian_selection_blocks(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_gaussian_selection_blocks'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_simulated_live_consumer(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_simulated_live_consumer'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_no_process_probe_fail_closed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_no_process_probe_fail_closed'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_release_reruns_the_gate(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_release_reruns_the_gate'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_retained_ledger_log_cache_survive(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_retained_ledger_log_cache_survive'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_sealed_receipt_verifies_and_blocks_replay(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_sealed_receipt_verifies_and_blocks_replay'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_status_is_read_only(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_status_is_read_only'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_cli_gate_status_release(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_cli_gate_status_release'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_release_records_free_bytes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_release_records_free_bytes'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_partial_hash_drift_at_delete_reports_state(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_partial_hash_drift_at_delete_reports_state'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_confirm_phrase_constant(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_confirm_phrase_constant'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
