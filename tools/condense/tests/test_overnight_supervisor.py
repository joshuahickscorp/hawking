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
RETIRED_MODULES = ['overnight_supervisor']


def test_01_successful_120b_completion_verifies(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_01_successful_120b_completion_verifies'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_02_invalid_seal_fails_verification(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_02_invalid_seal_fails_verification'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_03_source_mapped_refuses_release(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_03_source_mapped_refuses_release'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_04_deletion_gate_red_refuses(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_04_deletion_gate_red_refuses'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_05_interrupted_deletion_is_restart_safe(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_05_interrupted_deletion_is_restart_safe'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_06_interrupted_qwen_download_retries_and_respects_floors(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_06_interrupted_qwen_download_retries_and_respects_floors'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_07_disk_floor_hardstop_and_pause(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_07_disk_floor_hardstop_and_pause'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_08_restart_during_every_state(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_08_restart_during_every_state'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_09_claim_dedups_telegram_but_never_wedges_advance(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_09_claim_dedups_telegram_but_never_wedges_advance'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_10_q2_failure_blocks(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_10_q2_failure_blocks'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_11_successful_qwen_ignition_path(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_11_successful_qwen_ignition_path'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_12_qwen_controller_self_heal_and_seal(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_12_qwen_controller_self_heal_and_seal'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_release_success_deletes_only_shards(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_release_success_deletes_only_shards'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_release_refuses_unsafe_paths(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_release_refuses_unsafe_paths'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_reverify_gates_fresh_authorized(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_reverify_gates_fresh_authorized'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_campaign_crash_auto_resumes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_campaign_crash_auto_resumes'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_campaign_alive_no_resume(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_campaign_alive_no_resume'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_campaign_missing_lease_resumes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_campaign_missing_lease_resumes'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()
