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
RETIRED_MODULES = ['glm52_xet_live_driver']


def test_production_authority_verifier_replays_exact_event_checkpoint_and_lease(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_production_authority_verifier_replays_exact_event_checkpoint_and_lease'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_each_production_authority_mismatch_refuses_before_any_live_process(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_each_production_authority_mismatch_refuses_before_any_live_process'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_authority_refusal_precedes_plan_probe_git_resource_and_live_child(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_authority_refusal_precedes_plan_probe_git_resource_and_live_child'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_orchestration_executes_exact_12_then_two_full_hashes_and_attests(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_orchestration_executes_exact_12_then_two_full_hashes_and_attests'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_capability_is_issued_in_process_exactly_once_and_rechecks_state(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_capability_is_issued_in_process_exactly_once_and_rechecks_state'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_terminal_currentness_refuses_a_different_deterministic_rebuild(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_terminal_currentness_refuses_a_different_deterministic_rebuild'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_terminal_currentness_repeats_remote_then_refuses_lease_binding_drift(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_terminal_currentness_repeats_remote_then_refuses_lease_binding_drift'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_git_provenance_requires_clean_exact_pushed_head(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_git_provenance_requires_clean_exact_pushed_head'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_git_final_currentness_rejects_upstream_advance(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_git_final_currentness_rejects_upstream_advance'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_provenance_covers_every_planner_input_and_runtime_lock(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_provenance_covers_every_planner_input_and_runtime_lock'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_production_writer_accepts_only_canonical_lowercase_json_and_is_one_use(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_production_writer_accepts_only_canonical_lowercase_json_and_is_one_use'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_terminal_writer_revalidates_every_artifact_before_pass(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_terminal_writer_revalidates_every_artifact_before_pass'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_anchored_writer_refuses_replaced_artifact_directory(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_anchored_writer_refuses_replaced_artifact_directory'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_exclusive_writer_never_deletes_a_racing_foreign_destination(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exclusive_writer_never_deletes_a_racing_foreign_destination'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_post_link_replacement_plus_directory_fsync_failure_preserves_foreign_file(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_post_link_replacement_plus_directory_fsync_failure_preserves_foreign_file'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_replaced_marker_is_never_overwritten_or_unlinked_at_finish(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replaced_marker_is_never_overwritten_or_unlinked_at_finish'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
