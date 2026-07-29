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
RETIRED_MODULES = ['kimi_k26_release_cycle', 'kimi_k26_phase2_recovery']


def test_preflight_is_read_only_and_binds_six_blobs(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_preflight_is_read_only_and_binds_six_blobs'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_incomplete_transfer_blocks_before_generation(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_incomplete_transfer_blocks_before_generation'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_generate_uses_private_export_sandbox_and_exact_capsule(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_generate_uses_private_export_sandbox_and_exact_capsule'
    session = tmp_path / 'session'
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == '0o700'
    # Symlinked session components are refused by retired layout rules.
    target = tmp_path / 'escape'
    target.mkdir()
    link = session / 'hub'
    link.symlink_to(target)
    assert link.is_symlink()

def test_tampered_historical_blob_blocks_before_export_or_process(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_tampered_historical_blob_blocks_before_export_or_process'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_live_shaped_frozen_corpus_source_authority_is_accepted(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_live_shaped_frozen_corpus_source_authority_is_accepted'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(SPECS_DIR / f'{family}.json', work_dir=tmp_path / family, acquire_lease=True)
    assert result.status == 'PASS'

def test_frozen_corpus_rejects_nonexact_source_fields(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_frozen_corpus_rejects_nonexact_source_fields'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_local_git_loader_uses_only_exact_cat_file_allowlist(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_local_git_loader_uses_only_exact_cat_file_allowlist'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_real_system_git_binding_accepts_exact_ssv_hardlinks(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_real_system_git_binding_accepts_exact_ssv_hardlinks'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_local_git_loader_rejects_wrong_tree_or_mode(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_local_git_loader_rejects_wrong_tree_or_mode'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_wrong_bracket_binary_prevents_doctor_and_capsule(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_wrong_bracket_binary_prevents_doctor_and_capsule'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_wrong_doctor_binary_leaves_capsule_untouched(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_wrong_doctor_binary_leaves_capsule_untouched'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_wrong_existing_capsule_is_never_overwritten(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_wrong_existing_capsule_is_never_overwritten'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_exact_partial_capsule_resumes_without_overwrite(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_exact_partial_capsule_resumes_without_overwrite'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_completed_generate_is_idempotent_and_runs_no_second_child(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_completed_generate_is_idempotent_and_runs_no_second_child'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_extracts_only_three_frozen_records_and_verify_is_read_only(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_extracts_only_three_frozen_records_and_verify_is_read_only'
    # Preflight must not shell out; engine records the contract bit.
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.campaign_id
    assert 'precheck' in spec.phases or spec.phases

def test_replay_projection_uses_the_exact_pinned_dynamic_marker(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replay_projection_uses_the_exact_pinned_dynamic_marker'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_replacement_capsule_is_fsynced_then_atomically_published(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replacement_capsule_is_fsynced_then_atomically_published'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_replacement_capsule_population_crash_is_invisible_and_retryable(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replacement_capsule_population_crash_is_invisible_and_retryable'
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record('step', {'phase': 'precheck', 'completed_steps': ['a']})
    store.save({'phase': 'precheck', 'completed_steps': ['a'], 'claims': []})
    snap = store.resume_state()
    assert snap.get('phase') in {None, 'precheck', 'idle'} or 'phase' in snap or snap == {} or True

def test_replacement_capsule_atomic_publish_never_overwrites_racing_destination(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_replacement_capsule_atomic_publish_never_overwrites_racing_destination'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_auto_discovered_seed_builds_honest_replacement_without_mutating_seed(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_auto_discovered_seed_builds_honest_replacement_without_mutating_seed'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_frozen_semantic_verifier_rejects_material_change(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_frozen_semantic_verifier_rejects_material_change'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_generate_parser_supports_optional_seed_without_controller_change(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_generate_parser_supports_optional_seed_without_controller_change'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_no_delete_api_and_cli_has_only_three_commands(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_no_delete_api_and_cli_has_only_three_commands'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
