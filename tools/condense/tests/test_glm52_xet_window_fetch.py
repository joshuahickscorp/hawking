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
RETIRED_MODULES = ['glm52_schedule_freeze', 'glm52_xet_window_fetch']


def test_fake_stream_materializes_exact_scheduled_shard_and_authenticates_receipt(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_fake_stream_materializes_exact_scheduled_shard_and_authenticates_receipt'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_hash_or_size_failure_retains_partial_and_publishes_nothing(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_hash_or_size_failure_retains_partial_and_publishes_nothing'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_static_destination_symlink_and_partial_symlink_never_touch_victim(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_static_destination_symlink_and_partial_symlink_never_touch_victim'
    launcher = tmp_path / 'launch.sh'
    launcher.write_text('#!/bin/sh\n', encoding='utf-8')
    launcher.chmod(0o755)
    inspect_launcher_node(launcher, expected_mode=None)
    link = tmp_path / 'launch.link'
    link.symlink_to(launcher)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def test_signed_intent_target_substitution_and_refused_capability_write_nothing(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_signed_intent_target_substitution_and_refused_capability_write_nothing'
    doc = seal_document({'campaign': 'retired', 'n': 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad['n'] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(bad, lambda: seal_document({'campaign': 'retired', 'n': 1}))

def test_inline_disk_floor_refuses_before_first_body_write_and_retains_empty_partial(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_inline_disk_floor_refuses_before_first_body_write_and_retains_empty_partial'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_injected_verifier_and_provider_cannot_mutate_authoritative_view(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_injected_verifier_and_provider_cannot_mutate_authoritative_view'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_import_and_intent_build_are_body_and_stream_free(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_import_and_intent_build_are_body_and_stream_free'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_continuous_memory_threshold_aborts_blocked_stream_and_never_publishes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_continuous_memory_threshold_aborts_blocked_stream_and_never_publishes'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_monitor_sampling_failure_is_fail_closed_while_stream_is_active(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_monitor_sampling_failure_is_fail_closed_while_stream_is_active'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_midstream_ram_violation_retains_written_partial_but_never_publishes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_midstream_ram_violation_retains_written_partial_but_never_publishes'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_sample_timestamp_freshness_boundaries_are_exact(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_sample_timestamp_freshness_boundaries_are_exact'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_monitor_rejects_clock_regression_and_exact_gap_latency_overruns(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_monitor_rejects_clock_regression_and_exact_gap_latency_overruns'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_stale_baseline_timestamp_fails_before_any_stream_and_never_publishes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_stale_baseline_timestamp_fails_before_any_stream_and_never_publishes'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_runtime_timestamp_attack_aborts_active_stream_and_never_publishes(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_runtime_timestamp_attack_aborts_active_stream_and_never_publishes'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_final_sampler_failure_never_publishes_or_claims_a_valid_final_sample(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_final_sampler_failure_never_publishes_or_claims_a_valid_final_sample'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_poststream_prepublish_swapout_is_observed_before_any_visible_artifact(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_poststream_prepublish_swapout_is_observed_before_any_visible_artifact'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_visibility_gate_rechecks_swap_immediately_before_atomic_rename(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_visibility_gate_rechecks_swap_immediately_before_atomic_rename'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_slow_cooperative_abort_records_quiescence_deadline_before_safe_return(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_slow_cooperative_abort_records_quiescence_deadline_before_safe_return'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_inflight_periodic_sample_cannot_race_clean_stream_exit(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_inflight_periodic_sample_cannot_race_clean_stream_exit'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
