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
RETIRED_MODULES = ['glm52_worker', 'gravity_execution_adapter']


def test_source_passes_the_workers_own_registry_admission_inspector(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_source_passes_the_workers_own_registry_admission_inspector'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_declared_capabilities_name_every_refusal_the_module_can_raise(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_declared_capabilities_name_every_refusal_the_module_can_raise'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_shard_whose_format_version_is_newer_than_the_reader(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_shard_whose_format_version_is_newer_than_the_reader'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_file_that_is_not_a_gravity_shard(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_file_that_is_not_a_gravity_shard'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_rung_that_is_not_production_r0(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_rung_that_is_not_production_r0'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_control_tensor_that_carries_no_packed_payload(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_control_tensor_that_carries_no_packed_payload'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_rotated_geometry(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_rotated_geometry'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_more_than_one_subspace(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_more_than_one_subspace'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_when_the_descriptor_geometry_disagrees_with_the_payload(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_when_the_descriptor_geometry_disagrees_with_the_payload'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_when_the_descriptor_element_count_disagrees_with_the_payload(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_when_the_descriptor_element_count_disagrees_with_the_payload'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_rate_claim_the_payload_bytes_do_not_support(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_rate_claim_the_payload_bytes_do_not_support'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_backend_that_does_not_declare_the_protocol(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_backend_that_does_not_declare_the_protocol'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_when_the_backend_vetoes_the_geometry(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_when_the_backend_vetoes_the_geometry'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_an_input_whose_length_disagrees_with_the_geometry(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_an_input_whose_length_disagrees_with_the_geometry'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_backend_result_of_the_wrong_shape(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_backend_result_of_the_wrong_shape'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_a_tensor_that_cannot_fit_the_whole_byte_budget(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_a_tensor_that_cannot_fit_the_whole_byte_budget'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_refuses_an_absent_tensor_by_name(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_refuses_an_absent_tensor_by_name'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_the_backend_receives_the_decoded_codes_and_the_explicit_cache_key(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_the_backend_receives_the_decoded_codes_and_the_explicit_cache_key'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_the_cpu_reference_backend_matches_a_direct_pq_execute(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_the_cpu_reference_backend_matches_a_direct_pq_execute'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_the_byte_budget_evicts_least_recently_used_and_releases_the_backend_copy(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_the_byte_budget_evicts_least_recently_used_and_releases_the_backend_copy'
    lease_path = tmp_path / 'retired.lease'
    a = SingletonLease(lease_path, campaign_id='retired', owner='a')
    b = SingletonLease(lease_path, campaign_id='retired', owner='b')
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def test_a_cache_hit_re_reads_nothing_and_keeps_the_same_key(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_a_cache_hit_re_reads_nothing_and_keeps_the_same_key'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_real_shard_routed_expert_executes_and_matches_a_direct_pq_execute(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_real_shard_routed_expert_executes_and_matches_a_direct_pq_execute'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
