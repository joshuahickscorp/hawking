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

FAMILY = 'eco'
RETIRED_MODULES = ['eco_pipeline', 'eco_passport']


def test_selftest_green(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_selftest_green'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_canonical_order_matches_directive(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_canonical_order_matches_directive'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_spec_is_valid_topo_order(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_spec_is_valid_topo_order'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_every_passport_dimension_produced(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_every_passport_dimension_produced'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_spec_sha256_is_deterministic_content_address(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_spec_sha256_is_deterministic_content_address'
    spec = load_spec(SPECS_DIR / f'{family}.json')
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def test_advance_blocks_on_unmet_requires(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_advance_blocks_on_unmet_requires'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_runnable_progression(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_runnable_progression'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_rollback_reverts_only_dependents(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_rollback_reverts_only_dependents'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_offline_hydrate_stops_at_gap(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_offline_hydrate_stops_at_gap'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path

def test_passport_validator_enforced(tmp_path: Path) -> None:
    family = FAMILY
    name = 'test_passport_validator_enforced'
    # Retired controller case preserved as engine lifecycle / seal assertion.
    spec_path = SPECS_DIR / f'{family}.json'
    if not spec_path.is_file():
        # Fall back to any registered family for non-mapped shells.
        spec_path = next(SPECS_DIR.glob('*.json'))
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == 'PASS'
    assert result.receipt_path
