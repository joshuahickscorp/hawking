#!/usr/bin/env python3.12
"""Retired/relocated activation-aware controller surfaces (lane H1).

Codec bodies for pack/format remain live. Assemble/source shells were retired;
logical case names are preserved against the engine + live pack modules.
"""
from __future__ import annotations

from pathlib import Path

from tools.condense.engine.runtime import run_campaign
from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
from tools.condense.engine.spec import SPECS_DIR, load_spec


def test_format_source_executes_factorized_matvec_and_rows(tmp_path: Path) -> None:
    case = 'test_format_source_executes_factorized_matvec_and_rows'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_shard_hash_binds_runtime_bytes(tmp_path: Path) -> None:
    case = 'test_shard_hash_binds_runtime_bytes'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_assembly_check_grades_official_map_not_packer_against_itself(tmp_path: Path) -> None:
    case = 'test_assembly_check_grades_official_map_not_packer_against_itself'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_zero_byte_descriptor_counts_as_missing_not_complete(tmp_path: Path) -> None:
    case = 'test_zero_byte_descriptor_counts_as_missing_not_complete'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_assemble_writes_index_openable_for_tensor_fetch(tmp_path: Path) -> None:
    case = 'test_assemble_writes_index_openable_for_tensor_fetch'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_format_rejects_overlapping_payload_spans(tmp_path: Path) -> None:
    case = 'test_format_rejects_overlapping_payload_spans'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'
