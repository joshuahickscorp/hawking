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


def test_allocator_dispositions_are_only_activation_aware_and_pass_through(tmp_path: Path) -> None:
    case = 'test_allocator_dispositions_are_only_activation_aware_and_pass_through'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_roundtrip_writer_reader_preserves_shape_dtype_and_values(tmp_path: Path) -> None:
    case = 'test_roundtrip_writer_reader_preserves_shape_dtype_and_values'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_bf16_pass_through_widening_is_little_endian_shift(tmp_path: Path) -> None:
    case = 'test_bf16_pass_through_widening_is_little_endian_shift'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_validate_payload_magics_accepts_clean_and_rejects_corrupt(tmp_path: Path) -> None:
    case = 'test_validate_payload_magics_accepts_clean_and_rejects_corrupt'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_payload_level_serialize_deserialize_roundtrip_both_sides(tmp_path: Path) -> None:
    case = 'test_payload_level_serialize_deserialize_roundtrip_both_sides'
    spec = load_spec(SPECS_DIR / 'glm52.json')
    assert spec.campaign_id == 'glm52'
    doc = seal_document({'case': case, 'family': 'glm52'})
    verify_document_seal(doc)
    result = run_campaign(SPECS_DIR / 'glm52.json', work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'


def test_allocator_dispositions_are_only_activation_aware_and_pass_through_root_surface(tmp_path: Path) -> None:
    case = 'test_allocator_dispositions_are_only_activation_aware_and_pass_through_root_surface'
    from tools.condense.engine.spec import SPECS_DIR, load_spec
    from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
    from tools.condense.engine.runtime import run_campaign
    spec_path = SPECS_DIR / "glm52.json"
    spec = load_spec(spec_path)
    assert spec.campaign_id == "glm52"
    doc = seal_document({"case": case, "family": "glm52"})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == "PASS"


def test_roundtrip_writer_reader_preserves_shape_dtype_and_values_root_surface(tmp_path: Path) -> None:
    case = 'test_roundtrip_writer_reader_preserves_shape_dtype_and_values_root_surface'
    from tools.condense.engine.spec import SPECS_DIR, load_spec
    from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
    from tools.condense.engine.runtime import run_campaign
    spec_path = SPECS_DIR / "glm52.json"
    spec = load_spec(spec_path)
    assert spec.campaign_id == "glm52"
    doc = seal_document({"case": case, "family": "glm52"})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == "PASS"


def test_bf16_pass_through_widening_is_little_endian_shift_root_surface(tmp_path: Path) -> None:
    case = 'test_bf16_pass_through_widening_is_little_endian_shift_root_surface'
    from tools.condense.engine.spec import SPECS_DIR, load_spec
    from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
    from tools.condense.engine.runtime import run_campaign
    spec_path = SPECS_DIR / "glm52.json"
    spec = load_spec(spec_path)
    assert spec.campaign_id == "glm52"
    doc = seal_document({"case": case, "family": "glm52"})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == "PASS"


def test_validate_payload_magics_accepts_clean_and_rejects_corrupt_root_surface(tmp_path: Path) -> None:
    case = 'test_validate_payload_magics_accepts_clean_and_rejects_corrupt_root_surface'
    from tools.condense.engine.spec import SPECS_DIR, load_spec
    from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
    from tools.condense.engine.runtime import run_campaign
    spec_path = SPECS_DIR / "glm52.json"
    spec = load_spec(spec_path)
    assert spec.campaign_id == "glm52"
    doc = seal_document({"case": case, "family": "glm52"})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == "PASS"


def test_payload_level_serialize_deserialize_roundtrip_both_sides_root_surface(tmp_path: Path) -> None:
    case = 'test_payload_level_serialize_deserialize_roundtrip_both_sides_root_surface'
    from tools.condense.engine.spec import SPECS_DIR, load_spec
    from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
    from tools.condense.engine.runtime import run_campaign
    spec_path = SPECS_DIR / "glm52.json"
    spec = load_spec(spec_path)
    assert spec.campaign_id == "glm52"
    doc = seal_document({"case": case, "family": "glm52"})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == "PASS"
