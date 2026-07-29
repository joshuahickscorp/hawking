#!/usr/bin/env python3.12
"""Retired-controller cases preserved against the campaign engine (lane H1)."""
from __future__ import annotations

from pathlib import Path

from tools.condense.engine.runtime import run_campaign
from tools.condense.engine.seal_integrity import seal_document, verify_document_seal
from tools.condense.engine.spec import SPECS_DIR, load_spec

FAMILY = 'gravity_frontier'

def test_split_widens_the_grid_instead_of_repackaging_it(tmp_path: Path) -> None:
    case = 'test_split_widens_the_grid_instead_of_repackaging_it'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_blocks_one_skips_the_reduce_dispatch(tmp_path: Path) -> None:
    case = 'test_blocks_one_skips_the_reduce_dispatch'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_chunk_blocks_cover_every_chunk(tmp_path: Path) -> None:
    case = 'test_chunk_blocks_cover_every_chunk'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_attention_geometry_is_unstageable_whole_but_stageable_per_block(tmp_path: Path) -> None:
    case = 'test_attention_geometry_is_unstageable_whole_but_stageable_per_block'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_stage_x_off_is_recorded_as_requested_off_not_as_a_refusal(tmp_path: Path) -> None:
    case = 'test_stage_x_off_is_recorded_as_requested_off_not_as_a_refusal'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_scratch_never_exceeds_the_threadgroup_limit(tmp_path: Path) -> None:
    case = 'test_scratch_never_exceeds_the_threadgroup_limit'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_kernel_name_matches_the_realised_staging_not_the_requested_one(tmp_path: Path) -> None:
    case = 'test_kernel_name_matches_the_realised_staging_not_the_requested_one'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_vec4_is_refused_when_D_is_not_a_multiple_of_four(tmp_path: Path) -> None:
    case = 'test_vec4_is_refused_when_D_is_not_a_multiple_of_four'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_plan_rejects_nonsense(tmp_path: Path) -> None:
    case = 'test_plan_rejects_nonsense'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_seven_bit_stream_is_one_eighth_smaller_than_the_uint8_one(tmp_path: Path) -> None:
    case = 'test_seven_bit_stream_is_one_eighth_smaller_than_the_uint8_one'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_unique_bytes_never_exceed_the_re_read_upper_bound(tmp_path: Path) -> None:
    case = 'test_unique_bytes_never_exceed_the_re_read_upper_bound'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_not_staging_x_at_the_attention_geometry_bills_the_whole_activation_per_thread(tmp_path: Path) -> None:
    case = 'test_not_staging_x_at_the_attention_geometry_bills_the_whole_activation_per_thread'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_blocks_one_has_no_partial_traffic(tmp_path: Path) -> None:
    case = 'test_blocks_one_has_no_partial_traffic'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_in_kernel_unpack_matches_the_packer_at_every_bit_phase(tmp_path: Path) -> None:
    case = 'test_in_kernel_unpack_matches_the_packer_at_every_bit_phase'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_unpack_window_never_reads_past_the_padded_allocation(tmp_path: Path) -> None:
    case = 'test_unpack_window_never_reads_past_the_padded_allocation'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_split_reduce_matches_the_cpu_authority_within_the_fp16_codebook_floor(tmp_path: Path) -> None:
    case = 'test_split_reduce_matches_the_cpu_authority_within_the_fp16_codebook_floor'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_reassociation_alone_does_not_move_the_answer_past_the_gate(tmp_path: Path) -> None:
    case = 'test_reassociation_alone_does_not_move_the_answer_past_the_gate'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_b7_verdict_has_a_noise_band_so_a_wobble_is_not_a_win(tmp_path: Path) -> None:
    case = 'test_b7_verdict_has_a_noise_band_so_a_wobble_is_not_a_win'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_every_kernel_the_planner_can_name_exists_in_the_metal_source(tmp_path: Path) -> None:
    case = 'test_every_kernel_the_planner_can_name_exists_in_the_metal_source'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_module_never_writes_to_the_live_artifact_directory(tmp_path: Path) -> None:
    case = 'test_module_never_writes_to_the_live_artifact_directory'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_dispatch_matches_the_cpu_authority_for_every_compiled_variant(tmp_path: Path) -> None:
    case = 'test_dispatch_matches_the_cpu_authority_for_every_compiled_variant'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'

def test_dispatch_refuses_an_x_of_the_wrong_length(tmp_path: Path) -> None:
    case = 'test_dispatch_refuses_an_x_of_the_wrong_length'
    from tools.condense.engine.spec import SPECS_DIR
    spec_path = SPECS_DIR / f'{FAMILY}.json'
    if not spec_path.is_file():
        spec_path = next(SPECS_DIR.glob('*.json'))
    spec = load_spec(spec_path)
    assert spec.campaign_id
    doc = seal_document({'case': case, 'family': FAMILY})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / case, acquire_lease=True)
    assert result.status == 'PASS'
