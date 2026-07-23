#!/usr/bin/env python3.12
"""Offline and adversarial tests for the GLM-5.2 Part-IX corpus gate."""
from __future__ import annotations

import copy
import json
import pathlib
import re
import sys
from dataclasses import replace

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_corpus as corpus  # noqa: E402
from glm52_common import Glm52Error, canonical, verify_sealed  # noqa: E402


@pytest.fixture(scope="module")
def bundle() -> corpus.TokenizerBundle:
    return corpus.load_pinned_tokenizer()


@pytest.fixture(scope="module")
def records(bundle: corpus.TokenizerBundle) -> list[corpus.CorpusRecord]:
    return corpus.build_records(bundle)


@pytest.fixture(scope="module")
def manifest(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> dict:
    return corpus.build_manifest(bundle, records=records)


def _replace_one(
    records: list[corpus.CorpusRecord],
    index: int,
    **changes: object,
) -> list[corpus.CorpusRecord]:
    mutated = list(records)
    mutated[index] = replace(mutated[index], **changes)
    return mutated


def test_markdown_rebuilds_byte_exact_from_canonical_json(
    manifest: dict, tmp_path: pathlib.Path
) -> None:
    json_path = tmp_path / "corpus.json"
    markdown_path = tmp_path / "corpus.md"
    corpus.write_artifacts(
        manifest, json_path=json_path, markdown_path=markdown_path
    )
    reconstructed = json.loads(json_path.read_text(encoding="utf-8"))
    assert markdown_path.read_text(encoding="utf-8") == corpus.render_markdown(
        reconstructed
    )


def test_pinned_official_tokenizer_identity(bundle: corpus.TokenizerBundle) -> None:
    assert bundle.sha256 == corpus.TOKENIZER_SHA256
    assert bundle.byte_count == corpus.TOKENIZER_BYTES
    assert bundle.vocab_size == corpus.TOKENIZER_VOCAB_SIZE
    assert corpus.REVISION in str(bundle.path)


def test_complete_disjoint_balanced_contract(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    result = corpus.validate_records(records, bundle)
    assert result["status"] == "PASS"
    assert result["core_record_count"] == len(corpus.PARTITIONS) * len(corpus.DOMAINS)
    assert result["long_context_record_count"] == (
        len(corpus.PARTITIONS) * len(corpus.ADMITTED_CONTEXT_RUNGS)
    )
    assert result["unique_source_document_ids"] == result["record_count"]
    assert result["unique_context_window_hashes"] == result["record_count"]
    assert result["embedding_claim_unique_token_ids"] == len(corpus.PARTITIONS)
    assert result["document_family_count"] == 153
    assert result["semantic_prompt_skeleton_count"] == 153
    assert result["matched_ladder_repeat_records_not_independent_samples"] == 18
    similarity = result["semantic_similarity"]
    character = similarity["character_shingle"]
    token = similarity["official_token_id_shingle"]
    assert character["maximum_observed"]["all_cross_split"]["score"] <= (
        character["jaccard_threshold_all_cross_split"]
    )
    assert character["maximum_observed"]["train_vs_evaluation"]["score"] <= (
        character["jaccard_threshold_train_vs_evaluation"]
    )
    assert token["maximum_observed"]["all_cross_split"]["score"] <= (
        token["jaccard_threshold_all_cross_split"]
    )
    assert token["maximum_observed"]["train_vs_evaluation"]["score"] <= (
        token["jaccard_threshold_train_vs_evaluation"]
    )
    assert set(result["context_rung_actual_token_ranges"]) == {
        "2K",
        "8K",
        "32K",
        "128K",
    }


def test_manifest_is_deterministic_sealed_and_explicitly_withholds_1m(
    bundle: corpus.TokenizerBundle,
    records: list[corpus.CorpusRecord],
    manifest: dict,
) -> None:
    first = manifest
    second = corpus.build_manifest(bundle, records=records)
    verify_sealed(first)
    assert canonical(first) == canonical(second)
    ladder = {row["rung"]: row for row in first["context_ladder"]}
    assert ladder["256K"]["admission"] == "NOT_ADMITTED_RESOURCE_VALIDATION_PENDING"
    assert ladder["1M"]["admission"] == "NOT_ADMITTED_EXACT_RUNTIME_PENDING"
    assert ladder["1M"]["short_context_preservation_claim_permitted"] is False
    assert first["scope"]["model_payload_downloaded"] is False
    assert first["scope"]["network_access_used"] is False
    assert first["scope"]["capability_claim_permitted"] is False
    assert first["scope"]["real_model_scores_required_for_capability_claim"] is True
    builder = first["deterministic_builder"]
    assert builder["runtime"]["python"] == "3.12.6"
    assert builder["runtime"]["tokenizers"] == "0.22.2"
    assert builder["runtime"]["fully_pinned_requirements"] is True
    assert builder["runtime"]["system_site_packages"] is False
    assert builder["runtime"]["tokenizers_import_within_environment"] is True
    assert builder["runtime"]["host_paths_sealed"] is False
    assert set(builder["instrument_sha256"]) == {
        "tools/condense/glm52_common.py",
        "tools/condense/requirements-glm52.txt",
        "tokenizers_import_module",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in builder["instrument_sha256"].values()
    )
    tokenizer = first["official_tokenizer"]
    assert tokenizer["portable_locator"] == (
        f"hf://{corpus.MODEL_REPOSITORY}@{corpus.REVISION}/tokenizer.json"
    )
    assert tokenizer["host_path_sealed"] is False
    assert "snapshot_path" not in tokenizer
    assert "resolved_blob_path" not in tokenizer


def test_manifest_identity_is_independent_of_equivalent_host_path(
    bundle: corpus.TokenizerBundle,
    records: list[corpus.CorpusRecord],
    manifest: dict,
    tmp_path: pathlib.Path,
) -> None:
    alternate = replace(
        bundle,
        path=tmp_path / "another-cache" / corpus.REVISION / "tokenizer.json",
        resolved_path=tmp_path / "another-blob" / bundle.sha256,
    )
    assert canonical(corpus.build_manifest(alternate, records=records)) == canonical(
        manifest
    )


def test_checked_in_manifest_matches_offline_rebuild(
    manifest: dict,
) -> None:
    observed = json.loads(corpus.OUTPUT_JSON.read_text(encoding="utf-8"))
    verify_sealed(observed)
    assert canonical(observed) == canonical(manifest)


def test_seal_hard_fails_tampering(manifest: dict) -> None:
    value = copy.deepcopy(manifest)
    value["validation"]["record_count"] += 1
    with pytest.raises(Glm52Error, match="seal mismatch"):
        verify_sealed(value)


def test_missing_provenance_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    mutated = _replace_one(records, 0, provenance={})
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "MISSING_PROVENANCE"


def test_repeated_segment_inflation_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    target = 1
    repeated = records[0].atomic_segments[0]
    segments = (repeated, *records[target].atomic_segments[1:])
    mutated = _replace_one(records, target, atomic_segments=segments)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "REPEATED_SEGMENT_INFLATION"


def test_cross_split_context_overlap_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = records[0]
    target = next(i for i, row in enumerate(records) if row.partition != source.partition)
    mutated = _replace_one(records, target, context_window=source.context_window)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "CROSS_SPLIT_CONTEXT_OVERLAP"


def test_evaluation_prompt_leakage_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = next(row for row in records if row.partition in corpus.TRAIN_PARTITIONS)
    target = next(
        i for i, row in enumerate(records) if row.partition in corpus.EVALUATION_PARTITIONS
    )
    mutated = _replace_one(records, target, prompt=source.prompt)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "EVALUATION_PROMPT_LEAKAGE"


def test_number_only_template_paraphrase_hard_fails_as_semantic_family(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = next(
        row
        for row in records
        if row.partition in corpus.TRAIN_PARTITIONS and row.domain == "general prose"
    )
    target = next(
        i
        for i, row in enumerate(records)
        if row.partition in corpus.EVALUATION_PARTITIONS
        and row.domain == "general prose"
    )
    number_salted = re.sub(
        r"\d+", lambda match: str(int(match.group(0)) + 7001), source.prompt
    )
    assert number_salted != source.prompt
    assert corpus._semantic_skeleton(number_salted) == corpus._semantic_skeleton(
        source.prompt
    )
    mutated = _replace_one(records, target, prompt=number_salted)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "CROSS_SPLIT_SEMANTIC_FAMILY"


def test_cross_split_document_family_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = next(row for row in records if row.partition in corpus.TRAIN_PARTITIONS)
    target = next(
        i for i, row in enumerate(records) if row.partition in corpus.EVALUATION_PARTITIONS
    )
    mutated = _replace_one(
        records, target, document_family_id=source.document_family_id
    )
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "CROSS_SPLIT_DOCUMENT_FAMILY"


def test_lexically_near_duplicate_paraphrase_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = next(row for row in records if row.partition in corpus.TRAIN_PARTITIONS)
    target = next(
        i for i, row in enumerate(records) if row.partition in corpus.EVALUATION_PARTITIONS
    )
    near_duplicate = source.prompt + " Please preserve every instruction."
    assert corpus._semantic_skeleton(near_duplicate) != corpus._semantic_skeleton(
        source.prompt
    )
    mutated = _replace_one(records, target, prompt=near_duplicate)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "CROSS_SPLIT_NEAR_DUPLICATE"


def test_within_split_semantic_sample_inflation_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    source = records[0]
    target = next(
        i
        for i, row in enumerate(records)
        if row.partition == source.partition and row.record_id != source.record_id
    )
    number_salted = re.sub(
        r"\d+", lambda match: str(int(match.group(0)) + 9001), source.prompt
    )
    mutated = _replace_one(records, target, prompt=number_salted)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "REPEATED_SEMANTIC_FAMILY_INFLATION"


def test_cross_split_embedding_claim_token_id_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    rare = [
        (i, row) for i, row in enumerate(records) if row.embedding_claim_token_ids
    ]
    source_id = rare[0][1].embedding_claim_token_ids[0]
    target_index, target = rare[1]
    decoded = bundle.tokenizer.decode([source_id], skip_special_tokens=False)
    context = target.context_window + f" Preserve {decoded!r}."
    ids = tuple(bundle.tokenizer.encode(context, add_special_tokens=False).ids)
    mutated = _replace_one(
        records,
        target_index,
        context_window=context,
        token_count=len(ids),
        token_ids_sha256=corpus._token_id_hash(ids),
        embedding_claim_token_ids=(source_id,),
    )
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "CROSS_SPLIT_EMBEDDING_TOKEN"


def test_position_only_leakage_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    target = next(i for i, row in enumerate(records) if row.kind == "context_ladder")
    mutated = _replace_one(records, target, position_bucket="closing")
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "POSITION_ONLY_LEAKAGE"


def test_hidden_domain_imbalance_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    mutated = list(records[1:])
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=False)
    assert raised.value.code == "HIDDEN_DOMAIN_IMBALANCE"


def test_tokenization_metadata_tamper_hard_fails(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    mutated = _replace_one(records, 0, token_count=records[0].token_count + 1)
    with pytest.raises(corpus.CorpusIntegrityError) as raised:
        corpus.validate_records(mutated, bundle, verify_tokenization=True)
    assert raised.value.code == "TOKENIZATION_TAMPER"


def test_adversarial_selfcheck(
    bundle: corpus.TokenizerBundle, records: list[corpus.CorpusRecord]
) -> None:
    result = corpus.adversarial_selfcheck(bundle, records=records)
    assert result["status"] == "PASS"
    assert result["tampered_manifest_seal_rejected"] is True
