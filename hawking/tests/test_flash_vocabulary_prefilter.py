from __future__ import annotations

import hashlib
import json
import os
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from hawking.agentos import flash_tensor_probe
from hawking.agentos import flash_vocabulary_prefilter as vocabulary_prefilter
from hawking.control import build_parser
from hawking.flash_next import PINNED_REVISION, REPO_ID
from tools.odyssey import specimen_open


SYNTHETIC_VOCAB_SIZE = 32
SYNTHETIC_HIDDEN_SIZE = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _sha256(path)


def _write_tensor(path: Path, tensor_name: str, *, shape: list[int]) -> None:
    payload_bytes = 2
    for dimension in shape:
        payload_bytes *= dimension
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, payload_bytes],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + (b"\x00" * payload_bytes))


def _synthetic_token_surfaces() -> list[str]:
    """Four distinct rows for each 2-by-2-by-2 E13 factorial cell."""
    return [
        # rare / Latin / short, then long
        "a", "b", "c", "d",
        "alpha", "bravo", "cider", "delta",
        # rare / Han / short, then long
        "汉", "字", "语", "文",
        "漢字", "语言", "中文", "漢語",
        # common / Latin / short, then long
        "e", "f", "g", "h",
        "eagle", "fable", "giant", "honey",
        # common / Han / short, then long
        "话", "言", "学", "习",
        "汉语", "文字", "学习", "话语",
    ]


def _pin_synthetic_source(monkeypatch, specimen: Path) -> None:
    """Bind the preflight's pinned facts to the synthetic metadata-only source."""
    monkeypatch.setattr(vocabulary_prefilter, "PINNED_CONFIG_SHA256", _sha256(specimen / "config.json"))
    monkeypatch.setattr(vocabulary_prefilter, "PINNED_INDEX_SHA256", _sha256(specimen / "model.safetensors.index.json"))
    monkeypatch.setattr(vocabulary_prefilter, "PINNED_TOKENIZER_SHA256", _sha256(specimen / "tokenizer.json"))
    tokenizer = json.loads((specimen / "tokenizer.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        vocabulary_prefilter,
        "PINNED_TOKENIZER_MODEL_VOCAB_ENTRY_COUNT",
        len(tokenizer["model"]["vocab"]),
    )
    monkeypatch.setattr(
        vocabulary_prefilter,
        "PINNED_TOKENIZER_ADDED_TOKEN_ENTRY_COUNT",
        len(tokenizer["added_tokens"]),
    )
    monkeypatch.setattr(
        vocabulary_prefilter,
        "PINNED_TOKENIZER_ADDRESSABLE_TOKEN_COUNT",
        len(tokenizer["model"]["vocab"]) + len(tokenizer["added_tokens"]),
    )
    roles = {}
    for role, tensor_name in (
        ("input_embedding", vocabulary_prefilter.INPUT_EMBEDDING),
        ("output_head", vocabulary_prefilter.OUTPUT_HEAD),
    ):
        header = flash_tensor_probe.load_tensor_header(specimen, tensor_name)
        roles[role] = {
            key: header[key]
            for key in (
                "shard_name",
                "shard_size",
                "header_bytes",
                "dtype",
                "shape",
                "data_offsets",
                "payload_bytes",
            )
        }
    monkeypatch.setattr(vocabulary_prefilter, "PINNED_ROLE_HEADERS", roles)


def _rewrite_tensor_geometry(
        specimen: Path, monkeypatch, *, vocab_size: int,
) -> None:
    """Change only synthetic config/header geometry; never inspect tensor payloads."""
    _write_json(
        specimen / "config.json",
        {
            "tie_word_embeddings": False,
            "text_config": {
                "vocab_size": vocab_size,
                "hidden_size": SYNTHETIC_HIDDEN_SIZE,
                "tie_word_embeddings": False,
            },
        },
    )
    _write_tensor(
        specimen / "embedding.safetensors",
        vocabulary_prefilter.INPUT_EMBEDDING,
        shape=[vocab_size, SYNTHETIC_HIDDEN_SIZE],
    )
    _write_tensor(
        specimen / "head.safetensors",
        vocabulary_prefilter.OUTPUT_HEAD,
        shape=[vocab_size, SYNTHETIC_HIDDEN_SIZE],
    )
    _pin_synthetic_source(monkeypatch, specimen)


@pytest.fixture
def e13_specimen(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_tensor_probe.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    _write_json(
        manifests / f"{flash_tensor_probe.LAKE_SLUG}.json",
        {
            "repo": REPO_ID,
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 5,
        },
    )
    _write_json(
        specimen / "config.json",
        {
            "tie_word_embeddings": False,
            "text_config": {
                "vocab_size": SYNTHETIC_VOCAB_SIZE,
                "hidden_size": SYNTHETIC_HIDDEN_SIZE,
                "tie_word_embeddings": False,
            },
        },
    )
    token_surfaces = _synthetic_token_surfaces()
    _write_json(
        specimen / "tokenizer.json",
        {
            "model": {"vocab": {surface: index for index, surface in enumerate(token_surfaces)}},
            "added_tokens": [],
        },
    )
    embedding_shard = "embedding.safetensors"
    head_shard = "head.safetensors"
    _write_tensor(
        specimen / embedding_shard,
        vocabulary_prefilter.INPUT_EMBEDDING,
        shape=[SYNTHETIC_VOCAB_SIZE, SYNTHETIC_HIDDEN_SIZE],
    )
    _write_tensor(
        specimen / head_shard,
        vocabulary_prefilter.OUTPUT_HEAD,
        shape=[SYNTHETIC_VOCAB_SIZE, SYNTHETIC_HIDDEN_SIZE],
    )
    _write_json(
        specimen / "model.safetensors.index.json",
        {
            "weight_map": {
                vocabulary_prefilter.INPUT_EMBEDDING: embedding_shard,
                vocabulary_prefilter.OUTPUT_HEAD: head_shard,
            }
        },
    )
    monkeypatch.setattr(flash_tensor_probe, "LAKE_ROOT", lake)
    _pin_synthetic_source(monkeypatch, specimen)
    return specimen


def _control_root(repo: Path) -> Path:
    return repo / "receipts" / "headless"


def _valid_plan(specimen: Path, source: dict, repo: Path) -> dict:
    selected = list(range(SYNTHETIC_VOCAB_SIZE))
    surfaces = vocabulary_prefilter._selected_token_surfaces(specimen / "tokenizer.json", selected)
    control_root = _control_root(repo)
    frequency_path = control_root / "e13" / "frequency-evidence.json"
    script_path = control_root / "e13" / "script-evidence.json"
    producer_path = repo / vocabulary_prefilter.RUST_E13_PRODUCER_REL
    producer_path.parent.mkdir(parents=True, exist_ok=True)
    producer_path.write_text("synthetic reviewed Rust E13 producer\n", encoding="utf-8")
    producer_sha256 = _sha256(producer_path)
    producer = {
        "schema": vocabulary_prefilter.RUST_E13_PRODUCER_SCHEMA,
        "implementation_sha256": producer_sha256,
        "runtime": vocabulary_prefilter.RUST_E13_PRODUCER_RUNTIME,
        "model_loaded": False,
        "tensor_payload_bytes_read": 0,
        "gpu_session_started": False,
    }
    counts = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        100, 110, 120, 130, 140, 150, 160, 170,
        180, 190, 200, 210, 220, 230, 240, 250,
    ]
    scripts = [
        "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin",
        "Han", "Han", "Han", "Han", "Han", "Han", "Han", "Han",
        "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin", "Latin",
        "Han", "Han", "Han", "Han", "Han", "Han", "Han", "Han",
    ]
    frequency_hash = _write_json(
        frequency_path,
        {
            "schema": vocabulary_prefilter.FREQUENCY_EVIDENCE_SCHEMA,
            "provenance": "tokenized_corpus_frequency.v1",
            "tokenizer_sha256": source["tokenizer_sha256"],
            "corpus_sha256": "a" * 64,
            "corpus_token_count": 4000,
            "producer": producer,
            "rows": [
                {"token_id": token_id, "count": count}
                for token_id, count in enumerate(counts)
            ],
        },
    )
    script_hash = _write_json(
        script_path,
        {
            "schema": vocabulary_prefilter.SCRIPT_EVIDENCE_SCHEMA,
            "provenance": "tokenizer_script_classification.v1",
            "tokenizer_sha256": source["tokenizer_sha256"],
            "classifier": {
                "algorithm": vocabulary_prefilter.TOKENIZER_SCRIPT_CLASSIFIER_ALGORITHM,
                "implementation_sha256": producer_sha256,
            },
            "producer": producer,
            "rows": [
                {
                    "token_id": token_id,
                    "script": scripts[token_id],
                    **surfaces[token_id],
                    # The synthetic fixture's wire surface is ordinary UTF-8.
                    # Production must bind these independently decoded values.
                    "decoded_token_surface_sha256": surfaces[token_id]["tokenizer_raw_surface_sha256"],
                    "decoded_utf8_byte_length": surfaces[token_id]["tokenizer_raw_surface_utf8_byte_length"],
                }
                for token_id in selected
            ],
        },
    )

    strata = []
    definitions = [
        ("rare", "Latin", "short", 0, 1, 1, 10, 1, 1),
        ("rare", "Latin", "long", 4, 5, 1, 20, 5, 5),
        ("rare", "Han", "short", 8, 9, 1, 20, 3, 3),
        ("rare", "Han", "long", 12, 13, 1, 20, 6, 6),
        ("common", "Latin", "short", 16, 17, 100, 170, 1, 1),
        ("common", "Latin", "long", 20, 21, 100, 170, 5, 5),
        ("common", "Han", "short", 24, 25, 100, 250, 3, 3),
        ("common", "Han", "long", 28, 29, 100, 250, 6, 6),
    ]
    for band, script, byte_band, first_fit, second_fit, freq_low, freq_high, byte_low, byte_high in definitions:
        heldout = [second_fit + 1, second_fit + 2]
        strata.append(
            {
                "id": f"{band}-{script.lower()}-{byte_band}",
                "frequency_band": band,
                "frequency_range": {"min_inclusive": freq_low, "max_inclusive": freq_high},
                "script": script,
                "byte_length_band": byte_band,
                "byte_length_range": {"min_inclusive": byte_low, "max_inclusive": byte_high},
                "fit_token_ids": [first_fit, second_fit],
                "heldout_token_ids": heldout,
            }
        )
    return {
        "schema": vocabulary_prefilter.PLAN_SCHEMA,
        "row_budget": SYNTHETIC_VOCAB_SIZE,
        "source": {
            key: source[key]
            for key in (
                "repo",
                "pinned_revision",
                "config_sha256",
                "index_sha256",
                "tokenizer_sha256",
                "model_lake_manifest_sha256",
                "role_headers_sha256",
                "tokenizer_addressability_sha256",
            )
        },
        "row_universe": {
            "schema": vocabulary_prefilter.ROW_UNIVERSE_SCHEMA,
            "tensor_row_count": source["vocabulary_geometry"]["tensor_row_count"],
            "tokenizer_addressable_token_count": source["vocabulary_geometry"]["addressable_token_count"],
            "excluded_tensor_tail_row_count": source["vocabulary_geometry"]["tensor_rows_without_tokenizer_surface"],
            "e13_eligible_model_vocab_token_count": source["vocabulary_geometry"]["e13_eligible_model_vocab_token_count"],
            "excluded_added_control_token_count": source["vocabulary_geometry"]["e13_excluded_added_control_token_count"],
            "excluded_added_control_token_ids_sha256": source["vocabulary_geometry"]["e13_excluded_added_control_token_ids_sha256"],
            "tail_policy": vocabulary_prefilter.ROW_UNIVERSE_TAIL_POLICY,
            "added_token_policy": vocabulary_prefilter.ROW_UNIVERSE_ADDED_TOKEN_POLICY,
        },
        "frequency_evidence": {
            "artifact_path": "e13/frequency-evidence.json",
            "artifact_sha256": frequency_hash,
        },
        "script_evidence": {
            "artifact_path": "e13/script-evidence.json",
            "artifact_sha256": script_hash,
        },
        "strata": strata,
        "shuffled_token_label_null": {
            "algorithm": vocabulary_prefilter.SPLIT_PRESERVING_NULL_ALGORITHM,
            "seed": 7,
        },
    }


def _write_plan(repo: Path, plan: dict) -> Path:
    path = _control_root(repo) / "e13" / "plan.json"
    _write_json(path, plan)
    return path


def _baseline(specimen: Path, repo: Path) -> dict:
    return vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=specimen,
        repo_root=repo,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=repo / "baseline.json",
    )


def test_e13_withholds_without_plan_after_header_only_identity_preflight(e13_specimen, tmp_path, monkeypatch):
    boundaries = {}
    for shard in (e13_specimen / "embedding.safetensors", e13_specimen / "head.safetensors"):
        with shard.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
        boundaries[shard.resolve()] = 8 + header_bytes
    original_open_binary = specimen_open._open_binary

    class HeaderOnlyReader:
        def __init__(self, handle, boundary):
            self._handle = handle
            self._boundary = boundary

        def read(self, size=-1):
            position = self._handle.tell()
            if size < 0 or position + size > self._boundary:
                raise AssertionError("E13 attempted to read safetensors payload bytes")
            return self._handle.read(size)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def guarded_open_binary(path, *args, **kwargs):
        handle = original_open_binary(path, *args, **kwargs)
        boundary = boundaries.get(path.resolve())
        if boundary is not None:
            return HeaderOnlyReader(handle, boundary)
        return handle

    monkeypatch.setattr(specimen_open, "_open_binary", guarded_open_binary)
    result = _baseline(e13_specimen, tmp_path)

    assert result["status"] == "WITHHELD_MISSING_VOCABULARY_CONTROL_PLAN"
    assert result["model_loaded"] is False
    assert result["gpu_session_started"] is False
    assert result["tensor_payload_bytes_read"] == 0
    assert result["row_extraction_performed"] is False
    assert result["source_identity"]["vocabulary_geometry"]["vocab_size"] == SYNTHETIC_VOCAB_SIZE
    assert result["source_identity"]["vocabulary_geometry"]["addressable_token_count"] == SYNTHETIC_VOCAB_SIZE
    assert result["source_identity"]["vocabulary_geometry"]["tensor_rows_without_tokenizer_surface"] == 0
    assert result["bounded_extraction_budget"]["bytes_total"] == SYNTHETIC_VOCAB_SIZE * SYNTHETIC_HIDDEN_SIZE * 2 * 2
    assert result["bounded_extraction_budget"]["payload_read_now"] == 0
    assert result["seal_format"] == vocabulary_prefilter.RECEIPT_SEAL_FORMAT
    assert result["producer"]["module_sha256"]
    assert vocabulary_prefilter.verify_flash_vocabulary_prefilter_receipt(result) == result
    assert vocabulary_prefilter.verify_flash_vocabulary_prefilter_receipt_path(
        tmp_path / "baseline.json"
    ) == result


def test_e13_admits_exact_native_producer_without_reading_tensor_rows(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan_path = _write_plan(tmp_path, _valid_plan(e13_specimen, baseline["source_identity"], tmp_path))

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "withheld.json",
    )

    assert result["status"] == "READY_FOR_BOUNDED_ROW_EXTRACTION"
    assert result["model_loaded"] is False
    assert result["gpu_session_started"] is False
    assert result["tensor_payload_bytes_read"] == 0
    assert result["row_extraction_performed"] is False
    assert result["control_plan"]["fit_and_heldout_disjoint"] is True
    assert result["control_plan"]["frequency_evidence"]["path"].endswith("frequency-evidence.json")
    assert result["control_plan"]["shuffled_token_label_null"]["seed_derived_derangement"] is True
    assert result["control_plan"]["shuffled_token_label_null"]["assignment_count"] == SYNTHETIC_VOCAB_SIZE
    assert result["control_plan"]["shuffled_token_label_null"]["fit_and_heldout_partition_preserved"] is True
    assert result["evidence_admission"]["status"] == "ADMITTED_CANONICAL_RUST_PRODUCER"


def test_e13_refuses_forged_native_producer_identity(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    frequency_path = _control_root(tmp_path) / "e13" / "frequency-evidence.json"
    frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
    frequency["producer"]["implementation_sha256"] = "0" * 64
    plan["frequency_evidence"]["artifact_sha256"] = _write_json(frequency_path, frequency)
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "forged-producer.json",
    )

    assert result["status"] == "WITHHELD_EVIDENCE_PRODUCER_IDENTITY_MISMATCH"
    assert result["tensor_payload_bytes_read"] == 0


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (
            lambda plan, _: plan["source"].__setitem__("config_sha256", "0" * 64),
            "WITHHELD_SOURCE_BINDING_MISMATCH",
        ),
        (
            lambda plan, _: plan["frequency_evidence"].__setitem__("artifact_sha256", "0" * 64),
            "WITHHELD_EVIDENCE_ARTIFACT_HASH_MISMATCH",
        ),
        (
            lambda plan, _: plan["frequency_evidence"].__setitem__("rows", []),
            "WITHHELD_INLINE_EVIDENCE_REFUSED",
        ),
        (
            lambda plan, _: plan["shuffled_token_label_null"].__setitem__(
                "assignments",
                [
                    {"stratum_id": row["id"], "token_id": token_id, "shuffled_token_id": token_id}
                    for row in plan["strata"]
                    for token_id in row["fit_token_ids"] + row["heldout_token_ids"]
                ],
            ),
            "WITHHELD_SHUFFLED_NULL_NOT_SEED_DERIVED",
        ),
    ],
)
def test_e13_refuses_unbound_or_nonreproducible_controls(e13_specimen, tmp_path, mutate, status):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    mutate(plan, tmp_path)
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "withheld.json",
    )

    assert result["status"] == status
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_plan_outside_canonical_control_root(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    external_plan = tmp_path / "outside.json"
    _write_json(external_plan, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=external_plan,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "outside.json.receipt",
    )

    assert result["status"] == "WITHHELD_CONTROL_ARTIFACT_OUTSIDE_ROOT"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_frequency_artifact_that_breaks_count_conservation(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    frequency_path = _control_root(tmp_path) / "e13" / "frequency-evidence.json"
    frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
    frequency["rows"][0]["count"] = frequency["corpus_token_count"] + 1
    plan["frequency_evidence"]["artifact_sha256"] = _write_json(frequency_path, frequency)
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "count-inconsistent.json",
    )

    assert result["status"] == "WITHHELD_FREQUENCY_EVIDENCE_COUNT_INCONSISTENT"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_byte_length_confounded_strata(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    strata = deepcopy(plan["strata"])
    strata[-1]["byte_length_band"] = "short"

    with pytest.raises(vocabulary_prefilter._Withheld) as exc:
        vocabulary_prefilter._selected_ids(
            strata,
            vocab_size=SYNTHETIC_VOCAB_SIZE,
            row_budget=SYNTHETIC_VOCAB_SIZE,
        )

    assert exc.value.code in {"CONFOUNDED_BYTE_LENGTH_STRATA", "INCOMPLETE_FACTORIAL_STRATA"}


def test_e13_shuffled_null_never_crosses_fit_or_heldout(e13_specimen, tmp_path):
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    by_stratum, _, _ = vocabulary_prefilter._selected_ids(
        plan["strata"],
        vocab_size=SYNTHETIC_VOCAB_SIZE,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        addressable_token_count=SYNTHETIC_VOCAB_SIZE,
    )
    assignments = vocabulary_prefilter._derived_null_assignments(
        by_stratum,
        seed=plan["shuffled_token_label_null"]["seed"],
    )

    assert len(assignments) == SYNTHETIC_VOCAB_SIZE
    for assignment in assignments:
        partition = by_stratum[assignment["stratum_id"]][assignment["split"]]
        assert assignment["token_id"] in partition
        assert assignment["shuffled_token_id"] in partition
        assert assignment["token_id"] != assignment["shuffled_token_id"]


def test_e13_refuses_tensor_tail_selected_in_plan_before_payload_read(
        e13_specimen, tmp_path, monkeypatch,
):
    _rewrite_tensor_geometry(
        e13_specimen,
        monkeypatch,
        vocab_size=SYNTHETIC_VOCAB_SIZE + 2,
    )
    baseline = _baseline(e13_specimen, tmp_path)
    geometry = baseline["source_identity"]["vocabulary_geometry"]
    assert geometry["addressable_token_count"] == SYNTHETIC_VOCAB_SIZE
    assert geometry["tensor_rows_without_tokenizer_surface"] == 2
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    plan["strata"][-1]["heldout_token_ids"][-1] = SYNTHETIC_VOCAB_SIZE
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "tail-selection.json",
    )

    assert result["status"] == "WITHHELD_TOKEN_ID_NOT_ADDRESSABLE_BY_TOKENIZER"
    assert result["tensor_payload_bytes_read"] == 0
    assert result["row_extraction_performed"] is False


def test_e13_refuses_added_control_selected_in_plan_before_payload_read(
        e13_specimen, tmp_path, monkeypatch,
):
    _rewrite_tensor_geometry(
        e13_specimen,
        monkeypatch,
        vocab_size=SYNTHETIC_VOCAB_SIZE + 2,
    )
    tokenizer_path = e13_specimen / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"] = [
        {"id": SYNTHETIC_VOCAB_SIZE, "content": "<control-a>", "special": False},
        {"id": SYNTHETIC_VOCAB_SIZE + 1, "content": "<control-b>", "special": True},
    ]
    _write_json(tokenizer_path, tokenizer)
    _pin_synthetic_source(monkeypatch, e13_specimen)
    baseline = _baseline(e13_specimen, tmp_path)
    geometry = baseline["source_identity"]["vocabulary_geometry"]
    assert geometry["tokenizer_added_token_ids"] == [32, 33]
    assert geometry["tensor_rows_without_tokenizer_surface"] == 0
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    plan["strata"][-1]["heldout_token_ids"][-1] = SYNTHETIC_VOCAB_SIZE
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "added-control-selection.json",
    )

    assert result["status"] == "WITHHELD_TOKEN_ID_NOT_ELIGIBLE_FOR_E13_STRATA"
    assert result["tensor_payload_bytes_read"] == 0
    assert result["row_extraction_performed"] is False


def test_e13_records_exact_lexical_added_and_tensor_tail_partition(
        e13_specimen, tmp_path, monkeypatch,
):
    _rewrite_tensor_geometry(
        e13_specimen,
        monkeypatch,
        vocab_size=SYNTHETIC_VOCAB_SIZE + 4,
    )
    tokenizer_path = e13_specimen / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"] = [
        {"id": SYNTHETIC_VOCAB_SIZE, "content": "<control-a>", "special": False},
        {"id": SYNTHETIC_VOCAB_SIZE + 1, "content": "<control-b>", "special": True},
    ]
    _write_json(tokenizer_path, tokenizer)
    _pin_synthetic_source(monkeypatch, e13_specimen)

    result = _baseline(e13_specimen, tmp_path)
    geometry = result["source_identity"]["vocabulary_geometry"]
    partition = geometry["tokenizer_tensor_partition"]

    assert partition["schema"] == vocabulary_prefilter.TOKENIZER_TENSOR_PARTITION_SCHEMA
    assert partition["lexical_model_vocab"] == {
        "count": SYNTHETIC_VOCAB_SIZE,
        "id_range": [0, SYNTHETIC_VOCAB_SIZE - 1],
        "e13_lexical_eligible": True,
    }
    assert partition["added_control"]["count"] == 2
    assert partition["added_control"]["id_range"] == [32, 33]
    assert partition["added_control"]["e13_lexical_eligible"] is False
    assert partition["tokenizer_addressable"]["id_range"] == [0, 33]
    assert partition["tensor_only_tail"]["id_range"] == [34, 35]
    assert partition["tensor_only_tail"]["output_sampling_policy"] == vocabulary_prefilter.TAIL_SAMPLING_POLICY
    assert partition["tensor_rows"]["count"] == 36
    assert partition["tensor_rows"]["complete_artifact_accounting_policy"] == vocabulary_prefilter.TENSOR_ACCOUNTING_POLICY
    assert partition["tensor_rows"]["raw_source_logit_policy"] == vocabulary_prefilter.RAW_LOGIT_POLICY
    assert geometry["tensor_row_count"] == 36
    assert geometry["e13_eligible_model_vocab_token_count"] == 32
    assert result["bounded_extraction_budget"]["tensor_row_count"] == 36
    assert result["bounded_extraction_budget"]["e13_eligible_model_vocab_token_count"] == 32


def test_e13_refuses_tail_rows_claimed_by_frequency_evidence(
        e13_specimen, tmp_path, monkeypatch,
):
    _rewrite_tensor_geometry(
        e13_specimen,
        monkeypatch,
        vocab_size=SYNTHETIC_VOCAB_SIZE + 2,
    )
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    frequency_path = _control_root(tmp_path) / "e13" / "frequency-evidence.json"
    frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
    frequency["rows"].append({"token_id": SYNTHETIC_VOCAB_SIZE, "count": 1})
    plan["frequency_evidence"]["artifact_sha256"] = _write_json(frequency_path, frequency)
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "tail-evidence.json",
    )

    assert result["status"] == "WITHHELD_EVIDENCE_TOKEN_NOT_ADDRESSABLE_BY_TOKENIZER"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_added_controls_claimed_by_script_evidence(
        e13_specimen, tmp_path, monkeypatch,
):
    _rewrite_tensor_geometry(
        e13_specimen,
        monkeypatch,
        vocab_size=SYNTHETIC_VOCAB_SIZE + 2,
    )
    tokenizer_path = e13_specimen / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer["added_tokens"] = [
        {"id": SYNTHETIC_VOCAB_SIZE, "content": "<control-a>", "special": False},
        {"id": SYNTHETIC_VOCAB_SIZE + 1, "content": "<control-b>", "special": True},
    ]
    _write_json(tokenizer_path, tokenizer)
    _pin_synthetic_source(monkeypatch, e13_specimen)
    baseline = _baseline(e13_specimen, tmp_path)
    plan = _valid_plan(e13_specimen, baseline["source_identity"], tmp_path)
    script_path = _control_root(tmp_path) / "e13" / "script-evidence.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    script["rows"].append({"token_id": SYNTHETIC_VOCAB_SIZE})
    plan["script_evidence"]["artifact_sha256"] = _write_json(script_path, script)
    plan_path = _write_plan(tmp_path, plan)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        plan=plan_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "added-control-evidence.json",
    )

    assert result["status"] == "WITHHELD_EVIDENCE_TOKEN_NOT_ELIGIBLE_FOR_E13"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_source_metadata_symlink_before_reading_payload(e13_specimen, tmp_path):
    tokenizer = e13_specimen / "tokenizer.json"
    tokenizer.unlink()
    tokenizer.symlink_to(e13_specimen / "embedding.safetensors")

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "symlink.json",
    )

    assert result["status"] == "WITHHELD_SOURCE_METADATA_SYMLINK_REFUSED"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_refuses_hardlinked_tensor_at_a_json_metadata_path_before_reading_it(e13_specimen, tmp_path):
    tokenizer = e13_specimen / "tokenizer.json"
    tokenizer.unlink()
    os.link(e13_specimen / "embedding.safetensors", tokenizer)

    result = vocabulary_prefilter.run_flash_vocabulary_prefilter(
        root=e13_specimen,
        repo_root=tmp_path,
        row_budget=SYNTHETIC_VOCAB_SIZE,
        emit=tmp_path / "hardlink.json",
    )

    assert result["status"] == "WITHHELD_HARDLINKED_INPUT_REFUSED"
    assert result["tensor_payload_bytes_read"] == 0


def test_e13_rejects_a_small_safetensors_body_after_only_its_length_prefix(tmp_path, monkeypatch):
    disguised = tmp_path / "disguised.json"
    _write_tensor(disguised, "not_json", shape=[4, 4])
    original = vocabulary_prefilter._read_fd_up_to
    observed_limits = []

    def length_prefix_only(fd, *, maximum):
        observed_limits.append(maximum)
        if maximum > 8:
            raise AssertionError("a safetensors body disguised as JSON was read")
        return original(fd, maximum=maximum)

    monkeypatch.setattr(vocabulary_prefilter, "_read_fd_up_to", length_prefix_only)
    with pytest.raises(vocabulary_prefilter._Withheld) as exc:
        vocabulary_prefilter._read_bounded_bytes(
            disguised,
            maximum=vocabulary_prefilter.MAX_PLAN_BYTES,
            label="disguised control artifact",
        )

    assert exc.value.code == "SAFETENSORS_INPUT_REFUSED"
    assert observed_limits == [8]


def test_e13_compact_utf8_receipt_seal_rejects_tampering():
    sealed = vocabulary_prefilter._seal_receipt(
        {
            "schema": vocabulary_prefilter.SCHEMA,
            "status": "WITHHELD_TEST",
            "unicode_control": "漢字",
        }
    )

    assert vocabulary_prefilter.verify_flash_vocabulary_prefilter_receipt(sealed) == sealed
    assert b"\xe6\xbc\xa2\xe5\xad\x97" in vocabulary_prefilter._canonical_receipt_bytes(
        {key: value for key, value in sealed.items() if key != "seal_sha256"}
    )
    tampered = dict(sealed)
    tampered["status"] = "TAMPERED"
    with pytest.raises(ValueError, match="seal mismatch"):
        vocabulary_prefilter.verify_flash_vocabulary_prefilter_receipt(tampered)


def test_cli_exposes_e13_metadata_only_prefilter():
    args = build_parser().parse_args(["flash-vocabulary-prefilter", "--verify-receipt", "receipt.json"])

    assert args.command == "flash-vocabulary-prefilter"
    assert args.row_budget == vocabulary_prefilter.DEFAULT_ROW_BUDGET
    assert args.verify_receipt == "receipt.json"


def test_producer_identity_stays_canonical_under_main_module_execution(monkeypatch):
    monkeypatch.setattr(vocabulary_prefilter, "__name__", "__main__")

    identity = vocabulary_prefilter._producer_identity()

    assert identity["module"] == "hawking.agentos.flash_vocabulary_prefilter"
    assert identity["module_sha256"] == identity["source_closure"][identity["module"]]["sha256"]
