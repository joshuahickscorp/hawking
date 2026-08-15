"""Coverage assertions for the read-only DSV4F tensor schedule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from lab.operators.dsv4f_tensor_schedule import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_TENSOR_BYTES,
    ORGAN_CLASSES,
    TensorRow,
    classify_all,
    classify_tensor,
    iter_manifest_tensors,
    resolve_artifact_root,
)

REPO = Path(__file__).resolve().parents[2]
SCHEDULE_JSON = REPO / "workspace/campaign/records/runs/deepseek-v4/DSV4F_TENSOR_SCHEDULE.json"
SCHEDULE_MD = REPO / "workspace/campaign/records/runs/deepseek-v4/DSV4F_TENSOR_SCHEDULE.md"


@pytest.fixture(scope="module")
def artifact_root() -> Path:
    try:
        return resolve_artifact_root()
    except FileNotFoundError as exc:
        pytest.fail(f"sealed DSV4F artifact is required for this test: {exc}")


@pytest.fixture(scope="module")
def classified(artifact_root: Path) -> dict:
    manifest = artifact_root / "manifest.json"
    return classify_all(iter_manifest_tensors(manifest))


def test_dsv4f_classifier_covers_every_manifest_tensor(classified: dict) -> None:
    organs = classified["organs"]
    class_count_sum = sum(organs[name].tensor_count for name in ORGAN_CLASSES)
    assert classified["count"] == EXPECTED_TENSOR_COUNT
    assert class_count_sum == EXPECTED_TENSOR_COUNT
    assert class_count_sum == classified["count"]


def test_dsv4f_class_byte_masses_sum_to_manifest_total(classified: dict) -> None:
    organs = classified["organs"]
    byte_sum = sum(organs[name].byte_mass for name in ORGAN_CLASSES)
    assert classified["byte_sum"] == EXPECTED_TOTAL_TENSOR_BYTES
    assert byte_sum == EXPECTED_TOTAL_TENSOR_BYTES
    assert classified["shape_mismatch"] == 0
    assert classified["undetermined"] == []


def test_dsv4f_other_bucket_is_only_mtp_unique_projections(classified: dict) -> None:
    names = classified["organs"]["other"].names_if_other
    assert names == [
        "mtp.0.e_proj.scale",
        "mtp.0.e_proj.weight",
        "mtp.0.h_proj.scale",
        "mtp.0.h_proj.weight",
    ]


def test_dsv4f_known_name_taxonomy() -> None:
    cases = [
        ("embed.weight", "BF16", (129280, 4096), 1059061760, "embeddings"),
        ("head.weight", "BF16", (129280, 4096), 1059061760, "lm_head"),
        ("norm.weight", "BF16", (4096,), 8192, "norms"),
        ("hc_head_fn", "F32", (4, 16384), 262144, "mhc"),
        ("layers.0.ffn.experts.7.w1.weight", "I8", (2048, 2048), 4194304, "routed_expert"),
        ("layers.0.ffn.experts.7.w1.scale", "F8_E8M0", (2048, 128), 262144, "routed_expert"),
        ("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3", (4096, 2048), 8388608, "shared_expert"),
        ("layers.2.attn.wq_a.weight", "F8_E4M3", (1024, 4096), 4194304, "mla"),
        ("layers.2.attn.indexer.wq_b.weight", "F8_E4M3", (8192, 1024), 8388608, "indexer_compressor"),
        ("layers.0.ffn.gate.tid2eid", "I64", (129280, 6), 6205440, "hash_layers"),
        ("layers.5.ffn.gate.weight", "BF16", (256, 4096), 2097152, "router_gate"),
        ("layers.5.hc_attn_fn", "F32", (24, 16384), 1572864, "mhc"),
        ("mtp.0.ffn.experts.0.w3.weight", "I8", (2048, 2048), 4194304, "routed_expert"),
        ("mtp.0.e_proj.weight", "F8_E4M3", (4096, 4096), 16777216, "other"),
    ]
    for name, dtype, shape, nbytes, organ in cases:
        row = TensorRow(name=name, dtype=dtype, shape=shape, bytes=nbytes)
        assert classify_tensor(row).organ == organ, name


def test_dsv4f_fp4_logical_params_are_unpacked() -> None:
    row = TensorRow(
        name="layers.0.ffn.experts.0.w1.weight",
        dtype="I8",
        shape=(2048, 2048),
        bytes=4194304,
    )
    classified = classify_tensor(row)
    assert classified.logical_params == 2048 * 4096
    assert classified.stored_elements == 2048 * 2048


def test_dsv4f_schedule_files_exist_and_agree_with_classifier(classified: dict) -> None:
    assert SCHEDULE_JSON.is_file()
    assert SCHEDULE_MD.is_file()
    import json

    report = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    assert report["coverage"]["class_count_sum"] == EXPECTED_TENSOR_COUNT
    assert report["coverage"]["byte_mass_sum"] == EXPECTED_TOTAL_TENSOR_BYTES
    assert report["coverage"]["byte_residual"] == 0
    assert report["coverage"]["covers_all_tensors"] is True
    live_counts = Counter(
        {name: classified["organs"][name].tensor_count for name in ORGAN_CLASSES}
    )
    report_counts = Counter(
        {name: report["organs"][name]["tensor_count"] for name in ORGAN_CLASSES}
    )
    assert live_counts == report_counts
    assert sum(report_counts.values()) == EXPECTED_TENSOR_COUNT


def test_dsv4f_prior_moe_active_bytes_reproduce_from_geometry() -> None:
    import json

    report = json.loads(SCHEDULE_JSON.read_text(encoding="utf-8"))
    rec = report["prior_runtime_accounting_check"]["reconstructed_from_manifest_geometry"]
    assert rec["moe_prior_style_six_plus_shared_plus_gate"] == 102_760_448
    assert rec["moe_matches_prior"] is True
    assert rec["mla_plus_shared_plus_one_routed"] == 144_703_488
    assert rec["attention_equals_mla_plus_36mib"] is True
