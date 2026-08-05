"""Offline contracts for the DeepSeek-V4 public-Xet autotuner."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_xet_autotune as autotune  # noqa: E402
from lab.receipts import seal  # noqa: E402


def _digest(seed: str) -> str:
    return (seed * 64)[:64]


def _corpus() -> dict:
    categories = (
        "small_metadata_control",
        "medium_contiguous_fp8",
        "large_fp4_expert_payload",
        "cross_shard_streaming_window",
    )
    ranges = []
    for index, category in enumerate(categories, start=1):
        start = index * 1024
        length = 1024 * 1024
        ranges.append(
            {
                "category": category,
                "repository": autotune.REPOSITORY,
                "revision": autotune.REVISION,
                "shard": f"model-{index:05d}-of-00046.safetensors",
                "commit_hash": autotune.REVISION,
                "lfs_sha256": _digest(str(index)),
                "xet_file_hash": _digest(str(index + 4)),
                "file_size_bytes": 32 * 1024 * 1024,
                "start": start,
                "end": start + length,
                "length": length,
                "sha256": _digest(str(index + 8)),
                "tensor": None,
                "dtype": None,
                "descriptor": None,
                "discovery_response": {"host": "example.invalid", "http_status": 206, "wire_bytes": length},
            }
        )
    return seal(
        {
            "schema": autotune.CORPUS_SCHEMA,
            "status": "SEALED_FIXED_PUBLIC_CORPUS",
            "created_at": "2026-08-04T00:00:00Z",
            "source": {"repository": autotune.REPOSITORY, "revision": autotune.REVISION},
            "ranges": ranges,
            "constraints": {"source_shard_count": 4},
        }
    )


def test_broad_matrix_covers_requested_controls_without_cartesian_product() -> None:
    rows = autotune.broad_candidates()
    fixed = {
        row["fixed_download_concurrency"]
        for row in rows
        if row["id"].startswith("OFFICIAL_FIXED_")
    }
    files = {
        row["file_download_concurrency"]
        for row in rows
        if row["id"].startswith("OFFICIAL_FILES_")
    }
    assert fixed == set(autotune.FIXED_DOWNLOAD_CONCURRENCIES)
    assert files == set(autotune.FILE_DOWNLOAD_CONCURRENCIES)
    assert len(rows) < len(autotune.FIXED_DOWNLOAD_CONCURRENCIES) * len(
        autotune.FILE_DOWNLOAD_CONCURRENCIES
    )
    assert {row["transport"] for row in rows} >= {
        "official_hf_xet",
        "custom_direct_xet_range",
        "hub_http_without_xet",
        "direct_presigned_range",
    }


def test_profile_environment_isolated_zero_cache_and_duration_safe(tmp_path: Path) -> None:
    profile = autotune._profile(
        "TEST",
        adaptive=True,
        fixed=16,
        files=4,
        adaptive_min=4,
        adaptive_initial=16,
        adaptive_max=64,
    )
    environment = autotune.profile_environment(profile, tmp_path / "child")
    assert environment["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"
    assert environment["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] == "false"
    assert environment["HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE"] == "true"
    assert environment["HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"] == "true"
    assert environment["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] == "16"
    assert environment["HF_XET_CLIENT_RETRY_BASE_DELAY"].endswith("s")
    assert environment["HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"] == "16"
    assert Path(environment["HF_XET_CACHE"]).parent == tmp_path / "child"


def test_fixed_corpus_requires_source_identity_hashes_and_four_shards() -> None:
    corpus = _corpus()
    assert autotune.validate_fixed_corpus(corpus)["seal_sha256"] == corpus["seal_sha256"]
    broken = dict(corpus)
    broken.pop("seal_sha256")
    broken["ranges"] = list(corpus["ranges"])
    broken["ranges"][3] = dict(broken["ranges"][0])
    with pytest.raises(autotune.DeepSeekV4XetAutotuneError, match="duplicate"):
        autotune.validate_fixed_corpus(seal(broken))


def test_scheduler_shapes_are_bounded_to_eight_outer_source_windows() -> None:
    profile = autotune._profile("TEST", files=124)
    assert autotune._shape_workers("one_file_many_ranges", profile) == 1
    assert autotune._shape_workers("two_files_medium_range_concurrency", profile) == 2
    assert autotune._shape_workers("four_files_medium_range_concurrency", profile) == 4
    assert autotune._shape_workers("eight_files_low_per_file_concurrency", profile) == 8
    assert autotune._shape_workers("dynamic_work_stealing", profile) == 8


def test_ordered_jobs_rotate_corpus_without_changing_exact_ranges() -> None:
    corpus = _corpus()
    jobs = autotune._ordered_jobs(corpus, rounds=3, shape="dynamic_work_stealing")
    assert len(jobs) == 12
    assert {row["sha256"] for row in jobs} == {row["sha256"] for row in corpus["ranges"]}
    assert jobs[:4] != jobs[4:8]


def test_launcher_is_immutable_executable_and_uses_resume_guard(tmp_path: Path) -> None:
    launcher = tmp_path / autotune.LAUNCHER_NAME
    row = autotune._write_launcher(
        launcher,
        winner_path=tmp_path / autotune.WINNER_NAME,
        artifact=tmp_path / "full-43-layer-stream.gravity",
    )
    assert launcher.read_text(encoding="utf-8").startswith("#!/bin/zsh")
    assert "resume-real" in launcher.read_text(encoding="utf-8")
    assert stat.S_IMODE(os.lstat(launcher).st_mode) == 0o755
    assert row["mode"] == "0755"
