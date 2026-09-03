"""Fail-closed tests for the read-only Qwen3.8 protected verifier."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lab.qwen38_protected_run_verifier import (
    CAPTURE_SCHEMA,
    QWEN38_MODEL,
    QWEN38_SAY_HI_GREEDY_IDS,
    QWEN38_SAY_HI_PROMPT_IDS,
    Qwen38VerificationError,
    verify_qwen38_capture,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = artifact / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
                "complete_physical_bpw": 4.252735126866492,
            },
            sort_keys=True,
        )
    )
    runtime = tmp_path / "qwen38-runtime"
    runtime.write_bytes(b"protected runtime fixture\n")
    runtime.chmod(runtime.stat().st_mode | 0o111)
    kernel = tmp_path / "qwen_uniform_q4.metal"
    kernel.write_text("kernel void qwen_uniform_q4_fixture() {}\n")

    ids = list(QWEN38_SAY_HI_GREEDY_IDS[:16])
    raw_reps = (
        [37_000_000, 37_100_000, 37_200_000],
        [36_900_000, 37_050_000, 37_300_000],
        [37_020_000, 37_110_000, 37_220_000],
    )
    rep_medians = [sorted(values)[len(values) // 2] for values in raw_reps]
    headline = sorted(rep_medians)[len(rep_medians) // 2]
    capture: dict[str, object] = {
        "schema": CAPTURE_SCHEMA,
        "binding": {
            "artifact_manifest_sha256": _sha(manifest),
            "runtime_executable_sha256": _sha(runtime),
            "kernel_source_sha256": _sha(kernel),
        },
        "identity": {
            "model": QWEN38_MODEL,
            "prompt_ids": list(QWEN38_SAY_HI_PROMPT_IDS),
            "max_new_tokens": 16,
            "greedy_new_token_ids": ids,
            "fallbacks": 0,
        },
        "vehicle": {"artifact_root": str(artifact)},
        "authority": {
            "rep_median_complete_wall_ns": rep_medians,
            "headline_complete_wall_ns_per_token": headline,
            "headline_complete_tps": 1_000_000_000.0 / headline,
        },
        "warm_reps": [
            {
                "summary": {
                    "fallbacks": 0,
                    "new_token_ids": list(ids),
                    "n_steady_decode_steps": len(values),
                    "steady_decode": {"complete_wall_ns": {"all": list(values)}},
                }
            }
            for values in raw_reps
        ],
    }
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture))
    return artifact, runtime, kernel, capture_path, capture


def _rewrite(path: Path, capture: dict[str, object]) -> None:
    path.write_text(json.dumps(capture))


def test_verifier_derives_file_hashes_and_wall_from_raw_samples(tmp_path: Path) -> None:
    artifact, runtime, kernel, capture_path, _ = _fixture(tmp_path)

    result = verify_qwen38_capture(
        capture_path=capture_path,
        artifact_root=artifact,
        runtime_executable=runtime,
        kernel_source=kernel,
        max_wall_ns=37_200_000,
    )

    assert result["status"] == "PASS"
    binding = result["protected_binding"]
    assert binding["artifact_manifest_sha256"] == _sha(artifact / "manifest.json")
    assert binding["runtime_executable_sha256"] == _sha(runtime)
    assert binding["kernel_source_sha256"] == _sha(kernel)
    assert binding["candidate_hashes_were_not_authority"] is True
    assert result["measurement"]["derived_rep_median_complete_wall_ns"] == [
        37_100_000,
        37_050_000,
        37_110_000,
    ]
    assert result["measurement"]["derived_headline_complete_wall_ns_per_token"] == 37_100_000
    assert result["claim_boundary"]["gpu_work_launched"] is False
    assert result["claim_boundary"]["capture_origin_attested"] is False


@pytest.mark.parametrize("target", ["manifest", "runtime", "kernel"])
def test_candidate_hash_cannot_hide_changed_protected_file(
    tmp_path: Path, target: str
) -> None:
    artifact, runtime, kernel, capture_path, _ = _fixture(tmp_path)
    if target == "manifest":
        (artifact / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
                    "tampered": True,
                }
            )
        )
    elif target == "runtime":
        runtime.write_bytes(b"different runtime bytes\n")
        runtime.chmod(runtime.stat().st_mode | 0o111)
    else:
        kernel.write_text("kernel void tampered() {}\n")

    with pytest.raises(Qwen38VerificationError, match="independently hashed"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
        )


def test_top_level_or_repetition_greedy_drift_fails(tmp_path: Path) -> None:
    artifact, runtime, kernel, capture_path, capture = _fixture(tmp_path)
    warm_reps = capture["warm_reps"]
    assert isinstance(warm_reps, list)
    warm_reps[1]["summary"]["new_token_ids"][-1] = 999  # type: ignore[index]
    _rewrite(capture_path, capture)

    with pytest.raises(Qwen38VerificationError, match="greedy ids"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
        )


def test_any_silent_fallback_fails(tmp_path: Path) -> None:
    artifact, runtime, kernel, capture_path, capture = _fixture(tmp_path)
    identity = capture["identity"]
    assert isinstance(identity, dict)
    identity["fallbacks"] = 1
    _rewrite(capture_path, capture)

    with pytest.raises(Qwen38VerificationError, match="silent fallbacks"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
        )


def test_declared_wall_cannot_override_raw_samples_or_protected_limit(
    tmp_path: Path,
) -> None:
    artifact, runtime, kernel, capture_path, capture = _fixture(tmp_path)
    authority = capture["authority"]
    assert isinstance(authority, dict)
    authority["headline_complete_wall_ns_per_token"] = 1
    authority["headline_complete_tps"] = 1_000_000_000.0
    _rewrite(capture_path, capture)

    with pytest.raises(Qwen38VerificationError, match="declared headline wall"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
        )

    authority["headline_complete_wall_ns_per_token"] = 37_100_000
    authority["headline_complete_tps"] = 1_000_000_000.0 / 37_100_000
    _rewrite(capture_path, capture)
    with pytest.raises(Qwen38VerificationError, match="protected limit"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
            max_wall_ns=37_099_999,
        )


def test_runtime_must_be_an_actual_executable_file(tmp_path: Path) -> None:
    artifact, runtime, kernel, capture_path, _ = _fixture(tmp_path)
    runtime.chmod(runtime.stat().st_mode & ~0o111)
    assert not os.access(runtime, os.X_OK)

    with pytest.raises(Qwen38VerificationError, match="not executable"):
        verify_qwen38_capture(
            capture_path=capture_path,
            artifact_root=artifact,
            runtime_executable=runtime,
            kernel_source=kernel,
        )
