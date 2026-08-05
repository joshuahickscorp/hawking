"""Offline contracts for the long-running public-path confirmation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_xet_sustained as sustained  # noqa: E402


def test_curl_status_marker_and_error_redaction_are_bounded() -> None:
    assert sustained._CURL_MARKER.findall("x TG_CURL_TRANSFER:206:1048576.0 y") == [("206", "1048576.0")]
    assert "token=secret" not in sustained._safe_error("curl https://host/path?token=secret failed")
    assert "redacted-presigned-host" in sustained._safe_error("curl https://host/path?token=secret failed")


def test_curl_config_keeps_exact_range_stderr_status_and_url_out_of_argv() -> None:
    target = {"start": 100, "end": 200}
    metadata = {"signed_location": "https://example.invalid/signed"}
    config = sustained._curl_config(
        target,
        metadata,
        {"connect_timeout": 20.0, "read_timeout": 120.0, "max_attempts": 4, "base_delay": 0.25, "max_duration": 20.0},
    )
    assert 'range = "100-199"' in config
    assert "%{stderr}" in config
    assert 'url = "https://example.invalid/signed"' in config


def test_rank_requires_zero_errors_and_uncontended_exact_206() -> None:
    candidate = {
        "status": "PASS",
        "trial": {
            "steady_state": {"retry_count": 0, "http_status_checks": {"all_exact_206": True}, "sealed_and_evicted_bytes_per_second": 2.0},
            "host": {"before": {"contention_observed": False}, "after": {"contention_observed": False}},
        },
    }
    assert sustained._rank([candidate]) == [candidate]
    candidate["trial"]["steady_state"]["retry_count"] = 1
    assert sustained._rank([candidate]) == []


def test_sustained_run_is_longer_than_burst_control() -> None:
    assert sustained.DEFAULT_SUSTAINED_ROUNDS > sustained.DEFAULT_PROBE_ROUNDS
    assert sustained.DEFAULT_SUSTAINED_ROUNDS * 65 * 1024**2 > 100 * 1024**3


def test_retry_generation_is_immutable_and_validated() -> None:
    assert sustained._generated_name("result.json", "RETRY_AFTER_CONTENTION") == "result_RETRY_AFTER_CONTENTION.json"
    with pytest.raises(sustained.DeepSeekV4XetSustainedError):
        sustained._generated_name("result.json", "not-safe")


def test_balanced_rotating_batches_preserve_exact_multiplicity() -> None:
    targets = [{"shard": f"model-{number}", "length": number} for number in range(1, 9)]
    batches = sustained._balanced_rotating_batches(targets, rounds=12)
    assert len(batches) == 2
    assert all(len(batch) == sustained.CURL_BATCH_RANGES for batch in batches)
    assert sum(1 for batch in batches for row in batch if row["shard"] == "model-1") == 12
    assert batches[0][:8] != batches[0][8:16]


def test_successor_artifact_names_are_generation_safe() -> None:
    assert sustained._generated_name(sustained.SUSTAINED_ROOFLINE_NAME, "RETRY") == "TG_XET_PUBLIC_PATH_ROOFLINE_SUSTAINED_RETRY.json"
    assert sustained._generated_name(sustained.ACTIVE_CONFIG_NAME, "RETRY") == "TG_XET_REAL_STREAM_ACTIVE_CONFIG_RETRY.json"
