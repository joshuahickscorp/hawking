"""Doctor6 readability of a produced DSV4F source capture, when present."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.operators.dsv4f_activation_x_source_verify import verify_doctor6


def _default_run() -> Path:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "workspace/campaign/records/ascension-sandbox/physical/dsv4f/quality-diagnostics"
        / "activation-x-source-v1"
    )


def test_produced_source_capture_is_doctor6_readable() -> None:
    run = _default_run()
    if not (run / "capture-result.json").is_file():
        pytest.skip(f"source capture not produced yet: {run}")
    report = verify_doctor6(run)
    assert report["doctor6_key_count"] > 0
    assert report["all_organs_finite"] is True
    assert report["sample_organ"]["X_shape"] is not None
    assert report["sample_organ"]["X_shape"][1] == 4096
    assert report["all_layer"] is True
    n_fit = report["n_fit_distribution"]
    assert "p10" in n_fit and "p50" in n_fit
    assert n_fit["n_layer_expert_pairs"] == n_fit["layers"] * n_fit["experts"]
