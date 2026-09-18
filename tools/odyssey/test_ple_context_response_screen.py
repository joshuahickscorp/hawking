from __future__ import annotations

import hashlib
import json

import numpy as np

from tools.odyssey.ple_context_response_screen import (
    _compact_utf8_canonical_sha256,
    _read_sealed_receipt,
    build_symmetric_context_bank,
    spectral_energy_summary,
    summarize_symmetric_response,
)


def test_synthetic_context_bank_is_symmetric_and_deterministic() -> None:
    base = np.arange(1, 13, dtype=np.float32)
    first, first_binding = build_symmetric_context_bank(
        base, direction_count=3, relative_rms=0.05, seed=123
    )
    second, second_binding = build_symmetric_context_bank(
        base, direction_count=3, relative_rms=0.05, seed=123
    )
    np.testing.assert_array_equal(first, second)
    assert first_binding["context_bank_sha256"] == second_binding["context_bank_sha256"]
    assert first.shape == (7, 12)
    np.testing.assert_array_equal(first[0], base)
    for index in range(3):
        np.testing.assert_allclose(first[1 + 2 * index] + first[2 + 2 * index], 2.0 * base)
        perturbation = first[1 + 2 * index] - base
        assert abs(float(np.dot(perturbation, base))) < 1e-4


def test_spectrum_reports_the_low_dimensional_response_plane() -> None:
    responses = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]],
        dtype=np.float32,
    )
    summary = spectral_energy_summary(responses)
    assert summary["numerical_rank"] == 2
    assert summary["ranks_at_energy"]["0.9"] == 2
    assert len(summary["top_singular_values"]) == 4


def test_symmetric_response_reports_zero_curvature_for_a_linear_map() -> None:
    base = np.array([1.0, -2.0], dtype=np.float32)
    plus_one = np.array([2.0, -1.0], dtype=np.float32)
    minus_one = np.array([0.0, -3.0], dtype=np.float32)
    plus_two = np.array([0.0, -4.0], dtype=np.float32)
    minus_two = np.array([2.0, 0.0], dtype=np.float32)
    summary, deltas = summarize_symmetric_response(
        np.stack([base, plus_one, minus_one, plus_two, minus_two]), direction_count=2
    )
    assert deltas.shape == (4, 2)
    assert summary["normalized_symmetric_curvature"]["max"] == 0.0
    assert summary["local_delta_spectrum"]["numerical_rank"] == 2


def test_receipt_reader_accepts_declared_native_compact_utf8_seal(tmp_path) -> None:
    body = {
        "schema": "test.native.receipt.v1",
        "status": "PASS",
        "unicode": "Flash–Gravity",
    }
    body["seal_sha256"] = _compact_utf8_canonical_sha256(body)
    path = tmp_path / "native.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _document, binding = _read_sealed_receipt(
        path,
        expected_schema="test.native.receipt.v1",
        expected_status="PASS",
        label="native fixture",
    )

    assert binding["seal_format"] == "rust_serde_json_compact_utf8_sorted_v1"
    assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
