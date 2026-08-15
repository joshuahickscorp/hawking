#!/usr/bin/env python3.12
"""CPU tests for DSV4F offline host-activation export format the loader expects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab.operators.frankenstein_correspondence_loader import (
    LayerRange,
    LoadReport,
    load_dsv4f_layer_matrices_from_traces,
)


def _write_hash_only_trace(path: Path, example_id: str, prompt: str) -> None:
    doc = {
        "schema": "hawking.frankenstein.paired_functional_trace.v1",
        "example_id": example_id,
        "membership": "train",
        "prompt_text": prompt,
        "sides": {
            "dsv4f": {
                "present": True,
                "capture_status": "OK",
                "layers_run": [0, 1],
                "host_activation_handoff_permitted": False,
                "positions": [
                    {
                        "position": 0,
                        "token_id": 1,
                        "layers": [
                            {
                                "layer": 0,
                                "late_hidden_child_hc_sha256": "a" * 64,
                                "post_moe_sha256": "b" * 64,
                            },
                            {
                                "layer": 1,
                                "late_hidden_child_hc_sha256": "c" * 64,
                                "post_moe_sha256": "d" * 64,
                            },
                        ],
                    }
                ],
            }
        },
    }
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")


def test_npy_sidecar_export_loads(tmp_path: Path) -> None:
    """Mirrors gravity_deepseek_v4_fullseq_capture --export-host-activations output."""
    traces = tmp_path / "traces"
    act = tmp_path / "activations"
    traces.mkdir()
    act.mkdir()
    ids = ["pfv0:demo:a", "pfv0:demo:b"]
    for eid in ids:
        _write_hash_only_trace(traces / f"{eid}.json", eid, f"prompt for {eid}")

    rng = np.random.default_rng(0)
    for layer in (0, 1):
        mat = rng.standard_normal((2, 16)).astype(np.float32)
        np.save(act / f"L{layer:02d}.npy", mat)
        (act / f"L{layer:02d}.export.json").write_text(
            json.dumps(
                {
                    "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export.v1",
                    "layer": layer,
                    "site": "late_hidden",
                    "shape": [2, 16],
                    "dtype": "float32",
                    "mode": "offline_analysis_diagnostic",
                    "host_activation_handoff_permitted": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (act / "example_ids.json").write_text(
        json.dumps(
            {
                "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export_ids.v1",
                "example_ids": ids,
                "n": 2,
                "mode": "offline_analysis_diagnostic",
                "host_activation_handoff_permitted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mats, loaded, got_ids = load_dsv4f_layer_matrices_from_traces(
        traces, LayerRange(0, 2), site="late_hidden"
    )
    assert loaded == [0, 1]
    assert got_ids == ids
    assert len(mats) == 2
    assert mats[0].shape == (2, 16)
    assert mats[1].shape == (2, 16)
    assert np.isfinite(mats[0]).all()


def test_hash_only_without_export_blockers(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_hash_only_trace(traces / "v0_math_01.json", "v0_math_01", "Evaluate 1+1.")
    rep = LoadReport()
    mats, loaded, ids = load_dsv4f_layer_matrices_from_traces(
        traces, LayerRange(0, 2), report=rep
    )
    assert mats == []
    assert loaded == []
    assert ids == ["v0_math_01"]
    assert rep.dsv4f_hash_only is True
    assert any("export-host-activations" in b or "sha256" in b for b in rep.blockers)
