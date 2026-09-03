"""doctor6 collect reads Q80 post-SwiGLU packs the same way it reads hiddens."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    collect_expert_activations,
)


def _write_f32le(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(rows, dtype="<f4").tofile(path)


def test_packed_swiglu_collect_keyed_by_layer_expert(tmp_path: Path) -> None:
    run = tmp_path / "cap"
    # Two organs, 3 and 2 rows, width 512, deterministic fill.
    a = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    b = np.arange(1000, 1000 + 2 * 512, dtype=np.float32).reshape(2, 512)
    _write_f32le(run / "x/swiglu_hidden_routed/L03/E007.f32le", a)
    _write_f32le(run / "x/swiglu_hidden_routed/L10/E453.f32le", b)
    (run / "capture-result.json").write_text(
        json.dumps({"schema": "test", "probes": []})
    )

    stacked, prov = collect_expert_activations(run, x_kind="swiglu_hidden_routed")
    assert set(stacked) == {(3, 7), (10, 453)}
    assert stacked[(3, 7)].shape == (3, 512)
    assert stacked[(10, 453)].shape == (2, 512)
    np.testing.assert_array_equal(stacked[(3, 7)], a)
    np.testing.assert_array_equal(stacked[(10, 453)], b)
    assert prov["x_kind"] == "swiglu_hidden_routed"
    assert prov["packed_swiglu"] is True
    assert prov["swiglu_width"] == 512

    only, _ = collect_expert_activations(
        run, x_kind="swiglu_hidden_routed", wanted_keys={(10, 453)}
    )
    assert set(only) == {(10, 453)}


def test_default_collect_still_reads_hiddens(tmp_path: Path) -> None:
    run = tmp_path / "cap"
    hidden = np.arange(4 * 2048, dtype=np.float32).reshape(4, 2048)
    rel = "hidden/L00/p0/000000.f32le"
    _write_f32le(run / rel, hidden[0])
    cap = {
        "schema": "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1",
        "probes": [
            {
                "probe_id": "p0",
                "steps": [
                    {
                        "position": 0,
                        "layers": [
                            {
                                "layer": 0,
                                "selected_expert_ids": [3, 7],
                                "normalized_route_weights": [0.5, 0.5],
                                "router_input_hidden_f32le": {
                                    "relative_path": rel,
                                    "elements": 2048,
                                },
                                "hidden_retained": True,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (run / "capture-result.json").write_text(json.dumps(cap))
    stacked, prov = collect_expert_activations(run, cap)
    assert prov["x_kind"] == "router_input"
    assert stacked[(0, 3)].shape == (1, 2048)
    assert stacked[(0, 7)].shape == (1, 2048)
