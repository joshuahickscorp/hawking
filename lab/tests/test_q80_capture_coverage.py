"""Q80 capture coverage: distributions, holdout well-posedness, BF16 bind."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab.operators.q80_capture_coverage import (
    DOWN_RANK_TARGET,
    HIDDEN,
    N_PAIRS,
    SWIGLU,
    n_fit_after_holdout,
    occupancy_histogram,
    organ_verdict,
    summarize,
    swiglu_presence,
)
from lab.operators.q80_capture_index import (
    build_capture_index_from_layer_meta,
    inspect_index,
    try_walk_from_index,
)


def test_summarize_min_median_max():
    x = np.array([0, 1, 2, 3, 100], dtype=np.int32)
    d = summarize(x, "t")
    assert d["min"] == 0
    assert d["p50"] == 2
    assert d["max"] == 100
    assert d["zero"] == 1
    assert d["below_8"] == 4
    assert d["at_or_above_64"] == 1


def test_holdout_split_matches_n_lt_4_keeps_all():
    n = np.array([0, 1, 3, 4, 8, 100], dtype=np.int32)
    fit = n_fit_after_holdout(n)
    assert int(fit[0]) == 0
    assert int(fit[1]) == 1
    assert int(fit[2]) == 3
    # n=4: n_hold = max(1, min(2, round(1))) = 1 → fit 3
    assert int(fit[3]) == 3
    # n=8: n_hold = max(1, min(4, 2)) = 2 → fit 6
    assert int(fit[4]) == 6
    # n=100: n_hold = max(1, min(50, 25)) = 25 → fit 75
    assert int(fit[5]) == 75


def test_occupancy_histogram_zero_band():
    x = np.array([0, 0, 5, 200], dtype=np.int32)
    h = occupancy_histogram(x)
    zero = next(b for b in h if b["label"] == "0")
    assert zero["count"] == 2
    mid = next(b for b in h if b["label"] == "4-7")
    assert mid["count"] == 1


def test_organ_verdict_names_underdetermined():
    fit = np.zeros((48, 512), dtype=np.int32)
    fit[:, :] = 300
    fit[0, 0] = 0
    fit[1, 1] = 2048
    organs = organ_verdict(fit)
    by = {o["organ"]: o for o in organs}
    assert by["gate_proj"]["fitted_dim"] == HIDDEN
    assert by["up_proj"]["fitted_dim"] == HIDDEN
    assert by["down_proj"]["fitted_dim"] == SWIGLU
    assert by["down_proj"]["rank_target"] == DOWN_RANK_TARGET
    # 300 < 2048 for almost every pair
    assert by["gate_proj"]["underdetermined"] is True
    assert by["gate_proj"]["pairs_rows_ge_fitted_dim"] == 1
    assert by["gate_proj"]["pairs_rows_lt_fitted_dim"] == N_PAIRS - 1
    # 300 < 512 for down_proj except the one 2048 cell
    assert by["down_proj"]["underdetermined"] is True
    assert by["down_proj"]["pairs_rows_lt_rank"] == 1  # only the zero
    assert by["down_proj"]["pairs_rows_ge_fitted_dim"] == 1


def test_swiglu_presence_absent(tmp_path: Path):
    run = tmp_path / "cap"
    run.mkdir()
    (run / "capture-result.json").write_text("{}")
    info = swiglu_presence(run)
    assert info["packed_dir_present"] is False
    assert info["down_proj_x_on_disk"] is False


def test_swiglu_presence_packed(tmp_path: Path):
    run = tmp_path / "cap"
    packed = run / "x" / "swiglu_hidden_routed" / "L01"
    packed.mkdir(parents=True)
    (packed / "E265.f32le").write_bytes(b"\x00" * 16)
    (run / "capture-result.json").write_text("{}")
    info = swiglu_presence(run)
    assert info["n_packed_organ_files"] == 1
    assert info["down_proj_x_on_disk"] is True


def test_assemble_index_from_layer_meta(tmp_path: Path):
    run = tmp_path / "ext"
    hidden = run / "hidden" / "L00" / "p0"
    hidden.mkdir(parents=True)
    np.asarray([1.0, 2.0, 3.0, 4.0], dtype="<f4").tofile(hidden / "000000.f32le")
    meta = {
        "layer": 0,
        "tokens": [
            {
                "pi": 0,
                "pos": 0,
                "selected_expert_ids": [3, 7],
                "hidden_retained": True,
                "hidden": {
                    "relative_path": "hidden/L00/p0/000000.f32le",
                    "elements": 4,
                },
            },
            {
                "pi": 0,
                "pos": 1,
                "selected_expert_ids": [3],
                "hidden_retained": False,
                "hidden": None,
            },
        ],
    }
    (run / "layer_meta").mkdir()
    (run / "layer_meta" / "L00.json").write_text(json.dumps(meta))
    (run / "checkpoint.json").write_text(
        json.dumps({"probe_ids": ["p0"], "token_counts": [2]})
    )
    # Small scalars JSON so inspect_index can bind size+mtime.
    (run / "capture-result.json").write_text(
        json.dumps({"schema": "test", "bounded_storage": {"hidden_width": 4}})
    )
    built = build_capture_index_from_layer_meta(run)
    assert built["n_rows"] == 2
    assert built["n_tokens"] == 2
    assert built["n_keys"] == 2  # (0,3) and (0,7) — pos1 not retained
    assert inspect_index(run)[0] == "ok"
    walked = try_walk_from_index(run, wanted_keys=None, load_vectors=False)
    assert walked is not None
    counts, _ = walked
    assert counts.get((0, 3)) == 1
    assert counts.get((0, 7)) == 1
