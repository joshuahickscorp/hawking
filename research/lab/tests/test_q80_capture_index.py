"""Capture-index sidecar: JSON vs index equivalence, stale binding, fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    ALL_LAYER_RESULT_SCHEMA,
    ROW_CAP_SEED,
    collect_expert_activations,
    count_expert_activations,
)
from lab.operators.q80_capture_index import (
    SCHEMA,
    VALIDITY_BINDING,
    build_capture_index,
    inspect_index,
    try_walk_from_index,
)


def _write_hidden(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(path)


def _layer_row(layer: int, experts: list[int], rel: str | None, elements: int = 4) -> dict:
    hidden = None
    if rel is not None:
        hidden = {
            "bytes": elements * 4,
            "elements": elements,
            "relative_path": rel,
        }
    return {
        "hidden_retained": hidden is not None,
        "layer": layer,
        "normalized_route_weights": [1.0 / max(len(experts), 1)] * len(experts),
        "router_input_hidden_f32le": hidden,
        "selected_expert_ids": experts,
    }


def _build_all_layer(tmp: Path, *, pretty: bool = False) -> Path:
    run = tmp / ("pretty-run" if pretty else "compact-run")
    hidden = run / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    plan = [
        [
            (0, [1, 2], [0.0, 1.0, 2.0, 3.0]),
            (12, [3, 4], [4.0, 5.0, 6.0, 7.0]),
            (24, [5, 6], [8.0, 9.0, 10.0, 11.0]),
            (36, [7, 8], [12.0, 13.0, 14.0, 15.0]),
        ],
        [
            (0, [1, 3], [16.0, 17.0, 18.0, 19.0]),
            (12, [3, 5], [20.0, 21.0, 22.0, 23.0]),
            (24, [5, 7], [24.0, 25.0, 26.0, 27.0]),
            (36, [7, 1], [28.0, 29.0, 30.0, 31.0]),
        ],
        [
            (0, [1, 4], [32.0, 33.0, 34.0, 35.0]),
            (12, [3, 6], [36.0, 37.0, 38.0, 39.0]),
            (24, [5, 8], [40.0, 41.0, 42.0, 43.0]),
            (36, [7, 2], [44.0, 45.0, 46.0, 47.0]),
        ],
    ]
    steps = []
    for t_i, layers in enumerate(plan):
        layer_rows = []
        for layer, experts, vec in layers:
            rel = f"hidden/t{t_i}_L{layer:02d}.f32le"
            _write_hidden(run / rel, vec)
            layer_rows.append(_layer_row(layer, experts, rel))
        layer_rows.append(_layer_row(47, [99], None))
        steps.append(
            {
                "position": t_i,
                "input_token_id": 1000 + t_i,
                "hidden_retained_for_this_token": True,
                "layers": layer_rows,
            }
        )
    doc = {
        "schema": ALL_LAYER_RESULT_SCHEMA,
        "bounded_storage": {
            "strategy": "test",
            "hidden_width": 4,
            "hidden_rows_retained_total": 12,
            "total_tokens_executed": 3,
        },
        "capture_summary": {
            "all_layer_activation_capture": True,
            "total_tokens": 3,
            "hidden_rows_retained_total": 12,
        },
        "probes": [{"probe_id": "toy", "steps": steps}],
        "claim_boundary": {"test": True},
        "status": "TEST_CAPTURE",
    }
    payload = (
        json.dumps(doc, indent=2) if pretty else json.dumps(doc, separators=(",", ":"))
    )
    (run / "capture-result.json").write_text(payload)
    return run


def _build_l0(tmp: Path) -> Path:
    run = tmp / "l0-run"
    run.mkdir()
    _write_hidden(run / "h0.f32le", [1.0, 2.0, 3.0, 4.0])
    _write_hidden(run / "h1.f32le", [5.0, 6.0, 7.0, 8.0])
    doc = {
        "schema": "hawking.ascension.qwen30_broad_activation_layer0_route_capture_result.v1",
        "status": "TEST_L0",
        "probes": [
            {
                "probe_id": "p0",
                "steps": [
                    {
                        "position": 0,
                        "input_token_id": 7,
                        "selected_expert_ids": [2, 5],
                        "router_input_hidden_f32le": {
                            "relative_path": "h0.f32le",
                            "elements": 4,
                        },
                    },
                    {
                        "position": 1,
                        "input_token_id": 8,
                        "selected_expert_ids": [2],
                        "router_input_hidden_f32le": {
                            "relative_path": "h1.f32le",
                            "elements": 4,
                        },
                    },
                ],
            }
        ],
        "claim_boundary": {"l0": True},
    }
    (run / "capture-result.json").write_text(json.dumps(doc, separators=(",", ":")))
    return run


def test_index_matches_json_count_and_collect_compact(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path, pretty=False)
    cap = json.loads((run / "capture-result.json").read_text())
    built = build_capture_index(run)
    assert built["status"] == "WRITTEN"
    assert built["validity_binding"] == VALIDITY_BINDING
    assert inspect_index(run)[0] == "ok"

    json_counts, _ = count_expert_activations(run, cap)
    idx_counts, idx_prov = count_expert_activations(run, use_index=True)
    stream_counts, stream_prov = count_expert_activations(run, use_index=False)
    assert json_counts == idx_counts == stream_counts
    assert json_counts[(0, 1)] == 3
    assert (47, 99) not in json_counts
    assert idx_prov["capture_index"] is True
    assert idx_prov["capture_index_schema"] == SCHEMA
    assert stream_prov["streamed"] is True
    assert stream_prov.get("capture_index") is not True

    wanted = {(0, 1), (12, 3), (24, 5), (36, 7)}
    json_x, _ = collect_expert_activations(run, cap, wanted_keys=wanted)
    idx_x, idx_cprov = collect_expert_activations(
        run, wanted_keys=wanted, use_index=True
    )
    assert set(json_x) == set(idx_x) == wanted
    for key in wanted:
        assert json_x[key].tobytes() == idx_x[key].tobytes()
    assert idx_cprov["capture_index"] is True


def test_index_matches_pretty_printed_json(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path, pretty=True)
    cap = json.loads((run / "capture-result.json").read_text())
    build_capture_index(run)
    json_counts, _ = count_expert_activations(run, cap)
    idx_counts, _ = count_expert_activations(run, use_index=True)
    assert json_counts == idx_counts
    wanted = {(0, 1), (36, 7)}
    json_x, _ = collect_expert_activations(run, cap, wanted_keys=wanted)
    idx_x, _ = collect_expert_activations(run, wanted_keys=wanted, use_index=True)
    for key in wanted:
        assert json_x[key].tobytes() == idx_x[key].tobytes()


def test_row_cap_is_bit_identical_across_paths(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path)
    cap = json.loads((run / "capture-result.json").read_text())
    build_capture_index(run)
    wanted = {(0, 1), (12, 3)}
    json_x, _ = collect_expert_activations(
        run,
        cap,
        wanted_keys=wanted,
        max_rows_per_expert=2,
        row_sample_seed=ROW_CAP_SEED,
    )
    idx_x, _ = collect_expert_activations(
        run,
        wanted_keys=wanted,
        max_rows_per_expert=2,
        row_sample_seed=ROW_CAP_SEED,
        use_index=True,
    )
    assert set(json_x) == set(idx_x)
    for key in wanted:
        assert json_x[key].shape[0] == 2
        assert json_x[key].tobytes() == idx_x[key].tobytes()


def test_stale_sidecar_is_ignored(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path)
    build_capture_index(run)
    assert inspect_index(run)[0] == "ok"
    src = run / "capture-result.json"
    src.write_text(src.read_text() + "\n")
    assert inspect_index(run)[0] == "stale"
    assert try_walk_from_index(run, wanted_keys=None, load_vectors=False) is None
    counts, prov = count_expert_activations(run)
    assert prov.get("capture_index") is not True
    assert (0, 1) in counts


def test_missing_index_uses_json(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path)
    counts, prov = count_expert_activations(run)
    assert inspect_index(run)[0] == "missing"
    assert prov.get("capture_index") is not True
    assert counts[(0, 1)] == 3


def test_l0_capture_indexes(tmp_path: Path) -> None:
    run = _build_l0(tmp_path)
    cap = json.loads((run / "capture-result.json").read_text())
    built = build_capture_index(run)
    assert built["n_rows"] == 2
    json_counts, _ = count_expert_activations(run, cap)
    idx_counts, _ = count_expert_activations(run, use_index=True)
    assert json_counts == idx_counts
    assert json_counts[(0, 2)] == 2
    assert json_counts[(0, 5)] == 1
    json_x, _ = collect_expert_activations(run, cap)
    idx_x, _ = collect_expert_activations(run, use_index=True)
    for key in json_x:
        assert json_x[key].tobytes() == idx_x[key].tobytes()


def test_env_override_finds_index_for_other_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    run = _build_all_layer(tmp_path)
    built = build_capture_index(run)
    other = tmp_path / "other-run"
    other.mkdir()
    (other / "capture-result.json").write_bytes((run / "capture-result.json").read_bytes())
    # Same bytes but different mtime — bind would be stale if we used other's stamp
    # unless we copy mtime too. Force the other JSON to share size+mtime.
    import os

    src_st = (run / "capture-result.json").stat()
    os.utime(other / "capture-result.json", ns=(src_st.st_atime_ns, src_st.st_mtime_ns))
    monkeypatch.setenv("HAWKING_CAPTURE_INDEX", built["index_dir"])
    status, root, header = inspect_index(other)
    assert status == "ok"
    assert header is not None
    assert root == Path(built["index_dir"])


def test_already_present_skips_rebuild(tmp_path: Path) -> None:
    run = _build_all_layer(tmp_path)
    first = build_capture_index(run)
    second = build_capture_index(run)
    assert first["status"] == "WRITTEN"
    assert second["status"] == "ALREADY_PRESENT"
    forced = build_capture_index(run, force=True)
    assert forced["status"] == "WRITTEN"
