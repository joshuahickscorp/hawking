"""Memory-bounded prescribe loader: sample-first, materialize-only-sampled."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    ALL_LAYER_RESULT_SCHEMA,
    collect_expert_activations,
    count_expert_activations,
    subsample_expert_rows,
)
from lab.operators.doctor6.prescribe import (
    SAMPLE_SEED,
    _row_count,
    deterministic_sample,
)


def _write_hidden(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(path)


def _layer_row(layer: int, experts: list[int], rel: str, elements: int = 4) -> dict:
    return {
        "hidden_retained": True,
        "layer": layer,
        "router_input_hidden_f32le": {
            "bytes": elements * 4,
            "elements": elements,
            "relative_path": rel,
        },
        "selected_expert_ids": experts,
    }


def _build_capture(tmp: Path) -> Path:
    """Tiny all-layer capture: 4 bands × several experts, 3 hidden rows each."""
    run = tmp / "full-run"
    hidden = run / "hidden"
    hidden.mkdir(parents=True, exist_ok=True)
    probes = []
    # Three tokens. Each token routes to a distinct expert per band-layer.
    # Layers 0, 12, 24, 36 sit in the four prescribe bands.
    plan = [
        # token 0
        [(0, [1, 2], [0.0, 1.0, 2.0, 3.0]),
         (12, [3, 4], [4.0, 5.0, 6.0, 7.0]),
         (24, [5, 6], [8.0, 9.0, 10.0, 11.0]),
         (36, [7, 8], [12.0, 13.0, 14.0, 15.0])],
        # token 1
        [(0, [1, 3], [16.0, 17.0, 18.0, 19.0]),
         (12, [3, 5], [20.0, 21.0, 22.0, 23.0]),
         (24, [5, 7], [24.0, 25.0, 26.0, 27.0]),
         (36, [7, 1], [28.0, 29.0, 30.0, 31.0])],
        # token 2
        [(0, [1, 4], [32.0, 33.0, 34.0, 35.0]),
         (12, [3, 6], [36.0, 37.0, 38.0, 39.0]),
         (24, [5, 8], [40.0, 41.0, 42.0, 43.0]),
         (36, [7, 2], [44.0, 45.0, 46.0, 47.0])],
    ]
    steps = []
    for t_i, layers in enumerate(plan):
        layer_rows = []
        for layer, experts, vec in layers:
            rel = f"hidden/t{t_i}_L{layer:02d}.f32le"
            _write_hidden(run / rel, vec)
            layer_rows.append(_layer_row(layer, experts, rel))
        # One route-only layer (null hidden) must not increment counts.
        layer_rows.append(
            {
                "hidden_retained": False,
                "layer": 47,
                "router_input_hidden_f32le": None,
                "selected_expert_ids": [99],
            }
        )
        steps.append({"position": t_i, "layers": layer_rows})
    probes.append({"probe_id": "toy", "steps": steps})
    doc = {
        "schema": ALL_LAYER_RESULT_SCHEMA,
        "bounded_storage": {"strategy": "test"},
        "probes": probes,
        "claim_boundary": {"test": True},
    }
    (run / "capture-result.json").write_text(json.dumps(doc, separators=(",", ":")))
    return run


def test_row_count_accepts_arrays_and_ints() -> None:
    X = np.zeros((7, 3), dtype=np.float32)
    assert _row_count(X) == 7
    assert _row_count(7) == 7
    assert _row_count(np.int64(7)) == 7


def test_count_matches_collect_and_streaming_matches_in_memory(tmp_path: Path) -> None:
    run = _build_capture(tmp_path)
    cap = json.loads((run / "capture-result.json").read_text())
    counts_mem, _ = count_expert_activations(run, cap)
    counts_stream, stream_prov = count_expert_activations(run)
    stacked, _ = collect_expert_activations(run, cap)
    assert counts_mem == counts_stream
    assert counts_mem == {k: int(v.shape[0]) for k, v in stacked.items()}
    # Expert 1 at L0 is on every token → 3 rows. Expert 99 is route-only → absent.
    assert counts_mem[(0, 1)] == 3
    assert (47, 99) not in counts_mem
    assert stream_prov["streamed"] is True
    assert stream_prov["counts_only"] is True
    assert stream_prov["capture_schema"] == ALL_LAYER_RESULT_SCHEMA
    assert stream_prov["bounded_storage"] == {"strategy": "test"}


def test_wanted_keys_rows_are_bit_identical(tmp_path: Path) -> None:
    run = _build_capture(tmp_path)
    cap = json.loads((run / "capture-result.json").read_text())
    full, _ = collect_expert_activations(run, cap)
    wanted = {(0, 1), (12, 3), (24, 5), (36, 7)}
    bounded, prov = collect_expert_activations(run, wanted_keys=wanted)
    assert set(bounded) == wanted
    for key in wanted:
        np.testing.assert_array_equal(bounded[key], full[key])
    assert prov["wanted_keys"] is not None
    assert len(prov["wanted_keys"]) == 4


def test_deterministic_sample_on_counts_matches_arrays(tmp_path: Path) -> None:
    run = _build_capture(tmp_path)
    cap = json.loads((run / "capture-result.json").read_text())
    stacked, _ = collect_expert_activations(run, cap)
    counts, _ = count_expert_activations(run)
    from_arr = deterministic_sample(stacked, le_per_band=1, min_tokens=1)
    from_n = deterministic_sample(counts, le_per_band=1, min_tokens=1)
    assert [(o.layer, o.expert, o.component, o.n_routed) for o in from_arr] == [
        (o.layer, o.expert, o.component, o.n_routed) for o in from_n
    ]


def test_row_cap_is_seeded_and_order_preserving() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 4), dtype=np.float32)
    a, capped_a = subsample_expert_rows(
        X, max_rows=5, seed=SAMPLE_SEED, layer=3, expert=9
    )
    b, capped_b = subsample_expert_rows(
        X, max_rows=5, seed=SAMPLE_SEED, layer=3, expert=9
    )
    assert capped_a and capped_b
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, 4)
    # Relative order preserved (sorted indices).
    positions = []
    for row in a:
        matches = np.where((X == row).all(axis=1))[0]
        positions.append(int(matches[0]))
    assert positions == sorted(positions)
    uncapped, flag = subsample_expert_rows(
        X, max_rows=2048, seed=SAMPLE_SEED, layer=3, expert=9
    )
    assert flag is False
    assert uncapped is X


if __name__ == "__main__":
    import tempfile

    test_row_count_accepts_arrays_and_ints()
    test_row_cap_is_seeded_and_order_preserving()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_count_matches_collect_and_streaming_matches_in_memory(p)
        test_wanted_keys_rows_are_bit_identical(p)
        test_deterministic_sample_on_counts_matches_arrays(p)
    print("PASS test_doctor6_prescribe_membound")
