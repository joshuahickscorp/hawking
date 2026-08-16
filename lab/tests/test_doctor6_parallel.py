"""doctor6 organ-parallel prescribe/treat: determinism, seeding, memory gate."""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    ALL_LAYER_RESULT_SCHEMA,
)
from lab.operators.doctor6.capture_gate import check_per_organ_rows
from lab.operators.doctor6.prescribe import (
    PEAK_RSS_BUDGET_BYTES,
    _sens_ranks,
    available_parallelism,
    default_workers,
    deterministic_sample,
    map_in_stable_order,
    prescribe,
    resolve_workers,
)
from lab.operators.doctor6.treat import treat


_VOLATILE = ("recorded_at_unix", "elapsed_seconds", "peak_rss_bytes")


def _scientific(doc: dict) -> str:
    """JSON payload minus wall-clock / RSS fields that vary by run."""
    cleaned = {k: v for k, v in doc.items() if k not in _VOLATILE}
    return json.dumps(cleaned, sort_keys=True, default=str)


def _write_hidden(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(path)


def _write_f32_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


def _layer_row(layer: int, experts: list[int], rel: str, elements: int) -> dict:
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


def _build_capture(
    tmp: Path,
    *,
    hidden: int = 8,
    n_tokens: int = 4,
    extra_experts: bool = True,
) -> Path:
    """Four-band capture. Expert 1 is on every token so it is the unique
    eligible LE pair per band at min_rows=n_tokens."""
    run = tmp / "full-run"
    hidden_dir = run / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    band_layers = (0, 12, 24, 36)
    steps = []
    rng = np.random.default_rng(0xD0C70A)
    for t_i in range(n_tokens):
        layer_rows = []
        for layer in band_layers:
            vec = rng.standard_normal(hidden).astype(np.float32).tolist()
            rel = f"hidden/t{t_i}_L{layer:02d}.f32le"
            _write_hidden(run / rel, vec)
            experts = [1]
            if extra_experts:
                # One-off experts stay below a min_rows=n_tokens floor.
                experts = [1, 2 + (t_i % 3)]
            layer_rows.append(_layer_row(layer, experts, rel, hidden))
        steps.append({"position": t_i, "layers": layer_rows})
    doc = {
        "schema": ALL_LAYER_RESULT_SCHEMA,
        "bounded_storage": {"strategy": "test"},
        "probes": [{"probe_id": "toy", "steps": steps}],
        "claim_boundary": {"test": True},
    }
    (run / "capture-result.json").write_text(json.dumps(doc, separators=(",", ":")))
    return run


def _build_model(tmp: Path, *, hidden: int = 8, intermediate: int = 6) -> Path:
    model = tmp / "model"
    model.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    tensors: dict[str, np.ndarray] = {}
    weight_map: dict[str, str] = {}
    shard = "model-00001.safetensors"
    for layer in (0, 12, 24, 36):
        for expert in (1,):
            for component, shape in (
                ("gate_proj", (intermediate, hidden)),
                ("up_proj", (intermediate, hidden)),
                ("down_proj", (hidden, intermediate)),
            ):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{component}.weight"
                tensors[name] = rng.standard_normal(shape).astype(np.float32) * 0.08
                weight_map[name] = shard
    _write_f32_safetensors(model / shard, tensors)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return model


def _run_prescribe(tmp: Path, *, workers: int, hidden: int = 8) -> dict:
    capture = tmp / "full-run"
    if not (capture / "capture-result.json").is_file():
        _build_capture(tmp, hidden=hidden)
    model = tmp / "model"
    if not (model / "model.safetensors.index.json").is_file():
        _build_model(tmp, hidden=hidden)
    return prescribe(
        model_id="toy",
        model_dir=model,
        capture=capture,
        target_bpw=1.5,
        le_per_band=1,
        min_rows=4,
        device="cpu",
        qat_steps=2,
        qat_lr=1e-3,
        memory_bounded=True,
        max_rows_per_expert=8,
        workers=workers,
        out_path=tmp / f"rx_w{workers}.json",
    )


def test_map_in_stable_order_ignores_completion_order() -> None:
    items = [f"organ-{k}" for k in range(8)]

    def fn(i: int, item: str) -> dict:
        seed = int.from_bytes(hashlib.sha256(item.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        # Sleep is a function of identity, not of index, so workers=8 finish
        # in a different order than workers=1.
        time.sleep(0.002 + 0.003 * float(rng.random()))
        return {"item": item, "index": i, "draw": int(rng.integers(0, 1_000_000))}

    serial = map_in_stable_order(fn, items, workers=1)
    parallel = map_in_stable_order(fn, items, workers=8)
    assert serial == parallel
    assert [row["item"] for row in parallel] == items


def test_per_organ_seed_independent_of_shuffle() -> None:
    """Shuffling the *completion* order must not change per-organ draws."""
    items = [f"L{k}.E{k}.gate_proj" for k in range(6)]
    seen: list[str] = []

    def fn(_i: int, item: str) -> dict:
        seen.append(item)
        seed = int.from_bytes(hashlib.sha256(item.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        time.sleep((hash(item[::-1]) & 7) * 0.001)
        return {"item": item, "draw": int(rng.integers(0, 10_000))}

    by_workers = {
        w: {row["item"]: row["draw"] for row in map_in_stable_order(fn, items, workers=w)}
        for w in (1, 3, 6)
    }
    assert by_workers[1] == by_workers[3] == by_workers[6]


def test_sens_ranks_follow_input_order_not_completion() -> None:
    raw = [0.4, 0.1, 0.4, 0.9]
    ranks = _sens_ranks(raw)
    # argsort of the same array is deterministic; identity is the slot.
    assert ranks.shape == (4,)
    assert ranks[3] == 1.0  # highest sens
    assert ranks[1] == 0.0  # lowest sens


def test_default_workers_is_measured_not_a_ladder() -> None:
    gib = 1024**3
    full = default_workers(
        parallelism=28,
        free_memory_bytes=64 * gib,
        swap_used_bytes=0,
        n_items=100,
        peak_budget_bytes=PEAK_RSS_BUDGET_BYTES,
    )
    assert full == 28
    mem_bound = default_workers(
        parallelism=28,
        free_memory_bytes=2 * gib,
        swap_used_bytes=0,
        n_items=100,
        bytes_per_worker=768 * 1024**2,
        peak_budget_bytes=PEAK_RSS_BUDGET_BYTES,
    )
    assert 1 <= mem_bound < full
    swapped = default_workers(
        parallelism=28,
        free_memory_bytes=64 * gib,
        swap_used_bytes=2 * gib,
        n_items=100,
    )
    assert swapped < full
    tiny = default_workers(
        parallelism=28,
        free_memory_bytes=64 * gib,
        swap_used_bytes=0,
        n_items=3,
    )
    assert tiny == 3
    live = default_workers()
    assert 1 <= live <= available_parallelism()
    assert resolve_workers(8, n_items=12) == 8
    assert resolve_workers(8, n_items=3) == 3
    assert resolve_workers(0, n_items=5, parallelism=4, free_memory_bytes=64 * gib, swap_used_bytes=0) == 4


def test_swap_backoff_still_returns_input_order() -> None:
    state = {"n": 0}

    def swap() -> int:
        state["n"] += 1
        return 0 if state["n"] < 5 else 8 * 1024**3

    def fn(i: int, item: int) -> int:
        time.sleep(0.005)
        return item * 10 + i

    items = list(range(8))
    out = map_in_stable_order(
        fn, items, workers=4, swap_used=swap, swap_backoff_bytes=1
    )
    assert out == [fn(i, items[i]) for i in range(8)]


def test_memory_gate_refuses_empty_capture_even_with_workers(tmp_path: Path) -> None:
    run = tmp_path / "full-run"
    run.mkdir()
    (run / "capture-result.json").write_text(
        json.dumps(
            {
                "schema": ALL_LAYER_RESULT_SCHEMA,
                "probes": [],
                "claim_boundary": {"test": True},
            }
        )
    )
    for workers in (1, 8):
        rx = prescribe(
            model_dir=tmp_path / "no-model",
            capture=run,
            workers=workers,
            memory_bounded=True,
            le_per_band=1,
            min_rows=4,
        )
        assert rx["status"] == "REFUSED", rx
        assert rx["refusal"]["kind"] == "empty_capture"


def test_memory_gate_refuses_starved_organs() -> None:
    gate = check_per_organ_rows(
        {
            "model.layers.0.mlp.experts.1.gate_proj.weight": 3,
            "model.layers.0.mlp.experts.1.up_proj.weight": 8,
        },
        floor=64,
    )
    assert gate.ok is False
    assert gate.n_below_floor >= 1
    assert "starvation" in (gate.reason or "")


def test_deterministic_sample_still_sample_first(tmp_path: Path) -> None:
    run = _build_capture(tmp_path, n_tokens=4)
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        count_expert_activations,
    )

    counts, _ = count_expert_activations(run)
    organs = deterministic_sample(counts, le_per_band=1, min_tokens=4)
    assert len(organs) == 12  # 4 bands × 1 LE × 3 comps
    assert {(o.layer, o.expert) for o in organs} == {
        (0, 1),
        (12, 1),
        (24, 1),
        (36, 1),
    }


def test_prescribe_workers_1_and_8_identical(tmp_path: Path) -> None:
    rx1 = _run_prescribe(tmp_path, workers=1)
    rx8 = _run_prescribe(tmp_path, workers=8)
    assert rx1.get("organs"), rx1
    assert [r["tensor_name"] for r in rx1["organs"]] == [
        r["tensor_name"] for r in rx8["organs"]
    ]
    assert [r["chain"] for r in rx1["organs"]] == [r["chain"] for r in rx8["organs"]]
    assert _scientific(rx1) == _scientific(rx8)


def test_treat_workers_1_and_8_identical(tmp_path: Path) -> None:
    rx = _run_prescribe(tmp_path, workers=1)
    assert rx.get("organs"), rx
    # Tiny random organs fail the coherence screen (status REFUSED). Treat
    # short-circuits on REFUSED; force OK so the per-organ loop actually runs.
    rx_ok = dict(rx)
    rx_ok["status"] = "OK"
    rx_ok.pop("refusal", None)
    rx_path = tmp_path / "rx_ok.json"
    rx_path.write_text(json.dumps(rx_ok, indent=2, sort_keys=True, default=str) + "\n")
    t1 = treat(
        prescription_path=rx_path,
        device="cpu",
        qat_steps=2,
        workers=1,
        out_path=tmp_path / "t1.json",
    )
    t8 = treat(
        prescription_path=rx_path,
        device="cpu",
        qat_steps=2,
        workers=8,
        out_path=tmp_path / "t8.json",
    )
    assert [r["tensor_name"] for r in t1["organs"]] == [
        r["tensor_name"] for r in t8["organs"]
    ]
    assert _scientific(t1) == _scientific(t8)


def test_cli_exposes_workers() -> None:
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "lab.operators.doctor6", "prescribe", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--workers" in proc.stdout
    proc_t = subprocess.run(
        [sys.executable, "-m", "lab.operators.doctor6", "treat", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_t.returncode == 0, proc_t.stderr
    assert "--workers" in proc_t.stdout
