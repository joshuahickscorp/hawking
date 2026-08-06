"""Real-activation loader + correspondence seal (PENDING gate + measured path)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_cartography as carto  # noqa: E402
from lab.operators import frankenstein_correspondence_loader as loader  # noqa: E402
from lab.receipts import verify  # noqa: E402


def _sealed_body(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not str(k).startswith("_")}


def _write_glm_layer_npz(
    capture_dir: Path,
    layer: int,
    *,
    n_seq: int = 8,
    width: int = 16,
    seed: int = 0,
) -> Path:
    """Minimal teacher-forced layer shard matching production layout."""

    rng = np.random.default_rng(seed + layer)
    samples = rng.standard_normal((n_seq, 3, width)).astype(np.float32)
    l2 = np.linalg.norm(samples.reshape(n_seq, -1), axis=1).astype(np.float32)
    absmax = np.max(np.abs(samples), axis=(1, 2)).astype(np.float32)
    arrays = {
        "block_output/samples": samples,
        "block_output/l2": l2,
        "block_output/absmax": absmax,
        "block_output/mean": samples.mean(axis=1),
        "block_output/var": samples.var(axis=1),
        "block_output/sample_width": np.full((n_seq,), width, dtype=np.int32),
    }
    layers = capture_dir / "layers"
    layers.mkdir(parents=True, exist_ok=True)
    npz_path = layers / f"L{layer:02d}.npz"
    np.savez(npz_path, **arrays)
    # Minimal JSON receipt (unsealed is fine for loader; production is sealed).
    (layers / f"L{layer:02d}.json").write_text(
        json.dumps(
            {
                "schema": "hawking.frankenstein.glm_layer_capture_shard.v1",
                "layer_id": f"L{layer:02d}",
                "array_names": sorted(arrays),
                "npz_bytes": npz_path.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return npz_path


def _write_glm_corpus(capture_dir: Path, example_ids: list[str]) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    sequences = [{"example_id": eid, "prompt_text": f"prompt {eid}"} for eid in example_ids]
    (capture_dir / "FROZEN_CORPUS_L0.json").write_text(
        json.dumps(
            {
                "schema": "hawking.frankenstein.frozen_corpus.v1",
                "level": "L0",
                "n_sequences": len(sequences),
                "sequences": sequences,
                "fabricated": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dsv4f_traces_with_vectors(
    traces_dir: Path,
    example_ids: list[str],
    *,
    layers: list[int],
    dim: int = 12,
    seed: int = 1,
) -> None:
    """DSV4F fullseq-style traces that include host float vectors (future handoff)."""

    traces_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for ei, eid in enumerate(example_ids):
        positions = []
        for pos in range(2):
            layer_rows = []
            for L in layers:
                vec = rng.standard_normal(dim).astype(float).tolist()
                layer_rows.append(
                    {
                        "layer": L,
                        "token_position": pos,
                        "token_id": ei * 10 + pos,
                        "late_hidden": vec,
                        "post_moe": (rng.standard_normal(dim)).astype(float).tolist(),
                        # also keep a sha field to prove we prefer vectors
                        "late_hidden_child_hc_sha256": "a" * 64,
                    }
                )
            positions.append(
                {"position": pos, "token_id": ei * 10 + pos, "layers": layer_rows}
            )
        doc = {
            "schema": "hawking.frankenstein.paired_functional_trace.v1",
            "example_id": eid,
            "membership": "train",
            "prompt_text": f"prompt {eid}",
            "decoded_spans": [],
            "formal_actions": [],
            "tool_events": [],
            "repair_steps": [],
            "sides": {
                "glm": {"present": False, "side": "glm"},
                "dsv4f": {
                    "side": "dsv4f",
                    "present": True,
                    "capture_status": "OK",
                    "layers_run": layers,
                    "positions": positions,
                },
            },
            "fabricated": False,
        }
        (traces_dir / f"{eid}.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )


def _write_dsv4f_hash_only_traces(
    traces_dir: Path, example_ids: list[str], *, layers: list[int]
) -> None:
    """Current production shape: sha256 only, no float vectors."""

    traces_dir.mkdir(parents=True, exist_ok=True)
    for eid in example_ids:
        positions = [
            {
                "position": 0,
                "token_id": 0,
                "layers": [
                    {
                        "layer": L,
                        "token_position": 0,
                        "late_hidden_child_hc_sha256": "b" * 64,
                        "post_moe_sha256": "c" * 64,
                        "router_top6_route_ids": [1, 2, 3, 4, 5, 6],
                    }
                    for L in layers
                ],
            }
        ]
        doc = {
            "schema": "hawking.frankenstein.paired_functional_trace.v1",
            "example_id": eid,
            "membership": "train",
            "prompt_text": f"prompt {eid}",
            "sides": {
                "dsv4f": {
                    "present": True,
                    "capture_status": "OK",
                    "layers_run": layers,
                    "positions": positions,
                },
                "glm": {"present": False},
            },
            "fabricated": False,
        }
        (traces_dir / f"{eid}.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )


def test_parse_layer_range_dash_inclusive_and_colon_half_open() -> None:
    r = loader.parse_layer_range("0-1")
    assert r.indices() == [0, 1]
    r2 = loader.parse_layer_range("0:2")
    assert r2.indices() == [0, 1]
    r3 = loader.parse_layer_range("3")
    assert r3.indices() == [3]


def test_emit_pending_without_activations_still_honest(tmp_path: Path) -> None:
    """Existing gate: no arrays → PENDING, fabricated=False (do not break)."""

    layer = carto.emit_layer_correspondence(
        glm_layers=None,
        dsv4f_layers=None,
        out_path=tmp_path / "LAYER.json",
        write=True,
    )
    verify(_sealed_body(layer), label="layer pending")
    assert layer["status"] == "PENDING_REAL_ACTIVATIONS"
    assert layer["fabricated"] is False
    assert layer["matrix"] is None
    assert layer["executed"] is False

    phase = carto.emit_phase_alignment(
        glm_layers=None,
        dsv4f_layers=None,
        out_path=tmp_path / "PHASE.json",
        write=True,
    )
    verify(_sealed_body(phase), label="phase pending")
    assert phase["status"] == "PENDING_REAL_ACTIVATIONS"
    assert phase["fabricated"] is False


def test_real_arrays_emit_measured_not_pending(tmp_path: Path) -> None:
    """With real float matrices, emit real CKA/CCA/Procrustes (not PENDING)."""

    glm, dsv = carto.synthetic_paired_layers(
        n_glm=4, n_dsv=2, n_samples=32, d_glm=20, d_dsv=16, seed=7
    )
    layer = carto.emit_layer_correspondence(
        glm_layers=glm,
        dsv4f_layers=dsv,
        source="test_real_arrays",
        out_path=tmp_path / "GLM_DSV4F_LAYER_CORRESPONDENCE.json",
        write=True,
    )
    verify(_sealed_body(layer), label="layer measured")
    assert layer["status"] == "OK"
    assert layer["fabricated"] is False
    assert layer["matrix"] is not None
    assert len(layer["matrix"]) == 4
    assert len(layer["matrix"][0]) == 2
    # Real finite numbers in (0, 1] for planted correspondence.
    cka_00 = layer["matrices"]["cka"][0][0]
    assert isinstance(cka_00, float)
    assert 0.0 <= cka_00 <= 1.0 + 1e-6
    assert layer["sample_pair_metrics"]["cka"] == pytest.approx(cka_00, rel=0, abs=1e-9)
    assert "canonical_correlations" in layer["sample_pair_metrics"]["cca"]
    assert "relative_residual_energy" in layer["sample_pair_metrics"]["procrustes"]
    assert layer["claim_boundary"]["correspondence_numbers_measured"] is True
    assert layer["claim_boundary"]["synthetic_only"] is False

    phase = carto.emit_phase_alignment(
        glm_layers=glm,
        dsv4f_layers=dsv,
        source="test_real_arrays",
        out_path=tmp_path / "GLM_DSV4F_PHASE_ALIGNMENT.json",
        write=True,
    )
    verify(_sealed_body(phase), label="phase measured")
    assert phase["status"] == "OK"
    assert phase["fabricated"] is False
    assert phase["pairs"] is not None
    assert phase["monotonic"] is True


def test_loader_fixture_npz_and_vector_traces_run_end_to_end(tmp_path: Path) -> None:
    """Loader reads GLM NPZ + DSV4F float traces → seals measured correspondence."""

    ids = [f"shared_{i:02d}" for i in range(8)]
    glm_dir = tmp_path / "glm_capture"
    _write_glm_corpus(glm_dir, ids)
    for L in (0, 1, 2, 3):
        _write_glm_layer_npz(glm_dir, L, n_seq=len(ids), width=16, seed=10)

    dsv_root = tmp_path / "dsv_L0"
    traces = dsv_root / "traces"
    _write_dsv4f_traces_with_vectors(traces, ids, layers=[0, 1], dim=12, seed=11)
    receipt = dsv_root / "DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "hawking.frankenstein.dsv4f_fullseq_capture.v1",
                "status": "PASS",
                "scope": {"layers_run": [0, 1], "sequences": len(ids), "ladder": "L0"},
                "paired_traces": {
                    "dir": str(traces),
                    "n_traces": len(ids),
                    "rows": [{"example_id": e, "path": str(traces / f"{e}.json")} for e in ids],
                },
                "fabricated": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "cartography"
    result = loader.run_layer_correspondence(
        glm_capture_dir=glm_dir,
        dsv4f_receipts=[receipt],
        glm_layer_range="0-3",
        dsv4f_layer_range="0-1",
        align="intersection",
        out_dir=out,
        write=True,
        source_ok="fixture_paired_captures",
    )
    assert result["status"] == "OK"
    assert result["fabricated"] is False
    assert result["load_report"]["ok"] is True
    assert result["load_report"]["n_sequences"] == 8
    assert result["load_report"]["glm_layers_loaded"] == [0, 1, 2, 3]
    assert result["load_report"]["dsv4f_layers_loaded"] == [0, 1]

    layer_doc = json.loads((out / "GLM_DSV4F_LAYER_CORRESPONDENCE.json").read_text())
    verify(layer_doc, label="fixture layer seal")
    assert layer_doc["status"] == "OK"
    assert layer_doc["fabricated"] is False
    assert layer_doc["matrix"] is not None
    assert np.asarray(layer_doc["matrix"]).shape == (4, 2)
    # All CKA entries finite and in range.
    for row in layer_doc["matrices"]["cka"]:
        for v in row:
            assert np.isfinite(v)
            assert -1e-6 <= v <= 1.0 + 1e-6
    assert layer_doc["sample_pair_metrics"]["cca"]["mean_top_k"] is not None
    assert "procrustes" in layer_doc["matrices"]


def test_loader_hash_only_dsv4f_stays_pending(tmp_path: Path) -> None:
    """Production DSV4F hash-only traces must not invent correspondence numbers."""

    ids = [f"shared_{i:02d}" for i in range(6)]
    glm_dir = tmp_path / "glm_capture"
    _write_glm_corpus(glm_dir, ids)
    for L in (0, 1):
        _write_glm_layer_npz(glm_dir, L, n_seq=len(ids), width=8, seed=2)

    dsv_root = tmp_path / "dsv_hash"
    traces = dsv_root / "traces"
    _write_dsv4f_hash_only_traces(traces, ids, layers=[0, 1])
    receipt = dsv_root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "paired_traces": {"dir": str(traces), "n_traces": len(ids)},
                "scope": {"layers_run": [0, 1]},
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "cartography"
    result = loader.run_layer_correspondence(
        glm_capture_dir=glm_dir,
        dsv4f_receipts=[receipt],
        glm_layer_range="0-1",
        dsv4f_layer_range="0-1",
        align="intersection",
        out_dir=out,
        write=True,
    )
    assert result["status"] == "PENDING_REAL_ACTIVATIONS"
    assert result["fabricated"] is False
    assert result["load_report"]["dsv4f_hash_only"] is True
    assert any("sha256" in b.lower() or "float" in b.lower() for b in result["load_report"]["blockers"])

    layer_doc = json.loads((out / "GLM_DSV4F_LAYER_CORRESPONDENCE.json").read_text())
    verify(layer_doc, label="hash-only pending")
    assert layer_doc["status"] == "PENDING_REAL_ACTIVATIONS"
    assert layer_doc["fabricated"] is False
    assert layer_doc["matrix"] is None
    assert layer_doc["claim_boundary"]["correspondence_numbers_measured"] is False


def test_loader_missing_glm_npz_stays_pending(tmp_path: Path) -> None:
    """JSON receipts without NPZ payloads → PENDING with explicit NPZ blocker."""

    ids = [f"e{i}" for i in range(4)]
    glm_dir = tmp_path / "glm_json_only"
    _write_glm_corpus(glm_dir, ids)
    layers = glm_dir / "layers"
    layers.mkdir(parents=True, exist_ok=True)
    for L in (0, 1):
        (layers / f"L{L:02d}.json").write_text(
            json.dumps(
                {
                    "layer_id": f"L{L:02d}",
                    "npz_bytes": 1_000_000,
                    "array_names": ["block_output/samples"],
                }
            ),
            encoding="utf-8",
        )

    dsv_root = tmp_path / "dsv"
    traces = dsv_root / "traces"
    _write_dsv4f_traces_with_vectors(traces, ids, layers=[0, 1], dim=8)
    receipt = dsv_root / "r.json"
    receipt.write_text(
        json.dumps({"paired_traces": {"dir": str(traces)}, "scope": {"layers_run": [0, 1]}}),
        encoding="utf-8",
    )

    result = loader.run_layer_correspondence(
        glm_capture_dir=glm_dir,
        dsv4f_receipts=[receipt],
        glm_layer_range="0-1",
        dsv4f_layer_range="0-1",
        out_dir=tmp_path / "out",
        write=True,
    )
    assert result["status"] == "PENDING_REAL_ACTIVATIONS"
    assert result["fabricated"] is False
    assert len(result["load_report"]["glm_npz_missing"]) == 2
    assert result["load_report"]["glm_npz_present"] == []
    assert any("NPZ" in b or "npz" in b for b in result["load_report"]["blockers"])


def test_layer_range_limits_what_is_loaded(tmp_path: Path) -> None:
    ids = [f"s{i}" for i in range(6)]
    glm_dir = tmp_path / "glm"
    _write_glm_corpus(glm_dir, ids)
    for L in range(6):
        _write_glm_layer_npz(glm_dir, L, n_seq=len(ids), width=8, seed=L)

    dsv_root = tmp_path / "dsv"
    traces = dsv_root / "traces"
    _write_dsv4f_traces_with_vectors(traces, ids, layers=[0, 1, 2], dim=8)
    receipt = dsv_root / "r.json"
    receipt.write_text(
        json.dumps(
            {
                "paired_traces": {"dir": str(traces)},
                "scope": {"layers_run": [0, 1, 2]},
            }
        ),
        encoding="utf-8",
    )

    paired = loader.load_paired_activations(
        glm_capture_dir=glm_dir,
        dsv4f_receipts=[receipt],
        glm_layer_range="1-2",  # inclusive → layers 1,2
        dsv4f_layer_range="0-1",
        align="intersection",
    )
    assert paired.report.ok is True
    assert paired.glm_layer_indices == [1, 2]
    assert paired.dsv4f_layer_indices == [0, 1]
    assert len(paired.glm_layers) == 2
    assert len(paired.dsv4f_layers) == 2
