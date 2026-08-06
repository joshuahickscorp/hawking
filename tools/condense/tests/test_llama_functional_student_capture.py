from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "llama_functional_student_capture.py"
SPEC = importlib.util.spec_from_file_location("llama_functional_student_capture", MODULE_PATH)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


def trace(path: Path, surface: str, token_ids: list[int], width: int, offset: float = 0.0) -> None:
    path.write_text(json.dumps({
        "schema": capture.CHECKPOINT_SCHEMA, "model_id": "llama-test", "model_arch": "llama",
        "weights_path": "/weights.gguf", "prompt_token_ids": token_ids,
        "records": [{"position": i, "debug_vector": {"surface": surface, "values": [offset + i, offset + i + 0.5] if width == 2 else [offset + i]}} for i in range(len(token_ids))],
    }))


def test_assemble_seals_aligned_prompt_heldout_split(tmp_path: Path) -> None:
    weights = tmp_path / "teacher.gguf"; weights.write_bytes(b"teacher")
    inputs: list[Path] = []; targets: list[Path] = []
    for i in range(3):
        a, b = tmp_path / f"a{i}.json", tmp_path / f"b{i}.json"
        trace(a, "layer.0.ffn_norm", [1, 2 + i], 2, float(i))
        trace(b, "layer.0.ffn_out", [1, 2 + i], 2, float(i + 10))
        inputs.append(a); targets.append(b)
    out = tmp_path / "pairs.npz"
    receipt = capture.assemble(inputs, targets, input_surface="layer.0.ffn_norm", target_surface="layer.0.ffn_out", weights=weights, out=out, heldout_modulo=3)
    data = np.load(out)
    assert data["inputs"].shape == (6, 2)
    assert data["targets"].shape == (6, 2)
    assert data["heldout"].sum() == 2
    assert receipt["status"] == "CAPTURED_NOT_CAPABILITY_PROVEN"
    assert json.loads(out.with_suffix(".json").read_text())["tps_claim"] is None


def test_assemble_rejects_misaligned_tokenization(tmp_path: Path) -> None:
    weights = tmp_path / "teacher.gguf"; weights.write_bytes(b"teacher")
    source, target = tmp_path / "source.json", tmp_path / "target.json"
    trace(source, "layer.0.ffn_norm", [1, 2], 2)
    trace(target, "layer.0.ffn_out", [1, 3], 2)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        capture.assemble([source], [target], input_surface="layer.0.ffn_norm", target_surface="layer.0.ffn_out", weights=weights, out=tmp_path / "pairs.npz", heldout_modulo=2)


def test_load_trace_reads_paired_vector_schema_without_aliasing_accumulator(tmp_path: Path) -> None:
    trace_path = tmp_path / "paired.json"
    trace_path.write_text(json.dumps({
        "schema": capture.CHECKPOINT_SCHEMA, "model_id": "llama-test", "model_arch": "llama",
        "weights_path": "/weights.gguf", "prompt_token_ids": [1, 2],
        "records": [
            {"position": 0, "debug_vector": None, "debug_vectors": [{"surface": "layer.0.ffn_norm", "values": [1.0, 2.0]}, {"surface": "layer.0.ffn_out", "values": [3.0, 4.0]}]},
            {"position": 1, "debug_vector": None, "debug_vectors": [{"surface": "layer.0.ffn_norm", "values": [5.0, 6.0]}, {"surface": "layer.0.ffn_out", "values": [7.0, 8.0]}]},
        ],
    }))
    parsed = capture.load_trace(trace_path, "layer.0.ffn_norm")
    assert parsed["vectors"].shape == (2, 2)
    assert parsed["vectors"].tolist() == [[1.0, 2.0], [5.0, 6.0]]


def test_compact_paired_trace_preserves_vectors_and_removes_decimal_transport(tmp_path: Path) -> None:
    trace_path = tmp_path / "paired.json"
    trace_path.write_text(json.dumps({
        "schema": capture.CHECKPOINT_SCHEMA, "model_id": "llama-test", "model_arch": "llama",
        "weights_path": "/weights.gguf", "prompt_token_ids": [1],
        "records": [{"position": 0, "debug_vector": None, "debug_vectors": [{"surface": "layer.0.ffn_norm", "values": [1.0, 2.0]}, {"surface": "layer.0.ffn_out", "values": [3.0, 4.0]}]}],
    }))
    compact = capture.compact_paired_trace(trace_path, "layer.0.ffn_norm", "layer.0.ffn_out")
    assert compact.is_file() and not trace_path.exists()
    assert capture.load_trace(compact, "layer.0.ffn_norm")["vectors"].tolist() == [[1.0, 2.0]]
    assert capture.load_trace(compact, "layer.0.ffn_out")["vectors"].tolist() == [[3.0, 4.0]]


def test_load_trace_reads_resident_binary_capture(tmp_path: Path) -> None:
    path = tmp_path / "resident.bin"
    header = json.dumps({"schema": "hawking.tg.llama_resident_f32_capture.v1", "model_id": "llama-test", "model_arch": "llama", "weights_path": "/weights.gguf", "rows": 2, "width": 2, "input_surface": "layer.31.ffn_norm", "target_surface": "layer.31.ffn_out"}, separators=(",", ":")).encode()
    tokens = np.asarray([7, 9], dtype="<u4").tobytes()
    inputs = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="<f4").tobytes()
    targets = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype="<f4").tobytes()
    path.write_bytes(b"HLRFFN1\0" + len(header).to_bytes(4, "little") + header + tokens + inputs + targets)
    assert capture.load_trace(path, "layer.31.ffn_norm")["vectors"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert capture.load_trace(path, "layer.31.ffn_out")["prompt_token_ids"] == [7, 9]
