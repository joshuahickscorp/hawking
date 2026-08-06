#!/usr/bin/env python3
"""Capture an aligned, provenance-sealed Llama functional-student dataset.

This is deliberately a *teacher evidence* tool, not a packer and not a
benchmark.  A functional replacement may only be considered after it has
paired hidden-state inputs with the teacher's actual layer output, with a
prompt-level held-out split.  Weight-only fitting is not a capability test.

The Llama debug lane permits one vector surface per process.  Consequently we
replay each prompt twice, independently collect ``ffn_norm`` and ``ffn_out``,
then fail closed unless their prompt token IDs and positions align exactly.
The generated ``.npz`` is small activation evidence; it is never a model
artifact and contains no TPS claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCHEMA = "hawking.tg.llama_functional_student_capture.v1"
CHECKPOINT_SCHEMA = "hawking.tg.llama_checkpoint_summary.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path, expected_surface: str) -> dict[str, Any]:
    if path.suffix == ".bin":
        raw = path.read_bytes()
        if raw[:8] != b"HLRFFN1\x00" or len(raw) < 12:
            raise ValueError(f"{path}: not a resident f32 capture")
        header_len = int.from_bytes(raw[8:12], "little")
        header_end = 12 + header_len
        header = json.loads(raw[12:header_end])
        if header.get("schema") != "hawking.tg.llama_resident_f32_capture.v1":
            raise ValueError(f"{path}: unexpected resident capture schema")
        rows, width = int(header["rows"]), int(header["width"])
        tokens_end = header_end + rows * 4
        plane_bytes = rows * width * 4
        if len(raw) != tokens_end + plane_bytes * 2:
            raise ValueError(f"{path}: resident capture length does not match header")
        if expected_surface == header["input_surface"]:
            start = tokens_end
        elif expected_surface == header["target_surface"]:
            start = tokens_end + plane_bytes
        else:
            raise ValueError(f"{path}: resident capture lacks {expected_surface!r}")
        return {"model_id": header["model_id"], "model_arch": header["model_arch"], "weights_path": header["weights_path"], "prompt_token_ids": np.frombuffer(raw, dtype="<u4", count=rows, offset=header_end).tolist(), "vectors": np.frombuffer(raw, dtype="<f4", count=rows * width, offset=start).reshape(rows, width).copy()}
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        try:
            metadata = json.loads(str(data["metadata"][0]))
            key = "inputs" if expected_surface == metadata["input_surface"] else "targets" if expected_surface == metadata["target_surface"] else None
            if key is None:
                raise ValueError(f"{path}: compact trace lacks {expected_surface!r}")
            return {"model_id": metadata["model_id"], "model_arch": metadata["model_arch"], "weights_path": metadata["weights_path"], "prompt_token_ids": data["prompt_token_ids"].tolist(), "vectors": np.asarray(data[key], dtype=np.float32)}
        finally:
            data.close()
    document = json.loads(path.read_text())
    if document.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: unexpected checkpoint schema")
    records = document.get("records")
    prompt_ids = document.get("prompt_token_ids")
    if not isinstance(records, list) or not isinstance(prompt_ids, list):
        raise ValueError(f"{path}: checkpoint lacks records or prompt_token_ids")
    prompt_records = [row for row in records if row.get("position", -1) < len(prompt_ids)]
    if len(prompt_records) != len(prompt_ids):
        raise ValueError(f"{path}: prompt record count does not match prompt token IDs")
    vectors: list[list[float]] = []
    for position, row in enumerate(prompt_records):
        vector = row.get("debug_vector")
        if not isinstance(vector, dict) or vector.get("surface") != expected_surface:
            candidates = row.get("debug_vectors", [])
            vector = next((item for item in candidates if isinstance(item, dict) and item.get("surface") == expected_surface), None)
        if not isinstance(vector, dict) or vector.get("surface") != expected_surface:
            raise ValueError(f"{path}: position {position} lacks {expected_surface!r}")
        values = vector.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: position {position} has an empty debug vector")
        vectors.append(values)
    return {
        "model_id": document.get("model_id"),
        "model_arch": document.get("model_arch"),
        "weights_path": document.get("weights_path"),
        "prompt_token_ids": prompt_ids,
        "vectors": np.asarray(vectors, dtype=np.float32),
    }


def compact_paired_trace(path: Path, input_surface: str, target_surface: str) -> Path:
    """Replace JSON decimal vectors with a compact, self-describing f32 trace."""
    # A paired JSON trace has both vectors in every record.  Loading it twice
    # used two full decimal documents at peak exactly when long source traces
    # are largest.  Parse once and split its surfaces before creating the two
    # compact f32 planes; the generic loader stays available for distinct
    # input/target files and already-compact binary traces.
    if path.suffix == ".json" and input_surface != target_surface:
        document = json.loads(path.read_text())
        if document.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path}: unexpected checkpoint schema")
        records, prompt_ids = document.get("records"), document.get("prompt_token_ids")
        if not isinstance(records, list) or not isinstance(prompt_ids, list):
            raise ValueError(f"{path}: checkpoint lacks records or prompt token IDs")
        prompt_records = [row for row in records if row.get("position", -1) < len(prompt_ids)]
        if len(prompt_records) != len(prompt_ids):
            raise ValueError(f"{path}: prompt record count does not match prompt token IDs")
        planes: dict[str, list[list[float]]] = {input_surface: [], target_surface: []}
        for position, row in enumerate(prompt_records):
            vectors = row.get("debug_vectors", [])
            if not isinstance(vectors, list):
                vectors = []
            for surface in planes:
                vector = next((item for item in vectors if isinstance(item, dict) and item.get("surface") == surface), None)
                if not isinstance(vector, dict) or not isinstance(vector.get("values"), list) or not vector["values"]:
                    raise ValueError(f"{path}: position {position} lacks non-empty {surface!r}")
                planes[surface].append(vector["values"])
        identity = {"model_id": document.get("model_id"), "model_arch": document.get("model_arch"), "weights_path": document.get("weights_path"), "prompt_token_ids": prompt_ids}
        source = {**identity, "vectors": np.asarray(planes[input_surface], dtype=np.float32)}
        target = {**identity, "vectors": np.asarray(planes[target_surface], dtype=np.float32)}
    else:
        source = load_trace(path, input_surface)
        target = load_trace(path, target_surface)
    for key in ("model_id", "model_arch", "weights_path", "prompt_token_ids"):
        if source[key] != target[key]:
            raise ValueError(f"cannot compact mismatched paired trace {path}: {key}")
    compact = path.with_suffix(".npz")
    metadata = json.dumps({"model_id": source["model_id"], "model_arch": source["model_arch"], "weights_path": source["weights_path"], "input_surface": input_surface, "target_surface": target_surface}, separators=(",", ":"))
    # Activations are entropy-rich f32 values: deflate consumes a CPU core for
    # negligible space reduction.  A raw NPZ stays under the explicit capture
    # budget and keeps the GPU teacher lane from stalling on host compression.
    np.savez(compact, inputs=source["vectors"], targets=target["vectors"], prompt_token_ids=np.asarray(source["prompt_token_ids"], dtype=np.uint32), metadata=np.asarray([metadata]))
    path.unlink()
    return compact


def pair_trace(input_path: Path, target_path: Path, input_surface: str, target_surface: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = load_trace(input_path, input_surface)
    target = load_trace(target_path, target_surface)
    for key in ("model_id", "model_arch", "weights_path", "prompt_token_ids"):
        if source[key] != target[key]:
            raise ValueError(f"trace mismatch for {key}: {input_path.name} vs {target_path.name}")
    if source["vectors"].shape[0] != target["vectors"].shape[0]:
        raise ValueError("trace row counts differ")
    return source["vectors"], target["vectors"], {
        "model_id": source["model_id"], "model_arch": source["model_arch"],
        "weights_path": source["weights_path"], "prompt_token_ids": source["prompt_token_ids"],
    }


def assemble(input_paths: list[Path], target_paths: list[Path], *, input_surface: str, target_surface: str, weights: Path, out: Path, heldout_modulo: int, trace_dir: Path | None = None) -> dict[str, Any]:
    if len(input_paths) != len(target_paths) or not input_paths:
        raise ValueError("need the same non-zero number of input and target traces")
    if heldout_modulo < 2:
        raise ValueError("heldout_modulo must be at least 2")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    prompt_rows: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    for index, (input_path, target_path) in enumerate(zip(input_paths, target_paths, strict=True)):
        x, y, metadata = pair_trace(input_path, target_path, input_surface, target_surface)
        if identity is None:
            identity = metadata
        elif any(identity[key] != metadata[key] for key in ("model_id", "model_arch", "weights_path")):
            raise ValueError("traces originate from different teacher identities")
        xs.append(x)
        ys.append(y)
        prompt_rows.append({
            "index": index,
            "tokens": len(metadata["prompt_token_ids"]),
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(metadata["prompt_token_ids"], separators=(",", ":")).encode()
            ).hexdigest(),
            "split": "heldout" if index % heldout_modulo == 0 else "fit",
            "input_trace_sha256": sha256(input_path),
            "target_trace_sha256": sha256(target_path),
        })
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    if x_all.shape[0] != y_all.shape[0] or x_all.shape[1] == 0 or y_all.shape[1] == 0:
        raise ValueError("invalid paired activation geometry")
    split = np.concatenate([
        np.full(x.shape[0], index % heldout_modulo == 0, dtype=np.bool_)
        for index, x in enumerate(xs)
    ])
    if not np.any(split) or np.all(split):
        raise ValueError("dataset must contain both fit and heldout prompts")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, inputs=x_all, targets=y_all, heldout=split)
    receipt = {
        "schema": SCHEMA,
        "status": "CAPTURED_NOT_CAPABILITY_PROVEN",
        "weights_sha256": sha256(weights),
        "weights_path": str(weights),
        "teacher": identity,
        "surfaces": {"input": input_surface, "target": target_surface},
        "dataset": {"path": str(out), "sha256": sha256(out), "input_shape": list(x_all.shape), "target_shape": list(y_all.shape), "fit_rows": int((~split).sum()), "heldout_rows": int(split.sum()), "trace_dir": str(trace_dir) if trace_dir else None},
        "prompts": prompt_rows,
        "capability_gate": "A student remains ineligible until it reports held-out teacher-output error and end-to-end generated-token quality against this sealed split.",
        "tps_claim": None,
    }
    receipt_path = out.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def chunk_prompt_files(prompts: list[str], root: Path, char_limit: int) -> list[Path]:
    """Pack prompts into independent long-context chunks for efficient capture."""
    if char_limit < 1:
        return []
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    current = ""
    for prompt in prompts:
        addition = prompt if not current else "\n\n" + prompt
        if current and len(current) + len(addition) > char_limit:
            chunks.append(current)
            current = prompt
        else:
            current += addition
    if current:
        chunks.append(current)
    paths: list[Path] = []
    for index, content in enumerate(chunks):
        path = root / f"chunk-{index:05}.txt"
        path.write_text(content)
        paths.append(path)
    return paths


def capture_files(binary: Path, weights: Path, prompt_files: list[Path], surface: str, trace_dir: Path, max_seq_len: int) -> list[Path]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces: list[Path] = []
    for index, prompt_file in enumerate(prompt_files):
        trace = trace_dir / f"checkpoint-{index:05}.json"
        env = os.environ.copy()
        env.pop("HAWKING_LLAMA_CHECKPOINT_SUMMARY_DIR", None)
        env["HAWKING_LLAMA_CHECKPOINT_SUMMARY_PATH"] = str(trace)
        env["HAWKING_LLAMA_CHECKPOINT_VECTOR"] = surface
        # The CLI's matched warmup currently requires at least one completion
        # token.  The assembler filters records to prompt positions, so this
        # one-token tail cannot enter the student evidence.
        command = [str(binary), "--profile", "exact", "generate", "--weights", str(weights), "--prompt-file", str(prompt_file),
                   "--max-new-tokens", "1", "--max-seq-len", str(max_seq_len),
                   "--temperature", "0", "--top-k", "0", "--top-p", "1", "--seed", "42"]
        completed = subprocess.run(command, env=env, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError(f"capture {surface} chunk {index} failed: {completed.stderr[-2000:]}")
        binary_trace = trace.with_suffix(".bin")
        if not trace.is_file() and not binary_trace.is_file():
            raise RuntimeError(f"capture {surface} did not write {trace} or {binary_trace}")
        trace = binary_trace if binary_trace.is_file() else trace
        selected = [item.strip() for item in surface.split(",") if item.strip()]
        traces.append(
            compact_paired_trace(trace, selected[0], selected[1])
            if len(selected) == 2
            else trace
        )
    return traces


def capture_corpus(binary: Path, weights: Path, prompts_file: Path, prompt_count: int, surface: str, trace_dir: Path, max_seq_len: int) -> list[Path]:
    env = os.environ.copy()
    env.pop("HAWKING_LLAMA_CHECKPOINT_SUMMARY_PATH", None)
    env["HAWKING_LLAMA_CHECKPOINT_SUMMARY_DIR"] = str(trace_dir)
    env["HAWKING_LLAMA_CHECKPOINT_VECTOR"] = surface
    # The CLI's matched warmup currently requires at least one completion
    # token.  The assembler filters records to prompt positions, so this
    # one-token tail cannot enter the student evidence.
    command = [str(binary), "--profile", "exact", "generate", "--weights", str(weights), "--prompts-file", str(prompts_file),
               "--max-new-tokens", "1", "--max-seq-len", str(max_seq_len),
               "--temperature", "0", "--top-k", "0", "--top-p", "1", "--seed", "42"]
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"capture {surface} failed: {completed.stderr[-2000:]}")
    traces = [trace_dir / f"checkpoint-{index:05}.json" for index in range(prompt_count)]
    missing = [str(path) for path in traces if not path.is_file()]
    if missing:
        raise RuntimeError(f"capture {surface} did not write expected traces: {missing[:3]}")
    return traces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="activation .npz output")
    parser.add_argument("--input-traces", type=Path, nargs="*", default=[])
    parser.add_argument("--target-traces", type=Path, nargs="*", default=[])
    parser.add_argument("--input-surface", default="layer.0.ffn_norm")
    parser.add_argument("--target-surface", default="layer.0.ffn_out")
    parser.add_argument("--heldout-modulo", type=int, default=5)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--hawking", type=Path, default=Path("target/release/hawking"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--chunk-chars", type=int, default=6000, help="pack local prompts into bounded long-context capture chunks; 0 keeps one prompt per generate")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")
    inputs, targets = list(args.input_traces), list(args.target_traces)
    if args.prompts_file:
        prompts = [line.strip() for line in args.prompts_file.read_text().splitlines() if line.strip()]
        if not prompts:
            parser.error("prompts file has no non-empty prompts")
        if not args.hawking.is_file():
            parser.error(f"hawking binary not found: {args.hawking}")
        # Keep the raw traces next to the sealed dataset.  They are the small,
        # auditable teacher evidence behind its hashes; a temporary directory
        # would make the manifest impossible to replay.
        root = args.out.parent / f"{args.out.stem}.traces"
        root.mkdir(parents=True, exist_ok=True)
        chunk_files = chunk_prompt_files(prompts, root / "chunks", args.chunk_chars)
        paired_surfaces = f"{args.input_surface},{args.target_surface}"
        if chunk_files:
            raw_inputs = capture_files(args.hawking, args.weights, chunk_files, paired_surfaces, root / "paired", args.max_seq_len)
        else:
            raw_inputs = capture_corpus(args.hawking, args.weights, args.prompts_file, len(prompts), paired_surfaces, root / "paired", args.max_seq_len)
        inputs = [
            path if path.suffix in {".npz", ".bin"} else compact_paired_trace(path, args.input_surface, args.target_surface)
            for path in raw_inputs
        ]
        # One paired trace provides both exact surfaces for every record.
        targets = inputs
        receipt = assemble(inputs, targets, input_surface=args.input_surface, target_surface=args.target_surface, weights=args.weights, out=args.out, heldout_modulo=args.heldout_modulo, trace_dir=root)
    else:
        receipt = assemble(inputs, targets, input_surface=args.input_surface, target_surface=args.target_surface, weights=args.weights, out=args.out, heldout_modulo=args.heldout_modulo)
    print(json.dumps({"status": receipt["status"], "dataset": receipt["dataset"], "receipt": str(args.out.with_suffix(".json"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
