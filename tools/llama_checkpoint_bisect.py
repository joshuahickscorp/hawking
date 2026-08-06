#!/usr/bin/env python3
"""Locate the first numerical divergence from llama.cpp's eval callback.

`llama-eval-callback` prints a full tensor dump, which is far too large to
retain in a performance lane.  This tool streams that output and keeps only
the scalar sums for the named Llama surfaces.  Hawking emits matching
per-token scalar sums only when the explicit debug path is enabled; those are
summed across the prompt and compared to llama.cpp's first (batched) eval.

It is a correctness diagnostic, not a benchmark.  A `DIVERGED` receipt never
counts as Llama parity, TG progress, or a TPS result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "hawking.tg.llama_checkpoint_bisect.v1"
SUMMARY_SCHEMA = "hawking.tg.llama_checkpoint_summary.v1"
SEQUENTIAL_ORACLE_SCHEMA = "hawking.tg.llama_sequential_oracle.v1"
HEADER_PREFIX = "common_debug_cb_eval:"
SUM_RE = re.compile(r"^\s*sum\s*=\s*([-+0-9.eE]+)\s*$")
LAYER_RE = re.compile(r"(?:attn_norm|Qcur|Kcur|Vcur|attn_out|ffn_inp|ffn_norm|ffn_gate|ffn_up|ffn_swiglu|ffn_out|l_out)-(\d+)")
STATS_JSON_PREFIX = "[stats-json] "

LAYER_FIELDS = (
    "attn_norm",
    "q_raw",
    "k_raw",
    "v_raw",
    "q_rope",
    "k_rope",
    "attn_out",
    "ffn_input",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_swiglu",
    "ffn_out",
    "layer_out",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_from_header(header: str) -> int | None:
    match = LAYER_RE.search(header)
    return int(match.group(1)) if match else None


def oracle_surface(header: str) -> str | None:
    """Map one eval-callback header to a canonical scalar surface name."""
    label = header.removeprefix(HEADER_PREFIX).split("=", 1)[0].strip()
    if label == "embd" and "GET_ROWS(token_embd.weight" in header:
        return "embedding"
    if label == "result_norm" and "MUL(" in header:
        return "final_norm"
    if label == "result_output" and "MUL_MAT(" in header:
        return "logits"

    layer = layer_from_header(label)
    if layer is None:
        return None
    prefix = f"layer.{layer}."
    if label == f"attn_norm-{layer}" and "MUL(" in header:
        return prefix + "attn_norm"
    if label == f"Qcur-{layer}":
        if "MUL_MAT(" in header and f"blk.{layer}.attn_q.weight" in header:
            return prefix + "q_raw"
        if "ROPE(" in header:
            return prefix + "q_rope"
    if label == f"Kcur-{layer}":
        if "MUL_MAT(" in header and f"blk.{layer}.attn_k.weight" in header:
            return prefix + "k_raw"
        if "ROPE(" in header:
            return prefix + "k_rope"
    if label == f"Vcur-{layer}" and "MUL_MAT(" in header and f"blk.{layer}.attn_v.weight" in header:
        return prefix + "v_raw"
    if label == f"attn_out-{layer}" and "MUL_MAT(" in header:
        return prefix + "attn_out"
    if label == f"ffn_inp-{layer}" and "ADD(" in header:
        return prefix + "ffn_input"
    if label == f"ffn_norm-{layer}" and "MUL(" in header:
        return prefix + "ffn_norm"
    if label == f"ffn_gate-{layer}" and "MUL_MAT(" in header:
        return prefix + "ffn_gate"
    if label == f"ffn_up-{layer}" and "MUL_MAT(" in header:
        return prefix + "ffn_up"
    if label == f"ffn_swiglu-{layer}" and "SWIGLU(" in header:
        return prefix + "ffn_swiglu"
    if label == f"ffn_out-{layer}" and "MUL_MAT(" in header:
        return prefix + "ffn_out"
    if label == f"l_out-{layer}" and "ADD(" in header:
        return prefix + "layer_out"
    return None


def parse_oracle_lines(lines: Iterable[str]) -> tuple[dict[str, float], int | None]:
    """Extract first batched-eval scalar sums without retaining tensor dumps."""
    surfaces: dict[str, float] = {}
    first_batch_size: int | None = None
    eval_index = -1
    pending: str | None = None
    pending_eval = -1
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if line.startswith(HEADER_PREFIX):
            if "GET_ROWS(token_embd.weight" in line:
                eval_index += 1
                batch_match = re.search(r"inp_tokens\{\s*(\d+)", line)
                if batch_match and first_batch_size is None:
                    first_batch_size = int(batch_match.group(1))
            pending = oracle_surface(line)
            pending_eval = eval_index
            continue
        sum_match = SUM_RE.match(line)
        if sum_match and pending is not None:
            if pending_eval == 0:
                surfaces.setdefault(pending, float(sum_match.group(1)))
            pending = None
    return surfaces, first_batch_size


def parse_oracle(command: list[str], timeout_seconds: int) -> tuple[dict[str, float], int | None]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        def streamed_lines() -> Iterable[str]:
            for line in process.stdout:
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError(f"llama-eval-callback exceeded {timeout_seconds}s")
                yield line

        surfaces, batch_size = parse_oracle_lines(streamed_lines())
        status = process.wait(timeout=max(1, timeout_seconds - int(time.monotonic() - started)))
    except Exception:
        process.kill()
        process.wait()
        raise
    if status != 0:
        raise RuntimeError(f"llama-eval-callback exited {status}")
    return surfaces, batch_size


def load_hawking_summary(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    document = json.loads(path.read_text())
    if document.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"unexpected Hawking summary schema: {document.get('schema')!r}")
    prompt_ids = document.get("prompt_token_ids")
    records = document.get("records")
    if not isinstance(prompt_ids, list) or not isinstance(records, list):
        raise ValueError("Hawking summary lacks prompt_token_ids or records")
    prompt_len = len(prompt_ids)
    prompt_records = [record for record in records if record.get("position", -1) < prompt_len]
    if len(prompt_records) != prompt_len:
        raise ValueError(
            f"Hawking summary holds {len(prompt_records)} prompt records, expected {prompt_len}"
        )
    token_ids = [record.get("token_id") for record in prompt_records]
    if token_ids != prompt_ids:
        raise ValueError("Hawking checkpoint token ids do not match its encoded prompt ids")

    aggregates: dict[str, float] = {
        "embedding": sum(float(record["embedding_sum"]) for record in prompt_records),
        "final_norm": sum(float(record["final_norm_sum"]) for record in prompt_records),
        "logits": sum(float(record["logits_sum"]) for record in prompt_records),
    }
    for field in LAYER_FIELDS:
        for layer in range(len(prompt_records[0].get("layers", []))):
            key = f"layer.{layer}.{field}"
            source_key = f"{field}_sum"
            aggregates[key] = sum(
                float(record["layers"][layer][source_key]) for record in prompt_records
            )
    return document, aggregates


def compare_sequential_logits(
    hawking_summary: dict[str, Any], sequential_oracle: dict[str, Any], abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Compare final logits after every exact prompt token.

    The eval callback emits one batched prefill graph, so its aggregate tensor
    checks cannot tell which prompt position first drifts.  The optional API
    oracle decodes the same encoded ids one at a time and makes that boundary
    explicit.  This remains a diagnostic: it does not relax K0 tolerances.
    """
    if sequential_oracle.get("schema") != SEQUENTIAL_ORACLE_SCHEMA:
        return [], None, "unexpected sequential oracle schema"
    prompt_ids = hawking_summary.get("prompt_token_ids")
    oracle_ids = sequential_oracle.get("prompt_token_ids")
    records = hawking_summary.get("records")
    oracle_records = sequential_oracle.get("records")
    if not isinstance(prompt_ids, list) or prompt_ids != oracle_ids:
        return [], None, "sequential oracle prompt token ids do not match Hawking"
    if not isinstance(records, list) or not isinstance(oracle_records, list):
        return [], None, "sequential oracle records are malformed"
    prompt_records = [record for record in records if record.get("position", -1) < len(prompt_ids)]
    if len(prompt_records) != len(prompt_ids) or len(oracle_records) != len(prompt_ids):
        return [], None, "sequential oracle record count does not match the prompt"
    comparisons: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for position, (hawking, oracle) in enumerate(zip(prompt_records, oracle_records)):
        if hawking.get("position") != position or oracle.get("position") != position:
            return [], None, "sequential oracle records are out of position order"
        if hawking.get("token_id") != oracle.get("token_id"):
            return [], None, "sequential oracle token ids do not match Hawking"
        expected = float(oracle["logits_sum"])
        actual = float(hawking["logits_sum"])
        abs_error = abs(actual - expected)
        allowed_error = abs_tolerance + rel_tolerance * abs(expected)
        row = {
            "position": position,
            "token_id": hawking["token_id"],
            "llama_cpp_logits_sum": expected,
            "hawking_logits_sum": actual,
            "abs_error": abs_error,
            "allowed_error": allowed_error,
            "sum_pass": abs_error <= allowed_error,
            "llama_cpp_greedy_token_id": oracle["greedy_token_id"],
            "hawking_greedy_token_id": hawking["greedy_token_id"],
            "greedy_pass": hawking["greedy_token_id"] == oracle["greedy_token_id"],
        }
        comparisons.append(row)
        if (not row["sum_pass"] or not row["greedy_pass"]) and first is None:
            first = row
    return comparisons, first, None


def compare_checkpoint_vector(
    hawking_summary: dict[str, Any],
    sequential_oracle: dict[str, Any],
    hawking_surface: str,
    sequential_checkpoint: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compare one explicit f32 surface without retaining it in the receipt."""
    records = hawking_summary.get("records")
    if not isinstance(records, list) or not records:
        return None, "Hawking summary has no records for vector comparison"
    vector = records[0].get("debug_vector")
    if not isinstance(vector, dict) or vector.get("surface") != hawking_surface:
        return None, f"Hawking did not capture requested vector surface {hawking_surface!r}"
    actual = vector.get("values")
    checkpoint = sequential_oracle.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None, "sequential oracle has no checkpoint capture"
    if checkpoint.get("name") != sequential_checkpoint or not checkpoint.get("captured"):
        return None, f"sequential oracle did not capture checkpoint {sequential_checkpoint!r}"
    if not checkpoint.get("f32"):
        return None, "sequential oracle checkpoint is not f32"
    expected = checkpoint.get("values")
    if not isinstance(actual, list) or not isinstance(expected, list):
        return None, "checkpoint vectors are malformed"
    if len(actual) != len(expected):
        return None, f"checkpoint vector length mismatch: Hawking={len(actual)} llama.cpp={len(expected)}"
    if not actual:
        return None, "checkpoint vectors are empty"
    errors = [abs(float(left) - float(right)) for left, right in zip(actual, expected)]
    worst_index, max_abs_error = max(enumerate(errors), key=lambda item: item[1])
    return {
        "hawking_surface": hawking_surface,
        "llama_cpp_checkpoint": sequential_checkpoint,
        "value_count": len(actual),
        "max_abs_error": max_abs_error,
        "max_abs_error_index": worst_index,
        "hawking_value": float(actual[worst_index]),
        "llama_cpp_value": float(expected[worst_index]),
        "l1_error": sum(errors),
        "rmse": (sum(error * error for error in errors) / len(errors)) ** 0.5,
    }, None


def ordered_surfaces(oracle: dict[str, float]) -> list[str]:
    keys = ["embedding"]
    layers = sorted(
        {int(match.group(1)) for key in oracle for match in [re.match(r"layer\.(\d+)\.", key)] if match}
    )
    for layer in layers:
        keys.extend(f"layer.{layer}.{field}" for field in LAYER_FIELDS)
    keys.extend(["final_norm", "logits"])
    return [key for key in keys if key in oracle]


def compare(
    oracle: dict[str, float], hawking: dict[str, float], abs_tolerance: float, rel_tolerance: float
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    comparisons: list[dict[str, Any]] = []
    missing = [surface for surface in ordered_surfaces(oracle) if surface not in hawking]
    first: dict[str, Any] | None = None
    for surface in ordered_surfaces(oracle):
        if surface not in hawking:
            continue
        expected = oracle[surface]
        actual = hawking[surface]
        abs_error = abs(actual - expected)
        allowed_error = abs_tolerance + rel_tolerance * abs(expected)
        row = {
            "surface": surface,
            "llama_cpp_sum": expected,
            "hawking_sum": actual,
            "abs_error": abs_error,
            "allowed_error": allowed_error,
            "pass": abs_error <= allowed_error,
        }
        comparisons.append(row)
        if not row["pass"] and first is None:
            first = row
    return comparisons, first, missing


def run_hawking(
    binary: Path,
    weights: Path,
    prompt: str,
    max_seq_len: int,
    trace_path: Path,
    timeout_seconds: int,
    force_cpu: bool,
    kq8_authority: bool,
    checkpoint_vector: str | None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["HAWKING_LLAMA_CHECKPOINT_SUMMARY_PATH"] = str(trace_path)
    if force_cpu:
        environment["HAWKING_FORCE_CPU"] = "1"
    if kq8_authority:
        environment["HAWKING_LLAMA_KQ8_AUTHORITY"] = "1"
    if checkpoint_vector is not None:
        environment["HAWKING_LLAMA_CHECKPOINT_VECTOR"] = checkpoint_vector
    command = [
        str(binary),
        "--profile",
        "exact",
        "generate",
        "--weights",
        str(weights),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "1",
        "--max-seq-len",
        str(max_seq_len),
        "--temperature",
        "0",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--seed",
        "42",
        "--trace-tokens",
        "--trace-dispatch",
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
        check=False,
    )


def run_sequential_oracle(
    binary: Path,
    weights: Path,
    prompt: str,
    max_seq_len: int,
    gpu_layers: str,
    timeout_seconds: int,
    checkpoint: str | None,
) -> tuple[dict[str, Any], list[str]]:
    command = [
        str(binary),
        "--model",
        str(weights),
        "--prompt",
        prompt,
        "--ctx-size",
        str(max_seq_len),
        "--gpu-layers",
        gpu_layers,
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", checkpoint])
    run = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    if run.returncode != 0:
        raise RuntimeError("sequential llama.cpp oracle failed:\\n" + (run.stderr[-4000:] or run.stdout[-4000:]))
    document = json.loads(run.stdout)
    if not isinstance(document, dict):
        raise ValueError("sequential llama.cpp oracle did not emit a JSON object")
    return document, command


def hawking_execution_stats(stderr: str) -> dict[str, Any] | None:
    for line in reversed(stderr.splitlines()):
        if line.startswith(STATS_JSON_PREFIX):
            value = json.loads(line.removeprefix(STATS_JSON_PREFIX))
            return value if isinstance(value, dict) else None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--hawking-bin", type=Path, default=Path("target/release/hawking"))
    parser.add_argument("--llama-oracle", default="llama-eval-callback")
    parser.add_argument(
        "--sequential-oracle",
        type=Path,
        help="optional llama.cpp API oracle built from tools/llama_sequential_oracle.cpp",
    )
    parser.add_argument(
        "--checkpoint-vector",
        help="optional Hawking canonical surface to capture elementwise, e.g. layer.17.v_raw",
    )
    parser.add_argument(
        "--sequential-checkpoint",
        help="matching llama.cpp graph tensor name, e.g. Vcur-17; requires --checkpoint-vector",
    )
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--llama-gpu-layers",
        default="all",
        help="value passed to llama.cpp -ngl (use 0 with --force-cpu)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="diagnostic only: disable Hawking Metal and require llama.cpp -ngl 0",
    )
    parser.add_argument(
        "--kq8-authority",
        "--q4k-q8k-authority",
        dest="kq8_authority",
        action="store_true",
        help="diagnostic only: use Hawking's experimental K-quant × Q8_K CPU authority matvec",
    )
    parser.add_argument("--abs-tolerance", type=float, default=0.002)
    parser.add_argument("--rel-tolerance", type=float, default=0.00001)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument(
        "--llama-cpp-build",
        default=None,
        help="exact llama.cpp build/tag used by both reference oracles",
    )
    parser.add_argument(
        "--ggml-build",
        default=None,
        help="exact ggml backend build/tag dynamically linked by the reference oracle",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    binary = args.hawking_bin.resolve()
    if not weights.is_file():
        raise SystemExit(f"weights does not exist: {weights}")
    if not binary.is_file():
        raise SystemExit(f"Hawking binary does not exist: {binary}; build it first")
    oracle_binary = shutil.which(args.llama_oracle)
    if oracle_binary is None:
        raise SystemExit(f"llama oracle not found on PATH: {args.llama_oracle}")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    if args.force_cpu and args.llama_gpu_layers != "0":
        raise SystemExit("--force-cpu requires --llama-gpu-layers 0 for a matched diagnostic")
    if (args.checkpoint_vector is None) != (args.sequential_checkpoint is None):
        raise SystemExit("--checkpoint-vector and --sequential-checkpoint must be supplied together")
    if args.checkpoint_vector is not None and args.sequential_oracle is None:
        raise SystemExit("--checkpoint-vector requires --sequential-oracle")

    with tempfile.TemporaryDirectory(prefix="hawking-llama-bisect-") as directory:
        trace_path = Path(directory) / "hawking-checkpoints.json"
        oracle_command = [
            oracle_binary,
            "-m",
            str(weights),
            "-p",
            args.prompt,
            "-n",
            "1",
            "-ngl",
            args.llama_gpu_layers,
            "--temp",
            "0",
        ]
        oracle_sums, oracle_prompt_len = parse_oracle(oracle_command, args.timeout_seconds)
        hawking_run = run_hawking(
            binary,
            weights,
            args.prompt,
            args.max_seq_len,
            trace_path,
            args.timeout_seconds,
            args.force_cpu,
            args.kq8_authority,
            args.checkpoint_vector,
        )
        if hawking_run.returncode != 0:
            raise RuntimeError(
                "Hawking checkpoint run failed:\n"
                + (hawking_run.stderr[-4000:] or hawking_run.stdout[-4000:])
            )
        summary, hawking_sums = load_hawking_summary(trace_path)

    sequential_comparisons: list[dict[str, Any]] = []
    first_sequential_divergence: dict[str, Any] | None = None
    sequential_error: str | None = None
    sequential_command: list[str] | None = None
    vector_checkpoint: dict[str, Any] | None = None
    vector_checkpoint_error: str | None = None
    if args.sequential_oracle is not None:
        sequential_binary = args.sequential_oracle.resolve()
        if not sequential_binary.is_file():
            raise SystemExit(f"sequential oracle does not exist: {sequential_binary}")
        sequential_document, sequential_command = run_sequential_oracle(
            sequential_binary,
            weights,
            args.prompt,
            args.max_seq_len,
            args.llama_gpu_layers,
            args.timeout_seconds,
            args.sequential_checkpoint,
        )
        sequential_comparisons, first_sequential_divergence, sequential_error = compare_sequential_logits(
            summary, sequential_document, args.abs_tolerance, args.rel_tolerance
        )
        if args.checkpoint_vector is not None:
            vector_checkpoint, vector_checkpoint_error = compare_checkpoint_vector(
                summary,
                sequential_document,
                args.checkpoint_vector,
                args.sequential_checkpoint,
            )

    hawking_prompt_len = len(summary["prompt_token_ids"])
    comparisons, first_divergence, missing_hawking = compare(
        oracle_sums, hawking_sums, args.abs_tolerance, args.rel_tolerance
    )
    missing_oracle = [surface for surface in ordered_surfaces(hawking_sums) if surface not in oracle_sums]
    oracle_complete = oracle_prompt_len == hawking_prompt_len and bool(oracle_sums)
    if not oracle_complete or missing_hawking or missing_oracle:
        status = "INCOMPLETE_ORACLE"
    elif first_divergence is None:
        status = "PARITY"
    else:
        status = "DIVERGED"
    execution = hawking_execution_stats(hawking_run.stderr)
    device_execution_proven = bool(
        execution
        and execution.get("device_id")
        and execution.get("dispatches_per_forward", 0) > 0
        and execution.get("cpu_reference_fallback_count") == 0
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "source": {
            "weights_path": str(weights),
            "sha256": sha256_file(weights),
            "revision": args.source_revision,
        },
        "reference_runtime": {
            "llama_cpp_build": args.llama_cpp_build,
            "ggml_build": args.ggml_build,
        },
        "contract": {
            "prompt": args.prompt,
            "prompt_bytes_utf8_hex": args.prompt.encode("utf-8").hex(),
            "max_seq_len": args.max_seq_len,
            "generation": {"max_new_tokens": 1, "temperature": 0, "top_k": 0, "top_p": 1},
            "oracle_command": oracle_command,
            "sequential_oracle_command": sequential_command,
            "hawking_binary": str(binary),
            "diagnostic_backend": "cpu" if args.force_cpu else "metal-hybrid",
            "kq8_authority": args.kq8_authority,
            "oracle_prompt_tokens": oracle_prompt_len,
            "hawking_prompt_tokens": hawking_prompt_len,
            "hawking_prompt_token_ids": summary["prompt_token_ids"],
        },
        "checkpoint": {
            "abs_tolerance": args.abs_tolerance,
            "rel_tolerance": args.rel_tolerance,
            "oracle_surface_count": len(oracle_sums),
            "hawking_surface_count": len(hawking_sums),
            "missing_from_hawking": missing_hawking,
            "missing_from_oracle": missing_oracle,
            "first_divergence": first_divergence,
            "comparisons": comparisons,
            "sequential_logits": {
                "enabled": args.sequential_oracle is not None,
                "error": sequential_error,
                "first_divergence": first_sequential_divergence,
                "comparisons": sequential_comparisons,
            },
            "vector_checkpoint": {
                "enabled": args.checkpoint_vector is not None,
                "error": vector_checkpoint_error,
                "comparison": vector_checkpoint,
            },
        },
        "hawking_run": {
            "returncode": hawking_run.returncode,
            "execution": execution,
            "stdout_tail": hawking_run.stdout[-1000:],
            "stderr_tail": hawking_run.stderr[-4000:],
        },
        "promotion": {
            "counts_as_llama_parity": status == "PARITY",
            "counts_as_tps_result": False,
            "device_execution_proven": device_execution_proven,
            "reason": "Checkpoint bisection is a correctness diagnostic, not a throughput benchmark. K0 additionally requires real GPU execution with zero CPU fallback.",
        },
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    else:
        sys.stdout.write(encoded)
    return {"PARITY": 0, "DIVERGED": 1, "INCOMPLETE_ORACLE": 2}[status]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"llama_checkpoint_bisect: {error}", file=sys.stderr)
        raise SystemExit(2) from error
