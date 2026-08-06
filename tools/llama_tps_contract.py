#!/usr/bin/env python3
"""Fail-closed matched Llama decode parity / TPS scorecard.

This validator never runs an inference engine.  It compares signed-style JSON
measurement receipts produced by a runner and rejects timing claims when their
model source, prompt protocol, correctness evidence, or real GPU execution do
not match.  A result below `PARITY` is not a throughput result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "hawking.tg.llama_tps_contract.v1"
MEASUREMENT_SCHEMA = "hawking.tg.decode_measurement.v1"
MIN_CONTEXT_TOKENS = 2048
CONTEXTS_FOR_SHIP = {8192, 32768}
IDENTITY_FIELDS = ("source_sha256", "source_revision", "quantization", "tokenizer_sha256")
PROTOCOL_FIELDS = (
    "prompt_bytes_utf8_hex",
    "prompt_token_ids",
    "bos_token_id",
    "eos_token_id",
    "context_tokens",
    "context_definition",
    "batch_size",
    "greedy",
    "warmup_tokens",
    "generated_tokens",
    "runs",
    "power_state",
    "k0",
)
REQUIRED_CORRECTNESS = (
    "exact_token_ids",
    "embeddings",
    "layer_checkpoints",
    "final_logits_topk",
    "greedy_output",
    "incremental_replay",
)
REQUIRED_METRICS = (
    "decode_tps",
    "decode_p50_ms",
    "decode_p95_ms",
    "decode_p99_ms",
    "ttft_ms",
    "prefill_tps",
    "gpu_dispatches_per_token",
    "bytes_per_token",
    "ops_per_token",
    "peak_memory_bytes",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def nested(value: dict[str, Any], name: str) -> dict[str, Any]:
    child = value.get(name)
    return child if isinstance(child, dict) else {}


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_receipt(receipt: dict[str, Any], implementation: str) -> list[str]:
    problems: list[str] = []
    if receipt.get("schema") != MEASUREMENT_SCHEMA:
        problems.append(f"schema must be {MEASUREMENT_SCHEMA}")
    if receipt.get("implementation") != implementation:
        problems.append(f"implementation must be {implementation}")
    identity = nested(receipt, "identity")
    protocol = nested(receipt, "protocol")
    correctness = nested(receipt, "correctness")
    metrics = nested(receipt, "metrics")
    for field in IDENTITY_FIELDS:
        if identity.get(field) in (None, ""):
            problems.append(f"identity.{field} is required")
    for field in PROTOCOL_FIELDS:
        if protocol.get(field) in (None, ""):
            problems.append(f"protocol.{field} is required")
    if protocol.get("batch_size") != 1:
        problems.append("protocol.batch_size must be 1")
    if protocol.get("greedy") is not True:
        problems.append("protocol.greedy must be true")
    if not number(protocol.get("context_tokens")) or protocol["context_tokens"] < MIN_CONTEXT_TOKENS:
        problems.append(f"protocol.context_tokens must be at least {MIN_CONTEXT_TOKENS}")
    if not number(protocol.get("warmup_tokens")) or protocol["warmup_tokens"] < 64:
        problems.append("protocol.warmup_tokens must be at least 64")
    if not number(protocol.get("generated_tokens")) or protocol["generated_tokens"] < 512:
        problems.append("protocol.generated_tokens must be at least 512")
    if not number(protocol.get("runs")) or protocol["runs"] < 5:
        problems.append("protocol.runs must be at least 5")
    for field in REQUIRED_CORRECTNESS:
        if correctness.get(field) is not True:
            problems.append(f"correctness.{field} must be true")
    if not isinstance(correctness.get("gpu_device"), str) or not correctness["gpu_device"].strip():
        problems.append("correctness.gpu_device must identify a real device")
    if not number(correctness.get("cpu_reference_fallback_count")) or correctness[
        "cpu_reference_fallback_count"
    ] != 0:
        problems.append("correctness.cpu_reference_fallback_count must be 0")
    for field in REQUIRED_METRICS:
        if not number(metrics.get(field)) or metrics[field] < 0:
            problems.append(f"metrics.{field} must be a non-negative number")
    if number(metrics.get("gpu_dispatches_per_token")) and metrics["gpu_dispatches_per_token"] <= 0:
        problems.append("metrics.gpu_dispatches_per_token must be > 0")
    if number(metrics.get("decode_tps")) and metrics["decode_tps"] <= 0:
        problems.append("metrics.decode_tps must be > 0")
    return problems


def compare_pair(hawking: dict[str, Any], llama: dict[str, Any]) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    h_identity, l_identity = nested(hawking, "identity"), nested(llama, "identity")
    h_protocol, l_protocol = nested(hawking, "protocol"), nested(llama, "protocol")
    for field in IDENTITY_FIELDS:
        if h_identity.get(field) != l_identity.get(field):
            differences.append({"field": f"identity.{field}", "hawking": h_identity.get(field), "llama_cpp": l_identity.get(field)})
    for field in PROTOCOL_FIELDS:
        if h_protocol.get(field) != l_protocol.get(field):
            differences.append({"field": f"protocol.{field}", "hawking": h_protocol.get(field), "llama_cpp": l_protocol.get(field)})
    h_metrics, l_metrics = nested(hawking, "metrics"), nested(llama, "metrics")
    h_tps, l_tps = h_metrics.get("decode_tps", 0.0), l_metrics.get("decode_tps", 0.0)
    h_p99, l_p99 = h_metrics.get("decode_p99_ms", 0.0), l_metrics.get("decode_p99_ms", 0.0)
    ratio = h_tps / l_tps if number(h_tps) and number(l_tps) and l_tps > 0 else 0.0
    latency_ratio = h_p99 / l_p99 if number(h_p99) and number(l_p99) and l_p99 > 0 else float("inf")
    return {
        "context_tokens": h_protocol.get("context_tokens"),
        "matched": not differences,
        "differences": differences,
        "decode_tps_ratio": ratio,
        "p99_ratio": latency_ratio,
        "surpass": ratio >= 1.10 and latency_ratio <= 1.0,
        "ship": ratio >= 1.25 and latency_ratio <= 0.85,
        "dom": ratio >= 1.50 and latency_ratio <= 0.70,
        "moonshot": ratio >= 2.0 and latency_ratio <= 0.70,
    }


def score(hawking_paths: list[Path], llama_paths: list[Path]) -> dict[str, Any]:
    if len(hawking_paths) != len(llama_paths):
        raise ValueError("pass the same number of --hawking and --llama-cpp receipts")
    if not hawking_paths:
        raise ValueError("at least one matched receipt pair is required")

    pairs: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    seen_contexts: set[int] = set()
    for hawking_path, llama_path in zip(hawking_paths, llama_paths):
        hawking = read_json(hawking_path)
        llama = read_json(llama_path)
        h_problems = validate_receipt(hawking, "hawking")
        l_problems = validate_receipt(llama, "llama.cpp")
        pair = compare_pair(hawking, llama)
        context = nested(hawking, "protocol").get("context_tokens")
        if isinstance(context, int):
            if context in seen_contexts:
                h_problems.append(f"duplicate context receipt: {context}")
            seen_contexts.add(context)
        pair.update({"hawking_receipt": str(hawking_path), "llama_cpp_receipt": str(llama_path), "hawking_problems": h_problems, "llama_cpp_problems": l_problems})
        pairs.append(pair)
        if h_problems or l_problems or not pair["matched"]:
            blocked.append(pair)

    if blocked:
        tier = "BLOCKED"
    elif all(pair["moonshot"] for pair in pairs):
        tier = "MOONSHOT"
    elif all(pair["dom"] for pair in pairs):
        tier = "DOM"
    elif CONTEXTS_FOR_SHIP.issubset(seen_contexts) and all(pair["ship"] for pair in pairs):
        tier = "SHIP"
    elif all(pair["surpass"] for pair in pairs):
        tier = "SURPASS"
    else:
        tier = "PARITY"
    return {
        "schema": SCHEMA,
        "status": tier,
        "contexts_seen": sorted(seen_contexts),
        "required_for_ship": sorted(CONTEXTS_FOR_SHIP),
        "pairs": pairs,
        "promotion": {
            "counts_as_llama_parity": tier != "BLOCKED",
            "counts_as_tps_surpass": tier in {"SURPASS", "SHIP", "DOM", "MOONSHOT"},
            "counts_as_ship": tier in {"SHIP", "DOM", "MOONSHOT"},
            "reason": "Each pair must bind identical source, tokenizer, prompt bytes, K0, decode protocol, correctness evidence, and actual GPU execution.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawking", action="append", required=True, type=Path, metavar="RECEIPT")
    parser.add_argument("--llama-cpp", action="append", required=True, type=Path, metavar="RECEIPT")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = score(args.hawking, args.llama_cpp)
    except ValueError as error:
        print(f"llama_tps_contract: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0 if result["status"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
