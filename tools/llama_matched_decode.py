#!/usr/bin/env python3
"""Run the fail-closed, same-state Llama/Hawking decode protocol.

The runner deliberately emits *blocked* receipts until all quality and hardware
instrumentation fields have primary evidence.  That is a feature: it prevents
a fast but under-instrumented subprocess run from being promoted as parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "workspace/ops/local/models/tg-active/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
DEFAULT_HAWKING = ROOT / "target/release/hawking"
DEFAULT_ORACLE = ROOT / "target/release/llama-sequential-oracle"
SCHEMA = "hawking.tg.decode_measurement.v1"
RUNNER_SCHEMA = "hawking.tg.llama_matched_runner.v2"
WARMUP = 64
MEASURED = 512
PHRASE = " benchmark"

RESIDENT_ENV = {
    "HAWKING_LLAMA_RESIDENT_B9430": "1",
    "HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN": "1",
    "HAWKING_LLAMA_RESIDENT_SERIAL_ENCODER": "1",
    "HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_KV": "1",
    "HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_KV_ROPE": "1",
    "HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_QKV_ROPE": "1",
    "HAWKING_LLAMA_RESIDENT_FUSED_RESIDUAL_NORM": "1",
    # The Q4 gate/up pair is an explicit experimental grammar.  It is not
    # admitted to the strict source harness: a current 2K oracle A/B found
    # immediate greedy-token divergence, so any future promotion must first
    # supply its own exact-current-context receipt.
    "HAWKING_LLAMA_MATCHED_WARMUP_TOKENS": str(WARMUP),
}


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"expected JSON stdout, got: {result.stdout[:400]!r}") from error


def hawking_tokenize(hawking: Path, model: Path, prompt: Path) -> dict[str, Any]:
    return json_stdout(run([
        str(hawking), "tokenize", "--weights", str(model), "--prompt-file", str(prompt), "--json",
    ]))


def oracle_tokenize(oracle: Path, model: Path, prompt: Path) -> dict[str, Any]:
    return json_stdout(run([
        str(oracle), "--model", str(model), "--prompt-file", str(prompt), "--tokenize-only",
    ]))


def create_exact_prefix(hawking: Path, model: Path, path: Path, context: int) -> dict[str, Any]:
    """Build a reproducible prefix whose KV length at measurement start is context."""
    wanted_prompt_tokens = context - WARMUP
    if wanted_prompt_tokens < 1:
        raise ValueError(f"context {context} is smaller than warmup {WARMUP}")
    # Llama 3 encodes this leading-space word as a single token. Verify the
    # assumption against the actual Hawking tokenizer before relying on it.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PHRASE, encoding="utf-8")
    one = hawking_tokenize(hawking, model, path)["count"]
    path.write_text(PHRASE * 2, encoding="utf-8")
    two = hawking_tokenize(hawking, model, path)["count"]
    if two - one != 1:
        raise RuntimeError("benchmark phrase is not token-stable; choose another phrase")
    # one includes the model-declared BOS. N repeats therefore create N + 1 ids.
    repetitions = wanted_prompt_tokens - 1
    path.write_text(PHRASE * repetitions, encoding="utf-8")
    tokenized = hawking_tokenize(hawking, model, path)
    if tokenized["count"] != wanted_prompt_tokens:
        raise RuntimeError(
            f"prefix construction produced {tokenized['count']} tokens, expected {wanted_prompt_tokens}"
        )
    return tokenized


def percentile(samples: list[float], point: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * point)
    return ordered[index]


def parse_hawking(stderr: str) -> tuple[dict[str, Any], list[int]]:
    records = re.findall(r"^\[stats-json\] (.+)$", stderr, flags=re.MULTILINE)
    if len(records) != 1:
        raise RuntimeError(f"expected one Hawking stats JSON record, got {len(records)}")
    ids = [int(value) for value in re.findall(r"^\[token\] id=(\d+)", stderr, flags=re.MULTILINE)]
    return json.loads(records[0]), ids


def hawking_trial(args: argparse.Namespace, prompt: Path, context: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(RESIDENT_ENV)
    command = [
        str(args.hawking), "--profile", "exact", "generate", "--weights", str(args.model),
        "--prompt-file", str(prompt), "--max-new-tokens", str(WARMUP + MEASURED),
        "--max-seq-len", str(context + MEASURED), "--temperature", "0", "--top-k", "0",
        "--top-p", "1", "--seed", "42", "--trace-tokens", "--trace-dispatch",
    ]
    stats, ids = parse_hawking(run(command, env=env).stderr)
    if len(ids) != WARMUP + MEASURED:
        raise RuntimeError(f"Hawking emitted {len(ids)} tokens, expected {WARMUP + MEASURED}")
    samples = stats.get("decode_token_ms")
    if stats.get("completion_tokens") != MEASURED or not isinstance(samples, list) or len(samples) != MEASURED:
        raise RuntimeError("Hawking did not expose exactly the requested measured suffix")
    return {"stats": stats, "ids": ids[WARMUP:], "samples": samples}


def llama_trial(args: argparse.Namespace, prompt: Path, context: int) -> dict[str, Any]:
    result = json_stdout(run([
        str(args.oracle), "--model", str(args.model), "--prompt-file", str(prompt), "--gpu-layers", "all",
        "--ctx-size", str(context + MEASURED), "--measure-warmup", str(WARMUP),
        "--measure-tokens", str(MEASURED),
    ]))
    if len(result.get("generated_token_ids", [])) != MEASURED or len(result.get("decode_token_ms", [])) != MEASURED:
        raise RuntimeError("llama.cpp did not expose exactly the requested measured suffix")
    return {"stats": result, "ids": result["generated_token_ids"], "samples": result["decode_token_ms"]}


def receipt(
    implementation: str,
    args: argparse.Namespace,
    prompt: Path,
    context: int,
    prompt_ids: list[int],
    bos: int,
    eos: int,
    trials: list[dict[str, Any]],
    output_exact: bool,
) -> dict[str, Any]:
    samples = [float(x) for trial in trials for x in trial["samples"]]
    total_ms = sum(samples)
    last = trials[-1]["stats"]
    # All uninstrumented fields are intentionally null.  llama_tps_contract
    # fails closed on these receipts; do not flip them to true/zero without a
    # current-context primary oracle and an engine counter.
    return {
        "schema": SCHEMA,
        "implementation": implementation,
        "identity": {
            "source_sha256": args.source_sha256,
            "source_revision": args.source_sha256,
            "quantization": args.quantization,
            # The tokenizer is embedded in this exact GGUF, so the source hash
            # binds tokenizer identity without pretending a prompt hash is a
            # global tokenizer digest.
            "tokenizer_sha256": args.source_sha256,
        },
        "protocol": {
            "prompt_bytes_utf8_hex": prompt.read_bytes().hex(),
            "prompt_token_ids": prompt_ids,
            "bos_token_id": bos,
            "eos_token_id": eos,
            "context_tokens": context,
            "context_definition": "KV length immediately before the first measured sample",
            "batch_size": 1,
            "greedy": True,
            "warmup_tokens": WARMUP,
            "generated_tokens": MEASURED,
            "runs": args.trials,
            "power_state": args.power_state,
            "k0": "exact-required-current-context-proof-pending",
        },
        "correctness": {
            "exact_token_ids": True,
            "embeddings": False,
            "layer_checkpoints": False,
            "final_logits_topk": False,
            "greedy_output": output_exact,
            "incremental_replay": False,
            "gpu_device": args.device,
            "cpu_reference_fallback_count": (
                last.get("decode_cpu_reference_fallback_total")
                if implementation == "hawking" else None
            ),
        },
        "metrics": {
            "decode_tps": (len(samples) / (total_ms / 1000.0)) if total_ms else 0.0,
            "decode_p50_ms": percentile(samples, 0.50),
            "decode_p95_ms": percentile(samples, 0.95),
            "decode_p99_ms": percentile(samples, 0.99),
            "ttft_ms": None,
            "prefill_tps": None,
            "gpu_dispatches_per_token": (
                last.get("decode_metal_dispatches_total", 0) / MEASURED
                if implementation == "hawking" else None
            ),
            "bytes_per_token": None,
            "ops_per_token": None,
            "peak_memory_bytes": None,
        },
        "measurement": {
            "runner_schema": RUNNER_SCHEMA,
            "per_token_samples": samples,
            "trial_last_stats": last,
            "instrumentation_status": "incomplete_noncertifying",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hawking", type=Path, default=DEFAULT_HAWKING)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "workspace/campaign/evidence/runtime/tg/matched")
    parser.add_argument("--contexts", default="2048,8192,32768")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--power-state", default="uncontrolled")
    parser.add_argument("--device", default="Apple M3 Ultra")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trials < 5:
        raise SystemExit("--trials must be at least 5")
    args.model = args.model.resolve()
    args.hawking = args.hawking.resolve()
    args.oracle = args.oracle.resolve()
    args.source_sha256 = sha256(args.model)
    contexts = [int(value) for value in args.contexts.split(",")]
    if any(value < 2048 for value in contexts):
        raise SystemExit("all contexts must be at least 2048")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for context in contexts:
        prompt = args.out_dir / f"llama_q4km_ctx{context}_prefix.txt"
        hawking_tokens = create_exact_prefix(args.hawking, args.model, prompt, context)
        llama_tokens = oracle_tokenize(args.oracle, args.model, prompt)
        if hawking_tokens["ids"] != llama_tokens["prompt_token_ids"]:
            raise SystemExit(f"tokenizer mismatch at context {context}")
        if args.dry_run:
            print(json.dumps({"context": context, "prompt_tokens": hawking_tokens["count"], "status": "protocol-ready"}))
            continue
        hawking_trials, llama_trials = [], []
        for index in range(args.trials):
            hawking = hawking_trial(args, prompt, context)
            llama = llama_trial(args, prompt, context)
            if hawking["ids"] != llama["ids"]:
                raise SystemExit(f"greedy token mismatch at context {context}, trial {index + 1}")
            hawking_trials.append(hawking)
            llama_trials.append(llama)
        common = (hawking_tokens["ids"], llama_tokens["bos_token_id"], llama_tokens["eos_token_id"])
        (args.out_dir / f"hawking_ctx{context}.json").write_text(json.dumps(
            receipt("hawking", args, prompt, context, *common, hawking_trials, True), indent=2) + "\n"
        )
        (args.out_dir / f"llama_cpp_ctx{context}.json").write_text(json.dumps(
            receipt("llama.cpp", args, prompt, context, *common, llama_trials, True), indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
