"""Tests for the fail-closed 2K/8K/32K Llama TPS scorecard."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import llama_tps_contract as contract  # noqa: E402


def receipt(implementation: str, context: int, tps: float, p99: float) -> dict:
    return {
        "schema": contract.MEASUREMENT_SCHEMA,
        "implementation": implementation,
        "identity": {
            "source_sha256": "a" * 64,
            "source_revision": "revision",
            "quantization": "Q4_K_M",
            "tokenizer_sha256": "b" * 64,
        },
        "protocol": {
            "prompt_bytes_utf8_hex": "68656c6c6f",
            "prompt_token_ids": [128000, 15339],
            "bos_token_id": 128000,
            "eos_token_id": 128009,
            "context_tokens": context,
            "context_definition": "KV length immediately before the first measured sample",
            "batch_size": 1,
            "greedy": True,
            "warmup_tokens": 64,
            "generated_tokens": 512,
            "runs": 5,
            "power_state": "ac-performance",
            "k0": "exact",
        },
        "correctness": {
            "exact_token_ids": True,
            "embeddings": True,
            "layer_checkpoints": True,
            "final_logits_topk": True,
            "greedy_output": True,
            "incremental_replay": True,
            "gpu_device": "Apple Test GPU",
            "cpu_reference_fallback_count": 0,
        },
        "metrics": {
            "decode_tps": tps,
            "decode_p50_ms": 5.0,
            "decode_p95_ms": 8.0,
            "decode_p99_ms": p99,
            "ttft_ms": 10.0,
            "prefill_tps": 500.0,
            "gpu_dispatches_per_token": 4.0,
            "bytes_per_token": 100.0,
            "ops_per_token": 100.0,
            "peak_memory_bytes": 1000,
        },
    }


def write(path: Path, data: dict) -> Path:
    path.write_text(__import__("json").dumps(data))
    return path


def test_unmatched_source_is_blocked_even_when_tps_is_high(tmp_path: Path) -> None:
    hawking = receipt("hawking", 8192, 200.0, 4.0)
    llama = receipt("llama.cpp", 8192, 100.0, 10.0)
    hawking["identity"]["source_sha256"] = "c" * 64
    result = contract.score([write(tmp_path / "h.json", hawking)], [write(tmp_path / "l.json", llama)])
    assert result["status"] == "BLOCKED"
    assert result["pairs"][0]["differences"][0]["field"] == "identity.source_sha256"


def test_8k_and_32k_receipts_can_qualify_for_ship(tmp_path: Path) -> None:
    pairs = []
    for context in (8192, 32768):
        pairs.append((
            write(tmp_path / f"h-{context}.json", receipt("hawking", context, 130.0, 8.0)),
            write(tmp_path / f"l-{context}.json", receipt("llama.cpp", context, 100.0, 10.0)),
        ))
    result = contract.score([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    assert result["status"] == "SHIP"
    assert result["promotion"]["counts_as_ship"] is True


def test_single_8k_surpass_cannot_be_called_ship(tmp_path: Path) -> None:
    result = contract.score(
        [write(tmp_path / "h.json", receipt("hawking", 8192, 130.0, 9.0))],
        [write(tmp_path / "l.json", receipt("llama.cpp", 8192, 100.0, 10.0))],
    )
    assert result["status"] == "SURPASS"


def test_2k_is_a_valid_diagnostic_context_but_cannot_ship_alone(tmp_path: Path) -> None:
    result = contract.score(
        [write(tmp_path / "h.json", receipt("hawking", 2048, 130.0, 9.0))],
        [write(tmp_path / "l.json", receipt("llama.cpp", 2048, 100.0, 10.0))],
    )
    assert result["status"] == "SURPASS"


def test_cpu_fallback_or_missing_dispatch_is_blocked(tmp_path: Path) -> None:
    hawking = receipt("hawking", 8192, 150.0, 7.0)
    hawking["correctness"]["cpu_reference_fallback_count"] = 1
    hawking["metrics"]["gpu_dispatches_per_token"] = 0
    result = contract.score(
        [write(tmp_path / "h.json", hawking)],
        [write(tmp_path / "l.json", receipt("llama.cpp", 8192, 100.0, 10.0))],
    )
    assert result["status"] == "BLOCKED"
    assert any("fallback" in item for item in result["pairs"][0]["hawking_problems"])
