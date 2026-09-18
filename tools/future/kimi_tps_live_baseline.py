#!/usr/bin/env python3
"""Measure the already-live local KIMI-derived server with a TPS contract."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "http://127.0.0.1:8013"
DEFAULT_MODEL = "/Users/scammermike/hawking-agents/kimi-vl-a3b-q8"
SCHEMA = "hawking.future.kimi_tps_live_baseline.v1"
CONTEXT_REPEATS = (0, 64, 256)
REPS = 3
MAX_TOKENS = 32


def _request(url: str, body: dict[str, Any], timeout: float = 120.0) -> tuple[dict[str, Any], float, dict[str, str]]:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return json.loads(raw), (time.perf_counter() - started) * 1000.0, headers


def _prompt(repeats: int) -> str:
    return (
        "You are participating in a fixed physical throughput benchmark. "
        + ("Context token. " * repeats)
        + "Reply with one short sentence describing the benchmark."
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def med(key: str) -> float | None:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        return statistics.median(values) if values else None

    prompt_n = med("prompt_n")
    predicted_n = med("predicted_n")
    prompt_ms = med("prompt_ms")
    predicted_ms = med("predicted_ms")
    wall_ms = med("wall_ms")
    return {
        "rep_count": len(rows),
        "prompt_tokens_median": prompt_n,
        "completion_tokens_median": predicted_n,
        "prefill_ms_median": prompt_ms,
        "decode_ms_median": predicted_ms,
        "wall_ms_median": wall_ms,
        "prefill_tps": (prompt_n * 1000.0 / prompt_ms) if prompt_n and prompt_ms else None,
        "decode_tps": (predicted_n * 1000.0 / predicted_ms) if predicted_n and predicted_ms else None,
        "end_to_end_tps": (predicted_n * 1000.0 / wall_ms) if predicted_n and wall_ms else None,
    }


def run(base_url: str, model: str, reps: int, max_tokens: int) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    started = time.time()
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            models_body = json.loads(response.read())
            server_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "status": "BASELINE_INCOMPLETE",
            "server": {"url": base_url, "error": f"{type(exc).__name__}: {exc}"},
            "claim_boundary": "No physical measurement was accepted because the live server contract could not be read.",
        }

    # Warm-up is explicit and excluded from all reported statistics.
    warmup_body = {"model": model, "messages": [{"role": "user", "content": "Warm up the fixed benchmark path."}], "max_tokens": 4, "temperature": 0.0, "stream": False}
    try:
        _request(f"{base_url}/chat/completions", warmup_body)
    except Exception as exc:
        return {
            "schema": SCHEMA,
            "status": "BASELINE_INCOMPLETE",
            "server": {"url": base_url, "models": models_body, "headers": server_headers},
            "warmup_error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": "No physical measurement was accepted because the live server warm-up failed.",
        }

    contexts = []
    for repeats in CONTEXT_REPEATS:
        rows = []
        for rep in range(reps):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": _prompt(repeats)}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            }
            try:
                result, wall_ms, headers = _request(f"{base_url}/chat/completions", body)
            except Exception as exc:
                rows.append({"rep": rep, "error": f"{type(exc).__name__}: {exc}"})
                continue
            usage = result.get("usage") or {}
            timings = result.get("timings") or {}
            rows.append({
                "rep": rep,
                "prompt_n": timings.get("prompt_n", usage.get("prompt_tokens")),
                "predicted_n": timings.get("predicted_n", usage.get("completion_tokens")),
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_ms": timings.get("predicted_ms"),
                "wall_ms": wall_ms,
                "server_predicted_tps": timings.get("predicted_per_second"),
                "server_prompt_tps": timings.get("prompt_per_second"),
                "peak_memory_gib": timings.get("peak_memory"),
                "server_headers": headers,
            })
        valid = [row for row in rows if "error" not in row]
        contexts.append({
            "context_repeats": repeats,
            "rows": rows,
            "summary": _summarize(valid) if valid else None,
            "valid_rep_count": len(valid),
        })

    return {
        "schema": SCHEMA,
        "status": "SERVER_BASELINE_COMPLETE_NOT_KIMI_QUALIFIED",
        "source_model": "KIMI_DERIVATIVE_SERVER",
        "raw_kimi_base_equivalence": False,
        "model": model,
        "server": {"url": base_url, "models": models_body, "headers": server_headers},
        "measurement_contract": {
            "endpoint": "/chat/completions",
            "context_repeats": list(CONTEXT_REPEATS),
            "reps_per_context": reps,
            "warmup_excluded": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "concurrency": 1,
            "path": "live mlx_vlm HTTP server timings plus client wall time",
            "prefill_and_decode_separate": True,
        },
        "contexts": contexts,
        "elapsed_s": round(time.time() - started, 3),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "claim_boundary": (
            "Real measurements from the already-live local KIMI-derived q8 server. "
            "These are not raw KIMI_BASE weights, not a reduced Noetic representation, "
            "not a capability-preservation receipt, not a qualified KIMI TPS milestone, "
            "and not promotion or Odyssey admission evidence."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.reps <= 0 or args.max_tokens <= 0:
        raise SystemExit("reps and max-tokens must be positive")
    result = run(args.url, args.model, args.reps, args.max_tokens)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["script"] = str(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "contexts": [row.get("summary") for row in result.get("contexts", [])]}, indent=2))
    return 0 if result["status"] != "BASELINE_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
