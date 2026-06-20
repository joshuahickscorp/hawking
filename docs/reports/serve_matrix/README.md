# Serving Matrix Reports

This directory is for Dismantle/Hawking long-context serving benchmarks.

Use the harness in plan-only mode first:

```bash
python tools/bench/serve_concurrency_matrix.py \
  --plan-only \
  --engine dismantle \
  --workload shared_agent \
  --concurrency 1,2,4 \
  --prompt-tokens 8192,32768 \
  --decode-tokens 128 \
  --out docs/reports/serve_matrix/dismantle_plan.jsonl \
  --report docs/reports/serve_matrix/dismantle_plan.md
```

Measured runs require either an existing OpenAI-compatible server:

```bash
python tools/bench/serve_concurrency_matrix.py \
  --base-url http://127.0.0.1:8080/v1 \
  --stream \
  --workload shared_agent \
  --concurrency 1,2,4 \
  --prompt-tokens 8192 \
  --out docs/reports/serve_matrix/dismantle_current.jsonl \
  --report docs/reports/serve_matrix/dismantle_current.md
```

or an explicit launch command / engine config. Do not compare reports without
the hardware tier, model, workload, prompt length, decode length, and stream
mode recorded.

Required headline metrics:

- TTFT P50/P95/P99
- aggregate decoded tokens/sec
- per-user decoded tokens/sec
- error/cancellation rate
- prefix/state reuse when the engine exposes it
- resident KV/state bytes when the engine exposes it
