# Serving Workloads

These JSON files describe OpenAI-compatible serving benchmark workloads for
`tools/bench/serve_concurrency_matrix.py`.

Core workloads:

- `shared_agent.json`: repeated long shared prefix with small user deltas.
- `long_prompt_burst.json`: many long prompts to expose prefill pressure.
- `mixed_latency.json`: one long class mixed with short interactive requests.
- `cache_miss_taxonomy.json`: exact, near, and divergent prefix cases.
- `spec_decode_gate.json`: speculative decoding on/off gate shape.

Generate issue-derived workloads from the pain ledger:

```bash
python tools/bench/workloads/from_pain_radar.py
```

Use plan-only mode before any measured run:

```bash
python tools/bench/serve_concurrency_matrix.py \
  --plan-only \
  --workload shared_agent \
  --concurrency 1,2,4 \
  --prompt-tokens 8192 \
  --out docs/reports/serve_matrix/shared_agent_plan.jsonl \
  --report docs/reports/serve_matrix/shared_agent_plan.md
```
