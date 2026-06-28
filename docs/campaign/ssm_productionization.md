# SSM Productionization Plan

This is the current top product frontier. The validated finding is not "one RWKV
benchmark was fast"; it is that the SSM/recurrent path removes the transformer KV
wall in long-context decode. Production work should turn that into a reliable model
selection path with quality gates, not another kernel hunt.

## Current evidence

- Qwen2.5-3B Q4_K_M: ~40 tps short -> 18.8 tps at ~2.5k -> 8.6 tps at ~8k.
- RWKV-7 0.4B SFT Q4_K_M: 118.6 / 110.6 / 119.4 tps across short / ~2.5k / ~8k.
- mamba2-370M: short and ~2.5k were observed around 11.7 tps; the later ~8k corroboration process exited but no saved
  stdout/log was found, so final numbers remain unverified.
- Interpretation: RWKV-7 is both smaller and flat across context. The durable product claim is long-context throughput
  and bounded memory state, not transformer decode micro-optimization.

## Ship criteria

Before SSM is recommended as a product default for long-context work:

1. **Repeatable speed matrix**
   - Short, ~2.5k, ~8k, and optionally ~16k prompts.
   - At least 3 warm trials each.
   - Run with active-agent contamination clearly labeled.
   - Record median `dec_tps`, prompt size, token count, model path, and command.

2. **Quality matrix**
   - Coding, JSON, math, multilingual, instruction-following, and summarization prompts.
   - Compare SSM output to a stronger teacher or accepted rubric; do not use speed alone as acceptance.
   - Include long-context retrieval-style prompts where the answer depends on information near the beginning of the context.

3. **Serving behavior**
   - Confirm `hawking serve` can route or expose RWKV/SSM models without Qwen-specific assumptions.
   - Confirm streaming token cadence and cancellation behavior.
   - Confirm memory state resets between independent requests and is retained only when explicitly intended.
   - Use `tools/ci/ssm_serve_smoke.sh` as the first gate: health, models, native SSE generation, metrics,
     and clean server shutdown.

4. **Model-selection policy**
   - Short, quality-sensitive user prompts: keep Qwen/default transformer path unless quality gates say otherwise.
   - Long-context throughput-sensitive prompts: prefer SSM path when quality gate passes.
   - Hybrid option: use SSM for fast long-context drafting/summarization, then Qwen for final high-precision answer.

5. **Failure fallback**
   - If SSM quality fails a prompt class, route that class back to Qwen.
   - If SSM model load or tokenizer path fails, fall back to transformer path with a logged reason.
   - Do not silently change output semantics based only on speed.

## Immediate next checks

Current top check:

```bash
# Reproduce/characterize the known RWKV serve admission gap.
tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf
```

Observed result in `reports/serve-smoke/20260622T022233Z/`: server load and basic HTTP endpoints pass, but native
`/v1/hawking/generate` times out after 180s. Metrics after the request show `queued_requests=1`, `requests_admitted=0`,
and `tokens_generated=0`. This means the immediate productionization bug is serve admission/decode for SSM, not raw
RWKV CLI decode speed.

The full runner also runs RWKV serve smoke by default after the SSM bench matrix. To collect the bench matrix while this
serve gap is still known-failing:

```bash
RUN_SERVE_SMOKE=0 tools/ci/overnight_hardening.sh
```

Then update:

- `docs/campaign/test_matrix.md`
- `docs/campaign/findings_summary.md`
- `docs/campaign/autonomous_run_log.md`

## Implementation lanes

1. **Documentation and routing policy**
   - Document the model-choice rules in user-facing or operator-facing docs.
   - Add an internal decision table: `ctx_tokens`, quality mode, latency mode, model family.

2. **Dedicated workload suite**
   - Add prompts that look like real long-context coding work:
     - "summarize this file and identify the bug"
     - "answer a question whose evidence appears near the start"
     - "extract JSON facts from a long mixed document"
   - Track both latency and quality.

3. **Serve-path hardening**
   - Smoke `hawking serve` with RWKV/SSM weights.
   - Verify streaming, cancellation, request isolation, and repeated-request stability.
   - Current read-only audit: the `hawking serve` CLI passes weights into `hawking_core::model::load_engine`,
     the same generic loader used by `hawking generate`, so SSM families should reach the server front door.
     The smoke gate is still required before calling this production-ready.

4. **Hybrid SSM -> Qwen flow**
   - Use SSM to produce cheap long-context notes or candidate summaries.
   - Feed the compacted result to Qwen for final answer quality.
   - Gate on end-to-end latency and answer correctness, not just SSM decode tps.

## Non-goals

- Do not use SSM speed to justify weakening Qwen output defaults.
- Do not revive spec-decode speed work unless the verifier/proposer overhead wall is removed.
- Do not wire int4-KV-PC as a substitute for SSM productionization; int4-KV is a transformer footprint lever,
  while SSM is the long-context speed path.
