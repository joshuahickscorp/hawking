# Path-to-90 B1 — PPL eval harness (close report)

**Date:** 2026-05-15
**Branch:** `claude/modest-williamson-57d50f` (continues `strange-proskuriakova-b5d48e`)
**Base:** `0b4bd3d` (session_closeout 2026-05-15)
**Status:** SHIPPED — pure tooling, no perf risk.

## What this ships

A reproducible perplexity oracle for KV-cache and expert-quant variants.

| Artifact | Purpose |
|---|---|
| `dismantle ppl-eval` subcommand | Rust-side per-sample NLL via `forward_tokens_batched_for_test` + log_softmax. Single forward pass per sample with `reset_kv_for_test` between samples. |
| `tools/bench/ppl_eval.py` | Python orchestrator with `prep` / `run` / `diff` subcommands. `prep` downloads WikiText-2 and slices a deterministic subset; `run` invokes the Rust subcommand; `diff` computes ΔPPL + per-sample NLL histogram and applies the ±0.5% gate. |
| `tests/data/wikitext2_256_samples.jsonl` | 256-paragraph deterministic slice of WikiText-2 raw test split (seed=20260515, min length 80 chars). Filter drops section headers and blank rows. Regenerable byte-identical via `ppl_eval.py prep`. |
| `reports/path_to_90/stage1_b1/baseline_fp16kv.jsonl` | The FP16-KV reference baseline — current default profile (`metal-default` post-A5/A4), max_tokens=128. Every future variant diffs against this file. |

## Baseline numbers

```
{
  "samples": 256,
  "tokens_scored": 26849,
  "avg_nll": 3.533374,
  "ppl": 34.2393,
  "model": "DeepSeek-V2-Lite-Chat",
  "profile": "metal-default",
  "elapsed_s": 1449.1
}
```

- ~24 min wall on M3 Pro (Claude running, slm idle, `nice -n 19 taskpolicy -b`).
- 26,849 scored tokens across 256 samples (mean ~105/sample; some paragraphs shorter than the 128-token cap).
- The absolute PPL value is not the figure of merit. It's specific to: Q4_K_M quant, 128-token windows reset between samples (high BOS-adjacent NLL), the V2-Lite-**Chat** SFT variant (often higher raw-text PPL than the base), and this particular 256-sample slice. The figure of merit is **ΔPPL vs this same baseline**, gated at ±0.5%.

## Reproducibility validation

| Property | Method | Result |
|---|---|---|
| Self-diff (same JSONL twice) | `tools/bench/ppl_eval.py diff` over re-run output | ΔPPL = +0.000%, per-sample ΔNLL = 0 across 4/4 |
| Cross-invocation determinism | Re-run on first 16 samples, compare `nll_sum` per sample against `baseline_fp16kv.jsonl` | **16/16 bit-identical** |
| Engine math sanity | NLL math = `logsumexp(logits) - logits[target]`, max-subtracted for stability | NLL > 0 for every sample (correct sign), avg ~3.5 (matches V2-Lite-Chat Q4_K_M order-of-magnitude on raw wiki text) |
| Cargo lib tests | `cargo test --workspace --lib --release` | 5 + 25 + 9 tests pass; no new failures |

The reproducibility check is the load-bearing claim. Future variant runs only need to demonstrate `|ΔPPL/baseline_PPL| ≤ 0.5%` against the same `baseline_fp16kv.jsonl` file to clear the quality gate.

## How to use this for an A2 / B2 / B3 lever

```bash
# 1. Implement the variant (e.g. Q8 latent-KV path under a feature flag).
# 2. Build the binary.
# 3. Run the variant with the matching kernel-profile, write to its own out file.

tools/bench/ppl_eval.py run \
    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.q8kv.json \
    --max-tokens 128 \
    --out reports/path_to_90/stage1_a2/q8_latent_kv.jsonl \
    --diff-baseline reports/path_to_90/stage1_b1/baseline_fp16kv.jsonl

# 4. The orchestrator prints ΔPPL + the PASS/FAIL line at ±0.5% threshold.
#    For the FULL quality bar, also run:
#       dismantle batch-hash --prompts <baseline-prompts> --tokens 64
#    against the variant profile and diff hash columns vs the baseline file.
#    Both must pass: bit-identical 64-token greedy AND ΔPPL within ±0.5%.
```

## Design choices

- **Why `forward_tokens_batched_for_test` and not `forward_tokens_for_test`?**
  Both return bit-identical logits on this engine (verified — the sequential and single-TCB fast-path produced identical NLL on the 4-sample smoke). The batched-TCB version is ~2× faster wall-clock by eliminating K-1 commit+wait round-trips per sample. To enable the fast path the harness sizes `EngineConfig::max_batch_size = max_tokens`.

- **Why max_tokens=128 default?**
  128 tokens × 256 samples × ~70 ms per scored token ≈ 25 min wall on M3 Pro. Long enough that mid-sample positions have meaningful context (~50+ tokens of conditioning before the high-context tail); short enough that a full baseline is comfortably under the path-to-90 session budget. Future runs can widen to 256/512 once a faster prefill-style path is added; the JSONL schema is compatible.

- **Why reset KV between samples, not slide a window?**
  Stage 0 attribution showed dispatch overhead dominates over per-token GPU compute; the per-sample reset cost is negligible vs the per-token forward cost. Sliding-window PPL (the standard HF eval) would require either prefilling a 1024-token context with a per-position decode tail, or implementing chunked prefill with intermediate logit extraction. Neither buys quality for the levers this oracle is gating (Q8/Q3 KV, expert tiering). Independent paragraphs is the simpler, equally-discriminating choice for ΔPPL on quant noise.

- **Why JSON-lines, not aggregated summary only?**
  Per-sample NLL is needed to compute the per-sample ΔNLL distribution in `diff`. Distribution shape (p10/p50/p90) catches the case where a quant variant has the same corpus average but adds a long tail of catastrophic samples — the kind of failure that escapes a single-number PPL gate.

- **Why not score the BOS token itself?**
  The model never produces BOS as a target — it's an input-only sentinel. Predicting it has no signal. Per-sample math: tokenize with BOS, forward `tokens[0..L-1]`, score `tokens[1..L]`. Total scored = L-1 per sample.

## Files committed

```
crates/dismantle/src/main.rs                                     (modified — +PplEval subcommand)
tools/bench/ppl_eval.py                                          (new)
tests/data/wikitext2_256_samples.jsonl                           (new — deterministic slice)
reports/path_to_90/stage1_b1/baseline_fp16kv.jsonl               (new — force-add)
reports/path_to_90/stage1_b1/baseline_fp16kv.log                 (new — force-add)
reports/path_to_90/stage1_b1/close.md                            (this file — force-add)
```

## Followups

- **A2 (Q8 latent KV)** is now unblocked — the gate it needs is in place. Implementation is bounded by the structure described in `dismantle-path-to-90-immutable-jellyfish.md §A2`.
- **B2 (WHT 3-bit KV)** likewise — once the static codebook is fit, this same harness produces the calibration ΔPPL.
- If `forward_tokens_batched_for_test` becomes a perf bottleneck for repeated eval runs (Stage 2 will need many calibration sweeps), the next optimization is exposing per-position logits from the real prefill path. That's a larger engine change and out of scope here — `max_tokens=128` runs in ~25 min today which is fast enough for the gate-pass/fail decision.
- The Chat-variant tokenizer adds a system-prompt template by default? No — verified by `encode_prompt_for_batch` returning the raw BOS-prefixed token sequence for our text-only paragraphs. If we want a base-model PPL number for comparison with published numbers we'd need a base GGUF, separate work.

## What this does NOT cover

- This is the **PPL** half of the quality bar only. The plan also requires **token-identical greedy at 64 tokens** for KV-cache variants. That half is already covered by `dismantle batch-hash --tokens 64` against the existing `tests/golden/_phase1_token_baseline_expanded.hashes`. Both gates must pass for a variant to ship.
- No perf measurement. This is pure correctness tooling. Future levers still need env-A + env-B measurement per the regression_validation.md methodology.
- No automated CI integration. The harness is a one-shot tool invoked when implementing/verifying a quant variant. Adding a daily CI job would multiply by ~24min/run; not warranted until Stage 2 cadence justifies it.
