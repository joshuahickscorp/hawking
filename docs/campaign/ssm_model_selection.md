# SSM Model-Selection Decision Table (operator-facing)

Operationalizes the routing lane of `ssm_productionization.md` into a concrete table. Grounded in measured evidence
(`test_matrix.md`); **honest status caveats are load-bearing — read them.**

## Status caveats (2026-06-22)
- **RWKV serve path is NOT production-ready yet.** Admission is fixed (`admitted=1`, no hang), but the multiseq decode
  emits an immediate-EOS/empty token (RWKV prefill→GPU-slot state handoff bug). Gate: `rwkv7_prefill_slot_multiseq_parity`.
  **Until that is green, route long-context serve traffic to Qwen** (slow but correct) or use RWKV via single-stream `generate`.
- **RWKV-7-0.4B-SFT is a SMALL model** (0.4B vs Qwen-3B). Its win is *speed + flat long-context*, NOT raw answer quality.
  A per-prompt-class quality gate (`Lane 4`) must pass before RWKV is the default for any quality-sensitive class.
- **Mamba-2-370M is NOT viable now** (unoptimized ~11 tps; 8k path errors). Excluded from routing until its kernel lands.

## Primary table — context length × priority
| ctx tokens | latency-priority | quality-priority | notes |
|---|---|---|---|
| **< 2k** (short) | RWKV-7 (119 tps) *if quality-gate passes for the class*; else Qwen | **Qwen-3B** (40 tps, full quality) | Qwen's KV wall is irrelevant short; the 3B quality usually wins for short asks. |
| **2k–8k** (mid) | **RWKV-7** (110–119 tps vs Qwen 8.6–18.8) | Hybrid (RWKV draft → Qwen finalize) or Qwen if correctness-critical | RWKV is ~6–14× faster here; the gap is the whole point. |
| **8k–32k+** (long) | **RWKV-7** (flat ~119 tps; Qwen → single-digit) | **Hybrid** (RWKV summarizes/extracts → Qwen answers) | Qwen is often impractical at depth (8.6 tps @8k, worse beyond). SSM bounded state shines. |

## Prompt-class overlay (apply on top of the table)
| prompt class | guidance |
|---|---|
| code summarization / bug-spotting over a long file | RWKV draft (fast over the whole file) → Qwen for the precise fix. Quality-gate RWKV before solo use. |
| long-context retrieval (evidence near the start) | RWKV viable IF the Lane-4 retrieval gate passes (SSM recall over distance is the key risk to test). Else Qwen. |
| JSON fact extraction (format-constrained) | Quality-gate first — small models drift on strict format. Prefer Qwen for hard schema until RWKV passes. |
| math / multi-step reasoning | Qwen-3B by default (0.4B reasoning is weak); RWKV only if the math sanity gate passes. |
| multilingual | Quality-gate per language; default Qwen until RWKV's World-tokenizer quality is verified per language. |
| short, quality-sensitive chat | Qwen-3B (default). |

## Hybrid flow (the recommended long-context product path)
1. **RWKV-7** ingests the full long context (flat, cheap) → produces compact notes / candidate summary / extracted facts.
2. **Qwen-3B** takes the *compacted* result (now short) → produces the final, high-precision answer at full quality.
3. Gate end-to-end on **answer correctness AND total latency**, not RWKV decode tps alone.
This keeps Qwen's quality where it matters while paying RWKV's flat long-context cost only on the bulk ingest.

## Fallback rules (conservative; no silent semantic change)
- RWKV fails a prompt-class quality gate → route that class to Qwen; log the reason.
- RWKV model-load / World-tokenizer unavailable → fall back to the transformer path with a logged reason.
- **Never** weaken Qwen output defaults to justify SSM speed. Routing is additive; defaults stay conservative.

## Decision pseudocode (operator reference)
```
if ctx < 2k:
    return RWKV if (latency_priority and class_quality_ok(class)) else Qwen
if ctx < 8k:
    return RWKV if latency_priority else (hybrid if class_allows_hybrid else Qwen)
# ctx >= 8k
return RWKV if latency_priority else hybrid   # Qwen alone is usually impractical at depth
# ALWAYS: if not serve_decode_ok(RWKV): substitute Qwen for any RWKV *serve* route (generate path still ok)
```

## Validation hooks
- Speed evidence: `tools/ci/ssm_product_gate.sh` (speed matrix) + `test_matrix.md`.
- Quality gate (Lane 4): TODO `tools/ci/ssm_quality_suite.sh` — per-class pass/fail vs Qwen/rubric.
- Serve correctness (Lane 1): `rwkv7_prefill_slot_multiseq_parity` must be green before any RWKV *serve* route.
