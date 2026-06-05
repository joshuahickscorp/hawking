# Continuous Batching Ship Plan
*Authored 2026-06-04. Goal: ship GPU continuous-batching for aggregate serving TPS.*

## Why this matters

Single-stream B=1 decode is structurally tapped (kernel track exhausted, confirmed
2026-06-04 with predec_4r bench). The one large live lever is **GPU continuous
batching**: the weight-amortizing GEMM (v3w) already ships; the multi-seq MHA is
built and verified correct; nothing is wired to the HTTP server. Target: 3.5–5.6×
aggregate TPS at B=8 over the B=1 anchor (~114–184 tokens/sec vs llama's 55).

## Current state (from June 3 review wave)

**Already built and verified:**
- v3w batched projections (weight GEMM amortized across B slots) ✓
- `mha_decode_f32_batched_multiseq` (multi-seq MHA with per-slot positions) ✓
- `rope_f32_batched_multiseq_tcb` (batched RoPE, per-slot positions) ✓
- `kv_scatter_append_multiseq_tcb` (batched K+V append) ✓
- `silu_mul` batched ✓
- Slot manager + `decode_ready_once` + per-slot Sampler ✓

**Known blockers (latent bugs, detonate when wired to serving):**
1. Arena reallocs on batch-size growth → wipes all in-flight KV
2. Slot indices are compacted batch-position, not stable slot-id → eviction
   corrupts neighbor slots
3. No multi-seq prefill (decode-from-0 only)
4. LM head is B sequential reads of ~622MB f16 (dominant un-amortized cost)
5. HTTP server still one-request-per-mutex `engine.generate`

## Task list

### CB-1 — Arena lifecycle fix [PREREQUISITE, S ~1 day]
Fix the two arena bugs that block everything else:
- Allocate DenseDecodeArena at `max_batch` capacity on first use; never realloc
- Use stable `slot_id`/region indexing (not compacted batch-position)
- Zero slot region on release (not free the arena)
Gate: `tests/multiseq_churn_parity.rs` (divergent-position + varying-B + slot-churn)

### CB-2 — GPU-batched LM head [M ~2 days, depends CB-1]
Route B slot activations through `gemm_q4_k_m_batched_v3w` in one dispatch
instead of B sequential per-slot reads of the 622MB f16 head (or Q4K_LMHEAD).
Estimated gain: dominant jump toward 3.5–5.6× target.

### CB-3 — Batch embed + layer-0 rmsnorm + RoPE [S ~4h, depends CB-1]
`rope_f32_batched_multiseq_tcb` already exists. Wire embed + rmsnorm to use
batched dispatch. Removes 4B dispatches/step.

### CB-4 — Single batched KV-append [S ~4h, depends CB-1]
`kv_scatter_append_multiseq_tcb` already exists. Verify it's the only call site
for KV append in the multi-seq path; remove any residual per-slot appends.
Eliminates ~574 dispatches/step at B=8.

### CB-5 — Multi-seq prefill path [M ~2 days, depends CB-1]
New request needs KV populated from prompt. Options: (a) decode-from-0 by
running the multi-seq path from pos=0, or (b) a dedicated multi-seq prefill
that reuses the v3w GEMM already batched in prefill. Parity: each slot's
prefill output == solo generate prefill.

### CB-6 — HTTP server wiring [M ~2 days, depends CB-5]
Change `dismantle-serve` from mutex→generate to admission queue + batched loop.
Per-slot SSE streaming. Depends on F0.3 (fixed arena, CB-1) + prefill (CB-5).
Spec: slot manager is already built; missing = admission loop + SSE.

## Sequencing

```
CB-1 ──┬── CB-2
       ├── CB-3
       ├── CB-4
       └── CB-5 ── CB-6
```

CB-2, CB-3, CB-4 are independent after CB-1 and can be built in parallel
(different functions/files in qwen_dense.rs).

## Gate

Correctness: `tests/multiseq_churn_parity.rs` — divergent positions, varying B,
slot eviction; each survivor's tokens must match its solo decode (b3sum).
Performance: `tools/bench/paired_lever.sh` B=1 vs B=4 vs B=8 aggregate tps.
Target: B=8 aggregate ≥ 3× B=1 single-stream.

## Bench claim (what we can say when done)

"dismantle serves 8 concurrent code-completion requests at Xk tok/min on M3 Pro,
compared to llama.cpp's single-stream Y tok/min — Z× higher serving throughput
from the same hardware."
