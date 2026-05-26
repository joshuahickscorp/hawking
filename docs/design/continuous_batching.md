# Continuous Request Batching for `dismantle serve`

Author: Joshua Hicks <joshuahicksboba@gmail.com>
Status: design / not implemented
Anchor commits: `68c0ece` (batched MHA decode), `4e73195` (per-op consolidation), `b9ce844` (B=8 GEMM widen), `e81f563` (v3 shmem staging), `5b80cce` (batched prefill wire-up), `6d58576` (batched Q4_K GEMM kernel).

## 0. TL;DR

The kernel-side groundwork for amortizing Q4_K weight reads across a
B-dimension is already on `main` and shipping behind
`DISMANTLE_QWEN_BATCH_PREFILL=1`. The `dismantle-serve` crate already
has a `Slot` / `Scheduler` / `BatchDriver` skeleton
(`crates/dismantle-serve/src/batch/{mod,scheduler,driver}.rs`) sketched
against an `Engine::forward_tokens_batched(tokens, positions)` trait
method that currently falls back to a sequential per-token loop. The
gap between today and a real vLLM-style continuous batcher is:

1. A per-request KV cache (the current `KvCache` at
   `crates/dismantle-core/src/cache/mod.rs:22` is a single contiguous
   buffer with one `seq_len` cursor shared across the engine).
2. A batched decode kernel that accepts an **arbitrary** per-batch
   position vector rather than the current contiguous
   `[p0..p0+B)` constraint hard-coded into
   `forward_tokens_batch_tcb`
   (`crates/dismantle-core/src/model/qwen_dense.rs:1571` on `main`)
   and `mha_decode_f32_batched`
   (`crates/dismantle-core/shaders/mha.metal:137` on `main`).
3. An HTTP-side request queue that maps inbound SSE / JSON requests
   onto slots and drives the engine in a step loop rather than
   one-shot per request.

Recommended path: **(α) decode-only batching first**, gated behind
`DISMANTLE_CB_ALPHA=1` and `--continuous-batching` on the CLI, then
**(β) prefill+decode interleaving** when there is enough demand to
justify the per-request KV refactor.

---

## 1. Current engine loop architecture

### 1.1 Code anchors

| File | Lines | Role |
|---|---|---|
| `crates/dismantle/src/main.rs` | 20-42, 246-265 | CLI surface for `dismantle serve` |
| `crates/dismantle-serve/src/lib.rs` | 14-62 | Engine load + axum wire-up |
| `crates/dismantle-serve/src/http/mod.rs` | 25-39, 117-247 | Routes + per-request blocking task |
| `crates/dismantle-serve/src/batch/mod.rs` | 13-139 | `Slot` + `SlotState` + `DecodeStep` |
| `crates/dismantle-serve/src/batch/scheduler.rs` | 14-146 | Slot manager (admit, decode_batch, apply_decode_logits) |
| `crates/dismantle-serve/src/batch/driver.rs` | 20-58 | `BatchDriver::decode_ready_once` → `Engine::forward_tokens_batched` |
| `crates/dismantle-core/src/engine.rs` | 168-261 | `trait Engine` (CB seam: `encode_prompt_for_batch`, `forward_tokens_batched`, `eos_id_for_batch`) |
| `crates/dismantle-core/src/cache/mod.rs` | 22-84 | Single-request `KvCache` |
| `crates/dismantle-core/src/model/qwen_dense.rs` | 290-386 (main: 683-849) | `QwenDense::generate` |
| `crates/dismantle-core/src/model/qwen_dense.rs` (main) | 1571-1962 | `forward_tokens_batch_tcb` (batched prefill, contiguous positions) |
| `crates/dismantle-core/src/metal/dense_decode_arena.rs` (main) | 8-141 | GPU-resident scratch + `k_cache_buf` / `v_cache_buf` |
| `crates/dismantle-core/shaders/mha.metal` (main) | 125-220 | `mha_decode_f32_batched` |

### 1.2 What happens today for one request

The HTTP layer takes the engine mutex per request and runs
`engine.generate(...)` from start to finish on a blocking task
(`http/mod.rs:217-247` for SSE, `:249-277` for JSON). Inside
`QwenDense::generate` (`qwen_dense.rs:290` here, `:683` on main) the
flow is:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Client POST /v1/chat/completions                                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ axum handler
                              ▼
        ┌───────────────────────────────────────────┐
        │ tokio spawn_blocking → engine.lock()      │  (parking_lot::Mutex)
        │   serializes every request behind the     │
        │   one Engine instance.                    │
        └───────────────────────────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │ QwenDense::generate                       │
        │   tokenizer.encode → prompt_ids           │
        │   kv.reset()  ← clears shared cache       │
        │                                           │
        │   PREFILL (one of two modes):             │
        │     a) DISMANTLE_QWEN_BATCH_PREFILL=1     │
        │        for i in 0..prompt_len step 8:     │
        │          forward_tokens_batch_tcb(...)    │
        │     b) sequential                          │
        │        for each token in prompt:          │
        │          forward_token{,_greedy_tcb}      │
        │                                           │
        │   DECODE (one path):                      │
        │     loop max_new_tokens:                  │
        │       forward_token{,_greedy_tcb}         │
        │       sample → record → stream            │
        │                                           │
        │   Done event → release mutex.             │
        └───────────────────────────────────────────┘
```

Sequence diagram for the prefill+decode interleaving inside one
forward step (batched prefill chunk, B≤8 contiguous tokens):

```
QwenDense::forward_tokens_batch_tcb (B contiguous tokens at [p0..p0+B))
  │
  ├─ embed_lookup × B           (B sequential dispatches into x_buf_batch)
  ├─ rmsnorm × B                (layer 0 attn_norm; later layers come from
  │                              the fused add_rmsnorm tail)
  │
  └─ per layer × n_layers:
       ├─ batched_proj(q_proj) ──► gemm_q4_k_m_batched_v3w_pinned_tcb
       │                            (1 dispatch, weight read once)
       ├─ batched_proj(k_proj) ──► same for K (Q6_K falls back to B× gemv)
       ├─ batched_proj(v_proj) ──► same for V
       ├─ add_inplace_broadcast × 3  (Q/K/V biases, 1 dispatch each over B rows)
       ├─ rope × 2B               (Q + K, per-batch position)
       ├─ memcpy_f32 × 2          (KV-cache append for [p0..p0+B))
       ├─ mha_decode_f32_batched  (1 dispatch, 2D grid (n_heads, B); each
       │                            TG computes seq_len = p0 + b + 1)
       ├─ batched_proj(o_proj)    (1 dispatch)
       ├─ add_rmsnorm_fused_batched (B rows, 1 dispatch)
       ├─ batched_proj(ffn_gate)  (1 dispatch)
       ├─ batched_proj(ffn_up)    (1 dispatch)
       ├─ silu_mul                (flat over B × intermediate, 1 dispatch)
       ├─ batched_proj(ffn_down)  (Q4K-requant fast path)
       └─ add_rmsnorm_fused_batched (B rows, 1 dispatch)
  │
  └─ tcb.commit_and_wait()       (one GPU sync per chunk)
  │
  └─ kv.seq_len += B             (CPU mirror cursor advances)
```

Two structural facts that matter for continuous batching:

1. **`QwenDense::kv` is a single `KvCache`.** Every byte at
   `kv.keys[layer][..]` belongs to whichever request is currently
   holding the engine mutex. `kv.reset()` at line 317 (here) /
   line 710 (main) wipes between requests. There is no concept of a
   per-request slot.
2. **`forward_tokens_batch_tcb` requires contiguous positions.** See
   the explicit check at `qwen_dense.rs:1591-1597` (main):
   ```
   if p != positions[0] + i { return Err("positions must be contiguous"); }
   ```
   and the matching kernel-side use of `args.p0 + batch_id + 1` for
   per-element seq_len in `mha.metal:154`. Two requests at different
   absolute decode positions cannot share one batched forward today.

### 1.3 What `dismantle-serve` already has for CB

The `batch` submodule is shaped correctly but inert. From
`scheduler.rs:14-146`:

- `Scheduler::admit` claims an idle slot and stuffs prompt IDs into it.
- `Scheduler::decode_batch(max)` returns up to N `DecodeStep`s
  (slot_id, last_token, current position) for slots in `Decoding` state.
- `BatchDriver::decode_ready_once` calls
  `engine.forward_tokens_batched(&tokens, &positions)` with whatever
  tokens / positions the ready slots carry, then routes resulting
  logits back to the per-slot sampler.

The HTTP layer never instantiates `BatchDriver`. It locks the mutex
and runs `engine.generate()` directly (`http/mod.rs:223,254`), so the
scheduler is dead code at runtime.

The trait method `Engine::forward_tokens_batched` (engine.rs:206) has
a default impl that calls `forward_tokens_for_test`, which is a
correctness loop:

```rust
// engine.rs:206-221
fn forward_tokens_batched(&mut self, tokens, positions) -> Result<Vec<Vec<f32>>> {
    self.forward_tokens_for_test(tokens, positions)
}
// qwen_dense.rs:412-428 (here)
fn forward_tokens_for_test(&mut self, tokens, positions) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(tokens.len());
    for (i, &token) in tokens.iter().enumerate() {
        out.push(self.forward_token(token, positions[i])?);
    }
    Ok(out)
}
```

→ Calling `forward_tokens_batched` today does **not** share weight
reads. It runs N back-to-back per-token forwards against the shared
`self.kv`, which is wrong for multi-request inputs because all tokens
get appended into the same KV slab. **This is the path the prior
`forward_tokens_batched` work documented under the "dead-lever fence"
in §6.**

---

## 2. The continuous-batching invariant set

### 2.1 Per-request KV slots

Two viable shapes:

- **(A) Contiguous-with-free-list.** Allocate one logical KV cache of
  shape `(S_max, n_kv_heads, head_dim)` per slot, identically to the
  current `KvCache` but `N_slots` of them, packed back-to-back in a
  single GPU `PinnedBuffer`. Slot `i` owns rows
  `[i * S_max, (i+1) * S_max)`. Free list = `Vec<u32>` of idle
  slot IDs. Simple, fragmentation-free, but caps each request at
  `S_max` tokens and wastes memory on short requests.

- **(B) Paged attention.** Page = `(P, n_kv_heads, head_dim)` block,
  typical `P = 16` or `32`. Each slot owns a `Vec<u32>` of page IDs
  (a "block table"). MHA kernel does an extra indirection per
  read. ~4× more complex shader changes but no waste on short
  requests and supports long contexts.

**Recommendation for v1 of CB: (A).** Reasons:
1. The `DenseDecodeArena::new` constructor already allocates
   `total_kv_bytes = n_layers * max_seq * kv_dim * 4 B` for a single
   slot (dense_decode_arena.rs:91, on main). Replicating this N times
   is straight extension: change shape to
   `(N_slots, n_layers, max_seq, kv_dim)` with `N_slots` fixed at
   load time from `--max-concurrent-requests`.
2. The contiguous layout means the existing
   `mha_decode_f32_batched` kernel needs **only one** new index: a
   `slot_id` per batch element, which becomes an offset added to the
   layer base address. No page-table indirection inside the inner
   loop.
3. Paged attention is the right answer for the next iteration when
   workloads have a wide prompt-length distribution. Punt to (β)+1.

**Memory check.** Qwen-3B Q4_K_M:
`n_layers=36, max_seq=4096, n_kv_heads=2, head_dim=128 → kv_dim=256 →
per-slot KV ≈ 36 × 4096 × 256 × 4 B × 2 (K+V) = 288 MB`. At
`N_slots=8` that is **2.3 GB** of resident KV cache. On an 18 GB M3
Pro this is fine but already constrains slot count. Default cap
should be `N_slots=4` (1.15 GB) with a CLI override.

### 2.2 Mixed-position batched MHA

The shader change is small. From `mha.metal:137` (main):

```metal
struct ArgbufMhaDecodeBatched {
    uint p0;          // shared base position
    ...
};
const uint SEQ = args.p0 + batch_id + 1u;   // contiguous assumption
```

Becomes:

```metal
struct ArgbufMhaDecodeBatchedCB {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint group_size;
    uint slot_stride_floats;   // n_layers * max_seq * kv_dim (per-slot stride)
    uint layer_stride_floats;  // max_seq * kv_dim
    float scale;
};

kernel void mha_decode_f32_batched_cb(
    constant ArgbufMhaDecodeBatchedCB& args [[buffer(0)]],
    device const float*  q          [[buffer(1)]],
    device const float*  k_cache    [[buffer(2)]],   // global KV pool
    device const float*  v_cache    [[buffer(3)]],
    device       float*  out        [[buffer(4)]],
    device const uint*   slot_ids   [[buffer(5)]],   // per-batch-elem slot
    device const uint*   positions  [[buffer(6)]],   // per-batch-elem pos
    threadgroup float*   shmem      [[threadgroup(0)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    ...)
{
    const uint h        = tg_id.x;
    const uint batch_id = tg_id.y;
    const uint slot     = slot_ids[batch_id];
    const uint SEQ      = positions[batch_id] + 1u;

    // K/V base for this slot + layer:
    device const float* k_base = k_cache + slot * args.slot_stride_floats
                                          + layer_offset_passed_via_argbuf;
    ...
}
```

Per-batch-element `seq_len` (= per-request decode position + 1) is the
only loop-bound change. The shmem `scores[SEQ]` allocation now varies
per TG; for the contiguous-position kernel each TG already sees a
different `SEQ` (the largest is `p0 + B`), so threadgroup memory was
already sized to the worst case. For CB we size to
`max_position_in_batch + 1`, passed in the argbuf as
`max_seq_in_batch` so the host can compute the shmem allocation.

### 2.3 Mixed prefill+decode in one forward

Two design choices, listed from cheapest to most general:

- **(α-pure) Decode-only batching.** All batch slots are in decode
  mode (`B=N` tokens, each at a different position, batched across
  slots). New requests run prefill alone, on the engine mutex, blocking
  decode for the duration of their prefill. Reuses
  `forward_tokens_batch_tcb` virtually unchanged (single-request, batched
  along the *prompt-token* dimension). Win: existing kernels.
- **(β) Per-batch-slot mode tag.** A new
  `forward_step_cb(slots: &[CbSlotInput])` enters one forward per
  scheduler step. Each `CbSlotInput` carries either:
  - `Decode { slot_id, token, position }` — one token at one position.
  - `PrefillChunk { slot_id, tokens: &[u32], base_position }` — up to
    `chunk_size` tokens (typical 16-64) that consume one row each.

  All slots' tokens are concatenated into the `B`-dim of the GEMM. The
  MHA kernel handles all rows uniformly because the new shader signature
  already takes per-row `(slot_id, position)`. The `B`-dim grows from
  "B prefill tokens of one request" to "Σ tokens across all active
  slots", with one cap: `B ≤ max_b_kernel` (currently 8 in
  `dense_decode_arena.rs:69`).

We recommend **β-unify**: every batch element is a single (slot,
token, position) triple. Decode-only is the degenerate case where each
slot contributes exactly one. Prefill is the case where one slot
contributes a 16-64 chunk. The kernel does not need to distinguish.

### 2.4 Request lifecycle

```
       admit                       evict-on-cancel
          │                              │
          ▼                              ▼
    ┌──────────┐  prefill ready    ┌──────────┐
    │ QUEUED   │ ────────────────► │ DECODING │
    └────┬─────┘                   └────┬─────┘
         │ slot available               │ EOS / max_tokens / stop string
         ▼                              ▼
    ┌──────────┐                   ┌──────────┐
    │PREFILLING│ ──── done ──────► │FINISHING │ ──► release slot ──► Idle
    └──────────┘                   └──────────┘
```

States already exist in `batch/mod.rs:14-19`:

```rust
pub enum SlotState { Idle, Prefilling, Decoding, Finishing, }
```

We need to add:
- `Queued` — admitted by HTTP but no slot has been claimed yet (when
  all slots are busy and admission policy is queue-not-reject).
- Per-state timestamps for p50/p99 latency math at the metrics seam.

---

## 3. Three implementation paths

### (α) Decode-only batching

**Scope:** Mix N concurrent in-flight requests' decode steps into one
forward. Prefill stays single-request and blocks decode during its
window.

**Files touched (rough):**

| File | Change | LOC delta |
|---|---|---|
| `crates/dismantle-core/src/cache/mod.rs` | Add `MultiSlotKvCache` (N × current shape, one shared `PinnedBuffer`) | +120 |
| `crates/dismantle-core/src/metal/dense_decode_arena.rs` | Size `k_cache_buf` / `v_cache_buf` by `N_slots`; add `slot_byte_offset(slot, layer)` | +40 |
| `crates/dismantle-core/shaders/mha.metal` | New `mha_decode_f32_batched_cb` kernel (per-elem slot+position vectors) | +90 |
| `crates/dismantle-core/src/kernels/mod.rs` | Wrapper for new kernel | +60 |
| `crates/dismantle-core/src/model/qwen_dense.rs` | New `forward_step_cb(slots: &[CbDecodeSlot])` method (decode-only variant of `forward_tokens_batch_tcb`); reuses every other dispatch unchanged | +260 |
| `crates/dismantle-core/src/engine.rs` | Trait method `forward_step_cb` with default returning `Unimplemented` | +20 |
| `crates/dismantle-serve/src/batch/scheduler.rs` | Switch admission to per-slot-KV; track `Queued` state | +80 |
| `crates/dismantle-serve/src/batch/driver.rs` | Replace `forward_tokens_batched` call with `forward_step_cb` | +60 |
| `crates/dismantle-serve/src/http/mod.rs` | New `cb_handler` task that owns the engine and drives `decode_ready_once` in a loop; per-request channel for streaming | +180 |
| `crates/dismantle/src/main.rs` | `--continuous-batching` flag, `--max-concurrent-requests N` flag | +25 |
| `crates/dismantle-core/tests/cb_alpha_decode_parity.rs` | N=1 CB output ≡ single-request output bit-identical; N=2 CB output for each slot ≡ what that slot would have produced run alone | +220 |

**Total: ~1,155 LOC** new, ~50 LOC modified. Mostly mechanical
shader + wrapper + arena resize; the qwen_dense forward step is a
near-copy of `forward_tokens_batch_tcb` with `p0` removed from
argbuf and a per-batch-elem slot index added to the KV append +
MHA dispatches.

**Risk surface:**
- KV append memcpy currently uses `kernels::memcpy_f32_off_tcb`
  (`qwen_dense.rs:1832-1849` main). For CB the destinations are
  *non-contiguous* (slot i, slot j, slot k may not be adjacent), so
  we either (i) issue B individual memcpys, or (ii) write a small
  `kv_scatter_append_cb` shader. (ii) is the right answer (one
  dispatch, B threads).
- f16 vs f32 precision drift: the existing kernels are f32 K/V; the
  new ones must stay f32 K/V to keep the parity gate aligned with
  the current bit-identical greedy test.
- Slot-cache slicing means `qwen_dense.kv` (CPU mirror, line 270
  here) goes away from the CB path. Hybrid prefill (CPU) → CB decode
  is not supported in α; CB requires `DISMANTLE_QWEN_TCB=1` end-to-end.

**Parity gates:**
1. `cb_alpha_decode_parity::n1_eq_singleton` — `forward_step_cb`
   with one slot at position `p` produces identical logits to
   `forward_token_greedy_tcb(t, p)`.
2. `cb_alpha_decode_parity::n2_independent` — Two
   sequentially-generated 16-token completions, then the same two
   requests interleaved through `forward_step_cb` at N=2, must produce
   bit-identical token IDs per slot.
3. `cb_alpha_decode_parity::n4_kv_isolation` — At N=4, scrambling
   one slot's KV before the next step must not affect the other three
   slots' next-token outputs.
4. `cb_alpha_decode_parity::eviction` — Killing slot 2 mid-decode
   and admitting a new request into slot 2 must not contaminate
   slots 0/1/3.

**Bench gates** (see §7 for thresholds): aggregate req/s must be ≥
1.8× single-request baseline at N=4, ≥ 3.0× at N=8 (the
weight-amortization ceiling is 4×–6× and we expect to leave some on
the table to kernel overhead).

### (β) Full prefill+decode interleaving (vLLM-style)

**Scope:** One unified `forward_step_cb(rows: &[CbRow])` where each
row is either a decode token or a prefill chunk-token. The scheduler
admits prefill chunks of new requests into the same forward pass as
decode tokens of older requests, subject to a token budget
(`max_tokens_per_step`, typically 256-512).

**Files touched (delta on top of α):**

| File | Change | LOC delta |
|---|---|---|
| `crates/dismantle-serve/src/batch/scheduler.rs` | Chunked prefill admission; `pack_step(token_budget)` returns the optimal mix of prefill chunks + decode tokens | +200 |
| `crates/dismantle-serve/src/batch/driver.rs` | Split returned logits: prefill chunks emit no token; decode rows emit one token each | +90 |
| `crates/dismantle-core/src/model/qwen_dense.rs` | Generalize `forward_step_cb` to accept variable-length per-slot inputs; KV scatter writes (slot_id, position) tuples | +180 |
| `crates/dismantle-core/shaders/mha.metal` | Verify the CB kernel scales to higher B (the existing v3w kernel was tuned for B≤8; β may want B≤16 or 32) | +40 |
| `crates/dismantle-core/src/metal/dense_decode_arena.rs` | Variable-B sizing (param `max_step_tokens`) | +30 |
| `crates/dismantle-serve/src/batch/cb_loop.rs` (new) | Top-level event loop: pull from request queue, pack step, dispatch, emit | +220 |
| Tests | β-specific parity + soak | +400 |

**Total: ~1,160 LOC** *on top of* α, so β-cumulative is ~2,300 LOC.

**Risk surface:**
- Variable `B` per step changes the `gemm_q4_k_m_batched_v3w` kernel's
  performance envelope. The v3w kernel was tuned at B=8; for B=12 or
  B=16 we need a microbench to confirm we're not on the wrong side of
  register-pressure cliff (see [[v110_path30_findings.md]] for the
  pattern where adding registers regressed −14%). Likely needs a
  v3w-wider variant.
- Scheduling policy decisions (FCFS vs. shortest-first vs.
  priority): start with FCFS; everything else is a follow-up.
- Mixed precision per-batch (some requests Q8-KV, others f32-KV) is
  out of scope for β — the cache is f32 across all slots.
- Pre-fill chunks that span the slot's `S_max` boundary: scheduler
  must enforce `position + chunk_len ≤ S_max` or reject.

**Parity gates:**
1. All α gates still pass when running with chunk_size=1
   (degenerates to α).
2. `cb_beta_mixed_parity` — One slot doing 47-tok prefill in 8-tok
   chunks, simultaneously with three slots in decode, must produce
   identical token-1 logits at the moment prefill completes vs. a
   reference run where the prefill ran alone first then decode joined.
3. `cb_beta_soak_4x16` — 4 concurrent 16-tok completions, no
   crashes / leaks / KV bleed after 1,000 step iterations.

### (γ) Speculative continuous batching

**Scope:** Each decode slot emits K draft tokens per step via the
ngram or eagle5 draft head; the verify forward runs at B = Σ
`(1 + accepted_draft)` over all slots.

**Files touched (delta on top of β):**

| File | Change | LOC delta |
|---|---|---|
| `crates/dismantle-core/src/speculate/{shared,ngram}.rs` | Per-slot draft state; batched draft generation | +180 |
| `crates/dismantle-core/src/model/qwen_dense.rs` | `forward_verify_cb(rows: &[CbVerifyRow])` — each row has 1 + K_draft tokens | +160 |
| `crates/dismantle-serve/src/batch/cb_loop.rs` | Mixed accept/reject path; per-slot rewind on rejection | +180 |
| Tests | Spec acceptance parity per-slot | +250 |

**Total: ~770 LOC** on top of β. Cumulative ~3,100 LOC.

**Risk surface:** spec-decode in dismantle has a long history of
regressions (see [[spec_decode_runtime_NOT_broken_2026_05_22]],
[[v110_path30_findings]]). The composition with CB is high-variance.
**Gate γ on β actually shipping decode_tps neutral or positive at
N=1.** If β at N=1 is slower than today's `--speculate exact-shared`,
γ does not compose to a win.

### Ranking

| Path | Aggregate throughput @ N=4 | LOC | Time to ship | Risk |
|---|---|---|---|---|
| α | ~1.8-2.5× | ~1,150 | 1 session | Low-medium (KV slot refactor + new shader) |
| β | ~3-5× | ~2,300 cum. | 3-4 sessions | Medium (variable-B kernel envelope, scheduling policy) |
| γ | ~4-7× | ~3,100 cum. | 5-7 sessions | High (composes a known-fragile system) |

**Recommended: ship α first.** It is the smallest change that
delivers a real per-request throughput win and unblocks the rest of
the stack. β is conditional on real demand for mixed prefill+decode
(i.e. a workload where prefill latency hides badly behind decode of
older requests). γ is conditional on β.

---

## 4. API contract for `dismantle serve`

### 4.1 New CLI flags (`crates/dismantle/src/main.rs:20-42`)

```
--continuous-batching            (default: off)
--max-concurrent-requests N      (default: 4; effective max = min(N, slot capacity from --memory-limit-mb))
--queue-depth Q                  (default: 32; reject with 429 when exceeded)
--max-prefill-chunk-tokens C     (default: 16; β-only; ignored under α)
--cb-step-budget T               (default: 256; β-only; max tokens dispatched per forward)
```

### 4.2 Slot capacity / OOM behavior

At engine load:

1. Compute per-slot KV bytes = `n_layers * max_seq * n_kv_heads * head_dim * 8` (K + V at f32).
2. Effective `N_slots = min(--max-concurrent-requests,
   (memory_budget - weight_bytes) / per_slot_kv_bytes)`.
3. If `N_slots < 1`, return `Error::Model("memory budget cannot fit
   even one KV slot")` at load time. Matches the existing
   `memory_limit_mb` precondition in `engine.rs:30-34`.

At runtime, KV cache slot exhaustion cannot happen — slots are
fixed-capacity. The runtime failure mode is **request queue full**:

- If `queue_depth + active_slots >= max_concurrent_requests + queue_depth`,
  the HTTP handler returns `429 Too Many Requests` with
  `Retry-After: ceil(p50_completion_seconds)`.

### 4.3 Request cancellation mid-decode

The `GenerateRequest::abort: Option<Arc<AtomicBool>>` channel
(`engine.rs:122-124`) is already wired for the single-request path
(`qwen_dense.rs:301-306` checks it each step). For CB:

- HTTP layer creates the `abort` flag at admission and flips it when
  the SSE stream is dropped (`axum` SSE close → `tokio::sync::mpsc`
  receiver dropped → `tx.blocking_send` returns Err → handler flips
  the flag).
- The CB loop checks `slot.abort_flag` *before* including the slot in
  the next batch (`pack_step` filter). An aborted slot transitions
  `Decoding → Finishing` and is released after its current
  in-flight step completes. The current in-flight step is **not**
  rolled back; we always finish the GPU dispatch we already launched.
- Worst-case cancellation latency: one step ≈ one decode-token-time
  ≈ 40-60 ms at N=4 (vs. ~45 ms single-request decode).

### 4.4 Per-request metrics surfaced via `/metrics`

| Metric | Type | Notes |
|---|---|---|
| `dismantle_active_slots` | gauge | 0..N |
| `dismantle_queue_depth` | gauge | |
| `dismantle_request_admit_total` | counter | |
| `dismantle_request_reject_total{reason}` | counter | `reason` ∈ `queue_full`, `abort_at_admit` |
| `dismantle_step_tokens_total` | counter | Σ tokens emitted across all steps |
| `dismantle_step_duration_seconds` | histogram | per-step wall time |
| `dismantle_request_latency_seconds{phase}` | histogram | `phase` ∈ `prefill`, `decode_first_token`, `total` |
| `dismantle_weight_bw_pct` | gauge | estimated weight-bandwidth saturation; computed from per-step batch size and the known 1.6 GB per-token-prefill weight read |

---

## 5. Migration plan

| Phase | Ships | Gate |
|---|---|---|
| **0** | Today: HTTP layer is single-request only. CB skeleton lives at `crates/dismantle-serve/src/batch/*` but is unused. `forward_tokens_batched` is the dead-lever loop. | (current) |
| **M1** | α implementation behind `--continuous-batching` (default off). `--continuous-batching=false` path is byte-identical to today. | All §3-α parity gates green. Aggregate req/s at N=4 ≥ 1.8× single-request baseline (i.e. ≥ 40 req-tok/s vs. today's 22 dec_tps). |
| **M2** | α becomes default for `dismantle serve` (HTTP code paths through CB even at N=1). Add `--no-continuous-batching` escape hatch. Drop the non-CB SSE code path (`http/mod.rs:217-247`) after one release of soak. | M1 metrics show <0.5% per-token latency overhead at N=1 vs. the legacy path. |
| **M3** | β behind `--cb-prefill-interleave`. α remains the default. | β passes its mixed-prefill parity + soak; bench shows aggregate req/s at N=4 ≥ 3.5× single-request (i.e. measurable lift over α). |
| **M4** | β becomes default; α deleted. | M3 soak clean for 2 weeks. |
| **M5** | γ behind `--speculate exact-shared --continuous-batching`. | γ acceptance rate per-slot ≥ standalone spec-decode acceptance rate, no per-slot decode regression. |

The α→β migration is intentionally gated on **observed** throughput
need: if N=4 saturates the user's traffic, β's chunked prefill is
mostly a TTFT improvement, not a throughput improvement. Build it
when the data calls for it.

---

## 6. Dead-lever fence

**`Engine::forward_tokens_batched` (engine.rs:206) and its
`QwenDense` impl (qwen_dense.rs:404-410) are NOT the
continuous-batching forward path.** They are a per-token correctness
loop that:

1. Calls `self.forward_token(token, position)` N times against the
   single `self.kv` buffer.
2. Appends every (token, position) tuple into the same KV cache.
3. Returns N logit vectors.

Two failure modes if you reach for this from a CB scheduler:

- **KV contamination.** Tokens from request A and request B both
  land in `self.kv.keys[layer]`. Request A's next-token logits depend
  on request B's K/V vectors. Silent wrong output.
- **No bandwidth amortization.** Each `forward_token` reads every
  weight from scratch — 1.6 GB × N reads vs. the 1.6 GB × 1 read that
  the batched-prefill kernels already achieve. Defeats the entire
  point of CB.

The right entry point in the new code is `forward_step_cb` (§3-α).
The legacy `forward_tokens_batched` should be marked
`#[deprecated(note = "use forward_step_cb under continuous batching")]`
when α lands, and removed in M2.

Likewise: do not confuse `forward_tokens_batch_tcb`
(qwen_dense.rs:1571 main) — which is the **batched-prefill** path for
a single request — with the CB step. The TCB path requires contiguous
positions belonging to one request; CB requires per-row (slot_id,
position) and crosses request boundaries.

---

## 7. Bench targets

All numbers below assume Qwen-3B-Q4_K_M, M3 Pro 18 GB, prompt = 47
tokens (the canonical benchmark prompt), completion = 16 tokens,
greedy temperature=0, TCB+vocab-prune-32K+Q4K-LM-head defaults from
[[qwen_dense_metal_pipeline]].

### 7.1 Single-request baseline (already shipped)

| Metric | Value | Source |
|---|---|---|
| Prefill ms (47 tok, batched B=8) | ~1,136 ms | [[p3_batched_prefill_shipped]] |
| Decode dec_tps | ~22.4 | [[qwen_dense_metal_pipeline]] |
| Aggregate req/s @ N=1 | ~0.40 | derived: 1 / (1.136 + 16/22.4) s = 1/1.85 |
| Per-request p50 latency | ~1,850 ms | |
| Weight-BW saturation | ~45% | derived: 1.6 GB × 24 tok / 1.85 s / 150 GB/s |

### 7.2 CB-α targets (decode-only batching)

| N | Aggregate req/s | p50 per-req latency | p99 per-req latency | Weight-BW sat |
|---|---|---|---|---|
| 1 | ≥ 0.40 (no regression) | ≤ 1,900 ms | ≤ 2,200 ms | ~45% |
| 2 | ≥ 0.70 | ≤ 2,200 ms | ≤ 2,600 ms | ~70% |
| 4 | ≥ 1.20 | ≤ 3,000 ms | ≤ 3,800 ms | ~85% |
| 8 | ≥ 1.80 | ≤ 4,500 ms | ≤ 6,000 ms | ~90% |

Throughput model: at N=4 each request's prefill stalls all decode
slots for ~1,100 ms. Steady-state decode tps at N=4 is ≈ 4 × (1 /
per-step-ms) ≈ 4 / 50 ms = 80 tok/s aggregate. Over a 16-tok
completion that's ~200 ms decode per request, plus 1,100 ms shared
prefill amortized across the 4 = 275 ms each, ≈ 475 ms per request →
~2.1 req/s burst, ~1.2 req/s sustained including prefill overhead.

### 7.3 CB-β targets (prefill+decode interleave)

| N | Aggregate req/s | p50 per-req latency | TTFT p50 |
|---|---|---|---|
| 1 | ≥ 0.40 | ≤ 1,900 ms | ≤ 1,200 ms |
| 2 | ≥ 0.85 | ≤ 2,000 ms | ≤ 1,300 ms |
| 4 | ≥ 1.50 | ≤ 2,600 ms | ≤ 1,500 ms |
| 8 | ≥ 2.50 | ≤ 3,800 ms | ≤ 2,000 ms |

β's win over α at high N comes mainly from TTFT (no prefill
stalls decode) and from packing prefill chunks into otherwise
under-utilized decode steps when one request is finishing.

### 7.4 Bench harness wiring

Add to `crates/dismantle-bench/`:

- `bin/cb_throughput.rs` — drives M client connections each
  submitting the canonical 47-tok prompt back-to-back. Emits aggregate
  req/s, p50/p99 per-request latency, weight-BW % (computed from
  total weight bytes × steps / wall-time / 150 GB/s peak), per-step
  histogram.
- Reports written to `reports/cb_throughput_${date}.json` matching
  the existing bench-report shape so [[wall_clock_audit_pattern]]
  trainers can paired-test against the single-request baseline.
- Gate: paired-trial (N=10) vs. M=1 baseline must show ≥ +1.0
  aggregate req/s at M=4 to ship α as default.

---

## 8. Open questions for the implementer

1. **f16 KV cache for CB.** Q8_KV runtime is wired
   ([[q8_kv_runtime_landed]]) but the new MHA kernel reads f32 K/V.
   The CB shader can stay f32 for parity; a follow-up can switch to
   Q8 across all slots simultaneously. Don't try to do both in one
   change.
2. **Tokenizer concurrency.** `Tokenizer::encode` is called from
   `encode_prompt_for_batch` (`qwen_dense.rs:392`). If the CB loop
   batches admissions, encoding should happen *off* the engine
   thread — punt to a dedicated tokenizer task.
3. **`forward_tokens_batch_tcb` reuse.** α's `forward_step_cb` can
   share ~80% of the dispatch sequence with `forward_tokens_batch_tcb`
   (qwen_dense.rs:1571 main). Refactor: extract a private
   `dispatch_one_layer(b, slot_offsets, positions, ...)` helper used
   by both. This keeps the contiguous-position prefill path alive
   (for single-request) while the CB path takes the same shader
   dispatches with the new argbuf shape.
4. **Speculation mode + CB.** Today
   `SpeculateMode::ExactShared` operates per-request inside
   `qwen_dense.rs::generate`. Composing with CB means draft
   acceptance must happen per-slot inside `forward_step_cb`'s output
   handler. Defer to γ.

---

## 9. References

- `memory/p3_batched_prefill_shipped.md` — kernel groundwork.
- `memory/qwen_dense_metal_pipeline.md` — single-request baseline.
- `crates/dismantle-core/src/model/qwen_dense.rs` (main branch) — `forward_tokens_batch_tcb` is the source-of-truth for the batched dispatch sequence.
- `crates/dismantle-core/shaders/mha.metal` (main branch) — `mha_decode_f32_batched` is the source-of-truth kernel to fork for `mha_decode_f32_batched_cb`.
- vLLM paper (Kwon et al. 2023) — the original "continuous batching"
  formulation with paged attention. We intentionally start without
  paging; see §2.1.
