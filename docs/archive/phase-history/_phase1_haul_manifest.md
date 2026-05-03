# Phase 1 — Haul 1 manifest

**Authoritative scope for Haul 1.** The runner reads the
`## Gate runner manifest` fenced block below; the prose around it
is for the agent (and human readers) to understand what each item
should accomplish.

## Items

### Item 1 — G1.1 — Metal scaffold + rmsnorm round-trip

**Goal:** dispatch the existing real `rmsnorm` MSL kernel from
Rust through `MetalContext`, produce fp16 output, diff against
the CPU reference (`crates/dismantle-core/src/kernels/mod.rs::rmsnorm`)
on a fixed 4096-element input.

**Files touched (max):**
- `crates/dismantle-core/src/kernels/mod.rs` (host binding for the new dispatch)
- `crates/dismantle-core/src/metal/mod.rs` (any helper additions for buffer creation, dispatching)
- `tests/correctness/phase1_kernel_parity.rs` (the parity test itself)
- `_phase1_kernel_baseline.hashes` (regenerated with the new row)
- (NO change to `shaders/common.metal` — the rmsnorm kernel is already real MSL)

**Validation:**
- `cargo test --release --test phase1_kernel_parity test_rmsnorm_matches_cpu`
- Diff between Metal output and CPU output `< 1e-3` per element (fp16 atol)
- pre/post/verify evidence triples present

**Halt conditions:**
- Metal device construction fails (`MetalContext::new()` returns an error)
- Shader compile fails (any error from `newLibraryWithSource:`)
- Pipeline state creation fails for `rmsnorm`
- Output diff exceeds atol
- RSS sentinel fires (>5 GB) — should be ~50 MB; anything more means a leak

**Commit subject:** `phase 1: G1.1 Metal scaffold + rmsnorm round-trip`

### Item 2 — G1.2 — LM-head GEMV → Metal

**Goal:** replace the LM-head GEMV in `model::deepseek_v2::forward_token`'s
final `lm_head` branch with a Metal dispatch. This is the largest
single per-token CPU op (vocab 102400 × hidden 2048).

**Files touched (max):**
- `crates/dismantle-core/shaders/common.metal` (new `gemv_f16` kernel)
- `crates/dismantle-core/src/kernels/mod.rs` (host binding)
- `crates/dismantle-core/src/model/deepseek_v2.rs` (call site only)
- `tests/correctness/phase1_kernel_parity.rs` (extend with gemv_f16 parity)
- `_phase1_kernel_baseline.hashes` (new gemv_f16 row)
- `_phase1_token_baseline.hashes` (re-capture; should match prior bytes)

**Validation:**
- `cargo test --release --test phase1_kernel_parity test_gemv_f16_matches_cpu`
- Token regression test from `phase1_token_regression.rs` — first 3 token IDs match locked baseline
- pre/post/verify triples present
- Memory guard returns 0 before integration smoke runs

**Halt conditions:**
- Parity diff > atol on synthetic test
- Token IDs differ from baseline (deterministic regression)
- Generation produces empty stdout
- RSS > 5 GB during integration smoke
- Memory guard pressure persists for >2.5 min

**Commit subject:** `phase 1: G1.2 LM-head GEMV → Metal`

### Item 3 — G1.3 — Attention `o_proj` GEMV → Metal

**Goal:** replace the `o_proj` GEMV at the tail of MLA attention
with a Metal dispatch. Per-token op of size hidden 2048 ×
(n_heads × v_head_dim) = 2048 × 2048 = 4M MACs.

**Files touched (max):**
- `crates/dismantle-core/shaders/attn.metal` (new `gemv_f32` kernel for `o_proj`)
- `crates/dismantle-core/src/kernels/mod.rs` (host binding)
- `crates/dismantle-core/src/model/deepseek_v2.rs` (the call in `attention()`)
- `_phase1_kernel_baseline.hashes` (new row for o_proj kernel)

**Validation:**
- Synthetic parity test on the o_proj-shaped GEMV
- Token regression: first 3 token IDs match baseline
- Smoke generate produces non-empty UTF-8

**Halt conditions:** same as G1.2.

**Commit subject:** `phase 1: G1.3 attention o_proj → Metal`

### Item 4 — G1.4 — `ffn_gate_inp` GEMV → Metal

**Goal:** replace the gate-logit GEMV in the MoE FFN. Tiny GEMV
(64 × 2048 = 131k MACs) but proves MoE-shaped weight access.

**Files touched (max):**
- `crates/dismantle-core/shaders/moe.metal` (new `gemv_f32` for gate logits)
- `crates/dismantle-core/src/kernels/mod.rs` (host binding)
- `crates/dismantle-core/src/model/deepseek_v2.rs` (the call in `ffn()` MoE branch)
- `_phase1_kernel_baseline.hashes` (new row for gate-logit kernel)

**Validation:**
- Synthetic parity test
- Token regression: first 3 token IDs match baseline
- **Routing decisions match** for first 3 tokens (top-K experts identical) — implicit in token-ID match because greedy temp=0 and top_k_routed selection are deterministic given same gate logits

**Halt conditions:**
- Routing changes (different experts selected) → token IDs differ → halt
- Other halt conditions same as G1.2/G1.3

**Commit subject:** `phase 1: G1.4 ffn_gate_inp GEMV → Metal`

## Cross-gate invariants (checked by run-gates.sh between items)

- Pre-existing `cargo test --workspace --lib` 15 tests still pass
- Smoke generation produces non-empty UTF-8
- Existing entries in `_phase1_kernel_baseline.hashes` still hash the same
  (regression detection)

## Halt budget

Per spec `§ Locked decision rules § 5`:

- **G1.1 (scaffold): 1 halt = end haul.**
- **G1.2 / G1.3 / G1.4 (GEMV ports): 2 halts in this group = end haul.**
  First port halt: continue to next independent item.

## Dependency map

```
G1.1 ──┐
       ├── G1.2 (independent of G1.3, G1.4)
       ├── G1.3 (independent of G1.2, G1.4)
       └── G1.4 (independent of G1.2, G1.3)
```

If G1.1 halts, items 2–4 do not run. If G1.2 halts but G1.1 passed,
G1.3 and G1.4 still attempt.

## What this haul does NOT do

- Wedge 2 (fused Q4_K_M dequant inside FMA loop). Naive Metal GEMV
  reads pre-dequanted fp32 from a scratch buffer. The fusion is a
  later haul.
- Wedge 1 (single-launch fused MoE kernel). Still per-expert
  dispatch.
- Attention beyond `o_proj` (no flash-attention, no MLA decompress
  kernel).
- GPU sampling. CPU sampling stays.
- Any change to quant code, tokenizer, engine API, or Cargo deps.
- Any push to remote.

## Honest scope caveat

After Haul 1 lands all 4 items, dismantle's decode tok/s might
land anywhere between **3 and 50 tok/s** depending on:
- How well the naive Metal GEMV uses simdgroup matrix ops
- How much the lazy-dequant scratch-buffer round-trip costs
- How aggressively macOS's resource arbiter throttles us under
  co-existence with slm training

Hitting llama.cpp's ~48 tok/s would be lucky. The real wins land in
Haul 2 (wedge 2: fused dequant) and beyond. Phase 1's job is *not*
to beat llama.cpp — it's to get dismantle off the CPU baseline so
the wedges have something real to optimize.

## Gate runner manifest

The runner (`tools/haul/run-gates.sh`) parses the fenced block
below. Format: `<gate-id> <validator-kind> <validator-args...>`,
one per line. Comments allowed (lines starting with `#`).

```
# Phase 1 haul 1 — 4 gates, hybrid halt budget per spec
G1.1 cargo-test test_rmsnorm_matches_cpu
G1.2 cargo-test test_gemv_f16_matches_cpu
G1.3 cargo-test test_gemv_f32_attn_matches_cpu
G1.4 cargo-test test_gemv_f32_moe_matches_cpu
```

The corresponding integration-smoke gates (token-baseline
regression) are not in this fenced block because they're
conditional on `coexist.sh probe` returning 0 — the agent runs them
opportunistically after each `cargo-test` gate passes, not as
mandatory items. If memory pressure prevents the integration smoke,
the gate is recorded as `PASS-PARITY-ONLY` per spec § CE-4.

## Self-contained scope check

A fresh agent reading just this manifest + `_phase1-spec.md` +
`CLAUDE.md` should be able to execute Haul 1 without any prior
context from this conversation. Test: ask "what do I do?" and the
docs answer. If they don't, the docs need amendment, not the haul.
