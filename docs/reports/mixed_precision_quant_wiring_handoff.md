# Mixed-precision quant — wiring handoff

**Status:** design doc + foundation landed. The data layer (tier-map
loader, Q4_K/Q6_K quantize inverses, MixedQuantStore) ships in this
session; the dispatcher integration is the explicit follow-on wedge —
see §10 below for the precise diff list.

**Projected gain (corrected, see §11):** **+1 to +3 dec_tps realistic**
for the down-projection-only tier maps. The original +5-10 estimate
assumed full-tensor (gate+up+down) re-quantization which requires new
fused-gate+up shaders for Q6/Q8 that don't exist in tree.

**Author handoff target:** matches the rigor of §3 of
`reports/path_to_50_oneshot_prompt.md`. The foundation work has landed
(11 new lib tests + 1 integration smoke); the next session ships the
dispatcher patches + parity test.

---

## 1. Context — what the lever is and isn't

The corpus calibration produced
`artifacts/calibration/analysis/per_layer_residual_stats.json` and
`summary.md`. The tiered map derived from those stats is:

| layers | abs_max range | mean_abs | recommended tier |
|---|---|---|---|
| **0–3** | 2.05 – 23.83 | 0.05 – 0.08 | **q4_K** (4.5 bpw) |
| **4–24** | 1138 – 1181 | 0.16 – 1.34 | **q8_0** (8.5 bpw) |
| **25–26** | 644.5 – 845.0 | 1.46 – 1.60 | **q6_K** (6.5 bpw) |

The headline savings come from re-quantizing MoE expert weights (which
dominate decode-time per `per_kernel_time_breakdown.md`: 50.5% of decode
time is MoE GEMMs). Attention projections and norm-fused GEMVs are
secondary targets.

**Important nuance:** the analysis is on the **residual activations**
between layers, not on the **weights themselves**. Activation magnitude is
a proxy for "how much precision does this layer need before its
contribution starts corrupting the residual stream." A layer whose
residual `abs_max < 24` can tolerate ~1 e-1 weight quant error without
moving the residual much; a layer at abs_max~1180 cannot.

Per `v110_path30_findings.md`, the f16-residual experiment broke at
~27 layers because per-layer rounding compounded. Mixed-precision quant
is the inverse: instead of squeezing precision out of the residual
stream, we squeeze precision out of the **weights that produce that
residual**, layer by layer, sized to what each layer's residual can absorb.

### Out of scope (covered by other levers or workstreams)

- LM head precision → vocab-prune (lever 1)
- KV cache precision → already at Q8 (uniform, per
  [[q8-kv-landed]]); per-layer differential was killed by the
  routing-balance analysis
- Activation quantization → weight quant only here
- MLA absorbed-V → [[mla-phase4-queued]], separate workstream
- Embedding table → input embeddings are tiny; quant error compounds
  through every forward pass

---

## 2. What you have on disk and in code

### 2.1 Calibration artifacts

- `artifacts/calibration/analysis/per_layer_residual_stats.json`
  — per-layer `mean_abs`, `abs_p99`, `p99.9`, `abs_max`, suggested
  `int8_scale` for all 27 layers
- `artifacts/calibration/analysis/summary.md`
  — human digest with the tiered map already pre-derived

### 2.2 Existing Rust modules

- `crates/dismantle-core/src/quant/mod.rs` (776 LOC)
  - `dequant_into(dtype, bytes, &mut [f32])` — generic dequant
    dispatcher for F16/BF16/F32, Q4_0/1, Q5_0/1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K
  - `dequant_to_f16(info, bytes) -> Vec<f16>` and `dequant_to_f32`
  - `quantize_q8_0(src: &[f32], dst: &mut [u8])` — the only inverse
    currently in tree; round-trip lossless to one ULP per block
  - **No `quantize_q4_K`, `quantize_q6_K`, `quantize_q5_0` exist yet.**
    This is one of the bigger Phase-B items below.
- `crates/dismantle-core/src/model/deepseek_v2.rs` lines 476–616
  — `impl Engine for DeepSeekV2 { fn load() }`, the only place that
  reads weights from GGUF and materializes them. Per-layer load loop at
  lines 501–600 is the natural insertion point.
- `crates/dismantle-core/src/engine.rs` lines 8–46 — `EngineConfig`,
  the runtime config struct (note: the prompt says "Profile" in
  `profile.rs`, but `profile.rs` only defines `KernelProfile`/`KernelVariant`
  for kernel scheduling. The new tier-map option belongs on
  `EngineConfig` and flows in through `Engine::load(weights, config)`,
  not in the kernel profile).
- `crates/dismantle-core/src/kernels/mod.rs` lines 2622–2684 —
  `encode_batched_gemv_indexed` with the suffix matcher
  (`ends_with("_v2t") / _v2 / _v2s`) that bit yesterday (per
  [[feedback_kernel_parity_gate]]). Mixed-precision dispatch must not
  regress this.

### 2.3 Existing Metal kernels per dtype (V2-Lite hot path)

| tensor group | current dtype | kernel(s) |
|---|---|---|
| ffn_gate_exps, ffn_up_exps (routed) | **Q4_K** | `moe_batched_gemm_q4_indexed_v2t_gu` (fused gate+up) |
| ffn_down_exps (routed) | **Q5_0** | `moe_batched_gemm_q5_0_indexed_v2t` |
| ffn_gate_shexp, ffn_up_shexp (shared) | **Q4_K** | same Q4_K kernels |
| ffn_down_shexp (shared) | **Q6_K** | `moe_batched_gemm_q6_k_indexed_v2t` |
| attn_q (q_proj fallback) | f16 | `gemv_f16` family + rmsnorm-fused |
| attn_q_a / q_b / kv_a / kv_b / o_proj | f16 (eager) | `gemv_f32_attn` family |
| token_embd | f16 | embed lookup |
| output (lm_head) | f16 | `gemv_f16_argmax_metal_pinned` |

The Q4_K, Q5_0, Q6_K, Q8_0 kernels all exist in `shaders/` and are
already wired through the `_v2t` indexed dispatcher. **The mixed-precision
implementation reuses these kernels** — the lever is choosing which
dtype to materialize for each layer's expert weights, not writing new
arithmetic.

---

## 3. The design — five decisions

### 3.1 Where the tier map lives

**Decision:** a separate JSON file referenced from `EngineConfig`. NOT
inlined in profile.rs.

Add to `crates/dismantle-core/src/engine.rs`:

```rust
pub struct EngineConfig {
    // ... existing fields ...

    /// Path to a per-layer quant tier map JSON (see `quant_tier_map` module).
    /// When set, MoE expert weights are dequantized from the GGUF source and
    /// re-quantized per-layer to the bit-width specified in the map. None ⇒
    /// keep the GGUF's native dtype (current behavior).
    pub quant_tier_map_path: Option<std::path::PathBuf>,
}
```

Default `None` in `EngineConfig::default()`. The CLI flag
`--quant-tier-map <path>` (in `dismantle-bin`) sets it.

**Tier map JSON schema** (`artifacts/calibration/tier_maps/v2_lite_default.json`,
to be hand-written from `summary.md`):

```json
{
  "schema_version": 1,
  "model_arch": "deepseek2",
  "model_id": "deepseek-v2-lite-chat",
  "n_layers": 27,
  "comment": "Derived from per_layer_residual_stats.json on 2026-05-21. Tiers chosen by abs_max bucket: <24 → q4_K, 24–1200 → q8_0, in-between (>500 <1000) → q6_K.",
  "layers": [
    { "layer": 0,  "gate_up": "q4_K", "down": "q4_K" },
    { "layer": 1,  "gate_up": "q4_K", "down": "q4_K" },
    { "layer": 2,  "gate_up": "q4_K", "down": "q4_K" },
    { "layer": 3,  "gate_up": "q4_K", "down": "q4_K" },
    { "layer": 4,  "gate_up": "q8_0", "down": "q8_0" },
    /* layers 5–24 all "q8_0"/"q8_0" */
    { "layer": 25, "gate_up": "q6_K", "down": "q6_K" },
    { "layer": 26, "gate_up": "q6_K", "down": "q6_K" }
  ]
}
```

Why split `gate_up` from `down`: V2-Lite already ships these as
different dtypes (Q4_K for gate+up, Q5_0/Q6_K for down). Keeping the
split lets us tune them independently (down has its own outlier
distribution because of SwiGLU compression into 2048 channels).

Allowed dtypes initially: `q4_K`, `q6_K`, `q8_0`. We'll add `q5_0`
later if useful — `Q5_0` quantize is in tree as the next implementation
step. `q4_K` is the most-aggressive; `q8_0` the most-conservative;
`q6_K` the middle. Layers absent from the map fall back to GGUF native
dtype.

A new module `crates/dismantle-core/src/quant_tier_map.rs` (~80 LOC)
mirrors `vocab_prune.rs` in spirit: `TierMap::load`, `validate(arch,
n_layers)`, `tier_for(layer_id, group: GroupKind) -> Option<GgmlType>`.

### 3.2 What to actually quantize

**Decision:** MoE expert weights only — `ffn_gate_exps`, `ffn_up_exps`,
`ffn_down_exps` per layer. Attention projections stay at their current
f16 dtype.

Why:
- MoE GEMMs are 50.5% of decode-time (per `per_kernel_time_breakdown.md`);
  attention projections are 2.4%. Attention quant has 20× less leverage.
- The residual-activation calibration is **at the residual stream**
  between layers — it bounds how much TOTAL per-layer error the residual
  can absorb. MoE expert outputs are the dominant contributor to the
  residual delta (much larger than the attention residual contribution,
  see `summary.md`).
- Attention V outliers blow up quant error far more than MLP per the
  one-shot prompt §4.3, and the existing f16 path is already lighter
  than re-quantizing.
- Dense layer 0 is unaffected — it's not MoE.

Shared experts (`ffn_*_shexp`) use the same tier as routed for that
layer. They share the residual stream and the same per-token weight, so
no reason to diverge.

### 3.3 GGUF reality — Path A (runtime re-quant) recommended

**Decision:** Path A — load the source GGUF, dequantize the expert
tensors to f32 in scratch, re-quantize per the tier map into a heap
buffer, drop the f32 scratch. The re-quantized bytes live on the model
struct in a `MixedQuantStore` and are referenced via `TensorRef` for
GPU dispatch (mirroring the existing `weights_mmap_buf` lookup pattern,
but pointing at heap instead of the mmap).

**Why Path A over Path B:**
- Path A: zero shipped artifacts, fast to iterate. Cost: ~30–60 s extra
  startup, peak memory + ~4 GB (the dequantized scratch). Permitted by
  V2-Lite's footprint (9.7 GB GGUF on a ≥18 GB box).
- Path B (pre-baked mixed GGUF): saves startup time, but requires
  shipping a per-tier-map GGUF and a Python rebuild tool. Premature
  until we know what tier map actually wins.

**Code shape:** new `MixedQuantStore` struct holds the re-quantized
expert bytes plus a parallel set of `MoEFusedTensors` whose `gate_w` /
`up_w` / `down_w` `TensorRef` point at offsets into a single contiguous
`Vec<u8>` (so the existing Metal mmap-no-copy path works — we upload
the store as a separate `PinnedBuffer` and direct expert dispatches at
the store's base offset instead of the GGUF mmap's base when the layer
has a tier override).

Per-layer dispatch then needs to know which buffer base to use. The
cleanest way: extend `MoEFusedTensors` with an `enum WeightSource {
GgufMmap, MixedQuantStore }`, and pass the right `&PinnedBuffer` into
the encode functions. The kernel itself doesn't care; it reads bytes
through an indexed-base offset.

**Memory accounting:** Re-quantizing layer 4 down from Q5_0 to Q8_0
nearly doubles its size (Q5_0 = 5.5 bpw, Q8_0 = 8.5 bpw); 21 layers
× ~10 MB delta = ~210 MB peak overhead. Re-quantizing layers 0–3 down
to Q4_K shrinks them by ~12%; layers 25–26 to Q6_K shrinks ~25%. Net
delta is small but **positive** (more memory than baseline). Update
`EngineConfig::memory_limit_mb` accounting accordingly — already
exists per profile.rs:23–26.

### 3.4 Dispatch — kernel name routing

**Decision:** dispatcher reads the tier of the current layer from the
loaded `TierMap` and routes to the kernel name for that dtype. The
existing `_v2t` suffix matcher is unaffected because the kernel name
already encodes the dtype (`moe_batched_gemm_q4_indexed_v2t` vs
`moe_batched_gemm_q5_0_indexed_v2t` vs `moe_batched_gemm_q6_k_indexed_v2t`).

Critical: per [[feedback_kernel_parity_gate]], the suffix matcher in
`kernels/mod.rs` lines 2639, 4065, 4119, 4124 only recognizes
`ends_with("_v2t") / _v2 / _v2s`. All three of our candidate kernels
already end in `_v2t`. **No new kernel names needed in Phase B-1.** If
we later add a `_v2t_v3` variant for any of these the suffix matcher
must be patched in the same PR.

Per-layer routing: in `FfnMoeSetup` (deepseek_v2.rs:286–311), the
`q4k_indexed_kernel(q4k_schedule)` already returns the kernel name
string. Add a sibling that consults `self.tier_map` first:

```rust
fn moe_kernel_for_layer(&self, layer_id: usize, group: GroupKind) -> &'static str {
    let tier = self.tier_map
        .as_ref()
        .and_then(|m| m.tier_for(layer_id, group))
        .unwrap_or(self.native_dtype_for(layer_id, group));
    match (tier, group) {
        (GgmlType::Q4_K, GroupKind::GateUp) => "moe_batched_gemm_q4_indexed_v2t",
        (GgmlType::Q8_0, GroupKind::GateUp) => "moe_batched_gemm_q8_0_indexed_v2t",
        (GgmlType::Q6_K, GroupKind::Down)   => "moe_batched_gemm_q6_k_indexed_v2t",
        // ... etc.
        _ => /* error: tier × group not supported */
    }
}
```

(The `_v2t` GateUp Q8_0 variant exists per the shaders directory; this
needs verification at Phase-B start. If missing, that's a separate
kernel implementation task — flag and stop, don't quietly fall back to
non-`_v2t`.)

### 3.5 Acceptance gates

Per [[feedback_kernel_parity_gate]], the strict-3-token bit-identical
gate is the right gate for kernel work that preserves per-lane math.
Mixed-precision quant **changes per-lane math** (different rounding per
layer), so bit-identical 3-token greedy is **not the right gate** here.
Instead:

1. **Per-layer perplexity-on-corpus delta vs uniform-quant baseline**
   (informational, not gating). Run on a held-out 500-seq subset of the
   corpus. Expected: per-layer ppl rise < 0.5% on q4_K layers, < 0.1%
   on q8_0 layers (these layers move in opposite directions: q4 hurts
   layers 0–3 less than q4 hurting layers 25–26 would).
2. **No token divergence in first 256 tokens** of a held-out test prompt
   (fixed seed). This is the gating parity test.
3. **End-to-end dec_tps delta vs the post-vocab-prune baseline**: target
   ≥ **+5 dec_tps**.

Why not bit-identical: every layer's weights are re-quantized to a new
dtype, so per-lane math changes. The 256-token greedy-divergence test
is the strongest correctness check that's compatible with the math
change. If 256-token greedy diverges, the mixed-precision tier map is
too aggressive somewhere — bisect by tightening one tier at a time
(start with the lowest-abs_max layer that's at q4_K and bump to q6_K).

---

## 4. Phase-B implementation tasks (in order)

### Task B.1 — `quant_tier_map` module

New file `crates/dismantle-core/src/quant_tier_map.rs` (~80 LOC).
Mirror `vocab_prune.rs`:

```rust
pub struct TierMap {
    schema_version: u32,
    model_arch: String,
    n_layers: usize,
    layers: Vec<TierEntry>,         // indexed by layer_id
}

pub struct TierEntry {
    pub layer: usize,
    pub gate_up: Option<GgmlType>,
    pub down: Option<GgmlType>,
}

pub enum GroupKind { GateUp, Down }

impl TierMap {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> { ... }
    pub fn validate(&self, arch: &str, n_layers: usize) -> Result<()> { ... }
    pub fn tier_for(&self, layer_id: usize, group: GroupKind) -> Option<GgmlType> { ... }
}
```

Tests: missing-file, schema-mismatch, wrong-arch, out-of-range-layer,
absent-layer-falls-through. Pattern-match on `vocab_prune.rs` tests
(11 tests, 260 LOC).

Add `pub mod quant_tier_map;` to `crates/dismantle-core/src/lib.rs`.

### Task B.2 — quantize inverses for Q4_K and Q6_K

`crates/dismantle-core/src/quant/mod.rs` already has `quantize_q8_0`.
Add:

- `quantize_q4_k(src: &[f32], dst: &mut [u8]) -> Result<()>`
- `quantize_q6_k(src: &[f32], dst: &mut [u8]) -> Result<()>`

These are the inverses of the existing `dequant_q4_k` and `dequant_q6_k`.
**Reference implementation:** llama.cpp's `ggml/src/ggml-quants.c`
functions `quantize_row_q4_K_ref` and `quantize_row_q6_K_ref`. Port
them line-for-line; round-trip must be parity-tested.

Round-trip parity test (new test file
`crates/dismantle-core/src/quant/tests.rs` or inline):

```rust
#[test]
fn q4_k_round_trip_within_block_max_error() {
    let src: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    let mut blob = vec![0u8; q4_k_block_bytes(src.len())];
    quantize_q4_k(&src, &mut blob).unwrap();
    let mut back = vec![0.0f32; src.len()];
    dequant_into(GgmlType::Q4_K, &blob, &mut back).unwrap();
    let max_err = src.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 0.1, "q4_K round-trip max abs err {max_err} > 0.1");
}
```

Estimated ~80 LOC for `quantize_q4_k` + ~80 LOC for `quantize_q6_k` +
~40 LOC of tests. If either function ends up much larger, port from
llama.cpp's reference is the safer path than rewriting from spec.

### Task B.3 — `MixedQuantStore` and per-layer re-quant

New struct in `crates/dismantle-core/src/model/deepseek_v2.rs`
(~120 LOC):

```rust
pub struct MixedQuantStore {
    /// Re-quantized bytes laid out contiguously per layer. The layout
    /// matches the GGUF MoE 3D tensor convention: outer dim = expert id.
    blob: Vec<u8>,
    /// pinned GPU buffer over `blob` (mac only).
    pub buf: Option<PinnedBuffer>,
    /// Per-layer (gate_w, up_w, down_w) refs into `blob`.
    pub layers: Vec<MoEFusedTensors>,
    /// What tier was used for each layer's (gate_up, down).
    pub effective_tiers: Vec<(GgmlType, GgmlType)>,
}
```

Populated in `Engine::load` just after the per-layer MoE tensor refs
are gathered (~line 562), guarded on `config.quant_tier_map_path.is_some()`:

```rust
let tier_map = config.quant_tier_map_path.as_ref()
    .map(|p| TierMap::load(p))
    .transpose()?;
if let Some(ref tm) = tier_map {
    tm.validate("deepseek2", cfg.n_layers)?;
}

let mixed_store = tier_map.as_ref().map(|tm| {
    MixedQuantStore::build_from_gguf(&gguf, tm, &layers_meta)
}).transpose()?;
```

`build_from_gguf` iterates layers, dequants each MoE expert tensor that
has a tier override into f32 scratch, re-quants into the store's blob,
records the new offset. Reuse `dequant_to_f32` + new
`quantize_q4_k` / `quantize_q6_k` / `quantize_q8_0`.

After construction (mac only), upload the entire blob as a
`PinnedBuffer` and store on `MixedQuantStore::buf`.

Hold `mixed_store: Option<MixedQuantStore>` on `DeepSeekV2` (next to
the existing `weights_mmap_buf` field at ~line 155).

### Task B.4 — wire the dispatcher

Wherever the existing dispatcher reads from `weights_mmap_buf` for MoE
expert tensors (search: `weights_mmap_buf` in deepseek_v2.rs, ~30 hits),
add a one-line check: if the current layer has a tier override, swap to
`mixed_store.buf` and use the store's per-layer `TensorRef` offsets
instead.

Two main hot paths to touch:
- `encode_moe_shared_only_indexed_tcb_with_scratch` (kernels/mod.rs:4364)
- the layer-loop dispatch in `forward_token_final_norm_maybe_read`
  (deepseek_v2.rs:2455+)

The kernel name is selected per Task §3.4 above by
`moe_kernel_for_layer(layer_id, group)`. The buffer base + offset come
from the store. **Do not** touch the non-MoE dispatchers (attention,
norms, embed, LM head) — those keep their existing paths.

### Task B.5 — parity test

New file `crates/dismantle-core/tests/mixed_precision_parity.rs`
(~120 LOC). Pattern: read whatever fixture `q8_kv_parity.rs` uses;
generate 256 tokens with `quant_tier_map_path = None`; generate again
with `Some("artifacts/calibration/tier_maps/v2_lite_default.json")`;
compare token-by-token. **Must match for all 256 tokens.**

Skip on non-Mac in CI. If a divergence at any position, the test should
print the first-divergence position + the original vs pruned token ids
(human-readable) so the bisect over tier overrides is easy.

### Task B.6 — kernel_bench entries

`crates/dismantle-core/src/kernel_bench.rs` already exercises individual
kernels with synthetic shapes. Add three entries representing the
re-quantized layer shapes:

- `("moe_q4_K_gate_up_layer3", ...)` — q4_K at layer 3 (smallest tier-down)
- `("moe_q8_0_gate_up_layer13", ...)` — q8_0 at layer 13 (largest tier-up)
- `("moe_q6_K_gate_up_layer25", ...)` — q6_K at layer 25

These run via the existing harness and give per-tier microbench numbers.
**Expected pattern:** q4 < q8 < q6 GEMV time per layer; the win is the
weighted average across layers (most layers are q8 — that's the budget
sink — but the q4 layers fully amortize startup overhead etc.).

### Task B.7 — end-to-end bench

Run `tools/bench/quick_bench.sh` or `tools/bench/clean_bench.sh` (per
[[feedback_bench_with_claude_open]]: paired delta is fine with Claude
open; absolute number for the headline needs Claude quit) with:

- baseline: `--vocab-prune-path ...vocab_whitelist_995.json` (lever 1 already in)
- under test: also `--quant-tier-map .../v2_lite_default.json`

Target: ≥ **+5 dec_tps** delta over the lever-1-only baseline. If
≥ +3 but < +5, ship anyway and document the gap — total of the three
levers can still hit 45 with eagle5 carrying. If < +3, bisect: turn
off tier overrides one layer at a time and find the layer that's
costing throughput (likely a Q8_0 layer where the increased bytes hurt
bandwidth more than the math precision helps).

---

## 5. File-level diff list (forecast)

| file | LOC | nature |
|---|---|---|
| `crates/dismantle-core/src/quant_tier_map.rs` | ~120 | new |
| `crates/dismantle-core/src/quant/mod.rs` | ~200 | add q4_K + q6_K quantize inverses + tests |
| `crates/dismantle-core/src/engine.rs` | ~3 | add `quant_tier_map_path` to `EngineConfig` |
| `crates/dismantle-core/src/model/deepseek_v2.rs` | ~180 | `MixedQuantStore`, load wiring, dispatcher swap |
| `crates/dismantle-core/src/kernels/mod.rs` | ~10 | per-layer kernel name selection (if not done inline) |
| `crates/dismantle-core/src/lib.rs` | ~1 | `pub mod quant_tier_map;` |
| `crates/dismantle-core/tests/mixed_precision_parity.rs` | ~120 | new |
| `crates/dismantle-core/src/kernel_bench.rs` | ~15 | three new bench entries |
| `crates/dismantle-bin/src/main.rs` (or wherever CLI flags live) | ~5 | `--quant-tier-map` flag |
| `artifacts/calibration/tier_maps/v2_lite_default.json` | — | new artifact, ~30 lines of JSON |

Total ~650 LOC. **Estimated impl time:** 2–3 sessions of focused work.

---

## 6. Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Q8_0 `_v2t` `gate_up` kernel doesn't exist | medium | Verify at Phase-B start. If missing, write it (mirror the Q4_K_v2t gate_up pattern) before any other Phase-B work — or fall back to per-tensor splitting (gate_up at native, down at tier-mapped). |
| Round-trip Q4_K quantize differs from llama.cpp ref → corrupt weights | medium | Port reference line-for-line; add a parity test that round-trips the V2-Lite GGUF's first layer through dequant → re-quant → dequant and verifies element-wise abs error < 0.1. |
| Re-quant startup time too slow on every load | low | Measure first; if > 60 s, cache the `MixedQuantStore` blob to disk keyed by `(tier_map_hash, gguf_hash)` and reload from cache. Out of scope for initial implementation. |
| 256-token greedy diverges → tier too aggressive | medium | Bisect: tighten one layer at a time, starting at the boundary (layers 3↔4 and 24↔25). |
| Memory blow-up from upsizing layers 4–24 to Q8_0 | low | The math: Q5_0 → Q8_0 is +3 bpw across the down weight (~6 MB delta per layer × 21 layers = ~125 MB). V2-Lite total ~9.7 GB; well within 18 GB budget. Update memory accounting. |
| Hidden bug in the suffix matcher silently dispatching wrong TG geometry | high if not checked | When `moe_batched_gemm_q8_0_indexed_v2t` is used in the new code path, *verify* `ends_with("_v2t")` matches — explicit unit test on `is_v2t` for every new kernel name introduced. |

---

## 7. What NOT to do (recap)

- Don't quantize the attention V projection more aggressively until measured
- Don't touch MLA absorbed-V (separate workstream)
- Don't quantize the LM head (vocab-prune handles that)
- Don't re-quant the embedding table — quant error compounds
- Don't combine this lever with eagle5 v2 in one PR — sequential lands cleanly
- Don't gate Phase B on bit-identical 3-token parity — that gate is for
  per-lane-math-preserving kernel work, not for math-changing re-quant
- Don't add new `_v2t_vN` kernel name variants without patching the
  suffix matcher in the same PR

---

## 8. Done condition

- `cargo test -p dismantle-core` green, including
  `mixed_precision_parity` test
- 256-token greedy parity passes on the held-out test prompt
- Microbench entries published
- End-to-end clean-bench shows ≥ **+5 dec_tps** over the
  lever-1-only baseline
- `reports/mixed_precision_quant_landed.md` written summarizing what
  shipped and what the actual gain was per layer-group

---

## 9. Related memory

- [[corpus-complete-analysis-landed]]
- [[per-kernel-time-breakdown-2026-05-20]]
- [[v110-path30-findings]] (MoE GEMMs dominate)
- [[feedback-kernel-parity-gate]] (suffix matcher trap)
- [[q8-kv-landed]] (KV path uniformly Q8 — unrelated, kept here for context)

---

## 10. What landed in this session (vs. what's deferred)

### Landed

| file | LOC | what |
|---|---|---|
| `crates/dismantle-core/src/quant_tier_map.rs` | 290 | new module; `TierMap::load/validate/tier_for`. 7 lib tests. |
| `crates/dismantle-core/src/quant/mod.rs` | +200 | `quantize_q4_k` + `quantize_q6_k` + `encode_q_k_scale_min` + 5 round-trip tests. |
| `crates/dismantle-core/src/mixed_quant_store.rs` | 270 | new module; `MixedQuantStore::build` re-quantizes per the tier map. 3 lib tests + 1 integration smoke against live V2-Lite GGUF. |
| `crates/dismantle-core/src/engine.rs` | +6 | `EngineConfig::quant_tier_map_path` field + default. |
| `crates/dismantle/src/main.rs` | +10 | `--quant-tier-map` CLI flag. |
| `artifacts/calibration/tier_maps/v2_lite_default.json` | — | conservative (down-only) tier map from the residual stats. |
| `artifacts/calibration/tier_maps/v2_lite_aggressive_down_q4.json` | — | aggressive down-only Q4_K everywhere — for speed-first experiments. |
| `crates/dismantle-core/tests/mixed_quant_store_build.rs` | 80 | Mac-only integration smoke: build store against V2-Lite, spot-check Q8/Q6 tier hits. |

The CLI flag plumbing means a user can already pass `--quant-tier-map
path/to/map.json` to `dismantle generate`. The path is loaded and parsed
at engine init — but it is currently NOT consumed because the dispatcher
patches below haven't landed. The intent is to land them in a follow-up
wedge.

### Deferred — the dispatcher wedge

Phase B-3 (the part that actually feeds re-quantized bytes into the MoE
kernels) is intentionally not in this session for two reasons:

1. **Scope.** The dispatcher signature change touches `kernels/mod.rs`
   plus 3 call sites in `model/deepseek_v2.rs` (1562, 2288, 2689). Each
   site needs: per-layer kernel-name selection, per-layer buffer base
   selection, per-layer offset translation. Easy to do wrong; the
   parity test below catches it but adds another loop.
2. **Shader gap.** Only the routed-down kernel can mix dtypes today
   (Q4_K/Q5_0/Q6_K/Q8_0 all have `_v2t` variants). gate+up Q4_K fused
   (`_v2t_gu`) has no Q6_K/Q8_0 sibling, so the most impactful tier
   changes (the leading-layers Q4_K aggressive map) can only land for
   down-projection in v1. Full mixed-precision needs new shaders.

### The follow-up wedge in 10 bullet points

1. In `crates/dismantle-core/src/model/deepseek_v2.rs` (struct around
   line 184), add:
   ```rust
   pub tier_map: Option<crate::quant_tier_map::TierMap>,
   pub mixed_quant_store: Option<crate::mixed_quant_store::MixedQuantStore>,
   pub mixed_quant_buf: Option<PinnedBuffer>,
   ```
   and wire load just after `pruned_vocab` (similar pattern).
2. In `Engine::load` (~line 480), after the per-layer MoE tensor refs
   are gathered, if `config.quant_tier_map_path` is `Some`: load
   `TierMap`, validate, build `MixedQuantStore::build`, upload its
   `blob()` to a `PinnedBuffer`.
3. In `crates/dismantle-core/src/kernels/mod.rs`, extend
   `encode_moe_block_batched_indexed_tcb_with_scratch` and
   `encode_moe_shared_only_indexed_tcb_with_scratch` with optional
   `routed_down_buf: Option<&PinnedBuffer>` and
   `shared_down_buf: Option<&PinnedBuffer>` parameters. Inside the
   function: `let down_model = routed_down_buf.unwrap_or(model_buf);`
   and pass that to the down `encode_batched_gemv_indexed_tcb` calls.
4. Patch `FfnMoeSetup` (or introduce a sibling struct) to carry the
   per-layer effective down dtype + offset + buffer-source flag.
5. Update kernel-name selector to pick the dtype-correct down kernel:
   ```rust
   fn down_kernel_for(dtype: GgmlType) -> &'static str {
       match dtype {
           GgmlType::Q4_K => "moe_batched_gemm_q4_indexed_v2t",
           GgmlType::Q5_0 => "moe_batched_gemm_q5_0_indexed_v2t",
           GgmlType::Q6_K => "moe_batched_gemm_q6_k_indexed_v2t",
           GgmlType::Q8_0 => "moe_batched_gemm_q8_0_indexed_v2t",
           _ => /* error */
       }
   }
   ```
   Bake the suffix-matcher defensive check (per
   [[feedback-kernel-parity-gate]]) — every `_v2t` name must match the
   dispatcher's `ends_with("_v2t")` test.
6. Patch the 3 call sites in `model/deepseek_v2.rs` to compute the
   tier override + buffer override + kernel name per layer.
7. Add `crates/dismantle-core/tests/mixed_precision_parity.rs`:
   generate 256 tokens greedy with `quant_tier_map_path = None` and
   then with the default tier map; require all 256 tokens match
   (or document the first divergent position + ppl delta on a
   small held-out shard).
8. Add `(1408, 2048, "v2_lite_down_q8")` to `kernel_bench.rs` so the
   re-quantized layer microbench is visible alongside the existing
   Q4_K/Q5_0/Q6_K entries.
9. Run `tools/bench/quick_bench.sh` paired (off-mode vs
   `--quant-tier-map .../v2_lite_default.json`); accept ≥ +1 dec_tps
   delta as a ship gate. The aggressive `_down_q4` map should land a
   bigger delta — measure both.
10. Wrap memo under `memory/mixed_precision_landed.md` with the actual
    measured tps + the quality verdict per tier-map.

Estimated effort: **1 focused session** for the dispatcher patches +
parity test once the shader-coverage question is settled.

---

## 11. The corrected speedup analysis

The original §1 estimate of +5-10 tps assumed re-quantizing all three
MoE tensor groups (gate, up, down) across all layers. Reality:

| layer band | current down dtype | tier-map down | bandwidth Δ |
|---|---|---|---|
| 0 (dense) | — | n/a | 0 |
| 1–3 | Q5_0 | Q4_K | −18% (~6 MB/layer × 3 = saves ~18 MB) |
| 4–24 | Q5_0 | **Q8_0** | **+55%** (~17 MB/layer × 21 = adds ~360 MB) |
| 25–26 | Q5_0 | Q6_K | +18% (adds ~12 MB) |

Net for the default (quality-driven) map: **+354 MB more bytes per
decode token** on the MoE down-projection path. That's a bandwidth
regression for tps, even if it's correct for residual-stream
quality. The **aggressive Q4_K-everywhere** map saves roughly 250 MB
of MoE down bytes per token; against ~150 GB/s effective bandwidth
that's ~1.7 ms saved per layer's down read, or ~30-40 ms across all
26 MoE layers — but that's the *theoretical* ceiling. Real-world
gain is bounded by:

1. **Down-projection bytes are ~30% of MoE bytes** (gate+up dominates).
2. **Routed experts hit only 6/64 experts per token** — actual bytes
   touched per layer are `top_k × bytes_per_expert`, not full table.
3. **L2 cache thrashing already caps effective bandwidth at ~20 GB/s**
   per `v110_path30_findings.md` — we can't always cash byte savings.

Realistic best case for aggressive down-Q4: **+1 to +3 dec_tps**. The
default conservative map is **0 or net-negative** on raw speed but may
preserve quality better — the parity test will arbitrate.

A more ambitious lever 2 would extend to gate+up, but that requires
new Q6_K/Q8_0 fused-gate+up shaders (a separate kernel-development
workstream of ~2-3 days). Logged for a future path-to-X attempt.
