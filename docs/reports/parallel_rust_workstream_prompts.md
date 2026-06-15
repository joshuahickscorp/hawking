# Parallel Rust workstream prompts — copy into fresh Claude Code sessions

3 independent supervised workstreams. Each can run in parallel; minimal file
conflicts. Total compound projection: ~24 → 45-55 dec_tps when all land.

Sessions in priority order:
- **Session A** — Mixed-precision Path A wedge (+3-5 tps, 1-2 days)
- **Session B** — Spec-decode runtime cost reduction (+8-15 tps, ~1 week — biggest lever)
- **Session C** — Q8 KV wiring (+2-5 tps, 2-3 days)

All three avoid the MPS GPU (used by background eagle5 training). They are
CPU+disk+Metal-shader work that runs alongside without contention.

Defer until later: Eagle5 head deployment (blocks on B); MoE GEMM kernel
improvements (needs diagnostics we haven't gathered yet).

Each prompt starts with `--- SESSION X PROMPT ---`. Copy from there.

---

## SESSION A PROMPT — Mixed-precision Path A wedge

--- SESSION A PROMPT ---

You are implementing the path-to-50 lever 2 wedge in dismantle (Apple Silicon
MoE inference engine for DeepSeek-V2-Lite-Chat). The foundation exists; this
session wires it into the GPU dispatch path. Expected gain: **+3-5 dec_tps**
(midpoint of the +1-3 to +5-10 projected range, narrowed by the
gate+up-shader limitation documented below).

Working dir: `/Users/scammermike/Downloads/dismantle`.
M3 Pro 18 GB. Baseline: ~22 dec_tps with Claude open / ~27 clean.
**A background autonomous pipeline is training eagle5 heads on MPS — do NOT
run `dismantle generate` or `dismantle bench` until that finishes** (status
at `artifacts/runs/overnight/extended_status.json`; wait until
`current_stage == complete && state == done`). Cargo work is fine in
parallel — it's CPU-bound.

### Foundation already landed (read these before editing)

- `crates/dismantle-core/src/quant_tier_map.rs` — `TierMap` loader.
  `TierMap::tier_for(layer, GroupKind) → Option<dtype>`.
- `crates/dismantle-core/src/mixed_quant_store.rs` — `MixedQuantStore::build`
  + `::build_cached` (~30-60 s build, then disk-cached). Holds re-quantized
  expert weights as a blob + per-tensor descriptors.
  **Limitation:** v1 only supports `Down` projection tier overrides.
  `GateUp` returns an error unless tier matches source GGUF dtype.
  Fused gate+up requires new Q4_K_v2t_gu cross-dtype shaders that are out
  of scope here.
- `crates/dismantle-core/src/profile.rs` — `vocab_prune_path`,
  `quant_tier_map_path` fields exist on `Profile`.
- `crates/dismantle-core/src/engine.rs:39` — `EngineConfig.quant_tier_map_path`
  field exists.
- `crates/dismantle/src/main.rs` — `--quant-tier-map-path` CLI flag already
  threaded to `EngineConfig` (see existing `--vocab-prune-path` for pattern).
- Tier maps at `artifacts/calibration/tier_maps/*.json`. Per-layer stats at
  `artifacts/calibration/analysis/per_layer_residual_stats.json`.

### The gap (what this session wires)

`MixedQuantStore` is built once at model load — but the model's per-layer
expert weight loader (`crates/dismantle-core/src/model/deepseek_v2.rs`
around line 558-595, the `LayerMode::Moe` branch) **never queries it**.
Expert weights still come from `Self::tensor_ref(&gguf, &lp("ffn_down_exps.weight"))?`
even when a tier map is set, and the kernel dispatcher routes to the source
dtype's GEMM kernel.

### Implementation steps

#### 1. Engine plumbing audit (~30 min)
Verify `EngineConfig.quant_tier_map_path` flows from CLI → `Engine::new` →
`Model::load`. Add a `MixedQuantStore` field to `Model` (or
`DeepseekV2Engine` or whatever the load target struct is — check the file).
Build it at model construction when `quant_tier_map_path.is_some()`:

```rust
let mixed_store = if let Some(tier_path) = &config.quant_tier_map_path {
    let tier_map = TierMap::load(tier_path)?;
    Some(MixedQuantStore::build_cached(
        &gguf,
        weights_path,
        &tier_map,
        tier_path,
        cfg.n_layers,
        cfg.first_k_dense_layers,
        cfg.n_routed_experts,
        false,  // include_shared: skip — only Down overrides supported v1
    )?)
} else {
    None
};
```

Hold `mixed_store: Option<MixedQuantStore>` on the model struct.

#### 2. Expert weight loader swap (~60 min)
`crates/dismantle-core/src/model/deepseek_v2.rs` ~line 579, where you see:
```rust
down_w: Self::tensor_ref(&gguf, &lp("ffn_down_exps.weight"))?,
```
Replace with a check: if `mixed_store` has an entry for
`(layer_id=li, GroupKind::Down, dtype)` (use `mixed_store.get(StoreKey {...})`),
return a `TensorRef` pointing into `mixed_store.blob()` instead of the GGUF
mmap. The resulting TensorRef must carry the new (smaller) dtype so the
dispatcher knows to route to the q4_K (or q6_K) kernel instead of q8_0.

Read `MixedQuantStore::get` and `StoredTensor` in
`crates/dismantle-core/src/mixed_quant_store.rs` (~line 100-160) to see
the exact byte-offset + dtype interface. The blob is owned by the
`MixedQuantStore`; the `TensorRef` borrow lifetime must respect that —
either hold the store on the model permanently, or `Arc` it.

#### 3. Kernel dispatcher routing (~60 min)
`crates/dismantle-core/src/kernels/mod.rs` already has per-dtype dispatch
(`moe_batched_gemm_q8_0_indexed` etc — see line ~1463). The MoE down GEMM
call site (find via `grep -n "moe_batched_gemm\|moe_indexed_gemm" crates/dismantle-core/src/model/`)
selects a kernel based on the `TensorRef.dtype`. With the swap from step 2,
this should "just work" — the new dtype will route to the q4_K_indexed
kernel instead of q8_0_indexed.

**Audit `kernels/mod.rs` for a `_v2t_v3` style suffix-matcher bug** (per
memory `feedback_kernel_parity_gate.md`): when the dispatch table looks up
a kernel name, ensure mixed-bit-width layers don't fall through to a
"kernel-not-found" path. Specifically check whether the dispatcher uses
exact name match vs prefix match vs suffix-tolerant match.

#### 4. Parity test (~30 min)
File: new `crates/dismantle-core/tests/mixed_precision_parity.rs`.

Acceptance: with `--seed 0 --temperature 0` (greedy), the **first 64 tokens
of a fixed prompt must be bit-identical** between mixed-precision and
uniform fp16 fallback. If divergence at token N: the requantization or
dispatch path is wrong.

Reference pattern: `crates/dismantle-core/tests/q8_kv_parity.rs` or
`tests/integration_greedy_64.rs`. Use `fresh_test_profile(weights_path)`
helper for the kernel profile (avoids shader-hash mismatch — already in
`profile.rs`).

#### 5. Cargo build + microbench (~10 min)
Wait until eagle5 v3/v4 training is done (status file shows
`complete`/`done`) before triggering `cargo build --release -p dismantle`.
Then:
```sh
./target/release/dismantle bench-kernel --all
```
Confirm q4_K_indexed and q6_K_indexed kernels appear with reasonable
timings vs q8_0_indexed.

#### 6. End-to-end bench (~30 min)
With Claude Code QUIT for clean numbers (per memory
`bench_contamination.md`):
```sh
WEIGHTS=models/deepseek-v2-lite-q4.gguf \
  PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json \
  TOKENS=64 bash tools/bench/quick_bench.sh
```
Compare baseline vs `--quant-tier-map-path artifacts/calibration/tier_maps/<chosen>.json`.
Target: **+1-5 dec_tps net** with no token divergence.

If the gain is < +1 tps: the tier map may be over-aggressive (too many q4
layers) or under-aggressive (too few). Iterate by editing the JSON —
`artifacts/calibration/analysis/per_layer_residual_stats.json` ranks
layers by abs_max (lower = more q4-tolerant).

### What NOT to do

- **No Claude git attribution.** User's global rule. No `Co-Authored-By`,
  no "Generated with Claude" footers.
- **No autonomous commits.** Always ask before `git commit`.
- **Don't touch `gate_up` tier overrides.** v1 of `MixedQuantStore` rejects
  them — accept that limitation. New shaders are a separate workstream.
- **Don't run `dismantle generate` while eagle5 v3/v4 is training.** Check
  `artifacts/runs/overnight/extended_status.json` first.
- **Don't enable the tier map by default.** Opt-in via `--quant-tier-map-path`
  only.
- **Don't quantize the embed table or LM head.** Out of scope; LM head is
  handled by vocab-prune.

### Done condition

- `cargo test -p dismantle-core` green
- `mixed_precision_parity` test bit-identical at 64 tokens
- Microbench confirms per-layer kernel routing (q4_K_indexed appears in
  the dispatch trace)
- End-to-end bench (Claude quit) shows ≥+1 dec_tps
- Memory note added: `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/mixed_precision_landed.md`

Estimated effort: **1-2 days focused work**.

Cross-references:
- `reports/path_to_50_oneshot_prompt.md` §4
- `reports/mixed_precision_quant_wiring_handoff.md` (if it exists from a prior session)
- memory: `corpus_complete_analysis_landed.md`, `v110_path30_findings.md`,
  `feedback_kernel_parity_gate.md`

End of Session A prompt.

---

## SESSION B PROMPT — Spec-decode runtime cost reduction

--- SESSION B PROMPT ---

You are reducing the per-step cost of spec-decode in dismantle. This is
the **biggest single tps lever** in the project right now — measured
gain projected at +8-15 dec_tps when it lands, IF you can bring the
draft+verify step cost from ~3-4× off-mode cost down to ~1.5-2×.

Working dir: `/Users/scammermike/Downloads/dismantle`.
M3 Pro 18 GB. Baseline: off-mode ~22-27 dec_tps; eagle4 K=4 currently
runs at 9-12 tps (NET NEGATIVE — that's the problem you're solving).
**A background pipeline is training eagle5 heads on MPS — wait for
`artifacts/runs/overnight/extended_status.json` to show
`current_stage == complete && state == done` before running any benches.**

### Critical context from prior diagnostics

The spec-decode RUNTIME is structurally healthy (per
`reports/spec_decode_runtime_NOT_broken_2026_05_22.md`). The issue is
**per-step cost**:

Bench evidence (`artifacts/runs/overnight/spec_decode_sweep.md`):

| Prompt | off (tps) | eagle4 K=4 (tps) | accept ratio | regression |
|---|---|---|---|---|
| Once upon a time | 24.04 | 9.26 | 24/130 = 15.6% | −62% |
| Capital of France | 22.25 | 12.07 | 42/46 = 47.7% | −46% |
| def fibonacci(n) | 21.80 | 11.73 | 38/61 = 38.4% | −46% |

Even at **47.7% draft acceptance**, spec-decode loses 46%. The math:
- For net-positive: `1 + avg_accept > T_spec / T_off`
- Measured: `1 + avg_accept ≈ 1.5-1.9`, but `T_spec / T_off ≈ 2.5-3.5`
- Need to drive `T_spec / T_off` down to ~1.5 or `avg_accept` way up

Both improvements help but **runtime cost reduction is more deterministic**
than head accept-rate improvements (which depend on eagle5 training
quality — uncertain).

### Investigation plan

#### 1. Profile the draft step (~2-3 hours)
Per-kernel timing of a single draft+verify cycle. Use
`DISMANTLE_TCB_TRACE=1` env or `--trace-dispatch`:
```sh
DISMANTLE_TCB_TRACE=1 ./target/release/dismantle generate \
    --weights models/deepseek-v2-lite-q4.gguf \
    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
    --speculate exact-shared --verify-window 4 \
    --prompt "Once upon a time" --max-new-tokens 8 --seed 0
```

Parse the trace to identify:
- Which kernels run during DRAFT but not OFF mode
- Which kernels run K times during VERIFY (one per draft token)
- The single most expensive kernel in the spec-decode-only path

Existing parser: `tools/bench/analyze_tcb_trace.py`. Extend if needed
to compute per-kernel ms summed across the decode trace.

#### 2. Identify the cost driver (~1-2 hours)
Hypotheses to test (in order of likelihood from
`memory/per_kernel_time_breakdown.md`):

a. **Verify-pass MoE GEMM** runs K times (once per draft token in the
   accept loop). At K=4, that's 4× the MoE GEMM cost vs off-mode.
   This alone could explain a 2-3× per-step cost ratio.

b. **Draft head forward** — eagle4 head is ~30M params; forward at
   each speculative step adds non-trivial cost. Check whether the head
   is dispatched on MPS or CPU.

c. **KV cache append/rollback** during accept/reject — speculative
   steps that get rejected must revert KV state. If this involves
   buffer copies, that's hidden overhead.

d. **Cache pollution** — target model's KV cache may be invalidated and
   recomputed for verify. Look for repeated prefill-like patterns.

#### 3. Pick the highest-impact cost driver and fix it (~3-4 days)
Most likely fix targets (in `crates/dismantle-core/src/speculate/`):

- **Batched verify**: combine K verify positions into one GEMM call
  instead of K separate dispatches. The target model has the K tokens
  already; a (K, hidden) × (hidden, vocab) GEMM is one kernel, vs K ×
  (1, hidden) × (hidden, vocab) GEMVs.
- **Skip MoE re-routing on accepted prefix**: when token i is accepted,
  the next verify position's MoE routing decision SHOULD be reusable
  from cache for the prefix. Verify the runtime does this.
- **Pipeline draft and verify**: if draft computes K tokens
  sequentially, hide the head forward latency by pre-fetching the next
  draft step while verify runs.

#### 4. Parity test (~1 hour)
Spec-decode must remain bit-identical-equivalent (greedy temp=0) to
off-mode at the same seed. Test the optimized path produces the same
output as before optimization.

File: extend `crates/dismantle-core/tests/spec_decode_parity.rs` if it
exists; create otherwise.

#### 5. Re-run the sweep (~30 min compute)
Re-run `tools/bench/spec_decode_sweep.sh` (Claude quit for clean
numbers). Target: **eagle4 K=4 tps ≥ off-mode tps**, ideally a small
positive gain (1-3 tps). That validates the runtime cost reduction
is real and unblocks eagle5 head deployment.

### What NOT to do

- **No git attribution to Claude.** User's global rule.
- **No autonomous commits.**
- **Don't touch eagle4 weight format** — it's frozen; runtime must
  consume it as-is.
- **Don't run benches while eagle5 v3/v4 training is on MPS.** Wait for
  pipeline completion.
- **Don't shrink the model itself** — that's a different workstream
  (mixed-precision Path A, Session A).
- **Don't break the off-mode path**. All optimizations gated on
  `if config.speculate_mode != SpeculateMode::Off`.

### Done condition

- Re-run sweep shows eagle4 K=4 ≥ off-mode tps on at least 2 of 3 prompts
- `spec_decode_parity` test green
- A `reports/spec_decode_runtime_cost_2026_*.md` writeup with:
  - The cost driver identified
  - The fix applied
  - Before/after measurements
  - Estimated remaining headroom (how much further could it go?)
- Memory note added: `spec_decode_cost_reduced.md`

After this lands, eagle5 head deployment (a separate ~1-day session)
should immediately deliver +5-10 net tps.

Estimated effort: **~1 week of focused work**.

Cross-references:
- `reports/spec_decode_runtime_NOT_broken_2026_05_22.md`
- `artifacts/runs/overnight/spec_decode_sweep.md`
- memory: `path_to_100_repath.md`,
  `feedback_bench_with_claude_open.md`, `bench_contamination.md`

End of Session B prompt.

---

## SESSION C PROMPT — Q8 KV production wiring

--- SESSION C PROMPT ---

You are landing the Q8 KV cache in production. The kernels are already
written and microbenched — this session wires them into the
DeepSeek-V2-Lite decode path. Expected gain: **+2-5 dec_tps**.

Working dir: `/Users/scammermike/Downloads/dismantle`.
M3 Pro 18 GB. Baseline: ~22-27 dec_tps depending on Claude state.
**A background pipeline is training eagle5 heads on MPS — wait for
`artifacts/runs/overnight/extended_status.json` to show
`current_stage == complete && state == done` before running benches.**

### Foundation already landed (read first)

Per memory `q8_kv_landed.md`: `mla_decode_q8kv` kernel is in
`crates/dismantle-core/src/kernels/`. Microbench shows
1.28-1.96× per-kernel speedup at V2-Lite shapes. **Wiring** is what's
missing — the production cache path still uses fp16 K/V tensors.

Read first:
- `crates/dismantle-core/src/kernels/` — Find the `mla_decode_q8kv`
  kernel definition + dispatcher entry.
- `crates/dismantle-core/src/cache/` — current fp16 KV cache impl.
- `crates/dismantle-core/src/attn/` (or wherever MLA lives) — KV cache
  read site during decode.
- `crates/dismantle-core/tests/q8_kv_parity.rs` — existing parity
  reference (per memory).
- `artifacts/calibration/analysis/expert_load_per_layer.json` — although
  routing balance was 0.987-0.995 (too uniform for per-layer KV
  precision tuning per `corpus_complete_analysis_landed.md`), the
  uniform-Q8-on-all-layers is the in-scope target.

### Implementation steps

#### 1. KV cache storage swap (~3-4 hours)
Find the KV cache struct + its allocator. Add an int8 storage variant
with per-block (or per-head) scale factors. When `engine_config.q8_kv ==
true` (add this field; default false), allocate int8 buffers instead of
fp16.

File-level plan (audit before assuming line numbers):
- `crates/dismantle-core/src/cache/mod.rs` or similar — add
  `KVCacheVariant::Q8 { ... }` enum variant.
- `crates/dismantle-core/src/engine.rs` — add `q8_kv: bool` field on
  `EngineConfig` and CLI flag `--q8-kv` to main.rs.
- `crates/dismantle-core/src/model/deepseek_v2.rs` — when allocating
  the cache at model load, dispatch to the q8 variant if config says so.

#### 2. Append path: fp16 → int8 quantize on write (~2 hours)
The decode loop writes new K/V each token. Add a kernel call (or use
existing `kv_append_q8_0_f32_metal` per memory note — that exists in
`crates/dismantle-core/src/kernels/mod.rs:1813`) that quantizes during
the append.

#### 3. Read path: route MLA decode through `mla_decode_q8kv` (~3-4 hours)
The MLA attention kernel currently reads fp16 K/V. Route to the q8 variant
when the cache is q8-flavor. Wire the per-block scale factor through.

#### 4. Parity test (~1 hour)
Already exists at `crates/dismantle-core/tests/q8_kv_parity.rs`. Make
sure it covers your wiring (test ranges over expected shapes; fixed
seed, greedy decode). Target: bit-identical OR within 1 ULP at 64 tokens
(Q8 introduces minor rounding; 100% bit-identical may be unattainable —
match the existing parity test's tolerance).

#### 5. Bench (~30 min compute)
With Claude quit:
```sh
WEIGHTS=models/deepseek-v2-lite-q4.gguf \
  PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json \
  TOKENS=64 bash tools/bench/quick_bench.sh
```
Baseline vs `--q8-kv`. Target: **+2-5 dec_tps** with parity test green.

### What NOT to do

- **No git attribution to Claude.** User's global rule.
- **No autonomous commits.**
- **Don't pursue Q8 KV layer-differential** — corpus analysis showed
  routing balance is too uniform across layers
  (`corpus_complete_analysis_landed.md`). Uniform Q8 across all layers
  is the in-scope target.
- **Don't enable q8 KV by default**. Opt-in via flag until parity is
  proven on full bench.
- **Don't run benches during eagle5 training.** Check status.json.
- **Don't quantize the MLA absorbed-V** — that's `mla-phase4-queued.md`
  separate workstream.

### Done condition

- `cargo test -p dismantle-core --test q8_kv_parity` green
- Microbench shows expected 1.28-1.96× kernel speedup is realized in
  production decode
- End-to-end bench (Claude quit) shows ≥+2 dec_tps with `--q8-kv` flag
- Memory note added: `q8_kv_production.md`

Estimated effort: **2-3 days**.

Cross-references:
- memory: `q8_kv_landed.md`, `corpus_complete_analysis_landed.md`
- `crates/dismantle-core/tests/q8_kv_parity.rs`

End of Session C prompt.

---

## How to launch

For each session:
1. Open a fresh Claude Code session at `/Users/scammermike/Downloads/dismantle`
2. Copy everything between the `--- SESSION X PROMPT ---` markers
3. Paste as the first message
4. The session reads its own context from the prompt + the files it
   references

All three sessions are independent. If you have CPU bandwidth, run them
in parallel. Best order if serial:
- A first (smallest, contained, real win)
- C in parallel (independent files)
- B after eagle5 training pipeline is done (Session B needs benches)

Sessions B and C heavily benefit from waiting on the autonomous eagle5
training to finish so bench numbers are clean.

## Note on the foundational diagnostics gap

We did NOT execute the "high-leverage foundational work" (per-tensor
calibration stats, comprehensive kernel microbench, pre-built parity
harnesses, trace-dispatch capture) in the autonomous run. The Session B
prompt above absorbs that work — it asks the agent to do trace-dispatch
profiling as step 1. The Session A and C prompts are well-scoped enough
that they don't strictly need the diagnostics, but they will benefit
from the trace-dispatch data Session B produces if you run B first.

If you want a pure-diagnostics session before any of the above, ask for
"Session D — Diagnostics Sweep" and it'll produce all the missing data
in ~3-4 hours (kernel microbench across all variants × shapes, per-tensor
calibration stats, trace-dispatch capture, parity-test harnesses).
