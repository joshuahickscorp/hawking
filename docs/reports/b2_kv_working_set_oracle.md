# B2 — KV-working-set (§8 L1.1) attention-mass oracle

**Step:** overnight queue B2 (`plans/overnight_build_queue_2026_05_31.md`)
**Date:** 2026-05-31
**Mode:** unattended, oracle-first. Outcome: **oracle instrument built +
run; eviction NOT built (HALT-WITH-DESIGN per the step's "don't build
capture + eviction both unattended" rule).**

## 1. Is attention measurable locally? — YES, via a built instrument

**No attention-capture path existed at HEAD.** The decisive feasibility
facts:

- **Production decode (TCB) discards attention weights.** The live path is
  `forward_token_greedy_tcb` → the Metal kernel `mha_decode_f32_tcb`
  (`crates/dismantle-core/shaders/mha.metal`). That kernel computes the
  per-position softmax in **threadgroup memory** (`shmem`, "scores"
  buffer), softmaxes in place, and consumes it in Phase 4 to produce the
  head output. Only the attention *output* (head_dim-sized) is written to
  a global buffer — the per-position weights never leave the GPU. So the
  production path cannot be observed without a new kernel that spills
  `scores`, which is not worth building for an oracle.
- **The CPU reference path materializes the weights.** `forward_token`
  (the non-TCB path, `qwen_dense.rs:2439`) calls `crate::attn::mha_decode_step`,
  which builds the full post-softmax `scores[0..ctx_len]` per head on the
  host. **Prefill uses this CPU path by DEFAULT** — `DISMANTLE_QWEN_TCB`
  is opt-in (`qwen_dense.rs:1450`), so a plain `generate` run prefills the
  whole prompt through `forward_token`. This is exactly where long-context
  attention concentration is observable.

**Instrument built (default-off, pure side-observer):**
- `crates/dismantle-core/src/attn/mod.rs` — added `mha_decode_step_weights`,
  which recomputes the per-head post-softmax distributions with arithmetic
  identical to `mha_decode_step` (same scale, same softmax). Oracle-only.
- `crates/dismantle-core/src/stateful/attn_capture.rs` — new module. A
  process-global accumulator (no `QwenDense` field/ctor change). Per
  (layer, query position) at ctx_len ≥ `min_ctx`, it head-averages the
  attention and folds in: min #positions to reach 90/99/99.9% mass
  (descending sort), mass on the first `SINK_SPAN=4` positions, mass on
  trailing recent windows {32,64,128}, and the sinks∪recent coverage.
  Dumps compact per-layer means to JSON on `flush`.
- Wired into `forward_token` behind `DISMANTLE_QWEN_ATTN_CAPTURE=1` only
  (`qwen_dense.rs`, right after the `mha_decode_step` call). Flushed once
  from the CLI `generate_main` (`crates/dismantle/src/main.rs`).
- `tools/bench/oracle_attn_mass.py` — offline reader → GO/NO-GO verdict.

Gate hygiene: bit-identical to production with the flag unset (it is an
`if crate::stateful::attn_capture::enabled()` guard). `cargo build
--release --workspace` green; `cargo test --workspace --lib` 81 core + 9
serve tests pass. `mha_decode_step` itself was **not** touched.

## 2. Concentration finding

Run: `DISMANTLE_QWEN_ATTN_CAPTURE=1` on a real code prompt (1043-token
`qwen_dense.rs` source — the long-context coding workload the lever
targets), CPU prefill path, M3 Pro. min_ctx filter 128 → **917 scored
query positions per layer, mean scored context 586 tokens, all 36
layers.** Output: `reports/bench/attn_capture.json`.

**Concentration is BROAD, not sparse.** Aggregated per-layer means:

- **#positions to reach 99% of attention mass = 78–92% of the context**
  (median frac99 **0.80**, worst layer **0.92**). To keep 99% of mass on
  the hardest layer you must retain ~539 of 586 positions — that is **not
  a bounded working set**, it is essentially the whole cache.
- **The StreamingLLM "sinks + recent window" structure does NOT hold.**
  Mass on sinks(4) ∪ recent(128) is only **18–73%** per layer (worst
  layer **0.18**, median 0.42). Even the widest recent window leaves most
  mass uncovered on most layers.
- Behavior is **heterogeneous across layers**: a few are sink-dominated
  (L5 sink-mass 0.44, top1 0.43; L6 0.34), a few recent-dominated (L1
  recent-128 0.79; L0 0.69), but most are diffuse (L2 frac99 0.91, L30
  0.92, L34 0.92). A bounded budget must hold on **every** layer
  simultaneously, and the worst layer governs → no single positional or
  cumulative policy clears the bar.
- mean top-1 weight 0.04–0.43 (often <0.2): attention is not peaky.

Per-layer table (excerpt; full data in the JSON / via the reader):

```
layer  ctx  top1  sink  s+r128  pos99  frac99
   0   586 0.217 0.007  0.687   449.2  0.797
   2   586 0.064 0.016  0.454   528.2  0.909   <- diffuse
   5   586 0.431 0.441  0.470   286.5  0.506   <- sink-heavy
  21   586 0.295 0.296  0.712   424.3  0.772
  30   586 0.081 0.085  0.307   535.5  0.919   <- worst
  35   586 0.043 0.019  0.309   467.8  0.831
```

### Verdict: **NO-GO — Type-1** (on Qwen2.5-3B + this context)

A bounded sinks+recent (or heavy-hitter) working set would drop
load-bearing context on most layers. This is a **measured property of the
model's attention distribution** at this context length, not a defect in
the StreamingLLM/H2O *form* — same kill class as block-256 FFN sparsity.
The diffuseness defeats *any* selection rule: if 99% mass spans ~80–92%
of positions, the "important" set is ~80–92% of the cache however you pick
it (positional H2O/SnapKV reframe dies identically — recorded in
`reports/dead_levers.md`).

**In-sample discipline — this is a SINGLE prompt, one domain (code), and
mid-length (~586 tokens scored).** The literature's StreamingLLM/H2O
finding is for *much* longer contexts (16K–128K) where sink/recent
structure sharpens, and often larger models. The honest read: **NO-GO at
the measured regime**, with a concrete, cheap named oracle to revisit
(longer captures / other domains / larger Qwen — see §4 kill note and the
dead-lever resurrection check). The instrument now exists to run that
revisit in minutes.

## 3. What was built vs. deferred

- **Built:** the oracle instrument + reader (above). This is the missing
  prerequisite the design doc (`plans/stateful_core_design_2026_05_30.md`
  §2.5) named as "long-context capture replay" but left unwired.
- **Deferred (HALT-WITH-DESIGN):** the eviction bodies. Building them
  means wiring `EvictionPlan` application into the **GPU decode arena**
  (`DenseDecodeArena` K/V buffers compaction in `forward_token_greedy_tcb`)
  — a live-decode-path change that is exactly the unattended risk the step
  forbids pairing with capture. Design below.

## 4. Build design (for the next attended session, if GO)

The stub trait is already shaped (`working_set.rs`): `KvEvictionPolicy`
(StreamingLLM/H2O/SnapKV/Lossless), `KvWorkingSet<P>`, `WorkingSetBudget`,
`WorkingSetMode::{Bounded,Lossless}`. The bodies + wiring:

1. **Policy bodies first (CPU, no GPU risk).** Implement
   `StreamingLlmPolicy` (drop positions outside `[0..sinks) ∪
   [ctx-recent, ctx)`) and, if the oracle says heavy positions are
   scattered (H2O-style), `H2OPolicy` (running per-position cumulative
   mass via `observe_attention`, keep recent + top-`heavy`). Unit-test the
   `EvictionPlan` invariants (never evicts a protected position; leaves
   retained ≤ budget) on synthetic distributions — no model load.
2. **Wire the OBSERVE half on the CPU path first.** `forward_token` already
   has the per-head weights available (the oracle proves it). Feed them to
   `observe_attention`; assert the plan matches an offline recomputation.
   Still no eviction — just plumbing + parity.
3. **Apply eviction in the CPU path** (`forward_token`): compact
   `self.kv.keys/values[li]` per the plan before the `seq_len` bump.
   Gate behind `WorkingSetMode::Bounded` + a new
   `DISMANTLE_QWEN_KV_WORKING_SET` flag, default-off. **Bit-identical
   gate:** when ctx < budget the plan is `keep_all`, so short-context
   output must be byte-for-byte unchanged (the Lossless escape hatch +
   the "no eviction below budget" invariant guarantee this — test it).
4. **Quality-vs-budget curve** (the second oracle the design names):
   decode a long prompt at budgets {sinks+128, +256, +512, ∞} and report
   token-drift / perplexity vs the lossless run. Only ship a default-off
   `Bounded` mode if the curve is flat to a usable budget.
5. **GPU decode-arena compaction (LAST, attended).** Port the apply step
   to `forward_token_greedy_tcb`'s `DenseDecodeArena` so the bounded
   working set holds at decode-time (the actual RAM/bandwidth win). This
   is the only GPU-path change and must clear its own parity gate.
6. **Compression path (future).** `EvictionAction::Compress` + the fused
   quantized-KV attention kernel (design doc §2.4) — out of scope until
   drop-eviction ships and the fused-QKV kernel exists.

**Kill-protocol note:** if the oracle is NO-GO (mass spread broadly), that
is a **Type-1** death on this model+workload (a measured property of
Qwen2.5-3B's attention, like the block-256 FFN sparsity kill) — the
`StreamingLlmPolicy`/`H2OPolicy` *form* doesn't change a spread-mass
reality. The `LosslessPolicy` escape hatch ships regardless (no-op, needs
no oracle). The named re-test oracle: re-run this instrument on
genuinely longer captures (16K–32K) where sink/recent structure is known
to sharpen — concentration at 1–3K may understate the long-context case.

## 5. Files changed

- `crates/dismantle-core/src/attn/mod.rs` (+`mha_decode_step_weights`)
- `crates/dismantle-core/src/stateful/attn_capture.rs` (NEW)
- `crates/dismantle-core/src/stateful/mod.rs` (+`pub mod attn_capture;`)
- `crates/dismantle-core/src/model/qwen_dense.rs` (capture call in
  `forward_token`, default-off guard)
- `crates/dismantle/src/main.rs` (one `flush()` after generate)
- `tools/bench/oracle_attn_mass.py` (NEW, reader)

All default-off. No committed kernel touched. No `plans/`/`colab/`/
`eagle5*` touched.
