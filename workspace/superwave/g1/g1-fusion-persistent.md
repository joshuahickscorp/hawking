# G1 fusion + persistence — dispatch/submission ceremony on Qwen3.8

Lane: `16-fusion-persistent`. HEAD `2eee9a004`. No GPU run this lane.
Every number is tagged MEASURED / PROJECTED / ESTIMATED / UNVERIFIED_CLAIM.
A component microbench is not a token-level claim.

---

## 0. Current Qwen3.8 execution genome (this commit)

HEAD `step()` re-encodes the full graph every token into one command buffer
(`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3292–3310`):

```
encode_embed; encode_layers; encode_terminal; commit_and_wait_timed
```

`encode_layers` is a host `for layer in 0..64` (`:2822–2829`). Host is not
waiting per layer. Host is encoding 964 dispatches per token.

Structural graph (`crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12–54`):

| block | dispatches | kernels |
|---|---:|---|
| mixer prefix | 9 | DeltaNet: rmsnorm, qkvz GEMV, ba GEMV, rearrange+conv, ba_to_decay, gated_delta_vi, gated_rmsnorm, out GEMV, add. GQA: rmsnorm, q/k/v GEMV, qk_norm_rope_cache, mha_decode, sigmoid_gate, o GEMV, add |
| dense MLP suffix | 6 | rmsnorm, **gate GEMV**, **up GEMV**, **silu_mul**, down GEMV, add |
| per layer | 15 | |
| token | 1 embed + 64×15 + 3 terminal = **964** | 1 CB |

Confirmed by G024 `dispatches.total=964`, `production_command_buffers=1`
(`receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` field `dispatches`).

Gate and up are **two separate** `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
launches plus a standalone `qwen80_silu_mul_f32`. They share the post-attn
RMSNorm output. Not fused on HEAD.

---

## 1. Hard negatives — original measurements, not later citations

### 1.1 8-layer f16 megakernel = 4.4× SLOWER than production

**Original source (not the Aug-9 citation):** commit `fe0fb94c5395e58bc5ad30b51b9d7ded406b81b1`
(2026-05-29), message body, M3 Pro, Qwen-3B-Q4_K_M, N=8:

> fused megakernel 44.4 ms/token vs 187 ms for the CPU-orchestrated per-op
> baseline (4.2x). But that baseline is not production: the TCB-batched decode
> runs all 36 layers in ~45 ms (~22 tps), i.e. ~1.26 ms/layer, so the
> megakernel's 5.55 ms/layer is ~4.4x SLOWER per layer than production.
>
> Root cause (two strikes, both bandwidth):
> 1. The POC dequantizes weights to f16 → 4x the bytes of production's Q4_K.
> 2. Single threadgroup (256 threads, 1 of ~18 cores) can't saturate the bus.
> The 48→1 dispatch reduction is real but overwhelmed. STOP … do NOT wire
> into decode. A win needs Q4_K inline decode AND a multi-threadgroup fused
> design.

**What actually failed (mechanism, not slogan):**

1. **Expand-to-f16 then GEMV.** `prep_megakernel_layer_f16` materializes f16
   weights (`crates/hawking-core/src/model/qwen_dense.rs` `MegakernelLayerWeightsF16`).
   Binding rule on this wave forbids that production path unless a complete-token
   measurement proves a net win. Here the complete-token-class comparison
   (per-layer vs production TCB) lost 4.4×.
2. **One threadgroup, grid (256,1,1).** Still the shipping launch:
   `megakernel.rs:419–422` and `:537–540` dispatch `qwen3b_megakernel_nlayer`
   with `(MK_TG_SIZE,1,1)` / `(MK_TG_SIZE,1,1)`, `MK_TG_SIZE=256`.
   Shader: `megakernel_qwen3b.metal:1093–1133` — `Grid = (1,1,1), TG size = 256`,
   `for (li = 0; li < args.n_layers; ++li) mk_layer_forward(...)`.
3. **Serial row GEMV inside that TG.** `megakernel_qwen3b.metal:228–238`:
   each of 256 threads owns `Q_DIM/256 = 8` rows and walks all `HIDDEN`
   columns in f16. Contrast Qwen3.8 production: `geo_tpr64_tg128` launches
   8704 TGs on the 17408-row gate (G024: 145 TGs/core, 97.6% of 411.51 GB/s
   honest decode ceiling).
4. **Persistent runner is the same kernel.** `MegakernelRunner::step`
   (`megakernel.rs:506–540`) only hoists the f16 upload. Still one TG,
   still f16, still `n_layers` in one dispatch.

The 4.2× “win” vs the CPU-orchestrated per-op baseline is a strawman.
The kill is vs production TCB-batched Q4_K decode.

Later citations (`QWEN30_COMPLETE_TOKEN_ASCENT_BUCKET_CLASSIFICATION_20260809T193400Z.json:87`,
`Q30_S_BUCKET_MECHANISM_TABLE.md:104`, `dead_levers` via survey) repeat
“4.4× slower” without re-measuring. They do not change the mechanism.

**KILLS:** any proposal that (a) expands Q4/low-bpw to f16/f32 weights then
generic GEMV, or (b) folds fat GEMVs into one/few threadgroups that loop
stages/layers. **REOPEN_IF:** native Q4 (or denser) consumed in-register
AND multi-TG occupancy ≥ `geo_tpr64_tg128` AND a complete-token A/B that
is not slower than the current 1-CB Q4 genome.

### 1.2 `use_resource` 2.62 µs vs `set_buffer` 4.5 µs is a bind micro-win, not a token lever

**Original source:** commit `a9a2f980effec5b600bd1b8a2fdd77069dffad7f` (2026-05-24):

> Microbench (n=100 trials, 200 dispatches per trial):
> use_resource pattern: 525 us median = 2.62 us/dispatch
> Compare to pso_transition_microbench (set_buffer baseline …): 4.5 us/dispatch.
> The use_resource pattern is FASTER … residency declaration happens once
> per encoder.

This is a **component bind-only** number on `use_resource_poc_add`, not a
token. `use_resource` is cheaper, not more expensive.

Aug-9 classification listed it as prior evidence that “residency batching
matters” (`…ASCENT_BUCKET_CLASSIFICATION…json:87`). The S-bucket table then
**rejected it as a primary S lever** (`Q30_S_BUCKET_MECHANISM_TABLE.md:77–88`):

> Expected recoverable ≪ 20 ms at 2.62 µs vs 4.5 µs × O(10³) binds.
> Cannot close a 427 ms topology gap alone. Status: Rejected as primary.

ESTIMATED upper bound if you naively multiply the delta:
(4.5 − 2.62) µs × 964 binds ≈ 1.81 ms. That arithmetic assumes every
dispatch still pays a full `set_buffer` and that the microbench kernel
resembles a Q4 GEMV bind. HEAD encode of those 964 dispatches is already
MEASURED at 0.886–0.919 ms (authority / G024) — the 1.81 ms product is
therefore an overestimate of remaining bind fat.

**KILLS as a TOKEN_NS primary.** **REOPEN_IF:** encode share returns above
~1 ms/token after ICB is landed and the leftover is demonstrated to be
per-bind, not wait-minus-gpu.

---

## 2. Measured token economics (Qwen3.8, this box)

All DIRTY_ENGINEERING. Not BASE_TRUE_TPS. Other work may have been on the GPU.

| quantity | value | class | pointer |
|---|---:|---|---|
| G0 TPS / TOKEN_NS | 26.4 / 37.9e6 | UNVERIFIED_CLAIM | task contract |
| authority wall | 38,216,792 ns (26.17 TPS) | MEASURED dirty | `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` `headline_32_new_tokens` |
| authority GPU | 36,987,458 ns | MEASURED dirty | same |
| authority encode | 0.8862 ms | MEASURED dirty | same `wall_minus_gpu_named.components_ms[0]` |
| authority wait−gpu | 0.4259 ms | MEASURED dirty | same `[1]` |
| authority submit | 0.0105 ms | MEASURED dirty | same `[2]` |
| authority named fixed sum | 1.3322 ms | MEASURED dirty | same `mean_wall_minus_gpu_ms` |
| G024 wall | 35,227,917 ns | MEASURED dirty | `G024_QWEN38_TOKEN_NS.json` / `TOKEN_NS_QWEN38.json` |
| G024 GPU | 33,912,333 ns | MEASURED dirty | same |
| G024 encode | 919,250 ns | MEASURED dirty | same |
| G024 submit | 12,084 ns | MEASURED dirty | same |
| G024 wait−gpu | 384,250 ns | MEASURED dirty | same |
| G024 intra-CB residual | 341,925 ns | MEASURED dirty | `unattributed_residual` |
| production | 401.6 / 411.51 GB/s = 97.6% | MEASURED dirty | G024 `measurement` |
| BPW | 4.252735126866492 | MEASURED pack | G024 / authority `vehicle.complete_physical_bpw` |
| CBs / dispatches | 1 / 964 | MEASURED | authority `correctness` |

Do not add G024 and authority into one token. They are two dirty sessions
of the same genome. Authority is the 32-new-token wall used as the ICB
“before”. G024 is the closed 12-row ledger.

G024 ranked (MEASURED, isolated families scaled onto production GPU):

| rank | component | ns | % wall |
|---:|---|---:|---:|
| 1 | weight_addressing | 21,293,103 | 60.44 |
| 2 | deltanet | 3,732,795 | 10.60 |
| 3 | gqa | 2,443,471 | 6.94 |
| 4 | normalization | 2,367,415 | 6.72 |
| 5 | weight_decode | 1,808,227 | 5.13 |
| 6 | dense_swiglu | 1,004,198 | 2.85 |
| 7 | host_preparation | 919,250 | 2.61 |
| 8–12 | kv + sync + head + residual + submit | 1,659,459 | 4.71 |

Ceremony that fusion-of-dispatch / ICB / persistence can even see:

```
encode + submit + wait−gpu + intra-CB residual
G024:     0.919 + 0.012 + 0.384 + 0.342 = 1.657 ms  (4.7% of 35.2 ms)
authority: 0.886 + 0.011 + 0.426 + (unnamed in residual) ≈ 1.33 ms named
```

10 ms TOKEN_NS requires deleting ~25–28 ms. Ceremony is not that mass.
Weight addressing alone is 21.3 ms of Q4 DRAM at 87–92% of every GEMV
class (`G024_QWEN38_TOKEN_NS.json` `probes` / `top_three_attacks[0]`).

---

## 3. Isolated tiny-kernel tax (component, not token)

From `QWEN38_TOKEN_NS_LEDGER.json` `isolated` (one CB per family, GPUEnd−GPUStart):

| family | median GPU ns | launches | ns/launch | class |
|---|---:|---:|---:|---|
| input_norms | 1,137,250 | 64 | 17,770 | occupancy-starved reduction |
| post_norms | 1,210,874 | 64 | 18,920 | same |
| final_norm | 19,291 | 1 | 19,291 | same |
| gated_rmsnorm_48 | 1,295,500 | 48 | 26,990 | 16-wide reductions (G024) |
| rope_cache_16 | 1,562,625 | 16 | 97,664 | G024: 24 threads |
| rearrange_48 | 350,999 | 48 | 7,312 | launch + leftover ALU |
| ba_to_decay_48 | 139,374 | 48 | 2,904 | tiny |
| silu_64 | 160,958 | 64 | 2,515 | tiny |
| mlp_residual_64 | 134,208 | 64 | 2,097 | tiny |
| mixer_residual_64 | 118,250 | 64 | 1,848 | tiny |
| sigmoid_16 | 43,625 | 16 | 2,727 | tiny |
| mlp_matvecs_64 | 15,853,666 | 192 | 82,571 | fat Q4 GEMV |
| dn_gemvs | 5,560,749 | 144 | 38,616 | fat Q4 GEMV |
| gqa_gemvs | 1,817,416 | 64 | 28,397 | fat Q4 GEMV |

Older isolated split, same vehicle (`qwen38-layer-dense-q4-swiglu.json` `isolated`):

| family | median GPU ns |
|---|---:|
| mlp_gate_64 | 4,923,208 |
| mlp_up_64 | 5,178,249 |
| mlp_down_64 | 5,426,750 |
| mlp_matvecs_64 | 15,791,708 |
| mlp_full_64 | 16,756,500 |

gate+up+down = 15.528 ms vs matvecs 15.792 ms. The three fat GEMVs are
the MLP. silu+norm+add sit in the ~0.96 ms `mlp_full − mlp_matvecs` gap
on that receipt.

Addr fractions (MEASURED diagnostic kernels, same launch geometry, G024 `probes`):

| class | addr/full | decode−addr | FMA remainder |
|---|---:|---:|---:|
| mlp | 0.872 | 0.084 | 0.045 |
| dn | 0.905 | 0.059 | 0.036 |
| gqa | 0.830 | 0.064 | 0.106 |
| lm_head | 0.916 | 0.037 | 0.047 |

Fat GEMVs are DRAM. Tiny kernels are launch + low occupancy (G024:
norms 122× bandwidth floor; DeltaNet tails 223×; GQA rope 313×).

---

## 4. Proposal-by-proposal

Share-kill test: does the proposal repeat (1) f16/float expansion of
weights, or (2) single-TG / occupancy-collapsed loop over fat GEMVs?

### P1. Fuse gate + up (shared input)

**HEAD state:** two `geo_tpr64_tg128` dispatches + `silu_mul` (`schedule.rs:41–47`).
Independent pair: ICB receipt already omits a barrier between them
(`QWEN38_FIXED_OVERHEAD_DELETED.json` `mechanism.barriers`).

**Does not share the 4.4× mechanism** if the fused kernel keeps
`geo_tpr64_tg128` (or equal TG map) and consumes group-64 Q4 codes+scales
directly. **Does share the kill** if it dequants both matrices to f16/f32
and calls a generic GEMV (`qwen_direct_packed_gate_up_swiglu_fused.metal`
is a Q30 **binary-packed** component kernel, not Qwen3.8 Q4, and is not
wired: file header “The production runtime is not wired to this file”).

**Cost (PROJECTED from isolated, not token-level):**
- Weight bytes unchanged: gate and up are each 17408×5120 × 4.25/8 ≈ 47.4 MB
  × 64 ≈ 3.03 GB each (`QWEN38_TOKEN_NS_LEDGER.json` `weight_bytes.mlp_bytes`
  = 9,091,153,920 for three equal projections).
- Input reuse: 5120 f32 = 20 KiB read twice → once. At 400 GB/s = 50 ns.
  Noise.
- Launch deleted: 64 fat GEMV launches. Intra-CB residual 342 µs / 964
  ≈ 0.35 µs/dispatch ESTIMATED if uniform → 22 µs. Fat GEMVs are not the
  occupancy-starved population; their per-launch GPU time is ~80 µs of
  real DRAM work, not launch tax.
- PROJECTED recover: **0.05–0.30 ms / token.** Not a 10 ms lever.

**KILLS if implemented as expand-then-GEMV.** Legal shape: one
row-owned (or tpr64) kernel, two Q4 bodies, `x[c]` in register, write
gate and up — or write only SwiGLU (P2).

**Cheapest falsifier (serialized GPU lane, not this lane):** isolated
64-layer fused-gate-up Q4 vs serial gate+up, GPUEnd−GPUStart, same
`geo_tpr64_tg128` occupancy. If fused ≥ serial + 0.2 ms, stop.

### P2. Fuse SwiGLU into the producing kernel

**HEAD state:** `qwen80_silu_mul_f32` is its own dispatch (`schedule.rs:45`).
Isolated `silu_64` = 160,958 ns.

**Does not share the 4.4× mechanism** as an epilogue of the Q4 producer
(P1). **Shares the Phase-2.2 “trivial-op fusion dead-for-tps” name**
(`dead_levers.md:18`) but not the evidence: that kill is rope/add/memcpy
sub-noise on a llama path. Isolated silu here is 161 µs, not sub-noise
in the tiny-kernel table, still far below 1 ms.

`dead_levers.md:25` “MoE megakernel deeper / gate+up+SiLU fused /
handoff 0.04% wall” is a **Qwen-MoE** kill. Qwen3.8 dense suffix is not
that genome. Do not transfer the kill.

**Cost (PROJECTED):**
- Delete 64 silu launches: 0.161 ms isolated GPU (component).
- Stop materializing gate and up (17408×4×2×64 = 8.9 MB/token). At
  400 GB/s ≈ 22 µs. ESTIMATED.
- PROJECTED recover: **0.16–0.25 ms** if folded into P1. Alone, 0.16 ms.

**Legal shape:** Q4 gate+up producer writes the activated intermediate
only. No expand-to-float weights. Down GEMV already consumes that
activation as f32 x.

**Cheapest falsifier:** fused producer vs (gate+up+silu) on 64 layers,
bit-id on the activation vector, GPU timestamps.

### P3. Fuse norms into the consumer

**HEAD state:** 129 `qwen80_residual_rmsnorm_f32` launches (64 input +
64 post + 1 final). Isolated 1.137 + 1.211 + 0.019 = 2.367 ms.
G024 already queued this as attack #3, expected **1.2–1.8 ms**,
“folding RMSNorm into the following GEMV or sharing one persistent
encoder”. Plus `gated_rmsnorm_48` 1.296 ms (attack #2, with ba/rearrange).

**Does not share the 4.4× mechanism** if RMS is a prologue of the
existing multi-TG Q4 GEMV (reduce `x` once, then the same tpr64 walk
uses `x_norm`). **Does share the kill** if “fuse norm into consumer”
means the megakernel stage A+B in one TG (`megakernel_qwen3b.metal`
stage outline A then B, 256 threads).

Existing `v1g_rmsnorm_gemv_fusion_parity.rs` is **f16w** attn GEMV,
DeepSeek-V2-Lite shapes, not Qwen3.8 Q4. Cannot ship that kernel as
the G1 consumer.

**Cost (PROJECTED, G024’s own numbers, isolated not production-fused):**
- 129 RMSNorms: 1.2–1.8 ms.
- gated_rmsnorm + ba + rearrange: 1.0–1.5 ms (cannot go below rec-state
  0.47 ms already in `kv_state`).
- GQA rope 1.563 ms is the alternate if norms are left alone.
- Together if both fold: **2.2–3.3 ms** PROJECTED.

This is the only fusion class with millisecond-class expectation.
It is still not 25 ms.

**Cheapest falsifier:** one layer, fused `rmsnorm+geo_tpr64` vs
two-dispatch, then 64-layer isolated family. If fused GEMV occupancy
drops below ~400 GB/s on the 17408×5120 gate, **KILLS** (you spent
the 21 ms bucket to save 2 ms).

### P4. Aggregate the many tiny kernels into one

Two different mechanisms hide under this sentence.

**P4a. Fold tiny elementwise/reduction into neighboring fat Q4 kernels
without changing the GEMV TG map.** Same as P2+P3 plus residuals,
sigmoid, ba_to_decay, rearrange. Isolated sum of those families
≈ 6.17 ms raw; G024’s overlapping estimate is 2.2–3.3 ms plus
silu/residuals/sigmoid ≈ 0.45 ms → **PROJECTED ceiling ~3.8 ms**
if nothing in the fold hurts GEMV occupancy. **Does not share the
4.4× kill.**

**P4b. One megakernel / one dispatch for a layer or a token, fat GEMVs
serialized through shared memory and barriers.** This **is**
`qwen3b_megakernel_nlayer`. **Shares the failed mechanism exactly.**
On Qwen3.8 the penalty would be worse, not better: production already
saturates the honest decode ceiling (97.6%), and a 1-TG walk of
17408×5120 f16 (or even Q4) cannot. ESTIMATED: repeating the 4.4×
per-layer tax on a 33.9 ms GPU token → ~150 ms. Not run (forbidden,
and Type-1).

`dead_levers.md:18` Phase-2.2 trivial-op fusion is P4a’s llama-path
cousin (sub-noise). On this genome rope and RMS are **not** sub-noise
(1.56 ms and 2.37 ms isolated). Do not apply that kill to P3 / P4a.
Do apply it to fusing `qwen_next_add_residual` (118–134 µs isolated).

**KILLS: P4b.** **REOPEN_IF:** the `fe0fb94c5` conditions (inline Q4 +
multi-TG) plus a complete-token A/B that does not regress the 21.3 ms
addressing bucket.

### P5. Persistent kernel, TGs resident across a layer or a token

**This is the megakernel.** `MegakernelRunner` (`megakernel.rs:450+`)
was explicitly “the shape a decode loop” — upload once, one fused
dispatch per step, `n_layers` looped on device (`metal:1126–1133`).
Measured 4.4× slower per layer. **Shares the failed mechanism.**

A “persistent” design that is **not** that kernel would be: many TGs
stay resident and pull work from a device queue (CUDA-style persistent
threads). Apple GPU has no shipping equivalent in this repo. No
measurement. ESTIMATED recover cannot exceed the tiny-kernel tax
(~2–4 ms) because fat GEMVs are already occupancy-saturated. Risk of
occupancy collapse is the 4.4× hole.

**KILLS** as a re-proposal of `qwen3b_megakernel_nlayer` /
`MegakernelRunner`. **REOPEN_IF:** a multi-TG persistent work queue
that preserves `geo_tpr64_tg128` occupancy, consumes packed weights
directly, and beats the current 1-CB token on GPU timestamps. That is
new science, not this POC.

### P6. Indirect command buffers — encode once, replay

**Not a re-proposal of a known negative on this genome. Measured win.
Not on HEAD.**

`ReplayableComputeGraph` exists on HEAD (`metal/mod.rs:3569–3575`) and
is explicitly **not** wired into decode:

> This is intentionally not wired into decode selection yet. … measured
> CPU encoding share remains below the ICB ship gate.

The Qwen3.8 wiring lives on `7400acf1b` (`grok/qwen38-kill-fixed-overhead-20260816-165048`),
**not an ancestor of `2eee9a004`**. Receipts were preserved onto HEAD.

Mechanism (`7400acf1b` `qwen38_hybrid_decode.rs:163–187, 2480–2540`):
- Default schedule `IndirectCommandBuffer`.
- 64 KiB scalar slab; intern static `set_bytes` payloads once.
- Per token host writes 3 u32s: token, position, `mha_seq_len=position+1`.
- `tcb.execute_replayable_graph` → `executeCommandsInBuffer` (`metal/mod.rs:4726–4731`).
- ICB `setBarrier` on producer→consumer; omitted on independent pairs
  (gate/up, qkvz/ba, k/v after q).
- 964 commands still execute. Dispatch count does not change.

**MEASURED** (`QWEN38_FIXED_OVERHEAD_DELETED.json`, dirty, same vehicle,
coherence PASS 3-prompt id-identical):

| named piece | before mean ns | after mean ns | Δ ns |
|---|---:|---:|---:|
| encode_host_prepare | 886,200 | 90,981 | −795,219 |
| wait_minus_gpu | 425,900 | 561,994 | **+136,094** |
| submit | 10,500 | 9,420 | −1,080 |
| tokenizer | 6,300 | 6,831 | +531 |
| epilogue | 1,800 | 1,708 | −92 |
| named fixed sum | 1,330,700 | 670,934 | −660,000 |
| headline wall ms | 38.217 | 36.684 | −1.533 |
| headline GPU ms | 36.987 | 36.012 | −0.975 |
| TPS | 26.17 | 27.26 | +1.09 |

Encode fell and wait rose. Reported in the receipt, not hidden.
Net named fixed still dropped. Wall Δ includes dirty GPU movement
(36.99 → 36.01) and is **not** a clean 1.53 ms ceremony claim.
The clean ceremony claim is **−0.66 ms named fixed** (MEASURED).

**Does not share the 4.4× mechanism.** No math fusion, no occupancy
change, no f16 expand. ICB replay of the existing Q4 TG map.

**Does share the `dead_levers.md:19` ICB Type-1 name.** That kill’s
evidence is “CPU encode 0.22 ms = 0.51% wall”; resurrection is
“Encode >1 ms/tok”. Qwen3.8 encode is MEASURED 0.886–0.919 ms
(2.3–2.6% of wall) — under the 1 ms letter, above the 0.22 ms
evidence. The Qwen3.8 lane ran it anyway and measured a net win.
That is a **conditional reopen that succeeded**, not a Type-1 retest
of the 0.22 ms path.

Q30 M5 (`Q30_S_BUCKET_MECHANISM_TABLE.md:93–101`) rejected ICB as
primary because encode was 0.22–0.51% and dynamic MoE routes force
rebind. Qwen3.8 is dense; the graph is static except 3 scalars. M5
does not transfer.

**IMPLEMENT_READY:** land `7400acf1b` (or equivalent) onto the G1
genome. Expected: encode ~91 µs, named fixed ~0.67 ms, wait−gpu
higher by ~136 µs. Does not move the 21.3 ms addressing bucket.
Does not hit 10 ms TOKEN_NS.

**KILLS as a G1-primary.** Do not re-bench ICB vs encode-every-token
to “discover” the 0.66 ms again. The receipt exists.

### P7. GPU-side chaining so the host is not in the loop per layer

**Already true on HEAD, and more true on the ICB branch.**

HEAD: 1 CB, `encode_layers` is a CPU loop that records 960 dispatches
into that CB, then one `commit_and_wait` (`hybrid_decode.rs:3292–3310`,
`:2822–2829`). There is no per-layer host readback on Qwen3.8 (dense,
no route ids). This is the opposite of Q30’s 48 host route-id waits
(`Q30_S_BUCKET_MECHANISM_TABLE.md` M1).

ICB branch: host writes 3 u32s and `executeCommandsInBuffer`. Barriers
are GPU-side. Host is not in the layer loop at encode time either.

A stronger reading — **device loops tokens**, host waits once per
generate — is unmeasured. Sampler is already `sample_argmax_f32`.
Would need on-device embed gather + scalar increment. PROJECTED
recover ≤ named fixed (0.67–1.33 ms/token): the GPU body (33.9–37.0 ms)
still runs. **Does not share the 4.4× kill** if GEMVs stay multi-TG Q4.
**Does share it** if “GPU loops the token” is implemented as
`n_layers` inside `qwen3b_megakernel_nlayer`.

**KILLS** as a re-proposal of “host is the per-layer problem on Qwen3.8”.
It is not. **REOPEN_IF:** a generate-level (not token-level) persistent
graph is proposed, and the serialized lane measures wait−gpu + encode
across an N-token generate vs N independent commits.

---

## 5. Share-kill matrix

| proposal | share 4.4× megakernel kill? | share ICB Type-1 / M5? | share use_resource-as-primary (M4)? | verdict |
|---|---|---|---|---|
| P1 gate+up, Q4 geo_tpr64 | NO | NO | NO | small PROJECTED; legal |
| P1 gate+up, expand-to-float GEMV | YES (strike 1) | NO | NO | **KILLS** (binding + 4.4×) |
| P2 SwiGLU epilogue on P1 | NO | NO | NO | small PROJECTED; legal |
| P3 RMS into Q4 consumer | NO | NO | NO | 1.2–1.8 ms PROJECTED; G024 #3 |
| P3 RMS inside 1-TG megakernel | YES (strike 2) | NO | NO | **KILLS** |
| P4a fold tinies into fat Q4 | NO | NO | NO | 2–4 ms PROJECTED ceiling |
| P4b layer/token megakernel | YES (both strikes) | NO | NO | **KILLS** |
| P5 persistent TG across layer/token as shipped | YES (this *is* the kill) | NO | NO | **KILLS** |
| P6 ICB replay of current graph | NO | name only; reopen succeeded | NO | **MEASURED_WIN**, not on HEAD |
| P7 GPU barriers / 1 CB | NO (already shipped) | NO | NO | done; not a proposal |
| P7 device token loop via megakernel | YES | NO | NO | **KILLS** |
| M4 use_resource batching as S/TOKEN_NS primary | NO | NO | YES | **KILLS as primary** |

---

## 6. What ceremony deletion can and cannot do

Let `C` = encode + submit + wait−gpu + intra-CB residual.

| genome | C | wall | GPU | note |
|---|---:|---:|---:|---|
| HEAD (G024) | 1.657 ms | 35.228 ms | 33.912 ms | MEASURED dirty |
| HEAD (authority 32-tok) | 1.332 ms named | 38.217 ms | 36.987 ms | MEASURED dirty |
| ICB branch | 0.671 ms named | 36.684 ms | 36.012 ms | MEASURED dirty |
| P4a all tinies (PROJECTED) | C unchanged | wall − 2.2 to 3.8 ms | GPU − same | isolated-based |
| ICB + P4a (PROJECTED) | ~0.67 ms | ~32–34 ms | ~33 ms | still 3× the 10 ms rung |
| 1.5 BPW linear on addressing only (PROJECTED) | C unchanged | addressing 21.3 × 1.5/4.2527 ≈ 7.51 ms | — | G024-style scale; **not** a measured 1.5 BPW token |

G024 itself: at 2.0/4.2527 BPW, addressing → ~10.01 ms, recover ~11.28 ms
from the GEMV-traffic class. That is the only existential lever
(`top_three_attacks[0].not_a_kernel_win = true`).

Authority density ladder (`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json`
`density_ladder`, linear in BPW, **optimistic** — assumes GPU scales,
fixed overhead does not): 2.0 BPW → 17.97 ms / 55.6 TPS from the 38.2 ms
wall; 3.0 BPW still misses 50 TPS. Linear scale is a roof conditioned
on this genome, not a floor. It already shows fusion/persistence cannot
substitute for bytes.

**FALSIFIED:** eliminating dispatch/submission ceremony by fusion,
persistence, ICB, or GPU chaining is sufficient to reach
`TOKEN_NS <= 10,000,000` on the current Q4 genome.

**SUPPORTED:** the same tools can delete a **named 0.7–4 ms** (ICB
MEASURED 0.66 ms + tiny-kernel fusion PROJECTED 2–4 ms) without
repeating the 4.4× kill, if and only if fat GEMVs stay multi-TG Q4.

G1 100 TPS / 10 ms still requires a different physical model (BPW and
the 223×/313× DeltaNet/GQA tails), which is out of this lane’s
mechanism set.

---

## 7. Implementation order if a later lane is allowed to touch code

1. Land ICB (`7400acf1b`). Already measured. Do not re-discover.
2. Fold RMS / gated_rmsnorm / rope into the **existing** Q4 or vi
   consumer (G024 #2/#3). Component A/B first, then one complete-token
   dirty pair. Abort if gate GEMV GB/s drops.
3. Optional: Q4 gate+up+SwiGLU producer (P1+P2). Expected <0.3 ms.
4. Never: `qwen3b_megakernel_*`, f16 megakernel weights,
   `MegakernelRunner` as decode, 1-TG persistent layer/token,
   expand-to-float then generic GEMV, use_resource batching as the
   TOKEN_NS story.

---

## 8. Open measurements (serialized GPU lane owns them)

Not run here. Cheapest next experiment, in order:

1. Confirm ICB still bit-ids on current HEAD after rebase (coherence
   gate, 3 sealed prompts). Code merge, not a new mechanism.
2. Isolated fused `rmsnorm+geo_tpr64` on the 17408×5120 gate vs two
   dispatches. Kill if GB/s falls >5% of the 401.6 production figure.
3. Do **not** re-run 8-layer f16 megakernel. Type-1.

---

## Completion report

STATUS
FALSIFIED (ceremony fusion/persistence as a path to TOKEN_NS<=10e6 on this genome). ICB land is IMPLEMENT_READY as a 0.66 ms named-fixed win. 8-layer f16 megakernel and 1-TG persistence remain MEASURED_NEGATIVE / KILLS.

CLAIMS
1. The 4.4× figure is 5.55 ms/layer f16 1-TG megakernel vs 1.26 ms/layer production Q4_K TCB on M3 Pro Qwen-3B N=8, not a Qwen3.8 result. Evidence: commit `fe0fb94c5` message.
2. Failed mechanism = f16 expansion + single TG 256 looping layers. Evidence: `fe0fb94c5`; `megakernel.rs:419–422`; `megakernel_qwen3b.metal:1093–1133`.
3. `use_resource` 2.62 µs is faster than `set_buffer` 4.5 µs (component). Evidence: `a9a2f980e`. Rejected as TOKEN_NS primary. Evidence: `Q30_S_BUCKET_MECHANISM_TABLE.md:77–88`.
4. HEAD Qwen3.8 is 1 CB / 964 dispatches, encode 0.886–0.919 ms, GPU 97.6% of 411.51 GB/s, addressing 21.3 ms. Evidence: `qwen38_hybrid_decode.rs:3292–3310`; `schedule.rs:12–54`; G024; authority.
5. Gate and up are unfused Q4 GEMVs. Evidence: `schedule.rs:41–47`; isolated mlp_gate/up 4.923 / 5.178 ms in `qwen38-layer-dense-q4-swiglu.json`.
6. P1/P2 PROJECTED recover 0.05–0.30 ms and 0.16–0.25 ms if Q4-native. Do not share 4.4×. Expand-to-float shares it.
7. P3/P4a PROJECTED 2.2–3.8 ms. G024 attacks #2/#3. Do not share 4.4× iff TG map preserved.
8. P4b/P5 share the 4.4× kill. KILLS. REOPEN_IF inline Q4 + multi-TG + complete-token not-slower.
9. P6 ICB MEASURED encode 886→91 µs, named fixed 1.331→0.671 ms, wait−gpu +136 µs, wall 38.217→36.684 ms, ids bit-identical. Code not on HEAD (`7400acf1b` not ancestor of `2eee9a004`). IMPLEMENT_READY to land. Not a 10 ms path.
10. P7 “host out of the layer loop” is already true (1 CB, no route wait). Re-proposing it as the wall is false. Evidence: `hybrid_decode.rs:2822–2829`, `:3292–3310`.
11. Ceremony ≤1.66 ms. 10 ms TOKEN_NS is FALSIFIED as a fusion/persistence outcome on this Q4 genome. Evidence: G024 ranked table + authority named fixed.

EVIDENCE
- `git log -1 --format=full fe0fb94c5395e58bc5ad30b51b9d7ded406b81b1` (4.4× + two strikes)
- `git log -1 --format=full a9a2f980effec5b600bd1b8a2fdd77069dffad7f` (2.62 vs 4.5 µs)
- `crates/hawking-core/shaders/megakernel_qwen3b.metal:1093–1133` (1 TG, n_layers loop)
- `crates/hawking-core/src/kernels/megakernel.rs:328–333, 419–422, 506–540`
- `crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12–54`
- `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:2822–2829, 3292–3310`
- `crates/hawking-core/src/metal/mod.rs:3569–3575, 4726–4731`
- `receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`
- `receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`
- `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json`
- `receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json`
- `receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json` `isolated.mlp_*`
- `receipts/q30-dispatch-gap/Q30_S_BUCKET_MECHANISM_TABLE.md` M3/M4/M5
- `workspace/docs/guides/dead_levers.md:17–19, 25`
- `7400acf1b:crates/hawking-core/src/model/qwen38_hybrid_decode.rs:163–187, 2480–2540`
- `git merge-base --is-ancestor 7400acf1b HEAD` → exit 1 (ICB code not on HEAD)

CHANGES
- created `workspace/superwave/g1/g1-fusion-persistent.md` only

TESTS
- `test -s workspace/superwave/g1/g1-fusion-persistent.md` (required; run at end of this lane)
- `wc -l workspace/superwave/g1/g1-fusion-persistent.md` (required)
- `git status --porcelain` (required; exactly one new untracked path)
- no GPU, no cargo test, no inference (forbidden)

RISKS
- G024 and authority are different dirty sessions; mixing them into one TOKEN_NS is a lie. This file keeps them separate.
- PROJECTED fusion ns come from isolated CBs. G024 confidence on that is MEDIUM.
- ICB wall Δ includes GPU movement; the ceremony claim is the named-fixed Δ only.
- Linear BPW scale in the authority density ladder is a genome-conditioned roof, not a measured 1.5 BPW token.

UNRESOLVED
- Occupancy is launch-geometry derived, not a hardware counter (G024 unresolved).
- ICB rebase bit-id on `2eee9a004` not verified (no GPU this lane).
- Generate-level (multi-token) device loop unmeasured.
- No Q4 `rmsnorm+geo_tpr64` fused kernel exists to time.

NEXT
- Serialized GPU lane: rebase/land `7400acf1b`, coherence-gate, do not re-profile ICB vs encode as if new.
- After that: isolated fused rmsnorm+Q4 gate A/B (P3 falsifier). Abort on GB/s drop.
- Do not resurrect `megakernel_qwen3b.metal` as a G1 decode vehicle.
