# G1 residency / reuse — Qwen3.8 on this M3 Ultra

Lane: `19-residency-reuse`. Write scope: this file only. No GPU run, no inference, no artifact, no live-Genesis touch.

Hypothesis under test: *weight bytes need not be re-read from DRAM every token* (cache residency, persistent no-rebind layout, cross-token reconstruction cache, or multi-stream amortize).

**STATUS: FALSIFIED** for any mechanism that would change single-stream `TOKEN_NS` by stopping the weight DRAM stream. The re-read is unique-once DRAM traffic of a 13.611_663_360 B GEMV payload against a measured ~64 MiB cache-hot cliff. Host rebinding is a ≤1 ms tax, not the stream.

Label key: `THIS_LANE` = command/source/arithmetic here. `RECEIPT_CLAIMED` = prior receipt, not re-timed this lane. `PROJECTED` = scale from a receipt under a stated assumption.

---

## 0. Hardware (THIS_LANE)

```
machdep.cpu.brand_string: Apple M3 Ultra
hw.memsize: 103079215104          # 96 GiB
hw.ncpu / physicalcpu / logicalcpu: 28 / 28 / 28
hw.perflevel0 (P): 20 cpus; L1I 196608; L1D 131072; L2 16777216; cpusperl2 5
hw.perflevel1 (E):  8 cpus; L1I 131072; L1D  65536; L2  4194304; cpusperl2 4
SPDisplaysDataType: Apple M3 Ultra, 60 GPU cores, Metal Supported
```

sysctl does **not** expose GPU L1 / GPU L2 / SLC capacity. CPU L2 (4×16 MiB P + 2×4 MiB E = 72 MiB) is not the GPU hierarchy. Cache-effective GPU size below is taken from the Q4 size-sweep receipt, not from a datasheet.

Published peak **819 GB/s** appears in `receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json` `hardware.published_peak_gb_s`. Not re-measured here.

---

## 1. Genome on HEAD `2eee9a004`

Dense Qwen3.8-27B (`qwen3_5` text). Not MoE. Hybrid 48 Gated-DeltaNet + 16 GQA, interval 4.

Source constants: `crates/hawking-core/src/model/qwen38_geometry.rs:20-42`

| field | value |
|---|---:|
| layers / hidden / intermediate | 64 / 5120 / 17408 |
| vocab, untied | 248320 |
| DN heads (K/V), d | 16 / 48, 128 |
| GQA heads (Q/KV), d | 24 / 4, 256 |

Production decode: `Qwen38HybridDecodeSession` + `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`. Default `concurrent_independent=false`, `deltanet_vi_parallel=true`. `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:232-236,856-913`.

Weights are **already process-resident** `MTLBuffer`s (`Qwen38HybridWeights::load` uploads codes+scales once; `attach` only allocates workspace/KV). `resident_bytes()` sums buffer lengths (`:676-685`). Sharing is `Arc` (`attach` at `:883-914`, `share_weights` at `:916-918`).

Per-token `step` **re-encodes the whole graph**:

```3292:3310:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            ...
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
```

Each Q4 GEMV rebinds codes/scales/x/y and `set_bytes` rows/cols/groups (`:1569-1590`).

Dispatch census (THIS_LANE, from `encode_deltanet` / `encode_gqa` / `encode_dense_mlp` / embed / terminal):

```
DN layer  = rms + qkvz + ba + rearr + ba_to_decay + gated_delta + gated_rmsnorm + out + residual = 9
GQA layer = rms + q + k + v + rope + mha + sigmoid + o + residual = 9
MLP       = rms + gate + up + silu + down + residual = 6
token     = 1 + 48*9 + 16*9 + 64*6 + 3 = 964
```

Matches RECEIPT_CLAIMED `964` in `TOKEN_NS_QWEN38.json` / `G024_QWEN38_TOKEN_NS.json` (`kernel_runtime_genome`: "1 CB / 964 dispatches").

Uniform Q4 body: 4-bit codes + f16 scale / 64 els = **4.25 BPW** (`UNIFORM_Q4_NOMINAL_BPW`, `crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:48`). Complete-catalog physical **4.252735126866492** RECEIPT_CLAIMED (`QWEN38_COMPLETE_TOKEN_WALL.json` `vehicle.complete_physical_bpw`; `G024` `measurement.bpw`).

---

## 2. Working set vs GPU cache

### 2.1 Bytes that move every token (DERIVED, matches receipt)

Q4 payload `rows*cols/2 + rows*(cols/64)*2` (`UNIFORM_Q4_GROUP_SIZE=64`). Geometry GEMV total **excluding embed**:

| organ | bytes | MiB |
|---|---:|---:|
| mlp gate or up (17408×5120) | 47_349_760 | 45.156 |
| mlp down (5120×17408) | 47_349_760 | 45.156 |
| DN qkvz (16384×5120) | 44_564_480 | 42.500 |
| DN ba (96×5120) | 261_120 | 0.249 |
| DN/GQA out (5120×6144) | 16_711_680 | 15.938 |
| GQA q (12288×5120) | 33_423_360 | 31.875 |
| GQA k or v (1024×5120) | 2_785_280 | 2.656 |
| lm_head (248320×5120) | 675_430_400 | 644.141 |
| **64×MLP + 48×DN + 16×GQA + lm_head** | **13_611_663_360** | **12981.6** |

This equals `HONEST_ROOF_WEIGHT_ADDRESSING.json` `denominator_correction.correct_attribution.bytes` exactly (delta 0).

Per-layer GEMV:

| layer | bytes | MiB |
|---|---:|---:|
| MLP | 142_049_280 | 135.47 |
| DeltaNet | 61_537_280 | 58.69 |
| GQA | 55_705_600 | 53.13 |

Embed table is another 675_430_400 B but only **one row** (5120 Q4 groups → 2_720 B) is gathered (`QWEN38_ACTIVE_BUDGET_MEASURED.json` `embed_excluded_why`; `TOKEN_NS_QWEN38.json` `unattributed_residual.bytes_read=2720`).

Manifest class sum RECEIPT_CLAIMED (`QWEN38_ACTIVE_BUDGET_MEASURED.json` `by_class_bytes`):

| class | bytes |
|---|---:|
| mlp | 9_091_161_600 |
| linear_attn | 2_961_704_064 |
| full_attn | 891_325_184 |
| lm_head | 675_430_440 |
| embed_table | 675_430_440 |
| norms | 2_642_952 |
| active (no embed table) | 13_622_264_240 |

The 10.6 MB gap vs 13_611_663_360 is headers + non-GEMV mixer tensors (honest roof `adjudication.three_byte_counts`). **Defended GEMV stream is 13.611 GB.**

Workspace (state, not weights) at `max_seq_len=128` RECEIPT_CLAIMED (`QWEN38_SHARED_SESSIONS.json` `workspace`): act 1_691_396 + DN state 156_893_184 + GQA KV 16_777_216 = **175_361_796 B**. Formula: `qwen38_workspace_bytes` (`qwen38_hybrid_decode.rs:331-400`). DN recurrent f32 is 48×48×128×128×4 = 150_994_944 B (reuse-matrix `state_budget`); plus conv.

### 2.2 Cache-effective size (RECEIPT_CLAIMED, GPU_PROTECTED_CPU_CONTENDED)

`HONEST_ROOF_WEIGHT_ADDRESSING.json` single-GEMV `addr_probe` (load-only, same kernel/launch as production):

| payload | median GB/s | median ns |
|---|---:|---:|
| 64 MiB (67_107_840 B) | **817.14** | 82_125 |
| 128 MiB | **675.30** | 198_750 |
| 256 MiB … 13.612 GB | 692–700 | — |
| 13.612 GB | **699.57** | 19_457_084 |

64 MiB hits published peak (817 vs 819). 128 MiB is already on the DRAM plateau. That is the measured cache-hot cliff for this access pattern.

Full kernel (addr+decode+FMA) at 64 MiB is **608.00 GB/s** vs **666.68 GB/s** at 13.612 GB. Cache-sized *complete* GEMV is occupancy-limited, not faster per byte.

`unique_once` sequential-reduce plateaus ~375 GB/s from 2 GiB through 13.6 GB (64 MiB: 268 GB/s). It is **not** the Q4 GEMV ceiling. G024's "97.6% of 411.51" is **superseded** (wrong bytes, wrong time, wrong control). Honest roof: sealed addressing **639.25 GB/s** = 13_611_663_360 / 21_293_102.5 ns = **91.4% of the 699.57 GB/s kernel roof**.

Tiled production-organ (47 MB gates, 287 dispatches covering 13.59 GB): full **551.59 GB/s**, addr **591.13 GB/s**. Tiling through the cliff does **not** beat one large stream.

### 2.3 Fraction that fits

```
cache-hot 64 MiB / GEMV working set = 67_107_840 / 13_611_663_360 = 0.004930  (0.493%)
working set / cliff                 = 202.83 ×
one MLP layer / cliff               = 2.117 ×   (does not fit)
one DN layer  / cliff               = 0.917 ×   (barely)
one GQA layer / cliff               = 0.830 ×   (barely)
lm_head alone / cliff               = 10.07 ×
```

**~0.5% of the per-token GEMV working set sits in the cache-hot regime. ~99.5% is DRAM.**

Even if one DN/GQA layer's GEMVs momentarily fit, the next MLP (135 MiB) plus the remaining 63 layers (~13.5 GB) evict them before the next token revisits that layer. Autoregressive layer-major decode cannot keep layer L hot across tokens.

A genome whose entire GEMV stream fit in 64 MiB would be 64/13611 ≈ 0.49% of current bytes ≈ **0.021 BPW** if uniform. That is not a representation; it is a different model.

**KILLS:** "the working set fits in the GPU cache hierarchy."
**REOPEN_IF:** a *clean* size sweep moves the addr_probe cliff to ≫64 MiB. That would change layer-fit, not the 13.6 GB conclusion. Cheapest experiment: rerun `honest_roof` 64/128/256 MiB points with the GPU lock and no CPU builds (`timing_label` today is `GPU_PROTECTED_CPU_CONTENDED`, `clean_box=false`).

---

## 3. Persistent resident layout, no per-token rebinding

### 3.1 What is already resident on HEAD

| object | when paid | per-token? |
|---|---|---|
| Q4 codes+scales, f32 mixer/norm | `Qwen38HybridWeights::load` | no (handle reuse) |
| workspace / KV / DN state | `attach` | no alloc; contents change |
| pipelines | first dispatch | no |
| command list | **`step` re-encodes 964 encoders** | **yes** |

Weights are not re-uploaded. They **are** re-bound (`set_buffer` + `set_bytes`) every token. 222 `set_buffer`/`set_bytes` call sites in `qwen38_hybrid_decode.rs` (THIS_LANE `rg -c`).

### 3.2 ICB replay — implemented, measured, **not on this branch**

`7400acf1b` (`qwen38-kill-fixed-overhead: ICB replay, encode 886us -> 91us`) is **not** an ancestor of HEAD:

```
git merge-base --is-ancestor 7400acf1b HEAD  → exit 1
```

Only branch: `grok/qwen38-kill-fixed-overhead-20260816-165048`.

That commit's `step` writes 3 u32s and `execute_replayable_graph`:

```
write_u32(scalars, token_off, token);
write_u32(scalars, position_off, position);
write_u32(scalars, seq_len_off, position+1);
tcb.execute_replayable_graph(&replay.graph)
```

964 stages interned once into `ReplayableComputeGraph`. Scalar slab 64 KiB. Default `HAWKING_QWEN38_SCHEDULE=icb`.

HEAD metal substrate exists but is explicitly **not** wired into decode (`crates/hawking-core/src/metal/mod.rs:3569-3574`): "intentionally not wired into decode selection yet… CPU encoding share remains below the ICB ship gate." ICB cannot capture `set_bytes` (`:3443-3446`); scalars must live in buffers.

RECEIPT_CLAIMED (`QWEN38_FIXED_OVERHEAD_DELETED.json`, DIRTY_ENGINEERING, 6 warm A/B reps):

| named host | before ns | after ICB ns |
|---|---:|---:|
| encode_host_prepare | 886_200 | 90_981 |
| wait_minus_gpu | 425_900 | 561_994 (+136_094; reported) |
| submit | 10_500 | 9_420 |
| tokenizer | 6_300 | 6_831 |
| epilogue | 1_800 | 1_708 |
| **named fixed sum** | **1.331 ms** | **0.671 ms** |
| complete wall | 38.217 ms | 36.684 ms |
| GPU | 36.987 ms | 36.012 ms |

Single-stream TOKEN_NS win: **~1.53 ms** (~4% of wall). GPU still 98.2% of wall.

### 3.3 Bind-location is not addressing (RECEIPT_CLAIMED, unmerged)

`f0f8dbd28` also not on HEAD (exit 1). Receipt `AUTO_QWEN38_WEIGHT_ADDRESSING_TRY14.json`:

- hypothesis `host_gpu_partition_of_addressing` → **REJECT**
- serial-group vs host production GPU median 36_277_791 vs 36_282_833 (**−5_042 ns**)
- isolated mlp addr_probe ICB+barriers 12_829_000 vs host 12_733_416 (**+96 µs**)
- isolated mlp **concurrent ICB, no barriers** 11_440_583 (**−1.293 ms**) — 192 GEMVs overlapping; **illegal on production** (layer data dependence)
- limiter: `unique_once_q4_dram`, **not** host bind / encoder cuts / ICB bind location / address ALU

**Achievable:** yes, persistent no-rebind layout (ICB + scalar slab). Weights already resident.
**Does it stop the DRAM re-read?** No.
**Does it move G1 TOKEN_NS (37.9 ms → 10 ms)?** No. Ceiling of deleting *all* remaining named host after ICB is ~0.67 ms.

**KILLS:** "rebinding is why weights are re-read."
**REOPEN_IF:** ICB graft is still a cheap ~0.8 ms encode deletion if wait-minus-gpu rise is killed. Not a G1 closer.

---

## 4. Cross-token reconstruction cache

Production path consumes packed Q4 **in-register** (`geo_tpr64_tg128`). Mixed path: "Packed bytes stay packed. … no reconstruct-to-Q4 path" (`qwen38_hybrid_decode.rs:1-7`).

Two receipts, different scales:

| scale | claim | source |
|---|---|---|
| occupancy-tile GEMV | recon_excess_ns = 0 on 32/33 codecs at tpr64; q4/q3/q2/binary match f32 15.1 µs | `QWEN38_RECONSTRUCTION_IS_FREE.json`, `QWEN38_RECON_MEASURED.json` — **component microbench, not token** |
| complete token | `weight_decode_reconstruction` = 1_808_227 ns (5.13% of 35.228 ms) = decode_probe − addr_probe on isolated class GEMVs | `TOKEN_NS_QWEN38.json` / `G024` |

Honest roof single-GEMV at 13.6 GB: addr 699.57, decode 683.80, full 666.68 → **ALU+decode tax 4.70%**. Reconstruction is a small tax on a bandwidth-saturated load.

If one cached *reconstructed f32* across tokens:

- language f32 = 26_895_998_464 × 4 = **107.58 GB** (does not fit in 96 GB)
- MLP alone f32 ≈ 68.45 GB vs packed MLP 9.091 GB (**7.53×**)
- next token would *read more* bytes (f32) than packed Q4
- only wins if unpack ALU ≫ extra DRAM. Measured unpack is 4.7% of GEMV. Extra ~8× bytes would be ~8× GEMV time.

What is actually token-invariant and already resident: packed weights, scales, pipelines, geometry scalars, RMSNorm weights (2.6 MB), embed table (gather one row). Nothing expensive-to-reconstruct remains except the weight elements themselves, which the kernel unpacks into registers and consumes.

What *does* change every token (must not be "cached" as static): hidden, DN S (layer-wise 3.0 MiB, total 144 MiB), conv state, GQA KV (+128 KiB/token), `token` / `position` / `seq_len`, sampled id.

**KILLS:** cache reconstructed weights across tokens as a TOKEN_NS lever.
**REOPEN_IF:** a codec whose *in-register* unpack exceeds the GEMV load at production launch *and* whose reconstructed form is *smaller* than packed (impossible for expand-to-float). Component tile "recon is free" is not a token claim.

---

## 5. Real cost of the per-token bind / address ceremony

### 5.1 Closed token ledger (RECEIPT_CLAIMED, DIRTY_ENGINEERING)

`TOKEN_NS_QWEN38.json` / `G024_QWEN38_TOKEN_NS.json`. Wall 35_227_917 ns, GPU 33_912_333 ns, vehicle `uniform-q4-v1`, 0 fallbacks, greedy 16-id identical. Sum of 12 components = wall ±1 ns.

| rank | component | ns | % wall | class |
|---|---|---:|---:|---|
| 1 | weight_addressing | 21_293_103 | 60.44 | GPU DRAM unique-once |
| 2 | deltanet | 3_732_795 | 10.60 | GPU ALU/launch |
| 3 | gqa | 2_443_471 | 6.94 | GPU ALU/launch |
| 4 | normalization | 2_367_415 | 6.72 | GPU launch |
| 5 | weight_decode_reconstruction | 1_808_227 | 5.13 | GPU ALU on same load |
| 6 | dense_swiglu | 1_004_198 | 2.85 | GPU |
| 7 | **host_preparation** | **919_250** | **2.61** | **CPU encode 964 encoders** |
| 8 | kv_state | 537_665 | 1.53 | GPU stream |
| 9 | synchronization | 384_250 | 1.09 | wait−gpu |
| 10 | terminal_head FMA | 383_535 | 1.09 | GPU |
| 11 | unattributed (embed 4999 + intra-CB 341925) | 341_925 | 0.97 | GPU |
| 12 | command_submission | 12_084 | 0.03 | CPU commit |

Pre-ICB complete-wall authority (`QWEN38_COMPLETE_TOKEN_WALL.json`): encode mean 886_210 ns (2.31% of 38.217 ms; 66.5% of wall−gpu). After ICB: encode 90_981 ns; named fixed 0.671 ms.

**Addressing is not bind ceremony.** It is the GEMV load (13.611 GB at 639 GB/s sealed / 700 GB/s kernel roof). Addr_probe is 83–92% of every GEMV class (G024). TRY14 isolated host encode of mlp catalog ~0.10 ms, "not inside GPU addressing."

### 5.2 Ceremony vs G1

| delete | TOKEN_NS left (PROJECTED from G024 35.228 ms) | hits 10 ms? |
|---|---:|---|
| all host encode+submit+sync (1.315 ms) | 33.91 ms | no |
| that + ICB-already-claimed 1.53 ms | ~33.7 ms | no |
| all non-addressing GPU (13.12 ms) + all host | **21.29 ms addressing alone** | no |
| addressing at 819 GB/s datasheet (PROJECTED) | 16.62 ms | no |
| addressing at 699.57 GB/s kernel roof | 19.46 ms | no |
| 13.611 GB in 10 ms requires | **1361 GB/s** | exceeds 819 |

Binding/addressing *ceremony* is not the wall. The *bytes* are.

---

## 6. Multi-stream amortize

### 6.1 What was measured (independent GEMVs, not GEMM)

`QWEN38_SHARED_SESSIONS.json` (lock held, `uniform-q4-v1`, seq=128, 4 sessions, `Arc` ptr_eq true, resident weights 14_297_675_776 B):

| quantity | value |
|---|---:|
| 1-session tok/s | 26.653 |
| 1-session median GPU | 36_099_333 ns |
| 4-session aggregate tok/s | **9.427** (worse) |
| 4-session per-stream wall | 107–111 ms/token |
| lm_head 1× GPU | 1_013_791 ns |
| lm_head 4× serial | 4_144_541 ns = 4.09× |
| lm_head 4× concurrent encoder | 4_022_124 ns = 3.97× |

`measure_shared_weight_fanout` (`qwen38_hybrid_decode.rs:3541-3585`) issues **N separate GEMVs** against the same codes/scales. Concurrent encoder may overlap *execution*; it does not fuse into a weight-stationary GEMM. 3.97× for N=4 is "no amortize."

4 in-process decode threads on one command queue **slow each other** (26.7 → 9.4 aggregate tok/s). Genesis body (`GENESIS_RESIDENT_BODY.md`): "Concurrent decode ceiling remains 1; do not add sessions to chase tokens/s." Process-pool children do **not** share artifact pages.

### 6.2 What a real amortize would be

Weight-stationary GEMM: read each Q4 group once, apply to N activations.

AI of one GEMV: 2 flops/el ÷ (34 B / 64 el) = **3.765 FLOP/byte** (THIS_LANE). Batch N scales AI ≈ N (weight-dominated).

No Q38 GEMM kernel exists on HEAD. `concurrent_independent` overlaps *different* weights (gate+up, qkvz+ba, q/k/v) — more DRAM streams, not one stream reused. Default OFF.

TRY14 concurrent ICB without barriers on an *isolated catalog* of 192 independent MLP GEMVs gained 1.29 ms. That overlap is **illegal** across production layers (hidden residual is data-dependent). Not a token lever.

### 6.3 Single-stream vs throughput

A legal N-activation GEMM across **distinct decode streams** raises **aggregate tok/s**. It does **not** shorten stream-0's complete-token time unless stream-0 *waits* to form the batch, which **increases** TOKEN_NS.

A single stream cannot legally produce N future activations (autoregressive). MTP tensors in this checkpoint = 0 (`QWEN38_REUSE_MATRIX.json` `mtp_tensors_in_this_checkpoint`). Speculative decoding is a different mechanism, not present.

**KILLS:** "N concurrent decode streams amortize the same weight read" on the *current* genome (independent GEMVs), for single-stream TOKEN_NS.
**THROUGHPUT-ONLY** (unimplemented): true GEMM across streams.
**REOPEN_IF:** (1) a packed-Q4 GEMM kernel whose N=k GPU time is ≪ k× GEMV, measured on this box; **and** (2) a single-stream source of k activations that does not wait on the critical path (spec/MTP). (1) alone is not a TOKEN_NS win.

---

## 7. What changes TOKEN_NS vs what only changes throughput

| mechanism | single-stream TOKEN_NS | aggregate tok/s | evidence |
|---|---|---|---|
| Fewer weight bytes (BPW) | **yes** — existential | yes | G024 rank 1; 21.29 ms addressing |
| ICB / no per-token rebind | **yes, ~0.8–1.5 ms** | same | ICB receipt; not on HEAD |
| Fuse RMSNorm / DN tails | **yes, ~1–3 ms PROJECTED** | same | G024 ranks 2–3; not reuse |
| Cache reconstructed f32 | **no** (worse bytes) | no | §4 |
| Fit working set in GPU cache | **no** (0.49% fits) | no | §2 |
| Share `Arc` weights | no (already) | memory only | shared-sessions |
| N independent GEMVs / N threads | **no, worse** | **worse** | 26.7 → 9.4 tok/s |
| Weight-stationary GEMM, N streams | **no** (unless you wait) | **yes if built** | unimplemented; fanout 3.97× |
| Prefill GEMM along sequence | not decode TOKEN_NS | prefill only | out of scope |

G0 claims in the contract (unverified here): BPW 4.2527 / TPS 26.4 / TOKEN_NS 37.9e6. Closest receipts: complete-wall 38.217 ms / 26.17 TPS; G024 35.228 ms / 33.912 ms GPU; ICB-after 36.684 ms / 27.26 TPS. All DIRTY_ENGINEERING.

G1 `TOKEN_NS <= 10e6` at current bytes is **impossible on the measured kernel roof** (19.46 ms addressing floor at 699.57 GB/s; 16.62 ms at 819 datasheet). Linear BPW scale of *addressing only* to fit 10 ms at 699.57 GB/s: 6.996e9 / 13.612e9 × 4.2527 ≈ **2.19 BPW**, and that ignores the other 14 ms of the token. Linear scale of *entire GPU* 33.912 ms × (1.5/4.2527) ≈ **12.0 ms** (PROJECTED; still >10 ms). Density is necessary and may not be sufficient.

---

## 8. KILLS / REOPEN_IF

| ID | verdict | REOPEN_IF |
|---|---|---|
| K1 | Working set in GPU cache | FALSIFIED (0.49%). Cliff ≫64 MiB on a *clean* sweep. |
| K2 | Rebind causes the re-read | FALSIFIED. Weights resident; bind ≤1 ms; TRY14 bind-location REJECT. |
| K3 | ICB / persistent graph | ACHIEVABLE, ~1.5 ms TOKEN_NS, **not on HEAD**. Graft is ceremony, not G1. |
| K4 | Cache reconstructed weights | KILLS. 7.5× bytes; unpack is 4.7%. |
| K5 | N streams amortize one GEMV | KILLS on current genome (3.97×). GEMM+single-stream batch is a different invention. |
| K6 | Ceremony is the 10 ms path | KILLS. Addressing alone is 21.29 ms. |

---

## 9. Cheapest next experiments (not run; GPU lane owns them)

1. Clean honest-roof 64/128/256 MiB addr+full (confirm cliff). Does not change 13.6 GB.
2. Graft ICB from `7400acf1b` and re-close complete wall (expect ~0.8–1.5 ms, bit-identical). Optional.
3. Packed-Q4 GEMM N=2,4,8 vs N GEMVs on one organ (lm_head 675 MB). Reports **throughput**. Only becomes TOKEN_NS if a single-stream batch source exists.
4. Do **not** re-run independent-session fanout; 3.97× is enough.

---

## 10. Evidence appendix (commands / excerpts)

### 10.1 This-lane hardware

```
machdep.cpu.brand_string: Apple M3 Ultra
hw.memsize: 103079215104
hw.ncpu: 28
hw.physicalcpu: 28
hw.perflevel0.physicalcpu: 20
hw.perflevel0.l1icachesize: 196608
hw.perflevel0.l1dcachesize: 131072
hw.perflevel0.l2cachesize: 16777216
hw.perflevel0.cpusperl2: 5
hw.perflevel1.physicalcpu: 8
hw.perflevel1.l1icachesize: 131072
hw.perflevel1.l1dcachesize: 65536
hw.perflevel1.l2cachesize: 4194304
hw.perflevel1.cpusperl2: 4
SPDisplays: Chipset Model Apple M3 Ultra; Total Number of Cores 60; Metal Supported
```

`ps` of live Genesis: `operation not permitted` in this sandbox. Not used.

### 10.2 Ancestry

```
git merge-base --is-ancestor 7400acf1b HEAD   → exit 1   # ICB not on HEAD
git merge-base --is-ancestor f0f8dbd28 HEAD   → exit 1   # TRY14 not on HEAD
```

### 10.3 Honest roof (receipt fields)

`receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json`

- `timing_label`: GPU_PROTECTED_CPU_CONTENDED
- `clean_box`: false
- `date`: 2026-08-17
- `hardware.gpu_cores`: 60; `published_peak_gb_s`: 819.0
- `verdict.sealed_weight_addressing_gb_s`: 639.2522348492898
- `verdict.measured_q4_addr_kernel_roof_gb_s`: 699.5736545106142
- `verdict.sealed_over_kernel_roof`: 0.9137740261194911
- `q4_single_gemv_addr_probe[64mib].spread.median_gb_s`: 817.1426484018265
- `q4_single_gemv_addr_probe[128mib].spread.median_gb_s`: 675.2990188679245
- `q4_single_gemv_full[64mib].spread.median_gb_s`: 607.998550396376
- `q4_single_gemv_full[gemv_payload_13p612gb].spread.median_gb_s`: 666.6814921907636
- `unique_once_sweep[gemv_payload_13p612gb].spread.median_gb_s`: 375.6517695934827
- `denominator_correction.correct_attribution.bytes`: 13611663360
- `denominator_correction.correct_attribution.time_ns`: 21293102.5

### 10.4 Token ledger

`receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json`

- `TOTAL_TOKEN_NS`: 35227917
- `TOTAL_GPU_BUSY_NS`: 33912333
- `components[host_preparation].ns_per_token`: 919250.0
- `components[weight_addressing].ns_per_token`: 21293102.524500456
- `components[weight_addressing].bytes_read`: 13611663360
- `components[weight_decode_reconstruction].ns_per_token`: 1808227.3508656735

`receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json` `claim`: "Weight addressing is the only existential bucket (21.293 ms, 60.4% of wall). It is DRAM traffic… the lever is BPW." Ceiling sentence citing 411.51 is superseded by §10.3.

### 10.5 Shared sessions

`receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json`

- `weights_ptr_shared`: true
- `resident_weight_bytes`: 14297675776
- `baseline_1_session.tokens_per_s`: 26.653226333067778
- `parallel_n_session.aggregate_tokens_per_s`: 9.42718104951944
- `fanout_1.gpu_ns`: 1013791
- `fanout_n_serial.gpu_ns`: 4144541
- `fanout_n_concurrent.gpu_ns`: 4022124

### 10.6 ICB / TRY14

`receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json` `verification_pasted.complete_wall_stdout`: COMPLETE_WALL_MS_PER_TOKEN 36.683916; ENCODE_HOST_PREPARE_NS 90981; NAMED_FIXED_SUM_MS 0.670934; ICB_COMMANDS 964.

`f0f8dbd28:receipts/ascent-2026-08-16/AUTO_QWEN38_WEIGHT_ADDRESSING_TRY14.json` `result`: REJECT; `killed_hypothesis`: host_gpu_partition_of_addressing; `production_gpu_ns_per_token.serial_minus_host`: -5042.

### 10.7 HEAD step / bind

See §1 citations: `qwen38_hybrid_decode.rs:3292-3310`, `:1569-1590`, `:856-913`.

### 10.8 Arithmetic (THIS_LANE python)

```
token dispatches = 1 + 48*9 + 16*9 + 64*6 + 3 = 964
GEMV total no embed = 13611663360  (delta vs honest roof: 0)
64MiB/ws = 0.0049301718845917745
mlp_layer/cliff = 2.1167315175097277
AI = 128/34 = 3.764705882352941
13.611GB @ 699.5736545106142 GB/s = 19457084 ns
13.611GB in 10 ms needs 1361.166336 GB/s
```

---

## Completion report

```
STATUS
FALSIFIED

CLAIMS
C1. Per-token GEMV working set is 13_611_663_360 B (embed excluded). THIS_LANE geometry Q4 byte sum; equals HONEST_ROOF bytes (delta 0).
C2. Measured cache-hot cliff for this Q4 addr_probe is ~64 MiB (817 GB/s) vs 128 MiB (675 GB/s). RECEIPT_CLAIMED honest roof. Fraction in cache-hot = 0.493%. One MLP layer is 2.12× the cliff.
C3. Weights are already process-resident MTLBuffers. HEAD `step` still re-encodes 964 dispatches and rebinds every GEMV. THIS_LANE source.
C4. Persistent no-rebind (ICB) is achievable and was measured at encode 886→91 µs, wall 38.22→36.68 ms. Not on HEAD (`7400acf1b` ancestor exit 1). Single-stream win ~1.5 ms. Does not stop DRAM re-read.
C5. Host bind/ICB bind-location is not addressing (TRY14 REJECT, −5 µs production GPU). RECEIPT_CLAIMED, commit not on HEAD.
C6. Caching reconstructed f32 across tokens KILLS: 7.5× MLP bytes, 107 GB language f32, unpack tax 4.7% of GEMV. Tile "recon is free" is a component microbench.
C7. N independent streams do not amortize a weight read (lm_head 4× concurrent = 3.97× GPU; 4 sessions 9.43 vs 26.65 tok/s). THROUGHPUT-ONLY GEMM is unimplemented and would not cut single-stream TOKEN_NS.
C8. Addressing ceremony (host encode) is 0.92 ms / 2.61% of the 35.23 ms G024 wall. Addressing *traffic* is 21.29 ms / 60.4%. Deleting ceremony cannot reach 10 ms. 13.611 GB in 10 ms needs 1361 GB/s > 819 datasheet.

EVIDENCE
sysctl / SPDisplays (this lane).
crates/hawking-core/src/model/qwen38_geometry.rs:20-42
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:232-236,331-400,676-685,856-913,1569-1590,3292-3310,3541-3585
crates/hawking-core/src/metal/mod.rs:3443-3446,3569-3574
crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:48
receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json
receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.md
receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json
receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json
receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json
receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json
receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json
receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json
receipts/ascent-2026-08-16/QWEN38_RECONSTRUCTION_IS_FREE.json
receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json
receipts/ascent-2026-08-16/QWEN38_REUSE_MATRIX.json
receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.md
git show 7400acf1b:crates/hawking-core/src/model/qwen38_hybrid_decode.rs (encode_token_icb)
git show f0f8dbd28:receipts/ascent-2026-08-16/AUTO_QWEN38_WEIGHT_ADDRESSING_TRY14.json
python Q4 byte sum / 964 dispatch census / AI / 10 ms GB/s (this lane)

CHANGES
workspace/superwave/g1/g1-residency-reuse.md (new)

TESTS
see following message; `test -s`, `wc -l`, `git status --porcelain`

RISKS
All GPU numbers are RECEIPT_CLAIMED (DIRTY_ENGINEERING or GPU_PROTECTED_CPU_CONTENDED). This lane did not re-time. Absolute roofs may move on a clean box; the 13.6 GB vs 64 MiB ratio and the ceremony-vs-traffic split will not invert.
G024 411.51 GB/s ceiling is superseded; do not reuse it.
ICB and TRY14 code are not on HEAD; receipts are.

UNRESOLVED
GPU L1/L2/SLC capacities not exposed by sysctl; cache-effective size inferred from the size sweep only.
No packed-Q4 GEMM roof (throughput experiment, not TOKEN_NS).
Live Genesis process not inspected (`ps` denied).
Energy / pJ per weight unread.

NEXT
Density on the 13.611 GB GEMV stream is the only existential single-stream lever. ICB graft is optional ceremony. Do not chase cache residency, recon caches, or multi-stream GEMV fanout for TOKEN_NS.
```
