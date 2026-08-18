# G1 GQA and attention geometry

Lane: `35-gqa-and-attention-geometry`. HEAD `0fbf2e2a5` (wave-1 reports landed).
No GPU run. No inference. No resident-process contact.
Every number is `SOURCE` / `RECEIPT` / `DERIVED` / `PROJECTED` / `UNMEASURED`.
A component microbenchmark is not a token-level claim.

G0 vehicle: `qwen38-27b/uniform-q4-v1` via `Qwen38HybridDecodeSession::step`.
Sealed ledger wall used for component math:
`receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` /
`G024_QWEN38_TOKEN_NS.json` `median_wall_ns = 35_227_917`.
Today's live G0 (wave-1 remeasure, not re-run here): `TOKEN_NS = 39_326_090`.
Relative deltas against either wall are valid; do not add them.

---

## 0. Where the 2.443 ms actually goes

The sealed `gqa` row is **not** attention arithmetic and **not** KV traffic.

`QWEN38_TOKEN_NS_LEDGER.json` `components[gqa]`:

```
ns_per_token          2443470.7102658837
pct_of_token_wall     6.936177095755857
bytes_read            2883584
bytes_written         327680
dispatches            64
gpu_occupancy         "mha_decode_f32 TG=128, 16 layers, seq≈19; rope 24 threads"
effective_gb_s        1.3142224240742257
theoretical_lower_bound_ns  7803.611090860491
measured_over_floor   313.1205132874512
method                isolated rope+mha+sigmoid+gqa FMA remainder
                      + 16/64 mixer residual, minus KV/state stream
```

`seal_components` (`qwen38_token_ns_ledger.rs:410-414`) reconstructs that
row from isolated families (scale = 1.0, isolated GPU sum 33_575_407 <
production GPU 33_912_333):

| piece | ns | % of gqa row | class | pointer |
|---|---:|---:|---|---|
| rope leftover after KV stream | 1,526,670 | 62.5 | 16× 24-thread kernel | isolated `rope_cache_16` 1,562,625 |
| mha leftover after KV stream | 651,164 | 26.6 | 16× `mha_decode_f32` | isolated `mha_16` 666,500 |
| GQA GEMV FMA remainder | 192,449 | 7.9 | 10.6% of `gqa_gemvs` | probe `gqa.fma_remainder_frac=0.10589` |
| sigmoid | 43,625 | 1.8 | 16× elementwise | isolated `sigmoid_16` |
| mixer residual × 16/64 | 29,562 | 1.2 | 16 of 128 adds | isolated `mixer_residual_64` 118,250 |
| **sum** | **2,443,471** | 100 | | matches receipt exactly |

DERIVED reconstruction (this lane, closed arithmetic on receipt fields):

```
rope=1562625; mha=666500; sigmoid=43625
gqa_fma = 0.1058916672164676 * 1817416 = 192449.21
mixer   = 118250 * 16/64              = 29562.50
kv_gqa  = 24375+26916                 = 51291
rope_after = rope - kv_gqa * rope/(rope+mha) = 1526669.82
mha_after  = mha  - kv_gqa * mha /(rope+mha) =  651164.18
gqa = 2443470.7102658837
```

So **62% of the GQA bucket is `qwen38_gqa_qk_norm_rope_cache_f32`**, a
1-threadgroup / 24-thread kernel. Attention FLOPs at seq≈19 are 3.74 M
FMA (`16×24×19×256×2`). At the measured `mha_16` 666,500 ns that is
**5.60 GFLOP/s** (DERIVED). Unique KV traffic at seq=19 is 2,490,368 B
read + 131,072 B write (`theoretical_state_bytes(19)`); sequential
stream of that pair measured 51,291 ns (`stream_gqa_key`+`stream_gqa_value`).
The 2.443 ms is therefore **launch geometry + serial ALU + a 6× GQA
over-read inside `mha_decode_f32`**, not compute or DRAM.

The 313× figure is vs `HONEST_DECODE_CEILING_GB_S = 411.51`, which
wave 1 falsified as this box's roof (`g1-roof-falsification.md`).
Same bytes against measured regimes (DERIVED, this lane):

| bandwidth | floor ns for 3,211,264 B | 2,443,471 / floor | label |
|---|---:|---:|---|
| 411.51 | 7,803.611 | **313.121×** | RECEIPT method (named, not this roof) |
| 639.25 | 5,023.487 | 486.4× | wave-1 measured effective addressing |
| 699.57 | 4,590.340 | 532.3× | wave-1 single-address |
| 819.0 | 3,920.957 | 623.2× | named peak, not a measured GQA regime |

No quoted GB/s was measured on the GQA activation kernels themselves.
The statement that stands without a roof: unique KV at seq=19 streams
in 51 µs; the gqa row is 2.44 ms; ratio **47.6× the measured KV stream**
(DERIVED). That is enough to reject a traffic diagnosis.

G024 framed rope as "encoder-per-tiny-kernel tax"
(`G024_QWEN38_TOKEN_NS.json` `top_three_attacks[2]`). That framing is
**false for this kernel**. Same encoder path
(`TokenCommandBuffer::dispatch_threads_inner` `metal/mod.rs:4861-4873`):

| family | ns/launch | TGs | threads | SOURCE |
|---|---:|---:|---:|---|
| `sigmoid_16` | 2,727 | 24 | 6,144 | isolated 43,625 / 16 |
| `silu_64` | 2,515 | 68 | 17,408 | isolated 160,958 / 64 |
| `mixer_residual_64` | 1,848 | 20 | 5,120 | isolated 118,250 / 64 |
| `rope_cache_16` | **97,664** | **1** | **24** | isolated 1,562,625 / 16 |
| `mha_16` | 41,656 | 24 | 3,072 | isolated 666,500 / 16 |

Encoder issue floor on this genome is the 2–3 µs class. Rope is 36×
that floor with 1/24 the threadgroups of sigmoid. ICB (964 → 964
commands, encode 886 µs → 91 µs on a later commit, **not on this
`step()`**) cannot delete a 97 µs 24-thread kernel. ICB is ceremony;
rope is occupancy.

---

## 1. Dispatch census (964)

`production_dispatches_per_token = 1 + 64×15 + 3 = 964`
(`qwen38_token_ns_ledger.rs:56-59`, test `production_dispatch_count_is_964`).
RECEIPT `dispatches`: embed 1, mixer_prefix 576, mlp_suffix 384,
terminal 3, total 964, `production_command_buffers` 1.

Mixer prefix is 9 kernels × 64 layers (`qwen38_64_layer_execution_schedule.rs:12-39`).
16 of 64 layers are GQA (`(layer+1)%4==0`, layers 3,7,…,63;
`qwen38_geometry.rs:82-93`). 48 are DeltaNet.

### 1.1 Named chain in the contract

| kernel | times | schedule slot |
|---|---:|---|
| `qwen38_gqa_qk_norm_rope_cache_f32` | 16 | GQA prefix #5 |
| `mha_decode_f32` | 16 | GQA prefix #6 |
| `qwen38_attention_apply_sigmoid_gate` | 16 | GQA prefix #7 |
| `qwen38_qkvz_rearrange_conv_l2_f32` | 48 | DN prefix #4 (conv lives here) |
| **named chain** | **96** | **10.0% of 964** |

### 1.2 Broader mixer envelopes (do not confuse with §1.1)

| envelope | times | of 964 | what |
|---|---:|---:|---|
| GQA mixer prefix (all 9) | 144 | 14.9% | rms + q/k/v GEMV + rope + mha + sigmoid + o + add |
| DN mixer prefix (all 9) | 432 | 44.8% | rms + qkvz/ba GEMV + rearrange + ba_to_decay + vi + gated_rms + o + add |
| GQA activation-only (ledger `gqa.dispatches`) | 64 | 6.6% | 16×(rope+mha+sigmoid+mixer residual) |
| GQA weight GEMVs | 64 | 6.6% | 16×(q+k+v+o); live in `weight_addressing`, not the gqa row |
| DN rearrange+ba+vi+gated_rms | 192 | 19.9% | ledger `deltanet.dispatches` |

Production `step()` (`qwen38_hybrid_decode.rs:3292-3310`) builds one
`TokenCommandBuffer`, encodes embed+64 layers+terminal, one
`commit_and_wait_timed`. Default Off path: **964 compute encoders
inside 1 CB** (`metal/mod.rs:4861-4873` `new_compute_command_encoder`
+ `end_encoding` per `dispatch_threads`). `begin_serial_group` /
`begin_concurrent_group` / ICB are unused
(`concurrent_independent` default false, `:867-870,911`).
`ReplayableComputeGraph` exists (`metal/mod.rs:3569-3574`) and is
explicitly "not wired into decode selection yet". Confirmed: no ICB
symbol in `qwen38_hybrid_decode.rs` (SOURCE, this lane).

`qwen38_fuse_split_qkvz_f32` / `qwen38_fuse_split_ba_f32` are **not**
on the G0 uniform-Q4 token. Pack-time `fuse_in_proj_qkvz` already
produced `in_proj_qkvz`. Split kernels fire only on mixed artifacts
missing the fused name (`g1-kernel-inventory.md` §6.2).

---

## 2. Per-kernel grid and occupancy

Occupancy below is **launch-derived vs 60 M3 Ultra cores**, not a
hardware counter. G024 `unresolved[0]`: "Occupancy is launch-geometry
derived (60 cores), not a hardware counter sample." This lane did not
fill that gap (GPU forbidden).

Simdgroup width on this device is 32 (`g1-direct-gemv-geometry.md`
`matvec-occupancy-230x.json` `thread_execution_width`).

### 2.1 `qwen38_gqa_qk_norm_rope_cache_f32`

SOURCE: `qwen38_device_activations.metal:108-193`.
Host: `encode_gqa` `qwen38_hybrid_decode.rs:2753-2778`.

| | |
|---|---|
| grid | `(24, 1, 1)` = `QWEN38_GQA_HEADS` |
| TG | `(24, 1, 1)` |
| TGs | **1** |
| threads | 24 |
| simdgroups | 24/32 = **0.75** (not a full simdgroup) |
| TG/core if spread | 1/60 = **0.017** |
| times | 16 |
| isolated | 1,562,625 ns (97,664 ns/launch) RECEIPT |

One thread per query head. Thread `head` serial-walks 256 dims for Q
RMS, then 256 dims for `(1+q_norm)` and rotate-half on the first 64.
Threads `head < 4` also do K RMS + K rope + V memcpy into the cache.
Hard-fails unless `24/4/256/64/θ=1e7/eps=1e-6`.

`pow` / `cos` / `sin` of `θ^{-2i/64} * sequence_slot` are computed
inside the dim loop, and the peer dim is recomputed when `dim` and
when `peer` is current (`:145-156` and `:174-186`). Frequencies are
identical across 24 heads and 16 layers. Recomputes per token:
`16×24×64 = 24,576` (DERIVED) against 32 unique frequencies.

This is already "norm + rope + cache write" in one pass. The missing
fusion is not those three stages. The missing geometry is **threads
on the 256-wide dim axis**.

### 2.2 `mha_decode_f32`

SOURCE: `mha.metal:602-681`. Host: `mha_decode_f32_tcb`
`kernels/mod.rs:10506-10560`. Comment on the wrapper says "TG size 64";
the constant is `TG_SIZE = 128` (`:10545`). Comment is stale. Code wins.

| | |
|---|---|
| grid | `(24×128, 1, 1) = (3072, 1, 1)` |
| TG | `(128, 1, 1)` |
| TGs | **24** (one per query head) |
| threads | 3,072 |
| simdgroups | 96 |
| TG/core if spread | 24/60 = **0.40** |
| shmem | `(seq_len + 128)×4` B = 588 B at seq=19 |
| shmem cap | ~8,064 tokens at 32 KB (`(32768/4)−128`) |
| times | 16 |
| isolated | 666,500 ns (41,656 ns/launch) RECEIPT |
| seq | ≈19 (ledger occupancy string; `state_bytes.gqa_kv_read_bytes_at_pos=2490368` ⇒ `2490368/(16×2×4096)=19`) |

Phase 1 (`mha.metal:627-634`): `for t = tid; t < SEQ; t += 128` scalar
dot of 256. At seq=19, **19 of 128 threads work, 85.2% idle**.
Phase 4 (`:673-680`): `for i = tid; i < 256; i += 128` then
`for t in 0..SEQ` scalar `V[t, kv_h, i]`. Every query head in a group
of 6 independently re-reads the same `kv_h`.

`mha_decode_flash_f32` exists (`mha.metal:720`, wrapper
`kernels/mod.rs:11134-11197`, `HAWKING_QWEN_FLASH_ATTN` default off).
**Zero call sites in `qwen38_hybrid_decode.rs`** (SOURCE). Flash
removes the O(seq) shmem cap. It does not change the 6× over-read,
the `[seq, kv_head, dim]` layout, or the 24-TG map. At seq=19 it is
one 128-token tile with 109 idle lanes — same occupancy hole as Phase 1.

### 2.3 `qwen38_attention_apply_sigmoid_gate`

SOURCE: `qwen38_device_activations.metal:250-266`.
Host: `qwen38_hybrid_decode.rs:2795-2806`.

| | |
|---|---|
| grid | `(6144, 1, 1)` = `24×256` |
| TG | `(256, 1, 1)` |
| TGs | 24 |
| threads | 6,144 |
| TG/core | 0.40 |
| times | 16 |
| isolated | 43,625 ns (2,727 ns/launch) RECEIPT |

`gated[h,d] = attn[h,d] * σ(q_proj[h, 256+d])`. `q_proj` is
`24 × 512` (`QWEN38_Q_PROJ_ROWS = 12_288`, workspace
`qwen38_hybrid_decode.rs:760` allocates `HEADS * HEAD_DIM * 2`).
Gate half is the second 256 of each head. Independent of weights.

### 2.4 `qwen38_qkvz_rearrange_conv_l2_f32` (conv path)

SOURCE: `qwen38_device_activations.metal:32-105`.
Host: `qwen38_hybrid_decode.rs:1934-1958` / `:2630-2654`.
DeltaNet only. Lives in the `deltanet` bucket, not `gqa`.

| | |
|---|---|
| grid | `(256, 16, 1)` |
| TG | `(256, 1, 1)` |
| TGs | 16 (one per key head) |
| threads | 4,096 |
| TG/core | 0.27 |
| TG mem | `4×256×4` B |
| times | 48 |
| isolated | 350,999 ns (7,312 ns/launch) RECEIPT |

Per key-head: causal conv-k=4 + silu on Q/K/V channels, L2 of Q and K,
repeat ×3 onto 48 value heads, copy Z. Hard-fails unless
`16/3/128/128/k=4`. Occupancy is not the rope disaster. Time is near
the tiny-kernel floor. 48 launches of a 16-TG kernel.

### 2.5 GQA GEMVs (weight-interacting; not in the 2.443 ms except 192 µs FMA)

`geo_tpr64_tg128`, `tg=(128,1,1)`, `grid=(ceil(rows/2)×128, 1, 1)`
(`qwen38_hybrid_decode.rs:264-268`, inventory §3.1).

| organ | rows | TGs | TG/core | times | isolated family |
|---|---:|---:|---:|---:|---|
| q_proj | 12,288 | 6,144 | 102.4 | 16 | part of `gqa_gemvs` 1,817,416 / 64 |
| k_proj | 1,024 | 512 | 8.53 | 16 | |
| v_proj | 1,024 | 512 | 8.53 | 16 | |
| o_proj | 5,120 | 2,560 | 42.7 | 16 | |

Addr fraction 0.830 of GQA GEMVs (`probes[gqa]`). These 64 dispatches
and ~1.51 ms addressing sit in `weight_addressing`, not `gqa`.
Gravity / GEMV-geometry lanes own them.

Q80 shipped the same three-kernel GQA tail
(`qwen80_uniform_q4_hybrid_decode.rs:3564-3638`):
`qwen80_gqa_qk_norm_rope_cache_f32` grid `(query_heads,)` TG
`(min(query_heads,16),)` — still one undersized TG — then the same
`mha_decode_f32_tcb` + sigmoid. Q38 forked the shader for 24/4/256/64/1e7
and kept the launch.

---

## 3. KV cache layout

### 3.1 Physical layout (SOURCE)

Workspace (`qwen38_hybrid_decode.rs:767-768,798-799`):

```
gqa_key, gqa_value : 16 * max_seq_len * 4 * 256 f32
```

Per-layer window via byte offset
`slot * max_seq_len * 4 * 256 * 4`
(`:2717-2719`). Write (`qwen38_device_activations.metal:170`):

```
cache_base = (sequence_slot * n_kv_heads + head) * head_dim
```

Read (`mha.metal:628,676`):

```
k_cache + (t * NKV + kv_h) * H_DIM
v_cache + (t * NKV + kv_h) * H_DIM
```

Layout is **`[layer][seq][kv_head][dim]`**, K and V in separate buffers,
f32. `attn.rs:16-18` documents the same `(seq_len, n_kv_heads, head_dim)`.

One KV head slot = 256×4 = 1,024 B contiguous. Consecutive tokens of
the same head are **4,096 B apart** (`NKV * H_DIM * 4 = 4*256*4`).

### 3.2 Access pattern at seq≈19 (DERIVED from the loops)

**Phase 1 (K dots).** Threads `tid=0..18` each own one token. Inner
loop walks 256 contiguous floats of that token's `kv_h` — coalesced
*within* a thread. Adjacent threads sit 4,096 B apart — **uncoalesced
across the simdgroup**. 109/128 threads idle.

**Phase 4 (V accumulate).** Threads `tid=0..127` own dims `i` and
`i+128`. Inner loop walks `t=0..18` at `V[t, kv_h, i]`, stride 4,096 B
— **strided scalar gathers**. Adjacent threads at fixed `t` hit
consecutive dims — coalesced 512 B then a 4 KB jump.

**GQA group = 6.** `kv_h = h / 6`. Six TGs reread the same K and V.

| traffic | bytes / token at seq=19 | pointer |
|---|---:|---|
| unique K+V read | 16×2×19×4×256×4 = **2,490,368** | `state_bytes.gqa_kv_read_bytes_at_pos` |
| if 24 heads each read their kv_h | 16×2×19×24×256×4 = **14,942,208** | DERIVED, 6.0× |
| unique K+V write | 16×2×4×256×4 = **131,072** | `state_bytes.gqa_kv_write_bytes` |
| measured sequential stream | 51,291 ns | `stream_gqa_key`+`stream_gqa_value` |

The layout **does** force strided / uncoalesced access. At seq=19 the
byte volume is still tiny (2.5 MB unique). Layout is a multiplier on
MHA, not the 1.53 ms rope term.

### 3.3 Workspace reuse blocks cross-layer fusion

`q_proj`, `k_proj`, `v_proj`, `query`, `attn`, `gated_attn` are **one
buffer each**, overwritten every GQA layer (`:784-789`). Same for
DeltaNet `qkvz` / `ba` / `repeated_*` / `conv_v` / `z`. KV cache,
`conv_state`, and `rec_state` are stacked by slot.

A 16-layer fused GQA tail therefore needs stacked Q/K/V/attn
workspaces (~1.3 MB f32, DERIVED: `16*(12288+1024+1024+6144+6144+6144)*4`)
or stays serial across layers. That is a workspace change, not a KV
layout change, and is independent of the weight codec.

---

## 4. What is already fused, what is recomputed

### 4.1 Already one pass

`qwen38_gqa_qk_norm_rope_cache_f32` already does Q RMSNorm, K RMSNorm,
partial RoPE (first 64 of 256, rotate-half), Q store, K cache write,
V cache write. Asking "can norm and rope and cache write fuse into
one pass?" — **they already did**. The remaining fuse is that kernel
with `mha_decode_f32` and the sigmoid epilogue, and/or spreading its
256-wide loops across a threadgroup.

`qwen38_qkvz_rearrange_conv_l2_f32` already fuses rearrange + conv-k=4
+ silu + L2 + head repeat.

### 4.2 Recomputed every token that could be cached

| work | data-dependent? | cached today? | verdict |
|---|---|---|---|
| RoPE `θ^{-2i/64}·pos` cos/sin (32 unique freqs) | no, only `pos` | no; 24,576 recomputes | **cacheable**. Table of 64+64 f32 per token, reused by 16×24 heads. ESTIMATED ALU ≪ 10 µs. Not the 1.56 ms. |
| Q RMS / `(1+q_norm)` / rope apply | yes (new q) | no | must recompute. Parallelize, do not cache. |
| K RMS / rope apply | yes (new k) | written to cache after | correct. Past K not recomputed. |
| V | no transform | memcpy `v_proj → value_cache[pos]` | **recopy**, not recompute. V GEMV dest could be the cache slot. ESTIMATED ~0. |
| Past K/V attention | n/a | yes, f32 cache | correct. Scores must be redone (new q). |
| q_norm / k_norm vectors | static per layer | re-read 256+256 f32 ×16 | ignore. |
| Conv taps / recurrent S | recurrent | yes, per DN slot | correct. Not GQA. |
| Q/K L2 inside rearrange | yes | no | must recompute. |

Nothing material is being recomputed that a cache would delete except
the RoPE frequency table, and that table is not why rope costs 1.56 ms.

---

## 5. qkvz / conv path (DeltaNet, requested)

48 layers. Isolated `rearrange_48` 350,999 ns. After `stream_conv_state`
19,000 ns is parked in `kv_state`, ~332 µs stays in `deltanet`
(`seal_components` `:404,404-409`). `ba_to_decay_48` 139,374 ns and
`gated_rmsnorm_48` 1,295,500 ns dominate that bucket, not rearrange.

Rearrange launch is 16 TGs × 256 threads — not occupancy-starved the
way rope is. 7.3 µs/launch is 2–3× the tiny-kernel floor, consistent
with conv+L2 ALU plus 48 encoder transitions.

G024 already queued "fuse gated_rmsnorm + ba + rearrange" as attack #2,
1.0–1.5 ms (`G024_QWEN38_TOKEN_NS.json` `top_three_attacks[1]`). That
is a DeltaNet-lane claim. This lane only notes: rearranging 48→1
without stacking `qkvz` is illegal (workspace reuse, §3.3). Folding
rearrange into the following `ba_to_decay` is a 48-dispatch cut at
~100 µs isolated PROJECTED. Weight-independent.

---

## 6. Ranked proposals

Predictions are **PROJECTED from isolated-family GPU timestamps
already inside the sealed gqa/deltanet rows (scale=1.0)**. They are
not a new token measurement. Dispatch cuts are SOURCE-countable.
"Predicted token" is sealed wall 35,227,917 minus the projected
component cut. Today's live 39,326,090 would move by the same delta
only if the isolated families still partition that session; **UNMEASURED**.

Do not sum overlapping rows. P1 ⊂ P2 ⊂ P3. P6 stacks with P1/P2.
P7/P8 are absorbed by P1.

### Class A — fuse / rewrite existing activation kernels.
**Independent of the weight representation.** Can land regardless of
gravity / GEMV / BPW conclusions. No expand-to-float W.

#### P1. Parallelize rope on the dim axis. Dispatch Δ = 0.

Keep 16 launches. Change grid to `(256, 24, 1)` TG `(256, 1, 1)` —
one TG per query head, 256 threads for 256 dims. RMS becomes a
256-wide tree reduce (same shape as `qwen80_residual_rmsnorm_f32`).
Then each thread does one dim of `(1+q_norm)` + rope + cache write.
KV heads 0..3 still own the K/V path.

| | |
|---|---|
| dispatch | 16 → 16 |
| isolated rope | 1,562,625 → **80,000–200,000** PROJECTED (16 × 5–12 µs; well-occupied tiny kernels on this genome are 2–3 µs, RMS-class is 18 µs for a 5120-d 1-TG reduce; 24 TGs of 256 should sit between those) |
| gqa row | 2,443,471 → **~960,000–1,080,000** |
| sealed token | 35.23 ms → **33.85–34.00 ms** |
| live token | 39.33 ms → **37.95–38.10 ms** PROJECTED iff the 1.53 ms transfers |

G024 expected "~1.2 ms" from rope (`top_three_attacks[2]`). This
proposal is that 1.2 ms, named as an occupancy rewrite rather than
an encoder collapse.

**KILLS if** a 24-TG rope family still measures ≥50 µs/launch.
**REOPEN_IF** hardware counters show the 97 µs is not the 24-thread
serial walk (then the diagnosis is wrong).

Cheapest falsifier: isolated `rope_cache_16` A/B, same CB shape, GPU
timestamps, bit-id on `query` + `gqa_key` + `gqa_value`. No token run
required. GPU lane owns it.

#### P2. Fuse rope + mha + sigmoid per GQA layer. Dispatch Δ = −32.

One kernel per GQA layer: parallel rope (P1 geometry) then the
existing 128-TG mha body then `out[i] *= σ(q_gate[i])` at the Phase-4
store. 16 launches remain, one per GQA layer. No workspace stack.
No KV layout change.

| | |
|---|---|
| named-chain dispatches | 48 → 16 (−32). 964 → 932 |
| sigmoid | 43,625 → 0 (epilogue) |
| rope | as P1 |
| mha | 666,500 unchanged at seq=19 (same 24 TGs, same layout) |
| gqa row | 2,443,471 → **~900,000–1,050,000** |
| sealed token | → **33.78–33.93 ms** |

`dispatch_threads_pair_in_one_encoder` (`metal/mod.rs:3377+`) can
stage this as two kernels / one encoder without a new shader, but
that only deletes encoder create, not the 24-thread rope. The shader
fuse is the real cut.

Independent of weights. q/k/v GEMVs still produce `q_proj`/`k_proj`/`v_proj`;
o GEMV still consumes `gated_attn`.

#### P3. One dispatch for all 16 GQA tails. Dispatch Δ = −47.

P2 plus stacked Q/K/V/attn workspace (~1.3 MB). Grid
`(128, 24, 16)` = 384 TGs for the mha body (6.4 TG/core — healthy).
Rope portion 16×24 TGs.

| | |
|---|---|
| named-chain GQA tails | 48 → 1 |
| 964 | → 917 |
| extra vs P2 | encoder×15 + any residual per-launch floor |
| PROJECTED extra save vs P2 | 30–80 µs (15 × 2–5 µs). Not millisecond-class. |
| requires | workspace stack (§3.3), not a KV layout change |

Do this after P2 measures. Not first.

#### P4. Sigmoid epilogue only. Dispatch Δ = −16.

Fold `qwen38_attention_apply_sigmoid_gate` into `mha_decode_f32` store.
16 → 0 sigmoid launches. Isolated 43,625 ns. Near the tiny-kernel
floor. **PROJECTED token −0.04 ms.** Independent of weights.
Absorbed by P2. Ship only if P2 is deferred.

#### P5. Swap in `mha_decode_flash_f32`. Dispatch Δ = 0.

Kernel and TCB wrapper already exist. Not called from Q38 `step()`.

At seq≈19: **KILLS as a token win.** Same 24 TGs, same layout, one
flash tile, 109/128 lanes idle, more barriers than the materialize
path. PROJECTED ≤0, possibly slightly negative.

**REOPEN_IF** production seq ≫ 128 (shmem cap of `mha_decode_f32` is
~8k; flash is the long-seq insurance). Flash does not fix 6× over-read
or rope occupancy.

`mha_decode_flash_f16kv` / `int4kv` additionally change the cache
element type. That is a layout+codec change. Independent of *weight*
representation. Not a seq-19 lever. F16 KV halves 2.49 MB → 1.25 MB;
at the measured stream that is −26 µs. Ignore until depth.

### Class B — cache layout change.
**Independent of the weight representation.** Does not move rope.
Stacks with Class A.

#### P6. Retile KV to `[layer][kv_head][dim][seq]` (or `[kv_head][seq][dim]`) and share K/V across the GQA group of 6.

Current `[seq][kv_head][dim]` makes consecutive tokens of one head
stride 4,096 B and makes six query TGs reread the same head.

| retile | Phase 1 (walk dim @ fixed t) | Phase 4 (walk t @ fixed dim) |
|---|---|---|
| `[kv_head][seq][dim]` | token blocks 1,024 B contig; inter-thread stride 1,024 B not 4,096 B | still stride `dim×4=1,024` B |
| `[kv_head][dim][seq]` | dim walk strides by `seq` | **t walk contig** — this is the inner Phase-4 loop |

Cannot have both without a TG-mem transpose. Prefer `[kv_head][dim][seq]`
if Phase 4 stays the inner loop; or keep `[kv_head][seq][dim]` and
vectorize Phase 1 as `float4` over dim (the llama b9430 kernels
already do `float4`, `mha.metal:46+`).

Sharing: one TG per `kv_h` (4 TGs) that keeps 6 queries in registers
and reads K/V once **drops occupancy 24→4** — do not. Keep 24 TGs
but load a K/V token-tile into TG memory once per group, or split
dim so `4 kv × N` TGs stay ≥24.

| | |
|---|---|
| dispatch | 16 → 16 |
| unique KV bytes | unchanged (2.49 MB) |
| reread | 14.9 MB → 2.49 MB |
| isolated mha | 666,500 → **150,000–300,000** PROJECTED at seq=19 (3.7 M FMA is ~4 µs at 1 TFLOP; 42 µs/launch → ~10–20 µs/launch if scalar+stride+6× are the 10× over compute-floor) |
| gqa row | −0.35 to −0.50 ms on top of P1/P2 |
| sealed token stacked on P2 | → **33.3–33.6 ms** |

Rope write must emit the new layout (`cache_base` today is
`(slot * nkv + head) * dim`). Both kernels change together.

**KILLS if** isolated `mha_16` after retile+vectorize stays within
15% of 666,500 ns (then MHA is not the stride/over-read).
**REOPEN_IF** seq grows: the 6× and the stride scale with seq; this
proposal's share of the token grows with context.

Cheapest falsifier: host-side layout permute of a captured K/V plus
the existing `mha_decode_f32` vs a one-file retiled shader, isolated
`mha_16` CB, seq=19 and seq=256. GPU lane.

### Class C — interacts with the weight representation.
Do **not** wait on these to land Class A/B. Gravity owns the 21.3 ms.
Listed so the split is explicit.

#### P7. Fuse q+k+v GEMVs (shared X). Dispatch Δ = −32 GEMV.

16×3 → 16. Bytes almost unchanged (q 12,288 + k 1,024 + v 1,024 rows
of the same 5,120-col Q4). X reuse is 20 KB. Isolated `gqa_gemvs`
1,817,416 / 64 = 28.4 µs each; issue save ESTIMATED 32×2 µs = 64 µs
plus 50 ns of X. **PROJECTED token ≪ 0.1 ms.** Interacts with the Q4
(or successor) decoder: one kernel must unpack three packed matrices.
`g1-direct-gemv-geometry.md` C3 is the gate+up cousin; same constraint
(keep `geo_tpr64` occupancy, consume codes in-register).

#### P8. Fuse input RMSNorm into q/k/v GEMV. Dispatch Δ = −16 of the 129 norms.

The 16 GQA `input_layernorm` launches are in the `normalization` row
(2.367 ms / 129), not `gqa`. Fusion-lane P3 already queued 1.2–1.8 ms
for all 129. Mentioned only because it touches the GQA prefix's first
slot. Interacts with GEMV occupancy: **KILLS if** the fused GEMV drops
below ~400 GB/s on q_proj 12,288×5,120 (fusion-lane kill condition).

#### P9. ICB / serial-group encoder. Dispatch Δ = 0.

964 commands still execute. Encode 886 → 91 µs MEASURED on commit
`7400acf1b`, **not an ancestor of this HEAD**
(`g1-residency-reuse.md` §3.2; `g1-fusion-persistent.md` P6).
Net named-fixed −0.66 ms, wait−gpu rose +136 µs. Independent of
weights. Does **not** move the 1.56 ms rope. Not a GQA proposal.

---

## 7. Combined independent path (A + B, no gravity)

| step | dispatch 964→ | gqa row 2.443 ms → | sealed token 35.23 ms → | weight? |
|---|---:|---:|---:|---|
| P1 rope occupancy | 964 | 0.96–1.08 | 33.85–34.00 | independent |
| + P2 fuse tails | 932 | 0.90–1.05 | 33.78–33.93 | independent |
| + P6 KV retile + share | 932 | 0.50–0.75 | 33.29–33.63 | independent |
| + P3 stack-16 (optional) | 917 | 0.47–0.70 | 33.26–33.60 | independent |

Ceiling of everything this lane can touch without touching weights:
**about 1.7–2.0 ms off the sealed wall.** That is the entire gqa
activation remainder plus the MHA leftover. It is **not** 100 TPS.
100 TPS still requires the 21.3 ms addressing cut (wave-1 roof:
even at 1.5 BPW, addressing ~7.51 ms and the non-addressing remainder
must fall too). This lane's job is that remainder's GQA slice.

qkvz/conv adds at most ~0.1–0.2 ms more if 48 rearranges collapse
(Class A, DN bucket). Gated-rmsnorm 1.30 ms is the DN-lane sibling
of P1 (16-wide reductions × 48) and is **not** double-counted here.

---

## 8. What this lane will not claim

- Hardware occupancy counters. UNMEASURED. Cheapest experiment:
  `MTLCounterSampleBuffer` or a GPU frame capture on one
  `qwen38_gqa_qk_norm_rope_cache_f32` dispatch and one `mha_decode_f32`
  at seq=19. GPU lane.
- Token-level A/B of any proposal. Isolated-family GPU is the
  evidence; production transfer is PROJECTED (scale was 1.0, so the
  isolated numbers already sit inside the sealed token, but a fused
  kernel can change encoder-gap accounting).
- Flash or f16/int4 KV as a seq-19 win.
- Generator+residual, mixed-sub-1.5 expand vehicles, Q80/DSV4F
  resurrection. Transfer science only: Q80 has the same 1-TG rope
  launch; llama b9430 has `float4` MHA; ICB receipts show ceremony
  is not this mass.
- Any low-BPW path that expands to float/Q4 then this GEMV.

---

## 9. Evidence

### 9.1 Receipts (git show; not materialized by sparse checkout)

`HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`

```
schema hawking.ascension.qwen38_token_ns_ledger.v1
bpw 4.252735126866492
kernel_runtime_genome Qwen38HybridDecodeSession
  + qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
  + qwen38_gated_delta_decode_vi
  + qwen38_qkvz_rearrange_conv_l2_f32
  + qwen38_gqa_qk_norm_rope_cache_f32
  deltanet_vi_parallel=true concurrent_independent=false
  1 production CB / 964 dispatches
median_gpu_ns 33912333
median_wait_ns 34296583
median_encode_ns 919250
median_submit_ns 12084
median_wall_ns 35227917
dispatches {embed:1, mixer_prefix:576, mlp_suffix:384, terminal:3, total:964, production_command_buffers:1}
state_bytes.gqa_kv_write_bytes 131072
state_bytes.gqa_kv_read_bytes_at_pos 2490368
closure.gpu_scale_applied 1.0
closure.isolated_family_sum_gpu_ns 33575407
closure.production_gpu_ns 33912333
components[gqa].ns_per_token 2443470.7102658837
components[gqa].measured_over_floor 313.1205132874512
components[gqa].dispatches 64
isolated.rope_cache_16.median_gpu_ns 1562625  dispatches 16  reps [1562625, 1570666, 1550500]
isolated.mha_16.median_gpu_ns 666500          dispatches 16  reps [744999, 653625, 666500]
isolated.sigmoid_16.median_gpu_ns 43625       dispatches 16  reps [44833, 40416, 43625]
isolated.gqa_gemvs.median_gpu_ns 1817416      dispatches 64
isolated.rearrange_48.median_gpu_ns 350999    dispatches 48
isolated.stream_gqa_key 24375
isolated.stream_gqa_value 26916
probes[gqa].addr_frac_of_full 0.8302650920366828
probes[gqa].fma_remainder_frac 0.1058916672164676
```

`HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`

```
ranked_by_ns[3] gqa 2443471 ns  6.94%  triage=research
top_three_attacks[2] "GQA rope is the alternate 1.2 ms if norms are left alone."
unresolved[0] "Occupancy is launch-geometry derived (60 cores), not a hardware counter sample."
```

Command that produced the reconstruction and floors (this lane, no GPU):

```
python3 - <<'PY'
# outputs recorded in §0 and §2
H=24; NKV=4; HD=256; LAYERS=16; SEQ=19
total_rw = 24*256*4*16 + 16*2*SEQ*4*256*4 + 5120*4*16  # 3211264
# 2443470.71 / (3211264 / 411.51e9 * 1e9) = 313.121
# kv_read = 16*2*19*4096 = 2490368  => seq = 19
# unique_kv * 6 = 14942208
# qk_fma = av_fma = 16*24*19*256 = 1867776; mha 666500 ns => 5.60 GFLOP/s
PY
```

Exact printed values:

```
group_size 6
kv_one_bytes 4096
kv_write 131072 kv_read 2490368
gqa_component_rw 3211264
floor 411.51 GB/s: 7803.611 ns; 2443470.71/floor = 313.121x
floor 639.25 GB/s: 5023.487 ns; 2443470.71/floor = 486.409x
unique_kv_per_layer 155648 if_24_heads_reread 933888 overread_x 6.0
qk_fma 1867776 av_fma 1867776 total_fma 3735552
mha_effective_GFLOP/s 5.604729182295574
gqa_recon 2443470.7102658837
rope TGs 1 threads 24 simdgroups 0.75 tg/core 0.016666666666666666
mha TGs 24 phase1_idle_frac 0.8515625
named_chain 96
mha_shmem_seq19 588
mha_shmem_cap_32kb 8064
rope_cos_sin_recomputes_per_token 24576
rope_ns_per_launch 97664.0625
mha_ns_per_launch 41656.25
sigmoid ns/launch 2726.5625
```

### 9.2 Source excerpts

`crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:29-39`

```
pub const QWEN38_GQA_MIXER_PREFIX_KERNELS: [&str; QWEN38_MIXER_PREFIX_DISPATCHES] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen38_gqa_qk_norm_rope_cache_f32",
    "mha_decode_f32",
    "qwen38_attention_apply_sigmoid_gate",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_next_add_residual",
];
```

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:2753-2806`
(production GQA tail launch)

```
tcb.dispatch_threads(
    "qwen38_gqa_qk_norm_rope_cache_f32",
    (QWEN38_GQA_HEADS as u32, 1, 1),
    (QWEN38_GQA_HEADS as u32, 1, 1),
    ...
)?;
mha_decode_f32_tcb(..., self.position + 1, QWEN38_GQA_HEAD_DIM,
                   QWEN38_GQA_HEADS, QWEN38_GQA_KV_HEADS)?;
tcb.dispatch_threads(
    "qwen38_attention_apply_sigmoid_gate",
    (query_dim, 1, 1),   // 6144
    (256, 1, 1),
    ...
)?;
```

`crates/hawking-core/src/kernels/mod.rs:10543-10559`

```
// TG=128 matches Qwen-3B head_dim (128), so Phase 4 ... full TG occupancy.
const TG_SIZE: u32 = 128;
let shmem_bytes = ((seq_len + TG_SIZE as usize) * std::mem::size_of::<f32>()) as u64;
tcb.dispatch_threads(
    "mha_decode_f32",
    (n_heads as u32 * TG_SIZE, 1, 1),
    (TG_SIZE, 1, 1),
    ...
)
```

Q38 `head_dim` is 256, not 128. Phase 4 therefore takes two passes
over dim (`i = tid; i < 256; i += 128`). The "full TG occupancy"
comment is a Qwen-3B leftover.

`crates/hawking-core/shaders/mha.metal:626-634,671-679`

```
// Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale
for (uint t = tid; t < SEQ; t += tg_size) {
    device const float* kt = k_cache + (t * NKV + kv_h) * H_DIM;
    float acc = 0.0f;
    for (uint i = 0; i < H_DIM; ++i) acc += q_h[i] * kt[i];
    scores[t] = acc * scale;
}
// Phase 4
for (uint i = tid; i < H_DIM; i += tg_size) {
    float acc = 0.0f;
    for (uint t = 0; t < SEQ; ++t) {
        device const float* vt = v_cache + (t * NKV + kv_h) * H_DIM;
        acc += scores[t] * vt[i];
    }
    out_h[i] = acc * inv_sum;
}
```

`crates/hawking-core/shaders/qwen38_device_activations.metal:126-130,170`

```
if (head >= n_heads || n_heads != 24u || n_kv_heads != 4u ||
    head_dim != 256u || rotary_dim != 64u ||
    rope_theta != 10000000.0f || rms_epsilon != 1.0e-6f) {
    return;
}
...
const uint cache_base = (sequence_slot * n_kv_heads + head) * head_dim;
```

`crates/hawking-core/src/metal/mod.rs:3569-3574,4861-4873`

```
/// This is intentionally not wired into decode selection yet.
pub struct ReplayableComputeGraph { ... }

let enc = cmd.new_compute_command_encoder();
...
enc.dispatch_threads(...);
enc.end_encoding();
```

`crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:118-136,573-586`

```
let kv_one = (QWEN38_GQA_KV_HEADS as u64) * (QWEN38_GQA_HEAD_DIM as u64) * 4;
let kv_write = (QWEN38_GQA_LAYERS as u64) * 2 * kv_one;
let kv_read = (QWEN38_GQA_LAYERS as u64) * 2 * seq_len.max(1) * kv_one;
...
"gqa" ... dispatches 16 * 4
"mha_decode_f32 TG=128, 16 layers, seq≈{seq_len}; rope 24 threads"
```

### 9.3 Wave-1 reports consumed, not re-derived

- `g1-token-anatomy.md` § isolated table, 313× claim, 964 encoder path
- `g1-kernel-inventory.md` §2–3.5, 3.7 census
- `g1-fusion-persistent.md` P3/P4/P6, megakernel kill, ICB
- `g1-traffic-anatomy.md` ICB-not-on-step, state-byte identity
- `g1-direct-gemv-geometry.md` GEMV launch, gqa q/k/v organs
- `g1-residency-reuse.md` ICB not ancestor
- `g1-roof-falsification.md` 411.51 is not the roof

---

## 10. Completion report

```
STATUS
SUPPORTED

CLAIMS
C1. The sealed gqa row is 2,443,471 ns / 6.94% / 313× vs 411.51 GB/s. 62.5% is rope (1,526,670 ns), 26.6% is mha (651,164 ns), 7.9% is GEMV FMA, 1.8% sigmoid, 1.2% residual. Evidence: QWEN38_TOKEN_NS_LEDGER.json components[gqa] + isolated + probes; reconstruction in §0.
C2. Named attention chain is 96 dispatches of 964 (16 rope + 16 mha + 16 sigmoid + 48 rearrange). GQA mixer prefix is 144. Ledger gqa.dispatches=64 counts rope+mha+sigmoid+residual only. Evidence: schedule.rs:12-39; ledger dispatches; inventory §3.7.
C3. Rope launches 1 TG × 24 threads (0.75 simdgroup, 0.017 TG/core). Isolated 97,664 ns/launch vs sigmoid 2,727 ns/launch on the same encoder path. Diagnosis is occupancy + serial 256-d RMS, not encoder tax and not ICB. Evidence: hybrid_decode.rs:2753-2756; metal/mod.rs:4861-4873; isolated families.
C4. mha_decode_f32 launches 24 TGs × 128 at seq≈19 with 85% Phase-1 idle, scalar dots, [seq,kv,dim] stride 4,096 B, and 6× K/V over-read. Unique KV 2.49 MB streams in 51 µs; mha is 667 µs (5.60 GFLOP/s). Evidence: mha.metal:626-679; kernels/mod.rs:10543-10559; state_bytes; stream_gqa_*.
C5. Norm+rope+cache-write is already one kernel. Cacheable leftover is the 32-frequency RoPE table (24,576 recomputes). That ALU is not the 1.56 ms. Evidence: qwen38_device_activations.metal:108-193.
C6. ICB and flash exist and are not on Q38 step(). ICB does not move rope. Flash KILLS at seq≈19. Evidence: metal/mod.rs:3569-3574; no ICB/FLASH symbol in qwen38_hybrid_decode.rs; mha.metal:720.
C7. Class A (P1/P2) is independent of weight representation and is the first GQA cut: PROJECTED −1.3 to −1.5 ms sealed, 964→932 for P2. Class B (P6) stacks −0.35 to −0.50 ms and needs a layout change. Combined independent ceiling ≈ 1.7–2.0 ms. Not 100 TPS. Evidence: §6–§7; G024 top_three_attacks[2] 1.2 ms prior.

EVIDENCE
HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json fields cited in §9.1
HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json ranked_by_ns[3], top_three_attacks[2]
crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12-39
crates/hawking-core/src/model/qwen38_geometry.rs:38-41,82-93
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:760-799,1934-1958,2753-2827,3292-3310
crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:56-59,118-136,410-414,573-586
crates/hawking-core/src/kernels/mod.rs:10506-10560,11134-11197
crates/hawking-core/src/metal/mod.rs:3569-3574,4861-4873
crates/hawking-core/shaders/qwen38_device_activations.metal:32-266
crates/hawking-core/shaders/mha.metal:602-681,720
workspace/superwave/g1/g1-token-anatomy.md:140-165,387-391
workspace/superwave/g1/g1-kernel-inventory.md:50-269
workspace/superwave/g1/g1-fusion-persistent.md:292-418
This-lane python (no GPU) printed values in §9.1

CHANGES
workspace/superwave/g1/g1-gqa-and-attention-geometry.md  (new, this file)

TESTS
test -s workspace/superwave/g1/g1-gqa-and-attention-geometry.md
wc -l workspace/superwave/g1/g1-gqa-and-attention-geometry.md
git status --porcelain

RISKS
Isolated-family GPU includes that family's own encoder gaps. Production intra-CB gap is 337 µs across 964 (350 ns/dispatch); isolated rope's 16 gaps are ~6 µs, so the 1.56 ms is almost all kernel. A fused kernel can still move gap accounting; token A/B required before calling MEASURED_WIN.
P1 projection band (80–200 µs) is not a hardware-occupancy measurement. If 97 µs is a hidden per-dispatch GPU minimum rather than the 24-thread walk, P1 dies. Sigmoid at 2.7 µs argues against a 97 µs minimum.
P6 occupancy must stay ≥24 TGs. A naive 4-TG (one per kv_head) rewrite would re-create the rope disaster on MHA.
DIRTY_ENGINEERING on every receipt. Box contamination ~3.8% on today's live number; relative deltas only.

UNRESOLVED
Hardware occupancy counters on rope and mha. Cheapest: GPU-lane counter sample / frame capture. This lane forbidden to take it.
Production seq beyond 19. Ledger identity is seq≈19. MHA and P6 scale with seq; rope does not.
Whether stacked-16 (P3) is worth the workspace change after P2 measures.
Whether V GEMV can legally target value_cache[pos] without breaking mixed / concurrent_independent paths.
Live 39,326,090 vs sealed 35,227,917 session split: this lane did not remeasure.

NEXT
GPU lane: isolated rope A/B of P1 (24 TGs × 256) vs HEAD, bit-id on query+KV, GPU timestamps. If ≥1.2 ms falls, implement P2 (fuse sigmoid+mha into that kernel). Then P6 retile at seq=19 and seq=256. Do not wait on gravity. Do not swap flash at seq=19.
```
