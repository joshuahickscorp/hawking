# G1 GQA geometry redo

Lane: `72-gqa-geometry-redo`. No GPU. No inference. No resident-process contact.
Every number is `SOURCE` / `RECEIPT` / `DERIVED` / `PROJECTED` / `UNMEASURED`.
A component microbenchmark is not a token-level claim.

G0 vehicle: `qwen38-27b/uniform-q4-v1` via `Qwen38HybridDecodeSession::step`.
Sealed wall used for component math:
`HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` /
`G024_QWEN38_TOKEN_NS.json` `median_wall_ns = 35_227_917`.
Live G0 today (not re-run here): `TOKEN_NS = 39_326_090`. Relative deltas only.

Wave-1 `g1-gqa-and-attention-geometry.md` is consumed, not copied. Three
corrections vs that file are marked **CORR** below.

---

## 0. Answer

**62.5% of the sealed 2,443,471 ns is `qwen38_gqa_qk_norm_rope_cache_f32`.**
That kernel already fuses Q/K RMS, partial RoPE, Q store, and KV cache write.
It launches **1 threadgroup × 24 threads**. Isolated 97,664 ns/launch.
The missing geometry is threads on the 256-wide dim axis, not another fuse of
those three stages, and not encoder-share.

Largest legal cut: rewrite that launch to `(256, 24, 1)` / TG 256 — one TG
per query head, 256-wide RMS tree, one dim per thread. **Pure kernel.
Dispatch Δ = 0. PROJECTED −1.24 to −1.43 ms** of the gqa row.
Does **not** share this morning’s last-TG-atomic or encoder-share kills.

The 313× figure is unique-once 3,211,264 B ÷ falsified 411.51 GB/s.
Same bytes ÷ measured addressing 639.25 GB/s = **486.4×**. Same bytes ÷
measured GQA-cache sequential stream 327.1 GB/s = **248.9×**. Neither is
DRAM headroom. The organ is occupancy.

---

## 1. Where the 2.443 ms goes

`seal_components` (`qwen38_token_ns_ledger.rs:410-414`) rebuilds the gqa
row from isolated families, scale = 1.0:

| piece | ns | % of row | class |
|---|---:|---:|---|
| rope leftover after KV stream | 1,526,669.82 | 62.48 | 16× 24-thread kernel |
| mha leftover after KV stream | 651,164.18 | 26.65 | 16× `mha_decode_f32` |
| GQA GEMV FMA remainder | 192,449.21 | 7.88 | 10.59% of `gqa_gemvs` |
| sigmoid | 43,625.00 | 1.79 | 16× elementwise |
| mixer residual × 16/64 | 29,562.50 | 1.21 | 16 of 64 adds |
| **sum** | **2,443,470.71** | 100 | matches RECEIPT exactly |

DERIVED, this lane, closed on receipt fields:

```
rope=1562625  mha=666500  sigmoid=43625
gqa_fma = 0.1058916672164676 * 1817416 = 192449.21026588368
mixer   = 118250 * 16/64              = 29562.5
kv_gqa  = 24375+26916                 = 51291
rope_after = rope - kv_gqa * rope/(rope+mha) = 1526669.8162956317
mha_after  = mha  - kv_gqa * mha /(rope+mha) =  651164.1837043683
gqa = 2443470.7102658837
```

Command output: `/tmp/gqa_geometry_numbers.py` § SEAL RECON (this lane, no GPU).

GQA weight GEMVs (1,817,416 ns isolated, 64 dispatches) live in
`weight_addressing` except the 192 µs FMA remainder. Gravity owns them.

`qwen38_qkvz_rearrange_conv_l2_f32` is **not** in this row. 48 DeltaNet
layers, isolated 350,999 ns, parked in `deltanet` after the conv-state
stream carve. Geometry is in §2.4 because the contract named it.

### 1.1 Isolated seq is 26, ledger bytes are seq=19  **CORR**

`ascension_qwen38_token_ns.rs:157` opens `max_seq_len=128`.
Default prompt `"Say hi."` chat-rendered is 11 ids (DERIVED:
`state_bytes.gqa_kv_read_bytes_at_pos = 2_490_368 = 16*2*seq*4096`
⇒ `seq=19`; `seq = prompt_len + max_new/2` at `:386` with
`max_new_tokens=16` ⇒ `prompt_len=11`).

`generate_greedy` (`qwen38_hybrid_decode.rs:3367`) resets, then 11 prompt
steps + 15 further decode steps. Isolated families run **after** rep 0
(`token_ns.rs:201-226`) with **no reset**. `position=26`.

| quantity | value | source |
|---|---|---|
| production mha `seq_len` | `position+1` | `hybrid_decode.rs:2788` |
| isolated mha `seq_len` | `position.max(1)=26` | `encode_mha` `:2048` |
| isolated rope slot | `position-1=25` | `encode_rope_cache` `:2028` |
| ledger unique-KV bytes | seq=19 | `:386` midpoint heuristic |
| occupancy string | `seq≈19` | ledger `:583` |

Wave-1 treated isolated `mha_16` as seq≈19. The 666,500 ns is seq=26.
Phase-1 idle is 79.7% at 26, 85.2% at 19. Production decode tokens in
the sealed sample sit at positions 11–25 (`production_steps`).

---

## 2. Per-kernel census

Production token = 1 + 64×15 + 3 = **964**
(`qwen38_token_ns_ledger.rs:56-59`). RECEIPT `dispatches` confirms.
Default Off path: `dispatch_threads` → `dispatch_threads` (Metal),
**one encoder per call** (`metal/mod.rs:4861-4873`). ICB / serial-group
/ concurrent-group unused on this `step()`.

### 2.1 `qwen38_gqa_qk_norm_rope_cache_f32` — 16 / 964

SOURCE: `qwen38_device_activations.metal:108-193`.
Host: `encode_gqa` `:2753-2778` and isolated `encode_rope_cache` `:2015-2041`.
Same launch.

| | |
|---|---|
| grid / TG | `(24,1,1)` / `(24,1,1)` = `QWEN38_GQA_HEADS` |
| TGs / threads | **1** / 24 |
| simdgroups | 24/32 = 0.75. `thread_execution_width=32` MEASURED `matvec-occupancy-230x.json` |
| TG/core if spread | 1/60 = 0.0167 |
| isolated | 1,562,625 ns, reps [1,562,625 / 1,570,666 / 1,550,500] RECEIPT |
| ns/launch | 97,664 DERIVED |

**Thread work (SOURCE, the dim loops).** Thread `head` owns one query head.
Serial 256-d Q RMS, then serial 256-d `(1+q_norm)` + rotate-half on first 64.
`pow`/`cos`/`sin` of `θ^{-2i/64}·pos` sit **inside** the dim loop
(`:145-156`). Peer dim is reloaded and renormalized when `dim` and when
`peer` is current. Threads `head<4` then do the same for K and memcpy V
into `cache_base = (sequence_slot * n_kv_heads + head) * head_dim` (`:170`).

Hard-fail unless `24/4/256/64/θ=1e7/eps=1e-6`.

**Idle of the launched grid.**

- Simdgroup pad: TG=24 is not a multiple of 32. 8/32 = **25%** of the
  one simdgroup unused. DERIVED from launch + MEASURED width.
- Q half: all 24 threads work (serial).
- K/V half: only heads 0–3 enter `:162`. **20/24 = 83.3% idle** for that
  entire path.
- Whole-device: 1 TG on 60 cores. 59/60 cores have no work.

**Bytes.** Unique R+W per launch 67,584 B (DERIVED from the loops:
Q-half 24,576 + q_norm 1,024 + K 4,096 + k_norm 1,024 + V 4,096 +
query 24,576 + Kc 4,096 + Vc 4,096). ×16 = **1,081,344 B**.
Issued loads if no L1 reuse: 104,448 + 32,768 writes = 137,216 B/launch,
×16 = 2,195,456 B (double-walk of Q/K plus 24× broadcast of `q_norm`
plus rotary peer rereads).

At 639.25 GB/s the unique-16 floor is **1,692 ns**. Isolated rope is
1,562,625 ns = **924×** that floor. DERIVED.

**Recomputes that do not depend on the token’s activations.**
32 unique frequencies (`rotary_dim/2`). Issued special-function triples:
Q `16×24×64 = 24,576` + K `16×4×64 = 4,096` = **28,672** / token.
Wave-1 counted Q only. **CORR.** A 32-entry cos/sin table per position
is cacheable. That ALU is not 1.56 ms — see §4.

Q80 ships the same 1-thread-per-head body
(`qwen80_device_activations.metal:317-333`) with
`TG = min(query_heads, 16)` (`qwen80_uniform_q4_hybrid_decode.rs:3565-3567`).
Q38 forked the constants and kept the map.

### 2.2 `mha_decode_f32` — 16 / 964

SOURCE: `mha.metal:602-681`. Host `mha_decode_f32_tcb`
`kernels/mod.rs:10506-10560`. Wrapper comment says “TG size 64”;
constant is `TG_SIZE = 128` (`:10545`). Comment is stale.

| | |
|---|---|
| grid / TG | `(24×128, 1, 1)=(3072,1,1)` / `(128,1,1)` |
| TGs / threads | **24** (one per query head) / 3,072 |
| simdgroups | 96 |
| TG/core | 24/60 = 0.40 |
| shmem | `(seq+128)×4` B. 588 B at 19, 616 B at 26. Cap ~8,064 tokens @ 32 KB |
| isolated | 666,500 ns, reps [744,999 / 653,625 / 666,500] RECEIPT |
| ns/launch | 41,656 DERIVED |
| isolated seq | **26** (see §1.1) |
| ledger byte seq | 19 |

`kv_h = h / 6`. Scale `1/sqrt(256)`. Layout read
`(t * NKV + kv_h) * H_DIM` — `[seq][kv_head][dim]`.

**Thread work by phase.**

| phase | what | who works at seq=26 | idle of 128 |
|---|---|---|---:|
| 1 | scalar 256-d dot vs K[t, kv_h] | tid `<26` | **79.7%** |
| 1 at seq=19 | same | tid `<19` | **85.2%** |
| 2 | tree-max of scores | 26, then 64/32/…/1 | high |
| 3 | exp + tree-sum | same as 2 | high |
| 4 | `out[i] += score[t]*V[t,kv_h,i]`, `i=tid; i<256; i+=128` | **all 128**, 2 passes | **0%** |

Wave-1 said “85% idle” as if the whole kernel. Phase 4 is full.
The occupancy hole is the score walk, not the V accumulate.

**Bytes at isolated seq=26 (DERIVED from the loops).**

| | unique / layer | issued / layer (6× GQA group) |
|---|---:|---:|
| Q | 24,576 | 26×24×1,024 = 638,976 |
| K | 106,496 | 638,976 |
| V | 106,496 | 638,976 |
| out | 24,576 | 24,576 |
| R+W | 262,144 | 1,941,504 |
| ×16 | 4,194,304 | 31,064,064 |

At ledger seq=19 unique R+W ×16 = 3,276,800; issued ×16 = 22,806,528.
Unique KV only at 19 = 2,490,368 (RECEIPT `gqa_kv_read_bytes_at_pos`).
6× over-read is SOURCE (`kv_h = h/6`, 6 query TGs share one kv head).

Consecutive tokens of one head sit `NKV*HD*4 = 4,096` B apart.
Phase 1: adjacent working threads are 4 KB apart (uncoalesced across
the simdgroup). Phase 4: inner walk is `t` at fixed dim — 4 KB stride
scalar gathers. Adjacent threads at fixed `t` hit consecutive dims
(512 B then a 4 KB jump).

FMA: `16×24×seq×256×2`. seq=19 → 3,735,552 (5.60 GFLOP/s at 666,500 ns).
seq=26 → 5,111,808 (7.67 GFLOP/s). DERIVED. Isolated time is the
seq=26 figure; the 5.60 number is the ledger-seq mix.

`mha_decode_flash_f32` exists (`mha.metal:720`). **Zero call sites in
`qwen38_hybrid_decode.rs`** (SOURCE). At seq=26 it is one 128-token tile
with 102 idle lanes — same Phase-1 hole. Flash removes the O(seq) shmem
cap. It does not fix 6× over-read, the layout, or rope.

### 2.3 `qwen38_attention_apply_sigmoid_gate` — 16 / 964

SOURCE: `qwen38_device_activations.metal:250-266`.
Host `:2795-2806`. Isolated `encode_sigmoid_gate` `:2064-2078`.

| | |
|---|---|
| grid / TG | `(6144,1,1)` = `24×256` / `(256,1,1)` |
| TGs / threads | 24 / 6,144 |
| TG/core | 0.40 |
| isolated | 43,625 ns, reps [44,833 / 40,416 / 43,625] |
| ns/launch | 2,727 DERIVED |
| idle of launched grid | **0%** (`elements == 24*256`, one thread each) |

`gated[h,d] = attn[h,d] * σ(q_proj[h, 256+d])`. Gate half is the second
256 of each `q_proj` row (`QWEN38_Q_PROJ_ROWS=12_288`). Unique R+W
73,728 B/launch ×16 = 1,179,648 B. Near the tiny-kernel floor on this
genome (silu 2,515 ns/launch, mixer residual 1,848).

### 2.4 `qwen38_qkvz_rearrange_conv_l2_f32` — 48 / 964 (DeltaNet)

SOURCE: `qwen38_device_activations.metal:32-105`.
Host `:2630-2654` / `:1934-1958`. `dispatch_threads((256, 16, 1), (256, 1, 1))`.

| | |
|---|---|
| TGs / threads | 16 (one per key head) / 4,096 |
| TG/core | 0.27 |
| TG mem | `4×256×4` B |
| isolated | 350,999 ns, 7,312 ns/launch |
| idle | Q/K conv `tid<128` → 50%; apply-scale `tid<128` → 50%; V/Z all 256 work (≥1 row of 384) |

Already fuses rearrange + conv-k=4 + silu + L2 + ×3 head repeat.
Hard-fail unless `16/3/128/128/k=4`. Unique R+W 573,440 B/layer
(qkvz 65,536 + conv_w 163,840 + conv_state R+W 245,760 + 4 outputs
98,304). ×48 = 27,525,120 B. Occupancy is not the rope disaster.
Time is 2–3× the tiny-kernel floor. Lives in `deltanet`, not `gqa`.

Folding this into vi via last-TG atomics, or wrapping it in one encoder
as a 1.0–1.5 ms win, is this morning’s MEASURED_NEGATIVE. See §5.

### 2.5 Named-chain dispatch totals

| kernel | times | of 964 |
|---|---:|---:|
| `qwen38_gqa_qk_norm_rope_cache_f32` | 16 | 1.66% |
| `mha_decode_f32` | 16 | 1.66% |
| `qwen38_attention_apply_sigmoid_gate` | 16 | 1.66% |
| `qwen38_qkvz_rearrange_conv_l2_f32` | 48 | 4.98% |
| **named chain** | **96** | **9.96%** |
| GQA mixer prefix (all 9) | 144 | 14.9% |
| ledger `gqa.dispatches` (rope+mha+sigmoid+residual) | 64 | 6.6% |
| GQA Q4 GEMVs (in `weight_addressing`) | 64 | 6.6% |

---

## 3. Honest ratio vs the 313× figure

Ledger method (`qwen38_token_ns_ledger.rs:476-477, 573-578`):

```
bytes_read  = 24*256*4*16 + gqa_kv_read_at_seq19 = 393,216 + 2,490,368 = 2,883,584
bytes_written = 5120*4*16 = 327,680
total = 3,211,264
floor = total / HONEST_DECODE_CEILING_GB_S(411.51)
measured_over_floor = 2,443,470.71 / 7,803.611 = 313.121
```

RECEIPT matches. 411.51 is the falsified Q80 unique-once 512 MiB point
(`g1-roof-falsification.md` R1; still hardcoded at ledger `:28`).

Same unique-once envelope, measured regimes (DERIVED, this lane):

| regime | GB/s | floor ns | 2,443,471 / floor | what it is |
|---|---:|---:|---:|---|
| named ledger roof | 411.51 | 7,803.611 | **313.12×** | RECEIPT method, not this box’s roof |
| measured addressing | 639.25 | 5,023.487 | **486.41×** | `TOKEN_NS_QWEN38.json` `weight_addressing.effective_gb_s` |
| single-address | 699.57 | 4,590.340 | 532.31× | wave-1 addressing probe |
| GQA-cache stream | 327.10 | 9,817.377 | **248.89×** | this organ’s tensors, see below |
| named peak | 819.0 | 3,920.957 | 623.18× | spec, not a measured GQA regime |

**CORR — what `stream_gqa_*` actually copied.**
`measure_f32_stream("gqa_key")` (`hybrid_decode.rs:2439-2445`) copies
`(gqa_key.length)/4` f32. Session `max_seq_len=128`, so one cache is
`16*128*4*256*4 = 8,388,608` B. Dest is half of key+value
(`token_ns.rs:175`, `gqa_n = (gqa_cache_f32_count+1)/2`) = same 8.39 MB.
Combined K+V stream = **16,777,216 B in 51,291 ns = 327.10 GB/s**.
That is the full resident 128-slot buffers, **not** unique-at-19
(2.49 MB). Wave-1’s “unique KV streams in 51 µs” is false.

Small-buffer sequential: 8.39 MB @ 24,375 ns = 344.15 GB/s (K),
311.66 GB/s (V). Rec-state stream of 151 MB is 646 GB/s R+W
(302 MB / 467,374 ns). GQA-cache copies are launch-inflated relative
to the large rec-state stream.

Honest statements, in order of how well they match the organ:

1. **Method-matched correction of 313×: 486× vs 639.25 GB/s** on the
   same unique-once 3,211,264 B.
2. **Organ-matched sequential regime: 249× vs 327.1 GB/s** on those
   same unique-once bytes (the GQA caches as actually streamed).
3. **Rope alone is 924× its unique 1.08 MB at 639.25 GB/s.**
4. **Same encoder path: rope 97,664 ns/launch vs sigmoid 2,727 = 35.8×.**
   Encoder issue floor on this genome is the 2–3 µs class. Rope is not
   encoder tax. G024 `top_three_attacks[2]` calling rope
   “encoder-per-tiny-kernel tax” is **false for this kernel**.

No quoted GB/s was measured on the rope or mha kernels themselves.
The statement that stands without a roof: after the ledger carves the
51 µs stream into `kv_state`, **the entire 2.443 ms remainder is
occupancy + serial ALU + 6× over-read + a 192 µs GEMV FMA scrap.**
It is not DRAM headroom.

---

## 4. Already fused / what is recomputed

Norm + rope + cache write is **already one pass**
(`qwen38_gqa_qk_norm_rope_cache_f32`). Asking “can those three fuse?”
is the wrong question. The missing geometry is the 256-wide dim axis.

| work | token-dependent? | cached? | verdict |
|---|---|---|---|
| RoPE 32 frequencies × pos | no (only `pos`) | no; 28,672 triples | **cacheable**. ESTIMATED ALU ≪ 10 µs once occupancy is fixed. Not the 1.56 ms. |
| Q RMS / `(1+q_norm)` / rope apply | yes | no | must recompute. Parallelize. |
| K RMS / rope | yes | written after | correct. Past K not recomputed. |
| V | memcpy only | written after | V GEMV dest could be the cache slot. ESTIMATED ~0. |
| Past K/V attention | n/a | yes, f32 | scores must be redone (new q). |
| q_norm / k_norm vectors | static | re-read 256+256 f32 ×16 | ignore. |

Workspace `q_proj`/`k_proj`/`v_proj`/`query`/`attn`/`gated_attn` are
**one buffer each**, overwritten every GQA layer (`:784-789`). A 16-layer
fused tail needs a stacked workspace (~1.3 MB f32, DERIVED
`16*(12288+1024+1024+6144+6144+6144)*4`) or stays serial across layers.
That is a workspace change, not a KV layout change.

---

## 5. Mechanism check vs this morning’s kills

DeltaNet tails (`g1-deltanet-geometry.md` D1/D2;
`g1-resident-harvest.md:374-379`):

| kill | mechanism | GQA analogue |
|---|---|---|
| **D1** fused_vi last-TG atomic | last-arriving TG needs device-scope visibility of `rec_out`; no `memory_order_device`; slower (8.11 vs 7.38 ms) and ids drifted | a fuse that waits on “last TG of another map” |
| **D2** one_encoder as 1.0–1.5 ms | limiter was 48 underfilled 16-wide rms launches, not encoder create; GPU == split; complete-token Δ 133–245 µs inside DIRTY | wrapping rope+mha+sigmoid in one encoder without rewriting the 24-thread walk |
| RMSNorm encoder-collapse (this morning, harvest `:388`) | 129 norms serial-encoder GPU == multi-encoder; limiter is 1-TG × 18 µs, not encoder | same |

Named REOPEN_IF from both: **remove the 1-TG launch tax itself**, still
a separate dispatch if needed.

| this-lane proposal | shares D1? | shares D2 / encoder-collapse? |
|---|---|---|
| P1 dim-parallel rope, 24 TGs × 256, still 16 dispatches | **NO** | **NO** — this **is** the named reopen |
| P2 one kernel, 24 TGs own a head end-to-end (rope prologue + mha + sigmoid epilogue) | **NO** — no cross-TG atomic | **NO** — real kernel merge, not encoder wrap |
| Encoder-share of the three, kernels unchanged | NO | **YES. KILL as a ms-class claim.** ESTIMATED save 32×2–3 µs isolated / ~11 µs production intra-CB gap |
| P3 one dispatch looping 16 layers in one kernel | if it uses last-TG handoff, **YES** | adjacent to megakernel D6 (`g1-fusion-persistent.md` P5, 4.4×) |
| Fuse sigmoid/mha into **o_proj GEMV** | NO | **YES D8** (tails-into-following-GEMV, prior −10.68 ms). KILL |
| 4 TGs (one per kv_head) for mha | NO | shares DeltaNet D4 occupancy shape (48-TG map lost 5.36×) |

P1/P2 are the legal fusion class DeltaNet already named: “epilogue of a
TG that already owns the work.” They do not wait on a device atomic.

---

## 6. Ranked geometry changes

Predictions are **PROJECTED from isolated-family GPU already inside the
sealed gqa row (scale=1.0)**. Not a new token measurement. Dispatch
cuts are SOURCE-countable. Do not sum overlapping rows: P1 ⊂ P2.
P6 stacks with P1/P2. P4 is absorbed by P2.

### R1. Parallelize rope on the dim axis — **pure kernel**. Rank 1.

Keep 16 launches. Grid `(256, 24, 1)` TG `(256, 1, 1)`. RMS = 256-wide
tree (same shape as `qwen80_residual_rmsnorm_f32` `:24-48`). Then each
thread does one dim of `(1+q_norm)` + rope + cache write. KV heads 0–3
still own the K/V path. Optional: 32-entry cos/sin table in TG mem
(kills the 28,672 recomputes; not the milliseconds).

| | |
|---|---|
| dispatch | 16 → 16 (964 unchanged) |
| isolated rope | 1,562,625 → **128,000–320,000** PROJECTED (16 × 8–20 µs). Residual-rmsnorm is 17.8 µs for a 5,120-d 1-TG reduce; 24 TGs of 256-d should sit between the 2.7 µs sigmoid floor and that 18 µs) |
| gqa row | 2,443,471 → **~1.01–1.20 ms** |
| sealed token | 35.23 ms → **33.80–33.99 ms** |
| live token | 39.33 ms → **37.90–38.09 ms** PROJECTED iff the 1.53 ms transfers |

G024 expected “~1.2 ms” from rope (`top_three_attacks[2]`). This is
that 1.2 ms, named as occupancy, not encoder collapse.

**KILLS if** a 24-TG rope family still measures ≥50 µs/launch
(≥800 µs isolated).
**REOPEN_IF** hardware counters show the 97 µs is not the 24-thread
serial walk.

Cheapest falsifier: isolated `rope_cache_16` A/B, same CB shape, GPU
timestamps, bit-id on `query` + `gqa_key` + `gqa_value`. No token run.
GPU lane. This lane must not.

### R2. Fuse rope + mha + sigmoid per GQA layer — **pure kernel**. Rank 2.

One kernel, 24 TGs, each TG owns one query head: R1 rope prologue,
existing 128-lane mha body (or 256), `out[i] *= σ(q_gate[i])` at the
Phase-4 store. No last-TG atomic. No workspace stack. No KV layout
change. Does not share D1/D2.

| | |
|---|---|
| named-chain GQA tails | 48 → 16 (−32). 964 → 932 |
| sigmoid | 43,625 → 0 |
| rope | as R1 |
| mha | 666,500 unchanged at seq=26 (same 24 TGs, same layout) |
| gqa row | → **~0.97–1.16 ms** |
| sealed token | → **33.76–33.95 ms** |

`dispatch_threads_pair_in_one_encoder` can stage two kernels / one
encoder without a new shader. That is D2. Do not ship it as the cut.
The shader fuse is the cut.

### R3. Retile KV to `[layer][kv_head][dim][seq]` and stop the 6× reread — **cache layout**. Rank 3.

Current `[seq][kv_head][dim]` makes consecutive tokens of one head
stride 4,096 B and makes six query TGs reread the same head.

`[kv_head][dim][seq]`: Phase-4 inner `t` walk becomes contiguous.
`[kv_head][seq][dim]`: Phase-1 dim walk becomes 1,024 B blocks
(llama b9430 already does `float4` over dim, `mha.metal:46+`).
Cannot have both without a TG-mem transpose. Prefer `[kv_head][dim][seq]`
if Phase 4 stays the inner loop.

Sharing: keep **24 TGs**. Do not drop to 4 TGs (one per kv_head) —
that is the DeltaNet 48-TG occupancy fail. Load a K/V token-tile into
TG memory once per group of 6, or split dim so `4 kv × N` TGs stay ≥24.

| | |
|---|---|
| dispatch | 16 → 16 |
| unique KV bytes | unchanged (2.49 MB @19 / 3.41 MB @26) |
| reread | 6.0× → 1.0× |
| isolated mha | 666,500 → **180,000–350,000** PROJECTED at seq=26 |
| gqa row stacked on R2 | −0.32 to −0.49 ms |
| sealed token stacked on R2 | → **33.27–33.63 ms** |
| needs | rope write `cache_base` and mha reads change together |

**KILLS if** isolated `mha_16` after retile+vectorize stays within 15%
of 666,500 ns.
**REOPEN_IF** production seq ≫ 26: 6× and stride scale with seq.

Cheapest falsifier: host permute of captured K/V + one-file retiled
shader, isolated `mha_16` at seq=26 and seq=256. GPU lane.

### R4. Sigmoid epilogue only — **pure kernel**. Rank 4 (absorbed).

Fold into `mha_decode_f32` store. Dispatch −16. Isolated 43,625 ns.
PROJECTED token −0.04 ms. Ship only if R2 is deferred.

### R5. Encoder-share of rope+mha+sigmoid, kernels unchanged — **KILL**.

Shares D2 and this morning’s RMSNorm encoder-collapse. GPU of the
24-thread walk does not move. PROJECTED ≤80 µs isolated, ~11 µs
production. Not millisecond-class.

### R6. `mha_decode_flash_f32` at seq≈26 — **KILL as a token win**.

Already exists. Same 24 TGs, same layout, one tile, 102/128 idle,
more barriers. PROJECTED ≤0. **REOPEN_IF** seq ≫ 128 (shmem cap).

### R7. One dispatch for all 16 GQA tails — optional, after R2.

Needs stacked Q/K/V/attn (~1.3 MB). Adjacent to megakernel D6 if it
loops layers inside one kernel. Extra vs R2 is encoder×15,
PROJECTED 30–80 µs. Not first.

### R8. ICB — not a GQA proposal.

964 commands still execute. Encode 886 → 91 µs MEASURED on a later
commit, **not this `step()`**. Does not move the 97 µs rope. Ceremony.

### Combined independent path (no gravity)

| step | 964→ | gqa 2.443 ms → | sealed 35.23 ms → | class |
|---|---:|---:|---:|---|
| R1 rope occupancy | 964 | 1.01–1.20 | 33.80–33.99 | pure kernel |
| + R2 fuse tails | 932 | 0.97–1.16 | 33.76–33.95 | pure kernel |
| + R3 KV retile + share | 932 | 0.48–0.84 | 33.27–33.63 | **layout** |
| + R7 stack-16 (optional) | 917 | 0.45–0.81 | 33.24–33.60 | workspace, not KV |

Ceiling of everything this lane can touch without touching weights:
**about 1.6–2.0 ms** off the sealed wall. Not 100 TPS. 100 TPS still
requires the 21.3 ms addressing cut. This is the gqa-activation remainder.

qkvz/conv: at most ~0.1–0.2 ms if 48 rearranges collapse — DeltaNet
bucket, and only via a geometry that does **not** share D1/D2
(e.g. still-separate 128-thread rms, their named reopen).

---

## 7. What this lane will not claim

- Hardware occupancy counters. UNMEASURED. Cheapest: GPU-lane
  `MTLCounterSampleBuffer` / frame capture on one rope dispatch and
  one mha at seq=26.
- Token-level A/B. Isolated-family GPU is the evidence; production
  transfer is PROJECTED (scale was 1.0).
- Flash or f16/int4 KV as a seq-26 win.
- Encoder-share or last-TG fusion as a millisecond GQA win.
- Generator+residual, mixed-sub-1.5 expand vehicles, Q80/DSV4F
  resurrection. Transfer only: Q80 has the same 1-thread rope body;
  llama b9430 has `float4` MHA; ICB is ceremony.
- Any low-BPW path that expands to float/Q4 then this GEMV.

---

## 8. Evidence

### 8.1 Receipts (`git show`, not materialized)

`HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`

```
schema hawking.ascension.qwen38_token_ns_ledger.v1
median_wall_ns 35227917  median_gpu_ns 33912333
dispatches {embed:1, mixer_prefix:576, mlp_suffix:384, terminal:3, total:964}
state_bytes.gqa_kv_write_bytes 131072
state_bytes.gqa_kv_read_bytes_at_pos 2490368
closure.gpu_scale_applied 1.0
closure.isolated_family_sum_gpu_ns 33575407
components[gqa].ns_per_token 2443470.7102658837
components[gqa].measured_over_floor 313.1205132874512
components[gqa].bytes_read 2883584  bytes_written 327680
components[gqa].effective_gb_s 1.3142224240742257
isolated.rope_cache_16 1562625  [1562625, 1570666, 1550500]
isolated.mha_16        666500   [744999, 653625, 666500]
isolated.sigmoid_16     43625   [44833, 40416, 43625]
isolated.gqa_gemvs    1817416
isolated.rearrange_48  350999
isolated.mixer_residual_64 118250
isolated.stream_gqa_key   24375
isolated.stream_gqa_value 26916
probes[gqa].fma_remainder_frac 0.1058916672164676
```

`HEAD:receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json`
`weight_addressing.effective_gb_s = 639.2522341137478`.

`HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`
`ranked_by_ns[3] gqa 2443471 ns 6.94%`.
`top_three_attacks[2]` names rope as “encoder-per-tiny-kernel tax”
and “alternate 1.2 ms”. Occupancy diagnosis replaces that framing.

`HEAD:receipts/ascent-2026-08-16/matvec-occupancy-230x.json`
`thread_execution_width: 32`, `device_name: Apple M3 Ultra`,
`max_total_threads_per_threadgroup: 1024`.

### 8.2 This-lane python (no GPU)

`python3 /tmp/gqa_geometry_numbers.py` printed values used in §1–§3, §6.

```
gqa 2443470.7102658837
rope_after 1526669.8162956317  mha_after 651164.1837043683
floor 411.51: 7803.611 ns  313.121x
floor 639.25: 5023.487 ns  486.409x
floor 327.10: 9817.377 ns  248.892x
rope unique16 1081344  floor639 1691.58 ns  923.77x
rope/sigmoid per launch 35.819
resident_one 8388608  stream both 327.099 GB/s
mha p1 idle seq19 0.8515625  seq26 0.796875
rope recomputes 28672  unique_freq 32
isolated mha fma seq26 5111808  7.670 GFLOP/s
```

### 8.3 Source

```
crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:17-39
crates/hawking-core/src/model/qwen38_geometry.rs:38-41,82-93
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:760-799,1934-1958,
  2015-2078,2431-2454,2630-2654,2753-2827,3292-3310,3367-3413
crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:28,56-59,118-136,
  338-414,476-477,573-586
crates/hawking-core/src/kernels/mod.rs:10506-10560
crates/hawking-core/src/metal/mod.rs:4861-4873
crates/hawking-core/shaders/qwen38_device_activations.metal:32-266
crates/hawking-core/shaders/mha.metal:602-681,720
crates/hawking-core/examples/ascension_qwen38_token_ns.rs:157,175,201-226,386
crates/hawking-core/shaders/qwen80_device_activations.metal:24-48,317-333
crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:3564-3567
```

### 8.4 Wave reports consumed, not re-derived

- `g1-token-anatomy.md` isolated table, 964 encoder path
- `g1-kernel-inventory.md` §2–3 census (TG-vs-thread miscount on vi noted, not reused)
- `g1-deltanet-geometry.md` D1/D2/D4/D6/D8
- `g1-resident-harvest.md:374-388` tails + RMSNorm encoder-collapse kills
- `g1-fusion-persistent.md` P4b/P5 megakernel, P6 ICB
- `g1-roof-falsification.md` 411.51 is not the roof; 639.25 is
- `g1-gqa-and-attention-geometry.md` wave-1 attempt; corrections in §1.1, §2.1 recomputes, §3 stream size

---

## 9. Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. Sealed gqa row 2,443,471 ns / 6.94% reconstructs as rope leftover 1,526,670 (62.5%), mha leftover 651,164 (26.6%), GEMV FMA 192,449 (7.9%), sigmoid 43,625 (1.8%), mixer residual 29,562 (1.2%). Evidence: QWEN38_TOKEN_NS_LEDGER.json components[gqa]+isolated+probes; reconstruction §1.
C2. Named chain is 96/964 (16 rope + 16 mha + 16 sigmoid + 48 rearrange). Rearrange is DeltaNet, not the gqa row. Evidence: schedule.rs:17-39; hybrid_decode.rs encode sites.
C3. Rope launches 1 TG × 24 threads (0.75 simdgroup, 0.017 TG/core). Isolated 97,664 ns/launch vs sigmoid 2,727 on the same encoder path (35.8×). Diagnosis is occupancy + serial 256-d RMS, not encoder tax and not ICB. Evidence: hybrid_decode.rs:2753-2756; metal/mod.rs:4861-4873; isolated families.
C4. Isolated mha_16 ran at seq=26 (position after first generate), not ledger seq=19. Phase-1 idle 79.7% at 26 / 85.2% at 19. Phase 4 is full. 6× K/V over-read and 4,096 B stride are SOURCE. Evidence: token_ns.rs:201-226; encode_mha :2048; generate_greedy :3367; mha.metal:626-679.
C5. stream_gqa_* copied the full max_seq=128 resident caches (8.39 MB each, 327.1 GB/s combined), not unique-at-19. Wave-1 “51 µs streams 2.49 MB” is false. Evidence: hybrid_decode.rs:2439-2454; token_ns.rs:157,175; isolated stream medians.
C6. 313× is vs falsified 411.51. Honest method-matched ratio is 486.4× vs measured 639.25. Organ-matched sequential regime is 248.9× vs 327.1 GB/s. Rope alone is 924× its unique 1.08 MB at 639.25. Not DRAM headroom. Evidence: §3; TOKEN_NS_QWEN38.json weight_addressing; this-lane python.
C7. Norm+rope+cache-write is already one kernel. Cacheable leftover is the 32-frequency table (28,672 triples). That ALU is not the 1.56 ms. Evidence: qwen38_device_activations.metal:108-193.
C8. Encoder-share of the GQA tails shares this morning’s D2 and RMSNorm encoder-collapse kills. Last-TG fusion into a later map shares D1. Fuse into o_proj shares D8. P1/P2 do not share any of those. Evidence: g1-deltanet-geometry.md D1/D2; g1-resident-harvest.md:374-388; §5.
C9. Ranked independent cuts: R1 dim-parallel rope (pure kernel, Δdisp=0, PROJECTED −1.24 to −1.43 ms); R2 fuse tails (pure kernel, 964→932); R3 KV retile+share (layout, stacks −0.32 to −0.49 ms). Combined ceiling ≈ 1.6–2.0 ms. Not 100 TPS. Evidence: §6.

EVIDENCE
HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json fields in §8.1
HEAD:receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json weight_addressing.effective_gb_s
HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json ranked_by_ns[3], top_three_attacks[2]
HEAD:receipts/ascent-2026-08-16/matvec-occupancy-230x.json thread_execution_width
pointers in §8.3
workspace/superwave/g1/g1-deltanet-geometry.md §5, §7 D1/D2
workspace/superwave/g1/g1-resident-harvest.md:374-388
workspace/superwave/g1/g1-roof-falsification.md R1, R18
this-lane python /tmp/gqa_geometry_numbers.py printed in §8.2

CHANGES
workspace/superwave/g1/g1-gqa-geometry-redo.md  (new, this file)

TESTS
test -s workspace/superwave/g1/g1-gqa-geometry-redo.md
wc -l workspace/superwave/g1/g1-gqa-geometry-redo.md
git status --porcelain

RISKS
Isolated-family GPU includes that family’s own encoder gaps. Production intra-CB gap is 337 µs / 964 ≈ 350 ns/dispatch; isolated rope’s 16 gaps are ~6 µs, so 1.56 ms is almost all kernel. A fused kernel can still move gap accounting; token A/B required before MEASURED_WIN.
R1 band (128–320 µs) is not a hardware-occupancy measurement. Sigmoid at 2.7 µs with 24 TGs argues against a 97 µs per-dispatch GPU minimum.
R3 must keep ≥24 TGs. A 4-TG (one per kv_head) rewrite recreates the rope disaster on MHA.
DIRTY_ENGINEERING on every receipt. Isolated mha is seq=26; production decode tokens are 12–26. Relative deltas only.

UNRESOLVED
Hardware occupancy counters on rope and mha. Cheapest: GPU-lane counter sample. Forbidden here.
Whether V GEMV can legally target value_cache[pos] without breaking mixed / concurrent_independent.
Live 39,326,090 vs sealed 35,227,917 session split: this lane did not remeasure.
maxThreadgroupMemoryLength on this MTLDevice (R3 tile size).

NEXT
GPU lane: isolated rope A/B of R1 (24 TGs × 256) vs HEAD, bit-id on query+KV, GPU timestamps. If ≥1.2 ms falls, implement R2. Then R3 at seq=26 and seq=256. Do not retry encoder-share. Do not swap flash at seq=26. Do not wait on gravity.
```
