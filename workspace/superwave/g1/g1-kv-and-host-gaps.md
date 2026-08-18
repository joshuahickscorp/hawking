# G1 — KV cache, sampling, host waits, GPU idle gaps

Lane: `66-kv-and-host-gaps`. No GPU. No inference. No live Genesis touch.
One new file. Every number is **MEASURED** (receipt / production_steps already
on disk), **DERIVED** (geometry arithmetic), **PROJECTED** (measured scaled
outside its range), or **ESTIMATED**. A component microbench is not a token.

Identity bookkeeping and per-token host round trips were dominant on three
prior genomes. On this genome they are present and **not** the wall.

---

## 0. Genome this is of

`Qwen38HybridDecodeSession` + `geo_tpr64_tg128` + `gated_delta_vi` +
`qwen38_gqa_qk_norm_rope_cache_f32` + `mha_decode_f32` + `sample_argmax_f32`.
1 production CB / 964 dispatches / 1 `waitUntilCompleted` / 1 four-byte host
read. `concurrent_independent=false`. ICB **not** on this step path
(`HAWKING_QWEN38_ICB` / `ReplayableComputeGraph` unused in
`qwen38_hybrid_decode.rs`; `git merge-base --is-ancestor 7400acf1b HEAD` →
exit 1).

Default `max_seq_len=128` (`ascension_qwen38_hybrid_greedy.rs:68`,
`ascension_qwen38_token_ns.rs:156`). Seated wall prompt is 11 tokens
`Say hi.` chat (`QWEN38_COMPLETE_TOKEN_WALL.json` `identity.prompt_len=11`).
Ledger `state_bytes.gqa_kv_read_bytes_at_pos=2490368` ⇒ live seq **19**
(DERIVED: `16*2*19*4096`).

Two walls, do not add:

| wall | ns | GPU | wall−gpu | source |
|---|---:|---:|---:|---|
| G024 TOKEN_NS (encode+submit+wait) | 35,227,917 | 33,912,333 | 384,250 | `QWEN38_TOKEN_NS_LEDGER.json` |
| complete-token (every recurring host) | 38,216,792 | 36,987,458 | 1,229,334 | `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` |

---

## 1. Where the GPU is idle inside a token, and why

Three different “idle”s. Do not collapse them.

### 1.1 Queue idle — GPU has no CB (MEASURED host Instants)

Complete-wall mean split of wall−gpu **1,332,224 ns**
(`authority_decomposition` / `wall_minus_gpu_named.components_ms`):

| when | ns | % of wall−gpu | why GPU is idle |
|---|---:|---:|---|
| encode_host_prepare | 886,210 | **66.52** | host builds 964 encoders before `commit` |
| wait_minus_gpu | 425,929 | 31.97 | `waitUntilCompleted` minus `GPUEnd−GPUStart`: start delay + completion notify |
| submit | 10,457 | 0.79 | `commit` |
| tokenizer_decode_new_token | 6,306 | 0.47 | host, after wait |
| commit_epilogue (timestamps/status) | 1,789 | 0.13 | after wait |
| sample_readback (4 B) | 877 | 0.07 | after wait |
| Instant gap + bookkeeping + position | 655 | 0.05 | after wait |

Encode is **67% of host**. GPU cannot start until encode+commit finish.
G024 encode 919,250 / wait−gpu 384,250 — same organs, different dirty session.

Timeline of `step()` (`qwen38_hybrid_decode.rs:3292-3310` +
`metal/mod.rs:5330-5371`):

```
encode 964 encoders     GPU idle
commit                  GPU starts
waitUntilCompleted      GPU runs ~34–37 ms, then notify
contents() u32 load     GPU idle
position += 1           GPU idle
```

One wait, one four-byte read. Confirmed.

### 1.2 Device bubble — CB is running, cores empty between encoders (MEASURED residual)

TCB default `dispatch_threads` does `new_compute_command_encoder` + dispatch +
`end_encoding` per call (`metal/mod.rs:4856-4873`). Qwen3.8 never calls
`enable_ordered_encoder` / `begin_serial_group`. Production is **964 compute
encoders inside 1 CB**.

G024 `unattributed_residual` = **341,925 ns**. Method string:
`embed_row_gather (4999) + intra_cb_encoder_transition_gap + host_tail (0)`.
Production GPU 33,912,333 − isolated exclusive sum 33,575,407 = **336,926 ns**
≈ 349 ns/encoder × 964. That is the intra-CB encoder tax, **inside**
`GPUEnd−GPUStart`. It is not wait−gpu.

`concurrent_independent=true` exists and is **off**. Prior isolated MLP win
lost the token (`qwen38-layer-dense-q4-swiglu.json`); do not flip it here.

### 1.3 Occupancy-starved — work is queued, few threads (MEASURED isolated, not idle)

Not queue-idle. Cores underfilled:

| kernel | launch | isolated GPU | note |
|---|---|---:|---|
| `qwen38_gqa_qk_norm_rope_cache_f32` | grid=24, TG=24 | 1,562,625 (16 layers) | 24 threads; only 4 write KV |
| `mha_decode_f32` | 24 TGs × TG=128 | 666,500 (16 layers, seq≈19) | Phase 1: 19 of 128 threads work |
| `sample_argmax_f32` | 1 TG × 256 | 335,499 | one TG walks 248,320 f32 |
| 129 RMSNorm + 48 gated_rmsnorm | TG=256 / 16 | 2.37 + 1.30 ms | G024 attacks #2/#3 |

These are compute/launch, not host round trips. Fusion lane already owns them.

### 1.4 Identity bookkeeping — **KILLS** as the wall on this genome

`state_update` 24 ns, `bookkeeping` 80 ns, `sample_readback` 877 ns
(MEASURED complete-wall means). One wait, not 48 route-id waits (Q30) and not
1,171 GLM `commit_and_wait`s. Presence confirmed; dominance **FALSIFIED**.
**REOPEN_IF** a path reintroduces per-layer host readback.

---

## 2. KV cache layout and bytes

Qwen3.8 has two state families. Only GQA grows with seq.

### 2.1 GQA cache (16 layers)

Storage: two f32 buffers `gqa_key`, `gqa_value`.

```
elems_per_layer = max_seq_len * n_kv_heads * head_dim
                = max_seq_len * 4 * 256
layer_byte_off  = slot * elems_per_layer * 4
slot            = layer / 4          // 16 exclusive slots
```

(`qwen38_hybrid_decode.rs:351-357,767-768,2717-2719`;
`qwen38_gqa_state_slot` at `qwen38_geometry.rs:106-109`.)

On-device index (`qwen38_device_activations.metal:170`):

```
cache_base = (sequence_slot * n_kv_heads + head) * head_dim
```

Layout per layer: **`[seq][kv_head=4][head_dim=256]` f32**.
MHA comment agrees (`kernels/mod.rs:10502-10503`):
`k_cache (seq_len, n_kv_heads, head_dim)`.

Not `[kv][seq][dim]`. Not quantized. Not paged.

Write (already inside the rope kernel, not a extra dispatch):

```
key_cache[cache_base + dim]  = RMSNorm(K) + rotate_half(first 64 of 256)
value_cache[cache_base + dim] = v_proj[kv_base + dim]   // raw copy
```

(`qwen38_device_activations.metal:162-191`.) K cannot skip this kernel
(needs RMS+RoPE). V is a copy of the v-GEMV output.

### 2.2 Bytes (DERIVED; matches ledger at seq=19)

`kv_one = 4 * 256 * 4 = 4,096` B (one K or V vector, one layer, one pos).

| quantity | formula | bytes |
|---|---|---:|
| GQA write / token | `16 * 2 * 4096` | **131,072** |
| GQA read / token at seq S | `16 * 2 * S * 4096 = 131,072 * S` | see table |
| rec_state resident (48 DN) | `48 * 48 * 128 * 128 * 4` | **150,994,944** |
| rec R+W / token | ×2, seq-independent | **301,989,888** |
| conv_state resident | `48 * 10240 * 3 * 4` | **5,898,240** |
| conv R+W / token | ×2 | **11,796,480** |

Live GQA traffic vs 13,618,141,856 active weight bytes:

| seq | GQA read | +write | % of active weights | 400 GB/s floor (KV only) |
|---:|---:|---:|---:|---:|
| 1 | 131,072 | 262,144 | 0.002% | 0.7 µs |
| 19 (ledger) | 2,490,368 | 2,621,440 | 0.019% | 6.6 µs |
| 128 (default max) | 16,777,216 | 16,908,288 | 0.124% | 42 µs |
| 2048 | 268,435,456 | 268,566,528 | 1.97% | 671 µs |
| 4096 | 536,870,912 | 537,001,984 | 3.94% | 1.34 ms |
| 8192 | 1,073,741,824 | 1,073,872,896 | 7.88% | 2.68 ms |

Ledger `state_bytes` at seq 19: conv R+W 11,796,480; rec R+W 301,989,888;
GQA write 131,072; GQA read 2,490,368; total R+W **316,407,808**. Matches.

Resident workspace (`qwen38_workspace_bytes`, this-lane recomputation):

| max_seq | GQA K+V | DeltaNet state | activations | total |
|---:|---:|---:|---:|---:|
| 128 | 16.0 MiB | 149.6 MiB | 1.61 MiB | **167.24 MiB** |
| 2048 | 256.0 MiB | 149.6 MiB | 1.61 MiB | 407.24 MiB |
| 4096 | 512.0 MiB | 149.6 MiB | 1.61 MiB | 663.24 MiB |
| 8192 | 1024.0 MiB | 149.6 MiB | 1.61 MiB | 1175.24 MiB |

`kv_is_the_seq_len_term` (`qwen38_hybrid_decode.rs:3878-3887`): activation
and DeltaNet bytes are invariant; GQA doubles 2048→4096. Test is the
authority that KV is the only seq-length term.

`mha_decode_f32` threadgroup: `(seq_len + 128) * 4` bytes. 32 KB cap ⇒
**seq ≤ 8,064**. Flash sibling `mha_decode_flash_f32` exists
(`mha.metal:720`) and is **not** the Qwen3.8 default.

### 2.3 Does growth change token time?

**At seated length (seq 11–25): yes, barely, MEASURED. At realistic long
context: UNMEASURED; linear extrapolation from the starved window is not a
token claim.**

Ledger `production_steps` (3 paired generates, positions 0–25, same 964-dispatch
step). Median GPU vs position, decode-only pos 11–25:

```
OLS: gpu_ns = 33,047,779 + 45,874 * position
R² = 0.868
pos11 median 33,523,750 → pos25 34,105,875   Δ +582,125 ns
```

Slope **45,874 ns per additional context token** over that 14-token window
(MEASURED, n=3 dirty reps). Isolated `mha_16` 666,500 / 19 = 35,079 ns/ctx
if the whole family were linear — same order.

16-new-token complete wall 38,543,084 vs 32-new-token 38,216,792
(`g015_length_confirmation`). **Opposite** direction, different session.
16 extra tokens × 46 µs = 0.74 ms PROJECTED, smaller than session GPU
movement (36.99 vs 37.14 ms). That pair does **not** resolve growth.

Why the short-window slope is not a 2k roof:

- TG=128, SEQ=19 ⇒ Phase 1 uses 19 threads. Occupancy-starved.
- At SEQ≥128 every Phase-1 thread works. Slope should **fall**, not hold.
- KV byte floor at 400 GB/s is 328 ns/ctx-token. Measured 45,874 is **140×**
  that floor — the slope is MHA ALU/occupancy, not DRAM of the cache.
- PROJECTED if slope held: seq 128 +5.0 ms, seq 2048 +93 ms vs seq 19.
  Label **PROJECTED, likely high**. Cheapest closer: isolated `mha_decode_f32`
  at seq ∈ {19, 128, 512, 2048} under the GPU lock. Not run here.

`kv_state` ledger row **537,665 ns** is the sequential f32 **stream probe**
of resident rec+conv+allocated GQA (capped by parent), **not** MHA.
`stream_rec_state` 467,374 + `stream_conv_state` 19,000 +
`stream_gqa_key` 24,375 + `stream_gqa_value` 26,916 = 537,665.
Rec state is 467 µs / 151 MB ≈ 646 GB/s COMPONENT. GQA cache is a rounding
error next to rec at short seq.

### 2.4 Coalesced?

**Write (rope kernel).** 24 threads, one per query head. Only `head < 4`
store. Each store walks 256 consecutive f32s at
`(pos * 4 + head) * 256`. Contiguous per head. Adjacent KV heads at the
same pos are adjacent 1 KiB vectors. Not a simdgroup-wide coalesced store
(one thread owns a head). Occupancy is the tax, not the store stride.

**Read Phase 1 (`mha.metal:627-634`).** Thread `tid` owns time `t`.

```
kt = k_cache + (t * NKV + kv_h) * H_DIM
```

Thread t vs t+1: stride `NKV * H_DIM * 4 = 4,096` B. **Uncoalesced** across
the TG. Inner `i=0..256` is sequential for that thread.

**Read Phase 4 (`mha.metal:673-679`).** Thread `tid` owns dim `i`. Adjacent
threads read adjacent floats of the same V vector — **coalesced**. Then all
threads jump `4,096` B to the next `t`.

6 query heads share each KV head (`GROUP=24/4`). Six TGs reread the same
K/V stream — reuse, not extra bytes.

A `[kv][seq][dim]` retile would cut Phase-1 stride from 4 KiB to 1 KiB and
still not coalesce (want 4 B). Not a G1 lever at seq=19 (6.6 µs floor vs
667 µs isolated). **REOPEN_IF** a long-context mha timestamp shows the
uncoalesced Phase-1 walk in the millisecond class **and** a layout A/B
beats it without hurting Phase 4.

### 2.5 Can the cache write fuse with the kernel that produces V?

**Already fused with the consumer of V, not with the producer.**

Producer: `geo_tpr64_tg128` on `v_proj.weight` → `v_proj` (1,024 f32).
Consumer/writer: `qwen38_gqa_qk_norm_rope_cache_f32` copies `v_proj` into
`value_cache[pos]`. No standalone “append KV” dispatch.

Fusing the copy into the v GEMV: legal. Destination packing at one `pos`
is `[kv_head][dim]`, same as `v_proj`. Would skip 16 × 4,096 = 65,536 B
of intermediate V (ESTIMATED 0.16 µs at 400 GB/s) and a few stores inside
a 24-thread kernel whose isolated family is 1.56 ms for Q+K RMS+RoPE.
**KILLS** as a TOKEN_NS lever. **REOPEN_IF** a fused `v_proj→cache` A/B
shows a complete-token delta, not a copy microbench.

K write cannot move into the k GEMV (RoPE+RMS live in the rope kernel).

---

## 3. Sampling and the four-byte host read

### 3.1 What sampling is

G0 is **device greedy argmax**. No temperature / top-p / top-k on this path
(`sample_argmax_f32_tcb` → `sample_argmax_f32`, grid/TG (256,1,1),
`sample.metal:48-75`, `kernels/mod.rs:14232-14248`). Vocab 248,320 f32
logits (993,280 B) + 4 B id.

Isolated `argmax` **335,499 ns** GPU (ledger `isolated`). Medium confidence
as a token-level number: one tiny CB. Ledger parks it in `terminal_head`
(383,535 ns = argmax + lm_head FMA remainder). Host `sample_readback`
**877 ns** is the 4-byte load, **not** the sampler
(`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` `sample_readback` 0.0009 ms).

### 3.2 Does the four-byte read force the wait?

**The host needing the id for the next embed `set_bytes` forces the wait.
The load itself is 877 ns after the wait has already returned.**

`encode_embed` binds the token as `set_bytes` u32
(`qwen38_hybrid_decode.rs:2541`). Next `step(token)` cannot be encoded
until the host holds the id. `contents()` after `waitUntilCompleted` is
the visibility fence on unified memory (`step` 3304-3308;
`commit_and_wait_split` 5370-5371).

Removing the 877 ns load and keeping the wait saves 877 ns. Noise.

Removing the wait requires the next embed to read `sampled` on device.

That kernel **exists** and is **not wired** on Qwen3.8:

```
qwen_uniform_q4_embedding_lookup_device_token   // qwen_uniform_q4.metal:604-622
```

Comment: “previous step's argmax id stays in a device buffer and is
gathered without a host round-trip.” Bit-identical to the host-token
gather for the same id. Registered in `metal/mod.rs:1232-1233`. Used by
Q80/Q30 complete runtime. **Zero hits** in `qwen38_hybrid_decode.rs`.

`commit_no_wait` exists (`metal/mod.rs:5177-5181`) for exactly this:
enqueue token N+1 while N runs. Qwen3.8 `step` calls `commit_and_wait_timed`.

`position` and `mha_seq_len=position+1` are host-known without the sampled
id. Only the token id is the data dependency.

Host still wants the id for EOS (`QWEN38_EOS_IM_END` / `END_OF_TEXT`) and
tokenizer decode (`generate_greedy` / `step_complete`). Those can defer to
a drain every N tokens or a device EOS flag. Unmeasured on Qwen3.8.

### 3.3 What device-side chaining recovers

| keep per-token wait for EOS/tokenizer? | encode | wait−gpu | readback | PROJECTED recover |
|---|---|---|---|---|
| yes, but encode-ahead + device token + `commit_no_wait` | hidden under GPU N | still paid | 877 ns | **≈886 µs** (encode) |
| no; drain ids at end / every N | hidden | deleted | deleted | **≈1.33 ms** complete-wall host (encode+wait−gpu+submit+tok+epilogue) |

GPU body 33.9–37.0 ms still runs. Chaining is a host-gap delete, not a
10 ms path.

ICB successor already cut encode to 90,981 ns and **raised** wait−gpu to
561,994 ns (`QWEN38_FIXED_OVERHEAD_DELETED.json`). After ICB, encode-ahead
is only ~91 µs; the leftover host is wait−gpu. Multi-token ICB + device
token is what deletes that. ICB still writes 3 u32s/token
`(token, position, mha_seq_len)` — token write dies if the id stays on
device; position/seq_len stay host-known.

---

## 4. Ranked removable host round trips and idle gaps

Predicted ns are **recoverable from the named organ**, not a promise the
complete wall moves by exactly that after second-order GPU noise.
ICB wall Δ included dirty GPU movement 36.987→36.012 ms; the **clean**
ICB ceremony claim is named-fixed **−660,000 ns**.

### 4.1 Achievable at HEAD (no ICB, no new shader family)

| # | gap | predicted recover | how | confidence |
|---:|---|---:|---|---|
| H1 | Encode/GPU non-overlap | **886,210 ns** PROJECTED | wire `embedding_lookup_device_token` + `commit_no_wait` + encode-ahead. Kernels and TCB primitive exist. | HIGH mechanism, MEDIUM ns (overlap hides encode; does not delete wait) |
| H2 | Intra-CB 964-encoder bubble | **337,000 ns** ESTIMATED ceiling | `begin_serial_group` / one encoder. Primitive exists on `CommandBatch`, not on the TCB path Qwen3.8 uses. Hazard-tracking parity required. | LOW. ICB already deletes this better. Do not spend the GPU lock on both. |
| H3 | Tokenizer per token | **6,306 ns** MEASURED | decode at drain, not every step | HIGH, noise |
| H4 | 4-byte `contents()` | **877 ns** MEASURED | follows from H1 | HIGH, noise |
| H5 | Fuse V-cache write into v GEMV | **< 10,000 ns** ESTIMATED | skip 65 KiB intermediate | HIGH it is tiny |
| H6 | Flip `concurrent_independent` | **negative on the token** MEASURED prior | isolated MLP won, token lost | KILLS |

H1+H3+H4 ceiling ≈ **0.89 ms** if EOS still waits every token.
H1 without per-token wait (device EOS or deferred drain) ceiling ≈ **1.33 ms**.

Not 10 ms. Not identity bookkeeping.

### 4.2 Needs ICB land and/or GPU-side chaining (not on HEAD)

| # | gap | predicted recover | how | confidence |
|---:|---|---:|---|---|
| I1 | 964× encode ceremony | **−795,219 encode, +136,094 wait−gpu, net −660,000 named fixed** MEASURED | land `7400acf1b`. 964 ICB commands, 64 KiB slab, 3 u32s/token, 1 `executeCommandsInBuffer`. Coherence PASS 3 prompts. | HIGH (receipt). IMPLEMENT_READY to land. Do not re-profile vs encode. |
| I2 | Intra-CB encoder bubble | folded into I1 GPU | 964 encoders → 1 execute | HIGH mechanism; ns inside dirty GPU Δ |
| I3 | Per-token `waitUntilCompleted` | **≤ 425,929 ns HEAD / ≤ 561,994 ns after I1** PROJECTED | device token + multi-token replay + deferred EOS. Host waits once per generate (or every N). | MEDIUM. Unmeasured on Qwen3.8. TCB `commit_no_wait` is the substrate. |
| I4 | ICB's leftover 3 u32 host writes | ESTIMATED ≪ 91 µs | token stays on device; only `position` / `mha_seq_len` remain, and those are CPU-known | HIGH they are small |
| I5 | Occupancy-starved rope / mha / argmax / RMS | 2–4 ms PROJECTED | fold into neighboring multi-TG Q4 (fusion lane P3/P4a). Not a host round trip. | MEDIUM isolated; abort if gate GEMV GB/s drops |

I1 is the only **MEASURED_WIN** in this set and is **not on HEAD**.

### 4.3 Will not move TOKEN_NS to 10 ms

Ceremony + bubbles + sampler, deleted perfectly:

```
HEAD complete wall     38.217 ms
− I1 named fixed         0.660 ms   →  36.68 ms   MEASURED ICB wall 36.684
− I3 leftover wait       0.562 ms   →  ~36.1 ms   PROJECTED
− H1 already inside I1
− fusion tinies        2–4 ms       →  ~32–34 ms  PROJECTED (fusion lane)
```

Still ~3× the 10 ms rung. Weight addressing 21.3 ms is the existential
bucket. This lane does not reopen that.

**FALSIFIED:** deleting host round trips / idle gaps / the 4-byte read is
sufficient for 100 TPS on this Q4 genome.

**SUPPORTED:** the same deletes are real, ranked, and bounded.

---

## 5. Negative results

| id | mechanism | verdict | REOPEN_IF |
|---|---|---|---|
| K1 | Identity bookkeeping / many host round trips dominate Qwen3.8 | **KILLS** | a path reintroduces per-layer readback |
| K2 | 4-byte sample read is the sync | **KILLS** (877 ns). The wait is the embed `set_bytes` data dep | never as a ms lever |
| K3 | Fuse V write into v GEMV as a TOKEN_NS lever | **KILLS** | complete-token A/B shows otherwise |
| K4 | KV DRAM at seq≈19 is a traffic organ | **KILLS** (0.019% of active bytes; 6.6 µs floor vs 667 µs mha) | seq where KV bytes enter the 1%+ band **and** mha timestamps scale with them |
| K5 | Retile KV for coalescing at seated length | **KILLS** as G1 | long-context mha A/B |
| K6 | Host chaining / ICB / sample as the 10 ms path | **KILLS** | never on this Q4 genome |
| K7 | `concurrent_independent=true` as idle-gap delete | **KILLS** on prior token A/B | a new complete-token win |
| K8 | Linear 46 µs/ctx slope to 2k as a measured token | **not a finding**. PROJECTED, occupancy-starved window | isolated mha sweep |

---

## 6. Independent arithmetic (this lane)

Geometry constants only + receipt ns. No tensors loaded.

```
=== STATE BYTES (f32) ===
rec_one                 3,145,728
conv_one                  122,880
rec_48                150,994,944
conv_48                 5,898,240
deltanet_resident     156,893,184
kv_one (K or V, 1 layer, 1 pos)    4,096
gqa_write/token             131,072
gqa_read seq=19           2,490,368     # matches ledger
gqa_read seq=128         16,777,216
gqa_read seq=2048       268,435,456
workspace max_seq=128     175,361,796   (167.24 MiB)
workspace max_seq=2048    427,020,036   (407.24 MiB)

=== PRODUCTION GPU vs POSITION (ledger production_steps, 3 reps) ===
decode-only OLS  gpu = 33,047,779 + 45,874 * pos    R2=0.868
pos11 median 33,523,750
pos25 median 34,105,875
delta +582,125 ns over 14 context tokens

=== HOST (complete-wall mean) ===
encode     886,210   66.5% of wall−gpu
wait−gpu   425,929   32.0%
submit      10,457
tokenizer    6,306
epilogue     1,789
readback       877
rest           655
sum      1,332,224
```

---

## 7. Evidence index

| claim | pointer |
|---|---|
| 1 CB, 964 disp, 1 wait, 4 B read | `qwen38_hybrid_decode.rs:3292-3310`; `metal/mod.rs:4856-4873,5370-5371` |
| encode 67% of host | `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` `wall_minus_gpu_named` |
| intra-CB 337 µs | ledger `unattributed_residual`; production GPU − isolated sum |
| GQA layout `[seq][4][256]` f32 | `qwen38_device_activations.metal:170`; `kernels/mod.rs:10502` |
| GQA bytes / seq | `qwen38_token_ns_ledger.rs:118-136`; ledger `state_bytes` |
| workspace vs seq | `qwen38_hybrid_decode.rs:320-399,3878-3887` |
| write fused into rope, V is copy | `qwen38_device_activations.metal:162-191` |
| MHA Phase 1 uncoalesced / Phase 4 coalesced | `mha.metal:627-634,673-679` |
| seq 11–25 GPU slope | ledger `production_steps` OLS, this file §2.3 / §6 |
| 16 vs 32 walls do not show growth | authority `g015_length_confirmation` vs `headline_32_new_tokens` |
| device argmax 335 µs, readback 877 ns | ledger `isolated.argmax`; authority `sample_readback` |
| device-token embed exists, unused | `qwen_uniform_q4.metal:604-622`; grep empty in hybrid_decode |
| `commit_no_wait` exists, unused | `metal/mod.rs:5177-5181` vs `step` 3304 |
| ICB MEASURED, not on HEAD | `QWEN38_FIXED_OVERHEAD_DELETED.json`; `7400acf1b` not ancestor |
| ICB 3 u32s, wait rose | same receipt `mechanism.per_token_host_writes`, `wait_minus_gpu.delta` |
| max_seq default 128 | `ascension_qwen38_hybrid_greedy.rs:68` |
| prompt_len 11, seq≈19 | complete-wall `identity.prompt_len`; ledger `gqa_kv_read=2490368` |
| mha shmem cap 8064 | `kernels/mod.rs:10545-10546`; `mha.metal:686-693` |
| identity bookkeeping 24–80 ns | authority `state_update`, `bookkeeping` |

---

## 8. Unmeasured on purpose

Not run (GPU lock is serialized; organism holds the device).

1. Isolated `mha_decode_f32` at seq 19/128/512/2048. Closes whether the
   46 µs/ctx slope dies when Phase 1 fills. Cheapest long-context closer.
2. Device-token + `commit_no_wait` encode-ahead complete-token A/B.
   Predicted ~0.89 ms if EOS still waits; ~1.33 ms if it does not.
3. ICB rebase bit-id on current HEAD (coherence gate, 3 sealed prompts).
   Mechanism already measured; this is a merge check.
4. Production-fused argmax timestamp (is 335 µs real or isolated-CB tax).

---

```
STATUS
SUPPORTED

CLAIMS
C1 SUPPORTED. One waitUntilCompleted and one 4-byte contents() per token. Evidence: qwen38_hybrid_decode.rs:3292-3310; metal/mod.rs:5370-5371.
C2 SUPPORTED. GPU queue-idle inside a token is encode 0.886 ms (67% of wall−gpu) then wait−gpu 0.426 ms (32%). Submit/tokenizer/epilogue/readback/bookkeeping sum to <20 µs except tokenizer 6.3 µs. Evidence: QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json wall_minus_gpu_named.
C3 SUPPORTED. Intra-CB device bubble is ~337 µs from 964 encoder create/end. Evidence: TCB dispatch_threads metal/mod.rs:4856-4873; ledger unattributed_residual / production−isolated GPU.
C4 SUPPORTED. Identity bookkeeping is not the wall (24–80 ns). Prior-campaign dominance does not transfer. Evidence: authority state_update, bookkeeping; one CB not N waits.
C5 SUPPORTED. GQA KV is f32 [seq][4][256] per layer × 16, 131,072 B write/token, 131,072×S read/token. Rec/conv state is 156.9 MB resident and seq-invariant. Evidence: metal:170; geometry; ledger state_bytes; this file §2.2 / §6.
C6 SUPPORTED. At seq=19 KV is 0.019% of active bytes and a 6.6 µs bandwidth floor. Isolated mha is 667 µs. KV DRAM is not a seated-length traffic organ. Evidence: §2.2; ledger isolated.mha_16.
C7 MEASURED over seq 11–25: production GPU rose 582 µs, OLS 45,874 ns/ctx-token, R²=0.87. PROJECTED beyond that window (occupancy-starved). 16-vs-32 complete walls do not confirm it. Evidence: ledger production_steps; authority g015 vs headline.
C8 SUPPORTED. Phase-1 KV reads are uncoalesced (4 KiB stride). Phase-4 V reads are coalesced. Write is contiguous per head, 4 of 24 threads. Evidence: mha.metal:627-679; rope kernel:170-191.
C9 SUPPORTED. Cache write is already fused into qwen38_gqa_qk_norm_rope_cache_f32. Fusing V into the v GEMV is legal and tiny (<10 µs ESTIMATED). Evidence: metal:162-191; encode_gqa 2742-2778.
C10 SUPPORTED. Sampling is device argmax 335,499 ns isolated + 877 ns host readback. The 4-byte load does not cause the wait; the next embed set_bytes does. Evidence: sample.metal:48-75; encode_embed:2541; authority sample_readback.
C11 SUPPORTED. Device-token embed and commit_no_wait exist and are unused on Qwen3.8. Wiring them is HEAD-legal. Predicted recover ≈0.89 ms with per-token wait kept, ≈1.33 ms if wait is deferred. Evidence: qwen_uniform_q4.metal:604-622; metal/mod.rs:5177-5181; grep empty in hybrid_decode.
C12 MEASURED_WIN, not on HEAD. ICB encode 886→91 µs, named fixed 1.331→0.671 ms, wait−gpu +136 µs, wall 38.217→36.684 ms, ids bit-identical. Evidence: QWEN38_FIXED_OVERHEAD_DELETED.json; 7400acf1b not ancestor of HEAD.
C13 FALSIFIED as a path to TOKEN_NS≤10e6. Perfect deletion of every host gap + ICB + chaining leaves ~36 ms, or ~32–34 ms with fusion tinies. Evidence: §4.3; G024 weight_addressing 21.3 ms.

EVIDENCE
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:320-399,767-799,2010-2061,2522-2546,2711-2792,3292-3355,3878-3887
- crates/hawking-core/src/model/qwen38_geometry.rs:20-41,106-109,280-286
- crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:118-136,397-402,612-653
- crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs:12-54
- crates/hawking-core/src/metal/mod.rs:1232-1233,3572-3574,4856-4873,5177-5181,5330-5371
- crates/hawking-core/src/kernels/mod.rs:10497-10560,14232-14248
- crates/hawking-core/shaders/qwen38_device_activations.metal:108-193
- crates/hawking-core/shaders/mha.metal:602-681
- crates/hawking-core/shaders/sample.metal:48-75
- crates/hawking-core/shaders/qwen_uniform_q4.metal:604-622
- crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs:68,103
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json (state_bytes, isolated, production_steps, components, median_*)
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json (headline, wall_minus_gpu_named, g015)
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL.json (identity.prompt_len=11)
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json
- git merge-base --is-ancestor 7400acf1b HEAD → exit 1
- this-lane python: workspace bytes, KV table, production_steps OLS (§6)

CHANGES
created workspace/superwave/g1/g1-kv-and-host-gaps.md
no other path touched

TESTS
see executor final message for exact test -s / wc -l / git status --porcelain

RISKS
- production_steps OLS is 14 tokens × 3 dirty reps. Session GPU already moves milliseconds across receipts. Do not treat 46 µs/ctx as a 2k forecast.
- Isolated argmax 335 µs may be isolated-CB inflated (same caveat as lm-head lane). Does not change C10/C13.
- ICB wait−gpu rose; chaining after ICB is the leftover, not a second encode win.
- H2 serial-encoder on TCB is unmeasured and can lose bit-identity. Prefer I1.

UNRESOLVED
- mha vs seq at 128/512/2048 (GPU lane).
- Device-token + commit_no_wait complete-token A/B (GPU lane).
- ICB rebase bit-id on this HEAD.
- Production-fused argmax ns.
- Device EOS flag (unwritten).

NEXT
Serialized GPU lane: isolated mha seq sweep, then land ICB (do not re-discover), then device-token encode-ahead if wait−gpu remains after ICB. Do not retile KV or fuse V-write until the mha sweep says the organ is real.
```
