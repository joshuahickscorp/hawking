# G1 addressing topology — catalog vs single-address

Lane: `36-addressing-topology`. HEAD `0fbf2e2a5`. No GPU, no inference, no live-process touch.
Every number is **MEASURED** (receipt or on-disk census), **DERIVED** (geometry/source arithmetic),
or **PROJECTED** (named MEASURED inputs, stated model). A component microbenchmark is not a token.

---

## 0. Verdict

The 24 percent (530.65 vs 699.57 GB/s) is real as a **COMPONENT**. It is **not** a
per-tensor file layout tax and it is **not** 5 ms of TOKEN_NS.

| quantity | value | tag | pointer |
|---|---:|---|---|
| single-address addr | 699.5736545106142 GB/s / 19,457,084 ns | MEASURED COMPONENT | `HONEST_ROOF_WEIGHT_ADDRESSING.json` `verdict.single_gemv_at_13p6gb.addr_gb_s` |
| 401-organ catalog addr | 530.6544688491846 GB/s / 25,650,709 ns | MEASURED COMPONENT | same, `verdict.production_catalog_at_13p6gb.addr_gb_s` |
| catalog − single | 6,193,625 ns / 24.15% rate loss | DERIVED | 25,650,709 − 19,457,084 |
| sealed token addressing | 21,293,102.5 ns / 639.2522348492898 GB/s | CITED TOKEN | same, `sealed_ledger_cited_not_rerun.weight_addressing_ns`; `verdict.sealed_weight_addressing_gb_s` |
| sealed / single roof | 91.38% | DERIVED | 639.252 / 699.574 |
| recoverable TOKEN_NS if addressing reaches the single-address roof | **1,836,018.5 ns** | PROJECTED | 21,293,102.5 − 19,457,084 |
| 0.24 × addressing bucket | 5,110,344.6 ns | ARITHMETIC, not a prediction | contract slogan |

The catalog probe that produced 530.65 already used **one codes slab + one scales
slab + host offsets** (`honest_roof.rs` `time_q4_catalog`, L583–627). Packing the
755 files into two buffers, by itself, **does not close the gap**. It is the
topology that was measured at 530.65.

The gap is **401 encoder-bounded GEMV launches vs 1**, plus mixed organ sizes.
Device has **no** indirection table. Host offset math is a running `u64` add
(catalog probe) or the constant `0` (production). Per-dispatch `set_buffer` of a
new codes/scales pair is the production bind path; the catalog probe rebinds the
**same** two buffers at changing Metal offsets.

**Layout change that closes the COMPONENT gap:** load-time (or optional disk)
execution-order two-slab + **one multi-organ `geo_tpr64_tg128` grid**. Same inner
loop as the single-address kernel. A 401-row launch table is read once per
threadgroup, not per group. **No artifact repack is required.** Optional
page-aligned disk slabs buy no-copy load, not the 24 percent.

**Predicted (PROJECTED, remainder held at 13,934,814.5 ns):**

| addressing rate | addr ns | TOKEN_NS | seated TPS |
|---|---:|---:|---:|
| 639.25 sealed (today) | 21,293,102.5 | 35,227,917 | 28.3866 |
| 699.57 single-address roof | 19,457,084 | 33,391,898.5 | 29.9474 |
| 650 (conservative organ-switch) | 20,941,021 | 34,875,835.5 | 28.6732 |

100 TPS is not in this lever. Density still required (`g1-roof-falsification.md`).

**KILLS:** “repack the 755 hashed files into one catalog to recover 24 percent /
5 ms.” That layout is what `time_q4_catalog` already ran.
**REOPEN_IF:** a GPU-locked addr_probe of the **production** 401 GEMVs in
execution order, on the live 804 buffers, lands at ~530 GB/s rather than the
isolated-class 639. Then the 6.19 ms COMPONENT becomes a TOKEN claim. This lane
was forbidden to run that.

---

## 1. Three topologies that must not be conflated

### 1.1 Production G0 (what `step()` actually binds)

Source: `qwen38_hybrid_decode.rs` `Qwen38HybridWeights::load` L508–580,
`Q4Weight` L421–426, `encode_q4_matvec_kernel` L1569–1591,
`TokenCommandBuffer::new` defaults `ordered_encoder_enabled=false`
(`metal/mod.rs` L2958–2960), default `dispatch_threads` L3353–3367.

- 755 catalog rows (`QWEN38_EXPECTED_CATALOG_TENSORS` = 402 + 353,
  `qwen38_pack.rs` L31–34). Manifest MEASURED 755 / 402 / 353.
- Each Q4 tensor → **two** `new_buffer_with_bytes_checked` (codes, scales).
- Each f32 tensor → **one** buffer.
- Weight allocations: **804 + 353 = 1,157**. DERIVED from load match arms.
- Workspace: **34** `PinnedBuffer`s (`Qwen38HybridWorkspace` L712–747).
- Session attach total: **1,191** MTLBuffers, plus pipelines.
- Every GEMV: `set_buffer(0, codes, 0)`, `set_buffer(1, scales, 0)`,
  `set_buffer(2, input, 0)`, `set_buffer(3, output, 0)`, then
  `set_bytes` of `rows`, `cols`, `groups_per_row`. Offset is always **0**.
- One new compute encoder per dispatch. 964 dispatches / token
  (`qwen38_64_layer_execution_schedule.rs` L12–54; ledger `dispatches.total=964`).
- 401 of those are `geo_tpr64_tg128` GEMVs (embed is a gather, not a GEMV).

### 1.2 Catalog probe (the 530.65 number)

Source: `honest_roof.rs` `Slab` L493–550, `time_q4_catalog` L583–627,
`measure_production_catalog` L805–829, `production_gemv_shapes` L266–286.

- One codes slab + one scales slab, sized for the full 13,611,663,360 B payload.
- 401 shapes: 192 MLP + 144 DN + 64 GQA + 1 lm_head, **class-grouped**, not
  execution order (all 64 layers of gate/up/down, then all 48 DN, then all 16
  GQA, then lm_head).
- Host running `c_off` / `s_off`. `set_buffer(..., offset)`.
- `dispatch_batch_timed` also starts with `ordered_encoder_enabled=false`
  (`metal/mod.rs` L3010–3011) → **401 encoders** in one CB.
- Synthetic unique bytes. Same kernel and launch as decode.
  Receipt note: “Not a model-quality run.”

MEASURED (`HONEST_ROOF_WEIGHT_ADDRESSING.json`
`q4_production_catalog_addr_probe`):

```
dispatches: 401
payload_bytes: 13611663360
topology: production_shape_catalog
spread.all_ns: [26727750, 25415834, 25252084, 25670959, 25650709]
spread.median_ns: 25650709
spread.median_gb_s: 530.6544688491846
```

Full kernel same topology: median 26,910,625 ns / 505.8100047843556 GB/s
(`q4_production_catalog_full`).

### 1.3 Single-address probe (the 699.57 number)

Source: `time_q4_single` L553–581; `rows_for_payload` produces **5,004,288**
rows × 5120 cols (`honest_roof.rs` test L1181). One dispatch, offsets 0, one
encoder. Same two slabs.

MEASURED (`q4_single_gemv_addr_probe` label `gemv_payload_13p612gb`):

```
dispatches: 1
rows: 5004288
cols: 5120
payload_bytes: 13611663360
spread.all_ns: [19637375, 19536625, 19347166, 19345541, 19457084]
spread.median_ns: 19457084
spread.median_gb_s: 699.5736545106142
```

Full: 20,417,041 ns / 666.6814921907636 GB/s.
Decode-probe: 19,906,000 ns / 683.7970139656385 GB/s.
ALU+decode tax vs addr: 4.70% (`verdict.single_gemv_at_13p6gb.alu_plus_decode_tax_vs_addr`).

This is a **bandwidth probe**, not a legal model GEMV. Inputs, outputs, and
shapes are not concatenable into one 5,004,288 × 5120 multiply.

---

## 2. On-disk catalog: how the 755 tensors are laid out

Live artifact census, this lane, **stat + manifest only** (no 14 GB slurp):

`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`

| field | value | tag |
|---|---|---|
| `manifest.json` bytes | 238,879 | MEASURED `stat` |
| `schema` | `hawking.ascent.qwen38_language_uniform_q4.v1` | MEASURED |
| `tensor_count` / q4 / f32 | 755 / 402 / 353 | MEASURED |
| `tensor_payload_bytes` | 14,297,694,680 | MEASURED |
| `complete_physical_bpw` | 4.252735126866492 | MEASURED |
| `fused_in_proj_layers` | 48 | MEASURED |
| `skipped_vision_tensors` | 333 | MEASURED |
| files under `tensors/` | 1,108 | MEASURED |
| listed files present, size match | 755 / 0 mismatch / stat sum 14,297,694,680 | MEASURED |
| extra `.f32bin` | 353 / 10,584,840 B | MEASURED (wave-1 traffic also) |

Pack write order is **execution-ish** (`pack_qwen38_language_uniform_q4`
L474–664): embed, then per layer (norms, mixer GEMVs + mixer f32, gate, up,
down), then final norm, lm_head. Manifest row order matches that.

Filenames are **not** sequential. `artifact_filename` is SHA-256 of the tensor
name (`qwen38_pack.rs` L270–274). This lane: **0 / 755** hash mismatches.
Lexicographic distance between execution-consecutive Q4 artifacts:
min 1, median 117, max 367. Only **2 / 401** consecutive Q4 pairs are
lex-adjacent.

Inodes (APFS) are monotonic in pack order (401 / 401 diffs > 0) but only
**36 / 401** consecutive Q4 pairs have adjacent inodes. f32 files are written
between GEMVs, so even pack-time writes do not produce a contiguous Q4 stream.

### 2.1 One Q4 file

HQ30UQ4, MEASURED on
`tensors/6ef2a12f6e0acacc…665446.hq30uq4`
(`language_model.model.layers.0.linear_attn.in_proj_ba.weight`, 261,160 B):

```
magic     HQ30UQ4\0
version   1
group     64
rank      2
elements  491520
dim       [96, 5120]
```

Header parser (`uniform_q4.rs` L41–151; `COMPLETE_BINARY_HEADER_BYTES=32`):

```
[32 B prefix][rank × u32 dims][f16 scales, groups×2][u8 codes, groups×32]
```

Rank-2 header = 40 B. Then scales, then codes. `scale_offset = 40`,
`sign_offset = 40 + scale_bytes`. ba: 96×80 = 7,680 groups → 15,360 B scales +
245,760 B codes + 40 = 261,160. Matches file size.

Load **strips the header** and splits scales vs codes into two Shared buffers
(`load` L539–557). After load, the on-disk adjacency of scales-then-codes is
destroyed.

`new_buffer_with_bytes_checked` is `newBufferWithBytes` / copy
(`metal/mod.rs` L2610–2628). No-copy exists (`new_buffer_no_copy_checked`,
16 KiB align, L2677–2682) and is unused on this path.

### 2.2 Organ byte table (DERIVED, matches `GEMV_PAYLOAD_BYTES`)

`q4_bytes = rows × ceil(cols/64) × 34`. Header not included.

| organ | n | rows × cols | TGs (ceil(rows/2)) | TGs/core | payload | n × payload |
|---|---:|---|---:|---:|---:|---:|
| embed (not GEMV stream) | 1 | 248320 × 5120 | 124160 | 2069 | 675,430,400 | 675,430,400 |
| mlp gate | 64 | 17408 × 5120 | 8704 | 145.07 | 47,349,760 | 3,030,384,640 |
| mlp up | 64 | 17408 × 5120 | 8704 | 145.07 | 47,349,760 | 3,030,384,640 |
| mlp down | 64 | 5120 × 17408 | 2560 | 42.67 | 47,349,760 | 3,030,384,640 |
| dn qkvz | 48 | 16384 × 5120 | 8192 | 136.53 | 44,564,480 | 2,139,095,040 |
| dn ba | 48 | 96 × 5120 | **48** | **0.80** | 261,120 | 12,533,760 |
| dn out | 48 | 5120 × 6144 | 2560 | 42.67 | 16,711,680 | 802,160,640 |
| gqa q | 16 | 12288 × 5120 | 6144 | 102.40 | 33,423,360 | 534,773,760 |
| gqa k | 16 | 1024 × 5120 | 512 | 8.53 | 2,785,280 | 44,564,480 |
| gqa v | 16 | 1024 × 5120 | 512 | 8.53 | 2,785,280 | 44,564,480 |
| gqa o | 16 | 5120 × 6144 | 2560 | 42.67 | 16,711,680 | 267,386,880 |
| lm_head | 1 | 248320 × 5120 | 124160 | 2069 | 675,430,400 | 675,430,400 |
| **GEMV sum** | **401** | | | | | **13,611,663,360** |

`backend_honest_roof_production_catalog_payload_matches_geometry` asserts
`192+144+64+1` shapes and this sum (`honest_roof.rs` L1172–1176).

ba is the only organ that does not fill 60 cores (48 TGs). k/v fill the GPU
but are thin (8.5 TGs/core). Everything else is fat.

---

## 3. What a token touches

### 3.1 Distinct allocations

| class | allocations | when created | per token? |
|---|---:|---|---|
| Q4 codes | 402 | load | rebound, not realloc |
| Q4 scales | 402 | load | rebound |
| f32 mixer/norm | 353 | load | rebound |
| workspace | 34 | attach | rebound; 3 dead on fused path (`hgravs_mid`, `split_*`) |
| **weight + workspace** | **1,191** | | |

f32 census from pack: DN layer 6 (input ln, post ln, conv1d, A_log, dt_bias,
norm) × 48 + GQA layer 4 (input ln, post ln, q_norm, k_norm) × 16 + final
norm = 288 + 64 + 1 = **353**. All are read every token (wave-1 traffic).

### 3.2 Buffer bindings per token (SOURCE)

401 GEMVs × 4 `set_buffer` = **1,604**. Of those, **802** are weight binds
that name a **different** codes/scales pair every launch. 401 × 3 `set_bytes`
= **1,203** geometry constants (`rows`/`cols`/`gpr`), recomputed on the host
as `cols.div_ceil(64)` (`encode_q4_matvec_kernel` L1578).

Embed gather: 2 weight binds + hidden, token via `set_bytes` (L2533–2545).
Does not stream the 675,430,400 B table; indexes one row (2,720 B). DERIVED.

HashMap name lookup every encode (`q4()` L1182–1187). Host ceremony; already
inside the measured 919,250 ns encode (`TOKEN_NS` `host_preparation`).

Default encoder path creates **964** encoders (`dispatch_threads` L3353–3367).
Intra-CB encoder-transition gap MEASURED 336,926 ns / 964 = 349.5 ns
(`g1-token-anatomy.md` §4). ×401 GEMVs = 140,150 ns. That is **not** the
6.19 ms catalog gap.

### 3.3 Execution-order GEMV sequence (SOURCE, `encode_deltanet` / `encode_gqa` / `encode_dense_mlp`)

Layer `ℓ`, mixer `(ℓ+1)%4==0` → GQA else DeltaNet:

```
DN:  qkvz, ba,  [rearrange, decay, gated_delta, gated_rmsnorm], out,
     [post-ln], gate, up, [silu], down
GQA: q, k, v,   [rope, mha, sigmoid], o,
     [post-ln], gate, up, [silu], down
then lm_head
```

401 GEMVs. Consecutive same `rows×cols`: **80 / 400** (64× gate+up, 16× k+v).
Those 80 pairs share input (`normalized`) and are the only free same-shape
fuses. qkvz+ba share input and cols but not rows. down / out / o / lm_head
have different inputs and cannot stack onto the previous GEMV.

**Are consecutive GEMVs adjacent in memory?**

| place | adjacent? | evidence |
|---|---|---|
| on disk, filename | no | hashed; 2/401 lex-adjacent |
| on disk, inode | almost never | 36/401 inode-adjacent |
| on disk, one file | scales then codes, yes | HQ30UQ4 layout |
| after load, one tensor | **no** | split into 2 buffers |
| after load, consecutive GEMVs | **no** | 804 independent `newBufferWithBytes` |
| catalog probe | codes adjacent, scales adjacent | one slab each |
| single-address | one contiguous 13.6 GB stream | one GEMV |

---

## 4. Addressing indirection per GEMV

### 4.1 Device: affine, no table

`qwen_uniform_q4.metal` L183–221 (production) and L227–266 (`addr_probe`):

```
rgb  = row * groups_per_row + (col / 64)
scale = scales[rgb]                          // half
packed = *(uint*)(codes + rgb * 32 + (local >> 1))
```

Two independent affine maps from `(row, col)` into two buffers. **No** device
lookup table, **no** per-tensor header walk, **no** extra DRAM word per group
beyond the 32+2 payload.

`groups_per_row` is a host `set_bytes` constant, not loaded from a table.
For every production organ, `cols % 64 == 0`, so `div_ceil` is exact.

Per thread, cols=5120, stride 512, 8-wide: **10** scale loads + **10** packed
loads per row. That is the payload. addr_probe sinks those loads and skips
the input vector (`(void)input` L265).

### 4.2 Host: rebind every dispatch; offset only in the catalog probe

| path | codes/scales bind | offset | geometry |
|---|---|---|---|
| production `encode_q4_matvec_kernel` | distinct pair per tensor | **0** | `rows`, `cols`, `div_ceil(cols,64)` |
| catalog `time_q4_catalog` | same two slabs | running `c_off`/`s_off` | same three u32s |
| single `time_q4_single` | same two slabs | **0** | one `(5004288, 5120, 80)` |

Host offset cost on the catalog probe is two `u64` adds per organ (L619–620).
Not a millisecond-class term.

Per-dispatch **buffer rebinding: YES** on both production and catalog.
Production rebinds **object identity** (new `MTLBuffer*`). Catalog rebinds
**offset** into one object. Single-address rebinds once.

### 4.3 Checked specifically (objective)

| question | answer | evidence |
|---|---|---|
| per-dispatch buffer rebinding? | **YES** | `encode_q4_matvec_kernel` L1583–1584; catalog L611–612; 401× |
| per-tensor offset on the host? | production **NO** (always 0); catalog probe **YES** (prefix sum of `code_bytes`/`scale_bytes`) | L1583 vs L619–620 |
| device indirection table? | **NO** | metal L208–209; buffers are raw `device const uchar*` / `half*` |

---

## 5. Where the 24 percent comes from

Catalog and single-address share: kernel, launch geo, two Shared slabs, unique
synthetic bytes, completed-CB GPU timestamps, dirty box
(`timing_label=GPU_PROTECTED_CPU_CONTENDED`, `clean_box=false`).

They differ by:

1. **401 encoders / 401 grids vs 1.** Default `dispatch_threads` ends the
   encoder after every launch. Occupancy drains. GPU issue restarts.
2. **Mixed organ sizes** vs one 5,004,288 × 5120 rectangle. ba = 48 TGs.
   k/v = 512 TGs. down/out = 2,560 TGs. gate = 8,704 TGs.
3. **Class-grouped order**, not execution order. Not the production interleave
   with RMSNorm / silu / DeltaNet.
4. **Same input/output buffers** reused 401 times in the probe (helps the
   catalog, does not explain a loss).

Same-shape tiling, 287 × 17408×5120 gates, 13,589,381,120 B
(`q4_tiled_production_organ` label `13p612gb_tiled_gate_addr`):

```
median_ns: 22988750
median_gb_s: 591.1317979446468
dispatches: 287
```

Constant-per-dispatch tax model (DERIVED, COMPONENT):

| topology | ns | bytes | implied ns @ 699.57 | residual | residual / N |
|---|---:|---:|---:|---:|---:|
| single | 19,457,084 | 13,611,663,360 | 19,457,084 | 0 | — |
| tiled 287 fat | 22,988,750 | 13,589,381,120 | 19,425,167 | 3,563,583 | **12,417 ns** |
| mixed 401 | 25,650,709 | 13,611,663,360 | 19,457,084 | 6,193,625 | **15,445 ns** |

≈ 12 µs/dispatch for fat same-shape. Extra ≈ 3 µs/dispatch for the mix
(tinies + col-width changes). 12.4 µs × 401 ≈ 5.0 ms of the 6.19 ms.
Mix ≈ 1.2 ms. Host encode (953 ns/dispatch) and encoder-create (350 ns) are
an order of magnitude smaller.

Isolated **class** CBs (the sealed TOKEN split) sit between tiled and single,
not at catalog 530:

| class | isolated full ns | addr frac | addr ns | bytes | addr GB/s | tax vs 699.57 |
|---|---:|---:|---:|---:|---:|---:|
| mlp 192 | 15,853,666 | 0.871692 | 13,819,500 | 9,091,153,920 | 658 | 4.3 µs/disp |
| dn 144 | 5,560,749 | 0.905087 | 5,032,900 | 2,953,789,440 | 587 | 5.6 µs/disp |
| gqa 64 | 1,817,416 | 0.830265 | 1,508,900 | 891,289,600 | 591 | 3.7 µs/disp |
| lm_head 1 | 1,017,458 | 0.915707 | 931,700 | 675,430,400 | 725 | beats roof |
| **sum** | | | **21,293,000** | 13,611,663,360 | **639** | |

Fractions and isolated ns: `QWEN38_TOKEN_NS_LEDGER.json` `probes` / `isolated`
(cited in `g1-token-anatomy.md` §3.1–3.2). Isolated exclusive GPU 33,575,407
vs production GPU 33,912,333; scale = 1.0. Production is **not** 4 ms slower
than isolated families, so production GEMVs do **not** pay the mixed-catalog
15 µs tax.

`adjudication.catalog_topology_tax` (receipt, L as printed):

> 401 production-shaped GEMVs in one CB achieve 530.7 GB/s addr / 505.8 GB/s
> full — 24% below the single-GEMV addr roof. Isolated class CBs (the ledger)
> sit between the two, at 639.

That sentence is the whole topology result.

### 5.1 What does **not** cause the 24 percent

| hypothesis | verdict | why |
|---|---|---|
| 755 hashed files / non-adjacent extents | **KILLS** as the 24% | catalog probe never opened those files |
| 804 distinct MTLBuffers | **KILLS** as the 24% | catalog probe used 2 and still lost 24% |
| host offset arithmetic | **KILLS** | two u64 adds |
| device indirection table | **KILLS** | there isn't one |
| codes/scales as two streams | **KILLS** as the 24% | single-address is also two streams and hits 699.57 |
| reconstruction ALU | **KILLS** | 4.7% of single-GEMV; `QWEN38_RECONSTRUCTION_IS_FREE` |
| 964 encoder creates (host) | **KILLS** as the 6 ms | 0.14 ms of GEMV encoder-gap; encode is 0.92 ms ceremony |

### 5.2 The “5 ms of the token” slogan

`0.24 × 21,293,103 = 5.11 ms` treats the sealed bucket as if it were at
catalog 530. It is at 639. The catalog→single delta is 6.19 ms **on the
probe that ran at 530**. The token identity already closed at isolated-class
rates. Maximum honest TOKEN recovery at the measured single-address roof:

```
21,293,102.5 − 19,457,084 = 1,836,018.5 ns
TOKEN_NS 35,227,917 → 33,391,898.5
TPS      28.3866    → 29.9474
```

Holding the complete-wall remainder 16,923,690 ns (`g1-roof-falsification.md`
§4 table, 38,216,792 − 21,293,102): 36.38 ms / 27.49 TPS. Same arithmetic
the roof lane already published for “addressing at 699.57”.

---

## 6. Concrete layout change

### 6.1 What to build (closes the COMPONENT 24%)

Name: **execution-order two-slab + one multi-organ `geo_tpr64_tg128` grid**.

1. **Codes slab** 12,810,977,280 B and **scales slab** 800,686,080 B, filled
   in **execution order** (the sequence in §3.3), embed **excluded**.
2. **Launch table**, 401 rows, ~48 B each ≈ 19 KiB, resident:
   `{row0, rows, cols, gpr, in_sel, out_sel, code_off, scale_off, out_off}`.
3. **One dispatch.** Grid = `sum_i ceil(rows_i / 2) × 128` threads, TG=128.
   DERIVED total GEMV rows = 4,152,320; TGs = 2,076,160; 34,603 TGs/core.
   Occupancy matches the single-address probe class (millions of TGs).
4. Each TG binary-searches the 401-row table (≤9 compares, 19 KiB, not DRAM)
   then runs the **existing** inner loop (`rgb`, scale, packed). Different
   `cols` already parameterized. Pad so a TG never straddles an organ.
5. Inputs: six small f32 vectors (`normalized`, `act`, `gated`, `gated_attn`,
   hidden, logits dest). Bind as an array or one activation slab with
   `in_sel`. 20–993 KB, not the 13.6 GB problem.
6. Keep embed as the gather kernel. Keep mixer f32 as 353 small buffers or
   one f32 slab — out of `weight_addressing`.

This is **not** the killed 1-TG megakernel (`g1-fusion-persistent.md` §1.1:
f16 expand + one 256-thread group looping layers, 4.4× slower). Multi-TG,
in-register Q4, same inner loop as `geo_tpr64_tg128`. The
`REOPEN_IF` on that kill is exactly this shape.

### 6.2 Achievable without a repack: **YES**

Load already `fs::read`s every file and copies into Metal. Replace the 804
`new_buffer_with_bytes_checked` calls with:

- pre-size two slabs from the manifest (sizes known: 12.811 + 0.801 GB)
- memcpy codes/scales to a running offset
- store `{offset, rows, cols}` per name
- drop the file `Vec`

Peak RAM: same class as today (one tensor transient + growing resident set).
Do **not** keep the 804 buffers and the slabs (that would double 13.6 GB).
Peak extra vs today: ≈ 0 if slabs replace per-tensor buffers.

Host encode of 401 GEMVs becomes encode of **1** GEMV-class dispatch plus
the other 563 kernels (still 564 encoders unless ICB/ordered_encoder lands
separately). Addressing bandwidth does not wait on that.

Disk 755-file pack can stay. Capability bits unchanged. No codec change.

### 6.3 Optional disk pack (not required for the 24%)

Page-aligned `weights.codes`, `weights.scales`, `launch.table` (16 KiB)
lets `new_buffer_no_copy_checked` bind. Deletes the load-time double copy
of 14.3 GB (wave-1 traffic §6.2). Bring-up / child-spawn only. TOKEN_NS
unchanged once resident.

Migration cost if you **do** repack: one sequential rewrite of 13.6 GB GEMV
payload + 10.6 MB f32. All 755 hashed files become 2–3 slabs. Old artifacts
keep working via the no-repack load path. **Do not** force a fleet rewrite
to chase this lever.

### 6.4 Stepping stone that does **not** close 24%

Row-stack same-input GEMVs with the **existing** kernel (no new shader):

| fuse | new shape | dispatches saved | new kernel? |
|---|---|---:|---|
| gate+up | 34816 × 5120 | 64 | no |
| qkvz+ba | 16480 × 5120 | 48 | no |
| q+k+v | 14336 × 5120 | 32 | no |
| **sum** | | **144** → 257 left | |

ba occupancy hole disappears (absorbed into qkvz). Isolated-class tax
4–6 µs × 144 ≈ 0.6–0.9 ms COMPONENT. TOKEN PROJECTED **0.2–0.4 ms**
(isolated MLP already only 4.3 µs/disp). Useful hygiene, not the 24%.

`ordered_encoder` (one encoder, 401 still-separate grids): saves the 350 ns
encoder-create, not the 12–15 µs GPU drain. PROJECTED ≪ 0.2 ms.
ICB: encode 919 → 91 µs (`QWEN38_FIXED_OVERHEAD_DELETED`, later genome, not
HEAD `step()`). Does not change 401 GPU launches.

### 6.5 Predicted bandwidth and token time

Assumptions, stated:

- Remainder of seated TOKEN_NS stays 13,934,814.5 ns (deltanet + gqa +
  decode + swiglu + ceremony + …). ICB/density not in this lever.
- One-dispatch multi-organ addr_probe matches single-address if the inner
  loop and unique-once stream match. Organ-switch + mixed `cols` is the
  only new term; 19 KiB table is not DRAM.
- Conservative 650 GB/s if that term is first-order. Optimistic = 699.57.

| case | tag | GB/s | addr ns | TOKEN_NS | TPS |
|---|---|---:|---:|---:|---:|
| G0 seated | CITED | 639.25 | 21,293,103 | 35,227,917 | 28.3866 |
| two-slab, still 401 encoders | PROJECTED = catalog COMPONENT | 530.65 | 25,650,709 | **39,585,524** | **25.26** |
| fuse-only, 257 launches | PROJECTED | ~590–640 | 21.3–23.0e6 | ~35.2–37.0e6 | ~27–28.4 |
| multi-organ 1-dispatch, conservative | PROJECTED | 650 | 20,941,021 | 34,875,836 | 28.67 |
| multi-organ 1-dispatch, roof | PROJECTED | 699.57 | 19,457,084 | 33,391,899 | 29.95 |

The two-slab-only row is a **regression** if someone “packs then fires 401
back-to-back” like the catalog probe. Isolated-class 639 is the better
401-launch regime. Do not replace production’s per-class-ish interleave
with a packed 401-GEMV blast and call it a win.

Complete-wall (38,216,792) at the roof: 36.38 ms / 27.49 TPS
(`g1-roof-falsification.md` §4). Still dirty-engineering, still not 100 TPS.

---

## 7. Migration

| item | no-repack (required path) | optional disk slabs |
|---|---|---|
| artifact | keep 755 hashed files | rewrite 13.6 GB once |
| loader | concat into 2 MTLBuffers | mmap + no-copy |
| `Q4Weight` | become `{rows, cols, code_off, scale_off}` views | same |
| kernel | new `geo_tpr64_tg128_catalog` (or retarget production) | same |
| capability | unchanged; same bytes, same decode | unchanged |
| load wall | similar (still copy) | lower (no 14.3 GB memcpy) |
| TOKEN_NS | the 1.84 ms if the 1-dispatch lands | same |

Fail-closed: if slab length ≠ 13,611,663,360, refuse attach. If table
`n != 401` or prefix-sum of payloads ≠ slab length, refuse. If any
`cols % 64 != 0`, refuse (today all divide).

---

## 8. What this lane did not do

- No Metal, no generate, no lock, no touch of the resident Genesis process.
- Did not re-derive BPW, TOKEN_NS, or the 530.65 / 699.57 / 639.25 figures.
- Did not slurp 14 GB; manifest + `stat` + 40-byte header peek of one ba file.
- Peak RSS of this analysis: manifest 239 KB + directory entries. Under 20 GB.

---

## 9. Required command output

```
$ test -s workspace/superwave/g1/g1-addressing-topology.md && echo OK
OK
```

```
$ wc -l workspace/superwave/g1/g1-addressing-topology.md
     638 workspace/superwave/g1/g1-addressing-topology.md
```

```
$ git status --porcelain
?? workspace/superwave/g1/g1-addressing-topology.md
```

---

## Completion report

STATUS: IMPLEMENT_READY

CLAIMS:
- Catalog topology that lost 24% is 401 encoder-bounded GEMVs on **two already-packed slabs**, not 755 files. Evidence: `honest_roof.rs` `time_q4_catalog` L583–627; `HONEST_ROOF_WEIGHT_ADDRESSING.json` `q4_production_catalog_addr_probe.topology=production_shape_catalog` `median_gb_s=530.6544688491846`.
- Single-address is 1 dispatch, 5,004,288 × 5120, 699.5736545106142 GB/s / 19,457,084 ns. Evidence: same receipt `q4_single_gemv_addr_probe` label `gemv_payload_13p612gb`; `honest_roof.rs` L1181.
- Production G0 is 1,157 weight buffers (402×2 + 353) + 34 workspace; 401 GEMVs rebind a distinct codes/scales pair at offset 0; 964 encoders. Evidence: `qwen38_hybrid_decode.rs` L421–426, L551–557, L1569–1591; `metal/mod.rs` L3353–3367.
- Consecutive execution GEMVs are not adjacent after load (804 allocations) and almost never adjacent on disk (2/401 lex, 36/401 inode). Evidence: this lane census of `uniform-q4-v1`; `artifact_filename` L270–274.
- Device addressing is affine `rgb = row*gpr + col/64` into two buffers. No device table. Evidence: `qwen_uniform_q4.metal` L202–209.
- Per-dispatch rebinding YES; production host offset always 0; catalog probe host offset is a prefix sum; no device indirection table. Evidence: §4.3.
- Sealed token addressing is 639.25 GB/s (91.4% of 699.57), not 530.65. Max TOKEN recovery at the single-address roof is 1.836 ms, not 5 ms. Evidence: receipt `verdict.sealed_weight_addressing_gb_s`; `adjudication.catalog_topology_tax`; arithmetic §5.2.
- Two-slab pack without dispatch collapse does not close the gap and can regress TOKEN if it becomes a 401-blast. Evidence: catalog probe **is** that pack (530.65); isolated-class 639 is the better 401-launch regime.
- Closing change: execution-order two-slab + one multi-organ geo_tpr64 grid. No repack required. Predicted TOKEN_NS 33.39 ms / 29.95 TPS at the roof, remainder held. Evidence: §6.

EVIDENCE:
- Receipt: `receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json` (git `HEAD`, not sparse-checked-out). Fields quoted in §0, §1.2, §1.3, §5.
- Live artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/{manifest.json,tensors}`. Census §2.
- Source: `crates/hawking-core/src/backend/honest_roof.rs`, `crates/hawking-core/src/model/qwen38_{hybrid_decode,pack,geometry,64_layer_execution_schedule}.rs`, `crates/hawking-core/src/metal/mod.rs`, `crates/hawking-core/shaders/qwen_uniform_q4.metal`, `crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs`.
- Wave-1: `g1-traffic-anatomy.md`, `g1-token-anatomy.md`, `g1-roof-falsification.md`, `g1-residency-reuse.md`, `g1-fusion-persistent.md`.

CHANGES:
- created `workspace/superwave/g1/g1-addressing-topology.md` only

TESTS:
- `test -s workspace/superwave/g1/g1-addressing-topology.md`
- `wc -l workspace/superwave/g1/g1-addressing-topology.md`
- `git status --porcelain`

RISKS:
- HONEST_ROOF is `GPU_PROTECTED_CPU_CONTENDED`; absolute 699.57 is provisional (`contamination_note` on the receipt). Relative catalog vs single on the same run is the 24%.
- One-dispatch multi-organ is PROJECTED from single-address, not measured. Mixed `cols` (5120/6144/17408) is the residual uncertainty (conservative 650).
- Megakernel kill is 1-TG + f16 expand. A careless “persistent kernel” that loops organs inside one TG would re-hit that kill.

UNRESOLVED:
- Production-token GEMV-only addr_probe (804 buffers, execution order, 401 encoders). Cheapest close of the REOPEN_IF. GPU lane owns it.
- Whether a clean box still prints 699.57.
- ICB on HEAD `step()` (ceremony, not this lever).

NEXT:
- GPU-authority lane: one locked multi-organ addr_probe vs `time_q4_single` vs production 401.
- Genome: implement §6.1 behind a fail-closed attach; do not ship two-slab+401-blast.
- Do not spend a fleet repack on this lever.
