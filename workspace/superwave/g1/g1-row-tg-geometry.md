# G1 row / threadgroup geometry — mixed-q3mlp-v1

Lane 113. No GPU. No generate. No edit of `q80_mixed_decode.metal` or
`qwen38_hybrid_decode.rs`. One new file.

Every number is tagged. A launch-derived TG count is not a TOKEN_NS.
The circulating ~31 TPS figure is a **PROJECTED** complete-token number
and is not a wall. PACK_REPORT itself projects 30.618 TPS from a scaled
G0 wall (`mixed-q3mlp-v1/PACK_REPORT.json` `projection.projected_tps`).
`11.47 GB / 639.25 GB/s = 17.95 ms` is an addressing floor, not a token.

---

## 0. Verdict

Ship one kernel: **TPR=64, RPT=2, TG=128, W=8** — G0
`geo_tpr64_tg128` launch. HGRAVU01 body already feeds it (bits 3 = 24 B/group,
bound 3; bits 4 = Q4 nibbles, bound 7). No repack.

Do **not** keep the incumbent Q80 occupancy tile (TPR=32, RPT=8, TG=256).
TG=256 is already on the bits-3 path and is the slow path. Large TG is not
the missing degree of freedom. TPR and extract width are.

Per-shape exception: `in_proj_a` / `in_proj_b` (48×5120) underfill at every
legal geometry in the requested set. Best tight fill is TPR=128 RPT=1 TG=128
(0.80 TG/core vs 0.40). Those 96 GEMVs are 0.11 % of streamed GEMV bytes.
A kernel that is right on MLP + lm_head and poor on a/b cannot lose the
token. The inverse — right on a/b, TPR=32 W=1 on lm_head — can.

`lm_head` (248320×5120) is the largest M. Any RPT that fills MLP also fills
lm_head. "Optimal on MLP, poor on lm_head" is not a live risk on this mix.

---

## 1. Labels

| tag | meaning |
|---|---|
| MEASURED | integer/float from this process (catalog parse, header peek, sha) |
| SOURCE | file:line in this tree |
| DERIVED | exact arithmetic on MEASURED / SOURCE |
| RECEIPT | quoted prior receipt, not re-run |
| PROJECTED | not a finding; not a token wall |

Occupancy = `ceil(M / RPT) / 60`. SOURCE
`crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:174-186`.
Not a hardware counter. 60 is GPU launch-math cores on this box, not the
contract's 28 CPU cores.

```174:186:crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
pub fn geo_tpr64_occupancy(rows: u64) -> OccupancyNote {
    let tg = 128u64;
    let threadgroups = rows.div_ceil(2);
    let threads = threadgroups.saturating_mul(tg);
    OccupancyNote {
        kernel: "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        rows,
        threadgroups,
        threads,
        gpu_cores: 60,
        threadgroups_per_core_if_spread: threadgroups as f64 / 60.0,
        note: "Not a hardware occupancy counter. Derived from launch geometry vs 60 M3 Ultra cores. A 17408-row gate launches 8704 TGs; the kernel is bandwidth-saturated, not occupancy-starved.",
    }
}
```

G0 manifest sha MEASURED this process, unchanged:

```
d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

`shasum -a 256 .../uniform-q4-v1/manifest.json`

---

## 2. Catalog — every GEMV shape

Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`
Parser: `/tmp/g1_row_tg_geometry.py` (this process, HQ38M20 layout from
`qwen38_hybrid_decode.rs:411-488`).

```
xxd -l 32 catalog.hq38m20
00000000: 4851 3338 4d32 3000 0100 0000 5303 0000  HQ38M20.....S...
00000010: 0201 0000 0000 0000 1fb5 0000 0000 0000
```

MEASURED header: magic `HQ38M20\0`, version 1, `n_tensors=0x0353=851`,
`n_segments=0x0102=258`, `name_blob=0xb51f=46367`, file 180124 B.
Matches PACK_REPORT `tensor_count=851` / `catalog_bytes=180124`.

Fused `in_proj_qkvz` / `in_proj_ba`: **0**. Split names only.

Codec 3 on all 851 rows. Header peek of every non-`other` tensor:
HEADER_FAIL=0. Bits census on 498 named weight tensors: bits=3 → 192,
bits=4 → 306 (305 GEMV + 1 embed).

353 `other` (norms, A_log, dt_bias, conv1d, q/k_norm): max 40960 elems,
all ≤ 65536. Load-time HGRAVU01 vector dequant. Not GEMV.
SOURCE `qwen38_hybrid_decode.rs:517-520`.

### 2.1 GEMV classes (MEASURED)

| class | n | M × K | bits | bound | group | nbytes each | Wbytes/row | body BPW | physical BPW |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mlp.gate | 64 | 17408 × 5120 | 3 | 3 | 64 | 36_208_920 | 2080 | 3.25 | 3.2500251321231617 |
| mlp.up | 64 | 17408 × 5120 | 3 | 3 | 64 | 36_208_920 | 2080 | 3.25 | 3.2500251321231617 |
| mlp.down | 64 | 5120 × 17408 | 3 | 3 | 64 | 36_208_920 | 7072 | 3.25 | 3.2500251321231617 |
| dn.in_proj_qkv | 48 | 10240 × 5120 | 4 | 7 | 64 | 27_853_079 | 2720 | 4.25 | 4.250042572021484 |
| dn.in_proj_z | 48 | 6144 × 5120 | 4 | 7 | 64 | 16_711_957 | 2720 | 4.25 | 4.250070444742838 |
| dn.in_proj_a | 48 | 48 × 5120 | 4 | 7 | 64 | 130_827 | 2720 | 4.25 | 4.25869140625 |
| dn.in_proj_b | 48 | 48 × 5120 | 4 | 7 | 64 | 130_827 | 2720 | 4.25 | 4.25869140625 |
| dn.out | 48 | 5120 × 6144 | 4 | 7 | 64 | 16_711_957 | 3264 | 4.25 | 4.250070444742838 |
| gqa.q | 16 | 12288 × 5120 | 4 | 7 | 64 | 33_423_639 | 2720 | 4.25 | 4.2500354766845705 |
| gqa.k | 16 | 1024 × 5120 | 4 | 7 | 64 | 2_785_554 | 2720 | 4.25 | 4.250418090820313 |
| gqa.v | 16 | 1024 × 5120 | 4 | 7 | 64 | 2_785_554 | 2720 | 4.25 | 4.250418090820313 |
| gqa.o | 16 | 5120 × 6144 | 4 | 7 | 64 | 16_711_957 | 3264 | 4.25 | 4.250070444742838 |
| lm_head | 1 | 248320 × 5120 | 4 | 7 | 64 | 675_430_686 | 2720 | 4.25 | 4.250001799593266 |
| embed | 1 | 248320 × 5120 | 4 | 7 | 64 | 675_430_686 | — | 4.25 | gather, not GEMV |

192 + 305 = 497 GEMVs/token. Distinct (M,K,bits) = 9.

L0 gate header MEASURED:

```
{"schema":"hawking.gravity.uniform_group.v1","representation":"uniform_q3_group_scale",
 "shape":[17408,5120],"elements":89128960,"bits":3,"group_size":64,"groups":1392640,
 "scale_bytes":2785280,"code_bytes":33423360,"scale_dtype":"float16","retained_padding_elements":0}
```

`code_bytes/groups = 33423360/1392640 = 24` B/group. DERIVED. Matches
`packed_byte_count` (`q80_mixed_decode.rs:83-91`) and the contract
("bits 3 is 24 bytes per group"). Bound SOURCE
`factor_layout_from_meta` `:1164` `bound = (1 << (bits-1)) - 1`
→ bits 3 → 3, bits 4 → 7.

L0 qkv / lm_head headers: `representation=uniform_q4_group_scale`, bits=4,
32 code bytes/group + 2 B f16 scale. Same nibble layout geo_tpr64 already
consumes, bound 7.

Payload sum MEASURED 12_149_632_429. G0-def complete BPW
`8 * 12149632429 / 26895998464 = 3.6138111608720234` MATCHES the frontier
figure. PACK_REPORT 3.6138647373176767 bills catalog bytes too.

Streamed GEMV bytes (catalog nbytes, embed excluded) DERIVED
11_472_705_646 ≈ 11.47 GB. 15.7 % under G0 13_611_663_360
(`honest_roof.rs:46`). MLP is 60.60 % of streamed GEMV bytes; lm_head
5.89 %; a+b 0.11 %; k+v 0.78 %.

---

## 3. What the candidate actually launches

### 3.1 Dispatch count 1156 — SOURCE, DERIVED

G0 schedule is 1 + 64×15 + 3 = **964**.
SOURCE `qwen38_64_layer_execution_schedule.rs:12-54`.

mixed-q3mlp has no fused QKVZ/BA, so `encode_deltanet_mixed` takes
`encode_split_deltanet_projections` (`qwen38_hybrid_decode.rs:3279-3280`):

4 GEMVs (qkv, z, b, a) + 2 fuses (`qwen38_fuse_split_qkvz_f32`,
`qwen38_fuse_split_ba_f32`) instead of 2 fused GEMVs. +4 dispatches × 48
DeltaNet layers = **+192**. 964+192 = **1156**. DERIVED.

`48×13 + 16×9 + 64×6 + 1 + 3 = 1156`. Ceremony RECEIPT 1_625_793 ns
(frontier). Dispatch count is not the 3.78×.

GEMV mix: 192 bits-3 + 305 bits-4. DERIVED from catalog + schedule.

### 3.2 Incumbent geometry — SOURCE

```1042:1044:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
    fn simd8_grid(rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1)
    }
```

```1687:1692:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), encode)
```

Both bits land on **TPR=32, RPT=8, TG=256, 8 simdgroups**. Reduction is
`simd_sum` only (one simdgroup owns the row).

bits==3 inner loop (`q80_mixed_decode.metal:869`):

```
for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u)
```

W=8, stride 256, **20 steps at K=5120**, then 8 independent scale loads
per step (`:882-889`). Comment at `:866-868` names the organ this was
written for: `down L is 2048x160`.

bits==4 inner loop (`:521-528`):

```
for (uint base = 0u; base < cols; base += kSimdWidth) {
    const uint col = base + simd_lane;
    partial += q80_uniform_value_wide(...) * input[col];
}
```

W=1, stride 32, **160 steps at K=5120**. `gk_uniform_value_wide`
(`gk_family.metal:179-205`) is a per-element bitfield extract + per-element
scale fetch.

G0 control (`qwen_uniform_q4.metal:204-210`): TPR=64, W=8, stride 512,
**10 steps at K=5120**, one scale per group of 64, `simd_sum` + 2-way TG add.

### 3.3 Incumbent vs G0 work (DERIVED, K=5120)

| | TPR | W | steps | elems/thread | scale fetches/step | reduce |
|---|---:|---:|---:|---:|---:|---|
| bits3 simd3 | 32 | 8 | 20 | 160 | 8 | simd |
| bits4 simd | 32 | 1 | 160 | 160 | 1 | simd |
| G0 geo_tpr64 | 64 | 8 | 10 | 80 | 1 / group | simd+tg(2) |

Same unique bytes. Twice the K-serial work. bits4 also 16× the loop trips.

Token TG sum incumbent DERIVED: **519_040** (rpt=8).
Token TG sum G0-class DERIVED: **2_076_160** (rpt=2).
Equal to G0 fused token GEMV TGs: split qkv+z has the same row sum as
fused qkvz (10240+6144=16384); a+b same as ba (48+48=96).

---

## 4. Geometry law

TIGHT iff `TPR × RPT = TG`. SLACK leaves `TG - TPR×RPT` idle threads in
every group. ILLEGAL iff `TPR × RPT > TG`.

Reduce: TPR=32 → simd only. TPR=64 → simd + 2-way TG add. TPR=128 → simd +
4-way TG add. SOURCE G0 kernel `:213-220`.

K ∈ {1024, 5120, 6144, 17408}. All divide `TPR × W` for TPR∈{32,64,128}
and W∈{1,8}. Last-tile idle lanes = 0 on every production K. DERIVED
`/tmp/g1_row_tg_geometry.out` `K_DIV`.

M of every GEMV class divides 2, 4, and 8. Last-TG idle rows = 0. DERIVED.

W=8 is the geo_tpr64-class extract (bits4: one uint of nibbles; bits3:
3 bytes / 8 codes, same as simd3). W=1 is the incumbent bits4 path.

---

## 5. Tight grid (W=8) — DERIVED launch math

GPU cores = 60. `TGs = ceil(M/RPT)`. `TG/core = TGs/60`.
`saturates` = TGs ≥ 60. Not a counter.

### 5.1 Legal tight set

| TPR | RPT | TG | SG/TG | reduce | steps K=5120 | elems/th | token TGs |
|---:|---:|---:|---:|---|---:|---:|---:|
| 32 | 4 | 128 | 4 | simd | 20 | 160 | 1_038_080 |
| 64 | 2 | 128 | 4 | simd+tg(2) | 10 | 80 | 2_076_160 |
| 128 | 1 | 128 | 4 | simd+tg(4) | 5 | 40 | 4_152_320 |
| 32 | 8 | 256 | 8 | simd | 20 | 160 | 519_040 |
| 64 | 4 | 256 | 8 | simd+tg(2) | 10 | 80 | 1_038_080 |
| 128 | 2 | 256 | 8 | simd+tg(4) | 5 | 40 | 2_076_160 |

Incumbent is row 4. G0 is row 2. RPT=8 is outside {1,2,4} but is what
ships.

ILLEGAL (do not emit): TPR=64 RPT=4 TG=128; TPR=128 RPT=2 TG=128;
TPR=128 RPT=4 TG=128 or 256.

SLACK (idle threads every TG — do not ship): TPR=32 RPT=1/2 any TG;
TPR=64 RPT=1 any TG; TPR=64 RPT=2 TG=256; TPR=32 RPT=4 TG=256;
TPR=128 RPT=1 TG=256.

### 5.2 Per-shape occupancy at the three useful tights + incumbent

`sat` = TGs≥60.

| shape | M | n | inc TGs (tpc) | tpr32/4/128 | tpr64/2/128 | tpr128/1/128 | tpr64/4/256 |
|---|---:|---:|---|---|---|---|---|
| mlp.gate/up | 17408 | 64+64 | 2176 (36.27) sat | 4352 (72.53) | **8704 (145.07)** | 17408 (290.13) | 4352 (72.53) |
| mlp.down | 5120 | 64 | 640 (10.67) sat | 1280 (21.33) | **2560 (42.67)** | 5120 (85.33) | 1280 (21.33) |
| dn.qkv | 10240 | 48 | 1280 (21.33) sat | 2560 (42.67) | **5120 (85.33)** | 10240 (170.67) | 2560 (42.67) |
| dn.z | 6144 | 48 | 768 (12.80) sat | 1536 (25.60) | **3072 (51.20)** | 6144 (102.40) | 1536 (25.60) |
| dn.a / b | 48 | 48+48 | 6 (**0.10**) no | 12 (0.20) no | 24 (**0.40**) no | **48 (0.80)** no | 12 (0.20) no |
| dn.out / gqa.o | 5120 | 48+16 | 640 (10.67) sat | 1280 (21.33) | **2560 (42.67)** | 5120 (85.33) | 1280 (21.33) |
| gqa.q | 12288 | 16 | 1536 (25.60) sat | 3072 (51.20) | **6144 (102.40)** | 12288 (204.80) | 3072 (51.20) |
| gqa.k / v | 1024 | 16+16 | 128 (2.13) sat | 256 (4.27) | **512 (8.53)** | 1024 (17.07) | 256 (4.27) |
| lm_head | 248320 | 1 | 31040 (517.3) sat | 62080 (1034.7) | **124160 (2069.3)** | 248320 (4138.7) | 62080 (1034.7) |

Idle last group: 0 rows, 0 col-lanes, every cell. DERIVED.

### 5.3 Arithmetic intensity (DERIVED)

`AI = 2K / (Wbytes + 4K/RPT + 4)`. Formula SOURCE
`g1-direct-gemv-geometry.md:224-232`. Wbytes from peeked body.

| class | Wbytes | AI RPT=1 | RPT=2 | RPT=4 | RPT=8 |
|---|---:|---:|---:|---:|---:|
| mlp.gate/up q3 | 2080 | 0.454 | 0.831 | 1.421 | 2.205 |
| mlp.down q3 | 7072 | 0.454 | 0.831 | 1.422 | 2.206 |
| bits4 K=5120 | 2720 | 0.441 | 0.790 | 1.305 | 1.938 |
| bits4 K=6144 | 3264 | 0.441 | 0.790 | 1.306 | 1.938 |

G0 Q4 at RPT=2 is RECEIPT bandwidth-saturated at 639–700 GB/s
(`g1-direct-gemv-geometry.md:105-124`). Raising RPT raises AI but this
genome's limiter on large M is not X-reuse once TPR=64 W=8 is on.
Raising RPT is how a/b occupancy dies.

---

## 6. Why TG=256 is slow (and not sufficient)

The bits-3 path already is TG=256. Frontier GEMV bucket 137_099_341 ns =
92.27 % of 148.6 ms token, 3.78× G0. Bytes are 15.7 % *fewer*. So TG=256
did not buy the roof.

Causes, in order, all SOURCE/DERIVED, none a new GPU run:

1. **TPR stayed 32.** TG=256 packed 8 rows, it did not split K further.
   Each thread still walks K/32 elements. G0 walks K/64. Doubling TG
   without doubling TPR does not shorten the column walk.

2. **bits4 is W=1.** 160 scalar `gk_uniform_value_wide` trips vs 10
   uint-nibble trips. Same 160 elements, 16× the loop, per-element scale.

3. **bits3 fetches 8 scales per 8 codes** (`:882-889`) even though
   group=64 so those 8 share one scale. Decode tax on a codec whose
   recon at tpr64 was RECEIPT-free
   (`g1-direct-gemv-geometry.md:131-137` uniform q3 tpr64 15125 ns,
   `recon_excess_ns=0`).

4. **The tile was written for HGRAVS factors**, not Q38 dense GEMVs.
   Comment `:866-868`. Rank-160 / 2048-col organs. Q38 MLP is
   17408×5120 and 5120×17408.

5. **RPT=8 starves the only small organs.** a/b: 6 TGs = 0.10 TG/core.
   G0-class: 24 TGs = 0.40. Still hungry, but 4× better. High RPT is the
   wrong direction for M=48.

6. **RECEIPT: same codecs at tg256 are slower on the discriminator
   organ.** `g1-direct-gemv-geometry.md:143` gate-class
   `same codecs at tg256 ~26541` vs tpr64 ~15125–15500 ns. That organ is
   512×2048 and launch-dominated — not a Q38 token — but it falsifies
   "bigger TG is free".

7. **X-reuse at RPT=8 is real (AI 2.20 vs 0.83) and does not matter
   while extract is 1-wide.** The genome is decode-taxed, not X-bound.

KILLS: "switch to TG=256" as the next kernel move. It is the current
move. REOPEN_IF an A/B at fixed TPR=64 W=8 shows TG=256 RPT=4 beating
TG=128 RPT=2 on a *full-size* down or lm_head (X-reuse thesis). That
test is lane 110's, not this one.

---

## 7. Per-shape recommendation

First-try extract for both bits: W=8, group-64 scale, no repack.
bits3 = simd3 3-byte/8-code at stride `TPR×8`. bits4 = G0 nibble unpack
bound 7. Word-law is already satisfied for bits4. bits3 stays the LSB
stream (C-3C in `g1-direct-gemv-geometry.md:342`); contract says that
body feeds geo_tpr64 directly.

| class | rec | TGs | TG/core | reduce | steps | idle | why |
|---|---|---:|---:|---|---:|---|---|
| mlp.gate, mlp.up | **64 / 2 / 128** | 8704 | 145.07 | simd+tg(2) | 10 | 0 | G0 launch. MEASURED BW-sat on this M,K. Do not drop to TPR=32. |
| mlp.down | **64 / 2 / 128** | 2560 | 42.67 | simd+tg(2) | 34 | 0 | K=17408 still 34 W=8 steps. TPR=128 halves steps, doubles TGs, 4-way TG reduce. No receipt it wins. |
| dn.in_proj_qkv | **64 / 2 / 128** | 5120 | 85.33 | simd+tg(2) | 10 | 0 | same K as gate |
| dn.in_proj_z | **64 / 2 / 128** | 3072 | 51.20 | simd+tg(2) | 10 | 0 | sat |
| dn.in_proj_a, b | **128 / 1 / 128** if a second kernel exists; else share 64/2/128 | 48 / 24 | 0.80 / 0.40 | tg(4) / tg(2) | 5 / 10 | 0 | only class with TGs<60 at every legal geometry. Bytes 0.11 %. |
| dn.out, gqa.o | **64 / 2 / 128** | 2560 | 42.67 | simd+tg(2) | 12 | 0 | K=6144, 12 steps |
| gqa.q | **64 / 2 / 128** | 6144 | 102.40 | simd+tg(2) | 10 | 0 | sat |
| gqa.k, v | **64 / 2 / 128** | 512 | 8.53 | simd+tg(2) | 10 | 0 | G0 uses this. Incumbent 2.13 is worse. |
| lm_head | **64 / 2 / 128** | 124160 | 2069.3 | simd+tg(2) | 10 | 0 | same as G0 isolated ~1.02 ms RECEIPT. Do not special-case. |
| embed | not a GEMV | — | — | — | — | — | `qwen38_hgravu_embedding_lookup` already bits-generic |

No shape wants incumbent TPR=32 W=1.

TPR=32 RPT=4 TG=128 (simd-only reduce) is the only other tight that is
not worse on occupancy for large M. It doubles elems/thread (160 vs 80)
and was not the geometry-sweep winner. Do not pick it first.

TPR=128 RPT=1 TG=128 is the occupancy-max tight for a/b and k/v. On
gate it launches 17408 TGs and a 4-way TG reduce. No receipt it beats
TPR=64 on a 13.6 GB-class stream. Reserve it for a/b only.

---

## 8. One compromise if the model shares a kernel

**TPR=64, RPT=2, TG=128, W=8.**

Predicted (DERIVED, not timed):

| class | TGs / dispatch | TG/core | sat | token TGs | reduce | steps (K) |
|---|---:|---:|---|---:|---|---|
| mlp.gate | 8704 | 145.07 | yes | 557_056 | simd+tg(2) | 10 |
| mlp.up | 8704 | 145.07 | yes | 557_056 | simd+tg(2) | 10 |
| mlp.down | 2560 | 42.67 | yes | 163_840 | simd+tg(2) | 34 |
| dn.qkv | 5120 | 85.33 | yes | 245_760 | simd+tg(2) | 10 |
| dn.z | 3072 | 51.20 | yes | 147_456 | simd+tg(2) | 10 |
| dn.a | 24 | **0.40** | **no** | 1_152 | simd+tg(2) | 10 |
| dn.b | 24 | **0.40** | **no** | 1_152 | simd+tg(2) | 10 |
| dn.out | 2560 | 42.67 | yes | 122_880 | simd+tg(2) | 12 |
| gqa.q | 6144 | 102.40 | yes | 98_304 | simd+tg(2) | 10 |
| gqa.k | 512 | 8.53 | yes | 8_192 | simd+tg(2) | 10 |
| gqa.v | 512 | 8.53 | yes | 8_192 | simd+tg(2) | 10 |
| gqa.o | 2560 | 42.67 | yes | 40_960 | simd+tg(2) | 12 |
| lm_head | 124160 | 2069.3 | yes | 124_160 | simd+tg(2) | 10 |
| **token** | | | | **2_076_160** | | |

Same token TG count as G0 fused. Occupancy on every organ with M≥1024
matches the production genome that is RECEIPT-saturated at 639–700 GB/s.

### 8.1 Who the compromise serves badly

| class | why poor | byte share of streamed GEMV | can it lose the token? |
|---|---|---:|---|
| dn.in_proj_a | 24 TGs, 0.40 /core | 0.055 % | no |
| dn.in_proj_b | 24 TGs, 0.40 /core | 0.055 % | no |

Not on this list:

- **lm_head.** 2069 TG/core, 5.89 % of bytes, identical to G0. A
  shared kernel that is right on MLP is automatically right on lm_head
  because lm_head M is 14× gate M.
- **gqa.k / v.** 8.53 TG/core. G0 lives here. 0.78 % of bytes.
- **mlp.\*.** 42–145 TG/core, 60.6 % of bytes.

The contract's named failure mode — "optimal on MLP, poor on lm_head" —
does not occur for any legal geometry in this set. lm_head cannot be the
occupancy victim. a/b can, and do not matter.

If lane 110 later adds a second specialization, give it to a/b
(TPR=128 RPT=1) or to C2 split-K (out of requested set; 4 TGs/row →
192 TGs = 3.2 /core on M=48). Do not retune the shared kernel around
them.

---

## 9. Predicted GEMV floor (PROJECTED, not a token)

Streamed GEMV bytes MEASURED-from-catalog 11_472_705_646.
Sealed G0 addressing roof RECEIPT 639.25 GB/s
(`13_611_663_360 / 21_293_102.5`).

`11472705646 / 639.25e9 = 17.95 ms` PROJECTED addressing only.

Current GEMV bucket RECEIPT 137.10 ms. Tax vs that floor ≈ 7.6×.
That tax is TPR=32 + W=1 + per-element scale, not density, not 1156 vs
964, not a serial reduction (both incumbent paths already `simd_sum`).

31 TPS / PACK_REPORT 30.618 TPS remain PROJECTED complete-token numbers.
They assume the G0 roof transfers to HGRAVU01 at tpr64. Transfer is
plausible (same launch, same K, register decode, recon_excess RECEIPT 0
at tpr64 for uniform q3/q4) and is **not measured on this mix**.

---

## 10. KILLS / REOPEN_IF

| | |
|---|---|
| KILLS | TG=256 as the fix. Already shipping. Slow. |
| KILLS | TPR=32 W=1 for bits 4. Named mechanism. |
| KILLS | SLACK launches (idle threads every TG). |
| KILLS | ILLEGAL TPR×RPT > TG. |
| KILLS | One-thread-per-row / serial family (`HAWKING_QWEN38_RECON_FUSE=0`). |
| KILLS | Reopening representation search. Floor is located. |
| KILLS | Treating 31 TPS or 17.95 ms as a measured token. |
| KILLS | Fusing a/b back just to raise occupancy. Fuse kernels already exist; the bytes are 0.11 %. |
| REOPEN_IF | Isolated full-organ A/B at fixed TPR=64 W=8 shows RPT=4 TG=256 < RPT=2 TG=128 on down or lm_head (X-reuse). |
| REOPEN_IF | Token profile after tpr64 lands still shows a/b+k/v as a named slice. Then C2 split-K or TPR=128 on those two shapes only. |
| REOPEN_IF | Hardware occupancy counters (not launch math) show HGRAVU01 bits-3 extract register-limited at TPR=64. Then TPR=32 RPT=4 TG=128 is the fallback tight. |

Cheapest experiment this lane did not run (GPU authority is lane 110):
one isolated CB, L0 gate 17408×5120 HGRAVU01 q3, tpr64/rpt2/tg128 vs
incumbent simd3. Expected direction: toward the 15 µs-class discriminator
times, not toward 137 ms. That is a component microbenchmark, not a token.

---

## 11. Evidence index

| claim | pointer |
|---|---|
| catalog 851 / 258 / 180124 / magic | `xxd -l 32` + parser stdout `CATALOG`; `qwen38_hybrid_decode.rs:31-33,411-488` |
| 192 q3 + 305 q4 GEMV + 1 embed + 353 other | parser `CLASS_COUNTS`, `GEMV_BITS`, `OTHER_n` |
| shapes / nbytes / bits / bound | parser `DISTINCT_GEMV` + header JSON peeks |
| no fused in_proj | parser `FUSED_PRESENT {}` |
| BPW 3.6138111608720234 | `8*12149632429/26895998464`; PACK_REPORT 3.6138647 bills catalog |
| 11.47 GB streamed | 12_149_632_429 − embed 675_430_686 − other 1_496_097 |
| G0 sha live | `shasum` this process, matches contract |
| simd8_grid / bits3 / bits4 bind | `qwen38_hybrid_decode.rs:1042-1044,1687-1692` |
| simd3 stride 256, 8 scales | `q80_mixed_decode.metal:845-907` |
| simd W=1 stride 32 | `q80_mixed_decode.metal:499-534`; `gk_family.metal:179-205` |
| G0 tpr64 | `qwen_uniform_q4.metal:181-221`; launch `:576-582` |
| 1156 vs 964 | schedule `:12-54` + split `:1807-1847,3279-3280` |
| occupancy formula | `qwen38_token_ns_ledger.rs:174-186` |
| TG/core tables | `/tmp/g1_row_tg_geometry.out` `TOKEN_TG_SUMS` |
| K divides every TILE | same file `K_DIV` |
| recon tpr64 free / tg256 slower | `g1-direct-gemv-geometry.md:131-143` RECEIPT |
| 639.25 GB/s roof | `g1-direct-gemv-geometry.md:109`; `honest_roof.rs:46-50` |
| 137.10 ms GEMV / 148.59 ms token | campaign frontier, not re-measured |
| 31 TPS / 30.618 TPS | PROJECTED; PACK_REPORT `projection` |
| G0 isolated lm_head 1_017_458 ns | `g1-token-anatomy.md:157` RECEIPT |

Parser command (this process, no Metal):

```
python3 /tmp/g1_row_tg_geometry.py
# EXIT:0   452 lines   PAYLOAD_SUM 12149632429
# incumbent token_tgs=519040
# tpr64_rpt2_tg128 token_tgs=2076160
# DISPATCH {g0:964, candidate_split:1156, gemv_bits3:192, gemv_bits4:305, delta:192}
```
