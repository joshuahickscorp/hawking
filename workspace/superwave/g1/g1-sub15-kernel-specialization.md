# G1 mixed-sub15-v1 kernel specialization

Lane: `63-sub15-kernel-specialization`. No GPU, no generate, no inference, no
repack. One new file. Every number is `MEASURED` (this process, this artifact
or this receipt), `RECEIPT` (quoted field), `SOURCE` (file:line), `DERIVED`
(exact arithmetic on those), or `ESTIMATED`.

A component microbenchmark is not a token-level claim. Predicted bytes/out
and dispatch deltas are not TOKEN_NS.

Artifact (MEASURED on this box):
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1`

Packed complete BPW `1.2910781930062503` is a RECEIPT (`PACK_REPORT.json:23`)
and independently MEASURED as `8 * 4340604637 / 26895998464`.

Native load is a sibling lane (`g1-sub15-native-gap.md`). This lane designs
the representation-specific consume of the four codecs that pack actually
contains.

---

## 0. Verdict

Nothing here needs a codebook gather, a dense expand, or a new codec family.

| codec | reconstruction | production bind | K-complete? |
|---|---|---|---|
| HGRAVB01 | register-only, `±s` from 1 bit + f16 mean-abs | `q80_binary_group_matvec_tg256` | **NO**. `lid*8` covers 2048 cols. |
| HGRAVR02 | register-only binary + CSR `±rms`; Rice is load-time only | `q80_binary_group_csr_matvec_tg256` | **NO**. Same 2048-col bind. |
| HGRAVS01 r160_b3 | register-only 3-bit `q-3`; two-stage `L@(R@x)` | two × `q80_hgravs01_factor_matvec_simd3` | **YES**. `col += 256`. |
| HQ30UQ4 | register-only nibble `q-8` + f16 absmax | `geo_tpr64_tg128` / embed lookup | **YES**. TILE=512 divides both K. |

Ideal binary/rice mapping for **this** model's K∈{5120,6144} is tpr64 ×
exclusive-byte W=8 × TILE=512 × TG=128 (10 steps / 12 steps), not TILE=2048.
Tiling siblings that already walk all K exist (`*_simd_bytes`).

Specialization rank (bytes/out + dispatch, occupancy preserved):

1. **K-complete tile** — must. Dispatch Δ = 0 vs HEAD bind; −864 vs a 3-pass
   2048-col rebind. Repeats no recorded negative.
2. **Fold scale into the packed stream** — same unique bytes, −1 buffer/GEMV.
   Repeats no recorded negative.
3. **In-register reconstruction that drops the scale-plane read** — decode is
   already register-only. Deleting the plane is a pack change, not a kernel
   change, and is almost free at the incumbent 2-rows/TG launch because X
   dominates. Repeats no recorded negative.
4. **Fused multi-codec kernel that walks a whole layer** — **KILLS**. This is
   the megakernel negative. REOPEN_IF inline packed consume **and** multi-TG
   occupancy ≥ `geo_tpr64_tg128` **and** a complete-token A/B that is not
   slower.

A legal non-layer fusion (gate HGRAVB01 + up HGRAVR02, shared X, K=5120) is
not item 4 and is listed separately.

Direct codebook-lookup GEMV is **KILLS** (460.041 µs RECEIPT). None of the
four packed codecs is a codebook.

---

## 1. K that a tile must finish

Authority: `qwen38_geometry.rs:20-52` plus the mixed-sub15 census
(`g1-sub15-native-gap.md` §3; this process re-read `packed/attn_rows.json`
and mixed-2p0 `catalog.hq38m20`).

| organ | n | M × K | packed codec |
|---|---:|---|---|
| mlp.gate_proj | 64 | 17408 × **5120** | HGRAVB01 g=128 |
| mlp.up_proj | 64 | 17408 × **5120** | HGRAVR02 rice_q1_rms_2pct |
| mlp.down_proj | 64 | 5120 × 17408 | HGRAVS01 r160_b3 (factors, not a dense K) |
| in_proj_qkv | 48 | 10240 × **5120** | HGRAVR02 |
| in_proj_z | 48 | 6144 × **5120** | HGRAVR02 |
| in_proj_a / b | 48+48 | 48 × **5120** | HGRAVR02 |
| out_proj | 48 | 5120 × **6144** | HGRAVR02 |
| q_proj | 16 | 12288 × **5120** | HGRAVR02 |
| k_proj / v_proj | 16+16 | 1024 × **5120** | HGRAVR02 |
| o_proj | 16 | 5120 × **6144** | HGRAVR02 |
| lm_head | 1 | 248320 × **5120** | HQ30UQ4 |
| embed | 1 | gather 1 × **5120** | HQ30UQ4 (not a GEMV) |

Every HGRAVB01 / HGRAVR02 GEMV has K∈{5120,6144}. Both divide 128, 256, 512,
1024. **2048 divides 6144 (3×) and does not divide 5120 (remainder 1024).**
`g1-direct-gemv-geometry.md:69` said 2048 divides none of {5120,17408,6144}.
The 6144 clause is false; 5120 and 17408 stand. DERIVED `6144/2048=3`.

Production Q4 genome (G0, not this pack's binary tiles): 64 threads/row,
2 rows/TG, TG=128, 4 simdgroups. SOURCE `qwen_uniform_q4.metal:181-221`,
`qwen38_hybrid_decode.rs:264-268`. TILE = 64×8 = 512.

Shipping mixed bind (recon-fuse ON, default): one 256-thread TG per row,
`col = lid*8` → 2048 columns, **no tile loop**. SOURCE
`q80_mixed_decode.metal:699-728,742-793`; bind
`qwen38_hybrid_decode.rs:1330-1360`. `decode_family.rs:16-20` names those
tg256 kernels as the occupancy tiles.

Siblings that already walk all K: `q80_binary_group_matvec_simd_bytes` /
`q80_binary_group_csr_matvec_bytes` (`:620-648,:796-828`, `base += 256`);
`gk_binary_group_serial_row` (`gk_family.metal:116-160`).
`q80_uniform8_matvec_tg256` already documents the missing loop
(`:991-1016` "Loops 2048-col tiles so 4096-col out_proj is covered").

---

## 2. HGRAVB01 — binary_g128

### 2.1 Layout (SOURCE + MEASURED)

Container `HGRAVB01` / `hawking.gravity.binary_sign_scale.v1`.
Body = `fp16 scales[rows * (K/128)] || LSB-first signs[rows*K bits]`.
`q80_mixed_decode.rs:13-15,183-229,1174-1207`.

Scale is **mean-abs** of the 128-wide group, not absmax (`:183-184,218`).
Sign bit 1 → `+s`, 0 → `−s` (`binary_group_weight` `:232-241`).
`cols % 128 == 0` is a loader hard-fail (`:197-200,1190-1194`).

L0 gate header MEASURED (mixed-2p0 segment, byte-identical MLP):
`shape [17408,5120]`, `group_size 128`, `scale_bytes 1392640`,
`sign_bytes 11141120`. 64× that = `89128960 + 713031680 = 802160640` body.
PACK_REPORT gate `802177344` is body + JSON headers (DERIVED header
`16704 / 64 = 261` B, matches `g1-heterogeneous-allocation.md` "header 261").

Wbytes/row DERIVED:

| K | groups | signs | scales | W/row | physical BPW |
|---:|---:|---:|---:|---:|---:|
| 5120 | 40 | 640 | 80 | **720** | 1.125 |
| 6144 | 48 | 768 | 96 | **864** | 1.125 |

### 2.2 Per-thread decode, production tg256 (wrong K)

SOURCE `q80_mixed_decode.metal:297-318,701-739`.

```
// one 256-thread TG owns one output row
col = lid * 8;                          // 0,8,…,2040
if (col + 8 > cols) skip;               // K=5120/6144: never true for lid<256
s    = float(scales[row * gpr + col/128]);          // 2 B
byte = signs[(row*cols + col) >> 3];                // 1 B, exclusive
acc  = 0;
#pragma unroll
for (j = 0; j < 8; ++j)
    acc += ((byte >> j) & 1 ? +s : -s) * x[col+j];  // 8 FMAs, 32 B of X
acc = simd_sum(acc);
// 8 simdgroup partials → lid 0 writes y[row]
```

Register-only. No table. No shuffle. One exclusive sign byte per lane.

Covers **2048 / K** of the row: 0.400 at K=5120, 0.333 at K=6144. DERIVED.
The other columns are not read. This is the mixed-2p0 generate confound
(`g1-sub15-native-gap.md:244-247`).

Per thread, this bind (MEASURED instruction mix, DERIVED bytes):
1 B signs + 2 B scale + 32 B X = **35 B**, 16 FLOPs, AI_thread = 0.457.
W actually addressed / row = `2048/8 + 2*(2048/128)` = 256+32 = **288 B**
at either K (the leftover groups are never touched).

### 2.3 Per-thread decode, K-complete tpr64 × W=8 (ideal for these K)

Same `q80_binary_byte_dot`, launch of `geo_tpr64_tg128`:

```
lane = split*32 + simd_lane;            // 0..63
for (col = lane*8; col < K; col += 512)
    acc += q80_binary_byte_dot(...);    // 1 B + 2 B + 8 FMAs
acc = simd_sum(acc);
if (split==0 && lane==0) y[row] = red[0]+red[1];
```

TILE=512 divides both K. Steps: **10** (K=5120), **12** (K=6144). Every
lane busy every step.

| K | steps | B/thread (1+2+32)×steps | FMAs/thread | AI_thread |
|---:|---:|---:|---:|---:|
| 5120 | 10 | **350** | 80 | 0.457 |
| 6144 | 12 | **420** | 96 | 0.457 |

Sibling already compiled: `q80_binary_group_matvec_simd_bytes` — 32 lanes,
8 rows/TG, TILE=256 (20 / 24 steps). Same byte_dot, K-complete.
Per thread 700 B / 840 B. More X re-read per thread, better X reuse
across the 8 rows.

### 2.4 Arithmetic intensity (organ, not thread)

`AI = 2K / (Wbytes + 4K/Rtg + 4)`. DERIVED. Formula from
`g1-direct-gemv-geometry.md:224-232`.

| K | W | AI Rtg=2 (C0) | AI Rtg=8 (simd_bytes) | AI Rtg=32 (C1) |
|---:|---:|---:|---:|---:|
| 5120 | 720 | 0.934 | 3.118 | 7.507 |
| 6144 | 864 | 0.934 | 3.119 | 7.511 |

Incumbent 2-rows/TG throws the density win away: X is 10240 B (K=5120)
against 720 B of W. Same fact the geometry paper named; it is load-bearing
for any 1.125 BPW consume.

Relative ALU (RECEIPT, not GB/s): `QWEN38_RECON_MEASURED.json` gate
`binary_g128/disc_binary_tpr64` median 15416 ns, `recon_excess_ns=0`
against `f32_tpr64` 15125 ns. Reconstruction of this codec is free at tpr64.

---

## 3. HGRAVR02 — binary + rice_q1_rms @ 2%

### 3.1 Layout (SOURCE + MEASURED)

Container `HGRAVR02` / `hawking.gravity.binary_outlier_residual.v2`.
Body = binary body || `u32 first_index` || rice(diffs) || `fp16 rms` ||
1-bit residual signs. `q80_mixed_decode.rs:16-19,383-387,1209-1251`.

Rice = unary quotient (1-bits) + 0 + k LSBs, LSB-first
(`:64-105,336-346`). `rice_k=5` on **every** attention rice and **every**
MLP up (MEASURED: 304/304 and 64/64 headers).

Token path **must not** Rice-decode. Host `expand_rice_indices` +
`rice_q1_row_ptr` at upload (`qwen38_hybrid_decode.rs:1088-1114`;
`q80_mixed_decode.rs:461-506`). Rice-as-GEMV is KILL
(`g1-direct-gemv-geometry.md` K3).

Residual value = `sign × rms` of the selected 2% (`q80_residual_q1_value`
`:107-115`; one `residual_scale_f16` per **tensor**, not per group).

### 3.2 MEASURED residual census

Attention, 304 `packed/attn/*.rice` headers, this process:

| | |
|---|---:|
| outlier_count sum | 144_756_064 |
| scale_bytes (binary f16) | 113_090_560 |
| sign_bytes (binary) | 904_724_480 |
| rice_bytes | 129_012_216 |
| residual_sign_bytes | 18_094_672 |
| file bytes | 1_165_098_376 |

Outliers/row is the 2% identity: **102.4** at K=5120, **122.88** at K=6144
(MEASURED `out/row` on every shape matches `0.02*K` to ≤0.017).

MLP up, 64 mixed-2p0 HGRAVR02 headers (byte-identical to this pack's MLP):

| | |
|---|---:|
| outlier_count sum | 114_085_120 |
| rice_k | 5 × 64 |
| binary scale / sign | 89_128_960 / 713_031_680 |
| rice_bytes | 101_576_736 |
| residual_sign_bytes | 14_260_672 |

CSR that the token kernel actually streams (DERIVED from those counts):

| class | u32 indices | row_ptr | residual signs | CSR total |
|---|---:|---:|---:|---:|
| attention | 579_024_256 | 5_393_600 | 18_094_672 | 602_512_528 |
| mlp up | 456_340_480 | 4_456_704 | 14_260_672 | 475_057_856 |

Packed rice (129 MB attn + 102 MB up) expands to **1.078 GB** of u32 CSR
indices. Token-time physical rate on the rice organs is therefore **not**
the packed 1.288 BPW:

| organ | packed BPW (RECEIPT) | token-time W (binary+CSR) | token-time BPW (DERIVED) |
|---|---:|---:|---:|
| attention GEMV | 1.28779 | 1_620_327_568 | **1.791** |
| mlp up | 1.28751 | 1_277_218_496 | **1.791** |

`1.791 = 8*(720 + 0.02*K*4 + 0.02*K/8)/K` at K=5120. The u32 index is the
tax for keeping Rice off the GEMV.

### 3.3 Per-thread decode, production csr_tg256 (wrong K)

SOURCE `q80_mixed_decode.metal:743-793`.

```
// same binary_byte_dot as §2.2 (2048 cols only)
// then lid==0, serially:
rscale = float(as_type<half>(residual_scale_bits));   // 2 B, tensor-wide
for (n = row_ptr[row]; n < row_ptr[row+1]; ++n) {     // ~102 or ~123
    col = indices[n] % cols;                          // 4 B gather
    acc += q80_residual_q1_value(signs, n, rscale) * x[col];
}
```

Binary half: register-only, identical to §2.2.
Residual half: register-only given expanded CSR (1-bit extract + one
broadcast rms). **Table fetch: none.** Irregular `x[col]` gathers, ~102
(K=5120) or ~123 (K=6144) per row, **one thread**. 255 of 256 lanes idle
for the sidecar.

fuse-OFF splits this into `gk_matvec_binary` + `q80_sparse_q1_apply_csr`
(`qwen38_hybrid_decode.rs:1361-1373`) — +1 dispatch per residual GEMV.

`q80_sparse_q1_apply_csr_simd` (`:466-496`) already strides the sidecar
across 32 lanes. The shipping fused tile does not use it.

### 3.4 Ideal mapping for K=5120 / 6144

1. Binary body: same as §2.3 (tpr64 × W=8 × TILE=512).
2. Residual: 32-lane strided CSR after `simd_sum`, not `lid==0` serial.
   ~3–4 outliers/lane at K=5120, ~4 at K=6144. DERIVED `102.4/32=3.2`.
3. Store **u16 column** in the CSR (K≤6144 < 65536; row is implicit in
   `row_ptr`). Cuts index traffic in half: attn 579→289.5 MB, up 456→228 MB.
   ESTIMATED −502 MB/token of addressing. Requires a load-time pack change
   of the already-expanded buffer, not a Rice-as-GEMV reopen.
4. Do not decode Rice on device. K3 stands.

Relative ALU (RECEIPT): gate `rice_q1_rms_2pct/csr_inregister` 15125 ns,
`recon_excess_ns=0`. The sidecar is free at tpr64 as ALU. It is not a
proof the 2% scatter is free as unique-once bandwidth on the 1.08 GB
index working set.

Wbytes/row at token time, K=5120: `720 + 102.4*4 + 102.4/8 ≈ 1142`.
AI Rtg=2: `10240 / (1142+10240+4) = 0.899`. DERIVED.

---

## 4. HGRAVS01 r160_b3 — two-stage q3

### 4.1 Layout (SOURCE + MEASURED)

Container `HGRAVS01` / `hawking.gravity.activation_weighted_svd_low_rank.v1`.
Execute `y = L @ (R @ x)`. Mid[160] is the only temporary. Never dense W.
`q80_mixed_decode.rs:20-25,615-628`; lock
`qwen38_hybrid_decode.rs:37-39,1116-1134`.

L0 down header MEASURED:

```
rank=160 factor_bits=3 factor_group_size=64
left  shape [5120, 160]  codes 307200  scales 25600  groups 12800
right shape [160, 17408] codes 1044480 scales 87040  groups 43520
left_body 332800  right_body 1131520  pad 0
```

64× bodies = 93_716_480. PACK_REPORT `93847197` is that plus JSON.

Value: LSB-first unsigned 3-bit, `q = code - 3`, `w = q * f16_absmax`.
`pack_uniform_factor` `:547-567`; simd3 unpack `:869-897`.
Groups run over the **flattened** factor. R is row-aligned
(`17408 % 64 == 0`). L is **not** (`160 % 64 == 32`); group 2 of row 0
continues into row 1. SOURCE comment `q80_mixed_decode.metal:231-232`
(Q80 L is 2048×160; Q38 L is 5120×160, same remainder).

### 4.2 Per-thread decode, production simd3 (K-complete on the factors)

SOURCE `q80_mixed_decode.metal:845-907`. Grid `ceil(rows/8)*256`, TG 256.
One simdgroup owns one factor row.

```
// precondition: (row*cols + col) % 8 == 0 so 8 codes = 3 aligned bytes
for (col = simd_lane*8; col+8 <= cols; col += 256) {
    byte0 = ((row*cols + col) * 3) >> 3;
    b0,b1,b2 = codes[byte0 .. byte0+3];          // 3 B
    q0 =  (b0      ) & 7;  q1 = (b0 >> 3) & 7;
    q2 = ((b0 >> 6) | (b1 << 2)) & 7;
    q3 =  (b1 >> 1) & 7;   q4 = (b1 >> 4) & 7;
    q5 = ((b1 >> 7) | (b2 << 1)) & 7;
    q6 =  (b2 >> 2) & 7;   q7 = (b2 >> 5) & 7;
    q  = int(qi) - 3;                            // bound = 3
    // 8 scale loads; g=64 and 8-aligned col ⇒ they alias to 1 half
    acc += float(q[j]) * s[j] * x[col+j];        // 8 FMAs
}
// remainder 1-wide q80_uniform_value_wide (not taken when cols % 8 == 0)
acc = simd_sum(acc); lane0 writes y[row]
```

Register-only. No table. 3-bit extract fails the 32-contiguous-uint word
law (`g1-direct-gemv-geometry.md` C-3C); it is the shipped consume.

Two dispatches (`dispatch_hgravs` `:1418-1454`):

| stage | shape | TGs (simd8_grid) | tiles/lane | unique B/working-thread | FMAs |
|---|---|---:|---:|---:|---:|
| R @ x | 160 × 17408 | 20 | 68 | 204 codes + 136 scale + 2176 X = **2516** | 544 |
| L @ mid | 5120 × 160 | 640 | 1 | 3 + 2 + 32 = **37** (20 of 32 lanes) | 8 |

`17408/256=68` exact. `160/8=20` working lanes, then `col += 256` exits.
Stage-1 occupancy is 20 TGs on a 60-core launch model
(`qwen38_token_ns_ledger.rs:183` `gpu_cores: 60`) → 0.33 TG/core. This is
the geometry-paper C2 case.

`q80_hgravs01_two_stage_matvec` fused-in-one-dispatch exists and is
**illegal on this model**: `right_cols > 512` returns
(`q80_mixed_decode.metal:561-568`, `kXCap=512`). Q38 R has 17408 cols.
Do not bind it. K6 in the geometry paper.

### 4.3 Bytes / AI

DERIVED per down organ, unique W:

| | bytes |
|---|---:|
| L q3 + f16 | 332_800 |
| R q3 + f16 | 1_131_520 |
| total | 1_464_320 |
| dense Q4 same organ | 47_349_760 |
| byte ratio | 32.34× |
| W / output row | 286 |
| FLOPs two-stage | 7_208_960 |
| FLOPs dense | 178_257_920 |
| FLOP ratio | 0.0404 |
| bytes/out (W + Xopt + mid + y) | 303.7 |
| AI_opt | 4.636 |

Relative ALU (RECEIPT): down `hgravs01_r160_q3` **71458 ns**,
`recon_excess_ns=67851.7`, vs `f32_tpr64` 7083 ns. Two `disc_uniform_bits_tpr64`
launches on a 160-row organ. Occupancy failure, not a byte-roof measurement.
Byte floor at the measured Q4-full 666.7 GB/s regime would be 2.20 µs
(`g1-direct-gemv-geometry.md:504-508`, ESTIMATED conditional).

### 4.4 Ideal mapping for this model's factors

K of the **dense** down is 17408, not 5120/6144. The factors are 17408 and 160.

- Stage 1 (M=160, K=17408): split-K, 4–8 TGs/row (geometry C2). TILE=256
  already divides 17408. Do not use C0 (1.33 TG/core) or C1 (worse).
- Stage 2 (M=5120, K=160): one tpr32 step. 20/32 lanes is acceptable on
  640 TGs. Optional: pad K 160→256 and pay 96 q3 zeros / row (ESTIMATED
  5120×36 B = 184 KB) to fill the simdgroup. Not required.
- Keep two dispatches until a C2 two-stage is written that does **not**
  cap K at 512 and does **not** serialize R through one TG.
- 3-bit word-law remap (C-3B) needs TILE 320/640; 160 divides neither.
  Keep C-3C LSB extract on the factors.

---

## 5. HQ30UQ4 — group-64 nibble

### 5.1 Layout (SOURCE)

Magic `HQ30UQ4\0`, group 64, 32 code bytes + 1 f16 absmax / group.
Even local → low nibble, odd → high. `q = nibble - 8` ∈ [−8, 7].
`qwen_uniform_q4.metal:4-11`; `uniform_q4.rs:3-6,15-18`.

Wbytes/row = `ceil(K/64)*34`. K=5120 → 80×34 = **2720**. K=6144 → **3264**.
lm_head and embed in this pack are K=5120 (MEASURED inodes shared with
`uniform-q4-v1`, `g1-sub15-native-gap.md:189-191`).

G0 scale plane `800,686,080` = `2/34` of `13,611,663,360`
(`g1-traffic-anatomy.md:162-163`). This pack's HQ30UQ4 scale plane is
embed+lm_head only: `2 * 248320 * 80 * 2 = 79,462,400` MEASURED-from-geometry.
Embed table is residency, not token GEMV traffic (one gathered row = 2720 B).

### 5.2 Per-thread decode, production geo_tpr64_tg128 (already K-complete)

SOURCE `qwen_uniform_q4.metal:166-221`.

```
lane = split*32 + simd_lane;                 // 0..63
row  = group_id*2 + team;                    // 2 rows / TG
for (col = lane*8; col < K; col += 512) {
    group = col / 64;  local = col % 64;
    s = float(scales[row*gpr + group]);                      // 2 B
    packed = *(uint*)(codes + (row*gpr+group)*32 + local/2); // 4 B
    // unpack8: 8 nibbles from that uint, same scale
    for (i = 0; i < 4; ++i) {
        byte = (packed >> 8*i) & 0xff;
        acc += float(int(byte & 0xf) - 8) * s * x[col+2*i];
        acc += float(int(byte >> 4)  - 8) * s * x[col+2*i+1];
    }
}
acc = simd_sum(acc);
if (split==0 && lane==0) y[row] = red[0]+red[1];
```

Register-only. No table. 32 contiguous aligned uints per simdgroup per
step (word law already held, `g1-direct-gemv-geometry.md:184-201`).

K=5120 → 10 steps; K=6144 → 12. Both legal. lm_head is the 10-step case.

| K | B/thread (4+2+32)×steps | FMAs | AI_thread | W/row | AI Rtg=2 | AI Rtg=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 5120 | **380** | 80 | 0.421 | 2720 | 0.790 | 3.044 |
| 6144 | **456** | 96 | 0.421 | 3264 | 0.790 | 3.045 |

### 5.3 Embed (not a GEMV)

`qwen_uniform_q4_embedding_lookup` `:589-602`. Grid `(5120,)`, TG 256.
Thread `id` decodes one hidden dim of one row: 1 nibble + 1 half.
Must-move 2720 B. Do not retarget this to tpr64.

### 5.4 Ideal mapping

Already the production genome. Do not replace lm_head's `geo_tpr64_tg128`
with tg256. Do not invent a codebook. C1 (32 rows/TG, staged X) is a
measurement for ≤2 bpw organs, not for this 4.25 BPW table
(`g1-direct-gemv-geometry.md:583-591`: at 4 bpw C1 is a near-tie).

---

## 6. Token-time traffic of a native consume (DERIVED)

After C1–C3 of the native-gap lane, unique GEMV bytes/token (CSR expanded,
embed table excluded):

| class | bytes |
|---|---:|
| gate HGRAVB01 | 802_160_640 |
| up HGRAVR02 binary+CSR | 1_277_218_496 |
| attn HGRAVR02 binary+CSR | 1_620_327_568 |
| down HGRAVS01 factors | 93_716_480 |
| lm_head HQ30UQ4 | 675_430_400 |
| **token GEMV** | **4_468_853_584** |

vs G0 Q4 `13_611_663_360` → ratio **0.3283**. Not a TOKEN_NS claim.
Packed on-disk rice is smaller than this because CSR u32 > rice bitstream.

Token scale-plane (excl. embed table) DERIVED:
`89128960 + 89128960 + 113090560 + 7208960 + 39731200 = 338_288_640` B
(7.57% of the 4.47 GB GEMV). G0's 800.7 MB was 5.88% of a 4.25 BPW
stream. Absolute scale bytes drop because most organs left Q4.

---

## 7. Dispatch shape of a native mixed-sub15 token (DERIVED)

G0: 964 dispatches, 401 GEMVs, 1 CB, 1 wait. SOURCE schedule + ledger.

This pack's attention is **unfused** (304 rice names, no `in_proj_qkvz` /
`in_proj_ba`). `encode_deltanet_mixed` takes `encode_split_deltanet_projections`
(`:2979-2995`): 4 GEMVs + 2 fuse kernels vs G0's 2 fused GEMVs.

| delta vs G0 964 | count | why |
|---|---:|---|
| split DN mixer | +192 | +4 / layer × 48 (`:13` vs `:9` mixer ops) |
| HGRAVS two-stage | +64 | `dispatch_hgravs` is 2 factor launches |
| **native fused-ON** | **1220** | 1 CB still |
| fuse-OFF residual split | +368 | +1 CSR apply × (64 up + 304 attn) → 1588. Do not. |
| if qkvz/ba later fused | 1028 | inventory's mixed-2p0-style number |

Not measured (no `step`). Cheapest seal: one
`enable_structural_kernel_trace()` after native load. GPU lane.

---

## 8. Specialization ranking

Predicted `bytes/out` is unique W + X at the named Rtg + 4-byte y, unless
noted. Predicted dispatch is vs the 1220 DERIVED native graph. Neither is
TOKEN_NS.

### R1. K-complete tile — avoids the multi-pass bind

**What.** Bind `*_simd_bytes` (or add the uniform8 `tile += 2048` loop, or
the tpr64 × W=8 walk of §2.3) for every HGRAVB01 / HGRAVR02 GEMV.
One dispatch walks all of K. No host rebind of 2048-col slices.
HGRAVS and HQ30UQ4 already K-complete; leave them.

**Does not share** the megakernel kill (multi-TG, packed, no expand).
**Does not share** the 460 µs codebook kill (no table).

| vs | bytes/out (binary K=5120, Rtg=2) | dispatch |
|---|---|---|
| HEAD tg256 (wrong) | 288 W + 10240 X = **10964 addressed of a 40% row**; y is numerically wrong | 0 |
| HEAD made correct by 3× rebind | 720 + 3×10240 = **31444** (X thrice) | **−864** (432 incomplete GEMVs × 2 extra passes) |
| R1 tpr64 W=8 | 720 + 10240 = **10964**, full K | **0** vs HEAD count |
| R1 simd_bytes Rtg=8 | 720 + 2560 = **3284** | 0 |

432 = 64 gate + 64 up + 304 rice. All have K∈{5120,6144}. DERIVED.

K=6144 + a naive 2048 loop is 3 exact tiles (legal). K=5120 + that loop
idles half the TG on the last 1024 cols. Prefer TILE=256/512.

**Must** before any native generate is treated as a codec verdict
(`g1-sub15-native-gap.md` C3). IMPLEMENT_READY, 20–40 host lines or
15×2 shader lines. No new family.

### R2. Fold the scale fetch into the packed stream

**What.** `TileHdr { half scale; uint words[TPR]; }` as already drawn in
`g1-direct-gemv-geometry.md:300-306`. One device pointer per body instead
of `set_buffer(signs); set_buffer(scales)`.

Scales are independent information (mean-abs / absmax). Folding does
**not** delete them. Unique bytes/out **unchanged**.

Dispatch Δ = **0**. Binds/GEMV −1 (binary today 4 buffers; residual 7;
Q4 4; factor 4 — SOURCE `encode_binary_args` `:1244-1258` etc.).

May cut dual-stream address tax. Catalog-vs-single-address is a MEASURED
24% topology loss on the Q4 genome (699.57 → 530.65 GB/s,
`g1-roof-falsification.md` R18/R20). That receipt is **not** this layout.
Recovering any of it here is ESTIMATED and unmeasured. Do not book it.

Does not share either recorded negative.

Requires a packer emit of a new physical layout. Not a 20-line bind flip.
Do this after R1, on a write-enabled pack lane, only if a serialized
addr_probe on R1 still shows a second-stream tax.

### R3. In-register reconstruction that removes the scale-plane read

**Kernel-only reading of the sentence:** already true. Every codec in this
pack reconstructs `w` in registers. `binary_byte_dot` holds `s` for 8 FMAs.
`unpack8` holds `s` for 8 FMAs. simd3 *issues* 8 scale loads that alias to
one half (CSE is the compiler's problem; the unique bytes are one).

You **cannot** reconstruct mean-abs or group-absmax from the bits alone.
Removing the plane is a **pack** change (unit scale, or one tensor-level
scale, or column scales that could pre-multiply X). Per-row-group scales
cannot be folded into X.

Predicted bytes/out if the plane is deleted and quality is ignored
(ESTIMATED pack, DERIVED arithmetic):

| organ / launch | with scale | no scale | Δ |
|---|---:|---:|---:|
| binary K=5120 Rtg=2 | 10964 | 10884 | **−80** |
| binary K=5120 Rtg=32 | 1364 | 1284 | **−80** |
| Q4 K=5120 Rtg=2 | 12964 | 12804 | **−160** |
| HGRAVS down W/row | 286 | 264 | **−22** |

At incumbent Rtg=2 the win is noise against 10 KB of X. It becomes a real
fraction only after C1 X-reuse. Do not build a new kernel whose only job
is "skip the scale load" on C0.

Does not share either recorded negative. Quality of a no-scale binary is
a different codec and is out of this lane.

### R4. Fused multi-codec kernel that walks a whole layer

**This is the megakernel.** `qwen3b_megakernel_nlayer`: Grid=(256,1,1),
one TG, `for li in 0..n_layers mk_layer_forward` (`megakernel_qwen3b.metal:1093-1133`;
dispatch `megakernel.rs:419-422`). MEASURED 4.4× slower per layer than
production TCB on M3 Pro Qwen-3B, two strikes: expand-to-f16, and one TG
that cannot saturate the bus (`g1-fusion-persistent.md` §1.1, commit
`fe0fb94c5`).

A Qwen3.8 layer walk of HGRAVB01 + HGRAVR02 + HGRAVS01 + HQ30UQ4 through
one/few TGs repeats **both** strikes in a worse geometry (17408-row gate,
17408-col down). ESTIMATED: the fusion paper's ~4.4× on a 33.9 ms GPU
token → ~150 ms. Not run (forbidden, and Type-1).

`gk_family.metal:3-5` already forbids a codec switch inside the FMA.

| | bytes/out | dispatch |
|---|---|---|
| predicted if occupancy preserved (it will not be) | W unchanged; mixer X 20 KB ×1 vs ×5 | 1220 → ~67 |
| predicted under the shipped megakernel mechanism | DRAM unsaturated; not a bytes/out win | 1220 → 1 |

**KILLS.** REOPEN_IF: packed in-register consume **and** multi-TG occupancy
≥ `geo_tpr64_tg128` on the 17408×5120 gate **and** a complete-token A/B
that does not regress the addressing bucket. That is new science, not
`MegakernelRunner`.

Legal cousin, **not** R4: fuse **gate + up only**. Same M, same K=5120,
shared X, codecs that share a binary body. One tpr64 (or C1) walk, two
sign streams, CSR tail on up, write two vectors or SwiGLU. Dispatch **−64**
(and −64 more if silu folds). bytes/out for the pair at Rtg=2: W=1862 +
X=10240 + y=8 = 12110 / 2 = **6055** vs 10964+11386 sequential. Does not
share the 4.4× mechanism iff the TG map stays multi-TG. Expand-then-GEMV
of that pair **does** share it. `qwen_direct_packed_gate_up_pair.metal`
is one-thread-per-row and is not this candidate.

---

## 9. KILLS and REOPEN_IF

| ID | statement | REOPEN_IF |
|---|---|---|
| K-MK | Whole-layer / persistent 1-TG multi-codec walk | packed + multi-TG ≥ geo_tpr64 + complete-token not-slower |
| K-PQ | Direct codebook-lookup GEMV (PQ/RVQ/gravity-VQ) | a new kernel ≤ 0.5 ms combined gate/up on real geometry **and** quality ≥ 0.99. Shipping `gravity_pq_matvec` already lost at 460.041 µs / 672.625 µs (`TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json`; `g1-vector-quantization.md` §9.2) |
| K-RICE | Rice bitstream as the GEMV stream | a simdgroup can load exclusive aligned Rice words and reconstruct with shift/mask only (not believed possible) |
| K-EXPAND | Expand packed → float/Q4 → generic GEMV | complete-token wall with expand < packed-direct wall on **this** artifact |
| K-2STAGE512 | `q80_hgravs01_two_stage_matvec` on Q38 | K-cap raised past 17408 **and** C2 occupancy on M=160 |
| K-TG256 | Shipping `*_tg256` as the Q38 binary/rice bind | never; it drops K. The *name* may stay if a 2048-tile loop is added |
| K-C0-1BIT | tpr64 TILE=2048 (uint 1-bit) as the 1-bit launch | TILE remade to 512 via exclusive bytes (R1) |

---

## 10. What this lane did not measure

- Any Metal load, dispatch, or timestamp.
- Bit-identity of `*_simd_bytes` vs `gk_binary_group_serial_row` on
  17408×5120 / 48×5120 (cheapest CPU check; not run).
- Token ns of a native 1.291 graph. PACK_REPORT `projected_tps=79.44` is
  retired (`g1-sub15-native-gap.md:464-466`).
- Whether u16 CSR or TileHdr recovers any of the 24% catalog topology loss.
- Coherence of the packed representation.

Cheapest next experiment (GPU lane, after native-gap C1–C3): one greedy
with `HAWKING_QWEN38_RECON_FUSE=0` (serial, K-complete, honest) **or**
with the R1 bind. Structural kernel trace to seal 1220 vs 1028.

---

## 11. Evidence

### 11.1 PACK_REPORT.json (RECEIPT, this process)

```
:23     complete_physical_bpw 1.2910781930062503
:27-68  per_tensor_class gate 802177344 / up 918036000 / down 93847197 /
        attn rice 1165098376 / embed+lm_head 675430440 each
:77-81  generate vehicle = reconstructed HQ30UQ4
```

### 11.2 Artifact headers (MEASURED, this process)

```
304/304 rice: magic HGRAVR02, index_mode=rice, value_bits=1, value_scale=rms,
              rice_k=5, group=128, outlier_sum=144756064
L0 in_proj_a header: shape [48,5120] outlier_count=4916 scale_bytes=3840
                     sign_bytes=30720 rice_bytes=4379 residual_bytes=615
mixed-2p0 catalog: 851 tensors, codec 0×64 1×64 2×64 3×659
L0 gate HGRAVB01 scale_bytes=1392640 sign_bytes=11141120
L0 up   HGRAVR02 outlier_count=1782580 rice_k=5
L0 down HGRAVS01 rank=160 bits=3 g=64
        left [5120,160] 332800  right [160,17408] 1131520
64 up outliers 114085120  rice_bytes 101576736
64 gate scale 89128960 sign 713031680
```

### 11.3 Kernels / binds (SOURCE)

```
q80_mixed_decode.metal:1-25,107-180,297-318,536-568,620-653,699-840,845-907
qwen_uniform_q4.metal:4-11,166-221,589-602
gk_family.metal:3-5,116-160
q80_mixed_decode.rs:13-41,149-241,383-506,615-628,1134-1332
qwen38_hybrid_decode.rs:37-45,264-268,704-709,1088-1148,1244-1454,1522-1562,2979-2995
decode_family.rs:16-20
megakernel_qwen3b.metal:1093-1133
megakernel.rs:419-422
qwen38_geometry.rs:20-52
```

### 11.4 Receipts / sibling G1 (RECEIPT, not re-run)

```
QWEN38_RECON_MEASURED.json
  gate binary_g128 tpr64 15416 ns excess 0
  gate rice csr_inregister 15125 ns excess 0
  down hgravs01_r160_q3 71458 ns excess 67851.7  (f32 7083)
TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json
  14336×4096 D=32 S=1 K=256 → 460.041 µs
  D=8 S=4 K=128 → 672.625 µs
g1-fusion-persistent.md §1.1  megakernel 4.4× (fe0fb94c5)
g1-roof-falsification.md R18/R20  699.57 vs 530.65 GB/s
g1-sub15-native-gap.md  C1–C3, 2048-col bind, no KERNEL-MISSING
g1-direct-gemv-geometry.md  AI formula, C0/C1/C2/C4/C5, K3/K4/K6
g1-kernel-inventory.md  964 / 401 / mixed +64 / split +192
g1-traffic-anatomy.md  G0 scale plane 800686080
g1-vector-quantization.md §9  codebook reject
```

### 11.5 Arithmetic identities used

```
bin_w(K)     = K/8 + 2*(K/128)                 # 720 @5120, 864 @6144
q4_w(K)      = 34*ceil(K/64)                   # 2720 @5120
AI(K,W,Rtg)  = 2K / (W + 4K/Rtg + 4)
hgravs_q3    = (M*R+R*K)*3/8 + 2*(ceil(M*R/64)+ceil(R*K/64))
             = 1464320 for (5120,17408,R=160)
6144/2048    = 3
5120/2048    = 2 remainder 1024
TILE∈{256,512,1024} divide both 5120 and 6144
token GEMV   = 4468853584   (CSR-expanded, no embed table)
dispatch     = 964 + 192 + 64 = 1220  (unfused attn, fuse-ON)
```

---

```
STATUS
IMPLEMENT_READY

CLAIMS
1. mixed-sub15-v1 GEMVs are only HGRAVB01, HGRAVR02, HGRAVS01 r160_b3, HQ30UQ4; every binary/rice GEMV has K∈{5120,6144}. Evidence: §1; PACK_REPORT.json:12-19; 304 rice headers; mixed-2p0 L0 MLP headers.
2. All four reconstruct in registers. None needs a codebook gather. Evidence: §2.2, §3.3, §4.2, §5.2; gravity_pq.metal is a different family.
3. Production binary/CSR tiles cover 2048 columns and drop 60% of K=5120 and 67% of K=6144. Tiling siblings already walk all K. Evidence: q80_mixed_decode.metal:722-728 vs :641-648 and :821-828; 2048∤5120, 2048∣6144.
4. Ideal binary/rice mapping for these K is tpr64 × exclusive-byte W=8 × TILE=512 (10 / 12 steps), CSR strided across 32 lanes, optional u16 col indices. Evidence: §2.3, §3.4; TILE divisibility §1.
5. HGRAVS is already K-complete as two simd3 launches and is occupancy-starved on the 160-row stage (71458 vs 7083 ns RECEIPT). two_stage_matvec is illegal (K-cap 512). Evidence: §4; QWEN38_RECON_MEASURED.json down hgravs01_r160_q3; metal:561-568.
6. HQ30UQ4 geo_tpr64 is already the right K=5120/6144 tile. Evidence: §5; metal:204 (col += 512).
7. Rank: R1 K-complete (must, Δdisp 0 or −864, no negative) > R2 fold-scale (same bytes, −1 bind) > R3 drop scale plane (pack change, −80 B/out at C0) > R4 whole-layer fused kernel (KILLS, megakernel). Evidence: §8.
8. Rice token-time is 1.791 BPW not 1.288, because CSR u32 (1.078 GB) replaces the rice bitstream. Evidence: §3.2 MEASURED outlier sums.
9. Direct codebook GEMV remains KILL at 460.041 µs. Evidence: TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json; g1-vector-quantization.md:348-354.

EVIDENCE
PACK_REPORT.json:23,27-68
this process: 304 rice headers + mixed-2p0 catalog walk (§11.2)
q80_mixed_decode.metal:297-318,620-728,743-907
qwen_uniform_q4.metal:166-221
qwen38_hybrid_decode.rs:704-709,1088-1148,1323-1454,2979-2995
q80_mixed_decode.rs:183-241,461-506,1174-1332
decode_family.rs:16-20
megakernel_qwen3b.metal:1093-1133
QWEN38_RECON_MEASURED.json gate/down variants
TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json (via g1-vector-quantization.md:344-354)
g1-sub15-native-gap.md, g1-direct-gemv-geometry.md, g1-fusion-persistent.md, g1-roof-falsification.md

CHANGES
workspace/superwave/g1/g1-sub15-kernel-specialization.md (this file only)

TESTS
test -s workspace/superwave/g1/g1-sub15-kernel-specialization.md
wc -l workspace/superwave/g1/g1-sub15-kernel-specialization.md
git status --porcelain

RISKS
R1 not landed ⇒ any native generate of this pack is a 40%/33% K-drop on every binary/rice GEMV and will be misread as a codec verdict. R4 reopen without the three conjuncts repeats fe0fb94c5. u16 CSR and TileHdr are pack changes; do not sneak them into a bind-only lane. 1220 dispatch is DERIVED, not traced. RECON_MEASURED ns are relative ALU, not GB/s.

UNRESOLVED
Native coherence of 1.291. Token ns of a K-complete consume. Whether simd_bytes ≡ serial on these shapes (CPU). Whether TileHdr or u16 CSR moves the 24% catalog topology loss. GPU lane.

NEXT
Native-gap C1–C3, then R1 bind (or fuse-OFF) on a write lane. Serialized GPU lane: one greedy + structural kernel trace. Do not write a layer megakernel. Do not add a codebook path.
```
