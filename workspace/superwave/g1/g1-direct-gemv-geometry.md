# G1 direct packed GEMV geometry (M3 Ultra, Qwen3.8)

Paper design. No GPU run in this lane. Every number is tagged
MEASURED / DERIVED / ESTIMATED. A roof is a measured regime, not
bytes ÷ datasheet.

This worktree's `workspace/superwave/g1/` contains only `README.md`
besides this file. Sibling G1 gravity-lane reports are not on disk
here. Representation families are the sealed gravity / ascent codec
inventory those lanes already publish (Qwen3.8 descent catalog,
Q38/Q80 mixed HGRAV* containers, residual-compact, PQ, strand/TQ,
GGML K-quants).

---

## 0. Labels

| tag | meaning |
|---|---|
| MEASURED | GPU timestamp or artifact byte count from a named receipt |
| DERIVED | exact arithmetic from source constants or from named MEASURED inputs |
| ESTIMATED | not computed from a sealed receipt; not used as a finding |

Datasheet `819 GB/s` is published peak, not a decode roof
(`crates/hawking-core/src/backend/honest_roof.rs:59`).
`411.51 GB/s` is the 512 MiB `unique_once` point and is a **refuted**
ceiling for this GEMV shape (`receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.md`).

A component organ time is not a token wall.
A discriminator that matches f32 time is not a streaming-bandwidth claim.

---

## 1. Machine and production genome

### 1.1 Hardware constants used by this paper

| quantity | value | tag | evidence |
|---|---|---|---|
| device | Apple M3 Ultra | MEASURED | `matvec-geometry-sweep.json` `device_name` |
| GPU cores (this box, launch math) | 60 | DERIVED | `qwen38_token_ns_ledger.rs:183` `gpu_cores: 60`. Not a hardware occupancy counter. |
| simdgroup / `thread_execution_width` | 32 | MEASURED | `matvec-occupancy-230x.json` `pipeline.*.thread_execution_width` |
| max threads / TG | 1024 | MEASURED | same receipt `max_total_threads_per_threadgroup` |
| unified memory | 96 GB | given | task contract; honest-roof MD restates 96 GiB |
| datasheet peak | 819 GB/s | published | `honest_roof.rs:59` — not a measured roof |

### 1.2 Qwen3.8 decode GEMV shapes

Authority: `crates/hawking-core/src/model/qwen38_geometry.rs`.

| organ | M × K | count / token | evidence |
|---|---|---|---|
| mlp gate, up | 17408 × 5120 | 64 each | `QWEN38_INTERMEDIATE`, `QWEN38_HIDDEN`, `QWEN38_LAYERS` |
| mlp down | 5120 × 17408 | 64 | same |
| DeltaNet qkvz | 16384 × 5120 | 48 | `QWEN38_QKVZ_ROWS` |
| DeltaNet ba | 96 × 5120 | 48 | `QWEN38_BA_ROWS` |
| DeltaNet out | 5120 × 6144 | 48 | `QWEN38_O_PROJ_COLS` |
| GQA q | 12288 × 5120 | 16 | `QWEN38_Q_PROJ_ROWS` |
| GQA k, v | 1024 × 5120 | 16 each | `QWEN38_KV_PROJ_ROWS` |
| GQA o | 5120 × 6144 | 16 | same o-proj |
| lm_head | 248320 × 5120 | 1 | `QWEN38_VOCAB` |

Dense decode. Every GEMV listed is read every token except the embed
table (one gathered row). Embed table is **not** GEMV traffic
(`qwen38_token_ns_ledger.rs:102`, `QWEN38_ACTIVE_BUDGET_MEASURED.json`).

K values that any tile must divide (or pad): 5120, 17408, 6144, 1024.
DERIVED: all four divide 256, 512, 1024. 640 divides 5120 only.
2048 divides none of {5120, 17408, 6144}.

### 1.3 Incumbent execution genome (Q4)

| item | value | tag | evidence |
|---|---|---|---|
| kernel | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` | MEASURED | `G024_QWEN38_TOKEN_NS.json` `kernel_runtime_genome` |
| launch | 64 threads/row, TG 128, 2 rows/TG, grid=`ceil(M/2)*128` | source | `qwen_uniform_q4.metal:181-221` |
| reduction | `simd_sum` per simdgroup, then 2-way TG add | source | same |
| pack | HQ30UQ4: group 64, 32 code bytes, even nibble low, `q = nibble-8`, f16 scale | source | `qwen_uniform_q4.metal:4-11`, `uniform_q4.rs:3-6` |
| complete BPW | 4.252735126866492 | MEASURED | `G024` `measurement.bpw`; `UNIFORM_Q4_V1_BPW` |
| GEMV payload | 13_611_663_360 B | DERIVED | `honest_roof.rs:46`; matches `theoretical_weight_bytes` Q4 census |
| production GPU median | 33_912_333 ns | MEASURED | `TOKEN_NS_QWEN38.json` `TOTAL_GPU_BUSY_NS` |
| production wall | 35_227_917 ns | MEASURED | same `TOTAL_TOKEN_NS` |
| weight_addressing | 21_293_102.5 ns, 60.44% of wall | MEASURED | `TOKEN_NS_QWEN38.json` component `weight_addressing` |
| decode-reconstruction | 1_808_227 ns, 5.13% of wall | MEASURED | same, method `(decode_probe − addr_probe)/full` |
| GEMV dispatches | 401 | MEASURED | addressing component `dispatches` |
| CB / token | 1 CB / 964 dispatches | MEASURED | `G024` `kernel_runtime_genome` |

Q4 bytes/row DERIVED (`q4_matrix_bytes` in `qwen38_token_ns_ledger.rs:51-54`):
`groups = ceil(K/64)`, `bytes/row = groups * 34`.

| organ | Q4 bytes / matrix | evidence |
|---|---|---|
| gate, up, down | 47_349_760 | DERIVED; 17408×80×34 and 5120×272×34 |
| qkvz | 44_564_480 | DERIVED |
| ba | 261_120 | DERIVED |
| o / dn_out | 16_711_680 | DERIVED |
| gqa q | 33_423_360 | DERIVED |
| gqa k/v | 2_785_280 | DERIVED |
| lm_head | 675_430_400 | DERIVED; equals `MANIFEST_LM_HEAD_BYTES` |
| **token GEMV sum** | **13_611_663_360** | DERIVED; `honest_roof.rs:46` |

### 1.4 Measured bandwidth regimes (this box, this genome)

| regime | GB/s | tag | what it is |
|---|---|---|---|
| Q4 geo_tpr64 **full** at 13.612 GB working set | 666.7 | MEASURED | honest-roof MD table, single GEMV |
| Q4 geo_tpr64 **addr** at 13.612 GB | 699.6 | MEASURED | same |
| Q4 geo_tpr64 **decode** at 13.612 GB | 683.8 | MEASURED | same |
| sealed addressing / 13.611 GB | 639.25 | DERIVED | `13_611_663_360 / 21_293_102.5` (`TOKEN_NS` + `honest_roof.rs:46-50`) |
| 401-organ catalog full | 505.81 | MEASURED | honest-roof MD |
| `unique_once` at 13.6 GB | 375.7 | MEASURED | honest-roof MD |
| sequential float4 control (geometry sweep) | 494.78 | MEASURED | `matvec-geometry-sweep.json` `control.gbps`; DIRTY_ENGINEERING |
| sequential control (occupancy lane) | 550.14 | MEASURED | `matvec-occupancy-230x.json` `control_sequential_gbps`; DIRTY |
| conflict control (occupancy lane) | 652.05 | MEASURED | same `control_conflict_gbps`; DIRTY |
| Q4 serial one-thread-per-row, Q80 512×2048 | 2.62 | MEASURED | occupancy `serial_gbps.q80_q4_gate` |
| Q4 tpr64 on that same small organ | 83.03 named / 70.74 generated | MEASURED | geometry-sweep `shipped.q4_named_winner_gbps` / `top_survivors.q4[0]` |
| binary serial, same small organ | 2.44 | MEASURED | geometry-sweep `shipped.binary_serial_gbps` |
| binary generated winner, same small organ | 16.46 | MEASURED | `top_survivors.binary[0]` |

Honest-roof verdict: on **this** genome, `weight_addressing` is
bandwidth-saturated at 639–700 GB/s of unique Q4 grouped traffic, 91.4%
of the single-GEMV addr roof. Decode+FMA tax on a 13.6 GB Q4 GEMV is
4.7% (`HONEST_ROOF_WEIGHT_ADDRESSING.md`). That tax is a Q4-tpr64
measurement, not a proof that every codec is free.

### 1.5 Reconstruction-at-tpr64 (relative, not a byte roof)

`QWEN38_RECON_MEASURED.json`, real BF16 hidden, production launch
(64 thr/row, TG 128, 2 rows/TG), GPUEnd−GPUStart.

Gate 17408×5120 median ns:

| variant | median_ns | recon_excess_ns |
|---|---:|---:|
| f32_tpr64 | 15125 | 0 |
| prod_q4_nibble_g64 | 15500 | 0 |
| uniform q4/q3/q2 tpr64 | 15208 / 15125 / 15374 | 0 |
| binary_g128 tpr64 | 15416 | 0 |
| ternary_t0.7_g128 | 15541 | 0 |
| additive_q2q2_g64 | 15125 | 0 |
| rice CSR in-register | 15125 | 0 |
| hadamard_q2_g128 | 17333 | 0 |
| same codecs at tg256 | ~26541 | 0 |

Down 5120×17408: f32 7083 ns; HGRAVS r160 q3 **71458 ns**,
`recon_excess_ns = 67851.7`. Only HGRAVS is not free at this launch.

These times cannot be converted to GB/s. Gate Q4 payload 47.35 MB in
15.1 µs would be ~3130 GB/s, above peak — the discriminator is
launch-dominated and/or cache-resident, not a unique-once stream.
Use them only as **relative** ALU/launch cost at fixed geometry.

`QWEN38_RECONSTRUCTION_IS_FREE.json` consequence: under tpr64, codec
choice among the cheap in-register family is not constrained by
reconstruction time. HGRAVS is the exception (two-stage occupancy).

---

## 2. Binding constraints

A candidate is a direct packed GEMV only if all five hold:

1. Packed bits stay packed in DRAM. No dense `M×K` temporary, no
   expand-to-float / expand-to-Q4 then generic GEMV
   (`q80_mixed_decode.rs:23`, `qwen38_hybrid_decode.rs:5-7`).
2. A simdgroup issues 32 contiguous aligned word loads. Lane `i` owns
   word `i`. No `simd_shuffle`, no 8 lanes reading the same byte and
   taking different bits.
3. Reconstruction is shift / mask / `int→float` / FMA in registers.
   No codebook gather. No Acklam / LUT.
4. Partial sums reduce with `simd_sum` (and a 2-way TG add if split-K
   uses two simdgroups). One team owns the output element; no atomics
   on the large-M path.
5. Tile width `TPR * W` divides every production K, or the packer
   states an explicit K-pad and the extra bytes are billed.

Constraint 2 is the packing law. Constraint 5 is why 3-bit and uint4
low-bit tiles are not free choices.

---

## 3. What the incumbent pack already is

### 3.1 Q4 nibble + tpr64 already satisfies the word law

`geo_tpr64_tg128` (`qwen_uniform_q4.metal:204-210`):

```
lane_in_row = split*32 + simd_lane          // 0..63
col         = lane_in_row * 8               // step 512
packed      = *(uint*)(codes + group*32 + local/2)
```

First simdgroup (split=0) issues cols 0,8,…,248 = groups 0–3.
Those four groups are 128 contiguous code bytes = 32 uints, one per
lane. Second simdgroup covers groups 4–7. Then both stride 512
columns = 256 code bytes.

So HQ30UQ4 + tpr64 + W=8 is already “32 contiguous aligned uints,
no shuffle”. The geometry-sweep winner is this launch, not a
different pack (`matvec-geometry-sweep.json` named winner kernel).

What it does **not** do: reuse X. `rows_per_tg = 2`, so each TG
re-reads `4*K` bytes of X for only two output rows.

### 3.2 Current 1-bit / 2-bit / 3-bit on-disk streams

| family | on-disk | word law? |
|---|---|---|
| `HGRAVB01` / `binary_g128` | LSB-first signs, f16 mean-abs / group 128 | yes at W=8 (one exclusive byte / lane), **byte** not uint |
| `HQ30UQ2` / `HQ30UQ3` | LSB bitstream, `bit0 = element * bits` (`uniform_qn.rs:149-152`, `qwen_uniform_qn.metal:15-28`) | **no** — 3-bit bit0 is not word-aligned; adjacent tpr64 lanes are 24 bits apart |
| HGRAVS factor body | same unsigned LSB, bits=3, group 64 | **no** (same extract) |
| `HGRAVU01` bits=8 | one byte / weight | yes, wasteful |
| ternary 2-bit | LSB extract, codes {0,+s,−s} (`recon.metal` `disc_ternary_tpr64`) | no (2-bit unaligned) unless remapped to exclusive uints of 16 codes |
| additive q2q2 | two LSB q2 streams, levels `(c-1.5)*scale` (`disc_additive_tpr64`) | no until remapped; **no table** |
| Hadamard q2 | `WH(x)` then uniform q2 on weights (`recon_disc` host: `disc_walsh_hadamard_x+disc_uniform_bits_tpr64`) | GEMV side = q2; WH is an X preprocess |
| rice / `HGRAVR02` | binary body + Rice deltas + residual signs | binary half yes; Rice half is variable-length — **not a GEMV tile** |
| gravity PQ | per-chunk codebook gather (`gravity_pq.metal:1-11`) | **table fetch** |
| strand bitslice | TG LUT or integer Acklam (`strand_bitslice.metal:35-48, 176-192`) | LUT = table; Acklam = not a shift/mask |
| Q4_K / Q6_K | 256-wide superblock, 6-bit scales+mins, nibble split (`quant.rs:602-639`) | not lane-exclusive aligned words |
| HGRAVS as dense W | forbidden | two-stage only (`q80_mixed_decode.rs:23`) |

### 3.3 X-reuse is the low-BPW limiter

Per output element, FLOPs = `2K` (mul+add).

```
Wbytes     = K * bits / 8 + 2 * ceil(K / G)
x_opt      = 4K / M          # X cached across the whole organ
x_tg       = 4K / rows_per_tg
bytes/out  = Wbytes + x_* + 4
AI         = 2K / bytes/out
```

DERIVED for gate/up (M=17408, K=5120), G as named:

| pack | Wbytes/out | AI_opt | AI at 2 rows/TG | AI at 32 rows/TG |
|---|---:|---:|---:|---:|
| Q4 G=64 | 2720 | 3.758 | 0.790 | 3.044 |
| q3 G=64 | 2080 | 4.911 | 0.831 | 3.759 |
| q2 G=64 | 1440 | 7.086 | 0.876 | 4.914 |
| ternary 2b G=128 | 1360 | 7.501 | 0.882 | 5.110 |
| binary G=128 | 720 | 14.121 | 0.934 | 7.507 |
| binary G=256 | 680 | 14.945 | 0.937 | 7.734 |

At 2 rows/TG, dropping 4→1 bit moves AI from 0.79 to 0.93. X is
`4*5120/2 = 10240` B versus 1440–5440 B of weights. **The incumbent
launch throws away the density win.** That is the geometric fact this
paper exists to name.

Down (K=17408): X is 69.6 KB and does not fit in a 32 KB TG. Must
stage X in ≤1024-float (4 KB) tiles. ESTIMATED TG capacity 32 KB
(Apple default; not counter-sampled here). 1024-float tile is a
choice that fits with registers and reduction scratch.

---

## 4. Pack law (one physical contract, several bit widths)

Name: **lane-word**.

```
TPR ∈ {32, 64}           # threads cooperating on one row
W   = 32 / bits          # weights packed in one uint, bits ∈ {1,2,4}
TILE = TPR * W           # columns consumed per step
```

| bits | W (uint) | TILE tpr32 | TILE tpr64 | divides {5120,17408,6144,1024}? |
|---|---:|---:|---:|---|
| 1 | 32 | 1024 | 2048 | tpr32 yes; tpr64 **no** |
| 2 | 16 | 512 | 1024 | both yes |
| 4 | 8 | 256 | 512 | both yes |

uint4 variant (`W4 = 128/bits`): TILE(tpr32,4-bit)=1024 yes; TILE(tpr32,2-bit)=2048 no;
TILE(tpr32,1-bit)=4096 no. So uint4 is only legal at 4-bit on these K.

3-bit is not in the table. See C-3.

### 4.1 Byte layout, one row

Little-endian. Codes for row `r` are a contiguous array of `K/W` uints:

```
for t in 0 .. K/TILE:
  for lane in 0 .. TPR:
    word[t * TPR + lane] = pack_lsb(W weights starting at
                                    t*TILE + lane*W, `bits` each)
```

`pack_lsb`: weight `j` occupies bits `[j*bits, (j+1)*bits)`.
Offset-binary: `code = q + bound`, `bound = (1<<(bits-1))-1` for
uniform; binary uses `bits=1` as `1 → +s`, `0 → −s`.

Scales: `half scale[ceil(K/G)]`, `G` divides `TILE`. Recommended
`G = TILE` (one scale broadcast per step) or `G = 64` if the
representation lane refuses to grow the group. One scale per TILE is
one aligned half; all 32 lanes may load the same address (broadcast
load, not a shuffle).

Optional fused tile header, 16-byte aligned:

```
struct TileHdr { half scale; half pad; uint words[TPR]; }  // TPR=32 → 132 B, pad to 144
```

Billed. Prefer a separate scale buffer matching today's
`device const half* scales` so existing upload paths stay.

### 4.2 Per-thread instruction sequence (uint, bits in {1,2,4})

One step, one row, lane `L`, tile `t`:

```
uint word  = codes[row_words + t*TPR + L];          // aligned 4B
half s     = scales[row_groups + (t*TILE)/G];       // broadcast
uint col0  = t*TILE + L*W;
float acc  = 0;
#pragma unroll
for (uint j = 0; j < W; ++j) {
    uint code = (word >> (j*bits)) & ((1u << bits) - 1u);
    int  q    = int(code) - bound;                  // binary: q = code ? 1 : -1
    acc = fma(float(q) * float(s), x[col0 + j], acc);
}
```

Then `acc = simd_sum(acc)`. If `TPR=64`, two simdgroups write
`red[simd_id]` and lane 0 of split 0 stores `red[0]+red[1]`
(identical to `qwen_uniform_q4.metal:213-220`).

No table. No shuffle. 32 (or 64) exclusive words.

Binary may skip `float(q)` and use `(word & (1u<<j)) ? s : -s`.

### 4.3 3-bit (C-3)

| option | packing | waste | TILE tpr32 | legal K? |
|---|---|---|---|---|
| C-3A | 10 codes / uint (30 bits), 2 pad | 6.25% | 320 | 5120 yes; 17408/6144/1024 **no** |
| C-3B | 10 codes / uint, K padded to multiple of 640 (tpr64) or 320 (tpr32) | 6.25% + pad | 320 / 640 | pad 17408→17600 (tpr32) or 17920 (tpr64) |
| C-3C | keep LSB stream, `wide_extract` | 0 | n/a | all K; **fails word law** |
| C-3D | store as 4-bit | 25% | 256 / 512 | all K; density = Q4 |

DERIVED pad cost C-3B tpr32 on down: pad K 17408→17600 = 192 weights
× 3/8 = 72 B/row × 5120 rows = 368_640 B (~1.0% of a q3 down payload
of 5120×7072 = 36.2 MB). Acceptable if the representation lane wants
true 3-bit.

C-3A without pad is illegal on down/o/gqa-kv. Do not ship it.

---

## 5. Candidates

All candidates consume lane-word (or C-3B/C-3C as noted).
`rows_per_simdgroup` means independent accumulators, same column walk.

### C0 — incumbent tpr64 / 2 rows / TG 128  (control)

Exactly `geo_tpr64_tg128`.

| | |
|---|---|
| threads / row | 64 = 2 simdgroups split-K |
| rows / TG | 2 (1 per 64-thread team) |
| TG | 128 |
| W | 8 (4-bit) or 16 (2-bit) or — illegal 32 (1-bit, TILE=2048) |
| weights / thread / step | W |
| weights / simdgroup / step | 32*W |
| weights / TG / step | 2 rows × 64 × W |
| steps, K=5120, 4-bit | 10 |
| steps, K=17408, 4-bit | 34 |
| reduce | simd_sum + TG add of 2 |
| X | device; no stage |
| launch M=17408 | 8704 TG, 145 TG/core (DERIVED, `geo_tpr64_occupancy`) |
| launch M=96 | 48 TG, 0.80 TG/core |

Bytes/out and AI: table in §3.3, column “2 rows/TG”.

This is the only candidate whose **full-organ** bandwidth on this box
is MEASURED (Q4, 13.6 GB working set, 666.7 GB/s full). It is the
control, not the low-BPW winner.

1-bit cannot use C0 without TILE=2048 (illegal) or dropping to W=8
byte loads (word law relaxed to exclusive bytes; legal, weaker).

### C1 — tpr32 / 32 rows / TG 256 / staged-X tiles   (large-M)

The density-preserving launch.

| | |
|---|---|
| threads / row | 32 = 1 simdgroup |
| rows / simdgroup | 4 independent accs |
| simdgroups / TG | 8 |
| rows / TG | 32 |
| TG | 256 |
| W | 32/bits (uint) |
| weights / thread / step | W × 4 rows |
| X | stage 1024 floats (4 KB) into TG, walk tiles |
| reduce | `simd_sum` × 4, lane 0 writes 4 outputs; no TG reduce |
| launch M=17408 | 544 TG, 9.07 TG/core |
| launch M=5120 | 160 TG, 2.67 TG/core |
| launch M=1024 | 32 TG, 0.53 TG/core — **do not use C1** |
| launch M=96 | 3 TG, 0.05 TG/core — **do not use C1** |

Per-step sequence, one simdgroup, 4 rows:

```
// once per 1024-col X tile (all 256 threads cooperate)
x_tg[lid] = input[tile + lid];  // 1024 floats, 4 waves
threadgroup_barrier();

for (t in tile .. tile+1024/TILE):
    col0 = t*TILE + lane*W;
    s[r] = scales[row[r], t];          // 4 halfs, often the same group
    w[r] = codes[row[r], t, lane];     // 4 exclusive uints
    xj[0..W) from x_tg[col0 % 1024]
    for r in 0..4:
        acc[r] += unpack_dot(w[r], s[r], xj)
```

K=5120 = 5 × 1024: five X-stage barriers, then 5 (4-bit) or 10 (2-bit)
or 5 (1-bit, TILE=1024) code steps per tile.

DERIVED bytes/out gate 1-bit G=128: W=720, x_tg32=640, y=4 → 1364 B/out,
AI = 10240/1364 = 7.51 (vs C0 0.93). Same pack, 8.0× more arithmetic
intensity from X reuse alone.

Predicted time is **not** claimed. Falsify by measuring C1 vs C0 on
the same packed buffer: if C1 is not faster at 1-bit than C0 on
gate/up/down/qkvz/lm_head, the X-reuse thesis is false.

Occupancy risk is real on down (2.67 TG/core). If C1 down regresses
versus C0, drop to C1b: 8 rows/TG (R=1, 8 simdgroups), 640 TG on
down = 10.7 TG/core, x_tg = 4K/8 = 8704 B, AI_1bit_gate = 10240/(720+8704+4)=1.09
— still better than C0's 0.93, worse than C1's 7.51. Measure both.

### C2 — split-K many TGs / row   (tiny-M: ba, HGRAVS R, optionally gqa kv)

For M such that C0 launches < ~2 TG/core (ba=0.80, HGRAVS stage-1
R@x at 160 rows = 80 TG = 1.33 TG/core under C0).

| | |
|---|---|
| threads / row / TG | 128 = 4 simdgroups split-K |
| rows / TG | 1 |
| TGs / row | 4 (or 8) persistent split over K |
| TG | 128 |
| reduce | per-TG `simd_sum` + 4-way TG add into a `float mid[TGs_per_row]` **or** a second 1-TG reduce pass. No atomics. |
| X | stage the K-slice this TG owns (17408/4 = 4352 floats = 17.4 KB; 8-way split = 8.7 KB) |

ba 96×5120, 4 TGs/row → 384 TG = 6.4 TG/core.

HGRAVS `R` is 160×K. Same launch. Do not use
`q80_hgravs01_two_stage_matvec` on Q38: it caps `right_cols` at 512
(`q80_mixed_decode.metal:562-568`) and is a Q80-shaped kernel.

### C3 — fused gate+up, C1 launch

Gate and up share X and the same M,K. One dispatch, two code streams,
two acc sets (or 2 of the 4 row slots become the pair).

DERIVED gate+up pair, 1-bit, C1: Wbytes = 2×720, x = 640, y = 8 →
2088 B for 2 outputs, AI = 20480/2088 = 9.81 vs two C1 launches at
7.51 each plus a second dispatch.

Cuts 64 of 401 GEMV dispatches. Existing
`qwen_direct_packed_gate_up_pair_candidate` is one-thread-per-row
(`qwen_direct_packed_gate_up_pair.metal:51-75`) and is not this
candidate.

### C4 — two-stage HGRAVS native (not a dense GEMV)

`y = L @ (R @ x)`, L `[M,160]` q3, R `[160,K]` q3, group 64
(`q80_mixed_decode.rs:16-21`, Q38 mixed rank 160 /
`qwen38_hybrid_decode.rs:37-39`).

DERIVED bytes, down or gate (the two factors swap):

| | bytes |
|---|---:|
| L q3 | 332_800 |
| R q3 | 1_131_520 |
| total | 1_464_320 |
| dense Q4 same organ | 47_349_760 |
| byte ratio | 32.34× |

DERIVED FLOPs: `2*160*K + 2*M*160 = 7_208_960` vs dense `2*M*K = 178_257_920`
(ratio 0.0404).

Bytes/out: down 286 vs Q4 9248; gate 84 vs 2720.

Stage 1 (160 × K): C2 (tiny-M split-K). Pack R with C-3B (3-bit
lane-word) or C-3C (LSB, measured-free extract).
Stage 2 (M × 160): K=160 is one or two tpr32 steps. C0 or even
serial-enough (5120 threads) is fine. Stage mid[160] in TG (640 B).

MEASURED relative: `hgravs01_r160_q3` down = 71458 ns vs Q4 7500 ns
(~9.5× slower) using two `disc_uniform_bits_tpr64` launches
(`QWEN38_RECON_MEASURED.json`, `recon_disc` host lines 702-754).
That is C0 applied to a 160-row organ — the occupancy failure C2
exists to fix.

Byte floor ESTIMATED only as a falsification target, not a time
claim: 1_464_320 B at the **measured Q4-full 666.7 GB/s regime**
would be 2.20 µs. The 71 µs discriminator is 32× above that
conditional. If a C2 implementation does not approach the Q4 organ
time (7.5 µs class) on down, HGRAVS stays an occupancy tax and the
representation lane should not put rank-160 on the token path.

Do not expand L@R to dense W. That path is forbidden and would
move 32× more bytes.

### C5 — binary GEMV + CSR residual sidecar  (HGRAVR02)

Not one GEMV.

1. C1/C0 binary on the `HGRAVB01` body.
2. Second kernel: CSR residual, one entry = (col, sign×rms).
   Bind-time `expand_rice_indices` already exists
   (`q80_mixed_decode.rs:462`, `qwen38_hybrid_decode.rs:20`).
   Token path must not Rice-decode.

`disc_binary_csr_tpr64` was time-matched to f32 at tpr64
(`QWEN38_RECON_MEASURED` `rice_q1_rms_2pct/csr_inregister`).
That is relative ALU, not a 2% scatter bandwidth proof.

Rice-as-the-only-stream: KILL (§7).

### C6 — Hadamard as X-preprocess + C1 q2

Runtime is `y = W_wh @ WH(x)`, W_wh stored as uniform q2.
WH(x) is one K-vector butterfly, not a per-weight table.

GEMV = C1 2-bit. Extra cost is one WH of length 5120 or 17408.
MEASURED relative: hadamard 17333 vs f32 15125 ns on gate (~15%).
Group size 128 implies WH-128 tiles of X (fits the 1024-float stage
if 128 divides the tile; 1024/128=8).

If WH is implemented with shuffles inside a 128-wide group, that
violates constraint 2 **inside the preprocess**, not inside the GEMV.
Require a per-thread or TG-memory WH on 128 floats, no simd_shuffle
across packed-weight lanes.

---

## 6. Ranking

Rank is **predicted arithmetic intensity at legal occupancy**, plus
whether the family can hit that intensity without a table or expand.
Time ranks are not claimed.

### Large-M organs (M ≥ 4096): gate, up, down, qkvz, gqa q, o, lm_head

| rank | candidate | why |
|---|---|---|
| 1 | **C1** (R=4, staged X) | only launch that makes 1–2 bpw raise AI (7.5 vs 0.93 at 1-bit). Word law. In-register. |
| 2 | **C3** fused gate+up on C1 | same as C1 plus X paid once and −64 dispatches. Gate/up only. |
| 3 | **C1b** 8 rows/TG, staged X | fallback if C1 occupancy fails on down (2.67 TG/core). |
| 4 | **C0** tpr64 / 2 rows | MEASURED genome. Control. Density-blind because of X re-read. Required 4-bit / 2-bit; 1-bit only as exclusive bytes. |
| 5 | **C4** HGRAVS two-stage | 32× fewer bytes, 25× fewer FLOPs, but 160-row stage is a different kernel. Rank 1 on **byte** count; rank last until C2 makes stage-1 occupy. |
| 6 | **C5** binary+CSR | two kernels; residual is irregular. Use only if quality demands outliers. |
| 7 | **C6** WH + C1 q2 | C1 plus a WH. Only if the representation is already Hadamard. |

### Mid-M (gqa k/v, M=1024)

C0 (512 TG, 8.53/core) over C1 (32 TG, 0.53/core). X reuse at 32
rows/TG starves the box. Optional: C0 with `rows_per_tg=8`
(tpr32 × 8 sg, 128 TG, 2.13/core) as a measurement point.

### Tiny-M (ba M=96; HGRAVS R M=160)

**C2** first. C0 is 0.80–1.33 TG/core. C1 is worse. Serial
one-thread-per-row is the Q80 512-row pathology (2.6 GB/s MEASURED)
and is banned on small organs even though Q38 ba has 96 rows not 512.

### Token-level implication (not a TPS claim)

401 GEMVs/token. Addressing is 60% of the 35.23 ms wall (MEASURED).
C1 does not reduce bytes; it reduces **re-read of X** and so can
raise achieved GB/s of the *same* payload only if C0 is X-bound
rather than W-bound. Honest-roof says the 13.6 GB Q4 stream is
W-bound at 667 GB/s. Therefore:

- At **4 bpw**, C1 is predicted to be a small win or a tie (Wbytes
  2720 ≫ x_tg32 640). Measure; do not assume.
- At **≤2 bpw**, C1 is predicted to be the first-order win (Wbytes
  720–1440 vs x_tg2 10240). This is the G1-relevant regime.
- **Byte reduction** (representation) is still the only lever that
  moves the 13.6 GB addressing bucket. Geometry only harvests that
  reduction. `G024` action line is unchanged: density first.

PROJECTED (ratio only, method of `QWEN38_BANDWIDTH_BOUND.json`
`projection_robustness`, not a new roof):
`ms_at_target = 33.912 * (target_bpw / 4.2527)`.
At 1.5 bpw → 11.96 ms GPU if the genome stays W-bound at the same
achieved rate. That is a **conditional** on C0-class saturation
surviving the pack change. C1 existing is what makes “same rate”
plausible at 1.5 bpw. Without C1, X re-read can hold the time near
the 4 bpw wall while bytes drop — the failure mode this geometry
is designed to prevent.

---

## 7. Family consume / reject

“Consume” = a ranked candidate can execute the family under
constraints 1–5. “Reject” = museum for this GEMV, or needs a
different machine.

| family (gravity / ascent inventory) | C1/C0 lane-word | C2 | C3 | C4 | C5 | C6 | verdict |
|---|---|---|---|---|---|---|---|
| `uniform_q4_g64` / HQ30UQ4 | **yes** (already) | n/a | yes | no | no | no | executable. Control pack. |
| `uniform_q3_g64` LSB | C-3C yes; C-3B after remap | yes | yes | no | no | no | executable. Remap to C-3B for word law. |
| `uniform_q2_g64` / HQ30UQ2 | after remap to 16 codes/uint | yes | yes | no | no | no | executable. |
| `binary_g128` / `HGRAVB01` | yes (byte today; uint 32-bit after remap; C0 1-bit TILE illegal) | yes | yes | no | base of C5 | no | executable. Cheapest reconstruct. |
| `ternary_t0.7_g128` | after remap 16×2-bit / uint | yes | yes | no | no | no | executable. Same GEMV as q2, different `q` map {0,+s,−s}. |
| `additive_q2q2_g64` | two remapped q2 streams | yes | yes | no | no | no | executable. No table. Double Wbytes (= Q4). |
| `hadamard_q2_g128` | GEMV = q2 | yes | yes | no | no | **yes** | executable only as C6. WH-with-shuffle = reject. |
| `HGRAVU01` bits=8 | yes, W=4 uint | yes | yes | no | no | no | executable, 8 bpw. Not a G1 density vehicle. |
| `HGRAVS01` r160 b3 | no as dense W | stage-1 | no | **yes** | no | no | executable **only** as C4. Current two-stage kernel K-cap 512 = reject on Q38. |
| `HGRAVR02` / rice_q1_rms / residual-compact | binary half yes | n/a | binary half | no | **yes** | no | executable only as C5. Rice bitstream as GEMV = **KILL**. |
| gravity PQ / residual-PQ | no | no | no | no | no | no | **KILL** (codebook gather). REOPEN_IF card×sub floats stay in TG for the whole TG and indices are lane-word. |
| strand / TQ bitslice LUT | no | no | no | no | no | no | **KILL** (table). |
| strand computed Acklam | no | no | no | no | no | no | **KILL** (not shift/mask; 64×64→128 integer). REOPEN_IF a tpr64 probe shows it still matches f32 time **and** a complete-token run beats C1-binary. |
| Q4_K / Q5_K / Q6_K / Q8_0 / Q5_0 GGML | no as-is | no | pair kernel exists for Q5_0 (`gravity_pq.metal:129`) | no | no | no | **KILL** as token GEMV pack. REOPEN_IF remapped to lane-word with the same codebook. Science only. |
| DSV4F FP4 | if closed-form 4-bit like Q4 | — | — | no | no | no | executable **iff** decode is shift/mask, no LUT. Occupancy lane used a serial/simd FP4 split (`matvec-occupancy-230x.json`); not Q38-shaped. |
| gravity functional codec (seeded generator) | no | no | no | no | no | no | **KILL** for this GEMV. Different execution model (`gravity_functional_codec.py`). |
| expand-to-float then GEMV | — | — | — | — | — | — | **KILL**. Binding. |
| expand-to-Q4 then `geo_tpr64` | — | — | — | — | — | — | **KILL** unless a complete-token measurement shows the expand is still a net win. Preferred shape is representation-specific consume. |

Representation-side summary:

- **Build these:** affine uniform 1/2/3/4-bit and binary/ternary whose
  physical layout is lane-word (§4). Group size a divisor of TILE.
- **Build these if quality needs them:** HGRAVS, but only with C4+C2,
  never dense expand. HGRAVR02 only as C5 sidecar.
- **Do not build for this kernel:** PQ, strand LUT, GGML superblocks,
  Rice-as-GEMV, functional generators, any codec whose decode is a
  gather.
- **3-bit:** C-3B (pad) or live with C-3C (legal extract, fails word
  law). C-3D (store as 4-bit) deletes the 3-bit reason to exist.

---

## 8. KILLS and REOPEN_IF

| ID | statement | REOPEN_IF |
|---|---|---|
| K1 | One-thread-per-row on M≤512 | a measurement on that organ shows ≥50% of the Q4 667 GB/s regime |
| K2 | Expand packed → dense float/Q4 → generic GEMV | complete-token wall with expand < packed-direct wall on the same artifact |
| K3 | Rice / any variable-length bitstream as the primary GEMV stream | a simdgroup can load exclusive aligned words of Rice and reconstruct with shift/mask only (not believed possible) |
| K4 | PQ / residual-PQ codebook DRAM gather | codebook staged in TG/registers, indices lane-word, and a tpr64 probe matches f32 |
| K5 | Strand LUT | same as K4 |
| K6 | `q80_hgravs01_two_stage_matvec` on Q38 | kernel K-cap raised and C2 occupancy fixed; then it is C4, not this entry point |
| K7 | C0 as the G1 1.5 bpw production launch | C1 (or C1b) measured slower than C0 on gate+down at 1-bit **and** 2-bit on this box |
| K8 | Hadamard-128 via `simd_shuffle` of packed lanes | a no-shuffle TG-memory WH is written |
| K9 | 819 GB/s or 411.51 GB/s used as a planning floor | a new honest-roof receipt on **this** layout supersedes `HONEST_ROOF_WEIGHT_ADDRESSING.md` |

---

## 9. Falsification menu (measurement lane; this lane does not run these)

Cheap, ordered. Each line is one claim in §6–§7.

1. **C1 vs C0, same Q4 buffer, gate 17408×5120.**
   If C1 full GPU ns ≥ C0, X-reuse is not the 4 bpw limiter (expected
   near-tie). Record GB/s = `47_349_760 / ns`.
2. **C1 vs C0, remapped 1-bit, same organ.**
   If C1 is not materially faster than C0, §3.3 is wrong.
3. **C1 down, 1-bit, 160 TG.**
   If it regresses vs C0, ship C1b for down and keep C1 for gate/up/lm_head.
4. **C2 vs C0 on ba 96×5120 and on HGRAVS R 160×17408.**
   If C2 does not cut the 71 µs HGRAVS discriminator toward the 7.5 µs
   Q4 class, HGRAVS stays a museum for Q38 token path.
5. **C3 vs two C1 on gate+up.**
   Win must show up in isolated pair **and** in the 401-dispatch token
   (dispatch cut is the second term).
6. **C-3B vs C-3C on a q3 down.**
   Word-law remap must not lose to LSB extract by more than noise at
   tpr32. If it loses, keep C-3C and drop constraint 2 for 3-bit only.
7. Do not use `disc_*` ns as GB/s. Do not run these concurrent with
   another GPU authority lane.

---

## 10. Exact formulas the next lane can check without a GPU

Constants from `qwen38_geometry.rs` / `honest_roof.rs`.

```
q4_bytes(M,K) = M * ceil(K/64) * 34
token_q4_gemv  = 64*(q4(17408,5120)+q4(17408,5120)+q4(5120,17408))
               + 48*(q4(16384,5120)+q4(96,5120)+q4(5120,6144))
               + 16*(q4(12288,5120)+2*q4(1024,5120)+q4(5120,6144))
               + q4(248320,5120)
               = 13_611_663_360

lane_word_bytes(M,K,bits,G) = M * (K*bits/8 + 2*ceil(K/G))
AI_opt(M,K,bits,G) = 2K / (K*bits/8 + 2*ceil(K/G) + 4K/M + 4)
AI_tg (M,K,bits,G,Rtg) = 2K / (K*bits/8 + 2*ceil(K/G) + 4K/Rtg + 4)

hgravs_q3_bytes(M,K,R=160,G=64) =
    (M*R + R*K)*3/8 + 2*(ceil(M*R/G)+ceil(R*K/G))
  = 1_464_320 for (M,K) ∈ {(5120,17408),(17408,5120)}
```

C0 launch: `tg = 128`, `grid = ceil(M/2)*128`.
C1 launch: `tg = 256`, `grid = ceil(M/32)*256`.
C2 (4 TGs/row): `tg = 128`, `grid = M*4*128`.

If any of those identities fail against the source constants, this
paper is wrong before a kernel is written.

---

## Completion report

STATUS
IMPLEMENT_READY

CLAIMS
1. Production Qwen3.8 GEMV genome is `geo_tpr64_tg128` on HQ30UQ4; wall 35_227_917 ns; addressing 21_293_102.5 ns of 13_611_663_360 B. Evidence: `TOKEN_NS_QWEN38.json`, `honest_roof.rs:46-56`, `qwen_uniform_q4.metal:181-221`.
2. That launch already satisfies the 32-contiguous-uint word law for 4-bit nibble pack. Evidence: `qwen_uniform_q4.metal:199-210` plus group-major 32 B/group.
3. At ≤2 bpw the same launch is X-reread bound: AI 0.93 (1-bit, 2 rows/TG) vs 7.51 (32 rows/TG). Evidence: DERIVED table §3.3 from `qwen38_geometry.rs` constants.
4. Cheap in-register families (q4/q3/q2/binary/ternary/additive/rice-CSR) match f32 at tpr64; HGRAVS does not (71458 vs 7083 ns down). Evidence: `QWEN38_RECON_MEASURED.json`. Relative only.
5. Ranked large-M kernel is C1 (tpr32, 32 rows/TG, staged 1024-float X, lane-word pack). Tiny-M is C2. HGRAVS is C4+C2. Rice/PQ/strand-LUT/GGML-superblock/expand-then-GEMV are KILL. Evidence: §5–§8.
6. 819 GB/s and 411.51 GB/s are not decode floors for this genome. Measured Q4-full roof at 13.6 GB is 666.7 GB/s. Evidence: `HONEST_ROOF_WEIGHT_ADDRESSING.md`, `honest_roof.rs:59`.

EVIDENCE
- `crates/hawking-core/src/model/qwen38_geometry.rs` (materialized)
- `crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:51-54,74-103,174-186`
- `crates/hawking-core/src/backend/honest_roof.rs:46-59,266-285`
- `crates/hawking-core/shaders/qwen_uniform_q4.metal:1-11,181-221`
- `crates/hawking-core/shaders/qwen_uniform_qn.metal:1-28`
- `crates/hawking-core/shaders/q80_mixed_decode.metal:540-616`
- `crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:1-47`
- `HEAD:tools/qwen38_recon_disc/recon.metal` (sparse; `git show`)
- `HEAD:receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json`
- `HEAD:receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json`
- `HEAD:receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json`
- `HEAD:receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.md`
- `HEAD:receipts/ascent-2026-08-16/matvec-geometry-sweep.json`
- `HEAD:receipts/ascent-2026-08-16/matvec-occupancy-230x.json`
- `HEAD:docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md`

CHANGES
- created `workspace/superwave/g1/g1-direct-gemv-geometry.md` only

TESTS
- `test -s workspace/superwave/g1/g1-direct-gemv-geometry.md` (run at lane end)
- `wc -l workspace/superwave/g1/g1-direct-gemv-geometry.md` (run at lane end)
- `git status --porcelain` (run at lane end)

RISKS
- C1 down at 2.67 TG/core may regress; C1b is the named fallback.
- Honest-roof absolute GB/s is GPU_PROTECTED_CPU_CONTENDED; relative W-vs-X geometry does not depend on that scale.
- 32 KB TG capacity for staged X is ESTIMATED, not counter-sampled.
- Sibling G1 gravity-lane reports were absent; a family invented after this file is unclassified until it states pack bits, group, and whether decode is shift/mask.

UNRESOLVED
- No GPU measurement in this lane (contract). C1/C2/C3/C4 ranking is predicted AI plus measured relative recon, not a timed win.
- Whether 3-bit should pay C-3B pad or keep C-3C LSB extract.
- Whether gqa kv (M=1024) stays on C0 or a 8-row C0 variant.
- Energy / pJ per weight not in TOKEN_NS (`G024` unresolved list).

NEXT
- Measurement lane: §9 items 1–4, in that order, serialized against other GPU authority.
- Representation lane: emit lane-word physical layout for any codec it wants C1 to run; do not emit PQ / LUT / Rice-as-GEMV / dense HGRAVS.
