# G1 scale-fetch amortization

Lane: `112-scale-fetch-amortization`. Write scope: this file only.
No GPU, no generate, no pack, no live-G0 touch, no edit of
`q80_mixed_decode.metal` or `qwen38_hybrid_decode.rs`.

Every number is **SOURCE** (file:line), **DERIVED** (closed arithmetic on
SOURCE), **MEASURED** (this process, CPU only), **CITED** (prior receipt /
contract field, not re-run), or **ESTIMATED**. A projection is never a
measurement. The circulating ~31 TPS figure for a “fixed” kernel is a
**PROJECTED** complete-token wall from 11.47 GB at 639.25 GB/s plus
unchanged non-GEMV; it is not a token measurement.

G0 live manifest sha256
`d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`
was not opened.

---

## 0. Verdict

The incumbent bits=4 path issues **1.0 scale loads per output element**.
G0 `geo_tpr64_tg128` issues **0.125**. Unique DRAM is **1/64** in both
(one f16 per group of 64) if the same address coalesces. The named
per-element fetch is a **source-level load-use in a 1-wide loop**, not
extra unique bytes.

G0 already does the first amortization (one half load, reuse across 8
nibbles). The candidate is not on that kernel.

Under the production 64-TPR × W=8 × TILE=512 mapping, a group of 64 maps
to **8 lanes × 1 iteration**, not one thread iteration. Mapping a whole
group onto one thread forces W=64 ⇒ TILE=4096, which does **not** divide
K∈{5120,6144,17408}.

| candidate | issued / out elem | extra regs (SOURCE live) | pack change | keep “no repack”? |
|---|---:|---|---|---|
| **A-partial** hoist 1 scale / 8 codes, keep TILE=512 | **0.125** | +1 `float` vs incumbent 1-wide; = G0 | no | **YES** |
| A-full hoist 1 scale / 64 codes, W=64 | 0.015625 | +0 if the 8 uints are looped | no | YES, but TILE holes |
| **B** one lane loads, `simd_shuffle` to the 8-lane clique | 0.015625 | +0 vs A-partial | no | **YES** |
| **C** interleave scale into the code stream | ≤0.015625 only if also A/B | 0 | **YES** | **FORFEIT** |
| **D** recip / fused multiplier at upload | 0.125 if stacked on A, else 1.0 | 0 or +1 | resident widen or unused ALU | **no win** |

**First kernel (lane 110):** A-partial on a `geo_tpr64` launch.
bits=4: aligned `uint` load, `q = nibble - 7`. bits=3: existing 3-byte
extract, one scale, `q = code - 3`. Hardcode `group_size = 64`.
Do not repack. Do not delete the scale plane. Do not wait on B or C.

Deleting the scale plane **KILLS** capability (CITED uniqueness walk).
Amortize access. Do not remove scales.

---

## 1. What is bound

Candidate `mixed-q3mlp-v1` census (CITED contract +
`g1-bracket-bisection.md:183`): 851×HGRAVU01 = 498 Uniform matrices +
353 vectors dequantized to f32; q4=0. Of the 498 matrices: **192 bits=3
MLP** + **306 bits=4** of which **305 are GEMVs** and 1 is embed lookup.

Dispatch, default fuse ON (`env_opt_out` unset → true,
`lib.rs:207-214`, `qwen38_hybrid_decode.rs:41-45`):

```1681:1691:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
                let (name, grid) = if bits == 8 {
                    ...
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
                };
```

`simd8_grid` = `ceil(rows/8)*256`, TG 256 (`:1042-1044`).
One simdgroup (32 lanes) owns one row.

G0 production bind is **not** this path:

```547:582:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
pub const QWEN38_Q4_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
...
            Self::GeoTpr64Tg128 => {
                let tg = 128u32;
                let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
```

64 threads/row, 2 rows/TG, TG 128. Candidate token does not launch it
(CITED contract; `q4=0` in the mixed census).

GEMV K on this body is only `{5120, 6144, 17408}`
(`g1-group-partition-geometry.md:79`, `qwen38_geometry.rs:24-25`).
All `K % 64 == 0` (MEASURED below).

---

## 2. Layout: scales vs codes

### 2.1 HGRAVU01 (candidate Uniform)

On-disk body is one slab: `fp16 scales[groups] || LSB-first unsigned codes`
(`q80_mixed_decode.rs:17` HGRAVS comment is the same factor body;
`uniform_from_body` `:802-821`).

```
groups      = ceil(rows*cols / group_size)
scale_bytes = groups * 2
code_bytes  = ceil(groups * group_size * bits / 8)
bound       = (1 << (bits-1)) - 1     // bits=3 → 3; bits=4 → 7
```

SOURCE `factor_layout_from_meta` `:1149-1168`, `pack_uniform_factor`
`:547-576`, `gk_uniform_value_wide` `:194-206`.

Upload **splits** the slab into two MTLBuffers (`upload_mixed` Uniform
arm `:1435-1447`). Kernel signature is `buffer(0)=codes`,
`buffer(1)=scales`. Same two-buffer bind as G0 Q4.

Group index is **flat row-major**: `group = (row*cols + col) / 64`
(`gk_uniform_value_wide:202`). When `cols % 64 == 0` this equals
G0’s `row * (cols/64) + col/64`. Every Uniform GEMV K here is 0 mod 64,
including `in_proj_a/b` 48×5120. HGRAVS L is 5120×**160** (160%64=32)
and is **not** this lane’s Uniform path.

### 2.2 HQ30UQ4 (G0)

Per-row groups of 64, 32 code bytes, even nibble low / odd high,
`q = nibble - 8`, one f16 scale/group. Scales then codes in the file;
upload splits the same way (`uniform_q4.rs:3-11`,
`qwen38_hybrid_decode.rs:854-856`).

### 2.3 bits=4 HGRAVU01 **is** a nibble stream

LSB pack (`pack_unsigned` `:170-180`): bit 0 of code 0 lands in bit 0 of
byte 0. For `bits=4`, `bit0 = element*4` ⇒ even element = low nibble,
odd = high nibble. Same placement as HQ30UQ4. **Bias differs: 7 not 8.**

MEASURED this process (algorithm copied from `:149-180` and
`extract_unsigned` `:579-588`):

```
bits4 packed 1032547698badcfe
bytes [16, 50, 84, 118, 152, 186, 220, 254]
bits4 even_low_odd_high True nbytes 8
bits4 group64 bytes 32 aligned4 True
bits3 group64 bytes 24 aligned4 True
bits4 tpr64 uint offsets local= [0, 4, 8, 12, 16, 20, 24, 28] all%4 True
bits3 tpr64 byte0= [0, 3, 6, 9, 12, 15, 18, 21]
bound bits 3 3
bound bits 4 7
```

So the existing HGRAVU01 bits=4 body **feeds a G0-style `uint` load**
with no repack. bits=3 is 24 B/group; 8 codes = 3 B starting at
offsets 0,3,6,… inside the group — **not** 4-byte aligned. That extract
is already written (`simd3` `:869-881`). Word-law remapping of bits=3
is a pack change and is out of scope.

### 2.4 Scale plane is required

G0 GEMV scale plane **800,686,080** B = `400,343,040` groups × 2
(CITED `g1-byte-deletions.md:79-84`, `HONEST_ROOF` 13,611,663,360 =
codes 12,810,977,280 + scales 800,686,080).

Same group count on the candidate (DERIVED from the published 48 DN +
16 GQA + 64 MLP + lm_head shapes; split vs fused in_proj preserves
elements):

```
TOTAL groups 400343040 scale 800686080 codes 10671882240 W 11472568320
GEMVs bits3 192 bits4 305 all 497
```

`11,472,568,320` B matches the contract’s **11.47 GB/token**. MLP codes
shrink 8 bits/weight (32→24 B/group); the scale plane does **not**.

Deletion **KILLS**: 0 singleton tensors, 0 rows with unique≤4, scales
span `9.54e-7 … 0.213` (CITED `g1-byte-deletions.md:155-185`).
**REOPEN_IF** a generate-proven codec whose kernel does not stream a
per-group f16.

---

## 3. Current access pattern

### 3.1 Incumbent bits=4 — `q80_hgravs01_factor_matvec_simd`

SOURCE `:499-534` + `gk_uniform_value_wide` `:194-206`.

```
32 threads / row, TILE = 32, W = 1
col = base + simd_lane;  base += 32
every iteration:
    group  = element / group_size          // runtime buffer(6)
    code   = extract_wide(codes, element, bits)   // 1-wide
    scale  = float(scales[group])          // ONE half load
    acc   += (int(code)-bound) * scale * x[col]
```

Thread 0 visits cols `0,32,64,96,…`. Consecutive visits **0 and 32 are
the same group**. The source does not hoist. `group_size` is a buffer
uniform, so the compiler is not obliged to prove the alias.

**Issued / out element = 1.0** (SOURCE). Per row, K half loads.
Per thread, K/32 loads.

The load is **not data-dependent on the code**. Address is a function of
the loop index only. It can issue in parallel with the extract. It
cannot be DCE’d: `q * scale * x` needs the value.

### 3.2 Incumbent bits=3 — `q80_hgravs01_factor_matvec_simd3`

SOURCE `:843-907`. 32 threads/row, W=8, TILE=256, `col += 256`.

Already 8-wide extract (3 bytes). Then **eight** scale loads:

```882:889:crates/hawking-core/shaders/q80_mixed_decode.metal
        const float s0 = float(scales[(row_base + col) / group_size]);
        const float s1 = float(scales[(row_base + col + 1u) / group_size]);
        ...
        const float s7 = float(scales[(row_base + col + 7u) / group_size]);
```

For `group_size=64` and `col % 8 == 0`, `s0==…==s7` (DERIVED).
The source still issues 8. **Issued / out element = 1.0.**
Remainder 1-wide loop is dead: every K is 0 mod 8.

This is the proof the compiler does **not** automatically amortize when
`group_size` is a buffer.

### 3.3 G0 — `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`

SOURCE `qwen_uniform_q4.metal:181-221`.

```
64 threads / row, W = 8, TILE = 512, TG 128, 2 rows/TG
lane = split*32 + simd_lane            // 0..63
col  = lane*8;  col += 512
every iteration:
    scale  = float(scales[rgb])        // ONE half
    packed = *(uint*)(codes + rgb*32 + local/2)
    acc   += unpack8(packed, scale, x, col)   // 8 FMAs, same scale
```

**Issued / out element = 1/8 = 0.125.** Unique groups = K/64.

G0 **does** pay a scale load. It does not pay it per element.
`QWEN38_RECONSTRUCTION_IS_FREE` (CITED) says the nibble ALU at this
launch matches f32 time; it does not say the scale plane is free DRAM
(`2/34` of the 13.61 GB, PROJECTED 1,252,535 ns @ 639.25 GB/s,
`g1-byte-deletions.md:135-140`).

### 3.4 Issued vs unique, one row

DERIVED. Unique = K/64 in every legal mapping (one f16/group).

| K | groups | incumbent issued | G0 / A-partial issued | unique |
|---:|---:|---:|---:|---:|
| 5120 | 80 | 5120 | 640 | 80 |
| 6144 | 96 | 6144 | 768 | 96 |
| 17408 | 272 | 17408 | 2176 | 272 |

Per-thread issued: incumbent K/32 (160 / 192 / 544) vs G0 K/512
(10 / 12 / 34). **16× more scale loads per thread**, because 1-wide
**and** 32 TPR.

If L1 multicasts 32 lanes hitting one half, unique DRAM is already
1/64 on the incumbent. The 3.78× token (CITED 148,588,917 ns vs G0
39,326,090) is then **not** “64× scale traffic”. It is the 1-wide
load-use + 32-TPR + missing 8-wide extract. Scale amortization without
widening the extract leaves the 1-wide FMA loop in place.

No GPU this lane. That decomposition is SOURCE + DERIVED, not a
measured split of the 137,099,341 ns GEMV bucket (CITED contract).

---

## 4. 64-TPR × group-64

Production mapping (G0, the speed target):

```
TILE = 64 threads × 8 codes = 512
5120/512=10, 6144/512=12, 17408/512=34     all exact (MEASURED)
```

One TILE covers 8 groups. Lanes 0–7 share group 0, 8–15 share group 1,
…, 56–63 share group 7. **A group of 64 = 8 concurrent thread
iterations in the same loop step.** Not one thread iteration.

Hypothetical “one thread owns the group”:

```
W = 64, TILE = 64×64 = 4096
5120 % 4096 = 1024     16/64 lanes live on the tail
6144 % 4096 = 2048     32/64 live
17408 % 4096 = 1024    16/64 live
```

MEASURED remainders above. Occupancy holes on every K. This mapping is
legal (no pack change) and is **not** the first kernel.

8-TPR × W=64 would keep TILE=512 and give A-full, but abandons the
geometry-sweep winner (`qwen_uniform_q4.metal:181-182`) and is
UNMEASURED.

---

## 5. Candidates

Regs = SOURCE live scalars in the inner iteration, not a compiler
occupancy report. Occupancy delta is **UNMEASURED** (no GPU).

### A — hoist one scale per group into a register

**A-partial (keep TILE=512, W=8).** One `float scale` per iteration,
reuse on 8 codes. Issued 0.125. Extra regs vs incumbent 1-wide: +1
float + 1 `uint packed` (bits=4) or 3 `uchar` (bits=3). Pack change:
**no**. This **is** G0’s scale schedule.

On the incumbent 32-TPR 1-wide loop a weaker hoist is available without
changing mapping: each thread’s cols `c, c+32` share a group, so one
load covers two iterations. Issued 0.5. Not enough; still 1-wide.

**A-full (W=64).** Issued 0.015625. Pack change: **no**. TILE=4096
holes (§4). **KILL as default.** **REOPEN_IF** a launch with W=64
still matches G0 occupancy on the three K and a complete-token A/B
is not slower.

Specialize `group_size=64` as a literal / function constant so the
hoist is obvious. Leaving it in `buffer(6)` is why simd3 still writes
s0..s7.

### B — one lane loads, broadcast across the simdgroup

Metal already uses `simd_sum` / `simd_shuffle` (`mha.metal:197-200`).
`simd_shuffle(scale, leader)` is in-family. Broadcast is **within 32
lanes**, not across the two simdgroups of a 64-TPR row.

On tpr64 W=8: leader = `lane_in_row` floored to the 8-lane clique
(or `simd_lane & ~7` inside one split). Issued 1/64. Extra regs: 0
vs A-partial. Pack change: **no**.

On incumbent 32-TPR 1-wide: one load per 32-col half-group, issued
2/64 = 0.03125, still 1-wide extract.

Hardware multicast of 8 (or 32) lanes hitting the same half is
**ESTIMATED** to already collapse unique DRAM. B is a source-level
cleanup of issued loads / latency, not a byte deletion.

Do not block the first kernel on B.

### C — interleave scales into the code stream

Today: all scales, then all codes, then split to two buffers.
Interleaved: `[half scale][pad?][32 B codes]` per group, one buffer.

One aligned read of both: 32+2=34 B (bits=4) or 24+2=26 B (bits=3).
Neither is a power-of-two. Pad to 64 B is +30 B/group = **+3.75 BPW**
(DERIVED, bits=4) and **raises** unique bytes. Pad to 36 B is +0.25
BPW and still not one 32-B transaction.

Unique bytes with no pad: **unchanged**. Two-stream bind as the 24%
addressing hole is already **KILLS** (`g1-addressing-topology.md:419`,
single-address also two streams, 699.57 GB/s).

Pack change: **YES. FORFEITS “no repack”.** Prior lane: existing
HGRAVU01 body feeds geo_tpr64 directly (CITED contract;
nibble/`uint` identity MEASURED §2.3).

`g1-sub15-kernel-specialization.md:42` “fold scale into the packed
stream, −1 buffer/GEMV” is a bind-count idea, not a load-amortization
win, and still a repack.

**KILL for this iteration.** **REOPEN_IF** a generate-scored new pack
exists for another reason and the interleaved struct is free at tpr64.

### D — reciprocal or fused multiplier at upload, not per token

Decode is `float(q) * float(scale) * x` (`gk_uniform_value_wide:203-205`,
`unpack8:175-176`). There is **no divide** and no per-token scale
recompute. Pack already did `value/scale` (`pack_uniform_factor:560-563`).

| transform | effect | verdict |
|---|---|---|
| hoist `half→float` | absorbed by A | do that |
| store `1/scale` | turns a mul into a div | **KILL** |
| widen f16→f32 at upload | +800,686,080 B resident (DERIVED, same group count as G0) | **KILL** under 15 GB peak / 13.6 GB live G0 |
| per-group 16-entry dequant LUT | 32 B/group, doubles scale-class traffic | **KILL** |
| preprocess `sx[col]=scale[col/64]*x[col]` | writes K floats per GEMV | **KILL** (more traffic) |

No pack change for the f32-widen (upload-only). Still a residency
forfeit. ALU of `half→float` at tpr64 is already inside the
reconstruction-is-free envelope (CITED). **No independent D.**

---

## 6. First kernel (for lane 110; this lane does not type it)

Preserve no-repack. Preserve TILE=512. Specialize g=64.

bits=4 (305 GEMVs):

```
lane = split*32 + simd_lane;             // 0..63
for (col = lane*8; col < cols; col += 512) {
    group = row*(cols/64) + col/64;      // == (row*cols+col)/64
    scale = float(scales[group]);        // A-partial, ONE load
    packed = *(device const uint*)(codes + group*32 + (col%64)/2);
    // 8 nibbles even-low-odd-high; q = int(nibble) - 7   // NOT 8
    acc += unpack8_bound(packed, 7, scale, x, col);
}
```

`uint` address is 4-byte aligned (MEASURED offsets 0,4,…,28).
G0’s `- 8` is **wrong** on this body.

bits=3 (192 GEMVs): same launch and the same one scale load. Extract
stays the 3-byte simd3 unpack (`:869-881`), `q = code - 3`. Do not
invent an aligned-uint pack.

Optional B after A is correct: `simd_shuffle` from the clique leader.
Not required for the first complete-token A/B.

Do not change the G0 artifact. Do not add a third buffer.

---

## 7. KILLS / REOPEN_IF

| claim | verdict | REOPEN_IF |
|---|---|---|
| Delete the f16 scale plane, keep Q4/HGRAVU nibbles | **KILLS** capability | generate-proven codec with no per-group f16 |
| One scale per tensor / per row | **KILLS** (CITED 0 singletons, 0 rows unique≤4) | uniqueness walk flips |
| C interleave as the amortization | **KILLS** this iteration (repack; 34 B unaligned; two-stream already not the 24%) | new pack exists anyway **and** tpr64 A/B is not slower |
| D reciprocal | **KILLS** (no decode div) | kernel starts dividing |
| D f32 scale buffer | **KILLS** residency (+800,686,080 B) | live working set has that headroom **and** a measured GEMV win |
| A-full W=64 / TILE=4096 as default | **KILLS** as default (tail occupancy) | that launch matches G0 occupancy on all three K |
| Treat ~31 TPS as a measured wall | **not a measurement** | complete-token receipt on the new kernel |
| Scale issued-load ratio as a TOKEN_NS claim | **not measured** | locked addr/decode/full probes on one bits=4 organ + complete token |
| Reopen representation search / drop scales for BPW | **out of scope** | floor contract changes |

Cheapest next experiment (GPU, **not this lane**): one bits=4 attention
organ, incumbent `factor_matvec_simd` vs A-partial geo_tpr64 HGRAVU
(bound 7), addr_probe / decode_probe / full. Then one complete token.
Do not confound the live G0 lock.

---

## 8. Evidence

### 8.1 This process

```
python3  (LSB pack + TILE + issued-load arithmetic, 2026-08-17)
bits4 even_low_odd_high True
bits4 group64 bytes 32  bits3 group64 bytes 24
bound bits 3 3   bits 4 7
tile 512  remainders (0,0,0) on {5120,6144,17408}
tile 4096 remainders (1024,2048,1024)
issued  incumbent_1wide=1.0  A_partial_W8=0.125  unique=0.015625
K_mod_64 all 0
G0/candidate GEMV scale bytes 800686080
candidate W 11472568320   bits4 GEMVs 305  bits3 GEMVs 192
```

Command is the two `python3` heredocs in this lane’s session log.
No GPU. Peak RSS of those scripts is process-default (kilobytes).

### 8.2 Source pointers

| claim | pointer |
|---|---|
| bits=4 → 32-TPR 1-wide simd | `qwen38_hybrid_decode.rs:1687-1690`, `q80_mixed_decode.metal:499-534` |
| bits=3 → simd3 8-wide, 8 scale loads | `q80_mixed_decode.metal:843-889` |
| per-element `scales[group]` | `gk_family.metal:194-206` |
| G0 64-TPR 8-wide one scale + uint | `qwen_uniform_q4.metal:181-221` |
| HGRAVU body = scales \|\| codes, bound=(1<<(bits-1))-1 | `q80_mixed_decode.rs:547-576,802-821,1149-1168` |
| LSB pack | `q80_mixed_decode.rs:170-180` |
| two-buffer upload | `qwen38_hybrid_decode.rs:1435-1447` |
| fuse default ON | `lib.rs:207-214`, `qwen38_hybrid_decode.rs:41-45` |
| G0 launch | `qwen38_hybrid_decode.rs:547-582` |
| K set, g=64 exact | `g1-group-partition-geometry.md:79-93` |
| scale plane 800,686,080 + deletion KILL | `g1-byte-deletions.md:25-30,79-185` |
| two streams not the 24% | `g1-addressing-topology.md:419` |
| 851 U / 192 bits=3 / 659 bits=4 | `g1-bracket-bisection.md:183` |
| G0 not on candidate token | contract census `q4=0`; `QWEN38_Q4_MATVEC_KERNEL` only used for `weights.q4` |
| reconstruction free at tpr64 | `g1-direct-gemv-geometry.md:126-153` (CITED) |
| word law bits=4 yes, bits=3 no | `g1-direct-gemv-geometry.md:184-212` |

### 8.3 Not claimed

- TOKEN_NS delta of A/B/C/D. No GPU.
- Compiler register file after hoist. SOURCE live count only.
- L1 multicast collapsing incumbent unique DRAM. ESTIMATED.
- ~31 TPS. PROJECTED, not a wall.
- Quality of bound-7 vs bound-8. Pack already chose 7; kernel must match.
