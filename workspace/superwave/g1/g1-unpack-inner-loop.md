# G1 unpack inner loop — HGRAVU01 bits 3 / bits 4 under geo_tpr64

Lane: analysis only. No GPU. No edit of `q80_mixed_decode.metal` or
`qwen38_hybrid_decode.rs`. No repack. No live-G0 touch.

STATUS: **IMPLEMENT_READY**

## Verdict

Keep the production `geo_tpr64` map (`lane_in_row * 8`, stride 512).
Unpack **8 codes per iteration** for both bit widths. One scale. No
group-boundary branch. Qwen3.8 GEMV `cols` are all `% 64 == 0` and
`% 512 == 0`, so there is no tail.

| bits | load | extract | q | do not |
|---:|---|---|---|---|
| 3 | 3 `uchar` at `byte0 = 3*element0/8` | `(packed >> 3*i) & 7` | `q-3` | `*(device uint*)(codes+byte0)` |
| 4 | 1 aligned `uint` at `rgb*32 + local/2` | `(packed >> 4*i) & 15` | `q-7` | copy `unpack8`'s `- 8` |

8-wide bits 3 is the shipping `simd3` formula with the 8 scale fetches
collapsed to 1. 8-wide bits 4 is the shipping `geo_tpr64` `unpack8`
with bound 7, not 8. Same nibble wire as HQ30UQ4; different zero-point.

Second-best: bits 4 **32-wide `uint4`** under a *changed* stride-2048
map. Lost: breaks the proven 8-wide / 512 genome, 6/8 lanes misalign
if the map is *not* changed, occupancy unmeasured. Bits 3 second-best
is 1-wide `extract_wide` (shipping, correct, slow).

---

## 1. Pack law (SOURCE)

HGRAVU01 body = `fp16 scales[groups] || unsigned-LSB codes`.
`groups = ceil(elements / 64)`. Last group padded to 64 codes then
packed (`q80_mixed_decode.rs:545-566,83-90`). Kernel sees codes and
scales as two buffers (`qwen38_hybrid_decode.rs:1749-1759`).

```
bound = (1 << (bits-1)) - 1     # bits=3 → 3; bits=4 → 7
q     = int(code) - int(bound)
value = float(q) * float(scale)
bit0  = element * bits          # element = row*cols+col, flat
```

`pack_unsigned` writes each code LSB-first into a single bitstream
(`q80_mixed_decode.rs:170-180`). `extract_unsigned` / `gk_uniform_extract`
read the same way (`:579-588`, `gk_family.metal:208-221`).

| bits | codes/group | bits/group | **bytes/group** | bound | q range |
|---:|---:|---:|---:|---:|---|
| 3 | 64 | 192 | **24** | 3 | [-3,3] |
| 4 | 64 | 256 | **32** | 7 | [-7,8] |

Bits 4 nibble identity (DERIVED, CPU-verified 256/256): even element =
low nibble, odd = high nibble. Same wire as HQ30UQ4
(`qwen_uniform_q4.metal:7-9`). **Zero-point is not the same.**
HQ30UQ4 is `q = nibble - 8` ∈ [-8,7]. Copying `- 8` onto HGRAVU01
bits 4 is silent corruption (CPU: bound7 vs bound8 differ in **32/32**
8-wide windows).

`packed_byte_count` is `ceil(n*bits/8)` with **no** per-group pad to
32 B (`:83-90`). Bits 3 groups are 24 B tight. A 4-byte load of the
last 3 B of the last group is OOB (CPU: `byte0=93`, `uint_end=97`,
`payload=96`).

When `cols % 64 == 0`, flat groups == per-row groups:

```
rgb      = row * (cols/64) + col/64
byte3    = rgb * 24 + 3*(local/8)     # local%8==0
byte4    = rgb * 32 + local/2         # local%8==0 ⇒ %4==0
scale_i  = rgb
```

All 9 Qwen3.8 GEMV shapes satisfy `cols % 64 == 0` and `cols % 512 == 0`
(geometry `qwen38_geometry.rs:24-52`; CPU table in §9). Including
`ba` 96×5120.

---

## 2. geo_tpr64 map (SOURCE)

```
# qwen_uniform_q4.metal:195-210
lane_in_row = split*32 + simd_lane     # 0..63
col         = lane_in_row * 8          # then += 512
# 8 consecutive weights / iteration / thread
# 64 threads × 8 = 512 columns / pass
```

Incumbent candidate dispatch is **not** this kernel.

| bits | kernel | tpr | extract | stride | evidence |
|---:|---|---:|---|---:|---|
| 3 | `q80_hgravs01_factor_matvec_simd3` | 32 | 8-wide, **8 scales** | 256 | `:845-898`, `qwen38_hybrid_decode.rs:1687-1688` |
| 4 | `q80_hgravs01_factor_matvec_simd` | 32 | **1-wide** `extract_wide` | 32 | `:499-528`, `:1689-1690` |
| G0 Q4 | `..._geo_tpr64_tg128` | 64 | 8-wide, 1 scale, `q-8` | 512 | `qwen_uniform_q4.metal:183-210` |

`simd` is one simdgroup/row, `col = base + simd_lane`, one
`gk_uniform_value_wide` per column. That is the named 1-wide bits-4
path. `bits` is a buffer, so the `if (shift+bits > 8)` in
`gk_uniform_extract_wide` (`gk_family.metal:188-190`) does not fold.
For bits 4 that compare is never true (CPU: 0/64). Still paid.

---

## 3. Bits 3 arithmetic

8 consecutive codes starting at `e0 = 8m` occupy bits `24m .. 24m+23`
= **exactly 3 bytes**, byte-aligned. `3` and `8` coprime ⇒ 8-wide is
byte-aligned iff `e0 % 8 == 0`. The tpr64 map gives that for free
(`col = lane*8`, `cols % 8 == 0`).

Those 8 codes never cross a 64-group (CPU: 0 crosses on every
`e0 % 8 == 0` in 0..9992). So **one** scale. `simd3` still loads
eight (`q80_mixed_decode.metal:882-889`). Do not copy that.

Shipping formula (`:870-881`) ≡ 24-bit assemble (CPU: 32/32 windows):

```
packed = b0 | (b1<<8) | (b2<<16)
qi     = (packed >> 3*i) & 7          # i=0..7
```

`simd3`'s byte-crossing terms (`q2`, `q5`) are the same 24-bit window.

### Alignment classes under tpr64 (lane 0..7 of a group)

`byte0 = 3*(local/8)`. CPU:

| lane | local | byte0 | `byte0%4` | 24 bits in one aligned `uint`? |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | yes, shift 0 |
| 1 | 8 | 3 | 3 | **no**, needs 2 words (shift 24) |
| 2 | 16 | 6 | 2 | **no**, needs 2 words (shift 16) |
| 3 | 24 | 9 | 1 | yes, shift 8 |
| 4 | 32 | 12 | 0 | yes |
| 5 | 40 | 15 | 3 | no |
| 6 | 48 | 18 | 2 | no |
| 7 | 56 | 21 | 1 | yes, shift 8 |

**KILL** `const uint packed = *((device const uint*)(codes + byte0));`

MSL `device uint*` requires 4-byte alignment. Hardware that aligns
down yields the wrong 24 bits on 4/8 lanes (CPU). Lane 7's 4-byte
load also reads 1 B past the group; last group of the buffer is OOB
(CPU: `uint_end=97 > 96`).

`packed_uchar3` / three `uchar` loads are byte-aligned and in-bounds.

### Full-word 32-bit is not 8-wide

32 bits / 3 = **10 codes + 2 leftover**. `64 % 10 = 4`. Independent
10-from-each-`uint` matches the oracle on word 0 only; words 1..5
fail (CPU). Carry-chain is sequential and does not match the 8-col
stride (next iteration is +512 cols = +8 groups away). **KILL** as
the tpr64 extract.

### 128-bit

128 bits / 3 = 42 + 2 leftover. Does not divide 64. Group starts:
even `g` 16-aligned, **odd `g` at byte `24g ≡ 8 (mod 16)`** (CPU).
`*(device uint4*)(codes + 24*g)` on odd groups is a corruption risk.
**KILL** under this pack.

### 4-wide

4 codes = 12 bits. Byte-aligned iff `e0 % 8 == 0`. The other half of
`% 4` starts sit at bit-shift 4 (CPU). A 4-wide map (`lane*4`) makes
half the lanes mid-byte. Under the 8-wide map, two 4-wides are just
a split 8-wide. Lost.

---

## 4. Bits 4 arithmetic

`bit0 = 4*e` is always nibble-aligned. `extract_wide` never needs a
second byte (CPU: 0/64). 8 codes = 32 bits = one `uint`.

tpr64 `local ∈ {0,8,...,56}` ⇒ `local/2 ∈ {0,4,...,28}` ⇒ **uint
aligned**, and all 8 stay in one group (CPU: 0/64 misaligned, 0/64
cross). This is already how `geo_tpr64` loads
(`qwen_uniform_q4.metal:209`).

```
qi = ((packed >> 4*i) & 15) - 7     # HGRAVU01
qi = ((packed >> 4*i) & 15) - 8     # HQ30UQ4 only
```

`unpack8`'s byte split (`:173-176`) is the same 8 nibbles. Change
the `8`, not the addressing.

### 4-wide

4 nibbles = 1 `ushort`. Aligned when `local % 4 == 0`. Legal. Half
the amortisation of 8-wide on the same map. Lost.

### Full-word

32-bit full-word **is** 8-wide. 128-bit = 32 nibbles.

Under the **existing** 8-wide map, `uint4` at `local/2` is 16-aligned
only for `local ∈ {0,32}` — **6/8 lanes fail** (CPU). **KILL** unless
the map changes to 32-wide (`col = lane*32`, stride 2048). That map
is 16-aligned and stays in-group (CPU). It is the second-best, not
the pick: it abandons the measured G0 launch, 32 FMAs / 8 `float4`
of X per iteration, occupancy unknown. This lane did not run GPU.

---

## 5. Width table

Ops are **DERIVED** static MSL counts, not GPU ISA, not measured.
`extract_alu/wt` = shift+mask+sub+cvt. `code_ld` / `scale_ld` amortized.
`regs` = values live across the FMA body if the compiler reuses `q`.

### Bits 3

| width | codes/load | code_ld/wt | scale_ld/wt | extract_alu/wt | fma/wt | regs | group edge | alignment |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1-wide (incumbent for bits≠3; also `extract_wide` rem) | 1 | 1.25 | 1 | ~8 | 1 | ~6 | branch-free | byte; 16/64 need 2 B (`e%64 ∈ {2,5,10,13,...}`) |
| 4-wide | 4 / 12 bits | 0.50 | 0.25 | ~5 | 1 | ~8 | branch-free if `e0%8==0` | half of `%4` starts mid-byte |
| **8-wide (pick)** | **8 / 24 bits** | **0.375** | **0.125** | **4** | **1** | **5** | **branch-free** | 3×`uchar` only; uint-cast **kills** 4/8 lanes |
| full-word 32 | 10 + 2 bits | — | — | — | — | — | remainder 4; carry | **KILL** no-carry (CPU word1+) |
| full-word 128 | 42 + 2 bits | — | — | — | — | — | 128 ∤ 192 | **KILL** odd-group `uint4` |

8-wide regs: `packed, scale, acc, float4 x0, float4 x1`.

### Bits 4

| width | codes/load | code_ld/wt | scale_ld/wt | extract_alu/wt | fma/wt | regs | group edge | alignment |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **1-wide (incumbent)** | 1 | 1 | 1 | ~6 + live branch | 1 | ~6 | branch-free | byte |
| 4-wide | 4 | 0.25 | 0.25 | 4 | 1 | ~7 | branch-free | `ushort` ok if `local%4==0` |
| **8-wide (pick)** | **8** | **0.125** | **0.125** | **4** | **1** | **5** | **branch-free** | `uint` ok on tpr64 map |
| full-word 32 | 8 | = 8-wide | | | | | | same row |
| full-word 128 | 32 | 0.03125 | 0.03125 | 4 | 1 | ~14 | branch-free **if** 32-wide map | **KILL** on 8-wide map (6/8) |

1-wide vs 8-wide bits 4: **8×** code loads, **8×** scale loads, plus a
per-element `shift+bits>8` that never fires. That is the extract half
of the named 3.78× mechanism. Not a token-level claim.

---

## 6. Recommended Metal (specialized, `cols % 64 == 0`)

Same launch as `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
(tg 128, 4 SG, 2 rows/TG, 64 tpr). Hardcode bits/bound. Do not take
`bits` from a buffer.

```metal
// shared map
const uint lane_in_row = split * 32u + simd_lane;   // 0..63
const uint row = group_id * 2u + team;
float acc = 0.0f;
if (row < rows) {
    for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
        const float4 x0 = *((device const float4*)(input + col));
        const float4 x1 = *((device const float4*)(input + col + 4u));
        // col%8==0 ⇒ (col*sizeof(float))%16==0. MTLBuffer base is 16B+.
        // ... bits-specific body ...
    }
}
```

### 6.1 bits 3, bound 3, 24 B/group

```metal
        const uint element0 = row * cols + col;
        const uint byte0 = (element0 * 3u) >> 3u;
        const uint packed = uint(codes[byte0])
                          | (uint(codes[byte0 + 1u]) << 8u)
                          | (uint(codes[byte0 + 2u]) << 16u);
        const float scale = float(scales[element0 / 64u]);
        acc += float(int((packed       ) & 7u) - 3) * scale * x0.x;
        acc += float(int((packed >>  3u) & 7u) - 3) * scale * x0.y;
        acc += float(int((packed >>  6u) & 7u) - 3) * scale * x0.z;
        acc += float(int((packed >>  9u) & 7u) - 3) * scale * x0.w;
        acc += float(int((packed >> 12u) & 7u) - 3) * scale * x1.x;
        acc += float(int((packed >> 15u) & 7u) - 3) * scale * x1.y;
        acc += float(int((packed >> 18u) & 7u) - 3) * scale * x1.z;
        acc += float(int((packed >> 21u) & 7u) - 3) * scale * x1.w;
```

Equivalent address, row-aligned groups:
`byte0 = rgb*24u + 3u*((col & 63u) >> 3u)`, `rgb = row*(cols>>6) + (col>>6)`.

Three `uchar` loads, not a `uint` cast. `packed_uchar3` is also legal.

### 6.2 bits 4, bound 7, 32 B/group

```metal
        const uint rgb = row * (cols >> 6u) + (col >> 6u);
        const uint local = col & 63u;                 // 0,8,16,...,56
        const float scale = float(scales[rgb]);
        const uint packed = *((device const uint*)(
            codes + rgb * 32u + (local >> 1u)));      // 4-aligned
        acc += float(int((packed       ) & 15u) - 7) * scale * x0.x;
        acc += float(int((packed >>  4u) & 15u) - 7) * scale * x0.y;
        acc += float(int((packed >>  8u) & 15u) - 7) * scale * x0.z;
        acc += float(int((packed >> 12u) & 15u) - 7) * scale * x0.w;
        acc += float(int((packed >> 16u) & 15u) - 7) * scale * x1.x;
        acc += float(int((packed >> 20u) & 15u) - 7) * scale * x1.y;
        acc += float(int((packed >> 24u) & 15u) - 7) * scale * x1.z;
        acc += float(int((packed >> 28u) & 15u) - 7) * scale * x1.w;
```

Same load as production `:209`. Subtract **7**, not 8.

`float4` X is optional. Eight `input[col+i]` matches `unpack8` and is
enough. `float4` is legal on this map; not required for correctness.

No rem loop on Qwen3.8 GEMVs (`cols % 512 == 0`). Refuse or tail only
if a future tensor breaks `cols % 64 == 0`.

---

## 7. Second-best, and why it lost

**Bits 4: 32-wide `uint4`, `col = lane*32`, stride 2048.**
One 16 B load, 32 nibbles, one scale, two halves of a 64-group.
Lost because (1) it is a different column map than the G0 winner,
(2) under the *current* 8-wide map the same load corrupts 6/8 lanes,
(3) register / occupancy effect is unmeasured, this lane cannot GPU.
REOPEN_IF a locked `decode_probe` on the 17408×5120 organ shows
32-wide faster at equal numeric match.

**Bits 3: 1-wide `gk_uniform_extract_wide`.**
Already correct (`gk_family.metal:179-205`). Lost: 1.25 code-byte
loads + 1 scale + live 2-byte branch on 16/64 elements, vs 0.375
code-byte + 0.125 scale and no branch. That is the incumbent cost.

4-wide lost on both widths: legal subset of 8-wide, worse amortisation,
and bits 3 4-wide is mid-byte on half the starts.

---

## 8. KILLs / REOPEN_IF

| id | claim | evidence |
|---|---|---|
| K1 | `*(device uint*)(codes+byte0)` bits 3 | 4/8 tpr64 lanes unaligned; aligned-down window ≠ oracle (CPU §9) |
| K2 | `*(device uint4*)` at bits 3 group base | odd groups byte `24g % 16 == 8` (CPU) |
| K3 | 10-from-32 bits 3 without carry | word0 match, word1..5 fail; `64%10=4` (CPU) |
| K4 | `unpack8` `- 8` on HGRAVU01 bits 4 | bound is 7 (`:547`); 32/32 windows differ (CPU) |
| K5 | bits 4 `uint4` on 8-wide map | 6/8 lanes `byte % 16 != 0` (CPU) |
| K6 | 4-byte load of last bits 3 8-wide | last group OOB +1 B (CPU) |
| K7 | 8 scale fetches on 8-wide | 8-wide from `%8` never spans g=64 (CPU 0/1250) |
| K8 | reopen representation search | floor is located; out of scope |

REOPEN_IF `cols % 64 != 0` appears: fall back to flat `element` 1-wide
tail, or refuse. HGRAVS L `[2048,160]` is that case; it is not an
HGRAVU01 GEMV on this candidate.

REOPEN_IF a bits 3 decode_probe shows the 3 `uchar` loads as the
remaining limiter: then consider always-2-aligned-`uint` + shift,
which needs a +4 B buffer pad the packer does not emit
(`packed_byte_count` is exact). Do not do this without the pad.

---

## 9. Evidence

### 9.1 SOURCE excerpts

`gk_family.metal:179-205` 1-wide / wide extract + `q = code - bound`:

```
179:205:crates/hawking-core/shaders/gk_family.metal
static inline uint gk_uniform_extract_wide(...)
{
    const uint bit0 = element * bits;
    const uint byte0 = bit0 >> 3u;
    const uint shift = bit0 & 7u;
    uint packed = uint(codes[byte0]);
    if (shift + bits > 8u) {
        packed |= uint(codes[byte0 + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}
...
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
```

`q80_mixed_decode.metal:869-897` incumbent bits 3 8-wide (8 scales):

```
869:897:crates/hawking-core/shaders/q80_mixed_decode.metal
    for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u) {
        const uint byte0 = ((row_base + col) * 3u) >> 3u;
        const uint b0 = uint(codes[byte0]);
        const uint b1 = uint(codes[byte0 + 1u]);
        const uint b2 = uint(codes[byte0 + 2u]);
        const int q0 = int(b0 & 7u) - 3;
        const int q1 = int((b0 >> 3u) & 7u) - 3;
        const int q2 = int(((b0 >> 6u) | (b1 << 2u)) & 7u) - 3;
        const int q3 = int((b1 >> 1u) & 7u) - 3;
        const int q4 = int((b1 >> 4u) & 7u) - 3;
        const int q5 = int(((b1 >> 7u) | (b2 << 1u)) & 7u) - 3;
        const int q6 = int((b2 >> 2u) & 7u) - 3;
        const int q7 = int((b2 >> 5u) & 7u) - 3;
        const float s0 = float(scales[(row_base + col) / group_size]);
        const float s1 = float(scales[(row_base + col + 1u) / group_size]);
        const float s2 = float(scales[(row_base + col + 2u) / group_size]);
        const float s3 = float(scales[(row_base + col + 3u) / group_size]);
        const float s4 = float(scales[(row_base + col + 4u) / group_size]);
        const float s5 = float(scales[(row_base + col + 5u) / group_size]);
        const float s6 = float(scales[(row_base + col + 6u) / group_size]);
        const float s7 = float(scales[(row_base + col + 7u) / group_size]);
        partial += float(q0) * s0 * input[col];
        partial += float(q1) * s1 * input[col + 1u];
        partial += float(q2) * s2 * input[col + 2u];
        partial += float(q3) * s3 * input[col + 3u];
        partial += float(q4) * s4 * input[col + 4u];
        partial += float(q5) * s5 * input[col + 5u];
        partial += float(q6) * s6 * input[col + 6u];
        partial += float(q7) * s7 * input[col + 7u];
```

`q80_mixed_decode.metal:521-528` incumbent bits 4 1-wide:

```
521:528:crates/hawking-core/shaders/q80_mixed_decode.metal
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        ...
        partial += q80_uniform_value_wide(
            codes, scales, element, group_size, bits, bound) * input[col];
```

`qwen_uniform_q4.metal:167-210` G0 8-wide, bound 8:

```
167:210:crates/hawking-core/shaders/qwen_uniform_q4.metal
        sum += float(int(byte & 0x0fu) - 8) * scale * x[col + 2u * i];
        sum += float(int(byte >> 4u) - 8) * scale * x[col + 2u * i + 1u];
...
        for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
            ...
            const uint packed = *((device const uint*)(codes + rgb * 32u + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
```

`q80_mixed_decode.rs:547,563` bound and offset-binary:

```
547:563:crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs
    let bound = (1u16 << (bits - 1)) - 1;
    ...
            let signed = (value / denom).round().clamp(-(bound as f32), bound as f32) as i16;
            codes.push((signed + bound as i16) as u8);
```

`qwen38_hybrid_decode.rs:1687-1690` dispatch split:

```
1687:1690:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
```

`simd8_grid` = `ceil(rows/8)*256`, TG 256 = 8 SG × 32 = **32 tpr**
(`:1042-1043`, `:514-515`).

### 9.2 CPU verifier output (this lane)

Command: `python3` implementing `pack_unsigned_lsb` + `extract_unsigned`
+ simd3 + packed-24 + nibble + `unpack8` ± bound, run against 4 groups.
Full stdout:

```
=== PACK GEOMETRY ===
bits3 group_bytes=24 groups=4 payload=96
bits4 group_bytes=32 groups=4 payload=128
bound3=3 bound4=7

=== BITS3: bitloop vs wide vs simd3 vs packed24, all 8-aligned starts ===
bits3 8-wide matches: 32/32 fail=0

=== BITS3: 8-wide never crosses group-64 ===
group-cross 8-wide starts: 0

=== BITS3: 4-wide at %4 starts DOES cross? and byte-align ===
  e0= 0 bit0=  0 byte0= 0.00 shift=0 groups={0} byte_aligned=True
  e0= 4 bit0= 12 byte0= 1.50 shift=4 groups={0} byte_aligned=False
  e0= 8 bit0= 24 byte0= 3.00 shift=0 groups={0} byte_aligned=True
  e0=12 bit0= 36 byte0= 4.50 shift=4 groups={0} byte_aligned=False
  e0=16 bit0= 48 byte0= 6.00 shift=0 groups={0} byte_aligned=True
  e0=20 bit0= 60 byte0= 7.50 shift=4 groups={0} byte_aligned=False
  e0=24 bit0= 72 byte0= 9.00 shift=0 groups={0} byte_aligned=True
  e0=28 bit0= 84 byte0=10.50 shift=4 groups={0} byte_aligned=False
  e0=32 bit0= 96 byte0=12.00 shift=0 groups={0} byte_aligned=True
  e0=36 bit0=108 byte0=13.50 shift=4 groups={0} byte_aligned=False
  e0=40 bit0=120 byte0=15.00 shift=0 groups={0} byte_aligned=True
  e0=44 bit0=132 byte0=16.50 shift=4 groups={0} byte_aligned=False
  e0=48 bit0=144 byte0=18.00 shift=0 groups={0} byte_aligned=True
  e0=52 bit0=156 byte0=19.50 shift=4 groups={0} byte_aligned=False
  e0=56 bit0=168 byte0=21.00 shift=0 groups={0} byte_aligned=True
  e0=60 bit0=180 byte0=22.50 shift=4 groups={0} byte_aligned=False

=== BITS3: alignment class of tpr64 8-wide (lane*8) ===
  lane=0 e0= 0 byte0= 0 uint_aligned=True word=0 shift= 0 nwords=1
  lane=1 e0= 8 byte0= 3 uint_aligned=False word=0 shift=24 nwords=2
  lane=2 e0=16 byte0= 6 uint_aligned=False word=1 shift=16 nwords=2
  lane=3 e0=24 byte0= 9 uint_aligned=False word=2 shift= 8 nwords=1
  lane=4 e0=32 byte0=12 uint_aligned=True word=3 shift= 0 nwords=1
  lane=5 e0=40 byte0=15 uint_aligned=False word=3 shift=24 nwords=2
  lane=6 e0=48 byte0=18 uint_aligned=False word=4 shift=16 nwords=2
  lane=7 e0=56 byte0=21 uint_aligned=False word=5 shift= 8 nwords=1

=== BITS3: naive aligned-down uint* at byte0 CORRUPTS ===
  lane=0 byte0=0 aligned=0 extra_shift=0 spans_word=False raw24_eq=True shifted_eq=True
  lane=1 byte0=3 aligned=0 extra_shift=24 spans_word=True raw24_eq=True shifted_eq=False
  lane=2 byte0=6 aligned=4 extra_shift=16 spans_word=True raw24_eq=False shifted_eq=False
  lane=3 byte0=9 aligned=8 extra_shift=8 spans_word=False raw24_eq=False shifted_eq=True
  lane=4 byte0=12 aligned=12 extra_shift=0 spans_word=False raw24_eq=True shifted_eq=True
  lane=5 byte0=15 aligned=12 extra_shift=24 spans_word=True raw24_eq=True shifted_eq=False
  lane=6 byte0=18 aligned=16 extra_shift=16 spans_word=True raw24_eq=False shifted_eq=False
  lane=7 byte0=21 aligned=20 extra_shift=8 spans_word=False raw24_eq=False shifted_eq=True
lanes where aligned-down uint cannot yield 8 codes: 4/8

=== BITS3: 10-from-32 without carry CORRUPTS after word 0 ===
group0 words=['0x43c67543', '0x7543c675', '0xc67543c6', '0x43c67543', '0x7543c675', '0xc67543c6']
  word0 indep10=[3, 0, 5, 2, 7, 4, 1, 6, 3, 0] ref_if_e0=0:[3, 0, 5, 2, 7, 4, 1, 6, 3, 0] eq=True true_starts_in_word=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  word1 indep10=[5, 6, 1, 3, 4, 7, 0, 2, 5, 6] ref_if_e0=10:[5, 2, 7, 4, 1, 6, 3, 0, 5, 2] eq=False true_starts_in_word=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
  word2 indep10=[6, 0, 7, 1, 4, 2, 5, 3, 6, 0] ref_if_e0=20:[7, 4, 1, 6, 3, 0, 5, 2, 7, 4] eq=False true_starts_in_word=[22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  word3 indep10=[3, 0, 5, 2, 7, 4, 1, 6, 3, 0] ref_if_e0=30:[1, 6, 3, 0, 5, 2, 7, 4, 1, 6] eq=False true_starts_in_word=[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
  word4 indep10=[5, 6, 1, 3, 4, 7, 0, 2, 5, 6] ref_if_e0=40:[3, 0, 5, 2, 7, 4, 1, 6, 3, 0] eq=False true_starts_in_word=[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
  word5 indep10=[6, 0, 7, 1, 4, 2, 5, 3, 6, 0] ref_if_e0=50:[5, 2, 7, 4, 1, 6, 3, 0, 5, 2] eq=False true_starts_in_word=[54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
64 % 10 = 4 remainder codes if 10-wide

=== BITS3: carry-chain 10-wide over 6 words ===
full-group 64 from 24B match=True
after 6*10=60 codes, leftover_bits=12 leftover_codes=4

=== BITS3: 128-bit alignment at group starts ===
  group=0 byte=  0 align4=True align8=True align16=True
  group=1 byte= 24 align4=True align8=True align16=False
  group=2 byte= 48 align4=True align8=True align16=True
  group=3 byte= 72 align4=True align8=True align16=False
  group=4 byte= 96 align4=True align8=True align16=True
  group=5 byte=120 align4=True align8=True align16=False
  group=6 byte=144 align4=True align8=True align16=True
  group=7 byte=168 align4=True align8=True align16=False

=== BITS3: last 8-wide 4-byte over-read ===
  last e0=248 byte0=93 payload_len=96 bytes_needed=96 uint_end=97 oob_uint=True
  3-byte in-bounds=True

=== BITS4: bitloop vs wide vs nibble vs unpack8-bound7 ===
bits4 1-wide matches: 256/256 fail=0

=== BITS4: 8-wide unpack8-style vs shift4, bound 7 vs 8 ===
bits4 8-wide match bound7: 32/32 fail=0
bound7 vs bound8 differ at 32/32 windows (must be all)

=== BITS4: extract_wide second-byte branch taken? ===
bits4 wide 2-byte branch taken in 64 elements: 0
bits3 wide 2-byte branch: taken 16/64 elements [2, 5, 10, 13, 18, 21, 26, 29, 34, 37, 42, 45, 50, 53, 58, 61]

=== BITS4: 8-wide alignment and group stay ===
tpr64 8-wide bits4 uint-misaligned: 0/64  group-cross: 0/64

=== BITS4: 128-bit under 8-wide mapping alignment ===
  lane=0 local= 0 byte= 0 align16=True
  lane=1 local= 8 byte= 4 align16=False
  lane=2 local=16 byte= 8 align16=False
  lane=3 local=24 byte=12 align16=False
  lane=4 local=32 byte=16 align16=True
  lane=5 local=40 byte=20 align16=False
  lane=6 local=48 byte=24 align16=False
  lane=7 local=56 byte=28 align16=False
uint4-misaligned under 8-wide mapping: 6/8

=== BITS4: 128-bit under 32-wide mapping ===
  lane=0 col=0 group=0 local=0 byte=0 align16=True same_group_32=True
  lane=1 col=32 group=0 local=32 byte=16 align16=True same_group_32=True
  lane=2 col=64 group=1 local=0 byte=32 align16=True same_group_32=True
  lane=3 col=96 group=1 local=32 byte=48 align16=True same_group_32=True

=== QWEN38 GEMV COLS vs 64 and 512 ===
  gate/up     17408x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  down         5120x17408  cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  qkvz        16384x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  ba             96x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  dn.out       5120x6144   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  gqa.q       12288x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  gqa.kv       1024x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  gqa.o        5120x6144   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True
  lm_head    248320x5120   cols%64=0 cols%8=0 cols%512=0 elems%64=0 groups_row_aligned=True

=== SCALE INVARIANT: 8 consec from %8 never need 2 scales (g=64) ===
8-wide starts in 0..9992 that span 2 groups: 0
```

Op table in that run is DERIVED, not measured.

### 9.3 Not claimed

- No GPU timing. No token TPS. The circulating ~31 TPS for a fixed
  kernel is a **projection** from 11.47 GB at 639.25 GB/s, not a wall.
- Occupancy of 32-wide vs 8-wide: unknown. Cheapest: lane 110
  `decode_probe` A/B on one 17408×5120 organ.
- Whether three `uchar` loads coalesce as well as one `uint` on M3
  Ultra: unknown. Correctness ranks first; 3-byte is the only
  bits 3 load that is both aligned and in-bounds without a pad.
