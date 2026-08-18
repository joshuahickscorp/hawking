# G1 Uniform bits-3 kernel — why mixed-q3mlp-v1 is 3.78× slower than G0

Lane: `100-q3-uniform-kernel`. No GPU. No generate. No inference. No
repack. One new file. Live G0 / resident Genesis not touched.

Every number is **MEASURED**, **RECEIPT**, **SOURCE**, **DERIVED**,
**INFERRED**, or **ESTIMATED**. An isolated-organ microbenchmark is not
a token. A projected TPS is not a complete-token wall.

---

## 0. Verdict

The suspected cause is **CONFIRMED for the 192 bits=3 MLP GEMVs** and
**INCOMPLETE as the token genome**.

`dispatch_factor` at `qwen38_hybrid_decode.rs:1687-1692` (SOURCE):

- `bits == 3` → `q80_hgravs01_factor_matvec_simd3`, `simd8_grid`, TG 256.
  Kernel column stride is **256** (`q80_mixed_decode.metal:869`).
- `bits == 4` → `q80_hgravs01_factor_matvec_simd`, same launch, **1-wide**
  extract, column stride **32** (`q80_mixed_decode.metal:521-528`).
- G0 production is `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`,
  TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row, column stride **512**
  (`qwen_uniform_q4.metal:183-221`, `qwen38_hybrid_decode.rs:547-582`).

mixed-q3mlp-v1 is 192× bits=3 **and** 305× bits=4 GEMVs (catalog parse
this process). `q4=0`. Nothing on this token uses geo_tpr64.

The 148.6 ms wall is a **weight-GEMV explosion**. Ceremony did not
move. MEASURED wall−GPU = 1,625,793 ns. INFERRED GEMV bucket =
137,099,341 ns (92.3 % of wall) against G0 seated addressing
21,293,103 ns (60.44 %). Over 130 ms. Mechanism: **32 threads/row +
narrow columns/pass + per-element scale fetch on bits=4 + 1-wide
extract on every bits=4 GEMV**. Not uncoalesced happy-path reads, not
a serial reduction, not dispatch count (secondary).

The fix is a geo_tpr64-class pair that consumes the **existing**
HGRAVU01 packed bytes. No repack. bits=4 is the G0 Q4 layout with
`bound=7` instead of `nibble-8`. bits=3 is 24 code bytes/group, same
group geometry.

If that kernel class merely matched G0's, the artifact is a promotion
candidate on BPW and on projected TPS. That TPS is **PROJECTED**, not
a complete-token measurement.

---

## 1. Suspected cause — confirm / refute

### 1.1 Production switch (SOURCE)

```1680:1692:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
            if qwen38_recon_fuse_enabled() {
                let (name, grid) = if bits == 8 {
                    if cols >= 2048 {
                        ("q80_uniform8_matvec_tg256", tg256_grid(rows))
                    } else {
                        ("q80_uniform8_matvec_simd_bytes", simd8_grid(rows))
                    }
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), encode)
```

`qwen38_recon_fuse_enabled` is `env_opt_out("HAWKING_QWEN38_RECON_FUSE")`
(`qwen38_hybrid_decode.rs:43-45`). Default **ON** (`lib.rs:207-214`).
Lane-92 load log: `recon_fuse=ON` (RECEIPT
`g1-mlp-family-generate.md:522`).

`dispatch_uniform` is `dispatch_factor` with the tensor's `bits`
(`qwen38_hybrid_decode.rs:1742-1760`). Production `encode_*_mixed`
reaches it through `encode_named_matvec` (`1483-1518`, `3210-3574`).

### 1.2 simd3 launch and stride (SOURCE)

`simd8_grid(rows) = (rows.div_ceil(8)*256, 1, 1)` (`1042-1044`).
TG = (256,1,1). 8 simdgroups/TG, **1 row / simdgroup**, **32 threads/row**.

```859:869:crates/hawking-core/shaders/q80_mixed_decode.metal
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    if (row >= rows || bits != 3u) {
        return;
    }
    // ...
    for (uint col = simd_lane * 8u; col + 8u <= cols; col += 256u) {
```

Column stride **256**. 8-wide 3-byte unpack. Then **8 separate scale
loads** (`882-889`) even though `col % 8 == 0` and `group_size=64`
keep those 8 codes in one group (DERIVED: `local ∈ {0,8,...,56}` ⇒
`local+7 ≤ 63`).

`decode_family.rs:24-27` names this tile as the 3-bit occupancy tile
of the hgravs **factor** kernel — written for r160 bodies, reused
here for full Uniform matrices.

### 1.3 bits=4 path is not simd3 (SOURCE)

```513:528:crates/hawking-core/shaders/q80_mixed_decode.metal
    constexpr uint kSimdgroupsPerThreadgroup = 8u;
    constexpr uint kSimdWidth = 32u;
    const uint row = group_id * kSimdgroupsPerThreadgroup + simd_id;
    // ...
    for (uint base = 0u; base < cols; base += kSimdWidth) {
        const uint col = base + simd_lane;
        // ...
        partial += q80_uniform_value_wide(
            codes, scales, element, group_size, bits, bound) * input[col];
```

32 TPR, **1 weight / thread / pass**, stride 32, one `gk_uniform_extract_wide`
+ one scale per element (`gk_family.metal:179-205`).

### 1.4 G0 production genome (SOURCE + RECEIPT)

```181:221:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
    // 64 threads/row, 128-thread TG, 2 rows/TG
    // col = lane_in_row * 8; col += 512
```

`Qwen38MatvecKernel::GeoTpr64Tg128` launch: `grid = rows.div_ceil(2)*128`,
TG 128 (`qwen38_hybrid_decode.rs:547-582`). G024 genome string names
this kernel (`QWEN38_TOKEN_NS_LEDGER.json` `kernel_runtime_genome`).

### 1.5 What the artifact actually is (MEASURED this process)

Root: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`

HQ38M20 catalog parse (this process, `QWEN38_MIXED_RECORD_SIZE=128`
parser at `qwen38_hybrid_decode.rs:411-488`):

| quantity | value | tag |
|---|---:|---|
| tensors | 851, all codec 3 | MEASURED |
| complete physical BPW `8*nbytes/N` | 3.6138111608720234 | MEASURED |
| MLP BPW | 3.2500251321231617 | MEASURED |
| fused `in_proj_qkvz` names | **0** | MEASURED |
| split `in_proj_qkv` | 48 | MEASURED |

GEMV header census (this process, peek `HGRAVU01` JSON):

| organ | n | bits | g | shape | kernel via `dispatch_factor` |
|---|---:|---:|---:|---|---|
| gate, up | 64+64 | 3 | 64 | 17408×5120 | **simd3** |
| down | 64 | 3 | 64 | 5120×17408 | **simd3** |
| in_proj_qkv | 48 | 4 | 64 | 10240×5120 | **simd** |
| in_proj_z | 48 | 4 | 64 | 6144×5120 | **simd** |
| in_proj_a, b | 48+48 | 4 | 64 | 48×5120 | **simd** |
| out_proj | 48 | 4 | 64 | 5120×6144 | **simd** |
| q / k / v / o | 16×4 | 4 | 64 | G0 shapes | **simd** |
| lm_head | 1 | 4 | 64 | 248320×5120 | **simd** |
| embed | 1 | 4 | 64 | 248320×5120 | `qwen38_hgravu_embedding_lookup` |

Lane-92 load census (RECEIPT `g1-mlp-family-generate.md:143`):

```
uniform=498 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
```

498 = 497 GEMV + 1 embed. 353 HGRAVU01 vectors dequant to f32 at load
(`hgravu_is_vector`, `qwen38_hybrid_decode.rs:139-157, 976-977`). Not
a GEMV expand. `q4=0` ⇒ geo_tpr64 is never bound.

**Refute as sole cause:** 305 of 497 GEMVs, including lm_head, never
enter simd3. They enter the 1-wide simd tile. The suspected simd3
genome is the MLP half, not the token.

---

## 2. Dispatch record

No per-kernel GPU dump exists for the 148.6 ms run. Lane 92 recorded
the complete-token wall and the load census, not a TCB name list.
The record below is **reconstructed** from `encode_*_mixed` (SOURCE)
+ the catalog (MEASURED). It is the production graph that run executed.

`encode_layers` → `encode_deltanet` / `encode_gqa` / `encode_dense_mlp`
all divert to `*_mixed` when `mixed` is nonempty
(`qwen38_hybrid_decode.rs:2834-2842, 2886-2892, 2992-2994, 3107-3115`).
Split DN in_proj is taken because fused names are absent
(`3264-3284`, `1812-1847`).

### 2.1 Per-token dispatch census (DERIVED)

| class | kernel | n | launch |
|---|---|---:|---|
| embed | `qwen38_hgravu_embedding_lookup` | 1 | grid (5120,1,1) TG 256 |
| DN rmsnorm | `qwen80_residual_rmsnorm_f32` | 48 | TG 256 |
| DN GEMV qkv | simd bits=4 | 48 | simd8_grid(10240) |
| DN GEMV z | simd bits=4 | 48 | simd8_grid(6144) |
| DN fuse | `qwen38_fuse_split_qkvz_f32` | 48 | grid (16384,1,1) TG 256 |
| DN GEMV b, a | simd bits=4 | 48+48 | simd8_grid(48) = **6 TG** |
| DN fuse | `qwen38_fuse_split_ba_f32` | 48 | grid (96,1,1) TG 32 |
| DN rearrange / ba / delta / gnorm | same as G0 | 48×4 | same as G0 |
| DN out | simd bits=4 | 48 | simd8_grid(5120) |
| DN residual | `qwen_next_add_residual` | 48 | same as G0 |
| GQA rms + 4×GEMV + rope + mha + sig + residual | GEMVs = simd bits=4 | 16×9 | G0 extras; GEMV simd8_grid |
| MLP rms + gate/up simd3 + swiglu + down simd3 + residual | 64×6 | gate/up/down = simd3 |
| terminal rms + lm_head simd bits=4 + argmax | 3 | lm = simd8_grid(248320) |

Totals (DERIVED):

| | G0 | q3mlp |
|---|---:|---:|
| GEMV dispatches | 401 | **497** |
| fuse dispatches | 0 | **96** |
| other (rms/delta/gqa/swiglu/resid/embed/argmax) | 563 | 563 |
| **token dispatches** | **964** | **1156** |
| production CBs | 1 | 1 |

DN mixer prefix 13 vs G0 9 (`+4` = two extra GEMVs + two fuses).
`48*13 + 16*9 + 64*6 + 1 + 3 = 1156`.

### 2.2 GEMV launch geometry vs G0 (DERIVED)

`TGs = rows/8` (simd family) vs `rows/2` (geo). Threads/row 32 vs 64.
Col-pass width 256 (simd3) or 32 (simd) vs 512 (geo).

| organ | simd-family TGs | geo TGs | trips/thread K=5120 |
|---|---:|---:|---|
| gate/up 17408 | 2176 | 8704 | simd3: 20 of 8-wide; geo: 10 of 8-wide |
| down K=17408 | 640 | 2560 | simd3: 68; geo: 34 |
| qkv 10240 | 1280 | 5120 | simd: 160 of 1-wide; geo: 10 of 8-wide |
| a/b 48 | **6** | 24 | simd: 160; geo: 10 |
| lm_head 248320 | 31040 | 124160 | simd: 160; geo: 10 |

simd8_grid(48) = 6 threadgroups. Occupancy-starved. Secondary.

### 2.3 Bind log (RECEIPT, not a GEMV record)

```
qwen38-decode mixed bind: K-complete; recon_fuse=ON uses
q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes
when cols>2048 ...
```

(`g1-mlp-family-generate.md:522`). This string is emitted for every
mixed open (`qwen38_mixed_k_complete_bind_message`, `:124-136`). q3mlp
has **zero** binary/CSR tensors. It does not describe the Uniform
kernels actually dispatched. Authority for Uniform is `dispatch_factor`,
not this line.

---

## 3. Decompose 148.6 ms

### 3.1 What was measured on this artifact (RECEIPT)

Lane 92 complete-wall, 3 A/B pairs, 6 warm reps, 31 decode steps,
`timing_label=DIRTY_ENGINEERING` (`g1-mlp-family-generate.md:318-371`):

```
complete_wall  148,588,917 ns   6.7300 TPS
GPU            146,963,124 ns
wall − GPU       1,625,793 ns
rep medians    148424792 148588917 148460333 148480250 148721583 148600250
spread         0.297 ms
fallbacks      0
dense W        0
```

No addr_probe / decode_probe / isolated-class GEMV CB was run on
q3mlp. Those probes exist only as Q4 kernel names
(`QWEN38_Q4_ADDR_PROBE_KERNEL`, `measure_isolated_class_gemvs_kernel`
→ `encode_q4_matvec_kernel`). A G0-identical 12-row TOKEN_NS of this
token **does not exist**.

### 3.2 Ceremony — MEASURED, did not explode

| | G0 seated | q3mlp | tag |
|---|---:|---:|---|
| wall | 35,227,917 | 148,588,917 | RECEIPT / RECEIPT |
| GPU | 33,912,333 | 146,963,124 | RECEIPT / RECEIPT |
| wall−GPU | 1,315,584 | **1,625,793** | DERIVED / MEASURED |
| cited ceremony | 1,653,000 (4.69 %) | ≈ wall−GPU | RECEIPT / MEASURED |

G0 ceremony cited in the contract is host_prep + submit + sync +
named residual (`QWEN38_TOKEN_NS_LEDGER.json`: 919,250 + 12,084 +
384,250 + 341,925 = 1,657,509; cited 1,653,000 is that group).
q3mlp wall−GPU is the same class of host/sync tax and is **1.6 ms**.
+192 extra encoders cannot be the 109 ms gap.

Today's live G0 (contract): 39,326,090 ns / 25.4284 TPS. Ratio
148,588,917 / 39,326,090 = **3.778×** (DERIVED). That is the
headline slowness. It is GPU.

### 3.3 Non-GEMV GPU — APPLIED from G0 isolated, same shaders

These kernels and launches are identical on the mixed path
(rmsnorm, rearrange, ba, gated_delta_vi, gated_rmsnorm, rope, mha,
sigmoid, swiglu, residuals, argmax, f32 stream). Isolated medians
from `QWEN38_TOKEN_NS_LEDGER.json` `isolated` (RECEIPT):

| isolated | ns | n |
|---|---:|---:|
| input_norms + post_norms + final_norm | 2,367,415 | 129 |
| silu_64 + mlp_residual_64 | 295,166 | 128 |
| mixer_residual_64 | 118,250 | 64 |
| rearrange + ba_to_decay + gated_rmsnorm + gated_delta | 3,932,039 | 192 |
| rope + mha + sigmoid | 2,272,750 | 48 |
| argmax + embed | 340,498 | 2 |
| kv streams rec+conv+k+v | 537,665 | 4 |
| **sum** | **9,863,783** | |

Tag: **APPLIED**. Same SOURCE launch. Not re-timed on q3mlp. Box
load may move them ~10 %. They cannot absorb 109 ms.

New on q3mlp: 96 fuse kernels (`qwen38_fuse_split_qkvz_f32` /
`qwen38_fuse_split_ba_f32`). f32 copies of 16384 and 96. ESTIMATED
≪ 1 ms. Not the bucket.

### 3.4 Weight GEMV bucket — INFERRED, exploded

```
q3mlp GPU                 146,963,124   MEASURED
− applied non-GEMV          9,863,783   APPLIED
= GEMV + fuse + error     137,099,341   INFERRED
```

| | ns | % of 148.6 ms wall | tag |
|---|---:|---:|---|
| **weight GEMV (addr+decode+FMA)** | **137,099,341** | **92.27** | INFERRED |
| deltanet compute (no GEMV) | 3,932,039 | 2.65 | APPLIED isolated |
| gqa compute (no GEMV) | 2,272,750 | 1.53 | APPLIED isolated |
| ceremony (wall−GPU) | 1,625,793 | 1.09 | MEASURED |
| other non-GEMV (norm/swiglu/kv/argmax/embed/resid) | 3,658,994 | 2.46 | APPLIED isolated remainder |

G0 seated comparison (RECEIPT `QWEN38_TOKEN_NS_LEDGER.json`):

| G0 component | ns | % of 35.2 ms |
|---|---:|---:|
| weight_addressing | 21,293,102.52 | 60.44 |
| weight_decode_reconstruction | 1,808,227.35 | 5.13 |
| isolated GEMV sum mlp+dn+gqa+lm | 24,249,289 | — |

137.1 ms vs 21.29 ms addressing = **6.44×**. vs 24.25 ms isolated
GEMV = **5.65×**. The equivalent of G0 weight addressing is **over
130 ms**. That is the exploded bucket.

G0 `dense_swiglu` / `deltanet` / `gqa` / `terminal_head` **components**
each contain a GEMV FMA remainder. They are not reusable as
"compute stayed put" once the GEMV kernel changes. The isolated
non-GEMV table above is the honest reuse.

### 3.5 Which GEMV family — ESTIMATED, not isolated

No MLP-vs-mixer GPU split was taken. `measure_isolated_mlp_matvecs`
and `measure_isolated_mixer_gemvs` already divert to mixed
(`2010-2018`, `1950-1957`, `2104-2164`). They were not run.
`measure_isolated_lm_head` still calls `encode_q4_matvec` (`2166-2174`)
and would **refuse** this artifact (`q4=0`).

Issue-trip model (ESTIMATED, not a token):

| family | G0 isolated | trip×TPR vs geo | ESTIMATED ns |
|---|---:|---|---:|
| MLP 192× simd3 | 15,853,666 | 2× trips, ½ TPR → ~4× | ~63 ms |
| mixer+lm 305× simd | 8,395,623 | 16× trips, ½ TPR → ~8× | ~67 ms |
| sum | 24,249,289 | | ~130 ms |

Residual vs 137.1 ms ≈ 7 ms (split launches, fuses, box, model error).
Both families exploded. A bits=3-only kernel does not drain the
bucket. bits=4 1-wide extract on qkv / out / q / lm_head is the
same order as the 192 simd3 MLP GEMVs.

Q80 discriminator (RECEIPT `Q80_MATVEC_DISCRIMINATOR.json`) measured
the **serial** `q80_hgravs01_factor_matvec` (1 thread/row) at
16 GB/s packed on large Q8 organs and named `gk_uniform_extract`
as issue-bound. q3mlp is already on the simd tiles, not that serial
kernel. Those 16 GB/s figures are **not** this token. They only
show the family is reconstruction/issue limited when TPR is low
and extract is 1-wide — the same regime bits=4 simd still occupies
at TPR 32.

---

## 4. Mechanism

Named against the contract list. Authority: SOURCE loops + INFERRED
bucket + APPLIED G0 isolated.

| candidate | on this token |
|---|---|
| **threads per row** | **YES.** 32 vs G0 64. Half the latency-hiding along K. |
| **columns per pass** | **YES.** 256 (simd3) or 32 (simd) vs G0 512. |
| uncoalesced reads | NO on the happy path. simd3 32×8-wide tiles are 256 consecutive cols; simd 32 consecutive elements. Codes are a flat LSB stream. |
| **scale fetch per element** | **YES on bits=4** (one `scales[e/64]` per FMA). simd3 source issues 8 scale loads per 8-wide tile (`882-889`); CSE may collapse them (unverified). G0: one scale per 8-wide uint. |
| serial reduction | NO. Both families `simd_sum`. G0 adds a 2-way TG add. |
| low occupancy | CO-LIMITER on 48-row a/b only (6 TG). Large organs launch thousands of TGs. Not the 137 ms. |
| dispatch count | SECONDARY. 1156 vs 964. Ceremony MEASURED 1.6 ms. |

Not bytes. MLP payload is **lower** density (3.250 vs 4.253).
Attention/lm_head payload density matches G0 Q4. A bandwidth-bound
geo_tpr64 token at 639.25 GB/s (G0 addressing RECEIPT) would go
**down**, not up 3.78×.

Bytes/weight (DERIVED from L0 gate header this process):

| | codes | scales | total |
|---|---:|---:|---:|
| HGRAVU01 bits=3 g64 | 0.375 | 0.03125 | **0.40625** |
| HGRAVU01 bits=4 g64 | 0.5 | 0.03125 | **0.53125** |
| HQ30UQ4 g64 | 0.5 | 0.03125 | **0.53125** |

L0 gate header (MEASURED): `code_bytes=33423360`, `scale_bytes=2785280`,
`elements=89128960`, `groups=1392640`, `retained_padding_elements=0`,
body = scales ‖ codes, `scale_dtype=float16`.

Streamed GEMV catalog bytes (MEASURED, exclude embed table + 353
vectors): 12,149,632,429 − 675,430,686 − 1,496,097 = **11,472,705,646**.
G0 theoretical GEMV 13,611,663,360 (`g1-direct-gemv-geometry.md` §1.3).
Ratio 0.843. **15.7 % fewer streamed GEMV bytes** than G0. Complete
BPW ratio 3.6138/4.2527 = 0.850 (15.0 % fewer). Lower density,
slower token ⇒ execution genome.

---

## 5. Kernel design — geo_tpr64 for Uniform bits 3 (and 4)

One family, two specializations (or one Metal kernel with
function-constant `bits`). Do not expand to float or HQ30UQ4.

Name (proposed):

- `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`
- `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128` (bound=7)

G0 Q4 kernel **cannot** be reused for bits=4: it hardcodes
`q = nibble - 8` (`qwen_uniform_q4.metal:64, 175-176`). HGRAVU01
bound is `(1<<(bits-1))-1` = **7** (`q80_mixed_decode.rs:1164`).

### 5.1 Thread mapping — copy G0 (SOURCE `qwen_uniform_q4.metal:195-220`)

```
TG            = 128
simdgroups/TG = 4
rows/TG       = 2
threads/row   = 64          // 2 simdgroups
grid          = (ceil(rows/2)*128, 1, 1)

team          = simd_id / 2
split         = simd_id % 2
lane_in_row   = split*32 + simd_lane     // 0..63
row           = group_id*2 + team
col           = lane_in_row * 8 ; col += 512
```

Reduction: `simd_sum` per half-row, TG add of the two halves, lane 0
writes `output[row]`. Packed decode stays in registers. Never a dense W.

### 5.2 Per-thread decode — bits=3, existing packed bytes

Preconditions (DERIVED, this model, all GEMV K ∈ {5120,6144,17408}):

- `cols % 64 == 0` ⇒ groups are row-aligned.
- `cols % 8 == 0` and `lane_in_row*8 % 8 == 0` ⇒ each 8-wide tile
  starts on a byte boundary of the LSB bitstream
  (`bit0 = (row*cols+col)*3`, `col=8m` ⇒ `bit0=24m`).
- 64×3 bits = 24 code bytes/group.

Sequence, one 8-wide tile (same unpack as simd3 `:870-897`, one scale):

```
group   = col / 64
local   = col % 64                         // ∈ {0,8,...,56}
rgb     = row * (cols/64) + group
scale   = float(scales[rgb])               // one f16, not eight
byte0   = rgb * 24 + (local * 3) / 8       // 0,3,6,...,21
load 3 bytes at codes[byte0 .. byte0+2]
q0 = (b0     ) & 7  − 3
q1 = (b0 >> 3) & 7  − 3
q2 = ((b0 >> 6) | (b1 << 2)) & 7  − 3
q3 = (b1 >> 1) & 7  − 3
q4 = (b1 >> 4) & 7  − 3
q5 = ((b1 >> 7) | (b2 << 1)) & 7  − 3
q6 = (b2 >> 2) & 7  − 3
q7 = (b2 >> 5) & 7  − 3
acc += Σ_i float(qi) * scale * input[col+i]
```

`bound=3` is the header contract (`factor_layout_from_meta`). Do not
hardcode a different offset.

bits=4 sibling: `byte0 = rgb * 32 + local/2`, load `uint32` (8 nibbles),
`q = nibble - 7`. Nibble order = even local low, odd high = LSB-first
4-bit (`gk_uniform_extract_wide`). Same as HQ30UQ4 packing, different
offset.

### 5.3 Bytes moved per output element (DERIVED)

Unique W traffic, payload only (header stays on disk):

| organ | K | bits | B / out row |
|---|---:|---:|---:|
| gate, up, q, k, v, qkv, z, lm | 5120 | 3 or 4 | bits3: **2080**; bits4: **2720** |
| down | 17408 | 3 | **7072** |
| out, o | 6144 | 4 | **3264** |
| a, b | 5120 | 4 | **2720** |

Plus one f32 input vector of length K, reused across rows of the same
GEMV. G0 Q4 gate is 2720 B/row (`groups*34 = 80*34`). bits=3 gate is
2080 = 80×26 (24 code + 2 scale).

Token streamed W: 11.473 GB catalog GEMV (MEASURED, §4). At G0
addressing 639.25 GB/s (RECEIPT) that is **17.95 ms** PROJECTED
addressing, not a measured token.

### 5.4 Predicted dispatch count (DERIVED)

Same GEMV count. One dispatch per GEMV. **497** geo launches/token
on this pack (or **401** if a later pack emits fused qkvz/ba, which
`encode_deltanet_mixed` already prefers).

Plus the same 96 fuses (until fused pack) and 563 non-GEMV.
Token total stays **1156** until the pack fuses in_proj.

Grid grows (more TGs, fewer trips). Dispatch count does not.

### 5.5 Host bind

`dispatch_factor` (`1680-1692`) grows two arms:

```
bits == 3 → qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
            grid = ceil(rows/2)*128, TG 128
bits == 4 → qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
            same launch
else      → keep current simd / uniform8
```

Buffers already match: `codes, scales, input, output, rows, cols,
group_size, bits, bound` (`encode_factor_args` `:1576-1598`). geo
needs `groups_per_row = cols/64` as an extra u32 or derives it.
No new resident layout.

Keep simd3/simd as `HAWKING_QWEN38_RECON_FUSE=0` / diagnostic
fallback. Do not delete them; they remain the r160 factor path.

---

## 6. Layout — no repack

HGRAVU01 body (SOURCE `factor_layout_from_meta` `:1134-1168`,
MEASURED L0 gate header):

```
[HGRAVU01][u32 header_len][JSON][f16 scales[groups]][LSB packed codes]
groups = ceil(elements / 64)
code_bytes = ceil(groups * 64 * bits / 8)
bound = (1<<(bits-1)) - 1
```

LSB-first unsigned (`pack_unsigned_lsb`, `q80_mixed_decode.rs:93-95`).
Flat row-major elements. `retained_padding_elements=0` on L0 gate.

This stream **feeds geo_tpr64 directly** when `cols % 64 == 0` (true
for every GEMV on this model). bits=3 8-wide tiles are byte-aligned
(§5.2). bits=4 tiles are nibble-aligned the same way as HQ30UQ4.

**Pack layout must not change.** A repack would redo 12.15 GB and
change the economics. The only optional pack change is **fusing**
in_proj qkv+z and a+b into the names `encode_deltanet_mixed` already
checks — same codec, same bytes, fewer launches. Not required for
the kernel.

KILL: expand HGRAVU01 → HQ30UQ4 or f32, then G0 geo. Standing bind
rule. Two prior INCOHERENT verdicts in this campaign were that
confound. REOPEN_IF a complete-token A/B under GPU lock shows the
expand is still a net physical win. Do not assume it.

---

## 7. What promotion takes

q3mlp is already **COHERENT** on the campaign gate (RECEIPT lane 92:
France 128 contains Paris ×8; 17×19 256 contains 323 ×3; fallbacks 0;
dense W 0). Quality is not G0: it loops after the fact. Not a 6/6
oracle-32 seal.

A coherent 3.6138 BPW artifact that **merely matched the G0 kernel
class** is a promotion candidate because it moves ~15 % fewer bytes
at the incumbent addressing regime.

PROJECTED token if geo_tpr64 lands at G0's 639.25 GB/s on 11.473 GB
and non-GEMV stays APPLIED:

| piece | ns | tag |
|---|---:|---|
| addressing 11.473e9 / 639.25e9 | 17,947,000 | PROJECTED |
| decode+FMA ~ G0 3.0 ms × 0.843 | 2,530,000 | ESTIMATED |
| non-GEMV isolated | 9,863,783 | APPLIED |
| ceremony | 1,625,793 | MEASURED (today's dirty box) |
| **sum** | **~32.0 ms** | **PROJECTED ~31 TPS** |

Today's live G0 is 39.3 ms / 25.43 TPS. Seated G0 was 35.2 ms. The
32 ms figure is a **seated-genome projection on today's ceremony**.
It is not a complete-token wall. Do not write it on a scoreboard.

What it takes, in order:

1. Ship the two geo kernels + `dispatch_factor` arms. No repack.
2. Under the GPU lock, isolated CBs that already exist:
   `measure_isolated_mlp_matvecs`, `measure_isolated_mixer_gemvs`,
   `measure_isolated_mlp_full`, `step_decomposed`. One-line:
   `measure_isolated_lm_head` must call `encode_named_matvec`.
3. Complete-token A/B vs G0, 6-rep median, same harness as lane 92.
   Authority is that wall, not the projection above.
4. Keep fallbacks=0, expanded_to_q4=0, expanded_to_float_gemv=0.
5. Oracle-32 6/6 if promotion requires more than the campaign
   substring gate.

Optional, after (3): fused in_proj pack (−96 GEMV −96 fuse) to
return to 964 dispatches.

---

## 8. KILLS / REOPEN_IF

| item | status |
|---|---|
| "simd3 stride 256 is the whole token" | **INCOMPLETE.** 305 bits=4 GEMVs are 1-wide simd. REOPEN only as a MLP-only claim. |
| expand-to-Q4-then-geo as production | **KILL.** Bind rule. REOPEN_IF complete-token A/B shows net physical win. |
| "lower BPW must be faster" | **KILL** under this genome. Roof is conditioned on execution, not the spec sheet. REOPEN_IF geo_tpr64 is bound and the token is still slower — then reopen representation. |
| new pack / new 2 BPW artifact to fix speed | **KILL** for this question. The bytes are already there. REOPEN_IF geo_tpr64 is bound and the remaining gap is bytes. |
| serial `q80_hgravs01_factor_matvec` as the cause | **KILL.** recon_fuse=ON. That kernel is the `HAWKING_QWEN38_RECON_FUSE=0` path. |
| 1-bit/2-bit RTN, VQ, generator+residual, entropy under this quantizer | still dead. Unchanged premise. |

---

## 9. Cheapest experiment this lane did not run

GPU lock is held by serialized lanes. This lane must not take it.

Cheapest: `ascension_qwen38_token_ns` (or the isolated methods in
§7.2) against `mixed-q3mlp-v1`. That yields MEASURED mlp / mixer /
terminal GEMV ns on the real kernels. Then bind the geo pair and
repeat. Complete-wall last.

`measure_isolated_class_gemvs_kernel(..., addr_probe)` is Q4-only.
simd3 already has `q80_hgravs01_factor_matvec_simd3_addr_probe`
(`q80_mixed_decode.metal:1585`). Wire those for a G0-style
addr/decode split **after** the geo kernels exist; do not treat
the Q80 factor addr_probe as a Qwen3.8 token.

---

## 10. Evidence

### 10.1 Catalog parse (this process)

```
ver 1 n_tensors 851 n_segments 258
codec counts Counter({3: 851})
roles: gate 64, up 64, down 64,
       in_proj_qkv 48, z 48, a 48, b 48, out 48,
       q 16, k 16, v 16, o 16, embed 1, lm_head 1, other 353
fused qkvz names: 0
complete BPW 3.6138111608720234
L0 gate HGRAVU01 bits=3 g=64 code_bytes=33423360 scale_bytes=2785280
L3 q    HGRAVU01 bits=4 g=64
```

### 10.2 Lane 92 wall (RECEIPT)

`/Users/scammermike/.claude-grok/worktrees/92-mlp-family-generate-20260817-121917/workspace/superwave/g1/g1-mlp-family-generate.md:32-36,143,342-347`

### 10.3 G0 TOKEN_NS (RECEIPT)

`receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json` via `git show`.
`median_wall_ns=35227917`, `median_gpu_ns=33912333`, components as §3.4.

### 10.4 Source pointers

- `dispatch_factor` bits split: `qwen38_hybrid_decode.rs:1680-1692`
- simd3 stride 256: `q80_mixed_decode.metal:845-908`
- simd 1-wide: `q80_mixed_decode.metal:499-534`
- geo_tpr64: `qwen_uniform_q4.metal:181-221`
- HGRAVU01 layout + bound: `q80_mixed_decode.rs:1134-1168, 1305-1329`
- mixed census / named matvec: `qwen38_hybrid_decode.rs:75-91, 1483-1518`
- split in_proj: `qwen38_hybrid_decode.rs:1812-1847, 3264-3284`

---

## Completion

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. bits==3 Uniform GEMVs dispatch q80_hgravs01_factor_matvec_simd3
    at simd8_grid / TG 256 / col stride 256. CONFIRMED.
    evidence: qwen38_hybrid_decode.rs:1687-1688;
              q80_mixed_decode.metal:859-869
C2. bits==4 Uniform GEMVs (attention, split in_proj, lm_head) dispatch
    q80_hgravs01_factor_matvec_simd, 32 TPR, 1-wide extract, stride 32.
    CONFIRMED. 305 such GEMVs on this artifact.
    evidence: qwen38_hybrid_decode.rs:1689-1690;
              q80_mixed_decode.metal:521-528;
              catalog parse §1.5
C3. q4=0. G0 geo_tpr64 is not on this token. CONFIRMED.
    evidence: lane-92 census; classify_qwen38_mixed_payload codec 3
C4. Ceremony is 1.63 ms. Weight GEMV bucket is 137.1 ms (92 % of wall).
    INFERRED from MEASURED GPU minus APPLIED G0 non-GEMV isolated.
    evidence: §3
C5. Mechanism is TPR 32, narrow col-pass, bits=4 per-element scale
    and 1-wide extract. Not bytes, not serial reduce, not dispatch
    count. SUPPORTED by SOURCE + INFERRED bucket.
C6. Existing HGRAVU01 body feeds geo_tpr64 directly. No repack.
    bits=4 is Q4 nibble layout with bound 7. SUPPORTED.
    evidence: §5–§6
C7. Matching G0 kernel class PROJECTS ~32 ms / ~31 TPS on 11.47 GB.
    Not a complete-token wall. Promotion still needs the A/B in §7.

EVIDENCE
See §10 and the file:line pointers on every claim.

CHANGES
Created workspace/superwave/g1/g1-q3-uniform-kernel.md only.

TESTS
test -s workspace/superwave/g1/g1-q3-uniform-kernel.md
wc -l workspace/superwave/g1/g1-q3-uniform-kernel.md
git status --porcelain

RISKS
INFERRED 137.1 ms GEMV bucket assumes G0 isolated non-GEMV still
applies. Box dirt and the 96 new fuses are unmeasured. ESTIMATED
MLP-vs-mixer split is an issue-trip model, not isolated GPU.
measure_isolated_lm_head is still Q4-only.

UNRESOLVED
No q3mlp addr/decode probe. No isolated mlp/mixer/lm GPU on this
artifact. PROJECTED TPS unverified. Oracle-32 6/6 not claimed.

NEXT
GPU-lock lane: bind geo_tpr64 bits=3 and bits=4, isolated CBs, then
complete-wall A/B. Do not pack a new artifact first.
```
