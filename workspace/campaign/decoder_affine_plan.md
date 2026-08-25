# Qwen3.8 AFFINE group-32 2-bit integration plan

Status: plan only. No decoder or shader code is changed by this document.
Goal: add a native AFFINE matvec kind (`w = q * scale + bias`, group-32, 2-bit
mantissa, fp16 scale, fp16 bias) alongside the existing HGRAVU01 absmax path
(`w = (code - bound) * scale`, group-64, bits 3/4) so the Hawking Qwen3.8
Metal decoder can consume the external MLX 2-bit representation without
reconstructing a dense `W`.

Authority for the reconstruction is the contract formula, not HGRAVU01:

```
w[i] = float(q[i]) * float(scale[g]) + float(bias[g])
y[r] += w[i] * x[c]
```

`q` is the unsigned 2-bit mantissa in `{0,1,2,3}`. This is **not**
`q / levels * absmax` and **not** HGRAVU01's signed `(code - bound) * scale`.

Source geometry that every GEMV K in this model already satisfies
(`crates/hawking-core/src/model/qwen38_geometry.rs`):

| constant | value | `K % 32` | `K % 64` |
|---|---|---|---|
| `QWEN38_HIDDEN` | 5120 (L14) | 0 | 0 |
| `QWEN38_INTERMEDIATE` | 17408 (L15) | 0 | 0 |
| `QWEN38_O_PROJ_COLS` | 6144 (L55) | 0 | 0 |

So group-32 tiles and the existing geo_tpr64 8-wide / 512-stride thread map
are both K-complete on every Qwen3.8 GEMV.

The external MLX weights live at
`QWEN38_SOURCE_REPOSITORY = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX"`
(`qwen38_geometry.rs` L13). This plan does not fetch that repo (network is
closed). Pack-time must pin `mx.dequantize` / `QuantizedLinear` against the
native oracle before a catalog row is admitted.

---

## 1. Current dispatch spine (absmax HGRAVU01)

The mixed path is `catalog.hq38m20` → codec → `MixedCatalogLane` →
`MixedMlpNativeKind` (MLP only) → `MixedGpuWeight` → GEMV dispatch.

```
catalog row.codec
        │
        ▼
classify_qwen38_mixed_payload          qwen38_hybrid_decode.rs:160
        │
        ├── Packed(0) ──► MixedMlpNativeKind::Binary     mixed_mlp_native_kind_from_lane:217
        ├── Packed(1) ──► Residual                       :218
        ├── Packed(2) ──► Hgravs                         :219
        ├── Packed(3) ──► Uniform  (HGRAVU01)            :220
        ├── Hq30Uq4   ──► None  (MLP lock refuses)       :222
        ├── F32v2     ──► None                           :223
        └── HgravuVector ──► None (dequant to f32, not GEMV)
        │
        ▼
assert_mixed_mlp_role_allowed          :228
        │  gate  Binary | Uniform
        │  up    Residual | Uniform
        │  down  Hgravs | Uniform
        ▼
load_mixed / upload_mixed              :950 / :1341
        │
        ▼
encode_named_matvec                    :1541
        └── encode_mixed_matvec        :1559
                └── MixedGpuWeight::Uniform
                        └── dispatch_uniform                 :1800
                                └── qwen38_hgravu01_geo_tpr64_launch  :564
                                        bits=3 → qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
                                        bits=4 → qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
                                └── else dispatch_factor     :1720
                                        (signed q * scale, no bias)
```

Token graph does not special-case Uniform vs Binary vs Residual at the
layer level. `encode_dense_mlp_mixed` (L3441), `encode_mlp_matvecs_only_mixed`
(L3836), `encode_mixer_gemvs_only_mixed` (L3776), `encode_deltanet_mixed`
(L3493), `encode_gqa_mixed`, and `encode_terminal_mixed` (L3754) all call
`encode_named_matvec`. Adding Affine at the weight / dispatch layer is
enough for those organs to run. Embed is the exception
(`encode_embed_mixed` L3388 matches `MixedGpuWeight::Uniform` only).

### 1.1 Absmax reconstruction (what Affine must not reuse)

CPU oracle `uniform_factor_value`
(`qwen_complete_binary/q80_mixed_decode.rs:591-598`):

```
let code = extract_unsigned(&packed.codes, element, packed.bits);
let signed = i32::from(code) - i32::from(packed.bound);
signed as f32 * scale
```

`pack_uniform_factor` (same file L529-564) stores absmax/bound as the
group scale and writes offset-binary codes. `bound = (1 << (bits-1)) - 1`.
For bits=2 that would be `bound=1`, `q ∈ {-1,0,1}` — not MLX's `{0,1,2,3}`.

Device HGRAVU01 geo_tpr64 (`q80_mixed_decode.metal:922-1040`) does the
same signed subtract:

```
hgravu01_q3_unpack8:  float(int(bits) - bound) * scale * x[col]
hgravu01_q4_unpack8:  float(int(nibble) - bound) * scale * x[col]
```

Bindings today (`encode_factor_args` L1634-1656):

| index | value |
|---|---|
| 0 | codes |
| 1 | scales |
| 2 | input |
| 3 | output |
| 4 | rows |
| 5 | cols |
| 6 | group_size |
| 7 | bits |
| 8 | bound |

There is no bias buffer. `GpuUniform` (L825-833) has `codes` + `scales`
only. `factor_layout_from_meta` (L1134-1169) bills
`body == scale_bytes + code_bytes` and will refuse a body that also
carries bias.

`qwen38_hgravu01_geo_tpr64_launch` (L564-581) returns `None` unless
`recon_fuse && group_size == 64 && cols % 64 == 0 && bits ∈ {3,4}`.
The unit test at L4728 pins `group_size=32` as a **refuse** for this
function. Affine must not reopen that gate.

Incumbent serial / simd path `gk_uniform_value`
(`gk_family.metal:243-255`) is also signed `q * scale`. Affine must not
fall through to `dispatch_factor`.

### 1.2 Catalog envelope (what Affine reuses)

`HQ38M20` v1, 128-byte records (`qwen38_hybrid_decode.rs:32-37`,
`parse_qwen38_mixed_catalog` L412-489). Record byte 6 is `codec: u8`.
The decoder does not read byte 7 (`organ`); the packer does
(`tools/qwen38_sub15_pack.py` `write_catalog`, `<IHBBBB` at name_off /
name_len / codec / organ / ndim / pad).

Existing codecs in the packer (`tools/qwen38_sub15_pack.py:84-88`):

| codec | name | container |
|---|---|---|
| 0 | Binary | `HGRAVB01` |
| 1 | Residual | `HGRAVR02` |
| 2 | Hgravs | `HGRAVS01` |
| 3 | Uniform / HQ30UQ4 | `HGRAVU01` or `HQ30UQ4\0` |
| 4 | f32v2 | raw f32 envelope |

`classify_qwen38_mixed_payload` (L166-201) refuses anything else:
`unknown mixed codec {other}`. Tests at L4406 and L4612 pin codec 5 as
unknown. Those tests must move to codec 6 when codec 5 becomes Affine.

Gravity containers are `8-byte magic || u32le header_len || JSON || body`
(`split_gravity_container`, `q80_mixed_decode.rs:135-147`). Affine stays
inside that envelope.

`HGRAVA01` is **already taken** by the additive-residual experiment
(`lab/operators/qwen38_recon_pack.py` `split_additive`, expected magic
`b"HGRAVA01"`, body = base_scales || residual_scales || base_codes ||
residual_codes). Do not reuse it.

---

## 2. Decision: new codec 5, new magic `HGRAVF01`, new kernel

Do **not** overload codec 3 (already `HGRAVU01` vs `HQ30UQ4`).
Do **not** stuff bias into `HGRAVU01` (`factor_layout_from_meta` would
have to grow a second personality; `dispatch_uniform` would have to
inspect bits/group and silently change reconstruction).
Do **not** edit `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` or
`qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128`. Those kernels
are the sealed absmax path.

| item | value |
|---|---|
| Catalog codec | `5` |
| Magic | `HGRAVF01` (8 bytes) |
| Schema | `hawking.gravity.affine_scale_bias.v1` |
| Representation | `affine_q2_group32_fp16_scale_bias` |
| bits | 2 |
| group_size | 32 |
| q | unsigned mantissa `code & 3` |
| reconstruction | `w = float(q) * scale + bias` |
| scale | one IEEE fp16 per group |
| bias | one IEEE fp16 per group |
| packing | LSB-first, 4 codes/byte, 8 bytes/group |
| Catalog version | stay `HQ38M20` v1 (record size unchanged) |
| Kernel (default) | `qwen_affine_q2_group32_matvec_geo_tpr64_tg128` |
| Kernel (serial) | `qwen_affine_q2_group32_matvec` |

`HGRAVF01` is a new gravity-family sibling of `HGRAVU01`. The `F` is
affine (scale+offset). It is not `HGRAVA01`.

If a live MLX blob uses `w = scale * (q - zero)` rather than
`w = q * scale + bias`, convert **at pack time**:
`native_scale = scale`, `native_bias = -zero * scale` (or copy `bias`
through if MLX already stores the offset form). Seal the conversion by
matching `mx.dequantize` on one organ before writing the catalog row.
The kernel implements only the contract form.

---

## 3. Every function / match-arm that must change

Cited paths are relative to the worktree. Line numbers are from this
checkout (branch `decoder-affine-integration`).

### 3.1 Lane → kind → role (CPU, no Metal)

File: `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`

| site | lines | change |
|---|---|---|
| `MixedCatalogLane` | 69-73 | add `Affine` |
| `MixedCatalogCensus` | 80-92 | add `pub affine: usize` |
| `classify_qwen38_mixed_payload` match | 166-201 | new `5 =>` arm: magic `HGRAVF01` → `MixedCatalogLane::Affine`; anything else on codec 5 refuses |
| `MixedMlpNativeKind` | 208-213 | add `Affine` |
| `mixed_mlp_native_kind_from_lane` | 215-226 | `MixedCatalogLane::Affine => Some(Affine)`; keep `Hq30Uq4`/`F32v2`/`HgravuVector` as `None` |
| `mixed_mlp_role_allowed` | 228-244 | allow `Affine` on **all three** MLP roles, same as `Uniform` |
| `assert_mixed_mlp_native_kinds` error | 264-266 | message today is `is not {label} or HGRAVU01`; extend to `or HGRAVU01 or HGRAVF01` |
| `assert_mixed_mlp_native_catalog` | 313-314 | **must stop skipping `codec > 3`**. Today `if row.codec > 3 { continue; }` would treat an Affine MLP as **missing** and refuse with `silent dense/Q4 fallback`. Accept `0..=3` and `5`. |
| `census_qwen38_mixed_catalog` match | 333-364 | `Ok(MixedCatalogLane::Affine) =>` validate layout, `census.affine += 1` |
| `unknown_codec_5_still_refuses` | 4406-4414 | retarget to codec **6** |
| `catalog_roundtrip_codec_4_census_and_codec_5_refuses` | 4612-4637 | retarget refuse to codec 6; add a codec-5 Affine census case |
| `mixed_mlp_uniform_is_admitted_on_every_role` | 4427-4430 | keep; add `mixed_mlp_affine_is_admitted_on_every_role` |
| `hq30uq4_on_mlp_is_not_uniform_and_still_refuses` | 4480-4511 | keep (HQ30UQ4 stays non-native) |
| `hgravu01_geo_tpr64_bind_is_bits_3_and_4_only` | 4716-4754 | **keep**. `launch(4, 32, …)` stays `None`. Add a sibling test for the Affine launch. |

`hgravu_is_vector` (L140-158) already returns false for every GEMV suffix
Affine will sit on. Do not classify Affine vectors; Affine is a GEMV
kind. Small tensors stay codec 4 f32v2 or HGRAVU01-vector as today.

### 3.2 On-disk layout / CPU oracle

File: `crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs`

| site | lines | change |
|---|---|---|
| `MAGIC_UNIFORM` family | 38-46 | add `MAGIC_AFFINE: [u8; 8] = *b"HGRAVF01"` and `SCHEMA_AFFINE = "hawking.gravity.affine_scale_bias.v1"` |
| new `AffineFactorPacked` | after `UniformFactorPacked` L72-81 | `rows, cols, bits=2, group_size=32, groups, scales_f16, biases_f16, codes` — **no `bound`** |
| new `pack_affine_factor` | sibling of `pack_uniform_factor` L529 | per group: least-squares or MLX-copied `(scale, bias)`; `q = round((w-bias)/scale).clamp(0,3)`; LSB-pack |
| new `affine_factor_value` | sibling of `uniform_factor_value` L591 | `q = extract_unsigned(codes, i, 2); w = q as f32 * scale + bias` |
| new `affine_factor_matvec_f32` | sibling of L600 | never forms dense `W` |
| `MixedGpuKind` | 1094-1119 | add `Affine { scale_off, scale_bytes, bias_off, bias_bytes, code_off, code_bytes, group_size, bits }` |
| `mixed_gpu_layout` match | 1174-1332 | new `5 =>` arm; **do not** call `factor_layout_from_meta` (that function requires `body == scales + codes` at L1154) |
| `MixedPackedTensor` | 1073-1081 | add `Affine(AffineFactorPacked)` |
| `MixedPackedTensor::from_codec_payload` | 1336-1346 | `5 => Affine(...)` |
| `rows_cols` / `cpu_matvec` / `gather_row` | 1349-1371+ | Affine arms; `cpu_matvec` calls `affine_factor_matvec_f32` |
| `parse_uniform_q8_container` | 1047-1070 | **no change**. bits=8 group=64 stays the Q80 non-expert contract. |
| tests around L1598 | — | add `parse_affine_container_roundtrip` mirroring `parse_uniform_q8_container_roundtrip` |

`factor_layout_from_meta` (L1134) and `uniform_from_body` (L788) stay
absmax-only. A new `affine_layout_from_meta` bills:

```
groups      = rows * (cols / 32)     // require cols % 32 == 0; no silent pad
scale_bytes = groups * 2
bias_bytes  = groups * 2
code_bytes  = groups * 8             // 32 * 2 bits
body        = scales || biases || codes
```

JSON header keys (mirror HGRAVU01, plus bias):

```json
{
  "schema": "hawking.gravity.affine_scale_bias.v1",
  "representation": "affine_q2_group32_fp16_scale_bias",
  "shape": [rows, cols],
  "elements": rows * cols,
  "bits": 2,
  "group_size": 32,
  "groups": groups,
  "scale_bytes": groups * 2,
  "bias_bytes": groups * 2,
  "code_bytes": groups * 8,
  "source": "mlx_quantized_linear"
}
```

Refuse `bits != 2` or `group_size != 32` at parse. Affine is one
contract, not a family.

### 3.3 Upload / resident GPU objects

File: `crates/hawking-core/src/model/qwen38_hybrid_decode.rs` (`#[cfg(target_os = "macos")]` `device` module)

| site | lines | change |
|---|---|---|
| new `GpuAffine` | next to `GpuUniform` L825 | `codes, scales, biases, rows, cols, group_size, bits` |
| `MixedGpuWeight` | 835-840 | add `Affine(GpuAffine)` |
| `MixedGpuWeight::resident_bytes` | 843-861 | `codes + scales + biases` |
| `Qwen38HybridWeights::load_mixed` classify match | 972-1038 | new `MixedCatalogLane::Affine =>` upload three buffers; `census.affine += 1`. Do **not** route through `upload_mixed` (that function's codec switch L1347-1356 only knows 0..=3). |
| census `eprintln!` | 1041-1052 | print `affine=` |
| `assert_mixed_mlp_native` weight match | 1332-1337 | `MixedGpuWeight::Affine(_) => MixedMlpNativeKind::Affine` |
| `upload_mixed` | 1341-1507 | **no codec-5 arm**. Affine is a lane, like `Hq30Uq4` / `F32v2`. |

### 3.4 GEMV dispatch

Same file.

| site | lines | change |
|---|---|---|
| `encode_named_matvec` | 1541-1557 | no change (already forwards any mixed name) |
| `encode_mixed_matvec` match | 1571-1576 | add `MixedGpuWeight::Affine(body) => self.dispatch_affine(...)` |
| new `encode_affine_args` | next to `encode_factor_args` L1634 | see bindings in §4 |
| new `dispatch_affine` | next to `dispatch_uniform` L1800 | geo_tpr64 if `qwen38_affine_q2_geo_tpr64_launch` returns Some; else serial `qwen_affine_q2_group32_matvec`. **Never** call `dispatch_factor` / `encode_factor_args`. |
| new consts + `qwen38_affine_q2_geo_tpr64_launch` | next to L556-581 | `group_size==32 && cols%32==0 && recon_fuse`; same grid as HGRAVU01: `tg=128`, `grid = rows.div_ceil(2) * 128` |
| `qwen38_hgravu01_geo_tpr64_launch` | 564-581 | **no change** |
| `dispatch_uniform` | 1800-1840 | **no change** |
| `encode_embed_mixed` | 3388-3438 | **no change**. Affine embed is out of scope; keep the refuse `embed is neither HGRAVU01 nor HQ30UQ4`. |
| `has_weight` | 1929-1931 | no change (`mixed.contains_key`) |
| `encode_dense_mlp_mixed` / `encode_mlp_matvecs_only_mixed` / mixer mixed | 3441 / 3836 / 3776 | no change |

`measure_named_matvec` (L2180) and `measure_isolated_mlp_one_proj`
(L2221) call `encode_q4_matvec`, **not** `encode_named_matvec`. They
cannot see an Affine organ. The one-organ swap test must call
`encode_named_matvec` (or a new `measure_named_mixed_matvec` that
forwards to it) and `read_f32_workspace("gate"|"up"|"down")` (L2144).

### 3.5 Metal

| site | change |
|---|---|
| `crates/hawking-core/shaders/q80_mixed_decode.metal` after L1050 | add the two kernels in §4. Do not touch `hgravu01_q3_unpack8` / `hgravu01_q4_unpack8`. |
| `crates/hawking-core/src/metal/mod.rs` L333, L439 | no registry edit. `SHADER_Q80_MIXED_DECODE` is `include_str!` of that file; `all_shader_sources` already concatenates it. |
| `hgravu01_geo_tpr64_bind_is_bits_3_and_4_only` L4729-4732 | add the same `SHADER_Q80_MIXED_DECODE.contains("kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128")` check in the new Affine launch test. |
| `qwen_uniform_qn.metal` | do not reuse. It is signed `q * scale`, group-128 Lane-N. |
| `qwen_uniform_q4.metal` geo_tpr64 | do not modify. HQ30UQ4 group-32 is already refused (L4744-4748). |

### 3.6 Packer / catalog writer (not in the sparse checkout; present in git)

| site | change |
|---|---|
| `tools/qwen38_sub15_pack.py` L84-88 | `CODEC_AFFINE = 5` |
| `decode_mixed_payload` L146-159 | arm 5 → affine dequant (for pack-time checks only; decoder never uses this on the token path) |
| `write_hq38m20` L736+ | one-organ swap recipe (see §6); do not change the sealed sub15 census `{0:64, 1:368, 2:64, 3:2, 4:353}` on the default path |
| `lab/operators/qwen38_recon_pack.py` `split_additive` | leave `HGRAVA01` alone |
| new helper (Rust `pack_affine_factor` is the oracle; Python may wrap it) | ingest MLX `weight` (packed u32) + `scales` + `biases` |

Q80 mixed (`qwen80_mixed_hybrid_decode.rs`, `qwen80_mixed_catalog.rs`) is
out of scope. `mixed_gpu_layout` is shared: adding codec 5 there is
correct and refuses a Q80 catalog that is not Affine. Do not add Affine
to Q80 role locks.

---

## 4. Kernel: group-32 affine vs group-64 absmax

### 4.1 Why a new kernel, not a geo_tpr64 patch

`qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`
(`q80_mixed_decode.metal:962-1003`) and
`qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128` (L1009-1050):

- address `group = col >> 6` (group-64)
- load one `half` scale, no bias
- subtract `bound` (3 or 7)
- bind `group_size, bits, bound` as buffers 6-8
- require `group_size == 64 && (cols & 63) == 0`

Patching those kernels to optionally add bias and switch group-32 would
couple two reconstructions on one occupancy tile and break the existing
`hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` test (L4758),
which binds the same `encode_factor_args` layout.

### 4.2 Thread map (reuse geo_tpr64 occupancy)

Keep the sealed map documented at `q80_mixed_decode.metal:912-920`:

```
TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row
col = lane_in_row * 8, stride 512
grid = ceil(rows/2)*128, tg = 128
```

Group-32 alignment of that map:

- `lane_in_row * 8` is 0, 8, 16, …, 504. Every 8-wide tile sits inside
  one group of 32 (`col % 32 ∈ {0,8,16,24}`).
- Stride 512 = 16 groups of 32.
- Qwen3.8 `K ∈ {5120, 6144, 17408}` are all multiples of 512, so the
  tail loop is empty (same as today's HGRAVU01 GEMVs).

Do **not** require `cols % 64 == 0` in the Affine launch (the occupancy
still works when only `cols % 32 == 0`). On this model both hold.

### 4.3 Unpack

8 weights = 16 bits = one `ushort` / two bytes, LSB-first, matching
`extract_unsigned` / `pack_unsigned` (`q80_mixed_decode.rs:170-181`,
`579-588`):

```
byte 0: q0 bits[1:0], q1 [3:2], q2 [5:4], q3 [7:6]
byte 1: q4..q7
```

8 bytes/group = two `uint` = 32 codes. If MLX stores 16 codes per
`uint32` LSB-first (its usual 2-bit word), the native body is those
bytes unchanged. Pack-time must prove that on one organ; if MLX is
MSB-first inside the word, swizzle once in the packer, never in the
token kernel.

### 4.4 Proposed kernels

Append to `q80_mixed_decode.metal` after the HGRAVU01 geo_tpr64 block
(after L1050). Signatures:

```metal
// Serial family (HAWKING_QWEN38_RECON_FUSE=0). One thread per row.
// Grid (rows,1,1), TG (256,1,1). Same refuse-not-reconstruct rule.
kernel void qwen_affine_q2_group32_matvec(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    uint row                         [[thread_position_in_grid]]);

// G0 occupancy. Grid ceil(rows/2)*128, TG 128.
kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half*  scales      [[buffer(1)]],
    device const half*  biases      [[buffer(2)]],
    device const float* input       [[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint& rows             [[buffer(5)]],
    constant uint& cols             [[buffer(6)]],
    constant uint& group_size       [[buffer(7)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]]);
```

Inner 8-wide (contract association, not a hoist of `bias * sum(x)`):

```metal
static inline float affine_q2_unpack8(
    uint packed16, float scale, float bias,
    device const float* x, uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        const float w = float(q) * scale + bias;
        sum += w * x[col + i];
    }
    return sum;
}
```

Addressing inside the geo loop (`col` is the 8-aligned column):

```
groups_per_row = cols >> 5;          // not >> 6
group          = col >> 5;
local          = col & 31u;
rgb            = row * groups_per_row + group;
scale          = float(scales[rgb]);
bias           = float(biases[rgb]);
byte0          = rgb * 8u + (local >> 2u);   // 4 codes/byte
packed16       = ushort load at codes[byte0]
acc           += affine_q2_unpack8(packed16, scale, bias, input, col);
```

Guard: `row < rows && group_size == 32u && (cols & 31u) == 0u`.
No `bits` / `bound` buffers.

Reduction is the existing 2-split simd_sum + `red[4]` pair
(`q80_mixed_decode.metal:995-1002`). Same association of partials as
HGRAVU01 geo, different `w`.

### 4.5 Host bindings

`encode_affine_args` — **do not reuse** `encode_factor_args`:

| index | value |
|---|---|
| 0 | `GpuAffine.codes` |
| 1 | `GpuAffine.scales` |
| 2 | `GpuAffine.biases`  **new** |
| 3 | input activation |
| 4 | output |
| 5 | rows |
| 6 | cols |
| 7 | group_size (32) |

Serial and geo kernels share this ABI. `dispatch_affine` only changes
the kernel name and grid.

`qwen38_affine_q2_geo_tpr64_launch(group_size, rows, cols)`:

```
if !qwen38_recon_fuse_enabled() || group_size != 32 || cols % 32 != 0 {
    return None;   // serial kernel, not a reconstruct-to-Q4 fallback
}
Some(("qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
      (rows.div_ceil(2)*128, 1, 1), (128, 1, 1)))
```

---

## 5. On-disk catalog additions

The `HQ38M20` table does not grow. One `u8` codec and a new gravity
container carry scale+bias.

### 5.1 Catalog row

Same 128-byte record (`QWEN38_MIXED_RECORD_SIZE = 128`, L34).

| field | offset | Affine value |
|---|---|---|
| name | blob | existing tensor name, e.g. `language_model.model.layers.0.mlp.gate_proj.weight` |
| codec | rec[6] | **5** |
| organ | rec[7] | keep the role (`ORGAN_GATE=0` / `UP=1` / `DOWN=2` in the packer) |
| shape | rec[12..] | `[rows, cols]` unchanged |
| segment_id / offset / nbytes | rec[36..] | point at the `HGRAVF01` payload |
| codec_bpw | packer only | `3.0` = 2 code bits + 32 scale/bias bits / 32 weights |
| sha256 | packer only | hash of the new container |

`QWEN38_MIXED_CATALOG_VERSION` stays 1. `QWEN38_MIXED_SCHEMA` stays
`hawking.ascension.qwen38_mixed_representation_candidate.v1`. A new
catalog version is not required to introduce a payload codec.

### 5.2 Container body

```
[0, 8)     HGRAVF01
[8, 12)    u32le JSON header length
[12, 12+H) JSON (keys in §3.2)
then body:
  fp16le scales[groups]
  fp16le biases[groups]
  uint8  codes[groups * 8]
```

`groups = rows * (cols / 32)`. Example L0 `mlp.gate_proj`
(`[17408, 5120]`):

| piece | count | bytes |
|---|---|---|
| scales | 17408 * 160 = 2_785_280 | 5_570_560 |
| biases | 2_785_280 | 5_570_560 |
| codes | 2_785_280 * 8 | 22_282_240 |
| body |  | 33_423_360 |
| vs HGRAVU01 q3 g64 | scales 1_392_640*2 + codes 24 B/g | ~36.2 MB |

Affine-2bit is smaller than HGRAVU01 q3 on this organ (~3.00 vs ~3.25
bpw) because group-32 scale+bias (1.00 bpw) plus 2-bit codes beats
group-64 scale (0.25 bpw) plus 3-bit codes.

### 5.3 Ingest from MLX

Typical MLX `QuantizedLinear` tensors (names vary by export):

- packed codes: `uint32`, 16 × 2-bit values per word, last-axis groups
  of 32 (along `in_features` = Hawking `cols`)
- `scales`: fp16, shape `[rows, cols/32]`
- `biases`: fp16, same shape

Packer steps:

1. Load the three arrays for one organ.
2. Flatten scales/biases row-major to match `rgb = row * (cols/32) + col/32`.
3. Prove bit order: decode group 0 with LSB-first 2-bit; compare to
   `mx.dequantize` (or a recorded f32 dump of that group). If it fails,
   try word-swizzle; do not guess in the kernel.
4. Convert zero-point form to `w = q*s + b` if needed (§2).
5. Wrap `HGRAVF01` + JSON + `scales||biases||codes`.
6. Write a segment blob; patch one catalog row to codec 5.

Do not expand to f32 and re-`pack_uniform_factor`. That would be
absmax, not MLX.

### 5.4 Admission traps (must fail closed)

| input | required refuse |
|---|---|
| codec 5, magic ≠ `HGRAVF01` | classify error (same tone as L178-181) |
| codec 5, `bits≠2` or `group_size≠32` | `mixed_gpu_layout` error |
| codec 5, `cols % 32 != 0` | layout error; no pad reinterpret |
| codec 5, body ≠ scales+biases+codes | layout error |
| Affine MLP row with `assert_mixed_mlp_native_catalog` still skipping `codec>3` | would look "missing" — **fix the skip** |
| Affine organ dispatched via `dispatch_uniform` / `gk_uniform_value` | silent wrong math — **forbidden** |
| Reconstruct Affine → Q4 / f32 GEMV | `expanded_to_q4` / `expanded_to_float_gemv` must stay 0 |

---

## 6. One-organ swap test plan

Organ: `language_model.model.layers.0.mlp.gate_proj.weight`
(`[17408, 5120]`, DeltaNet layer 0, `mlp.gate_proj` is the Binary-or-Uniform
role today). One GEMV, first layer, first MLP projection. Swapping this
does not touch SwiGLU, down-proj, or the mixer.

Do **not** expect greedy token identity against `mixed-q3mlp-v1` or
against a full-model MLX 2-bit generate. Only that organ changes.

### 6.1 Build the swap catalog

1. Start from a catalog that already admits
   (`mixed-q3mlp-v1` preferred — every MLP is already HGRAVU01, so the
   role lock is Uniform|Affine; or `mixed-2p0-v1` where L0 gate is
   Binary and Affine is also allowed).
2. Copy the tree (hardlink segments).
3. Pack L0 gate from the MLX 2-bit tensor into `HGRAVF01` (§5.3).
4. Write `segments/<new>.hq38seg` (or overwrite a dedicated swap
   segment). Do not mutate the original mixed segment in place.
5. Patch that one catalog row: `codec=5`, `nbytes`, `sha256`,
   `codec_bpw=3.0`, `segment_id/offset` if the blob moved.
6. `assert_mixed_mlp_native_catalog(swap_root)` must pass.
7. `census_qwen38_mixed_catalog` must show `affine == 1`,
   `refused == 0`, `expanded_to_q4 == 0`, `expanded_to_float_gemv == 0`,
   and the previous codec count for that row decremented by 1.

### 6.2 CPU oracle vs MLX

On the host, without Metal:

1. Read the `HGRAVF01` payload; `affine_factor_value` over the whole
   organ is too big to materialize. Instead gather row 0 and row
   `17407` (or 32 random rows).
2. Compare those rows to `mx.dequantize` (or a pre-dumped f32 slice
   of the same MLX tensor).
3. Pass: `max_abs <= 2 * fp16_ulp(scale) + fp16_ulp(bias)` on every
   compared element (f16 scales/bias are the stored truth; q is exact).
4. Fail closed if bit-order or `q*s+b` vs `(q-z)*s` was guessed wrong.

### 6.3 Kernel vs CPU oracle

Mirror `hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` (L4758)
but Affine has no signed incumbent. Compare:

- serial `qwen_affine_q2_group32_matvec`
- geo `qwen_affine_q2_group32_matvec_geo_tpr64_tg128`
- `affine_factor_matvec_f32`

on the swapped organ with the same `x[i] = (i % 17) as f32 * 0.125 - 1.0`
used at L4828. Pass: serial == geo at 0 abs (same association), and
`max_abs(gpu, cpu) == 0` for serial (identical f32 ops). Geo vs CPU may
differ by simd reduction order; record `max_abs` / `max_rel` the same
way the HGRAVU01 table prints (L4791). A geo/CPU gap larger than the
HGRAVU01 q3 gap on the same shape is a bug.

Also run this on a tiny synthetic `[4, 64]` fixture that does not need
the campaign artifact.

### 6.4 Isolated GEMV vs external decode of that organ

This is the "diff vs the external decode" gate.

1. Open `Qwen38HybridDecodeSession` on the swap catalog.
2. Write a known `normalized` (the same `x` as §6.3, length 5120) into
   the workspace, or encode a real token through `encode_embed` + L0
   input RMSNorm and **snapshot** `workspace.normalized` (`read_f32_workspace`).
3. Dispatch only L0 `mlp.gate_proj` via `encode_named_matvec` (not
   `measure_isolated_mlp_one_proj` — that is Q4-only, L2239).
4. `read_f32_workspace("gate", 17408)`.
5. External: MLX `QuantizedLinear` for that organ, same `x`, fp32 out.
6. Diff `gate` vs MLX `y`. Pass: `max_abs` within the geo/CPU envelope
   of §6.3. Token ids are not compared here.

A second external path if MLX Python is not on the host: dump `y` from
a previously recorded MLX run (same `x` hash) and diff the file. The
plan requires the dump's `x` sha256 to match the Hawking snapshot.

### 6.5 Rest of the graph did not move

1. On the **unswapped** catalog, snapshot L0 `workspace.up` and
   `workspace.down` after one `encode_dense_mlp_mixed` with the same
   residual input.
2. On the swap catalog, same input, same snapshots.
3. `up` and `down` must be bit-identical (those organs are untouched).
4. `gate` must **differ** (proves the swap actually bound Affine, not
   the old HGRAVU01/Binary blob).
5. `census.affine == 1` on the swap tree; `== 0` on the parent.

### 6.6 What this test does not claim

- Full-model greedy match to MLX generate.
- Affine on attention / lm_head / embed (same kernel would serve them
  via `encode_named_matvec`, but they are not in the swap).
- Throughput. Occupancy is copied from geo_tpr64; no TPS receipt.

### 6.7 Suggested test names

Add next to `mixed_catalog_contract_tests`:

- `classify_codec_5_hgrafv01_is_affine`
- `classify_codec_5_wrong_magic_refuses`
- `unknown_codec_6_still_refuses` (replaces codec-5 refuse)
- `mixed_mlp_affine_is_admitted_on_every_role`
- `assert_mixed_mlp_native_catalog_accepts_codec_5`
- `affine_q2_geo_tpr64_bind_is_group32_only`
- `affine_layout_refuses_missing_bias`
- `affine_factor_value_is_q_times_scale_plus_bias` (contrast a bits=2
  HGRAVU01 pack of the same group — they must disagree)
- `#[cfg(target_os = "macos")] affine_geo_matches_cpu_oracle_on_fixture`
- `#[cfg(target_os = "macos")] one_organ_l0_gate_affine_matches_mlx_dump`
  (skip if the MLX dump / swap catalog is absent, same pattern as
  L4524-4528)

---

## 7. Implementation order

1. **Oracle + container** in `q80_mixed_decode.rs` (magic, pack, value,
   layout, tests). No Metal yet. Proves `w = q*s+b` ≠ absmax bits=2.
2. **Classify / role / census / catalog skip** in
   `qwen38_hybrid_decode.rs` (the `codec > 3` skip is a load-bearing
   footgun). Unit tests without Metal.
3. **Kernels** in `q80_mixed_decode.metal` + launch helper +
   `GpuAffine` + `dispatch_affine`. Fixture parity on macOS.
4. **Packer** writes one `HGRAVF01` organ from MLX; swap catalog.
5. **One-organ swap** (§6) against the MLX dump.
6. Only then consider attention / lm_head / more organs. Same kernel.

---

## 8. Evidence (real reads / greps)

### 8.1 Lane and role lock

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:69
pub enum MixedCatalogLane {
    Packed(u8),
    Hq30Uq4,
    F32v2,
    HgravuVector,
}

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:208
pub enum MixedMlpNativeKind {
    Binary,
    Residual,
    Hgravs,
    Uniform,
}

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:215
pub fn mixed_mlp_native_kind_from_lane(lane: MixedCatalogLane) -> Option<MixedMlpNativeKind> {
    match lane {
        MixedCatalogLane::Packed(0) => Some(MixedMlpNativeKind::Binary),
        MixedCatalogLane::Packed(1) => Some(MixedMlpNativeKind::Residual),
        MixedCatalogLane::Packed(2) => Some(MixedMlpNativeKind::Hgravs),
        MixedCatalogLane::Packed(3) => Some(MixedMlpNativeKind::Uniform),
        MixedCatalogLane::Packed(_)
        | MixedCatalogLane::Hq30Uq4
        | MixedCatalogLane::F32v2
        | MixedCatalogLane::HgravuVector => None,
    }
}

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:228
fn mixed_mlp_role_allowed(suffix: &str, kind: MixedMlpNativeKind) -> bool {
    match suffix {
        "mlp.gate_proj.weight" => matches!(kind, Binary | Uniform),
        "mlp.up_proj.weight"   => matches!(kind, Residual | Uniform),
        "mlp.down_proj.weight" => matches!(kind, Hgravs | Uniform),
        _ => false,
    }
}

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:313
        if row.codec > 3 {
            continue;
        }
```

### 8.2 Classify refuse + codec 5 tests

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:198
        other => Err(mixed_error(format!(
            "{name} unknown mixed codec {other}; refusing silent fallback"
        ))),

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:4406
    fn unknown_codec_5_still_refuses() {
        let err = classify_qwen38_mixed_payload(5, b"xxxxxxxx", "tensor.x", &[1])
            .expect_err("codec 5 must refuse");
```

### 8.3 HGRAVU01 launch refuses group-32

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:564
pub fn qwen38_hgravu01_geo_tpr64_launch(...) {
    if !qwen38_recon_fuse_enabled() || group_size != 64 || cols % 64 != 0 {
        return None;
    }
    let name = match bits {
        3 => QWEN38_HGRAVU01_Q3_GEO_TPR64,
        4 => QWEN38_HGRAVU01_Q4_GEO_TPR64,
        _ => return None,
    };

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:4728
        assert!(qwen38_hgravu01_geo_tpr64_launch(4, 32, 5120, 5120).is_none());
```

### 8.4 Uniform GPU object and dispatch

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:825
    struct GpuUniform {
        codes: PinnedBuffer,
        scales: PinnedBuffer,
        rows: u32, cols: u32, group_size: u32, bits: u32, bound: u32,
    }

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1571
            match weight {
                MixedGpuWeight::Binary(body) => self.dispatch_binary(...),
                MixedGpuWeight::Residual(body) => self.dispatch_residual(...),
                MixedGpuWeight::Hgravs(body) => self.dispatch_hgravs(...),
                MixedGpuWeight::Uniform(body) => self.dispatch_uniform(...),
            }

crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1634
        fn encode_factor_args(..., codes, scales, input, output,
                              rows, cols, group_size, bits, bound)
        // buffers 0..8; no bias
```

### 8.5 Absmax oracles

```
crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:41
pub const MAGIC_UNIFORM: [u8; 8] = *b"HGRAVU01";
pub const SCHEMA_UNIFORM: &str = "hawking.gravity.uniform_group.v1";

crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:591
pub fn uniform_factor_value(...) -> f32 {
    let signed = i32::from(code) - i32::from(packed.bound);
    signed as f32 * scale
}

crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:1154
    if body_len != scale_bytes + code_bytes { return Err(...) }

crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:1174
pub fn mixed_gpu_layout(codec: u8, payload: &[u8]) -> Result<MixedGpuLayout> {
    match codec { 0 => Binary, 1 => Residual, 2 => Hgravs, 3 => Uniform,
                  other => Err(unknown mixed codec) }
```

### 8.6 Device HGRAVU01 geo_tpr64

```
crates/hawking-core/shaders/q80_mixed_decode.metal:962
kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output      [[buffer(3)]],
    constant uint& rows       [[buffer(4)]],
    ... group_size, bits, bound [[buffer(6..=8)]],
    ...
            const uint group = col >> 6u;
            const float scale = float(scales[rgb]);
            acc += hgravu01_q3_unpack8(codes, byte0, scale, qbound, input, col);

crates/hawking-core/shaders/gk_family.metal:243
static inline float gk_uniform_value(...) {
    const int q = int(code) - int(bound);
    return float(q) * float(scales[group]);
}
```

### 8.7 Shader compile path

```
crates/hawking-core/src/metal/mod.rs:333
pub const SHADER_Q80_MIXED_DECODE: &str = include_str!("../../shaders/q80_mixed_decode.metal");
// all_shader_sources() pushes SHADER_Q80_MIXED_DECODE at L439
```

### 8.8 Catalog writer codecs

```
tools/qwen38_sub15_pack.py:84
CODEC_BINARY = 0
CODEC_RESIDUAL = 1
CODEC_HGRAVS01 = 2
CODEC_UNIFORM4 = 3
CODEC_F32 = 4
```

### 8.9 HGRAVA01 is additive residual, not affine

```
lab/operators/qwen38_recon_pack.py
def split_additive(payload: bytes) -> dict[str, Any]:
    header, body = _parse_container(payload, expected_magic=b"HGRAVA01")
    # body = base_scales || residual_scales || base_codes || residual_codes
```

### 8.10 Token-graph mixed MLP already goes through encode_named_matvec

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3455
            self.encode_named_matvec(..., "mlp.gate_proj.weight", normalized, gate)?;
            self.encode_named_matvec(..., "mlp.up_proj.weight",   normalized, up)?;
            ... swiglu ...
            self.encode_named_matvec(..., "mlp.down_proj.weight", act, down)?;
```

### 8.11 Isolated Q4 measure will not see Affine

```
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:2239
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, suffix),
                        input,
                        output,
                    )?;
```

---

## 9. Non-goals

- Editing any file under `crates/` as part of delivering this plan.
- Changing Q80 mixed catalogs or Q80 role locks.
- Affine embed / a new `qwen38_hgrafv_embedding_lookup`.
- Reopening HQ30UQ4 group-32 (explicitly refused at L4744).
- Reusing `qwen_uniform_qn.metal` (signed, group-128).
- Throughput claims, megakernel fusion, concurrent gate+up.
- Bumping `HQ38M20` to v2.

---

## 10. Done when (implementation follow-on, not this document)

1. Codec 5 / `HGRAVF01` classifies, layouts, role-locks, and censuses.
2. `dispatch_affine` binds bias and runs `w = q*scale+bias` group-32.
3. HGRAVU01 geo_tpr64 q3/q4 still match the incumbent on
   `mixed-q3mlp-v1`.
4. One L0 `mlp.gate_proj` swap catalog diffs that GEMV against the
   external MLX dequant of the same organ.
5. `expanded_to_q4` and `expanded_to_float_gemv` stay zero.
