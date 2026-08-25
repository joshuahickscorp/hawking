# EXEC lever map — Qwen3.8 mixed-q3mlp-v1 MLP GEMV

Read-only analysis. No crates / `.rs` / `.metal` were modified.
Sources: `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`,
`crates/hawking-core/shaders/q80_mixed_decode.metal`,
`crates/hawking-core/shaders/qwen_uniform_q4.metal`,
`crates/hawking-core/src/lib.rs`,
`crates/hawking-core/src/decode_family.rs`,
`crates/hawking-core/src/metal/mod.rs`.
The contract's `workspace/campaign/metal_shared_op_blueprint.md` is **not
in this tree** (not on disk, not in `git ls-tree HEAD -- workspace/campaign`).
The dispatch chain below is taken from the live decoder, with the G1 note
`workspace/superwave/g1/g1-hgravu01-geo-tpr64.md` used only as a measured
receipt (via `git show`; not in the sparse checkout).

---

## 0. Verdict (read this first)

**The current mixed-q3mlp-v1 patient already runs the fast
`geo_tpr64` tiled kernel for every Uniform HGRAVU01 MLP GEMV under the
default environment.** It is not stuck on `simd3`.

`HAWKING_QWEN38_RECON_FUSE` does **not** choose `geo_tpr64` vs `simd3`.
It is a default-ON opt-out that chooses the occupancy-tile family vs the
G023 serial family. The `simd3` vs `geo_tpr64` split lives **inside** the
fuse-ON path, in `dispatch_uniform` → `qwen38_hgravu01_geo_tpr64_launch`.

| Question | Answer |
|---|---|
| What does the patient dispatch for MLP today (default env)? | `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` |
| What would `HAWKING_QWEN38_RECON_FUSE=0` dispatch? | serial `gk_matvec_hgravs` / `q80_hgravs01_factor_matvec` — slower than both |
| Is there an env that selects `simd3` while fuse is ON? | **No.** Geometry miss is the only fuse-ON fallback. |
| Fast path reachable by flag/env alone? | **YES** — already the default. |
| ~3–4× leftover on this patient by flipping a flag? | **NO** under default env. |

### YES / NO / CONDITIONAL

**YES** — the fast path is reachable on mixed-q3mlp-v1 by env alone.

Exact lever:

- **unset** `HAWKING_QWEN38_RECON_FUSE` (default ON via `env_opt_out`), **or**
- set it to any non-disable value (`1`, `true`, …).

Disable tokens (`0` / `false` / `off` / `no`) **leave** the fast path and
drop Uniform MLP onto the serial factor kernel.

No code change is required for this patient's geometry
(`bits=3`, `group_size=64`, `cols ∈ {5120, 17408}`, both `% 64 == 0`).

A leftover ~3–4× vs `simd3` is **not** sitting unused on the current
default patient. That delta was already claimed when `geo_tpr64` was
wired (G1 informal whole-token GPU: ~46–48 ms/token vs ~163 ms/token
incumbent). It is **not** an isolated MLP-kernel measurement, and it is
**not** recoverable a second time by a flag.

---

## 1. Dispatch chain for mixed-q3mlp-v1 MLP GEMV

### 1.1 Production encode (layer suffix)

Mixed catalog present → `encode_dense_mlp` (3040) delegates to
`encode_dense_mlp_mixed` (3441). Each of gate / up / down is
`encode_named_matvec`:

```3455:3483:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
            )?;
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                ...
            )?;
            ...
            self.encode_named_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
```

`encode_named_matvec` (1541) prefers mixed over Q4. For HGRAVU01 Uniform
it lands here:

```1571:1576:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
            match weight {
                MixedGpuWeight::Binary(body) => self.dispatch_binary(...),
                MixedGpuWeight::Residual(body) => self.dispatch_residual(...),
                MixedGpuWeight::Hgravs(body) => self.dispatch_hgravs(...),
                MixedGpuWeight::Uniform(body) => self.dispatch_uniform(tcb, body, input, output),
            }
```

G1 census on this artifact (and the decoder's load-time print):
`tensors=851 binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353`.
Every MLP GEMV is `MixedGpuWeight::Uniform`. HGRAVS r160 never sees
these tensors; `dispatch_factor` is only the Uniform fallback.

### 1.2 Exact branch that chooses `geo_tpr64` vs `simd3`

**Function:** `Qwen38HybridDecodeSession::dispatch_uniform`
**File:** `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
**Lines:** 1800–1840

```1800:1840:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        fn dispatch_uniform(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuUniform,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if let Some((name, grid, tg)) = qwen38_hgravu01_geo_tpr64_launch(
                body.bits,
                body.group_size,
                body.rows,
                body.cols,
            ) {
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_factor_args(...)
                });
            }
            self.dispatch_factor(
                tcb, &body.codes, &body.scales, input, output,
                body.rows, body.cols, body.group_size, body.bits, body.bound,
            )
        }
```

The predicate is `qwen38_hgravu01_geo_tpr64_launch` (564–581):

```564:581:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
pub fn qwen38_hgravu01_geo_tpr64_launch(
    bits: u32, group_size: u32, rows: u32, cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    if !qwen38_recon_fuse_enabled() || group_size != 64 || cols % 64 != 0 {
        return None;
    }
    let name = match bits {
        3 => QWEN38_HGRAVU01_Q3_GEO_TPR64,   // "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128"
        4 => QWEN38_HGRAVU01_Q4_GEO_TPR64,   // "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128"
        _ => return None,
    };
    let tg = 128u32;
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    Some((name, (grid, 1, 1), (tg, 1, 1)))
}
```

Constants (556–559):

- `QWEN38_HGRAVU01_Q3_GEO_TPR64` = `"qwen_uniform_q3_group64_matvec_geo_tpr64_tg128"`
- `QWEN38_HGRAVU01_Q4_GEO_TPR64` = `"qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128"`

If that returns `None`, `dispatch_factor` (1720–1758) is the incumbent:

```1738:1757:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
            if qwen38_recon_fuse_enabled() {
                let (name, grid) = if bits == 8 {
                    ...
                } else if bits == 3 {
                    ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
                } else {
                    ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), encode)
            } else {
                tcb.dispatch_threads(
                    crate::decode_family::matvec_hgravs(),
                    (rows, 1, 1),
                    (256, 1, 1),
                    encode,
                )
            }
```

`simd8_grid` (1095–1097) = `(rows.div_ceil(8) * 256, 1, 1)`, TG `(256, 1, 1)`.

`decode_family::matvec_hgravs()` (136–138 of `decode_family.rs`) is
`gk_matvec_hgravs` when `HAWKING_DECODE_FAMILY` is ON (default), else
legacy `q80_hgravs01_factor_matvec`. That name switch is **serial only**.
It does not retarget `geo_tpr64`.

### 1.3 Decision table for Uniform MLP

| `HAWKING_QWEN38_RECON_FUSE` | bits | group | `cols % 64` | kernel | TG | grid |
|---|---:|---:|---|---|---|---|
| ON (default) | 3 | 64 | 0 | `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` | 128 | `ceil(rows/2)*128` |
| ON | 4 | 64 | 0 | `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128` | 128 | `ceil(rows/2)*128` |
| ON | 3 | ≠64 or `cols%64≠0` | — | `q80_hgravs01_factor_matvec_simd3` | 256 | `ceil(rows/8)*256` |
| ON | 4 | ≠64 or `cols%64≠0` | — | `q80_hgravs01_factor_matvec_simd` | 256 | `ceil(rows/8)*256` |
| ON | 8 | any | — | `q80_uniform8_matvec_*` | 256 | tile/simd |
| OFF (`=0`) | any | any | — | `gk_matvec_hgravs` / `q80_hgravs01_factor_matvec` | 256 | `(rows,1,1)` |

mixed-q3mlp-v1 MLP is the first row. Attention Uniform organs on this
patient are bits=4 group-64 and take the second row (not MLP, listed
only so the 3–4× whole-token number is not misread as MLP-only).

### 1.4 Patient geometry (from the in-tree parity test)

`hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` (4758) opens
`campaign_qwen38("mixed-q3mlp-v1")` =
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`.

G1 measured table (same artifact):

| tensor | bits | shape | incumbent if geo refused |
|---|---:|---|---|
| L0 `mlp.gate_proj` | 3 | 17408 × 5120 | simd3 |
| L0 `mlp.up_proj` | 3 | 17408 × 5120 | simd3 |
| L0 `mlp.down_proj` | 3 | 5120 × 17408 | simd3 |

`QWEN38_HIDDEN = 5120`, `QWEN38_INTERMEDIATE = 17408`, `QWEN38_LAYERS = 64`
(`qwen38_geometry.rs:20–25`). `5120/64 = 80`, `17408/64 = 272`. Both
widths bind. The unit test at 4717 binds exactly this gate shape:

```
qwen38_hgravu01_geo_tpr64_launch(3, 64, 17408, 5120)
  → ("qwen_uniform_q3_group64_matvec_geo_tpr64_tg128",
     (17408.div_ceil(2)*128, 1, 1) = (1_114_112, 1, 1),
     (128, 1, 1))
```

Role lock (`assert_mixed_mlp_native_kinds`, 250–276) admits Uniform on
all three MLP roles. `mixed_q3mlp_and_q4down_pass_mlp_admission` (4521)
requires this catalog to admit.

---

## 2. What `HAWKING_QWEN38_RECON_FUSE` actually switches

### 2.1 The flag

```42:46:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
/// Default **on**. Consume packed mixed codes on the Q80 occupancy tiles.
/// `HAWKING_QWEN38_RECON_FUSE=0` selects the G023 serial family names.
pub fn qwen38_recon_fuse_enabled() -> bool {
    crate::env_opt_out("HAWKING_QWEN38_RECON_FUSE")
}
```

`env_opt_out` (`crates/hawking-core/src/lib.rs:207–215`):

```207:215:crates/hawking-core/src/lib.rs
pub fn env_opt_out(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}
```

Unset → ON. `=1` / `true` / anything else → ON. Only `0`/`false`/`off`/`no`
turns it off.

Load-time stderr (`qwen38_mixed_k_complete_bind_message`, 125–137, printed
at 1053) reports `recon_fuse=ON|OFF`. That message talks about **binary /
CSR** occupancy tiles (`q80_binary_group_matvec_simd_bytes` /
`q80_binary_group_csr_matvec_bytes` when `cols>2048`). It does **not**
mention `geo_tpr64`. Operators reading that line cannot tell Uniform MLP
which kernel they got.

### 2.2 Every fuse site in this decoder

| Site | fuse ON | fuse OFF |
|---|---|---|
| `qwen38_hgravu01_geo_tpr64_launch:570` | allow geo bind | **force None** → Uniform falls through |
| `dispatch_uniform:1807` | geo if launch Some | `dispatch_factor` serial |
| `dispatch_factor:1738` | simd3 / simd / uniform8 occupancy | serial `matvec_hgravs()` |
| `dispatch_binary:1665` | `q80_binary_group_matvec_{simd_bytes,tg256}` | serial `matvec_binary()` |
| `dispatch_residual:1693` | fused CSR occupancy tile | serial binary + `q80_sparse_q1_apply_csr` |

Prior note that "RECON_FUSE selects geo_tpr64 vs simd3" is **false**.
Correct statement:

- fuse ON + Uniform bits∈{3,4} + group 64 + `cols%64==0` → **geo_tpr64**
- fuse ON + those predicates fail + bits==3 → **simd3**
- fuse OFF → **serial**, never simd3, never geo

There is no second env (`HAWKING_QWEN38_GEO_*`, etc.) in this crate that
retargets the Uniform MLP kernel.

`HAWKING_DECODE_FAMILY` (`decode_family.rs:103–105`) only renames the
serial G023 symbols. It does not reach `dispatch_uniform`'s geo branch.

### 2.3 Related unused kernel (not an env)

`qwen_uniform_q3_group64_matvec_geo_tpr64_tg128_alignedload` exists in
`qwen_uniform_q4.metal:1530` (same TG map, two aligned `uint` loads of
the 3-byte unpack). It is listed in `metal/mod.rs:1271` as a compiled
name and is **never** selected by `qwen38_hgravu01_geo_tpr64_launch`.
Reaching it is a **code change** (swap the bits=3 name), not a flag.

---

## 3. Both kernels — names, signatures, geometry, why geo is faster

Both live in `crates/hawking-core/shaders/q80_mixed_decode.metal`, compiled
into the default library:

```
crates/hawking-core/src/metal/mod.rs:333
pub const SHADER_Q80_MIXED_DECODE: &str = include_str!("../../shaders/q80_mixed_decode.metal");
```

included in the shader table at `metal/mod.rs:439`.

### 3.1 Fast path — `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`

Signature (`q80_mixed_decode.metal:961–974`):

```metal
// Grid: ceil(rows/2)*128, TG 128. bits must be 3, group_size 64, cols % 64 == 0.
kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
```

Thread map (shader comment 910–920 and body 976–1002):

- TG size **128** = 4 simdgroups
- **2 rows / TG** (`row = group_id * 2 + team`, `team = simd_id / 2`)
- **64 threads / row** (`lane_in_row = split * 32 + simd_lane`)
- `col = lane_in_row * 8`, stride **512**
- `simd_sum` then 2-way threadgroup add (`red[4]`)
- One FP16 scale per 64-col group; 24 code bytes/group; `q = code - bound`
- Address by `(row, group)`, not `element * bits`

This is the same launch class as the Q4 geometry-sweep winner:

```181:182:crates/hawking-core/shaders/qwen_uniform_q4.metal
// Geometry-sweep winner for Q4 gate [512, 2048]: 64 threads/row, 128-thread
// TG, 2 rows/TG. Packed decode stays in registers. Grid: ceil(rows/2)*128.
```

Host launch (`qwen38_hgravu01_geo_tpr64_launch:578–580`) matches:
`tg=128`, `grid = ceil(rows/2)*128`.

### 3.2 Slow incumbent — `q80_hgravs01_factor_matvec_simd3`

Signature (`q80_mixed_decode.metal:843–857`):

```metal
// 3-bit specialized factor: 8 codes / lane, 256-col tiles.
// Grid: ceil(rows/8)*256, TG 256. bits must be 3, bound 3.
kernel void q80_hgravs01_factor_matvec_simd3(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    constant uint& rows             [[buffer(4)]],
    constant uint& cols             [[buffer(5)]],
    constant uint& group_size       [[buffer(6)]],
    constant uint& bits             [[buffer(7)]],
    constant uint& bound             [[buffer(8)]],
    uint group_id                    [[threadgroup_position_in_grid]],
    uint simd_lane                   [[thread_index_in_simdgroup]],
    uint simd_id                     [[simdgroup_index_in_threadgroup]])
```

Thread map (859–907):

- TG size **256** = 8 simdgroups
- **1 row / simdgroup**, **8 rows / TG** (`row = group_id * 8 + simd_id`)
- **32 threads / row** (one simdgroup)
- `col = simd_lane * 8`, stride **256**
- `simd_sum`; lane 0 writes the row
- 8-wide LSB unpack, then a 1-wide remainder via `q80_uniform_value_wide`
- **Per-element** scale lookup: `scales[(row_base + col + i) / group_size]` eight times
- Address via `gk_packed_lsb_byte(row_base + col, 3)` (element-linear)

`decode_family.rs:24–25` names this the occupancy tile of 3-bit
hgravs/uniform (`MATVEC_HGRAVS_TILES`).

Slower still (fuse OFF): `q80_hgravs01_factor_matvec` (233–254) — one
thread per row, serial `for (col = 0; col < cols; ++col)` left-to-right
association. Grid `(rows, 1, 1)`, TG 256.

### 3.3 Why geo is faster (from the shader, not a new bench)

On a 17408 × 5120 gate (this patient's MLP):

| | simd3 | geo_tpr64 |
|---|---|---|
| threads / row | 32 | 64 |
| cols / thread / iter | 8 | 8 |
| K-stride | 256 | 512 |
| K-loop trips (`K=5120`) | 20 | 10 |
| rows / TG | 8 | 2 |
| TG | 256 | 128 |
| scale loads / 8-wide | 8 (per-element `/ group_size`) | 1 (group scale) |
| code addressing | `gk_packed_lsb_byte(element, 3)` | `rgb * 24 + (local*3 >> 3)` |
| remainder path | 1-wide `value_wide` | none (`cols % 64 == 0`) |

geo does the same 8-wide 3-bit unpack (`hgravu01_q3_unpack8`, 922–943)
but:

1. **Twice the threads per row**, half the loop trips — the Q80
   `geo_tpr64_tg128` occupancy that won the 512×2048 sweep
   (`qwen38_hybrid_decode.rs:604–607`).
2. **One scale per group** instead of eight integer divides into the
   scale buffer (simd3 lines 882–889 vs geo 989–990).
3. **Group-addressed 24-byte tiles** instead of a per-element bit
   extract plus a 1-wide remainder (simd3 899–903).
4. **No `element * bits` path**, so the bits=4 sibling cannot wrap on
   `lm_head` (comment at 1007–1008). MLP q3 never hit that wrap
   (`floor(2^32 / 3 / 5120) ≫ 17408`).

simd3 is still an occupancy tile vs serial. geo is the same codec with
the sweep-winning thread map.

---

## 4. The ~3–4× claim — what was actually measured

No isolated MLP-kernel timer exists in the decoder or in the G1 receipt.
The number the campaign already recorded is **whole-token GPU** on this
same artifact, geo binary vs lane-91 incumbent (no geo q3/q4 names):

From `workspace/superwave/g1/g1-hgravu01-geo-tpr64.md` (INFORMAL / DIRTY,
Genesis resident, "not a complete-token wall"):

- geo median ≈ **46–48 ms/token**
  (`hgravu01-geo-france128.json`: `median_gpu_ns_per_token = 46346625`)
- incumbent simd ≈ **163 ms/token**
- ratio **163 / 46.35 ≈ 3.5×**

That delta includes every Uniform GEMV on mixed-q3mlp-v1 (192 MLP q3 +
attention q4 + `lm_head` q4), plus mixer / SwiGLU / residual. MLP is
the largest share of bytes, but the 3–4× is **not** an MLP-only figure.

G1 also recorded that the geo generate changed greedy vs the overflowing
incumbent `lm_head` (France → `<think>` then EOS; incumbent looped
"Paris"). That was a **bits=4 `lm_head` addressing bug**, not an MLP
numeric difference. Current source extract is overflow-safe
(`gk_packed_lsb_byte`, `gk_family.metal:179–187`); the in-tree test now
**requires** `lm_head` bit-identity (`qwen38_hybrid_decode.rs:5033–5047`).

A second 3–4× is not available from a flag on this patient.

---

## 5. Correctness: bit-identical or Doctor re-validation?

### MLP q3 (the EXEC lever)

In-tree test `hgravu01_geo_tpr64_matches_incumbent_on_real_tensors`
(4758–5055) launches incumbent simd3 and geo in one command buffer on
the real mixed-q3mlp-v1 tensors and asserts:

```5050:5054:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
                assert_eq!(
                    max_abs, 0.0,
                    "{name} max_abs={max_abs} is not bit-identical to incumbent"
                );
```

G1 measured table (same input `x[i] = (i%17)*0.125 - 1.0`):

| tensor | max_abs vs simd3 | max_rel | rms | n>\|d\|>1e-2 |
|---|---:|---:|---:|---:|
| L0 gate 17408×5120 | 0 | 0 | 0 | 0 |
| L0 up 17408×5120 | 0 | 0 | 0 | 0 |
| L0 down 5120×17408 | 0 | 0 | 0 | 0 |

**MLP geo_tpr64 is bit-identical to simd3.** Association matches the
CPU serial oracle `uniform_factor_value` on the small organs the test
walks (`n <= 48`). Switching simd3 → geo on MLP does **not** need a
Doctor numeric re-validation.

### Non-MLP caveat (do not mix into the MLP lever)

Historically, bits=4 `lm_head` (248320×5120) **was not** bit-identical:
incumbent `element * bits` wrapped in uint32 at row **209715**. geo
addressed by `(row, group)` and matched CPU; simd did not. That extract
is now `gk_packed_lsb_byte` and the same test requires
`first_bad == None` and `max_abs == 0` on `lm_head` too. That is a
**bits=4 addressing** story, already closed in this source. It is not
an MLP q3 numeric delta.

Fuse OFF serial (`q80_hgravs01_factor_matvec`) is the
left-to-right association baseline (comment at 230–232). Occupancy
tiles (simd3 and geo) both `simd_sum` partials; for this codec they
have been measured bit-identical to each other on MLP.

---

## 6. How to claim the fast path (operator)

1. Do **not** set `HAWKING_QWEN38_RECON_FUSE=0`.
2. Open the mixed catalog. Stderr must contain
   `recon_fuse=ON` (from `qwen38_mixed_k_complete_bind_message`) and
   `uniform=498` / `hgravs=0` (from the census print at 1041–1052).
3. That is sufficient: `dispatch_uniform` will take the `Some` branch
   for every MLP GEMV on this patient.
4. There is no additional config key, catalog flag, or kernel enum to
   flip. `Qwen38MatvecKernel` (608–623) retargets **HQ30UQ4 / Q4**
   launch geometry only, not HGRAVU01 q3.

To **leave** the fast path (debug only):
`HAWKING_QWEN38_RECON_FUSE=0` → serial factor. That is a slowdown.

To **force simd3 while fuse is ON** (A/B the 3–4×): no env exists.
Minimal code change would be making `qwen38_hgravu01_geo_tpr64_launch`
return `None` (or skipping the `if let Some` in `dispatch_uniform`).
This contract forbids that change.

---

## 7. What the static 64-layer schedule does **not** tell you

`qwen38_64_layer_execution_schedule.rs:41–47` lists the MLP suffix as
three copies of `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.
That is the **HQ30UQ4 / dense-Q4** name. Mixed-q3mlp-v1 never dispatches
that symbol for MLP; it dispatches
`qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` via `encode_named_matvec`.
The schedule is asserted intact on session open (875, 1278) as a
dispatch-count / mixer-prefix lock, not as the mixed kernel selector.

---

## 8. Evidence (raw locator output)

```
$ rg -n "fn dispatch_uniform|fn dispatch_factor|fn qwen38_hgravu01_geo_tpr64_launch|fn qwen38_recon_fuse_enabled|HAWKING_QWEN38_RECON_FUSE" \
    crates/hawking-core/src/model/qwen38_hybrid_decode.rs
43:/// `HAWKING_QWEN38_RECON_FUSE=0` selects the G023 serial family names.
44:pub fn qwen38_recon_fuse_enabled() -> bool {
45:    crate::env_opt_out("HAWKING_QWEN38_RECON_FUSE")
564:pub fn qwen38_hgravu01_geo_tpr64_launch(
1720:        fn dispatch_factor(
1800:        fn dispatch_uniform(

$ rg -n "kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128|kernel void q80_hgravs01_factor_matvec_simd3|kernel void qwen_uniform_hgravu_q4" \
    crates/hawking-core/shaders
crates/hawking-core/shaders/qwen_uniform_q4.metal:1530:kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128_alignedload(
crates/hawking-core/shaders/q80_mixed_decode.metal:845:kernel void q80_hgravs01_factor_matvec_simd3(
crates/hawking-core/shaders/q80_mixed_decode.metal:962:kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128(
crates/hawking-core/shaders/q80_mixed_decode.metal:1009:kernel void qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128(

$ git ls-tree -r --name-only HEAD -- workspace/campaign | rg metal_shared_op_blueprint
# (empty — blueprint not in this tree)
```

---

## 9. Acceptance checklist

- [x] Exact dispatch branch: `dispatch_uniform` @
      `qwen38_hybrid_decode.rs:1800`, predicate
      `qwen38_hgravu01_geo_tpr64_launch` @ 564 / call @ 1807;
      fallback `dispatch_factor` @ 1720 (`simd3` @ 1746).
- [x] Both kernel names + signatures + TG geometry + why geo is faster,
      cited from `q80_mixed_decode.metal:845` and `:962`.
- [x] `HAWKING_QWEN38_RECON_FUSE` is default-ON occupancy vs serial
      (`qwen38_recon_fuse_enabled` @ 44–45, `env_opt_out` @ `lib.rs:207`).
      It is **not** a geo-vs-simd3 switch.
- [x] **YES**: fast path by env alone — leave the var unset (or not a
      disable token). **NO** leftover 3–4× on the default patient.
- [x] Correctness: MLP q3 geo is **bit-identical** to simd3
      (`max_abs == 0`). No Doctor re-validation for the MLP lever.
      Historical `lm_head` mismatch was bits=4 addressing, now fixed.
