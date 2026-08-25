# Metal shared-operator MLP — implementation blueprint

**Status:** design only. This document does not change the decoder or any
`.rs` / `.metal` file. It is the implementation-ready plan for a new
mixed-catalog MLP execution kind on the Qwen3.8 hybrid decoder.

**Vehicle:** `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
(5950 lines) + group-64 GEMV kernels in `crates/hawking-core/shaders`.

**q3 patient (live on this host):**
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`

---

## 0. What we are adding

One SwiGLU operator `(G, U, D)` reused across all 64 layers, plus a tiny
per-layer FiLM code `(γ_L, β_L)`. Per layer:

```
inter = silu(x @ G.T) * (x @ U.T)     # width m, not native 17408
inter = inter * γ_L + β_L
y     = inter @ D.T                   # back to hidden 5120
```

`G, U ∈ R^{m×5120}`, `D ∈ R^{5120×m}`, `m ∈ {6144, 8192}` (both
multiples of the group-64 kernel constraint). Weights stay HGRAVU01
group-64 packed. The three Metal buffers are allocated once and rebound
every layer; only the input activation and the FiLM code change.

This is a **new selectable mixed-catalog kind**, not a replacement for
Binary / Residual / Hgravs / Uniform. A pack may mix: 63 native layers +
one shared-op layer (the one-layer-swap test), or all 64 layers on the
shared operator.

Physics: the q3 patient streams **108.6 MB of distinct MLP bytes per
layer** (6.95 GB / token across 64 layers). The shared operator streams
**~38.3 MB once**, then reuses it; 64 FiLM codes add 3.1 MB. Apple
Silicon SLC can hold 38 MB. It cannot hold 6.95 GB.

---

## 1. Exact per-layer MLP GEMV dispatch sites

Token entry is `Qwen38HybridDecodeSession::step` (line 3861). It encodes
embed → layers → terminal into one `TokenCommandBuffer`:

```3861:3871:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            // ...
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
```

The layer loop is `encode_layers` (3352). Mixer first, then MLP:

```3352:3359:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        fn encode_layers(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            for layer in 0..QWEN38_LAYERS {
                match qwen38_mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer)?,
                    Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer)?,
                }
                self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
            }
```

`QWEN38_LAYERS = 64`, `QWEN38_HIDDEN = 5120`, `QWEN38_INTERMEDIATE = 17408`
(`qwen38_geometry.rs` 20–25). Layer tensor names come from
`qwen38_layer_name(layer, suffix)` →
`language_model.model.layers.{layer}.{suffix}` (geometry.rs:119).

### Mixed catalog (the path this kind lives on)

`encode_dense_mlp` (3040) redirects to `encode_dense_mlp_mixed` whenever
`self.weights.mixed` is non-empty (3046–3048). That is the **production
MLP graph** for HQ38M20 artifacts, including the q3 patient:

```3441:3490:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        fn encode_dense_mlp_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            self.encode_rmsnorm(/* post_attention_layernorm → workspace.normalized */)?;
            self.encode_named_matvec(tcb, &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized, &self.workspace.gate)?;
            self.encode_named_matvec(tcb, &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized, &self.workspace.up)?;
            tcb.dispatch_threads(crate::decode_family::swiglu_f32(), /* n=17408 */ ...)?;
            self.encode_named_matvec(tcb, &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act, &self.workspace.down)?;
            qwen_next_add_residual_tcb(tcb, input, &self.workspace.down, &self.workspace.hidden, QWEN38_HIDDEN)
        }
```

GEMV fan-in is `encode_named_matvec` (1541). Mixed hits
`encode_mixed_matvec` (1559). Uniform (HGRAVU01, the q3-patient MLP kind)
hits `dispatch_uniform` (1800), which binds geo_tpr64 when
`qwen38_hgravu01_geo_tpr64_launch` returns `Some`:

```1800:1826:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        fn dispatch_uniform(
            &self, tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuUniform, input: &PinnedBuffer, output: &PinnedBuffer,
        ) -> Result<()> {
            if let Some((name, grid, tg)) = qwen38_hgravu01_geo_tpr64_launch(
                body.bits, body.group_size, body.rows, body.cols,
            ) {
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_factor_args(enc, &body.codes, &body.scales,
                        input, output, body.rows, body.cols,
                        body.group_size, body.bits, body.bound)
                });
            }
            self.dispatch_factor(/* incumbent simd / simd3 / uniform8 / serial */)
        }
```

Launch gate (564–580): recon-fuse ON, `group_size == 64`, `cols % 64 == 0`,
bits ∈ {3, 4}. Grid = `ceil(rows/2)*128`, threadgroup = 128.

### Isolated / profiling siblings (must take the same SharedOp branch)

These are not on the token path but they encode the same three GEMVs and
will lie if they keep looking up per-layer names on a SharedOp layer:

| Function | Lines | What it encodes |
|---|---|---|
| `encode_mlp_matvecs_only` | 2119–2141 | redirects mixed → `encode_mlp_matvecs_only_mixed` |
| `encode_mlp_matvecs_only_mixed` | 3836–3858 | gate + up + down, **no** rmsnorm / swiglu / residual |
| `measure_isolated_mlp_full` | 2203–2209 | 64 × `encode_dense_mlp` |
| `measure_isolated_mlp_matvecs` | 2212–2218 | 64 × `encode_mlp_matvecs_only` |
| `measure_isolated_mlp_one_proj` | 2221–2247 | 64 × `encode_q4_matvec` on one role (Q4 catalog only) |
| `step_decomposed` | 2931–2957 | per-layer `encode_dense_mlp` for class timing |

`generate_greedy` (3928) is the example-runner entry: it only calls
`session.step`, so it inherits whatever `encode_dense_mlp_mixed` does.

### Non-mixed Q4 catalog (out of SharedOp scope)

`encode_dense_mlp` Q4 arm (3057–3082) uses `encode_independent_q4_pair` +
`encode_q4_matvec` against `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.
SharedOp is a **mixed-catalog** kind. Do not add it to the Q4 arm.

---

## 2. `MixedMlpNativeKind::SharedOp` thread-through

Today the kind is a **codec-lane → role lock**. SharedOp is an
**execution kind selected by tensor name**, sitting on top of the
existing HGRAVU01 (codec 3) packing. No new HQ38M20 codec. Codec 5
already refuses (`unknown_codec_5_still_refuses`, line 4406).

### 2.1 Current chain (q3 patient walks this)

```
catalog.hq38m20 row
  → parse_qwen38_mixed_catalog (412)
  → classify_qwen38_mixed_payload (160)
       codec 3 + magic HGRAVU01 + not a vector → MixedCatalogLane::Packed(3)
  → mixed_mlp_native_kind_from_lane (215)
       Packed(3) → MixedMlpNativeKind::Uniform
  → is_mixed_mlp_gemv_name (296) filters gate/up/down
  → assert_mixed_mlp_native_kinds (250)
       per layer, per suffix, mixed_mlp_role_allowed (228)
       Uniform is legal on all three roles
  → load_mixed (950) upload_mixed (1341) → MixedGpuWeight::Uniform
  → Qwen38HybridDecodeSession::assert_mixed_mlp_native (1330)
       maps MixedGpuWeight::* → MixedMlpNativeKind::*
  → encode_dense_mlp_mixed → encode_named_matvec
       → encode_mixed_matvec match Uniform → dispatch_uniform
```

`mixed_mlp_native_kind_from_lane` today:

```215:226:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
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
```

Role lock (`mixed_mlp_role_allowed` 228–244):

| suffix | allowed kinds |
|---|---|
| `mlp.gate_proj.weight` | Binary \| Uniform |
| `mlp.up_proj.weight` | Residual \| Uniform |
| `mlp.down_proj.weight` | Hgravs \| Uniform |

`assert_mixed_mlp_native_kinds` (250–276) **requires all 64 × 3
per-layer names**. A SharedOp artifact that omits
`layers.{L}.mlp.gate_proj.weight` currently dies with
`missing {name}; refusing silent dense/Q4 fallback`. That is the lock
that must grow a SharedOp arm.

### 2.2 Naming additions (minimal; no record-layout change)

HQ38M20 already carries arbitrary UTF-8 names in the name blob
(`parse_qwen38_mixed_catalog` 470–475). Add names, not codecs.

**Shared operator (codec 3, magic `HGRAVU01`, group_size=64, bits=3
default / 4 allowed):**

| catalog name | shape | role |
|---|---|---|
| `language_model.model.shared_mlp.gate_proj.weight` | `[m, 5120]` | G |
| `language_model.model.shared_mlp.up_proj.weight` | `[m, 5120]` | U |
| `language_model.model.shared_mlp.down_proj.weight` | `[5120, m]` | D |

These **already** match `is_mixed_mlp_gemv_name` (`ends_with("mlp.gate_proj.weight")`
etc.) and `hgravu_is_vector` treats them as GEMVs (so they stay Packed(3),
not HgravuVector). `assert_mixed_mlp_native_kinds` looks up
`qwen38_layer_name(layer, suffix)` and will **not** see them until the
admission loop is taught the shared names.

**Per-layer FiLM (codec 4, f32v2 — same lane as RMSNorm vectors):**

| catalog name | shape |
|---|---|
| `language_model.model.layers.{L}.mlp.shared_op.gamma` | `[m]` |
| `language_model.model.layers.{L}.mlp.shared_op.beta` | `[m]` |

Codec 4 is already classified as `MixedCatalogLane::F32v2` and uploaded
into `Qwen38HybridWeights.f32s` (1018–1027). `mlx_residual_norm_to_delta_named`
only rewrites `*norm.weight` / `model.norm.weight` names (53–57), so
FiLM values are stored as packed.

Identity / plumbing packs may set `γ = 1`, `β = 0` and `m = 17408` with
G/U/D aliased to one native layer's HGRAVU01 bodies.

### 2.3 New helpers (add next to the existing kind functions)

```
const SHARED_MLP_GATE: &str = "language_model.model.shared_mlp.gate_proj.weight";
const SHARED_MLP_UP:   &str = "language_model.model.shared_mlp.up_proj.weight";
const SHARED_MLP_DOWN: &str = "language_model.model.shared_mlp.down_proj.weight";

fn is_shared_op_weight_name(name: &str) -> bool
fn is_shared_op_film_name(name: &str) -> bool          // *.mlp.shared_op.{gamma,beta}
fn shared_op_film_name(layer: usize, which: &str) -> String
    // qwen38_layer_name(layer, "mlp.shared_op.gamma" | "mlp.shared_op.beta")

pub fn mixed_mlp_native_kind_from_row(name: &str, lane: MixedCatalogLane)
    -> Option<MixedMlpNativeKind>
    // if is_shared_op_weight_name(name) && lane == Packed(3) => Some(SharedOp)
    // else mixed_mlp_native_kind_from_lane(lane)
```

Do **not** make `mixed_mlp_native_kind_from_lane(Packed(3))` return
SharedOp. That would reclassify every Uniform MLP tensor in mixed-q3mlp-v1.

### 2.4 Every function that needs a new match arm

Exhaustive against today's `match` / `enum` sites in this decoder.
Sites that stay Uniform-as-payload (no new arm) are listed so implementers
do not invent them.

#### Must add `SharedOp` (admission + encode)

| # | Function | Lines | Arm |
|---|---|---|---|
| 1 | `MixedMlpNativeKind` | 208–213 | add `SharedOp` |
| 2 | `mixed_mlp_native_kind_from_row` | **new**, beside 215 | name+lane → SharedOp |
| 3 | `mixed_mlp_role_allowed` | 228–244 | SharedOp is **not** a per-suffix native kind. Leave this function's three arms unchanged. Admission of SharedOp happens one level up. |
| 4 | `assert_mixed_mlp_native_kinds` | 250–276 | rewrite the layer loop (see §2.5) |
| 5 | `is_mixed_mlp_gemv_name` | 296–300 | also true for the three `shared_mlp.*` names (already true via `ends_with`) and **false** for film names (already false) |
| 6 | `assert_mixed_mlp_native_catalog` | 305–322 | classify via `mixed_mlp_native_kind_from_row`; also record that shared weights exist and which layers have both film vectors |
| 7 | `Qwen38HybridDecodeSession::assert_mixed_mlp_native` | 1330–1338 | pass a lookup that returns `SharedOp` for the three shared names (they live on `weights.shared_mlp`, not in the per-name `mixed` map) |
| 8 | `encode_dense_mlp_mixed` | 3441–3491 | `if self.layer_uses_shared_op(layer) { encode_shared_op_mlp(...) } else { existing }` |
| 9 | `encode_mlp_matvecs_only_mixed` | 3836–3858 | same branch: three GEMVs against shared G/U/D, skip film (this helper is GEMV-only) |
| 10 | `Qwen38HybridWeights` | 866–871 | add `shared_mlp: Option<SharedMlpOp>`, `film_gamma: HashMap<usize, PinnedBuffer>`, `film_beta: HashMap<usize, PinnedBuffer>` |
| 11 | `Qwen38HybridWeights::load_mixed` | 950–1061 | divert `shared_mlp.*` into `SharedMlpOp`; divert `mlp.shared_op.{gamma,beta}` into the film maps; do **not** insert shared G/U/D 64 times |
| 12 | unit tests | 4416–4545 | `filled_mlp_kinds` stays native-only; add `shared_op_admits_when_film_and_operator_present`, `shared_op_one_layer_swap_admits`, `shared_op_without_film_refuses`, `native_q3_patient_still_admits` |

#### Do **not** add a SharedOp arm (payload stays Uniform / f32)

| Function | Lines | Why |
|---|---|---|
| `mixed_mlp_native_kind_from_lane` | 215 | lane-only; SharedOp is name-qualified |
| `classify_qwen38_mixed_payload` | 160–201 | no new codec |
| `MixedCatalogLane` | 69–74 | reuse Packed(3) + F32v2 |
| `MixedGpuWeight` | 835–840 | G/U/D stay `Uniform(GpuUniform)` |
| `upload_mixed` | 1341–1506 | codec 3 already builds `GpuUniform` |
| `encode_mixed_matvec` | 1559–1576 | shared GEMVs call `dispatch_uniform` directly |
| `dispatch_uniform` / `encode_factor_args` | 1800 / 1634 | reused as-is |
| `dispatch_binary` / `dispatch_residual` / `dispatch_hgravs` / `dispatch_factor` | 1658–1798 | native kinds only |
| `census_qwen38_mixed_catalog` | 325 | optional `shared_op` counter; not required to decode |
| mixer / embed / terminal / GQA / DeltaNet | — | out of scope |

`#[derive(Eq, PartialEq)]` on `MixedMlpNativeKind` means every test
literal stays compiling; adding a variant is non-breaking for existing
`matches!` arms that list Uniform/Binary/Residual/Hgravs.

### 2.5 Admission rewrite (`assert_mixed_mlp_native_kinds`)

Keep the function public and the 64-layer walk. Change the body to a
per-layer alternative:

```
shared = lookup(SHARED_MLP_GATE/UP/DOWN) are all Some(SharedOp)
         and the three GpuUniform geometries agree
         (gate.rows == up.rows == down.cols == m,
          gate.cols == up.cols == down.rows == 5120,
          group_size == 64, bits ∈ {3,4}, m % 64 == 0,
          m ∈ 64..=QWEN38_INTERMEDIATE)

for layer in 0..64:
    has_film = film_gamma[layer] and film_beta[layer] both present, length m
    has_native = the existing three-suffix role lock would pass for this layer

    if has_film && shared:
        admit SharedOp for this layer          # one-layer-swap OR full
    else if has_native:
        admit native kind                      # q3 patient unchanged
    else:
        refuse with a name that says which of
        (native triple, shared operator, film pair) was missing
```

A full shared-op pack has `shared=true`, `has_film` on all 64, and
**zero** `layers.{L}.mlp.{gate,up,down}_proj.weight` rows. A one-layer
swap has `shared=true`, `has_film` on exactly layer L, and native triples
on the other 63. The q3 patient has `shared=false` and native on all 64
— today's tests (`mixed_mlp_uniform_is_admitted_on_every_role`,
`mixed_q3mlp_and_q4down_pass_mlp_admission`) keep passing.

`lookup` today is `Fn(&str) -> Option<MixedMlpNativeKind>`. Either
extend it (film is not a MixedMlpNativeKind) or change the signature to
take a small `MixedMlpAdmission` view. Prefer a new struct over
overloading the kind enum with film.

### 2.6 Load-time residency object

```
struct SharedMlpOp {
    gate: GpuUniform,   // rows=m, cols=5120
    up:   GpuUniform,   // rows=m, cols=5120
    down: GpuUniform,   // rows=5120, cols=m
    width: u32,         // m
}
```

`load_mixed` (978–984) currently does
`mixed.insert(row.name.clone(), upload_mixed(...)?)` for every Packed
row. Shared names go to `shared_mlp = Some(SharedMlpOp { ... })`
instead. Film codec-4 rows already land in `f32s`; after the loop,
harvest them into `film_gamma` / `film_beta` by parsing the layer index
out of the name so encode does not string-format on the hot path.

`resident_bytes` (1063) must add `shared_mlp` + film. Today it only
sums `q4` + `f32s` + `mixed`. If film stays in `f32s` it is already
counted; `shared_mlp` is not.

---

## 3. Metal kernel plan

### 3.1 Reuse the group-64 HGRAVU01 geo_tpr64 kernels for all three GEMVs

Shared-op G/U/D are HGRAVU01, so they use the **HGRAVU01** kernels in
`q80_mixed_decode.metal`, **not** the HQ30UQ4 kernel in
`qwen_uniform_q4.metal`. The bind signatures differ.

**bits=3** — `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`
(`q80_mixed_decode.metal:962`):

```962:974:crates/hawking-core/shaders/q80_mixed_decode.metal
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

**bits=4** — `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128`
(`q80_mixed_decode.metal:1009`), identical buffer map.

Host bind is already `encode_factor_args` (1634–1656):

```
buf0 codes | buf1 scales | buf2 input | buf3 output
bytes4 rows | bytes5 cols | bytes6 group_size | bytes7 bits | bytes8 bound
```

Launch via existing `qwen38_hgravu01_geo_tpr64_launch(bits, 64, rows, cols)`
(564–580). Constants:

```556:559:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
pub const QWEN38_HGRAVU01_Q3_GEO_TPR64: &str =
    "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_HGRAVU01_Q4_GEO_TPR64: &str =
    "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128";
```

Do **not** bind `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
(`qwen_uniform_q4.metal:183`). That kernel takes `groups_per_row` at
buffer 6 and has no bits/bound. It is the HQ30UQ4 / Q4-catalog path
(`encode_q4_matvec_kernel` 1933). Mixing the two unpack conventions
(nibble-8 vs bound=7) is a silent numerical bug.

Kernel constraint `cols % 64 == 0` (shader 983 / 1030, host 570):

| GEMV | rows | cols | ok at m=6144 / 8192 |
|---|---|---|---|
| `x @ G.T` | m | 5120 | 5120 = 80×64 |
| `x @ U.T` | m | 5120 | same |
| `inter @ D.T` | 5120 | m | 6144 = 96×64, 8192 = 128×64 |

Grid at m=6144: G/U = `ceil(6144/2)*128 = 393216`, D = `ceil(5120/2)*128 = 327680`.
Native G/U at 17408 is `1_114_112`. Shared GEMVs are cheaper to launch
as well as cheaper to stream.

### 3.2 Elementwise: silu\*up and FiLM

**silu(gate)\*up already exists.** Do not fold it into geo_tpr64 (that
kernel writes one output row from codes/scales/x; it has no second
vector and no affine). Do not edit `gk_swiglu_f32` (G023 family, shared
with Q80).

```597:606:crates/hawking-core/shaders/gk_family.metal
kernel void gk_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* output     [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint id                   [[thread_position_in_grid]])
{
    if (id >= n) return;
    output[id] = gk_silu_f32(gate[id]) * up[id];
}
```

Host: `decode_family::swiglu_f32()` (168) → `"gk_swiglu_f32"` when
`HAWKING_DECODE_FAMILY` is default-on. Today's mixed MLP already
dispatches it at `n = QWEN38_INTERMEDIATE` (3467–3476). SharedOp
dispatches the same kernel at `n = m`.

**FiLM is new.** There is no `x * γ + β` kernel in the Qwen3.8 graph.
Closest cousins: `qwen_next_add_residual` (add only,
`qwen_next.metal:328`) and `gravity_layernorm_affine_f32` (wrong file,
wrong contract). Add a tiny kernel next to the other Q38 activations:

File: `crates/hawking-core/shaders/qwen38_device_activations.metal`
(already compiled; `SHADER_QWEN38_DEVICE_ACTIVATIONS` at
`crates/hawking-core/src/metal/mod.rs:371`).

```
kernel void qwen38_mlp_film_f32(
    device const float* inter   [[buffer(0)]],
    device const float* gamma   [[buffer(1)]],
    device const float* beta    [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& n            [[buffer(4)]],
    uint id                      [[thread_position_in_grid]])
{
    if (id >= n) return;
    output[id] = inter[id] * gamma[id] + beta[id];
}
```

Launch: grid `(m, 1, 1)`, tg `(min(m, 256), 1, 1)` — same pattern as
`gk_swiglu_f32` in `encode_dense_mlp_mixed` (3468–3469). In-place is
legal (`inter` and `output` both `workspace.act`) if the device hazard
tracker is on; the token path already relies on that for ordered
encoders (`metal/mod.rs:3394–3397`). Prefer `act` → `act` to avoid a
fourth mid-width buffer.

Register the name in
`qwen38_device_activation_kernels_are_trace_named_and_compiled`
(`metal/mod.rs:2194–2218`) so the compile/trace contract sees it.

**Fold vs new kernel (decision):** new tiny kernel. Folding FiLM into
`gk_swiglu_f32` would change a shared Q80/Q38 family entry. Folding
into geo_tpr64 would couple a 9-buffer GEMV to two extra f32 vectors
and break the reuse goal. A later fused `qwen38_swiglu_film_f32` (one
dispatch instead of two) is a dispatch-count nicety, not a residency
requirement — do not build it in v1.

### 3.3 SharedOp encode sequence (`encode_shared_op_mlp`)

Workspace already allocates `gate` / `up` / `act` at
`QWEN38_INTERMEDIATE = 17408` f32s (`Qwen38HybridWorkspace::allocate`
1179–1181). `m ≤ 8192` writes a prefix. No workspace resize.

```
encode_rmsnorm(input, layers.{L}.post_attention_layernorm.weight,
               workspace.normalized, 5120)                  # existing

dispatch_uniform(shared.gate, normalized, gate)             # x @ G.T → R^m
dispatch_uniform(shared.up,   normalized, up)               # x @ U.T → R^m

gk_swiglu_f32(gate, up, act, n=m)                           # silu*up
qwen38_mlp_film_f32(act, γ_L, β_L, act, n=m)                # FiLM

dispatch_uniform(shared.down, act, down)                    # inter @ D.T → R^5120
qwen_next_add_residual(input, down, hidden, 5120)           # existing
```

`dispatch_uniform` on a `GpuUniform` borrowed from `shared_mlp` is the
whole residency trick: same `PinnedBuffer` pointers every layer.

Gate and up are independent. `concurrent_independent` (1259–1262) is
off by default on `step` for bit-identity. v1 stays serial, matching
`encode_dense_mlp_mixed`. Do not invent a concurrent pair here.

### 3.4 Buffer binding cheat-sheet

| dispatch | buf0 | buf1 | buf2 | buf3 | scalars |
|---|---|---|---|---|---|
| geo_tpr64 G | `shared.gate.codes` | `shared.gate.scales` | `normalized` | `gate` | rows=m, cols=5120, gs=64, bits, bound |
| geo_tpr64 U | `shared.up.codes` | `shared.up.scales` | `normalized` | `up` | same |
| `gk_swiglu_f32` | `gate` | `up` | `act` | — | n=m |
| `qwen38_mlp_film_f32` | `act` | `film_gamma[L]` | `film_beta[L]` | `act` | n=m |
| geo_tpr64 D | `shared.down.codes` | `shared.down.scales` | `act` | `down` | rows=5120, cols=m, gs=64, bits, bound |
| `qwen_next_add_residual` | `input` | `down` | `hidden` | — | elements=5120 |

---

## 4. Residency plan

### 4.1 What is already resident

`Qwen38HybridWeights` is one Metal copy of the catalog. Sessions hold
`Arc<Qwen38HybridWeights>` and allocate only workspace / KV (`attach`
1277, comment at 864–865). `load_mixed` uploads **every** catalog row
once at open. The q3 patient therefore already has all 192 MLP tensors
in VRAM at once (6.95 GB of MLP). Residency of capacity is not the
bug.

The bug is **address diversity per token**. Each layer rebinds a
different `MTLBuffer`. The GPU streams 108.6 MB of new bytes per layer
from DRAM. SLC cannot hold 6.95 GB, so every layer is a cold stream.

### 4.2 What forces per-layer weight binding today

1. **Catalog shape.** mixed-q3mlp-v1 has 192 distinct MLP rows
   (`layers.{0..63}.mlp.{gate,up,down}_proj.weight`), each with its own
   segment and `nbytes=36208920`. Parsed by `parse_qwen38_mixed_catalog`
   (412) and uploaded independently.

2. **HashMap key.** `load_mixed` 978–984 inserts under `row.name`.
   Encode looks up `qwen38_layer_name(layer, "mlp.gate_proj.weight")`
   (3457, 3463, 3480, 3843, 3849, 3855). 64 names → 64 buffer pairs.

3. **Admission.** `assert_mixed_mlp_native_kinds` (258–274) refuses a
   missing per-layer name. You cannot legally ship one G/U/D triple
   today.

4. **`encode_named_matvec` (1548–1552)** is a name → buffer map. It has
   no notion of "this layer aliases the shared operator".

Nothing in Metal requires this. `dispatch_uniform` only needs a
`&GpuUniform`. The same three `GpuUniform`s can be passed for every
layer.

### 4.3 What changes so G/U/D persist

| stage | today | shared-op |
|---|---|---|
| catalog | 192 MLP GEMV rows | 3 `shared_mlp.*` rows + 64×2 film (full) or 189 native + 3 shared + 2 film (one-layer swap) |
| `load_mixed` | 192 `mixed.insert(layer_name, Uniform)` | 3 uploads into `SharedMlpOp`; film into `f32s` / film maps |
| VRAM | 192 distinct code+scale pairs | 3 code+scale pairs + 128 tiny f32 vectors |
| encode | `mixed.get(layer_name)` every GEMV | `shared_mlp.gate/up/down` every GEMV; `film_gamma[L]` / `film_beta[L]` once |
| first layer | cold stream 108.6 MB | cold stream 38.3 MB |
| layers 1..63 | cold stream 108.6 MB each | same 38.3 MB, SLC-hot |

`Arc` cloning already shares the weight set across sessions
(`share_weights` 1308, `measure_shared_weight_fanout` 4113). SharedOp
rides that: one `SharedMlpOp` per process, N sessions.

Do **not** re-upload G/U/D inside `step` / `encode_shared_op_mlp`. Do
**not** copy them into workspace. Rebind the existing `PinnedBuffer`s.

### 4.4 One-layer-swap residency

Layer L binds `shared_mlp.{gate,up,down}` + `film_*[L]`.
Layers ≠ L bind `mixed[layers.{k}.mlp.*]` as today.
Working set per token = 38.3 MB shared + 49 KB film_L + 63 × 108.6 MB
native. That is a **correctness** configuration, not the physics win.
The physics win is the full-64 SharedOp pack.

### 4.5 What we deliberately do not change

- HQ38M20 magic / version / 128-byte record (`QWEN38_MIXED_CATALOG_*`
  at 32–35).
- Codec numbers 0–4.
- `GpuUniform` layout (`codes`, `scales`, `rows`, `cols`, `group_size`,
  `bits`, `bound`) at 825–833.
- Workspace formula `qwen38_workspace_bytes` (697). Film is a weight,
  not workspace. Mid-width buffers stay 17408 so a mixed
  native+SharedOp session still fits native SwiGLU.

---

## 5. One-layer-swap test plan vs the q3 patient

### 5.1 Patient

Artifact: `mixed-q3mlp-v1` (campaign path above).

Live catalog census (parsed from `catalog.hq38m20` on this host,
2026-08-18):

- magic `HQ38M20\0`, version 1, 851 tensors, 258 segments
- 192 MLP rows, all codec 3
- every `gate_proj` / `up_proj`: shape `(17408, 5120)`, `nbytes=36208920`
- every `down_proj`: shape `(5120, 17408)`, `nbytes=36208920`
- L0 gate header (HGRAVU01): `bits=3`, `group_size=64`,
  `code_bytes=33423360`, `scale_bytes=2785280`,
  `representation=uniform_q3_group_scale`
- no `shared_mlp` / `film` names

Admission already covered by
`mixed_q3mlp_and_q4down_pass_mlp_admission` (4521) and
`hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` (4758).

### 5.2 Runner (existing; do not write a new one)

`crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs`

Entry: `Qwen38HybridDecodeSession::open` → `generate_greedy` (line 277–280
of the example; decoder 3928). Surfaces:

- `GENERATED_TEXT_VERBATIM`
- `NEW_TOKENS` / `generated_token_ids`
- `FALLBACKS` (must stay 0)
- `--prompts-file` coherence mode (`run_prompts_file`, example 620)
- `--complete-wall` for dispatch count / gpu_ns (example 332)

Build (repo convention):

```
cargo build --release --example ascension_qwen38_hybrid_greedy \
  --target-dir workspace/ops/build/rust
```

### 5.3 Three rungs, in this order

**Rung A — bind identity (bit-identical).**
Build a catalog that keeps all 64 native MLP triples **and** adds:

- `shared_mlp.{gate,up,down}` = **byte-identical copies** (or catalog
  aliases into the same segment offset) of layer L's three HGRAVU01
  bodies, so `m = 17408`
- `layers.{L}.mlp.shared_op.gamma` = ones(17408)
- `layers.{L}.mlp.shared_op.beta`  = zeros(17408)

Admission: layer L is SharedOp, others native. Encode on L runs the
SharedOp sequence at native width with γ=1, β=0. Numerically this is
`silu(x@G_L.T)*(x@U_L.T)` then `@ D_L.T` — the same three GEMVs +
`gk_swiglu_f32` the patient already runs. Expect:

- greedy token ids equal to the patient on the same prompt
- `read_f32_workspace("logits", QWEN38_VOCAB)` after the first `step`
  matches the patient within geo_tpr64 self-parity (the incumbent-vs-geo
  test at 4758 is the tolerance authority; do not invent a looser one)
- `FALLBACKS=0`

If Rung A disagrees, the bind is wrong. Do not proceed.

**Rung B — one-layer reduced-m swap (quality).**
Same as A, but `shared_mlp.*` is a real m∈{6144,8192} operator and
`(γ_L, β_L)` is the FiLM fit for layer L. Native triples remain on
layers ≠ L. Layer L's native triple may be omitted (saves 108.6 MB VRAM)
or left in the catalog unused.

Compare against the patient with the existing runner:

```
# patient
./tools/gpu_lane_lock.sh qwen38-sharedop-patient \
  workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy \
  --artifact-root .../mixed-q3mlp-v1 \
  --tokenizer .../qwen38-27b/bf16/tokenizer.json \
  --prompt "Say hi." --max-new-tokens 16 \
  --out workspace/campaign/records/runs/qwen38-27b/sharedop-patient-sayhi.json

# swap layer L
./tools/gpu_lane_lock.sh qwen38-sharedop-swap-L \
  workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy \
  --artifact-root .../mixed-q3mlp-sharedop-L{L}-m6144-v1 \
  --tokenizer .../qwen38-27b/bf16/tokenizer.json \
  --prompt "Say hi." --max-new-tokens 16 \
  --out workspace/campaign/records/runs/qwen38-27b/sharedop-L{L}-sayhi.json
```

Diff `new_token_ids` and `generated_text`. Repeat with
`--prompts-file .../qwen38-27b/coherence_prompts.txt` (already on the
campaign disk) via the example's prompts-file path.

Logit probe (not in the example; use a unit test or a one-shot that
opens two sessions, `step`s the same prompt token, and
`read_f32_workspace("logits", 248320)`): report max-abs, RMS,
argmax agreement. A single swapped layer is allowed to move logits;
the gate is **coherence** (greedy text still English / still answers
the prompt), not bit-identity. Record the numbers. Do not hide a
collapse behind "approx".

Suggested L for the first swap: **layer 0** (first MLP, after a
DeltaNet mixer) and **layer 3** (first GQA layer). Mixer kind is
`(layer+1) % 4 == 0` → GQA (`qwen38_mixer_kind`, geometry.rs:85–96).
Two mixers, one swap each, proves SharedOp is mixer-agnostic.

**Rung C — full 64-layer SharedOp (physics).**
Only after Rung B is coherent. All 64 layers take the shared operator.
This is the DRAM-traffic win. Same greedy / coherence commands. Compare
`complete-wall` `dispatches` and `gpu_ns` against the patient
(§6).

### 5.4 What the pack side must emit (out of decoder scope, in test scope)

The decoder does not train G/U/D or FiLM. The swap catalog is a pack
job. Constraints the decoder will enforce at `load_mixed` / admission:

- three shared tensors, codec 3, magic `HGRAVU01`, `group_size=64`,
  bits 3 or 4, shapes as in §2.2, `m % 64 == 0`
- film vectors codec 4, length `m`, present for every SharedOp layer
- no reconstruct-to-Q4, no dense W (`dequant_hgravu_vector` still
  refuses >65536 elements)

Rung A can be assembled with a catalog rewrite that points three new
names at layer L's existing segment offsets (HQ38M20 already supports
absolute segment paths, `resolve_mixed_segment_path` 400–409). No
re-quant.

---

## 6. Active MLP bytes / token and dispatch count

### 6.1 Byte model

HGRAVU01 group-64 body (what Metal streams — `GpuUniform.codes + scales`):

```
groups      = rows * cols / 64
scale_bytes = groups * 2                 # f16
code_bytes  = groups * 64 * bits / 8
body        = scale_bytes + code_bytes
```

Confirmed against L0 gate on mixed-q3mlp-v1: `bits=3`,
`groups=1392640`, `scale_bytes=2785280`, `code_bytes=33423360`,
`body=36208640`. Catalog `nbytes=36208920` includes the 280-byte
HGRAVU01 container (`12 + header_len 268`); that header is **not**
bound in `dispatch_uniform`.

| pack | per matrix body | ×3 (one layer) | ×64 (one token, unique bytes) |
|---|---:|---:|---:|
| q3 patient, m=17408, bits=3 | 36,208,640 | **108,625,920 (108.6 MB)** | **6,952,058,880 (6.952 GB)** |
| shared, m=6144, bits=3 | 12,779,520 | 38,338,560 (38.3 MB) once | **38,338,560 + film** |
| shared, m=8192, bits=3 | 17,039,360 | 51,118,080 (51.1 MB) once | **51,118,080 + film** |
| shared, m=6144, bits=4 | 16,711,680 | 50,135,040 (50.1 MB) once | 50,135,040 + film |

FiLM is f32:

```
per layer = 2 * m * 4
m=6144 → 49,152 B;  ×64 = 3,145,728 B (3.15 MB)
m=8192 → 65,536 B;  ×64 = 4,194,304 B (4.19 MB)
```

**Active MLP bytes / token (unique working set the GPU must be able to
hold hot):**

| path | unique MLP bytes / token | vs patient |
|---|---:|---:|
| current mixed-q3mlp (all 64 native) | 6,952,058,880 | 1.00× |
| shared-op m=6144 q3 + 64 films | 38,338,560 + 3,145,728 = **41,484,288 (39.6 MiB)** | **168× less** |
| shared-op m=8192 q3 + 64 films | 51,118,080 + 4,194,304 = **55,312,384 (52.8 MiB)** | **126× less** |
| one-layer swap m=6144 | 38,338,560 + 49,152 + 63×108,625,920 = **6,881,720,672** | ~1% (correctness only) |

Default target is **m=6144, bits=3**: that is the contract's "~38 MB".
bits=4 is the quality fallback; it still fits SLC better than 6.95 GB
but misses the 38 MB headline.

`Qwen38HybridWeights::resident_bytes` on a full SharedOp pack drops MLP
VRAM from 6.95 GB to 41 MB. That is a capacity win on top of the traffic
win. One-layer-swap packs keep nearly all native VRAM.

### 6.2 Dispatch count

`TokenCommandBuffer::dispatch_threads` increments `compute_dispatches`
once per kernel (`metal/mod.rs:3377 / 3405 / 3422`).
`CommandBufferTiming.dispatches` is what `generate_greedy` / complete-wall
records.

**Per-layer MLP, q3 patient (`encode_dense_mlp_mixed`):**

| # | kernel | count |
|---|---|---|
| 1 | `qwen80_residual_rmsnorm_tg` (or `_f32`) | 1 |
| 2 | geo_tpr64 gate | 1 |
| 3 | geo_tpr64 up | 1 |
| 4 | `gk_swiglu_f32` | 1 |
| 5 | geo_tpr64 down | 1 |
| 6 | `qwen_next_add_residual` | 1 |
| | **per layer** | **6** |
| | **64 layers** | **384** |

HGRAVS down (legacy mixed-2p0) is 2 factor GEMVs; the q3 patient is
Uniform, so down is one geo_tpr64. Isolated GEMV helper
`encode_mlp_matvecs_only_mixed` is 3 dispatches/layer = 192/token
(no rmsnorm / swiglu / residual).

**Per-layer MLP, SharedOp v1:**

| # | kernel | count |
|---|---|---|
| 1 | rmsnorm | 1 |
| 2 | geo_tpr64 G (shared buffers) | 1 |
| 3 | geo_tpr64 U (shared buffers) | 1 |
| 4 | `gk_swiglu_f32` n=m | 1 |
| 5 | `qwen38_mlp_film_f32` | 1 |
| 6 | geo_tpr64 D (shared buffers) | 1 |
| 7 | `qwen_next_add_residual` | 1 |
| | **per layer** | **7** |
| | **64 layers** | **448** |

SharedOp adds **+1 dispatch/layer (+64/token)** for FiLM. That is not
the win. The win is 168× less unique DRAM. A later fused swiglu+film
kernel would restore 384; do not block v1 on it.

**One-layer swap:** 63×6 + 7 = **385** MLP dispatches/token.

`measure_isolated_mlp_matvecs` stays 192 GEMVs either way (3 per layer).
The difference is that 192 SharedOp GEMVs touch 3 buffers, not 192.

Complete-wall `dispatches` also counts mixer + embed + terminal. Those
are unchanged. Report MLP-only numbers from `step_decomposed.mlp` /
`measure_isolated_mlp_full` so a mixer change cannot masquerade as an
MLP regression.

---

## 7. Implementation order (decoder PR, after this blueprint)

1. `MixedMlpNativeKind::SharedOp` + `mixed_mlp_native_kind_from_row` +
   admission rewrite + unit tests against synthetic catalogs
   (`write_tiny_hq38m20` at 4577 is the fixture helper). q3 patient
   admission tests must stay green.
2. `SharedMlpOp` fields + `load_mixed` divert + `resident_bytes`.
3. `qwen38_mlp_film_f32` + compile/trace test arm.
4. `encode_shared_op_mlp` + branches in `encode_dense_mlp_mixed` and
   `encode_mlp_matvecs_only_mixed`.
5. Rung A identity catalog + greedy token match vs mixed-q3mlp-v1.
6. Rung B one-layer reduced-m swap + coherence_prompts.txt.
7. Rung C full-64 pack + bytes/dispatch receipt.

Non-goals remain: no attention / DeltaNet work, no HQ38M20 record
redesign, no fused GEMV+SwiGLU megakernel, no new codec.

---

## 8. Evidence (commands and excerpts used)

### 8.1 Dispatch / kind map (`rg` on the decoder)

```
MixedMlpNativeKind                         208
mixed_mlp_native_kind_from_lane            215
mixed_mlp_role_allowed                     228
assert_mixed_mlp_native_kinds              250
is_mixed_mlp_gemv_name                     296
assert_mixed_mlp_native_catalog            305
QWEN38_HGRAVU01_Q3_GEO_TPR64               556
qwen38_hgravu01_geo_tpr64_launch           564
load_mixed                                 950
assert_mixed_mlp_native                   1330
upload_mixed                              1341
encode_named_matvec                       1541
encode_mixed_matvec                       1559
encode_factor_args                        1634
dispatch_uniform                          1800
encode_mlp_matvecs_only                   2119
encode_dense_mlp                          3040
encode_layers                             3352
encode_dense_mlp_mixed                    3441
encode_mlp_matvecs_only_mixed             3836
step                                      3861
generate_greedy                           3928
```

### 8.2 Kernel symbols

```
crates/hawking-core/shaders/q80_mixed_decode.metal:962
  kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128(
crates/hawking-core/shaders/q80_mixed_decode.metal:1009
  kernel void qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128(
crates/hawking-core/shaders/qwen_uniform_q4.metal:183
  kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(   # DO NOT USE
crates/hawking-core/shaders/gk_family.metal:597
  kernel void gk_swiglu_f32(
crates/hawking-core/shaders/qwen_next.metal:328
  kernel void qwen_next_add_residual(
```

### 8.3 q3 patient catalog (parsed 2026-08-18)

```
exists True size 180124
magic b'HQ38M20\x00'
version 1 n_tensors 851 n_segments 258
mlp_rows 192
  down_proj.weight: n=64 codecs={3} shapes={(5120, 17408)} nbytes={36208920}
  gate_proj.weight: n=64 codecs={3} shapes={(17408, 5120)} nbytes={36208920}
  up_proj.weight:   n=64 codecs={3} shapes={(17408, 5120)} nbytes={36208920}
total_mlp_nbytes 6952112640
L0 gate magic b'HGRAVU01' header_len 268
{"bits":3,"code_bytes":33423360,"elements":89128960,"group_size":64,
 "groups":1392640,"representation":"uniform_q3_group_scale",
 "scale_bytes":2785280,"schema":"hawking.gravity.uniform_group.v1",
 "shape":[17408,5120]}
```

Body streamed per GEMV = 33,423,360 + 2,785,280 = 36,208,640.
×3 = 108,625,920 bytes/layer. ×64 = 6,952,058,880 bytes/token.

### 8.4 Geometry / family

```
crates/hawking-core/src/model/qwen38_geometry.rs:20  QWEN38_LAYERS = 64
crates/hawking-core/src/model/qwen38_geometry.rs:24  QWEN38_HIDDEN = 5_120
crates/hawking-core/src/model/qwen38_geometry.rs:25  QWEN38_INTERMEDIATE = 17_408
crates/hawking-core/src/decode_family.rs:38          SWIGLU_F32 = "gk_swiglu_f32"
crates/hawking-core/src/decode_family.rs:168         pub fn swiglu_f32()
```

---

## 9. Risks the implementer must not "fix" by falling back

- Missing SharedOp film or operator **refuses**. The decoder's law is
  "a missing codec fails the run; there is no reconstruct-to-Q4 path"
  (file header, lines 6–7). SharedOp inherits that.
- `HAWKING_QWEN38_RECON_FUSE=0` makes `qwen38_hgravu01_geo_tpr64_launch`
  return `None` (570) and drops Uniform onto `dispatch_factor`
  (incumbent simd3). SharedOp still works; it just loses the geo_tpr64
  occupancy tile. Do not special-case fuse-off.
- `m` not divisible by 64, or down.cols != gate.rows, refuses at
  admission. Do not pad in the kernel.
- Do not dequant G/U/D to f32 (`dequant_hgravu_vector` is the vector
  path and already refuses >65536 elements). Packed bytes stay packed.
