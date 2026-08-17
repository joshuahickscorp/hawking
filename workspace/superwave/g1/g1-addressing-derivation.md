# G1 addressing derivation

Lane: tensor identity → device address on the 401 GEMV dispatches.
Not buffer packing / catalog layout (sibling). No GPU run this lane.

## Verdict

The 24.15% catalog-vs-single gap (530.65 vs 699.57 GB/s) is access
pattern plus dispatch topology. It is not address derivation.

Host derivation is `format!` + `HashMap<String>` + `div_ceil` +
`set_bytes` of values that never change. It lives in
`host_preparation` (MEASURED 919,250 ns), which the ledger already
carves out of `weight_addressing`.

Device derivation is `row * groups_per_row + col/64`, then two
direct loads. Zero catalog walk. Zero pointer chase. Same integer
path in the catalog probe and the single-address probe.

A per-layer descriptor resolved at load is a runtime-only change.
Predicted effect on the `weight_addressing` bucket: **0 ns**.
It does not close the 24% gap. The catalog probe already had
pre-resolved offsets and still lost 24%.

The "5 ms" figure is the catalog-minus-single COMPONENT delta
applied to the sealed token bucket. Wrong attribution. Sealed
addressing already sits at 639.25 GB/s, not 530.65. Closable
token-level gap to the single-address roof is **1,836,019 ns
PROJECTED**, and that remaining gap is still topology, not
derivation.

---

## 0. Vocabulary

Two different things are named "catalog" in this campaign.

| name | what it is | on the token path? |
|---|---|---|
| on-disk catalog | `Qwen38CatalogRow` / `catalog.hq38m20` name→file or name→(segment,offset) | no — parsed at `load` only |
| honest-roof "catalog addressing" | 401 production-shaped GEMVs in one CB, sequential `set_buffer` offsets into one synthetic slab | the 530.65 GB/s COMPONENT |

This lane traces derivation. The 530.65 number is a topology
microbenchmark that already skipped host name lookup.

---

## 1. End-to-end: tensor identity → first weight byte

### 1.1 Load (once)

G0 vehicle is uniform-Q4, not mixed.

1. `Qwen38HybridWeights::load` reads the pack manifest
   (`qwen38_hybrid_decode.rs:514`). 755 catalog rows
   (`QWEN38_EXPECTED_Q4_TENSORS=402` +
   `QWEN38_EXPECTED_F32_TENSORS=353`, `qwen38_pack.rs:31-35`).
2. Each Q4 row is a **separate file**. Payload is split into
   scales `[scale_offset, sign_offset)` and codes
   `[sign_offset, payload_bytes)` (`qwen38_hybrid_decode.rs:539-558`).
   Two `MTLBuffer`s. Offset 0 forever.
3. Inserted into `HashMap<String, Q4Weight>` keyed by the
   manifest name (`:551-558`).
4. Mixed `catalog.hq38m20` parse (`:96-173`) and
   `read_catalog_payload` seek (`:176-191`) run only if that
   file exists. Resident G0 is the Q4 pack. Either way the
   on-disk catalog is not consulted after `load` returns.

`Q4Weight` (`:421-426`): `{rows, cols, codes, scales}`. No
device-side table. No per-token offset field.

### 1.2 Token (`step`)

```
3292:3310:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            ...
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
```

`encode_layers` (`:2822-2829`): 64× `(mixer + dense_mlp)`.
No ICB. `concurrent_independent` defaults false (`:911`).
One compute encoder per `dispatch_threads` (`metal/mod.rs:4856-4873`).

401 GEMVs (`honest_roof.rs:266-285`, test `:1173-1176`):

| class | n | shape (rows × cols) |
|---|---:|---|
| mlp gate/up | 64+64 | 17408 × 5120 |
| mlp down | 64 | 5120 × 17408 |
| dn qkvz | 48 | 16384 × 5120 |
| dn ba | 48 | **96 × 5120** |
| dn out | 48 | 5120 × 6144 |
| gqa q | 16 | 12288 × 5120 |
| gqa k/v | 16+16 | 1024 × 5120 |
| gqa o | 16 | 5120 × 6144 |
| lm_head | 1 | 248320 × 5120 |

402nd Q4 tensor is the embed table. It is a gather, not a GEMV,
and is not in `weight_addressing`.

### 1.3 One GEMV bind

Every production GEMV goes through this:

```
1569:1590:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        fn encode_q4_matvec_kernel(...) {
            let weight = self.q4(name)?;
            let groups_per_row = weight.cols.div_ceil(UNIFORM_Q4_GROUP_SIZE) as u32;
            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let (grid, tg) = self.matvec_kernel.launch(rows);
            tcb.dispatch_threads(kernel, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(&weight.codes), 0);
                encoder.set_buffer(1, Some(&weight.scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows ...);
                encoder.set_bytes(5, 4, &cols ...);
                encoder.set_bytes(6, 4, &groups_per_row ...);
            })
        }
```

`name` is allocated every call:

```
116:118:crates/hawking-core/src/model/qwen38_geometry.rs
pub fn qwen38_layer_name(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}
```

Call sites on the G0 `step` path (not the `measure_*` path):

- MLP: `post_attention_layernorm`, `gate`, `up`, `down` (`:2562-2588`)
- DeltaNet: `input_layernorm`, `qkvz`, `ba`, plus f32
  `conv1d` / `A_log` / `dt_bias` / `norm`, then `out_proj` (`:2616-2694`)
- GQA: `input_layernorm`, `q/k/v/o_proj`, f32 `q_norm`/`k_norm` (`:2723-2809`)
- terminal: literal `"language_model.lm_head.weight"` (`:2844-2848`)

`q4(name)` is `HashMap::get` (`:1182-1187`). Default hasher
(SipHash). Then `groups_per_row = cols.div_ceil(64)` and
`grid = rows.div_ceil(2) * 128` (`:262-267`) are recomputed
from fields that were fixed at upload.

`set_buffer(..., 0)`: the Metal offset is always zero. Production
does not derive a byte offset on the host. The sibling "catalog
layout" question is whether those 401+401 buffers should have
been one slab. This lane only notes that the bind is already
`(buffer, 0)`.

### 1.4 Device

```
183:211:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(...)
{
    const uint row = group_id * 2u + team;
    const uint rgb0 = row * groups_per_row;
    for (uint col = lane_in_row * 8u; col < cols; col += 512u) {
        const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;
        const uint rgb = rgb0 + group;
        const float scale = float(scales[rgb]);
        const uint packed = *((device const uint*)(codes + rgb * 32u + (local >> 1u)));
        acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
    }
}
```

`addr_probe` (`:227-254`) is the same address + the same two
loads, with the loads sunk so LLVM cannot DCE them. No nibble
unpack, no `input[]`, no FMA. That is the definition of the
`weight_addressing` bucket
(`qwen38_token_ns_ledger.rs:529-535`).

Indirections before the first code byte, per thread:

1. `row = group_id*2 + team` (thread ids)
2. `rgb0 = row * groups_per_row` (one mul, once)
3. `group = col / 64`, `rgb = rgb0 + group`
4. `scales[rgb]` (half)
5. `codes[rgb*32 + (local>>1)]` (uint, 8 weights)

Zero metadata loads. Zero row_ptr. Zero catalog index. The
scale plane is a second contiguous stream, not a pointer table.

Iterations per thread = `cols/512` (DERIVED from the loop):
10 at K=5120, 12 at K=6144, 34 at K=17408 (down_proj).
Across 401 GEMVs that is 3,202,744,320 thread-iterations
(DERIVED; see appendix). Each iteration: 4 integer ops + 2
device loads. Same arithmetic in catalog and single-address
probes — they are the same function.

### 1.5 What is *not* on the G0 token path

- On-disk catalog walk. Load only.
- Mixed CSR (`row_ptr` / `indices`, `encode_csr_args` `:1282-1297`).
  G0 `weights.mixed` is empty; `encode_layers` takes the Q4 branch.
- ICB / `ReplayableComputeGraph`. Substrate exists
  (`metal/mod.rs:3570-3574`: "intentionally not wired into
  decode selection yet"). `qwen38_hybrid_decode.rs` has zero
  references to it. `step` builds a fresh TCB every token.
- `KernelArgBuffer` (`metal/argbuf.rs`). GEMV scalars go out
  as three `set_bytes`, every dispatch.
- 2048-col organ tiling. The kernel was *tuned* on Q80
  512×2048 organs (`qwen38_hybrid_decode.rs:239-240`). G0
  launches the full tensor.

Token-varying GEMV arguments: **none**. Workspace `MTLBuffer`
handles are session-stable. `rows/cols/groups_per_row/grid/tg`
are load-stable. `codes`/`scales` are load-stable. The only
per-token integers on the whole graph are embed `token`, GQA
`position`, and KV `cache_off`. None of those are GEMV weight
addresses.

---

## 2. Host work per GEMV dispatch

Per call, on G0 `step` (code, not a timing):

| step | mutates per token? | evidence |
|---|---|---|
| `format!(language_model.model.layers.{L}.{suffix})` | no | `qwen38_geometry.rs:116-118` |
| `HashMap<String,Q4Weight>::get` (SipHash) | no | `:1182-1187`, `:502` |
| `cols.div_ceil(64)`, `rows.div_ceil(2)*128` | no | `:1578-1581`, `:262-267` |
| `pipeline(fn_name)`: `Mutex` + `HashMap<String>` + `clone` CPS | no (cache hit) | `metal/mod.rs:2480-2508` |
| `new_compute_command_encoder` + `set_label` | ceremony | `metal/mod.rs:4861-4866` |
| `set_compute_pipeline_state` | ceremony | `:4867` |
| 4× `set_buffer(..., 0)` | no (same handles) | `:1583-1586` |
| 3× `set_bytes` of `{rows,cols,gpr}` | no | `:1587-1589` |
| `dispatch_threads` + `end_encoding` | ceremony | `:4869-4873` |

The ledger already put this in a different row:

```
host_preparation  919,250 ns  2.609%  CPU Instant around encode
weight_addressing 21,293,102.5 ns  60.444%  "none in kernel; host bind is in host_preparation"
```

MEASURED, `QWEN38_TOKEN_NS_LEDGER.json`:
`median_encode_ns=919250`, `dispatches.total=964`.
919,250 / 964 = **953.6 ns/dispatch** MEASURED
(`g1-token-anatomy.md:304`). GEMV share if uniform:
401/964 × 919,250 = **382,385 ns ESTIMATED**.

ICB on commit `9c87c500` (not this tree, not G0) cut encode
886,200 → 90,981 ns MEASURED (`QWEN38_FIXED_OVERHEAD_DELETED.json`
`named_fixed_components_ns.encode_host_prepare`). That is the
encoder-create tax, not HashMap/format. Host derivation is a
slice of the leftover ~91 µs plus whatever ICB still writes
(3 u32s/token: token, position, mha_seq_len).

No host-only microbenchmark of `format!`+SipHash vs encoder
create exists for this GEMV. Cheapest: `HAWKING_TCB_TRACE=cpu`
on `step` with a pre-resolved descriptor vs current. This lane
is forbidden to run GPU/organism work; a CPU-only unit bench
of 401 `format!`+`HashMap::get` would bound it. Not run.

---

## 3. The 24% gap is not derivation

### 3.1 The three COMPONENT points

All three: same kernel `*_addr_probe`, same unique synthetic
Q4 bytes, GPU timestamps after `wait`,
`GPU_PROTECTED_CPU_CONTENDED`, date 2026-08-17.
Source: `HEAD:receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json`.
Not re-run this lane.

| topology | dispatches | median_ns | GB/s | vs single |
|---|---:|---:|---:|---:|
| `single_gemv` 5004288×5120 | 1 | 19,457,084 | 699.5736545106142 | — |
| `tiled_production_organ` 287× (17408×5120) | 287 | 22,988,750 | 591.1317979446468 | −15.50% |
| `production_shape_catalog` 401 mixed | 401 | 25,650,709 | 530.6544688491846 | −24.15% |

ALU tax (full vs addr) at 13.6 GB:

| topology | addr GB/s | full GB/s | tax |
|---|---:|---:|---:|
| single | 699.57 | 666.68 | 4.70% |
| catalog | 530.65 | 505.81 | 4.68% |

Same reconstruction tax. If device address *math* were the
catalog penalty, decode/full would blow out on the catalog
only. They do not.

### 3.2 How to tell derivation from access

| test | result | implies |
|---|---|---|
| Clock | catalog vs single is `GPUStartTime/GPUEndTime` (`honest_roof.rs:444-450`, `:623-624`) | host `format!`/HashMap/`div_ceil` cannot be in the 24% |
| Host state of the catalog probe | `Vec<GemvShape>` + running `c_off`/`s_off` (`:583-626`). No names. No HashMap. | host derivation already removed; 24% remains |
| Device state of both probes | same `*_addr_probe` function | device derivation identical |
| ALU tax | 4.70% single, 4.68% catalog (DERIVED from receipt) | gap is not extra integer math |
| Same-shape tile | 287 identical 17408×5120 organs at 591.13 GB/s | 15.50 of the 24.15 points is "many launches of one shape vs one launch" |
| Mixed catalog vs tile | 530.65 vs 591.13 = −10.23% | remaining points are shape mix (ba=96 rows / 48 TGs, k/v=1024, down K=17408) + 114 extra dispatches |

Receipt says this in one sentence
(`adjudication.catalog_topology_tax`):

> 401 production-shaped GEMVs in one CB achieve 530.7 GB/s
> addr / 505.8 GB/s full — 24% below the single-GEMV addr
> roof. Isolated class CBs (the ledger) sit between the two,
> at 639.

`execution_headroom.dispatch_topology`:

> 401 mixed organs leave ~24% vs one GEMV. Tiny ba (96x5120)
> and encoder boundaries are genome, not ALU.

### 3.3 The "5 ms" number

| quantity | ns | kind |
|---|---:|---|
| catalog − single | 25,650,709 − 19,457,084 = **6,193,625** | COMPONENT MEASURED |
| 0.24146 × sealed 21,293,102.5 | **5,131,638** | PROJECTED, mis-applied |
| sealed − single | 21,293,102.5 − 19,457,084 = **1,836,019** | PROJECTED token-level closable if addressing reached 699.57 |
| catalog − sealed | **4,357,607** | catalog is *worse* than the seated token split |

Sealed 639.25 GB/s is isolated-class GEMV GPU × addr_frac,
then scaled onto production GPU
(`qwen38_token_ns_ledger.rs:389-394`, `:529-535`). Isolated
classes already use production `(buffer, 0)` binds and still
beat the synthetic 401-offset catalog (639 vs 531). Token
addressing is not sitting at 530.65.

Per extra dispatch (DERIVED, COMPONENT, not a law):

- catalog: (25,650,709 − 19,457,084) / 400 = **15,484 ns**
- tiled:   (22,988,750 − 19,457,084) / 286 = **12,348 ns**

Not constant. Shape mix costs extra. Do not treat 15 µs as a
token-level dispatch tax without a production-CB counter
sample.

CommandBatch used by the catalog timer has
`ordered_encoder_enabled=false` (`honest_roof.rs` via
`metal/mod.rs:3004-3011`), so it also pays 401 encoder
boundaries. Production `step` does too. That is topology,
sibling-adjacent, not derivation.

---

## 4. The change that pre-resolves addressing

Runtime only. No artifact format change. Applies to G0 today
and to every future candidate that still binds through
`encode_q4_matvec_kernel`.

### 4.1 Shape

Built once at the end of `Qwen38HybridWeights::load`:

```
struct ResolvedGemv {
    codes: PinnedBuffer,          // already owned
    scales: PinnedBuffer,
    rows: u32,
    cols: u32,
    groups_per_row: u32,          // cols.div_ceil(64)
    grid: (u32, u32, u32),        // rows.div_ceil(2)*128
    tg: (u32, u32, u32),          // (128,1,1)
}

struct LayerBinds {
    input_norm: PinnedBuffer,
    post_norm: PinnedBuffer,
    mixer: MixerBinds,            // DN {qkvz, ba, out, conv, a_log, dt_bias, norm}
                                  // or GQA {q, k, v, o, q_norm, k_norm}
    gate: ResolvedGemv,
    up: ResolvedGemv,
    down: ResolvedGemv,
}

struct ResolvedGraph {
    layers: [LayerBinds; 64],
    lm_head: ResolvedGemv,
    embed: /* existing Q4Weight is enough */,
    final_norm: PinnedBuffer,
}
```

Replace `qwen38_layer_name` + `self.q4(name)` on the `step`
path with `&self.graph.layers[layer].gate` etc.

```
fn encode_resolved_gemv(
    tcb: &mut TokenCommandBuffer<'_>,
    w: &ResolvedGemv,
    input: &PinnedBuffer,
    output: &PinnedBuffer,
) -> Result<()> {
    tcb.dispatch_threads(QWEN38_Q4_MATVEC_KERNEL, w.grid, w.tg, |enc| {
        enc.set_buffer(0, Some(&w.codes), 0);
        enc.set_buffer(1, Some(&w.scales), 0);
        enc.set_buffer(2, Some(input), 0);
        enc.set_buffer(3, Some(output), 0);
        enc.set_bytes(4, 4, &w.rows as *const u32 as *const _);
        enc.set_bytes(5, 4, &w.cols as *const u32 as *const _);
        enc.set_bytes(6, 4, &w.groups_per_row as *const u32 as *const _);
    })
}
```

Optional next tightening, still runtime: intern `{rows,cols,gpr}`
in a `KernelArgBuffer` per GEMV (one `set_buffer` instead of
three `set_bytes`; `metal/argbuf.rs:1-18` already describes
this). Still 0 ns on the GPU addressing bucket.

Optional next tightening, still runtime: feed the 401
`ResolvedGemv` records to `ReplayableComputeGraph`. GEMV
stage arguments are a closed set — ICB can capture them
entirely (`per_token_host_writes` on the later genome did
not include any GEMV field). That deletes encoder create
for those 401. It does **not** turn 401 launches into 1.

Keep the HashMap for diagnostics / `measure_*`. `step` must
not touch it.

### 4.2 Predicted effect

| bucket | predicted Δ | kind | why |
|---|---|---|---|
| `weight_addressing` 21,293,103 ns | **0 ns** | predicted | GPU addr_probe of codes+scales. Descriptor does not change the kernel, the bytes, or the launch count. Catalog probe already had resolved offsets and still ran at 530.65. |
| `host_preparation` 919,250 ns | tens of µs, not hundreds | ESTIMATED | ICB evidence: encoder create is the 953 ns/dispatch, not SipHash. 401 `format!`+get is a slice of 382,385 ns ESTIMATED GEMV-encode share. |
| catalog 24% / "5 ms" | **0 ns** | predicted | see §3 |
| sealed → single-addr roof | **0 ns from this change** | predicted | remaining 1,836,019 ns PROJECTED is 401-launch vs 1-launch access |

Does a per-layer descriptor remove per-dispatch work
*entirely*? It removes derivation (name, hash, div, static
scalars). It does not remove encoder create / set_pipeline /
dispatch / end_encoding. Those are bind ceremony. ICB
removes the ceremony (MEASURED 886,200 → 90,981 ns encode
on a later genome). Neither removes the 401 DRAM streams.

### 4.3 What would move the bucket

Not this lane's change. Sibling / genome:

- fewer larger GEMVs (gate+up already share an input;
  `encode_independent_q4_pair` is still two dispatches)
- one-address / persistent stream (the 699.57 COMPONENT)
- byte reduction (mandatory; 13.6 GB / 699.57e9 = 19.46 ms
  load-only, complete token still has 13.93 ms remainder)

ICB GPU effect on addressing: UNKNOWN. Later-genome complete
token GPU 36.987 → 36.012 ms DIRTY (−975 µs). Cannot assign
that to GEMVs. Cheapest experiment: ICB-encode the 401-shape
`time_q4_catalog` and compare GPU timestamps to the current
401-encoder catalog. Serialized GPU lane. Not this lane.

---

## 5. KILLS / REOPEN_IF

**KILL** "pre-resolve addresses to close the 24% catalog gap."
The catalog probe is already pre-resolved and is the 24%.
REOPEN_IF a locked GPU run shows `time_q4_catalog` with
HashMap names vs with `GemvShape` offsets differing by more
than timestamp noise. Predicted difference: 0 on GPU.

**KILL** "device threads pointer-chase a catalog / descriptor
table on the G0 token path."
`geo_tpr64_tg128` does `codes[rgb*32 + local/2]`.
REOPEN_IF that kernel grows a device-side organ table walk.

**KILL** "host recompute of rows/cols/gpr is inside
`weight_addressing`."
Ledger: `cpu_involvement = "none in kernel; host bind is in
host_preparation"` (`qwen38_token_ns_ledger.rs:529`).
REOPEN_IF the component definition changes.

**KILL** "G0 walks `catalog.hq38m20` or seeks segments per
token."
`step` never opens a file. `parse_qwen38_mixed_catalog` is
`load_mixed` only.
REOPEN_IF `step` grows a catalog read.

**Not killed.** ICB as a ceremony cut. MEASURED on
`9c87c500`, not seated as G0, not derivation.

**Not killed.** Mixed Residual CSR pointer chase. Not on G0.
REOPEN_IF mixed-sub15 becomes the generate path: then
`row_ptr[row]` is a real device indirection and needs its
own probe.

---

## 6. Evidence

### 6.1 Receipt extracts (git show, not re-run)

```
# HONEST_ROOF_WEIGHT_ADDRESSING.json — catalog / single / tiled
catalog addr  topology=production_shape_catalog dispatches=401
  payload_bytes=13611663360 median_ns=25650709 median_gb_s=530.6544688491846
  all_ns=[26727750, 25415834, 25252084, 25670959, 25650709]
single addr   topology=single_gemv dispatches=1 rows=5004288 cols=5120
  payload_bytes=13611663360 median_ns=19457084 median_gb_s=699.5736545106142
  all_ns=[19637375, 19536625, 19347166, 19345541, 19457084]
tiled 13p612gb_tiled_gate_addr topology=tiled_production_organ
  organs=287 organ_rows=17408 organ_cols=5120
  payload_bytes=13589381120 median_ns=22988750 median_gb_s=591.1317979446468
verdict.catalog_topology_tax:
  "401 production-shaped GEMVs in one CB achieve 530.7 GB/s addr
   / 505.8 GB/s full — 24% below the single-GEMV addr roof.
   Isolated class CBs (the ledger) sit between the two, at 639."
timing_label=GPU_PROTECTED_CPU_CONTENDED
gpu_timestamp_authority=completed MTLCommandBuffer GPUStartTime/GPUEndTime
```

```
# QWEN38_TOKEN_NS_LEDGER.json
median_gpu_ns=33912333 median_encode_ns=919250
median_submit_ns=12084 median_wait_ns=34296583 median_wall_ns=35227917
dispatches.total=964 production_command_buffers=1
weight_addressing ns=21293102.524500456 disp=401 pct=60.443830739411744
  cpu="none in kernel; host bind is in host_preparation"
host_preparation ns=919250.0 disp=964 pct=2.609436146905876
probes.mlp  addr_frac=0.8716916907999576
probes.dn   addr_frac=0.9050872853807689
probes.gqa  addr_frac=0.8302650920366828
probes.lm_head addr_frac=0.9157069126980056
```

```
# QWEN38_FIXED_OVERHEAD_DELETED.json (other commit 9c87c500, DIRTY)
encode_host_prepare before_mean=886200 after_mean=90981 delta=-795219
icb_commands=964 per_token_host_writes=[token, position, mha_seq_len]
complete_token_wall before_gpu_ms=36.987458 after_gpu_ms=36.01225
```

### 6.2 Arithmetic (DERIVED from those scalars)

```
single_gb              699.5736545106142
catalog_gb             530.6544688491846
sealed_gb              639.2522348492898
tiled_gb               591.1317979446468
catalog_vs_single      0.24146018731879892
catalog_minus_single   6193625 ns
sealed_minus_single    1836018.5 ns
24pct_of_sealed        5131637.7 ns   # mis-applied projection
per_disp_catalog       15484.06 ns
per_disp_tiled         12348.48 ns
host encode GEMV share 382385 ns      # 401/964 * 919250, ESTIMATED
```

### 6.3 Source pointers

- Load / HashMap / Q4 split: `qwen38_hybrid_decode.rs:498-580`
- Mixed catalog parse (load only): `:96-191`
- `step` / `encode_layers`: `:3292-3310`, `:2822-2829`
- GEMV bind: `:1569-1590`
- Name format: `qwen38_geometry.rs:116-118`
- Launch: `qwen38_hybrid_decode.rs:261-267`
- Default flags: `:911` concurrent off
- TCB encoder-per-dispatch: `metal/mod.rs:4801-4873`
- Pipeline cache: `metal/mod.rs:2478-2508`
- ICB unwired: `metal/mod.rs:3572-3574`
- Kernel + addr_probe: `qwen_uniform_q4.metal:183-266`
- Catalog vs single timers: `honest_roof.rs:553-626`, `:805-829`
- Production shapes: `honest_roof.rs:266-286`
- Bucket definition: `qwen38_token_ns_ledger.rs:510-535`
- 402 Q4 + 353 f32: `qwen38_pack.rs:31-35`
- Argbuf unused by GEMV: `metal/argbuf.rs:1-18`

---

## 7. Completion report

```
STATUS
CLAIMS
EVIDENCE
CHANGES
TESTS
RISKS
UNRESOLVED
NEXT
```

STATUS: SUPPORTED

CLAIMS

1. On-disk catalog is load-only. Token path never parses it or
   seeks a segment. Evidence: `qwen38_hybrid_decode.rs:96-191`
   (parse), `:508-580` (load), `:3292-3302` (step has no catalog
   call).
2. Host derivation per GEMV is `format!` + `HashMap<String>` +
   `div_ceil` + `set_bytes` of load-stable fields. Evidence:
   `:116-118`, `:1182-1187`, `:1569-1590`.
3. Device derivation is `row*gpr + col/64`, then `scales[rgb]`
   and `codes[rgb*32 + local/2]`. Zero pointer chase. Evidence:
   `qwen_uniform_q4.metal:183-211` and addr_probe `:227-254`.
4. No GEMV bind argument varies per token. Evidence: workspace
   allocated once (`attach` `:890`); bind list `:1583-1589`;
   token/position only on embed/GQA (`:2541`, `:2766-2771`).
5. 24.15% catalog vs single is topology/access, not derivation.
   Evidence: both probes are GPU timestamps of the same kernel;
   catalog probe has no HashMap (`honest_roof.rs:583-626`);
   ALU tax 4.70% vs 4.68%; tiled same-shape takes 15.50 of the
   24.15 points (HONEST_ROOF json, appendix §6).
6. Seated token addressing is 639.25 GB/s, not 530.65. Closable
   gap to single-addr is 1,836,019 ns PROJECTED, not ~5 ms.
   Evidence: `SEALED_WEIGHT_ADDRESSING_NS` / `GEMV_PAYLOAD_BYTES`
   (`honest_roof.rs:44-50`); receipt `sealed_weight_addressing_gb_s`.
7. Per-layer `ResolvedGemv` is runtime-only and predicts **0 ns**
   on `weight_addressing`. Evidence: claim 5; ledger
   `cpu_involvement` on that row (`qwen38_token_ns_ledger.rs:529`).
8. ICB exists and is not on G0 `step`. Evidence:
   `metal/mod.rs:3572-3574`; `rg` of `qwen38_hybrid_decode.rs`
   for `ReplayableComputeGraph` / `execute_replayable` is empty.

EVIDENCE

- `git show HEAD:receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json`
  parsed in §6.1 (catalog/single/tiled/verdict).
- `git show HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`
  parsed in §6.1 (encode, probes, components).
- `git show HEAD:receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json`
  parsed in §6.1 (ICB encode cut, not G0).
- Source excerpts cited by path:line in §§1–4.
- DERIVED arithmetic in §6.2 from those scalars.

CHANGES

- Created `workspace/superwave/g1/g1-addressing-derivation.md`.
- No other path touched. No GPU. No organism. No format change.

TESTS

- `test -s workspace/superwave/g1/g1-addressing-derivation.md`
- `wc -l workspace/superwave/g1/g1-addressing-derivation.md`
- `git status --porcelain`
  (exact outputs pasted in the session completion report)

RISKS

- HONEST_ROOF is `GPU_PROTECTED_CPU_CONTENDED`. Absolute 699.57
  is provisional. The *ratio* catalog/single/tiled is the
  discriminator and all three share that contamination.
- Isolated-class 639.25 is a TOKEN split, not a 401-in-one-CB
  production addr_probe. Do not call 639.25 a catalog-topology
  measurement.
- ICB encode cut is a different commit and a DIRTY wall. Do
  not subtract 795 µs from G024 `host_preparation`.

UNRESOLVED

- Host nanoseconds of `format!`+SipHash vs encoder create.
  Cheapest: CPU-only 401-lookup bench, or `HAWKING_TCB_TRACE=cpu`
  A/B after the descriptor lands. Not run.
- ICB GPU timestamps on the 401-shape catalog. Cheapest:
  ICB-wrap `time_q4_catalog`. Serialized GPU lane.
- Whether production 401-encoder boundaries inside the
  isolated class CBs are a visible slice of the 21.29 ms.
  Cheapest: `ProdCbGpu` per-encoder samples on isolated mlp/dn/gqa.

NEXT

- Implement `ResolvedGraph` at load (runtime, ~80–150 lines in
  `qwen38_hybrid_decode.rs`). Expect ceremony hygiene, not a
  `weight_addressing` win.
- Do not spend a GPU lane on "does pre-resolve close 24%".
  Already answered by the existing catalog probe.
- Topology work (sibling): fewer launches, one-address stream,
  ba=96 occupancy. That is the 1.84 ms PROJECTED token-level
  remainder vs 699.57, and the 6.19 ms COMPONENT catalog gap.
