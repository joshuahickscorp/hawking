# G1 byte deletions — Qwen3.8 G0

Lane: `37-byte-deletions`. Write scope: this file only. No GPU, no inference, no
artifact mutation, no live-Genesis touch.

Every number is **MEASURED** (command or receipt field), **DERIVED** (geometry or
closed arithmetic on measured inputs), **PROJECTED** (measured regime × stated
assumption), or **ESTIMATED** (order-of-magnitude, not a token claim).

G0 vehicle: `uniform-q4-v1` at
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`.
Complete physical BPW **MEASURED** `4.252735126866492`. Live TOKEN_NS **CITED**
`39,326,090`. Sealed addressing **CITED** `21,293,102.5` ns at `639.2522348492898`
GB/s (`HONEST_ROOF_WEIGHT_ADDRESSING.json` `sealed_ledger_cited_not_rerun` /
`verdict.sealed_weight_addressing_gb_s`).

---

## 0. Verdict

All three wave-1 wastes are real. Only one of them is per-token DRAM.

| finding | bytes | when | TOKEN_NS | class |
|---|---:|---|---|---|
| Q4 group-64 f16 scale plane | **800,686,080** | every token, 401 GEMVs | **PROJECTED** 1,252,535 ns @ sealed 639.25 GB/s | TRAFFIC |
| `fs::read` + `newBufferWithBytes` | **28,595,370,456** host (two walks of the catalog) | load only | **0** once resident | LOAD |
| 353 orphan `*.f32bin` | **10,584,840** | disk only | **0** | DISK |

Deleting the scale plane from the current HQ30UQ4 kernel **KILLS** capability
(0 singleton tensors; per-row unique never 1; scales span `9.54e-7 … 0.213`).
The 1.25 ms is recovered only by a successor codec that does not stream
per-group f16. **REOPEN_IF** that codec is generate-proven on this body.

Hash-per-token and clone-tree-on-open (the DSV4F identity pair) are **absent**
on the Qwen38 `step` path. Index bloat is **present** as a 755-file catalog +
1,157 weight `MTLBuffer`s + `HashMap<String>` lookup, and is **not** the TOKEN_NS
wall. The wall is 13.611_663_360 B unique-once DRAM.

---

## 1. Scale plane — 800,686,080 B — TRAFFIC

### 1.1 It exists

HQ30UQ4 is 32 code bytes + 2 f16 scale bytes per group of 64 weights.

```3:6:crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs
//! Layout matches `qwen_uniform_q4.metal`:
//!   * magic `HQ30UQ4\\0`, version 1, group_size 64
//!   * FP16 scale per group of 64 flat elements
//!   * 32 code bytes per group (even nibble low, odd high; q = nibble - 8)
```

```112:125:crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs
    let groups = elements
        .checked_add(group_size - 1)
        ...
        / group_size;
    let scale_bytes = groups
        .checked_mul(2)
        ...
    let code_bytes = groups
        .checked_mul(UNIFORM_Q4_CODE_BYTES_PER_GROUP)
```

`UNIFORM_Q4_CODE_BYTES_PER_GROUP = 32`. Kernel consumes both planes every GEMV:

```208:210:crates/hawking-core/shaders/qwen_uniform_q4.metal
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
```

### 1.2 Byte count — MEASURED from manifest shapes + file headers

Python over `manifest.json` + `lstat` + 6-file header seek (not a 14 GB slurp):

```
q4 402  f32 353
not_div64 0
ALL q4 groups 420208640 codes 13446676480 scales 840417280 headers 16080
GEMV (no embed) groups 400343040 codes 12810977280 scales 800686080 sum 13611663360
EMBED groups 19865600 codes 635699200 scales 39731200 sum 675430400
2/34 of gemv_sum 800686080
q4_formula_mismatch 0
```

Header spot-check (first 64 bytes of each file). Layer-0 gate:

```
magic=HQ30UQ4\0 ver=1 gsz=64 rank=2 elems=89128960 dims=[17408,5120]
after=40 groups=1392640 scale_bytes=2785280 code_bytes=44564480
expect=47349800 fsz=47349800
```

Same closed form on embed, lm_head, down, q_proj, out_proj. `scale_offset = 40`
for every rank-2 matrix.

Defended GEMV payload is this sum:

```44:46:crates/hawking-core/src/backend/honest_roof.rs
/// Bytes the Q4 GEMV kernels actually stream per token (codes + f16 scales).
/// This is the defended denominator for `weight_addressing`.
pub const GEMV_PAYLOAD_BYTES: u64 = 13_611_663_360;
```

Receipt `HONEST_ROOF_WEIGHT_ADDRESSING.json` `byte_count_adjudication.defended_bytes=13611663360`.
Matches. Embed table is **not** in that number. Embed gather is 80 groups × 34 B
= **2,720** B/token, of which **160** B are scales (**DERIVED**).

Resident scale plane including the unused embed-table scales: **840,417,280** B
(**DERIVED**). Per-token DRAM is the GEMV slice **800,686,080** plus the 160 B
embed-row slice.

### 1.3 It is per-token DRAM, not residency

Weights are already `MTLBuffer`s. `step` rebinds codes + scales every GEMV:

```1582:1589:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
            tcb.dispatch_threads(kernel, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(&weight.codes), 0);
                encoder.set_buffer(1, Some(&weight.scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &cols as *const u32 as *const _);
                encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
```

addr_probe is written to stream exactly those two planes
(`qwen_uniform_q4.metal:223-226`). Sealed `weight_addressing.bytes_read =
13,611,663,360` (`TOKEN_NS_QWEN38.json`, cited in `g1-traffic-anatomy.md` §4.2).
13.6 GB does not fit in SLC. Unique-once DRAM.

**PROJECTED** addressing time of the scale plane at the sealed regime:

```
800,686,080 / 639.2522348492898e9 s = 1,252,535 ns
= 21,293,102.5 × (2/34)
```

That is **3.185%** of live TOKEN_NS `39,326,090` if and only if the rest of the
token is unchanged. It is **not** a measured token delta. A component
microbenchmark that deleted the scale load without a matching codec would DCE
or NaN, not give this number.

### 1.4 It is not independent weight information

The scale is decode metadata: `q = (nibble-8) * scale`. Codes without scales
are uncalibrated integers in `[-8,7]`.

Uniqueness walk of **every** Q4 scale plane (800.7 MB GEMV + 39.7 MB embed,
seek-read scales only):

```
q4_tensors_scanned 402
global_unique_f16_bit_patterns 1903
global_min 9.5367431640625e-07  global_max 0.213134765625
zero_groups 0  nan_groups 0  inf_groups 0
singleton_tensors 0  le16 0  le256 0  le1024 400
uniq_hist {'257-1024': 400, '1025-4096': 2}
```

Per class, mean unique scales per tensor 405–1244. Lowest non-embed unique:
315 (`layers.23.mlp.up_proj`). Highest: 1416 (`layers.16.linear_attn.in_proj_qkvz`).
Embed: 1244 unique. Mode fraction ~1.1–1.5%. No zero groups.

Per-row unique on five tensors (full tensor except lm_head first 4096 rows):

| tensor | gpr | tensor unique | row unique min/mean/max | rows with unique≤4 |
|---|---:|---:|---|---:|
| L0 out_proj | 96 | 588 | 52 / 65.0 / 91 | 0 |
| L0 gate | 80 | 480 | 41 / 57.9 / 76 | 0 |
| L0 down | 272 | 607 | 98 / 113.1 / 178 | 0 |
| L3 q_proj | 80 | 467 | 44 / 57.1 / 73 | 0 |
| lm_head (4096/248320) | 80 | 731 | 43 / 55.3 / 75 | 0 |

**KILLS** (current kernel, keep Q4 nibbles, drop the scale buffer or replace
with 1.0): every GEMV output is multiplied by the wrong factor, range 10⁻⁶ to
0.21. Tokens will not match.

**KILLS** (one f16 per tensor, 802 B): 0/401 GEMVs are singleton.

**KILLS** (one f16 per row, 8,304,640 B, **DERIVED** `gemv_rows=4,152,320 × 2`):
0 scanned rows have unique 1 or even ≤4.

**Lossless 11-bit codebook is possible.** 1903 < 2048 global f16 values.
Index plane `400,343,040 × 11/8 = 550,471,680` B + codebook `1,903 × 2 = 3,806`
B. Save **250,210,594** B. **PROJECTED** `250,210,594 / 639.2522348492898e9 =
391,412` ns. New pack + new kernel. Quality holds only if the bit-pack is
exact. ALU tax **UNMEASURED**. `QWEN38_RECONSTRUCTION_IS_FREE` licenses
“nibble unpack is free vs f32”, not “11-bit gather is free”. A u16 index is
the same 2 B/group as storing the f16 — **zero** saving.

### 1.5 What breaks if it is removed

- Production kernel `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` reads
  `buffer(1)` as `device const half* scales`. Missing bind is a Metal error.
- `parse_uniform_q4_header` rejects payload size ≠ `40 + 2G + 32G` and rejects
  non-finite scales (`uniform_q4.rs:129-141`).
- `Q4Weight` is two buffers (`qwen38_hybrid_decode.rs:421-426`). Loader slices
  `[scale_offset:sign_offset]` and `[sign_offset:payload_bytes]`.
- Oracle-32 / greedy identity vs G0: **KILLS**.
- Complete BPW would become `8 × (14,297,694,680 − 800,686,080) / 26,895,998,464
  = 4.0147` **DERIVED** if only GEMV scales left the catalog and embed-table
  scales stayed. Not a quality statement.

**REOPEN_IF:** a generate-proven G1 codec whose production kernel does not
stream a per-group f16 (global scale, implicit scale, or a representation
that is not Q4-absmax). Binding: reject expand-to-float/Q4 then generic GEMV.

---

## 2. Load-time double copy — 14.3 GB × 2 — LOAD only

### 2.1 It exists

```534:557:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
                let payload = fs::read(&path).map_err(|error| {
                    Error::Model(format!("cannot read {}: {error}", path.display()))
                })?;
                match row.kind.as_str() {
                    "q4" => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        ...
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
```

```2610:2628:crates/hawking-core/src/metal/mod.rs
        pub fn new_buffer_with_bytes_checked(&self, bytes: &[u8]) -> Result<Buffer> {
            ...
            Ok(self.inner.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        }
```

`newBufferWithBytes` / `new_buffer_with_data` **copies**. The `Vec<u8>` from
`fs::read` is then dropped. After load, one Shared copy remains.

f32 path is a **triple** copy: `fs::read` → `read_qwen38_f32_payload` builds a
`Vec<f32>` element-wise (`qwen38_pack.rs:751-767`) → `new_buffer_with_bytes_checked`
of the cast slice. 10,582,016 B resident. Load only.

### 2.2 Byte count — MEASURED / DERIVED

```
listed_disk_bytes     14297694680   # fs::read total, MEASURED lstat == manifest
headers               18904         # 402×40 + 353×8, DERIVED, matches census
resident_weight_bytes 14297675776   # GENESIS_RESIDENT_BODY.json resident_load
host_traffic_at_load  28595370456   # 14297694680 + 14297675776, DERIVED
```

`GENESIS_RESIDENT_BODY.json` `resident_load`:

```
load_ns                              4908153958
ready_wall_ns                        5577653625
load_count                           1
resident_weight_bytes                14297675776
workspace_bytes                      175361796
rss_bytes                            15511666688
phys_footprint_bytes                 15385816488
ioaccelerator_graphics_dirty_bytes   14473019392
```

Peak extra during load is **not** 2×14.3 GB: tensors are processed one-by-one, so
transient `Vec` ≈ largest file (embed/lm_head `675,430,440`) plus the growing
Metal set. Matches RSS 15.51 GB after load.

`28,595,370,456 / 4,908,153,958 ns = 5.82 GB/s` **DERIVED**. That is not a DRAM
memcpy roof (this box’s GEMV addr_probe is 530–700 GB/s). The 4.91 s warm load
is **not** explained by the second copy. Candidates, **ESTIMATED** split, not
timed this lane:

- 755 sequential `open`+`read` of a sha256-named catalog (index bloat at load)
- `parse_uniform_q4_header` walks **all 420,208,640** scales and checks
  `is_finite()` (`uniform_q4.rs:135-141`) — 840,417,280 B CPU touch
- 1,157 `MTLBuffer` allocations
- the actual `newBufferWithBytes` memcpy of 14.3 GB (tens of ms at ≥200 GB/s
  host memcpy, **ESTIMATED**)

Box-cold load is a different number: receipt `established_box_cold_load_s=50`,
explicitly “not re-measured; previous lock holder had just uploaded”.

### 2.3 Not TOKEN_NS

After `load_count=1`, `step` does not `fs::read` and does not
`new_buffer_with_bytes` for weights. Confirmed: no `Sha256` / `hash_invocations`
in `qwen38_hybrid_decode.rs`. Weights stay `Arc<Qwen38HybridWeights>`.

Zero-copy API exists and is used on Q80 / DSV4F / GGUF, **not** on Qwen38:

```2677:2700:crates/hawking-core/src/metal/mod.rs
        pub const NO_COPY_PAGE_ALIGN: usize = if cfg!(target_arch = "aarch64") {
            16 * 1024
        } else {
            4 * 1024
        };
        pub fn new_buffer_no_copy_checked(&self, bytes: &[u8], label: &str) -> Result<Buffer> {
            const ALIGN: usize = MetalContext::NO_COPY_PAGE_ALIGN;
            if bytes.is_empty()
                || (bytes.as_ptr() as usize) % ALIGN != 0
                || bytes.len() % ALIGN != 0
```

Qwen38 callers of `new_buffer_no_copy*`: **none**. Callers elsewhere:
`qwen80_device_expert_table.rs`, `gravity_deepseek_v4_streamed_native.rs`,
`llama.rs` / `qwen_dense.rs` / `deepseek_v2.rs` / `mixtral.rs` (GGUF mmap).

### 2.4 Current files cannot no-copy

Alignment census of 402 Q4 files, 16 KiB page:

```
scale_off_16k  0/402     # all start at byte 40
scale_len_16k  354/402
code_off_16k   0/402     # 40 + scale_bytes
code_len_16k   402/402
pad_if_header_page_plus_len_pad  6619440
```

**DERIVED** pad to put header in its own 16 KiB page and pad each plane’s
length to 16 KiB: **6,619,440** B (~0.046% of the catalog). Then mmap the
file, bind scales and codes with `new_buffer_no_copy_checked`, keep the mmap
alive for process lifetime.

### 2.5 What breaks if the copy is removed

- Without a kept mmap (or a page-aligned pack), `new_buffer_no_copy` is
  unsound: the `Vec` dies at the end of the loop.
- Unaligned window: `new_buffer_no_copy_checked` fail-closes (no silent copy).
- `new_buffer_from_verified_bytes` **falls back** to a copy — do not use it
  if the point is to refuse a silent pack.
- Capability: preserved by construction if the bound bytes are the same
  codes/scales. Needs one greedy-id check after land.
- TOKEN_NS: **0**. This buys load / child-spawn / “table already fits”
  headroom, not TPS.

**REOPEN_IF:** a GPU counter shows the kernel over-fetches because codes and
scales live in two address spaces (TLB). Cheapest experiment: locked
addr_probe, one-buffer interleaved layout vs two-buffer. Forbidden this lane.

---

## 3. Orphan `*.f32bin` — 10,584,840 B — DISK only

### 3.1 It exists

```
tensors/ files          1108
by_ext                  .hq30uq4=402  .f32v2=353  .f32bin=353
listed                  755
listed_disk_bytes       14297694680  missing=0  size_mismatch=0
extras                  353  {'.f32bin': 353}  extra_bytes=10584840
listed_f32bin           0
listed_f32v2            353
```

Current packer:

```30:30:crates/hawking-core/src/model/qwen38_pack.rs
pub const QWEN38_F32_EXT: &str = "f32v2";
```

```270:274:crates/hawking-core/src/model/qwen38_pack.rs
fn artifact_filename(name: &str, ext: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(name.as_bytes());
    format!("{:x}.{ext}", digest)
}
```

Same stem, two extensions. `Qwen38HybridWeights::load` joins `tensors_dir` +
`row.artifact`. Manifest artifacts are `*.hq30uq4` / `*.f32v2` only. Load
never opens `*.f32bin`. `rg f32bin` in `*.rs` / `*.py` at HEAD: **no runtime
hit**, only wave-1 reports.

### 3.2 192 byte-identical, 161 pre-delta — MEASURED

Full pairwise `read_bytes` of all 353 pairs (21.2 MB, under the 20 GB cap):

```
identical 192  ident_bytes 7908864
different 161  diff_bytes_bin 2675976  diff_bytes_v2 2675976
size_eq 353  same_header 161  delta_minus_one_first 161
diff_all_plus_one 161  diff_not_plus_one 0  max_abs_err_from_1 0.0
class_ident {'dt_bias': 48, 'A_log': 48, 'linear_attn.norm': 48, 'conv1d': 48}
class_diff  {'input_layernorm': 64, 'post_attention_layernorm': 64,
             'final_norm': 1, 'q_norm': 16, 'k_norm': 16}
size_ident  {200: 96, 520: 48, 163848: 48}
size_diff   {20488: 129, 1032: 32}
```

161/161 different pairs: **every** f32 is exactly `bin = v2 + 1.0`
(`max_abs_err_from_1 = 0.0`). That is `mlx_residual_norm_to_delta`:

```427:439:crates/hawking-core/src/model/qwen38_pack.rs
/// MLX stores residual / q_norm / k_norm as already-materialized scales
/// (mean ~1). Q80 kernels apply `(1+w)` expecting HF delta-from-one. Convert
/// at pack time. Gated DeltaNet norm stays conventional (ones-init, no +1).
fn mlx_residual_norm_to_delta(name: &str, values: &[f32]) -> Option<Vec<f32>> {
    let is_residual = name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("model.norm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight");
```

64+64+1+16+16 = 161. 48×4 mixer tensors were not converted and stayed
byte-identical. `.f32bin` is the pre-conversion sidecar. `.f32v2` is what
the runtime loads.

### 3.3 Hardlinks — deleting one directory does not free the bytes

```
f32bin nlink hist   {3: 353}
f32v2  nlink hist   {3: 353}
```

Same inode in `uniform-q4-v1`, `mixed-sub15-v1`, `mixed-2p0-materialized`
(353/353 `same_inode_as_g0`). Unlink of G0’s 353 names drops nlink 3→2.
Data is freed only when all three catalogs drop the names.

### 3.4 What breaks if they are removed

- Production `load`: nothing. Not in the manifest.
- `complete_physical_bpw`: nothing. Numerator is listed payload only
  (`8 * 14297694680 / 26895998464`).
- A human or a future reader that opens `*.f32bin` and expects HF-delta
  norms: 161 files would look like MLX `(1+δ)`. That reader does not exist
  at HEAD.
- Sibling catalogs that still point at the same inodes: directory entries
  remain valid until they unlink.

TOKEN_NS: **0**. Resident: **0**. Disk: **10,584,840** B after all three
unlinks.

---

## 4. Same defect class, checked

Prior campaigns: identity bookkeeping and index bloat dominated three times.
Checked on this genome, not assumed absent.

### 4.1 Read every token, does not change per token

| object | bytes/token | changes? | must-move? |
|---|---:|---|---|
| Q4 codes | 12,810,977,280 | no | yes — operands |
| Q4 scales | 800,686,080 | no | yes **for this codec** |
| f32 mixer/norms | 10,582,016 | no | yes — other kernels |
| embed table | 2,720 moved / 675,430,440 resident | no | gather one row |
| `rows,cols,groups_per_row,eps,theta,heads,dims` via `set_bytes` | tens of bytes × 964 | no | no — ICB intern |
| HashMap name strings | 0 weight bytes | no | no — `qwen38_layer_name` `format!` every call |

`qwen38_layer_name` (`qwen38_geometry.rs:116-118`) heap-allocates
`language_model.model.layers.{i}.{suffix}` on every encode. 401 GEMVs +
norms + mixer f32 lookups per token. This is the encode tax
(**MEASURED** 919,250 ns G024 / 886,200 ns complete-wall), not DRAM.

### 4.2 Buffer that exists only to satisfy an API

| buffer | bytes | bound on G0 `step`? |
|---|---:|---|
| load `Vec<u8>` | ≤675,430,440 transient | no, dropped |
| `split_qkv` | 10,240 × 4 = 40,960 | no |
| `split_b` / `split_a` | 48 × 4 × 2 = 384 | no |
| `hgravs_mid` | 160 × 4 = 640 | no (mixed HGRAVS only) |
| **dead workspace** | **41,984** | no |
| codes + scales as two `MTLBuffer`s | 0 extra unique | yes, kernel signature |

Dead workspace **DERIVED** from

```43:46:crates/hawking-core/src/model/qwen38_geometry.rs
pub const QWEN38_IN_PROJ_QKV_ROWS: usize = 10_240;
pub const QWEN38_IN_PROJ_Z_ROWS: usize = 6_144;
pub const QWEN38_IN_PROJ_A_ROWS: usize = 48;
pub const QWEN38_IN_PROJ_B_ROWS: usize = 48;
```

plus `QWEN38_MIXED_HGRAVS_RANK = 160` (`qwen38_hybrid_decode.rs:37`).
Allocated at `Qwen38HybridWorkspace::allocate` `:800-809`. G0
`encode_deltanet` (`:2620-2628`) binds fused `in_proj_qkvz` / `in_proj_ba`
directly. Manifest `fused_in_proj_layers=48`. Split path is the unfused
fallback (`encode_split_deltanet_projections`). `hgravs_mid` is bound only
from the mixed HGRAVS encode. **41,984 B resident, 0 TOKEN_NS.**

Two-buffer split: 402 × 2 + 353 = **1,157** weight buffers. Extra
`set_buffer` × 401 GEMVs/token. Unique bytes unchanged. TLB tax
**UNMEASURED**, **ESTIMATED** ≪ 100 µs.

### 4.3 Value computed per token from constant inputs

- `groups_per_row = cols.div_ceil(64)` (`:1578`)
- launch grid from constant `rows` (`matvec_kernel.launch(rows)`)
- `QWEN38_RMS_EPS`, `QWEN38_ROPE_THETA`, head/dim `set_bytes`
- 964 encoder create + bind + dispatch + end_encoding

ICB infrastructure is on HEAD (`ReplayableComputeGraph` at
`crates/hawking-core/src/metal/mod.rs:3575`) and is **not** called from
`Qwen38HybridDecodeSession::step` (`:3292-3302`).
`rg ReplayableComputeGraph|executeCommandsInBuffer` in
`qwen38_hybrid_decode.rs`: none.

`git merge-base --is-ancestor 7400acf1b HEAD` → exit 1.
`7400acf1b` = `qwen38-kill-fixed-overhead: ICB replay, encode 886us -> 91us`.

Receipt `QWEN38_FIXED_OVERHEAD_DELETED.json` (later genome, dirty box):

```
encode_host_prepare  886200 → 90981    delta -795219
wait_minus_gpu       425900 → 561994   delta +136094
named_fixed_sum      1.3307 → 0.670934 ms
complete_token_wall  38.216792 → 36.683916 ms
coherence            PASS, 0 fallbacks, greedy ids bit-identical
```

**MEASURED** ceremony cut on that genome. **Not** seated G0. Net named-fixed
**659,766 ns**. Wall Δ includes GPU movement; do not spend the 1.53 ms wall
as a ceremony claim.

### 4.4 Duplicate storage of the same bytes under two names

| pair | identical? | bytes | live? |
|---|---|---:|---|
| `.f32bin` / `.f32v2` (192 mixer) | yes, byte | 7,908,864 | only v2 |
| `.f32bin` / `.f32v2` (161 norms) | no, `bin = v2+1` | 2,675,976 | only v2 |
| embed / lm_head | **no** (head/mid/tail sample differ; different inode) | 2 × 675,430,440 | both, untied |
| G0 Q4 files vs each other | 402 unique inodes | — | — |
| G0 `.f32bin` vs mixed-sub15 / mixed-2p0-materialized | same inode, nlink=3 | 10,584,840 once | none of the bins |

Embed ≠ lm_head. Not a delete.

### 4.5 Identity bookkeeping / index bloat

DSV4F’s three hits, applied here:

| DSV4F defect | Qwen38 G0 | evidence |
|---|---|---|
| hash-per-token (~10 GiB/token) | **ABSENT** | `rg Sha256\|sha256_hex\|hash_invocations` in `qwen38_hybrid_decode.rs` → none |
| clone-tree-on-open | **ABSENT** | `fs::read` of listed artifacts, no clone |
| mmap-index / 755-name catalog | **PRESENT**, load-time | 755 sha256 filenames; 4.91 s warm load at 5.82 GB/s effective |
| labeled_sha of strings | **PRESENT**, receipts only | `g1-baseline-audit.md`; not on `step` |
| `HashMap<String, Q4Weight>` + `format!` per GEMV | **PRESENT**, ceremony | swallowed by 0.89–0.92 ms encode |
| 401-organ vs single-address | **PRESENT**, COMPONENT | see §4.6 |

Identity bookkeeping is **not** the TOKEN_NS wall on this genome. Traffic is.

### 4.6 Catalog topology (index of addressing, not of hashes)

`HONEST_ROOF_WEIGHT_ADDRESSING.json` `verdict`:

```
sealed_weight_addressing_gb_s              639.2522348492898
measured_q4_addr_kernel_roof_gb_s          699.5736545106142
measured_q4_addr_catalog_gb_s              530.6544688491846
single_gemv_at_13p6gb.addr median_ns       19457084
production_catalog_at_13p6gb.addr median_ns 25650709
```

24% is **catalog-probe vs single-GEMV-probe** (`530.65 / 699.57 = 0.759`).
It is **not** “live token vs single-GEMV”. Live attributed addressing is
already 639.25, between the two probes.

**PROJECTED** if sealed 639.25 became single-addr 699.57:
`21,293,102.5 − 19,457,084 = 1,836,018` ns. **UNMEASURED** as a token wall.
Production is a 964-dispatch mixed graph, not one 13.6 GB GEMV and not the
401-organ addr_probe.

**REOPEN_IF:** GPU lane measures complete-token wall of a concatenated
single-address layout against seated G0. Forbidden this lane.

### 4.7 Sibling stored-but-dead (not G0, same class)

`mixed-sub15-v1/packed/attn/`: **304** `*.rice` + 304 json,
**1,165,098,376** B rice (**MEASURED** `stat`). Matches
`PACK_REPORT.json` `per_tensor_class.attention_gemv_rice.bytes=1165098376`
`tensors=304`. Uniform loader reads `manifest.json` artifacts only
(the 755 Q4/f32 files, many hardlinked to G0). Rice is not in that
manifest. Native mixed reader is a different vehicle; wave 1 already
tagged the expand-to-Q4 verdict as confounded.

`mixed-2p0-materialized`: another 14.3 GB Q4 tree, 353 f32bin + 353 f32v2
hardlinked to G0, 210/402 hq30uq4 hardlinked. Expand-vehicle copy. Not a
G0 delete.

Do not unlink rice as a G0 action. Do not treat mixed-sub15’s 1.29 BPW as
a G0 byte deletion.

### 4.8 Headers, logits, gate/up scratch — not the 802 MB class

| object | bytes | class |
|---|---:|---|
| HQ30UQ4 + f32v2 headers | 18,904 | stored only, stripped at upload |
| full-vocab logits | 993,280 W+R | must, this sampler |
| gate+up before silu | 2×69,632 W then 3× R/W per layer | must unless silu-fused GEMV |
| unused device-token embed kernel | 0 | bind choice, ICB-adjacent |

---

## 5. Ranked deletions

Risk units (capability-preserving unless marked KILLS):

- **R1** unlink unused files, no runtime
- **R2** loader / alloc, same bytes in the kernel
- **R3** genome / layout, needs remeasure, should preserve capability
- **R4** codec + kernel, capability unknown or KILLS

`ns/R` = TOKEN_NS recovered / risk. `0` token-ns items sort after any
nonzero, then by risk ascending (do the cheap ones). KILLS sort last
regardless of projected ns.

| rank | action | TOKEN_NS | label | R | ns/R | do |
|---:|---|---:|---|---:|---:|---|
| 1 | Land ICB replay of the 964-dispatch graph; intern static `set_bytes` | **659,766** named-fixed | MEASURED later genome `QWEN38_FIXED_OVERHEAD_DELETED` | 3 | 219,922 | YES. Rebase `7400acf1b`. Not a byte delete. Same class as “constant sent every token”. |
| 2 | Concatenate 401 GEMVs into a single-address (or few-address) layout | **UNMEASURED** token; **PROJECTED** 1,836,018 if 639.25→699.57 | COMPONENT roofs | 3 | — | Experiment, GPU lane. Do not spend the 24% catalog-probe gap as TOKEN_NS. |
| 3 | 11-bit global scale codebook (1903 values) | **PROJECTED** 391,412 | lossless re-encode | 4 | 97,853 | Not first. ALU tax unknown. Quality holds only if exact. |
| 4 | Page-align pack + mmap no-copy bind | **0** token; load **ESTIMATED** tens–hundreds of ms of the 4.91 s | LOAD | 2 | 0 | YES for spawn / child. Pad 6,619,440 B. Keep mmap for process life. |
| 5 | Stop allocating `hgravs_mid`+`split_*` on fused G0 | **0** | resident 41,984 | 2 | 0 | YES. Fallback split path must still allocate if unfused catalog appears. |
| 6 | Skip load-time `is_finite` walk of 420 M scales | **0** token; load **ESTIMATED** up to ~1–2 s CPU | LOAD CPU | 2 | 0 | YES if packer already checked. Fail closed on NaN at first GEMV is worse. Prefer packer seal. |
| 7 | Unlink 353 `*.f32bin` at all 3 nlink sites | **0** | disk 10,584,840 | 1 | 0 | YES hygiene. G0-only unlink does not free the bytes. |
| 8 | Merge codes+scales into one `MTLBuffer` | **UNMEASURED** ≪ | bind/TLB | 3 | ~0 | Skip until a counter shows over-fetch. |
| 9 | Drop per-group scales from HQ30UQ4 / one scale per row / per tensor | **KILLS** | — | 4 | — | Do not. **REOPEN_IF** a generate-proven scale-free codec. |
| 10 | Delete 12.81 GB code plane | not a waste | density | 4 | — | G1 codec. Not this lane. Binding: no expand-to-Q4 vehicle. |

Rank 1 is the only **MEASURED** TOKEN_NS recovery that does not require a
new codec. It is ceremony, 1.7% of live TOKEN_NS, and was already measured
by the fusion-persistent lane. Do not re-profile ICB vs encode as if new.

The only **TRAFFIC** byte delete on this list that is not KILLS is the
11-bit codebook (rank 3): 250,210,594 B, 391 µs **PROJECTED**, R4. Below
ICB on ns/R.

The 800,686,080 B plane is the live traffic lever. It is not a hygiene
unlink. It dies when the codec dies.

---

## 6. Ordered execution (capability-preserving prefix)

1. **Unlink** `uniform-q4-v1/tensors/*.f32bin` **and** the same names under
   `mixed-sub15-v1/tensors/` and `mixed-2p0-materialized/tensors/`. Confirm
   `nlink→0` and `stat` 10,584,840 B reclaimed. Then `ls tensors | wc` =
   755. Manifest unchanged. BPW unchanged.
2. **Stop allocating** the four dead workspace buffers when
   `in_proj_qkvz` is present and mixed HGRAVS is absent. Keep the fields
   for the unfused / mixed fallbacks or `Option` them. `resident_bytes`
   / `qwen38_workspace_bytes` equality check must be updated in the same
   change or attach will fail (`got != expected` is a hard error).
3. **Land ICB** (`7400acf1b`) on this tree. Coherence-gate. Do not re-bench
   against encode-every-token as a discovery. Expected named-fixed
   ~0.66 ms if the later-genome number still holds.
4. **Pack realign** (16 KiB header page + padded planes) + mmap +
   `new_buffer_no_copy_checked`. Fail closed. One greedy-id check.
5. **GPU lane** (not this lane): single-address catalog vs seated G0 token
   wall. Only then spend rank 2.
6. **Do not** strip scales from HQ30UQ4. **Do not** fold to per-row or
   per-tensor scale. **Do not** 11-bit-pack until a kernel exists and a
   locked token wall shows the 391 µs is not eaten by gather ALU.

---

## 7. What this lane did not do

- No Metal, no generate, no lock, no touch of pid 74869 / the resident sock.
- Did not slurp 14 GB payloads. Scale walk was 840 MB of f16. f32 pair
  census was 21 MB. Peak analysis RSS ≪ 20 GB.
- Did not time load with a patched no-copy. The 4.91 s split is ESTIMATED.
- Did not re-derive BPW, TOKEN_NS, or the 13,611,663,360 defended payload.
- Did not treat mixed-sub15 rice or mixed-2p0-materialized as G0 deletes.

---

## 8. Required command output

```
$ test -s workspace/superwave/g1/g1-byte-deletions.md && echo OK
OK
```

```
$ wc -l workspace/superwave/g1/g1-byte-deletions.md
     764 workspace/superwave/g1/g1-byte-deletions.md
```

```
$ git status --porcelain
?? workspace/superwave/g1/g1-byte-deletions.md
```

---

## Completion report

STATUS: IMPLEMENT_READY

CLAIMS:
- Scale plane is **800,686,080** B of f16 group scales, **2/34** of the
  defended GEMV payload, streamed every token. Evidence: §1.2 census;
  `honest_roof.rs:44-46`; header spot-check; kernel `:208-210`.
- That plane is TRAFFIC, not residency. Evidence: `encode_q4_matvec_kernel`
  rebinds scales every GEMV; addr_probe comment; sealed
  `bytes_read=13611663360`.
- **PROJECTED** 1,252,535 ns @ sealed 639.25 GB/s. Not a token measurement.
- Deleting it from HQ30UQ4 **KILLS**. Evidence: §1.4 uniqueness (0
  singletons, 0 rows with unique≤4, range 9.54e-7–0.213).
- Load double-copy is **14,297,694,680 + 14,297,675,776 =
  28,595,370,456** B host, **0** TOKEN_NS. Evidence: `load` `:534-557`;
  `new_buffer_with_bytes_checked` `:2610-2628`;
  `GENESIS_RESIDENT_BODY.json` `resident_weight_bytes`.
- Current files cannot no-copy: 0/402 scale offsets 16 KiB aligned. Pad
  **6,619,440** B would suffice. Evidence: §2.4.
- 353 `*.f32bin`, **10,584,840** B, 192 identical / 161 exact `+1.0`,
  nlink=3 across three catalogs, not in the manifest. Evidence: §3.
- Hash-per-token **ABSENT**. 755-file index bloat **PRESENT** at load, not
  dominant at token. Evidence: §4.5–4.6.
- ICB is the only MEASURED TOKEN_NS recovery in this class (659,766 ns
  named-fixed) and is not on HEAD. Evidence: `7400acf1b` not ancestor;
  `QWEN38_FIXED_OVERHEAD_DELETED.json`.
- Dead workspace **41,984** B allocated, never bound on fused G0.
  Evidence: §4.2.

EVIDENCE:
- Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/{manifest.json,tensors}`
- Receipts via `git show HEAD:…`: `receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.json` `resident_load`; `HONEST_ROOF_WEIGHT_ADDRESSING.json` `verdict` / `q4_production_catalog_addr_probe` / `q4_single_gemv_addr_probe`; `QWEN38_FIXED_OVERHEAD_DELETED.json` `named_fixed_components_ns` / `named_fixed_sum`; `QWEN38_ACTIVE_BUDGET_MEASURED.json`
- Source: `qwen38_hybrid_decode.rs`, `qwen38_pack.rs`, `uniform_q4.rs`, `qwen_uniform_q4.metal`, `metal/mod.rs`, `honest_roof.rs`, `qwen38_geometry.rs`
- Wave-1: `g1-traffic-anatomy.md`, `g1-artifact-inventory.md`, `g1-roof-falsification.md`, `g1-token-anatomy.md`, `g1-fusion-persistent.md`, `g1-baseline-remeasure.md`
- This-lane commands: dir/manifest census; f32 pair + residual `+1.0` walk; Q4 header seek; scale uniqueness walk; nlink/sibling census; `git merge-base --is-ancestor`

CHANGES:
- created `workspace/superwave/g1/g1-byte-deletions.md` only

TESTS:
- `test -s workspace/superwave/g1/g1-byte-deletions.md`
- `wc -l workspace/superwave/g1/g1-byte-deletions.md`
- `git status --porcelain`

RISKS:
- Spending the 24% catalog-probe gap as a token claim.
- Unlinking f32bin in only one catalog (nlink 3→2, 0 bytes freed).
- 11-bit scale pack whose gather ALU eats the 391 µs.
- ICB rebase bit-id on current HEAD not verified (no GPU this lane).
- `qwen38_workspace_bytes` hard-equals attach: dropping dead buffers
  without updating the formula fails load.

UNRESOLVED:
- Token wall of a single-address catalog (GPU lane).
- Split of the 4.91 s warm load (755-file vs finite-check vs memcpy).
- Whether two-buffer codes/scales over-fetch (counter).
- ICB named-fixed Δ on this exact tree after rebase.

NEXT:
- Hygiene: unlink the 353 `*.f32bin` at all three nlink sites.
- Genome: land `7400acf1b` ICB, coherence-gate, do not rediscover.
- Loader: 16 KiB realign + no-copy. Fail closed.
- Do not strip HQ30UQ4 scales. G1 density is a new codec, not this delete.
- GPU lane owns any catalog-topology or codebook token measurement.
