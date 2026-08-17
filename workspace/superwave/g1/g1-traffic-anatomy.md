# G1 traffic anatomy — Qwen3.8

Lane: `18-traffic-anatomy`. Tree: `2eee9a004`. No GPU, no inference, no artifact mutation.
Every number is tagged **MEASURED** (command or receipt), **DERIVED** (geometry/source), or **ESTIMATED**.
A component microbenchmark is not a token-level claim.

G0 vehicle under study: `uniform-q4-v1` (complete physical BPW **MEASURED** `4.252735126866492`).
G0 TPS / TOKEN_NS in the campaign brief are unverified claims. Receipts that already measured them are cited as such, not re-run.

---

## 0. Four-count (G0, one decode token, fused-Q4 genome)

| count | bytes | label | what it is |
|---|---:|---|---|
| stored in the G0 artifact | `14,297,694,680` listed payload + `10,584,840` orphan `.f32bin` + `238,879` manifest | **MEASURED** disk | files on disk under `uniform-q4-v1` |
| must be active per token | `13,611,663,360` GEMV codes+scales + `10,582,016` f32 mixer/norms + `2,720` embed row + state + activations | **DERIVED** from geometry + catalog | unique bytes the G0 kernels are written to touch |
| actually crossing DRAM per token (weights) | `13,611,663,360` GEMV | **MEASURED** in prior receipt, **not re-run** | addr_probe streams exactly those bytes; no f32 expansion buffer |
| served from cache / residency, not re-streamed | embed table `675,427,720` + (after load) the whole 14.3 GiB stays in unified DRAM | **DERIVED** | SLC cannot hold 13.6 GiB unique-once; the economic change is SSD→DRAM, not DRAM→SLC |

**Gap (bytes that move minus bytes that must move), production decode, this codec:**

- Weight path: **≈ 0**. G0 consumes HQ30UQ4 in-register (`geo_tpr64_tg128`). There is no per-token expand-to-f32 staging buffer. **KILLS** the hypothesis that Qwen3.8 G0 still pays an 802 MB *expansion* staging tax per token.
- Codec addressing that *does* move and is not independent weight information: **`800,686,080` B** of f16 group scales (`2/34` of the GEMV payload). Same order as the historical “802 MB” figure.
- Load-time (not per-token, once the Genesis body is resident): every listed tensor is `fs::read` then `newBufferWithBytes` — **two host copies of `14,297,675,776` B**. Q80 already has a no-copy mmap bind; Qwen38 does not use it.
- Stored-but-dead: **353** sibling `.f32bin` files, **`10,584,840` B**.

That gap — scale-plane + load double-copy + dead `.f32bin` + unused split workspace — is the traffic lever that is *not* “delete bytes from the MLP”.

---

## 1. Genome identity

**MEASURED** from `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`:

```
schema                    hawking.ascent.qwen38_language_uniform_q4.v1
tensor_count              755
q4_tensors                402
f32_tensors               353
skipped_vision_tensors    333
fused_in_proj_layers      48
source_weight_elements    26895998464
tensor_payload_bytes      14297694680
complete_physical_bpw     4.252735126866492
```

This is the same BPW constant compiled into the ledger:

```30:32:crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
pub const UNIFORM_Q4_V1_BPW: f64 = 4.252735126866492;
pub const ACTIVE_BUDGET_BYTES: u64 = 13_622_264_240;
pub const EMBED_TABLE_BYTES: u64 = 675_430_440;
```

Architecture (census receipt, not re-derived): dense Qwen3_5 text, 64 layers, 48 Gated-DeltaNet + 16 GQA, hidden 5120, intermediate 17408, vocab 248320, untied lm_head. Not MoE. Vision tower skipped.

Source BF16 (the thing the packer read, **not** the G0 runtime artifact):

```
BF16_SHARDS 11 sum 54713606485
```

`du -sk` on this box:

```
53457532   .../qwen38-27b/bf16
13977284   .../qwen38-27b/uniform-q4-v1
6848148    .../qwen38-27b/mixed-2p0-v1
8879152    .../qwen38-27b/mixed-2p0-materialized
```

(`du -sk` is 1024-blocks; BF16 shard sum `54,713,606,485` B is the honest stored-source number.)

---

## 2. Stored bytes

### 2.1 Listed catalog (what load reads)

**MEASURED** Python census of `manifest.json` + `tensors/`:

```
listed_files          755
listed_disk_bytes     14297694680
manifest_json_bytes   238879
```

Class split of listed payload (same totals as `QWEN38_ACTIVE_BUDGET_MEASURED.json`):

```
mlp          n=192 bytes= 9091161600 elems=17112760320 bpw=4.250003590303309
linear_attn  n=336 bytes= 2961704064 elems= 5562051072 bpw=4.259873238359218
full_attn    n= 96 bytes=  891325184 elems= 1677729792 bpw=4.250148925054077
embed        n=  1 bytes=  675430440 elems= 1271398400 bpw=4.250000251691366
lm_head      n=  1 bytes=  675430440 elems= 1271398400 bpw=4.250000251691366
norm         n=129 bytes=    2642952 elems=     660480 bpw=32.0125
```

`linear_attn` here is the name-class, not the GEMV class: 144 fused Q4 matrices + 192 f32 mixer tensors (conv1d / A_log / dt_bias / linear_attn.norm).

Q4 on-disk layout: 40-byte HQ30UQ4 header + 32 code bytes + 2 f16 scale bytes per group of 64 weights.

```
q4_header_over_geo     16080     (= 402 × 40)
f32_len_prefix_est      2824     (= 353 × 8)
headers_total_est      18904
resident_if_no_headers 14297675776
```

`14,297,694,680 − 18,904 = 14,297,675,776`. That last number is **exactly** `resident_weight_bytes` in `GENESIS_RESIDENT_BODY.json` (`14,297,675,776`). Headers stay on disk; Metal buffers hold codes/scales/f32 values only.

### 2.2 Orphan stored copies (dead)

```
all_files     1108
extra         353
extra_ext     {'.f32bin': 353}
extra_bytes   10584840
F32BIN_PAIRS  identical 192  different 161
```

`artifact_filename` hashes the tensor *name*, so `.f32v2` (manifest) and `.f32bin` (orphan) share a stem:

```270:274:crates/hawking-core/src/model/qwen38_pack.rs
fn artifact_filename(name: &str, ext: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(name.as_bytes());
    format!("{:x}.{ext}", digest)
}
```

Current packer writes only `.f32v2` (`QWEN38_F32_EXT = "f32v2"`). The 353 `.f32bin` files are leftovers. 161 content-diffs match the residual-norm set (`w → w−1` in `mlx_residual_norm_to_delta`): 129 layernorms of 20,488 B + 32 q/k norms of 1,032 B. Load never opens `.f32bin` (`load` joins `row.artifact` from the manifest).

Stored waste: **`10,584,840` B**. Not per-token DRAM.

### 2.3 What is *not* in the G0 artifact

- Vision: 333 tensors skipped (census + manifest `skipped_vision_tensors`).
- BF16 source shards: 11 × ~5 GB, **`54,713,606,485` B**. Pack-time only.

---

## 3. Bytes that must be active per token

Dense model: every GEMV weight is used every token except the embedding table (one gathered row).

### 3.1 Defended GEMV payload (codes + scales)

Adjudicated in-tree:

```44:46:crates/hawking-core/src/backend/honest_roof.rs
/// Bytes the Q4 GEMV kernels actually stream per token (codes + f16 scales).
/// This is the defended denominator for `weight_addressing`.
pub const GEMV_PAYLOAD_BYTES: u64 = 13_611_663_360;
```

**DERIVED** from listed Q4 shapes, excluding embed:

```
gemv_codes    12810977280
gemv_scales     800686080
gemv_sum      13611663360
```

Matches `theoretical_weight_bytes()` unit test:

```704:708:crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
        assert_eq!(b.mlp_bytes, 9_091_153_920);
        assert_eq!(b.linear_attn_bytes, 2_953_789_440);
        assert_eq!(b.full_attn_bytes, 891_289_600);
        assert_eq!(b.lm_head_bytes, 675_430_400);
```

`9,091,153,920 + 2,953,789,440 + 891,289,600 + 675,430,400 = 13,611,663,360`.

Three larger “active” numbers exist and are **wrong for weight_addressing** (same file, `adjudicate_byte_counts`):

| name | bytes | why not GEMV traffic |
|---|---:|---|
| `ACTIVE_BUDGET_BYTES` / manifest minus embed | `13,622,264,240` | includes 40 B headers + mixer f32 |
| bandwidth receipt remainder | `13,621,829,601` | stale embed subtract `675,865,079` (real embed table is `675,430,440`) |
| ledger geometry-active | `13,618,141,856` | GEMV + f32 norms + embed row |

Manifest-minus-geometry extras reconcile **exactly**:

- MLP: `7,680` = 192 × 40 headers
- lm_head: `40`
- linear non-GEMV: `7,908,864` = conv1d `7,864,704` + A_log `9,600` + dt_bias `9,600` + linear_attn.norm `24,960`
- full non-GEMV: `33,024` = 32 × `1,032` q/k RMS

Those mixer/norm f32 bytes **are** must-active (other kernels), just not `weight_addressing`.

### 3.2 Embed row

`q4_matrix_bytes(1, 5120) = 2,720`. Kernel:

```589:602:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_embedding_lookup(
    device const uchar* codes [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device float* output       [[buffer(2)]],
    ...
    const uint element = token * hidden + id;
    output[id] = qwen_uniform_q4_value(codes, scales, element, group_size);
}
```

Binds the whole 675,430,400 B table; indexes one row (80 groups × 34 B). **DERIVED** must-move `2,720` B. Rest of the table is residency, not token traffic.

G0 `encode_embed` still passes `token` via `set_bytes`. A device-token sibling kernel exists (`qwen_uniform_q4_embedding_lookup_device_token`) and is unused here.

### 3.3 f32 mixer / norms

f32 listed payload `10,584,840` − 353 × 8 prefixes = **`10,582,016` B** resident. All of it is read every token (64-layer walk). **DERIVED**.

### 3.4 Recurrent / KV state

Geometry (`qwen38_geometry.rs` tests):

- rec: `48 × 48 × 128 × 128` f32 = `150,994,944` B
- conv: `48 × 10,240 × 3` f32 = `5,898,240` B
- GQA write: `16 × 2 × 4 × 256 × 4` = `131,072` B/token
- GQA read: that × `seq_len`

Every DeltaNet layer reads+writes its own slot, and every token visits all 48. Must-move rec+conv R+W = `2 × 156,893,184 = 313,786,368` B.

Sealed TOKEN_NS `kv_state` row: `bytes_read = 159,383,552`, `bytes_written = 157,024,256`, sum `316,407,808`.
`313,786,368 + 16×2×19×4×256×4 + 16×2×4×256×4 = 316,407,808`. That receipt was taken at **seq ≈ 19**. **MEASURED** (receipt) / **DERIVED** (closed arithmetic).

### 3.5 Activations (workspace)

`qwen38_workspace_bytes` is enforced at attach (`got != expected` is a hard error). **DERIVED** from the same formula:

```
activation_total           1,691,396
seq=1    workspace       158,715,652
seq=128  workspace       175,361,796
seq=2048 workspace       427,020,036
```

`175,361,796` is the `workspace_bytes` in `GENESIS_RESIDENT_BODY.json` (`max_seq_len_measured = 128`).

Dead on the fused-Q4 production path (allocated, not bound by `encode_mixer`/`encode_dense_mlp` when `in_proj_qkvz` exists):

```
hgravs_mid  640
split_qkv   40,960
split_b     192
split_a     192
DEAD        41,984
```

Split buffers are live only if the catalog is unfused (`encode_split_deltanet_projections`). G0 manifest has `fused_in_proj_layers: 48`.

Activation *traffic* (read+write of live f32 workspace, **ESTIMATED** from the 15-dispatch schedule, not GPU-traced this lane) is tens of MB plus the 314 MB state walk. It is not 802 MB.

### 3.6 Dispatch shape (not bytes, but the genome)

```56:59:crates/hawking-core/src/model/qwen38_token_ns_ledger.rs
pub fn production_dispatches_per_token() -> u64 {
    1 + (QWEN38_LAYERS as u64) * (QWEN38_FULL_LAYER_DISPATCHES as u64)
        + QWEN38_TERMINAL_HEAD_KERNELS.len() as u64
}
```

`1 + 64×15 + 3 = 964`. Test: `production_dispatch_count_is_964`.

`step()` on this commit builds a fresh `TokenCommandBuffer` and calls `encode_embed` + `encode_layers` + `encode_terminal`. No ICB, no serial-group encoder. `concurrent_independent` defaults **false**.

ICB infrastructure exists (`ReplayableComputeGraph` / `executeCommandsInBuffer` in `crates/hawking-core/src/metal/mod.rs`) and a receipt claims it deleted 964 encoder creates (`QWEN38_FIXED_OVERHEAD_DELETED.json`, other commit). **SOURCE on `2eee9a004` does not call it from Qwen38 `step()`.** Host encode tax is still 964 `set_buffer`/`set_bytes`/`dispatch`/`end_encoding` cycles. That is time, not extra weight bytes; it *can* flush activation cache lines between encoders (**ESTIMATED**, not measured here).

---

## 4. Bytes that actually cross DRAM per token

### 4.1 Weights — no expansion staging

Production kernel unpacks in registers:

```183:211:crates/hawking-core/shaders/qwen_uniform_q4.metal
kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes       [[buffer(0)]],
    device const half* scales       [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float* output            [[buffer(3)]],
    ...
            const float scale = float(scales[rgb]);
            const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));
            acc += qwen_uniform_q4_unpack8(packed, scale, input, col);
```

addr_probe is the same addressing with the load sunk so LLVM cannot DCE it, and **no input-vector load**:

```223:226:crates/hawking-core/shaders/qwen_uniform_q4.metal
// Diagnostic: same launch geometry as geo_tpr64_tg128, but only the
// addressing + DRAM load of scales and packed codes. The loaded values
// are sunk into `acc` so the compiler cannot DCE the traffic. No nibble
// unpack, no input-vector load, no FMA.
```

`HONEST_ROOF_WEIGHT_ADDRESSING.json` (date 2026-08-17, GPU timestamps, **not re-run this lane**):

- defended bytes `13,611,663,360`
- sealed `weight_addressing` `21,293,102.5` ns → **639.25 GB/s**
- production catalog 401 GEMVs addr_probe median **530.65 GB/s**
- single 13.612 GB GEMV addr_probe median **683.80 GB/s**
- decode reconstruction attributed **1.808 ms** (~5% of that token wall)

`QWEN38_RECONSTRUCTION_IS_FREE.json`: at tpr64, 32/33 codecs match f32 control time. Reconstruction is not a byte-staging tax on this launch.

**KILLS (this genome, per-token):** “low-BPW then expand to float/Q4 then generic GEMV.” G0 is already representation-specific Metal. Mixed path comment in `qwen38_hybrid_decode.rs` lines 4–7: packed bytes stay packed; missing codec fails closed; no reconstruct-to-Q4.

### 4.2 What the sealed TOKEN_NS row-set counted

From `QWEN38_TOKEN_NS_LEDGER.json` (production GPU `33,912,333` ns, wall `35,227,917` ns — **MEASURED** in that receipt):

| component | bytes_read | bytes_written | note |
|---|---:|---:|---|
| weight_addressing | 13,611,663,360 | 0 | defended GEMV |
| weight_decode_reconstruction | 0 | 0 | ALU only |
| deltanet | 4,718,592 | 2,162,688 | activations; state carved out |
| gqa | 2,883,584 | 327,680 | activations + short KV |
| dense_swiglu | 8,912,896 | 5,767,168 | gate/up/act/down |
| normalization | 5,283,840 | 2,641,920 | |
| kv_state | 159,383,552 | 157,024,256 | rec+conv+KV at seq≈19 |
| terminal_head | 675,430,400 | 993,284 | **do not add** to addressing; comment says lm_head traffic lives in addressing/decode |
| unattributed_residual | 2,720 | 20,480 | embed gather + encoder-gap |

**Do-move unique weight bytes ≈ must-move unique weight bytes.** The 675,430,400 terminal_head `bytes_read` is a classification double-count if summed naively.

### 4.3 Roofs — conditioned, not datasheet

Do **not** treat `bytes / 819e9` or `bytes / 411.51e9` as a floor for this workload.

- `411.51 GB/s` is the 512 MiB point of a sequential `unique_once` *read-reduce*, different kernel, `uint32` nbytes (cannot even take a 13.6 GB `nbytes`). The 1024 MiB point of the same sweep was `301.63 GB/s` and was discarded (`HONEST_ROOF_WEIGHT_ADDRESSING.json`).
- `535–637 GB/s` is a **64 MiB reuse** roofline (`Q80_DECODE_SHAPE_BANDWIDTH.json`, `claim_boundary.reuse_64mib_x_iters_is_a_cache_friendly_roofline_not_decode`).
- Qwen38’s own GEMV addr_probe on the real 13.6 GB payload is **530–684 GB/s** depending on 401-dispatch vs one-dispatch topology. That is a **measured regime for this kernel**, still not a physics floor: it is conditioned on `geo_tpr64_tg128`, Shared storage, 401 launches, current barriers.

`CORRECTION_ROOF_IS_CONDITIONED.json`: a roof is conditioned on the current execution genome. The 406 GB/s “at the wall” claim divided 13.6 GB by *whole-token* GPU time. Honest attribution is 13.6 GB / 21.3 ms addressing.

This lane did **not** re-measure GPU traffic. DRAM-cross for weights is **receipt-MEASURED**, not live-MEASURED.

---

## 5. Bytes served from cache / residency

Three caches, three answers.

### 5.1 SSD / page cache (between processes)

After the Genesis body is up, load is paid once. `GENESIS_RESIDENT_BODY.json`:

```
resident_weight_bytes                  14297675776
workspace_bytes                          175361796
rss_bytes                              15511666688
phys_footprint_bytes                   15385816488
ioaccelerator_graphics_dirty_bytes     14473019392
load_count                             1 across three serves
```

`ioaccelerator_graphics_dirty ≈ 14.47 GB` ≈ the Metal Shared weight set. RSS `15.51 GB` is host + some of that. **MEASURED** (receipt). Process-pool children do **not** share those pages (`QWEN38_SHARED_SESSIONS.md`). One process, N sessions, one `Arc<Qwen38HybridWeights>` does.

Four independent GEMVs against the same `lm_head` still pay ~4× GPU time. Residency saves memory, not DRAM.

### 5.2 Unified DRAM (between tokens)

14.3 GB << 96 GB. The entire G0 table is resident. Per-token economics are **unique-once DRAM**, not streaming from SSD. That is the change vs a “page the shard each token” assumption.

Host-admission comment (`qwen38_host_admission.rs`) quotes `8.77 GB RSS/child` against a `14.3 GB` Metal set. Those two ledgers do not close: Metal Shared can sit in IOAccelerator dirty without showing in RSS. Machine-wide child cost `10.2 GB` still undercounts `14.3 GB`. **UNRESOLVED** accounting, not a second smaller artifact.

### 5.3 On-chip SLC / L2 (within a token)

13.6 GB unique-once does not fit in SLC (unpublished; tens of MB, not GB). **ESTIMATED** cache-served *weight* bytes after first use of each group: **≈ 0**.

What *can* be cache-served:

- Input vector `5,120 × 4 = 20,480` B, reread across rows of one GEMV (addr_probe deliberately does **not** load it).
- Embed table minus one row: `675,427,720` B resident, `2,720` B streamed.
- Tiny f32 mixer tensors (~10.6 MB total) after the preceding 200 MB GEMV have already blown SLC — treat as DRAM.

A G1 1.5 BPW table is `26,895,998,464 × 1.5 / 8 = 5,042,999,712` B **DERIVED**. Still >> SLC. Same unique-once DRAM regime, smaller working set. Cache economics do **not** flip at 1.5 BPW.

---

## 6. The 802 MB class and the double-fetch class

### 6.1 Three different “802 MB” numbers (do not conflate)

| number | bytes | what it actually is | class |
|---|---:|---|---|
| G0 Q4 scale plane | **`800,686,080`** | 2 f16 bytes × (GEMV groups). `2/34` of `13,611,663,360` | **DERIVED**. Moves every token. Addressing metadata, not independent weights |
| mixed-2p0 `mlp_gate_proj` stored | **`802,177,344`** | `PACK_REPORT.json` organ size at 1.125 BPW | **MEASURED** stored. Not per-token staging |
| linear `out_proj` stored (G0) | **`802,162,560`** | 48 × Q4 `5120×6144` | **MEASURED** stored, **must-active** (out_proj GEMV) |
| DSV4F indexer/compressor | **`801,075,968`** | `DSV4F_TENSOR_SCHEDULE.md`. Not on the BOS graph | prior-campaign stored/optional decode tax |

Prior Hawking “802 MB staging on a model whose table already fit” is the **same class** as: bytes that exist only to satisfy a codec/API and still stream even though the table is resident. On Qwen38 G0 that class is the **scale plane**, not an f32 scratch slab.

DSV analog (steal the science, do not resurrect the campaign): `host.memcpy` of `10,280,249,760` B into Metal scratch after mmap, on a table that fits (`DSV4F_HOST_WALL_BASELINE.json`, `disappear_under_resident_packed: YES`). Qwen38 G0 pays the same *shape* at **load** (`fs::read` + `new_buffer_with_data`), then **stops**. Per-token it does not memcpy the table again.

### 6.2 Every shard fetched twice

**Load (G0, 755 tensors) — SOURCE:**

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

`new_buffer_with_bytes_checked` is `newBufferWithBytes` / `new_buffer_with_data`, `StorageModeShared` — a **copy** (`crates/hawking-core/src/metal/mod.rs` 2610–2628).

Zero-copy already exists and is used on Q80 experts:

```2640:2675:crates/hawking-core/src/metal/mod.rs
        /// **Zero-copy** buffer view over a borrowed mmap region.
        pub unsafe fn new_buffer_no_copy(...)
        pub fn new_buffer_from_verified_bytes(...)
        pub fn new_buffer_no_copy_checked(...)
```

Qwen38 load never calls it. Fail-closed no-copy needs 16 KiB pointer+length alignment; HQ30UQ4 payload after a 40-byte header is **not** page-aligned unless the file is laid out for it.

Host traffic at one load: file read `14,297,694,680` + MTL copy `14,297,675,776` = **`28,595,370,456` B**. Transient `Vec<u8>` per tensor, then dropped. Peak extra RAM during load ≈ largest tensor (`lm_head`/`embed` `675,430,440`) plus the growing Metal set — not 2×14 GB simultaneously if tensors are processed one-by-one.

**Pack-time source shards:** `SourceIndex::open` reads each safetensors **header**, then `read_f32` `pread`s each tensor from the same 11 shards (`qwen38_pack.rs` 123–182). That is header+payload, not a double payload fetch, unless the kernel page-cache already holds the shard and the subsequent `pread` is a cache hit (then it is DRAM, not SSD). Not per-token.

**Disk duplicates:** 353 `.f32bin` siblings — stored twice, fetched once (manifest path only).

### 6.3 Other waste that is *not* 802 MB

| waste | bytes | when | evidence |
|---|---:|---|---|
| HQ30UQ4 + f32 headers | 18,904 | stored only | §2.1 |
| unused split/hgravs workspace | 41,984 | every session | §3.5 |
| 964 encoder creates | 0 weight bytes | every token | `step()` 3292–3302; ICB unused |
| `set_bytes` of static geometry (rows/cols/eps) | tens of bytes × 964 | every token | `encode_q4_matvec_kernel` 1587–1589. ICB receipt interned these; this tree still sends them |
| full-vocab logits | 993,280 W + 993,280 R | every token | workspace `logits`; argmax. Must for this sampler. A fused `..._final_norm_lm_head_simdgroup8` exists in the same metal file and is **not** bound by `encode_terminal` |
| gate+up materialised before silu | `2 × 69,632` W then `3 × 69,632` R/W | every layer | `encode_dense_mlp` 2566–2585. Must unless silu is fused into the GEMV |
| codes and scales as two buffers | 0 extra unique bytes | every GEMV | two allocations, two binds, two address spaces. Possible extra TLB; **ESTIMATED** |

---

## 7. 96 GB economics vs streaming

```
unified_memory_bytes  103,079,215,104    HONEST_ROOF hardware block
G0 listed payload      14,297,694,680    13.3% of 96 GiB
G0 + seq=128 workspace 14,473,037,572
G0 + seq=2048 workspace 14,724,714,716
Genesis RSS after load 15,511,666,688    receipt
```

The table fits. A streaming design that page-faults 13.6 GB/token would be a category error on this box. The live Genesis organism already holds one copy.

What residency does **not** buy:

- It does not make the second token free. Unique-once GEMV still walks 13.6 GB of DRAM.
- It does not put 13.6 GB in SLC.
- Sharing the `Arc` across sessions does not amortize `lm_head` (`QWEN38_SHARED_SESSIONS.md`: 1× GPU `1,013,791` ns vs 4× concurrent `4,022,124` ns ≈ 3.97×).
- Concurrent decode ceiling remains 1 (same receipt). Extra sessions are memory, not TPS.

Peak-RAM note for this lane: census walked filenames and a 239 KB manifest. Did not slurp 14 GB.

---

## 8. Gap, lever, G1 implication

Let

- `S` = stored listed = `14,297,694,680`
- `A_w` = must-active GEMV = `13,611,663,360`
- `A_aux` = f32 + embed row = `10,584,736`
- `A_state(seq)` = `313,786,368 + 131,072×seq` (KV read+write; write is one slot)
- `D_w` = DRAM weight stream ≈ `A_w` (this genome)
- `C_embed` = `675,427,720` resident, not streamed

**`D_w − A_w ≈ 0`.** The expansion-staging lever is already dead on G0.

Remaining byte levers, largest first:

1. **Density of the 12.81 GB code plane.** mixed-2p0 already cut MLP to 0.848 BPW and left attention+embed+norm at 4.250 BPW (`PACK_REPORT.json`, `complete_physical_bpw 2.0856`). That pack is **MEASURED incoherent** (`QWEN38_DENSITY_ROOT_CAUSE.json`, `QWEN38_COHERENCE_FLOOR_BRACKETED.json`). Attention is 5.20 / 7.01 GB of that pack (74%). G1 `< 1.5` complete BPW is an attention-codec problem, not an MLP-staging problem.
2. **Scale plane `800,686,080` B/token.** 5.88% of GEMV traffic. Exists only because group-64 Q4 carries a per-group f16. A codec that folds or drops scales deletes this without touching capability *if* quality holds. Not free: it is the decode metadata the current kernel multiplies by.
3. **Load double-copy `14.3 GB`.** Irrelevant to TOKEN_NS once resident; relevant to bring-up, child spawn, and the “table fits so why memcpy” class. Needs a page-aligned catalog layout before `new_buffer_no_copy_checked` can bind.
4. **Dead `.f32bin` `10.6 MB`.** Delete-only. No runtime effect.
5. **Execution genome that is *not* bytes.** Sealed TOKEN_NS: deltanet `3.73 ms` at **1.84 GB/s** (`measured_over_floor 223`), gqa `2.44 ms` at **1.31 GB/s** (`313×`), normalization `2.37 ms` at **3.35 GB/s** (`123×`). Those rows are occupancy/dispatch, not DRAM. `CORRECTION_ROOF_IS_CONDITIONED` is the binding sentence: attacking only BPW leaves ~10 ms of non-GEMV GPU on the table. G1 `TOKEN_NS <= 10,000,000` is **not** reachable by density alone if those rows stay.

Projected (not measured) GEMV addressing at constant 639 GB/s if codes+scales scale with BPW:

| complete BPW | GEMV bytes **DERIVED** | addr ns @ 639 GB/s **PROJECTED** |
|---|---:|---:|
| 4.2527 (G0) | 13,611,663,360 | 21,293,000 (receipt, not projected) |
| 2.0856 (mixed-2p0, incoherent) | ~6.68e9 | ~10.5e6 |
| 1.5 (G1 target) | ~4.80e9 | ~7.5e6 |

Even the 1.5-BPW addressing projection is only the GEMV row. Adding sealed non-GEMV `~12.6 ms` (deltanet+gqa+swiglu+norm+kv+head+sync+residual from that ledger) overshoots 10 ms. Those components must move too, or the 10 ms target is a genome change, not a pack change.

**REOPEN_IF** (expansion staging): a path that materializes f32/Q4 scratch larger than the packed organ before the GEMV. Current mixed decoder refuses that path. Do not reintroduce it.

**REOPEN_IF** (DRAM-cross): a GPU counter or `MTLCounterSampleBuffer` run on this box shows GEMV bytes loaded ≠ `13,611,663,360` (double-fetch inside the kernel, or scale/code over-fetch from poor alignment). Cheapest experiment: one locked addr_probe vs decode_probe vs full `geo_tpr64` on the production 401-shape catalog — already in `honest_roof.rs`. This lane was forbidden to run it.

**REOPEN_IF** (ICB): `step()` is rebound to `ReplayableComputeGraph` and a locked wall re-measures encode. Until then, treat `QWEN38_FIXED_OVERHEAD_DELETED` as a *different commit’s* claim, not this tree’s production path.

---

## 9. What this lane did not do

- No Metal, no generate, no lock, no touch of the live Genesis process.
- Did not re-derive the 11-shard HF census; used `QWEN38_ARCH_CENSUS.json` + on-disk shard `stat`.
- Did not open mixed-2p0 as a production genome; only stole its stored-byte organ split.
- Peak RSS of this analysis: manifest + directory entries. Under 20 GB.

---

## 10. Required command output

```
$ test -s workspace/superwave/g1/g1-traffic-anatomy.md && echo OK
OK
```

```
$ wc -l workspace/superwave/g1/g1-traffic-anatomy.md
     594 workspace/superwave/g1/g1-traffic-anatomy.md
```

```
$ git status --porcelain
?? workspace/superwave/g1/g1-traffic-anatomy.md
```

Census command that produced §0–§2 numbers: Python over
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/{manifest.json,tensors}`
(output pasted in those sections).

---

## Completion report

STATUS: SUPPORTED

CLAIMS:
- G0 stored listed payload is `14,297,694,680` B at complete BPW `4.252735126866492`. Evidence: `uniform-q4-v1/manifest.json` fields `tensor_payload_bytes`, `complete_physical_bpw`; Python census in §2.
- G0 Metal-resident weights are `14,297,675,776` B (headers stripped). Evidence: `14,297,694,680 − 18,904`; `GENESIS_RESIDENT_BODY.json` `resident_weight_bytes`.
- Must-active GEMV per token is `13,611,663,360` B. Evidence: `honest_roof.rs` `GEMV_PAYLOAD_BYTES`; geometry unit test; Python `gemv_sum`.
- Of that, `800,686,080` B are f16 group scales (`2/34`). Evidence: Python `gemv_scales`; kernel comment 32 code + 1 half scale per 64 weights.
- Production decode does not expand packed weights to f32/Q4 scratch. Evidence: `geo_tpr64_tg128` in-register unpack; `qwen38_hybrid_decode.rs` lines 4–7; `QWEN38_RECONSTRUCTION_IS_FREE.json`.
- Embed table `675,430,440` B is stored/resident; `2,720` B must move. Evidence: manifest row + embedding kernel.
- Load fetches every listed tensor twice (file `Vec` + `newBufferWithBytes`). Evidence: `qwen38_hybrid_decode.rs` 534–557; `metal/mod.rs` 2610–2628. No-copy API unused.
- 353 orphan `.f32bin` files duplicate f32 storage (`10,584,840` B). Evidence: `1108 − 755` files; pair census identical 192 / different 161.
- ICB is not on the Qwen38 `step()` path in `2eee9a004`. Evidence: `step` 3292–3302; no ICB symbol in `qwen38_hybrid_decode.rs`.
- 96 GB residency makes SSD-streaming a false model; unique-once DRAM remains. Evidence: 14.3 GB vs 96 GB; Genesis `ioaccelerator_graphics_dirty_bytes`; `Q80_DECODE_SHAPE_BANDWIDTH.json` reuse-vs-unique-once split.
- `D_w − A_w ≈ 0` for this codec; the live byte lever is the scale plane + density of the 12.81 GB codes + non-GEMV occupancy. Evidence: §4–§8.

EVIDENCE:
- On-disk artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/`
- Receipts (git, not sparse-checked-out; `git show HEAD:…`): `receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json`, `QWEN38_BANDWIDTH_BOUND.json`, `QWEN38_ARCH_CENSUS.json`, `QWEN38_TOKEN_NS_LEDGER.json`, `HONEST_ROOF_WEIGHT_ADDRESSING.json`, `GENESIS_RESIDENT_BODY.json`, `QWEN38_SHARED_SESSIONS.md`, `QWEN38_RECONSTRUCTION_IS_FREE.json`, `QWEN38_DENSITY_ROOT_CAUSE.json`, `QWEN38_FIXED_OVERHEAD_DELETED.json`, `CORRECTION_ROOF_IS_CONDITIONED.json`, `Q80_DECODE_SHAPE_BANDWIDTH.json`, `DSV4F_BYTE_REDUCTION.md`, `DSV4F_HOST_WALL_BASELINE.json`
- Source: `crates/hawking-core/src/model/qwen38_{pack,geometry,hybrid_decode,token_ns_ledger,64_layer_execution_schedule,host_admission}.rs`, `crates/hawking-core/src/backend/honest_roof.rs`, `crates/hawking-core/src/metal/mod.rs`, `crates/hawking-core/shaders/qwen_uniform_q4.metal`
- mixed organ split: `.../mixed-2p0-v1/PACK_REPORT.json`

CHANGES:
- created `workspace/superwave/g1/g1-traffic-anatomy.md` only

TESTS:
- `test -s workspace/superwave/g1/g1-traffic-anatomy.md`
- `wc -l workspace/superwave/g1/g1-traffic-anatomy.md`
- `git status --porcelain`

RISKS:
- ICB receipt vs this tree disagree; treating ICB as shipped would understate host encode waste.
- TOKEN_NS `terminal_head.bytes_read` will double-count lm_head if summed blindly.
- Process RSS vs Metal dirty disagree; admission math that trusts RSS undercounts.
- DRAM-cross for activations/state is schedule-derived, not counter-sampled this lane.

UNRESOLVED:
- Live GPU byte counters on this box (forbidden this lane).
- Why child RSS is 8.77 GB against a 14.3 GB Shared set.
- Whether 16 KiB realignment of HQ30UQ4 files would let no-copy bind without a pack rewrite.
- Whether 964 encoder boundaries cause extra activation writebacks (needs `ProdCbGpu` or Instruments).

NEXT:
- GPU lane: one locked addr/decode/full catalog probe to confirm `D_w = 13,611,663,360` (the REOPEN_IF).
- Genome lane: wire or falsify ICB on `step()` against this tree.
- Density lane: attention codec; do not stage-expand MLP.
- Hygiene: delete the 353 `.f32bin` orphans (pack-only, no runtime).
