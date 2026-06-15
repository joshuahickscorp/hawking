# STRAND production runtime — `strand-decode-kernel` build plan

**Goal.** Grow `crates/strand-decode-kernel` from a CPU correctness reference (today: just
`lib.rs` with `decode_weights_q12` / `matvec` / `footprint_bytes`) into the on-device **runtime**:
mmap a `.strand` v2 file, decode any tensor to Q12/f32 reusing the `strand-quant` decoder, run a
CPU decode→GEMV, and (macOS) dispatch the existing `shaders/strand_trellis_gemv.metal` kernel with a
GPU-vs-CPU bit-identity test. **Additive only** — no existing public fn, test, or wire format
changes. The GPU **bandwidth measurement** (the M3 gate in `shaders/README.md` §1–§4) is a manual
post-sweep step and is explicitly **not** run in this wave.

This plan is authored against verified sources (symbols/line numbers cited inline). The single
make-or-break risk is GPU contention with the live sweep — see [§7 Risks](#7-risks--safety).

---

## 0. What already exists (verified, do not duplicate)

- **Loader primitive (the one wire-format owner).** `strand_quant::format::read_strand_v2_header`
  (`format.rs:556`) returns `StrandV2Header { flags, source_sha256, tensors: Vec<TensorHeaderV2> }`.
  Each `TensorHeaderV2` (`format.rs:511`) carries `name, shape, rht_seed, l_bits, k_bits, vec_dim,
  has_rht_seed, tail_biting, has_affine_min, block_len, total, n_blocks, table_offset,
  payload_offset, payload_bytes, sideinfo_offset, sideinfo_bytes, table: Vec<BlockOffsetRecord>`.
  All region offsets are **absolute file bytes, page-aligned (PAGE=4096)**; the parser already
  validates magic/version/alignment/bounds. This is "the canonical lean parser every external
  consumer delegates to" (`format.rs:550-555`) — **we call it; we never re-parse**.
- **On-disk per-block record.** `format::BlockOffsetRecord` (`format.rs:242`, `SIZE = 16`):
  `{ bit_offset: u64 (tensor-payload-relative), init_state: u32, scale_q: i32 }`.
- **Owned full read.** `format::read_strand_v2` (`format.rs:697`) → `Vec<OwnedTensorV2>`, each
  `{ base: OwnedTensor { name, shape, rht_seed, l_bits, k_bits, vec_dim, enc: EncodedTensor },
  block_len, table }`. `OwnedTensor.enc` decodes bit-identically to v1 (proven by
  `strand_v2_round_trip_matches_v1_q12`, `format.rs:1029`).
- **Decoders.** `decode::decode_tensor_fixed(enc, cfg)` (`decode.rs:95`, reference) and
  `decode::decode_lean(enc, cfg)` (`decode.rs:268`, aligned-word lean path) — **byte-identical**,
  guarded by `decode_lean_is_bit_identical` (`decode.rs:491`, sweeps fold/no-fold, k∈{2,3,4},
  tail-biting, affine-min). `reconstruct_q(scale_q, quantile_q) = (i64 product) >> SCALE_SHIFT(16)`
  (`decode.rs:37`). `Q12_TO_F32 = 2^-12` (`decode.rs:461`, exact).
- **CPU runtime today.** `lib.rs`: `decode_weights_q12(enc, cfg, lut)` (`lib.rs:22`),
  `matvec(enc, cfg, lut, out, in, x)` (`lib.rs:34`, Q12·(1/4096)·x), `footprint_bytes` (`lib.rs:61`).
- **GPU kernel + the 16B↔80B distinction.** `shaders/strand_trellis_gemv.metal` (fused B.6a) and
  `strand_trellis_gemv_predec` (B.6b). The on-disk record is 16 B; the **GPU-side `BlockEntry` is
  80 B** (`metal:58-66`), built by the *host loader at upload time* — it pre-expands `eff[8]` from
  `scale_q` + the side-info sub-scales and rebases `bit_offset` to absolute-in-the-row-blob. The
  file stays dense; the kernel gets the pre-expanded record (`shaders/README.md:49-71`).
- **`metal` is already a locked, compiled dependency.** `crates/strand-quant/Cargo.toml:26-28`
  pins `metal = "0.27"` + `objc = "0.2"` under `[target.'cfg(target_os="macos")'.dependencies]`,
  and `strand-quant/src/metal_backend.rs` actively uses it (`encode.rs:61-64` inits a
  `OnceLock<MetalViterbi>`). `Cargo.lock:597` has `metal 0.27.0` resolved. **The live
  `quantize-model` binary was already built with the metal crate present** → adding the same pin to
  `strand-decode-kernel` pulls an already-built rlib, **zero lockfile churn, zero strand-quant
  recompile**. Established gating idiom (`metal_backend.rs:199-203`): `Device::system_default()?`
  then `device.new_library_with_source(MSL, &CompileOptions::new())`.

---

## 1. Module layout

```
crates/strand-decode-kernel/
  Cargo.toml                 # + memmap2 dep; + metal/objc (macos-only); strand-quant path dep already present
  src/
    lib.rs                   # UNCHANGED public fns; add `pub mod loader; pub mod gemv; #[cfg(macos)] pub mod metal;`
    loader.rs                # NEW — mmap + read_strand_v2_header → zero-copy region slices; reconstruct EncodedTensor view
    gemv.rs                  # NEW — CPU decode→GEMV over a loaded tensor (extends matvec to the v2/mmap path)
    metal.rs                 # NEW, #[cfg(target_os="macos")] — compile the .metal, bake 80B BlockEntry, dispatch, GPU==CPU test
    bin/kernel-bench.rs      # UNCHANGED
  shaders/                   # UNCHANGED (strand_trellis_gemv.metal, README.md)
```

Rationale for three modules (not one): `loader.rs` is **pure byte-parsing + `memmap2`** with no
Metal symbol, so it compiles on every target (matches the dismantle reader's "compiles on every
target, no Metal symbol" requirement, `STRAND-dismantle-integration.md:75`). `gemv.rs` depends only
on `loader` + `strand_quant::decode`, so the CPU runtime is fully testable with `cargo test` on
non-macOS / GPU-less CI. `metal.rs` is the only `#[cfg(target_os = "macos")]` + `unsafe` surface,
isolated exactly as `strand-quant` isolates `metal_backend.rs` (`lib.rs:60`
`pub(crate) mod metal_backend;`).

---

## 2. `Cargo.toml` edits (exact)

`crates/strand-decode-kernel/Cargo.toml` — append to `[dependencies]` and add a macOS target table:

```toml
[dependencies]
strand-quant = { path = "../strand-quant" }   # already present (Cargo.toml:8)
memmap2 = "0.9"                                # NEW — mmap the .strand v2 file (read-only)

# Metal path. SAME pin as strand-quant (Cargo.toml:26-28) so the lockfile is unchanged
# and the already-compiled metal/objc rlibs are reused. macOS-only; the rest of the crate
# (loader/gemv + the public CPU API) builds with no GPU on every target.
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.27"
objc  = "0.2"
```

- `memmap2 = "0.9"` is the version `STRAND-dismantle-integration.md:75` names (`memmap2::Mmap`,
  `unsafe { Mmap::map }`). It is **not** in `Cargo.lock` yet → this is the only new lockfile entry
  (a tiny, leaf, pure-Rust crate; resolving it does not recompile strand-quant).
- `metal`/`objc` are pinned identically to `strand-quant` → **no version bump, no lock change** for
  them. (Note: dismantle pins `metal = "0.29"`; we deliberately match the *strand* workspace's 0.27
  to avoid touching the lock the live binary was built against. A 0.27↔0.29 reconciliation, if ever
  wanted for the dismantle path, is a separate hardening item — `STRAND-dismantle-integration.md:6`
  shim, out of scope here.)

**Why this is sweep-safe:** `cargo check -p strand-decode-kernel` and `cargo test -p
strand-decode-kernel` compile only this crate + already-built deps; they never invoke
`cargo build --release` and never touch `target/release/quantize-model`. See §7.

---

## 3. `loader.rs` — mmap + zero-copy region slices

**Public surface (all additive):**

```rust
use memmap2::Mmap;
use strand_quant::format::{read_strand_v2_header, StrandV2Header, TensorHeaderV2, BlockOffsetRecord};
use strand_quant::encode::EncodedTensor;
use strand_quant::TrellisConfig;

/// An mmap'd .strand v2 file + its parsed header. The mmap is kept alive for the
/// lifetime of the loader so all returned slices are zero-copy borrows of the file.
pub struct StrandModel {
    mmap: Mmap,
    header: StrandV2Header,
}

/// Zero-copy view of ONE tensor's on-file regions (no decode, no copy).
pub struct TensorView<'a> {
    pub hdr: &'a TensorHeaderV2,
    pub payload: &'a [u8],     // mmap[payload_offset .. +payload_bytes]  (== EncodedTensor.bits)
    pub sideinfo: &'a [u8],    // mmap[sideinfo_offset .. +sideinfo_bytes]  (empty if sideinfo_offset==0)
    pub table: &'a [BlockOffsetRecord],   // &hdr.table — the parsed 16B records
}

impl StrandModel {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self>;     // File::open + unsafe { Mmap::map(&f) }
    pub fn from_mmap(mmap: Mmap) -> Result<Self, String>;             // calls read_strand_v2_header(&mmap)
    pub fn header(&self) -> &StrandV2Header;
    pub fn tensor_names(&self) -> impl Iterator<Item = &str>;
    pub fn view(&self, name: &str) -> Option<TensorView<'_>>;         // slice payload/sideinfo by absolute offset
    pub fn config_for(&self, h: &TensorHeaderV2) -> TrellisConfig;    // see §3.2
    pub fn encoded_tensor(&self, name: &str) -> Option<EncodedTensor>;// see §3.3 (owned EncodedTensor view)
}
```

### 3.1 `open` / `from_mmap`

```rust
pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
    let f = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };          // same idiom as STRAND-dismantle-integration.md:80
    Self::from_mmap(mmap).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
pub fn from_mmap(mmap: Mmap) -> Result<Self, String> {
    let header = read_strand_v2_header(&mmap)?;    // THE canonical parser; bounds already checked here
    Ok(Self { mmap, header })
}
```

Because `read_strand_v2_header` already validated `payload_offset + payload_bytes <= flen` and the
page-alignment of every region (`format.rs:635-652`), `view` can slice **without** re-checking
bounds — but we still slice through `mmap.get(a..b)` (returning `Option`) so a malformed file can
never panic the runtime. The `table` is taken straight from `hdr.table` (already parsed; no second
read of the on-disk 16B records).

### 3.2 `config_for` — rebuild a `TrellisConfig` from header fields

The decoders need a `&TrellisConfig`, but the v2 header stores only `l_bits`, `k_bits`, `block_len`,
`vec_dim` (the file does **not** store the whole config — by design). `TrellisConfig` is a plain
`#[derive]`-able struct of public fields (`trellis.rs:14-23`: `pub l_bits: u32`, `pub k_bits: u32`,
`pub block_len: usize`, and a `vec_dim` field exposed via `vec_dim()` `trellis.rs:141`). Construct it
directly from the header:

```rust
pub fn config_for(&self, h: &TensorHeaderV2) -> TrellisConfig {
    TrellisConfig {
        l_bits: h.l_bits as u32,
        k_bits: h.k_bits as u32,
        block_len: h.block_len as usize,
        vec_dim: h.vec_dim,        // (the private/public field — confirm exact field name when editing trellis.rs:14)
    }
}
```

> **Verified (`trellis.rs:14-36`):** all four fields are `pub` (`l_bits: u32`, `k_bits: u32`,
> `block_len: usize`, `vec_dim: u32`), so the struct literal above **compiles as-is** — no new
> constructor is strictly required. (`TrellisConfig::new(l, k, block_len)` exists at `trellis.rs:51`
> but does not take `vec_dim`.) **Optional polish:** add an additive
> `TrellisConfig::from_parts(l, k, block_len, vec_dim)` (4 lines, clamps like `new`) so the runtime
> doesn't hard-code field layout; pure CPU, sweep-safe via `cargo check -p strand-quant`. Not a
> blocker for the wave.

The decode LUT comes from `strand_quant::codebook::codebook_lut(l_bits)` (`codebook.rs:481`) exactly
as `decode_tensor_fixed` does internally (`decode.rs:99`).

### 3.3 `encoded_tensor` — owned `EncodedTensor` for the CPU decode path

For the CPU GEMV we need an `EncodedTensor` (`bits`, `blocks: Vec<BlockMeta>`, `total`, flags). Two
options were considered:

| Option | Cost | Verdict |
|---|---|---|
| (A) call `read_strand_v2(&mmap)` and pull the one tensor's `OwnedTensorV2.base.enc` | parses **all** tensors, copies every payload | **rejected** — defeats mmap; O(model) copy per tensor |
| (B) reconstruct one tensor's `EncodedTensor` from its `TensorView` (payload + side-info + table) | one tensor only; `bits` is a `to_vec()` of the payload slice; `blocks` rebuilt from table + side-info | **chosen** |

**Justification:** `read_strand_v2` already contains the exact, tested per-tensor reconstruction
(`format.rs:804-901`): it walks `n_blocks`, slices each block's `sub_scales` (constant `sub_stride
= ceil(6*(block_len/32)/8)` for full blocks, tail for the last), and — if `has_affine_min` — the
`min_base_q` i32 array + packed `mins`, then builds `BlockMeta { scale_q, sub_scales, min_base_q,
mins, init_state, n }`. `loader.rs` **reproduces that single-tensor walk against the `TensorView`
slices** (not the whole file). To avoid hand-copying that logic (the drift the whole "one owner"
design forbids, `format.rs:550-555`), the clean move is:

> **Preferred additive `strand-quant` helper:** add `format::owned_tensor_v2_from_view(hdr:
> &TensorHeaderV2, payload: &[u8], sideinfo: &[u8]) -> Result<OwnedTensorV2, String>` that factors
> the existing `read_strand_v2` inner loop (`format.rs:804-901`) into a per-tensor function, and have
> the existing `read_strand_v2` call it in its loop. This is a **pure refactor of existing tested
> code into a reusable unit** (additive: the old fn keeps its signature/behavior, the round-trip
> tests still pass), and it keeps `loader.rs` from re-implementing the side-info slice math. Then
> `encoded_tensor` is `self.view(name).map(|v| owned_tensor_v2_from_view(v.hdr, v.payload,
> v.sideinfo).map(|o| o.base.enc))`.

If that refactor is deemed out of scope for the wave, the fallback is to call `read_strand_v2` once,
cache the `Vec<OwnedTensorV2>` in `StrandModel`, and index it — correct and simple, but copies all
payloads. **Recommend the helper** (zero-copy, single-owner-preserving, ~15 lines moved).

### 3.4 loader tests (in `loader.rs #[cfg(test)]`, sweep-safe — `strand-decode-kernel` only)

1. `open_round_trips_header` — build a tiny v2 archive in-test via
   `strand_quant::format::write_strand_v2` (the test pattern at `format.rs:1029`: 4×256 weights,
   `for_bpw(3.0)`, STRICT), write to a `tempfile`-less temp path (use
   `std::env::temp_dir()`), `StrandModel::open`, assert `header().tensors.len()`, names, shapes,
   `all_strict()`.
2. `view_slices_match_payload` — assert `view(name).payload == &archive[payload_offset..+bytes]`
   and `== enc.bits` (the encoder's bitstream), and `table.len() == n_blocks`,
   `table[0].bit_offset == 0`.
3. `encoded_tensor_decodes_identically_to_v1` — **the loader's core gate**: `decode_tensor_fixed(
   &model.encoded_tensor(name).unwrap(), &model.config_for(h))` equals `decode_tensor_fixed(&enc,
   &cfg)` of the originally-encoded tensor (mirrors `format.rs:1075-1077`). This proves the mmap
   loader path is bit-identical to the in-memory encode.

---

## 4. `gemv.rs` — CPU decode → GEMV over a loaded tensor

The existing `lib.rs::matvec` takes an `&EncodedTensor` + `&TrellisConfig`. `gemv.rs` adds the
**v2/mmap-driven** entry points on top, reusing `decode_lean` (the production decode path; the kernel
ports its arithmetic) rather than re-deriving anything:

```rust
use crate::loader::{StrandModel, TensorView};
use strand_quant::decode::decode_lean;     // production lean path (decode.rs:268), bit-identical to fixed

/// Decode tensor `name` to Q12 ints (row-major [out*in]). Reuses decode_lean.
pub fn decode_tensor_q12(model: &StrandModel, name: &str) -> Option<Vec<i32>>;

/// y = W·x for tensor `name`, W decoded on the fly. Mirrors lib.rs::matvec (Q12·(1/4096)·x)
/// but pulls (enc, cfg, shape) from the loaded header. `x.len() == in_features`.
/// NOTE: x is RHT-space activation iff the tensor was RHT-encoded — the caller applies the
/// per-tensor RHT once (host responsibility; see shaders/README.md "RHT on the activation" and
/// the determinism caveat). This fn does the integer decode + float MAC only.
pub fn matvec_named(model: &StrandModel, name: &str, x: &[f32]) -> Option<Vec<f32>>;
```

`out_features = shape[0]`, `in_features = shape[1]` from `hdr.shape` (row-major `[out, in]`,
`format.rs:309-312`). Implementation = `decode_lean(&enc, &cfg)` then the identical accumulation loop
as `lib.rs:47-54`. **No new arithmetic** — this is the existing `matvec` body fed from the loader, so
it inherits `decode_lean`'s bit-identity guarantee for free.

**`gemv.rs` tests (sweep-safe):**
1. `matvec_named_matches_lib_matvec` — for a tiny in-test v2 archive,
   `gemv::matvec_named(&model, name, &x) == lib::matvec(&enc, &cfg, None, out, in, &x)` within
   `1e-6` (same float order). This ties the v2 runtime path to the existing reference.
2. `decode_tensor_q12_matches_decode_lean` — `gemv::decode_tensor_q12 == decode_lean(&enc,&cfg)`
   exactly (integer equality).

---

## 5. `metal.rs` — compile, bake 80B `BlockEntry`, dispatch, GPU==CPU test

`#[cfg(target_os = "macos")]` module, `#![allow(unsafe_code)]` (Metal FFI), mirroring
`strand-quant/src/metal_backend.rs` structure exactly.

### 5.1 Compile the shader (reuse the established idiom)

```rust
use metal::{Device, CompileOptions, ComputePipelineState, CommandQueue, MTLResourceOptions, MTLSize};

const GEMV_MSL: &str = include_str!("../shaders/strand_trellis_gemv.metal");  // the authored kernel

pub struct StrandGpu {
    device: Device,
    queue: CommandQueue,
    fused: ComputePipelineState,     // function "strand_trellis_gemv"
    predec: ComputePipelineState,    // function "strand_trellis_gemv_predec"
}
impl StrandGpu {
    /// Returns None on a GPU-less host (CI), exactly like MetalViterbi::new (metal_backend.rs:199-203).
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let lib = device.new_library_with_source(GEMV_MSL, &CompileOptions::new()).ok()?;
        let f = lib.get_function("strand_trellis_gemv", None).ok()?;
        let p = lib.get_function("strand_trellis_gemv_predec", None).ok()?;
        let fused = device.new_compute_pipeline_state_with_function(&f).ok()?;
        let predec = device.new_compute_pipeline_state_with_function(&p).ok()?;
        let queue = device.new_command_queue();
        Some(Self { device, queue, fused, predec })
    }
}
```

`include_str!` (not a copy) keeps `shaders/strand_trellis_gemv.metal` the single source of the MSL.

### 5.2 Build the GPU-side 80 B `BlockEntry` from the on-disk 16 B record + side-info

This is the host loader's job per `shaders/README.md:49-71` and `metal:43-66`. The struct (host-side,
`#[repr(C)]`, **must `const _: () = assert!(size_of::<BlockEntry>() == 80)`**):

```rust
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockEntry {
    bit_offset: u32,   // ABSOLUTE bit pos in the per-tensor w_bits blob (== on-disk bit_offset; tensor-relative IS absolute-in-blob)
    init_state: u32,   // BlockOffsetRecord.init_state (baked unconditionally — README invariant 2)
    scale_q: i32,      // BlockOffsetRecord.scale_q  (debug / future un-folded path)
    eff: [i32; 8],     // PRE-EXPANDED eff_scale_q(scale_q, sub_code[s])  (README invariant 3)
    n: u16,            // block weight count (last block of a row may be < 256)
    d: u16,            // vec_dim (1 at deploy; reserved for B.7)
    _pad: u32,         // pad to 80 B
}
```

**Baking algorithm (one tensor → `Vec<BlockEntry>` + the flat `w_bits` blob = the payload slice):**

For each of the `n_blocks` `BlockOffsetRecord`s in `hdr.table`:
- `bit_offset = rec.bit_offset as u32` — the on-disk value is **already tensor-payload-relative**,
  and buffer(0) `w_bits` is exactly that one tensor's payload blob (`TensorView.payload`), so
  tensor-relative == absolute-in-blob. (`shaders/README.md:55-57`, `metal:55-57`.)
- `init_state = rec.init_state`, `scale_q = rec.scale_q`.
- `eff[s]` for `s in 0..n_sub`: unpack this block's 6-bit sub-scale codes from the **side-info
  SUB-SCALES half** and fold:
  - block `b`'s sub-scale bytes start at `sideinfo[b * sub_stride ..]` with
    `sub_stride = ceil(6 * (block_len/32) / 8)` for full blocks (`format.rs:806-810`,
    `STRAND-format-v2-spec.md:255-256`); the **last** block uses its own `n_sub =
    ceil(n_last/32)`. (We have `n_last = total - (n_blocks-1)*block_len`, `format.rs:849-853`.)
  - `let codes = strand_quant::encode::unpack_sub_scales(&sub_bytes, n_sub);` (`encode.rs:140`)
  - `eff[s] = strand_quant::decode::eff_scale_q(rec.scale_q, codes[s]);` (`decode.rs:59`)
  - zero-fill `eff[n_sub..8]` (deploy block_len=256 ⇒ n_sub=8 ⇒ no padding; short final block
    pads the unused tail — never read because `j >> 5 < n_sub` for `j < n`).
- `n = rec_block_n as u16` (from the same `n_last`/`block_len` rule), `d = hdr.vec_dim as u16`,
  `_pad = 0`.

> **Reuse, do not re-derive.** The side-info slice math (`sub_stride`, last-block tail, the
> `mins`-half base) is **already implemented and tested** inside `read_strand_v2`
> (`format.rs:804-901`). If §3.3's `owned_tensor_v2_from_view` helper is added, `metal.rs` bakes
> straight from the resulting `OwnedTensorV2.base.enc.blocks[b].sub_scales` +
> `.scale_q` + `.init_state` + `.n` — i.e. **call `eff_scale_q` over the already-unpacked
> `BlockMeta`**, and never touch raw side-info bytes. This is the cleanest path and is what the test
> in §5.4 should exercise. (Affine-min: at the 3-bit deploy point `has_affine_min == false`
> (`shaders/README.md:83`, `STRAND-format-v2-spec.md:262`); if a 4-bit tensor is ever baked,
> `BlockEntry` needs a parallel `off[8] = eff_min_q(min_base_q, mins_code[s])` (`decode.rs:80`) and
> the kernel an `+ e->off[j>>5]` — **flag, assert `!has_affine_min` in the baker for now**.)

### 5.3 Upload buffers + dispatch (matches the kernel's binding table exactly)

Buffer table from `metal:103-117` / `shaders/README.md:28-47`:

| idx | contents | source |
|---|---|---|
| 0 `w_bits` | `TensorView.payload` (the per-tensor blob; **4-byte aligned** — payload_offset is page-aligned so the blob base is aligned) | `device const uchar*` |
| 1 `x_rht` | `RHT(x)` length `cols` — **host applies RHT once** (see note) | `device const float*` |
| 2 `y` | output length `rows` | `device float*` |
| 3 `rows` | `shape[0]` | `constant uint&` |
| 4 `cols` | `shape[1]` (multiple of 256) | `constant uint&` |
| 5 `tbl` | the baked `Vec<BlockEntry>`, **row-major, stride `bpr = cols/256`** | `device const BlockEntry*` |
| 6 `k_bits` | `hdr.k_bits` (3) | `constant uint&` |
| 7 `l_bits` | `hdr.l_bits` (7) | `constant uint&` |
| 8 `lut_q12` | `codebook_lut(l_bits)` (`codebook.rs:481`) | `device const int*` |
| tg(0) `sh_lut` | `2^L` ints (set length, no buffer) | threadgroup |
| tg(1) `sh_red` | 256 floats | threadgroup |
| tg(2) `sh_wq12` | **predec only**, `cols` ints | threadgroup |

Dispatch: **one threadgroup per output row, 256 threads/threadgroup** — grid `MTLSize{ width: rows,
height:1, depth:1 }`, threadgroup `MTLSize{ width:256, height:1, depth:1 }` (`metal:8-13`,
`shaders/README.md:30-32`). Buffers via `new_buffer_with_data(..., StorageModeShared)` (the
`metal_backend.rs:360` idiom); threadgroup memory via `set_threadgroup_memory_length(i, bytes)`.

> **`BlockEntry` table is row-major with stride `bpr`** (`metal:141`
> `&tbl[(uint64_t)gid * bpr + b]`). The on-disk v2 table is **block-sequential per tensor**
> (`bit_offset` monotonic over all blocks, `format.rs:435-443`); for a STRICT tensor
> (`in_features % block_len == 0`, `all_strict()` true) block index `gid*bpr + b` **is** the
> sequential block index, so the baked `Vec<BlockEntry>` is just the table in file order. Assert
> `hdr.shape[1] % 256 == 0` and `all_strict()` before dispatch (RAGGED tensors are not a deploy
> target — `STRAND-format-v2-spec.md`/`format.rs:333-345`).

> **RHT caveat (carry into the test + any caller).** The kernel dots decoded **RHT-space** weights
> against `x_rht`; the host must compute `x_rht = rht_forward(x, RhtConfig::from_seed(hdr.rht_seed))`
> with **block=256, row-restart, same seed** (`shaders/README.md:90-110`). This per-token FWHT is
> **float** and outside the integer-decode guarantee. For the **§5.4 correctness test we sidestep
> RHT entirely** by asserting on the *decoded Q12 integers* (kernel-internal, pre-MAC), which are
> bit-identical regardless of activation — the only thing this wave proves on GPU.

### 5.4 GPU correctness test (GPU-GATED, **no timing**)

`#[cfg(target_os = "macos")] #[test] fn gpu_q12_matches_cpu_decode_lean()`:

```rust
let Some(gpu) = StrandGpu::new() else { eprintln!("no Metal device; skipping"); return; };
```

(`Device::system_default().is_none()` → clean skip on GPU-less CI, exactly the prompt's gate and
`metal_backend.rs:199`'s pattern.)

Steps:
1. Encode a tiny STRICT tensor in-test (e.g. `rows=2, cols=256` ⇒ `bpr=1`; and a `rows=2, cols=512`
   ⇒ `bpr=2` case to exercise the row stride), `for_bpw(3.0)` (k=3,L=7,d=1, `has_affine_min=false`),
   write v2, `StrandModel::open`.
2. **CPU reference:** `let cpu_q12 = decode_lean(&enc, &cfg);` (`decode.rs:268`) — the bytes the
   kernel must reproduce.
3. **GPU:** dispatch a **debug variant** that writes the decoded Q12 (pre-MAC) for the row to an
   output buffer. The shipped kernel folds decode into the MAC; to read the Q12 out, use
   `strand_trellis_gemv_predec` (`metal:225`) which **already materializes `sh_wq12[col0+j] =
   (int)(((long)es*(long)q)>>16)` into a `cols`-wide shmem tile** (`metal:278`) — add a tiny
   sibling kernel **in a test-only MSL string** (or extend predec behind a debug buffer) that copies
   `sh_wq12` to a `device int*` instead of doing Pass B. Assert `gpu_q12[row*cols + i] ==
   cpu_q12[row*cols + i]` for all `i` (**exact integer equality** — the moat; `metal:25-29`,
   `shaders/README.md:151-157`).
   - *Lighter alternative that needs no new kernel:* run `strand_trellis_gemv` with a **one-hot
     `x_rht`** (`x_rht[c] = 1.0` at one column, else 0) so `y[row] = (decoded Q12 at (row,c)) *
     (1/4096)`; recover the Q12 as `round(y[row] * 4096)` and compare to `cpu_q12[row*cols+c]`.
     Sweep `c` over the row. This reuses the **shipped** kernel unmodified and still proves Q12
     identity (float touch is one exact power-of-two scale). **Recommend this** for the wave — zero
     new MSL, and it also exercises the real fused decode loop end-to-end.
4. (Optional, still no perf claim) assert the full `y` from `strand_trellis_gemv` matches
   `lib::matvec` / `gemv::matvec_named` on the same `x` within a float-reduction tolerance
   (`1e-4` rel) — the `shaders/README.md:157` "kernel y matches CPU matvec" sanity. Skip if RHT
   wiring is deferred (use a tensor encoded with RHT off so `x_rht == x`).

**This test asserts correctness only. No `MTLCounterSampleBuffer`, no bandwidth, no Q4_K
comparison** — those are the manual M3 steps in `shaders/README.md:146-205`, explicitly out of this
wave.

---

## 6. Test plan (what runs in this wave vs. what's deferred)

| test | location | gate | runs in wave? |
|---|---|---|---|
| `open_round_trips_header` | `loader.rs` | none | ✅ `cargo test -p strand-decode-kernel` |
| `view_slices_match_payload` | `loader.rs` | none | ✅ |
| `encoded_tensor_decodes_identically_to_v1` | `loader.rs` | none | ✅ (the loader correctness gate) |
| `matvec_named_matches_lib_matvec` | `gemv.rs` | none | ✅ |
| `decode_tensor_q12_matches_decode_lean` | `gemv.rs` | none | ✅ |
| `gpu_q12_matches_cpu_decode_lean` | `metal.rs` | `StrandGpu::new().is_some()` | ✅ **iff a Metal device is free** — but see §7: prefer to author + `cargo check`, run the GPU test only when the sweep is done |
| existing `matvec_matches_manual_decode`, `footprint_scales_with_bpw` | `lib.rs` | none | ✅ unchanged |
| M3 bandwidth / roofline / Q4_K head-to-head | `shaders/README.md` §1–§4 | manual | ❌ **deferred — documented M3 manual step** |

`cargo test -p strand-decode-kernel` is a **separate crate from the live `strand-quant`/`quantize-
model`** binary and is the prompt's sanctioned command. The GPU test self-skips with no device;
**on this machine it must not be run while the sweep holds the GPU** (§7) — author it, `cargo check`
it, and let the human run it post-sweep alongside the README measurement.

---

## 7. Risks + safety

1. **Live sweep contention (the only critical risk).** pid 89192 holds all 12 CPU cores + the GPU +
   the `scratch/qwen-7b/*.safetensors` files + the `target/release/quantize-model` binary.
   - `cargo check -p strand-decode-kernel` and `cargo test -p strand-decode-kernel` compile **only
     this crate + already-built rlibs** (strand-quant is already built; metal/objc already in the
     lock and built). They do **not** run `cargo build --release`, do **not** rebuild strand-quant,
     and do **not** touch `quantize-model`. ✅ allowed by the brief.
   - **Do NOT run the `gpu_q12_matches_cpu_decode_lean` GPU test while the sweep runs** — it would
     submit Metal command buffers and contend the GPU the encoder uses (`encode.rs:61` MetalViterbi).
     Author it and `cargo check`; the human runs it post-sweep. The CPU loader/gemv tests are GPU-
     free and safe anytime.
   - Use **small synthetic tensors** in every test (≤ a few KB, encoded in-test). Never read or bake
     `scratch/qwen-7b`.
2. **Lockfile churn.** Only `memmap2 = "0.9"` is new (leaf, pure-Rust; `Cargo.lock` is already
   `M`-dirty per git status, so adding one leaf entry is low-impact). `metal`/`objc` match
   strand-quant's existing 0.27 pin ⇒ **no change** to those entries → the live binary's deps are
   untouched. Run `cargo check -p strand-decode-kernel` (not `update`) so only the needed entry is
   added.
3. **`metal` 0.27 vs dismantle's 0.29.** Deliberately pin 0.27 to match the *strand* workspace and
   avoid disturbing the lock. The dismantle-side integration (a `strand-quant` path dep there,
   `STRAND-dismantle-integration.md:425-432`) is a **separate repo / separate wave**; this plan
   touches only the strand workspace. No dismantle file is edited or built (per the brief).
4. **`TrellisConfig` construction coupling.** `config_for` reads private-ish layout. Mitigation:
   add the 4-line additive `TrellisConfig::from_parts` to `strand-quant` (CPU-only, `cargo check
   -p strand-quant` is sweep-safe) so the runtime depends on a stable constructor, not field order.
5. **Side-info slice drift.** The bake re-uses `read_strand_v2`'s tested per-tensor walk via the
   proposed `owned_tensor_v2_from_view` factor-out (a pure refactor of `format.rs:804-901`), so the
   runtime never re-implements `sub_stride`/tail/`mins`-half math — preserving the "one wire-format
   owner" invariant (`format.rs:550-555`). If the refactor is skipped, fall back to caching
   `read_strand_v2` (copies payloads, but still single-owner). Either way: **no second parser.**
6. **RHT determinism (carried, not solved here).** The host activation RHT is float and outside the
   integer guarantee (`shaders/README.md:104-110`). This wave's GPU test asserts on **Q12 integers**
   (RHT-independent), so it is unaffected; any real GEMV caller must pin RHT block=256/seed/row-
   restart. Flagged for the integration/hardening pass.
7. **Affine-min at >3-bit.** `BlockEntry` has no `off[8]` yet; the baker must `assert!(
   !hdr.has_affine_min)` (3-bit deploy ⇒ false). A 4-bit deploy needs the parallel `eff_min_q`
   expansion + a kernel `+ e->off[j>>5]` (`decode.rs:80`, `shaders/README.md:83-86`). Flagged.

---

## 8. Sequencing (suggested)

1. `Cargo.toml`: add `memmap2` + macOS `metal`/`objc`. `cargo check -p strand-decode-kernel`.
2. (strand-quant, additive) `TrellisConfig::from_parts` + `format::owned_tensor_v2_from_view`
   (factor of the existing `read_strand_v2` loop). `cargo check -p strand-quant`.
3. `loader.rs` + its 3 tests. `cargo test -p strand-decode-kernel`.
4. `gemv.rs` + its 2 tests. `cargo test -p strand-decode-kernel`.
5. `metal.rs` (`#[cfg(macos)]`) + `StrandGpu::new` + bake + dispatch + the one-hot `x_rht` Q12
   test. **`cargo check -p strand-decode-kernel`** only (do **not** run the GPU test until the
   sweep is done).
6. Hand off the GPU correctness run + the `shaders/README.md` §1–§4 M3 bandwidth measurement to the
   human, post-sweep.
