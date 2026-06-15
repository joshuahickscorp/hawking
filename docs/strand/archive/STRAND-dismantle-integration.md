# STRAND ⇄ dismantle integration — slotting `.strand` v2 in as a peer format to GGUF

**Status:** design only. No code in this doc. Additive throughout: every existing dismantle
path (GGUF + Q4K_FAST sidecar + AWQ + W4A8) stays bit-identical and on by default. STRAND is a
*new* opt-in weight source gated behind an env flag, exactly like `DISMANTLE_QWEN_Q4K_FAST`.

**Why this shape.** dismantle already has the seam for this. It carries one foreign quantized
layout beside GGUF today — the Q4K_FAST `.dismantle` sidecar (`crates/dismantle-core/src/q4k_fast.rs`,
baked by `tools/awq_bake/`). `.strand` v2 is the *second* such layout, and the four touch-points
are the same four Q4K_FAST already uses: a reader module, a baker tool, a Metal GEMV kernel, and one
arm in the `gemv_proj!` dispatch macro. We follow that precedent verbatim so the diff is small and
the existing golden hashes never move. The make-or-break is **§3's Metal decode gate**, not the
plumbing — see `docs/STRAND-metal-decode-gate.md`.

The grounding for "custom format = the vehicle for sub-4-bit, *if* decode stays bandwidth-bound"
is dismantle's own `paradigmshift.md` Part IV (lines 155–185) and §V.3 (lines 271–307, the QTIP /
"validate the kernel before committing the format" / "Q3_K died compute-bound at 24% peak"
asterisk). This integration is the build that answers that asterisk.

---

## 0. The two sides, named

**STRAND artifact (source of truth for bits):** `crates/strand-quant/src/format.rs`
- `write_strand(&[PackedTensor]) -> Vec<u8>` / `read_strand(&[u8]) -> Vec<OwnedTensor>`, magic
  `b"STRQ"`, `VERSION = 1`. `OwnedTensor { name, shape, rht_seed, l_bits, k_bits, vec_dim, enc }`.
- `enc: EncodedTensor { bits: Vec<u8>, blocks: Vec<BlockMeta>, total, has_rht_seed, tail_biting,
  has_affine_min }` (`crates/strand-quant/src/encode.rs:173`).
- `BlockMeta { scale_q: i32, sub_scales: Vec<u8>, min_base_q: i32, mins: Vec<u8>, init_state: u32,
  n: u32 }` (`encode.rs:90`) — **all the per-block fields a GPU needs for random access already
  exist here.** v2 only re-lays them; it does not re-quantize.
- Decode contract: `decode::reconstruct_q(scale_q, quantile_q) = ((scale_q as i64) * (quantile_q
  as i64)) >> SCALE_SHIFT(=16)` (`crates/strand-quant/src/decode.rs:37`). Per the 2026-06-08 gate
  refinement, the i64/native-`32×32→64` product is **load-bearing** (i32 overflows) — the kernel
  must keep it.

**dismantle host (consumes bits):** workspace `crates/{dismantle,dismantle-core,dismantle-serve,
dismantle-bench}` + `tools/{q4k_fast,awq_bake}`.
- GGUF reader: `crates/dismantle-core/src/gguf/{mod.rs,reader.rs}` — `GgufFile::open`,
  `TensorInfo { name, dims, dtype, data_offset, byte_size }`, `GgmlType` (closed `#[repr(u32)]`
  enum, `from_u32` **errors** on unknown tags).
- Backend seam: `crates/dismantle-core/src/backend/mod.rs` — `Backend`/`ComputeBackend`,
  `BackendGemv::gemv`, `Op::Gemv`, `WeightKind` (already has a `Q4kFast` precedent variant),
  `GemvSpec`. Concrete impl `backend/metal.rs` (`MetalBackend`).
- Dense dispatch: `crates/dismantle-core/src/model/qwen_dense.rs` — the `gemv_proj!` macro
  (`~5004`) switches on `$tref.dtype` and threads fast/predec side-tables keyed by `$tref.offset`;
  `ensure_q4k_fast_cache` (`3895`) pins a sidecar as one `PinnedBuffer`; `load_q4k_fast_tensor`
  (`8406`).
- Kernels + shaders: `crates/dismantle-core/src/kernels/mod.rs` (`gemv_q4k_fast_v1_pinned_tcb`,
  `3447`) → MSL `crates/dismantle-core/shaders/quant.metal` (`gemm_q4k_fast_v1`, line 1049).
  Shaders are `include_str!`'d as consts in `metal/mod.rs` and concatenated by
  `all_shader_sources()`; a kernel name resolves through `MetalContext::pipeline(fn_name)`
  (`metal/mod.rs:727`) → `library.get_function`.
- Bench: `crates/dismantle-bench/` (suites `decode`, `prefill`, `competitive`, `bandwidth`),
  `tools/bench/measure_joules.sh` (J/tok via `macmon`).

---

## 1. Where the STRAND reader lives, and the seam it implements

### 1a. Module placement — a sibling of `gguf/`

New module **`crates/dismantle-core/src/strand/`**, peer to `src/gguf/`:

```
crates/dismantle-core/src/
  gguf/            (exists)  mod.rs, reader.rs
  strand/          (NEW)
    mod.rs                 # re-exports; pub use reader::{StrandFile, StrandTensorInfo, StrandQuant}
    reader.rs              # mmap-backed .strand v2 reader (mirrors gguf/reader.rs)
```

Register in `crates/dismantle-core/src/lib.rs` next to the existing `pub mod gguf;` with
`pub mod strand;`. **Platform-neutral, `#[cfg]`-free** (like `gguf` and `backend/mod.rs`): the
reader is pure byte-parsing + `memmap2::Mmap`, no Metal symbol, so it compiles on every target and
in CI on non-macOS.

`reader.rs` mirrors `gguf/reader.rs` structurally:
- `StrandFile { mmap: Mmap, version: u32, tensors: HashMap<String, StrandTensorInfo>, tensor_order:
  Vec<String> }`, with `StrandFile::open(path)` / `from_mmap(mmap)` (same `unsafe { Mmap::map }`
  idiom and the same EOF-guarded `Cursor` as `gguf::reader::Cursor`).
- `StrandTensorInfo { name, shape: Vec<u64>, l_bits, k_bits, vec_dim, flags, n_blocks,
  block_table_offset: u64, bits_offset: u64, bits_len: u64 }` — the v2 random-access descriptor
  (see §1c). This is the STRAND analog of GGUF `TensorInfo`; offsets are **absolute into the mmap**,
  exactly like `TensorInfo.data_offset`.
- `StrandFile::tensor_bytes(name) -> Option<&[u8]>` and a new `block_table(name) -> Option<&[u8]>`
  returning zero-copy slices of the mmap (mirrors `GgufFile::tensor_bytes`).

**Reuse, do not fork, the wire decode.** The byte schema is owned by `strand-quant::format`. The
dismantle reader links `strand-quant` as a workspace dep and calls into a new
`format::read_strand_v2_header(&[u8])` (a v2 sibling of `read_strand`, header-only, no payload
copy). This keeps a single source of truth for the layout and means the encoder and the dismantle
loader can never drift. (See §6 for the `strand-quant` dependency note.)

### 1b. The seam it implements — `GgmlType`-parallel `StrandQuant`, surfaced as a new `WeightKind`

STRAND is **not** a `GgmlType`. That enum is closed and `from_u32` rejects unknown tags
(`gguf/reader.rs:81`); jamming a synthetic tag in would be a lie about the on-disk container and
risks colliding with a future upstream ggml type. Instead, STRAND rides the **existing** extension
point the backend seam already defines for "a foreign layout that is not a GGUF dtype":
`backend::WeightKind`, which today carries `Q4kFast` precisely for "same dispatch geometry, distinct
memory layout, sourced from a sidecar" (`backend/mod.rs:112–128`).

Add one variant:

```rust
// crates/dismantle-core/src/backend/mod.rs — WeightKind
/// STRAND trellis-coded weights (.strand v2): k-bit Viterbi index stream +
/// per-block {scale_q, sub_scales, init_state} side table. Integer, float-free
/// decode (reconstruct_q = (scale_q*quantile_q)>>16). Distinct kernel family.
StrandTrellis,
```

and the matching logical-op capability stays `Op::Gemv` (no new verb — STRAND is a weight storage
class behind the single GEMV verb, the same design rule the seam states for the 31 GGUF GEMV
kernels). A backend that lacks the STRAND kernel returns `false` from `supports(Op::Gemv)`? No —
`supports` is per-*op*, not per-weight-kind; capability for STRAND is expressed by whether
`ensure_strand_cache` found a sidecar, identical to how Q4K_FAST self-gates. No seam-trait signature
changes: `BackendGemv::gemv` already takes a `GemvSpec` whose `weight: WeightKind` selects the
kernel family inside the impl body.

### 1c. `.strand` v2 — the random-access deploy layout (design, not new quant)

`.strand` v1 (`STRQ`) is a **sequential** per-tensor stream: great for CPU round-trip, but a GPU
threadgroup must jump to any `(row, block)` without walking the stream. v2 adds, per tensor, a
**page-aligned block-offset table** so block `b` is O(1) addressable. This is the
`docs/STRAND-metal-decode-gate.md` §"Format v2 implication" plan, made concrete:

```
file:
  [4]   magic   = b"STRQ"
  [4]   version = 2 (LE)              # v1 reader still recognizes v1; v2 is a new code path
  [4]   n_tensors (LE)
  per tensor (header section, tightly packed):
     name_len u32, name bytes,
     n_dims u32, dims[u64...],
     rht_seed u64, l_bits u8, k_bits u8, vec_dim u8, flags u8,   # flags = format::flags::*
     n_blocks u32,
     block_table_offset u64,          # ABS file offset, 16 KB page-aligned
     bits_offset       u64,           # ABS file offset, 16 KB page-aligned
     bits_len          u64
  [pad to 16 KB page]
  per tensor (data section, page-aligned):
     block_table[n_blocks] : array of fixed-stride BlockEntry { 
         bit_offset u64,   # bit position of this block's first symbol within `bits`
         init_state u32,   # BlockMeta.init_state (decoder seed; needed since random access
                           #   cannot replay the trellis from tensor start)
         scale_q    i32,   # BlockMeta.scale_q
         min_base_q i32,   # BlockMeta.min_base_q (0 unless affine-min)
         n          u32,   # weights in this block (last block may be short)
         sub_off    u32,   # offset into a packed sub_scales/mins arena (or inline if vec_dim==1)
     }                      # fixed 32-byte stride ⇒ block b is table_base + b*32, coalesced read
     sub_arena : concatenated 6-bit-packed sub_scales (+ mins if affine-min)
     bits      : the EncodedTensor.bits stream, verbatim from v1
```

Key properties:
- **Built, not re-quantized.** Every field is already in `BlockMeta` / `EncodedTensor`. v2 is a
  layout change in `strand-quant::format` (a new `write_strand_v2`), gated alongside the existing
  `write_strand`. Bit-for-bit, the decoded weights of v2 == v1 == `decode_tensor_fixed`.
- **`bits` verbatim.** The symbol stream is unchanged; only a side index is added. So a v2 file's
  payload bpw equals v1's (`EncodedTensor::payload_bpw`), and the on-disk size delta is just the
  block table (~32 B / block ≈ negligible at 256-weight blocks ≈ 0.01 bpw).
- **mmap-ready + page-aligned**: `block_table_offset`/`bits_offset` on 16 KB boundaries so the
  whole file pins as one `MTLBuffer.newBufferWithBytesNoCopy` (the no-copy idiom GGUF already uses,
  `gguf/reader.rs:2`) and threadgroups index without straddling pages.
- **Determinism floor**: the reader validates `flags` against `strand-quant::format::flags`
  (`TAIL_BITING|AFFINE_MIN|HAS_RHT`) and refuses unknown bits (fail-fast, like `GgmlType::from_u32`).

> Risk / TODO (post-training hardening): the v2 writer (`write_strand_v2`) is the one *new* piece of
> `strand-quant` code this plan needs. It is pure CPU/Rust, cheap, and must ship with a
> round-trip test `read_strand_v2(write_strand_v2(enc)) == enc` and a cross-check that
> `decode_tensor_fixed` over the v2-reconstructed blocks bit-matches v1. This is independent of the
> live training sweep.

---

## 2. `tools/strand_bake` — safetensors/GGUF → `.strand` v2 (mirrors `tools/awq_bake`)

A new offline workspace tool, modeled 1:1 on `tools/awq_bake/` (which dequantizes Q4_K from a GGUF,
transforms, re-quantizes, and emits a `.dismantle` sidecar). Add `tools/strand_bake` to the
workspace `members` list in the root `Cargo.toml`.

```
tools/strand_bake/
  Cargo.toml      # name = "strand_bake_tool"; [[bin]] name = "strand_bake"
                  # deps: anyhow, serde, serde_json, sha2 (workspace) +
                  #       strand-quant { path = "../../crates/strand-quant" }
                  #       dismantle-core { path = "../../crates/dismantle-core" }  (for GgufFile)
  src/main.rs
```

CLI (mirrors `awq_bake_sidecar <input.gguf> <smoothing.json> <output.dismantle>`):

```
strand_bake <input.{gguf|safetensors}> <output.strand>
            [--bpw 3.34] [--mixed mixed.json] [--tensors blk.*.ffn_down,...]
```

Pipeline per tensor (same four-step shape as `awq_bake/src/main.rs:175–207`):
1. **Source read.** GGUF path: `GgufFile::open` + `tensor_bytes(name)` + `quant::dequant_into` to
   f32 row-major `[rows, cols]` (the exact call `awq_bake` already uses,
   `awq_bake/src/main.rs:181`). safetensors path: read f32/bf16 directly (no dequant). This is the
   *bake-time* float touch — fully offline, never on the decode path.
2. **Pick config.** `TrellisConfig::for_bpw(bpw)` (or `for_bpw_l`) from `strand-quant::trellis`.
   For mixed precision honor a `--mixed` JSON mapping tensor-name → bpw (the deployable lever per
   the settled verdict: 4-bit on attn + `down_proj`, 3-bit elsewhere ⇒ ~3.7 bpw). The JSON shape
   mirrors `awq_bake`'s `smoothing_factors: HashMap<String, Vec<f32>>` but is `bpw: HashMap<String,
   f64>` plus a default.
3. **Encode.** `strand-quant::encode::encode_tensor(&w, &cfg) -> EncodedTensor`. This is where the
   Viterbi/RHT/affine-min machinery runs (all already built and measured: 9.42 PPL @ 3.34 bpw).
4. **Emit v2.** Collect `PackedTensor`s and call `strand-quant::format::write_strand_v2(...)`
   (§1c). Stamp a `src_hash` = first-8-bytes-of-SHA-256 of the source file into the v2 header
   (mirrors `awq_bake`'s `src_hash_from_sha256_first8`, `awq_bake/src/main.rs:107–110`) — this is
   the determinism / staleness key §5 checks at load.

Tensor-name mapping reuses `awq_bake`'s convention: GGUF names are `blk.{i}.{attn_q|attn_k|attn_v|
attn_output|ffn_gate|ffn_up|ffn_down}.weight` (`awq_bake/src/main.rs:55–64`); non-projection tensors
(embeddings, norms, `output.weight`/LM-head, biases) are **passed through untouched** — they are NOT
in the `.strand` file, so at load they fall back to their native GGUF dtype. This keeps STRAND
additive: a `.strand` covering only the FFN/attn projections of a GGUF, with the GGUF still
supplying everything else.

> Decision: `strand_bake` is the STRAND analog of the `awq_bake_sidecar` *standalone binary*, NOT
> the `dismantle bake-sidecar` subcommand. `bake-sidecar` in `crates/dismantle/src/main.rs` (`2091`)
> bakes predec scales *in-engine* and is GGUF-specific; STRAND baking runs the QTIP encoder and is
> better as a separate tool (no Metal/engine dependency, runs on any box, matches how `awq_bake`
> already lives outside the engine). A thin `dismantle bake-strand` subcommand that shells to
> `strand_bake` can be added later if desired, but is not required for the integration.

---

## 3. `strand_trellis_gemv.metal` — placement and decode dispatch (THE GATE)

### 3a. Where the kernel slots

New shader file **`crates/dismantle-core/shaders/strand_trellis_gemv.metal`**, beside the existing
`quant.metal`/`moe.metal`/etc. Wire it in exactly like the others:
1. `metal/mod.rs`: add `pub const SHADER_STRAND_TRELLIS: &str =
   include_str!("../../shaders/strand_trellis_gemv.metal");` next to `SHADER_QUANT` (line 9).
2. `metal/mod.rs::all_shader_sources()`: append `SHADER_STRAND_TRELLIS` to the concatenation
   (`metal/mod.rs:19`). The one runtime `new_library_with_source` (`metal/mod.rs:668`) then compiles
   it into the same `MTLLibrary`; the kernel becomes resolvable by name through
   `MetalContext::pipeline("strand_trellis_gemv")` with **zero** build-system change (no `metallib`
   artifact — the project compiles MSL at runtime, `metal/mod.rs:4–6`).

The kernel name `strand_trellis_gemv` is what the host dispatcher passes as the `KERNEL` const
(§3b), and `library.get_function` looks up (`metal/mod.rs:737`).

### 3b. Kernel design — modeled on `gemm_q4_k_m_fused`, decode fused in the FMA loop

The template is `quant.metal`'s `gemm_q4_k_m_fused` (line 85) — the exact one
`docs/STRAND-metal-decode-gate.md` §"Metal GEMV kernel design" already specs against:

- **One threadgroup per output row.** Buffers `(0)=weights/bits (1)=x (2)=y (3)=rows (4)=cols`,
  plus `(5)=block_table` and `threadgroup(0)=shmem` for the codebook LUT. Same binding convention
  as `gemv_q4k_fast_v1_pinned_tcb` (`kernels/mod.rs:3484–3490`).
- **Random access via the v2 block table.** `row = threadgroup_position_in_grid`; the row's blocks
  are `block_table[row * blocks_per_row .. ]`, each a fixed 32-byte `BlockEntry` (coalesced load).
  Threads split the row's blocks; each thread seeks `bit_offset`, seeds `state = init_state`, and
  walks `n` symbols.
- **Inner loop (the ~9-ops/weight lean count from the gate doc):**
  ```
  cooperatively load the 2^L Q12 codebook LUT into shmem (once per threadgroup)
  for each block assigned to this thread:
     state = entry.init_state;  es = entry.scale_q;  off = entry.min_base_q-derived offset
     for each weight j in block (ALIGNED 32-bit-word reads of k-bit symbols):
        sym = next k bits                  # register-resident, NOT per-weight unaligned read_bits
        state = trellis_step(state, sym)   # the deterministic transition
        q   = LUT[ state-derived index ]   # Q12 quantile (sub-scale folded in at bake → a load)
        w   = (es * q) >> 16               # native 32x32->64 — KEEP i64 product (i32 overflows)
        acc = fma(w_as_float, x[j], acc)
  ```
- **The two real levers (per the 2026-06-08 refinement, `STRAND-metal-decode-gate.md:127–147`),
  not the dropped i32 one:**
  1. **Aligned/vectorized bitstream reads** — unpack `k`-bit symbols from 32-bit words in registers
     (the biggest tax is the *current* per-weight unaligned `read_bits ≈ 5 ops`; killing it is the
     decisive win).
  2. **Per-sub-block scale-fold** — bake the sub-scale into the LUT so reconstruct is a load, and
     keep the native 32×32→64 multiply (the i32 reconstruct is **wrong** — `2^18 × 2^14 = 2^32`
     overflows i32; the i64/native product is load-bearing).

### 3c. The decode-dispatch hook in `gemv_proj!`

`gemv_proj!` (`qwen_dense.rs:5004`) switches on `$tref.dtype` (a `GgmlType`) and resolves
fast/predec side-tables by `$tref.offset`. STRAND is *not* a `GgmlType`, so the hook sits **before**
the `match $tref.dtype` — a per-tensor "is there a STRAND override for this offset?" check that
short-circuits to the trellis kernel, otherwise falls through to the existing GGUF arms unchanged
(this is precisely how the `q4k_fast_ref` arm already pre-empts the base Q4_K kernel,
`qwen_dense.rs:5091–5110`):

```rust
// inside gemv_proj!, before `match $tref.dtype { ... }`
if let Some((strand_buf, tbl_off, bits_off, n_blocks)) = strand_ref
    .and_then(|(buf, map)| map.get(&$tref.offset).map(|m| (buf, m.tbl, m.bits, m.nblk)))
{
    kernels::gemv_strand_trellis_pinned_tcb(
        &mut tcb, strand_buf, tbl_off, bits_off, n_blocks, $rows, $cols, $x, $out,
    )?;
} else { match $tref.dtype { /* ...existing GGUF arms, untouched... */ } }
```

- **Host loader** `ensure_strand_cache(&mut self)` is a near-clone of `ensure_q4k_fast_cache`
  (`qwen_dense.rs:3895`): probe `<gguf>.strand` / `models/<stem>.strand`, `StrandFile::open`, pin
  the whole file as one `PinnedBuffer` via `ctx.new_buffer_with_bytes(&bytes)`, and build
  `strand_offsets: HashMap<usize(src GGUF offset), StrandRef{tbl,bits,nblk}>` by walking the same
  per-layer `proj_to_name` table (`qwen_dense.rs:3962–3970`). New fields `strand_buf:
  Option<PinnedBuffer>` and `strand_offsets: Option<HashMap<...>>` on the model, parallel to
  `q4k_fast_buf`/`q4k_fast_offsets` (`qwen_dense.rs:277–280`).
- **Gate:** `let strand_active = crate::env_on("DISMANTLE_QWEN_STRAND");` near the existing
  `q4k_fast_active` (`qwen_dense.rs:4277`); `ensure_strand_cache()` runs once when the flag is set
  and the buffer is `None`. Default-unset ⇒ no STRAND path touched ⇒ all golden hashes unchanged.
- **New kernel host fn** `kernels::gemv_strand_trellis_pinned_tcb` mirrors
  `gemv_q4k_fast_v1_pinned_tcb` (`kernels/mod.rs:3447`): validates `n_blocks`/byte bounds against
  `strand_buf.length()`, computes `(n_tg, tg_size)`, and calls `tcb.dispatch_threads(
  "strand_trellis_gemv", ...)` binding the 6 buffers.
- **Backend-seam parity (optional, additive).** For callers that go through the typed seam rather
  than the macro, `MetalBackend::gemv` (`backend/metal.rs:181`) gets one arm: `WeightKind::
  StrandTrellis => kernels::gemv_strand_trellis_pinned_tcb(...)`. The existing `metal.rs` GEMV
  byte-offset caveat (its doc item 1) applies identically — STRAND addresses weights as a slice of
  one pinned buffer, which is exactly what the v2 block table encodes.

> **This is the gate, and it is the only real GPU experiment in the whole plan.** Steps §1–§2 and
> §4's harness are CPU/Rust and cheap. §3 must be *measured* on the M3 (next section) before the
> format is declared deployable — `paradigmshift.md` §V.3 asterisk, restated. CPU `decode_lean`
> (the existing `decode.rs` work) is a correctness scaffold, **not** the ridge proof; the ridge
> proof is this kernel + an Instruments bandwidth trace.

---

## 4. Measurement harness — bytes/token, tps, J/token, determinism (reuse dismantle's bench path)

The metric is **bytes/token down ⇒ tps up ⇒ J/token down**, *iff* §3 is bandwidth-bound (`metric =
bytes/token + determinism, NOT PPL`). Reuse dismantle's existing bench surfaces — do not build a new
one:

1. **tps (the headline):** `crates/dismantle-bench` `decode` suite. Run via the engine the same way
   `measure_joules.sh` invokes the locked fast path, A/B on the STRAND env flag:
   - **A (baseline):** existing Q4_K_M GGUF path (`DISMANTLE_QWEN_Q4K_FAST=1`, the shipped config).
   - **B (STRAND):** `DISMANTLE_QWEN_STRAND=1` + a `<model>.strand` baked by §2.
   Compare median `decode_tps` (the suite already reports a sorted median, `suites/decode.rs:51`).
   Use the `competitive` suite (`suites/competitive.rs`) for the three-way vs llama.cpp / MLX
   context.

2. **J/token (the moat axis):** `tools/bench/measure_joules.sh` verbatim, once per arm. It already
   computes `J/tok = avg_power_W * decode_wall_s / tokens` from `macmon` (sudo-free) and is a pure
   measurement tool (no perf change). Pass the STRAND env in `BASE_ENV` for arm B. This is the
   "runs cool / sips power" number `paradigmshift.md` centers.

3. **bytes/token (the *why*):** two complementary reads —
   - **On-disk truth:** the `.strand` file size (its byte length *is* the real bytes/weight — the
     metric STRAND optimizes, per `format.rs:85–88`), vs the Q4_K_M GGUF tensor bytes for the same
     projections. Bake-time, no GPU.
   - **Decode-time truth (the gate):** the **`bandwidth` suite**
     (`crates/dismantle-bench/src/suites/bandwidth.rs`) — **currently a stub** (`phase1_pending:
     true`). Filling it for STRAND is the natural home for the ridge measurement: weight bytes
     moved/token and **% of M3 peak bandwidth** for `strand_trellis_gemv` vs `gemm_q4k_fast_v1`.
     Bandwidth-bound (high % peak, scaling with byte cut) ⇒ thesis holds; compute-bound (flat at
     low % peak, the Q3_K-at-24% failure mode) ⇒ kernel needs the §3b levers before the format
     ships. Cross-check against the existing `crates/strand-decode-kernel/bin/kernel-bench.rs`
     (CPU reference GMAC/s + `footprint_bytes` ratio) for an apples-to-apples byte-traffic sanity
     number.

4. **Instruments / Metal trace:** drive the same `decode` bench under Metal System Trace
   (dismantle's `--trace-dispatch` flag / `BenchOptions.trace_dispatch`, `bench/lib.rs:28`) to read
   the GPU memory-bandwidth counter directly — the decisive bandwidth-bound confirmation the gate
   doc calls for (`STRAND-metal-decode-gate.md:145`).

Report shape (one row per arm, all from the above tools): `{bpw, bytes/token, decode_tps,
avg_power_W, J/token, %peak_bw, golden_hash}`.

---

## 5. Integration sequence + the determinism bit-identity gate

**Sequence (each step independently landable + additive; nothing breaks if a later step is absent):**

1. **`strand-quant` v2 writer/reader** (`format::write_strand_v2` / `read_strand_v2`), block-offset
   table per §1c. Pure CPU. Test: `read_strand_v2(write_strand_v2(enc)) == enc`, and
   `decode_tensor_fixed` over v2-reconstructed blocks bit-matches the v1 path. **No dismantle change
   yet.**
2. **`crates/dismantle-core/src/strand/`** reader (§1) + `WeightKind::StrandTrellis` (§1b). Compiles
   on all targets; no dispatch wired. Test: `StrandFile::open` round-trips a `strand_bake` output;
   `cargo check -p dismantle-core`.
3. **`tools/strand_bake`** (§2). Produces a real `<model>.strand` from the Qwen2.5 GGUF. Verify file
   size = expected bpw; `src_hash` stamped.
4. **`strand_trellis_gemv.metal` + `gemv_strand_trellis_pinned_tcb` + the `gemv_proj!` pre-empt
   arm + `ensure_strand_cache`** (§3). Behind `DISMANTLE_QWEN_STRAND`, default-off.
5. **Harness run** (§4): A/B baseline vs STRAND; fill the `bandwidth` suite; Instruments trace.
   **This is the go/no-go on the format** — if compute-bound, iterate §3b levers (aligned reads,
   scale-fold) before declaring `.strand` v2 the deploy format.

**The determinism / bit-identity gate (the moat — `MOAT = density × determinism × float-free
decode`):**

- **G0 — additive proof (regression).** With `DISMANTLE_QWEN_STRAND` *unset*, every existing
  dismantle golden hash and parity test is byte-identical to pre-integration `main`. This is what
  "never break existing dismantle paths" means, mechanically. (`cargo test -p dismantle-core`
  post-training; do NOT run during the live sweep.)
- **G1 — CPU↔CPU exactness.** `read_strand_v2 ∘ write_strand_v2` reconstructs the *same* integer
  weights as v1 `decode_tensor_fixed` (the float-free decode is a pure integer function of the
  bits, `decode.rs`). Asserted in `strand-quant` tests, no GPU.
- **G2 — GPU↔CPU bit-identity (the headline determinism gate).** The Metal `strand_trellis_gemv`
  output must equal the CPU reference `strand_quant::decode::decode_tensor_fixed` →
  `strand-decode-kernel::matvec` for a fixed `(weights, x)` at a real decode shape (e.g. a Qwen
  `ffn_down` row), to the **bit** on the integer reconstruct and within the kernel's documented
  fp-accumulation tolerance on the final `fma`. This is a new parity test in
  `crates/dismantle-core/tests/` modeled on the existing `q4k_fast_parity.rs` /
  `gemm_q4k...predec_parity.rs` family (which assert the Q4K_FAST kernel is bit-identical to the
  source-Q4_K kernel — the exact precedent, `q4k_fast.rs:54–59`). **The whole "bit-identical on
  phone/WASM/MCU/FPGA" claim reduces to G2 passing.**
- **G3 — staleness fail-fast.** The v2 `src_hash` is checked against the source GGUF/safetensors at
  `ensure_strand_cache` load; mismatch ⇒ hard error (mirrors `sidecar::check_sidecar_compatibility`'s
  `GgufHashMismatch` being fatal, `sidecar.rs:228`, and `GgmlType::from_u32`'s reject-unknown
  discipline). No silent use of stale bits.

**Determinism guarantee that makes G2 achievable:** the reconstruct is integer-only —
`reconstruct_q = (scale_q i64 * quantile_q i64) >> 16`, the LUT is Q12 integer, the trellis
transition is integer state — so CPU and GPU compute the *same* `i32` weight before the single
float multiply-accumulate against `x`. The only permitted float divergence is fp32 `fma` ordering
in the accumulation, which the parity test bounds exactly as the existing predec/fast parity tests
do. Keep the native `32×32→64` multiply on GPU (i32 overflows — `STRAND-metal-decode-gate.md:129`),
or G2 fails on large-magnitude blocks.

---

## 6. Notes / risks for the post-training hardening pass

- **`strand-quant` as a dismantle dependency.** The two repos are separate workspaces today
  (`/Users/scammermike/Downloads/strand` vs `/Users/scammermike/Downloads/dismantle`). The cleanest
  wiring is a path dep `strand-quant = { path = "../../strand/crates/strand-quant" }` (or vendored)
  so the dismantle reader and `strand_bake` share the *one* `format` schema and `decode`
  reference — never a hand-copied parser (that is the drift the Q4K_FAST design avoided by keeping
  `q4k_fast.rs` as the single owner of its layout). Decide vendor-vs-path at hardening; the design
  holds either way.
- **`bandwidth` suite is a stub** (`suites/bandwidth.rs`) — filling it for STRAND is both required
  (§4.3) and the lowest-friction place to host the ridge proof; it currently returns
  `phase1_pending`.
- **v2 writer is the only new `strand-quant` code** and is independent of the live
  `quantize-model` sweep (pure CPU layout; safe to add to source now, compiles via `cargo check -p
  strand-quant`).
- **Mixed-precision JSON** for `strand_bake --mixed` should reuse the settled lever (4-bit attn +
  `down_proj`, 3-bit else ⇒ ~3.7 bpw → ~7.7–8.0 PPL); the tensor-name keys are the GGUF
  `blk.{i}.{site}.weight` names `awq_bake` already maps.
- **Open gate risk (the only make-or-break):** if `strand_trellis_gemv` measures compute-bound on
  the M3 (Apple SIMD model, the §V.3 asterisk), the byte savings do NOT convert to tps/J and the
  format is not yet deployable on Apple — iterate the two §3b levers (aligned 32-bit-word reads,
  per-sub-block scale-fold) and re-measure before shipping `.strand` v2 as the default weight source.
  This is the same trap dismantle's dead Q3_K kernel fell into; the kernel, not the bit-width, is on
  trial.
