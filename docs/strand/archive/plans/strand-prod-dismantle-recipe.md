# STRAND → dismantle integration recipe (production)

**Status:** ready-to-apply recipe. **You apply this in `~/Downloads/dismantle` yourself** — this
doc never builds or edits dismantle. Every change is **additive + default-off**: with
`DISMANTLE_QWEN_STRAND` unset, the dismantle binary compiles and runs byte-identically to current
`main` (all golden hashes unchanged). STRAND is the *second* foreign weight layout beside GGUF,
slotted in through the same four touch-points the Q4K_FAST `.dismantle` sidecar already uses: a
reader module, a baker tool, a Metal GEMV kernel, and one pre-empt arm in the `gemv_proj!` macro.

This recipe supersedes the stale parts of `docs/STRAND-dismantle-integration.md`. That earlier doc
was written before `strand-quant::format` v2 existed and **drifted** from what was actually built.
The reconciliations below are load-bearing — the wire format has exactly ONE owner now
(`strand-quant::format`), and several structs in the prior wave's scaffold contradict it.

All symbols/line numbers cited were read in both repos on 2026-06-09.

---

## 0. The single source of truth, and the three structs that get confused

The `.strand` v2 wire format is owned by **`crates/strand-quant/src/format.rs`** (read in full).
The canonical lean parser external consumers MUST delegate to is
`format::read_strand_v2_header(&[u8]) -> Result<StrandV2Header, String>` (format.rs:556). Everything
below is keyed off the **real** v2 layout, which differs from the prior integration doc:

| field | prior doc (`STRAND-dismantle-integration.md` §1c) — STALE | **actual `format.rs` (authoritative)** |
|---|---|---|
| magic | `b"STRQ"` version=2 | `MAGIC_V2 = b"STR2"`, `VERSION_V2 = 2` (format.rs:222-224) |
| page | "16 KB page-aligned" | `PAGE = 4096` (format.rs:227) |
| file header | `magic/ver/n_tensors` only | fixed **56 B**: magic[4], ver u32, header_bytes u32, n_tensors u32, **flags u32**, **source_sha256[32]**, reserved u32 (format.rs:350-360) |
| per-block record | 32-byte `BlockEntry {bit_offset u64, init_state u32, scale_q i32, min_base_q i32, n u32, sub_off u32}` | **16-byte** `BlockOffsetRecord {bit_offset u64, init_state u32, scale_q i32}` (`SIZE = 16`, format.rs:240-259). `min_base_q`/`n`/`sub_scales` are **NOT** in the table — they are side-info / derived |
| regions | `block_table` then `sub_arena` then `bits` | per tensor, each page-aligned: **(a) table**, **(b) payload** (= v1 `enc.bits` verbatim), **(c) sideinfo** (sub_scales for all blocks, then if affine-min: 4-aligned per-block `min_base_q` i32s + packed mins) (format.rs:428-498) |
| ragged | not addressed | `flags_v2::ALL_STRICT` bit0 set iff every 2-D tensor has `in_features % block_len == 0`; `strict=false` emits RAGGED with the flag clear (format.rs:330-346) |

**Three different structs, named precisely (this is the crux of part (c)):**

1. **On-disk record — `strand_quant::format::BlockOffsetRecord`, 16 B.** What is in the file. Owned
   by strand-quant. Read only via `read_strand_v2_header`.
2. **GPU-side record — `BlockEntry`, 80 B, `#[repr(C)]`.** Defined in
   `crates/strand-decode-kernel/shaders/strand_trellis_gemv.metal:58-66` and documented in
   `crates/strand-decode-kernel/shaders/README.md:59-71`. This is **built by the host loader at
   upload time**, NOT read from disk. It expands `eff[8]` (pre-folded effective sub-scales, Q16)
   from `scale_q` + the side-info sub-scales, bakes `init_state` unconditionally, and rewrites the
   tensor-relative `bit_offset` to absolute within the row blob. `static_assert(sizeof==80)`.
3. **STALE record — the hand-rolled 32-byte `BlockEntry` in the orphaned
   `crates/dismantle-core/src/strand/reader.rs:161-207`.** It has `min_base_q`/`n`/`sub_off` fields,
   declares `STRAND_BLOCK_ENTRY_STRIDE = 16` while its struct is 32 B (an internal contradiction),
   and re-implements the parser. It contradicts BOTH (1) and (2). **Delete it** (part (c) below).

The file stays dense (16 B/block on disk); the kernel still gets the fat 80 B record it wants
because the loader expands it once per load. Disk size delta vs v1 is just the 16 B table per block
(≈0.5 bpw at 256-weight blocks → ~0.0005 bpw amortized).

---

## (a) Dependency decision: dismantle DEPS `strand-quant` (path), not vendor

**Recommendation: add `strand-quant` as a path dependency of `dismantle-core` AND of
`tools/strand_bake`. Do NOT vendor a copy of the reader, and do NOT depend on
`strand-decode-kernel`.**

Reasons:

- **Single owner of the wire format.** `read_strand_v2_header` (format.rs:556) is explicitly
  written as "the canonical lean parser every external consumer delegates to, so the `.strand` v2
  wire format has exactly ONE owner." Vendoring re-introduces exactly the drift this recipe is
  fixing — the prior doc's stale 32 B `BlockEntry`/16 KB-page sketch is what happens when a second
  copy of the schema exists. The Q4K_FAST precedent keeps `q4k_fast.rs` the *single* owner of its
  layout for the same reason (`q4k_fast.rs:54-59` parity contract).
- **The encoder and loader can never drift.** `tools/strand_bake` already calls
  `strand_quant::encode::encode_tensor` + `format::write_strand_v2`; if dismantle-core parses with a
  different copy, a writer change silently breaks the loader. One dep = compile-time coupling.
- **The reconstruct math is load-bearing and must not be forked.** `reconstruct_q` is the i64
  `(scale_q*quantile_q)>>16` native 32×32→64 product (decode.rs:37); the doc-comment in
  reader.rs:373 already warns "do NOT fork the reconstruct math (i32 overflows)." The CPU parity
  reference (G2 below) must call `strand_quant::decode::decode_tensor_fixed` / `decode_lean`
  directly.
- **Why `strand-quant`, not `strand-decode-kernel`.** `strand-decode-kernel`
  (`crates/strand-decode-kernel/src/lib.rs`) is the *CPU* reference runtime (`decode_weights_q12`,
  `matvec`, `footprint_bytes`) and itself just deps `strand-quant`. dismantle's production decode is
  the **Metal** kernel; it does not need the CPU matvec on the hot path. dismantle-core depends on
  `strand-quant` (format + decode reference for parity tests); the **dev-dependency**
  `strand-decode-kernel` is optional and only useful if you want its `matvec` as a second
  cross-check in the G2 parity test. Keep the production dep surface to `strand-quant` alone.

Path-dep mechanics (the two workspaces are siblings under `~/Downloads/`):

- `crates/dismantle-core/Cargo.toml` — add under `[dependencies]`:
  ```toml
  strand-quant = { path = "../../../strand/crates/strand-quant" }
  ```
  (climb `dismantle-core` → `crates` → `dismantle` → `Downloads`, then into `strand/crates/`.)
- `tools/strand_bake/Cargo.toml` — uncomment the already-present line (strand_bake/Cargo.toml:40),
  which is the exact same path from one level deeper:
  ```toml
  strand-quant = { path = "../../../strand/crates/strand-quant" }
  ```

**Caveat to verify at apply time (cheap, do it first):** `strand-quant` pulls in `strand-gguf` +
`wide` (strand-quant/Cargo.toml) and, on macOS, `metal 0.27` + `objc 0.2`. dismantle pins
`metal 0.29` / `objc2`. Cargo allows two `metal` major-0 minors to coexist (they are distinct
crates as far as resolution is concerned), but it inflates the build. If that is unwanted, gate
strand-quant's macOS metal/cloud-gpu deps so the *format + decode* path (which is what dismantle needs)
compiles without them — those deps exist for strand-quant's own GPU Viterbi *encoder*, which
dismantle never invokes. A minimal `default-features` split in strand-quant (feature-gate the
`[target.'cfg(target_os = "macos")'.dependencies] metal`/`objc`) is the clean fix; out of scope for
this additive recipe but flag it. Confirm with `cargo tree -p dismantle-core -i metal` after wiring.

---

## (b) The exact diffs (all in `~/Downloads/dismantle`, all additive)

### b1. `crates/dismantle-core/src/lib.rs` — register the module

Add next to `pub mod gguf;` (lib.rs:5). The module is platform-neutral (pure byte-parse +
`memmap2`), so it compiles on every target:
```rust
pub mod strand;
```

### b2. `crates/dismantle-core/src/backend/mod.rs` — `WeightKind::StrandTrellis`

Add ONE variant to the **open** `WeightKind` enum (backend/mod.rs:112-128) — NOT to the closed
`GgmlType` (gguf/reader.rs `from_u32` rejects unknown tags; a synthetic ggml tag would be a lie
about the container and could collide with a future upstream type). `WeightKind` already carries
`Q4kFast` for exactly this "foreign layout, same GEMV verb" role:
```rust
    /// STRAND trellis-coded weights (.strand v2): k-bit Viterbi index stream +
    /// per-block {scale_q, sub_scales, init_state} side table. Integer, float-free
    /// decode (reconstruct_q = (scale_q*quantile_q)>>16). Distinct kernel family;
    /// self-gates on whether `ensure_strand_cache` found a sidecar, like Q4kFast.
    StrandTrellis,
```
The logical op stays `Op::Gemv` — no new verb, no `BackendGemv::gemv` signature change. `supports()`
is per-*op*, so it still returns `true` for `Op::Gemv`; STRAND capability is expressed by whether a
sidecar loaded, identical to Q4K_FAST. **Matching arm in `backend/metal.rs`:** `MetalBackend::gemv`
switches on `spec.weight` at metal.rs:195 with `WeightKind::Q4K | WeightKind::Q4kFast =>` at
metal.rs:205. Add an additive arm:
```rust
    WeightKind::StrandTrellis => kernels::gemv_strand_trellis_pinned_tcb(/* see b5 */)?,
```
**This `metal.rs` arm is OPTIONAL / partial.** The arm at metal.rs:195-215 carries a documented
"SEAM GAP" (metal.rs:200-208): `GemvSpec` gives one `weight` buffer with **no offset and no sidecar
table**, which is why even `Q4kFast`'s predec/fast variants fall through to the base kernel there.
STRAND needs the per-tensor `StrandTensorPin` (table offset, n_blocks, rht_seed) which `GemvSpec`
cannot carry today — so the **real integration is the `gemv_proj!` macro path** (b5), and this
`metal.rs` arm is best left returning the same not-yet-wired behavior or a clear `Error::Metal`
("StrandTrellis requires the gemv_proj! pin path") until `GemvSpec` grows a sidecar handle. Add the
variant so the `match spec.weight` stays exhaustive; do not claim it as the production path.

### b3. `crates/dismantle-core/src/model/qwen_dense.rs` — fields + cache loader

**New fields** on the model struct, parallel to `q4k_fast_buf`/`q4k_fast_offsets`
(qwen_dense.rs:277-280). The value type carries the per-tensor v2 geometry the dispatcher needs:
```rust
    /// .strand v2 deploy file pinned whole as one MTLBuffer (DISMANTLE_QWEN_STRAND).
    /// `strand_offsets` maps a GGUF source offset (the key the gemv_proj! macro has
    /// from `$tref.offset`) → that tensor's v2 region geometry inside the pinned blob.
    #[cfg(target_os = "macos")]
    pub(crate) strand_buf: Option<crate::metal::PinnedBuffer>,
    #[cfg(target_os = "macos")]
    pub(crate) strand_offsets: Option<std::collections::HashMap<usize, StrandTensorPin>>,
```
with a small plain struct (define near the field, or in `strand/mod.rs` and re-export):
```rust
/// Per-tensor v2 geometry, resolved at load, that the GEMV host fn binds.
#[derive(Clone, Copy)]
pub(crate) struct StrandTensorPin {
    pub bits_offset: usize,   // ABS byte offset of payload in the pinned blob (page-aligned)
    pub bits_len: usize,
    pub gpu_table_offset: usize, // ABS byte offset of the loader-built 80 B BlockEntry[] (see b4)
    pub n_blocks: usize,
    pub rows: usize,
    pub cols: usize,
    pub rht_seed: u64,        // host applies RHT(x) once per GEMV (see b5)
    pub k_bits: u8,
    pub l_bits: u8,
}
```
Initialize both new fields to `None` at the two struct-construction sites that set
`q4k_fast_buf: None` (qwen_dense.rs:1284-1286).

**`ensure_strand_cache(&mut self)`** — a near-clone of `ensure_q4k_fast_cache`
(qwen_dense.rs:3895-3983), `#[cfg(target_os = "macos")]`. Differences from the Q4K_FAST clone:
1. Probe `<weights>.strand` and `models/<stem>.strand` (drop the AWQ candidate branch — STRAND has
   no AWQ variant). Return `Ok(())` (feature off) if none exists, exactly like
   q4k_fast (qwen_dense.rs:3943).
2. Parse via the **canonical** delegate, not a hand-rolled header:
   ```rust
   let bytes = std::fs::read(&sidecar_path)?;
   let hdr = strand_quant::format::read_strand_v2_header(&bytes)
       .map_err(|e| Error::Model(format!("parse .strand v2: {e}")))?;
   ```
3. **G3 staleness gate (see (c) and §G).** Compare `hdr.source_sha256` (the full 32-byte digest in
   the v2 file) against `sha2::Sha256::digest(mmap_of_source_gguf)`; on mismatch return a hard
   `Error::Model` (mirrors `sidecar::check_sidecar_compatibility` → `GgufHashMismatch` being fatal,
   sidecar.rs:217/246).
4. **Build the GPU-side 80 B `BlockEntry[]` per tensor and the pinned blob.** This is the loader
   expansion that part (c)/the shader README:53-57 describe. For each tensor in `hdr.tensors`:
   reconstruct `eff[s] = eff_scale_q(record.scale_q, sub_scale_code[s])` for the 8 sub-blocks
   (call into strand-quant's fold so the math isn't forked — expose
   `decode::eff_scale_q` / the sub-scale unpack as `pub` if not already; see (c) note), copy
   `init_state` unconditionally, and convert `bit_offset` to absolute. Concatenate, per tensor:
   `[payload bytes (page-aligned)] [BlockEntry[n_blocks] (page-aligned)]` into one `Vec<u8>`, record
   each tensor's `StrandTensorPin`, then pin once with `ctx.new_buffer_with_bytes(&blob)`
   (qwen_dense.rs:3979). Map by **GGUF source offset** using the same per-layer `proj_to_name` walk
   as q4k_fast (qwen_dense.rs:3962-3976) so `strand_offsets` is keyed by `$tref.offset`.
   `rht_seed`/`k_bits`/`l_bits` come straight off `hdr.tensors[i]`.

> Note: the 80 B `BlockEntry` is built host-side because the kernel wants pre-folded `eff[8]` and
> absolute offsets (README:80-86). At the **3-bit deploy point `has_affine_min == false`**, so no
> `off[8]` is needed; if a 4-bit `--mixed` tensor is ever baked, the loader must also expand
> `off[s] = eff_min_q(...)` (decode.rs:80) into an extended record and the kernel must add it — this
> is the README:83-86 hardening flag, carry it as a `debug_assert!(!has_affine_min)` for now.

### b4. Where the gate lives in the forward pass (lazy-load + ref bind)

Next to the `q4k_fast_active` block (qwen_dense.rs:4277-4280):
```rust
    let strand_active = crate::env_on("DISMANTLE_QWEN_STRAND");
    if strand_active && self.strand_buf.is_none() {
        self.ensure_strand_cache()?;
    }
```
Then bind an immutable ref for the macro body, next to where `q4k_fast_ref` is bound
(qwen_dense.rs:4440-4448), AFTER the `ensure_*` mutable-borrow calls:
```rust
    let strand_ref = if strand_active {
        self.strand_buf.as_ref().zip(self.strand_offsets.as_ref())
    } else {
        None
    };
```
**Mutual exclusion (additive guards, same style as the F16_KV/INT4_KV refusals
qwen_dense.rs:4311-4373):** STRAND replaces the Q4_K weight bytes entirely, so it is incompatible
with the predec / q4k_fast / W4A8 / AWQ levers that read or re-decode Q4_K. Add early:
```rust
    if strand_active && (predec_active || q4k_fast_active || w4a8_active_early || awq_active_early) {
        return Err(Error::Model(
            "DISMANTLE_QWEN_STRAND=1 is incompatible with Q4K_PREDEC/Q4K_FAST/W4A8/AWQ \
             (STRAND replaces the Q4_K weight stream); unset the others".into(),
        ));
    }
```

### b5. The `gemv_proj!` pre-empt arm

`gemv_proj!` (qwen_dense.rs:5004-5153) dispatches on `$tref.dtype` — a **`GgmlType`**. STRAND is not
a `GgmlType`, so the override must sit **before** the `match $tref.dtype`, short-circuiting to the
trellis kernel and otherwise falling through to the untouched GGUF ladder. This is exactly how the
`q4k_fast_ref` arm pre-empts the base Q4_K kernel inside that match (qwen_dense.rs:5091-5110), only
hoisted one level out because the key is "is this offset STRAND?" rather than a dtype:
```rust
    macro_rules! gemv_proj {
        ($site_w4a8:expr, $tref:expr, $pinned_f16:expr, $rows:expr, $cols:expr,
         $x:expr, $x_i8:expr, $x_sc:expr, $out:expr) => {{
            // STRAND pre-empt: a .strand v2 override for this tensor's GGUF offset
            // short-circuits the whole GgmlType ladder. Default-off ⇒ strand_ref is
            // None ⇒ this arm vanishes and the macro is byte-identical to today.
            if let Some(pin) = strand_ref
                .and_then(|(buf, map)| map.get(&$tref.offset).copied().map(|p| (buf, p)))
            {
                let (strand_buf, p) = pin;
                kernels::gemv_strand_trellis_pinned_tcb(
                    &mut tcb,
                    strand_buf,
                    &p,            // StrandTensorPin: bits/table offsets, n_blocks, k/l, rht_seed
                    $x,            // host RHTs this once inside the host fn (see RHT note)
                    $out,
                )?;
            } else {
                match $tref.dtype {
                    /* ...every existing GGUF arm, COMPLETELY UNCHANGED (5007-5151)... */
                }
            }
        }};
    }
```
Note the macro signature and all call sites (qwen_dense.rs:5374/5635/5972/6255 etc.) are untouched —
the override is keyed off `$tref.offset`/`$rows`/`$cols` it already receives.

**New kernel host fn** `kernels::gemv_strand_trellis_pinned_tcb` — a sibling of
`gemv_q4k_fast_v1_pinned_tcb` (kernels/mod.rs:3447-3491). Skeleton:
```rust
    pub fn gemv_strand_trellis_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        strand_buf: &PinnedBuffer,
        pin: &crate::model::qwen_dense::StrandTensorPin,
        x_buf: &PinnedBuffer,   // f32 activation; host applies RHT(x) → x_rht below
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "strand_trellis_gemv";
        if pin.cols % 256 != 0 { /* Err: STRICT deploy invariant, README:109 */ }
        // bounds: pin.bits_offset + pin.bits_len  and the table region must be
        //         within strand_buf.length() (mirror kernels/mod.rs:3473-3478).
        // RHT: x_rht = rht_forward(x, RhtConfig::from_seed(pin.rht_seed)) ONCE per GEMV,
        //      256-wide block, row-restart (README:90-110). This is FLOAT preprocessing,
        //      same status as the q·(1/4096)·x MAC — NOT covered by the integer-decode
        //      guarantee. Write x_rht into a scratch PinnedBuffer; bind it as buffer(1).
        const TG: u32 = 256;
        let n_tg = (pin.rows as u32).div_ceil(1); // one TG per row (README:30)
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(strand_buf), pin.bits_offset as u64); // w_bits
            enc.set_buffer(1, Some(&x_rht_buf), 0);                      // x_rht
            enc.set_buffer(2, Some(out_buf), 0);                         // y
            enc.set_u32(3, pin.rows as u32);
            enc.set_u32(4, pin.cols as u32);
            enc.set_buffer(5, Some(strand_buf), pin.gpu_table_offset as u64); // BlockEntry[]
            enc.set_u32(6, pin.k_bits as u32);
            enc.set_u32(7, pin.l_bits as u32);
            enc.set_buffer(8, Some(&lut_buf), 0); // 2^L Q12 codebook, codebook_lut(l_bits)
        })
    }
```
Buffer indices match `strand_trellis_gemv.metal` (README:34-44). The `lut_buf` (frozen Q12 codebook,
`strand_quant::codebook::codebook_lut(l_bits)`) and the `x_rht_buf` scratch are pinned once and held
on the model alongside `strand_buf` — add them as sibling `Option<PinnedBuffer>` fields built in
`ensure_strand_cache`. **The RHT seed/segmentation must match the encoder EXACTLY (256-wide,
row-restart, `RhtConfig::from_seed`); a block-size mismatch silently corrupts `y` — README:108-110
calls this the single most likely integration bug.**

### b6. Wire `strand_trellis_gemv.metal` into the shader library

The kernel source lives in strand (`crates/strand-decode-kernel/shaders/strand_trellis_gemv.metal`).
**Copy it into dismantle** at `crates/dismantle-core/shaders/strand_trellis_gemv.metal` (the shader
is `include_str!`'d, so it must be inside dismantle-core's source tree — it cannot be an external
path). Then two one-liners in `crates/dismantle-core/src/metal/mod.rs`:
1. Next to `SHADER_QUANT` (metal/mod.rs:9):
   ```rust
   pub const SHADER_STRAND_TRELLIS: &str =
       include_str!("../../shaders/strand_trellis_gemv.metal");
   ```
2. Append it to the `all_shader_sources()` array (metal/mod.rs:19-31):
   ```rust
       SHADER_STRAND_TRELLIS,
   ```
The single runtime `new_library_with_source` (metal/mod.rs:666-669) then compiles it into the same
`MTLLibrary`; `MetalContext::pipeline("strand_trellis_gemv")` (metal/mod.rs:727-744) resolves it by
name with **zero** build-system change (no `metallib` artifact — MSL compiles at runtime).

> Drift note: `all_shader_sources()` is hashed into the profile signature
> (`profile.rs:633 short_hash(metal::all_shader_sources())`). Appending a new shader **changes that
> hash**. That hash is a profile/cache key, not a golden-output hash — verify it is not asserted in a
> golden test; if it is, the change is still additive (new kernel source) but the expected hash
> constant must be updated in the same commit. Confirm with
> `grep -rn "all_shader_sources\|short_hash" crates/dismantle-core/src` before applying.

---

## (c) Reconcile the orphaned `crates/dismantle-core/src/strand/reader.rs`

The scaffold `reader.rs` (read in full) is a **hand-rolled v2 parser** with a stale schema. It must
become a thin delegator to `strand_quant::format::read_strand_v2_header`. Concretely:

**Delete (these contradict the single owner):**
- The local `Cursor` primitive reader (reader.rs:405-437).
- `StrandFile::from_mmap`'s hand parse (reader.rs:235-330) — magic/version/flag re-checks, the
  per-tensor descriptor walk, `check_region`/`table_bytes`.
- The local constants `STRAND_MAGIC`/`STRAND_VERSION_V2`/`STRAND_BLOCK_ENTRY_STRIDE`
  (reader.rs:74-83) and the `mod flags` mirror (reader.rs:88-97) — these duplicate
  `format::MAGIC_V2`/`VERSION_V2`/`BlockOffsetRecord::SIZE`/`format::flags_v2`.
- **The 32-byte `BlockEntry` struct + `read_at` (reader.rs:154-207).** This is the STALE struct from
  §0. Its `STRIDE` is declared 16 while its body is 32 B; its `min_base_q`/`n`/`sub_off` fields are
  NOT on disk. Replace all its uses with `format::BlockOffsetRecord` (16 B). The **80 B** GPU
  `BlockEntry` is a separate concern that lives in `ensure_strand_cache` (b3) + the .metal file — it
  is built at upload, never parsed from disk, so it does NOT belong in `reader.rs`.

**Keep/rewrite (the mmap + zero-copy surface, mirroring `gguf/reader.rs`):**
- `StrandFile { mmap, version, tensors: HashMap<String, StrandTensorInfo>, tensor_order }` and
  `StrandFile::open` (reader.rs:209-227) — but `from_mmap` now does:
  ```rust
  pub fn from_mmap(mmap: Mmap) -> Result<Self> {
      let hdr = strand_quant::format::read_strand_v2_header(&mmap[..])
          .map_err(|e| Error::Gguf(format!("strand: {e}")))?;
      // map hdr.tensors -> StrandTensorInfo (offsets are ABS file bytes, like
      // gguf::TensorInfo::data_offset). bits_offset = h.payload_offset, etc.
      ...
  }
  ```
- `StrandTensorInfo` (reader.rs:118-143) — keep the shape, drop fields that don't come from the
  canonical header; populate from `TensorHeaderV2` (format.rs:512-532): `payload_offset`/
  `payload_bytes` → `bits_offset`/`bits_len`, `table_offset`/`n_blocks`, `sideinfo_offset`/
  `sideinfo_bytes` (the loader needs sideinfo to expand `eff[8]`), plus `k_bits`/`l_bits`/`vec_dim`/
  `rht_seed`/`block_len`/`tail_biting`/`has_affine_min`.
- `tensor_bits(name)` / `block_table(name)` zero-copy slices (reader.rs:339-353) — keep; they slice
  `mmap` at the canonical offsets. (`block_table` now returns the on-disk 16 B-record region.)
- `StrandQuant::Trellis` (reader.rs:107-113) — keep as the decode-family tag, mapped to
  `WeightKind::StrandTrellis` at the call site.

**Update the module docs** (reader.rs:1-64, mod.rs:1-29) to point at the real schema (`STR2`, 16 B
`BlockOffsetRecord`, 4096-byte pages, separate payload/sideinfo regions) and to state plainly: *this
reader delegates all parsing to `strand_quant::format::read_strand_v2_header`; the 80 B GPU
`BlockEntry` is built by `ensure_strand_cache`, not here.* Delete the stale ASCII sketch
(reader.rs:22-44) and the obsolete "wiring step" TODO list since this recipe is that wiring.

**strand-quant fold helpers — already `pub`, no change needed (verified).** The 80 B-record build
in `ensure_strand_cache` needs the effective-scale fold, and all the helpers are already public:
`decode::eff_scale_q(scale_q, code)` (decode.rs:59), `decode::eff_min_q` (decode.rs:80, for the
4-bit affine path), `encode::unpack_sub_scales` (encode.rs:140), `encode::SUB_BLOCK` (encode.rs:77),
and `codebook::codebook_lut(l_bits)` (codebook.rs:481). So the loader can call into strand-quant for
the fold without forking the math, and **step 1 below is a verify, not an edit** (these were public
as of 2026-06-09).

---

## Ordered apply plan (each step independently landable; nothing breaks if a later step is absent)

> Build discipline while the 7B sweep is LIVE: only `cargo check -p strand-quant` (light, no release
> binary) and `cargo test -p strand-decode-kernel` (separate crate) are safe in the strand repo.
> Do NOT `cargo build --release` / `cargo build -p strand-quant` / `cargo test -p strand-quant`.
> Do NOT build/test anything in dismantle while the sweep runs. The dismantle steps below are
> authored now, **built/tested by you post-sweep.**

1. **strand-quant: expose the fold helpers `pub`** (part (c) tail) — `eff_scale_q`,
   `unpack_sub_scales`/`SUB_BLOCK`, `codebook_lut`. Verify `cargo check -p strand-quant`. (The v2
   writer/reader/header already exist and are tested — format.rs:994-1185.) **Sweep-safe.**
2. **dismantle deps** (part (a)) — add the `strand-quant` path dep to `dismantle-core/Cargo.toml`
   and uncomment it in `tools/strand_bake/Cargo.toml`; run `cargo tree -p dismantle-core -i metal`
   to confirm the metal-version coexistence is acceptable (else feature-gate strand-quant's macOS
   deps — part (a) caveat). *Post-sweep.*
3. **`lib.rs` + `backend/mod.rs`** (b1, b2) — `pub mod strand;` and `WeightKind::StrandTrellis` +
   the `metal.rs` arm. Compiles on all targets; no dispatch wired yet. *Post-sweep:*
   `cargo check -p dismantle-core`.
4. **Reconcile `strand/reader.rs`** (part (c)) — delete the stale parser/struct, delegate to
   `read_strand_v2_header`, rewrite `StrandTensorInfo` from `TensorHeaderV2`. The reader's own smoke
   test (reader.rs:462) must be rewritten to build its fixture via
   `format::write_strand_v2(&[..], sha, true)` instead of hand-laying bytes. *Post-sweep.*
5. **`tools/strand_bake`** — flip the `STRAND_BAKE_TODO` markers (strand_bake/src/main.rs:227-243) to
   real `encode_tensor` + `write_strand_v2` calls. **Fix the `src_hash` mismatch:** the scaffold
   computes a `u64` via `from_be_bytes` (strand_bake/src/main.rs:141), but `write_strand_v2` takes
   `source_sha256: [u8; 32]` (format.rs:324) — pass the **full 32-byte `Sha256::digest`**, not a
   u64, and have `ensure_strand_cache` compare the full digest (G3). Produce a real `<model>.strand`
   from the Qwen2.5 GGUF; assert file size ≈ expected bpw. *Post-sweep.*
6. **Metal kernel + host fn + fields + cache + gemv arm** (b3-b6) — copy the `.metal` into
   dismantle-core/shaders, wire `all_shader_sources()`, add the model fields + `ensure_strand_cache`
   + `gemv_strand_trellis_pinned_tcb` + the `gemv_proj!` pre-empt + the mutual-exclusion guard.
   Behind `DISMANTLE_QWEN_STRAND`, default-off. *Post-sweep.*
7. **Gates G0-G3 + harness** (§ below). The Metal decode gate (G2 + bandwidth %peak) is the
   **go/no-go** on the format; if compute-bound, iterate the README:188-199 levers before declaring
   `.strand` v2 the deploy format. *Post-sweep, GPU must be free.*

---

## Determinism / bit-identity gates (the moat = density × determinism × float-free decode)

- **G0 — additive proof.** With `DISMANTLE_QWEN_STRAND` unset, every dismantle golden hash + parity
  test is byte-identical to current `main`. The pre-empt arm vanishes (`strand_ref == None`), the
  new `WeightKind` arm is never reached, the only always-on change is one extra shader in the
  library compile (and its profile-hash key — verify per b6). `cargo test -p dismantle-core`
  *post-sweep*.
- **G1 — CPU↔CPU (already green in strand-quant).** `read_strand_v2(write_strand_v2(enc))`
  reconstructs the same `EncodedTensor` as v1 and `decode_tensor_fixed` is bit-identical — proven by
  `strand_v2_round_trip_matches_v1_q12` and `strand_v2_header_matches_full_read` (format.rs:1028,
  1098). No new work.
- **G2 — GPU↔CPU bit-identity (the headline gate).** New parity test in
  `crates/dismantle-core/tests/`, modeled on the Q4K_FAST parity family (`q4k_fast.rs:54-59`
  contract): bake ONE Qwen `ffn_down`-shape tensor to v2, dispatch `strand_trellis_gemv`, and assert
  the decoded Q12 weights equal `strand_quant::decode::decode_tensor_fixed` / `decode_lean`
  **bit-for-bit** on the integer reconstruct, and `y` matches `strand_decode_kernel::matvec` within
  the documented fp-accumulation tolerance. **The entire "bit-identical on phone/WASM/MCU/FPGA"
  claim reduces to G2 passing.** Keep the native 32×32→64 product on GPU — i32 overflows
  (`Q16·Q12=Q28`, README:139-140), G2 fails on large-magnitude blocks otherwise.
- **G3 — staleness fail-fast.** `ensure_strand_cache` compares the v2 `source_sha256` (32 B) against
  the source GGUF's `Sha256::digest`; mismatch ⇒ hard `Error::Model` (mirrors `GgufHashMismatch`
  fatal, sidecar.rs:217/246). No silent use of stale bits.

**Harness (reuse dismantle's bench; build nothing new):** A/B `decode` suite (baseline
`DISMANTLE_QWEN_Q4K_FAST=1` vs B `DISMANTLE_QWEN_STRAND=1` + `<model>.strand`), `measure_joules.sh`
for J/tok, and **fill the stubbed `bandwidth` suite** (`suites/bandwidth.rs`, `phase1_pending`) with
weight-bytes/token + **% of measured M3 peak bandwidth** for `strand_trellis_gemv` vs
`gemm_q4k_fast_v1`. Bandwidth-bound + byte win realized ⇒ ship; compute-bound at low %peak ⇒ the
Q3_K trap, iterate the kernel levers first.

---

## Risks (ranked)

1. **The decode gate (make-or-break).** `strand_trellis_gemv` may measure **compute-bound** on M3
   (the serial state-walk + symbol-pop, the same wall dismantle's dead Q3_K kernel hit at 24% peak).
   Then STRAND's ~26% byte cut does NOT convert to tps/J and `.strand` v2 is not Apple-deployable as
   a *speed* play — the moat stands on determinism + on-device-fit alone. Measure (G2 + bandwidth
   %peak) before declaring deploy; iterate README:188-199 levers (aligned `uint*` reads, 2 syms/iter,
   predec variant, LUT in constant memory).
2. **RHT segmentation mismatch.** The host RHT(x) MUST be 256-wide, row-restart, `from_seed(rht_seed)`
   — byte-for-byte the encoder's segmentation (README:108-110). Any mismatch silently corrupts `y`.
   Single most likely integration bug. Assert `in_features % 256 == 0` (true for all Qwen2.5-7B
   tensors).
3. **`metal` crate version skew.** strand-quant pins `metal 0.27`/`objc 0.2`; dismantle `0.29`/`objc2`.
   They can coexist but bloat the build; feature-gate strand-quant's macOS encoder deps so dismantle
   pulls only `format`+`decode` (part (a) caveat). Verify with `cargo tree -i metal`.
4. **`src_hash` type + endianness mismatch.** The strand_bake scaffold's `from_be_bytes` u64
   (strand_bake/src/main.rs:141) is wrong for the v2 header which stores a full 32-byte
   `source_sha256` (format.rs:324). Fix in step 5 or G3 silently never triggers / always triggers.
5. **80 B GPU record vs 16 B disk record confusion.** The loader (b3) MUST expand the 16 B
   `BlockOffsetRecord` + side-info sub-scales into the 80 B `BlockEntry` at upload; do NOT try to
   read 80 B from disk (the stale reader.rs struct, part (c)) or bind the 16 B record to the kernel.
   `static_assert(sizeof(BlockEntry)==80)` on the host record builder.
6. **affine-min at 4-bit.** The shipped 3-bit point has `has_affine_min == false`; the kernel + 80 B
   record omit the offset path. A 4-bit `--mixed` tensor needs `off[8]` expanded in the loader and
   added in the kernel (README:83-86). Guard with `debug_assert!(!has_affine_min)` until implemented.
7. **profile-hash key drift.** Appending `SHADER_STRAND_TRELLIS` changes
   `all_shader_sources()`'s hash (profile.rs:633). It's a cache key, not golden output, but if any
   test pins it, bump the constant in the same commit (b6 drift note).
