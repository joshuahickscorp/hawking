# STRAND → dismantle wiring — the executable recipe

**This is the step-by-step, copy-pasteable companion to** `docs/plans/strand-prod-dismantle-recipe.md`
(the design) and `docs/STRAND-dismantle-integration.md` (the original, now-superseded sketch). The
design doc explains *why* each touch-point is shaped the way it is and reconciles the three confused
per-block structs; **this doc is the literal sequence of edits you apply by hand in
`~/Downloads/dismantle`**, ordered so each step compiles before the next. It does not duplicate the
rationale — read the recipe first, then execute here.

**You apply every edit below in `~/Downloads/dismantle` yourself.** This doc never builds or edits
dismantle. Every change is **additive + default-off**: with `DISMANTLE_QWEN_STRAND` unset, the
dismantle binary compiles and runs byte-identically to current `main` (all golden hashes unchanged).
The only always-on change is one extra shader string in the runtime library compile (step 8) — its
only observable effect is the `profile.rs:633` cache-key hash, addressed there.

All line numbers were read in both repos on 2026-06-09 and are cited inline so you can confirm before
each paste. Anchors that shift as you edit are noted as "after step N".

> **Build discipline while the 7B sweep (pid 89192) is LIVE.** In the strand repo only
> `cargo check -p strand-quant` (light, no release binary) and `cargo test -p strand-decode-kernel`
> (a separate crate) are safe. Do **NOT** run `cargo build --release`, `cargo build -p strand-quant`,
> or `cargo test -p strand-quant`. Do **NOT** build or test anything in dismantle. Steps 2–8 are
> *authored now, compiled/tested by you post-sweep* with the exact commands given per step.

> **2026-06-15 update (read first).** `strand-quant` and `strand-decode-kernel` are now **excluded
> from the root workspace** and are self-contained packages. The `cargo … -p <crate>` commands
> throughout this doc no longer resolve — run them as
> `cargo check --manifest-path crates/strand-quant/Cargo.toml` (and likewise for
> `strand-decode-kernel`). dismantle still consumes `strand-quant` by path dependency exactly as
> Step 1 describes. The Step 0 symbol **line numbers are pre-refactor (2026-06-09)** — the helpers
> are all still `pub` but moved files (e.g. `read_strand_v2_header` is now in `sideinfo_wire.rs`,
> `SUB_BLOCK` in `selfdesc.rs`, `codebook_lut` in `codebook.rs`); re-grep by symbol name rather
> than trusting the citations.

---

## Step 0 — verify the strand-quant fold helpers are `pub` (sweep-safe; expected no-op)

The loader in step 6 builds the 80-byte GPU `BlockEntry` by folding sub-scales into `eff[8]` via
strand-quant — do **not** fork that math. All required helpers are already `pub` as of 2026-06-09
(verified):

| symbol | location | role in the wiring |
|---|---|---|
| `format::read_strand_v2_header(&[u8]) -> Result<StrandV2Header, String>` | format.rs:556 | the canonical lean parser; the dismantle reader + loader delegate to it |
| `decode::reconstruct_q(scale_q, quantile_q) -> i32` | decode.rs:37 | the load-bearing i64 `(scale_q*quantile_q)>>16`; the kernel mirrors it, do not re-derive |
| `decode::eff_scale_q(scale_q, code) -> i32` | decode.rs:59 | folds one 6-bit sub-scale code into an effective Q16 scale → one `eff[s]` |
| `decode::eff_min_q(min_base_q, code) -> i32` | decode.rs:80 | 4-bit affine-min offset fold (only if a `--mixed` 4-bit tensor is baked) |
| `encode::unpack_sub_scales(bytes, n) -> Vec<u8>` | encode.rs:140 | unpacks the 6-bit sub-scale codes from a block's side-info slice |
| `encode::SUB_BLOCK` (= 32) | encode.rs:77 | weights per sub-block ⇒ 8 sub-blocks per 256-block ⇒ `eff[8]` |
| `codebook::codebook_lut(l_bits) -> &'static [i32]` | codebook.rs:481 | the frozen 2^L Q12 codebook the kernel binds as buffer(8) |

Confirm (strand repo, sweep-safe):
```bash
cargo check -p strand-quant
```
If any helper were private, make it `pub` (additive, no behavior change) and re-run. Expected: clean,
no edit needed. The v2 writer/reader/header are already implemented and tested
(format.rs:994–1185).

---

## Step 1 — dismantle deps: add the `strand-quant` path dep (part (a))

dismantle becomes a consumer of the single wire-format owner. **Path, not vendor; `strand-quant`,
not `strand-decode-kernel`** (rationale: recipe part (a)).

**1a.** `crates/dismantle-core/Cargo.toml` — add under `[dependencies]`:
```toml
# Single owner of the .strand v2 wire format + the load-bearing i64 reconstruct.
# Path climbs dismantle-core -> crates -> dismantle -> Downloads, into strand/crates/.
strand-quant = { path = "../../../strand/crates/strand-quant" }
```

**1b.** `tools/strand_bake/Cargo.toml` — uncomment the already-present line at strand_bake/Cargo.toml:40
(the path is the same string from one level deeper: tools/strand_bake → tools → dismantle →
Downloads → strand/crates):
```toml
strand-quant = { path = "../../../strand/crates/strand-quant" }
```
and add `tools/strand_bake` to the workspace root `Cargo.toml` `members` array (root Cargo.toml:3,
next to `"tools/awq_bake"` at line 9):
```toml
    "tools/strand_bake",
```

**Verify the `metal` version skew is tolerable** (recipe part (a) caveat — strand-quant pins
`metal 0.27`/`objc 0.2`, dismantle `0.29`/`objc2`):
```bash
cargo tree -p dismantle-core -i metal
```
Two `metal` minors can coexist (distinct crates to the resolver) but bloat the build. If unwanted,
feature-gate strand-quant's `[target.'cfg(target_os = "macos")'.dependencies] metal`/`objc` so
dismantle pulls only `format`+`decode` (those deps exist for strand-quant's own GPU Viterbi
*encoder*, which dismantle never invokes). That feature split is out of scope for this additive
recipe — flag it if the tree is heavy.

*Post-sweep gate:* `cargo check -p dismantle-core` (will still pass — nothing references the dep yet).

---

## Step 2 — `lib.rs` + `backend/mod.rs` (b1, b2)

**2a.** `crates/dismantle-core/src/lib.rs` — register the module next to `pub mod gguf;` (lib.rs:5).
The module is platform-neutral (byte-parse + `memmap2`), so no `#[cfg]`:
```rust
pub mod strand;
```

**2b.** `crates/dismantle-core/src/backend/mod.rs` — add ONE variant to the **open** `WeightKind`
enum, right after `Q4kFast` (backend/mod.rs:127, the enum ends at line 128):
```rust
    /// STRAND trellis-coded weights (.strand v2): k-bit Viterbi index stream +
    /// per-block {scale_q, sub_scales, init_state} side table. Integer, float-free
    /// decode (reconstruct_q = (scale_q*quantile_q)>>16). Distinct kernel family;
    /// self-gates on whether `ensure_strand_cache` found a sidecar, like Q4kFast.
    StrandTrellis,
```
No `Op` / `supports()` / `BackendGemv::gemv` signature change — STRAND is a weight storage class
behind the single `Op::Gemv` verb, exactly like `Q4kFast`.

**2c.** `crates/dismantle-core/src/backend/metal.rs` — `MetalBackend::gemv` matches `spec.weight`
at metal.rs:195; the Q4_K family arm is `WeightKind::Q4K | WeightKind::Q4kFast =>` at metal.rs:205.
Adding the enum variant makes that `match` non-exhaustive, so add an arm to keep it compiling:
```rust
            // STRAND rides the gemv_proj! pin path (per-tensor sidecar table /
            // rht_seed), which GemvSpec cannot carry (see the metal.rs:197-204
            // "SEAM GAP"). The typed-seam entry is therefore a hard error until
            // GemvSpec grows a sidecar handle — NOT the production path.
            WeightKind::StrandTrellis => Err(Error::Metal(
                "WeightKind::StrandTrellis requires the gemv_proj! pin path \
                 (GemvSpec carries no per-tensor table/rht_seed); not reachable via the typed seam"
                    .into(),
            )),
```
This is intentionally a stub: the SEAM GAP doc-comment at metal.rs:197-204 already states the
predec/fast Q4_K variants fall through to the base kernel because `GemvSpec` has no sidecar table —
STRAND needs strictly more (a per-tensor `rht_seed` + table offset), so its real integration is the
macro path (step 7), and the seam arm just preserves exhaustiveness.

*Post-sweep gate:* `cargo check -p dismantle-core` — compiles on all targets; no dispatch wired yet.

---

## Step 3 — reconcile the orphaned `strand/reader.rs` (part (c))

The scaffold `reader.rs` (read in full) hand-rolls a v2 parser with a **stale 32-byte `BlockEntry`**
that contradicts both the 16-byte on-disk `format::BlockOffsetRecord` and the 80-byte GPU record.
Replace the hand-parse with delegation to the single owner.

**Delete** (these duplicate / contradict `strand_quant::format`):
- the `Cursor` primitive reader (reader.rs:405-437);
- the local constants `STRAND_MAGIC`/`STRAND_VERSION_V2`/`STRAND_BLOCK_ENTRY_STRIDE` (reader.rs:74-83)
  and the `mod flags` mirror (reader.rs:88-97);
- **the 32-byte `BlockEntry` struct + `read_at` + `block_entry` (reader.rs:154-207, :355-362)** — the
  stale struct (its `STRIDE` is declared 16 while its body is 32 B; its `min_base_q`/`n`/`sub_off`
  fields are not on disk);
- `StrandFile::from_mmap`'s hand parse (reader.rs:235-330), `check_region`/`table_bytes`
  (reader.rs:384-401), and the smoke test that hand-lays bytes (reader.rs:462-523).

**Rewrite** `reader.rs` to this (drop-in; the module keeps its mmap + zero-copy surface, now sourced
from the canonical header). Note `StrandTensorInfo` now carries the v2 region geometry from
`TensorHeaderV2` (format.rs:512-532), and offsets are absolute file bytes like
`gguf::TensorInfo::data_offset`:

```rust
//! `.strand` v2 reader — mmap-backed, random-access tensor index.
//!
//! Delegates ALL parsing to `strand_quant::format::read_strand_v2_header` — the
//! single owner of the `.strand` v2 wire format (MAGIC `STR2`, 4096-byte pages,
//! 16-byte on-disk `BlockOffsetRecord`, separate page-aligned table/payload/
//! sideinfo regions). This reader hands out zero-copy mmap slices + per-tensor
//! geometry; it does NOT build the 80-byte GPU `BlockEntry` — that is the
//! loader's job (`ensure_strand_cache`, qwen_dense.rs), built at upload time.
//! Mirrors `gguf/reader.rs` structurally.

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Decode family for dispatch. STRAND is NOT a `GgmlType` (closed; `from_u32`
/// rejects unknown tags) — it rides `backend::WeightKind::StrandTrellis`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrandQuant {
    /// k-bit Viterbi index stream + per-block {scale_q, sub_scales, init_state};
    /// integer float-free decode (`reconstruct_q = (scale_q*quantile_q)>>16`).
    Trellis,
}

/// One entry in the v2 tensor index. Offsets are ABSOLUTE file bytes
/// (page-aligned), the STRAND analog of `gguf::TensorInfo::data_offset`.
/// Fields are populated straight from `strand_quant::format::TensorHeaderV2`.
#[derive(Debug, Clone)]
pub struct StrandTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub l_bits: u8,
    pub k_bits: u8,
    pub vec_dim: u8,
    pub rht_seed: u64,
    pub has_rht_seed: bool,
    pub tail_biting: bool,
    pub has_affine_min: bool,
    pub block_len: u32,
    /// Total weight count (so a short final block decodes to the right length).
    pub total: usize,
    pub n_blocks: usize,
    /// ABS offset of the on-disk 16-byte `BlockOffsetRecord[n_blocks]` table.
    pub table_offset: usize,
    /// ABS offset of the verbatim-v1 `bits` symbol stream (page-aligned).
    pub payload_offset: usize,
    pub payload_bytes: usize,
    /// ABS offset of the side-info (6-bit sub_scales [+ affine-min]); 0 if none.
    pub sideinfo_offset: usize,
    pub sideinfo_bytes: usize,
    pub quant: StrandQuant,
}

impl StrandTensorInfo {
    /// Which decode family this tensor needs (maps to `WeightKind::StrandTrellis`).
    pub fn quant(&self) -> StrandQuant {
        self.quant
    }
}

/// The mmap-backed `.strand` v2 file. Mirrors [`crate::gguf::GgufFile`].
pub struct StrandFile {
    pub mmap: Mmap,
    pub version: u32,
    /// File-level flags (bit0 = ALL_STRICT) from the v2 header.
    pub flags: u32,
    /// Source digest stamped by the baker; checked against the GGUF at load (G3).
    pub source_sha256: [u8; 32],
    pub tensors: HashMap<String, StrandTensorInfo>,
    pub tensor_order: Vec<String>,
}

impl StrandFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path.as_ref())?;
        // Safety: read-only for the lifetime of `StrandFile` (same contract as GgufFile::open).
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap)
    }

    /// Parse a `.strand` v2 file already mmap'd, delegating to the single owner.
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let hdr = strand_quant::format::read_strand_v2_header(&mmap[..])
            .map_err(|e| Error::Gguf(format!("strand v2: {e}")))?;

        let mut tensors = HashMap::with_capacity(hdr.tensors.len());
        let mut order = Vec::with_capacity(hdr.tensors.len());
        for h in &hdr.tensors {
            order.push(h.name.clone());
            tensors.insert(
                h.name.clone(),
                StrandTensorInfo {
                    name: h.name.clone(),
                    shape: h.shape.clone(),
                    l_bits: h.l_bits,
                    k_bits: h.k_bits,
                    vec_dim: h.vec_dim,
                    rht_seed: h.rht_seed,
                    has_rht_seed: h.has_rht_seed,
                    tail_biting: h.tail_biting,
                    has_affine_min: h.has_affine_min,
                    block_len: h.block_len,
                    total: h.total,
                    n_blocks: h.n_blocks,
                    table_offset: h.table_offset,
                    payload_offset: h.payload_offset,
                    payload_bytes: h.payload_bytes,
                    sideinfo_offset: h.sideinfo_offset,
                    sideinfo_bytes: h.sideinfo_bytes,
                    quant: StrandQuant::Trellis,
                },
            );
        }
        Ok(Self {
            mmap,
            version: strand_quant::format::VERSION_V2,
            flags: hdr.flags,
            source_sha256: hdr.source_sha256,
            tensors,
            tensor_order: order,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&StrandTensorInfo> {
        self.tensors.get(name)
    }

    /// Zero-copy slice of the verbatim-v1 `bits` symbol stream (mirrors
    /// `GgufFile::tensor_bytes`). The Metal kernel reads this slice of the pin.
    pub fn tensor_bits(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        self.mmap.get(t.payload_offset..t.payload_offset + t.payload_bytes)
    }

    /// Zero-copy slice of the ON-DISK 16-byte `BlockOffsetRecord` table region.
    /// (The 80-byte GPU `BlockEntry[]` is built by the loader, NOT here.)
    pub fn block_table(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        let len = t.n_blocks * strand_quant::format::BlockOffsetRecord::SIZE;
        self.mmap.get(t.table_offset..t.table_offset + len)
    }

    /// Zero-copy slice of the side-info region (6-bit sub_scales [+ affine-min]).
    /// The loader needs this to expand `eff[8]` per block.
    pub fn sideinfo(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.get(name)?;
        if t.sideinfo_offset == 0 {
            return None;
        }
        self.mmap.get(t.sideinfo_offset..t.sideinfo_offset + t.sideinfo_bytes)
    }
}
```

Update `mod.rs` (the module docs at mod.rs:1-24 still say `STRQ`/v1-owned; correct them) and keep the
re-export (mod.rs:28) — the public surface is unchanged except `StrandTensorInfo`'s fields:
```rust
pub use reader::{StrandFile, StrandQuant, StrandTensorInfo};
```

**Rewrite the reader's smoke test** to build its fixture via the canonical writer instead of
hand-laid bytes (the old test asserted the stale 32-byte layout, reader.rs:462-523):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_v2_via_canonical_writer() {
        use strand_quant::encode::encode_tensor;
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
        use strand_quant::TrellisConfig;

        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let shape = [4u64, 256u64]; // in_features 256 % block_len 256 == 0 (STRICT)
        let pt = PackedTensorV2 {
            base: PackedTensor {
                name: "blk.0.ffn_down.weight",
                shape: &shape,
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc: &enc,
            },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2(&[pt], [7u8; 32], true).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("strand_v2_reader_{}.strand", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        let sf = StrandFile::open(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(sf.version, 2);
        assert_eq!(sf.source_sha256, [7u8; 32]);
        let ti = sf.tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!(ti.shape, vec![4, 256]);
        assert_eq!(ti.k_bits, cfg.k_bits as u8);
        assert_eq!(ti.n_blocks, enc.blocks.len());
        // table region is exactly n_blocks * 16 bytes, payload is verbatim v1 bits.
        assert_eq!(
            sf.block_table("blk.0.ffn_down.weight").unwrap().len(),
            enc.blocks.len() * 16
        );
        assert_eq!(sf.tensor_bits("blk.0.ffn_down.weight").unwrap(), &enc.bits[..]);
    }
}
```

*Post-sweep gate:* `cargo test -p dismantle-core strand::` (the reader's own test; needs step 1's dep).

---

## Step 4 — `tools/strand_bake`: flip the TODOs + fix the `src_hash` type bug

The scaffold's source-read + tensor-selection are real (strand_bake/src/main.rs:182-225); two steps
are gated behind `STRAND_BAKE_TODO` and a `bail!`. Flip them, and **fix the load-bearing type bug**:

**The bug.** `src_hash_first8` returns a `u64` (`u64::from_be_bytes`, main.rs:136-142), and the
scaffold's TODO sketch passes that `u64` to `write_strand_v2` (main.rs:239). But
`write_strand_v2(tensors, source_sha256: [u8; 32], strict)` (format.rs:324) takes the **full 32-byte
digest**. Passing a `u64` does not compile; truncating to 8 bytes would make the G3 staleness gate
(step 6) compare against a different field than it stamps. **Stamp and check the full 32-byte
`Sha256::digest`.**

**4a.** Replace `src_hash_first8` (main.rs:133-142) with a full-digest helper:
```rust
/// Full SHA-256 of the source file — the staleness key stamped into the v2
/// header (format.rs:324) and checked at `ensure_strand_cache` load (G3).
fn source_sha256(path: &Path) -> Result<[u8; 32]> {
    let bytes = fs::read(path).with_context(|| format!("hash source {}", path.display()))?;
    Ok(Sha256::digest(&bytes).into())
}
```
and at the call site (main.rs:154-155):
```rust
    let src_sha = source_sha256(&args.input)?;
    eprintln!("[strand_bake] source_sha256 = {}", hex::encode(src_sha));
```
(add `hex` to the tool's deps, or format the bytes manually — cosmetic.)

**4b.** Replace the gated encode+emit block (main.rs:227-243) and the trailing `bail!`
(main.rs:251-255) with the real pipeline. Accumulate `EncodedTensor`s + their `block_len`, then emit
v2 once after the loop:
```rust
    // (declare before the `for name in &gguf.tensor_order` loop, main.rs:192)
    let mut encoded: Vec<(String, [u64; 2], strand_quant::encode::EncodedTensor,
                          u8, u8, u8, u64, u32)> = Vec::new();
```
inside the loop, replacing the TODO (main.rs:227-243):
```rust
        // 3. Encode (QTIP Viterbi/RHT/affine-min). bake-time float touch only.
        let cfg = strand_quant::TrellisConfig::for_bpw(bpw);
        let enc = strand_quant::encode::encode_tensor(&w, &cfg);
        // rht_seed: STRAND derives it per-tensor (FNV-1a over the name, |1).
        // If `encode_tensor` did not store one (`enc.has_rht_seed == false`),
        // pass 0; the kernel's host-side RHT(x) only fires when has_rht_seed.
        let rht_seed = if enc.has_rht_seed {
            strand_quant::rht::seed_for_name(name) // see note below
        } else {
            0
        };
        encoded.push((
            name.clone(),
            [rows as u64, cols as u64],
            enc,
            cfg.l_bits as u8,
            cfg.k_bits as u8,
            cfg.vec_dim() as u8,
            rht_seed,
            cfg.block_len as u32,
        ));
        n_selected += 1;
```
after the loop, replacing the final `bail!` (main.rs:251-255):
```rust
    if encoded.is_empty() {
        bail!("strand_bake: no projection tensors selected; nothing to write");
    }
    let packed: Vec<strand_quant::format::PackedTensorV2> = encoded
        .iter()
        .map(|(name, shape, enc, l, k, vd, seed, bl)| strand_quant::format::PackedTensorV2 {
            base: strand_quant::format::PackedTensor {
                name,
                shape,
                rht_seed: *seed,
                l_bits: *l,
                k_bits: *k,
                vec_dim: *vd,
                enc,
            },
            block_len: *bl,
        })
        .collect();
    // STRICT: every Qwen2.5 projection has in_features % 256 == 0, so strict=true
    // must succeed (and asserts the kernel's linear (row,block) map holds).
    let file = strand_quant::format::write_strand_v2(&packed, src_sha, true)
        .map_err(|e| anyhow::anyhow!("write_strand_v2: {e}"))?;
    fs::write(&args.output, &file)
        .with_context(|| format!("write {}", args.output.display()))?;
    eprintln!(
        "[strand_bake] wrote {} ({} bytes, {} tensors, {} passthrough)",
        args.output.display(), file.len(), n_selected, n_passthrough
    );
    Ok(())
```

> **`rht_seed` note (verify at apply).** The kernel's host RHT(x) needs the *exact* seed the encoder
> used. The README (strand-decode-kernel/shaders/README.md:94,103) says the encoder derives it as
> `FNV-1a(tensor name) | 1` and writes it to the v2 header (`quantize-model.rs:1028`). Confirm the
> public symbol name in strand-quant (`rht::seed_for_name` or equivalent — grep `rht.rs`); if the
> encoder stores the seed back onto `EncodedTensor` or returns it from `encode_tensor`, use that
> instead of recomputing. The seed in the `.strand` header (`hdr.tensors[i].rht_seed`, read by the
> loader in step 6) is the source of truth at decode — the bake just has to write the same value the
> kernel will RHT with.

*Post-sweep:* produce a real artifact and sanity-check its size vs expected bpw:
```bash
cargo run -p strand_bake_tool --release -- \
    models/qwen2.5-3b-instruct-q4_k_m.gguf models/qwen2.5-3b.strand --bpw 3.34
# expect file bytes ≈ (Σ rows*cols of selected projections) * 3.34/8 + tables/sideinfo
```

---

## Step 5 — THE KERNEL SECTION (rewritten 2026-06-11 around the PROVEN G4 bitslice shape)

> **Supersedes the old per-row recipe in this step.** The original `strand_trellis_gemv`
> (one threadgroup per output row) measured **8-25% of peak, ALU-bound** — it is in will.md §4's
> dead-GPU ledger together with occupancy/multirow (18-29%), vecread, predec, and windowed (8-10%).
> The kernel dismantle copies is **`strand_bitslice.metal`** — the G4 shape, measured
> **60.6-74.0% of the empirically measured peak** (12.7-15.9 Gw/s decode, 3.3-3.9× the 12-core CPU;
> fused B=1 35.7-40.6 Gw/s effective) on the M3 with **673 identity cells byte-identical to
> `decode_tensor_fixed`**. Numbers: `docs/STRAND-speed-roadmap.md` §"G4 FINAL". The old per-row
> kernel may still be copied for debug parity, but it is NOT the production path.

### 5.0 — the proven shape (what makes it fast; do not "improve" it back into a dead kernel)

- **Grid = ALL blocks of the tensor.** One thread owns one 256-weight block-stream END-TO-END.
  Dispatch: `ceil(n_blocks/256)` threadgroups × 256 threads. Full occupancy at ANY tensor shape —
  no idle lanes, no per-row reduction tree, no barrier after the LUT stage.
- **Chain state in registers.** `state/acc/have/word_idx` are thread-private registers; the only
  shared resources are the TG-resident LUT and the device payload/output streams.
- **LUT residency rule:** the frozen 2^L Q12 codebook is staged ONCE per threadgroup into
  threadgroup memory (`2^L × 4` bytes): L=7 → 512 B; **L=12 (the 2-bit reopen) → 16 KB**; L=13 →
  32 KB is the hard TG-memory wall on M3-class. One cooperative copy loop + ONE
  `threadgroup_barrier`, executed by every thread BEFORE the tail guard (threads past `n_blocks`
  must not skip the barrier).
- **Table residency rule:** the 80-byte `BitsliceEntry` per block lives in device memory and is
  read once per thread (sequential per-lane). It is 0.3125 B/w at the 256-weight block — at the
  fused points it is 43-53% of all traffic, so keep it 80 B and resist padding it further.
- **Payload reads are uncoalesced by GPU standards, cache-resident by construction**: adjacent
  lanes' streams are offset by 256·k bits, so a TG's 256 streams cover one contiguous ~16-24 KB
  payload window walking forward in lockstep. Measured: this is NOT the wall; do not add a
  transposed staging tile (it cannot coexist with the 16 KB L=12 LUT in 32 KB TG memory).

### 5.1 — buffer layout (the `strand_bitslice_decode` binding table)

| buffer | type | contents |
|---|---|---|
| 0 | `device const uchar*` | the tensor's contiguous k-bit symbol stream — **padded to a 4-byte word boundary + 8 zero bytes** (the `WordReader` zero-pad contract; the kernel's whole-word loads at the stream tail must stay in bounds) |
| 1 | `device int*` | decoded Q12 out, `total × 4` B (decode-only path; the fused/GEMM kernels replace this with partials) |
| 2 | `device const BitsliceEntry*` | one 80-B record per block, stream order |
| 3 | `constant uint&` | `n_blocks` |
| 4 | `constant uint&` | `k_bits` (2..4) |
| 5 | `constant uint&` | `l_bits` (4..14; LUT must have exactly 2^L entries) |
| 6 | `device const int*` | frozen Q12 codebook, `codebook_lut(l_bits)` |
| tg 0 | `threadgroup int*` | `2^L × 4` B, set via `set_threadgroup_memory_length` |

The GPU-side record (MSL struct == Rust `#[repr(C)]` `metal.rs::BitsliceEntry`, **80 B,
probe-asserted at init** — the kernel exports `strand_bitslice_entry_sizeof` and the host refuses
to run on mismatch; replicate that probe, it is what makes the stride un-divergeable):

```text
BitsliceEntry (80 B, 4-byte aligned, LE):
  uint bit_offset    // ABSOLUTE bit position of the block's first k-bit symbol in buffer(0)
  uint init_state    // baked start state — host did the tail-bite prescan (block_init_state);
                     // the kernel NEVER prescans
  uint out_off       // first output index (prefix sum of n over blocks)
  uint n             // weights in this block (<= 256)
  int  eff[8]        // pre-expanded effective sub-scales, Q16 (eff_scale_q), one per 32 weights
  int  off[8]        // pre-expanded affine offsets (eff_min_q); ALL ZERO when affine-min off —
                     // the +0 is bit-exact, so ONE kernel covers both encode branches
```

Bake reference: `metal.rs::bake_bitslice_entries` (block_plans prefix sums + `SideInfo::hoist` +
`block_init_state`). The loader builds this at upload from the on-disk 16-byte
`BlockOffsetRecord` + side-info — never read 80 B from disk. **Bake ONCE at model load** (the
prepared discipline): per-call rebuild costs 13.8× the dispatch (measured: 63 ms cold vs 4.6 ms
prepared on ffn_down).

### 5.2 — the kernel family to copy (all in `strand_bitslice.metal`)

| kernel | role |
|---|---|
| `strand_bitslice_decode` | decode-only, Q12 ints out (weight materialization / debug parity) |
| `strand_bitslice_gemv_partials` + `strand_bitslice_reduce_rows` | fused B=1 token path: one float partial per block, then a per-row reduce in FIXED ascending block order (no atomics — deterministic float order, documented in the shader header) |
| `strand_bitslice_gemm_partials_b4/b16/b64` + `strand_bitslice_reduce_rows_gemm` | prompt phase; host transposes the activation tile to `xt[col*B + b]`. **Use B=16: measured sweet spot (227-255 GMAC/s, 5.6-6.3× CPU fused-NEON). B=64 REGRESSES on M3-class (64 f32 regs/thread kill occupancy; 1.4-1.9× CPU only) — tile bigger prompts as ceil(B/16) B=16 passes** |
| `strand_bitslice_entry_sizeof` | the host stride probe (assert == 80 at init) |

Fused-path layout precondition (host-asserted): `cols % 256 == 0` and block_len == 256 so no
block straddles a row (block g covers `row = out_off/cols`, columns `[out_off%cols, +n)`). All
Qwen2.5 projections satisfy it.

**5a.** Copy `crates/strand-decode-kernel/shaders/strand_bitslice.metal` (verbatim, the whole
family) to `crates/dismantle-core/shaders/strand_bitslice.metal` — it is `include_str!`'d, so it
must live inside dismantle-core's tree.

**5b.** `crates/dismantle-core/src/metal/mod.rs` — add the const next to `SHADER_QUANT`
(metal/mod.rs:9):
```rust
pub const SHADER_STRAND_BITSLICE: &str =
    include_str!("../../shaders/strand_bitslice.metal");
```
and append it to the `all_shader_sources()` array (metal/mod.rs:19-31). The single runtime
`new_library_with_source` (metal/mod.rs:666) compiles it into the same `MTLLibrary`;
`MetalContext::pipeline("strand_bitslice_decode")` etc. resolve by name with **zero**
build-system change (MSL compiles at runtime, no `metallib` artifact).

### 5.3 — the identity-gate protocol dismantle MUST replicate (non-negotiable)

The house pattern (`bin/gate-bitslice.rs`); perf numbers from an ungated kernel are inadmissible:

1. **Decode identity, byte-level:** GPU Q12 out `== strand_quant::decode::decode_tensor_fixed`
   exactly (`assert_eq!` on the `Vec<i32>`), across the encode-lever matrix — k∈{2,3,4}, both
   fold branches, L=12, tail-biting × affine-min, edge lengths (short final block, sub-block
   tail). In strand this is 673 cells; dismantle needs at least its deploy configs × variants.
2. **Fused/GEMM identity via one-hot lanes:** drive the fused kernel with a one-hot activation
   (lane b = 1.0 at column c_b, else 0) ⇒ `y[r*B+b] = Q12(r,c_b)/4096` with only an exact
   power-of-two float op; recover `round(y*4096)` and assert `== decode_tensor_fixed` for every
   row × lane. This holds the MAC kernels themselves to byte-level Q12 identity.
3. **Float y order is documented per kernel, never claimed cross-device:** per-(block,lane)
   j-ascending partials, then block-ascending row reduce. Metal fast-math may contract mul+add
   to fma inside a partial — same caveat llama.cpp carries. The Q12 stream is the moat.
4. **The stride probe** (5.1) and the GPU-free skip (no Metal device ⇒ clean skip, never a fake
   pass).

### 5.4 — expected numbers per rung (M3 Pro, measured 2026-06-11; judge against the
*empirically measured* streaming peak — 98.0 GB/s on this box — never the datasheet)

| rung | 3-bit (k3 L7) | 2-bit (k2 L12) |
|---|---|---|
| decode-only (ffn_down 67.9 Mw) | 5.36 ms = 12.66 Gw/s = 60.6% peak | 4.27 ms = 15.89 Gw/s = 74.0% peak |
| vs CPU rayon 12-core | 3.29× | 3.88× |
| fused B=1 (token) | 1.67 ms = 40.6 Gw/s eff | 1.90 ms = 35.7 Gw/s eff |
| GEMM B=16 (prompt) | 227 GMAC/s, 0.299 ms/col | 256 GMAC/s, 0.266 ms/col |
| prepared dispatch (resident buffers) | 4.56 ms (13.8× vs per-call rebuild) | 4.59 ms |
| 24-tensor token loop, ONE commit | 10.58 ms/token (12.7 Gw/s) | 10.38 ms/token |
| per-commit overhead if NOT batched | ~0.23 ms/tensor | ~0.25 ms/tensor |

Integration rules these numbers force: (1) **prepare at load** (bake+upload once; the 13.8×);
(2) **one command buffer per token across ALL tensors** (at 7B's 196 projections, per-tensor
commits would add ~45 ms/token — more than the decode itself); (3) the fused paths' resident
bill is payload + table + LUT ≈ 0.56-0.69 B/w (no 4 B/w out buffer needed).

### 5.5 — the activation-RHT call sequence (THE OLD RECIPE IN THIS DOC WAS WRONG — fixed here)

> **Correction (pinned by test).** Earlier revisions of this doc (and
> `strand-decode-kernel/shaders/README.md`) said: compute `x_rht = rht_forward(x, seed)` ONCE
> per GEMV and dot every row against it. **That is wrong for every row after the first.** The
> encoder's Rademacher signs are drawn from the GLOBAL flat index
> (`rht.rs::sign_at(seed, row*in + col)`), so each row r is rotated by a DIFFERENT signed
> Hadamard R_r, and `<R_r(w_r), R_0(x)> ≠ <w_r, x>` for r > 0. The divergence is pinned by
> `outlier_mac.rs::single_rotation_recipe_diverges_for_multirow` (O(1) wrong, not noise-level).
> **The correctness reference is `outlier_mac::matvec_rht`**: broadcast x to every row and push
> it through the encoder's own `rht_forward_rows_inplace(seed)` — per-row signs by construction:
>
> `y[r] = <q_r, FWHT(s_r ⊙ x)>`, `s_r` = signs at flat indices `r·in .. r·in+in`.

Cost arithmetic (be honest before wiring): per-row rotation = rows FWHTs = `rows·in·log2(in)`
FLOPs ≈ 815M for ffn_down vs 67.9M MACs — **~12× the MAC**. So the production routes, in order:

1. **RHT-off tensors** (`!has_rht_seed`): the fused kernels take x directly — nothing to do.
2. **Per-column-sign encoder flag** (signs drawn per column index, identical for every row):
   makes the single-rotation trick EXACT and unlocks the fused GPU path at zero per-token cost.
   This is an **encode-bits change**: per house rules it ships as an OFF-BY-DEFAULT flag with an
   A/B quality gate and a KAT re-anchor note. Not built yet; the named route for dismantle-scale
   serving of RHT-on artifacts.
3. **Reference fallback** (correctness, evals, small models): `matvec_rht`'s broadcast-rotate —
   or decode + inverse-RHT the weights once at load (`patched_weights`) and serve dense f32,
   surrendering the compression but keeping bit-identical provenance.

### 5.6 — the OUTL call sequence (decided by arithmetic: CPU-patch boundary)

The outlier channel REPLACES weights (it is not an add), and the bulk stream's decoded value at
an outlier position is only approximately zero — the exact sparse form needs the **residual**
(`outlier_mac.rs` module doc):

```text
y[r] = (W_bulk · x)[r]  +  Σ_{outlier i=(r,c)} (val_i − w_bulk_i) · x[c]
```

Call sequence per tensor:
1. **Load time:** `outlier_residuals(model, name)` — one bulk decode + inverse RHT, residual =
   `val − w_bulk` at each outlier index. 12 B/outlier resident (0.12 B/w at the 1% channel).
2. **Per token:** GPU bulk fused GEMV → y (resident, unified memory) → **CPU rayon sparse add**
   `y[row] += resid · x[col]` against the ORIGINAL (unrotated) x.

The arithmetic that picked CPU: at 7B, 1% = 68M residuals ≈ 0.55-0.8 GB streamed + 136 MFLOP
per token ⇒ **5.6-8.3 ms at the measured 98 GB/s ≈ 3-5% of the ~172-196 ms fused-GPU token
wall** — and on unified memory the CPU reads y in place (zero readback). A GPU sparse add must
be CSR-per-row ordered to stay float-deterministic (atomics forbidden) and would save ≤5%;
build it only if profiling shows the CPU↔GPU sync latency (not bandwidth) binding. Float-order
note: the sparse adds land AFTER the row's dense sum — `matvec_rht` and `matvec_patched` agree
to float tolerance, not bit-exactly; pick ONE order per deployment and stick to it.

> **Drift note (the one always-on change).** `all_shader_sources()` is hashed into the profile
> signature: `profile.rs:633 short_hash(metal::all_shader_sources().as_bytes())`. Appending a shader
> **changes that hash**. It is a profile/cache key, not a golden-output hash — but confirm no test
> pins it:
> ```bash
> grep -rn "all_shader_sources\|short_hash" crates/dismantle-core/src crates/*/tests
> ```
> If a test asserts the literal hash, update the expected constant in the same commit (still additive
> — new kernel source).

*Post-sweep gate:* `cargo check -p dismantle-core` (the shader compiles into the library only at
runtime, so a host `cargo check` still passes; the MSL itself is validated by the G2 test in step 9).

---

## Step 6 — model fields + `ensure_strand_cache` (b3) + the lazy-load/ref-bind gate (b4)

All edits in `crates/dismantle-core/src/model/qwen_dense.rs`, `#[cfg(target_os = "macos")]` where they
touch `PinnedBuffer`/`MetalContext`.

**6a. New fields** parallel to `q4k_fast_buf`/`q4k_fast_offsets` (qwen_dense.rs:277-280). The pin holds
the payload + the loader-built 80-byte `BlockEntry[]` + the frozen LUT + an `x_rht` scratch:
```rust
    /// .strand v2 deploy artifact pinned whole (DISMANTLE_QWEN_STRAND). Holds,
    /// per tensor: [payload bytes (page-aligned)] [BlockEntry[n_blocks] 80 B
    /// (page-aligned)]. `strand_offsets` maps a GGUF source offset (the key the
    /// gemv_proj! macro has from `$tref.offset`) → that tensor's pin geometry.
    #[cfg(target_os = "macos")]
    pub(crate) strand_buf: Option<crate::metal::PinnedBuffer>,
    #[cfg(target_os = "macos")]
    pub(crate) strand_offsets: Option<std::collections::HashMap<usize, StrandTensorPin>>,
    /// Frozen 2^L Q12 codebook (one per distinct l_bits seen), pinned once.
    #[cfg(target_os = "macos")]
    pub(crate) strand_lut: Option<crate::metal::PinnedBuffer>,
    /// Per-block partials scratch for the bitslice fused pair (§5.2), sized to
    /// the largest tensor's n_blocks × max-B floats, reused across tensors.
    /// (Was an RHT(x) scratch in the superseded per-row recipe — §5.5.)
    #[cfg(target_os = "macos")]
    pub(crate) strand_partials: Option<crate::metal::PinnedBuffer>,
```
with the per-tensor geometry struct (place it above the model `impl`, or in `strand/mod.rs` and
re-export — its fields are exactly what the host fn in step 7 binds):
```rust
/// Per-tensor v2 geometry resolved at load, keyed by GGUF source offset.
#[derive(Clone, Copy)]
pub(crate) struct StrandTensorPin {
    pub bits_offset: usize,      // ABS byte offset of payload in `strand_buf`
    pub bits_len: usize,
    pub gpu_table_offset: usize, // ABS byte offset of the 80-byte BlockEntry[] in `strand_buf`
    pub n_blocks: usize,
    pub rows: usize,
    pub cols: usize,
    pub rht_seed: u64,           // host RHTs x once per GEMV iff has_rht_seed
    pub has_rht_seed: bool,
    pub k_bits: u8,
    pub l_bits: u8,
    pub has_affine_min: bool,    // debug_assert!(false) at 3-bit deploy (step 7)
}
```
Initialize the new fields to `None` at the two struct-construction sites that set `q4k_fast_buf: None`
(near qwen_dense.rs:1284-1286):
```rust
            #[cfg(target_os = "macos")]
            strand_buf: None,
            #[cfg(target_os = "macos")]
            strand_offsets: None,
            #[cfg(target_os = "macos")]
            strand_lut: None,
            #[cfg(target_os = "macos")]
            strand_partials: None,
```

> **BITSLICE UPDATE (2026-06-11).** The loader sketch below builds the OLD per-row kernel's
> 80-byte `BlockEntry` (`bit_offset/init_state/scale_q/eff[8]/n/d/_pad`). For the production
> bitslice kernel (step 5) build the §5.1 **`BitsliceEntry`** instead (`bit_offset/init_state/
> out_off/n/eff[8]/off[8]`, also 80 B) — mirror `strand-decode-kernel/src/metal.rs::
> bake_bitslice_entries` (prefix-sum `out_off`, `SideInfo::hoist` for eff/off, host-side
> `block_init_state` tail-bite recovery so the kernel never prescans). The page-alignment,
> G3 hash gate, and offset-map plumbing below carry over unchanged. Also pad each payload
> upload to a 4-byte word boundary + 8 zero bytes (the WordReader contract, §5.1 buffer 0).

**6b. `ensure_strand_cache`** — a near-clone of `ensure_q4k_fast_cache` (qwen_dense.rs:3895-3983).
Differences: probe `.strand` (no AWQ branch), parse via the canonical header, run G3, and **build the
80-byte GPU `BlockEntry[]` at upload** by folding sub-scales with the (verified `pub`) strand-quant
helpers. The field names below match the real `ensure_q4k_fast_cache` body: `self.metal_ctx`,
`self._weights_path`, `self.config.n_layers`, `self.layers[li].{q_proj,k_proj,v_proj,o_proj,ffn_gate,
ffn_up,ffn_down}.offset`, `ctx.new_buffer_with_bytes(&blob)`.

```rust
    #[cfg(target_os = "macos")]
    fn ensure_strand_cache(&mut self) -> Result<()> {
        if self.strand_buf.is_some() {
            return Ok(());
        }
        let ctx = match self.metal_ctx.as_ref() {
            Some(c) => c,
            None => return Ok(()),
        };
        // Probe <weights>.strand, then models/<stem>.strand. No AWQ variant.
        let wp = &self._weights_path;
        let stem = wp.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        candidates.push(wp.with_extension("strand"));
        candidates.push(std::path::PathBuf::from(format!("models/{stem}.strand")));
        let sidecar_path = match candidates.iter().find(|p| p.exists()).cloned() {
            Some(p) => p,
            None => return Ok(()), // No sidecar; feature stays off (like q4k_fast:3943).
        };

        let bytes = std::fs::read(&sidecar_path)
            .map_err(|e| Error::Model(format!("read .strand {}: {e}", sidecar_path.display())))?;
        let hdr = strand_quant::format::read_strand_v2_header(&bytes)
            .map_err(|e| Error::Model(format!("parse .strand v2: {e}")))?;

        // G3: staleness fail-fast. The v2 header carries the FULL source digest
        // (format.rs:359/586); compare against the source GGUF (mirrors
        // sidecar.rs:217/246 GgufHashMismatch being fatal). No silent stale bits.
        {
            use sha2::{Digest, Sha256};
            let src = std::fs::read(wp)
                .map_err(|e| Error::Model(format!("read source for G3 hash: {e}")))?;
            let actual: [u8; 32] = Sha256::digest(&src).into();
            if actual != hdr.source_sha256 {
                return Err(Error::Model(format!(
                    ".strand source_sha256 mismatch for {}: artifact was baked from a \
                     different source GGUF (re-run strand_bake)",
                    sidecar_path.display()
                )));
            }
        }

        // Build, per tensor: [payload (page-aligned)] [BlockEntry[n_blocks] (page-aligned)].
        // The 80-byte GPU record is built HERE (loader), never read from disk.
        const PAGE: usize = strand_quant::format::PAGE; // 4096
        const REC: usize = 80; // sizeof(GPU BlockEntry); static-checked below.
        fn align_up(x: usize, a: usize) -> usize { (x + a - 1) & !(a - 1) }

        let mut blob: Vec<u8> = Vec::new();
        // name -> (bits_offset, bits_len, gpu_table_offset, geometry)
        let mut by_name: std::collections::HashMap<String, StrandTensorPin> =
            std::collections::HashMap::with_capacity(hdr.tensors.len());

        for h in &hdr.tensors {
            // (a) payload, page-aligned.
            let pad = align_up(blob.len(), PAGE);
            blob.resize(pad, 0);
            let bits_offset = blob.len();
            blob.extend_from_slice(
                bytes
                    .get(h.payload_offset..h.payload_offset + h.payload_bytes)
                    .ok_or_else(|| Error::Model("strand: payload slice OOB".into()))?,
            );
            let bits_len = h.payload_bytes;

            // (b) build the 80-byte BlockEntry[] (page-aligned). eff[8] folded from
            //     scale_q + side-info sub-scales via strand_quant (do NOT fork the math).
            let pad = align_up(blob.len(), PAGE);
            blob.resize(pad, 0);
            let gpu_table_offset = blob.len();
            // side-info slice (sub_scales for every block, then affine-min if present).
            let sideinfo = if h.sideinfo_offset != 0 {
                bytes
                    .get(h.sideinfo_offset..h.sideinfo_offset + h.sideinfo_bytes)
                    .ok_or_else(|| Error::Model("strand: sideinfo slice OOB".into()))?
            } else {
                &[][..]
            };
            // 3-bit deploy point: no affine-min path in the kernel. Hardening flag.
            debug_assert!(!h.has_affine_min,
                "strand: has_affine_min=true needs the 4-bit off[8] loader+kernel path (not built)");
            // sub-scale stride for a FULL block: ceil(6 * (block_len/SUB_BLOCK) / 8) bytes.
            let n_sub_full = (h.block_len as usize).div_ceil(strand_quant::encode::SUB_BLOCK);
            let ss_stride_full = (6 * n_sub_full).div_ceil(8);
            let mut ss_cursor = 0usize;
            let bl = h.block_len as usize;
            for (b, rec) in h.table.iter().enumerate() {
                // weights in this block (last may be short).
                let nb = if b + 1 < h.n_blocks { bl } else { h.total - (h.n_blocks - 1) * bl };
                let n_sub = nb.div_ceil(strand_quant::encode::SUB_BLOCK);
                let ss_bytes = (6 * n_sub).div_ceil(8);
                let codes = if h.sideinfo_offset != 0 {
                    let slice = sideinfo
                        .get(ss_cursor..ss_cursor + ss_bytes)
                        .ok_or_else(|| Error::Model("strand: sub_scale slice OOB".into()))?;
                    strand_quant::encode::unpack_sub_scales(slice, n_sub) // 6-bit codes
                } else {
                    Vec::new()
                };
                ss_cursor += ss_bytes;
                let _ = ss_stride_full; // documents the constant-stride invariant for full blocks

                // eff[8]: fold each sub-scale code into an effective Q16 scale.
                // Pad to 8 with the super-scale (unity-fold) for short final blocks.
                let mut eff = [rec.scale_q; 8];
                for s in 0..n_sub.min(8) {
                    let code = codes.get(s).copied().unwrap_or(strand_quant::encode::SUB_SCALE_UNITY);
                    eff[s] = strand_quant::decode::eff_scale_q(rec.scale_q, code);
                }

                // bit_offset is tensor-relative in the on-disk record; the kernel
                // wants the bit position within THIS tensor's flat w_bits blob,
                // which starts at `bits_offset` (byte-aligned page). The payload
                // bytes are copied verbatim, so the tensor-relative bit offset IS
                // the absolute bit offset within the per-tensor blob the kernel
                // binds at buffer(0) (set_buffer offset = bits_offset). Keep it.
                let bit_offset = rec.bit_offset as u32;

                // emit 80-byte #[repr(C)] BlockEntry (LE): uint bit_offset, uint
                // init_state, int scale_q, int eff[8], ushort n, ushort d, uint _pad.
                blob.extend_from_slice(&bit_offset.to_le_bytes());
                blob.extend_from_slice(&rec.init_state.to_le_bytes());
                blob.extend_from_slice(&rec.scale_q.to_le_bytes());
                for e in eff {
                    blob.extend_from_slice(&e.to_le_bytes());
                }
                blob.extend_from_slice(&(nb as u16).to_le_bytes()); // n
                blob.extend_from_slice(&1u16.to_le_bytes());          // d = 1 (scalar)
                blob.extend_from_slice(&0u32.to_le_bytes());          // _pad
            }
            // static check: each record is exactly 80 bytes.
            debug_assert_eq!((blob.len() - gpu_table_offset), h.n_blocks * REC);

            // shape is row-major [out_rows=rows, in_cols=cols] (format.rs:311).
            let rows = h.shape.first().copied().unwrap_or(0) as usize;
            let cols = h.shape.get(1).copied().unwrap_or(0) as usize;
            by_name.insert(
                h.name.clone(),
                StrandTensorPin {
                    bits_offset,
                    bits_len,
                    gpu_table_offset,
                    n_blocks: h.n_blocks,
                    rows,
                    cols,
                    rht_seed: h.rht_seed,
                    has_rht_seed: h.has_rht_seed,
                    k_bits: h.k_bits,
                    l_bits: h.l_bits,
                    has_affine_min: h.has_affine_min,
                },
            );
        }

        // Map by GGUF source offset using the same proj_to_name walk as q4k_fast
        // (qwen_dense.rs:3962-3970), so strand_offsets is keyed by $tref.offset.
        let mut offsets = std::collections::HashMap::new();
        let cfg = &self.config;
        for li in 0..cfg.n_layers {
            let proj_to_name: &[(usize, &str)] = &[
                (self.layers[li].q_proj.offset,   "attn_q.weight"),
                (self.layers[li].k_proj.offset,   "attn_k.weight"),
                (self.layers[li].v_proj.offset,   "attn_v.weight"),
                (self.layers[li].o_proj.offset,   "attn_output.weight"),
                (self.layers[li].ffn_gate.offset, "ffn_gate.weight"),
                (self.layers[li].ffn_up.offset,   "ffn_up.weight"),
                (self.layers[li].ffn_down.offset, "ffn_down.weight"),
            ];
            for (src_off, name_suf) in proj_to_name {
                let full = format!("blk.{li}.{name_suf}");
                if let Some(pin) = by_name.get(&full) {
                    offsets.insert(*src_off, *pin);
                }
            }
        }

        // Pin the assembled blob once; build the LUT + x_rht scratch.
        let buf = ctx.new_buffer_with_bytes(&blob);
        // LUT: assume one l_bits across the file (uniform deploy). If --mixed mixes
        // l_bits, key a HashMap<u8, PinnedBuffer> instead and bind per tensor.
        let l_bits = hdr.tensors.first().map(|h| h.l_bits as u32).unwrap_or(7);
        let lut = strand_quant::codebook::codebook_lut(l_bits);
        let lut_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(lut));
        // partials scratch sized to the largest tensor's n_blocks (× 16 if the
        // B=16 prompt GEMM shares it), reused each GEMV (§5.2).
        let max_blocks = by_name.values().map(|p| p.n_blocks).max().unwrap_or(0);
        let partials = ctx.new_buffer_with_bytes(&vec![0u8; max_blocks * 16 * 4]);

        self.strand_buf = Some(buf);
        self.strand_offsets = Some(offsets);
        self.strand_lut = Some(lut_buf);
        self.strand_partials = Some(partials);
        Ok(())
    }
```

**6c. Lazy-load gate** next to `q4k_fast_active` (qwen_dense.rs:4277-4278):
```rust
        #[cfg(target_os = "macos")]
        let strand_active = crate::env_on("DISMANTLE_QWEN_STRAND");
        #[cfg(target_os = "macos")]
        if strand_active && self.strand_buf.is_none() {
            self.ensure_strand_cache()?;
        }
```

**6d. Mutual-exclusion guard** (additive, same style as the F16_KV/INT4_KV refusals at
qwen_dense.rs:4311-4373). STRAND replaces the Q4_K weight bytes wholesale, so it is incompatible with
every lever that reads/re-decodes Q4_K. The env flags are all in scope by this point: `predec_active`
(:4242), `q4k_fast_active` (:4277), `w4a8_active_early` (:4284), `awq_active_early` (:4300):
```rust
        #[cfg(target_os = "macos")]
        if strand_active && (predec_active || q4k_fast_active || w4a8_active_early || awq_active_early) {
            return Err(Error::Model(
                "DISMANTLE_QWEN_STRAND=1 is incompatible with Q4K_PREDEC/Q4K_FAST/W4A8/AWQ \
                 (STRAND replaces the Q4_K weight stream); unset the others".into(),
            ));
        }
```

**6e. Bind the immutable refs** for the macro body, next to where `q4k_fast_ref` is bound
(qwen_dense.rs:4440-4445), AFTER the `ensure_*` mutable-borrow calls:
```rust
        #[cfg(target_os = "macos")]
        let strand_ref = if strand_active {
            match (self.strand_buf.as_ref(), self.strand_offsets.as_ref(),
                   self.strand_lut.as_ref(), self.strand_partials.as_ref()) {
                (Some(b), Some(m), Some(l), Some(x)) => Some((b, m, l, x)),
                _ => None, // sidecar absent ⇒ feature off ⇒ fall through to GGUF
            }
        } else {
            None
        };
```

*Post-sweep gate:* `cargo check -p dismantle-core` (fields + loader compile; the macro arm in step 7
is what actually *uses* `strand_ref`).

---

## Step 7 — the `gemv_proj!` pre-empt arm (b5) + the kernel host fn

**7a.** `gemv_proj!` (qwen_dense.rs:5004) dispatches on `$tref.dtype` (a `GgmlType`) — the very first
token in the body is `match $tref.dtype {` at qwen_dense.rs:5007. STRAND is **not** a `GgmlType`, so
its override must sit **before** that match (the `q4k_fast_ref` arm at qwen_dense.rs:5091-5110
pre-empts the *base Q4_K kernel* but lives *inside* the `GgmlType::Q4_K` arm — STRAND hoists one level
out because its key is "is this offset STRAND?", not a dtype). Wrap the existing match:

```rust
            macro_rules! gemv_proj {
                ($site_w4a8:expr, $tref:expr, $pinned_f16:expr, $rows:expr, $cols:expr,
                 $x:expr, $x_i8:expr, $x_sc:expr, $out:expr) => {{
                    // STRAND pre-empt: a .strand v2 override for this tensor's GGUF
                    // offset short-circuits the whole GgmlType ladder. Default-off ⇒
                    // strand_ref is None ⇒ this block vanishes and the macro is
                    // byte-identical to today.
                    #[cfg(target_os = "macos")]
                    let __strand_hit = strand_ref.and_then(|(buf, map, lut, xrht)| {
                        map.get(&$tref.offset).copied().map(|p| (buf, p, lut, xrht))
                    });
                    #[cfg(not(target_os = "macos"))]
                    let __strand_hit: Option<()> = None;

                    #[cfg(target_os = "macos")]
                    if let Some((strand_buf, p, lut_buf, partials_buf)) = __strand_hit {
                        kernels::gemv_strand_bitslice_pinned_tcb(
                            &mut tcb, strand_buf, lut_buf, partials_buf, &p, $x, $out,
                        )?;
                    } else {
                        match $tref.dtype {
                            /* ...every existing GGUF arm, COMPLETELY UNCHANGED
                               (qwen_dense.rs:5007-5151)... */
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        let _ = __strand_hit;
                        match $tref.dtype {
                            /* ...same existing GGUF arms... */
                        }
                    }
                }};
            }
```
The macro signature and all call sites (qwen_dense.rs:5374/5635/5972/6255 etc.) are untouched — the
override is keyed off `$tref.offset`/`$x`/`$out`, which the macro already receives. (`&mut tcb` is the
local the existing arms use; `$x`/`$out` are `&PinnedBuffer`.) If duplicating the `match` block for
the two `#[cfg]`s is unpalatable, factor the GGUF ladder into a helper `gemv_proj_gguf!` and call it
in both arms — purely cosmetic.

**7b. New kernel host fn** `kernels::gemv_strand_bitslice_pinned_tcb` — same `TokenCommandBuffer`
shape as `gemv_q4k_fast_v1_pinned_tcb` (kernels/mod.rs:3447-3491), but it encodes the BITSLICE
fused pair (§5.2): `strand_bitslice_gemv_partials` (grid = blocks) then
`strand_bitslice_reduce_rows` (grid = rows, fixed ascending-block float order, no atomics). The
Rust reference for both dispatches is `strand-decode-kernel/src/metal.rs::matvec_dispatch` —
mirror it, including the buffer indices. It needs a per-tensor `partials` scratch
(`n_blocks × 4` B, allocate once at load next to the pin):

```rust
    /// STRAND bitslice fused GEMV (G4): one thread per 256-weight block-stream
    /// (decode walk identity-gated to decode_tensor_fixed via one-hot lanes),
    /// then a per-row fixed-order reduce. Two encoders in the SAME token command
    /// buffer — never one commit per tensor (measured +0.23-0.25 ms/commit).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_strand_bitslice_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        strand_buf: &PinnedBuffer,        // [payload (word-padded +8)][BitsliceEntry[]] per tensor
        lut_buf: &PinnedBuffer,           // 2^L Q12 codebook (codebook_lut(l_bits))
        partials_buf: &PinnedBuffer,      // n_blocks floats, resident scratch
        pin: &crate::model::qwen_dense::StrandTensorPin,
        x_buf: &PinnedBuffer,             // f32 activation — see the RHT routing note below
        out_buf: &PinnedBuffer,           // y, length = rows
    ) -> Result<()> {
        if pin.cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "bitslice gemv requires cols % 256 == 0 (no block straddles a row); got {}",
                pin.cols
            )));
        }
        // bounds checks on payload/table regions exactly as the q4k_fast sibling (:3473).
        let n_blocks = pin.n_blocks as u32;
        let bpr = (pin.cols / 256) as u32;

        // pass 1: per-block partials. grid = ceil(n_blocks/256) TGs x 256 threads.
        tcb.dispatch_threadgroups("strand_bitslice_gemv_partials",
            (n_blocks.div_ceil(256), 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(strand_buf), pin.bits_offset as u64);      // w_bits
            enc.set_buffer(1, Some(x_buf), 0);                                 // x
            enc.set_buffer(2, Some(partials_buf), 0);                          // partials
            enc.set_buffer(3, Some(strand_buf), pin.gpu_table_offset as u64);  // BitsliceEntry[]
            enc.set_u32(4, n_blocks);
            enc.set_u32(5, pin.cols as u32);
            enc.set_u32(6, pin.k_bits as u32);
            enc.set_u32(7, pin.l_bits as u32);
            enc.set_buffer(8, Some(lut_buf), 0);                               // Q12 LUT
            enc.set_threadgroup_memory_length(0, (1u64 << pin.l_bits) * 4);    // sh_lut
        })?;
        // pass 2: per-row reduce in fixed block order (deterministic float order).
        tcb.dispatch_threadgroups("strand_bitslice_reduce_rows",
            ((pin.rows as u32).div_ceil(256), 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(partials_buf), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_u32(2, pin.rows as u32);
            enc.set_u32(3, bpr);
        })
    }
```

For the prompt phase, the sibling fn dispatches `strand_bitslice_gemm_partials_b16` +
`strand_bitslice_reduce_rows_gemm` against a host-transposed `xt[col*16 + b]` tile
(reference: `metal.rs::gemm_dispatch`; partials scratch = `n_blocks × 16` floats). **B=16 only
on M3-class** — §5.2's measured B=64 occupancy regression.

> **Activation-RHT routing (THE CORRECTED RECIPE — §5.5).** Do NOT compute one
> `rht_forward(x, seed)` and bind it for all rows — that single-rotation recipe silently
> corrupts every row after the first (per-row global-flat-index signs; divergence pinned by
> `outlier_mac.rs::single_rotation_recipe_diverges_for_multirow`). Routing: `!has_rht_seed`
> tensors bind x directly (the code above); RHT-on tensors need either the off-by-default
> per-column-sign encoder flag (route 2, exact single rotation — not built yet) or the
> `outlier_mac::matvec_rht` reference path (route 3, ~12× MAC cost — evals only).
> After the GPU bulk y: the OUTL sparse residual add runs on CPU per §5.6
> (`outlier_residuals` at load; `y[row] += resid·x[col]` per token, original x).

> **Accessor-name caveat (verify at apply).** `dispatch_threadgroups`/`set_u32`/
> `set_threadgroup_memory_length` are placeholders for dismantle's encoder API — grep
> `kernels/mod.rs` for how an existing two-pass kernel encodes consecutive dispatches into one
> `TokenCommandBuffer` and use those exact methods. Metal guarantees pass-2 sees pass-1's
> device writes across encoders in the same command buffer.

*Post-sweep gate:* `cargo check -p dismantle-core` then the G2 parity test (step 9).

---

## Step 8 — full additive-proof build (G0)

With `DISMANTLE_QWEN_STRAND` unset, the dismantle binary must be byte-identical to current `main`. The
pre-empt block vanishes (`strand_ref == None`), the new `WeightKind` arm is never reached, and the
only always-on change is the extra shader string (step 5) + its profile-hash key:
```bash
cargo build -p dismantle-core            # post-sweep, after steps 1-7
cargo test  -p dismantle-core            # every existing golden hash + parity test
```
If the `profile.rs:633` hash is asserted anywhere (grep from step 5), update that one constant in the
same commit. No other golden output may move — that is the mechanical meaning of "additive".

---

## Step 9 — the verification gate: GPU↔CPU bit-identity (G2), modeled on q4k_fast parity

This is the headline determinism gate and the precedent is explicit:
`q4k_fast.rs:54-59` — "Output of `gemv_q4k_fast_v1_pinned_tcb` on a Q4K_FAST tensor MUST be
bit-identical to `gemv_q4_k_m_v3_8r_pinned_tcb` on the source Q4_K tensor (per the session's parity
test at q_proj decode shape)." The parity-test family lives in `crates/dismantle-core/tests/` (e.g.
`gemm_q4k_v4r_predec_parity.rs`). Add `crates/dismantle-core/tests/strand_trellis_parity.rs`:

```rust
// GPU↔CPU bit-identity for the STRAND trellis kernel — the gate the whole
// "bit-identical on phone/WASM/MCU/FPGA" claim reduces to. Mirrors the Q4K_FAST
// parity contract (q4k_fast.rs:54-59).
#![cfg(target_os = "macos")]

#[test]
fn strand_trellis_gpu_matches_cpu_decode() {
    use strand_quant::encode::encode_tensor;
    use strand_quant::TrellisConfig;

    // One Qwen ffn_down-shape tile: rows x cols, cols % 256 == 0 (STRICT).
    let rows = 8usize;
    let cols = 512usize; // 2 blocks/row; keep small for a fast test
    let weights: Vec<f32> =
        (0..rows * cols).map(|i| ((i as f32) * 0.0007).sin() * 0.6).collect();
    let cfg = TrellisConfig::for_bpw(3.0); // k=3, l=7 deploy point

    // CPU reference: integer Q12 weights, bit-exact and platform-independent.
    let enc = encode_tensor(&weights, &cfg);
    let q_cpu: Vec<i32> = strand_quant::decode::decode_tensor_fixed(&enc, &cfg);

    // GPU: bake to v2, pin, dispatch `strand_bitslice_decode` (the DECODE-ONLY
    // kernel — its whole job is emitting the Q12 ints) and assert the output
    // buffer equals q_cpu BYTE-FOR-BYTE. This is gate-bitslice's protocol §5.3
    // verbatim; in strand it passes 673 config×variant cells. Then hold the
    // FUSED pair to the same Q12 identity via one-hot activations (§5.3 item 2):
    // y[r] = Q12(r,c)/4096 exactly, recover round(y*4096) == q_cpu[r*cols+c].
    //
    // General-x y matches the CPU reference only within float tolerance — the
    // documented per-(block)-partials + fixed-order row reduce grouping (§5.2);
    // never claim cross-device float equality, the Q12 stream is the contract.
    //
    // (Fill in the MetalContext bring-up by copying the existing
    //  gemm_q4k_v4r_predec_parity.rs harness; the asserts are the contract above.
    //  Reference implementation of the whole gate: strand's bin/gate-bitslice.rs.)
    let _ = (q_cpu, enc);
}
```
The CPU↔CPU exactness (G1) is already green in strand-quant
(`strand_v2_round_trip_matches_v1_q12` / `strand_v2_header_matches_full_read`, format.rs:1028/1098) —
no new work. G3 is the hash check in step 6b.

**Run when the GPU is free** (post-sweep): `cargo test -p dismantle-core --test strand_trellis_parity`.

The **decode gate verdict is already in** (2026-06-11, strand repo, M3, idle, machine-stamped):
the bitslice shape measured **60.6-74.0% of the empirically measured streaming peak** —
decisively past the ≥50% revival line that was committed before running (every prior kernel:
8-29%). Expected numbers per rung are tabulated in §5.4; dismantle's own A/B vs
`gemm_q4_k_m_fused` on its harness is still owed before any tps ship claim (different host
plumbing — **measure in situ, do not argue from strand's numbers**), but the kernel-level
go/no-go on this silicon generation is GO.

---

## Apply order (each step compiles before the next)

1. strand-quant helpers `pub` (verify; sweep-safe, `cargo check -p strand-quant`).
2. dismantle deps (`strand-quant` path dep × 2 + workspace member; `cargo tree -i metal`).
3. `lib.rs` + `backend/mod.rs` + `backend/metal.rs` arm (`cargo check -p dismantle-core`).
4. reconcile `strand/reader.rs` (`cargo test -p dismantle-core strand::`).
5. `tools/strand_bake` (flip TODOs + fix `src_hash`; bake a real `.strand`, check size).
6. Metal kernel copy + `all_shader_sources()` (+ profile-hash grep).
7. model fields + `ensure_strand_cache` + gate/ref-bind (`cargo check -p dismantle-core`).
8. `gemv_proj!` pre-empt arm + `gemv_strand_trellis_pinned_tcb` (`cargo check -p dismantle-core`).
9. G0 full additive proof (`cargo test -p dismantle-core`) → G2 parity → decode/bandwidth gate.

> The numbering in the apply order folds steps 5/6 (kernel) and 7/8 (host fn) together; the section
> bodies above are split for readability. Steps 1–4 are pure CPU/Rust and cheap; the GPU experiment
> is only the G2 + bandwidth gate at the end.

---

## TODO / risks carried into the hardening pass (ranked)

1. ~~The decode gate is make-or-break.~~ **RESOLVED 2026-06-11: the G4 bitslice shape measured
   60.6-74.0% of measured peak (12.7-15.9 Gw/s, 3.3-3.9× the 12-core CPU)** — the compute-bound
   fear was true for the per-row kernel (8-25%, now dead) and false for the bitslice grid. The
   remaining owed measurement is dismantle's in-situ A/B vs `gemm_q4_k_m_fused` (step 9).
2. **Activation-RHT for RHT-on tensors (§5.5) — the biggest open integration item.** The
   single-rotation recipe this doc used to give is WRONG (per-row signs; divergence pinned by
   test). Production needs either the off-by-default per-column-sign encoder flag (encode-bits
   change: A/B gate + KAT re-anchor, not yet built) or RHT-off artifacts; the
   `matvec_rht` broadcast-rotate reference costs ~12× the MAC and is for evals only. Any
   seed/segmentation mismatch still silently corrupts `y` — the seed source of truth is the
   descriptor (`hdr.rht_seed`), never a name re-derivation.
3. **Encoder/dispatch accessor names (step 7b).** `dispatch_threadgroups`/`set_u32`/
   `set_threadgroup_memory_length` are placeholders — bind to dismantle's real encoder API (grep
   an existing two-pass shmem kernel). The bitslice pair needs `sh_lut` length `2^L × 4` B on
   pass 1 and nothing on pass 2.
4. **`metal` crate version skew (step 1).** strand-quant pins `metal 0.27`/`objc 0.2`; dismantle
   `0.29`/`objc2`. They coexist but bloat the build; feature-gate strand-quant's macOS encoder deps so
   dismantle pulls only `format`+`decode`. Verify with `cargo tree -i metal`.
5. **`src_hash` type bug (step 4) — load-bearing.** The scaffold's `u64` `src_hash_first8`
   (main.rs:141) does not match `write_strand_v2`'s `[u8; 32]` (format.rs:324); it won't compile and
   would desync the G3 gate. Fixed to the full 32-byte digest in step 4 — keep stamp and check in
   lock-step.
6. **80-byte GPU record vs 16-byte disk record (step 6b).** The loader expands the 16-byte
   `BlockOffsetRecord` + side-info into the 80-byte `BlockEntry` at upload; never read 80 B from disk
   (the stale reader.rs struct deleted in step 3) or bind the 16-byte record to the kernel. The
   `debug_assert_eq!((blob.len()-gpu_table_offset), n_blocks*80)` is the guard.
7. **affine-min only safe at 3-bit (steps 6b/7b).** The shipped 3-bit point has
   `has_affine_min == false`; the loader + kernel omit `off[8]`. A 4-bit `--mixed` tensor needs
   `eff_min_q` (decode.rs:80) expanded into an extended record and added in the kernel
   (README:83-86). Guarded by `debug_assert!(!has_affine_min)` until built.
8. **profile-hash key drift (step 5).** Appending `SHADER_STRAND_TRELLIS` changes
   `all_shader_sources()`'s hash (profile.rs:633). It's a cache key, not golden output, but if a test
   pins it, bump the constant in the same commit.
9. **`bit_offset` relative-vs-absolute (step 6b).** The on-disk record's `bit_offset` is
   tensor-relative (format.rs:243-248). Because the loader copies the payload verbatim and binds
   buffer(0) at `bits_offset`, the tensor-relative bit offset IS the absolute offset within the
   per-tensor blob the kernel sees — kept as-is. If the kernel is ever changed to bind one shared
   buffer for all tensors (no per-tensor offset), this must become an absolute bit offset across the
   blob. Documented inline.

---

## Appendix — tokens/s arithmetic (decode-primitive ceilings, NOT end-to-end inference)

Basis: the measured M3 fused-B=1 effective rates (3-bit 40.6 Gw/s, 2-bit 35.7 Gw/s, §5.4);
tok/s = rate ÷ nominal Gw/token. The 3090-class column is **arithmetic, not a measurement** —
no CUDA port of the bitslice kernel exists. The fused kernel is ALU-bound on M3 (21.7-29.8% of
peak bytes moved), so the 3090 column scales by ~5-6× integer-ALU class ratio; the 936 GB/s
bandwidth ceiling sits far above and never binds. Deduct from every ceiling: attention/KV/
norms/sampling, the activation-RHT route (§5.5), OUTL ~3-5% (§5.6), and ~0.23-0.25 ms per
command-buffer commit unless the whole token is batched (§5.4 rule 2).

| model (Gw/token) | M3-class 3-bit | M3-class 2-bit | 3090-class 3-bit (arith.) | 3090-class 2-bit (arith.) |
|---|---|---|---|---|
| 0.5B (0.5) | ~81 tok/s | ~71 | ~405-487 | ~357-428 |
| 7B (7) | **~5.8** | ~5.1 | ~29-35 | ~25-31 |
| 14B (14) | ~2.9 | ~2.5 | ~14-17 | ~13-15 |

Prompt phase (B=16 GEMM, measured 227-255 GMAC/s on M3): 7B ≈ **31-37 prompt tok/s**
decode-primitive on the M3 GPU — vs ~9.8 on the CPU fused-NEON path at B=64. Context: llama.cpp
Q4 token decode on M3-class silicon ≈ 20-30 tok/s, prefill ≳100 — the gap at 7B is now ~4-5×
on token decode and ~3× on prompt, down from the CPU era's ~50×.

---

## Cross-references (do not duplicate)

- **`docs/STRAND-speed-roadmap.md` §"G4 FINAL"** — the bitslice revival + productionization
  numbers this doc's kernel section (step 5) is built on, with the full measurement protocol.
- **`crates/strand-decode-kernel/src/bin/gate-bitslice.rs`** — the reference implementation of
  the identity-gate protocol (§5.3): 673-cell decode matrix, one-hot GEMM lanes, prepared
  identity, machine-stamped perf only after identity passes in-process.
- **`crates/strand-decode-kernel/src/metal.rs`** — `BitsliceGpu` / `BitslicePrepared` /
  `bake_bitslice_entries` / `matvec_dispatch` / `gemm_dispatch`: the host-side reference for
  steps 6-7.
- **`crates/strand-decode-kernel/src/outlier_mac.rs`** — the activation-RHT correctness
  reference (`matvec_rht`), the pinned single-rotation divergence test, and the OUTL residual
  machinery (`outlier_residuals`) behind §5.5-5.6.
- **`docs/plans/strand-prod-dismantle-recipe.md`** — the design + the three-confused-structs
  reconciliation + the dependency decision (part (a)). Read first.
- **`docs/STRAND-dismantle-integration.md`** — the original sketch. **Superseded** on the wire format
  (it says magic `STRQ`/16 KB pages/32-byte records; the real format is `STR2`/4096-byte pages/16-byte
  `BlockOffsetRecord` — recipe §0). Kept for the §3/§4 measurement-harness narrative and the
  paradigmshift.md grounding.
- **`crates/strand-decode-kernel/shaders/README.md`** — the OLD per-row kernel's gate doc.
  **Its single-rotation RHT recipe (lines 90-110) is WRONG for multirow tensors** (§5.5) and its
  PASS/MARGINAL/FAIL ladder is superseded by the G4 verdict; kept for the on-disk-16B-vs-GPU-80B
  resolution narrative.
- **`docs/STRAND-metal-kernel-impl.md`**, **`docs/STRAND-format-v2-spec.md`** — the kernel build sheet
  and the wire-format spec the implementation matches.
