# STRAND production BAKE producer — design plan

_Status: DESIGN (doc only). Additive. Does not change any existing public fn, the v1
`--packed-out` path, the quant pipeline, or the integer decode contract._

Goal: turn a safetensors model into a `.strand` **v2** archive (`STR2`) — the
GPU-random-access deploy artifact the Metal GEMV kernel `mmap`s — by reusing the
quantizer that already exists.

The v2 serializer is **already built and tested** in
`crates/strand-quant/src/format.rs`:
`write_strand_v2(&[PackedTensorV2], source_sha256: [u8;32], strict: bool) -> Result<Vec<u8>, String>`
(format.rs:324), with `PackedTensorV2 { base: PackedTensor, block_len: u32 }`
(format.rs:265), and round-trip + STRICT-reject tests (format.rs:1028, 1098, 1153).
**Nothing in `format.rs` needs to change.** The work is purely producer-side: feed it
the right `PackedTensorV2`s from the quant loop and stamp the source hash.

---

## Decision: extend `quantize-model.rs`, do NOT add a new bin

**Recommendation: add a `--packed-v2-out <path>` flag to the existing
`crates/strand-quant/src/bin/quantize-model.rs`.** Reasons:

1. **The whole pipeline is already there.** `quantize-model.rs` has the zero-dep
   safetensors parser (`SafeTensors`/`StTensor`/`to_f32`, lines 209-263), tensor
   classification (`is_quantizable_linear` 421, `mp_bits` 157), the per-tensor
   `TrellisConfig` resolver (`resolve_cfg` 1068), the parallel Viterbi/RHT/affine-min
   quant loop (1158-1196), deterministic header-order sort (1202), and an **existing
   `--packed-out` v1 emitter** (1205-1238) that already builds `PackedTensor` from
   `TensorResult`. v2 is the same data plus a `block_len` field and a different writer
   call. A new bin would duplicate ~600 lines of parser + CLI + quant orchestration.

2. **v1 and v2 share `TensorResult` verbatim.** `TensorResult` already carries
   `enc: Option<EncodedTensor>, l_bits, k_bits, rht_seed` (lines 768-780) and the v1
   path consumes exactly those. v2 needs one more datum per tensor — `block_len` — which
   is a one-line add to `TensorResult` (it is `cfg.block_len`, always 256 on every
   shipped `for_bpw*` config; trellis.rs:97/106/117).

3. **The separate `tools/strand_bake`** described in
   `docs/STRAND-dismantle-integration.md` §2 is a *dismantle-repo* tool (it needs
   `dismantle-core::GgufFile` for the GGUF path and lives in dismantle's workspace). It
   is the GGUF→strand baker. The *safetensors→strand* producer belongs in
   `strand-quant` next to the quantizer, and the flag here is exactly what `strand_bake`
   would call into for the safetensors branch. Building the flag here does **not** block
   `strand_bake`; it gives it the reusable core. (If a standalone safetensors-only
   binary is later wanted, factor the emit block into a `pub fn` — not needed now.)

**Why not a new bin:** the only thing a new bin buys is a narrower `--help`. It costs a
duplicated safetensors parser and a second copy of the quant loop that will drift from
the live one. Rejected.

---

## SHA-256: vendor a tiny safe implementation in `strand-quant`

`source_sha256: [u8;32]` is the v2 header provenance field (spec §3, format.rs:359). It
is the determinism / staleness key a consumer checks at load
(`docs/STRAND-dismantle-integration.md` §2 / §5; dismantle's `awq_bake` stores
`first-8-bytes-of-SHA-256` of the source, `awq_bake/src/main.rs:107-110`).

**Repo reality (verified):**
- No sha256 implementation exists anywhere under `crates/` (grep: zero hits).
- No hashing crate is a dependency (`sha2`/`ring`/`blake3`/`digest` absent from every
  `crates/*/Cargo.toml` and from `Cargo.lock`).
- `strand-quant` is deliberately near-zero-dep (deps: `strand-gguf`, `wide`, optional
  `cudarc`, and macOS `metal`/`objc`) and the bin is `#![forbid(unsafe_code)]`
  (quantize-model.rs:54) with its own hand-rolled zero-dep safetensors + JSON parsers.

**Decision: vendor a ~70-line pure-safe-Rust SHA-256 in a new module
`crates/strand-quant/src/sha256.rs`** rather than pull the `sha2` crate (+`digest`,
+`generic-array`, +`typenum`, +`cpufeatures`) into a tool crate that has avoided every
external parser so far. Rationale:
- Matches the crate's established "vendor the small thing, no dep" pattern (the
  safetensors and HSDI parsers are already hand-rolled).
- Keeps `#![forbid(unsafe_code)]` intact (a scalar FIPS-180-4 SHA-256 is trivially safe
  Rust; `sha2`'s asm paths are not needed for a one-shot hash of a file we already read).
- The dismantle `strand_bake` tool independently uses the `sha2` workspace crate
  (`awq_bake/Cargo.toml:19`); the two producers must agree only on the **bytes hashed
  and the digest**, not the implementation. FIPS-180-4 SHA-256 of the same input file is
  identical regardless of impl, so `digest[0..8]` from our vendored hasher equals
  `awq_bake`'s `first8` — the dismantle staleness check still matches. (A unit test pins
  the empty-string and `"abc"` NIST vectors so this can never silently diverge.)

`sha256.rs` public surface (additive, new file — no existing symbol touched):

```rust
//! Minimal, dependency-free FIPS-180-4 SHA-256 (safe Rust) for stamping the
//! `.strand` v2 source_sha256 provenance field. One-shot only.
#![forbid(unsafe_code)]
pub fn sha256(data: &[u8]) -> [u8; 32] { /* standard block compression */ }
```

Register it in `crates/strand-quant/src/lib.rs` (one line: `pub mod sha256;` next to
`pub mod format;` at lib.rs:56). The hash is computed over the **raw bytes of the input
safetensors file** — which `SafeTensors::open` already reads into `self.bytes`
(quantize-model.rs:227-228) — so no second file read is needed (see edit site C).

---

## How `source_sha256` is computed

`source_sha256 = sha256(<entire input .safetensors file bytes>)`. The producer hashes
the same `--in` file it parses; `SafeTensors` already owns `bytes: Vec<u8>` of the whole
file (quantize-model.rs:218, 227). Add a `bytes()` accessor (or hash inline in `main`
right after `SafeTensors::open`, where `st.bytes` is in scope through a small accessor).
This is bit-for-bit the artifact's provenance: any change to the source weights changes
the digest, which the deploy-time loader compares against the engine's GGUF/safetensors
hash to reject a stale `.strand` (dismantle integration §5). The full 32 bytes ship;
consumers wanting dismantle's compact key take `digest[0..8]`.

---

## The exact `PackedTensorV2` construction from `TensorResult`

The v1 path (quantize-model.rs:1205-1238) already builds this from `TensorResult`:

```rust
PackedTensor {
    name: &r.name, shape: &r.shape, rht_seed: r.rht_seed,
    l_bits: r.l_bits, k_bits: r.k_bits, vec_dim: 1, enc, // enc: &EncodedTensor
}
```

v2 wraps that and adds `block_len` (the one new `TensorResult` field):

```rust
// after the v1 --packed-out block, a parallel --packed-v2-out block:
if let Some(packed_path) = &args.packed_v2_out {
    let mut pts: Vec<format::PackedTensorV2> = Vec::new();
    let mut skipped = 0usize;
    for r in &quant_results {
        match &r.enc {
            Some(enc) => pts.push(format::PackedTensorV2 {
                base: PackedTensor {
                    name: &r.name,
                    shape: &r.shape,
                    rht_seed: r.rht_seed,
                    l_bits: r.l_bits,
                    k_bits: r.k_bits,
                    vec_dim: 1,            // v2 producer is scalar-only, same as v1 path
                    enc,
                },
                block_len: r.block_len,    // NEW field on TensorResult (= cfg.block_len)
            }),
            None => skipped += 1,          // vector-trellis / salient tensors: not in v1/v2
        }
    }
    let src = sha256::sha256(st.bytes());  // accessor added in edit site A2
    let bytes = format::write_strand_v2(&pts, src, args.strict_v2)
        .unwrap_or_else(|e| panic!("write_strand_v2: {e}"));
    fs::write(packed_path, &bytes).expect("write .strand v2 archive");
    // report bytes/weight exactly like the v1 path (lines 1224-1236)
    return;
}
```

Notes that make this correct (all verified against the code):
- `r.enc` is `Some` **only on the clean scalar, non-salient path**
  (quantize-model.rs:1013-1017: `want_packed && cfg.vec_dim()==1 && salient_pct==0.0`).
  So `vec_dim: 1` is always right here, and the `None => skipped` branch mirrors v1's
  "skipped N non-scalar tensors" message (1231-1235). The v2 writer would otherwise have
  no LUT for vector tensors — consistent with the v1 limitation.
- `r.shape` is the **original** `[out, in]` shape (set in `quantize_one`,
  quantize-model.rs:1020, from `job.shape`). `write_strand_v2` reads `shape[1]` as
  `in_features` for the STRICT divisibility check (`in_features_of`, format.rs:310) — so
  passing the original shape is exactly what the STRICT invariant needs.
- `block_len`: `TrellisConfig.block_len` is `usize` (trellis.rs:23) and is `256` on every
  shipped config. `PackedTensorV2.block_len` is `u32`, so store `cfg.block_len as u32`.
  `write_strand_v2` falls back to 256 if it ever sees 0 (`enc_block_len`, format.rs:983),
  but we pass the real value so the table length and STRICT check are exact.

---

## STRICT-flag default = `true`

`--packed-v2-out` defaults `strict = true` (deploy invariant: every 2-D tensor satisfies
`in_features % block_len == 0`, so the kernel's `(row, block)` map is linear and needs no
per-row base array — spec §7, format.rs:319-323). For the shipped quant targets this
always holds: `block_len = 256` and every Qwen2.5-7B projection has `in_features`
divisible by 256 (4096/3584/2560/18944/…). STRICT therefore **catches a misconfiguration
loudly** (e.g. an odd-dim tensor, or a future non-256 block_len) instead of silently
emitting a RAGGED file the Metal kernel can't take the linear path on.

Provide an escape hatch `--ragged-v2` (or `--no-strict-v2`) that flips it to `false` for
models with an odd `in_features` (e.g. 896-dim tensors on Qwen2.5-0.5B): RAGGED still
round-trips and decodes bit-identically, it just clears the `ALL_STRICT` file flag
(format.rs:346) so a consumer knows the linear map doesn't universally hold. Default
stays `true`.

On a STRICT violation `write_strand_v2` returns `Err` naming the first offending tensor
(format.rs:336-341); surface it via `panic!`/`eprintln!` so the bake fails fast with the
tensor name.

---

## Exact edit sites (file + line)

All edits are in **`crates/strand-quant/src/bin/quantize-model.rs`** unless noted. Each
is additive; no existing line is removed or repurposed.

### New file
- **`crates/strand-quant/src/sha256.rs`** — vendored safe SHA-256 (`pub fn sha256(&[u8])
  -> [u8;32]`) + NIST-vector unit test.
- **`crates/strand-quant/src/lib.rs:56`** — add `pub mod sha256;` immediately after the
  existing `pub mod format;`.

### A. `SafeTensors` raw-bytes access (so we hash without a second read)
- **A2 — quantize-model.rs:~240** (inside `impl SafeTensors`, next to `fn raw`): add
  ```rust
  fn bytes(&self) -> &[u8] { &self.bytes }
  ```
  (`self.bytes` is the full file, populated at open, line 227-231.)

### B. CLI: new flags + struct fields + parser arms
- **B1 — `struct Args`, after `packed_out: Option<String>` (line 541):** add
  ```rust
  /// Path for the packed STRAND **v2** (`STR2`) GPU-random-access archive.
  packed_v2_out: Option<String>,
  /// v2 deploy invariant: require in_features % block_len == 0 for every 2-D
  /// tensor (default true). Cleared by --ragged-v2.
  strict_v2: bool,
  ```
- **B2 — `parse_args` locals, after `let mut packed_out = None;` (line 601):** add
  `let mut packed_v2_out = None;` and `let mut strict_v2 = true;`.
- **B3 — `parse_args` match, after the `"--packed-out"` arm (line 640):** add
  ```rust
  "--packed-v2-out" => packed_v2_out = Some(it.next().expect("--packed-v2-out needs a path")),
  "--ragged-v2" | "--no-strict-v2" => strict_v2 = false,
  ```
- **B4 — `parse_args` `--help` text (line 640-691 region):** add a line documenting
  `--packed-v2-out <path>` and `--ragged-v2` (the v2 deploy archive).
- **B5 — output-required assertion, line 698:** extend so `--packed-v2-out` also
  satisfies "an output was requested":
  `assert!(measure_only || !output.is_empty() || packed_out.is_some() || packed_v2_out.is_some(), ...)`.
- **B6 — `Args { … }` constructor, line 704-727:** add `packed_v2_out,` and `strict_v2,`.

### C. `TensorResult`: carry `block_len`
- **C1 — `struct TensorResult`, after `k_bits: u8` (line 779):** add `pub block_len: u32,`
  (or non-pub; same module). Doc: "per-tensor block_len for v2 layout (= cfg.block_len)."
- **C2 — `quantize_one` return literal, line 1018-1029:** add
  `block_len: cfg.block_len as u32,` to the `TensorResult { … }` (`cfg` is the
  `&TrellisConfig` arg already in scope, line 784).

### D. The v2 emit block
- **D1 — quantize-model.rs:1238**, immediately after the existing `--packed-out` block's
  closing `return;` (line 1237) and before the sidecar-JSON section (line 1242): insert
  the `if let Some(packed_path) = &args.packed_v2_out { … }` block shown above
  (build `Vec<PackedTensorV2>`, hash `st.bytes()`, call `write_strand_v2`, write file,
  log bytes/weight like 1224-1236, `return`).
- **D2 — imports, quantize-model.rs:66:** no change needed —
  `use strand_quant::format::{self, PackedTensor};` already imports `format` (so
  `format::PackedTensorV2` and `format::write_strand_v2` resolve) and `PackedTensor`.
  Add `use strand_quant::sha256;` near the other `use strand_quant::…` lines (63-70).

### E. (optional, recommended) test
- A `#[cfg(test)]` unit test in `sha256.rs` asserting the two NIST vectors
  (`sha256("")` and `sha256("abc")`) so the digest can never silently diverge from
  `awq_bake`'s `sha2`-crate digest. (Format-level v2 round-trip is already covered by the
  three tests in format.rs:1028/1098/1153.)

---

## Risks / sharp edges

1. **Vector-trellis & salient tensors are silently skipped.** `r.enc` is `None` for
   `--vec-dim > 1` or `--salient-patch` runs (quantize-model.rs:1013). The v2 producer
   (like v1) drops those tensors and reports a `skipped N` count. **Mitigation:** mirror
   the v1 skip message exactly; if `pts.is_empty()` after the loop, `panic!` with a clear
   "no scalar tensors to pack — v2 archive needs the scalar path (no --vec-dim/--salient)"
   rather than writing an empty archive. (A v2 file with vector LUTs is out of scope; the
   format carries no codebook — same constraint as v1.)

2. **STRICT will reject odd-dim models.** A model with any `in_features % 256 != 0`
   2-D projection (e.g. Qwen2.5-0.5B's 896-dim tensors) fails the default STRICT bake
   with a clear per-tensor error. **Mitigation:** that's intentional (fail-fast for the
   Metal linear-map invariant); `--ragged-v2` is the documented escape hatch and still
   produces a correct, decodable file.

3. **SHA-256 divergence from dismantle.** If the vendored hasher had a bug it would
   stamp a digest dismantle's `sha2`-crate loader rejects. **Mitigation:** the NIST-vector
   unit test (edit site E) pins correctness; both sides hash the *same input file bytes*,
   and SHA-256 is impl-independent, so `digest[0..8] == awq_bake first8` by construction.

4. **`block_len as u32` truncation.** `block_len` is `usize`; the cast is safe for all
   shipped configs (256) and any realistic block size. A pathological `block_len >
   u32::MAX` is impossible (the Viterbi table is `O(block_len·2^L)` — it would OOM long
   before). No guard needed; note it in the field doc.

5. **Determinism.** Output order is already header-order (the `sort_by_key` at
   quantize-model.rs:1202 runs before both emit blocks), and the quant loop is a pure
   function of input regardless of thread scheduling (documented at 1152-1154). The v2
   archive bytes are therefore deterministic for a given input + flags. No new
   nondeterminism introduced. The source hash is over the input file (stable), not over
   the emitted archive.

6. **Build/run safety (this machine).** A 7B sweep is LIVE on `target/release/quantize-model`
   (pid 89192). These edits are authored only; **do not** `cargo build --release` /
   `cargo build -p strand-quant` / `cargo test -p strand-quant` until the sweep is done —
   they would clobber the live binary or oversubscribe the 12 cores. `cargo check -p
   strand-quant` is the safe compile check (no release binary emitted). The new
   `sha256.rs` test runs under `cargo test -p strand-quant` only (defer until sweep ends).

---

## Summary of the change surface

| Area | File | Edit |
|---|---|---|
| SHA-256 | `crates/strand-quant/src/sha256.rs` (NEW) | vendored safe FIPS-180-4 `sha256(&[u8])->[u8;32]` + NIST test |
| module reg | `crates/strand-quant/src/lib.rs:56` | `pub mod sha256;` |
| raw bytes | `quantize-model.rs:~240` | `fn bytes(&self) -> &[u8]` on `SafeTensors` |
| CLI flags | `quantize-model.rs:541,601,640,~665,698,704` | `--packed-v2-out`, `--ragged-v2`; `packed_v2_out`/`strict_v2` field+local+arm+assert+ctor |
| result field | `quantize-model.rs:779,1018` | `TensorResult.block_len` + `cfg.block_len as u32` |
| emit | `quantize-model.rs:1238` | v2 block: build `Vec<PackedTensorV2>`, hash `st.bytes()`, `write_strand_v2(&pts, src, args.strict_v2)`, write, log, return |
| import | `quantize-model.rs:63-70` | `use strand_quant::sha256;` |

No edits to `format.rs` (writer already complete), `encode.rs`, `decode.rs`,
`trellis.rs`, or any dismantle file.
