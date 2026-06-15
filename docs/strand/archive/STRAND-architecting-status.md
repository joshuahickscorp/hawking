# STRAND architecting status — the capstone (what's built, what's next)

_Written 2026-06-08, mid-training. Snapshot of the design+scaffold wave that produced the v2
format, the lean decode, the Metal gate kernel, and the dismantle integration stubs. This is the
hand-off to the next **GPU-free** session: the completion ledger, the exact ordered hardening
checklist, and the single highest-leverage next action._

> **✅ UPDATE 2026-06-08 (post-wave) — the #1 BLOCKER is RESOLVED.** The v2 wire-format divergence
> (§3) is closed: `strand_quant::format` is the single owner, and the canonical lean parser
> **`read_strand_v2_header` is now implemented + cargo-checked + tested** (`format.rs`; a test proves
> it agrees byte-for-byte with `read_strand_v2`). The kernel `README.md` now labels the 80 B
> `BlockEntry` as a **GPU-side, load-time** struct (on-disk truth = the lean 16 B
> `BlockOffsetRecord`). The dismantle `reader.rs` constants were corrected to the authoritative
> schema (`STR2` / 4 KiB / 16 B record) with a delegate-to-`read_strand_v2_header` TODO. **Remaining
> for the hardening session:** (a) wire the `strand-quant` path dep into dismantle and replace
> `reader.rs`'s hand-rolled parse with the delegation; (b) run the deferred `cargo test -p
> strand-quant`; (c) **the Metal gate on the M3** — now the sole make-or-break.

**Settled context (do NOT re-litigate — see `STRAND-product-spec.md`):** the deployable format is
**uniform 3-bit-deterministic** (9.42 PPL @ 3.34 bpw, integer float-free decode), mixed-precision
(4-bit on attn + `down_proj`) was *thought* to reach a ~7.7–8.0 band — but that "result" was a
bug (uniform-4-bit; see the CORRECTION in `STRAND-product-spec.md`, fixed in `f2f7716`) and is
being re-measured. ONLY uniform-3 (9.42@3.34) and uniform-4 (7.81@4.50) are validated. Moat =
density × determinism × float-free decode. The metric for the dismantle fusion is **bytes/token
(tps ↑, J ↓) + determinism, NOT PPL**. The one make-or-break is the **Metal decode gate**: does
trellis-GEMV stay bandwidth-bound on M3, or fall into dismantle's dead-Q3_K compute-bound trap.

---

## 1. Completion checklist (status per piece)

Legend: **DESIGNED** (doc only) · **SCAFFOLDED** (source written, marked stub/not-wired) ·
**CHECK-PASSED** (compiles via `cargo check`, exercised by a test) · **DEFER** (left for the
hardening / GPU session).

### A. `strand-quant` — the format + decode (the bits)

| Piece | File | Status | Evidence / note |
|---|---|---|---|
| `.strand` v2 writer `write_strand_v2` | `crates/strand-quant/src/format.rs:324` | **CHECK-PASSED** | `cargo check -p strand-quant` clean; round-trip test at `format.rs:841`. |
| `.strand` v2 reader `read_strand_v2` | `format.rs:514` | **CHECK-PASSED** | validates magic/version/alignment/bounds; reconstructs `EncodedTensor` + table. |
| `BlockOffsetRecord` (16 B: `{bit_offset u64, init_state u32, scale_q i32}`) | `format.rs:242` | **CHECK-PASSED** | `#[repr(C)]`, `SIZE=16`, padding-free. **`STR2` magic, 4 KiB `PAGE`.** |
| `PackedTensorV2` / `OwnedTensorV2` | `format.rs:265` / `:276` | **CHECK-PASSED** | mirror v1 `PackedTensor`/`OwnedTensor` + `block_len` + reconstructed `table`. |
| `strand_v1_to_v2` transcoder | `format.rs:762` | **CHECK-PASSED** | produces v2 from a shipped v1 archive (needs per-tensor `block_len`). |
| v2 round-trip test (`enc` bit-match + table == cursor) | `format.rs:841` | **CHECK-PASSED** | asserts `read_strand_v2(write_strand_v2(..)).base.enc == ts.enc`. |
| RAGGED-reject / STRICT test | `format.rs:921` | **CHECK-PASSED** | STRICT errors on `in%block≠0`; RAGGED clears `ALL_STRICT`. |
| `decode_lean` + `decode_lean_with_lut` | `crates/strand-quant/src/decode.rs:268` / `:275` | **CHECK-PASSED** | aligned-word `WordBitReader` (`decode.rs:211`), scale-fold gate (`SUB_BLOCK≥num_states`), native i64 reconstruct. Vector path delegates. |
| `decode_lean_is_bit_identical` test | `decode.rs:491` | **CHECK-PASSED (compiles)** — **DEFER run** | varies bpw {3,2,4} × seed; covers short final block, sub-block tail, u32-unaligned. **Must be RUN in the hardening pass** (`cargo test` is forbidden mid-sweep). |

### B. The Metal kernel + gate (the one real GPU experiment)

| Piece | File | Status | Note |
|---|---|---|---|
| `strand_trellis_gemv.metal` (3-bit deploy geometry) | `crates/strand-decode-kernel/shaders/strand_trellis_gemv.metal` (296 L) | **DESIGNED + SCAFFOLDED** | full MSL: one-TG/row, 256 threads, aligned `load_u32_le` reads, shmem LUT, native 32×32→64 reconstruct, tree-reduce. **Not compiled by any build** (runtime-MSL, no `metallib`). |
| Kernel measurement plan (microbenches, counters, PASS/MARGINAL/FAIL) | `shaders/README.md` (§0–§5) | **DESIGNED** | commits the verdict thresholds BEFORE running (no goalpost-moving). |
| M3 host harness (bake → upload → `MTLCounterSampleBuffer` time) | — | **DEFER (not started)** | the gate cannot run without it. See §2 step 4. Candidate home: `metal` crate shim beside `strand-decode-kernel/bin/kernel-bench.rs`, or dismantle's `bandwidth` suite. |
| CPU byte-traffic reference (`footprint_bytes`, `matvec` GMAC/s) | `crates/strand-decode-kernel/src/bin/kernel-bench.rs` | **CHECK-PASSED** | `cargo check -p strand-decode-kernel` clean; apples-to-apples byte-ratio sanity, NOT the ridge proof. |

### C. dismantle integration (additive, behind an env flag)

| Piece | File | Status | Note |
|---|---|---|---|
| `.strand` v2 reader module `strand/{mod,reader}.rs` | `dismantle/crates/dismantle-core/src/strand/` (28 + 527 L) | **SCAFFOLDED — NOT WIRED** | header + offset-table parse is self-contained (only `crate::{Error,Result}` + `memmap2`, already a dep). **`pub mod strand;` is NOT in `lib.rs`** → orphaned source; dismantle compiles unchanged. Decode/GPU dispatch are marked `TODO`. |
| `WeightKind::StrandTrellis` | `dismantle/.../backend/mod.rs:113` | **DESIGNED ONLY** | spec'd in `reader.rs:429` as a commented companion; **the enum still has only `Q4kFast`**, no `StrandTrellis` variant added. |
| `tools/strand_bake` (GGUF/safetensors → .strand v2) | `dismantle/tools/strand_bake/` (256 L) | **SCAFFOLDED — NOT IN BUILD** | CLI + pipeline shape present; encode+emit gated behind `STRAND_BAKE_TODO` `bail!`s. **NOT in root `Cargo.toml` `members`**; `strand-quant` path dep commented out. |
| `strand_trellis_gemv.metal` in dismantle shaders | `dismantle/.../shaders/` | **DEFER** | the kernel currently lives only in `strand-decode-kernel/shaders/`; copying/`include_str!`-wiring into dismantle is step 4 of §2. |
| `gemv_proj!` pre-empt arm + `ensure_strand_cache` + `gemv_strand_trellis_pinned_tcb` | `dismantle/.../qwen_dense.rs`, `kernels/mod.rs` | **DESIGNED** (integration doc §3c) | no host code yet. |
| Parity tests G0/G1/G2/G3 | `dismantle/.../tests/` | **DESIGNED** | G1 lives in `strand-quant`; G2 (GPU↔CPU bit-identity) is the headline, modeled on `q4k_fast_parity.rs`. |

### D. Docs (the design surface — all DESIGNED, internally near-complete)

`STRAND-format-v2-spec.md`, `STRAND-metal-kernel-impl.md`, `STRAND-dismantle-integration.md`,
`STRAND-product-spec.md`, `STRAND-metal-decode-gate.md`, `STRAND-density-roadmap.md`,
`crates/strand-decode-kernel/shaders/README.md` — all present and consistent with the scaffolded
code **except for the v2 wire-format discrepancy in §3 below**, which is the top hardening item.

---

## 2. HARDENING CHECKLIST (exact, ordered — for the next GPU-free session)

Run these in order. Steps 1–3 are pure CPU/Rust and safe to do **before** the M3 frees; step 4 is
the gate and needs the GPU idle. Nothing here should run while `quantize-model` is sweeping
(`cargo test` / `cargo build --release` are forbidden mid-sweep; `cargo check -p <crate>` is safe).

**Step 0 — confirm the sweep is done.** `pgrep -fl quantize-model` returns nothing. Only then run
any `cargo build`/`cargo test`. (Until then you may still do steps 1a/2a edits + `cargo check`.)

**Step 1 — RECONCILE the v2 wire format (BLOCKER, see §3). Do this first; everything downstream
parses these bytes.**
- Decide the single authoritative schema. Recommendation: **adopt the shipped `strand-quant`
  `format.rs` schema** (`STR2` magic, 16-byte `BlockOffsetRecord`, 4 KiB `PAGE`, tensor-relative
  `bit_offset`) — it is the one that compiles, has round-trip tests, and is the leaner kernel
  record. The dismantle `reader.rs` (`STRQ`+v2, 32-byte `BlockEntry`, 16 KiB pages, absolute
  `bit_offset`) must be rewritten to match it.
- Add `pub fn read_strand_v2_header(&[u8])` to `strand-quant::format` (header-only, no payload
  copy) — the integration doc §1a already specifies the dismantle reader should *delegate* to it
  rather than re-parse. This kills the drift permanently (one schema owner).
- `cargo check -p strand-quant` after the edit (safe mid-sweep).

**Step 2 — RUN the `strand-quant` test suite (after Step 0).**
- `cargo test -p strand-quant` — must go green, with special attention to:
  - `decode_lean_is_bit_identical` (`decode.rs:491`) — the determinism contract G1. A single
    counterexample is a release blocker.
  - the v2 round-trip + RAGGED tests (`format.rs:841`, `:921`).
- If `decode_lean` diverges on the fold path (`bpw=2` ⇒ `SUB_BLOCK≥num_states`), debug the
  `folded[sb*ns+state]` indexing against `decode_tensor_fixed` before anything else.

**Step 3 — finish + wire the dismantle reader (CPU, additive, compiles on all targets).**
- After Step 1's schema reconcile, register the module: add `pub mod strand;` to
  `dismantle/crates/dismantle-core/src/lib.rs` (next to `pub mod gguf;`).
- Add the `strand-quant` path dep to `dismantle-core/Cargo.toml`:
  `strand-quant = { path = "../../../strand/crates/strand-quant" }` and have `reader.rs` delegate
  header parsing to `read_strand_v2_header`.
- Add `WeightKind::StrandTrellis` to `backend/mod.rs` (the variant is spec'd in `reader.rs:429`).
- `cargo check -p dismantle-core` (debug-only; safe even mid-sweep). Resolve any drift.
- Implement `tools/strand_bake`: add `tools/strand_bake` to root `Cargo.toml` `members`, uncomment
  its `strand-quant` dep, replace the `STRAND_BAKE_TODO` `bail!`s with the real
  `encode_tensor` + `write_strand_v2` calls. Then bake a real `<qwen2.5-7b>.strand` (3-bit) from
  the GGUF and assert its byte size ≈ `rows·cols·0.375 + table` (the on-disk bytes/weight truth).

**Step 4 — THE M3 GATE (needs the GPU idle; this is the only real experiment).** Per
`shaders/README.md` §0–§5 and `STRAND-metal-kernel-impl.md` §C — commit the thresholds first:
1. **Correctness gate FIRST.** Build the minimal `metal`-crate host harness (beside
   `strand-decode-kernel/bin/kernel-bench.rs`, or fill dismantle's stub `bandwidth` suite). Dump
   the kernel's decoded `w` for one `ffn_down` block and assert it equals `decode_lean`'s for that
   block, **bit-for-bit**, before trusting any timing. A fast wrong kernel is worse than none.
2. **Peak microbenches on THIS M3** (not datasheet): a streaming-load kernel → real `BW_peak`
   (~100–150 GB/s on M3 Pro); a fused-multiply loop → real `OPS_peak`. These feed the ridge `I*`.
3. **Time `strand_trellis_gemv` with `MTLCounterSampleBuffer` `.timestamp`** (not wall-clock) at
   batch 1 on real Qwen2.5-7B shapes (one wide `cols≥11008` e.g. `ffn_down`/`up`, one narrow
   `cols≈2048` to exercise the B.6 occupancy fallback). Mirror for `gemm_q4_k_m_fused` head-to-head.
4. **Compute** achieved BW (% of `BW_peak`), achieved OPS (% `OPS_peak`), intensity `I =
   work_ops/bytes_read`, and the ridge `I* = OPS_peak/BW_peak`. One Instruments "Metal System
   Trace" frame capture on the wide tensor to read the memory-bandwidth counter directly.
5. **Verdict (no goalpost-moving):**
   - **PASS** → batch-1 BW ≥ 60% `BW_peak` AND `I < I*` AND tps beats Q4_K by ≥ 25% (the 4.5→3.34
     byte ratio). Build the dismantle shim (step 5).
   - **MARGINAL** → bandwidth-bound but tps win < 25%: apply the levers harder (pop 2 symbols/iter,
     B.6b predec-to-shmem staging, fold LUT into constant memory), re-measure. Ship after one pass.
   - **FAIL** → `I ≥ I*` OR BW < 40% peak with compute at ceiling = the Q3_K trap. Do NOT build the
     shim; the density moat then stands on **determinism + on-device-fits** alone (the honest
     downside the roadmap already prices in).

**Step 5 — dismantle fusion (ONLY if step 4 is PASS/MARGINAL).** Copy `strand_trellis_gemv.metal`
into `dismantle/crates/dismantle-core/shaders/`, `include_str!` it in `metal/mod.rs` +
`all_shader_sources()`. Add `gemv_strand_trellis_pinned_tcb` (mirror
`gemv_q4k_fast_v1_pinned_tcb`), the `gemv_proj!` pre-empt arm, and `ensure_strand_cache` (mirror
`ensure_q4k_fast_cache`), all behind `DISMANTLE_QWEN_STRAND` (default-off ⇒ all golden hashes
unchanged = gate **G0**). Then run the A/B harness (§4 of the integration doc): `decode` suite for
tps, `measure_joules.sh` for J/token, fill the `bandwidth` suite for %peak. Land **G2** (GPU↔CPU
bit-identity parity test in `dismantle/.../tests/`, modeled on `q4k_fast_parity.rs`) — the whole
"bit-identical on phone/WASM/MCU/FPGA" claim reduces to G2 passing.

---

## 3. The blocking discrepancy the hardening pass MUST resolve first

**The two sides do not agree on the v2 wire format, and the dismantle reader will fail to parse a
file the strand writer actually produces.** This is the cross-repo drift the integration doc itself
warned against, and it gates the entire bit-identity chain.

| field | `strand-quant/format.rs` (SHIPPED, tests pass) | dismantle `strand/reader.rs` (scaffold stub) |
|---|---|---|
| magic | **`b"STR2"`** (`MAGIC_V2`, `format.rs:222`) | **`b"STRQ"` + version=2** (`reader.rs:64`) |
| per-block record | **16 bytes** `{bit_offset u64, init_state u32, scale_q i32}` | **32 bytes** `{…, +min_base_q, n, sub_off, pad}` (`reader.rs:72`) |
| page granularity | **4096** (`PAGE`, `format.rs:227`) | **16384** (`STRAND_PAGE`, `reader.rs:69`) |
| `bit_offset` frame | **tensor-payload-relative** (`format.rs:242`) | **absolute / stream-relative** (`reader.rs:149`) |
| side-info (sub_scales/mins) | separate **SIDE-INFO page region** + hoisted `scale_q` in record | `sub_off` index into an inline arena per record |

The `format.rs` schema is authoritative (it compiles, round-trips, and is the leaner kernel
record). **Action: rewrite dismantle `reader.rs` to the `format.rs` schema and have it delegate to
a new `read_strand_v2_header`** (§2 step 1). Until this converges, steps 3–5 build on bytes that
don't match, and G2 is unreachable. _Note: the Metal kernel's `BlockEntry` struct in
`STRAND-metal-kernel-impl.md` §B.3 also lists the 32-byte pre-expanded-`eff[8]` variant — the
kernel record and the writer record must be reconciled in the same pass; decide whether eff-scales
are pre-expanded into the table (kernel-friendlier, fatter) or read from the SIDE-INFO region
(leaner, matches shipped `format.rs`). Pick one and make all three — writer, reader, kernel —
agree._

---

## 4. The single highest-leverage next action

**Reconcile the v2 wire format to the shipped `strand-quant/format.rs` schema and add
`read_strand_v2_header`, then rewrite dismantle `reader.rs` to delegate to it (§2 step 1 + §3).**

Why this one, above even running the gate: every other deliverable — `strand_bake`'s output, the
dismantle reader, the Metal kernel's `BlockEntry`, and the G2 bit-identity gate that *is* the moat
— consumes these exact bytes. Three independent components currently disagree on the byte layout,
so any GPU measurement or parity test built today would be measuring against a file format that
won't load end-to-end. It is pure CPU/Rust, safe to do **right now** mid-sweep (`cargo check` only),
unblocks steps 3–5, and converts the scaffold from "three drafts of a format" into one real format
with a single owner. The M3 gate (the genuinely decisive experiment) should be run on the
reconciled bytes, not the divergent ones.

---

## 5. Risks / TODO carried into hardening (consolidated)

- **RHT-on-activation float determinism** (`STRAND-metal-kernel-impl.md` §B.1): the per-token FWHT
  on `x` is float, NOT covered by the integer-decode guarantee. Weights stay bit-identical; the
  activation transform must be pinned to the encoder's block=256 / row-restart / per-tensor seed or
  `y` silently corrupts. Decide the public determinism claim: "weights bit-identical, activation
  float" (same status as every GGUF GEMV).
- **`acc` width in the kernel** (§B.4): single `uint` accumulator assumes `k≤8`; widen to `ulong`
  if a research config breaks the ≤32-bit OR window. Add a debug assert.
- **Tail-biting init_state baking**: v2 must write `BlockMeta.init_state` unconditionally so the
  kernel skips the sequential pre-scan; the CPU `decode_lean` still re-derives it for tail-bitten
  blocks (advisory in the record) — verify the writer path stores it even when `enc.tail_biting`.
- **Buffer-0 4-byte alignment**: the aligned-`uint*` read path needs each tensor's `w_bits`
  4-byte-aligned; make it a writer invariant, not a kernel branch. (The shipped `format.rs` pads
  payload to a full PAGE, so this holds — confirm in the reconcile.)
- **`bandwidth` suite stub** in dismantle (`suites/bandwidth.rs`, `phase1_pending: true`) — the
  natural home for the ridge measurement; fill it in step 4.
- **`decode_lean_is_bit_identical` has not been RUN** (only compiled) — Step 2 must execute it
  post-sweep; it is the G1 determinism contract.
