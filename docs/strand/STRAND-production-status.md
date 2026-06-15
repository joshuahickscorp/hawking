# STRAND — production status ledger

_Snapshot 2026-06-09, branch `sub4bit-innovation`. The honest "what runs today" record for the
deploy path: cloud **bake** → on-device **load** → integer **decode** → **GEMV**, plus the Metal
GPU harness. Every claim below was re-verified against the working tree (commands re-run after the
release sweep terminated), not trusted from the wave summaries. Where something only `cargo check`ed
versus actually executed, it says so._

> Scope note: this ledger covers the **runtime/deploy plumbing** built in the production wave
> (`strand-quant` v2 emit + `strand-decode-kernel` loader/gemv/metal/e2e). It does **not** re-litigate
> the quant *quality* story (sub-4-bit R-D curve) — that lives in `STRAND-product-spec.md` and the
> research playbook, and is unaffected by this wave. The two mixed-precision rows in the product spec
> remain RETRACTED/PROVISIONAL (commit `f2f7716`); nothing here depends on them.

---

## 0. TL;DR

The end-to-end **deploy mechanism is real and green on this machine**: you can quantize a
safetensors model to a page-aligned `.strand` (STR2) archive with a provenance hash, mmap-load it
through the runtime crate, and decode weights to Q12 **bit-identically** on both CPU and the Apple
M3 GPU. What is *not* done is the part that needs resources this session didn't own: the **Metal
performance MEASUREMENT** (correctness is proven, speed is not), the **dismantle wiring** (recipe
written, not applied — needs the dismantle repo to settle), and shipping a real 7B archive end-to-end.

---

## 1. PRODUCTION-READY now (verified compiling AND tested)

Everything in this section was confirmed on 2026-06-09 after the release sweep was confirmed dead
(`pgrep quantize-model` → empty, `pgrep cargo` → empty). Commands and their real output are quoted.

### 1a. Bake — `.strand` v2 (STR2) emit from `quantize-model`
- **What:** `--packed-v2-out <path>` on `crates/strand-quant/src/bin/quantize-model.rs` quantizes every
  linear weight, then writes the **STR2** deploy archive: per-tensor page-aligned (`PAGE=4096`)
  block-offset table + a `source_sha256` of the entire input `.safetensors` file, then stops (`--out`
  not required). STRICT-by-default (`strict_v2 = true`); `--ragged-v2` / `--no-strict-v2` relaxes it
  for odd-dim models (e.g. 896-dim 0.5B).
- **New code:** dependency-free FIPS 180-4 SHA-256 at `crates/strand-quant/src/sha256.rs`
  (`pub fn sha256(&[u8]) -> [u8;32]`), `pub mod sha256;` in `lib.rs`, and the v2 emit block in
  `quantize-model.rs` (`SafeTensors::bytes()` accessor, `Args.{packed_v2_out,strict_v2}`, `TensorResult.block_len`,
  the `PackedTensorV2` build loop + `write_strand_v2` call at ~line 1287). The v2 writer/reader
  (`write_strand_v2`/`read_strand_v2`/`read_strand_v2_header`, `MAGIC_V2=b"STR2"`) already lived in
  `format.rs` from the prior architecting wave; this wave only added the *emit caller*.
- **Compiles:** `cargo check -p strand-quant` → `Finished dev profile … in 0.04s`, zero errors, zero warnings.
- **Tested (the deferred suite, now RUN):** `cargo test -p strand-quant --lib` →
  **`78 passed; 0 failed`** (110.4s — this runtime is exactly why it was deferred while the sweep held
  the cores). By name, the new/relevant tests are green:
  - `sha256::tests::{nist_abc, empty, nist_two_block, exact_block_boundary, first8_of_abc_matches_digest_prefix}`
    — NIST vectors + the cross-tool digest-prefix lock (`digest[0..8]` == dismantle `awq_bake`'s `first8`).
  - `format::tests::{strand_v2_round_trip_matches_v1_q12, strand_v2_header_matches_full_read,
    strand_v2_strict_rejects_ragged_in_features, strand_round_trip_decodes_identically}`.
- **Caveat:** the bake was **not executed on a real 7B** this session. The v2 path is proven by the
  format round-trip tests + the kernel-crate e2e test (§1e), not by a multi-GB production run. Vector /
  salient tensors are still silently skipped in v2 (mirrors v1; `panic!` only if *all* tensors are
  non-scalar) — the in-format LUT for those is a future addition.

### 1b. Load — v2 mmap loader (`strand-decode-kernel::loader`)
- **What:** `crates/strand-decode-kernel/src/loader.rs` — `StrandModel::{open, from_mmap, header,
  tensor_names, tensor_header, view, config_for, encoded_tensor, encoded_tensor_checked}`, `TensorView<'a>`,
  pure `encoded_tensor_from_view`. `open` delegates to the canonical `read_strand_v2_header`; the
  per-tensor side-info walk reproduces tested `read_strand_v2` logic against a single tensor's mmap view.
- **Depends on:** `memmap2 = "0.9"` (the only new lock package; `Cargo.lock` memmap2 = 0.9.10).
- **Tested:** part of the 13/13 below — `loader::tests::{open_round_trips_header,
  view_slices_match_payload, encoded_tensor_decodes_identically_to_v1, encoded_tensor_with_affine_min_round_trips}`
  all pass.

### 1c. Decode — integer Q12, CPU (`strand-decode-kernel::gemv` + the original `lib.rs`)
- **What:** `gemv::decode_tensor_q12` reuses `strand_quant::decode::decode_lean` (the aligned-read lean
  decoder) off the mmap; `lib.rs::decode_weights_q12` wraps the reference `decode_tensor_fixed`. Both are
  pure integer (`reconstruct_q = (Q16·Q12)>>16`), no float on the decode path.
- **Tested:** `gemv::tests::decode_tensor_q12_matches_decode_lean` proves `decode_lean` ≡ reference; the
  e2e test (§1e) proves the loader-reconstructed `EncodedTensor` and the production `decode_tensor_q12`
  path both equal `decode_tensor_fixed` bit-for-bit on a real on-disk archive.

### 1d. CPU GEMV (`strand-decode-kernel::gemv::matvec_named`, `lib.rs::matvec`)
- **What:** fused decode→`y = Wx` (Q12 weights × `1/4096` × x). Integer decode + float MAC.
- **Tested:** `gemv::tests::{matvec_named_matches_lib_matvec, matvec_named_rejects_bad_x_len}` +
  `tests::matvec_matches_manual_decode`. Green.
- **Speed (reference only):** the e2e report measured **41.2 Mweights/s** for `decode_tensor_q12` over
  4 synthetic tensors (M3 Pro). This is the **scalar correctness baseline**, explicitly not a SIMD/GPU
  fast path — do not quote it as throughput.

### 1e. End-to-end test (`strand-decode-kernel::e2e`)
- **What:** `e2e::tests::e2e_v2_runtime_decode_is_bit_exact` builds a synthetic 4-tensor model
  (mixed bpw 3/2/4), encodes, writes a **real `.strand` v2 to a tempfile**, mmap-loads through the
  runtime loader, and asserts bit-exact Q12 via two independent runtime paths vs `decode_tensor_fixed`
  + full header round-trip. `e2e_footprint_bytes_per_token_report` checks the footprint story
  (3 bpw = 5.33× smaller than bf16, monotone in bpw).
- **Tested:** both green. Reported `1.364 bytes/weight` on the tiny 4-tensor archive — inflated by
  page padding + per-tensor headers at 45 K weights; the footprint test confirms the asymptote
  (7B @ 3 bpw ≈ 5.3× smaller than bf16). Not a production-archive overhead number.

### 1f. Metal kernel — COMPILES, and CPU↔GPU bit-identity PROVEN (correctness only)
- **What:** `crates/strand-decode-kernel/src/metal.rs` (macOS-only) — `StrandGpu::{new,
  gpu_blockentry_sizeof, gemv_fused}`, `bake_block_entries`, `gpu_matvec_named`. Compiles
  `shaders/strand_trellis_gemv.metal` via `include_str!`, bakes the GPU-side `BlockEntry` from the
  on-disk 16-byte `BlockOffsetRecord` + side-info, dispatches one threadgroup/row × 256 threads.
  Depends on `metal = "0.27"` + `objc = "0.2"` (same pin as `strand-quant` → zero metal/objc lock churn).
- **Tested ON THE REAL GPU (Apple M3 Pro) — these are RUN, not skipped:**
  `cargo test -p strand-decode-kernel --lib` →
  ```
  test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
  ```
  including the two GPU-gated tests:
  - `metal::tests::gpu_q12_matches_cpu_decode_lean` — first asserts the GPU's own
    `sizeof(BlockEntry)` (read via a probe kernel) equals the host `repr(C)` size (so the
    `tbl[gid*bpr+b]` stride cannot diverge), then drives the shipped fused kernel with a one-hot
    `x_rht` and asserts `round(y*4096) == decode_lean[...]` for every probed (row,col) at bpr=1
    (256 cols) and bpr=2 (512 cols).
  - `metal::tests::gpu_y_matches_cpu_matvec` — full GPU `y` vs CPU `matvec_named` within 1e-4 rel
    (RHT off).
  - **Confidence:** these tests self-skip (`return`) when `Device::system_default()` is `None`. They
    did **not** skip — they passed on the M3 Pro (the wave summary plus this re-run both show 13 real
    passes with no "no Metal device" path taken).
- **`kernel-bench` binary builds:** `cargo build -p strand-decode-kernel --bin kernel-bench` → `Finished`.

---

## 2. PENDING — and exactly why

| # | Item | Status | Blocker / why pending |
|---|------|--------|----------------------|
| P1 | **Metal performance MEASUREMENT** (bandwidth / roofline / Q4_K head-to-head) | NOT run | This is the make-or-break gate, and it needs the **GPU free and undisturbed** to produce trustworthy timings. The wave proved GPU *correctness* (§1f) but explicitly deferred the timing harness in `shaders/README.md` §1–§4 / `STRAND-metal-decode-gate.md`. **Risk:** STRAND decode could be compute-bound at low bpw and lose the bandwidth advantage to Q4_K — must be measured before any "faster on device" claim. |
| P2 | **Apply the dismantle wiring** (`StrandTrellis` `WeightKind`, `gemv_proj!` pre-empt, loader, parity test) | Recipe written, **not applied** | Needs the **user's `dismantle` repo to settle** — it had uncommitted work and a live sweep; dismantle was correctly left untouched this session. The full copy-pasteable recipe is `docs/STRAND-dismantle-wiring.md` (57 KB) with per-step `cargo check/test -p dismantle-core` gates. One real bug pre-fixed in the doc: the scaffold's `tools/strand_bake/src/main.rs` returns a `u64` hash where `write_strand_v2` wants `[u8;32]` — would not compile as-is. |
| P3 | **`cargo test -p strand-quant`** | ✅ **NO LONGER PENDING** — RUN this session | Was deferred while the release sweep held the cores (the suite is 110s). Sweep is now dead; **all 78 lib tests pass** (§1a). This row is closed. |
| P4 | **Ship a real model end-to-end** (bake a 7B → `.strand` → load → decode → token) | NOT done | No production bake executed; the v2 path is proven by format round-trips + the synthetic e2e (§1e), not a multi-GB run. First real bake is step 4 of the checklist. |
| P5 | **Activation RHT in the GEMV caller** | NOT wired | `gemv::matvec_named` / `metal::gpu_matvec_named` do integer decode + float MAC only. A real GEMV must compute `x_rht = rht_forward(x, seed, block=256, row-restart)` per tensor. The correctness tests sidestep RHT (one-hot integers / RHT-off tensors). API confirmed present: `strand_quant::rht::{RhtConfig, RhtConfig::from_seed, rht_forward}`. Note: `seed_for_name` does **not** exist — the baker must read the seed the encoder stored, not recompute via a helper. |
| P6 | **Affine-min (>3-bit) on GPU** | NOT done | `bake_block_entries` `assert!(!has_affine_min)` + `assert_eq!(vec_dim, 1)` → today's GPU path is the **3-bit deploy point only**. A 4-bit GPU deploy needs `off[8] = eff_min_q(...)` on `BlockEntry` + a kernel `+ e->off[j>>5]`; the vector trellis needs a `d`-dim kernel. CPU decode already handles affine-min (`encoded_tensor_with_affine_min_round_trips` passes). |
| P7 | **Stale shader doc: `sizeof(BlockEntry)` says 80, actually 52** | Known, **not fixed** here | `strand_trellis_gemv.metal:54,65` and `shaders/README.md:59` claim 80 B / 16-byte align + `static_assert(sizeof==80)`. The **real** MSL/`repr(C)` size is **52 B** (max field align 4). The runtime is correct (matches the real 52 B and self-verifies via the probe assertion), but the comments would mislead the next host implementer. Flagged as background task `task_0404ffea` (comment-only fix; do not change field order/types). |

Smaller flagged risks (in-code, non-blocking): `block_len as u32` cast (safe for all real configs ≈256);
`metal` 0.27 (strand) vs 0.29 (dismantle) skew, deliberately unreconciled across repos; the `predec`
kernel compiles but is not dispatched (`#[allow(dead_code)]`).

---

## 3. The single ordered "to ship" checklist

Each step has a concrete gate. Steps 1–2 are this-repo and can be done now; 3+ depend on the GPU
being free / the dismantle repo settling.

1. **Land this wave on a branch.** Commit the additive runtime + v2 emit (`strand-quant`:
   `sha256.rs`, `lib.rs`, `quantize-model.rs`; `strand-decode-kernel`: `loader.rs`, `gemv.rs`,
   `metal.rs`, `e2e.rs`, `lib.rs`, `Cargo.toml`; `Cargo.lock`). **Gate:** `cargo check -p strand-quant`
   + `cargo test -p strand-decode-kernel` (13/13) + `cargo test -p strand-quant --lib` (78/78) — all
   already green on this tree.

2. **Fix the shader doc defect (P7).** Comment-only: `80 → 52`, drop the false "16-byte alignment" /
   `static_assert(==80)` language in `strand_trellis_gemv.metal` + `shaders/README.md`. **Gate:**
   `cargo test -p strand-decode-kernel --lib gpu_` (still 2/2 — the probe assertion is the real
   source of truth and is unaffected).

3. **Run the Metal performance gate (P1).** With the GPU idle, run the `shaders/README.md` §1–§4
   bandwidth / roofline / Q4_K head-to-head on the M3. **Gate / GO-NO-GO:** STRAND 3-bit decode+GEMV
   must be bandwidth-bound and beat Q4_K's bytes/token at iso-quality. If it's compute-bound and
   loses, the GPU deploy stops here and the value is CPU/WASM/MCU bandwidth only. *This is the gate
   that decides whether the GPU story ships at all.*

4. **Bake one real model end-to-end (P4).** Quantize a settled model to v2:
   ```
   cargo run -p strand-quant --release --bin quantize-model -- \
     --in <model.safetensors> --bits 3 --packed-v2-out <model>-strand-q3.strand
   ```
   (add `--ragged-v2` for odd-dim models). **Gate:** the run ends with
   `wrote N tensors -> … (v2/STR2, strict=true, … bytes/weight, src_sha256[0..8]=…)`, and a CPU
   `StrandModel::open` + `decode_tensor_q12` round-trip on the produced file matches a reference
   `decode_tensor_fixed`.

5. **Wire the activation RHT into the GEMV caller (P5).** Add per-tensor `x_rht = rht_forward(x, …)`
   ahead of `matvec_named` / `gpu_matvec_named`, reading the encoder-stored seed (not a helper).
   **Gate:** a new test asserting full `y` (RHT on) vs a CPU reference within tolerance, with
   determinism pinned across two runs.

6. **Apply the dismantle recipe (P2)** once the dismantle repo is clean. Follow
   `docs/STRAND-dismantle-wiring.md` step-by-step (`StrandTrellis` `WeightKind`, hoist the STRAND
   pre-empt above the `match $tref.dtype` in `gemv_proj!`, the loader mirroring `ensure_q4k_fast_cache`,
   the G3 staleness check using the **full 32-byte** `source_sha256`). **Gate:** each step's
   `cargo check/test -p dismantle-core`, ending with the ported GPU↔CPU bit-identity parity test
   (the dismantle analog of `gpu_q12_matches_cpu_decode_lean`).

7. **Add affine-min to the GPU path (P6)** only if a >3-bit GPU deploy is needed — `off[8]` on
   `BlockEntry` + kernel offset add. **Gate:** a 4-bit GPU Q12 bit-identity test (mirrors the 3-bit one).

---

## 4. One-line verdict

**Bake / load / CPU-decode / CPU-GEMV / Metal-compile-and-correctness are PRODUCTION-READY and green
(78 + 13 tests pass on M3 Pro).** The gating unknown is the **Metal performance measurement** (step 3)
— correctness is proven, speed is not — followed by the first real-model bake (step 4) and the
dismantle wiring (step 6). The deferred `strand-quant` test debt is paid off; the only known stale
artifact is the cosmetic shader byte-size comment (step 2).
