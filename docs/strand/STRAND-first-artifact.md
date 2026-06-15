# STRAND first trained-2-bit artifact — `qwen05b-pv2-2bit.strand`

_The dismantle-integration rehearsal: the first deploy-grade `.strand` archive baked from
TRAINED (PV) shadows, mmap-load-verified through the runtime loader, attested with an embedded
SPV3 provenance trailer, and — since the 2026-06-11 honest-artifact re-bake — carrying the
**OUTL sparse-outlier section**, so the archive decodes to the SAME weights the 26.77-PPL
lineage was measured on. Baked by `scripts/bake-attested.sh`._

**RE-BAKE NOTE (2026-06-11, the honest-artifact cluster):** the original same-day bake
emitted STR2 only — the outlier channel was billed but NOT stored, so that artifact decoded
to bulk-only weights that were *not* the 26.77-PPL model (its own caveat 1). This re-bake
closes that: STR2 + OUTL + SPRV v2 (R2 descriptor binding), with an archive-vs-recon
**byte-equality gate** run over all 168 tensors. The model_root is **unchanged** — by
construction: the root hashes the (deterministic) Q12 bulk plane; the new bindings live in
the SPRV v2 per-record descriptor digests.

## What was baked

| field | value |
|---|---|
| model | Qwen2.5-0.5B (896-dim, ragged) |
| shadows | `scratch/qwen-05b/qat-pv2-hf/model.safetensors` |
| lineage | **PV "train through what you ship"**: pv arm 300 steps (78.95 → 27.02, 4 requant boundaries, chunked-KD) → pv2 arm +600 steps, requant-100 → **PPL 26.77 post-requant** (the 0.5B 2-bit PV asymptote; bf16 = 12.55, PTQ-only floor = 80.7) |
| quant config | `--bits 2 --l 12 --outlier-channel 1` (the canon 2-bit reopen config), `STRAND_NO_GPU=1`, CPU SIMD encode |
| archive | `scratch/artifacts/qwen05b-pv2-2bit.strand` — `.strand` v2 (STR2) **+ OUTL section + SPRV v2 trailer**, **RAGGED** (`--ragged-v2`; see caveats) |
| tensors in archive | 168 quantized projection tensors (122 pass-through tensors NOT in STR2 — caveat 4) |
| archive size | 134,493,904 bytes (134.5 MB; was 120.7 MB bulk-only — the +13.8 MB IS the outlier channel) |
| **billed bytes/weight (audit O7: the file is the bill)** | **0.3759 B/w = file_len / 357,826,560 weights** — every byte counted: headers, offset tables, page padding, OUTL, SPRV. (Bulk-only scalar floor 0.292 B/w; the gap = the 1% outlier channel + v2 wire overhead.) |
| outlier channel | OUTL section: **168/168 tensors, 3,578,232 (idx,val@8bit) entries** (1% of weights), canonical integer wire form (`outlier_wire.rs`, shared with strand-delta) |
| **model_root (SPV3)** | `6e1b0e4e36a5aa53ade2b95d372a56eeb96d6572457f50d96e93950db401be54` — **identical to the pre-re-bake root** (Q12 plane unchanged; encode determinism re-confirmed across bakes) and now ALSO stored + self-verified in the embedded SPRV v2 trailer |
| source_sha256 | `cd0c53558c90789d…` (full hash in the attestation transcript) |

The attestation transcript lives next to the archive:
`scratch/artifacts/qwen05b-pv2-2bit.attestation.txt` (+ `.quant.log`, `.recon.log`,
`.attest.log` with every per-tensor root).

## How it was verified (the attest stanza)

`crates/strand-decode-kernel/src/bin/attest-strand.rs` — runs the exact consumer path:

1. **mmap load** via `StrandModel::open` (the runtime loader; all parsing delegates to
   `format::read_strand_v2_header` + `outlier_wire::read_outl_bytes`, the single wire-format
   owners).
2. **Billing headline**: `file_len / n_weights` printed from the artifact itself.
3. **SPRV v2 self-verification**: `verify_archive(Vectors)` — re-decodes the self-test
   vector blocks AND re-derives every tensor's **descriptor digest** (name, shape, rht_seed,
   l/k/d, flags, block_len, total, outlier-channel digest) from the live descriptors,
   comparing against the stored records. A flipped `rht_seed` byte, resized shape, or
   tampered outlier channel now FAILS verification (the audit's R2 hole, closed; tamper KATs
   in `provenance_io.rs::sprv_v2_descriptor_tamper_detection`).
4. **Decode spot check** on 3 tensors (first/middle/last): `encoded_tensor_checked` →
   `decode_tensor_fixed_with_lut` asserted **bit-identical** to `decode_lean_with_lut`
   (the determinism law gate — no number is printed before it passes).
5. **THE O1 GATE — `--recon-check`**: every archive tensor decoded through the patched path
   (`outlier_mac::patched_weights`: bulk Q12 → ×1/4096 → inverse row-aware RHT with the
   stored seed → outlier replacement) and **byte-compared against the recon safetensors**
   quantize-model wrote from the same shadows + config. **PASS, all 168 tensors** —
   archive-only decode IS the recon, bit for bit.
6. **Provenance**: per-tensor SPV3 roots in file order → `model_root_from_tensor_roots`,
   asserted equal to the SPRV-stored root.

## How a runtime consumes it

- **Full weights**: `outlier_mac::patched_weights(model, name)` — the corrected
  weight-space tensor (what the PPL was measured on).
- **GEMV**: `outlier_mac::matvec_patched` (materialized reference, bit-pinned MAC order) or
  `outlier_mac::matvec_rht` (dense bulk path on the RHT-side activation + sparse outlier
  add using `outlier_residuals` precomputed at load — the `y = W_bulk·x + Σ resid·x[col]`
  form; note the residual subtlety in `outlier_mac.rs`'s module doc).
- **Activation-RHT warning (measured)**: the one-rotation recipe previously documented in
  `shaders/README.md` (`x_rht = rht_forward(x, seed)` once per GEMV) is **wrong for every
  row after the first** — the encoder's Rademacher signs are drawn from the GLOBAL flat
  index, so each weight row is rotated by a different signed Hadamard. The divergence is
  pinned by `outlier_mac::tests::single_rotation_recipe_diverges_for_multirow`;
  `matvec_rht` implements the correct per-row form. Integrators (dismantle first) must use
  the API, not the README sketch.

Per `docs/STRAND-dismantle-wiring.md` (the executable recipe — apply by hand in
`~/Downloads/dismantle`, this repo never edits there): the reader delegates to
`read_strand_v2_header`, and everything stays additive + default-off behind
`DISMANTLE_QWEN_STRAND`. **Attestation contract:** any consumer that decodes this archive's
bulk plane to Q12 must reproduce the model root bit-for-bit, and `verify_archive` must pass
— a mismatch means a broken decode path or tampered metadata, full stop.

## Honest caveats (current)

1. **RAGGED archive, not STRICT.** The 0.5B's 896-dim tensors violate the STRICT
   `in_features % 256 == 0` invariant, so the `ALL_STRICT` flag is clear. The CPU loader path
   handles ragged fine; the Metal GEMV's linear `(row, block)` map assumes STRICT — GPU
   consumption of this specific artifact is not claimed. A 7B bake (256-aligned) would be STRICT.
2. **Quality context:** PPL 26.77 is the recon-plane number from the PV eval at 64-window
   screening; 2.13× bf16 (12.55). It is the best trained 2-bit on this box, not a claim of
   parity. An archive-only PPL eval was not re-run — it is now *provably redundant*: the
   archive decode byte-equals the recon the 26.77 was measured on (gate 5 above).
3. **Attested weights ≠ attested inference.** The SPRV roots cover the decoded weights; the
   float MAC/activation path is ordinary float GEMV (the integer-activation gate experiment
   is the open upgrade path).
4. **Pass-through tensors (embeddings, norms, lm_head) are not in the archive** — STR2 carries
   the quantized projection tensors only; a full inference deployment pairs the archive with
   the original non-projection tensors (dismantle already loads those from its own model file).
5. **Vector-trellis rungs (1.5/1-bit) still have no archive format** — the learned-LUT
   region remains v3/STR3 work (audit formats O2).

_Removed caveats (closed by the re-bake): "the outlier channel is NOT in the archive"
(O1 — it is now, with a byte-equality gate); "archive version is v2 + an out-of-band root"
(O3 — the SPRV v2 trailer is embedded and self-verified, with R2 descriptor binding)._
