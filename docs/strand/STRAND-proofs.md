# STRAND-proofs.md — the proof lane: what is PROVEN vs tested vs assumed

_Created 2026-06-11. Owner files: `crates/strand-quant/src/proofs.rs` (in-crate theorem
module, built under `cfg(any(test, kani))`), `crates/strand-quant/tests/exhaustive.rs`
(from-spec equivalence), this ledger. Scope: the **scalar integer decode path**
(`decode_lean`, `decode_tensor_fixed`, and their shared arithmetic). All timing on this
box is advisory (contended); correctness results are machine-independent._

The decode is integer-only, branch-light, and small — unusually provable. This lane
upgrades it from "tested" to "proven at explicit bounds", with the honest residue named.
Vocabulary used below, strictly:

- **PROVEN** — exhaustive enumeration of the full finite domain, OR a corner check
  completed by a stated monotonicity lemma, OR Kani bounded model checking (symbolic,
  full assumed domain). No sampling counted as proof.
- **TESTED** — property/spot checks that would catch a class of bug but do not cover
  the domain.
- **ASSUMED** — preconditions that depend on input data or on code inspection; each one
  is pinned to a demonstrated boundary case where possible.

---

## 1. Decoder equivalence against the SPEC (tests/exhaustive.rs)

A **clean-room reference decoder** (~60 lines) was written from
`docs/STRAND-format-v2-spec.md` (bit order, payload walk, tail-biting/init_state rule)
plus the `decode`/`encode` module doc-comments (reconstruct/eff-scale/eff-min formulas,
sub-block keying) — *not* from the implementation. Every check is three-way:
`spec-reference == decode_lean == decode_tensor_fixed`. Equality against the reference
means the *documented* format semantics hold, not merely that two decoders agree.

Tensors are constructed directly (symbols packed by hand), so coverage is over **all**
bit-streams, including streams no encoder would emit.

### PROVEN (exhaustive, exact counts asserted in the tests)

| space | coverage |
|---|---|
| (L=4, k=2) | ALL 16 states × ALL 2^(2n) streams, n = 1..6 → 87,360 tensors |
| (L=4, k=3) | ALL 16 states × ALL 2^(3n) streams, n = 1..4 → 74,880 tensors |
| (L=5, k=2) | ALL 32 states × ALL 2^(2n) streams, n = 1..6 → 174,720 tensors |
| (L=5, k=3) | ALL 32 states × ALL 2^(3n) streams, n = 1..4 → 149,760 tensors |
| total non-tail-biting | **486,720 tensors** (budget bound: n·k ≤ 12 per tier) |
| tail-biting, n·k < L | full (state × stream) product (stored init_state is live) |
| tail-biting, n·k ≥ L | ALL streams × stored states {0, 2^L−1}, **plus the independence theorem**: output identical for both stored states (the spec's "init_state is dead under tail-biting" claim, proven over all streams at these L,k,n) |
| 6-bit sub-scale codes | ALL 64×64 code pairs over a 33-weight (partial second sub-block) geometry, × 2 (L,k) corners × 4 scales — the full code space |
| 6-bit affine-min codes | ALL 64×64 code pairs (both polarities by construction) × 4 bases within the proven add-bound |
| per-case scale axis | rotation through {0, ±1, ±2^16, 4096, i32::MAX, i32::MIN} (each (state,stream) bucket sees one; arithmetic totality over the full i32 scale is proven separately in §2, so the rotation is a wiring check, not the load-bearing proof) |

### TESTED (boundary geometries, not exhaustive)

- Multi-block tensors `[256, 256, tail]` for tail ∈ {1,2,5,31,32,33,63,256} × all four
  (L,k) × tail-biting on/off × affine on/off (128 geometries, pseudo-random streams):
  short final block (n·k < L stored-state path), sub-block tails, u32-unaligned total
  bit lengths (the lean word-refill edge).
- f32 wrapper: `decode_tensor == fixed × 2^-12` bit-for-bit at all 8 rotation scales.

### NOT covered (out of scope, named)

- The **vector trellis** (`vec_dim > 1`, `decode_tensor_fixed_with_lut_vec`) — both
  production paths delegate to the *same* function there, so a three-way proof needs a
  from-spec vector reference; not written.
- L ≥ 6 (state spaces grow 2^L; see §5 for what full-L would take), k=4 streams, n
  beyond the per-tier bound, explicit non-frozen LUTs (`--dist`), the encode side.

## 2. Arithmetic theorems (src/proofs.rs)

Domain bounds first, then totality. "Oracle" = i128 recomputation of the exact
mathematical value.

### PROVEN

| theorem | method |
|---|---|
| **Quantile domain bound**: every entry of every frozen LUT (L ∈ [4,14], 65,504 entries) lies in ±Q_CLAMP = ±24576 (6σ·Q12), and each codebook LUT is a permutation of its quantile LUT | exhaustive, count asserted |
| **reconstruct_q totality**: for ALL scale_q ∈ i32 × quantile_q ∈ [−24576, 24576]: the i64 product never overflows (max |prod| = 2^31·24576 < 2^46) and the result fits i32 (max |r| = RECON_ABS_MAX = 805,306,368 < 2^31) and equals the oracle | corner proof + monotonicity lemma (|s·q| = |s|·|q| non-decreasing in each |arg| ⇒ corners bind) + ~20M boundary-sweep pairs + **full 2^32 scale enumeration at the 4 extreme quantiles** (`--ignored`, release, ~16 s measured-advisory 2026-06-11). The Kani harness for this theorem exists but its symbolic 32×15-bit multiplier does NOT converge on this box (documented in `proofs.rs`); the full-domain claim rests on the exhaustive + monotonicity lane, which needs no solver |
| **eff_scale_q totality**: ALL scale_q ∈ i32 × ALL 256 codes: i64 product ≤ 2^37, result fits i32 — the binding corner (i32::MIN, mult 64) → exactly i32::MIN, representable, no wrap. Code aliasing (codes ≥ 64 ≡ code & 63) is part of the proven contract | exhaustive over codes × boundary scales + corners/monotonicity + **Kani symbolic, full domain (no assumption at all)** |
| **eff_min_q totality on the encoder domain** (min_base_q ≥ 0): ALL 256 codes, oracle-exact, \|offset\| ≤ base | exhaustive over codes × boundary bases + **Kani symbolic** |
| **The add-bound implication**: with \|recon\| ≤ 805,306,368 and \|off\| ≤ min_base_q, the decoder's un-widened `recon + off` i32 add cannot overflow iff min_base_q ≤ 1,342,177,279 (conservative; exact tight bound 1,342,177,280 — the positive recon max is 805,306,367 and i32::MIN absorbs one extra). Tightness demonstrated one past the bound | corner arithmetic, both sign pairings |
| **WordBitReader == read_bits, 2-byte buffers**: ALL 2^16 contents × k ∈ 1..8 at offset 0 (through end + 8 zero-pad symbols), and ALL contents × ALL 16 offsets × k ∈ {1,3,8} | unconditionally exhaustive (full input space at this width) |
| **WordBitReader == read_bits, 6-byte buffers**: ALL 2^48 contents × ALL k ∈ [1,8] × ALL offsets < 16, 8 symbols drained (crosses the u32 word boundary and the zero-pad refill) | **Kani bounded model checking** (symbolic) |
| **WordBitReader == read_bits, 16-byte buffers, full (k ≤ 8, offset < 64) grid**: zero buffer + all 128 single-bit basis buffers, every symbol through end + pad | exhaustive basis; completed by the bit-linearity lemma (below) |
| **Empty-buffer zero contract**: full (k, offset) grid | exhaustive (finite, tiny) |

### The one lemma (code inspection, spot-verified)

The 16-byte full-grid claim rests on: *both `read_bits` and `WordBitReader` are
bit-selectors* — every output bit is a copy of exactly one input bit or constant 0
(shifts/masks/ORs of disjoint ranges; no carries). Bit-selectors are GF(2)-linear, and
two linear maps agreeing on a basis agree on all 2^128 buffers. Inspection is the proof
of the lemma; `word_reader_xor_homomorphism` (1,024 structured XOR triples × 4 k × 6
offsets) would catch any carry/arithmetic coupling instantly but is TESTED, not proof.
The Kani 6-byte result is lemma-free and covers the same refill mechanics symbolically.

### Kani status

`cargo kani` installed cleanly (kani-0.67.0). Four harnesses live in
`proofs.rs::kani_harnesses` (`cfg(kani)`); module registered as `cfg(any(test, kani))`.
Run: `cargo kani -p strand-quant --harness <name>`. Verified 2026-06-11 (times
advisory, contended box):

| harness | verdict | time |
|---|---|---|
| `word_reader_matches_read_bits_symbolic` (2^48 buffers × k ∈ [1,8] × start < 16) | **VERIFICATION SUCCESSFUL** | 8.2 s |
| `eff_scale_q_total_symbolic` (full i32 × u8, no assumptions) | **VERIFICATION SUCCESSFUL** | 14.1 s |
| `eff_min_q_total_symbolic` (b ≥ 0 × all 256 codes) | **VERIFICATION SUCCESSFUL** | 403.5 s |
| `reconstruct_q_total_symbolic` | DOES NOT CONVERGE (SAT-hard 32×15-bit symbolic multiplier; 3 formulations killed at 10–35 min). Kept, sound, run explicitly only. The theorem is independently PROVEN by the exhaustive lane (corner+monotonicity + full 2^32 enumeration) — Kani adds nothing load-bearing here | — |

## 3. The preconditions ledger (ASSUMED — each pinned)

| precondition | where it comes from | pin |
|---|---|---|
| `quantile_q ∈ [−24576, 24576]` | frozen LUT clamp (±6σ) | PROVEN exhaustively for the frozen tables; **assumed** for any explicit `--dist` LUT a caller passes — an unclamped custom LUT voids the overflow theorems |
| `min_base_q ≥ 0` | encoder stores the positive magnitude (`choose_affine_min`: `base_abs ≥ 0`, saturating f64→i32 round) | code inspection; out-of-domain behaviour pinned: `eff_min_q(i32::MIN, 0x3F)` **wraps to i32::MIN** (sign flip) — demonstrated in `eff_min_q_i32_min_wrap_is_out_of_domain` |
| `min_base_q ≤ 1,342,177,279` (no overflow in the decoder's `recon + off` i32 add) | equals per-sub-block \|mean\| ≤ ~327,680 in real units; LLM weights are \|w\| < ~100 | data-dependent — cannot be proven from code; the arithmetic implication itself IS proven (§2) |
| `blk.n ≥ 1` per block | encoder never emits empty blocks | not enumerated (a zero-n block decodes to nothing; harmless by inspection, unproven) |
| `lut.len() == 2^L` (scalar) | `codebook_lut` returns the frozen table of exactly that length | masked indexing makes out-of-bounds impossible for the frozen path; a SHORT caller-passed LUT would panic, not corrupt |
| scalar path only (`vec_dim == 1`) | this lane's scope | vector path unproven (§1) |

## 4. What this does and does not claim (no overclaiming)

**The honest statement:** the scalar decode arithmetic is *proven total* over its real
domains (full i32 scales, clamped quantiles, encoder-domain min bases), the lean
bit-reader is *proven equivalent* to the spec reader exhaustively at 16-bit width,
symbolically at 48-bit width, and basis-exhaustively (one inspection lemma) at 128-bit
width — and the two production decoders are *proven equal to the written spec* on every
one of ~1.1M exhaustively enumerated small-L tensors plus the full 6-bit side-info code
spaces. That is **exhaustive-at-small-L + property-at-bounds**, not a full-L functional
correctness proof.

What a counterexample anywhere above would mean: the float-free-decode moat broken
(decode divergence) or UB-class arithmetic on the deterministic path (overflow). None
found; every enumeration count is asserted so silent coverage shrink fails the suite.

## 5. What a full-L proof would take

- **L ∈ [6,14] equivalence**: the state×stream product explodes (2^L·2^(nk)); the
  honest routes are (a) Kani with a symbolic stream and bounded n — the decode loop is
  unwindable, the obstacle is the 2^L LUT as a symbolic array (CBMC handles it, but
  expect minutes-to-hours per (L,n) at L ≥ 10); or (b) a *universal* argument: the
  decoder's per-step function depends on L only through `mask` and `lut` — prove the
  step function once symbolically (state, sym, mask as free variables with
  `mask = 2^L−1`), then the per-L claim follows by induction on n. (b) is the right
  next rung: one Kani harness over a symbolic step, plus a paper induction recorded
  here, would cover ALL L ∈ [4,14] and all n at once.
- **Vector trellis**: needs a from-spec vector reference decoder first (the spec text
  for `d > 1` lives in `trellis.rs` doc-comments; spec-doc coverage is thinner — a
  v3-spec section would make the clean-room exercise meaningful).
- **The encode side** is deliberately out of scope forever (float Viterbi is licensed
  to be non-deterministic; only the emitted integers matter, and those are what the
  decode theorems consume).
- **rustc/LLVM trust**: all of the above proves the Rust source. Bit-identity of the
  *binary* across platforms remains covered by the existing KAT/attestation lane
  (replay sweeps), which is the right tool for that layer.
