# STRAND → GGUF / llama.cpp Bridge — DECISION DOC

Status: **APPROVED — SAFE-WITH-GUARDRAILS** (build only after the 6 mandatory guardrails below are wired)
Scope: one-way, post-attestation, dequantize-to-bf16 exporter in its own leaf crate.
Audience: moat review → implementation.
Date: 2026-06-13.

This doc LEADS with the moat verdict and the non-negotiable guardrails (the owner's #1 concern),
THEN gives the design, THEN the explicit "what is lost / how it stays separate from the
deterministic ship lane" boundary. Every load-bearing code claim below was verified read-only
against the tree (file:line cited).

---

## PART A — MOAT VERDICT (read this first)

### A.0 Verdict

**SAFE-WITH-GUARDRAILS. Build it.** Not KILL.

The bridge is *structurally incapable* of cannibalizing the deterministic format at the code
level — that protection is the strongest kind available in a Rust workspace (dependency-graph +
type-system, not reviewer vigilance). KILL would be wrong: refusing any on-ramp cedes ecosystem
reach for **zero** protection benefit, because the firewall already makes the moat untouchable by
the bridge.

The entire residual risk is at the **product / defaults / framing** level — *plus one concrete
code hole* (the seal check is too shallow, §A.2 G1). It is NOT at the architecture level.

### A.1 Why the architecture is moat-safe (verified, not asserted)

| Claim | Verified against tree |
|---|---|
| Single reconstruction contract, no second decoder to drift | `patched_weights` at `crates/strand-decode-kernel/src/outlier_mac.rs:10` is the exact fn `archive-to-safetensors.rs:24` already calls. |
| Producer/attestor lane is the dependency *root* (can't reach a leaf) | `crates/strand-quant/Cargo.toml` has **no workspace deps** — `write_strand_v2`, `append_outl`, `append_sprv_computed` all live here and depend on nothing in-tree. |
| `strand-decode-kernel` is downstream-only | `crates/strand-decode-kernel/Cargo.toml`: `strand-quant = { path = "../strand-quant" }` (one-way). |
| Media lane is separate | `strand-container` depends on `strand-core`, not the quant lane. |
| A leaf `strand-bridge` is therefore a TRUE sink | Nothing in the ship lane can `use strand_bridge::*` — contamination is a **compile error**, not a policy. |

Because the moat (`will.md` §2/§9: *density × determinism × float-free portability*; pure-integer
`state → LUT → (scale_q × q) >> 16` decode, GPU==CPU bit-exact) is defended by the dependency
graph, the bridge cannot touch it even if a future contributor wanted it to.

### A.2 — THE GUARDRAILS (mandatory, concrete, unmissable)

These are gate-blocking. The bridge does not ship until **all six** are wired and CI-enforced.

---

#### G1 — VERIFICATION HOLE — FIX BEFORE ANY EXPORT SHIPS  *(highest priority)*

**The bug in the original design:** §B says the front door calls
`require_sealed()` → `verify_archive(VerifyDepth::Vectors)`. **`Vectors` is a SAMPLED spot-check, not a full verify.**

Verified at `crates/strand-quant/src/provenance_io.rs`:
- `VerifyDepth::Vectors` (line 574-580) only runs `verify_test_vectors` over the *stored* sample
  vectors. It does **not** recompute every block, and it does **not** recompute `tensor_root`.
- `VerifyDepth::Full` (line 582-606) recomputes **all** `block_hashes`, checks them against the
  full leaf list, AND recomputes `tensor_root_from_hashes(...) == rec.tensor_root` ("decoded
  weights are not the committed weights").
- `attest-strand.rs:103` itself self-verifies with only `Vectors` and relies on a *separate*
  external `--recon-check` for full coverage.

So as originally specified, the stamp's `strand.source_model_root` would assert a fidelity the
bridge **never actually checked** — a corrupt block *outside the sample* would sail through and
circulate carrying a model_root pointer. That dilutes the "provable" brand exactly where the
brand IS the product.

**FIX (both halves required):**
1. `require_sealed()` MUST call `verify_archive(path, VerifyDepth::Full)` —
   `crates/strand-quant/src/provenance_io.rs:582` — not `Vectors`.
2. The exporter MUST **re-derive `tensor_root` (and `model_root`) over the EXACT dense bytes it
   is about to write**, and refuse to emit if they disagree with the sealed roots. Only then is
   "verified-faithful derivative of a signed model" a *true* statement and the §C.5 stamp honest.

---

#### G2 — NEVER THE DEFAULT, NEVER IN THE SHIP BINARY

- The bridge is a **separate binary** (`strand-export`) in a **separate leaf crate**
  (`strand-bridge`), invoked explicitly.
- It is **NOT** a subcommand of `quantize-model` (the producer:
  `crates/strand-quant/src/bin/quantize-model.rs` — calls `write_strand_v2`:1148,
  `append_outl`:1168, `append_sprv_computed`:1213) and **NOT** on the main `strand` CLI primary path.
- **Delete `archive-to-safetensors.rs` only AFTER** lifting its 58-line body into the crate, so no
  ad-hoc float exporter lingers in the tree.
- **CI hard gate:** a `cargo metadata` assertion proving **no dependency path** from
  `strand-quant` / `strand-container` producer targets into `strand-bridge`. The firewall is
  enforced mechanically, not by reviewer vigilance. (LOC/density pressure — `will.md` §5.13 —
  creates a standing temptation to "simplify" by merging the bridge into an existing binary; this
  gate is what stops it.)

---

#### G3 — INDELIBLE PROVENANCE + LOUD, NON-SUPPRESSIBLE DOWNGRADE BANNER

- Every output (GGUF KV + safetensors `__metadata__`) carries: `strand.derived=true`,
  `strand.attestable=false`, `strand.source_model_root=<hex>`, and an explicit `strand.lost=...`
  string. Full block in §C.5.
- The CLI prints a **non-suppressible** banner on every run naming what is surrendered:
  bit-identical float-free decode, cross-device determinism, in-file attestation, uniform-bpw density.
- **All docs frame this as "try-via-GGUF, then upgrade-to-deterministic."** The bridge is an
  adoption on-ramp and demo/interop tool, explicitly **NOT a deployment target**. Native
  float-free decode (dismantle / `strand-decode-kernel`) is the **only** production path.

---

#### G4 — GUARD THE "STRAND == GGUF QUALITY" FALSE IMPRESSION

- The `gguf-requant` tier (Q4_K_M etc.) is **double-quantization** (STRAND loss + K-quant loss
  *stacked*) AND re-inherits the 896-dim tiling tax STRAND was built to avoid (verified:
  `docs/STRAND-vs-gguf-isobpw.md` — "Q2_K is not 2-bit here — it is 4.20 bpw on the projection
  weights"). It carries the **loudest** warning and is **never** the suggested tier.
- **Any** published benchmark of a bridged GGUF MUST state it is a downgraded derivative scored
  through llama.cpp's float path — **never** presented as STRAND's own number. The honest frame is
  already established (`docs/STRAND-vs-gguf-isobpw.md`: iso-bpw, same harness, GGUF dequantized
  through the SAME eval). **A bridged-GGUF PPL must never leak into STRAND's headline canon**
  (`will.md` §3). This is doubly dangerous because the "smaller/better-than-GGUF on PPL" thesis is
  already disproven ~5× (`will.md` §4 / MEMORY); a leaked number re-invites the exact head-to-head
  STRAND was repositioned away from.

---

#### G5 — INPUT MUST BE SEALED (keep the default strict)

- Default: **refuse unsealed archives** (front door = G1's Full verify).
- If `--allow-unsealed` is offered at all, it MUST: (a) require a second, even-louder stamp
  `strand.source_attested=false`; (b) be **absent** from every quick-start/demo doc; and (c) still
  run a full integer-decode self-consistency pass (`decode == decode_lean`, per `will.md` §5.2) so
  an unsealed export is at least internally bit-consistent. Exporting an unattested archive that
  then circulates as "a STRAND model" is the single fastest way to dilute the provable brand.

---

#### G6 — CARRY-THROUGH CORRECTNESS (non-moat but ship-blocking)

STRAND only quantizes the 7 projection matrices; embeddings / norms / biases are **not** in its
quantized set (confirmed `will.md` §7 + isobpw doc). A loadable-but-wrong GGUF (missing/mismatched
non-quantized tensors) would produce broken outputs **attributed to STRAND**. Therefore:
- Source the non-quantized tensors from the **original HF dir** (the bridge takes both the
  `.strand` and the base HF dir, exactly like `tools/gguf/gguf_to_hf.py` takes a `base_hf_dir`).
- **Reuse the existing reverse name map** inverted —
  `tools/gguf/gguf_to_hf.py::build_reverse_map` (verified present, line 28). Do **not** reinvent
  tensor naming.

### A.3 Moat-erosion risks the guardrails are answering (so we don't forget *why*)

1. **Cannibalization (top risk, BEHAVIORAL not architectural).** Users take the one-command GGUF
   path, run it in tooling they already have, and never adopt the deterministic native decoder — at
   which point STRAND's whole reason-for-being is invisible *to that user*. The code firewall does
   nothing here; only **defaults + framing + making the deterministic path the obviously-better
   destination** defend it. → G2, G3.
2. **Brand dilution via false-fidelity stamp.** The shallow `Vectors` check lets a subtly-wrong
   tensor circulate under a `model_root` pointer. → G1.
3. **"STRAND == GGUF quality" misimpression.** Every bridged GGUF runs through llama.cpp's float
   decode and (requant tier) is double-quantized + pays the 896-dim tax. → G4.
4. **Determinism-contract erosion by proximity.** `will.md` §5.2 makes "every new decode path
   asserts bit-identical vs `decode_lean` before any number is reported" sacred. A float exporter in
   the tree normalizes a non-bit-exact path; it must never drift toward the producer. → G2 (no
   write-API, CI dep gate), G5 (self-consistency even when unsealed).
5. **Maintenance/support burden competing with the core.** A GGUF on-ramp invites a long tail of
   ecosystem-compat churn (convert script version drift, tensor-naming drift, quant-type requests)
   that pulls time from the actual frontier (PV/QAT 2-bit, `will.md` §8). The surface is small (we
   already carry the upstream script for the reverse direction) but the **support expectation** is
   not. → scope as **best-effort interop, unsupported for production** (G3 framing).
6. **Reverse-contamination (low, named for completeness).** A bridged GGUF re-ingested via
   `quantize-model` goes through a full fresh RHT+trellis+attest and carries none of the original's
   roots — so there is no "trust the GGUF's STRAND-ness" path. The only residual is **human**:
   `strand.source_model_root` is just a *pointer*; verifying it still requires the original sealed
   `.strand`. Docs must say so, so the one-way pointer is never mistaken for portable attestation.

---

## PART B — DESIGN

### B.1 What the bridge must reproduce (the reconstruction contract)

The canonical "tensor → dense weights" path already exists and is proven byte-exact against the
encoder recon (`patched_decode_byte_equals_recon_path`, `outlier_mac.rs`). It is
`crates/strand-decode-kernel/src/outlier_mac.rs:10` `patched_weights`:

1. `decode_q12_fast(enc, cfg)` → `Vec<i32>` Q12 fixed-point (pure-integer trellis:
   `state → codebook_lut[state]`, then `reconstruct_q(eff_scale_q(scale_q, sub_scale), q) + eff_min`,
   all `i64`/`i32`; `crates/strand-quant/src/decode.rs`).
2. `w = q12 * (1.0/4096.0)` → `f32` (the single, well-defined Q12→f32 step; `QUANTILE_SHIFT = 12`)
   — `outlier_mac.rs:18`.
3. If `has_rht_seed`: `rht_inverse_rows_inplace(...)` back to weight space
   (`crates/strand-quant/src/rht.rs`) — `outlier_mac.rs:20-24`.
4. If an `OUTL` channel exists: overwrite `w[i] = dequant_val` per stored outlier (sparse
   high-magnitude side-channel) — `outlier_mac.rs:26-37`.

**The bridge reuses `patched_weights` verbatim** — the same fn `archive-to-safetensors.rs:24`
calls. No second decoder, no chance to diverge from the attested weights: the exporter's input is
the exact dense tensor the attestation chain signs.

Review note: steps 1–2 are bit-identical on every device (the moat). Step 3 (inverse RHT) is f32
no-FMA IEEE-754, bit-exact only for even-power-of-2 RHT block widths; odd-power widths (e.g.
896→128) are approximate ~1e-6 (wave wu9rh330z). This is *already* true of STRAND's own f32 weight
view — the bridge does not introduce it — but it matters for §C item 6: in llama.cpp everything is
float from dequant onward, so that approximation lives inside a fully-float pipeline with no
integer anchor.

### B.2 The mapping decision — bf16, NOT a GGUF quant type

Two candidates:

- **(A) STRAND-trellis → a native GGUF quant type** (re-encode STRAND's symbol stream into
  `block_q*`: f16 block scale + packed quants; K-quants are 256-element superblocks, legacy are
  32-element).
- **(B) STRAND → dense bf16 → standard GGUF** (dequant via `patched_weights`, write clean
  HF-shaped safetensors, then run the *unmodified* upstream
  `tools/gguf/convert_hf_to_gguf.py`).

**Choose (B).** Reasons:

- **(A) destroys the reason STRAND exists and gains nothing.** STRAND's value is
  density-at-uniform-bpw-on-any-dim + integer determinism. A GGUF `block_q` cannot represent the
  trellis (state-machine codebook + sub-scales + tail-biting + RHT); you'd have to *re-quantize*
  the dequantized weights into K-quant's own scheme — i.e. (B) with extra steps and **worse**
  quality. There is no structure-preserving embedding: different codecs, not different encodings of
  one codec.
- **The 896-dim edge proves it.** `docs/STRAND-vs-gguf-isobpw.md` shows K-quants *silently fall
  back* to 32-blocked legacy on 896-dim tensors (Q2_K → 4.20 bpw actual). Forcing STRAND into a
  GGUF quant type **inherits that tiling tax** — the exact weakness STRAND avoids.
- **(A) also requires patching llama.cpp's C++ kernels** (a `GGML_TYPE_STRAND` dequantizer): a
  hostile fork, float and non-deterministic anyway (carries none of the moat), needing
  upstreaming/maintenance forever. Dead on cost.
- **(B) is honest, low-surface, already half-built.** `archive-to-safetensors.rs` is 58 lines and
  produces the safetensors; `convert_hf_to_gguf.py` is upstream tooling we already carry for the
  reverse direction. The bridge is "wire these two + a provenance stamp + a hard lane separation."

**Output tiers (user picks one):**

| tier | what it writes | use case |
|---|---|---|
| `--emit safetensors` | F32/bf16 HF safetensors (dense) | feed any HF-stack tool, or our own eval harness |
| `--emit gguf-f16` | bf16 safetensors → `convert_hf_to_gguf.py` → F16/BF16 GGUF | run in llama.cpp at full dequantized fidelity (largest file) |
| `--emit gguf-requant Q4_K_M` (etc.) | bf16 → convert → `llama-quantize` to requested type | smallest llama.cpp file; **double-quantized**, lowest fidelity, loudest warning (G4) |

`gguf-requant` exists only because users will ask; it is the most-warned, never-suggested tier (G4).

### B.3 Architecture & where it lives

New leaf crate `crates/strand-bridge` (depends *downstream-only* on `strand-decode-kernel` +
`strand-quant` read APIs):

```
crates/strand-bridge/
  Cargo.toml             # deps: strand-decode-kernel (loader, outlier_mac), strand-quant (READ-ONLY)
                         #       NO edge from strand-quant/strand-container back to this crate
  src/lib.rs             # export_dense(model: &StrandModel) -> DenseModel  (reuses patched_weights)
  src/safetensors_out.rs # writer (lift archive-to-safetensors.rs body here, de-duplicated)
  src/provenance_stamp.rs# the DERIVED / NON-ATTESTABLE metadata block (§C.5)
  src/seal_gate.rs       # require_sealed(): VerifyDepth::Full + re-derive roots over emitted bytes (G1)
  src/bin/strand-export.rs # CLI: strand-export <in.strand> <base_hf_dir> --emit <tier> --out <path>
tools/gguf/strand_to_gguf.py # thin orchestrator: strand-export → convert_hf_to_gguf.py / llama-quantize
```

**Dependency direction (critical):** `strand-bridge` is a sink. Nothing in the ship lane
(`strand-quant` producer path, `strand-container`, SPRV/attestation) ever depends on it →
contamination is a compile error (G2 CI gate makes it mechanical).

**Why a separate crate, not a `quantize-model` subcommand:** `quantize-model.rs` is the *producer*
of attested archives (`write_strand_v2`:1148 + `append_outl`:1168 + `append_sprv_computed`:1213).
A lossy float exporter in the same binary invites a future path that wires float weights back into
the seal. A separate binary with **no write-API to the archive format** makes that impossible. This
formalizes-and-hardens the existing `archive-to-safetensors` read-only-bin precedent, then deletes
the ad-hoc bin (G2).

**Data flow:**

```
                       [ DETERMINISTIC SHIP LANE — the moat ]
 weights ─quantize-model→ .strand v2 ─append_sprv→ SEALED .strand ─verify_archive→ attested
                                                          │  (read-only mmap, input only)
                                                          ▼
                       [ ADOPTION LANE — strand-bridge, a SINK ]
  SEALED .strand ─require_sealed(Full + re-derive roots, G1)→ patched_weights → dense bf16/f32
       + base HF dir (non-quant tensors, G6) ─→ HF safetensors(+stamp §C.5)
                                                          │
                                                          ▼
                                  convert_hf_to_gguf.py (upstream) ─→ .gguf
                                                          │ (optional, loudest-warned)
                                                          ▼
                                      llama-quantize Q4_K_M/… ─→ .gguf (DOUBLE-QUANT, G4)
```

### B.4 The pod step (no local compute — hard safety constraint)

Nothing built/run locally (PV training run PID 28690 holds ~15/18 GB). Route to the pending pod:

1. `cargo build -p strand-bridge --release` (new crate; CPU-only, no Metal/cloud-GPU).
2. Functional check on the existing 0.5B sealed archive:
   `strand-export <qwen05b.strand> <base_hf> --emit safetensors --out /tmp/derived.safetensors`,
   then diff against `archive-to-safetensors` output → **byte-identical** (proves the lift changed
   nothing).
3. **G1 root re-derivation check:** assert the exporter's re-derived `tensor_root`/`model_root` over
   the emitted dense bytes equals the sealed roots; assert it *refuses* when a block is perturbed.
4. `--emit gguf-f16` → run `llama-perplexity` (or our eval) on the GGUF and compare PPL to the
   STRAND-native eval of the same archive. Expectation: GGUF-f16 PPL ≈ STRAND f32-view PPL **minus**
   f16 rounding of dense weights + inference-time numeric drift (this delta is the measured cost of
   §C "non-portable float dequant"). **Label any number per G4** — never STRAND's own.
5. **CI assertion:** `cargo metadata` shows no path from `strand-quant`/`strand-container` producer
   targets into `strand-bridge` (G2).

---

## PART C — WHAT IS LOST / HOW IT STAYS SEPARATE (the boundary)

### C.1 The contamination firewall — five independent barriers (defense-in-depth)

1. **One-way dependency edge (compile-time).** `strand-bridge` is a leaf sink; ship-lane crates do
   not list it. No symbol the deterministic encoder/attestor can call reaches float-export code. A
   reviewer can grep workspace `Cargo.toml`s and prove it; CI proves it mechanically (G2).
2. **No write path to the archive format.** Bridge imports only *reader* surfaces
   (`StrandModel::open`, `view`, `encoded_tensor_checked`, `patched_weights`). It does **not**
   import `write_strand_v2`, `append_outl`, `append_sprv_computed`, or `descriptor_digest`. It
   cannot produce or re-seal a `.strand` — only foreign formats. "Bridge output gets attested as
   STRAND" is structurally unreachable.
3. **Input must be sealed; output can never be re-imported as ground truth.** Front door =
   `require_sealed()` → **`VerifyDepth::Full` + re-derive roots over the emitted bytes** (G1) and
   refuse otherwise. On the way out, every artifact gets the DERIVED stamp. Re-ingesting a bridged
   GGUF goes through normal `quantize-model` (fresh RHT+trellis+attest); there is no "trust the
   GGUF's STRAND-ness" path — the GGUF carries no STRAND roots. Root-of-trust flows **out, never in**.
4. **Distinct artifact identity.** STRAND = `STR2`/`.strand` + `SPRV` trailer +
   `DOMAIN_DESC`/`DOMAIN_OUTL` digest domains. Bridge outputs = `.gguf` (GGUF magic) /
   `.safetensors`. Zero file-level ambiguity. The stamp uses a *separate* domain tag
   (`STRAND-DERIVED\0`) so it can never collide with or be mistaken for a real `model_root`.
5. **Tests + CI as the immune system.** Bridge tests assert: (a) no dep edge into producer/attestor
   (`cargo metadata`); (b) exporting an *unsealed* archive errors; (c) dense export byte-equals
   `patched_weights` for every tensor (reuse `attest-strand.rs --recon-check` logic); (d) the stamp
   is present in every output. The ship lane's determinism replay/golden suite never imports the
   bridge → a bridge change can never alter a golden.

**Net:** the moat is protected by type-system + dependency-graph, not reviewer vigilance — the
strongest separation a Rust workspace allows.

### C.2 Provenance stamp (travels with every bridged file) — see G3

Embedded in GGUF KV-metadata and safetensors `__metadata__`:

```
strand.derived           = true
strand.bridge_version    = <semver>
strand.source_model_root = <hex of the SPRV model_root it was derived from>
strand.source_sha256     = <archive source_sha256>
strand.lost              = "bit-exact float-free deterministic decode; see boundary §C.3"
strand.attestable        = false       # this file is NOT attestable; only the source .strand is
strand.export_tier       = gguf-f16 | gguf-requant:Q4_K_M | safetensors
# if --allow-unsealed was used (G5):
strand.source_attested   = false
```

`source_model_root` is a one-way pointer back to the moat: shows *which* attested model this
descends from, and that the file itself carries none of STRAND's guarantees. It is a **label**, not
an attestation — verifying it requires the original `.strand` (docs must say this, risk #6).

### C.3 What going STRAND → GGUF surrenders (loudest in user-facing docs)

In order of moat-importance:

1. **Bit-identical, float-free, integer decode — GONE.** STRAND decode is
   `state → integer LUT → (scale_q × q) >> 16`, pure `i32`. llama.cpp's dequant for **every** quant
   type is floating-point (`d (f16) × q`, float-accumulated) → weights are **not** bit-identical
   across devices/builds (depend on CPU/GPU rounding, SIMD width, FMA contraction). STRAND's single
   defining property dies the instant weights enter ggml.
2. **Cross-device determinism / reproducibility — GONE.** Float dequant + non-associative inference
   accumulation → same GGUF gives subtly different logits on CPU vs Metal vs cloud-GPU vs different
   llama.cpp builds. STRAND's `gpu_q12_matches_cpu_decode_lean` (GPU==CPU bit-exact) has no analogue.
3. **Attestability — GONE.** `SPRV` trailer, `tensor_root`/`model_root`, `verify_archive` chain do
   not survive conversion. Best we can do is the §C.2 label pointing back at the source `.strand`.
   No in-file proof; provenance requires keeping + verifying the original.
4. **Density at uniform bpw on any dim — GONE (possibly worse).** STRAND hits a requested uniform
   bpw on any dim via row-aware RHT (896-dim edge). The honest path dequantizes to **bf16/f16 (16
   bpw — much larger)**; a later `llama-quantize` hits GGUF's tiling tax (Q2_K → 4.20 actual bpw on
   896-dim) **and** is now double-quantized (STRAND + K-quant losses stacked) → strictly worse than
   either format alone at that size. There is no "keep STRAND's bits in GGUF" — different codec (§B.2).
5. **RHT / outlier structure — flattened.** STRAND stores compact per-tensor RHT seed + sparse
   outlier channel; after dequant these are baked into dense weights. The GGUF is larger and carries
   none of STRAND's structural compactness or the outlier side-channel's targeted precision.
6. **Integer-domain safety net for odd-power RHT widths — GONE.** STRAND inference can stay in the
   integer Q12 domain (matvec_rht decodes Q12 and scales), so the ~1e-6 odd-power-width RHT-inverse
   approximation never compounds at runtime. In llama.cpp everything is float from dequant onward, so
   that approximation lives in a fully-float pipeline with no integer anchor.

**Summary for users:** *The bridge is an adoption on-ramp, not a deployment target. Use it to run a
STRAND model inside tooling you already have (llama.cpp / ollama / HF). The moment you do, you trade
away every property that made STRAND worth choosing — determinism, float-free portability,
attestation, density. For production STRAND deployment use the native float-free decoder (dismantle
/ strand-decode-kernel); use the GGUF bridge only for interop, demos, and ecosystem reach.*

### C.4 Open questions for the moat review (resolved into guardrails where possible)

- **Allow reading unsealed archives at all?** Resolved → G5: require sealed by default; `--allow-unsealed`
  only with the louder `strand.source_attested=false` stamp, absent from demos, and a full
  `decode == decode_lean` self-consistency pass.
- **Vocabulary/embeddings/norms.** Resolved → G6: carry through from the original HF dir; reuse the
  existing inverted name map.
- **Naming.** Resolved → G6: reuse `tools/gguf/gguf_to_hf.py::build_reverse_map` inverted (verified
  line 28). Do not reinvent.

### C.5 Key files referenced (all absolute, verified)

- Reconstruction contract: `/Users/scammermike/Downloads/strand/crates/strand-decode-kernel/src/outlier_mac.rs` (`patched_weights`:10; byte-exact proof `patched_decode_byte_equals_recon_path`)
- Precedent to formalize then delete: `/Users/scammermike/Downloads/strand/crates/strand-decode-kernel/src/bin/archive-to-safetensors.rs` (58 lines; `patched_weights`:24)
- Integer decode (the moat): `/Users/scammermike/Downloads/strand/crates/strand-quant/src/decode.rs`
- RHT (inverse on export): `/Users/scammermike/Downloads/strand/crates/strand-quant/src/rht.rs`
- **Verify depths (G1):** `/Users/scammermike/Downloads/strand/crates/strand-quant/src/provenance_io.rs` (`VerifyDepth`:91-97; `Vectors`=sampled:574-580; `Full`=recompute all + tensor_root:582-606; `verify_archive`:614)
- Self-verify uses only Vectors: `/Users/scammermike/Downloads/strand/crates/strand-decode-kernel/src/bin/attest-strand.rs:103`
- Format v2 + sections + reader: `/Users/scammermike/Downloads/strand/crates/strand-quant/src/format.rs`
- Loader / mmap / OUTL surfacing: `/Users/scammermike/Downloads/strand/crates/strand-decode-kernel/src/loader.rs`
- Producer (bridge must NOT live here): `/Users/scammermike/Downloads/strand/crates/strand-quant/src/bin/quantize-model.rs` (`write_strand_v2`:1148, `append_outl`:1168, `append_sprv_computed`:1213)
- Attestation chain (must NOT be importable by bridge): `/Users/scammermike/Downloads/strand/crates/strand-quant/src/provenance.rs`, `provenance_io.rs`
- Upstream GGUF tooling to reuse: `/Users/scammermike/Downloads/strand/tools/gguf/convert_hf_to_gguf.py`, `/Users/scammermike/Downloads/strand/tools/gguf/gguf_to_hf.py` (`build_reverse_map`:28)
- Dependency firewall evidence: `crates/strand-quant/Cargo.toml` (no workspace deps), `crates/strand-decode-kernel/Cargo.toml` (`strand-quant` downstream-only), `crates/strand-container/Cargo.toml` (`strand-core` lane)
- Moat definition / loss grounding: `/Users/scammermike/Downloads/strand/docs/will.md` §2/§3/§4/§5.2/§7/§9, `/Users/scammermike/Downloads/strand/docs/STRAND-vs-gguf-isobpw.md` (896-dim: Q2_K → 4.20 bpw)
