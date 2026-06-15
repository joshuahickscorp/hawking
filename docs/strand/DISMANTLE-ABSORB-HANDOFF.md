# DISMANTLE ⇽ STRAND — quant-track absorption + strand cleanup (executable handoff)

**You are the dismantle session.** The user has linked the STRAND repo into your session as a
sibling folder (expected at `~/Downloads/strand`, i.e. `../strand` from `~/Downloads/dismantle`).
Your job, in order:

1. **ABSORB** the entire STRAND **quant track** into dismantle (it currently lives in strand as
   two excluded crates + experimental/research docs).
2. **AUDIT the whole nine yards for accuracy** — do not trust this document; *verify every claim,
   number, and byte* against the actual code and re-run the gates. This doc is a map, not gospel.
3. **ERASE** the quant track from the strand folder once absorption is confirmed (you have write
   access to the linked folder). Leave strand as the clean, determinism-only core codec.

Everything below is the state of strand at the absorb-from anchor. **Re-verify before relying on
anything** — line numbers drift, and the point of this task is accuracy.

---

## 0. Pre-flight (verify first)

```bash
# in the linked strand folder:
git -C ../strand rev-parse --short HEAD          # expect 070b541 (or a descendant)
git -C ../strand tag -l quant-handoff -n1         # the absorb-from anchor; quant is intact here
git -C ../strand status --short                   # expect clean
```

- **Absorb-from ref:** tag `quant-handoff` (HEAD `070b541`). If strand HEAD has moved, absorb from
  the tag (`git -C ../strand checkout quant-handoff` in a worktree) so you get the quant intact.
- **The boundary is clean (verified):** the strand *core* crates (`strand-core`, `strand-container`,
  `strand-cli`, `strand-ffi`, `strand-bench`, `strand-wasm`) have **zero** dependency edges into the
  quant crates. The only references to `strand-quant`/`strand-decode-kernel` anywhere in core are the
  three lines in the root `Cargo.toml` `exclude` list. **Confirm this yourself** before erasing:
  ```bash
  grep -rniE "strand[-_]quant|strand[-_]decode[-_]kernel" ../strand/crates/strand-core \
    ../strand/crates/strand-container ../strand/crates/strand-cli ../strand/crates/strand-ffi \
    ../strand/crates/strand-bench ../strand/crates/strand-wasm
  # expect: no matches. If anything matches, STOP and report — erasure is not clean.
  ```

---

## 1. Inventory — exactly WHAT to absorb (and what is CORE, never touch)

### ABSORB (the quant track):
| path | what | size |
|---|---|---|
| `../strand/crates/strand-quant/` | the quantizer: encode (float, offline), **integer-deterministic decode**, RHT, trellis/codebook, the STR2 wire format, side-info (C2/SDSQ), de-bias, outlier wire, provenance/SPRV | 63 `.rs`, ~31.7k lines |
| `../strand/crates/strand-decode-kernel/` | the on-device runtime: mmap loader, CPU/NEON decode→GEMV, Metal G4 bitslice GPU path, `outlier_mac` (RHT+outlier serving), all the `gate-*` microbenchmarks | 52 `.rs`, ~19.6k lines |
| `../strand/experimental/` | **all quant**: `docs/` (27 — specs, results, the wiring recipe), `scripts/` (50 — quant/PPL/cloud ops), `ops/` (4), `tools/` (2), `configs/` (6). `shaders/` is now **empty** (the two encode shaders were moved into `crates/strand-quant/shaders/`). | dir |
| `../strand/research/` | 42 quant research/strategy docs (R-D curves, supercondenser sprint, competitive verdicts) | dir |
| `../strand/Cargo.toml` | the 3 `exclude`-list lines + comment for the quant crates (remove during erase) | 3 lines |

### CORE — DO NOT ABSORB, DO NOT ERASE (this is the strand product that stays):
- `crates/strand-core`, `crates/strand-container`, `crates/strand-cli`, `crates/strand-ffi`,
  `crates/strand-bench`, `crates/strand-wasm`
- root `scripts/` — `bench.sh`, `golden`, `prove-determinism.sh`, `roundtrip-demo.sh`, `samples/`
  (these are the **core** determinism/bench proofs, NOT quant)
- root `docs/` — `BENCHMARKS.md`, `STRAND-media-speed-roadmap.md`, etc. (core docs)
- `README.md`, `SPEC.md`, the core `.gitignore` entries

> ⚠️ The naming is a trap: `experimental/docs/` and `experimental/scripts/` are **quant** (absorb);
> root `docs/` and root `scripts/` are **core** (keep). Check the path prefix every time.

---

## 2. How dismantle consumes strand-quant today (the existing contract)

dismantle already path-deps `strand-quant` (it does **not** use `strand-decode-kernel` — that crate
is strand's own reference runtime + gate harness; dismantle has its own kernel). See the existing
recipe `../strand/experimental/docs/STRAND-dismantle-wiring.md` — that is the **authoritative,
step-by-step wiring procedure (Steps 0–8)**; read it first. Note its 2026-06-15 header: the strand
crates are now excluded from strand's root workspace and are **self-contained** (concrete dep
versions, no `workspace=true`), so they path-dep / vendor cleanly into dismantle's own workspace.

Current path-deps in dismantle (from the recipe):
- `crates/dismantle-core/Cargo.toml`: `strand-quant = { path = "../../../strand/crates/strand-quant" }`
- `tools/strand_bake/Cargo.toml`: same path.

**The handoff API (7 load-bearing `pub` helpers).** All verified present & `pub` at the anchor, but
they MOVED files in the repo refactor — the recipe's 2026-06-09 line numbers are stale. **Re-grep by
symbol name; do not trust line numbers:**
| symbol | role | found at anchor (re-verify) |
|---|---|---|
| `format::read_strand_v2_header` | the canonical STR2 lean header parser | `sideinfo_wire.rs` |
| `decode::reconstruct_q` | the load-bearing i64 `(scale_q*quantile_q)>>16` | `decode.rs` |
| `decode::eff_scale_q` | fold a 6-bit sub-scale code → effective Q16 scale | `decode.rs` |
| `decode::eff_min_q` | 4-bit affine-min offset fold | `decode.rs` |
| `encode::unpack_sub_scales` | unpack 6-bit sub-scale codes | `encode.rs` |
| `encode::SUB_BLOCK` (=32) | weights per sub-block | `selfdesc.rs` |
| `codebook::codebook_lut` | the frozen 2^L Q12 codebook | `codebook.rs` |

```bash
for s in read_strand_v2_header reconstruct_q eff_scale_q eff_min_q unpack_sub_scales SUB_BLOCK codebook_lut; do
  echo "== $s =="; grep -rn "pub fn $s\|pub const $s" ../strand/crates/strand-quant/src/ | head -1
done
```

---

## 3. The format & decode contract (verify each against code)

- **Format = STR2 v2** (`MAGIC_V2 = b"STR2"`, `VERSION_V2 = 2`), page-aligned (`PAGE=4096`),
  per-tensor seek table + `source_sha256` of the whole input safetensors. Two writers share a core
  (`write_strand_v2_inner`): `write_strand_v2` (inline scale_q) and `write_strand_v2_packed`
  (SDSQ 12-byte records, `SCALEQ_IN_SDSQ` flag). Both delegate; **do not** duplicate the flag logic.
- **Per-tensor flag byte** (offset `p+3` in each tensor record): bit0 `has_rht_seed`, bit1
  `tail_biting`, bit2 `has_affine_min`, **bit3 `rht_cols`** (see §4). Additive — archives without a
  bit are byte-identical to the old format.
- **RHT = ON in shipped artifacts** (`quantize-model` default `rht=true`; `--no-rht` disables). The
  decoder must apply the RHT inverse/serving transform — this is **the activation-RHT-into-GEMV
  wiring, and it is THE thing dismantle must mirror in its own kernel.** Reference impl:
  `strand-decode-kernel/src/outlier_mac.rs` (`patched_weights`, `bulk_weights`, `matvec_rht`).
- **OUTL = ON** (1% top-|w| pre-RHT outliers; `outlier_wire.rs`, `OUTL` magic). It is an
  **OVERWRITE, not an add** (`outlier_mac.rs`: `w[i] = v`) applied after decode. dismantle's GEMV
  must do the sparse overwrite, not a residual add, unless using the residual form in `matvec_rht`.
- **Decode is integer-deterministic and bit-identical** across CPU/GPU/WASM — this is the moat.
  Float only appears in the offline encoder and the final MAC (documented per-kernel float caveat).

---

## 4. The column-sign RHT serving path (NEW — landed this session, commit 070b541)

This is the freshest piece and the one most relevant to dismantle's tps. **Audit it carefully.**

- **Why:** per-ROW RHT needs a different activation transform per output row (the ~1 tok/s serving
  wall). Per-COLUMN RHT shares the sign by column, so **one** `rht_forward(x)` serves every row →
  the cheap-serving path (~6–10 tok/s hybrid vs ~1). Quality is proven equal to per-row
  (`crates/strand-quant/tests/rht_serving_feasibility.rs`: col-vs-row penalty ≈ −0.0%/−0.1%).
- **Format:** flag **bit3 = `rht_cols`** on `TensorHeaderV2`. New writer
  `format::write_strand_v2_rht(tensors, sha, strict, scaleq_in_sdsq, rht_cols: &[bool])` stamps it;
  the plain writers delegate with an empty mask (byte-identical). `quantize-model --rht-cols
  --packed-v2-out` now emits it (v1 `--packed-out` still blocked — no flag in v1).
- **Decode (reference, `outlier_mac.rs`):** when `hdr.rht_cols`:
  - `patched_weights`/`bulk_weights` invert with `rht_inverse_cols_inplace` (not `_rows_`);
  - `matvec_rht` transforms the activation ONCE (`rht_forward_cols_inplace`, ≡ `rht_forward` for a
    single in_features vector) and reuses it for all rows.
- **Contract dismantle must mirror in its kernel:** `y = decode(W_col_rht) · rht_forward(x)`,
  one transform per tensor, weights NOT inverted. Identity proven:
  `matvec(decoded_col_weights, rht_forward(x)) == matvec(rht_inverse_cols(decoded), x)`.
- **Tests to read:** `outlier_mac::tests::col_rht_single_rotation_serves_all_rows` (proves it works
  for all rows) and its foil `single_rotation_recipe_diverges_for_multirow` (proves row-RHT can't
  use one transform). The `rht_*_cols*` primitives are in `strand-quant/src/rht.rs`.

---

## 5. Baseline numbers — the scorecard (VERIFY each; sources cited)

These are prior measured runs in strand's own docs (2026-06-09…15), **not re-measured at handoff.**
Part of your audit is to confirm or refresh them. Be honest about model family — they differ.

| axis | number | source doc | note |
|---|---|---|---|
| bf16 PPL (Qwen2.5-7B, 146w) | **7.74** | `experimental/docs/STRAND-product-spec.md` | the R-D ceiling |
| **3-bit quality** (Llama2-7B `mp_light` down@4/rest@3, L=12, 1% outl) | **+5.8% / 0.0562 nats** (5.5353→5.8552), PROVEN, no training, bit-identical | `research/STRAND-item2-ship-3bit.md` (`ppl_mp_light_l12_out1.json`) | **lossless-class; ties SOTA at 3-bit.** Archived Llama2-7B recon — re-confirm on the canon harness (`ops/eval-ppl.py`, ctx 2048) for a live point |
| 2-bit quality at scale (Qwen2.5 PTQ, l=12 + outlier ch) | 7B **10.54** / 14B **8.92** / 32B **6.61** PPL, no collapse, ~**0.31** loss-tax | `research/STRAND-supercondenser-sprint.md` + scorecard | above the ≤0.15 target → needs selective-PV (open cloud gate) |
| 2-bit target | loss-tax **≤0.15** (now 0.324) | supercondenser-sprint | the PV frontier |
| 3-bit target | loss-tax **≤0.05** (now 0.056) | supercondenser-sprint | essentially met |
| size (3-bit CPU fastpath) | **0.4175 B/w** vs Q4_K 0.5625 B/w | `experimental/docs/STRAND-cpu-deploy.md` | **smaller at equal/better quality** |
| 4-bit beats Q4_K at | **~7.8% rel-RMS** | `experimental/docs/STRAND-gate-results.md` | the size win |
| B=1 fused decode (M3) | **34.6 Gw/s ≈ 18% of peak, ALU-bound** | `research/STRAND-dismantle-integration-closeout.md` | the speed wall |
| hybrid 7B decode ceiling | **~6–10 tok/s** (per-column RHT) / ~1 (per-row) | closeout | **below Q4_K 20–30** |
| dismantle today / llama.cpp | ~31 dec_tps / ~50 | closeout | gap is runtime, not format |
| core codec (non-quant) | 7.36× ratio, ~300× slower (~0.6 MB/s) | `docs/BENCHMARKS.md` | the separate "clean strand" story |

**Honest positioning (do not overclaim):** STRAND-into-dismantle is a **deterministic + compact +
competitive-quality FFN footprint option, default-OFF, off dismantle's tps/J critical path.** It
WINS on quality (3-bit lossless-class), size (smaller bpw), and determinism (bit-identical decode).
It **LOSES on speed** (B=1 ALU-bound; the col-RHT path lifts per-row→per-column but stays under
Q4_K). The "faster than Q4_K / beats llama.cpp" projection was **refuted on M3** by the local gate
sweep — see `experimental/docs/STRAND-speed-moonshot-research.md` (every micro-opt dead; the live
speed levers are system-level: whole-token command buffer + token-multiplying decode, both
dismantle-side).

---

## 6. ABSORB plan (into dismantle)

1. **Decide the vendor layout.** Recommended: move `strand-quant` and `strand-decode-kernel` into
   dismantle's tree (e.g. `dismantle/vendor/strand-quant`, `dismantle/vendor/strand-decode-kernel`)
   and add them to dismantle's workspace `members`, OR keep them as a pinned git/path dep — your
   call based on dismantle's build model. They are **self-contained** (concrete deps), so either works.
2. **Update the path-deps** (`dismantle-core/Cargo.toml`, `tools/strand_bake/Cargo.toml`) to point at
   the new location instead of `../../../strand/crates/strand-quant`.
3. **Bring the docs you need** into dismantle (the wiring recipe, the product spec, the result
   ledgers, the supercondenser sprint, this handoff). Leave behind nothing dismantle will want later
   — once strand erases the quant track, those docs are gone from strand (recoverable only via the
   `quant-handoff` tag / git history).
4. **Wire the activation-RHT-into-GEMV** in dismantle's kernel per §3–§4 (this was always the
   blocking integration code task). Mirror `outlier_mac` for both row and **col** RHT.
5. **Follow `STRAND-dismantle-wiring.md` Steps 1–8** for the bake→load→decode→GEMV touch-points and
   the per-tensor hybrid (attention-Q4_K + FFN-STRAND, default-OFF behind `DISMANTLE_QWEN_STRAND`).

---

## 7. AUDIT — "the whole nine yards" (the user's explicit mandate)

Treat this section as a checklist; **report findings, don't silently trust.**

- [ ] **Determinism / byte-identity preserved post-move.** Re-run strand's own proofs against the
      absorbed copy: `cargo test --manifest-path <quant>/Cargo.toml` and
      `cargo test --manifest-path <decode-kernel>/Cargo.toml --lib` must be green
      (anchor: strand-quant **174 lib + 26 c2 + 15 debias** green; decode-kernel **72 lib** green).
      The STR2 round-trip + golden Q12 byte-identity must hold bit-for-bit after vendoring.
- [ ] **Numbers re-checked.** For each row in §5, open the cited doc/JSON and confirm the figure, or
      re-measure. Flag any you cannot substantiate. Especially: the **0.0562** 3-bit anchor is a
      Llama2-7B archived recon — re-confirm on the canon harness for a live Qwen-7B `.strand` point.
- [ ] **The col-RHT contract holds** (§4): `rht_serving_feasibility.rs` passes; the `outlier_mac`
      col tests pass; your dismantle-kernel mirror reproduces `matvec_rht` col output within the
      documented float tolerance.
- [ ] **No code/doc lost in the move.** Diff file counts and the symbol inventory (§2) before/after.
- [ ] **Dependency hygiene.** The vendored crates build inside dismantle's workspace with no
      `workspace=true` leakage and no new lock surprises (`wide`, `rayon`, `memmap2`, `metal`/`objc`).
- [ ] **No float introduced into the integer decode path.** The `(eff*q)>>16` i64 reconstruct and the
      trellis recurrence must stay integer. Never run `cargo clippy --fix` on this code (see §9).
- [ ] **Contract facts** (§3): RHT-on default, OUTL-overwrite-not-add, STR2 v2, col-RHT bit3 — each
      confirmed against code, not this doc.
- [ ] **Honest positioning** (§5) reflected in whatever dismantle docs/claims you write — footprint +
      determinism + competitive quality, NOT a speed win.

---

## 8. ERASE the quant track from strand (after absorption is confirmed)

Only after §7 passes. You have write access to the linked folder. In `../strand`:

```bash
cd ../strand
git switch -c remove-quant-track            # do NOT commit straight to main
git rm -r crates/strand-quant crates/strand-decode-kernel
git rm -r experimental research
git rm DISMANTLE-ABSORB-HANDOFF.md           # this file
# Edit Cargo.toml: delete the 3 quant lines from `exclude` (and the comment), and the
# `crates/strand-container/fuzz` exclude stays. Tidy .gitignore (drop the per-crate
# Cargo.lock + quant-only ignores if any).
```

Then **prove the core is still whole** (this is non-negotiable):
```bash
cargo test --workspace                       # core only; expect all green (was 253+35+18+17+16+14+...)
bash scripts/prove-determinism.sh            # expect "DETERMINISM PROOF: PASS"
cargo build --workspace                      # no dangling refs to the removed crates
```
Grep to confirm nothing in core still references the removed crates (expect zero):
```bash
grep -rniE "strand[-_]quant|strand[-_]decode[-_]kernel|experimental/|research/" \
  crates Cargo.toml README.md docs scripts | grep -v target
```
Commit on the branch with a plain message (NO AI/co-author trailers — the user forbids them):
```
Remove the quant track (absorbed into dismantle)

strand-quant, strand-decode-kernel, experimental/, and research/ moved into the dismantle
repo. The core deterministic codec is unaffected (zero dependency edges into quant; core
tests + determinism proof green). Recover the quant track from tag `quant-handoff` if needed.
```
Leave it on the branch for the user to review/merge — **do not force-push strand's main.**
Note: strand's `origin/main` (217877a) is 3 PR-merge commits + missing the local session's 5
commits; the user will reconcile the remote separately. Don't touch the remote.

---

## 9. Gotchas & landmines (learned the hard way — heed these)

- **NEVER `cargo clippy --fix` the quant/codec code.** It reorders float ops → breaks bit-exact
  decode + the golden determinism test. A prior run did a blanket `--fix` (~7000 changes) and left CI
  red; it was reverted. Use `cargo fmt` + lint *suppress*, never `--fix`.
- **`EncodedTensor` has 25+ construction sites and no `Default`.** Do not add fields to it; thread
  new state through the format/`PackedTensor` layer instead (that's why `rht_cols` is a writer-mask
  param + a `TensorHeaderV2` field, not an `EncodedTensor` field).
- **`rht_seed_for(name)` is name-based** (FNV over the tensor name) → reproducible/byte-identical
  regardless of shard order. Requant sharding relies on this.
- **Metal is NET-SLOWER than CPU for the full quant model** (CPU 12-way beats a single M-series GPU
  ~8× for whole-model quantize). Quantize on CPU. The Metal path is the *decode/serving* G4 kernel.
- **B=1 decode is latency/occupancy-bound (~18% of peak), NOT bandwidth-bound** on M3 — no
  codebook/recurrence micro-opt closes the gap to Q4_K (all proven dead). tps wins are system-level
  (whole-token command buffer to kill per-tensor commits; token-multiplying decode).
- **18 GB MPS OOM ceiling**; fp16 PPL → NaN (use bf16); fp32 PV-shadow OOMs at 32B (needs bf16 +
  8-bit Adam). 24 GB 3090 OOMs loading 7B alone (~23.5 GB) → 7B QAT needs an A100.
- **`isolation:'worktree'` doesn't reliably fork the current branch** — verify any agent/worktree
  branch's base (`git merge-base`) before trusting a determinism-critical build off it.
- **Two v2 writers share `write_strand_v2_inner`** — flag logic lives there once; don't fork it.

---

## 10. Definition of done

1. Quant track (both crates + experimental/ + research/ + needed docs) lives in dismantle and builds
   inside dismantle's workspace; activation-RHT (row + col) wired into dismantle's GEMV.
2. Every §7 audit box checked, findings reported, numbers substantiated or re-measured, byte-identity
   + determinism re-proven on the absorbed copy.
3. strand reduced to the clean core: `cargo test --workspace` green + `prove-determinism.sh` PASS,
   no dangling quant refs, change on a `remove-quant-track` branch for the user.
4. A short written report back to the user: what moved, what you verified, any number that didn't
   substantiate, and the state of both repos.
