# STRAND quant-track absorption — status & audit record

**Date:** 2026-06-15 · **Source:** `~/Downloads/strand` @ tag `quant-handoff` (annotated tag obj
`9498ac3` → commit `0499ece`, which includes col-RHT serving `070b541` + the handoff doc).
**Branch:** `absorb/strand-quant-track`.

This records what was absorbed into dismantle, what was verified, and which handoff claims did **not**
substantiate. The companion procedure is [`STRAND-dismantle-wiring.md`](STRAND-dismantle-wiring.md)
(Steps 0–9); the original handoff is [`DISMANTLE-ABSORB-HANDOFF.md`](DISMANTLE-ABSORB-HANDOFF.md).

## What moved (source only — no blobs, no `target/`)

| from strand | to dismantle | tracked files | size |
|---|---|---|---|
| `crates/strand-quant/` | `vendor/strand-quant/` | 66 | 1.6M |
| `crates/strand-decode-kernel/` | `vendor/strand-decode-kernel/` | 58 | 924K |
| `experimental/docs/` | `docs/strand/` | 27 | — |
| `experimental/{scripts,ops,tools,configs}` | `tools/strand/` | ~95 | 1.2M |
| `research/*.md` + small ledgers | `docs/strand/research/` | 22 | ~340K |
| `DISMANTLE-ABSORB-HANDOFF.md` | `docs/strand/` | 1 | — |

Vendored via `git archive` (tracked tree only), so **no `target/` (was 482M+236M), no per-crate
`Cargo.lock`, no `.strand`/safetensors/gguf blobs**. Total absorbed ≈ **5M**.

> **NOT absorbed — by design:** `research/`'s four model-checkpoint dirs (`isobpw` 9.7G, `mp-frontier`
> 5.6G, `pv-deep` 3.7G, `down-protect` 1.9G = **~21 GB**) are gitignored in strand and are PPL/eval
> artifacts, not source. Only the strategy markdown + small ledgers came over. The macOS installer
> `.app` build binaries under `experimental/scripts/packaging` were pruned too.

## Workspace wiring

- `vendor/strand-quant` + `vendor/strand-decode-kernel` are in the dismantle workspace **`exclude`**
  list — self-contained packages (concrete deps, no `workspace=true`), kept OUT of the default build
  so `cargo build --workspace` and all golden hashes are **unchanged**. Build/test standalone with
  `cargo test --manifest-path vendor/strand-quant/Cargo.toml`.
- `strand-quant` pins `metal 0.27 / objc 0.2`; dismantle's product uses `metal 0.29 / objc2 0.5`.
  They coexist (different crate majors) — only relevant once the kernel/baker actually link it.
- `tools/strand_bake` updated from "not implemented, separate project" → an honest **seam** that
  points at the absorbed `vendor/strand-quant` (path-dep one uncomment away). Still a non-member;
  the baker pipeline (GGUF f32 → `encode_tensor` → `write_strand_v2`) is wiring Step 4, not done.

## Gates (determinism / byte-identity re-proven on the absorbed copy)

| gate | anchor | result |
|---|---|---|
| `strand-quant` full test suite | 174 lib + 26 + 15 debias | **174 lib pass / 0 fail**, all integration green, **0 failures** across 24 binaries (incl. 165s/87s/113s exhaustive-determinism tests; `rht_serving_feasibility` col-vs-row proof; `debias_determinism` 15 pass/2 ignored) |
| `strand-decode-kernel --lib` | 72 lib | **72 pass / 0 fail** (one transient crates.io `metal` fetch blip on the first concurrent run; clean on retry) |
| dismantle product `cargo check --workspace` | unchanged | **clean (4.28s, 0 errors)** — exclude + strand_bake edits don't touch the product build |

## Contract verified against code (not trusted from the doc)

- **§2 helpers** — all 7 present & `pub`: `read_strand_v2_header` (format.rs:506, *not* sideinfo_wire),
  `reconstruct_q`/`eff_scale_q`/`eff_min_q` (decode.rs), `unpack_sub_scales` (encode.rs:64),
  `SUB_BLOCK=32` (encode.rs:25; selfdesc carries it as const-id tag #3 → value 32),
  `codebook_lut -> &'static [i32]` (codebook.rs:310).
- **§3 format** — `MAGIC_V2 = b"STR2"`, `VERSION_V2 = 2`, `PAGE = 4096`; flag byte additive bits
  bit0 `f|=1` rht_seed, bit1 `f|=2` tail_biting, bit2 `f|=4` affine_min, **bit3 `f|=8` rht_cols**.
- **RHT default ON** (`let mut rht = true` + `--no-rht`); **OUTL is overwrite** `w[i] = v` (not `+=`).
- **§4 col-RHT** — `rht_forward_cols_inplace`/`rht_inverse_cols_inplace` (rht.rs); `outlier_mac`
  inverts weights with `_cols_` and transforms the activation **once** with `rht_forward_cols_inplace`
  for the cheap per-column serving path. Matches the handoff exactly.

## Scorecard audit (§5) — substantiated vs flagged

**Confirmed** in the absorbed docs:
- 3-bit **0.0562 nats / +5.8%**, PROVEN — `research/STRAND-item2-ship-3bit.md` (Llama2-7B archived
  recon; handoff's own caveat: re-confirm on a live Qwen-7B `.strand` point).
- loss-tax **2-bit 0.324 / 3-bit 0.056** vs targets **≤0.15 / ≤0.05** — `STRAND-supercondenser-sprint.md`.
- size **0.4175 B/w vs Q4_K 0.5625** — `STRAND-cpu-deploy.md`.
- **34.6 Gw/s ≈ 18% peak, ALU-bound → 6–10 tok/s (col-RHT) / ~1 (row), below Q4_K 20–30** —
  `STRAND-dismantle-integration-closeout.md`; bf16 PPL **7.74** — `STRAND-product-spec.md`.

**Flagged (did not fully substantiate):**
1. The specific 2-bit PPLs **7B 10.54 / 14B 8.92 / 32B 6.61** were **not found** in the cited
   sprint doc (only the 0.324 loss-tax was). Needs the "scorecard" source.
2. **"4-bit beats Q4_K at ~7.8% rel-RMS"** is **not** substantiated in `STRAND-gate-results.md`; that
   doc explicitly disclaims rel-RMS as "the wrong metric at low state count."
3. The **0.4175 size win is CPU-fastpath ONLY** — the GPU `BlockEntry` path is **0.578 B/w > Q4_K
   0.5625** (cpu-deploy.md:8,84). The handoff's §5 row understated this.

## Honest positioning (verified, carried forward)

STRAND-into-dismantle = **deterministic + compact (CPU path) + competitive-quality FFN footprint
option, default-OFF, off the tps/J critical path.** It WINS on 3-bit quality (lossless-class),
CPU-path size, and bit-identical decode. It **LOSES on speed** — the FFN GEMV is B=1 ALU-bound
(measured d=2 vector kernel only 1.18×, below the 1.3× bar); col-RHT lifts per-row→per-column but
stays under Q4_K. *"Footprint/determinism is bought with throughput."*

## Pass 2 — TQ: GEMV / activation-RHT wiring

The dismantle-side serving project is **TQ (Trellis-Quant)** — feature `tq`, module
`dismantle_core::tq`, artifact extension **`.tq`** (the on-disk wire magic stays `STR2`; `.tq` is
TQ's project identity). Built on the absorbed `strand-quant` codec.

**Slice 1 — DONE & merged** (PR #2): CPU serving reference in `tq.rs` — integer Q12 decode
(`decode_q12` → `strand_quant::decode_tensor_fixed`) + the row/col/none activation-RHT `matvec_rht`
mirroring `outlier_mac.rs` + `apply_outlier_overwrites` (the OUTL `w[i]=v` overwrite). Tests: Q12
decode determinism + match to float decode; the **col-RHT one-transform-serves-all-rows identity**;
OUTL overwrite-not-add.

**Slice 2 — DONE & merged** (PR #3): `tq::read_strand(bytes) -> Vec<StrandTensor>` — parse a `.tq`/STR2
archive into decode-ready tensors (zip lean header for `rht_cols`/shape/seed with the SDSQ-applied
payloads). `StrandTensor::{decode_q12, matvec}`. Round-trip test: encode → `write_strand_v2` →
`read_strand` → decode is **bit-identical** to the direct decode. End-to-end **file → decode → serve**
CPU path on main.

**Default build byte-identical** throughout — `strand-quant` is not pulled without the `tq` feature
(`cargo tree`: 0). The `tq` module is the parity oracle the GPU kernel will be gated against.

**Remaining:**
- **Baker (Step 4)** — `tq_bake`: GGUF → select projections → `encode_tensor` → `write_strand_v2` →
  `<name>.tq`. Locally only a Q4_K model is present, so the f32 encode source comes from a Q4_K→f32
  dequant (plumbing validation, not a quality artifact); a real bake needs an f16/f32 source.
- **Metal kernel (Steps 5–9)** — port the G4 bitslice decode→GEMV, build the loader block table,
  dispatch behind `DISMANTLE_QWEN_TQ` (default-off), pass the **GPU↔CPU bit-identity gate** against
  `tq::matvec_rht`. Reference: `vendor/strand-decode-kernel/{outlier_mac.rs,shaders/strand_bitslice.metal}`.
  The long pole — needs sustained GPU iteration.
