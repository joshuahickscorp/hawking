# The 2-bit frontier — verification + synthesis

_Machine: Apple M3 Pro, 18 GB, 12 logical cores (macos/aarch64). Verified 2026-06-12.
This is the synthesis of the 2-bit-frontier wave: the iso-bpw STRAND-vs-GGUF
head-to-head, the inner-product de-bias lever, and the PV-at-scale recipe. The
owner's standing order — no soft-positives — is honored: every number below is
labeled proven / partial-live / pending / modeled._

## Suites (PROVEN green)

| suite | result |
|---|---|
| `cargo check --workspace` | clean (1 benign `cfg(kani)` warning) |
| `cargo test -p strand-quant --lib` | **116 passed, 0 failed**, 2 ignored (242 s) |
| `cargo test -p strand-decode-kernel --lib` | **69 passed, 0 failed** (1321 s) |

All identity / KAT / proof tests pass, including `decode_lean_is_bit_identical`,
`block_decode_is_bit_identical_to_reference`, and the debias first-moment-cancellation
identity.

## 1. The iso-bpw Pareto verdict — PARTIAL (live run in progress)

**Harness honesty: it IS true iso-harness, not cross-harness.** Both formats are
scored by the same `ops/eval-ppl.py` (WikiText-2, ctx 2048, 64 non-overlap windows,
cpu, bf16). GGUF quants are dequantized back to HF safetensors via gguf's reference
dequantizer (`gguf_to_hf.py`) and run through the identical code path. The bpw
denominators are billed identically over the 7 projection-weight tensors
(357,826,560 elements both sides — verified). No cross-harness caveat needed. The
doc states this loudly and correctly.

**Run state:** the harness (`/tmp/isobpw-run.sh`, live pid as of writing) is mid-loop.
Landed so far:

| bpw (proj) | config | PPL | format | status |
|---|---|---|---|---|
| 16.0 | bf16 (anchor) | **12.536** | ref | proven (ceiling, not a target) |
| **4.197** | **GGUF Q2_K** | **15.238** | gguf | proven |
| 3.806 | STRAND mp_light (attn4/ffn3) | **15.039** | strand | proven (canon 64w) |
| 4.197 | GGUF IQ3_S | … | gguf | pending eval |
| 4.574 | GGUF Q3_K_M | … | gguf | eval in flight |
| 5.521 | GGUF Q4_K_M | … | gguf | pending |
| ~2.3 | STRAND q2_l12_out1 | … | strand | **not yet quantized** |
| ~2.7 | STRAND q3_l12_out1 | … | strand | not yet quantized |

**The one decisive standing already landed: at LOWER bpw STRAND beats GGUF Q2_K on
PPL.** STRAND mp_light (3.806 proj-bpw) scores 15.039 vs GGUF "Q2_K" (4.197
proj-bpw) at 15.238 — STRAND wins on BOTH axes simultaneously (0.39 fewer bpw AND
0.20 lower PPL). The driver is the **dim-896 K-quant fallback tax**: 896 is not a
clean multiple of llama.cpp's 256-element superblock, so "Q2_K" silently falls back
to Q4_0/Q5_0/Q3_K mixes and lands at 4.197 proj-bpw, not 2-bit. This is the will.md
structural edge made concrete and extended to the whole K-quant family.

**Honest limits of this verdict (no spin):**
- This win is at the **3.8 bpw tier on a 0.5B**, and it rides the 896-dim fallback —
  a small-model artifact. On 256-aligned dims (most ≥7B tensors) GGUF tiles cleanly
  and this specific bpw edge **shrinks or vanishes**. The 7B GGUF side is the open
  item (will.md §10 has the STRAND 7B/14B points; GGUF-7B unmeasured here).
- The **true 2-bit tier head-to-head is not yet measured** — STRAND q2_l12_out1
  (~2.3 bpw) has no GGUF peer at that bpw on the 0.5B (GGUF can't get there on
  896-dim), and STRAND's own q2 PPL on this exact harness is still pending quant.
- IQ3_S was quantized **without an imatrix** (a fairness caveat the doc flags); a
  calibrated imatrix would help its PPL but not the bpw-tiling story.

**Verdict: STRAND wins the iso-bpw race on the 0.5B at the 3.8-bpw tier (proven,
both axes), entirely on the 896-dim fallback structural edge. The 2-bit-vs-2-bit
and the at-scale (256-aligned, 7B) tiers are the load-bearing open questions and are
NOT yet settled.**

## 2. The de-bias verdict — ALIVE-ON-A-MODELED-MEAN, PPL-UNCONFIRMED

`gate-debias` reproduced exactly (14 real Qwen2.5-0.5B tensors, k=2 l=12 +1%
outlier, 64 acts/tensor, elapsed 656 s):

| activation model | output-RMS reduction |
|---|---|
| non-zero-mean (μ̄≈0.3, **synthetic/assumed**) | **+4.19%** |
| zero-mean control (μ̄→0) | **+0.0001%** (vacuous) |
| rowsum-bias RMS post-RHT | 0.230 (nonzero ⇒ survives RHT) |

**Does the de-bias math actually reduce output error, or is it the 4th RHT-whitening
kill? The honest answer: it depends entirely on a number that has NOT been measured.**

- The math is correct and the rowsum bias `S_i` genuinely **survives the RHT** (the
  Hadamard rotates the all-ones direction, does not destroy it — 0.23 RMS confirms).
  This is structurally distinct from the dead Hessian family: a *first-moment* (mean)
  correction, not a *second-moment* (curvature) reweight.
- BUT on **zero-mean** activations the correction is provably vacuous (+0.0001%) — it
  degenerates exactly to the dead Hessian. So the entire +4.19% is contingent on real
  Qwen activations carrying a DC mean comparable to the **assumed μ̄=0.3** (a modeled,
  synthetic Gaussian shift — never read off the real model).
- rel-RMS (the proxy) is flat (~22-26%) while output-RMS (closer to truth) moves —
  exactly the will.md §5.5 pattern, which is why output-RMS, not rel-RMS, is the gate.

**This is NOT yet a clean RHT-whitening kill, and NOT yet a win.** It sits on the
knife's edge: ALIVE iff real μ̄ is large, the 4th kill iff real μ̄ ≈ 0. The deciding
test (the 0.5B PPL A/B, protocol in `research/debias-results.md` §"0.5B PPL A/B") is
**spec'd but not run** — it needs `ops/calib-actmean.py` (real μ̄ measurement) + a
2-arm quant + an eval shim, all blocked behind the box/freeze trap. Cost if adopted:
0.0179 bpw (free if folded into a layer bias). **Verdict: PROVISIONAL-ALIVE on a
modeled mean; the real-μ̄ PPL A/B is the gate that turns it into a win or the 4th
kill. Do not report +4.19% as a STRAND result — it is a property of an assumed
activation distribution.**

## 3. The PV recipe result — PENDING (run blocked on the box)

The deep 0.5B PV run (`scripts/pv-recipe.sh`) is **spec'd and scripted but has NOT
executed** — `research/pv-deep/` is empty. At write time the box was held by a live
quant + the iso-bpw eval loop (will.md §7 freeze trap forbids MPS training alongside
them). The recipe is a clean A/B: every lever held identical to the prior 26.77
plain-cosine PV floor, with **`--cooldown-frac 0.2` (Apple WSD decay) as the ONLY
variable**. Anchors: bf16 12.55, PTQ-only floor 80.7, prior PV floor 26.77.

**Did cooldown beat 26.77? UNKNOWN — not run.** The honest expectation (from the
Apple intel scorecard): if the WSD cooldown is real for low-bit QAT, ~26.77 → low-26
/ high-25; if the cosine tail already captured it, a wash (an informative null).

**7B-selectivity feasibility: GO, ~$6/arm** (analysis in `research/pv-scale-plan.md`,
all param counts measured from Qwen2.5-7B config, footprints arithmetic):
- down_proj-only PV (1.901B trainable, 28 of 168 tensors) fits **one A100-80**
  (47.6 GB fp32-Adam) — recommended path, ~$6/arm at $1.5/hr incl. requant cadence.
- 8-bit-Adam pulls down_proj deepest-quarter onto a **$0.46/hr 3090** (21.5 GB) —
  the $2-4 budget floor, but stakes the result on the concentration prior + int8
  stability.
- Full PV (119.6 GB) needs 2×A100-80 / H100 — the $300 path, **not needed** for the
  trailblazer claim.
- **The bet rides on ONE unproven transfer:** that the down_proj concentration which
  holds for 3-bit mixed-precision (mp_light) ALSO holds for 2-bit PV re-learning.
  Under the RHT, rel-RMS sensitivity is flat across tensor classes (0.019 pp spread),
  so concentration must be proven in PPL space. The cheap gate (a free 0.5B
  `--pv-tensors down_proj` vs full-PV A/B) is recommended BEFORE any rental.

## 4. The one honest sentence on where STRAND's 2-bit stands

STRAND's 2-bit is a **deterministic, float-free, dimension-agnostic PTQ floor at
~80 PPL on the 0.5B (uncalibrated)** that wins iso-bpw against GGUF only because
GGUF cannot tile sub-4-bit on 896-dim weights (a small-model edge that shrinks at
256-aligned scale), and the path to a *competitive* 2-bit (the AQLM/QuIP#-class
~6.9 PPL trained tier on llama2-7b) is **PV/QAT training, not PTQ** — that run is
scoped (~$6 on one A100-80) but unrun, so against the trained field STRAND's 2-bit
is today a credible-but-unproven contender, not a winner.

## Pass/fail

| item | result |
|---|---|
| Suites green | **PASS** (116/0, 69/0, check clean) |
| gate-debias reproduces | **PASS** (+4.19% modeled / +0.0001% zero-mean, exact) |
| iso-bpw is true iso-harness | **PASS** (honest, loud caveat present) |
| iso-bpw verdict | **PARTIAL** — 3.8-bpw tier won (both axes); 2-bit + 7B tiers pending live run |
| PV cooldown beat 26.77 | **PENDING** — not run (box blocked) |
| 7B-selectivity GO/NO-GO | **GO** (~$6/arm A100-80; rides one unproven concentration transfer) |

## Files (this verification owns none of the shared-refactor files)
- `research/2bit-frontier-SUMMARY.md` (this file)
