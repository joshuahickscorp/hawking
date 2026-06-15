# STRAND master plan — all channels (2026-06-09)

_The unifying invariant across every channel: the **decode stays integer-only and bit-identical across
hardware** (the moat). The deep research proved this is not a tax but a **precondition** — bit-exact
parallel decode and the speed it unlocks are mathematically available ONLY in integer arithmetic; float
codebooks are locked out. "Learned codebook" is fine (decode is a frozen integer LUT); "model training"
(QAT) is the one flagged frontier that trades training-free for usable 1-bit._

Verified state: 4/3-bit ship today (7.81 / 9.42 PPL), density at the floor (+0.7–2.7%) and beating Q4_K
at every rung, CPU decode 6× via block-parallel. Open: **sub-2-bit quality collapses** (2-bit 213, 1-bit
worse), the **GPU/parallel speed** (serial wall now known-breakable), and the last **density + side-info**.

---

## Channel A — SPEED (break the serial-decode wall → bandwidth-bound)

**Goal:** decode that scales to the bandwidth limit on CPU and GPU, bit-exact.

- **A1 · GATE — is the decode even serial? (the windowed-shift-register test).** STRAND's decode state is
  `state[i] = (state[i-1] << k | sym[i]) & mask` — a SHIFT REGISTER of width `l` bits, whose closed form is
  `state[i] = the l-bit window of the symbol stream ending at bit (i+1)·k`: an **independent windowed read per
  weight**, NOT a true Viterbi/min-plus recurrence (that's the ENCODE). So the decode is plausibly
  **embarrassingly parallel per weight** (each lane reads its own `l`-bit window + LUT), and the GPU "serial
  wall" we measured is likely a per-block-serial KERNEL artifact, not the decode's nature. **Test: a
  per-weight-windowed decode — verify bit-identical to the serial decode, then measure bandwidth** (CPU first,
  then GPU). If bandwidth-bound, the GPU speed path is solved *simply*, no tropical scan needed (the only cost
  is ~`l/k ≈ 2.3×` redundant reads of overlapping windows — still far under the bandwidth ceiling).
  *Fallback:* only if a real residual dependency survives (e.g. tail-biting wrap forcing a per-block init), the
  tropical-semiring rank-convergence route (research doc) applies — measure STRAND's rank-convergence depth then.
- **A2 · Bitslicing** (lowest risk, fully moat-safe): pure-integer 32-stream batch decode = our CPU
  block-parallel extended to GPU (21 Gbps V100 precedent). Build + bench CPU & GPU.
- **A3 · Windowed / tail-biting** parallel decode with overlap sized for **provable sufficient-history**
  (we add the formal sizing the channel-coding papers only show empirically). Build + bench vs the 6× baseline.
- **A4 · Re-run the Metal gate** with the parallel decode — does 3-bit GEMV now clear bandwidth-bound?
- **Determinism:** A1/A2/A3 all integer-exact (A1 if convergence bounded; A3 if overlap proven). Tensor
  cores excluded (float). **Unlocks:** the GPU speed path I'd wrongly shelved; faster CPU.

## Channel B — QUALITY (make sub-2-bit usable — the hard frontier)

**Goal:** 2-bit and ~1-bit that don't collapse. Honest fork — determinism-compatible first, training as the flagged path.

- **B1 · GATE (determinism-compatible) — higher-dimensional vector trellis.** TCQ cost is `O(2^L·T)` —
  *linear in dimension* — so STRAND can push `vec_dim` far past d=2/3 (the collapsed q1) into the d≥4–8+
  regime where the **space-filling/shaping gap closes** (QTIP reports >3× gap reduction). Learned-but-FROZEN
  codebook ⇒ decode stays integer-deterministic. **Measure sub-2-bit rel-RMS + PPL at d=4,8,… vs the d=2/3
  collapse.** The open research question made concrete: *is the gap-closing realizable with a frozen integer
  LUT, or only with float?* This is the single most valuable quality experiment.
- **B2 · PTQ refinements:** better incoherence (rotation vs Hadamard, structured), Hessian-guided per-tensor
  bit allocation pushing tolerant tensors to 1-bit while protecting the sensitive few (bitmap-indexed salient,
  not the dead per-weight-index salient). Determinism-safe.
- **B3 · Training frontier (flagged):** QAT / BitNet-ternary — the *honest* route to genuinely usable 1-bit;
  STRAND is the **deterministic runtime** for it. Trades training-free for quality; a strategic choice, pursued
  only if B1/B2 can't reach usable 2-bit.
- **Unlocks:** if B1 closes the gap with a frozen LUT, **usable 2-bit/1.5-bit deterministic** — the bleeding-edge
  moat. If not, the honest answer is "format+runtime ready, quality is B3/training."

## Channel C — COMPRESSION / DENSITY FLOOR (closer to the physical floor)

**Goal:** fewer bytes/weight at iso-quality.

- **C1 · Close the shaping gap** = the *same* higher-dim vector trellis as B1. The ~0.2–0.35 bpw above the
  Shannon R-D bound is the VQ space-filling gap; higher dimension is the determinism-compatible lever (capped by
  the ~1.53 dB / 0.255-bit ultimate shaping gain). B1's frozen-LUT verdict decides how much is recoverable.
- **C2 · Side-info minimization:** we sit +0.7–2.7% above the floor, almost all of it the per-block scales
  (32-bit super-scale + 8×6-bit sub-scales = 80 bits/256-block). Predictive/shared/entropy-coded scales →
  shave toward the information-theoretic minimum. Determinism-safe (decode reads the same).
- **Unlocks:** the genuinely-densest deterministic representation at each quality.

## Channel D — RESEARCH (close the open theory)

- **D1 · Follow-up deep-research on Axes 2 & 3** — the first pass answered SPEED but barely touched
  compression and nothing on side-info: the quantitative shaping-gain limit + empirical TCQ-to-R-D gap vs
  trellis state/dimension, ECTCQ, the incoherence-transform-as-Gaussianizer math (Axis 2); the information-
  theoretic minimum side-info, predictive/entropy scale coding, fixed-point state/scale precision bounds, and
  the ops/quality ranking of FWHT vs random rotations vs structured transforms (Axis 3). Feeds B1/B2/C1/C2.

---

## Sequencing — the decisive gates first (parallelizable)

| now (gates) | tells us |
|---|---|
| **A1** rank-convergence | is the deterministic ~24× speed revival real for STRAND? |
| **B1 / C1** higher-dim vector trellis | does a frozen-LUT high-dim trellis fix sub-2-bit quality AND close the density gap? |
| **D1** Axes 2/3 research | the compression + side-info theory that directs B/C |

Then the build-out per gate result: A2/A3 (parallel decode) if A1 clears · B2/B3 (PTQ refine / training) per
B1 · C2 (side-info) always-worth-doing · A4 (re-gate Metal) after A2/A3.

**The honest priors going in:** A1 is high-upside and small (do it). B1 is the make-or-break for the quality
moat — if a frozen high-dim trellis works, sub-2-bit deterministic is real; if it needs float/training, that's
the honest boundary and B3 becomes the path. C2 is a sure small win. D1 de-risks B/C.
