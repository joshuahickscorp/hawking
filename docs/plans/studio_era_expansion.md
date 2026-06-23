# Studio-Era Expansion — Hardening + Aggression for Hawking Condense

Forward plan for when the Mac Studio (96 GB+) lands. The mechanics live in
[parameter_sweep_pipeline.md](parameter_sweep_pipeline.md) (the floor-search harness:
`ladder.py` / `sweep.py` / `sweep_render.py`). This doc is the **strategy**: what to push,
what to harden, and how each model run feeds back into the methodology. Expand the open
hooks as concrete findings land.

## The reframe (what the Studio changes)
- **Now (19 GB):** the demo I can actually run is **7B naked vs 32B condensed** — the parent
  that fits normally vs the big model that *only* runs because Hawking condensed it. That IS
  the RAM-cliff story at testable scale. The 0.5B/7B ladder audit maps the bit-floor on what fits.
- **Studio (96 GB+, bf16):** unlocks the full multi-family ladder (to 235B/405B/671B), the
  **bit-floor-vs-scale curve** as the headline science, a stronger doctor at scale, and the
  1-bit-at-scale frontier. bf16 NOT f32 (32B f32 = 128 GB won't fit 96; bf16 = 64 GB fits —
  bf16 is what keeps 32B/405B in the native-condense tier).

## The core contract (the one hypothesis the whole thing tests)
**Floor-search, not a fixed grid:** for each model, climb *effective* bpw across recipes
(single-bake AWQ · residual b1+b2) and **stop at the lowest eff-bpw the doctor holds near-1:1.**
`floor × params = smallest full-capability artifact = highest tps`. The hypothesis the ladder
exists to falsify: **the bit-floor DESCENDS as params rise** (0.5B floors ~3-bit; 32B maybe ~2;
405B maybe ~1). If that curve holds, the *biggest* models compress *hardest* → **1-bit-405B is
the headline.** The deliverable is that curve, per model, overlaid — the gap between columns is
the redundancy result, not noise.

## "Harnessing the models to improve our science" (the feedback loop)
Every model run is not just a datapoint — it *calibrates the method*. Each floor we find tunes
the doctor and the recipe ladder; each failure (1-bit-on-0.5B catastrophic; uniform-QAT
diverges) prunes a dead-end permanently. The substrate ladder is deliberate:
- **0.5B** = worst-case stress (no redundancy → pessimistic floor, fast iteration).
- **7B** = the honest mid substrate (the 1-bit floor judged here, not on 0.5B).
- **32B+** = the redundancy payoff (where low-bit should start winning).
- **405B/671B** = the capstone (the fit-cliff; 1-bit-or-nothing).
The contrast across them *is* the science: it measures how redundancy buys compressibility.

## Aggression axes to push on the Studio (each a lever, ranked by headline value)
1. **Drive the aggressive end at scale.** 1-bit / ~1.34-bit on 235B/405B; the **405B
   single-1-bit-or-nothing** cliff (fits 68 GB at 1-bit; every higher rung overflows 96 GB);
   671B at the ~1.0-bpw absolute edge.
2. **The doctor at scale = the gating capability.** Residual (`residual_bake.py`, train-free,
   full-rank, codec-native) is the primary heal and supersedes LoRA — *but it costs +bpw*, so
   at the extreme-fit frontier (405B ≤1.34 bpw) it can't afford a residual pass. There,
   **single-bit viability is the open question**, and **block-wise full-rank QAT** stays the
   unbuilt ceiling-breaker worth attempting *at scale* (it failed on 0.5B's trellis, but the
   redundancy hypothesis says big models may behave differently — test, don't assume).
3. **Recipe-ladder breadth:** AWQ α-sweep · residual allocations (3+2/2+2/2+1/1+1) · AWQ×residual
   stacked · **mixed-precision** (`--mp-config` per-layer bits — keep output-sensitive tensors
   higher, push tolerant ones lower) · **c2f-outlier** entropy coding (the baker's ~0.15 bpw
   side-info lever). Mixed-precision is the next big density lever after residual.
4. **Multi-eval rigor:** graduate from one ppl passage to **downstream task accuracy** (the real
   capability check), per-domain — so "near-1:1" means capability-preserved, not just ppl-preserved.

## Hardening (the rigor invariants — stay honest as we get aggressive)
- **Effective-vs-nominal bpw, always.** Report the baker's `AGGREGATE effective bpw` (RHT +
  outlier + residual-pass overhead included); residual sums both passes. Nominal is a lie at the
  margin — the 405B fit-cliff turns on the real number.
- **Quality gates:** `~1:1 ≤ +2%` · `beats-llama ≤ +8%` (the Q4_K line: 4.5 bpw, ~+8%). A floor
  only counts if it clears a stated gate.
- **Two streams, never conflated:** A = **quality** (residual proves it; no serve dependency),
  B = **serve-tps** (single-bake, native GPU `.tq`). The **residual two-part `.tq` serve path is
  the key unbuilt gap** → a top Studio priority (today residual proves quality but doesn't serve;
  the 405B WIN deliberately uses single-bake so it doesn't depend on it).
- **Small-param skew flag:** the 0.5B 1-bit cell is reported but never sets the verdict; 1-bit is
  judged on the big substrate.
- Disk discipline; GPU jobs sequential; bf16 everywhere at scale.

## Sequencing
- **Now:** 0.5B + 7B ladder audit → first concrete bit-floor datapoints. Re-run 7B with the bf16
  fix (the earlier f16 run nan'd). Then the 32B-condensed vs 7B-naked cliff demo (needs the native
  `.tq` serve path).
- **Studio:** full ladder → the bit-floor-vs-scale curve → 235B/405B fit-cliff → block-wise doctor
  attempt for 1-bit-at-scale → the headline (1-bit-405B if the curve holds).

## Sub-1-bit frontier (fractional bpw) — the deep version of the bit-floor hypothesis
**We are already fractional.** Effective bpw is continuous (the baker reports 2.594, 4.81 …);
the integer `--bits` is just the trellis quantization level, not the artifact's real density.
Going *below 1 bpw* is a different regime needing a different mechanism than the bit knob:
- **Large-block codebook/trellis:** encode K weights per index → `idx_bits/K` bpw (STRAND's
  "k bits per d weights" packed lever; `--bits` floors at 1 *per step*, but k/d can be <1).
- **Sparsity:** prune most weights, store survivors + positions → amortized
  `bpw ≈ p·(b + log2(1/p))` (90% sparse @2-bit ≈ 0.5 bpw). The surer sub-1-bit path.

**Information theory — there is no fixed 1-bit floor.** The floor is `MDL(function)/n_weights`,
which *shrinks as params grow*. So the bit-floor-descends hypothesis **predicts sub-1-bit for
big-enough models**: a 405B computing a function a ~70B could approximate carries >5× redundancy
⇒ <1 bpw is plausible. The doctor (residual/AWQ heal) **shifts the whole curve down** — making
sub-1-bit viable where *raw* sub-1-bit isn't, even when we deliberately grade off strict 1:1.

**How low, realistically (per model, doctor-assisted):**
- ~1.5–2 bpw: near-lossless on big models (residual already hits ~1:1 at the 3+2 equivalent).
- ~1 bpw: the edge (the 405B-fits story); doctor-dependent.
- ~0.5 bpw (1/2): plausible ONLY big + sparse + doctor, grading a few % off 1:1 = **the novel offer**.
- ~0.1 bpw (1/10): almost certainly breaks — too little of the function's information survives,
  except extreme overparam or specific tolerant layers (mixed-precision territory).

**DISCIPLINE (the shipability honesty):** sub-1-bit is a RESEARCH frontier — prove the quality
curve (Stream A), keep it SEPARATE from the shippable core (2–3-bit single-bake serves today;
the residual/sparse `.tq` serve path is unbuilt). Judge it on 7B+/big models, NEVER the 0.5B
(floors ~3-bit). Mechanically it's the SAME floor-search, just extended below 1 bpw with
sparsity + large-block recipes — the curve continues; the doctor sets how far.

## Open hooks — fill as findings land (this is the "expand once concrete" part)
- [ ] **Does the bit-floor descend with scale?** (the curve from 0.5B → 7B → 32B). The whole thesis.
- [ ] **Where does residual stop paying?** (the eff-bpw cliff per model — when +bpw stops buying ~1:1).
- [ ] **Is 1-bit viable at scale with the doctor?** (the 405B unlock; block-wise QAT retried at scale).
- [ ] **Is the residual `.tq` serve path built?** (gates Stream B for the residual recipes).
- [ ] **Does ppl-1:1 == capability-1:1?** (multi-eval validation before any "lossless" claim).
