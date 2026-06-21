# Throughput-Pivot Campaign — live autonomous run (started 2026-06-21)

**Why:** spec-decode (EH free market AND trained EAGLE) is conclusively NET-NEGATIVE on
this engine — the per-cycle overhead wall (proven: 87% accept → still 0.91×). Speed must
come from the bandwidth-bound decode levers (quantization / KV / dispatch / kernel), per the
throughput bible. This campaign architects + tests that pivot, unattended, for ~4 hours.

**Goal artifacts (what the user returns to):**
1. A clean, measured decode-tps baseline (multiple prompts/trials).
2. The best throughput CONFIG from existing opt-in levers (a measured sweep).
3. A ranked, validated ROADMAP of new optimization candidates (lever · expected gain ·
   implementation sketch · risk · measurement), grounded in the real hot path + the bible.

**Hard rails (unattended safety):**
- NO auto-merge of risky kernel/code changes. Prototypes are designed + patched + benched,
  reviewed by the user — never committed to main.
- Every measured change passes a quality gate (bit-identity / token-regression) before it
  counts as a win.
- GPU work is SEQUENTIAL (no concurrent model jobs → no OOM).
- Only config-level (env-flag) changes may be auto-benched; code changes are design-only.

## Lanes
- **Lane A (GPU, the test):** decode-tps baseline + opt-in-lever config sweep → best config.
- **Lane B (CPU, the architect):** chained workflows → ranked optimization roadmap.

## Round chain (the watchdog advances this; update STATE below each round)
- **R0 — audit + baseline:** workflow audits the decode hot path (qwen_dense.rs) + reads the
  throughput bible (docs/plans/bible_*) → ranked candidate levers; one agent measures a clean
  tps baseline. OUT: candidate list + baseline.
- **R1 — research:** deep-research Apple-Silicon / Metal small-model decode SOTA (2026) →
  techniques applicable to Hawking; cross-check vs R0 candidates.
- **R2 — config sweep (Lane A):** bench the existing opt-in levers (predec, Q4K_FAST,
  f16-scales, vocab-prune, profiles, KV opts) + key combinations, tps + quality gate → best config.
- **R3 — synthesize roadmap:** merge R0+R1+R2 → ranked, validated roadmap (the deliverable).
- **R4+ — design top candidates:** for each top roadmap item, an agent produces a grounded
  implementation patch + a measurement plan (design-only, no merge). Loop until time/budget.

## STATE (watchdog updates this on every wake)
- 2026-06-21: campaign created. Parity re-verify running (router commit pending pass).
- R0 DONE (wf wm11hndt4). Baseline ~31 tps anchor / ~35-40 release; gap ~1.6× to llama.cpp (kernel-bound).
  Ranked roadmap:
  - #1 flip PREDEC_F16SCALES default-on — **LIKELY DEAD** (tried e613dde, failed quality oracle
    0.792<0.90; stays opt-in). Re-verify oracle once to confirm, else dismiss.
  - #2 **Q6_K predec for default ffn_down** (the #1 GPU consumer ~46%, never got the +34% predec
    win — Q4_K-only). HIGH gain, Stage-1 BIT-IDENTICAL → quality free. ← TOP REAL PICK, designing now.
  - #3 continuous batched decode B=8 — AGGREGATE tps, not single-stream (off-goal).
  - #4 per-channel int4-KV (long-ctx BW/footprint). #5 MST diff vs llama.cpp (high ceiling/high risk).
    #6 GQA KV coalescing (long-ctx only).
  NEXT: (a) design Q6_K predec patch [wf, running]; (b) commit router on parity-pass;
  (c) R2 config sweep + #1 oracle re-verify on GPU-free; (d) R1 SOTA research.

## Watchdog
- A heartbeat ticker re-invokes the manager ~every 25 min; workflow/bench completions also
  re-invoke it. On each wake: read STATE → if a round finished, record its output + launch the
  next; if a round is mid-flight, no-op; re-arm the ticker. Campaign ends at ~4h or budget.
