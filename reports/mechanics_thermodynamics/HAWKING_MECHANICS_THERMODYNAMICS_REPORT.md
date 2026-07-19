# HAWKING MECHANICS & THERMODYNAMICS — FINAL REPORT (run-all)

updated 2026-07-19T05:17:59Z  branch codex/hawking-mechanics-thermodynamics  base f5521233  Generation F e0857398 frozen

## 1-2. Handoff / G2
Starting main f5521233. G2 COMPLETE (10/10 sealed, lease released) -> non-interference satisfied; Generation-M experiments admitted.

## 3-5. Generation M
`HAWKING_GENERATION_M.json` (66f89835). Apple M3 Ultra 20P+8E 96GB MPS torch2.6.

## 4. Energy instrumentation
UNAVAILABLE - powermetrics needs sudo (not bypassed, Bible section 33/68). All 20 thermodynamics rows UNAVAILABLE; NO invented estimates. Timing MEASURED but contaminated by concurrent MoP CPU load; paired/relative ratios preferred.

## 6-14. Stage results (real, layer-0 routed experts, reps=5 paired)
- B0 source-native direct-compact baseline: sealed (the reference).
- B1 bounded reconstruction control: sealed; no dense shadow (temp < 0.5x dense).
- M1 lookup-linear PQ: POSITIVE - 2.5-3.3x faster CPU vs B0 compact at ZERO quality cost; M1==B1 ~1e-7; CPU/Metal parity ~1e-7. (Dense BLAS faster but INADMISSIBLE.)
- M2 shared lookup-linear MoE: POSITIVE mechanical - 75% table-builds avoided, 64-69% fewer FLOPs, 3-7% faster, matched-or-better quality.
- M3 shared MoE + islands: mlp2 improves 0.144 (magnitude), mlp1 no regression -> islands are the mlp2 lever.
- M4 fused PQ+islands+Doctor: launches 27->19, temp reduced, exact-match fusion (0.0).
- M5 conditional Doctor: NEGATIVE - gate never fires (Doctor always material at deep sub-bit), 2.7-2.9x slower. Sealed honest negative.
- M6 residual/additive lookup: WINS mlp1 (0.654 vs 0.805) at equal bits; M4 wins mlp2.
- M7 bit-oriented: DEFERRED (section 77).

## 15-24. Quality / mechanics / parity / islands / Doctor
Quality NEGATIVE at sub-bit (inherited from Gravity): rel_err 0.65-0.88 mlp1 (high-rank, collapses) / 0.20-0.36 mlp2. Mechanical wins are quality-NEUTRAL. CPU/Metal parity ~1e-7 (Metal Quality Law). Islands help mlp2; Doctor always material (M5 negative). No dense shadow anywhere (asserted).

## 25-28. Energy / lifecycle
UNAVAILABLE; lifecycle break-even not computable (no per-token energy).

## 29-30. Pareto / champion
`HAWKING_MECHANICS_PARETO.json`: 6-candidate frontier, 0 inadmissible-dense. Quality-preserving-speed champions: **mlp1 -> M6 residual_additive_lookup; mlp2 -> M4 fused_pq_islands_doctor; base mechanical -> M2 shared-table**. A clean tensor-class-dependent split at equal bits.

## 31-34. Generation M / merge / handoff / rollback
Generation M sealed (66f89835). Shadow-replay green (Gen-M lookup == Gen-F direct, rel-err 2.3e-7; artifact BPW + quality unchanged; no dense shadow). NOT merged to main (house rule: no push/merge without approval); durable checkpoint committed to the branch. Rollback: git reset --hard f5521233.

## Required statement
Hawking Mechanics has executed every mandatory stage B0/B1 through M1-M6 with M7 formally deferred. Hawking Thermodynamics has honestly classified Apple energy as UNAVAILABLE (no sudo) and timing as contaminated, inventing no estimates. Generation M contains the selected quality-preserving execution grammars (tensor-class-dependent: M6 for mlp1, M4 for mlp2, M2 shared-table base), and the result is a MECHANICAL win on a Gravity-NEGATIVE representation - no capability pass, no Event Horizon. The tournament, Pareto frontier, and Generation-M closure are sealed; the 120B->685B->1T Frontier ladder is unaffected and ready to resume, and the champion grammars are ready to apply when a quality-positive representation is found.
