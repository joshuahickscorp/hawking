# Path-to-150 phase plan — index

After the L8 iter 4 break (K=2 vector gate clears 33% chain accept,
vs scalar's 7% ceiling), the path to 150 dec_tps decomposes into
discrete engineering phases. Each has its own plan doc. This file
is the index + dependency graph.

## Status snapshot (commit `7f1a034`)

- **Off baseline**: 26.78 dec_tps
- **Eagle4 chain K=2 (iter 4 step 400)**: 33.3% accept → projected ~35-40 dec_tps
- **Architectural blocker (scalar gate)**: CLEARED
- **Auto-chain queue running**: iter 4 → iter 5 (K=4 vector) → iter 6 (K=4 + chain-reg, fallback)

## Phase plan & dependencies

```
NOW running (autonomous compute, no engineering needed)
└── iter 4/5/6 + L7.1 clean bench  →  50-60 dec_tps  [+1-5 hr]
       │
       ▼
   Phase L5 — chain-decode pipeline restructure
   (uses already-shipped SharedEvent + TCB-on-secondary)  →  +3-8 dec_tps  [+3-6 hr code]
       │
       ▼
   Phase L7 — kernel rewrites
   (L7.2 single-kernel fusion + Stage 0.5 broader MLX)    →  60-90 dec_tps  [+2-3 days code]
       │
       ▼
   Phase E — tree decode
   (new verify kernel + branching head decoder)            →  100-115 dec_tps  [+1-2 weeks code]
       │
       ▼
   Phase F — EAGLE-3 medusa multi-token head
   (new head architecture + new training pipeline)         →  130-150 dec_tps  [+2-4 weeks code]
```

## Per-phase plans

| Phase | Plan file | Engineering est. | Expected dec_tps |
|---|---|---|---|
| L5 | [phase_l5_chain_pipeline.md](phase_l5_chain_pipeline.md) | 3-6 hr | +3-8 |
| L7 | [phase_l7_kernel_rewrites.md](phase_l7_kernel_rewrites.md) | 2-3 days | +15-30 |
| E | [phase_e_tree_decode.md](phase_e_tree_decode.md) | 1-2 weeks | +30-50 |
| F | [phase_f_medusa.md](phase_f_medusa.md) | 2-4 weeks | +20-40 |

## Cross-cutting

| Doc | Purpose |
|---|---|
| [acceleration_patterns.md](acceleration_patterns.md) | Patterns distilled from L8 iter 1→4 to avoid wasted compute on future phases. **Read this BEFORE starting any phase.** |

## Working principle for future sessions

Each phase plan is written for a focused future session to execute
end-to-end. The acceleration_patterns doc is the meta-lever — apply
its checklist before each phase to avoid the kind of multi-iter
troubleshoot loop L8 went through (4 iters before vector gate cracked
the ceiling).

Phases L5, L7, E, F are largely INDEPENDENT once their prerequisites
are met. They can be tackled in any order or in parallel by different
sessions / branches. The recommended order (above) maximizes
compounding dec_tps gains.
