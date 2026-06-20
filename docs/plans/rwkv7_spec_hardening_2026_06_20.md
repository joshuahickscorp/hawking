# RWKV-7 Spec-Decode Hardening
**Date:** 2026-06-20 14:54 UTC

## Assumptions

| Metric | Value |
|---|---:|
| target_tps | 31.00 |
| llama_tps | 50.00 |
| K | 4 |
| verify_equiv | 1.15 target-forwards |
| accept_floor | 60% |
| required margin | 1.10x llama |

## Variant Physics

| Variant | Params M | Draft TPS est | Accept | PPL | Effective TPS | vs llama | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| draft_35m_probe | 34.7 | 400 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_50m_probe | 49.9 | 400 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_75m_probe | 76.1 | 400 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_100m | 100.1 | 400 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_150m | 166.2 | 375 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_200m | 200.2 | 246 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |
| draft_300m | 300.5 | 146 | pending | pending | pending | pending | **PENDING**: no accept-rate eval yet |

## Break-Even Accept

| Variant | Accept needed to match llama |
|---|---:|
| draft_35m_probe | 61.17% |
| draft_50m_probe | 61.17% |
| draft_75m_probe | 61.17% |
| draft_100m | 61.17% |
| draft_150m | 61.96% |
| draft_200m | 67.97% |
| draft_300m | 77.85% |

## Compression Doctrine

1. Promote the smallest draft that passes accept floor, predicted TPS margin, and measured verify cost.
2. If the smallest passing draft is not the smallest configured probe, immediately train the next smaller probe before extending larger models.
3. With untied 65k input/output embeddings and 256-aligned width, the practical configured floor is about 35M parameters; going below that requires changing the architecture, not just the layer count.

## Shrink Rule

1. Do not shrink yet: no evaluated draft clears the spec physics gate.
2. First improve accept rate with target-logit KD, then re-run this hardening pass.
3. Once a draft passes, launch the nearest smaller configured probe with `DRAFT_VARIANTS="draft_75m_probe draft_50m_probe"` before scaling anything up.

## Hard Rule

No runtime spec-decode promotion unless an evaluated draft clears accept floor, predicted TPS margin, and measured K-wide verify cost.
