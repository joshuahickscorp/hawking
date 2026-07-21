# Kimi K2.6 Tested-Region Closure

- Decision: **TESTED_LINEAR_REPAIR_REGION_CLOSED**
- Independent held-out tokens: `8672`
- Tested ceiling: `0.98` complete BPW
- Best retained candidate: `P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY`
- Best retained BPW: `0.9085909525553385`

## Causal closure

- Diagnosis: `UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER`
- Teacher indices+weights rescue: `0.162511`
- Teacher weighted-MoE rescue: `0.362110`
- Teacher hidden rescue: `1.000000`
- Route-weight rescue after crossing: `0.463332`
- Route-weight rescue with route matched: `0.009399`

## Closed families

- first-divergence low-margin state protection
- low-margin router-logit correction
- pre-router low-rank hidden repair
- weighted-MoE-output low-rank repair
- pre-router plus post-MoE hybrid
- calibration-CV post-MoE shrinkage
- upstream compact-output linear residual

## Next architecture

Test a representation-side nonlinear structural allocation at F0/F1 that directly reduces compact expert-output state error before any router. Do not spend more bits on generic downstream low-rank Doctor paths; require disjoint-score F1 evidence before F2.
