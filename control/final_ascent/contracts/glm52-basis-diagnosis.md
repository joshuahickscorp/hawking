# Independent GLM-5.2 basis diagnosis

## Role

Act as an independent falsifier. Do not implement or edit files. Read the current
repository at final-ascent HEAD, including receipts committed after
`299208e6e2ed22d6b74c6bbd391dc631b4e29ec5`, and the exact evidence named below.
The controller's current hypothesis may be wrong.

## Governing evidence

- `HAWKING_RESUME_CHECKPOINT.md`
- `GLM52_GENERATION_B_CAPABILITY_VERDICT.json`
- `GLM52_BYTE_ATTRIBUTION.json`
- `GLM52_MATH_PRESERVE_V2_PLAN.json`
- `GLM52_REBUILD_PILOT_RECEIPT.json`
- `GLM52_REAL_ACTIVATION_SWEEP.json`
- `QUALITY_CALIBRATION_CURVE.json` and `.md` if present in another reachable
  Grok worktree/commit; otherwise state that they are absent at this HEAD
- `HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json`
- `tools/condense/glm52_activation_aware_pack.py`
- `tools/condense/tests/test_glm52_activation_aware_pack.py`
- the retained real teacher capsules under
  `/Users/scammermike/Library/Application Support/hawking/GLM52Gravity/source_fetch/teacher/capsules`

## Current controller hypothesis to falsify

Generation B's allocation objective was invalid because it minimized BPW subject
only to beating a constant-mean null. In addition, its basis builder centered
`X_fit` before SVD, discarding the mean activation direction. The smallest
decisive next experiment is a real-capsule, held-out comparison of:

1. centered residual basis,
2. uncentered SVD basis,
3. explicit mean direction plus centered residual basis.

The comparison must use the same fit/holdout rows, tensors, ranks, byte accounting,
and capability-calibrated quality metrics. Routed-expert promotion scores must
use only held-out rows whose real capsule `topk_indices` routed to that expert;
the existing all-row expert score is a competing causal defect, not valid
promotion evidence. A full 282-shard traversal is forbidden until a bounded arm
passes preregistered floors.

## Required report

Return:

1. an independent diagnosis of the earliest causal divergence;
2. at least three competing explanations, including one in which basis centering
   is not the dominant defect;
3. the exact smallest experiment that distinguishes them;
4. tensor/organ sampling that is representative and actually executable from
   the five resident immutable shards in the source-rehydration receipt plus
   real capsules, including high-traffic experts at early/middle/late layers;
5. metrics, nulls, physical-byte accounting, floors, and kill/promotion rules;
6. risks of leakage, unfair basis rank, double-counted explicit mean storage,
   wrong projection side, Gaussian fallback, or output-space-only inference;
7. whether `down_proj` can be evaluated with reconstructed real routed SwiGLU
   intermediates rather than Gaussian or output-only inference;
8. remaining uncertainty after the bounded experiment.

Do not accept reconstruction error, `beats_null`, or a fixture as promotion
evidence. Do not flip `ODYSSEY_LAUNCH_AUTHORIZED`,
`RAMANUJAN_RESEARCH_AUTHORIZED`, or `HIDE_KERNEL_TURN`.
