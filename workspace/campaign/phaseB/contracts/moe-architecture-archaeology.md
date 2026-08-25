# Task contract — moe-architecture-archaeology

## Objective
Write a measured-from-source architecture + representation-opportunity map for the two Odyssey MoE candidates,
plus an honest assessment of what our Qwen3.8-hybrid decoder would need to run either natively. Read-only,
deliverable is one markdown report. No weight downloads (weights are being fetched separately); use configs +
framework source + the Qwen3.8 legacy index.

## Context
Steer S042 opens a two-model MoE campaign. Candidate A = huihui-ai/Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated
(model_type qwen3_moe: 48L, hidden 2048, 128 experts, top-8, moe_intermediate 768, ~3B active/30B). Candidate B =
ArliAI/GLM-4.5-Air-Derestricted (model_type glm4_moe: 46L, hidden 4096, 128 routed + 1 shared expert, top-8,
moe_intermediate 1408, first_k_dense_replace 1, ~12B active/106B). Our native decoder
crates/hawking-core/src/model/qwen38_hybrid_decode.rs is Qwen3.8-hybrid-specific (dense SwiGLU MLP + GQA + DeltaNet).
Qwen3.8 science is indexed in receipts/ascent-2026-08-18/QWEN38_LEGACY_INDEX.json (affine2/HGRAVF01 kernel + native
GEMV are UNIVERSAL; the operator-distillation + density-doesnt-buy-TPS findings are dense-specific). The mlx_lm and
transformers model classes for qwen3_moe + glm4_moe are installed under ~/.grok-vision/lib/python3.12/site-packages.

## Read these sources
- READ ~/.grok-vision/lib/python3.12/site-packages/mlx_lm/models/qwen3_moe.py and glm4_moe.py (and transformers
  equivalents if present) for the exact expert/router/shared-expert/attention topology + forward structure.
- READ crates/hawking-core/src/model/qwen38_hybrid_decode.rs (dispatch + kind machinery) to assess the MoE gap.
- READ receipts/ascent-2026-08-18/QWEN38_LEGACY_INDEX.json + DENSITY_LEVER_HONEST.json + G26_RUNTIME_DIAGNOSIS.json
  to know which Qwen3.8 negatives are UNIVERSAL (do not re-test) vs dense-specific (reopenable on MoE).

## Deliverable
- Write workspace/campaign/moe_arch_map.md.

## Do not touch
- Read-only. No code changes. No mlx/model execution (no weights yet). No weight downloads.

## Acceptance criterion
Done when workspace/campaign/moe_arch_map.md exists, is non-empty, and covers: (1) exact expert/router/
shared-expert/attention topology of BOTH models cited to the framework source; (2) the per-token ACTIVE vs STORED
parameter split for each (which tensors are touched per token); (3) what our qwen38 decoder would need to run each
natively (new kinds, router dispatch, expert gather) named against real functions; (4) which representation
mechanisms are newly-viable on MoE (per-expert codec, cold-expert compression, router-aware, Matryoshka experts)
and which Qwen3.8 negatives stay closed (cite the legacy index). Conservative: label each opportunity
UNIVERSAL/MOE-SPECIFIC/UNRESOLVED. Must pass verify.

## Verify
- VERIFY: test -s workspace/campaign/moe_arch_map.md

## Required evidence
Quote exact class/function names + line numbers from the framework source and our decoder. No hand-waving.

## Limits
- Max turns: 40. No commits other than the branch you are on. No push/merge/deploy. External network: not permitted.

## Required completion report
End with: ## SUMMARY / ## FILES CHANGED / ## DECISIONS MADE / ## ASSUMPTIONS / ## DEVIATIONS FROM CONTRACT / ## TESTS RUN / ## KNOWN LIMITATIONS / ## REMAINING WORK / ## CONFIDENCE
