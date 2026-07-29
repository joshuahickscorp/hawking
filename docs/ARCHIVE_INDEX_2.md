# Archive index — floor prune (2026-07-28)

Working-tree floor prune of no-reader root receipts and later closed-campaign
artifacts. **Nothing is lost:** full content lives in git history at the annotated
tag `pre-floor-prune-20260728`. Restore any file with:

```
git checkout pre-floor-prune-20260728 -- <path>
# or
git show pre-floor-prune-20260728:<path>
```

## T1 — root receipts with no executable-code readers — 196 files

Pruned **196 tracked root files** (~58,934,431 bytes,
~2,068,345 lines). Reader map was built by scanning every `.py`/`.rs`/`.sh`/`.toml`/`.ts`/`.tsx`
source under the repo (excluding `.git`, `target`, `models`, `node_modules`) for the
exact basename. Files with any hit stayed. Also kept: build/config, the hard path-loaded
receipts (`tools/odyssey/known_failures.py`, `crates/hawking-adapters/src/evidence.rs`,
campaign/prometheus/condense path constants), and the 14 `must_not_delete` themes from
`HAWKING_CONSOLIDATION_INVENTORY.json` (acceleration verdicts, KIMI scientific law /
long-run / gravity finals, GLM52 functional + Math-Preserve + corrected law, ascension/
odyssey chain, DEEPSEEK contraction pilot, doctor negative atlas, HIDE/Fabric scaffolds, etc.).

## T1 / HAWKING — 100 files

- `HAWKING_ASCENSION_FINAL.json` — FUNCTIONAL_PARTIAL_ONLY on DeepSeek, same terminal category as GLM by a different path
- `HAWKING_ASCENSION_FINAL.md` — Hawking Ascension — final position
- `HAWKING_ASCENSION_FUNCTIONAL_GRAVITY_CONTINUATION.md` — HAWKING ASCENSION: FUNCTIONAL GRAVITY CONTINUATION
- `HAWKING_ASCENSION_STATUS.json` — JSON receipt
- `HAWKING_ASCENSION_STATUS.md` — Hawking Ascension status
- `HAWKING_ATTENTION_PRELUDE_ICB_REPLAY_20260727.json` — ATTENTION_PRELUDE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_ATTENTION_PRE_SCORE_ICB_GROUPING_20260727.json` — ATTENTION_PRE_SCORE_ICB_GROUPING_INTEGRATED_DEFAULT_OFF
- `HAWKING_BASE_RUNTIME_G2_RESULT.json` — hawking.gravity.base_runtime_g2.v1
- `HAWKING_BASE_RUNTIME_ROOT_CAUSE.json` — hawking.gravity.base_runtime_root_cause.v1
- `HAWKING_BASE_RUNTIME_WARM_RESULT.json` — hawking.gravity.base_runtime_warm.v1
- `HAWKING_BF16_NATIVE_RECEIPT.json` — lm_head.weight
- `HAWKING_BRIDGE_ADAPTERS_FINISH.json` — FINISHED
- `HAWKING_BYTE_CATEGORY_LEDGER.json` — JSON receipt
- `HAWKING_BYTE_SURPLUS_ROOT_CAUSE.json` — hawking.gravity.byte_surplus_root_cause.v1
- `HAWKING_CANONICAL_AUTHORITY_MAP.json` — MULTIPLE_LIVE
- `HAWKING_COMMAND_GRAPH_AUDIT_20260726.json` — hawking.gravity.command_graph_audit.v1
- `HAWKING_COMPACT_ATTENTION_ICB_REPLAY_20260727.json` — COMPACT_ATTENTION_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_COMPACT_ATTENTION_REPLAY_GRID_AUDIT_20260727.json` — COMPACT_ATTENTION_REPLAY_BOUNDARY_ENUMERATED_NO_OVERDISPATCH_ADMITTED
- `HAWKING_COMPACT_MLA_ADMISSION_PREFLIGHT_20260727.json` — QUALIFIED_DEFAULT_OFF_PREALLOCATION_ADMISSION
- `HAWKING_COMPACT_MLA_APPEND_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_CACHE_OWNER_20260727.json` — PRODUCTION_UNREACHABLE_OWNER_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_FIVE_DISPATCH_DAG_20260727.json` — SYNTHETIC_COMPLETE_ATTENTION_DAG_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_LIVE_CANDIDATE_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_NOT_PERFORMANCE_PROMOTED
- `HAWKING_COMPACT_MLA_PARITY_PROOF_20260727.json` — hawking.gravity.compact_mla.cpu_architecture_proof.v1
- `HAWKING_COMPACT_MLA_PRODUCTION_FEASIBILITY_AUDIT_20260727.json` — hawking.gravity.compact_mla.production_feasibility_audit.v1
- `HAWKING_COMPACT_MLA_THREE_DISPATCH_CHAIN_20260727.json` — SYNTHETIC_CHAIN_PROOF_NOT_PROMOTED
- `HAWKING_COMPACT_RANKED_ATTENTION_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_CONSOLIDATION_INVENTORY.md` — Hawking consolidation inventory
- `HAWKING_DEVICE_ATTENTION_PRELUDE_CLOSED_20260727.json` — COMPLETE_TOKEN_PARITY_QUALIFIED_DEFAULT_OFF_NOT_FLAGSHIP_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_ATTENTION_RESIDUAL_CLOSED_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_DEVICE_GRAPH
- `HAWKING_DEVICE_DSA_CLOSED_INDEXER_GRAPH_20260727.json` — COMPLETE_TOKEN_PARITY_QUALIFIED_DEFAULT_OFF_NOT_FLAGSHIP_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_DSA_LIVE_CANDIDATE_20260727.json` — PARITY_QUALIFIED_GRAPH_CLOSED_DEFAULT_OFF_NOT_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_DSA_POST_SCORE_ICB_REPLAY_20260727.json` — DEVICE_DSA_POST_SCORE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_DSA_PRE_SCORE_ICB_REPLAY_20260727.json` — DEVICE_DSA_PRE_SCORE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_DSA_RADIX_TOPK_20260727.json` — BOUNDED_32K_EXACT_AND_MEASURED_DEFAULT_OFF_NOT_FLAGSHIP_PROMOTED
- `HAWKING_DEVICE_EXPERT_ADDRESSING_AUDIT_20260727.json` — READ_ONLY_AUDIT_CACHE_INDEXED_HIT_PATH_RECOMMENDED
- `HAWKING_DEVICE_EXPERT_EXECUTION_PERMUTATION_20260727.json` — BOUNDED_PREREQUISITE_QUALIFIED_NO_PERFORMANCE_CLAIM
- `HAWKING_DEVICE_EXPERT_RESIDUAL_CLOSED_20260727.json` — BOUNDED_COMPLETE_TOKEN_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_COMPLETE_WAVE_20260727.json` — BOUNDED_COMPLETE_WAVE_PROOF_QUALIFIED_DEFAULT_UNUSED
- `HAWKING_DEVICE_EXPERT_TABLE_HETEROGENEOUS_20260727.json` — HETEROGENEOUS_SELECTED_ROUTE_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_HIT_PROOF_20260727.json` — BOUNDED_INDIRECT_PROJECTION_PROOF_QUALIFIED_DEFAULT_UNUSED
- `HAWKING_DEVICE_EXPERT_TABLE_ICB_REPLAY_20260727.json` — DEVICE_EXPERT_WAVE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_NATIVE_BF16_20260727.json` — NATIVE_BF16_INDIRECT_PROJECTION_QUALIFIED_LOW_LEVEL_ONLY
- `HAWKING_DEVICE_EXPERT_TABLE_PACKED_R0_20260727.json` — PACKED_R0_INDIRECT_PROJECTION_QUALIFIED_LOW_LEVEL_ONLY
- `HAWKING_DEVICE_EXPERT_TABLE_PERSISTENT_LEASE_20260727.json` — PERSISTENT_SELECTED_ROUTE_LEASE_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_PRODUCTION_WIRING_20260727.json` — PRODUCTION_WIRED_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_FINAL_HEAD_ICB_REPLAY_20260727.json` — DEVICE_FINAL_HEAD_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_FINAL_NORM_HEAD_GRAPH_20260727.json` — BOUNDED_PARITY_QUALIFIED_DEFAULT_OFF_WITH_INHERITED_NATIVE_HEAD_BLOCKER
- `HAWKING_DEVICE_ROUTER_SELECTION_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_DEVICE_ROUTER
- `HAWKING_DSA_INDEX_OWNERSHIP_SPLIT_20260727.json` — OWNERSHIP_PREREQUISITE_NOT_PROMOTED
- `HAWKING_EXPERT_WAVE_CONCURRENT_RUNTIME_20260727.json` — SYNTHETIC_REPLICATED_AND_BOUNDED_COMPLETE_TOKEN_QUALIFIED_DEFAULT_OFF
- `HAWKING_EXPERT_WAVE_PERSISTENT_SCRATCH_CANDIDATE_20260727.json` — hawking.gravity.expert_wave_persistent_scratch_candidate.v1
- `HAWKING_FINAL_ASCENT_PROCESS_QUIESCENCE_RECEIPT.json` — hawking.final_ascent.process_quiescence_receipt.v1
- `HAWKING_G2_DOMINANT_COST.json` — hawking.gravity.g2_dominant_decode_cost.v1
- `HAWKING_GPU_CACHE_REVIEW.json` — VERIFIED BY CLAUDE. Tests re-run independently. Not merged; end-to-end residency still unmeasured.
- `HAWKING_GRAVITY_CROSS_MODEL_TRANSFER.md` — Cross-model transfer: the next-parent protocol is now amplification-first
- `HAWKING_GRAVITY_RUNTIME_GAPS.json` — hawking.gravity.base_runtime_gaps.v1
- `HAWKING_GRAVITY_SERVE_FIRST_GENERATION.json` — hawking.serve.gravity.first_generation.v1
- `HAWKING_GRAVITY_SERVE_PROMPT_PATH_AUDIT_20260727.json` — PROMPT_DROP_HYPOTHESIS_EXCLUDED
- `HAWKING_HIDE_HANDOFF_CONTRACT.json` — FROZEN_AT_THE_BOUNDARY, implementation deferred until a parent admits a servable compact representation
- `HAWKING_HIDE_RUNTIME_CONTRACT.json` — JSON receipt
- `HAWKING_ICB_COMPLETE_TOKEN_PERFORMANCE_GATE_20260727.json` — ICB_COMPLETE_TOKEN_DEFAULT_ON_REJECTED
- `HAWKING_LADDER_V3.md` — Hawking Ladder V3
- `HAWKING_LEDGER_BF16.json` — JSON receipt
- `HAWKING_LM_HEAD_ROOT_CAUSE.json` — hawking.gravity.lm_head_root_cause.v1
- `HAWKING_LONG_CONTEXT_SCRATCH_RECEIPT_20260727.json` — hawking.gravity.long_context_scratch.v1
- `HAWKING_MEMO_VERIFIED_RESULT.json` — JSON receipt
- `HAWKING_MODEL_FEEL_PARITY_RESULTS.json` — hawking.model_feel_parity.v1
- `HAWKING_MODEL_VAULT.json` — CONTRACT_SEALED_NOT_EXECUTED
- `HAWKING_MOTHERLOAD_REPORT.md` — Hawking Motherload Completion — Session Report
- `HAWKING_NEXT_PARENT_ADMISSION.json` — hawking.next_parent_admission.v1
- `HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md` — Hawking null-corrected metric contract
- `HAWKING_ORCHESTRATION_POLICY.md` — Orchestration policy — standing delegation zones
- `HAWKING_PARALLEL_GATE_ASSESSMENT.json` — substrate sealed
- `HAWKING_PHYSICAL_ACCOUNTING_PHASE1_20260727.json` — hawking.gravity.physical_accounting.phase1.v1
- `HAWKING_POSITIONED_ROPE_REPLAY_ABI_20260727.json` — n_heads
- `HAWKING_POST_ATTENTION_FUSION_REJECTED_20260727.json` — REJECTED_NUMERIC_PARITY_V2_1_NOT_PROMOTED
- `HAWKING_PQ_K_TRANSPOSE_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_PQ_V_ROWS_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_PROFILER_RECOVERY_RECEIPT.json` — hawking.gravity.profiler_recovery.v1
- `HAWKING_PROVIDER_CAPABILITY_MATRIX.json` — hawking.provider_capability_matrix.v1
- `HAWKING_RECLAMATION_SURVEY.md` — I'll survey the Python/Rust split read-only and write `RECLAMATION_SURVEY.md` + `.json`. Starting with layout and the gr
- `HAWKING_RECLAMATION_VERDICT.json` — REVISED DOWN -- optimistic
- `HAWKING_REPLAYABLE_COMPUTE_GRAPH_ICB_20260727.json` — REPLAYABLE_COMPUTE_GRAPH_SUBSTRATE_DEFAULT_OFF
- `HAWKING_REPRESENTATION_READINESS_MATRIX.json` — native functional organ replacement
- `HAWKING_RESIDENT_ATTENTION_LAYOUT_VARIANT_20260727.json` — DEFAULT_INERT_LAYOUT_PREREQUISITE_NOT_PROMOTED
- `HAWKING_RESIDENT_KV_STATE_STATIC_20260727.json` — hawking.gravity.resident_kv_state.static_projection.v1
- `HAWKING_RESIDENT_TPS.json` — JSON receipt
- `HAWKING_RESOURCE_MODE.json` — hawking.resource_mode.v1
- `HAWKING_RESUME_NEXT_SESSION.sh` — #!/usr/bin/env bash
- `HAWKING_ROUTE_SEGMENT_PRIMITIVES_20260726.json` — hawking.route_segment_primitives.v1
- `HAWKING_ROUTE_SEGMENT_REORDER_20260726.json` — hawking.route_segment_reorder.v1
- `HAWKING_RUNTIME_ASCENSION_AUDIT.json` — PROVEN
- `HAWKING_SHARED_TREE_PROTOCOL.json` — hawking.shared_tree_protocol.v1
- `HAWKING_STORAGE_CLEANUP_RECEIPT.json` — hawking.storage_cleanup_receipt.v1
- `HAWKING_TG_COST_LEDGER.json` — JSON receipt
- `HAWKING_TG_LEDGER_FINDINGS.json` — hawking.gravity.tg_ledger_findings.v1
- `HAWKING_TG_LEDGER_POSTMEMO.json` — JSON receipt
- `HAWKING_TG_LEDGER_POSTMEMO_FINDINGS.json` — hawking.gravity.tg_ledger_postmemo.v1
- `HAWKING_WARM_NOHASH_RESULT.json` — JSON receipt

## T1 / KIMI — 27 files

- `KIMI_K26_CAUSAL_ATLAS.json` — JSON receipt
- `KIMI_K26_DEVICE_CLEANSE_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_DOCTOR_BYTE_AUCTION.json` — JSON receipt
- `KIMI_K26_FINAL_GC_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_FINAL_STORAGE_REPORT.md` — Kimi K2.6 Final Storage Report
- `KIMI_K26_FIRST_CHECKPOINT.json` — PASS
- `KIMI_K26_HANDOFF_PRECHECK.json` — hawking.kimi_k26.handoff_precheck.v1
- `KIMI_K26_HANDOFF_PRECHECK.md` — Kimi K2.6 handoff precheck
- `KIMI_K26_LOCAL_SOURCE_DETECTION.json` — GREEN
- `KIMI_K26_NEXT_EXPERIMENT.json` — ADVANCING
- `KIMI_K26_P1_F1_FUNCTIONAL_CODEBOOKS.json` — JSON receipt
- `KIMI_K26_P1_F1_GRAMMAR_ISLANDS_BRACKET.json` — JSON receipt
- `KIMI_K26_P1_F2_SPARSE_PROPAGATION.json` — JSON receipt
- `KIMI_K26_PARALLEL_EXECUTION_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_PARALLEL_EXECUTION_PLAN.json` — contextual_native_seam
- `KIMI_K26_PARALLEL_EXECUTION_PLAN_PE02.json` — nonlinear_f0_f1_tournament
- `KIMI_K26_PARALLEL_EXECUTION_PLAN_PE03.json` — m5_exact_rate_ladder
- `KIMI_K26_RESIDENT_STORAGE_GATE.json` — hawking.kimi_k26.resident_storage_gate.v1
- `KIMI_K26_ROLLBACK.json` — SEALED
- `KIMI_K26_SCIENCE_PUBLICATION_MANIFEST.json` — hawking.kimi_k26.science_publication_manifest.v1
- `KIMI_K26_SOURCE_ADMISSION.json` — Modified MIT License
- `KIMI_K26_SOURCE_FORMAT_LEDGER.json` — PASS
- `KIMI_K26_STATE.json` — JSON receipt
- `KIMI_K26_TEXT_CORE_CLAIM.json` — PASS
- `KIMI_K26_TOURNAMENT.json` — SEALED
- `KIMI_PHONE_STATUS.json` — RUNNING
- `KIMI_PHONE_STATUS.md` — Kimi K2.6 Doctor Prime

## T1 / ROOT_MD — 22 files

- `120B_GATE_F_CLOSEOUT_PRECHECK.md` — 120B GATE-F CLOSEOUT PRECHECK
- `ADAPTIVE_TRANSFER_LADDER_PRECHECK.md` — ADAPTIVE TRANSFER LADDER PRECHECK
- `BASE_RUNTIME_MAXIMIZED_GATE.md` — BASE_RUNTIME_MAXIMIZED — hard promotion gate
- `CONDENSE_AUDIT.md` — CONDENSE_AUDIT.md · /goal condense · branch condense/run-20260703 · 2026-07-03
- `CONDENSE_DOCS_REVIEW.md` — CONDENSE_DOCS_REVIEW.md · staged doc deletions for the grader · condense/run-20260703
- `CONDENSE_LEDGER.md` — CONDENSE_LEDGER.md · /goal condense run · branch condense/run-20260703
- `M1ULTRA_RUN_REPORT.md` — M1 Ultra Run Report: the hawking maximization run
- `MODEL_LADDER_IGNITION_PRECHECK.md` — MODEL LADDER IGNITION PRECHECK
- `MOP_RAMANUJAN_ASSESSMENT_SURVEY.md` — MOP → Ramanujan Reuse Assessment
- `NUCLEAR_PASTA_LEDGER.md` — NUCLEAR PASTA execution ledger
- `NUMERIC_PARITY_V2.md` — Numeric Parity Contract V2
- `PROVIDER_FOUNDRY_V2_PRECHECK.md` — PROVIDER FOUNDRY V2 PRECHECK
- `RAMANUJAN_COGNITION_AMENDMENT.md` — Continuum amendment — the discovery loop
- `RAMANUJAN_COGNITION_CLAUDE.md` — Ramanujan's cognitive architecture — independent pass (Claude)
- `RAMANUJAN_COGNITION_GROK.md` — Ramanujan cognition — independent frontier ideation
- `RAMANUJAN_COGNITION_PROPOSAL.md` — Ramanujan cognitive architecture — consolidated proposal
- `SPEED_RESEARCH_WALL_CLOCK.md` — Speed research: where the wall clock goes, and what removes it
- `SUBBIT_CAPABILITY_DENSITY_PRECHECK.md` — Sub-bit capability-density reset — precheck
- `SUBBIT_CLOSURE_LEDGER.md` — Sub-bit closure ledger — Qwen3-235B-A22B-Instruct-2507
- `TEMPORAL_GRAVITY.md` — Temporal Gravity — minimize causal execution
- `VULTURE_HANDOFF_PRECHECK.md` — VULTURE HANDOFF PRECHECK
- `WATCHLIST.md` — WATCHLIST.md — lapping-condition spec

## T1 / GLM52 — 19 files

- `GLM52_A4_REAL_EXPERT_BANK.json` — hawking.representation.a4_real_bank.v1
- `GLM52_AA_PACK_ALLOCATOR_FIX.json` — hawking.glm52.aa_pack_allocator_fix.v1
- `GLM52_AA_PACK_DRY_RUN.json` — lm_head.weight
- `GLM52_ACTIVATION_AWARE_METAL_RUNTIME_20260727.json` — hawking.glm52.activation_aware_metal_runtime.v1
- `GLM52_ACTIVATION_AWARE_RUNTIME_ABI_AUDIT_20260727.json` — hawking.glm52.activation_aware_runtime_abi_audit.v1
- `GLM52_ASSEMBLY_COVERAGE.json` — hawking.glm52.assembly_coverage.v1
- `GLM52_ASSEMBLY_RESULT.json` — JSON receipt
- `GLM52_BASIS_TRANSFER.json` — hawking.representation.basis_transfer.v1
- `GLM52_CLAIM_A_STATUS.json` — The iso-memory frontier and the RandomPolicy control
- `GLM52_EXPERT_SWEEP.json` — hawking.representation.expert_sweep.v1
- `GLM52_GENERATION_A_DELETION_RECEIPT.json` — hawking.glm52.generation_a_deletion_receipt.v1
- `GLM52_GPU_BASE_TPS.json` — hawking.gravity.glm_base_tps.v1
- `GLM52_PASS3_BYTE_PRECLEARANCE.json` — hawking.prometheus.pass3_byte_preclearance.v1
- `GLM52_PASS3_OPTIMIZATION_AUDIT.json` — hawking.prometheus.pass3_optimization_audit.v1
- `GLM52_POSTPACK_RUNTIME_GATES_20260727.json` — PREPARED_NOT_EXECUTED
- `GLM52_RATE_SWEEP_ALL_ORGANS.json` — hawking.representation.rate_sweep.v1
- `GLM52_SELECTED_GENERAL_ARTIFACT.json` — hawking.glm52.selected_artifact.v1
- `GLM52_TABULA_RASA_LIVE_AUDIT.json` — JSON receipt
- `GLM52_TABULA_RASA_LIVE_AUDIT.md` — GLM-5.2 tabula rasa live audit

## T1 / OTHER — 14 files

- `120B_GATE_F_CLOSEOUT_PRECHECK.json` — hawking.gpt_oss_120b.gate_f_closeout_precheck.v1
- `ADAPTIVE_TRANSFER_LADDER_PRECHECK.json` — hawking.adaptive_transfer_ladder.precheck.v1
- `FIT_KERNEL_V3_REVIEW.json` — VERIFIED BY CLAUDE. Benchmarked and tested independently. Not merged; opt-in only.
- `MODEL_LADDER_IGNITION_PRECHECK.json` — hawking.model_ladder.ignition_precheck.v1
- `MODEL_RELEASE_INTERIM_PARENT.json` — GREEN
- `MOP_RAMANUJAN_TRANSFER_ASSESSMENT.json` — hawking.ramanujan.mop_transfer_assessment.v1
- `NUCLEAR_PASTA_STATE.json` — JSON receipt
- `NUMERIC_PARITY_V2_1_HARNESS.json` — absolute_error_near_zero
- `PROMETHEUS_ARCHITECTURE.json` — NOT_SEALED
- `PROVIDER_FOUNDRY_V2_PRECHECK.json` — hawking.provider_foundry_v2.precheck.v1
- `S1_FAILURE_DECOMPOSITION.json` — JSON receipt
- `SUBBIT_CAPABILITY_DENSITY_PRECHECK.json` — LEVER_ALIVE
- `SUBBIT_CLOSURE_STATE.json` — admitted_next
- `VULTURE_HANDOFF_PRECHECK.json` — RUNNING (do not touch; do not modify its scientific code while alive)

## T1 / OTHER_MODEL — 6 files

- `QWEN235B_DEGRADATION_ATLAS.json` — CLOSURE. This is a PRIOR LIBRARY for the next parent, not a set of universal truths. Every entry names the parent it was
- `QWEN235B_DOCTOR_HANDOFF.json` — hawking.doctor.handoff.v1
- `QWEN235B_TREATMENT_RESPONSE_ATLAS.json` — CLOSURE. Priors for the next parent, not universal truths.
- `QWEN35_397B_ORGAN_INVENTORY.json` — hawking.qwen35_moe.inventory.v1
- `QWEN35_397B_WINDOW_PLAN.json` — hawking.qwen35_moe.window_plan.v1
- `QWEN_FULL_COURSE_PRECHECK.json` — SEALED

## T1 / DOCTOR — 4 files

- `DOCTOR_DIAGNOSIS_ONTOLOGY.json` — JSON receipt
- `DOCTOR_GENERATION_3_REGISTRY.json` — JSON receipt
- `DOCTOR_PRIME_PRECHECK.json` — FIXED and GUARDED. corpus_integrity.py splits by document/context-hash and refuses a shared embedding row via assert_lay
- `DOCTOR_TREATMENT_LIBRARY.json` — hawking.doctor.treatment_library.v1

## T1 / ROOT_YAML — 1 files

- `pnpm-lock.yaml` — lockfileVersion: '9.0'

## T1 / DEEPSEEK — 1 files

- `DEEPSEEK_V4_FLASH_MOE_PROBE_DECISION.json` — the embedding-seeded shortcut is INVALID for DeepSeek's functional existence test, proven by the shared-expert control; 

## T1 / GRAVITY — 1 files

- `GRAVITY_DEGENERATE_ATTRIBUTION.json` — hawking.gravity.degenerate_attribution.v1

## T1 / ROOT_SH — 1 files

- `run_7b_ladder.sh` — #!/usr/bin/env bash

## T2 — superseded campaign narration — 127 files

Pruned **127 tracked markdown files** (~2,073,871 bytes, ~30,329 lines).
Includes root `HAWKING_*_STATUS.md` (tools rewrite on next run), one-shot `docs/plans/**` campaign plans
without executable readers, `docs/hide-impl/**` campaign writeups without readers, other non-living `docs/*.md`
narrative without readers, and superseded `control/final_ascent/contracts/*-revision-N.md` (kept highest N and all
contracts with no revision suffix). Living reference kept: README/ARCHITECTURE/MODELS, docs/{dead_levers,serve,
BENCHMARKS,env_flags,kernels}, docs/gravity/**, docs/hide-bible/**, docs/ARCHIVE_INDEX*.md.
Restore: `git checkout pre-floor-prune-20260728 -- <path>`.

## T2 / control/final_ascent/contracts (superseded revisions) — 27 files

- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-1.md` — Revision 1: close rank, leakage, byte, and lifecycle gaps
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-10-adversarial-ledger-identity.md` — GLM-5 rare-route pilot — Revision 10 adversarial ledger/identity closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-11-science-witness-closure.md` — GLM-5 rare-route pilot — Revision 11 science/witness closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-12-immutable-closure.md` — GLM-5.2 rare-route basis pilot — Revision 12 immutable closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-2.md` — Revision 2: close deployed-representation, provenance, and recovery gaps
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-3.md` — Revision 3: eliminate false-positive science and lifecycle gates
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-4-lifecycle.md` — Revision 4B: implement the real reader and recoverable lifecycle
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-4-science.md` — Revision 4A: make science and provenance validation semantic
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-5-science.md` — Revision 5: close remaining science and provenance false positives
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-6-lifecycle.md` — Revision 6: close remaining lifecycle, recovery, and replay false positives
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-7-science-cleanup.md` — Revision 7: remove remaining science/provenance fail-open paths
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-8-lifecycle-science-closure.md` — Revision 8: close lifecycle recovery and streamed-reader evidence
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-9-atomic-epoch-replay.md` — GLM-5 2-shard rare-route pilot — Revision 9 atomic epoch/replay closure
- `control/final_ascent/contracts/tg-cheap-hotpath-residual-router-scalars-revision-1.md` — Temporal Gravity cheap hot-path closure — Revision 1 physics/no-regression
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-1.md` — TG device-resident ordinary three-batch MLP — Revision 1
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-2.md` — TG device-resident three-batch MLP — Revision 2 live-Metal correction
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-3.md` — TG device-resident three-batch MLP — Revision 3 device-only correction
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-1.md` — Temporal Gravity GLM live-token path into HIDE — Revision 1
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-2.md` — Temporal Gravity GLM live-token path into HIDE — Revision 2
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-3.md` — Temporal Gravity → HIDE live-token path — Revision 3
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-1.md` — Near-tie fallback Revision 1: real FP64 authority and durable evidence
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-2.md` — Temporal Gravity Numeric Parity near-tie fallback — Revision 2
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-3.md` — Temporal Gravity Numeric Parity near-tie fallback — Revision 3
- `control/final_ascent/contracts/tg-router-bias-residency-bind-once-revision-1.md` — Temporal Gravity router-bias residency — Revision 1
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-1.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 1
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-2.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 2
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-3.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 3

## T2 / docs (other narrative) — 4 files

- `docs/architecture.md` — Hawking architecture map (2026-06-21)
- `docs/design/continuous_batching.md` — Continuous Request Batching for `hawking serve`
- `docs/mixtral.md` — Mixtral 8×7B support
- `docs/rwkv7.md` — On-device instruct post-train: RWKV7-g1-0.4B (M3 Pro, $0, overnight)

## T2 / docs/hide-impl — 16 files

- `docs/hide-impl/FIRST_RECEIPT_HARNESS.md` — First Mandatory Implementation Receipt - Harness Spec
- `docs/hide-impl/PHASE0_CRATE_MAP.md` — Phase 0: Crate Map (target vs current, reconciled)
- `docs/hide-impl/PHASE0_SOURCE_LEGAL_INTAKE.md` — Phase 0: Source and Legal Intake Ledger
- `docs/hide-impl/consolidation/HIDE_CODEX_DEPTH_MAP.md` — HIDE Codex Depth Map
- `docs/hide-impl/consolidation/HIDE_CONTROL_DENSITY_SCORECARD.md` — HIDE Control Density Scorecard
- `docs/hide-impl/consolidation/HIDE_DEAD_DUPLICATE_CONTROL_REPORT.md` — HIDE Dead / Duplicate / Misleading / Mock Control Report
- `docs/hide-impl/consolidation/HIDE_FINAL_CONSOLIDATION_REPORT.md` — HIDE Final Consolidation Report
- `docs/hide-impl/consolidation/HIDE_GROK_BUILD_DEPTH_MAP.md` — HIDE x grok-build depth map
- `docs/hide-impl/consolidation/HIDE_OPENCODE_DEPTH_MAP.md` — HIDE opencode Depth Map
- `docs/hide-impl/consolidation/HIDE_PRODUCTIVITY_DENSITY_BASELINE.md` — HIDE Productivity Density Baseline
- `docs/hide-impl/consolidation/HIDE_SURFACE_WITHOUT_BACKEND_REPORT.md` — HIDE Surface Without Backend Report
- `docs/hide-impl/consolidation/HIDE_UI_CONTROL_CENSUS.md` — HIDE UI Control Census
- `docs/hide-impl/consolidation/HIDE_WORKFLOW_BEFORE_AFTER.md` — HIDE Workflow Before and After
- `docs/hide-impl/preview/HIDE_FOURTH_ADVERSARIAL_REPORT.md` — HIDE fourth adversarial report
- `docs/hide-impl/preview/HIDE_PREVIEW_CLOSEOUT_REPORT.md` — HIDE Preview Closeout Report
- `docs/hide-impl/preview/HIDE_PREVIEW_GALLERY.md` — HIDE Preview Gallery

## T2 / docs/plans — 75 files

- `docs/plans/APPENDIX.md` — The Appendix
- `docs/plans/APPENDIX_HANDOFF.md` — Appendix handoff and incorporation contract
- `docs/plans/CONDENSATION_DOCTOR_V2.md` — Condensation Doctor v2 — Program Synthesis for Capability Restoration
- `docs/plans/DOCTOR_V5.md` — Condensation Doctor v5 — Canonical Capability-First Specification
- `docs/plans/DOCTOR_V5_3BPW_RESOURCE_STOP_RECOVERY.md` — Doctor V5 14B/3bpw resource-stop recovery staging
- `docs/plans/DOCTOR_V5_AGGRESSIVE_ADMISSION_V2.md` — Doctor V5 aggressive admission v2 (unbound scaffold)
- `docs/plans/DOCTOR_V5_BLOCKED_CELL_RECOVERY.md` — Doctor V5 14B/4bpw blocked-cell recovery
- `docs/plans/DOCTOR_V5_ELASTIC_HOST_SPRINT.md` — Doctor V5 elastic phases and host sprint isolation
- `docs/plans/DOCTOR_V5_MOUNTAIN_LADDER.md` — Doctor V5 post-120B mountain ladder
- `docs/plans/DOCTOR_V5_PARALLEL_ACCELERATION_HANDOFF.md` — Doctor V5 unbound GPT-OSS and higher-tier acceleration handoff
- `docs/plans/DOCTOR_V5_PHYSICAL_ADAPTER_AUTHORITY.md` — Doctor V5 physical adapter authority
- `docs/plans/DOCTOR_V5_PHYSICAL_POST120_NEXT_GOAL_PROMPT.md` — Restart-proof next-goal prompt
- `docs/plans/DOCTOR_V5_POST120_ACCELERATION.md` — Doctor V5 120B and post-120B acceleration handoff
- `docs/plans/DOCTOR_V5_POST_120B.md` — Doctor V5 post-120B handoff
- `docs/plans/DOCTOR_V5_REMAINING_SCRATCH_LEDGER.md` — Doctor V5 phase-aware remaining-scratch ledger
- `docs/plans/DOCTOR_V5_RESEARCH_PASSES.md` — Condensation Doctor v5 — Three Expansion Passes and Adversarial Proof Plan
- `docs/plans/DOCTOR_V5_SINGLE_DEVICE_SPRINT.md` — Doctor V5 single-device sprint handoff
- `docs/plans/DOCTOR_V5_TELEGRAM_NOTIFICATIONS.md` — Doctor V5 Telegram rung notifications
- `docs/plans/FRONTIER_ECOSYSTEM_SCAFFOLD.md` — Condenser Ecosystem Frontier: scaffold status
- `docs/plans/HAWKING_GLM52_GENERATION_B_STREAMING_FRONTIER_ACTION_PLAN.md` — /goal
- `docs/plans/HAWKING_GRAVITY_FORGE.md` — Hawking Gravity Forge
- `docs/plans/HAWKING_GRAVITY_RUNBOOK.md` — Hawking Gravity Runbook
- `docs/plans/HAWKING_LAPTOP_WAVE_2026_07_08.md` — Hawking laptop wave - 2026-07-08
- `docs/plans/HAWKING_SEED_ARCHITECTURE.md` — Hawking Seed — architecture specification + 3-candidate design (Phase 1 oracle)
- `docs/plans/HIDE_CONDENSER_GOAL_PROMPT.md` — HIDE condenser goal prompt
- `docs/plans/HIDE_REFINEMENT_RUN_REPORT.md` — HIDE refinement run report
- `docs/plans/M1ULTRA_GOAL_PROMPT.md` — M1 ULTRA GOAL PROMPT: the iterative maximization loop for the delivered box
- `docs/plans/M1ULTRA_POTENTIAL_AUDIT.md` — M1 ULTRA POTENTIAL AUDIT: every evaluated category graded on the delivered box's ceiling
- `docs/plans/STUDIO_DEEP_AUDIT_2026_07_08.md` — Studio deep audit - 2026-07-08
- `docs/plans/STUDIO_GO.md` — STUDIO GO — the one-command entry point for the Hawking frontier program
- `docs/plans/STUDIO_MODEL_LADDER.md` — Studio Model Ladder — Condensation, Not Quantization
- `docs/plans/SUCCESSOR_RUNBOOK.md` — Successor condenser: operational + rollback runbook
- `docs/plans/TABULA_RASA_SHIPPING_PLAN.md` — Hawking Tabula Rasa Shipping Plan
- `docs/plans/TRAINING_LADDER_V5.md` — Hawking Training Ladder v5 — Capability-First Condensation
- `docs/plans/agentic_tool_system_audit_2026_07_11.md` — Agentic Tool System: scaffold audit + rating (2026-07-11)
- `docs/plans/apple_fit_frontier_2026_06_22.md` — Apple Fit Frontier (2026-06-22)
- `docs/plans/bible_active.md` — > **THROUGHPUT BIBLE — ACTIVE (lean working doc).** Split 2026-05-31 from `throughput_bible_2026_05_30.md`. This is the 
- `docs/plans/bible_archive.md` — > **ARCHIVE / REFERENCE — the long companion to [bible_active.md](bible_active.md)** (split 2026-05-31 from this file's 
- `docs/plans/computational_efficiency_paradigms_2026_07_11.md` — Beyond FLOPS: a cross-layer research agenda for computational efficiency
- `docs/plans/condense_autopilot_2026_06_27.md` — Condense Autopilot - 2026-06-27
- `docs/plans/condense_naming_migration_2026_06_22.md` — Condense Naming Migration (2026-06-22)
- `docs/plans/doctor_capability_and_speed_roadmap.md` — Doctor → ~1:1 quality + Speed, BEFORE the 32B (2026-06-23)
- `docs/plans/doctor_maximization_plan.md` — Doctor Maximization Plan — Recovery to Outrun Compression's Decay
- `docs/plans/g1a_v2_expansion_results_2026_06_20.md` — G1a V2 Expansion Chain Results
- `docs/plans/h_autoverify_protocol.md` — Event Horizon Auto-Handoff Protocol (2026-06-21)
- `docs/plans/hawking_capability_frontier_2026_06_28.md` — Hawking IDE — Capability Frontier & Build Roadmap
- `docs/plans/hawking_capability_frontier_2026_06_28_HANDOFF_PROMPT.md` — Build-session handoff prompt (paste into the integrated/coding session)
- `docs/plans/hawking_event_horizon_phase0_blueprint.md` — I have everything grounded against the real source. The skeletons in the designs match the actual signatures (`verify_dr
- `docs/plans/hawking_event_horizon_proposal_engine.md` — Hawking Event Horizon — Unified Speculative Proposal Engine
- `docs/plans/hawking_event_horizon_status.md` — Hawking Event Horizon — As-Built Status (2026-06-20)
- `docs/plans/hawking_gravity_maximal_fidelity_ladder.md` — Hawking Gravity: maximal-fidelity ladder (as close to raw models as the box allows)
- `docs/plans/hawking_handoff_2026_06_28.md` — Hawking — Maximal Handoff & Codebase Understanding (2026-06-28)
- `docs/plans/hawking_ide_claude_research_handoff_2026_07_19.md` — Claude handoff: independent Hawking IDE frontier pass
- `docs/plans/hawking_ide_frontier_2026_07_19.md` — Hawking IDE frontier dossier
- `docs/plans/hawking_shippability_masterplan_2026_06_22.md` — Hawking — Shippability Master Plan (2026-06-22)
- `docs/plans/hide_command_system_maximalist_2026_06_29.md` — HIDE — The Maximalist Command System (trigger grammar + local-unlimited depth)
- `docs/plans/hide_deep_audit_2026_07_16.md` — HIDE deep audit and facet ladder
- `docs/plans/hide_executor_plan_v2_enriched_2026_06_29.md` — HIDE Executor — Enriched Master Plan (v2)
- `docs/plans/hide_handoff_2026_06_28.md` — HIDE handoff prompt (paste into the dedicated HIDE chat) — 2026-06-28
- `docs/plans/hide_master_build_plan_2026_06_29.md` — HIDE / Hawking — Master Build Plan (2026-06-29)
- `docs/plans/hide_refinement_roadmap_2026_07_05.md` — HIDE refinement roadmap
- `docs/plans/hide_research_menu_2026_06_29.md` — HIDE Agentic-Frontier Research Menu (2026-06-29)
- `docs/plans/hide_ship_readiness.md` — HIDE — Ship Readiness (living status, 2026-06-29)
- `docs/plans/hide_ship_status_2026_06_29.md` — HIDE — Ship Status (2026-06-29)
- `docs/plans/hide_sota_frontier_and_regrade_2026_07_16.md` — HIDE SOTA frontier read + regraded ladder (condenser pass)
- `docs/plans/hide_ux_audit_and_research_2026_06_29.md` — HIDE — Deep Audit + UX Research (2026-06-29)
- `docs/plans/parameter_sweep_pipeline.md` — Hawking Condense — Parameter-Sweep Testing Pipeline (2026-06-23)
- `docs/plans/q6k_predec_design.md` — Q6_K predec ffn_down — implementation design (campaign R-design, 2026-06-21)
- `docs/plans/quintessential_engine_2026_06_29.md` — Hawking — the quintessential local inference engine (unified plan, 2026-06-29)
- `docs/plans/ratios_roadmap_2026_06_21.md` — Ratios Roadmap — speed · compression · density (validated 2026-06-21)
- `docs/plans/spec_decode_reentry_appendix_2026_07_14.md` — Speculative-decode re-entry appendix — 2026-07-14
- `docs/plans/spec_decode_studio_readiness_2026_07_12.md` — Speculative decoding on the 96 GB Studio: evidence and readiness
- `docs/plans/storage_stripdown_resident_first_2026_07_20.md` — Storage stripdown and the resident-first resequencing
- `docs/plans/throughput_pivot_campaign.md` — Throughput-Pivot Campaign — live autonomous run (started 2026-06-21)
- `docs/plans/tq_compute_for_memory_appendix_2026_07_14.md` — TQ compute-for-memory appendix — 2026-07-14

## T2 / root HAWKING_*_STATUS.md — 5 files

- `HAWKING_CONTINUUM_STATUS.md` — HAWKING CONTINUUM STATUS
- `HAWKING_FINAL_ASCENT_STATUS.md` — HAWKING FINAL ASCENT STATUS
- `HAWKING_LIGHT_ONLY_STATUS.md` — HAWKING LIGHT-ONLY STATUS
- `HAWKING_MOTHERLOAD_STATUS.md` — Hawking Motherload Completion Status
- `HAWKING_PARALLEL_STATUS.md` — HAWKING PARALLEL CONTINUATION STATUS

## T3 — closed-campaign condense modules — 62 files

Pruned **62 files** (~1,174,292 bytes, ~24,624 lines) from sealed campaigns
(kimi/qwen/gravity_frontier/second_light/deepseek_v4/frontier prefixes) where no non-test live
module imports or shells out to them. Reader proof: `rg` over tools/, crates/, ramanujan/, app/
for each module stem; hits inside other closed modules or `tools/condense/tests/test_*` counterparts
do not count as live readers. **Kept 18 modules** with live readers (e.g. `gptoss_block` via
mech/forge, `qwen_real_forward` via doctor_causal_harness, `deepseek_v4_adapter` via export.rs,
overnight_supervisor path constants, emergency_detached_campaign hashes).
Restore: `git checkout pre-floor-prune-20260728 -- <path>`.

### T3 kept (live readers) — 18 modules

- `tools/condense/deepseek_v4_adapter.py` — e.g. `crates/hawking-adapters/src/export.rs:646:                    "tools/condense/deepseek_v4_adapter.py`
- `tools/condense/gptoss_block.py` — e.g. `tools/condense/forge_f2_fixture.py:83:        import gptoss_block as gb`
- `tools/condense/gptoss_gravity_run.py` — e.g. `tools/condense/gravity_forge.py:4:The naive Gravity 120B run (gptoss_gravity_run.py) proved a BASELI`
- `tools/condense/gptoss_moe_runtime.py` — e.g. `tools/condense/gravity_forge.py:539:    import gptoss_moe_runtime as rt`
- `tools/condense/gptoss_real_forward.py` — e.g. `tools/condense/vulture_harvest.py:925:            "coherence-validated forward (gptoss_real_forward.`
- `tools/condense/gptoss_subbit_packer.py` — e.g. `tools/foundry/gravity_potency.py:301:            "engine": "tools/condense/gravity_forge.py + gptoss`
- `tools/condense/gravity_frontier_correction_wave.py` — e.g. `tools/condense/overnight_supervisor.py:270:        subprocess.run([PY, str(ROOT / "tools/condense/gr`
- `tools/condense/gravity_frontier_g4_controller.py` — e.g. `tools/condense/seal_120b_conclusion.py:616:            f"{PY} tools/condense/gravity_frontier_g4_con`
- `tools/condense/kimi_k26_phase2_recovery.py` — e.g. `tools/condense/emergency_detached_campaign.py:55:    "tools/condense/kimi_k26_phase2_recovery.py": "`
- `tools/condense/kimi_k26_phase2_release.py` — e.g. `tools/condense/emergency_detached_campaign.py:56:    "tools/condense/kimi_k26_phase2_release.py": "9`
- `tools/condense/kimi_k26_release_cycle.py` — e.g. `tools/condense/emergency_detached_campaign.py:57:    "tools/condense/kimi_k26_release_cycle.py": "10`
- `tools/condense/qwen_adaptive_k.py` — e.g. `tools/condense/doctor_byte_auction.py:62:import qwen_adaptive_k as AK  # noqa: E402`
- `tools/condense/qwen_correction_wave.py` — e.g. `tools/condense/overnight_supervisor.py:557:QWEN_CTRL = ROOT / "tools/condense/qwen_correction_wave.p`
- `tools/condense/qwen_download_worker.py` — e.g. `tools/condense/overnight_supervisor.py:554:QWEN_DL_WORKER = ROOT / "tools/condense/qwen_download_wor`
- `tools/condense/qwen_function_aware_codec.py` — e.g. `tools/condense/doctor_byte_auction.py:63:import qwen_function_aware_codec as FAC  # noqa: E402`
- `tools/condense/qwen_real_forward.py` — e.g. `tools/condense/doctor_causal_harness.py:71:import qwen_real_forward as Q  # noqa: E402`
- `tools/condense/qwen_structural_plan.py` — e.g. `tools/condense/doctor_byte_auction.py:64:import qwen_structural_plan as SP  # noqa: E402`
- `tools/condense/qwen_subhalfbit_search.py` — e.g. `tools/condense/doctor_byte_auction.py:65:import qwen_subhalfbit_search as SHB  # noqa: E402`

## T3 / deepseek_v4_* — 8 files

- `tools/condense/deepseek_v4_amplification.py`
- `tools/condense/deepseek_v4_cascade.py`
- `tools/condense/deepseek_v4_contextual_probe.py`
- `tools/condense/deepseek_v4_moe.py`
- `tools/condense/deepseek_v4_primitive_parity.py`
- `tools/condense/deepseek_v4_reference.py`
- `tools/condense/deepseek_v4_release.py`
- `tools/condense/deepseek_v4_source.py`

## T3 / frontier_* — 1 files

- `tools/condense/frontier_giant_scaffold.py`

## T3 / gravity_frontier_* — 12 files

- `tools/condense/gravity_frontier_controller.py`
- `tools/condense/gravity_frontier_g1.py`
- `tools/condense/gravity_frontier_g2_controller.py`
- `tools/condense/gravity_frontier_g2_ignite.py`
- `tools/condense/gravity_frontier_g2_program.py`
- `tools/condense/gravity_frontier_g2_status.py`
- `tools/condense/gravity_frontier_g3_controller.py`
- `tools/condense/gravity_frontier_g3_program.py`
- `tools/condense/gravity_frontier_g3_status.py`
- `tools/condense/gravity_frontier_ignite.py`
- `tools/condense/gravity_frontier_program.py`
- `tools/condense/gravity_frontier_status.py`

## T3 / kimi_* — 2 files

- `tools/condense/kimi_k26_download_supervisor.py`
- `tools/condense/kimi_k26_stale_download_cleanup.py`

## T3 / qwen_* — 18 files

- `tools/condense/qwen_bpw_budget.py`
- `tools/condense/qwen_calibration_corpus.py`
- `tools/condense/qwen_checkpoint_notifier.py`
- `tools/condense/qwen_codec_portfolio.py`
- `tools/condense/qwen_compressibility_train.py`
- `tools/condense/qwen_distill_gate.py`
- `tools/condense/qwen_doctor_gen2.py`
- `tools/condense/qwen_function_aware_probe.py`
- `tools/condense/qwen_generated_params.py`
- `tools/condense/qwen_gravity_campaign.py`
- `tools/condense/qwen_layer0_codec.py`
- `tools/condense/qwen_layerwise_qat.py`
- `tools/condense/qwen_q2_bootstrap.py`
- `tools/condense/qwen_q2_diagnose.py`
- `tools/condense/qwen_qat_disjoint.py`
- `tools/condense/qwen_router_distill.py`
- `tools/condense/qwen_routing_calibration.py`
- `tools/condense/qwen_shannon_bound.py`

## T3 / second_light_* — 13 files

- `tools/condense/second_light_controller.py`
- `tools/condense/second_light_controller_evidence.py`
- `tools/condense/second_light_first_light_seal.py`
- `tools/condense/second_light_gates.py`
- `tools/condense/second_light_ignite.py`
- `tools/condense/second_light_pack.py`
- `tools/condense/second_light_pq_evidence.py`
- `tools/condense/second_light_precheck.py`
- `tools/condense/second_light_program.py`
- `tools/condense/second_light_quality_contract.py`
- `tools/condense/second_light_readiness.py`
- `tools/condense/second_light_source_manifest.py`
- `tools/condense/second_light_status.py`

## T3 / tools/condense/tests — 8 files

- `tools/condense/tests/test_gravity_frontier.py`
- `tools/condense/tests/test_gravity_frontier_g2.py`
- `tools/condense/tests/test_gravity_frontier_g3.py`
- `tools/condense/tests/test_kimi_k26_download_supervisor.py`
- `tools/condense/tests/test_kimi_k26_stale_download_cleanup.py`
- `tools/condense/tests/test_qwen_bpw_budget.py`
- `tools/condense/tests/test_qwen_gravity_campaign.py`
- `tools/condense/tests/test_second_light_controller.py`


### T3 follow-up restores (gate repair)

Restored from `pre-floor-prune-20260728` after pytest collection failed:
- `tools/condense/kimi_k26_download_supervisor.py` (+ test) — imported by kept `kimi_k26_phase2_recovery`
- `tools/condense/qwen_bpw_budget.py` (+ test) — imported by kept `qwen_subhalfbit_search`

## T4 — adapter artifacts single location — 9 files

Root copies of the nine adapter codegen deliverables were byte-identical to
`crates/hawking-adapters/generated/` (verified with `shasum -a 256`). Readers
(`tools/adapters/verify_grades.py`, `tools/campaign/light_status.py`,
`tools/campaign/final_ascent_status.py`) now point at `generated/`. Codegen no
longer writes repo-root duplicates; drift test requires generated-only placement.

Restore root copies if needed: `git checkout pre-floor-prune-20260728 -- HAWKING_ADAPTER_*.json HAWKING_CANONICAL_EVENTS.json HAWKING_BRIDGE_SURFACE.json HAWKING_CLI_SURFACE.json HAWKING_SCHEMA_MIGRATIONS.json`

#
### T3 follow-up restores (gate repair)

Restored from `pre-floor-prune-20260728` after pytest collection failed:
- `tools/condense/kimi_k26_download_supervisor.py` (+ test) — imported by kept `kimi_k26_phase2_recovery`
- `tools/condense/qwen_bpw_budget.py` (+ test) — imported by kept `qwen_subhalfbit_search`

## T4 / root duplicates removed — 9 files

- `HAWKING_ADAPTER_ABI.json`
- `HAWKING_ADAPTER_REGISTRY.json`
- `HAWKING_ADAPTER_CAPABILITY_MATRIX.json`
- `HAWKING_ADAPTER_TEST_MATRIX.json`
- `HAWKING_ADAPTER_MIGRATION_MAP.json`
- `HAWKING_CANONICAL_EVENTS.json`
- `HAWKING_BRIDGE_SURFACE.json`
- `HAWKING_CLI_SURFACE.json`
- `HAWKING_SCHEMA_MIGRATIONS.json`

## T5 — stale studio_run entrypoint

`tools/condense/studio_run.py` was deleted in clean-slate collapse `0a970800` (legacy capsule /
hawking-lab pack). No in-tree successor implements start/drain/resume/--go-plan. `hawking studio`
is not a subcommand of the active `hawking` binary. `GO.md` rewritten to refuse the dead command
and name live `hawking` / `tools/campaign/*` entrypoints only. `BASELINES.md` marks `hawking studio`
and `studio_run.py` command lines as sealed/not active. No replacement command invented.


## vendor/strand-decode-kernel — archived 2026-07-28

53 files, 18,151 Rust LOC, 836 KB. STRAND's reference decode runtime and gate harness,
absorbed with the quant track and never a hawking build dependency: the root `Cargo.toml`
listed it under `exclude`, and `cargo metadata` reports zero reverse dependencies on the
package. Nothing links it.

It also stopped being a mirror. `crates/hawking-core` carries the product port
(`tq_gpu.rs`, `tq.rs`, `shaders/strand_bitslice.metal`), and the two shaders have diverged:

    crates/hawking-core/shaders/strand_bitslice.metal   1324 lines
    vendor/strand-decode-kernel/shaders/strand_bitslice.metal    549 lines

The port added `CompactBitsliceEntry` (40 B/block) and expand helpers that the vendor copy
never received. The port's stated contract is bit-identity to
`strand_quant::decode::decode_tensor_fixed`, not to this crate's GPU binary.

Gate run before removal, on the local M3 Ultra:

    cargo test -p hawking-core --features tq --test tq_trellis_parity
    5 passed; 1 ignored (trellis_k1_gpu_decode_parity, k=1 GPU path not yet validated)
    including bitslice_gpu_decode_matches_cpu_oracle_over_matrix (321s, real Metal)

    cargo test -p hawking-core --features tq --test qwen_tq_serve_parity
    0 passed; 1 ignored -- NOT RUN. The test needs models/Qwen2.5-3B-Instruct-Q4_K_M.gguf,
    which is not on this disk. The serve-trajectory half of the gate is unproven here.

`vendor/strand-quant` is untouched and stays: it is live for encode via `tools/tq_bake`,
`hawking-core`'s optional `tq` feature, and `hide-backend`'s optional `tq` feature.

Provenance comments in `crates/hawking-core/src/{tq.rs,tq_gpu.rs,kernels/mod.rs,metal/mod.rs}`
still cite `vendor/strand-decode-kernel/...` paths. They were left as-is: they record where
the port came from, and that history is what this index points at.

Restore: `git checkout pre-r3-vendor-drop-20260728 -- vendor/strand-decode-kernel`
and re-add the `exclude` entry in the root `Cargo.toml`.
## T2 — historical campaign reports archive (2026-07-28) — 358 files

Working-tree prune of historical `reports/condense/` campaign bodies with **no
executable-code reader** of the file basename under `tools/`, `crates/`,
`ramanujan/`, `adapters/`, `odyssey/`, or `app/` (`.py`/`.rs`/`.sh`/`.toml`).
**Nothing is lost:** full content lives in git history at the annotated tag
`pre-reports-archive-20260728`. Restore any file with:

```
git checkout pre-reports-archive-20260728 -- <path>
# or
git show pre-reports-archive-20260728:<path>
```

Pruned **358 tracked files** (~1,576,570 bytes, ~29,420 lines).
Reader map: exact basename scan of executable sources. **Any basename hit kept**
the file. Slices named by the lane (`second_light`, `general_frontier`,
`gravity_forge`) were partially archived; fully no-reader siblings
`gravity_frontier` and `deepseek_v4_flash` were archived entirely.

### Kept in tree (not archived)

- `reports/condense/glm52_generation_b` — **45 files, live**. Bound by
  `glm52_pilot.py`, pack-v2 headers (`GLM52_SOURCE_SHARD_HEADERS.json`),
  rate ladder, window plan, functional gauntlet/auction/roofline, generation-A
  seal, odyssey inventory fixtures, and related generation-B tools.
- Reader-hit files inside partially archived slices (36 files): second_light (4),
  general_frontier (15), gravity_forge (17). Basenames appear in executable tools.
- Untouched sibling slices with live readers: `breakthrough`, `kimi_k26`,
  `storage_stripdown`, `subbit_frontier`.
- Mega matrices left at repo root (T3): `GLM52_SHARD_DEPENDENCY_GRAPH.json`,
  `GLM52_ROUTE_POPULATION_CENSUS.json`, `PROMETHEUS_MATH_ALLOCATION_MANIFEST.json`.

## reports/condense/second_light — 199 files archived (~501,489 bytes, ~14,934 lines)

- `reports/condense/second_light/GPT_OSS_120B_FIRST_LIGHT_CALIBRATION.json` — GPT_OSS_120B_FIRST_LIGHT_CALIBRATION.json
- `reports/condense/second_light/GPT_OSS_120B_FIRST_LIGHT_DOSSIER.md` — GPT-OSS-120B FIRST-LIGHT CALIBRATION DOSSIER
- `reports/condense/second_light/GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json` — GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json
- `reports/condense/second_light/GPT_OSS_120B_PQ_READINESS.json` — GPT_OSS_120B_PQ_READINESS.json
- `reports/condense/second_light/GPT_OSS_120B_QUALITY_CONTRACT.json` — GPT_OSS_120B_QUALITY_CONTRACT.json
- `reports/condense/second_light/GPT_OSS_120B_SECOND_LIGHT_REPORT.md` — GPT-OSS-120B :: SECOND-LIGHT PQ BASELINE ARTIFACT
- `reports/condense/second_light/GPT_OSS_120B_SECOND_LIGHT_REPRODUCTION.json` — hawking.gpt_oss_120b.second_light_reproduction.v1
- `reports/condense/second_light/SECOND_LIGHT_FIRST_CANDIDATE.json` — hawking.second_light.first_candidate_artifact.v1
- `reports/condense/second_light/SECOND_LIGHT_IGNITION_RECEIPT.json` — SECOND_LIGHT_IGNITION_RECEIPT.json
- `reports/condense/second_light/SECOND_LIGHT_PRECHECK.json` — SECOND_LIGHT_PRECHECK.json
- `reports/condense/second_light/SECOND_LIGHT_PRECHECK.md` — SECOND LIGHT PRECHECK
- `reports/condense/second_light/SECOND_LIGHT_REPORT.md` — HAWKING SECOND LIGHT :: Required Report
- `reports/condense/second_light/checkpoints/r0000.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0001.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0002.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0003.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0004.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0005.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0006.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0007.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0008.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0009.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0010.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0011.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0012.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0013.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0014.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0015.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0016.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0017.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0018.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0019.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0020.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0021.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0022.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0023.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0024.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0025.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0026.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0027.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0028.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0029.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0030.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0031.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0032.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0033.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0034.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0035.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0036.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0037.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0038.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0039.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0040.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0041.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0042.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0043.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0044.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0045.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0046.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0047.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0048.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0049.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0050.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0051.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0052.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0053.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0054.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0055.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0056.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0057.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0058.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0059.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0060.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0061.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0062.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0063.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0064.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0065.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0066.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0067.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0068.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0069.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0070.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0071.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0072.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0073.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0074.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0075.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0076.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0077.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0078.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0079.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0080.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0081.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0082.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0083.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0084.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0085.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0086.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0087.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0088.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0089.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0090.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0091.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0092.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0093.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0094.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0095.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0096.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0097.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0098.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0099.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0100.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0101.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0102.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0103.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0104.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0105.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0106.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0107.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0108.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0109.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0110.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0111.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0112.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0113.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0114.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0115.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0116.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0117.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0118.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0119.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0120.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0121.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0122.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0123.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0124.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0125.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0126.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0127.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0128.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0129.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0130.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0131.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0132.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0133.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0134.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0135.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0136.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0137.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0138.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0139.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0140.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0141.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0142.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0143.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0144.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0145.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0146.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0147.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0148.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0149.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0150.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0151.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0152.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0153.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0154.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0155.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0156.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0157.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0158.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0159.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0160.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0161.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0162.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0163.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0164.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0165.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0166.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0167.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0168.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0169.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0170.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0171.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0172.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0173.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0174.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0175.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0176.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0177.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0178.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0179.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0180.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0181.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/checkpoints/r0182.json` — hawking.second_light.row_checkpoint.v1
- `reports/condense/second_light/evidence/ADVERSARIAL_VERIFICATION.json` — hawking.second_light.adversarial_verification.v1
- `reports/condense/second_light/evidence/CONTROLLER_STATUS.json` — hawking.successor.resource_sample.v1
- `reports/condense/second_light/evidence/SEED_FROZEN.json` — hawking.second_light.seed_frozen.v1
- `reports/condense/second_light/evidence/STAGED_GATES.json` — STAGED_GATES.json

## reports/condense/general_frontier — 81 files archived (~322,631 bytes, ~7,283 lines)

- `reports/condense/general_frontier/ADAPTIVE_LADDER_PARENT_REGISTRY.json` — ADAPTIVE_LADDER_PARENT_REGISTRY.json
- `reports/condense/general_frontier/FULL_FRONTIER_IGNITION_PRECHECK.json` — hawking.full_frontier_ignition.precheck.v1
- `reports/condense/general_frontier/G2_IGNITION_RECEIPT.json` — hawking.full_frontier.g2_ignition_receipt.v1
- `reports/condense/general_frontier/G2_LAUNCH_READINESS.json` — hawking.full_frontier.g2_readiness.v1
- `reports/condense/general_frontier/G4/checkpoints/code_py__original.json` — code_py__original.json
- `reports/condense/general_frontier/G4/checkpoints/code_py__rvq1.0.json` — code_py__rvq1.0.json
- `reports/condense/general_frontier/G4/checkpoints/gen_paris__original.json` — gen_paris__original.json
- `reports/condense/general_frontier/G4/checkpoints/gen_paris__rvq1.0.json` — gen_paris__rvq1.0.json
- `reports/condense/general_frontier/G4/checkpoints/gen_science__original.json` — gen_science__original.json
- `reports/condense/general_frontier/G4/checkpoints/gen_science__rvq1.0.json` — gen_science__rvq1.0.json
- `reports/condense/general_frontier/G4/checkpoints/instr_list__original.json` — instr_list__original.json
- `reports/condense/general_frontier/G4/checkpoints/instr_list__rvq1.0.json` — instr_list__rvq1.0.json
- `reports/condense/general_frontier/G4/checkpoints/math_add__original.json` — math_add__original.json
- `reports/condense/general_frontier/G4/checkpoints/math_add__rvq1.0.json` — math_add__rvq1.0.json
- `reports/condense/general_frontier/G4/checkpoints/reason_syllogism__original.json` — reason_syllogism__original.json
- `reports/condense/general_frontier/G4/checkpoints/reason_syllogism__rvq1.0.json` — reason_syllogism__rvq1.0.json
- `reports/condense/general_frontier/G4_IGNITION_RECEIPT.json` — hawking.full_frontier.g4_ignition_receipt.v1
- `reports/condense/general_frontier/GENERAL_FRONTIER_BACKEND_PARITY/CUDA_PROVISIONING_PLAN.json` — CUDA_PROVISIONING_PLAN.json
- `reports/condense/general_frontier/GENERAL_FRONTIER_CLOUD_COSTS/CLOUD_BLOCKED_RECEIPT.json` — hawking.general_frontier.cloud_blocked.v1
- `reports/condense/general_frontier/GENERAL_FRONTIER_CLOUD_COSTS/CLOUD_BUDGET_SCHEMA.json` — CLOUD_BUDGET_SCHEMA.json
- `reports/condense/general_frontier/GENERAL_FRONTIER_LEDGER.md` — GENERAL FRONTIER LEDGER
- `reports/condense/general_frontier/GENERAL_FRONTIER_PRECHECK.json` — hawking.general_frontier.precheck.v1
- `reports/condense/general_frontier/GENERAL_FRONTIER_PROGRAMS/G2_COMPLETE_LAYER_PROGRAM.json` — G2_COMPLETE_LAYER_PROGRAM.json
- `reports/condense/general_frontier/GENERAL_FRONTIER_PROGRAMS/G3_CROSS_LAYER_PROGRAM.json` — G3_CROSS_LAYER_PROGRAM.json
- `reports/condense/general_frontier/GIANT_PARENT_PREPARATION.json` — GIANT_PARENT_PREPARATION.json
- `reports/condense/general_frontier/GPT_OSS_120B_G4_UNTREATED_REPORT.md` — GPT-OSS-120B G4 Untreated Control Report
- `reports/condense/general_frontier/HAWKING_FRONTIER_ATLAS.jsonl` — hawking.gpt_oss_120b.second_light_baseline.v1
- `reports/condense/general_frontier/HAWKING_FRONTIER_ATLAS_SCHEMA.json` — HAWKING_FRONTIER_ATLAS_SCHEMA.json
- `reports/condense/general_frontier/HAWKING_FRONTIER_GENERATION_F.json` — hawking.frontier.generation_f.v1
- `reports/condense/general_frontier/KIMI_1T_FULLDISK_DECISION_DRAFT.json` — KIMI_1T_FULLDISK_DECISION_DRAFT.json
- `reports/condense/general_frontier/OVERNIGHT_HANDOFF/SAFETY_AUDIT.md` — Overnight Supervisor - Adversarial Safety Audit
- `reports/condense/general_frontier/QWEN35_397B_ADAPTER_PLAN.json` — QWEN35_397B_ADAPTER_PLAN.json
- `reports/condense/general_frontier/QWEN35_397B_PREP_SEALED.json` — hawking.qwen35_397b.prep_sealed.v1
- `reports/condense/general_frontier/QWEN35_397B_SOURCE_ADMISSION_DRAFT.json` — hawking.qwen35_397b.source_admission.v1
- `reports/condense/general_frontier/QWEN3_235B_Q0_RECEIPT.json` — hawking.qwen3_235b.q0_source_feasibility.v1
- `reports/condense/general_frontier/QWEN3_235B_Q1_RECEIPT.json` — hawking.qwen3_235b.q1_bounded_decode.v1
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/PROBE__gen_paris__S64_structural.json` — PROBE__gen_paris__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/PROBE__gen_science__S64_structural.json` — PROBE__gen_science__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__D1_route_only.json` — code_py__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__D2_recon_only.json` — code_py__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__S2A_adaptive_k.json` — code_py__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__S32_recon_first.json` — code_py__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__S64_doctor.json` — code_py__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__S64_gamma.json` — code_py__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/code_py__S64_structural.json` — code_py__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__D1_route_only.json` — gen_paris__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__D2_recon_only.json` — gen_paris__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__S2A_adaptive_k.json` — gen_paris__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__S32_recon_first.json` — gen_paris__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__S64_doctor.json` — gen_paris__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__S64_gamma.json` — gen_paris__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_paris__S64_structural.json` — gen_paris__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__D1_route_only.json` — gen_science__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__D2_recon_only.json` — gen_science__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__S2A_adaptive_k.json` — gen_science__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__S32_recon_first.json` — gen_science__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__S64_doctor.json` — gen_science__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__S64_gamma.json` — gen_science__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/gen_science__S64_structural.json` — gen_science__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__D1_route_only.json` — instr_list__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__D2_recon_only.json` — instr_list__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__S2A_adaptive_k.json` — instr_list__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__S32_recon_first.json` — instr_list__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__S64_doctor.json` — instr_list__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__S64_gamma.json` — instr_list__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/instr_list__S64_structural.json` — instr_list__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__D1_route_only.json` — math_add__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__D2_recon_only.json` — math_add__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__S2A_adaptive_k.json` — math_add__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__S32_recon_first.json` — math_add__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__S64_doctor.json` — math_add__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__S64_gamma.json` — math_add__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/math_add__S64_structural.json` — math_add__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__D1_route_only.json` — reason_syllogism__D1_route_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__D2_recon_only.json` — reason_syllogism__D2_recon_only.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__S2A_adaptive_k.json` — reason_syllogism__S2A_adaptive_k.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__S32_recon_first.json` — reason_syllogism__S32_recon_first.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__S64_doctor.json` — reason_syllogism__S64_doctor.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__S64_gamma.json` — reason_syllogism__S64_gamma.json
- `reports/condense/general_frontier/QWEN_GRAVITY/checkpoints/reason_syllogism__S64_structural.json` — reason_syllogism__S64_structural.json
- `reports/condense/general_frontier/QWEN_GRAVITY/telegram_delivered.json` — telegram_delivered.json

## reports/condense/gravity_forge — 28 files archived (~74,768 bytes, ~1,758 lines)

- `reports/condense/gravity_forge/FORGE_AUDIT.json` — FORGE_AUDIT.json
- `reports/condense/gravity_forge/FORGE_BASELINE_NEGATIVE.json` — hawking.gravity_forge.baseline_negative.v1
- `reports/condense/gravity_forge/FORGE_F2_RESIDUAL.json` — hawking.gravity_forge.f2_fixture.v1
- `reports/condense/gravity_forge/FORGE_READINESS.json` — FORGE_READINESS.json
- `reports/condense/gravity_forge/FORGE_RUN.json` — hawking.gravity_forge.run_contract.v1
- `reports/condense/gravity_forge/condensation/ACCRETION_PRECHECK.json` — hawking.accretion_precheck.v1
- `reports/condense/gravity_forge/condensation/ARCHB_CHECKPOINT.json` — hawking.archb_checkpoint.v1
- `reports/condense/gravity_forge/condensation/CODEBASE_ESCAPE_RECEIPT.json` — CODEBASE_ESCAPE_RECEIPT.json
- `reports/condense/gravity_forge/condensation/CONDENSATION_PLAN.md` — Stage B condensation plan (CLEAN SLATE Sections 11-27) - grounded in the measured census
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_F2_EVIDENCE.json` — hawking.gpt_oss_120b_f2.v1
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_GRAVITY_PROGRAM.json` — hawking.gpt_oss_120b_gravity_program.v1
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_IGNITION.json` — hawking.120b_ignition.v1
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_READINESS.json` — hawking.gpt_oss_120b_readiness.v2
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_RUN_DOSSIER.json` — hawking.120b_run_dossier.v1
- `reports/condense/gravity_forge/condensation/GPT_OSS_120B_RUN_DOSSIER.md` — GPT-OSS-120B — scientific run dossier (layer 0, 128 experts)
- `reports/condense/gravity_forge/condensation/HAWKING_OWNED_GRAPH.json` — HAWKING_OWNED_GRAPH.json
- `reports/condense/gravity_forge/condensation/HAWKING_OWNED_GRAPH.md` — Hawking owned graph (deduplicated, honest)
- `reports/condense/gravity_forge/condensation/HAWKING_RELEASE_CLOSURE.json` — hawking.release_closure.v1
- `reports/condense/gravity_forge/condensation/HAWKING_REUNIFIED_ARCHITECTURE.md` — Hawking reunified architecture — one Seed, one pack universe
- `reports/condense/gravity_forge/condensation/HAWKING_REUNIFIED_GRAPH.json` — hawking.reunified_graph.v1
- `reports/condense/gravity_forge/condensation/HAWKING_REUNIFIED_METRICS.json` — HAWKING_REUNIFIED_METRICS.json
- `reports/condense/gravity_forge/condensation/HAWKING_SEED_CANDIDATE_COMPARISON.json` — HAWKING_SEED_CANDIDATE_COMPARISON.json
- `reports/condense/gravity_forge/condensation/HAWKING_SEED_CANDIDATE_COMPARISON.md` — Hawking Seed — A / B / C comparison and final selection
- `reports/condense/gravity_forge/condensation/HAWKING_SEED_C_METRICS.json` — hawking.seed_metrics.v1
- `reports/condense/gravity_forge/condensation/PRE_SEED_RELEASE.json` — hawking.pre_seed_release.v1
- `reports/condense/gravity_forge/condensation/SEED_PREDECESSOR_AUDIT.json` — hawking.seed_predecessor_audit.v1
- `reports/condense/gravity_forge/giant_adapters/deepseek-v3.2-685b.json` — deepseek-v3.2-685b.json
- `reports/condense/gravity_forge/giant_adapters/deepseek-v4-pro-1.6t.json` — deepseek-v4-pro-1.6t.json

## reports/condense/gravity_frontier — 40 files archived (~119,369 bytes, ~3,790 lines)

- `reports/condense/gravity_frontier/FRONTIER_SELECTION.json` — hawking.gravity_frontier.frontier_selection.v1
- `reports/condense/gravity_frontier/GPT_OSS_120B_FRONTIER_QUALITY_CONTRACT.json` — hawking.gpt_oss_120b.frontier_quality_contract.v1
- `reports/condense/gravity_frontier/GPT_OSS_120B_GRAVITY_FRONTIER_PROGRAM.json` — GPT_OSS_120B_GRAVITY_FRONTIER_PROGRAM.json
- `reports/condense/gravity_frontier/GPT_OSS_120B_GRAVITY_FRONTIER_READINESS.json` — hawking.gpt_oss_120b.gravity_frontier_readiness.v1
- `reports/condense/gravity_frontier/GRAVITY_FRONTIER_GEOMETRY_RESULT.json` — GRAVITY_FRONTIER_GEOMETRY_RESULT.json
- `reports/condense/gravity_frontier/GRAVITY_FRONTIER_IGNITION_RECEIPT.json` — hawking.gravity_frontier.ignition_receipt.v1
- `reports/condense/gravity_frontier/GRAVITY_FRONTIER_RELEASE_CLOSURE.json` — hawking.gravity_frontier.release_closure.v1
- `reports/condense/gravity_frontier/GRAVITY_FRONTIER_STATE.json` — hawking.gravity_frontier.state.v1
- `reports/condense/gravity_frontier/checkpoints/t0000.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0001.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0002.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0003.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0004.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0005.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0006.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0007.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0008.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0009.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0010.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0011.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0012.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0013.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0014.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0015.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0016.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0017.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0018.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0019.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0020.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0021.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0022.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0023.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0024.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0025.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0026.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0027.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0028.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0029.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0030.json` — hawking.gravity_frontier.trial_checkpoint.v1
- `reports/condense/gravity_frontier/checkpoints/t0031.json` — hawking.gravity_frontier.trial_checkpoint.v1

## reports/condense/deepseek_v4_flash — 10 files archived (~558,313 bytes, ~1,655 lines)

- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_AMPLIFICATION_L05.json` — hawking.deepseek_v4.amplification.v1
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_AMPLIFICATION_L20.json` — hawking.deepseek_v4.amplification.v1
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_AMPLIFICATION_L38.json` — hawking.deepseek_v4.amplification.v1
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_CASCADE_STUDENT.json` — DEEPSEEK_V4_CASCADE_STUDENT.json
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_CONTEXTUAL_PROBE_L05.json` — DEEPSEEK_V4_CONTEXTUAL_PROBE_L05.json
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_MOE_DECOMPOSE_L20.json` — hawking.deepseek_v4.moe_decomposition.v1
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_MOE_PROBE_L20.json` — DEEPSEEK_V4_MOE_PROBE_L20.json
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_MOE_PROBE_L40.json` — DEEPSEEK_V4_MOE_PROBE_L40.json
- `reports/condense/deepseek_v4_flash/DEEPSEEK_V4_PRIMITIVE_PARITY.json` — hawking.deepseek_v4.primitive_parity.v1
- `reports/condense/deepseek_v4_flash/pre_moe_hidden_L05.npy` — pre_moe_hidden_L05.npy

## F1 — condense controller retirement (2026-07-28) — 73 modules deleted from archive, 104 restored as live

The campaign-engine lane had moved 173 files / 102,159 lines into
`tools/condense/archive/` and left 21-line shims that `exec`'d those bodies.
That was relocation, not condensation. Lane F1 made retirement real.

**Annotated tag:** `pre-controller-retirement-20260728` → commit `53435e75f8e80f2b1351f5da0fd4dbea0449f567`

Working-tree state immediately before deletions. Restore any removed path with:

```
git checkout pre-controller-retirement-20260728 -- <path>
# or by commit if the annotated tag is not yet on this clone:
git checkout 53435e75f8e80f2b1351f5da0fd4dbea0449f567 -- <path>
git show pre-controller-retirement-20260728:<path>
```

If the annotated tag ref is missing on this clone (sandbox could not write the shared
`.git` directory from the worktree), recreate it with:

```
git tag -a pre-controller-retirement-20260728 53435e75f8e80f2b1351f5da0fd4dbea0449f567 -m "Working tree before the archived controllers are removed"
```

### F1 / deleted (superseded or unreferenced) — 68 modules, ~23690 lines

Archive bodies and live shims removed. Engine specs keep fixture, receipt, reproduction,
and reopen. Git history holds the code.

- `tools/condense/archive/_bench_fetch_workers.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/_bench_fetch_workers_nodisk.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/codebase_census.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/condense_reachability.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/corpus_integrity.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/decode_parity_harness.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/deepseek_v4_adapter.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/doctor_byte_auction.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/doctor_causal_harness.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/doctor_gen3.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/doctor_treatment_abi.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/forge_actaware_experiment.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/forge_controller_integration.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/forge_f2_fixture.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/forge_giant_adapters.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/forge_pre_run_readiness.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_allocation_probe.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_concurrency_autotune.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_flagship_screening.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_auction.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_cascade.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_controller.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_integration.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_roofline.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_functional_student.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_generation_a_seal.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_lowrank.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_maturity.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_metric_correction.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_moe_student_fit.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_pilot.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_pilot_fetch.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_pilot_seal.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_precheck.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_protect_head_embed.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_rate_ladder.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_readiness_gate.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_rollback_seal.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_source_release.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_teacher_rechain.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/glm52_window_plan.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gptoss_gravity_run.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_attention.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_breakthrough_baseline.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_container_freeze.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_decode.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_functional_metal.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_lab_lease.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_pq_fixture.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_profiler_acceptance.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_runtime.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/gravity_scale_correction.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/hawking_compat.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/hawking_contraction_pilot.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/hawking_tps_budget.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/mech_fidelity_c.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/mech_fidelity_d.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/qwen35_moe_adapter.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/qwen_adaptive_k.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/qwen_function_aware_codec.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/qwen_structural_plan.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/qwen_subhalfbit_search.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/resident_first_ladder.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/safetensors_to_gravity.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/storage_stripdown.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/storage_stripdown_controller.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/succ_eta.py` — deleted; restore via `pre-controller-retirement-20260728`
- `tools/condense/archive/test_activation_aware_roundtrip.py` — deleted; restore via `pre-controller-retirement-20260728`

### F1 / deleted archive duplicates of already-full live modules — 5 modules, ~8537 lines

- `tools/condense/archive/glm52_external_baselines.py` — duplicate of live module; restore via `pre-controller-retirement-20260728` if needed
- `tools/condense/archive/glm52_terminal_proofs.py` — duplicate of live module; restore via `pre-controller-retirement-20260728` if needed
- `tools/condense/archive/glm52_window_execution.py` — duplicate of live module; restore via `pre-controller-retirement-20260728` if needed
- `tools/condense/archive/glm52_xet_live.py` — duplicate of live module; restore via `pre-controller-retirement-20260728` if needed
- `tools/condense/archive/gravity_execution_adapter.py` — duplicate of live module; restore via `pre-controller-retirement-20260728` if needed

### F1 / restored as live (still referenced by tests or live code) — 104 modules, ~78465 lines

These are **not** retired. Bodies moved from `archive/` back to `tools/condense/<module>.py`.
Reported as relocation (earns nothing as condensation).

- `tools/condense/activation_aware_format.py` — restored from archive (tests/live imports)
- `tools/condense/bounded_cache.py` — restored from archive (tests/live imports)
- `tools/condense/doctor_campaign_supervisor.py` — restored from archive (tests/live imports)
- `tools/condense/doctor_v5_gptoss_mxfp4.py` — restored from archive (tests/live imports)
- `tools/condense/doctor_v5_telegram_rung_notifier.py` — restored from archive (tests/live imports)
- `tools/condense/eco_activation.py` — restored from archive (tests/live imports)
- `tools/condense/eco_admission.py` — restored from archive (tests/live imports)
- `tools/condense/eco_cli.py` — restored from archive (tests/live imports)
- `tools/condense/eco_import.py` — restored from archive (tests/live imports)
- `tools/condense/eco_passport.py` — restored from archive (tests/live imports)
- `tools/condense/eco_pipeline.py` — restored from archive (tests/live imports)
- `tools/condense/eco_planner.py` — restored from archive (tests/live imports)
- `tools/condense/eco_status.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_activation_aware_assemble.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_activation_aware_pack.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_activation_aware_pack_v2.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_activation_aware_source.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_basis_pilot.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_campaign_contract.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_capability_gate.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_compact_mla_fixture.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_composition_gate.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_external_baselines.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_flagship_oracle.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_functional_gauntlet.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_gravity_fixture.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_gravity_source.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_long_context_gate.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_moe_student.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_pilot_source_release.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_rehydrate_window.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_resource_policy.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_route_population_census.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_runtime_parity_gate.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_runtime_speed_gate.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_schedule_freeze.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_shard_probe.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_terminal_proofs.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_window_execution.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_xet_live.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_xet_live_driver.py` — restored from archive (tests/live imports)
- `tools/condense/glm52_xet_window_fetch.py` — restored from archive (tests/live imports)
- `tools/condense/gptoss_block.py` — restored from archive (tests/live imports)
- `tools/condense/gptoss_moe_runtime.py` — restored from archive (tests/live imports)
- `tools/condense/gptoss_real_forward.py` — restored from archive (tests/live imports)
- `tools/condense/gptoss_subbit_packer.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_bench_lab.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_execution_adapter.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_flop_ledger.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_forge_run.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_frontier_correction_wave.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_frontier_g4_controller.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_functional_codec.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_global_allocator.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_kernel_select.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_llama_reference.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_metal.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_metal_lab_a.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_metal_lab_b.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_moe_layer.py` — restored from archive (tests/live imports)
- `tools/condense/gravity_real_fixtures.py` — restored from archive (tests/live imports)
- `tools/condense/hawking_null_metric.py` — restored from archive (tests/live imports)
- `tools/condense/kimi_k26_download_supervisor.py` — restored from archive (tests/live imports)
- `tools/condense/kimi_k26_phase2_recovery.py` — restored from archive (tests/live imports)
- `tools/condense/kimi_k26_phase2_release.py` — restored from archive (tests/live imports)
- `tools/condense/kimi_k26_release_cycle.py` — restored from archive (tests/live imports)
- `tools/condense/kimi_k26_stale_download_cleanup.py` — restored from archive (tests/live imports)
- `tools/condense/mech_measure.py` — restored from archive (tests/live imports)
- `tools/condense/mech_run_all.py` — restored from archive (tests/live imports)
- `tools/condense/overnight_supervisor.py` — restored from archive (tests/live imports)
- `tools/condense/qwen3_moe_adapter.py` — restored from archive (tests/live imports)
- `tools/condense/qwen_bpw_budget.py` — restored from archive (tests/live imports)
- `tools/condense/qwen_correction_wave.py` — restored from archive (tests/live imports)
- `tools/condense/qwen_download_worker.py` — restored from archive (tests/live imports)
- `tools/condense/qwen_real_forward.py` — restored from archive (tests/live imports)
- `tools/condense/seal_120b_conclusion.py` — restored from archive (tests/live imports)
- `tools/condense/size_frontier.py` — restored from archive (tests/live imports)
- `tools/condense/source_release_readiness.py` — restored from archive (tests/live imports)
- `tools/condense/studio_manifest.py` — restored from archive (tests/live imports)
- `tools/condense/succ_admission.py` — restored from archive (tests/live imports)
- `tools/condense/succ_audit.py` — restored from archive (tests/live imports)
- `tools/condense/succ_calibrate.py` — restored from archive (tests/live imports)
- `tools/condense/succ_cli.py` — restored from archive (tests/live imports)
- `tools/condense/succ_engine.py` — restored from archive (tests/live imports)
- `tools/condense/succ_events.py` — restored from archive (tests/live imports)
- `tools/condense/succ_frontier.py` — restored from archive (tests/live imports)
- `tools/condense/succ_gc.py` — restored from archive (tests/live imports)
- `tools/condense/succ_gravity.py` — restored from archive (tests/live imports)
- `tools/condense/succ_gravity_policy.py` — restored from archive (tests/live imports)
- `tools/condense/succ_gravity_receipts.py` — restored from archive (tests/live imports)
- `tools/condense/succ_harvest.py` — restored from archive (tests/live imports)
- `tools/condense/succ_press.py` — restored from archive (tests/live imports)
- `tools/condense/succ_queue.py` — restored from archive (tests/live imports)
- `tools/condense/succ_retire.py` — restored from archive (tests/live imports)
- `tools/condense/succ_state.py` — restored from archive (tests/live imports)
- `tools/condense/succ_telegram.py` — restored from archive (tests/live imports)
- `tools/condense/succ_transition.py` — restored from archive (tests/live imports)
- `tools/condense/succ_twin.py` — restored from archive (tests/live imports)
- `tools/condense/succ_watch.py` — restored from archive (tests/live imports)
- `tools/condense/succ_watchdog.py` — restored from archive (tests/live imports)
- `tools/condense/tg_active_byte_budget.py` — restored from archive (tests/live imports)
- `tools/condense/tg_k11_reconcile.py` — restored from archive (tests/live imports)
- `tools/condense/tg_k11_synthetic_schedule.py` — restored from archive (tests/live imports)
- `tools/condense/vulture_harvest.py` — restored from archive (tests/live imports)

### F1 / summary

- eliminated (archive lines gone, not re-homed): ~32227
- relocated (archive → live): ~78465
- archive directory now holds only `tools/condense/archive/README.md`
- receipt: `tools/condense/engine/fixtures/f1_retirement_receipt.json`

- `tools/condense/test_activation_aware_roundtrip.py` — restored as live test (was archived test module; keeps 5 logical cases)


## G1 — lab-tree stub reversal retirements (2026-07-29)

After undoing the stub-and-archive laundering under `tools/bench|training|strand`,
five archive-only modules had **no live twin** and no importers. They were a
duplicate of the live `strand_eval` package (`tools/strand/tools/strand_eval/`
and `tools/strand/scripts/strand_eval/`). Bodies deleted from the working tree;
content remains in git history.

**Tag / restore:**

```
# intended annotated tag (sandbox may block tag write on this worktree):
git tag -a pre-unstub-20260729 -m "Working tree before the lab-tree stub reversal"
# restore any retired path:
git show pre-unstub-20260729:<path>
# or from the commit this lane started on:
git show 655f77c5d927c203aa5c5a85b0e448913d22a88a:<path>
```

| path | lines | summary | fixture | receipt | reopen when |
|------|------:|---------|---------|---------|-------------|
| `tools/strand/archive/strand_eval_scripts_dup/__init__.py` | 63 | strand_eval — THE canonical PPL eval module (audit measurement.md §3.1/§3.2). | none (duplicate of live strand_eval package) | none | If a consumer is found that imported strand_eval_scripts_dup specifically rather than tools/strand/tools/strand_eval |
| `tools/strand/archive/strand_eval_scripts_dup/cli.py` | 95 | strand_eval.cli — the one CLI over the canon eval + ledger. | none (duplicate of live strand_eval package) | none | If a consumer is found that imported strand_eval_scripts_dup specifically rather than tools/strand/tools/strand_eval |
| `tools/strand/archive/strand_eval_scripts_dup/core.py` | 331 | strand_eval.core — the eval engine + the by-construction identity helpers. | none (duplicate of live strand_eval package) | none | If a consumer is found that imported strand_eval_scripts_dup specifically rather than tools/strand/tools/strand_eval |
| `tools/strand/archive/strand_eval_scripts_dup/ledger.py` | 200 | strand_eval.ledger — the results ledger + the tells as code (audit 3.2). | none (duplicate of live strand_eval package) | none | If a consumer is found that imported strand_eval_scripts_dup specifically rather than tools/strand/tools/strand_eval |
| `tools/strand/archive/strand_eval_scripts_dup/qat_shim.py` | 71 | strand_eval.qat_shim — drop-in eval for scripts/strand-qat.py (copy #3 retires). | none (duplicate of live strand_eval package) | none | If a consumer is found that imported strand_eval_scripts_dup specifically rather than tools/strand/tools/strand_eval |

**Reproduction (pre-delete identity check):** these five files lived only under
`tools/strand/archive/strand_eval_scripts_dup/` and were not referenced by any
live module, spec, or shell entrypoint after the unstub. Live eval entrypoints
remain `tools/strand/tools/strand_eval/cli.py` and `tools/strand/scripts/strand_eval/cli.py`.


## H1 — campaign controller → engine specs (2026-07-29)
Lane H1 converted remaining campaign controller shells into engine specs and **deleted** the controller bodies (not archived). **85 modules / 66,442 lines** eliminated. Each deletion keeps specification, fixture, receipt pointer, reproduction command and reopen condition. Restore with `git show HEAD~0:tools/condense/<module>.py` from the pre-H1 tree (or the parent of the H1 commit).

New specs: `succession`, `eco`, `mechanics`, `tg`. Updated: `kimi_k26`, `glm52`, `gravity_frontier`, `qwen`, `gptoss`. Engine operator added: `host_metrics.parse_swap_used` (inlined former emergency-controller helper used by `glm52_source_fetch`).

| path | lines | replaced_by_spec |
|---|---:|---|
| `tools/condense/kimi_k26_phase2_recovery.py` | 3420 | `engine/specs/kimi_k26.json` |
| `tools/condense/glm52_worker.py` | 3307 | `engine/specs/glm52.json` |
| `tools/condense/glm52_window_execution.py` | 3137 | `engine/specs/glm52.json` |
| `tools/condense/kimi_k26_phase2_release.py` | 3109 | `engine/specs/kimi_k26.json` |
| `tools/condense/glm52_notifications.py` | 2982 | `engine/specs/glm52.json` |
| `tools/condense/kimi_k26_download_supervisor.py` | 2771 | `engine/specs/kimi_k26.json` |
| `tools/condense/glm52_basis_pilot.py` | 2339 | `engine/specs/glm52.json` |
| `tools/condense/kimi_k26_release_cycle.py` | 2270 | `engine/specs/kimi_k26.json` |
| `tools/condense/glm52_xet_window_fetch.py` | 2141 | `engine/specs/glm52.json` |
| `tools/condense/emergency_detached_campaign.py` | 1820 | `engine/specs/gravity_frontier.json` |
| `tools/condense/glm52_xet_live_driver.py` | 1657 | `engine/specs/glm52.json` |
| `tools/condense/glm52_route_population_census.py` | 1631 | `engine/specs/glm52.json` |
| `tools/condense/mech_run_all.py` | 1418 | `engine/specs/mechanics.json` |
| `tools/condense/vulture_harvest.py` | 1329 | `engine/specs/gptoss.json` |
| `tools/condense/glm52_campaign_contract.py` | 1239 | `engine/specs/glm52.json` |
| `tools/condense/glm52_pilot_source_release.py` | 1211 | `engine/specs/glm52.json` |
| `tools/condense/kimi_k26_stale_download_cleanup.py` | 1111 | `engine/specs/kimi_k26.json` |
| `tools/condense/doctor_v5_telegram_rung_notifier.py` | 1085 | `engine/specs/gptoss.json` |
| `tools/condense/succ_gravity.py` | 1073 | `engine/specs/succession.json` |
| `tools/condense/succ_press.py` | 1011 | `engine/specs/succession.json` |
| `tools/condense/gravity_metal_lab_a.py` | 934 | `engine/specs/gravity_frontier.json` |
| `tools/condense/mech_measure.py` | 928 | `engine/specs/mechanics.json` |
| `tools/condense/succ_gc.py` | 895 | `engine/specs/succession.json` |
| `tools/condense/seal_120b_conclusion.py` | 857 | `engine/specs/gravity_frontier.json` |
| `tools/condense/overnight_supervisor.py` | 777 | `engine/specs/gravity_frontier.json` |
| `tools/condense/succ_twin.py` | 769 | `engine/specs/succession.json` |
| `tools/condense/succ_transition.py` | 711 | `engine/specs/succession.json` |
| `tools/condense/succ_watchdog.py` | 698 | `engine/specs/succession.json` |
| `tools/condense/glm52_schedule_freeze.py` | 692 | `engine/specs/glm52.json` |
| `tools/condense/gravity_global_allocator.py` | 688 | `engine/specs/gravity_frontier.json` |
| `tools/condense/source_release_readiness.py` | 644 | `engine/specs/gptoss.json` |
| `tools/condense/qwen_real_forward.py` | 618 | `engine/specs/qwen.json` |
| `tools/condense/succ_telegram.py` | 618 | `engine/specs/succession.json` |
| `tools/condense/eco_planner.py` | 614 | `engine/specs/eco.json` |
| `tools/condense/succ_gravity_policy.py` | 558 | `engine/specs/succession.json` |
| `tools/condense/qwen3_moe_adapter.py` | 545 | `engine/specs/qwen.json` |
| `tools/condense/glm52_external_baselines.py` | 534 | `engine/specs/glm52.json` |
| `tools/condense/qwen_correction_wave.py` | 533 | `engine/specs/qwen.json` |
| `tools/condense/succ_admission.py` | 527 | `engine/specs/succession.json` |
| `tools/condense/test_activation_aware_roundtrip.py` | 525 | `engine/specs/glm52.json` |
| `tools/condense/glm52_resource_policy.py` | 494 | `engine/specs/glm52.json` |
| `tools/condense/gravity_frontier_correction_wave.py` | 483 | `engine/specs/gravity_frontier.json` |
| `tools/condense/succ_cli.py` | 482 | `engine/specs/succession.json` |
| `tools/condense/gravity_execution_adapter.py` | 448 | `engine/specs/gravity_frontier.json` |
| `tools/condense/eco_import.py` | 401 | `engine/specs/eco.json` |
| `tools/condense/succ_harvest.py` | 386 | `engine/specs/succession.json` |
| `tools/condense/succ_frontier.py` | 381 | `engine/specs/succession.json` |
| `tools/condense/glm52_compact_mla_fixture.py` | 378 | `engine/specs/glm52.json` |
| `tools/condense/studio_manifest.py` | 375 | `engine/specs/glm52.json` |
| `tools/condense/glm52_activation_aware_assemble.py` | 368 | `engine/specs/glm52.json` |
| `tools/condense/eco_activation.py` | 362 | `engine/specs/eco.json` |
| `tools/condense/glm52_capability_gate.py` | 361 | `engine/specs/glm52.json` |
| `tools/condense/glm52_runtime_parity_gate.py` | 347 | `engine/specs/glm52.json` |
| `tools/condense/gravity_frontier_g4_controller.py` | 340 | `engine/specs/gravity_frontier.json` |
| `tools/condense/tg_k11_synthetic_schedule.py` | 311 | `engine/specs/tg.json` |
| `tools/condense/succ_gravity_receipts.py` | 308 | `engine/specs/succession.json` |
| `tools/condense/eco_pipeline.py` | 299 | `engine/specs/eco.json` |
| `tools/condense/succ_queue.py` | 298 | `engine/specs/succession.json` |
| `tools/condense/glm52_long_context_gate.py` | 292 | `engine/specs/glm52.json` |
| `tools/condense/doctor_campaign_supervisor.py` | 289 | `engine/specs/gravity_frontier.json` |
| `tools/condense/glm52_runtime_speed_gate.py` | 286 | `engine/specs/glm52.json` |
| `tools/condense/tg_active_byte_budget.py` | 281 | `engine/specs/tg.json` |
| `tools/condense/eco_passport.py` | 278 | `engine/specs/eco.json` |
| `tools/condense/succ_engine.py` | 257 | `engine/specs/succession.json` |
| `tools/condense/glm52_gravity_fixture.py` | 241 | `engine/specs/glm52.json` |
| `tools/condense/succ_watch.py` | 241 | `engine/specs/succession.json` |
| `tools/condense/eco_admission.py` | 237 | `engine/specs/eco.json` |
| `tools/condense/glm52_activation_aware_source.py` | 232 | `engine/specs/glm52.json` |
| `tools/condense/succ_audit.py` | 232 | `engine/specs/succession.json` |
| `tools/condense/eco_status.py` | 231 | `engine/specs/eco.json` |
| `tools/condense/succ_retire.py` | 231 | `engine/specs/succession.json` |
| `tools/condense/succ_calibrate.py` | 229 | `engine/specs/succession.json` |
| `tools/condense/eco_cli.py` | 227 | `engine/specs/eco.json` |
| `tools/condense/tg_k11_reconcile.py` | 226 | `engine/specs/tg.json` |
| `tools/condense/succ_state.py` | 222 | `engine/specs/succession.json` |
| `tools/condense/gravity_forge_run.py` | 211 | `engine/specs/gravity_frontier.json` |
| `tools/condense/glm52_gravity_source.py` | 190 | `engine/specs/glm52.json` |
| `tools/condense/glm52_rehydrate_window.py` | 190 | `engine/specs/glm52.json` |
| `tools/condense/gravity_llama_reference.py` | 187 | `engine/specs/gravity_frontier.json` |
| `tools/condense/qwen_bpw_budget.py` | 186 | `engine/specs/qwen.json` |
| `tools/condense/succ_events.py` | 186 | `engine/specs/succession.json` |
| `tools/condense/glm52_composition_gate.py` | 165 | `engine/specs/glm52.json` |
| `tools/condense/size_frontier.py` | 132 | `engine/specs/glm52.json` |
| `tools/condense/glm52_flagship_oracle.py` | 118 | `engine/specs/glm52.json` |
| `tools/condense/qwen_download_worker.py` | 97 | `engine/specs/qwen.json` |

**H1 summary:** eliminated 66,442 lines across 85 modules; relocated 0; non-engine modules 111,424 → under 45k target.
