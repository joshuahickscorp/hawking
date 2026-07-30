# HAWKING Evidence Index (A-EVID-R1)

Audit map for Core A narrative release. Not a history archive.
Rollback base: `0250cdb2ba035303da723e2a2eb41edcaf1fc45e`. Recover: `git show 0250cdb2ba035303da723e2a2eb41edcaf1fc45e:<path>`.

## Retained operative authorities

- `KIMI_K26_GRAVITY_FINAL.md` (1100) — sealed gravity final. glm52_terminal_proofs FROZEN sha.
- `KIMI_K26_NEXT_PARENT_TRANSFER.md` (60) — sealed parent transfer. same terminal-proof reader.
- `docs/dead_levers.md` (492) — kill ledger. BB BC-ACCEL-012; BC-ACCEL-012.
- `BASELINES.md` (296) — source pins. BB BC-SOURCE-010; BC-SOURCE-005..010.
- `ACCELERATION_ARCHAEOLOGY.md` (214) — accel archaeology. BC-ACCEL-009.
- `HAWKING_250K_GRAPH_FINDINGS.md` (191) — graph findings. graph mission authority.
- `README.md` (185) — user entry. BC-CLI/ARTIFACT/GENERATION/SERVER evidence.
- `docs/serve.md` (158) — server ops. BC-SERVER-001..004,013.
- `docs/gravity/GRAVITY_CONTAINER_SPEC.md` (153) — Gravity container. BB BC-ARTIFACT-014.
- `ARCHITECTURE.md` (135) — architecture. BC-ARTIFACT/BRIDGE/GENERATION/SERVER.
- `docs/BENCHMARKS.md` (112) — bench + hotness. BC-CLI-011; extract_other.py read.
- `FABRIC_BRIDGE_ARCHAEOLOGY.md` (98) — bridge archaeology. BC-BRIDGE-002.
- `docs/env_flags.md` (60) — env flags. MIG-021; multi-BC.
- `MODELS.md` (46) — model/quant. BC-ARTIFACT-002; BC-GENERATION-*.
- `docs/kernels.md` (28) — kernels + hotness. extract_other.py read.
- `GLM52_V2_PROGRAM_FEASIBILITY.md` (167) — Core C dual write. glm52_activation_aware_pack_v2.py.
- `GLM52_CORPUS_INTEGRITY.md` (71) — Core C dual write. glm52_corpus.py.
- `GLM52_STREAMING_SCHEDULE.md` (37) — Core C dual write. glm52_contract.py.
- `GLM52_REFERENCE_PARITY.md` (16) — Core C dual write. glm52_parity.py.
- `GLM52_XET_AUTOTUNE_PLAN.md` (16) — Core C dual write. glm52_xet_autotune.py.

Also retained untouched: Gravity schema/compat/vectors JSON; constitution;
blackbox matrix; migrations; performance baselines; F1 seals; capability;
`control/**/*.md`; graph payload/viewer.

## Sealed KIMI digests

- `KIMI_K26_GRAVITY_FINAL.md` — `04370b55d1073923877989874cbf336869c949cd4892dbe3e1a845c5e2fc0752`
- `KIMI_K26_NEXT_PARENT_TRANSFER.md` — `8c34b679524327e4a3ff61bc82fe7d451f62c42f3d200e7aafe0423a10a01a34`

## Released families

Pure delete; no facade at old paths. Graph dumps/doc-comments are inventory-only.

### archive_indexes (2 files / 1688 LOC)
Successor: git history; this index. Reason: historical catalogs; no product/gate reader.
- `docs/ARCHIVE_INDEX.md` — 328
- `docs/ARCHIVE_INDEX_2.md` — 1360

### hide_bible_historical (5 files / 2164 LOC)
Successor: HIDE product code; HIDE_ARCHAEOLOGY*.json. Reason: superseded HIDE bible narrative.
- `docs/hide-bible/DESIGN_DOCTRINE.md` — 559
- `docs/hide-bible/HIDE_PLAN.md` — 1077
- `docs/hide-bible/README.md` — 24
- `docs/hide-bible/SCAFFOLD_AUDIT.md` — 448
- `docs/hide-bible/SCAFFOLD_STATUS.md` — 56

### hide_impl_consolidation_narrative (4 files / 1004 LOC)
Successor: hide-protocol/backend code; doc-comments. Reason: consolidation narrative; comment-only refs.
- `docs/hide-impl/consolidation/HIDE_BACKEND_WITHOUT_SURFACE_REPORT.md` — 210
- `docs/hide-impl/consolidation/HIDE_COMMAND_REGISTRY_SPEC.md` — 324
- `docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md` — 278
- `docs/hide-impl/consolidation/HIDE_DONOR_PORT_LEDGER.md` — 192

### plans_comment_only (8 files / 3840 LOC)
Successor: crate/tool module docs; soft cliff_watchdog corpus. Reason: historical plans; no hard runtime open.
- `docs/plans/CONDENSER_ECOSYSTEM_FRONTIER.md` — 1899
- `docs/plans/PROMETHEUS_LEG_PLAN.md` — 690
- `docs/plans/agentic_tool_system_2026_07_11.md` — 506
- `docs/plans/condense_frontier_2026_06_22.md` — 277
- `docs/plans/condense_master_plan_2026_06_22.md` — 262
- `docs/plans/eh_verify_kernel_losslessness_2026_06_21.md` — 69
- `docs/plans/hide_release_autoupdate.md` — 80
- `docs/plans/native_tq_serving_impl.md` — 57

### campaign_status_narrative (13 files / 2301 LOC)
Successor: control/rungs/*; HAWKING_250K_GRAPH_FINDINGS.md; JSON twins. Reason: superseded campaign status/handoff prose.
- `HAWKING_250K_ARCHITECTURE_DECISION.md` — 202
- `HAWKING_250K_STATUS.md` — 348
- `HAWKING_300K_DECISION_REQUIRED.md` — 96
- `HAWKING_ASCENSION_CLOSED.md` — 96
- `HAWKING_FINAL_ASCENT_CONTINUATION_GOAL.md` — 59
- `HAWKING_RECOMPOSITION_HANDOFF.md` — 143
- `HAWKING_RESUME_CHECKPOINT.md` — 149
- `HAWKING_VNEXT_ARCHITECTURE_A1.md` — 240
- `HAWKING_VNEXT_ARCHITECTURE_A2.md` — 633
- `HISTORY_PUBLICATION_PACKET.md` — 106
- `ODYSSEY_PROMOTION_GATE.md` — 99
- `REBUILD_PERFORMANCE_BASELINE_RECONCILIATION.md` — 70
- `docs/HISTORY_REWRITE_20260728.md` — 60

### hide_archaeology_md_json_twins (2 files / 660 LOC)
Successor: HIDE_ARCHAEOLOGY.json / HIDE_ARCHAEOLOGY_V2.json. Reason: MD twin of retained archaeology JSON.
- `HIDE_ARCHAEOLOGY.md` — 370
- `HIDE_ARCHAEOLOGY_V2.md` — 290

### kimi_nonsealed_status (5 files / 352 LOC)
Successor: KIMI_*.json; sealed KIMI MD pair retained. Reason: non-sealed status; sealed pair kept.
- `KIMI_K26_DEVICE_CLEANSE_FINAL.md` — 36
- `KIMI_K26_FINAL_CHAPTER_STATUS.md` — 30
- `KIMI_K26_LONG_RUN_FINAL.md` — 218
- `KIMI_K26_LONG_RUN_REGION_CLOSURE.md` — 30
- `KIMI_K26_LONG_RUN_STATUS.md` — 38

### glm52_md_json_twins_non_writer (6 files / 591 LOC)
Successor: sibling *.json receipts; Core C dual MD retained. Reason: non-writer MD with JSON twin.
- `GLM52_BASIS_PILOT_RECEIPT.md` — 183
- `GLM52_CORRECTED_SCIENTIFIC_LAW.md` — 184
- `GLM52_FUNCTIONAL_DECISION.md` — 83
- `GLM52_HANDOFF_PRECHECK.md` — 24
- `GLM52_ROUTE_POPULATION_CENSUS.md` — 77
- `GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.md` — 40

### stretch_user_docs (10 files / 963 LOC)
Successor: git log; numeric_parity.rs; GRAVITY_CONTAINER_SPEC.md; performance JSON. Reason: user/ops prose with code/JSON/git successors.
- `CHANGELOG.md` — 224
- `NUMERIC_PARITY_V2_1.md` — 152
- `GRAVITY_FUNCTIONAL_CODEC_SPEC.md` — 144
- `HAWKING_MODEL_FEEL_PARITY_CONTRACT.md` — 70
- `docs/SPEED.md` — 93
- `docs/autotune.md` — 81
- `docs/profile.md` — 74
- `CONTRIBUTING.md` — 57
- `FAILURES.md` — 41
- `GRAVITY_EXTERNAL_BASELINE_MATRIX.md` — 27

## Totals and rollback

- Released: 55 paths / 13563 LOC
- Apparatus: this file only (counted); ledger `control/A-EVID-R1-ledger.json` (uncounted)
- Rollback: `git checkout 0250cdb2ba035303da723e2a2eb41edcaf1fc45e -- <path>` or revert the A-EVID-R1 commit

## Reader classification

- Product/gate hard open of released MD: none
- Soft corpus: cliff_watchdog `cat … 2>/dev/null` of two plan MD + README
- Doc-comments only: app/crates plan and consolidation path strings
- BB/MIG/constitution/F1: no released path pin; dual-written Core C MD retained
- Historical citations in retained docs recover via pre-rung git show

## Six-bucket intent

```text
eliminated            13563
rewritten                 0
generated                 0
relocated                 0
compatibility_facade      0
added_apparatus         148
```

