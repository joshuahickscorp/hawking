# Hawking consolidation inventory

**Mode:** read-only audit. Delete nothing. Move nothing. Refactor nothing.
**Commit:** `d9bca273e86ca189a7b550f80734db7bf9ca94ad` (`campaign/glm52-generation-b`)
**At:** 2026-07-26
**Method:** entry-points-inward static tracing + cross-check of `HIDE_ARCHAEOLOGY.md`, `FABRIC_BRIDGE_ARCHAEOLOGY.md`, `ACCELERATION_ARCHAEOLOGY.md`, `HAWKING_RECLAMATION_SURVEY.md`, `docs/dead_levers.md`, `CONDENSE_AUDIT.md`.
**Evidence level:** STATIC_SOURCE_READING + committed archaeology/receipts.

**Standing bias:** default **keep**. Sealed negatives are scientific history, not dead code. Goal is **genuine duplicate authorities**.

**Active lanes (overlap only, do not consolidate into):** HIDE wiring, Fabric, Bridge/adapters/events, Odyssey, speculation safety, Temporal Gravity / Math-Preserve (forbidden paths in `_COMMON.md`).

**Write note:** OS denied writes under the campaign checkout (`Operation not permitted`). Canonical copies:
- `/tmp/hawking_consolidation_inventory/HAWKING_CONSOLIDATION_INVENTORY.{md,json}`
- session scratchpad (same filenames)
- `~/.grok/hawking_consolidation_inventory/`

```bash
cp /tmp/hawking_consolidation_inventory/HAWKING_CONSOLIDATION_INVENTORY.{md,json} \
   /Users/scammermike/Downloads/hawking/
```

---

## Role authorities

### 1. public CLI
- **Live:** `hawking` — `crates/hawking/src/main.rs:11-23` Cli, `:127-675` Cmd, dispatch `:693+`
- **Also live:** `hide-serve` `crates/hide-serve/src/main.rs:17-28`; `hide-acp-server`; `hide-sdk-codegen` (codegen)
- **Campaign:** `hawking-seed-c` `crates/hawking-seed-c/src/main.rs:36-52` (run|f2|gravity-run|status|inspect|verify|drain|resume)
- **Subcommands:** Serve, Generate, Tokenize, Bench, Autotune, BenchQ4kShapes, Doctor, ProfileRank, Stats, Version, BatchHash, ShaderHash, BakeSidecar, Verify, Press, Fit, SpecOracle
- **Shadowed:** BakeSidecar plan-only (`main.rs:557-558`); Press dry-run only (`:597-601`); extracted studio/bench packs
- **Live entry:** generate/serve → `hawking_serve::run` (`main.rs:841`) or `load_engine` (`main.rs:2329,3106`) → `hawking-core/src/model/mod.rs:26`
- **Cost/risk:** low value to merge bins; high if seed-c merged without golden migration

### 2. model registry
| Candidate | Path | Live? |
|---|---|---|
| load_engine | model/mod.rs:26-110 | **YES** sole production factory |
| GravityEngine | gravity_engine.rs:59-150 | **YES** via magic at mod.rs:67-70 |
| RoleRegistry | hawking-orch/registry.rs:13-104 | **YES** roles, not weights |
| seed-c providers::Registry | providers/registry.rs:72+ | campaign-only |
| PRODUCTION_EXECUTION_ADAPTER_REGISTRY | tools/condense/glm52_worker.py:111-114 | **empty by contract** |
| HttpModelProvider | hide-backend | **YES** when HIDE_MODEL_WEIGHTS boots |

Live path: `hawking-serve/src/lib.rs:523` → load_engine. Shipping arches: llama*, mistral, deepseek2*, qwen2*, qwen*moe*, rwkv7; gravity llama + glm_moe_dsa. Extracted: gemma2/phi3/olmoe/mamba2/mixtral (mod.rs:15-17).

### 3. adapter ABI
- **Live execute:** Engine trait + load_engine + GravityEngine
- **Shadow metadata:** ArchAdapter `hawking-seed-c/src/providers/adapters.rs:96-299` (build_plan does **not** execute)
- **Campaign:** seed-c adapter.rs IR plan; tools/condense/*_adapter.py; gravity_execution_adapter unregistered
- **Cost/risk:** high if forced early — Bridge/adapter lane owns; keep empty PRODUCTION registry

### 4. context compiler
- **Live:** `hawking_context::ContextCompiler` compiler.rs:272+ on every SubmitTurn (run_turn_core/kernel)
- **Also live:** Project Brain SqliteMemoryStore; host MemoryLedger (different product)
- **Cost/risk:** high to merge memory planes for little gain

### 5. tool / effect registry
- **Live:** hide_core ToolRegistry + `hide-backend/src/tools.rs:11` build_default_tool_registry + builtins
- **Live MCP:** `host.rs:614-667` register_mcp_servers_at_boot (supersedes older HIDE archaeology UNWIRED)
- **Partial:** kernel parse→run only if HIDE_KERNEL_TURN=1
- **Separate ABI:** hide-extension-registry capability Registry (not execute dispatcher)

### 6. session / event authority
| Authority | Path | Live? |
|---|---|---|
| SessionRegistry | services.rs:152, open_or_create:415 | YES |
| hide_core::Event | event.rs:52 | YES durable log |
| UiEvent / UiEventBus | api.rs:78, ui_bus.rs | YES Wire-B |
| hide-protocol Items/Methods | hide-protocol | schema; transport partial |
| StreamEvent | engine.rs:188 | YES inference |
| seed-c state::Event | state.rs:52 | campaign |
| JSONL ledgers | root | archive |

Six competitors; intended durable winner is hide_core::Event. Bridge lane owns consolidation — do not delete.

### 7. acceleration registry
- No AccelerationRegistry type — SpeculateMode + flags + generate loops
- Opt-in live: user_ngram, ExactShared, EH partial, eagle5 load, replay_oracle CLI
- Unwired/stub: retrieval, suffix_automaton, parallel_draft, eagle_proposer, tree, cross_tokenizer (ACCELERATION_ARCHAEOLOGY)
- Lane: speculation safety; sealed negatives bind

### 8. Fabric planner
- **Reachable:** fleet_run in HANDLED_CUSTOM_NAMES host.rs:497; arm :1308-1333; method :2843-2880
- **Scaffold honesty:** FixedResourceProbe fake 32GiB, with_fake_worktrees(), AgentKernel::new StubPlanner (:2848-2866)
- Unwired: choose_pattern (patterns.rs:87 unit-only), merge tournament
- RuntimePlanner live only on build_turn_kernel (:2383) when HIDE_KERNEL_TURN=1
- Fabric/HIDE lane owns — no deletes

### 9. artifact loader
- Live funnel: load_engine → gravity magic or GgufFile::open
- Also: GravityShard multi-shard, TQ loaders, safetensors (eagle5/seed), sidecar honor read-only
- seed-c own gguf/gravity: intentional isolation (Cargo.toml comment)
- Cost/risk high to merge loaders

### 10. provenance authority
- hide_core::types::Provenance types.rs:58 (live HIDE)
- hide_extension_registry Provenance manifest.rs:227 (capability pins)
- seed-c evidence receipts; campaign seal_sha256; cost_ledger notes
- No single store; low priority unify

---

## Duplicates / dead adapters / stale names / forks / copied arch / orphan schemas

**Duplicates (keep unless noted):** load_engine vs RoleRegistry; ToolRegistry vs extension-registry; SessionRegistry vs unwired hide-state; Event/UiEvent/StreamEvent until Bridge; ArchAdapter vs Engine (document dual ABI); InMemory vs Sqlite index (Sqlite unwired); MemoryLedger vs Project Brain; seed-c Gguf vs core Gguf.

**Dead adapters:** in-tree gemma2/phi3/olmoe/mamba2/mixtral **extracted** (not deleted); ArchAdapter builtins declarative-only; PRODUCTION registry empty by contract; gravity_execution_adapter unregistered; MODELS.md/ARCHITECTURE.md stale present-claims.

**Stale names:** Eagle5 v1; functional full-admit; Claim A as filed; Gen-B raw cosine; Nuclear Pasta label; HIDE archaeology MCP/fleet UNWIRED claims (partially stale).

**Benchmark forks:** hawking-bench competitors; kernel_bench; gravity_glm examples; seed-c golden; spec-oracle; test load_engine helpers.

**Copied architecture:** LlamaDense vs seed ArchAdapter/plan; GravityLlama vs GGUF llama; gravity_glm vs glm52 Python pack; qwen_dense single path.

**Orphan schemas:** hide-state capsules unwired; hide-protocol schema authority with partial transport; Tauri dual schemas deliberate; campaign JSON schemas archive.

---

## Flags (146 HAWKING_*/HIDE_* refs)

Legend: default-off-and-live | default-on-or-profile | dead-or-refuse-scaffold | test-only | lane-owned

| Flag | Class | Evidence |
|---|---|---|
| HIDE_KERNEL_TURN | default-off-and-live | host.rs:6298-6301; must stay off |
| HIDE_KERNEL_AUTONOMY | default-off-and-live | default SuggestOnly |
| HIDE_MODEL_WEIGHTS / ADDR | default-off-and-live | supervisor boot |
| HAWKING_QWEN_USER_DRAFT | default-off-and-live | qwen_dense + CLI |
| HAWKING_QWEN_EVENT_HORIZON | default-off-and-live | EH path |
| HAWKING_QWEN_SPEC_GOVERNOR | default-off-and-live | qwen_dense.rs:1786 |
| HAWKING_SPEC_DECODE | default-off-and-live | engine SpeculateMode |
| HAWKING_FORCE_CPU / TRACE_DISPATCH / COST_LEDGER | default-off-and-live | core |
| HAWKING_GLM_GPU_RESIDENT_STATE | default-off-and-live + sealed negative | RESIDENT_STATE_NEGATIVE |
| HAWKING_GLM_GPU_EXPERT_WAVE | default-off-and-live + sealed negative | EXPERT_WAVE_NEGATIVE |
| HAWKING_GLM_GPU_LM_HEAD | default-off-and-live | TG lane-owned |
| HAWKING_QWEN_Q4K_PREDEC / pair / fuse | default-on-or-profile | RuntimeProfile |
| VOCAB_PRUNE / FFN_DOWN_Q4K / f16-scales | profile fast; exact force-off | lever_plan |
| HAWKING_EH_SAM / EH_PARALLEL_DRAFT | dead-or-refuse-scaffold | not registered / stub |
| HAWKING_BACKEND_SEAM | default-off-and-live | qwen_dense.rs:5226 |
| HAWKING_TEST_WEIGHTS* / LLAMA_CLI / MLX_* | test-only | tests/bench |

Policy: low file-ref count ≠ dead. Full list in JSON `flags`.

---

## Unwired scaffolds vs archaeology

| Scaffold | This audit | HIDE arch | FABRIC | Agree? |
|---|---|---|---|---|
| fleet_run | WIRED scaffold (fake probe/worktrees/StubPlanner) | REAL_UNWIRED | reachable+fake | **Disagree HIDE reachability; agree FABRIC quality** |
| fleet merge/tournament | UNWIRED | REAL_UNWIRED | — | Agree |
| choose_pattern | UNWIRED | — | unit-only | Agree |
| MCP boot | **WIRED** host.rs:614-667 | REAL_UNWIRED | REAL_WIRED | **Disagree HIDE; agree FABRIC** |
| ACP BackendHost handler | Deferred | PARTIAL | PARTIAL | Agree |
| hide-state capsules | Unwired | REAL_UNWIRED | — | Agree |
| SqliteCodeIndex at open | Unwired | PARTIAL | — | Agree |
| Initialize hide-serve route | Missing | noted | — | Agree |
| SDK Transport | Deferred | — | PARTIAL | Agree |
| Anthropic /v1/messages | Missing | — | MISSING | Agree |
| Spec retrieval/SAM/tree/… | Unwired/stub | — | ACCEL | Agree ACCEL |
| BakeSidecar write / Press bake | Unimplemented | — | — | New |
| MixedQuantStore requant | read-only honor | — | dead_levers | Agree keep |

---

## Scientific history clusters (~273 root receipts)

| Cluster | N | Dates | Status | Superseded by | Reproduction | Relevance | Must survive |
|---|---|---|---|---|---|---|---|
| KIMI_K26 | 79 | 07-21..22 | CLOSED / BOUNDARY | parent→GLM52 | long-run/gravity finals | scientific law | **YES** |
| GLM52 | 44 | 07-21..26 | FUNCTIONAL_PARTIAL_ONLY; Claim A blocked | corrected law | pack/Math-Preserve | flagship parent | **YES** |
| HAWKING continuum/ascension/motherload | ~59 | 07-23..26 | ASCENSION_CLOSED; ODYSSEY_READY | HIDE/Odyssey next | ledgers/gates | control plane | **YES** |
| DEEPSEEK_V4 | 11 | 07-23 | FAVOURABLE_NOT_UNIFORMLY_CONTRACTIVE | cascade | flash pilot | functional reopen | **YES** |
| HIDE archaeology/reviews | 9 | 07-25..26 | recon | future wiring receipts | static | product map | keep |
| Acceleration | 3 | 07-26 | bug fixed; ledger stale | TQ re-receipt | EH/parity | spec science | **YES** |
| Resident/expert-wave negatives | 2 | 07-26 | NEGATIVE sealed | — | GLM GPU flags | falsified collapses | **YES** |
| DOCTOR/SUBBIT/GRAVITY | ~14 | 07-21..23 | prechecks/atlases | doctor gen3 | doctor CLIs | negative transfer | **YES** |
| PROMETHEUS/PASS3 | 2+ | 07-24..25 | allocation | continuum pass3 | workers | math alloc | keep |
| NUCLEAR_PASTA | 2 | 07-17..18 | historical | extracted crates | — | LOC ledger | keep |
| CONDENSE_AUDIT | 3 | 07-03 | completed | — | go-plan | almost nothing removable | keep |
| NUMERIC_PARITY | 3 | 07-26 | V2.1 | — | harness | math contract | **YES** |
| RAMANUJAN/MOP | 6 | 07-26 | assessment | step 14 | assess tools | training gov | keep |
| ODYSSEY | fence/gate | — | CANONICAL gate | controller flip | — | training fence | **YES** |
| Provider/ladder/prechecks | ~10 | 07-19..23 | prechecks | ladder V3 | — | admission | keep |
| Standing docs | many | ongoing | canonical | — | — | operating law | **YES** |

### Sealed negatives that must survive any consolidation
1. Kimi K2.6 law + long-run + gravity finals
2. GLM FUNCTIONAL_PARTIAL_ONLY + corrected scientific law + Math-Preserve chain
3. RESIDENT_STATE + EXPERT_WAVE negatives
4. Eagle5/EAGLE-3 NO-GO (dead_levers)
5. Acceleration losslessness diagnosis/fix narrative
6. Ascension closed + Odyssey gate + ODYSSEY_READY
7. DeepSeek V4 contraction pilot
8. Doctor negative-transfer atlas
9. Empty PRODUCTION_EXECUTION_ADAPTER_REGISTRY contract
10. docs/dead_levers.md Type-1 kill set

---

## Ranked actions (later lane)

| # | Action | Blast | Blocked by | Evidence | Recommendation |
|---|---|---|---|---|---|
| 1 | Document dual ABI (load_engine vs ArchAdapter vs empty PRODUCTION) | docs | — | mod.rs:26; adapters.rs; glm52_worker | **do docs** |
| 2 | Refresh HIDE_ARCHAEOLOGY fleet_run + MCP | docs | — | host.rs:497,614,1308,2843 | **do docs** |
| 3 | Fix MODELS.md/ARCHITECTURE.md extracted-arch claims | docs | — | FABRIC_BRIDGE; mod.rs:15-17 | **do docs** |
| 4 | Fabric Os probe + real worktrees + RuntimePlanner | hide-backend/fleet | fabric lane | host.rs:2848-2866 | lane-owned |
| 5 | Wire merge tournament | hide-fleet | (4) | merge.rs | lane-owned |
| 6 | Event projection Event→UiEvent | hide/* | bridge | six competitors | lane-owned |
| 7 | ACP TurnHandler→BackendHost | hide-acp | HIDE wiring | DeferredTurnHandler | lane-owned |
| 8 | SqliteCodeIndex at open | index | product | InMemory only | optional |
| 9 | hide-state RPC or demote claims | hide-state | product | NotImplemented | document/wire; no delete |
| 10 | Spec register or ARCHIVE with dead_levers | speculate | spec safety | ACCEL arch | keep stubs |
| 11 | Archive idle campaign Python after gates | tools/ | Math-Frozen+HIDE+Fabric+Bridge | RECLAMATION | archive only after |
| 12 | True orphans after human rg | tiny | human | CONDENSE_AUDIT | prefer archive |
| 13 | Unify Provenance types | hide crates | low prio | two structs | defer forever |
| 14 | Merge seed-c gguf into core | seed-c/core | API drift | Cargo.toml | **do not** |
| 15 | Flip HIDE_KERNEL_TURN default on | host | approval + standing law | host.rs:6290+ | **forbidden** |

## must_not_delete
- `_COMMON.md` forbidden paths
- All sealed-negative / scientific-law clusters above
- Empty PRODUCTION registry contract + tests
- dead_levers + kill evidence
- Active-lane scaffolds (HIDE/Fabric/Bridge/Odyssey/speculation)
- Dual event types until Bridge
- Extracted-adapter pack references / load_engine error strings

## What this audit did not do
No deletions, moves, refactors, flag flips. No cargo test/heavy generate. No open of sealed Math-Preserve artifact. No line-count target.

## Companion
Machine-readable: `HAWKING_CONSOLIDATION_INVENTORY.json` (`hawking.consolidation.inventory.v1`).
