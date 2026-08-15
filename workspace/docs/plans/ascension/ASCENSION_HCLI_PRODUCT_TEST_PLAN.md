# Ascension HCLI product-test plan

**Bible:** §33 (HCLI product tests), Final Directive  
**Status:** PLAN + harness scaffold (no live model work in this lane)  
**Primary metric:** `verified_tasks_completed_per_hour`  
**Agent speed definition:**

```text
question
→ trustworthy evidence
→ implemented result
→ verified result
```

Token generation alone is **not** the product metric.

Also record: per-agent latency, aggregate accepted TPS, patch acceptance, tests
passed, retries, regressions, human intervention, thermal stability, search
latency, tool-selection latency, memory retrieval cost, planning failure rate.

---

## 1. Purpose

Generalize the DeepSeek-V4 diagnostic HCLI live suite already run against the
local diagnostic endpoint into a **family-agnostic product-test harness** that
any admitted Gravity/HCLI family (Qwen 30B, Qwen Next 80B, later ladder models)
can satisfy without rewriting the case catalog.

This plan does **not** claim any product case is done for Qwen. The DeepSeek
diagnostic run is **evidence that the harness pattern works**, not a TG or
production promotion.

---

## 2. Existing suite audit (what already exists)

### 2.1 Core packer + contracts

| Artifact | Role |
|----------|------|
| `lab/operators/deepseek_v4_gravity.py` → `hcli_live_suite_receipt()` | Privacy-preserving aggregate sealer: binds evidence files to a sealed diagnostic artifact, hashes prompts, redacts completions, never promotes diagnostic capability |
| Schema `hawking.gravity.deepseek_v4.hcli_live_suite.v1` | Family-specific live-suite receipt |
| Status `HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY` | Explicit non-promotion claim boundary |
| CLI `deepseek_v4_gravity.py hcli-live-suite` | Operator entrypoint |
| `tools/condense/tests/test_deepseek_v4_hcli_live_suite.py` | Contract tests: prompt hashing, artifact identity match, mismatch rejection |
| `tools/condense/seal_deepseek_v4_hcli_encoding_contract.py` | Separate encoding/tokenizer admission (not product turns) |

### 2.2 Live evidence already collected (DeepSeek layer-4 diagnostic)

Directory: `workspace/campaign/evidence/hide/deepseek-v4-live.sBqM7r/`  
Aggregate: `hcli-live-suite-receipt-v2.json`  
Related: `deepseek-v4-swarm.NvBh4u/`, `deepseek-v4-write-smoke.Yz13sy/`, `chain-audit-fixed.A28L17/`

| Bible §33 case | Existing evidence file(s) | Observed shape | Promotion claim |
|----------------|---------------------------|----------------|-----------------|
| chat | `normal-turn.json` | `hcli.command.v1` / `run` | none (diagnostic) |
| repo context | `repo-context-turn.json`, `repo-source-ingest.json`, `attached-evidence-turn.json` | `hcli.command.v1` + `hcli.source.context.v1` | none |
| coding | `coding-task-turn.json` | `hcli.command.v1` / `run` | none |
| planner/act/verify | `hcli-agent-planner-act-verify-receipt.json`, `hcli-agent-planner-act-verify-6-receipt.json`, `chain-audit-fixed.A28L17/receipt.json` | `hide.headless.audit.v1` | none; statuses include `step_limit` |
| tool calls | planner receipts / audit event chains | audit/tool events | none |
| structured JSON | `structured-json-turn.json` | `hcli.command.v1` / `run` | none |
| session restart | `recovery-before-turn.json`, `recovery-after-turn.json`, `recovery-session-before.json`, `recovery-session-after.json` | session continuity pair | none |
| endpoint restart | `endpoint-health-after-restart.json`, `capabilities.json` | health + capabilities | none |
| context compaction | `context-compaction-turn.json`, `compaction-source-ingest.json` | source omitted when window full | none |
| read-safe swarm | `deepseek-v4-swarm.NvBh4u/hcli-read-safe-swarm-receipt.json` | `hcli.parallel_analysis_swarm.v1` status `incomplete` | none |
| isolated write-agent | `deepseek-v4-write-smoke.Yz13sy/hcli-isolated-write-smoke-receipt.json` | `hide.headless.audit.v1` status `step_limit` | none |
| continuous batching | `hcli-diagnostic-cpu-benchmark-receipt.json`, `base-tps-gate-receipt.json` | benchmark + `BASE_TRUE_TPS_WITHHELD` | **withheld** |
| search/retrieval | *not yet a first-class product case in the DeepSeek pack* | pending Agent OS retrieval gateway | n/a |
| memory operations | *not yet first-class* | pending Memory OS lane | n/a |
| skill execution | *not yet first-class* | pending Skill Foundry lane | n/a |
| document perception | *partial via source/context ingest* | pending Perception service lane | n/a |

### 2.3 Pattern to preserve (do not throw away)

1. **Privacy:** prompt text and completions never enter the aggregate receipt; only hashes + status + artifact seal claims.
2. **Identity binding:** every evidence file that claims a runtime artifact must match the sealed Gravity manifest `seal_sha256`.
3. **Claim boundary:** live suite seals evidence; it does **not** assert full model, Metal, numeric parity, or base-true TPS.
4. **Separate original JSON:** aggregate is an index; originals remain SHA-bound and inspectable.
5. **Diagnostic honesty:** status strings make non-eligibility explicit (`diagnostic_cpu_only_not_tg_eligible`, `BASE_TRUE_TPS_WITHHELD`).

---

## 3. Family-agnostic harness design

### 3.1 Code scaffold (this lane)

| Path | Role |
|------|------|
| `tools/condense/hcli_product_test_harness.py` | Case catalog (§33), family-neutral receipt packer interface, DeepSeek evidence map, metric envelope |
| `tools/condense/tests/test_hcli_product_test_harness.py` | Catalog integrity + schedule/state cross-links + DeepSeek map coverage (no live endpoint) |
| `workspace/docs/plans/ascension/ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json` | Durable machine-readable case list |

DeepSeek remains the first adapter:

- Keep `hcli_live_suite_receipt()` as the production sealer for DeepSeek diagnostic receipts.
- New family adapters (Qwen 30B / 80B) implement the same evidence shapes (`hcli.command.v1`, `hide.headless.audit.v1`, swarm schema) and call a family-parameterized packer later.
- Do **not** rename or break `hawking.gravity.deepseek_v4.hcli_live_suite.v1` receipts already sealed.

### 3.2 Target family-neutral receipt shape

```text
schema: hawking.gravity.hcli_product_suite.v1
family: qwen3_coder_30b | qwen3_coder_next | deepseek_v4_diagnostic | ...
artifact.seal_sha256: <Gravity seal>
endpoint: capabilities + health summary
cases[]: { id, status, evidence_sha256, prompt_hashes, claim_boundary }
metrics: { verified_tasks_completed_per_hour, ... }
prompt_disclosure.mode: hash_only
claim_boundary: never promotes TG / production without controller cert
```

DeepSeek v1 receipts remain valid historical evidence; the neutral schema is the
**forward** contract for multi-family product gates (schedule step 26).

### 3.3 Case catalog (bible §33)

| id | Required evidence kinds | Pass heuristic (product, not diagnostic smoke) |
|----|-------------------------|-----------------------------------------------|
| `chat` | `hcli.command.v1` run with turn completion | Session turn completes; no policy crash |
| `repo_context` | run + `hcli.source.context.v1` selected sources | Repo/source injection recorded; hashes bind |
| `coding` | run and/or headless audit with code goal | Coding goal executes under sandbox policy |
| `planner_act_verify` | `hide.headless.audit.v1` with plan→act→verify chain | Planner steps + verification events present |
| `tool_calls` | tool-gateway events in audit/tool receipts | Tool selection + result bound in receipt |
| `structured_json` | run whose completion/schema is JSON-constrained | Structured output validates against schema |
| `session_restart` | before/after session pair | Session identity continuity after restart |
| `endpoint_restart` | healthz + capabilities after process restart | Ready + same artifact seal |
| `context_compaction` | source context with omit/truncate policy | Compaction decision recorded (inject/omit) |
| `read_safe_swarm` | `hcli.parallel_analysis_swarm.v1` | Multi-lane read-only; no effectful dispatch |
| `isolated_write_agent` | headless audit with write isolation | Writes confined to isolated workspace |
| `continuous_batching` | benchmark receipt | Batch/scheduler metrics; TPS only if earned |
| `search_retrieval` | retrieval-gateway receipt | Query → evidence pack with latency |
| `memory_ops` | memory-OS receipt | Store/recall/evict with cost ledger |
| `skill_execution` | skill-foundry receipt | Skill install/run/verify chain |
| `document_perception` | perception service receipt | Doc → structured perception without network fetch by default |

### 3.4 Metric envelope

Primary:

```text
verified_tasks_completed_per_hour
  = count(cases with verification_authority = pass within wall window)
    / wall_hours
```

Secondary fields (all optional until instrumented):

```text
per_agent_latency_ms
aggregate_accepted_tps
patch_acceptance_rate
tests_passed
retries
regressions
human_intervention_count
thermal_stability
search_latency_ms
tool_selection_latency_ms
memory_retrieval_cost
planning_failure_rate
```

Agent speed scoring must use the four-stage chain in the header, not raw TPS.

---

## 4. Mapping to programme schedule / completion states

| When | Gate |
|------|------|
| Schedule step **9** (Agent OS foundations) | Scaffold + offline contract tests must exist |
| Schedule step **26** (production retrieval/memory/skills/perception) | Full case catalog runnable on dual-Qwen Option-C |
| Completion state **HCLI_AGENT_PIPELINE_READY** | Controller-certified product suite on production path |
| Completion state **HAWKING_APPLE_PRODUCTION_RELEASE_READY** | Product suite + ladder + authorities sealed |

Companion lanes own subsystem detail (do not block this plan on them):

- Scheduler / planning → `HCLI_AGENT_SCHEDULER_PLAN.md`, `HCLI_PLANNING_DIAGNOSTIC_PLAN.md`
- Retrieval / tools → `HCLI_RETRIEVAL_GATEWAY_PLAN.md`, `HCLI_TOOL_GATEWAY_PLAN.md`
- Memory / skills → `HCLI_MEMORY_OS_PLAN.md`, `HCLI_SKILL_FOUNDRY_PLAN.md`
- Perception / comms → `HCLI_PERCEPTION_SERVICE_PLAN.md`, `HCLI_COMMUNICATION_BUS_PLAN.md`
- Sandbox / verify → `HCLI_EXECUTION_SANDBOX_PLAN.md`, `HCLI_VERIFICATION_AUTHORITY_PLAN.md`
- Option-C / residency → `HCLI_OPTION_C_PLAN.md`, `HCLI_RESIDENCY_MODES_PLAN.md`

---

## 5. CUDA / backend neutrality

Per bible §34, product tests must not hard-code Metal-only success:

- Preserve receipt schema, benchmark contracts, parity/capability suite hooks, scheduler API.
- Apple claims remain Apple-specific.
- A later CUDA backend reuses the same case catalog and metric envelope.

---

## 6. Non-goals (this lane)

- No live Qwen endpoint runs
- No CUDA work
- No frankenstein operator or evidence mutation
- No marking schedule steps or completion states complete
- No promotion of DeepSeek diagnostic smoke to production readiness

---

## 7. Implementation order (future execution, not this scaffold)

1. Keep DeepSeek packer + contracts green (regression anchor).
2. Land catalog JSON + harness module (this lane).
3. Add Qwen adapter once schedule step 11–13 produce a sealed serve artifact.
4. Expand cases for search/memory/skill/perception as those Agent OS plans land.
5. Wire verification authority (bible §22) as the only `pass` issuer for product metrics.
6. Controller certifies `HCLI_AGENT_PIPELINE_READY` only after sealed suite + review.

---

## 8. How to run (today)

Offline contracts only:

```bash
# DeepSeek live-suite receipt contracts (existing)
python -m pytest tools/condense/tests/test_deepseek_v4_hcli_live_suite.py -q

# Family-agnostic catalog / harness contracts (this lane)
python -m pytest tools/condense/tests/test_hcli_product_test_harness.py -q
```

Live suite against a real endpoint is operator-driven (not CI by default) and must
use the privacy packer so prompt material never lands in the aggregate.
