# Chapter 11 — Bleeding-Edge Capabilities and Moonshots

> **Purpose (one line).** Collect, specify, and rank the research-frontier capabilities that compound HIDE's local-plane advantage beyond what any prior chapter fully designs — personalization flywheels, self-improving loops, on-device RL, world models, latent agent communication, learned retrieval, and a suite of AI-native UX primitives — then hand the team a prototype-first shortlist with exact integration points so none of this stays at the level of aspiration.

**Status:** DESIGN — research frontier / prototype phase. Every item in this chapter is staged *after* the shell and all prior chapters' core designs. Tags used throughout: **[RESEARCH-PROVEN]** = published with reproducible results, build risk is implementation; **[SPECULATIVE]** = compelling but requires further validation on HIDE's stack; **[MOONSHOT]** = fundamental research gap, high reward, unknown timeline. Within each, **impact** (1–5) and **prototype feasibility** (1–5) are scored and the §11.10 table aggregates them.

Each section carries at least one concrete Rust schema, function signature, or protocol definition — the chapter's standing commitment that these are engineering plans, not wishes.

---

## Table of contents

1. [§11.1 — Personalization Flywheel (Model↔Harness Co-evolution)](#111--personalization-flywheel-modelharness-co-evolution)
2. [§11.2 — Self-Improving Agent Workflows (DSPy/ADAS/GEPA on-device)](#112--self-improving-agent-workflows-dspyadasgepa-on-device)
3. [§11.3 — Autonomous Benchmark & Eval Generation](#113--autonomous-benchmark--eval-generation)
4. [§11.4 — Latent / Compressed Inter-Agent Communication](#114--latent--compressed-inter-agent-communication)
5. [§11.5 — KV-Cache Handoff Between Agents](#115--kv-cache-handoff-between-agents)
6. [§11.6 — Learned Retrieval (Meta-Learning over the Codebase Index)](#116--learned-retrieval-meta-learning-over-the-codebase-index)
7. [§11.7 — RLEF (Reinforcement Learning from Execution Feedback) On-Device](#117--rlef-reinforcement-learning-from-execution-feedback-on-device)
8. [§11.8 — World Model / Project Simulation](#118--world-model--project-simulation)
9. [§11.9 — AI-Native Environment Design](#119--ai-native-environment-design)
10. [§11.10 — Ranked Table and Prototype-First Shortlist](#1110--ranked-table-and-prototype-first-shortlist)
- [Appendix A — Cross-references and dependency graph](#appendix-a--cross-references-and-dependency-graph)

---

## §11.1 — Personalization Flywheel (Model↔Harness Co-evolution)

Ch.06 §4.10 already defines the capture→curate→train→deploy loop and calls it "the unfair advantage cloud cannot offer." This section deepens the design to the **record schema, the curation pipeline, the three training phases, and the privacy invariant** — making it buildable rather than aspirational.

### 11.1.1 What the harness records

Every agent turn in ch.02 produces structured telemetry as a side-effect of the loop. The flywheel taps this telemetry without modifying the loop. The minimal sufficient record:

```rust
/// Written to `.hide/personal/records/<date>/<ulid>.jsonl` after each task.
/// Never leaves the device. User can inspect, filter, or delete the whole dir.
#[derive(Serialize, Deserialize)]
struct PersonalizationRecord {
    /// Stable identifier for session; groups records from one interactive session.
    session_id:          Ulid,
    /// Microsecond-precision wall-clock of the tool-call observation.
    observed_at_us:      u64,
    /// High-level task class: EditCode | WriteTest | Refactor | ExplainCode |
    /// CommitMsg | Diagnose | Research — inferred from the plan step.
    task_type:           TaskClass,
    /// BLAKE3 of the final system prompt + user message (not the content itself).
    prompt_hash:         [u8; 32],
    /// BLAKE3 of the set of file paths + their sizes at call time.
    context_fingerprint: [u8; 32],
    /// What happened to the suggestion.
    outcome:             Outcome,
    /// The diff HIDE proposed (raw unified diff, secrets already scrubbed).
    diff_proposed:       String,
    /// The diff the user actually kept (empty if outcome == Rejected).
    diff_accepted:       String,
    /// Total wall-clock from plan step start to user accept/reject signal.
    latency_ms:          u32,
    /// Decode throughput of the generation that produced diff_proposed.
    tok_s:               f32,
    /// Free-form reason recorded when outcome == Rejected or Modified
    /// (harvested from undo events, explicit feedback, or inline comment).
    reject_reason:       Option<String>,
    /// Which model role served this request (hero / fast-draft / …).
    model_role:          String,
    /// Active adapter ids at the time of generation, empty = base.
    active_adapters:     Vec<String>,
}

#[derive(Serialize, Deserialize)]
enum Outcome {
    /// User accepted the diff without modification.
    Accepted,
    /// User accepted a manually-edited version of the diff.
    Modified { edit_distance_chars: u32 },
    /// User rejected (Ctrl-Z, explicit dismiss, or replaced entirely with different code).
    Rejected,
    /// Session ended before explicit accept/reject (treat as implicit partial signal).
    Abandoned,
}
```

The record is appended to the JSONL log only *after* the outcome is known — i.e., after the user commits, accepts, or explicitly rejects. The harness listens to ch.01's event log for `diff.accepted`, `diff.rejected`, and `diff.modified` event kinds (already emitted by ch.03's diff tool); no new hook into the model is required for Phase 1.

**Secrets scrubbing** runs on `diff_proposed` and `diff_accepted` before the record is written: the same redaction layer ch.10 §4.8 applies to the event log. API keys, tokens, and patterns matching the keychain-pattern library are replaced with `<REDACTED>` before persistence. The `diff_proposed` is further hashed and checked against a `.hide/personal/.scrub_patterns` blocklist the user can extend.

### 11.1.2 Data curation pipeline

Raw records are noisy: the user accepted a suggestion that was already in the file (false positive), or rejected because the model was fast but the user wanted a different approach (not a quality signal). The curate pass runs nightly as a background daemon (`hide-personalize curate`) and produces a clean SFT dataset:

```
~/.hawking/personal/
  records/          # raw JSONL, append-only, user-deletable
  dataset/          # curated, versioned SFT records
    v001/
      train.jsonl   # (prompt_context, target_diff) pairs for SFT
      pref.jsonl    # (prompt_context, chosen_diff, rejected_diff) pairs for DPO
      held_out.jsonl
  adapters/         # trained LoRA checkpoints
    personal_v001.safetensors
  eval/             # accept-rate measurement on held_out
```

Curation rules (applied in order):

1. **Keep `Accepted` records as positive SFT examples.** Input = the context manifest (reconstructed from `context_fingerprint` and the event log replay), output = `diff_accepted`.
2. **Discard records where `diff_accepted` is empty or `edit_distance_chars` > 0.8 × len(diff_proposed)** — the user rewrote so much that the model's proposal was noise.
3. **Pair `Accepted` vs `Rejected` records on the same `prompt_hash`** (same prompt, different rollouts — possible when best-of-N is on) → DPO preference pair.
4. **Discard records where latency_ms is an outlier** (p95 × 3) — timeout artifacts.
5. **Cap dataset at 10 000 positive records** (recency-weighted) to keep LoRA fine-tune short.
6. **Deduplicate** on `diff_accepted` content hash — don't overfit to frequently-accepted boilerplate.

### 11.1.3 Three training phases

**Phase 1 — Silent collection (ship immediately).** The record writer is pure shell: it taps existing ch.02 telemetry events and writes JSONL. No model change. No user-visible change. The user can find and inspect every record in `.hide/personal/records/`. This runs from day one.

**Phase 2 — Offline LoRA (after Phase 1 has ≥ 500 accepted records).** *Hawking Condense* (`tools/condense/doctor_qat.py` extended) runs the LoRA fine-tune on the curated dataset overnight. The recipe: LoRA rank 16, target modules `q_proj|v_proj|o_proj|gate_proj|up_proj|down_proj`, 3 epochs, learning rate 3e-4 with cosine decay, batch size 4, gradient checkpointing on. Takes ~2–4 hours on M-series for a 7B. The trained LoRA is written to `~/.hawking/personal/adapters/personal_v<N>.safetensors`. The adapter ships to the role registry as `personal`, gated on the accept-rate oracle (below).

**Phase 3 — Re-bake into `.tq` (after Phase 2 succeeds consistently).** *Hawking Condense* folds the personal LoRA into a new `.tq` checkpoint via QAT-style merge: the LoRA delta is applied to the full-precision weights before the trellis quantization step. The resulting model is the user's private condensed artifact — smaller on disk than a separate LoRA + base, faster to serve, and personalized at the weight level. This is the heavy option; it requires a re-condense run (~4–8 hours), so it runs as an optional weekend job.

### 11.1.4 Deploy and the accept-rate gate

The personal adapter is deployed only if it passes the **accept-rate oracle**: run it against `held_out.jsonl` (50 records withheld during curation) and measure whether the adapter's outputs score ≥ 5% higher accept rate than the base model on the same prompts. If not, the old adapter stays. The gate prevents the flywheel from amplifying a transient bad pattern.

Once gated, the router (ch.06 §4.4) automatically includes `personal_v<N>` in `AdapterSelection.adapters` for every `EditCode`, `WriteTest`, and `Refactor` request, blended at `scale: 0.6` alongside any language adapter. The version is recorded in the context manifest (ch.06 Appendix A.5), so replay is deterministic.

### 11.1.5 Privacy invariant

The invariant is structural, not policy: *training data is written only to `~/.hawking/personal/`; no network egress path exists from the curate or train pipeline; the `.tq` artifacts are local files.* Ch.10 §4.7's egress monitor treats any outbound traffic from the `hide-personalize` process as a security event. The user-facing privacy model is simple: "your model learns from your code; your code never leaves your machine."

**[RESEARCH-PROVEN / low–med build for Phase 1–2]** **[SHELL-TODAY for capture.]**

---

## §11.2 — Self-Improving Agent Workflows (DSPy/ADAS/GEPA on-device)

### 11.2.1 The problem with hand-written prompts

Every prompt in HIDE's agent loop — the Planner's task-decomposition prompt, the Verifier's failure-analysis prompt, the Skill Library's retrieval query — was written by a human and is frozen at ship time. DSPy [Khattab et al. 2023] showed that treating each prompt as an optimizable module signature yields measurable, automatic improvements. ADAS [Chen et al. 2024] showed that a meta-agent can design *new agent topologies* that outperform hand-crafted ones. On-device, where evaluation is free and private, there is no reason to leave this leverage unclaimed.

### 11.2.2 Prompt modules as typed signatures

Every agent prompt in HIDE is refactored into a **PromptModule**: a name, a typed signature (inputs → outputs), a current prompt template, and a version history.

```rust
/// Stored under `.hide/skill-lib/prompts/<module_name>/`.
#[derive(Serialize, Deserialize, Clone)]
struct PromptModule {
    name:             String,              // e.g. "planner.decompose"
    schema_version:   u32,
    /// Typed inputs the module receives.
    input_schema:     serde_json::Value,   // JSON Schema
    /// Typed output the module must produce.
    output_schema:    serde_json::Value,
    /// The current active template (a Jinja-style string with {{var}} slots).
    template:         String,
    /// Immutable version history; new optimized templates append here.
    history:          Vec<PromptVersion>,
    /// The metric this module is optimized for (accuracy / accept_rate / latency_ms / …).
    metric:           OptimizationMetric,
    /// Minimum number of eval tasks required before an optimization is accepted.
    min_eval_n:       u16,
}

#[derive(Serialize, Deserialize, Clone)]
struct PromptVersion {
    version:          u32,
    template:         String,
    score:            f64,
    eval_n:           u16,
    promoted_at_us:   u64,
    promoted_by:      PromotedBy,         // Human | AutoDSPy | ADAS
}

#[derive(Serialize, Deserialize, Clone)]
enum PromotedBy { Human, AutoDSPy { optimizer_run_id: Ulid }, ADAS { run_id: Ulid } }
```

### 11.2.3 The optimization loop (DSPy-style, on-device)

The optimizer runs as part of `hawking-eval` (the eval harness from the plan doc). It is triggered by the user ("optimize planner prompts") or on a schedule (weekly, overnight).

The loop for a single module:

1. **Sample mini-eval set.** Pull 10–20 tasks from the local eval harness (§11.3) that exercise the module.
2. **Bootstrap few-shot candidates.** Run the current template on a subset; record (input, output, score) tuples.
3. **Propose candidate templates.** Use the fast-draft model (0.5B, cheap) to paraphrase and mutate the template in N directions (N=8). Each candidate is a `PromptVersion` at this stage.
4. **Evaluate candidates.** Run each candidate against the mini-eval set in parallel (ch.09 fan-out, isolated worktrees). The metric is read from `PromptModule.metric`.
5. **Select and gate.** The winning candidate must beat the current template by ≥ 1% on the metric and ≥ min_eval_n tasks. Human approval is optionally required (`require_human_gate: bool` in module config).
6. **Promote.** Append to `history`, update `template`, emit a `prompt.promoted` event (ch.01 event log). The agent kernel picks up the new template on the next cold-start or hot-reload.

This entire loop runs locally, uses the existing `hawking-eval` environment and ch.09 scheduling, and produces measurable improvements over time without any cloud dependency.

### 11.2.4 ADAS — meta-agent design of new topologies

ADAS runs at a higher level: instead of optimizing a prompt template, it proposes, evaluates, and archives entirely new **agent loop variants** — different Planner→Executor topologies, different search strategies, different verification orderings.

The meta-agent is itself a HIDE agent run, but with a special system prompt that grants it:
- Read access to the existing loop implementations in `hawking-agent/src/`.
- Write access to `.hide/adas/candidates/` (isolated, never the active loop).
- Access to the eval harness to measure candidate performance.

The meta-agent proposes a new topology as a **LoopVariant descriptor**:

```rust
#[derive(Serialize, Deserialize)]
struct LoopVariant {
    id:             Ulid,
    description:    String,
    /// Diff relative to the base loop (a unified diff against hawking-agent/src/).
    patch:          String,
    /// Score on the eval harness before promotion.
    eval_score:     Option<f64>,
    /// Human must approve before this variant becomes active.
    approved:       bool,
    status:         VariantStatus,
}

#[derive(Serialize, Deserialize)]
enum VariantStatus {
    Candidate,
    Evaluating,
    EvalPassed { score: f64 },
    EvalFailed { reason: String },
    Promoted,
    Archived,
}
```

An `EvalPassed` variant moves to `Promoted` only after **explicit human approval** — ADAS never self-promotes. The governance rule is: the meta-agent proposes, the eval harness measures, the human decides. This hard gate prevents runaway self-modification.

### 11.2.5 GEPA — tool selection inner loop

GEPA (Generate→Evaluate→Prune→Adapt) is the per-task, single-session version of ADAS. When the agent is mid-task and uncertain which tool to call next, a GEPA pass:

1. **Generates** 3 candidate next-tool-calls (cheap, fast-draft).
2. **Evaluates** each using the in-process LSP (type-check, lint) or a quick test run.
3. **Prunes** candidates that produce immediate errors.
4. **Adapts** the selection policy for this task class (temporary, session-scoped weight update on the tool-selection prompt).

GEPA requires no new infrastructure — it is a narrow best-of-N + oracle gate running inline in the ACT phase (ch.02 §4.8), triggered when tool-call uncertainty is above threshold (entropy > GEPA_ENTROPY_THRESH, default 1.8 nats).

**Integration with the Skill Library:** when a promoted LoopVariant or PromptModule consistently outperforms on a specific task class, it is registered as a **Skill** (ch.02 §4.11), inheriting the Skill Library's retrieval/versioning machinery. The boundary between "a better prompt for this task" and "a skill for this task" is a single flag (`SkillEntry.source: PromptOpt | Manual | ADAS`).

**[RESEARCH-PROVEN for DSPy / SPECULATIVE for ADAS on-device / SPECULATIVE for GEPA in agent loops]**

---

## §11.3 — Autonomous Benchmark & Eval Generation

### 11.3.1 The thesis-gate problem

The eval harness (`hawking-eval`, referenced throughout the bible) is only as good as its task set. If the task set is a fixed snapshot (SWE-bench-lite, a curated checklist), the thesis gate becomes a fixed number that drifts from reality as the codebase evolves. What HIDE needs is a **living, continuously-growing task set that is automatically mined from the user's actual codebase**.

### 11.3.2 The EvalMiner daemon

`EvalMiner` is a background daemon that watches the project state (via the Living Index daemon, ch.05 §4.9) and the `.hide/log` event log (ch.01) for patterns that imply a good eval task candidate.

```rust
/// Runs as a background tokio task inside the hawking-agent process.
/// Wakes on Living-Index invalidation events.
struct EvalMiner {
    index:        Arc<dyn Index>,          // ch.05 query API
    event_log:    Arc<dyn EventLog>,       // ch.01
    task_store:   Arc<EvalTaskStore>,      // SQLite table in .hide/eval/tasks.db
    config:       EvalMinerConfig,
}

#[derive(Clone)]
struct EvalMinerConfig {
    /// Minimum confidence required to auto-add a task (without human confirmation).
    auto_add_confidence_threshold:  f32,   // default 0.95
    /// Max tasks auto-added per day (rate-limit).
    max_auto_add_per_day:           u32,   // default 20
    /// Patterns that block a file from contributing eval tasks (e.g. generated code).
    file_blocklist:                 Vec<Glob>,
}
```

Mining heuristics (applied on each index invalidation):

| Heuristic | Oracle type | Confidence |
|---|---|---|
| Function has no test file linkage (ch.05 §4.4 test-graph) | Test pass/fail | 0.90 — high; easy to determine |
| TODO/FIXME comment with an associated failing CI check | Build or test | 0.95 if CI state is available |
| Spec file exists but its assertions fail on current HEAD | Test pass/fail | 0.99 — machine-determinable |
| Deprecation annotation with no migration in the call graph | Type-check | 0.85 |
| A recently-broken test (event log: test was green, now red) | Test pass/fail | 0.99 |
| A function whose cyclomatic complexity crossed a threshold | Lint | 0.80 — subjective; needs human |

For each mined candidate:

```rust
#[derive(Serialize, Deserialize)]
struct EvalTaskCandidate {
    id:               Ulid,
    task_description: String,             // human-readable what to fix/write
    oracle:           OracleSpec,         // how pass/fail is determined
    source_files:     Vec<PathBuf>,       // affected files
    confidence:       f32,               // 0.0–1.0
    mined_at_us:      u64,
    status:           CandidateStatus,
}

#[derive(Serialize, Deserialize)]
enum OracleSpec {
    BuildGreen,
    TestPass   { test_filter: String },
    TypeCheck  { paths: Vec<PathBuf> },
    LintClean  { rules: Vec<String> },
    Custom     { command: String, expected_exit: i32 },
}

#[derive(Serialize, Deserialize)]
enum CandidateStatus {
    PendingHuman,
    AutoAdded,
    HumanApproved,
    HumanRejected { reason: String },
    Active,
    Retired { retired_at_us: u64 },
}
```

Candidates with `confidence ≥ auto_add_confidence_threshold` are added directly to the active task set (no human gate needed — the oracle is machine-determinable). Lower-confidence candidates surface in the HIDE UI as a "suggested eval tasks" panel where the user confirms or rejects with one click.

### 11.3.3 SWE-bench adapter and bootstrapping

The task store is bootstrapped with a **SWE-bench-lite adapter**: a one-time import that reads SWE-bench-lite's JSON task format and converts each into an `EvalTaskCandidate` with `oracle: BuildGreen` or `oracle: TestPass`. This gives ~300 seed tasks on day one. The `EvalMiner` then grows the set organically from the user's codebase, so the harness compounds in relevance over time.

### 11.3.4 `hawking-eval --live` mode

```
hawking-eval --live [--filter task_class=WriteTest] [--max-tasks 50]
```

In live mode, `hawking-eval` streams results against the current active task set, updating as new tasks are auto-added. Each result is a structured event (`eval.task.result { task_id, outcome, latency_ms, model_role, adapter }`) in the ch.01 event log, so the UI can render a live thesis-gate dashboard.

The live mode also detects **regression** automatically: if a task that was passing on the last run now fails, it surfaces a `eval.regression.detected` event (severity=high) that blocks an ongoing swarm run from merging (ch.09 §4.4 oracle gate).

**[SPECULATIVE for auto-generation / RESEARCH-PROVEN for oracle-gated harnesses]**

---

## §11.4 — Latent / Compressed Inter-Agent Communication

### 11.4.1 The token-waste problem

In every current multi-agent system (OpenAI multi-agent, LangGraph, AutoGen), agents communicate by passing natural-language summaries: the Planner writes a paragraph, the Executor reads it, re-interprets it, and may re-derive what the Planner already computed. This loses precision (a summary always compresses away distinctions), wastes tokens (the summary is re-tokenized and re-prefilled at every hop), and introduces drift (the Executor's interpretation ≠ the Planner's intent). Because HIDE controls both sender and receiver, this chapter is the one place in the IDE landscape where a better protocol is structurally available.

### 11.4.2 Three approaches, ranked by build readiness

#### Approach 1 — Structured JSON handoffs (build first, [PROVEN])

The minimum viable upgrade: replace natural-language inter-agent messages with typed schemas. Already partially in place via ch.09's `PlanStepResult` and ch.02's `SubagentReturn` — this approach canonicalizes and extends them.

```rust
/// The canonical handoff envelope between any two named agents in HIDE.
/// Compatible with ch.01's Event envelope (wraps as payload in event kind "agent.handoff").
#[derive(Serialize, Deserialize)]
struct AgentHandoff {
    schema_version: u32,                   // bumped on breaking field changes
    from:           AgentId,
    to:             AgentId,
    /// Monotonically increasing within a run; used for in-order delivery check.
    seq:            u64,
    /// Typed, schema-validated payload. The schema is determined by (from, to) pair
    /// registered in HandoffRegistry at startup.
    payload:        serde_json::Value,
    /// Optional KV-cache handle (§11.5) for the receiver to restore shared prefix.
    kv_ref:         Option<KvHandle>,
    /// Optional compressed thought vector (§11.4.2 Approach 3, future).
    thought_vec:    Option<ThoughtVectorRef>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
struct AgentId(String);    // e.g. "planner:0", "worker:3", "merger:0"

#[derive(Serialize, Deserialize, Clone)]
struct KvHandle { store_key: String, fork_seq: u64 }  // ch.05 KvStore key + position

#[derive(Serialize, Deserialize, Clone)]
struct ThoughtVectorRef { blob_key: [u8; 32], dim: u32 }  // CAS key + vector dim
```

The **HandoffRegistry** maps `(AgentId::kind_from, AgentId::kind_to)` to a JSON Schema; the handoff is schema-validated before dispatch, guaranteeing the receiver never sees malformed input. This alone eliminates an entire class of inter-agent coordination failure — the Planner can no longer accidentally write "See above" as a handoff; the schema requires a `Vec<PlanStep>` and a `Vec<ContextRef>`.

**Cost reduction (concrete).** A natural-language Planner→Executor summary for a mid-size task is typically 800–1200 tokens. A `PlanStepResult` JSON payload for the same task is 200–350 tokens. At 1000-token savings per hop × 16 workers in a fan-out swarm × 3 hops per run = **48 000 fewer tokens per swarm run**, translating to ~4–6 fewer prefill seconds per swarm run at 7B scale.

#### Approach 2 — KV-cache handoff (medium-term, [SPECULATIVE])

The Planner has already prefilled its context — the system prompt, the codebase context, the user intent — and produced planning tokens. The Executor, starting fresh, re-prefills that same shared prefix from scratch. With KV-cache handoff, the Executor *restores the Planner's KV state* and decodes only from the fork point.

The mechanism is `KvStore::checkpoint(key, seq_len)` → `KvStore::restore(key)` (ch.04 §4.5 already specifies the `KvStore` trait). The Planner calls `checkpoint("planner-0-fork", prefix_len)` at the end of planning; the `AgentHandoff` carries `kv_ref: Some(KvHandle { store_key: "planner-0-fork", fork_seq: prefix_len })`. The Executor calls `restore` before its first generation step, skipping the entire prefill of the shared prefix.

**Cost model.** Prefill scales O(seq_len²) in attention. At a 4k shared prefix (a typical codebase context + system prompt), the Executor saves ~120ms per run. For 16 workers: 16 × 120ms = 1.92 seconds saved per fan-out invocation — meaningful in interactive swarms where the user is watching. At 8k shared prefix, savings double.

**Gate:** requires KV-handles API (ch.06 Appendix B #5) to be wired in `hawking-serve`. Prototype in the single-model single-worker case first; extend to fan-out once stable.

#### Approach 3 — Compressed thought vectors (moonshot, [MOONSHOT])

A learned compressor converts an agent's final hidden state (the residual stream of the last token before the `</think>` tag) into a fixed-size latent vector. The receiver decompresses it into its own KV prefix, effectively bootstrapping its "memory" of the sender's reasoning without retokenizing or re-generating.

This requires joint training of a compressor `f: hidden_state → R^d` and a decompressor `g: R^d → KV_prefix` on a dataset of (planning trace, execution trace) pairs. It is related to the "compressed chain-of-thought" and "recurrent latent" lines of research [Hao et al. 2024] but has not been demonstrated at the precision required for code agents. Estimated timeline: 6–12 months of Condense-side research after Approach 1 and 2 are shipping.

**[PROVEN for Approach 1 / SPECULATIVE for Approach 2 / MOONSHOT for Approach 3]**

---

## §11.5 — KV-Cache Handoff Between Agents

### 11.5.1 Motivation

Every agent in a HIDE swarm receives the same codebase context block at the start of its run: the system prompt (≈500 tokens), the repo-map (≈1000 tokens), the plan context (≈500 tokens), the tool schema list (≈300 tokens). At 2300 shared tokens × 7B model, each prefill costs ~40ms on M3 Pro (measured). For a 16-agent swarm, that is 640ms of pure redundant prefill — before any agent does any useful work.

The KV-cache handoff eliminates this by broadcasting the shared prefix's KV state to all workers before they start.

### 11.5.2 KvShareGroup protocol

```rust
/// Registered with the Governor (ch.09 §4.6) when a fan-out swarm is launched.
#[derive(Serialize, Deserialize, Clone)]
struct KvShareGroup {
    /// Key under which the prefix KV state is stored in the KvStore.
    prefix_key:   KvKey,
    /// Agents that share this prefix.
    members:      Vec<AgentId>,
    /// Token position at which each member diverges (all members share 0..fork_seq).
    fork_seq:     u64,
    /// Expiry: the KV state is evicted after this many ms of non-use.
    ttl_ms:       u64,
}

/// Extension to GenerateRequest (ch.06 §4.1) to accept a KV seed.
/// Sent to hawking-serve's /v1/hawking/generate.
/// hawking-serve restores the KV state before generation starts.
struct GenerateRequest {
    // ... existing fields ...
    /// If Some, skip prefill of tokens 0..kv_seed.fork_seq; restore from store.
    kv_seed: Option<KvHandle>,
}
```

The Governor's **swarm-launch path** (ch.09 §4.6):

1. **Planner completes.** Its KV state is checkpointed: `kv_store.checkpoint(prefix_key, fork_seq)`.
2. **KvShareGroup registered.** All N workers inherit the same `prefix_key`.
3. **Workers launched.** Each `GenerateRequest` carries `kv_seed: Some(KvHandle { store_key: prefix_key, fork_seq })`. `hawking-serve` routes each request to the `copy_kv_prefix_to_slot` method already in `hawking-core/src/engine.rs` (verified in-tree), which copies the stored KV blocks into the request's decode slot.
4. **Workers diverge.** Each worker's first generated token is unique; KV state diverges from `fork_seq` onward. The shared blocks are reference-counted; eviction happens only when all members have completed.

### 11.5.3 Broadcast prefix pattern in practice

```
Planner (fork_seq=2300)
  ├── Worker-00: sees [0..2300] from KV store, generates from token 2301
  ├── Worker-01: same
  ├── ...
  └── Worker-15: same

RAM cost: one copy of the 2300-token KV block in the store + N incremental slots.
vs. today: N copies × 2300 prefill = N × the cost.
At N=16: 16× prefill cost → ~1× (the store copy) + 16 × incremental.
```

The `copy_kv_prefix_to_slot` in-tree function already implements the copy primitive. The missing pieces are: (a) the `KvStore::checkpoint` / `restore` API in `hawking-serve`'s HTTP surface (not yet exposed), and (b) the Governor's `KvShareGroup` registration and lifecycle. Neither is a fundamental research problem — both are plumbing.

**Prototype target:** single model, 2-worker case, measure the prefill savings directly. The in-tree `copy_kv_prefix_to_slot` function is the integration point.

**Gate: KV-handles API (ch.06 Appendix B #5).** Ship after that API lands.

**[PROVEN-IN-PROD conceptually (prefix caching is in-tree) / med build for the handoff API]**

---

## §11.6 — Learned Retrieval (Meta-Learning over the Codebase Index)

### 11.6.1 The fixed-policy problem

Ch.05's Living Index offers several retrieval strategies: BM25 (lexical), embedding cosine (semantic), call-graph proximity, test-file linkage, recency, PageRank repo-map weight. Today the `IndexQuery` uses a static priority: embedding + BM25 re-rank, with call-graph fallback for specific tool calls like `find_callers`. This works reasonably well as a default, but it is not optimal for any specific query type or any specific codebase.

The insight: **the right retrieval policy for "find the function that causes this error message" is different from the policy for "find code similar to this pattern"** — and both are different from "what module should I read before editing this file." A learned router that adapts per query type and per codebase produces systematically better context, which produces systematically better agent outputs.

### 11.6.2 The training signal

After each completed agent task, HIDE knows which retrieved context spans actually appeared in the final accepted diff (or were cited in the plan reasoning). A span that appeared in the agent's final output was *useful*; one that was retrieved but never used was *noise*. This is the supervision signal — fully automatic, no human labeling.

```rust
/// Written to .hide/retrieval-log/<date>/<ulid>.jsonl after each task.
#[derive(Serialize, Deserialize)]
struct RetrievalOutcomeRecord {
    task_id:             Ulid,
    query_text:          String,
    query_type:          QueryType,        // FindCallers | FindSimilar | FindCauses | …
    codebase_fingerprint:[u8; 32],         // BLAKE3 of repo-map at query time
    /// Strategies tried and the spans each returned.
    retrievals:          Vec<RetrievalAttempt>,
    /// The set of file:line ranges that actually appeared in the final diff.
    used_spans:          Vec<Span>,
}

#[derive(Serialize, Deserialize)]
struct RetrievalAttempt {
    strategy:   RetrievalStrategy,
    spans:      Vec<Span>,
    rank:       u32,    // position this strategy's results appeared in the final context
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
enum RetrievalStrategy { Bm25, EmbeddingCosine, CallGraphProximity, TestFileLinkage, Recency, PageRankWeight }

#[derive(Serialize, Deserialize, Clone)]
struct QueryType { kind: String, detected_language: Option<String> }
```

### 11.6.3 The meta-retrieval classifier

A tiny classifier (50MB, MobileBERT or a custom 4-layer transformer trained on `RetrievalOutcomeRecord` data) learns `(query_type, codebase_fingerprint_bucket, retrieval_strategy) → P(used_in_output)`. At query time, the classifier picks the strategy with the highest predicted usefulness probability.

```rust
/// Added to the ch.05 Index trait as an optional field.
/// Starts as None (uniform random, exploratory); fills in after ~200 tasks.
pub trait Index: Send + Sync {
    // ... existing methods ...
    fn meta_router(&self) -> Option<&dyn MetaRouter>;
}

pub trait MetaRouter: Send + Sync {
    /// Given a query text and type, return the retrieval strategy to use first.
    /// Falls back to the static priority ordering if confidence < threshold.
    fn route(&self, query: &str, qtype: &QueryType, confidence_min: f32) -> RetrievalStrategy;
    /// Update the classifier with a new outcome record (online SGD step).
    fn update(&mut self, record: &RetrievalOutcomeRecord);
}
```

**Online learning:** after each task, `meta_router.update(record)` applies a single SGD gradient step on the classifier. The classifier is tiny (50MB), so each step takes ~10ms on CPU — negligible. No retraining pipeline, no batching, no cloud. The classifier improves monotonically with each completed task.

**Cold start:** seed the classifier with prior weights trained on a synthetic benchmark derived from the codebase-intelligence heuristics: `CallGraphProximity` wins for `FindCallers` queries; `EmbeddingCosine` wins for `FindSimilar` queries; `Bm25` wins for exact-symbol queries. These are the sensible defaults; the learned classifier refines them per user's codebase.

**Exploration policy:** ε-greedy (ε=0.1 during the first 500 tasks, decaying to 0.02). The classifier occasionally tries a non-optimal strategy to keep its signal fresh. This is invisible to the user — the retrieval quality is still good; the suboptimal strategy is tried infrequently and its outcome is recorded.

**[SPECULATIVE (novel combination) / med build / high impact]**

---

## §11.7 — RLEF (Reinforcement Learning from Execution Feedback) On-Device

### 11.7.1 Why RLEF is locally tractable

Reinforcement Learning from Human Feedback (RLHF) requires human raters. Reinforcement Learning from AI Feedback (RLAIF) requires a frontier reward model. RLEF requires neither: the reward comes from the *execution environment* — build pass/fail, test pass/fail, type-check, linter — which is deterministic, free, and already integrated into HIDE's oracle system (ch.02 §4.6). On-device, the environment is always running. The reward is always available. The model is small enough to compute a gradient step on a Mac. This combination makes on-device RLEF feasible for the first time.

### 11.7.2 The RLEF training loop

```
TASK SAMPLING
  └─ Draw task from eval harness (§11.3) or recent failure log
GENERATION
  └─ Agent run produces a diff attempt (up to max_attempts=4)
ORACLE EVALUATION
  └─ Build / test / typecheck / lint oracle (ch.02 §4.6) → reward signal
REWARD SHAPING
  └─ +1.0 (all oracles green) | -1.0 (build breaks) | -0.5 (test fail) |
     -0.25 (lint-only fail) | -0.75 (timeout)
PPO/GRPO STEP
  └─ Compute policy gradient on the LoRA adapter using the reward signal
CHECKPOINT
  └─ Write adapter to ~/.hawking/rlef/adapters/<run_id>_step<N>.safetensors
PPL GATE
  └─ If held-out PPL degraded > 0.5 nats → rollback to prior checkpoint
MERGE SCHEDULE
  └─ Every 100 gradient steps → merge adapter candidate into role registry
       (gated on accept-rate oracle, §11.1.4)
```

The **training dataset** at any given point: 100 tasks × 4 attempts each = 400 `(context, response, reward)` tuples per overnight run. At 7B scale with LoRA rank 16, a PPO gradient step on a 400-sample batch takes ~8–12 minutes on M3 Pro (estimated from the existing `doctor_qat.py` QAT timings — similar compute profile). A full overnight run of 100 gradient steps = ~20 hours: fits a weekend run without blocking interactive use.

### 11.7.3 GRPO as the preferred algorithm

GRPO (Group Relative Policy Optimization, [Shao et al. 2024]) is preferred over PPO for on-device RLEF because:

1. **No value model.** PPO requires training a separate value function (doubles memory). GRPO uses group-relative rewards (compare within a group of samples on the same prompt). On a memory-constrained laptop, removing the value model is a significant footprint win.
2. **Simpler implementation.** GRPO's loss is a single grouped KL-constrained objective; PPO's clipping and advantage estimation add implementation complexity.
3. **Already demonstrated on code tasks.** DeepSeek-R1-Zero used GRPO on code reasoning tasks and produced a measurable improvement in task-completion rate.

The GRPO loss (simplified):

```
L_GRPO = -E[( r_i - mean(r_g) ) / std(r_g) × log π(a_i | s_i)] + β × KL(π || π_ref)
```

where `r_i` is the execution reward for attempt `i`, `r_g` is the reward group for attempts on the same task, `π_ref` is the base model (frozen reference), and `β` is the KL penalty (default 0.02).

### 11.7.4 The `hawking-rlef` daemon

```rust
/// Background daemon; spawned by the Governor on an explicit user opt-in.
/// Writes progress to .hide/rlef/runs/<run_id>/
struct RlefDaemon {
    eval_env:    Arc<EvalEnvironment>,      // §11.3 eval harness
    model_role:  String,                    // "hero" — the role being trained
    adapter_id:  String,                    // current active LoRA to fine-tune
    config:      RlefConfig,
    ppl_baseline:f64,                       // held-out PPL before this run
}

#[derive(Serialize, Deserialize)]
struct RlefConfig {
    tasks_per_batch:   u32,    // default 20
    attempts_per_task: u32,    // default 4
    max_grad_steps:    u32,    // default 100 per overnight run
    ppl_rollback_nats: f64,    // default 0.5
    lora_rank:         u32,    // default 16
    learning_rate:     f64,    // default 1e-5
    kl_penalty:        f64,    // default 0.02
    grpo_group_size:   u32,    // must equal attempts_per_task
    reward_shape:      RewardConfig,
}

#[derive(Serialize, Deserialize)]
struct RewardConfig {
    build_pass:   f32,    // +1.0
    test_pass:    f32,    // +1.0
    build_fail:   f32,    // -1.0
    test_fail:    f32,    // -0.5
    lint_only:    f32,    // -0.25
    timeout:      f32,    // -0.75
}
```

The daemon is off by default; enabled via a user opt-in in Settings. When running, it uses only idle GPU time (the Governor (ch.09 §4.6) allocates it at the lowest priority, preemptible by any interactive request). It logs progress to `.hide/rlef/runs/<run_id>/log.jsonl` and surfaces a "RLEF training — step N/100" badge in the status bar (§11.9).

**PPL gate implementation:**

```rust
fn ppl_gate(adapter: &Path, held_out: &[TrainExample], baseline: f64, threshold: f64) -> bool {
    let current_ppl = compute_ppl(adapter, held_out);  // forward pass, no grad
    current_ppl <= baseline + threshold  // true = keep, false = rollback
}
```

If the gate fails, the adapter checkpoint is moved to `.hide/rlef/archived/` (not deleted — the user can inspect it) and the prior adapter is restored. A `rlef.ppl_rollback` event is emitted to the ch.01 log.

**[RESEARCH-PROVEN at scale / SPECULATIVE on-device at 7B / high impact / high build]**

---

## §11.8 — World Model / Project Simulation

### 11.8.1 The idea

Before spending GPU tokens executing a multi-step plan, predict whether the plan will succeed. In robotics, a world model is a learned `f(state, action) → next_state` that lets the robot reason about outcomes before touching the physical environment. In HIDE, the "world" is the project state (file hashes, test results, type errors, lint warnings, dependency graph) and the "actions" are agent tool calls (edit_file, run_build, run_tests). The project IS the simulator.

### 11.8.2 Tier 1 — Lightweight static simulation (build first, [RESEARCH-PROVEN])

Before running `edit_file(path, diff)`, HIDE can **statically predict** the post-edit type-check and lint result using the in-process LSP and tree-sitter without touching the filesystem or running the compiler. This is the simulation tier that is already mostly built — it is just not wired into the planning phase.

```rust
/// Called by the Governor (ch.02 §4.3) before committing to a plan step that
/// involves edit_file. Returns a prediction before GPU time is spent on generation.
pub trait StaticSimulator: Send + Sync {
    /// Given a proposed unified diff, predict whether applying it would
    /// (a) typecheck clean, (b) pass linter, (c) not break the symbol graph.
    /// Uses in-process LSP + tree-sitter incremental parse — no subprocess.
    fn predict_edit(
        &self,
        path: &Path,
        diff: &str,
        context: &ProjectSnapshot,
    ) -> SimulationResult;
}

#[derive(Serialize, Deserialize, Clone)]
struct SimulationResult {
    /// The simulator's best guess at the edit's outcome.
    predicted_outcome: PredictedOutcome,
    /// Confidence that the prediction is correct (0.0–1.0).
    confidence:        f32,
    /// Specific issues detected (if any).
    issues:            Vec<SimulatedIssue>,
}

#[derive(Serialize, Deserialize, Clone)]
enum PredictedOutcome {
    LikelyClean,
    PredictedTypeError { description: String },
    PredictedLintViolation { rules: Vec<String> },
    PredictedBuildBreak { module: String },
    Unknown,  // simulator cannot reason about this edit
}

#[derive(Serialize, Deserialize, Clone)]
struct SimulatedIssue { kind: String, file: PathBuf, line: u32, message: String }
```

**How it works:** apply the diff to an in-memory copy of the parse tree (tree-sitter's incremental edit API), then run the headless LSP diagnostic request on the modified buffer. This takes ~5–20ms per edit, far cheaper than a real compilation cycle (~500ms–5s). If the simulation predicts a type error, the Governor flags the plan step for re-planning before the agent even starts generating the edit.

**Integration point:** `Governor::pre_simulate_step(step: &PlanStep, snapshot: &ProjectSnapshot) -> Option<SimulationResult>` — called in the `SELECT_STEP → ACT` transition (ch.02 §4.2 state machine). If `predicted_outcome != LikelyClean && confidence > 0.7`, the Governor emits a `sim.predicted_failure` event and re-enters `PLAN` state with the simulation result as additional context.

### 11.8.3 Tier 2 — Dynamic project simulation (moonshot, [MOONSHOT])

A learned forward model `g(state, action) → next_state` trained on the user's git history. State = `(file_content_hashes: HashMap<PathBuf, [u8;32]>, test_results: HashMap<String, TestOutcome>, lint_errors: Vec<LintError>, type_errors: Vec<TypeError>)`. Action = a unified diff. The model learns: "when I apply a diff like this to a codebase in this state, the new state tends to look like that."

This is the level of Dreamer V3 applied to software projects — compelling but requires a large training corpus (the user's full git history, ideally 1000s of commits with measured pre/post CI results) and a non-trivial learned model architecture. It is a research project in its own right. Timeline estimate: 12–18 months after Tier 1 is validated. Valuable enough to design now so the schema and interface are right when the research matures.

The key design constraint: the `SimulationResult` schema (above) is shared between Tier 1 and Tier 2. The `StaticSimulator` trait implementation switches from the tree-sitter/LSP backend to the learned model backend transparently. The Governor never knows which tier is active.

**[RESEARCH-PROVEN for Tier 1 / MOONSHOT for Tier 2]**

---

## §11.9 — AI-Native Environment Design

This section specifies UX and system primitives that only make sense inside a process-owned AI IDE — features that require in-process access to logits, event logs, GPU state, and OS APIs, none of which a cloud-backed IDE can expose.

### 11.9.1 Inline probability display (logit heat-map)

Monaco decorators can apply color spans to any range of text. With logprob readback from the runtime (ch.06 Appendix B #1), HIDE can color each generated token by its entropy: low-entropy tokens (the model was confident) = green underline; medium-entropy = amber; high-entropy (uncertain, possible hallucination) = red underline.

The display is opt-in (a toggle in the editor toolbar, default off). When enabled, the runtime's `want_logprobs: true` is added to every `GenerateRequest`, and the `top_k` logprobs (k=5) are stored alongside the token stream in the event log. The decorator is drawn on the Monaco diff view, not the main editor, so it does not interfere with the user's own code.

**Integration point:** `DiffView.decorate_with_logprobs(spans: Vec<LogprobSpan>)` in the Tauri front-end, fed by a `token.logprob` sub-event in the ch.01 stream. This requires the runtime-side logprob API (ch.06 Appendix B #1) and is **[RUNTIME-SIDE — LATER]** until that lands.

### 11.9.2 Live context budget bar

The Context Compiler (ch.04 §4.2) already produces a **ContextManifest** listing every source and its token count. The budget bar is a Monaco gutter widget that renders the manifest as a live stacked bar chart: each colored segment is a context source (system prompt / codebase / retrieved code / plan / tool output / memory / scratchpad) with its token count and percentage of the total window.

The user can **drag the boundary** between segments to reallocate budget in real time — dragging "retrieved code" wider tells the Context Compiler to retrieve more code at the cost of a smaller scratchpad. The reallocation is stored as a per-project preference in `.hide/profiles/` (ch.06 §4.6). This is the literal "lever we can offer people" the product thesis calls out.

**Implementation:** the manifest is already emitted as a `context.compiled` event (ch.04). The React front-end subscribes and renders it as an SVG horizontal bar using existing Monaco decoration APIs. The drag-to-resize is pure front-end state; on drag-end it writes a `ContextBudgetOverride` to the active profile.

### 11.9.3 Agent undo to any checkpoint

Every git-worktree snapshot (ch.03) is addressable. The ch.01 event log records the worktree state at every `diff.committed` event. The IDE timeline (a panel showing the sequence of agent actions and outcomes) lets the user click any prior event → HIDE checks out that worktree state → the editor reflects it.

This is not a simple Ctrl+Z — it is **time-travel** to any named point in the session. The user sees "agent created auth.rs at 14:32 → build failed → agent fixed it at 14:35 → user accepted at 14:37." Clicking "14:32" restores the worktree to before the fix.

**Implementation:** the timeline panel reads the ch.01 event log projection (a SQLite view over the append-only log, ch.10 §4.2). Each entry has a `worktree_ref` tag (the git commit SHA of the worktree at that point). Clicking an entry dispatches `agent.restore_to { event_seq: u64 }` to the Tauri host, which calls `git checkout <sha>` in the worktree path.

### 11.9.4 Parallel draft racing

For ambiguous tasks (the Planner's uncertainty is above a threshold, or the user explicitly requests it), the Governor launches 2–3 parallel agent runs on different plan strategies (ch.09 fan-out). The results are rendered side-by-side in Monaco's split diff view. The user picks one or merges manually. This is the "parallel draft" feature that cloud IDEs cannot offer — each draft would cost money at cloud rates; at HIDE rates, each draft costs a few seconds of local GPU time.

The UX: a "Racing drafts (3)" badge appears in the status bar. When the first draft completes, it is shown immediately (no wait for slower drafts). Each draft panel has an "Adopt" button (closes others, applies the diff) and a "Merge into this" button (opens a three-way merge editor).

**Implementation:** standard ch.09 fan-out with `n_workers=3`, `topology=BestOfN`. The fan-out already produces parallel SSE streams; the front-end arranges them in split view using Monaco's `createEditorDiffWidget`. No new Rust required — this is a UI composition of existing swarm + diff primitives.

### 11.9.5 Thermal-aware auto-throttle

The Governor (ch.09 §4.6) already monitors RAM and thermal state. When the chip hits ≥85°C (read via macOS `IOKit` power management APIs, already used in ch.06 §4.11), the Governor:

1. Reduces `max_batch_size` by 50% for the next 60 seconds.
2. Cancels pending fan-out requests above the reduced batch size (returns them to queue head).
3. Emits a `governor.thermal_throttle { temp_c: f32, action: ReducedBatch }` event.
4. Displays a "Cooling down (82°C)" badge in the status bar.

The badge disappears when temperature drops below 78°C and batch size is restored. This is a **user-delightful** behavior — the IDE explicitly communicates that it is protecting the machine, which builds trust in sustained use. Cloud IDEs cannot offer this because they have no visibility into the user's hardware thermal state.

### 11.9.6 Offline-first mode indicator

HIDE functions with zero internet at all times. The only operation requiring network is initial model download. An "Offline" indicator in the status bar shows:
- Green dot: all capabilities active (model loaded, embedder ready).
- Amber dot: model not loaded (needs download), degraded mode active.
- No network icon: no network needed — always shown as a feature, not a warning.

When a user first runs HIDE without a network connection, the app opens fully functional with a cached model. The initial model download (if the model is not yet cached) is the only gated operation, and it surfaces an explicit download prompt rather than a mysterious failure.

### 11.9.7 Explain-this-diff

Right-click any diff hunk in the IDE → "Explain this change" → HIDE opens a narrow panel showing the agent's reasoning for this specific change, grounded in the retrieved context that produced it.

**Implementation:** the ch.01 event log records `reasoning.step { diff_hunk_id, retrieved_spans, agent_thought }` for every edit. The `diff_hunk_id` is the BLAKE3 of the hunk header + first changed line. On right-click, the front-end looks up the hunk's id in the event log and renders the stored reasoning. No new generation is needed — the reasoning is already recorded.

This is only possible because HIDE stores every reasoning step locally. A cloud IDE could offer it, but only by paying to re-generate the explanation at query time.

### 11.9.8 Time-travel replay

The ch.01 event log is append-only and supports deterministic replay (`replay_to_seq(seq)` in the event store). In time-travel replay mode, the user scrubs a timeline slider and the entire IDE state — editor content, agent status, context bar, diff panel — reconstructs to exactly what it was at any prior `seq`. This is the "show me what the agent was thinking when it made this mistake" feature.

**Implementation:** `EventLog::replay_to_seq(seq: u64, sink: &mut dyn FnMut(Event))` streams the log subset to a replay projector that rebuilds derived state in a read-only sandbox (no side effects fired, per ch.01 T3). The front-end receives replayed events over the same `ipc::Channel<T>` it uses for live events; the UI renders them identically. A `REPLAY` badge in the status bar prevents confusion with live state.

**[PROVEN for most primitives / med build for the UI layers]**

---

## §11.10 — Ranked Table and Prototype-First Shortlist

### 11.10.1 Ranked table

All moonshots in this chapter, scored on a 1–5 scale. Weighted overall score = 0.35 × Impact + 0.35 × Prototype Feasibility + 0.30 × Build Readiness.

| # | Moonshot | Impact | Proto Feasibility | Build Readiness | **Overall** | Key Dependency | Proto Timeline |
|---|---|---|---|---|---|---|---|
| 1 | **§11.1 Phase 1 — Personalization capture** | 5 | 5 | 5 | **5.00** | None — taps existing telemetry | 1–2 weeks |
| 2 | **§11.9.2 — Live context budget bar** | 4 | 5 | 5 | **4.65** | ContextManifest (ch.04, exists) | 1–2 weeks |
| 3 | **§11.4 Approach 1 — Structured JSON handoffs** | 4 | 5 | 5 | **4.65** | None | 1–2 weeks |
| 4 | **§11.3 — EvalMiner + living task set** | 5 | 4 | 4 | **4.35** | Living Index (ch.05, designed) | 2–3 weeks |
| 5 | **§11.9.4 — Parallel draft racing** | 4 | 4 | 5 | **4.30** | ch.09 swarm (designed) | 2–3 weeks |
| 6 | **§11.8 Tier 1 — Static simulation** | 4 | 4 | 4 | **4.00** | Headless LSP (ch.05 §4.9) | 2–4 weeks |
| 7 | **§11.5 — KV-cache handoff** | 4 | 3 | 4 | **3.70** | KV-handles API (ch.06 B#5) | 3–5 weeks |
| 8 | **§11.6 — Learned retrieval router** | 4 | 3 | 3 | **3.35** | ≥200 completed tasks of data | 4–6 weeks |
| 9 | **§11.2 — DSPy prompt optimization** | 4 | 3 | 3 | **3.35** | §11.3 eval harness | 4–6 weeks |
| 10 | **§11.9.7 — Explain-this-diff** | 3 | 4 | 4 | **3.65** | Event log reasoning storage | 2–3 weeks |
| 11 | **§11.9.8 — Time-travel replay** | 3 | 3 | 4 | **3.35** | Ch.01 replay semantics (designed) | 3–5 weeks |
| 12 | **§11.1 Phase 2 — Offline LoRA** | 5 | 2 | 3 | **3.35** | Phase 1 ≥500 records + Condense trainer | 6–10 weeks |
| 13 | **§11.7 — RLEF on-device** | 5 | 2 | 2 | **3.00** | Phase 2 LoRA infra + eval harness | 8–16 weeks |
| 14 | **§11.2 — ADAS meta-agent** | 4 | 2 | 2 | **2.70** | §11.3 + stable eval harness | 6–12 weeks |
| 15 | **§11.4 Approach 2 — KV handoff** | 4 | 2 | 3 | **3.00** | KV-handles API + §11.5 | 6–10 weeks |
| 16 | **§11.9.1 — Inline logit heat-map** | 3 | 3 | 2 | **2.70** | Logprob API (ch.06 B#1) | 3–5 weeks after API |
| 17 | **§11.8 Tier 2 — Dynamic world model** | 5 | 1 | 1 | **2.35** | Large git history + research | 12–18 months |
| 18 | **§11.4 Approach 3 — Thought vectors** | 5 | 1 | 1 | **2.35** | Joint training, novel research | 12–24 months |

### 11.10.2 Prototype-first shortlist

The six items to prototype first — chosen for high impact, high feasibility, no deep dependencies, and meaningful learning value that gates later tiers.

---

#### Prototype A — Personalization capture (§11.1 Phase 1)

**Why first:** zero new infrastructure; directly compounding. Phase 2 and 3 are gated on having records. Every day without Phase 1 running is a day of training data not collected.

**Exactly how to prototype:**
- File to touch: `crates/hawking-agent/src/telemetry.rs` (create if absent; or extend the existing event-fan-out in `loop.rs`).
- Subscribe to ch.01 event kinds `diff.accepted`, `diff.rejected`, `diff.modified` in the loop's post-verify phase.
- For each event, construct a `PersonalizationRecord` (§11.1.1), run the secrets scrubber (`crates/hawking-core/src/redact.rs`), and append to `.hide/personal/records/<date>/<ulid>.jsonl`.
- Add a `hide personalize inspect` CLI subcommand that pretty-prints the last N records.
- **Demo:** run HIDE on a small Rust project, accept 5 diffs, run `hide personalize inspect` — see the structured records with hashes, outcomes, and scrubbed diffs. No model change needed.

---

#### Prototype B — Live context budget bar (§11.9.2)

**Why first:** the ContextManifest already exists in ch.04's design; this is pure front-end. Delivers immediate visible value that demonstrates "local superpowers" to users and stakeholders.

**Exactly how to prototype:**
- File to touch: `src-tauri/src/context.rs` — emit a `context.compiled` Tauri event carrying the `ContextManifest` (§11.9.2 references ch.04's manifest schema).
- File to touch: `src/components/ContextBudgetBar.tsx` (create) — a React SVG component that renders a horizontal stacked bar. Each segment is a `<rect>` colored by source kind, with a tooltip showing token count.
- Subscribe the component to the `context.compiled` event stream via Tauri's `listen()`.
- Wire a drag-resize handler: `onMouseUp` writes a `ContextBudgetOverride { source_kind, new_fraction }` back to Rust via `invoke("update_context_budget", ...)`.
- **Demo:** open HIDE on a large codebase, trigger an agent run, watch the bar update live as the Context Compiler packs the window. Drag "retrieved code" larger and see the next run use more code context.

---

#### Prototype C — Structured JSON handoffs (§11.4 Approach 1)

**Why first:** reduces token waste immediately with no runtime dependency. The `AgentHandoff` schema replaces freetext inter-agent messages in the existing ch.09 fan-out paths.

**Exactly how to prototype:**
- File to touch: `crates/hawking-agent/src/handoff.rs` (create) — define the `AgentHandoff`, `AgentId`, and `HandoffRegistry` structs (§11.4.2).
- File to touch: `crates/hawking-agent/src/subagent.rs` — replace the current `return_summary: String` field in `SubagentReturn` (ch.02 §4.10) with `handoff: AgentHandoff`.
- Implement `HandoffRegistry::validate(from_kind, to_kind, payload)` using the `jsonschema` crate for validation.
- Register the existing `PlanStepResult` and `OracleVerdict` schemas as the Planner→Executor and Executor→Merger handoff schemas.
- **Demo:** run a 3-agent planner→worker→merger chain; inspect the `.hide/log` event stream; verify no freetext summary fields appear, only typed JSON payloads. Measure the token count of the handoff vs the prior freetext summary.

---

#### Prototype D — EvalMiner with two heuristics (§11.3)

**Why first:** gates both §11.2 (DSPy optimization needs an eval set) and §11.7 (RLEF needs an eval environment). Building it early maximizes the task set size before those features land.

**Exactly how to prototype:**
- File to touch: `crates/hawking-agent/src/eval_miner.rs` (create) — implement `EvalMiner` with two heuristics only: (1) "function with no test file linkage" and (2) "spec file exists but assertions fail on HEAD."
- Wire the Living Index query for heuristic 1: `index.test_coverage(symbol)` (ch.05 §4.4 test-graph API) — any function with `coverage: None` in a file with a corresponding `*_test.rs` or `test_*.py` is a candidate.
- Wire heuristic 2: `eval_env.run_oracle(OracleSpec::TestPass { test_filter: "*" })` on the failing spec files.
- Write candidates to `.hide/eval/tasks.db` with `status: PendingHuman`.
- Emit a `eval.candidate.mined { candidate_id, heuristic, confidence }` event.
- **Demo:** run `hide eval mine` on a real project; see a list of "untested functions" and "failing specs" populated in `.hide/eval/tasks.db`; approve two of them; run `hawking-eval --live --max-tasks 2` and watch the thesis-gate score update.

---

#### Prototype E — Parallel draft racing (§11.9.4)

**Why first:** directly demonstrates the "cloud cannot do this" thesis to users. High WOW factor with low build cost — ch.09 fan-out already exists.

**Exactly how to prototype:**
- File to touch: `crates/hawking-agent/src/governor.rs` — add a `ParallelDraftConfig { n_drafts: u32, trigger: Trigger }` field where `Trigger` is either `UserRequest` (explicit) or `HighPlannerUncertainty { entropy_threshold: f32 }`.
- When triggered, launch N=3 fan-out agents on the same task with different plan seeds (`SamplingParams.seed` varied).
- File to touch: `src/components/DraftRacing.tsx` — render 3 side-by-side Monaco diff panels, each subscribed to its own agent's SSE stream. Show a "First ready" badge when the first completes. Implement "Adopt" (apply one diff, cancel others) and "Close others" buttons.
- **Demo:** open HIDE on a task with an ambiguous implementation choice ("add pagination to this list"), trigger parallel drafts, see 3 different approaches in split view, adopt the best one. Time the total UX vs. sequential attempts.

---

#### Prototype F — Static simulation for edit prediction (§11.8 Tier 1)

**Why first:** directly reduces wasted agent iterations. Every false-start (the agent generates an edit, applies it, watches the build fail, re-plans) costs 3–10 seconds. Static simulation catches a subset of those failures before generation.

**Exactly how to prototype:**
- File to touch: `crates/hawking-agent/src/simulator.rs` (create) — implement `StaticSimulator` with a single backend: apply the diff to an in-memory tree-sitter parse tree (`ts-rs` or the existing tree-sitter bindings in ch.05), then request diagnostics from the headless LSP (`tower-lsp` in-process, already designed in ch.05 §4.9).
- Wire into `Governor::pre_simulate_step`: call `simulator.predict_edit(path, diff, snapshot)` before the `ACT` state transition. If `predicted_outcome: PredictedTypeError && confidence > 0.7`, emit `sim.predicted_failure` and re-enter `PLAN`.
- Add a `SimulatedIssue` display to the plan-tree UI (ch.07 plan panel): a ⚠ badge on plan steps with predicted failures.
- **Demo:** ask HIDE to add a field to a Rust struct without updating all match arms. Watch the simulator flag the predicted exhaustive-match error before the edit is generated. See the planner re-route to include the match arm updates.

---

### 11.10.3 Build ordering and sequencing recommendation

```
Week 1–2:  Prototypes A + B + C  (no dependencies, pure shell or front-end)
Week 3–4:  Prototype D (needs Living Index §4.4 test-graph query to be implemented)
Week 5–6:  Prototype E (needs ch.09 fan-out to be stable under 3-way concurrency)
Week 7–10: Prototype F (needs headless LSP from ch.05 §4.9 to be in-process)
Week 10+:  Phase 2 personalization (§11.1) — gated on ≥500 Phase 1 records
Week 12+:  DSPy optimization (§11.2) — gated on §11.3 eval set being large enough
Week 16+:  KV-cache handoff (§11.5) — gated on ch.06 KV-handles API
Month 6+:  RLEF (§11.7) — gated on LoRA serving infrastructure from ch.06 §4.9
```

This ordering ensures every prototype is buildable with what exists at the time it is started, and each one produces a measurable data or infrastructure artifact that enables the next tier.

---

## Appendix A — Cross-references and dependency graph

| This chapter's item | Depends on | Provides to |
|---|---|---|
| §11.1 Phase 1 (capture) | ch.02 telemetry events; ch.01 event log | §11.1 Phase 2 (training data); §11.6 (retrieval signal) |
| §11.1 Phase 2 (LoRA) | Phase 1 ≥500 records; ch.06 §4.9 LoRA serving | §11.7 RLEF (adapter infra); §11.1 Phase 3 |
| §11.2 DSPy | §11.3 eval harness; ch.02 Skill Library | §11.2 ADAS; ch.02 §4.11 (better Skill entries) |
| §11.3 EvalMiner | ch.05 Living Index §4.4; ch.01 event log | §11.2 (optimization target); §11.7 (RL environment) |
| §11.4 Approach 1 | ch.02 §4.10 SubagentReturn; ch.09 fan-out | §11.4 Approach 2; ch.09 all topologies |
| §11.4 Approach 2 | §11.5 KV handoff | §11.4 Approach 3 (future research) |
| §11.5 KvShareGroup | ch.06 Appendix B #5 (KV-handles API) | §11.4 Approach 2 |
| §11.6 MetaRouter | ch.05 §4.11 IndexQuery; §11.1 (outcome records) | ch.05 all retrieval; ch.04 Context Compiler |
| §11.7 RLEF | §11.3 eval harness; §11.1 Phase 2 (LoRA infra) | §11.1 Phase 3 (trained adapter → re-bake) |
| §11.8 Tier 1 | ch.05 headless LSP; ch.02 Governor ACT transition | §11.8 Tier 2 (same interface) |
| §11.9 logit heat-map | ch.06 Appendix B #1 (logprob API) | — |
| §11.9 context bar | ch.04 ContextManifest | — |
| §11.9 parallel drafts | ch.09 fan-out; ch.07 Monaco split view | — |
| §11.9 time-travel | ch.01 replay semantics; ch.03 worktree snapshots | — |
