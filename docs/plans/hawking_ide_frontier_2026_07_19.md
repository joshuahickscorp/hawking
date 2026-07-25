# Hawking IDE frontier dossier

Date: 2026-07-19  
Research cutoff: 2026-07-19  
Repository inspected: branch codex/hawking-mechanics-thermodynamics at 5d1bf1b9  
Scope: Hawking IDE / HIDE, the local-first agent platform that runs beside the Hawking inference runtime  
Status: research and architecture direction, not an implementation claim

## Executive decision

Hawking IDE should be built as a local-first agent operating system whose durable project state survives models, context windows, process restarts, and local/cloud switching. Hawking is its native low-latency inference and execution-state engine. Cloud models are optional capability escalators, not the owners of project memory.

The product objective is lexicographic:

1. capability density;
2. speed;
3. verified performance and quality;
4. usefulness to the person operating it.

That ordering changes the architecture. HIDE should not maximize model size, prompt length, number of agents, or raw tokens per second in isolation. It should maximize verified software-development capability per resident byte, model-visible token, joule or dollar, and minute of user waiting, subject to hard correctness and security floors.

The most important corrections to the older HIDE story are:

- Active context is always finite. A million-token cloud window is large, not infinite.
- Weight compression does not by itself extend a model's trained or positional context window.
- Recurrent state is compressed model state with bounded recall, not a lossless transcript.
- Durable local files, events, artifacts, indexes, and checkpoints are the effectively unbounded layer.
- Retrieval, exact source evidence, compaction, KV reuse, and recurrent execution state are distinct mechanisms and must not be conflated.
- The current repository contains the HIDE frontend and Hawking runtime, but the roughly 50K-line HIDE backend was sealed out of the active workspace on 2026-07-17. The production vertical slice is currently broken.
- The fastest agent is the one that avoids unnecessary inference and tool round trips, not simply the one with the highest decode tokens per second.
- No “fastest” or “best” claim is earned until a reproducible end-to-end HIDE evaluation proves it.

The recommended near-term direction is therefore:

1. recover the minimum HIDE spine and make one real, secure, execution-grounded vertical slice;
2. make the context and prompt format deterministic, cache-stable, exact, and observable;
3. expose Hawking session affinity, prefix reuse, state save/load/fork, real context ceilings, and latency telemetry;
4. add a capability-dense hybrid local coding-model lane, with Qwen3-Coder-Next as the first architecture-fit target to investigate;
5. remove model turns from deterministic tool orchestration through persistent tools, programmatic calls, safe parallelism, and constrained decoding;
6. build private rotating evaluations and critical-path traces before multiplying agents or adding speculative features.

## 0. Evidence and status language

This dossier deliberately separates facts from proposals.

| Label | Meaning |
|---|---|
| VERIFIED REPO | Confirmed in the active tree, git history, or the verified sealed pack. |
| PRIMARY SOURCE | Reported by an official project, specification, peer-reviewed venue, or the original paper. It may still be workload-specific. |
| INFERENCE | Architecture conclusion derived from repository and research evidence. |
| EXPERIMENT | Promising but not yet reproduced in Hawking on Apple Silicon and not a product claim. |

Component status uses the following terms:

| Status | Meaning |
|---|---|
| ACTIVE | Present in the current workspace and reachable from a current shipping path. |
| ACTIVE, UNEXPOSED | Present, but its useful capability is not available to HIDE or the HTTP boundary. |
| PACKED | Recoverable in the sealed HIDE pack or git history, absent from the active workspace. |
| UI-ONLY | A frontend surface or event shape exists without a working backend capability. |
| PROPOSED | Not implemented. |
| RESEARCH BET | Worth an isolated experiment; evidence is too weak for a roadmap promise. |

Old HIDE documents claim large multi-agent research passes, but the linked per-topic source briefs were never committed. Their methods and numbers are leads, not retained evidence. This pass rechecked the important claims against current primary sources.

## 1. Repository truth: HIDE is recoverable, but not currently whole

### 1.1 What exists now

VERIFIED REPO:

| Area | Current state | Evidence |
|---|---|---|
| React/Tauri HIDE frontend | Present | app/ is a React 19, Zustand, Monaco, xterm, Tauri v2 application. |
| HIDE wire contract | Present | app/src/wire.ts defines intents, events, context, fleet, fork, compact, inline-edit, and tool projections. |
| HIDE transport | Present but backend-missing | app/src/ipc.ts targets POST /v1/hide/intent, GET or WebSocket /v1/hide/events, and POST /v1/hide/connector on port 8744. |
| Desktop sidecar launcher | Present but target-missing | app/src-tauri/src/main.rs attempts to start hide-serve, which is no longer an active workspace binary. |
| Hawking inference runtime | Active | hawking-core and hawking-serve implement models, Metal kernels, continuous batching, OpenAI-shaped chat/tool I/O, metrics, and caching primitives. |
| RWKV recurrent checkpoint/fork primitives | Active, unexposed | Engine and RWKV code include save, load, fork, serialization, and a conditional parity test; no HIDE-facing HTTP route exists. |
| HIDE backend and support crates | Packed | 13 crates were sealed out of the active workspace. |

The sealed manifest is packs/hawking-hide-desktop.json. It records:

- 13 crates;
- 164 files;
- 50,351 content lines;
- source commit 5a99d0e2d7bf7ea822fd41a74f713008bacba1a5;
- archive SHA-256 f0c75f9309120f8375256d560bd6670261aec20793d4dde14a85607990ccfa8c;
- rollback command git checkout 5a99d0e2d7bf -- crates Cargo.toml.

The manifest's offline_cache path is stale, but the archive is present at:

    /Users/scammermike/Downloads/hawking-packs/historical/hawking-hide-desktop/pack.tar.gz

The archive hash matches the manifest. Git history independently preserves the same source.

### 1.2 Intended architecture before extraction

The intended chain was:

    HIDE React/Tauri
        -> hide-serve
        -> hide-backend / hide-kernel
        -> hawking-context / hawking-index / hawking-orch
        -> hide-tools / hide-security / hide-fleet
        -> Hawking HTTP runtime
        -> hawking-core Metal or CPU model execution

The active tree now has the first and final parts but not the middle. Production HIDE cannot honestly be described as a working agent IDE until that seam is restored.

### 1.3 What the packed system contains

VERIFIED REPO, historical source:

- hawking-context: a deterministic reserve-then-fill compiler, manifests, recall-aware degradation, memory storage, and checkpoint protocol shapes;
- hawking-index: tree-sitter parsing, symbols, BLAKE3/Merkle identities, SQLite FTS, graph and hybrid retrieval;
- hawking-orch: adapters, routing, scheduling, grammar scaffolding, and tool-spec decoding;
- hide-kernel: planner, verifier, tools, event loop, repair/replan logic, and execution oracles;
- hide-tools: filesystem, edits, search, process, git, shell, and MCP pieces;
- hide-security: redaction, encrypted storage, sandboxing, and hash-chain support;
- hide-fleet: jobs, worktrees, resource scheduling, merge, isolation, and remote concepts;
- hide-backend: composition, event bus, replay, time travel, and UI projections;
- hide-serve: the missing desktop service boundary;
- hawking-research and hawking-eval: research and evaluation support.

These are useful assets, not proof of a working product. The historical production host bypassed most of them.

### 1.4 The historical shipping path was mostly a facade

VERIFIED REPO, source commit 5a99d0e2:

- the default AgentKernel used a stub planner, no runtime, and no tools;
- the host generated from the raw prompt with empty message history;
- output was capped at 256 tokens;
- no system instructions, compiled repository context, tool loop, files, or prior assistant turns reached the model;
- the context compiler ran, but its compiled prompt was discarded;
- compact_context was logged or dropped rather than performed;
- rich frontend projections often represented mock or optimistic state;
- the planned Hawking state/KV HTTP routes did not exist.

This is why the correct recovery strategy is not “restore every crate and declare HIDE back.” It is to recover only the pieces needed for a tested vertical slice, then reconnect other components behind evidence.

### 1.5 Active Hawking strengths to preserve

VERIFIED REPO:

- a Rust-native runtime with Metal kernels;
- continuous batching and a low-readback greedy path;
- an OpenAI-compatible chat/tool wire shape;
- a Hermes/Qwen-style tool renderer and lenient parser;
- RAM and disk prefix/state cache scaffolding;
- serializable, fixed-size RWKV recurrent state with fork semantics;
- model architecture modules for dense, MoE, DeepSeek-style, and RWKV paths;
- server and engine metrics;
- a recorded M3 Pro result around 31 tokens/s, which is a hardware/model/build-specific observation, not an IDE-speed claim.

### 1.6 Active Hawking gaps that directly matter to HIDE

VERIFIED REPO:

1. There is no HTTP session/state save, load, fork, or rollback surface.
2. There is no session-to-slot affinity contract.
3. The serve path hardcodes max_seq_len to 4096 and exposes no Serve CLI override.
4. Direct-admit prefix reuse misses the common batch-size-one path.
5. The system prompt KV bank is currently a routing hint, not detached reusable KV.
6. Tool parsing shapes API output, but no live IDE tool-dispatch loop exists.
7. Tool-bearing streaming buffers to completion before parsing.
8. Stop strings, batched JSON constraints, and batched speculative decoding are incomplete or dormant.
9. The context endpoint can report an estimated multiplier without proving that the associated runtime path is active.
10. Hawking Serve defaults to a LAN-visible address and has no authentication, which is unsafe for an execution-capable local IDE.
11. Historical claims that weight compression yields a multiplied context window are not valid without separate positional, KV-memory, and recall evidence.

## 2. The objective function

### 2.1 Capability density comes first

Capability density should be measured as a Pareto surface, not one vanity scalar.

Candidate operational definition:

    verified task utility
    -----------------------------------------------
    resident model bytes × model-visible tokens ×
    critical-path seconds × energy or dollar cost

That expression is useful for intuition, but release decisions should be lexicographic:

1. meet correctness, security, and task-success floors;
2. maximize the number and difficulty of verified workflows supported within the device envelope;
3. minimize p95 end-to-end critical-path latency;
4. minimize memory, tokens, energy, and remote cost;
5. minimize user attention and intervention.

A tiny but unreliable model is not capability-dense. A huge model that solves a task while consuming the machine and waiting minutes is not capability-dense either.

### 2.2 The four scoreboards

| Priority | Questions | Primary metrics |
|---|---|---|
| Capability density | How much verified work fits locally and composes cleanly? | private task success, capability coverage, success per resident GB, success per non-cached visible token, quantization delta, security-floor pass rate |
| Speed | How quickly does useful, correct work appear? | keystroke-to-suggestion p50/p95, TTFT, inter-token latency, model-to-tool gap, state restore, time to first valid patch, edit-to-green p50/p95 |
| Performance/quality | Is the work correct, stable, and reproducible? | compile/test/regression outcomes, tool-call validity, retrieval precision/recall/utilization, success@time, multi-run reliability |
| Usefulness | Does it reduce human effort without creating cleanup? | accepted-edit rate, undo rate, intervention count, clarification burden, human attention minutes, time saved against human baseline |

Raw tokens/s remains a Hawking metric. It is only one term in HIDE wall clock:

    queue + context construction + cache lookup + prefill + decoding
    + model/tool gaps + tool execution + verification + user approval

The optimization target is the critical path, not aggregate activity.

## 3. The correct “unbounded context” model

### 3.1 Four layers, not one context window

The product can feel continuous and effectively unbounded without making a false claim about active attention.

| Layer | What it holds | Bound |
|---|---|---|
| Durable project universe | repository snapshots, events, decisions, memory, artifacts, test results, traces, tool outputs | bounded by storage and retention policy |
| Active model context | the exact evidence and instructions in the current inference call | bounded by the selected model |
| Reusable compute state | transformer KV prefixes, recurrent state, or an execution-state capsule | bounded by memory, model identity, and compatibility |
| Working task state | plan, acceptance criteria, current patch, failures, unresolved hypotheses, next actions | compact, structured, reconstructable |

PRIMARY SOURCE: InfiAgent keeps reasoning context bounded while externalizing persistent state into a file-centric workspace snapshot plus recent actions. Coding Agents are Effective Long-Context Processors reports that agents can use files and executable tools to process corpora far beyond an attention window. ContextBench shows that simply exploring more context does not mean the model uses it well.

INFERENCE: HIDE should promise continuity, provenance, and reconstructability. It should not promise infinite active context, perfect recall, or lossless semantic compression.

### 3.2 Local and cloud share one truth format

Cloud and device execution should differ in policy, not in the owner of truth.

Canonical local state:

- append-only task and UI event ledger;
- content-addressed repository and artifact identities;
- current working-tree and patch transaction;
- symbol, lexical, semantic, git, test, and trace indexes;
- versioned project instructions and architecture maps;
- durable memory records with provenance and supersession;
- structured checkpoints;
- evaluation and telemetry receipts.

Provider-specific hot state:

- previous-response or interaction IDs;
- live WebSocket connections;
- remote cache keys and affinity;
- server compaction blocks;
- Hawking session/slot IDs and state capsule references.

If a provider disappears, HIDE can reconstruct from canonical state. If the device is offline, the same task and artifacts continue with Hawking. If a cloud model is selected for a difficult step, it receives the smallest necessary scoped context, not ownership of the project history.

### 3.3 The context compiler contract

Each turn should build a versioned ContextPack:

1. stable policy and repository invariants;
2. stable, small tool namespace manifest;
3. task contract and acceptance criteria;
4. repository map and current snapshot identity;
5. ranked exact source spans with file, symbol, commit, and retrieval provenance;
6. current diff and transaction state;
7. test failures, logs, traces, and runtime evidence;
8. compact durable decisions and unresolved questions;
9. recent action window;
10. immediate query last.

Every item carries:

- content identity;
- source and trust domain;
- token count from the actual target tokenizer;
- inclusion reason and score;
- freshness or invalidation condition;
- whether it was explored, presented, cited, edited, or test-relevant.

Summaries orient. Exact source spans authorize edits. A model should never mutate code based only on a lossy summary when the authoritative bytes are available locally.

### 3.4 Compaction is a checkpoint, not an essay

A HIDE checkpoint should include:

- task and acceptance criteria;
- repository and worktree identity;
- hard constraints and invariants;
- decisions and rejected alternatives;
- touched files and symbols;
- current patch;
- tests executed and their results;
- unresolved hypotheses;
- evidence/artifact references;
- next actions;
- prompt/tool/model ABI version;
- permissions and trust-domain state.

Continuity evaluation must reconstruct a task from the checkpoint only and compare it to an uncompacted control. Measure forgotten constraints, repeated tool work, wrong-file edits, and task-success delta.

## 4. Target architecture: eight cooperating planes

    Human control surface
        |
        v
    Experience plane: Workstation / IDE / Chat / Context Stack
        |
        v
    Task and agent plane: flat inner loop + durable task DAG
        |
        +---- Action plane: typed tools, transactions, sandbox, MCP/LSP
        |
        +---- Context plane: indexes, compiler, memory, checkpoints
        |
        +---- Verification plane: tests, builds, linters, runtime oracles, review
        |
        v
    Model plane: Hawking local lanes + cloud adapters + model/effort router
        |
        v
    State and serving plane: prompt ABI, KV/recurrent capsules, batching, caches
        |
        v
    Observability and evaluation plane across every boundary

Security and provenance are not a ninth optional plane. They wrap every boundary.

### 4.1 Durable truth plane

Responsibilities:

- single-writer append-only event log with crash-safe framing;
- content-addressed artifacts and exact repository snapshots;
- idempotency keys for tool calls and patch transactions;
- explicit supersession rather than destructive memory rewrites;
- replay, fork, rollback, and time-travel semantics;
- local encryption and retention controls;
- portable task checkpoints.

This plane is the source of continuity. A transcript is one projection of it, not the database.

### 4.2 Context and state plane

Responsibilities:

- token-true deterministic context compilation;
- lexical, AST/symbol, LSP, graph, semantic, git, test, and trace retrieval;
- prompt-prefix canonicalization and versioning;
- transformer prefix/radix KV reuse;
- recurrent or hybrid execution-state save/load/fork;
- compaction with continuity tests;
- memory extraction, consolidation, provenance, and revalidation;
- local/cloud checkpoint portability.

HIDE needs separate identities for:

- ContextPack;
- PromptABI;
- ToolRegistryABI;
- ModelWeights;
- Tokenizer and chat template;
- Engine build and state format;
- Repository snapshot;
- Permission/trust policy.

A reusable state must bind to the compatible identities. Silent cross-version reuse is a correctness bug.

### 4.3 Model plane

Three lanes are preferable to one universal model:

| Lane | Purpose | Likely policy |
|---|---|---|
| Reflex | tab completion, classification, retrieval ranking, small transforms | smallest local model or deterministic algorithm meeting the quality floor |
| Local agent | interactive tool use, edits, tests, repository work | capability-dense local coding model on Hawking |
| Escalation | ambiguous architecture, hard debugging, high-risk review | strongest permitted local or cloud model |

Model choice and effort are distinct axes. Effort controls files read, tests run, tools used, alternative hypotheses, and verification depth. A larger model may be faster or cheaper overall on genuinely hard work if it avoids failed loops.

Route at meaningful phases:

- triage;
- exploration;
- planning;
- patch generation;
- test diagnosis;
- review;
- final explanation.

Start with transparent rules. Learn a router only after HIDE has enough outcome-labelled trajectories.

### 4.4 Action plane

The interactive agent's inner loop should be flat and execution-grounded:

    observe -> decide -> invoke typed action -> receive bounded evidence
    -> update task state -> verify -> continue or stop

Planning and verification may happen inside that loop. They should not be a rigid twelve-stage state machine that every task must traverse.

Tools must be:

- typed and schema-versioned;
- cache-stably registered;
- dynamically discoverable;
- explicit about read/write/network/secret effects;
- deadline- and cancellation-aware;
- idempotent or transactionally staged where possible;
- bounded in output, with full data spilled to content-addressed artifacts;
- provenance-labelled and treated as untrusted model input.

Programmatic tool calling should handle deterministic loops, joins, filtering, pagination, and safe fan-out without returning to the model after each result. The model should spend turns on semantic decisions.

### 4.5 Verification plane

“Done” requires objective evidence appropriate to the task:

- patch applies transactionally;
- project builds or typechecks;
- targeted tests pass;
- regression suite does not worsen;
- lints/security/architecture constraints pass;
- requested behavior is exercised;
- high-risk changes receive an independent review;
- the final diff stays within requested scope.

Verification feedback re-enters the flat loop as evidence. Self-judgment is a weak signal, not the accept gate.

### 4.6 Fleet plane

The Workstation should manage durable deliverables, not a wall of chat sessions.

Each task has:

- acceptance criteria;
- dependency edges;
- isolated worktree or write lease;
- resource/model/effort policy;
- current evidence and state checkpoint;
- retry and stall policy;
- integration owner;
- human-attention state.

Parallelize independent read-heavy work: repository exploration, documentation lookup, test diagnosis, and alternative hypotheses. Isolate write-heavy workers. Use a single integration/review path. Return compact evidence packets, not whole transcripts.

Large migrations benefit from:

- a rulebook;
- dependency-aware work queues;
- implement/review/fix roles;
- a shared build/test daemon;
- mechanically resumable work derived from disk state;
- upstream fixes to repeated failure patterns;
- adversarial review only where expected value is positive.

### 4.7 Experience plane

The existing Workstation, IDE, Chat, and Context Stack model is directionally strong. It needs truthful backend semantics.

Core product interactions:

- inline FIM completion and next-edit prediction;
- explicit task creation with acceptance criteria;
- streamed, reviewable patch transactions;
- checkpoint, fork, rollback, cancel, steer, and resume;
- test and runtime evidence attached to changes;
- an attention inbox for blockers, approvals, conflicts, and completed deliverables;
- a calm activity projection that does not equate busyness with progress;
- review changes across files before commit;
- exact provenance for what context influenced a change;
- latency modes such as Instant, Interactive, Thorough, and Background.

The Context Stack should explain inclusion and provenance without presenting a dishonest “percent of infinite context” meter. It should answer:

- what does the model currently know?
- why was this included?
- what was excluded?
- what evidence is stale?
- what state will survive compaction or provider switching?

### 4.8 Observability and evaluation plane

Every run needs one trace spanning:

- queue and routing;
- provider or Hawking connection;
- cache lookup/read/write;
- context retrieval and packing;
- prefill and TTFT;
- decoding and inter-token latency;
- model-to-tool gaps;
- tool execution, retries, and cancellation;
- patch transactions;
- builds/tests/oracles;
- subagent scheduling and joins;
- user approvals and interventions;
- checkpoint and compaction.

Use OpenTelemetry-compatible concepts, but do not capture prompts, source, tool arguments, or results by default. Sensitive content should remain local, sampled explicitly, redacted, and retention-limited.

## 5. Frontier findings and concrete Hawking implications

### 5.1 Long context: maximize effective evidence, not nominal tokens

PRIMARY SOURCE:

- ContextBench contains 1,136 tasks from 66 repositories across eight languages and measures retrieval recall, precision, and efficiency. It finds only marginal retrieval gains from sophisticated scaffolds, a model preference for recall over precision, and a gap between explored and utilized context.
- NoLiMa and RULER show that nominal context length does not guarantee reliable semantic retrieval.
- Coding Agents are Effective Long-Context Processors reports that filesystem and executable-tool use can outperform direct long-context baselines on massive corpora.
- InfiAgent demonstrates bounded active reasoning plus persistent file-centric state.
- Current provider guidance acknowledges TTFT and retrieval degradation as prompts grow.
- SWE-Explore reports that agentic code exploration can beat classical retrieval under fixed line budgets, and that region ranking and coverage predict downstream repair.
- CORE-Bench reports that general embeddings degrade on agentic code retrieval and that task-specific retrieval training can help.
- A July 2026 preprint reports that compressed exact source can outperform summaries for acting on code in its evaluated setup. The result is promising but not yet a general rule.

INFERENCE:

- Keep repository truth outside the prompt.
- Use an agentic explorer under a hard line/token budget.
- Favor exact, ranked source slices over whole-file stuffing.
- Measure retrieved-to-utilized overlap and tokens per solved task.
- Treat a cloud million-token window as an escalation option, not the default memory design.
- Keep a small read/search-only context worker as an optional retrieval stage, but require it to beat deterministic lexical, symbol, graph, and RepoMap baselines.

### 5.2 Prompt caching is a product-level ABI

PRIMARY SOURCE:

- Anthropic states that Claude Code's harness is built around prompt caching and treats hit-rate degradation as an incident.
- Cache hits require an exact shared prefix. Static system content and stable tools come first; project and session material follow; dynamic conversation comes last.
- Changing models, tool order, or definitions mid-session breaks reuse.
- Cache-safe compaction forks from the identical parent prefix and appends the compaction request.
- OpenAI's Codex agent loop uses the same prefix principle and appends state changes rather than mutating earlier messages.

INFERENCE:

Create a PromptABI with:

- deterministic byte/token serialization;
- stable tool ordering;
- versioned system and project instruction blocks;
- append-only dynamic updates;
- model/session affinity;
- a cache-key explanation;
- hit/miss and invalidation telemetry;
- test vectors proving equivalent inputs serialize identically.

The tool registry is part of that ABI. Deferred tool stubs remain stable; full schemas append after discovery.

### 5.3 The strongest capability-density model lead is hybrid recurrent MoE

PRIMARY SOURCE:

Qwen3-Coder-Next is an open-weight coding-agent model with:

- 80B total parameters and 3B activated per token;
- 48 layers;
- a repeating hybrid layout with three Gated DeltaNet plus MoE blocks followed by one gated-attention plus MoE block;
- 512 experts, 10 activated plus one shared expert;
- a native 262,144-token context;
- FIM support;
- a dedicated tool-call format and tokenizer requirements;
- training with executable environments and agentic feedback.

Its vendor-reported coding results are promising, but public coding benchmarks are currently compromised by contamination and broken tasks. The architecture fit matters more than its leaderboard number.

INFERENCE:

Qwen3-Coder-Next should be the first serious architecture-fit study for Hawking's local-agent lane because it combines:

- low active compute;
- sparse experts;
- recurrent/linear-attention state;
- periodic exact attention;
- native coding/tool/FIM behavior;
- a total parameter footprint that can plausibly fit high-memory Apple systems after quantization.

This is not an immediate “support” claim. It requires:

- Gated DeltaNet kernels and state semantics;
- the exact sparse MoE route;
- periodic attention and KV layout;
- tokenizer, special tokens, chat template, tool parser, and FIM contract;
- quantized weight and state formats;
- correctness comparison with a reference runtime;
- Apple-specific prefill, decode, memory, power, and task-quality measurements.

Pure RWKV remains valuable as a fast fixed-state lane. It should not be assumed to dominate a hybrid model on exact retrieval-heavy repository work.

EXPERIMENT: BaseRT, a July 2026 native-Metal preprint and implementation, reports up to 1.56× decode throughput over llama.cpp and 1.35× over MLX in its tested M3/M4 Pro, model, and quantization combinations. This supports Hawking's native-Metal direction, but Hawking should reproduce kernel, model-load, prefill, decode, power, and end-to-end agent results on identical hardware before borrowing the claim.

### 5.3.1 Route from trajectory evidence, not only the initial prompt

PRIMARY SOURCE:

- RouteLLM established a useful strong/weak model-routing baseline, but its historical results are query-level and workload-specific.
- LLMRouterBench finds real model complementarity while also finding that complex routers often fail to beat simple baselines; model-pool selection can matter more.
- TwinRouterBench evaluates routing from partial agent trajectories, tool logs, and diffs and reports a 53% spend reduction at matched resolution in its limited study.
- SWE-Router argues that a cheap model should explore first and that routing should use files, tests, and trajectory evidence. It is a recent preprint, not a production standard.

INFERENCE:

- route separately for exploration, patching, diagnosis, review, and explanation;
- include tests, uncertainty, repeated failures, security risk, cache affinity, queue time, and model complementarity;
- start with a transparent policy and an intentionally small model pool;
- collect counterfactual traces before training a router;
- optimize success@time and success@cost under quality floors, not cheapest-call percentage.

### 5.4 Reuse has three different mechanisms

#### Transformer or hybrid prefix KV

PRIMARY SOURCE: SGLang's RadixAttention, vLLM prefix caching, and TensorRT-LLM demonstrate automatic shared-prefix reuse, continuous batching, structured output, and cache-aware scheduling.

Hawking implication:

- replace a single special-case system-prefix hint with a token-prefix radix or content-addressed cache;
- perform reuse on direct admission, especially batch size one;
- schedule by cache affinity without starving interactive work;
- isolate caches by workspace and trust domain;
- observe admitted, reused, loaded, evicted, and recomputed tokens.

#### Hierarchical KV storage

PRIMARY SOURCE: Strata, published at OSDI 2026, shows that naive GPU/CPU/SSD KV hierarchy becomes I/O-bound due to fragmentation, loading stalls, and schedulers that ignore cache-load time. Its SGLang implementation reports up to 5× throughput over vLLM-LMCache and 3.75× over TensorRT-LLM in its evaluated workloads.

Hawking implication:

- co-design cache layout, transfer size, and scheduler;
- model cache-load latency in admission decisions;
- keep short interactive requests from regressing;
- treat reported datacenter throughput as architectural evidence, not an Apple result.

#### Recurrent or complete execution state

VERIFIED REPO: Hawking has RWKV state serialization and fork primitives, but they are not HIDE HTTP capabilities and GPU-to-CPU capture costs need measurement.

EXPERIMENT: Execution-State Capsules / FlashRT snapshots the closed set of KV, recurrent, convolution, MTP, and metadata buffers at a committed graph boundary. A June 2026 single-author CUDA study reports byte-exact restore, token-identical greedy decoding, sub-millisecond resident restore, and 3.9× to 27× TTFT improvement over cold prefill at 2K to 16K.

Hawking implication:

- do not call the current CPU RWKV checkpoint a complete execution-state capsule;
- build an explicit state inventory and committed-boundary protocol;
- include model, weights, tokenizer, prompt ABI, tool ABI, engine, quantization, and security-domain identities;
- invalidate aborted or partially applied states;
- support GPU-resident, host-resident, and disk tiers;
- parity-test next-token logits and long continuations;
- benchmark Metal capture, fork, restore, and memory pressure.

This is a high-value experiment because local interactive batch-one latency matches the paper's intended serving point. It is not yet a validated Hawking moat.

### 5.5 Long transformer context requires KV work, not a weight multiplier

EXPERIMENT: Open-TQ-Metal, a 2026 single-author preprint, reports fused INT4 KV attention on Apple Silicon, 3.2× KV-memory reduction, and large attention-kernel speedups at 128K in its tested models.

INFERENCE:

- separate weight quantization, KV/state quantization, positional context support, and semantic recall in every API and UI;
- expose native trained context, configured runtime limit, memory-feasible limit, and measured recall envelope separately;
- never report native × weight multiplier as an effective context window;
- reproduce any compressed-KV result in Hawking with quality, top-token, perplexity, and coding-task gates before adopting it.

### 5.6 Tool speed is largely a round-trip problem

PRIMARY SOURCE:

- Anthropic reports examples where tool definitions consumed 55K to 134K tokens.
- Deferred tool search reduced initial definition load in its example and improved large-catalog tool selection in internal evaluations.
- Programmatic Tool Calling keeps intermediate results outside model context and replaces many model round trips with one sandboxed control program. Anthropic reports a 37% average token reduction on its complex research tasks and elimination of 19 or more inference passes in a 20-call example.
- Tool-use examples improved complex parameter accuracy in Anthropic's internal test.
- XGrammar-2 adds dynamic tag dispatch and cross-grammar caching, reporting more than 6× faster compilation and near-zero end-to-end overhead in its evaluated serving systems.
- ToolSpec uses schema and retrieved historical calls as speculative drafts and reports up to 4.2× generation speedup.
- AsyncFC overlaps decoding and function execution using symbolic futures without model fine-tuning.

INFERENCE:

The HIDE tool plane should have:

- three to five always-present core tools;
- a stable deferred namespace index;
- on-demand schemas and concise examples only where ambiguity justifies them;
- a programmatic tool-control sandbox;
- safe concurrent reads;
- dependency-aware futures with cancellation;
- deterministic result commit order;
- constrained tool-call decoding with dynamic tag dispatch;
- schema/history-based tool-call speculation behind a feature flag;
- artifact handles instead of dumping large outputs into context.

Naive grammar masking can suppress tools if the model must first choose between prose and a tool tag. Use two-stage or tag-dispatch semantics and measure tool-selection recall, not merely JSON validity.

Persistent connections and provider hot state are separate speed levers:

- OpenAI reports up to roughly 40% lower end-to-end time from WebSocket Responses mode on its tool-heavy rollouts with more than 20 calls. The continuation cache is connection-local and the result is not universal.
- OpenAI, Anthropic, and Gemini expose different combinations of previous-response IDs, stateless messages, server compaction, and stateful interaction IDs.
- MCP supports persistent subprocess or Streamable HTTP transports, and experimental MCP Tasks provide durable handles for long-running work.

HIDE should keep provider connections, local MCP/LSP/DAP servers, search indexes, and build/test workers warm. Provider continuation IDs and hot connections belong in transient provider state. They must never replace the canonical local task and artifact ledger.

### 5.7 Speculation must be lossless or explicitly stageable

Potential local speed lanes:

- suffix decoding for repetitive agent/edit output;
- prompt or file-as-draft verification for localized edits;
- schema-aware tool-call drafts;
- model-native multi-token prediction;
- safe local speculative reads;
- best-of-N branches from cheap state forks.

Rules:

- token speculation must prove target-distribution or greedy-token equivalence for the configured mode;
- edit speculation must remain a reviewable transaction;
- tool speculation is allowed only for authorized, local/private, side-effect-free, idempotent reads;
- never speculatively send external requests, secrets, messages, writes, deletes, purchases, or credentials;
- measure acceptance, wasted work, rollback cost, memory overhead, critical-path savings, and task-quality delta.

PASTE and ToolSpec are promising 2026 preprints, not Hawking performance receipts. “Ghost Tool Calls” additionally warns that even discarded external requests can leak predicted user intent.

### 5.8 Flat inner loop, structured outer work

PRIMARY SOURCE:

- OpenAI describes the Codex loop as repeated model output, tool execution, observation, and context management.
- Anthropic's migration workflow emphasizes an explicit rulebook, dependency map, resumable queue, implement/review/fix roles, and objective build/test gates.
- OpenAI Symphony uses an issue/task tracker as a control plane, per-task workspaces, retries, stall detection, and resumable agents.

INFERENCE:

- use a flat execution-grounded inner loop for a single task;
- use a durable DAG for multi-task or multi-agent coordination;
- do not encode every cognitive phase as a mandatory kernel state;
- make plans, hypotheses, tests, and decisions explicit artifacts inside the loop;
- stop or escalate on objective criteria, budget, stall, or repeated failure.

### 5.9 Security is a capability enabler and a phase-zero gate

PRIMARY SOURCE:

- Anthropic reports users approved roughly 93% of permission prompts, illustrating approval fatigue.
- Its OS sandbox reduced prompts by 84% in internal use while enforcing filesystem and network boundaries.
- It documents project-local configuration executing before a trust decision, user-pasted prompt exfiltration, malicious tool output, approved-domain exfiltration, and persistent-memory poisoning.
- It concludes that probabilistic model defenses cannot replace deterministic environment boundaries.

HIDE security contract:

- loopback-only authenticated services;
- a folder trust boundary before parsing or executing local configuration, hooks, skills, or MCP definitions;
- canonical path and symlink resolution before scope checks;
- workspace-scoped read/write mounts;
- read-write-no-delete and read-only modes;
- process sandboxing for every execution path, including the terminal;
- network denied by default, with an egress broker based on capabilities rather than domain names alone;
- secrets never entering the sandbox;
- per-session scoped credentials and revocation;
- immutable provenance labels on all tool and external content;
- inspection/sanitization before untrusted output enters model context;
- transactional writes and recoverable deletes;
- append-log crash repair and a single-writer lock;
- memory records treated as an injection persistence surface;
- audit export without exposing sensitive content by default.

Approvals are for meaningful exceptions and irreversible effects, not for every low-risk step. The sandbox must make common work safe enough to proceed without fatigue.

### 5.10 Interoperability prevents the IDE from becoming a closed harness

PRIMARY SOURCE:

- MCP standardizes model applications connecting to tools, resources, and context.
- ACP standardizes communication between code editors and local or remote coding agents.
- LSP and DAP already standardize language intelligence and debugging.

INFERENCE:

HIDE should:

- host MCP behind its security/tool gateway;
- support ACP so external agents can use the HIDE surface and Hawking agents can use other editors;
- consume LSP for definitions, references, diagnostics, and symbols rather than rebuilding all language semantics;
- consume DAP for runtime state;
- expose an OpenAI-compatible model boundary where useful, but keep the richer HIDE task/event/state protocol separate;
- version all adapter capabilities and degrade honestly.

The user should be able to choose Hawking local, a cloud provider, or an external ACP agent without losing project state, review, provenance, or checkpoints.

### 5.11 Product frontier: review, restore, steer, and deliverables

PRIMARY SOURCE:

- Zed exposes external agents via ACP, thread history, checkpoints, review-changes views, and model-dependent capabilities.
- Cursor background agents use isolated remote workspaces and make background work a first-class object, though their network/privacy model demonstrates why HIDE needs explicit local trust boundaries.
- OpenAI Symphony treats the task tracker and per-task workspace as the durable control plane.
- Modern GitHub/VS Code agents converge on project instructions, skills, hooks, MCP, specialized agents, and policy gates.

INFERENCE:

The differentiator is not another chat composer. It is:

- local, durable, inspectable project intelligence;
- near-zero-cost state continuity and forking where Hawking supports it;
- a truthful Context Stack;
- sub-second reflex actions;
- execution-grounded interactive work;
- background deliverables with calm review;
- one consistent safety, provenance, and rollback model across local and cloud agents.

### 5.12 Harness and product lessons worth internalizing

These are observed product and engineering signals, not neutral performance rankings.

#### Model-specific harness profiles

PRIMARY SOURCE: Cursor reports that model and harness jointly determine quality; it provisions models with the edit format they saw in training, tunes prompts and tools per model/version, and observes that mid-chat switching causes an out-of-distribution history plus a cache miss. It recommends staying with one model or using a fresh subagent when possible.

HIDE implication:

- keep the event/task/artifact protocol provider-neutral;
- make prompts, edit tools, tool-call format, reasoning controls, compaction, retry, and stop behavior a versioned ModelProfile;
- do not force every model through one generic edit or tool interface;
- use a portable structured checkpoint when a model handoff is necessary;
- prefer a fresh specialist subagent to corrupting a long, cache-warm parent session.

#### Online outcome metrics

PRIMARY SOURCE: Cursor measures latency, token efficiency, tool calls, cache hits, and a Keep Rate: the fraction of agent-generated code still present after a fixed interval. It also reports that tool errors persist in context and can degrade later decisions.

HIDE implication:

- measure Keep Rate at one, seven, and thirty days where consent and repository retention allow;
- separately record accepted-as-is, accepted-after-human-edit, agent-repaired, reverted, and escaped-defect outcomes;
- classify every tool failure by model, tool, environment, provider, user abort, timeout, and unknown harness bug;
- prevent repeated bad tool results from silently accumulating in active context;
- use delayed outcomes to improve tools, context, routing, and verification before training a larger model.

#### Specialized subagents and worktrees

PRIMARY SOURCE: Current Claude Code subagent profiles can separately specify models, tools, MCP servers, skills, hooks, persistent memory, effort, background execution, and worktree isolation. Agent teams use a dependency-aware shared task list and direct mailboxes, while the documentation explicitly warns that teams cost more tokens and work best on independent tasks.

HIDE implication:

- make AgentProfile and TaskContract explicit data, not prompt folklore;
- isolate write agents by worktree or VM;
- give read-only specialists small tool/context envelopes;
- keep task dependencies and mailboxes durable and validated;
- expose worker value, duplication, blockers, and critical-path contribution in the Workstation;
- retain a single-agent baseline because more agents are not automatically better.

#### Actor and evaluator separation

PRIMARY SOURCE: Claude Code's goal feature uses a separate fast evaluator after each turn, but that evaluator only judges evidence surfaced in the transcript and does not independently execute tools.

HIDE implication:

- use deterministic test/build/policy oracles first;
- use a separate model evaluator for ambiguous acceptance criteria and scope review;
- allow the evaluator to inspect exact artifacts and oracle receipts, not only prose claims;
- never let a prompt evaluator overrule a failing deterministic gate;
- bound loops by time, turns, and resource budget.

#### Revalidated memory

PRIMARY SOURCE: GitHub Copilot Memory stores repository facts with source citations and checks those citations against the current branch before using a fact.

HIDE implication:

- every repository memory record needs evidence references and a revalidation rule;
- corrections should supersede rather than erase history;
- stale or unverified facts remain available for audit but do not enter active context as truth;
- scope memory by repository, user, organization, and trust domain.

#### Small deterministic repository maps remain a serious baseline

PRIMARY SOURCE: Aider builds a graph-ranked repository map that selects relevant signatures under an active token budget, defaulting near 1K tokens. mini-SWE-agent remains a useful control for whether harness complexity is actually helping.

HIDE implication:

- ship a deterministic RepoMap baseline before a learned context system;
- compare every learned retriever or context agent to that small, cheap control;
- require each layer of orchestration to earn its latency and complexity through held-out results.

#### Review artifacts, not hidden autonomy

Across Zed, Claude Code, Cursor, GitHub/VS Code, Google Antigravity, Devin/Windsurf, and Codex/Symphony, the product surface is converging on:

- durable tasks rather than disposable chats;
- reviewable plans, patches, tests, screenshots, and recordings;
- checkpoints and restore;
- background workers with status and intervention;
- isolated workspaces;
- project instructions, skills, hooks, MCP, and specialized agents;
- local/remote continuity.

HIDE should make artifacts the unit of review. A plan, diff, test receipt, runtime trace, screenshot, or benchmark result can be inspected and replayed. A glowing “agent working” indicator cannot.

## 6. Capability inventory and desired frontier

| Facet | Current truth | Desired frontier |
|---|---|---|
| Desktop shell | UI present; production sidecar missing | reliable launch, model/runtime discovery, health/restart, honest degraded state |
| Inline coding | no verified native FIM path | FIM, next edit, low-latency ranking, syntax/type-aware acceptance |
| Interactive agent | historical live path single-shot; backend packed | flat execution-grounded loop with real history, context, tools, cancel, resume |
| Context | strong compiler packed but historically discarded | token-true ContextPack actually sent, exact evidence, provenance, recall/utilization metrics |
| Memory | storage ideas packed, product behavior unproven | append-only facts/events, supersession, hybrid retrieval, revalidation, poisoning defense |
| Tools | parser/API shapes active; dispatcher packed/unwired | persistent typed tool runtime, dynamic discovery, programmatic calls, transactions |
| Structured output | scaffolds dormant on batch path | tag-dispatch constrained decode with validity and selection-recall gates |
| State reuse | RWKV primitives active but unexposed | session affinity, prefix radix, state save/load/fork/rollback, compatibility ABI |
| Fleet | UI and packed scheduler concepts | task DAG, worktrees/write leases, build daemon, evidence joins, stall/retry |
| Verification | small execution smoke receipts only | objective task-specific oracles, private rotating suite, adversarial review |
| Security | packed components plus current gaps | trust-before-config, sandbox all execution, egress/secret boundary, crash-safe ledger |
| Interoperability | historical MCP client; no product integration | MCP gateway, ACP agent boundary, LSP/DAP ingestion, provider-neutral checkpoints |
| Observability | Hawking metrics, no full task trace | local-sensitive OTel-compatible critical path and capability receipts |
| Personalization | concepts only | outcome-labelled preferences/macros/memory, reversible and eval-gated |

## 7. Priority-ordered build program

This is an implementation ladder, not authorization to begin implementation in this research pass.

### Phase 0: restore truth and measurement

Goal: one honest baseline.

1. Freeze and document the HIDE pack recovery decision.
2. Reintroduce only the crates needed for one vertical slice in a dedicated, testable workspace boundary.
3. Make Tauri reliably launch, health-check, restart, and shut down hide-serve and Hawking.
4. Establish workspace trust before reading project-local executable configuration.
5. Bind services to authenticated loopback.
6. Add crash-safe event framing, tail repair, and a workspace single-writer lock.
7. Add end-to-end trace IDs across UI, HIDE, Hawking, tools, tests, and state.
8. Build a small private golden suite from real Hawking/HIDE tasks.
9. Report the current baseline without “infinite,” “fastest,” or multiplied-context claims.

Exit gate:

- a stranger can launch the app, select or install a model, submit a task, see a persisted response, restart, and replay it;
- failures are explicit;
- the same task produces a complete critical-path trace;
- no unsandboxed command path exists.

### Phase 1: capability-dense vertical slice

Goal: a real local coding agent before broad feature recovery.

1. Restore the context compiler, index, typed tools, transactional editing, and execution oracle.
2. Replace the stub/single-shot path with the flat model-tool-observation loop.
3. Persist user, assistant, tool, patch, verification, cancel, and completion events.
4. Feed the compiled ContextPack to the model.
5. Use the actual model tokenizer and expose honest context limits.
6. Implement repository read/search, patch, build/test, and git status as the minimal trusted tool set.
7. Add FIM inline completion with the current best supported local model.
8. Start the Qwen3-Coder-Next Hawking architecture feasibility branch, isolated from the vertical-slice ship path.

Exit gate:

- the agent can solve a private multi-file Rust or TypeScript task through the real app;
- all edits are transactional and reviewable;
- tests, cancel, resume, replay, and restart work;
- context receipts show exactly what reached the model.

### Phase 2: make the repeated loop fast

Goal: eliminate avoidable critical-path work.

1. Define and test PromptABI and ToolRegistryABI.
2. Fix direct-admit prefix reuse and build token-prefix radix caching.
3. Add session/slot affinity.
4. Expose Hawking state save/load/fork with compatibility and parity gates.
5. Keep MCP, LSP, search, test, and build services warm.
6. Add deferred tool discovery and small stable namespaces.
7. Add programmatic tool fan-out/reduction.
8. Honor stops and stream structured tool calls.
9. Wire tag-dispatch constrained decoding to the batched path.
10. Benchmark suffix/file-as-draft and schema-aware speculation behind flags.

Exit gate:

- warm repeated turns demonstrate measured TTFT and edit-to-green improvement without task-quality regression;
- state fork/restore is parity-correct and resource-bounded;
- cache invalidation is explainable;
- tool latency is decomposed into generation, dispatch gap, execution, and result processing.

### Phase 3: make quality scale with effort

Goal: spend extra compute only where it buys verified success.

1. Separate model selection from effort selection.
2. Add transparent phase-aware routing and quality-triggered escalation.
3. Add structured checkpoints and continuity evals.
4. Add hybrid memory with provenance, time/entity links, supersession, and revalidation.
5. Add test-generation and regression oracles where appropriate.
6. Add best-of-N only for tasks whose expected quality gain exceeds latency and compute cost.
7. Use state-forked branches where compatible and isolated text checkpoints elsewhere.
8. Compare quantized Hawking output against reference precision per capability.

Exit gate:

- success@time improves on private tasks;
- escalation triggers are calibrated;
- compaction/memory does not materially reduce success or violate constraints;
- increased effort has a positive measured return.

### Phase 4: Workstation and fleet

Goal: parallel deliverables, not chat-tab multiplication.

1. Promote tasks, dependencies, checkpoints, worktrees, and artifacts to first-class state.
2. Parallelize independent exploration and diagnosis.
3. Serialize or isolate writes.
4. Add a shared build/test service with deduplication.
5. Add retry, stall, resume, crash recovery, and attention-inbox semantics.
6. Add implement/review/fix roles and an integration owner.
7. Stop redundant workers and record wasted work.
8. Make the overnight digest a projection of verified deliverables.

Exit gate:

- parallel work reduces critical-path wall clock on eligible tasks;
- merge conflicts and duplicate work remain within a defined budget;
- every background change is reviewable, reproducible, and attributable.

### Phase 5: local learning flywheel

Goal: improve Hawking/HIDE from real, consented outcomes.

1. Record outcome labels without retaining sensitive source by default.
2. Learn retrieval/ranking, tool selection, effort, and routing before attempting broad model fine-tuning.
3. Harvest execution-labelled Rust and TypeScript trajectories.
4. Train or adapt only behind frozen evals and rollback.
5. Keep personalization local, inspectable, revocable, and separately evaluated for drift.

Exit gate:

- learned components beat simple baselines on held-out private tasks;
- no privacy or cross-project leakage;
- every learned policy can be disabled without corrupting task state.

## 8. Evaluation system

### 8.1 Why public leaderboards cannot be the product objective

PRIMARY SOURCE:

- OpenAI's 2026 audit estimates roughly 30% of SWE-Bench Pro tasks are broken and retracts its earlier recommendation.
- Its earlier audit found SWE-bench Verified increasingly contaminated and misaligned.
- Anthropic reports a six-percentage-point Terminal-Bench swing caused by infrastructure setup and warns against interpreting small differences without matched resources.

Therefore:

- public benchmarks are useful regression signals;
- headline scores are not evidence that HIDE is fast, correct, or useful;
- private rotating real-work tasks are the primary release gate;
- every benchmark result must pin task revision, harness, model, quantization, context policy, tools, compute envelope, cache state, trial count, and confidence interval.

### 8.2 Evaluation lanes

#### Per-change fast lane

- wire-format and tool round-trip tests;
- tokenizer/chat/FIM golden tests;
- context compiler determinism and budget;
- prefix/state parity;
- patch transaction and rollback;
- stop/structured-output correctness;
- cache invalidation;
- sandbox filesystem/network escape tests;
- cancel/replay/crash repair;
- targeted small coding tasks.

#### Nightly lane

- private Rust and TypeScript issue suite;
- multi-file edit-format tasks;
- ContextBench-style retrieval recall, precision, utilization, and efficiency;
- BFCL-style single/multi-turn tool use, hallucination, and latency;
- Terminal-Bench 2.x as a noisy regression indicator with infrastructure receipts;
- Aider Polyglot-style edit compliance;
- long-horizon resume and compaction continuity;
- prompt/tool/memory injection suite;
- quantized versus reference-quality deltas.

#### Release lane

- rotating, contamination-audited private repository tasks;
- repeated trials with confidence intervals;
- cold and warm cache;
- interactive and background latency envelopes;
- human acceptance, undo, intervention, and time-saved study;
- security red team;
- crash/power-loss recovery;
- model/provider outage and offline fallback.

### 8.3 Required trace fields

Minimum non-sensitive record:

- task, snapshot, run, trace, and parent IDs;
- model, provider, quantization, tokenizer, template, engine, prompt ABI, and tool ABI versions;
- context item identities, token counts, sources, trust domains, and retrieval scores;
- cache hit/write/eviction and reused-token counts;
- queue, retrieval, prefill, TTFT, decode, tool gap, tool, verification, and total times;
- state checkpoint/fork/restore sizes and times;
- tool names, effect classes, status, retries, and artifact handles;
- tests/oracles and outcomes;
- user interventions and approvals;
- final patch identity and accept/undo outcome.

Sensitive content capture is opt-in, local, redacted, and retention-limited.

## 9. Research bets and kill criteria

| Bet | Why it matters | First experiment | Kill or pause when |
|---|---|---|---|
| Qwen3-Coder-Next on Hawking | strong hybrid capability-density fit | reference parity, memory, prefill/decode, FIM/tool/private-task run | Apple performance or quant quality fails the local-agent envelope |
| Complete Metal execution-state capsules | near-zero repeated prefill and cheap fork | inventory all live buffers; committed-boundary snapshot; parity and latency | capture/restore cost or memory erases TTFT benefit |
| Token-prefix radix cache | broad transformer/hybrid reuse | batch-one admission and shared repo prefix benchmark | hit rate is low or eviction harms interactive tail latency |
| Fused quantized KV/state | longer local context within memory | reproduce on one supported model with reference-quality gates | coding/recall delta exceeds floor or kernel win is not end-to-end |
| Programmatic tool control | removes model turns and context pollution | search/test fan-out with artifact spill | sandbox/control overhead exceeds saved latency or reliability worsens |
| Async futures | overlap decoding and tools | safe independent local reads with cancellation | dependency errors or wasted work exceed critical-path gain |
| ToolSpec/schema drafts | structured generation acceleration | one stable tool family under constrained decode | tool-selection recall drops or acceptance is too low |
| Suffix/file-as-draft | fast repetitive code edits | localized patch workload with exact verification | rewrite-heavy tasks collapse to baseline or memory cost is excessive |
| Phase/trajectory router | use cheap models without losing success | simple rules, then offline learned comparison | learned router does not beat transparent rules out of sample |
| Best-of-N state forks | turn cheap local branches into quality | hard tasks with execution tie-break | quality gain per second is inferior to one stronger model |
| Large agent fleets | background throughput | dependency DAG with read workers and one writer | duplicate work, conflicts, attention, or thermal contention dominate |

Feature flags, receipts, and rollback are mandatory for every bet.

## 10. Decisions that should be treated as settled

1. HIDE is a first-class Hawking platform, not leftover UI.
2. Durable local project state is canonical.
3. Active context remains bounded and evidence-selected.
4. Retrieval is first-class for recurrent, hybrid, transformer, local, and cloud models.
5. Hawking state and cache capabilities require an explicit versioned serving contract.
6. One flat execution-grounded inner loop replaces the mandatory cognitive FSM.
7. Outer multi-agent work uses a durable DAG, isolated writes, and objective joins.
8. Execution and tests dominate self-judgment as accept signals.
9. Prompt/tool prefix stability is a runtime ABI and monitored SLO.
10. Security containment lands before autonomous execution.
11. Public benchmark numbers cannot authorize product claims.
12. Capability claims require receipts from the real app path.

## 11. Open owner decisions

1. Which Apple hardware envelopes define the first supported tiers: laptop, Pro/Max, and Ultra?
2. Is Qwen3-Coder-Next the first hybrid target, or should another open coding model be evaluated alongside it before kernel work begins?
3. What are the latency contracts for Instant, Interactive, Thorough, and Background?
4. How much of the sealed HIDE backend should be restored versus rewritten around a smaller vertical slice?
5. Should the full-VM execution option exist for unattended fleet work, with Seatbelt as the interactive default?
6. Which cloud providers are permitted, and what data classifications may leave the device?
7. Which private real-world task corpus can be retained for evaluation and training?
8. What product name is canonical in code and docs: Hawking IDE, HIDE, or both?

These decisions affect product policy. They do not block the Phase 0 truth/measurement work.

## 12. Primary source ledger

Sources are grouped by the claim they support. Vendor measurements remain vendor measurements unless reproduced.

### Context, memory, and compaction

- [ContextBench: context retrieval in coding agents](https://arxiv.org/abs/2602.05892)
- [SWE-Explore](https://arxiv.org/abs/2606.07297)
- [CORE-Bench](https://arxiv.org/abs/2606.11864)
- [What Context Does a Coding Agent Actually Need to Act?](https://arxiv.org/abs/2607.09691)
- [Coding Agents are Effective Long-Context Processors](https://arxiv.org/abs/2603.20432)
- [InfiAgent, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1787/)
- [NoLiMa official repository](https://github.com/adobe-research/NoLiMa)
- [RULER official repository](https://github.com/NVIDIA/RULER)
- [Gemini long-context guidance](https://ai.google.dev/gemini-api/docs/long-context)
- [OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [OpenAI WebSocket Responses mode](https://developers.openai.com/api/docs/guides/websocket-mode)
- [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction)
- [Anthropic compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Anthropic context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Anthropic memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- [Mem0 paper](https://arxiv.org/abs/2504.19413)
- [Mem0 official repository](https://github.com/mem0ai/mem0)

### Prompt, prefix, and execution-state reuse

- [Anthropic: Prompt caching is everything](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything)
- [OpenAI: Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching)
- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang official repository](https://github.com/sgl-project/sglang)
- [Strata, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/xie-zhiqiang)
- [Execution-State Capsules](https://arxiv.org/abs/2606.20537)
- [FlashRT official repository](https://github.com/flashrt-project/FlashRT)
- [Open-TQ-Metal](https://arxiv.org/abs/2604.16957)

### Local model and inference architecture

- [Qwen3-Coder official repository](https://github.com/QwenLM/Qwen3-Coder)
- [Qwen3-Coder-Next model card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Qwen3-Coder-Next technical report](https://arxiv.org/abs/2603.00729)
- [BaseRT Apple-Silicon runtime paper](https://arxiv.org/abs/2607.00501)
- [vLLM official repository](https://github.com/vllm-project/vllm)
- [vLLM suffix decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/suffix/)
- [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [RouteLLM](https://arxiv.org/abs/2406.18665)
- [LLMRouterBench](https://arxiv.org/abs/2601.07206)
- [TwinRouterBench](https://arxiv.org/abs/2605.18859)
- [SWE-Router](https://arxiv.org/abs/2607.00053)

### Tools, structured generation, and latency

- [Anthropic advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
- [OpenAI tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
- [OpenAI programmatic tool calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling)
- [Anthropic programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- [XGrammar-2](https://arxiv.org/abs/2601.04426)
- [XGrammar official repository](https://github.com/mlc-ai/xgrammar)
- [ToolSpec](https://arxiv.org/abs/2604.13519)
- [AsyncFC](https://arxiv.org/abs/2605.15077)
- [PASTE](https://arxiv.org/abs/2603.18897)
- [Ghost Tool Calls](https://arxiv.org/abs/2606.02483)

### Agent harness, fleet, and product

- [OpenAI harness engineering](https://openai.com/index/harness-engineering/)
- [OpenAI Symphony specification and announcement](https://openai.com/index/open-source-codex-orchestration-symphony/)
- [Anthropic large-scale code migrations](https://claude.com/blog/ai-code-migration)
- [Cursor: continually improving the agent harness](https://cursor.com/blog/continually-improving-agent-harness)
- [Cursor: cloud-agent runtime lessons](https://cursor.com/blog/cloud-agent-lessons)
- [Claude Code subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code agent teams](https://code.claude.com/docs/en/agent-teams)
- [Claude Code goal evaluator](https://code.claude.com/docs/en/goal)
- [GitHub Copilot Memory](https://docs.github.com/en/copilot/concepts/agents/copilot-memory)
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [Agent Client Protocol](https://agentclientprotocol.com/get-started/introduction)
- [Zed Agent Panel](https://zed.dev/docs/ai/agent-panel)
- [Zed external agents and ACP](https://zed.dev/docs/ai/external-agents)
- [Aider repository map](https://aider.chat/docs/repomap.html)
- [mini-SWE-agent official repository](https://github.com/SWE-agent/mini-swe-agent)
- [Google Antigravity overview](https://antigravity.google/docs/overview)
- [Devin DeepWiki](https://docs.devin.ai/work-with-devin/deepwiki)
- [Windsurf Fast Context](https://docs.devin.ai/desktop/context-awareness/fast-context)

### Security

- [Anthropic: How we contain Claude across products](https://www.anthropic.com/engineering/how-we-contain-claude)
- [Anthropic: Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [MCP security guidance](https://modelcontextprotocol.io/specification/2025-11-25/basic/security_best_practices)

### Evaluation and observability

- [OpenAI: Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)
- [OpenAI: Why SWE-bench Verified no longer measures frontier coding](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [Anthropic: Infrastructure noise in agent evaluations](https://www.anthropic.com/engineering/infrastructure-noise)
- [BFCL leaderboard and methodology](https://gorilla.cs.berkeley.edu/leaderboard)
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)

## 13. Final synthesis

The durable competitive shape is not “an IDE with a local model.” It is a coherent local agent system in which:

- the whole project is addressable without being stuffed into attention;
- every model receives a small, exact, cited working set;
- repeated work reuses byte-stable prompt prefixes or compatible execution state;
- deterministic tool work happens without repeated model turns;
- agents act through typed, transactional, contained tools;
- tests and runtime evidence decide when work is done;
- local Hawking and cloud escalation share the same task truth;
- parallel agents are scheduled around dependencies and human attention;
- the product can replay, explain, fork, roll back, and resume;
- every speed and capability claim is backed by a real end-to-end receipt.

That is the architecture capable of being capability-dense first, fast second, high-performing third, and genuinely useful fourth.
